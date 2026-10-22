"""QLoRA fine-tuning of a small instruct LLM on Banking77 intent classification."""
import json
import os
import time

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (DataCollatorForSeq2Seq, Trainer, TrainingArguments,
                          set_seed)

import common as C

os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", C.MLFLOW_EXPERIMENT)


def build_dataset(train_ds, labels, tok):
    names = labels

    def encode(ex):
        prompt = C.render(tok, C.messages_finetuned(ex["text"]))
        answer = names[ex["label"]] + tok.eos_token          # <|im_end|> for Qwen
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        a_ids = tok(answer, add_special_tokens=False)["input_ids"]
        return {
            "input_ids": p_ids + a_ids,
            "attention_mask": [1] * (len(p_ids) + len(a_ids)),
            "labels": [-100] * len(p_ids) + a_ids,            # loss on the answer only
        }

    return train_ds.map(encode, remove_columns=train_ds.column_names)


def main():
    set_seed(C.SEED)
    train_ds, _, labels = C.load_banking77()
    tok = C.load_tokenizer()
    data = build_dataset(train_ds, labels, tok)

    model = C.load_base_model()
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    ))
    trainable, total = model.get_nb_trainable_parameters()
    print(f"trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    args = TrainingArguments(
        output_dir=os.path.join(C.OUT_DIR, "checkpoints"),
        num_train_epochs=2,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
        logging_steps=20,
        save_strategy="no",
        optim="paged_adamw_8bit",
        bf16=bf16,
        fp16=not bf16,
        gradient_checkpointing=True,
        report_to=["mlflow"],
        seed=C.SEED,
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=data,
        data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100),
    )

    t0 = time.time()
    trainer.train()
    train_seconds = time.time() - t0

    model.save_pretrained(C.ADAPTER_DIR)
    tok.save_pretrained(C.ADAPTER_DIR)
    stats = {
        "base_model": C.BASE_MODEL,
        "trainable_params": trainable,
        "total_params": total,
        "train_seconds": round(train_seconds, 1),
        "train_cost_usd": round(train_seconds / 3600 * C.GPU_PRICE_PER_HOUR, 4),
        "train_examples": len(data),
    }
    os.makedirs(C.OUT_DIR, exist_ok=True)
    with open(os.path.join(C.OUT_DIR, "train_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(stats)


if __name__ == "__main__":
    main()
