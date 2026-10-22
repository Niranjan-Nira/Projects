"""Shared config, data loading, prompt builders and model helpers."""
import difflib
import re

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ----------------------------- config ---------------------------------
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"   # swap to Qwen2.5-0.5B-Instruct for a faster run
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ADAPTER_DIR = "outputs/lora-adapter"
OUT_DIR = "outputs"
SEED = 42
RAG_K = 8                                    # retrieved examples per query
GPU_PRICE_PER_HOUR = 0.35                    # USD, roughly a cloud T4; edit to your GPU
MLFLOW_EXPERIMENT = "lora-banking77"

SYSTEM_MSG = "You are a banking intent classifier. Reply with only the intent label."


# ----------------------------- data -----------------------------------
def load_banking77():
    ds = load_dataset("PolyAI/banking77")
    labels = ds["train"].features["label"].names
    return ds["train"], ds["test"], labels


# ----------------------------- prompts --------------------------------
def messages_finetuned(query):
    """Short prompt: the model has learned the label space during fine-tuning."""
    return [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": f"Banking query: {query}\nIntent:"},
    ]


def messages_zeroshot(query, labels):
    """Prompting baseline: the full label list goes into every request."""
    label_block = "\n".join(labels)
    return [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": (
            f"Choose exactly one intent from this list:\n{label_block}\n\n"
            f"Banking query: {query}\nIntent:")},
    ]


def messages_rag(query, examples):
    """RAG baseline: retrieved labeled neighbours act as in-context examples."""
    ex_block = "\n".join(f"Query: {t}\nIntent: {l}" for t, l in examples)
    return [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": (
            f"Here are similar labeled queries:\n{ex_block}\n\n"
            f"Banking query: {query}\nIntent:")},
    ]


def render(tok, messages):
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ----------------------------- model ----------------------------------
def bnb_config():
    compute = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute,
    )


def load_tokenizer():
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_base_model():
    return AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb_config(), device_map={"": 0}
    )


# ----------------------------- output parsing -------------------------
def _norm(s):
    return re.sub(r"[\s\-]+", "_", s.strip().lower()).strip("_.")


def to_label(raw, labels):
    """Map raw generation to a label. Returns (label, was_valid)."""
    first_line = raw.strip().split("\n")[0]
    cand = _norm(first_line)
    if cand in labels:
        return cand, True
    close = difflib.get_close_matches(cand, labels, n=1, cutoff=0.0)
    return (close[0] if close else labels[0]), False
