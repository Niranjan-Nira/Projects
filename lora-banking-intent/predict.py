"""Quick demo: classify a query with the fine-tuned adapter.
Usage: python predict.py "I still haven't received my new card"
"""
import sys

import torch
from peft import PeftModel

import common as C
from benchmark import generate


def main():
    query = " ".join(sys.argv[1:]) or "I still haven't received my new card"
    _, _, labels = C.load_banking77()
    tok = C.load_tokenizer()
    model = PeftModel.from_pretrained(C.load_base_model(), C.ADAPTER_DIR).eval()
    raw = generate(model, tok, [C.render(tok, C.messages_finetuned(query))])[0]
    label, valid = C.to_label(raw, labels)
    print(f"query : {query}\nintent: {label}  (exact label: {valid})")


if __name__ == "__main__":
    main()
