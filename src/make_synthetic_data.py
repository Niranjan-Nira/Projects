"""Generate a small synthetic multi-label support-ticket dataset.

This exists only so the whole pipeline runs end to end out of the box.
For your resume, replace data/tickets.csv with a real public dataset
(same format: a `text` column plus one 0/1 column per label).
"""
import argparse
import random

import pandas as pd

TEMPLATES = {
    "billing": [
        "I was charged twice for my subscription this month",
        "Please send me a refund for the invoice dated last week",
        "My card was declined but the payment still shows as pending",
        "Why is my bill higher than the quoted price",
    ],
    "technical": [
        "The app crashes every time I open the dashboard",
        "I get a 500 error when I try to upload a file",
        "Login page keeps loading and never finishes",
        "The export feature is not working after the update",
    ],
    "account": [
        "I forgot my password and the reset email never arrives",
        "Please change the email address on my account",
        "I need to delete my account and all my data",
        "Two factor authentication locked me out",
    ],
    "shipping": [
        "My order has not arrived and tracking shows no update",
        "The package was delivered to the wrong address",
        "I received a damaged item and need a replacement",
        "How long will international delivery take",
    ],
    "feature_request": [
        "It would be great to have a dark mode option",
        "Can you add bulk export to the reports page",
        "Please support single sign on with our identity provider",
        "I would like an API endpoint for listing all projects",
    ],
}
FILLERS = ["", "Hi team,", "Hello,", "Urgent:", "Thanks in advance.", "Please help."]


def build(n: int, noise: float, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    labels = list(TEMPLATES)
    rows = []
    for _ in range(n):
        k = 1 if rng.random() < 0.75 else 2  # 25% of tickets are multi-label
        chosen = rng.sample(labels, k)
        parts = [rng.choice(TEMPLATES[c]) for c in chosen]
        text = f"{rng.choice(FILLERS)} {' and also '.join(parts)}".strip()
        row = {"text": text, **{c: int(c in chosen) for c in labels}}
        if rng.random() < noise:  # label noise: flip one random label
            flip = rng.choice(labels)
            row[flip] = 1 - row[flip]
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data/tickets.csv")
    a = ap.parse_args()
    df = build(a.n, a.noise, a.seed)
    df.to_csv(a.out, index=False)
    print(f"wrote {len(df)} rows to {a.out}")
