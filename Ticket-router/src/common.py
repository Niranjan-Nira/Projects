"""Shared helpers: data loading and a fixed train/val/test split.

Every model (baseline and transformer) uses the SAME split so that the
comparison is fair and the test set is never touched during tuning.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
TEXT_COL = "text"


def load_dataframe(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if TEXT_COL not in df.columns:
        raise ValueError(f"CSV must contain a '{TEXT_COL}' column")
    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)
    return df


def label_columns(df: pd.DataFrame) -> list:
    """Every column except `text` is a binary (0/1) label column."""
    return [c for c in df.columns if c != TEXT_COL]


def split(df: pd.DataFrame):
    """70% train / 15% validation / 15% test, deterministic."""
    train, temp = train_test_split(df, test_size=0.30, random_state=SEED)
    val, test = train_test_split(temp, test_size=0.50, random_state=SEED)
    return (x.reset_index(drop=True) for x in (train, val, test))


def to_matrix(df: pd.DataFrame, labels: list) -> np.ndarray:
    return df[labels].values.astype(int)
