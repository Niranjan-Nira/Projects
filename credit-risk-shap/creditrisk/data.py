"""Data for credit-default modelling.

Two sources, same schema (UCI "Default of Credit Card Clients"):
  * make_synthetic(): reproducible simulator with KNOWN structure (used when you have no data / offline)
  * load_uci(path):   the real UCI CSV/XLS  (https://archive.ics.uci.edu/dataset/350)

Protected attributes (audited, NOT used by the production model): SEX, AGE (-> age_group), MARRIAGE.
"""
import numpy as np
import pandas as pd

TARGET = "default"
PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYAMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]
PROTECTED = ["SEX", "AGE", "MARRIAGE"]
AUDIT_ATTRS = ["sex", "age_group", "marriage"]          # categorical audit columns created in prepare()


def _sigmoid(x): return 1 / (1 + np.exp(-x))


def make_synthetic(n=30000, seed=0, target_rate=0.22):
    """Simulator with a *deliberate, documented* structure so the audit has something to find:
       - default is driven by latent financial stress (-> repayment status, utilisation) and income,
         NOT by sex directly;
       - a pay-gap proxy: women have lower income => lower LIMIT_BAL (proxy path for sex);
       - young borrowers (<30) carry genuinely higher default risk (base-rate difference);
       - younger borrowers' repayment history is shorter/noisier (less reliable features => worse calibration)."""
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(35.5, 9.2, n).round(), 21, 70).astype(int)
    sex = rng.choice([1, 2], n, p=[0.40, 0.60])                    # 1=male, 2=female (UCI coding)
    edu = rng.choice([1, 2, 3, 4], n, p=[0.35, 0.47, 0.16, 0.02])
    mar = rng.choice([1, 2, 3], n, p=[0.45, 0.53, 0.02])
    stress = rng.normal(0, 1, n)                                    # latent financial stress
    log_inc = 10.6 + 0.012 * (age - 35) + 0.25 * (4 - edu) / 3 - 0.18 * (sex == 2) + rng.normal(0, 0.45, n)
    limit = np.clip((np.exp(log_inc - 10.6) * 140000 * rng.lognormal(0, 0.35, n)) // 10000 * 10000, 10000, 800000)
    young = age < 30
    logit = -1.75 + 0.95 * stress - 0.45 * (log_inc - 10.6) + 0.35 * young + 0.05 * (edu == 4)
    logit += (target_rate - _sigmoid(logit).mean()) * 4             # roughly centre base rate
    y = (rng.random(n) < _sigmoid(logit)).astype(int)

    # repayment status: -2..8, persistent AR(1)-like path driven by stress and (eventually) the outcome
    base = 0.9 * stress + 1.1 * (y - 0.22) + rng.normal(0, 0.9, n)
    pay = np.zeros((n, 6)); prev = base
    for t in range(6):
        noise_sd = np.where(young, 1.1, 0.8)
        cur = 0.65 * prev + 0.35 * base + rng.normal(0, 1, n) * noise_sd * 0.6
        pay[:, t] = np.clip(np.round(cur * 1.2 - 0.3), -2, 8); prev = cur
    util = np.clip(0.45 + 0.25 * stress + 0.15 * (y - .22) + rng.normal(0, 0.25, n), 0, 1.3)
    bills = np.column_stack([np.clip(limit * util * rng.lognormal(0, 0.15, n), 0, None) for _ in range(6)])
    payamt = np.column_stack([np.clip(bills[:, i] * np.clip(rng.beta(2, 5, n) - 0.08 * stress, 0, 1.2), 0, None) for i in range(6)])
    df = pd.DataFrame({"LIMIT_BAL": limit, "SEX": sex, "EDUCATION": edu, "MARRIAGE": mar, "AGE": age})
    for i, c in enumerate(PAY_COLS): df[c] = pay[:, i].astype(int)
    for i, c in enumerate(BILL_COLS): df[c] = bills[:, i].round()
    for i, c in enumerate(PAYAMT_COLS): df[c] = payamt[:, i].round()
    df[TARGET] = y
    return df


def load_uci(path):
    df = pd.read_excel(path, header=1) if str(path).endswith((".xls", ".xlsx")) else pd.read_csv(path)
    df = df.rename(columns={"default payment next month": TARGET, "default.payment.next.month": TARGET,
                            "PAY_1": "PAY_0"})
    df.columns = [c.strip() for c in df.columns]
    df = df.drop(columns=[c for c in ("ID",) if c in df.columns])
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4}); df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    return df


def prepare(df):
    """Feature engineering + categorical audit columns."""
    d = df.copy()
    d["util_ratio"] = (d["BILL_AMT1"] / d["LIMIT_BAL"].clip(lower=1)).clip(-1, 3)
    d["avg_util"] = (d[BILL_COLS].mean(1) / d["LIMIT_BAL"].clip(lower=1)).clip(-1, 3)
    d["pay_ratio"] = (d[PAYAMT_COLS].sum(1) / d[BILL_COLS].clip(lower=0).sum(1).clip(lower=1)).clip(0, 5)
    d["max_delinq"] = d[PAY_COLS].max(1); d["mean_delinq"] = d[PAY_COLS].mean(1)
    d["n_late"] = (d[PAY_COLS] > 0).sum(1)
    d["sex"] = d["SEX"].map({1: "male", 2: "female"})
    d["age_group"] = pd.cut(d["AGE"], [0, 29, 44, 200], labels=["<30", "30-44", "45+"]).astype(str)
    d["marriage"] = d["MARRIAGE"].map({1: "married", 2: "single", 3: "other"})
    return d


def feature_lists(d):
    """Return (features_with_protected, features_without_protected) using AGE/SEX/MARRIAGE numeric codes."""
    drop = {TARGET, "sex", "age_group", "marriage"}
    aware = [c for c in d.columns if c not in drop]
    unaware = [c for c in aware if c not in PROTECTED]
    return aware, unaware


def split(d, seed=0, test=0.25, cal=0.15):
    """train / calibration / test (stratified on target)."""
    from sklearn.model_selection import train_test_split
    dev, te = train_test_split(d, test_size=test, stratify=d[TARGET], random_state=seed)
    tr, ca = train_test_split(dev, test_size=cal / (1 - test), stratify=dev[TARGET], random_state=seed)
    return tr.reset_index(drop=True), ca.reset_index(drop=True), te.reset_index(drop=True)
