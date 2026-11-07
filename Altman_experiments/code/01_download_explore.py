"""
01_download_explore.py
======================
Download the IBM Synthetic AML dataset (Altman et al., NeurIPS 2023)
and print structural statistics needed for experiment planning.

Dataset: ealtman2019/ibm-transactions-for-anti-money-laundering-aml
We use the HI-Small subset: ~5M transactions, manageable memory footprint.

Outputs (data/)
---------------
  hi_small_trans.parquet   — transactions, compressed
  account_labels.parquet   — per-account binary label
  dataset_stats.txt        — summary statistics
"""

import os, gc
import numpy as np
import pandas as pd
import kagglehub

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA = f"{BASE}/data"
os.makedirs(DATA, exist_ok=True)

print("=" * 60)
print("  IBM AML Dataset — Download & Explore")
print("=" * 60)

# ── 1. Download ──────────────────────────────────────────────
print("\n[1/4] Downloading dataset …")
path = kagglehub.dataset_download("ealtman2019/ibm-transactions-for-anti-money-laundering-aml")
print(f"  Cached at: {path}")
files = sorted(os.listdir(path))
print(f"  Files: {files}")

# ── 2. Load HI-Small transactions ───────────────────────────
print("\n[2/4] Loading HI-Small_Trans.csv …")
# Find the file (name may vary slightly)
trans_file = next(f for f in files if "HI-Small" in f and "Trans" in f)
trans_path = os.path.join(path, trans_file)
print(f"  Reading {trans_file} …")

trans = pd.read_csv(trans_path, dtype={
    "From Bank": str, "Account": str,
    "To Bank":   str, "Account.1": str,
    "Is Laundering": np.int8,
})

# Normalise column names
trans.rename(columns={
    "From Bank": "from_bank", "Account": "from_acct",
    "To Bank":   "to_bank",   "Account.1": "to_acct",
    "Amount Received": "amount_recv", "Receiving Currency": "recv_currency",
    "Amount Paid":     "amount_paid", "Payment Currency":   "paid_currency",
    "Payment Format":  "pay_format",  "Is Laundering":      "is_laundering",
    "Timestamp":       "timestamp",
}, inplace=True)

# Build compound account IDs
trans["from_id"] = trans["from_bank"] + "_" + trans["from_acct"]
trans["to_id"]   = trans["to_bank"]   + "_" + trans["to_acct"]
trans.drop(columns=["from_bank","from_acct","to_bank","to_acct"], inplace=True)

print(f"  Rows: {len(trans):,}")
print(f"  Laundering transactions: {trans['is_laundering'].sum():,} ({trans['is_laundering'].mean():.2%})")
print(f"  Columns: {list(trans.columns)}")

# ── 3. Build account-level labels ───────────────────────────
print("\n[3/4] Building account labels …")
senders   = trans[["from_id","is_laundering"]].rename(columns={"from_id":"account_id"})
receivers = trans[["to_id",  "is_laundering"]].rename(columns={"to_id":  "account_id"})
all_accts = pd.concat([senders, receivers], ignore_index=True)

# An account is ML if ANY of its transactions is is_laundering=1
labels = all_accts.groupby("account_id")["is_laundering"].max().reset_index()
labels.columns = ["account_id", "label"]
labels["label"] = labels["label"].astype(np.int8)

n_accts   = len(labels)
n_ml      = labels["label"].sum()
print(f"  Total accounts: {n_accts:,}")
print(f"  ML accounts:    {n_ml:,} ({n_ml/n_accts:.2%})")
print(f"  Licit accounts: {n_accts - n_ml:,} ({(n_accts-n_ml)/n_accts:.2%})")

# ── 4. Save compressed ──────────────────────────────────────
print("\n[4/4] Saving compressed files …")
trans.to_parquet(f"{DATA}/hi_small_trans.parquet", index=False)
labels.to_parquet(f"{DATA}/account_labels.parquet", index=False)
print(f"  hi_small_trans.parquet  → {os.path.getsize(f'{DATA}/hi_small_trans.parquet')/1e6:.1f} MB")
print(f"  account_labels.parquet  → {os.path.getsize(f'{DATA}/account_labels.parquet')/1e6:.1f} MB")

# ── Stats report ─────────────────────────────────────────────
stats = f"""IBM AML HI-Small Dataset Statistics
=====================================
Source file  : {trans_file}
Transactions : {len(trans):,}
Accounts     : {n_accts:,}
  ML accounts  : {n_ml:,} ({n_ml/n_accts:.2%})
  Licit        : {n_accts-n_ml:,} ({(n_accts-n_ml)/n_accts:.2%})
Illicit txn  : {trans['is_laundering'].sum():,} ({trans['is_laundering'].mean():.2%})
Currencies   : {trans['recv_currency'].nunique()}
Pay formats  : {trans['pay_format'].nunique()}
"""
with open(f"{DATA}/dataset_stats.txt", "w") as f:
    f.write(stats)
print(stats)
print("Done. Run 02_build_features.py next.")
