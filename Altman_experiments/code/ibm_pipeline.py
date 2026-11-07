"""
ibm_pipeline.py
===============
Unified pipeline for any IBM AML dataset variant.
Runs all three stages (features, classification, SHAP) for a specified variant.

Usage:
    python ibm_pipeline.py --variant LI-Small
    python ibm_pipeline.py --variant HI-Small   # re-run for verification

Outputs to results/{variant_lower}/
  features/
    base_features.parquet   (n_accounts, 17)
    ftd_features.parquet    (n_accounts, 37)
    full_features.parquet   (n_accounts, 54)
    kcore.parquet           (n_accounts,)
  classification/
    cv_results.parquet
    significance_results.csv
    shap_importance.parquet
    figures/
      fig_cv_metrics.png
      fig_shap_top20.png
      fig_shap_groups.png
  data/
    {variant_lower}_trans.parquet
    {variant_lower}_labels.parquet
    dataset_stats.txt
"""

import os, gc, time, warnings, argparse
import numpy as np
import pandas as pd
import networkx as nx
import lightgbm as lgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import wilcoxon
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from matplotlib.patches import Patch
warnings.filterwarnings("ignore")

# ── Argument parsing ─────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--variant", default="LI-Small",
                    help="Dataset variant: HI-Small, LI-Small, HI-Medium, LI-Medium")
args = parser.parse_args()

VARIANT      = args.variant                        # e.g. "LI-Small"
VARIANT_KEY  = VARIANT.lower().replace("-", "_")   # e.g. "li_small"
KAGGLE_PATH  = os.path.expanduser(
    "~/.cache/kagglehub/datasets/ealtman2019/"
    "ibm-transactions-for-anti-money-laundering-aml/versions/8"
)

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA_DIR = f"{BASE_DIR}/data/{VARIANT_KEY}"
FEAT_DIR = f"{BASE_DIR}/results/{VARIANT_KEY}/features"
CLS_DIR  = f"{BASE_DIR}/results/{VARIANT_KEY}/classification"
FIG_DIR  = f"{CLS_DIR}/figures"

for d in [DATA_DIR, FEAT_DIR, CLS_DIR, FIG_DIR]:
    os.makedirs(d, exist_ok=True)

print("=" * 60)
print(f"  IBM AML Pipeline — {VARIANT}")
print("=" * 60)

# ═══════════════════════════════════════════════════════════
# STAGE 1 — Load and preprocess
# ═══════════════════════════════════════════════════════════
trans_path  = os.path.join(KAGGLE_PATH, f"{VARIANT}_Trans.csv")
trans_cache = f"{DATA_DIR}/{VARIANT_KEY}_trans.parquet"
label_cache = f"{DATA_DIR}/{VARIANT_KEY}_labels.parquet"

print(f"\n[1/7] Loading {VARIANT}_Trans.csv …")
t0 = time.time()

if os.path.exists(trans_cache) and os.path.exists(label_cache):
    print("  Using cached parquet files.")
    trans  = pd.read_parquet(trans_cache)
    labels = pd.read_parquet(label_cache)
else:
    trans = pd.read_csv(trans_path, dtype={
        "From Bank": str, "Account": str,
        "To Bank":   str, "Account.1": str,
        "Is Laundering": np.int8,
    })
    trans.rename(columns={
        "From Bank": "from_bank", "Account": "from_acct",
        "To Bank":   "to_bank",   "Account.1": "to_acct",
        "Amount Received": "amount_recv", "Receiving Currency": "recv_currency",
        "Amount Paid":     "amount_paid", "Payment Currency":   "paid_currency",
        "Payment Format":  "pay_format",  "Is Laundering":      "is_laundering",
        "Timestamp":       "timestamp",
    }, inplace=True)
    trans["from_id"] = trans["from_bank"] + "_" + trans["from_acct"]
    trans["to_id"]   = trans["to_bank"]   + "_" + trans["to_acct"]
    trans.drop(columns=["from_bank","from_acct","to_bank","to_acct"], inplace=True)
    trans.to_parquet(trans_cache, index=False)

    senders   = trans[["from_id","is_laundering"]].rename(columns={"from_id":"account_id"})
    receivers = trans[["to_id",  "is_laundering"]].rename(columns={"to_id":  "account_id"})
    all_accts = pd.concat([senders, receivers], ignore_index=True)
    labels    = all_accts.groupby("account_id")["is_laundering"].max().reset_index()
    labels.columns = ["account_id","label"]
    labels["label"] = labels["label"].astype(np.int8)
    labels.to_parquet(label_cache, index=False)

accounts = labels["account_id"].values
n_accts  = len(accounts)
n_ml     = labels["label"].sum()
print(f"  {len(trans):,} transactions | {n_accts:,} accounts | "
      f"ML: {n_ml:,} ({n_ml/n_accts:.2%})  ({time.time()-t0:.1f}s)")

stats_txt = (
    f"IBM AML {VARIANT} Dataset Statistics\n"
    f"{'='*40}\n"
    f"Transactions : {len(trans):,}\n"
    f"Accounts     : {n_accts:,}\n"
    f"  ML accounts  : {n_ml:,} ({n_ml/n_accts:.2%})\n"
    f"  Licit        : {n_accts-n_ml:,} ({(n_accts-n_ml)/n_accts:.2%})\n"
    f"Illicit txn  : {trans['is_laundering'].sum():,} "
    f"({trans['is_laundering'].mean():.2%})\n"
)
with open(f"{DATA_DIR}/dataset_stats.txt","w") as f:
    f.write(stats_txt)
print(stats_txt)

# ═══════════════════════════════════════════════════════════
# STAGE 2 — BASE features
# ═══════════════════════════════════════════════════════════
print("[2/7] Computing BASE features …")
t0 = time.time()

sent = trans.groupby("from_id").agg(
    n_sent       = ("amount_paid",  "count"),
    vol_sent     = ("amount_paid",  "sum"),
    avg_sent     = ("amount_paid",  "mean"),
    std_sent     = ("amount_paid",  "std"),
    max_sent     = ("amount_paid",  "max"),
    n_recv_banks = ("to_id",        "nunique"),
).fillna(0)

recv = trans.groupby("to_id").agg(
    n_recv       = ("amount_recv", "count"),
    vol_recv     = ("amount_recv", "sum"),
    avg_recv     = ("amount_recv", "mean"),
    std_recv     = ("amount_recv", "std"),
    max_recv     = ("amount_recv", "max"),
    n_sent_banks = ("from_id",     "nunique"),
).fillna(0)

diversity = pd.concat([
    trans[["from_id","recv_currency","pay_format"]].rename(columns={"from_id":"account_id"}),
    trans[["to_id",  "recv_currency","pay_format"]].rename(columns={"to_id":  "account_id"}),
], ignore_index=True).groupby("account_id").agg(
    n_currencies = ("recv_currency", "nunique"),
    n_formats    = ("pay_format",    "nunique"),
).fillna(0)

base = (labels[["account_id"]]
        .merge(sent,      left_on="account_id", right_index=True, how="left")
        .merge(recv,      left_on="account_id", right_index=True, how="left")
        .merge(diversity, on="account_id",                         how="left")
        .fillna(0))
base["sent_recv_ratio"] = base["vol_sent"] / (base["vol_recv"] + 1e-9)
base["degree"]          = base["n_sent"] + base["n_recv"]

base_cols = [c for c in base.columns if c != "account_id"]
print(f"  BASE: {len(base_cols)} dims  ({time.time()-t0:.1f}s)")
base.to_parquet(f"{FEAT_DIR}/base_features.parquet", index=False)
del sent, recv, diversity; gc.collect()

# ═══════════════════════════════════════════════════════════
# STAGE 3 — Adjacency lists
# ═══════════════════════════════════════════════════════════
print("[3/7] Building adjacency lists …")
t0 = time.time()
out_adj: dict = defaultdict(set)
in_adj:  dict = defaultdict(set)
edges_df = trans[["from_id","to_id"]]
edges_df = edges_df[edges_df["from_id"] != edges_df["to_id"]]
for f, t in zip(edges_df["from_id"], edges_df["to_id"]):
    out_adj[f].add(t)
    in_adj[t].add(f)
del edges_df; gc.collect()
print(f"  Done  ({time.time()-t0:.1f}s)")

# ═══════════════════════════════════════════════════════════
# STAGE 4 — k-core
# ═══════════════════════════════════════════════════════════
print("[4/7] Computing k-core …")
t0 = time.time()
G_und = nx.Graph()
edges_und = set()
for v in accounts:
    for u in out_adj[v]: edges_und.add((min(v,u), max(v,u)))
    for u in in_adj[v]:  edges_und.add((min(v,u), max(v,u)))
G_und.add_edges_from(edges_und)
G_und.remove_edges_from(nx.selfloop_edges(G_und))
del edges_und; gc.collect()
core_dict = nx.core_number(G_und)
del G_und; gc.collect()
print(f"  k-core: {len(core_dict):,} nodes  ({time.time()-t0:.1f}s)")
pd.DataFrame({"account_id": list(core_dict.keys()),
              "core":       list(core_dict.values())
              }).to_parquet(f"{FEAT_DIR}/kcore.parquet", index=False)

# ═══════════════════════════════════════════════════════════
# STAGE 5 — FTD features
# ═══════════════════════════════════════════════════════════
print("[5/7] Computing FTD features (36 dims) …")
t0 = time.time()

def compute_ftd(v, out_adj, in_adj, core_dict) -> dict:
    n1_out = out_adj.get(v, set()); n1_in  = in_adj.get(v, set())
    n2_out = set()
    for u in n1_out: n2_out.update(out_adj.get(u, set()))
    n2_out -= n1_out | {v}
    n2_in  = set()
    for u in n1_in:  n2_in.update(in_adj.get(u, set()))
    n2_in  -= n1_in  | {v}

    A1, A2 = len(n1_in), len(n2_in)
    D1, D2 = len(n1_out), len(n2_out)

    asym  = abs(D1-A1)/(A1+D1+1); asym2 = abs(D2-A2)/(A2+D2+1)
    exp_out = D2/(D1+1)

    rec_out_count = sum(1 for w in n2_out
        if sum(1 for u in n1_out if w in out_adj.get(u,set())) >= 2)
    P2_out = sum(len(out_adj.get(u,set())) for u in n1_out)
    rec_out_ratio = 1 - D2/(P2_out+1)

    rec_in_count = sum(1 for x in n2_in
        if sum(1 for u in n1_in if x in in_adj.get(u,set())) >= 2)
    P2_in = sum(len(in_adj.get(u,set())) for u in n1_in)
    rec_in_ratio = 1 - A2/(P2_in+1)

    if D1 >= 2:
        cont = sum(1 for u in n1_out if len(out_adj.get(u,set())) >= 1)/(D1+1)
        leaf = sum(1 for u in n1_out if len(out_adj.get(u,set())) == 0)/(D1+1)
        peel = cont*leaf
    else:
        cont = leaf = peel = 0.0

    Vv = {v}|n1_out|n1_in|n2_out|n2_in; nv = len(Vv)
    und_edges: set = set()
    for u in Vv:
        for w in out_adj.get(u,set()):
            if w in Vv: und_edges.add((min(u,w),max(u,w)))
    ev = len(und_edges)
    density = 2*ev/(nv*(nv-1)) if nv > 1 else 0.0

    parent = {u:u for u in Vv}
    def find(x):
        while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
        return x
    def union(x,y): parent[find(x)]=find(y)
    for (u,w) in und_edges: union(u,w)
    beta0 = len({find(u) for u in Vv})

    Vv_minus = Vv-{v}; parent2 = {u:u for u in Vv_minus}
    def find2(x):
        while parent2[x]!=x: parent2[x]=parent2[parent2[x]]; x=parent2[x]
        return x
    for (u,w) in und_edges:
        if u!=v and w!=v: parent2[find2(u)]=find2(w)
    cc_minus = len({find2(u) for u in Vv_minus}) if Vv_minus else 0
    bridge_frag = cc_minus - beta0

    neighbors_v = {u for u in Vv if (min(v,u),max(v,u)) in und_edges}
    kv = len(neighbors_v)
    tri_v = sum(1 for u in neighbors_v for w in neighbors_v
                if u<w and (min(u,w),max(u,w)) in und_edges)
    clust_v = 2*tri_v/(kv*(kv-1)) if kv>=2 else 0.0
    beta1 = ev-nv+beta0; beta1norm = beta1/(nv+1) if nv>0 else 0.0

    core_v   = core_dict.get(v,0)
    cores_Vv = [core_dict.get(u,0) for u in Vv]
    core_mean = float(np.mean(cores_Vv)) if cores_Vv else 0.0
    core_max  = float(np.max(cores_Vv))  if cores_Vv else 0.0

    deg_und = {u: sum(1 for w in out_adj.get(u,set()) if w in Vv)+
                  sum(1 for w in in_adj.get(u,set())  if w in Vv)
               for u in Vv}
    kappa_edges = {(u,w): 4-deg_und.get(u,0)-deg_und.get(w,0)
                   for (u,w) in und_edges}
    v_edges  = [(u,w) for (u,w) in und_edges if u==v or w==v]
    kappa_v    = np.mean([kappa_edges[e] for e in v_edges]) if v_edges else 0.0
    kappa_mean = np.mean(list(kappa_edges.values()))        if kappa_edges else 0.0
    kappa_min  = np.min(list(kappa_edges.values()))         if kappa_edges else 0.0

    Vv1 = {v}|n1_out|n1_in; und1: set = set()
    for u in Vv1:
        for w in out_adj.get(u,set()):
            if w in Vv1: und1.add((min(u,w),max(u,w)))
    nv1=len(Vv1); ev1=len(und1)
    parent3={u:u for u in Vv1}
    def find3(x):
        while parent3[x]!=x: parent3[x]=parent3[parent3[x]]; x=parent3[x]
        return x
    for (u,w) in und1: parent3[find3(u)]=find3(w)
    beta0_1=len({find3(u) for u in Vv1}); beta1_1=ev1-nv1+beta0_1
    d_beta0=beta0-beta0_1; d_beta1=beta1-beta1_1

    return {
        "A1":A1,"A2":A2,"D1":D1,"D2":D2,
        "asym":asym,"asym2":asym2,
        "div1":D1,"div2":D2,"exp_out":exp_out,
        "rec_out_count":rec_out_count,"rec_out_ratio":rec_out_ratio,
        "rec_in_count":rec_in_count,"rec_in_ratio":rec_in_ratio,
        "peel_score":peel,
        "nv":nv,"ev":ev,"density":density,
        "beta0":beta0,"cc_minus":cc_minus,"bridge_frag":bridge_frag,
        "tri":tri_v,"clust":clust_v,"beta1":beta1,"beta1_norm":beta1norm,
        "core":core_v,"core_mean":core_mean,"core_max":core_max,
        "kappa":kappa_v,"kappa_mean":kappa_mean,"kappa_min":kappa_min,
        "beta0_1":beta0_1,"beta0_2":beta0,"d_beta0":d_beta0,
        "beta1_1":beta1_1,"beta1_2":beta1,"d_beta1":d_beta1,
    }

BATCH = 5000; rows = []; n = len(accounts)
for i, v in enumerate(accounts):
    rows.append(compute_ftd(v, out_adj, in_adj, core_dict))
    if (i+1) % BATCH == 0 or (i+1) == n:
        elapsed = time.time()-t0
        rate    = (i+1)/elapsed
        eta     = (n-i-1)/rate
        print(f"  {i+1:>7,}/{n:,}  ({elapsed:.0f}s elapsed, ETA {eta:.0f}s)", flush=True)

ftd = pd.DataFrame(rows)
ftd.insert(0, "account_id", accounts)
print(f"  FTD shape: {ftd.shape}  ({time.time()-t0:.1f}s)")
ftd.to_parquet(f"{FEAT_DIR}/ftd_features.parquet", index=False)
del out_adj, in_adj, core_dict, rows; gc.collect()

# Merge full features
ftd_cols  = [c for c in ftd.columns if c != "account_id"]
full = (labels.merge(base, on="account_id", how="left")
               .merge(ftd,  on="account_id", how="left")
               .fillna(0))
full.to_parquet(f"{FEAT_DIR}/full_features.parquet", index=False)
print(f"  Full matrix: {full.shape}  |  base={len(base_cols)}  ftd={len(ftd_cols)}")
del base, ftd; gc.collect()

# ═══════════════════════════════════════════════════════════
# STAGE 6 — Classification + significance
# ═══════════════════════════════════════════════════════════
print("\n[6/7] Classification (5-fold CV × 5 seeds) + significance …")
t0 = time.time()

y      = full["label"].values.astype(np.int8)
X_base = full[base_cols].values.astype(np.float32)
X_full = full[base_cols + ftd_cols].values.astype(np.float32)
del full; gc.collect()

LGB_PARAMS = dict(
    objective="binary", metric="auc", n_estimators=500,
    learning_rate=0.05, num_leaves=63, min_child_samples=20,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    class_weight="balanced", n_jobs=-1, verbose=-1,
)

records = []
last_model_full = None; last_X_val_full = None

for seed in range(5):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_base, y)):
        for tag, X in [("base", X_base), ("full", X_full)]:
            clf = lgb.LGBMClassifier(**LGB_PARAMS, random_state=seed)
            clf.fit(X[tr_idx], y[tr_idx],
                    eval_set=[(X[va_idx], y[va_idx])],
                    callbacks=[lgb.early_stopping(30, verbose=False),
                               lgb.log_evaluation(period=-1)])
            proba = clf.predict_proba(X[va_idx])[:,1]
            pred  = (proba >= 0.5).astype(int)
            records.append({
                "seed":seed,"fold":fold,"feature_set":tag,
                "auc": roc_auc_score(y[va_idx], proba),
                "ap":  average_precision_score(y[va_idx], proba),
                "f1":  f1_score(y[va_idx], pred, zero_division=0),
            })
            if seed==4 and fold==4 and tag=="full":
                last_model_full = clf; last_X_val_full = X[va_idx]
    print(f"  Seed {seed} done  ({time.time()-t0:.0f}s)", flush=True)

cv_df = pd.DataFrame(records)
cv_df.to_parquet(f"{CLS_DIR}/cv_results.parquet", index=False)

summary = cv_df.groupby("feature_set")[["auc","ap","f1"]].agg(["mean","std"])
print("\n  CV summary:")
print(summary.to_string())

# Significance
metrics   = ["auc","ap","f1"]
sig_rows  = []
for m in metrics:
    b = cv_df[cv_df.feature_set=="base"][m].values
    f = cv_df[cv_df.feature_set=="full"][m].values
    stat, pval = wilcoxon(f, b, alternative="two-sided", zero_method="wilcox")
    sig_rows.append({"metric":m,
                     "base_mean":b.mean(),"base_std":b.std(),
                     "full_mean":f.mean(),"full_std":f.std(),
                     "delta":f.mean()-b.mean(),
                     "W_stat":stat,"p_raw":pval})
sig_df = pd.DataFrame(sig_rows)
p_sorted = np.sort(sig_df["p_raw"].values); n_t = len(p_sorted)
holm = [0.05/(n_t-i) for i in range(n_t)]
sig_df["p_holm_thresh"] = [holm[np.where(p_sorted==p)[0][0]] for p in sig_df["p_raw"].values]
sig_df["significant"]   = sig_df["p_raw"] < sig_df["p_holm_thresh"]
print("\n  Significance:")
print(sig_df[["metric","base_mean","full_mean","delta","p_raw","significant"]].to_string(index=False))
sig_df.to_csv(f"{CLS_DIR}/significance_results.csv", index=False)

# SHAP
print("\n  Computing SHAP …")
MAX_SHAP = 5000
rng = np.random.default_rng(42)
X_shap = last_X_val_full[rng.choice(last_X_val_full.shape[0], min(MAX_SHAP, last_X_val_full.shape[0]), replace=False)]
explainer   = shap.TreeExplainer(last_model_full)
shap_values = explainer.shap_values(X_shap)
if isinstance(shap_values, list): shap_values = shap_values[1]
mean_abs    = np.abs(shap_values).mean(axis=0)
shap_df = pd.DataFrame({
    "feature": base_cols + ftd_cols,
    "mean_abs_shap": mean_abs,
    "group": ["base"]*len(base_cols) + ["ftd"]*len(ftd_cols),
}).sort_values("mean_abs_shap", ascending=False)
shap_df.to_parquet(f"{CLS_DIR}/shap_importance.parquet", index=False)

grp = shap_df.groupby("group")["mean_abs_shap"].sum()
tot = grp.sum()
print(f"  SHAP mass — base: {grp.get('base',0)/tot:.1%}  ftd: {grp.get('ftd',0)/tot:.1%}")

# ═══════════════════════════════════════════════════════════
# STAGE 7 — Figures
# ═══════════════════════════════════════════════════════════
print("\n[7/7] Figures …")
colors = {"base":"#4878CF","full":"#D65F5F"}
metric_labels = {"auc":"AUC-ROC","ap":"Avg Precision","f1":"F1"}

fig, axes = plt.subplots(1,3,figsize=(12,4))
for ax, m in zip(axes, metrics):
    for fs in ["base","full"]:
        vals = cv_df[cv_df.feature_set==fs][m].values
        ax.boxplot(vals, positions=[0 if fs=="base" else 1], widths=0.4,
                   patch_artist=True, boxprops=dict(facecolor=colors[fs],alpha=0.7),
                   medianprops=dict(color="black",linewidth=2))
    row = sig_df[sig_df.metric==m].iloc[0]
    sig_str = f"p={row['p_raw']:.3f}" + (" *" if row["significant"] else "")
    ax.set_title(f"{metric_labels[m]}\n{sig_str}", fontsize=10)
    ax.set_xticks([0,1]); ax.set_xticklabels(["BASE","BASE+FTD"])
    ax.grid(axis="y", alpha=0.3)
plt.suptitle(f"IBM AML {VARIANT}: BASE vs BASE+FTD (5-fold CV × 5 seeds)", fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig_cv_metrics.png", dpi=150, bbox_inches="tight")
plt.close()

top20 = shap_df.head(20)
fig, ax = plt.subplots(figsize=(8,6))
bar_colors = ["#4878CF" if g=="base" else "#D65F5F" for g in top20["group"]]
ax.barh(top20["feature"][::-1], top20["mean_abs_shap"][::-1], color=bar_colors[::-1])
ax.set_xlabel("Mean |SHAP|")
ax.set_title(f"Top-20 Features by SHAP (IBM AML {VARIANT})")
ax.legend(handles=[Patch(facecolor="#4878CF",label="BASE"),
                   Patch(facecolor="#D65F5F",label="FTD")], loc="lower right")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig_shap_top20.png", dpi=150, bbox_inches="tight")
plt.close()

fig, ax = plt.subplots(figsize=(5,4))
fracs = [grp.get("base",0), grp.get("ftd",0)]
labels_ = [f"BASE\n{fracs[0]/tot:.1%}", f"FTD\n{fracs[1]/tot:.1%}"]
ax.pie(fracs, labels=labels_, colors=["#4878CF","#D65F5F"],
       startangle=90, wedgeprops={"alpha":0.8})
ax.set_title(f"SHAP Mass by Group\n(IBM AML {VARIANT})")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig_shap_groups.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"\nAll outputs saved to: results/{VARIANT_KEY}/")
print(f"Done — IBM AML {VARIANT} pipeline complete.")
