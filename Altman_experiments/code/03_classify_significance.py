"""
03_classify_significance.py
============================
Classify accounts (ML vs licit) using LightGBM with two feature sets:
  BASE  — 15 raw transaction statistics (no topology)
  FULL  — BASE + 36 FTD features (51 total)

Evaluation: stratified 5-fold CV × 5 seeds.
  Metrics: AUC-ROC, Average Precision, F1 (at 0.5 threshold).
  Significance: Wilcoxon signed-rank test (BASE vs FULL), Holm correction.
  SHAP: TreeExplainer on the last fold/seed model.

Outputs (results/classification/)
-----------------------------------
  cv_results.parquet        — per-fold-seed metrics for both feature sets
  significance_results.csv  — Wilcoxon + Holm summary table
  shap_importance.parquet   — mean |SHAP| per feature (FULL model)
  figures/
    fig_cv_metrics.png
    fig_shap_top20.png
    fig_shap_groups.png
"""

import os, gc, time, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
warnings.filterwarnings("ignore")

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
FEAT_DIR = f"{BASE_DIR}/results/features"
OUT      = f"{BASE_DIR}/results/classification"
FIG_DIR  = f"{OUT}/figures"
os.makedirs(FIG_DIR, exist_ok=True)

print("=" * 60)
print("  IBM AML — Classification & Significance Testing")
print("=" * 60)

# ── 1. Load features ─────────────────────────────────────────
print("\n[1/5] Loading features …")
t0 = time.time()
full_feat = pd.read_parquet(f"{FEAT_DIR}/full_features.parquet")

# Labels
y = full_feat["label"].values.astype(np.int8)

# Feature column groups
meta_cols   = {"account_id", "label"}
base_cols   = [c for c in pd.read_parquet(f"{FEAT_DIR}/base_features.parquet").columns
               if c != "account_id"]
ftd_cols    = [c for c in pd.read_parquet(f"{FEAT_DIR}/ftd_features.parquet").columns
               if c != "account_id"]
full_cols   = base_cols + ftd_cols

X_base = full_feat[base_cols].values.astype(np.float32)
X_full = full_feat[full_cols].values.astype(np.float32)

print(f"  Accounts: {len(y):,}  |  ML: {y.sum():,} ({y.mean():.2%})")
print(f"  BASE dims: {len(base_cols)}  |  FULL dims: {len(full_cols)}  ({time.time()-t0:.1f}s)")

# ── 2. CV × seeds ────────────────────────────────────────────
print("\n[2/5] Cross-validation (5 folds × 5 seeds) …")
t0 = time.time()

LGB_PARAMS = dict(
    objective        = "binary",
    metric           = "auc",
    n_estimators     = 500,
    learning_rate    = 0.05,
    num_leaves       = 63,
    min_child_samples= 20,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    reg_alpha        = 0.1,
    reg_lambda       = 1.0,
    class_weight     = "balanced",
    n_jobs           = -1,
    verbose          = -1,
)

records = []
last_model_base = None
last_model_full = None
last_X_val_base = None
last_X_val_full = None
last_y_val      = None

for seed in range(5):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_base, y)):
        for tag, X in [("base", X_base), ("full", X_full)]:
            clf = lgb.LGBMClassifier(**LGB_PARAMS, random_state=seed)
            clf.fit(
                X[tr_idx], y[tr_idx],
                eval_set=[(X[va_idx], y[va_idx])],
                callbacks=[lgb.early_stopping(30, verbose=False),
                           lgb.log_evaluation(period=-1)],
            )
            proba = clf.predict_proba(X[va_idx])[:, 1]
            pred  = (proba >= 0.5).astype(int)
            records.append({
                "seed": seed, "fold": fold, "feature_set": tag,
                "auc":  roc_auc_score(y[va_idx], proba),
                "ap":   average_precision_score(y[va_idx], proba),
                "f1":   f1_score(y[va_idx], pred, zero_division=0),
            })
            # Keep last fold/seed for SHAP
            if seed == 4 and fold == 4:
                if tag == "base":
                    last_model_base = clf
                    last_X_val_base = X[va_idx]
                else:
                    last_model_full = clf
                    last_X_val_full = X[va_idx]
                    last_y_val      = y[va_idx]

    elapsed = time.time() - t0
    print(f"  Seed {seed} done  ({elapsed:.0f}s)", flush=True)

cv_df = pd.DataFrame(records)
cv_df.to_parquet(f"{OUT}/cv_results.parquet", index=False)

# Summary
summary = cv_df.groupby("feature_set")[["auc","ap","f1"]].agg(["mean","std"])
print("\n  CV summary:")
print(summary.to_string())

# ── 3. Significance tests ─────────────────────────────────────
print("\n[3/5] Wilcoxon signed-rank tests + Holm correction …")

metrics = ["auc", "ap", "f1"]
sig_rows = []
for m in metrics:
    base_scores = cv_df[cv_df.feature_set == "base"][m].values
    full_scores = cv_df[cv_df.feature_set == "full"][m].values
    delta_mean  = full_scores.mean() - base_scores.mean()
    # Wilcoxon requires pairs
    stat, pval  = wilcoxon(full_scores, base_scores, alternative="two-sided",
                           zero_method="wilcox")
    sig_rows.append({
        "metric": m,
        "base_mean": base_scores.mean(), "base_std": base_scores.std(),
        "full_mean": full_scores.mean(), "full_std": full_scores.std(),
        "delta":  delta_mean,
        "W_stat": stat, "p_raw": pval,
    })

sig_df = pd.DataFrame(sig_rows)
# Holm correction
p_sorted = np.sort(sig_df["p_raw"].values)
n = len(p_sorted)
holm_thresh = [0.05 / (n - i) for i in range(n)]
sig_df["p_holm_thresh"] = [holm_thresh[np.where(p_sorted == p)[0][0]]
                            for p in sig_df["p_raw"].values]
sig_df["significant"] = sig_df["p_raw"] < sig_df["p_holm_thresh"]

print(sig_df[["metric","base_mean","full_mean","delta","p_raw","significant"]].to_string(index=False))
sig_df.to_csv(f"{OUT}/significance_results.csv", index=False)

# ── 4. SHAP analysis ─────────────────────────────────────────
print("\n[4/5] Computing SHAP values …")
t0 = time.time()

# Use a sample if too large
MAX_SHAP = 5000
rng = np.random.default_rng(42)
if last_X_val_full.shape[0] > MAX_SHAP:
    idx_s = rng.choice(last_X_val_full.shape[0], MAX_SHAP, replace=False)
    X_shap = last_X_val_full[idx_s]
else:
    X_shap = last_X_val_full

explainer   = shap.TreeExplainer(last_model_full)
shap_values = explainer.shap_values(X_shap)
if isinstance(shap_values, list):
    shap_values = shap_values[1]   # positive class

mean_abs_shap = np.abs(shap_values).mean(axis=0)
shap_df = pd.DataFrame({
    "feature":   full_cols,
    "mean_abs_shap": mean_abs_shap,
    "group":     (["base"] * len(base_cols)) + (["ftd"] * len(ftd_cols)),
})
shap_df = shap_df.sort_values("mean_abs_shap", ascending=False)
shap_df.to_parquet(f"{OUT}/shap_importance.parquet", index=False)

print(f"  SHAP done  ({time.time()-t0:.1f}s)")

# SHAP by group
group_shap = shap_df.groupby("group")["mean_abs_shap"].sum()
total_shap  = group_shap.sum()
print(f"  SHAP mass — base: {group_shap.get('base',0)/total_shap:.1%}  "
      f"ftd: {group_shap.get('ftd',0)/total_shap:.1%}")

# ── 5. Figures ────────────────────────────────────────────────
print("\n[5/5] Generating figures …")

# 5a: CV metric comparison (box plots)
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
metric_labels = {"auc": "AUC-ROC", "ap": "Avg Precision", "f1": "F1"}
colors = {"base": "#4878CF", "full": "#D65F5F"}
for ax, m in zip(axes, metrics):
    for fs in ["base", "full"]:
        vals = cv_df[cv_df.feature_set == fs][m].values
        ax.boxplot(vals, positions=[0 if fs == "base" else 1],
                   widths=0.4, patch_artist=True,
                   boxprops=dict(facecolor=colors[fs], alpha=0.7),
                   medianprops=dict(color="black", linewidth=2))
    row = sig_df[sig_df.metric == m].iloc[0]
    sig_str = f"p={row['p_raw']:.3f}" + (" *" if row["significant"] else "")
    ax.set_title(f"{metric_labels[m]}\n{sig_str}", fontsize=10)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["BASE", "BASE+FTD"])
    ax.grid(axis="y", alpha=0.3)
plt.suptitle("IBM AML: BASE vs BASE+FTD (5-fold CV × 5 seeds)", fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig_cv_metrics.png", dpi=150, bbox_inches="tight")
plt.close()

# 5b: Top-20 SHAP features
top20 = shap_df.head(20)
fig, ax = plt.subplots(figsize=(8, 6))
bar_colors = ["#4878CF" if g == "base" else "#D65F5F" for g in top20["group"]]
ax.barh(top20["feature"][::-1], top20["mean_abs_shap"][::-1], color=bar_colors[::-1])
ax.set_xlabel("Mean |SHAP|")
ax.set_title("Top-20 Features by SHAP (IBM AML, BASE+FTD model)")
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor="#4878CF", label="BASE"),
                   Patch(facecolor="#D65F5F", label="FTD")]
ax.legend(handles=legend_elements, loc="lower right")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig_shap_top20.png", dpi=150, bbox_inches="tight")
plt.close()

# 5c: SHAP mass by group (pie)
fig, ax = plt.subplots(figsize=(5, 4))
fracs = [group_shap.get("base", 0), group_shap.get("ftd", 0)]
labels_ = [f"BASE\n{fracs[0]/total_shap:.1%}", f"FTD\n{fracs[1]/total_shap:.1%}"]
ax.pie(fracs, labels=labels_, colors=["#4878CF", "#D65F5F"],
       autopct=None, startangle=90, wedgeprops={"alpha": 0.8})
ax.set_title("SHAP Mass by Feature Group\n(IBM AML, BASE+FTD model)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig_shap_groups.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"\nAll outputs saved to: {OUT}")
print("Done. Run 04_cross_dataset_comparison.py next.")
