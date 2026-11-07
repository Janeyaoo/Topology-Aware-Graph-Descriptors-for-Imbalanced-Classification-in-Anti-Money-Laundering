"""Supplementary analyses for the ESWA manuscript (2026-10-12).

Read-only with respect to all existing experiment artifacts: it loads saved
predictions, splits, feature caches and final models from the frozen tuned
batch and writes new tables to a separate output directory. No model is
fitted and no feature is re-extracted.

Parts
  stats    corrected resampled t-tests (Nadeau & Bengio, 2003) with Holm
           correction for the paired representation increments
  budget   Precision@k / Recall@k at fixed alert budgets per outer fold
  patterns detection rate per laundering typology at fixed alert budgets
  shap     TreeSHAP module shares for LightGBM / XGBoost final models
  timing   recorded fit times and measured inference time

Usage: python eswa_supplementary_analysis_20261012.py {stats,budget,patterns,shap,timing,all}
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

ROOT = Path(__file__).resolve().parents[2]
TUNED = ROOT / "Altman_experiments/results/account_gmkep56_tuned_20261008"
PREP = ROOT / "Altman_experiments/results/paper_eswa_preparation_20261011/tables"
DATA = ROOT / "Altman_experiments/data"
OUT = ROOT / "Altman_experiments/results/paper_eswa_supplementary_analysis_20261012"

DATASETS = ["hi_small", "li_small"]
MODELS = ["lightgbm", "xgboost", "random_forest"]
FSETS = ["base16", "gmke44", "gmkep56"]
REP = {"base16": "Base", "gmke44": "Base+Struct", "gmkep56": "Base+Struct+PH"}
SEEDS, FOLDS = range(5), range(5)
BUDGETS = [0.005, 0.01, 0.02]
# outer test / outer train size ratio for 5-fold CV
TEST_TRAIN_RATIO = 1 / 4
N_FOLDS_TOTAL = 25
MODULES = {"T": range(0, 16), "G": range(16, 32), "M": range(32, 37),
           "K": range(37, 40), "E": range(40, 44), "P": range(44, 56)}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fold_record(ds, model, fs, seed, fold):
    return json.loads((TUNED / ds / "folds" / f"{model}_{fs}_seed{seed}_fold{fold}.json").read_text())


def tuned_prediction(ds, model, fs, seed, fold):
    rec = fold_record(ds, model, fs, seed, fold)["strategies"]["tuned"]
    return pd.read_parquet(ROOT / rec["prediction_path"])


def corrected_t(d):
    """Corrected resampled t-test for repeated k-fold CV (Nadeau & Bengio, 2003)."""
    d = np.asarray(d, float)
    n = d.size
    var = d.var(ddof=1)
    if var == 0:
        return np.nan, np.nan
    t = d.mean() / np.sqrt((1 / n + TEST_TRAIN_RATIO) * var)
    return t, 2 * st.t.sf(abs(t), df=n - 1)


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def top_k_flags(score, frac):
    k = int(np.ceil(frac * score.size))
    order = np.argsort(-score, kind="stable")
    flags = np.zeros(score.size, bool)
    flags[order[:k]] = True
    return flags, k


# ---------------------------------------------------------------- stats
def run_stats():
    pf = pd.read_csv(PREP / "paired_feature_fold_values.csv", float_precision="round_trip")
    g32 = pd.read_csv(PREP / "fixed_graph32_fold_values.csv", float_precision="round_trip")
    df = pd.concat([pf, g32], ignore_index=True)
    df = df[df.strategy.isin(["tuned", "original_fixed"]) & (df.threshold_policy == "inner_max_f1")]
    rows = []
    for key, g in df.groupby(["dataset", "model", "strategy", "contrast_id"]):
        g = g.sort_values(["seed", "fold"])
        assert len(g) == N_FOLDS_TOTAL, key
        row = dict(zip(["dataset", "model", "strategy", "contrast_id"], key))
        row["contrast"] = g.contrast.iloc[0]
        for metric in ["ap", "f1", "auc"]:
            d = g[f"delta_{metric}"].to_numpy()
            t, p = corrected_t(d)
            row.update({f"mean_delta_{metric}": d.mean(), f"sd_delta_{metric}": d.std(ddof=1),
                        f"t_{metric}": t, f"p_{metric}": p,
                        f"wilcoxon_p_{metric}": st.wilcoxon(d).pvalue if np.any(d != 0) else np.nan,
                        f"n_pos_{metric}": int((d > 0).sum())})
        rows.append(row)
    res = pd.DataFrame(rows)
    # Holm families: tuned increments (12) and fixed-parameter increments (12 + 6 Graph32)
    for metric in ["ap", "f1", "auc"]:
        res[f"p_holm_{metric}"] = np.nan
        for strat in ["tuned", "original_fixed"]:
            m = res.strategy == strat
            res.loc[m, f"p_holm_{metric}"] = holm(res.loc[m, f"p_{metric}"].to_numpy())
    # sanity check against the published paired-increment table
    pub = pd.read_csv(PREP / "table03_paired_feature_increments.csv", float_precision="round_trip")
    checks = []
    for _, r in res.iterrows():
        if r.contrast == "44-32":
            continue
        cand = pub[(pub.dataset == r.dataset) & (pub.model == r.model) & (pub.strategy == r.strategy)
                   & (pub.contrast_id == r.contrast_id) & (pub.threshold_policy == "inner_max_f1")]
        if len(cand):
            checks.append(abs(float(cand["delta_ap_mean"].iloc[0]) - r.mean_delta_ap))
    res.to_csv(OUT / "stats_corrected_t_tests.csv", index=False)
    return {"rows": len(res), "max_abs_diff_vs_published_mean_delta_ap": max(checks) if checks else None,
            "n_checked": len(checks), "inputs": {str(PREP / "paired_feature_fold_values.csv"): sha(PREP / "paired_feature_fold_values.csv"),
                                                 str(PREP / "fixed_graph32_fold_values.csv"): sha(PREP / "fixed_graph32_fold_values.csv")}}


# ---------------------------------------------------------------- budget
def run_budget():
    from sklearn.metrics import average_precision_score
    fm = pd.read_csv(TUNED / "fold_metrics.csv", float_precision="round_trip")
    fm = fm[(fm.strategy == "tuned") & (fm.threshold_policy == "inner_max_f1")].set_index(["dataset", "model", "feature_set", "seed", "fold"])
    rows, max_ap_diff = [], 0.0
    for ds in DATASETS:
        for model in MODELS:
            for fs in FSETS:
                for seed in SEEDS:
                    for fold in FOLDS:
                        p = tuned_prediction(ds, model, fs, seed, fold)
                        y, s = p.y_true.to_numpy(), p.y_score.to_numpy()
                        ap = average_precision_score(y, s)
                        max_ap_diff = max(max_ap_diff, abs(ap - fm.loc[(ds, model, fs, seed, fold), "ap"]))
                        for b in BUDGETS:
                            flags, k = top_k_flags(s, b)
                            tp = int(y[flags].sum())
                            rows.append({"dataset": ds, "model": model, "feature_set": fs, "seed": seed, "fold": fold,
                                         "budget": b, "k": k, "n_test": y.size, "n_positive": int(y.sum()), "tp": tp,
                                         "precision_at_k": tp / k, "recall_at_k": tp / y.sum()})
    fold = pd.DataFrame(rows)
    fold.to_csv(OUT / "budget_fold_values.csv", index=False)
    summ = fold.groupby(["dataset", "model", "feature_set", "budget"]).agg(
        precision_mean=("precision_at_k", "mean"), precision_sd=("precision_at_k", lambda x: x.std(ddof=1)),
        recall_mean=("recall_at_k", "mean"), recall_sd=("recall_at_k", lambda x: x.std(ddof=1))).reset_index()
    summ.to_csv(OUT / "budget_summary.csv", index=False)
    # paired increments with corrected t-test
    prs = []
    for (ds, model, b), g in fold.groupby(["dataset", "model", "budget"]):
        piv = g.pivot_table(index=["seed", "fold"], columns="feature_set", values="recall_at_k")
        for lhs, rhs in [("gmke44", "base16"), ("gmkep56", "gmke44")]:
            d = (piv[lhs] - piv[rhs]).to_numpy()
            t, p = corrected_t(d)
            prs.append({"dataset": ds, "model": model, "budget": b, "contrast": f"{REP[lhs]} - {REP[rhs]}",
                        "mean_delta_recall": d.mean(), "sd_delta_recall": d.std(ddof=1), "p": p, "n_pos": int((d > 0).sum())})
    prs = pd.DataFrame(prs)
    for b in BUDGETS:
        m = prs.budget == b
        prs.loc[m, "p_holm"] = holm(prs.loc[m, "p"].to_numpy())
    prs.to_csv(OUT / "budget_paired_recall.csv", index=False)
    return {"fold_rows": len(fold), "max_abs_ap_diff_vs_fold_metrics": max_ap_diff,
            "tie_rule": "stable descending sort; ties at the cut-off resolved by row order"}


# ---------------------------------------------------------------- patterns
def parse_patterns(ds):
    fn = DATA / ds / ("HI-Small_Patterns.txt" if ds == "hi_small" else "LI-Small_Patterns.txt")
    rows, ptype = [], None
    for line in fn.read_text().splitlines():
        line = line.strip()
        if line.startswith("BEGIN"):
            ptype = line.split(" - ")[1].split(":")[0].strip()
            continue
        if not line or line.startswith("END"):
            continue
        c = line.split(",")
        rows += [(ptype, c[1] + "_" + c[2]), (ptype, c[3] + "_" + c[4])]
    return pd.DataFrame(rows, columns=["pattern", "account_id"]).drop_duplicates(), sha(fn)


def run_patterns():
    rows, meta = [], {}
    for ds in DATASETS:
        pat, digest = parse_patterns(ds)
        acc = pd.read_parquet(TUNED / ds / "accounts.parquet")
        assert pat.account_id.isin(acc.account_id).all()
        pos = set(acc.account_id[acc.label == 1])
        assert set(pat.account_id) <= pos
        other = pos - set(pat.account_id)
        groups = {p: set(g.account_id) for p, g in pat.groupby("pattern")}
        groups["OTHER (not in pattern file)"] = other
        groups["ALL POSITIVES"] = pos
        meta[ds] = {"pattern_file_sha256": digest, "pattern_accounts": int(pat.account_id.nunique()),
                    "positives": len(pos), "coverage": pat.account_id.nunique() / len(pos),
                    "accounts_per_group": {k: len(v) for k, v in groups.items()}}
        for model in MODELS:
            for fs in FSETS:
                for seed in SEEDS:
                    flagged = {b: set() for b in BUDGETS}
                    seen = 0
                    for fold in FOLDS:
                        p = tuned_prediction(ds, model, fs, seed, fold)
                        seen += len(p)
                        s = p.y_score.to_numpy()
                        for b in BUDGETS:
                            f, _ = top_k_flags(s, b)
                            flagged[b] |= set(p.account_id.to_numpy()[f])
                    assert seen == len(acc)
                    for b in BUDGETS:
                        for name, members in groups.items():
                            rows.append({"dataset": ds, "model": model, "feature_set": fs, "seed": seed, "budget": b,
                                         "pattern": name, "n_accounts": len(members),
                                         "detected": len(members & flagged[b]),
                                         "recall": len(members & flagged[b]) / len(members)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "pattern_seed_values.csv", index=False)
    summ = df.groupby(["dataset", "model", "feature_set", "budget", "pattern", "n_accounts"]).agg(
        recall_mean=("recall", "mean"), recall_sd=("recall", lambda x: x.std(ddof=1))).reset_index()
    summ.to_csv(OUT / "pattern_summary.csv", index=False)
    return meta


# ---------------------------------------------------------------- shap
def _shap_one(args):
    """TreeSHAP for one saved final model; result is written to its own file."""
    ds, model, seed, fold = args
    part = OUT / "shap_parts" / f"{ds}_{model}_seed{seed}_fold{fold}.npz"
    if part.exists():
        return part
    import joblib
    t0 = time.time()
    rec = fold_record(ds, model, "gmkep56", seed, fold)["strategies"]["tuned"]
    est = joblib.load(ROOT / rec["model_path"])
    split = np.load(TUNED / ds / f"splits/seed{seed}_fold{fold}.npz")
    test = split["outer_test"]
    X = np.ascontiguousarray(np.load(TUNED / ds / "arrays/features.npy", mmap_mode="r")[test])
    y = np.load(TUNED / ds / "arrays/labels.npy")[test]
    threads = int(os.environ.get("SHAP_THREADS", "3"))
    if model == "xgboost":
        import xgboost as xgb
        booster = est.get_booster()
        booster.set_param({"nthread": threads})
        dm = xgb.DMatrix(X)
        contrib = booster.predict(dm, pred_contribs=True)
        margin = booster.predict(dm, output_margin=True)
    else:
        contrib = est.booster_.predict(X, pred_contrib=True, num_threads=threads)
        margin = est.booster_.predict(X, raw_score=True, num_threads=threads)
    add_err = float(np.max(np.abs(contrib.sum(1) - margin)))
    phi = np.abs(contrib[:, :56])
    part.parent.mkdir(exist_ok=True)
    np.savez(part, all=phi.mean(0), pos=phi[y == 1].mean(0), additivity_max_err=add_err,
             n_test=len(y), n_pos=int(y.sum()), seconds=time.time() - t0)
    print(f"shap part {part.name} {time.time() - t0:.1f}s add_err={add_err:.2e}", flush=True)
    return part


def run_shap():
    """Compute missing per-model parts (optionally filtered by SHAP_DS / SHAP_MODEL), then aggregate if complete."""
    names = json.loads((TUNED / "protocol.json").read_text())["feature_columns"]["gmkep56"]
    jobs = [(ds, m, s, f) for ds in DATASETS for m in ["lightgbm", "xgboost"] for s in SEEDS for f in FOLDS]
    want = [j for j in jobs if os.environ.get("SHAP_DS", j[0]) == j[0] and os.environ.get("SHAP_MODEL", j[1]) == j[1]]
    for j in want:
        _shap_one(j)
    parts = [OUT / "shap_parts" / f"{ds}_{m}_seed{s}_fold{f}.npz" for ds, m, s, f in jobs]
    if not all(p.exists() for p in parts):
        return {"partial": True, "parts_done": sum(p.exists() for p in parts), "parts_total": len(parts)}
    out = []
    for (ds, m, s, f), p in zip(jobs, parts):
        z = np.load(p)
        out.append({"dataset": ds, "model": m, "seed": s, "fold": f, "all": z["all"], "pos": z["pos"],
                    "additivity_max_err": float(z["additivity_max_err"]), "seconds": float(z["seconds"])})
    feat_rows, mod_rows = [], []
    for r in out:
        for subset in ["all", "pos"]:
            v = r[subset]
            for i, n in enumerate(names):
                feat_rows.append({"dataset": r["dataset"], "model": r["model"], "seed": r["seed"], "fold": r["fold"],
                                  "subset": subset, "feature": n, "mean_abs_shap": v[i]})
            tot = v.sum()
            for mod, idx in MODULES.items():
                mod_rows.append({"dataset": r["dataset"], "model": r["model"], "seed": r["seed"], "fold": r["fold"],
                                 "subset": subset, "module": mod, "mean_abs_shap": v[list(idx)].sum(),
                                 "share": v[list(idx)].sum() / tot})
    pd.DataFrame(feat_rows).to_csv(OUT / "shap_feature_fold_values.csv", index=False)
    mods = pd.DataFrame(mod_rows)
    mods.to_csv(OUT / "shap_module_fold_values.csv", index=False)
    mods.groupby(["dataset", "model", "subset", "module"]).agg(
        share_mean=("share", "mean"), share_sd=("share", lambda x: x.std(ddof=1))).reset_index().to_csv(OUT / "shap_module_summary.csv", index=False)
    pd.DataFrame(feat_rows).groupby(["dataset", "model", "subset", "feature"]).mean_abs_shap.mean().reset_index().to_csv(OUT / "shap_feature_summary.csv", index=False)
    return {"models": len(out), "max_additivity_error": max(r["additivity_max_err"] for r in out),
            "mean_seconds_per_model": float(np.mean([r["seconds"] for r in out])),
            "note": "LightGBM/XGBoost TreeSHAP in raw-score (log-odds) units; random forest not computed"}


# ---------------------------------------------------------------- timing
def run_timing():
    """Per-model timing (set TIMING_MODEL); LightGBM and XGBoost must not share a process (OpenMP conflict)."""
    import joblib
    models = [os.environ["TIMING_MODEL"]] if os.environ.get("TIMING_MODEL") else MODELS
    for model in models:
        part = OUT / f"timing_fold_values_{model}.csv"
        if part.exists():
            continue
        rows = []
        for ds in DATASETS:
            X = np.load(TUNED / ds / "arrays/features.npy", mmap_mode="r")
            for fs in FSETS:
                ncol = {"base16": 16, "gmke44": 44, "gmkep56": 56}[fs]
                for seed in SEEDS:
                    for fold in FOLDS:
                        j = fold_record(ds, model, fs, seed, fold)
                        tr = j["strategies"]["tuned"]
                        # when the default configuration is selected, the tuned model is the anchor model
                        # (its fit times are recorded as 0), so the anchor's pure fit times are used
                        fit_src = j["strategies"]["same_protocol_anchor"] if tr["physical_reuse_anchor"] else tr
                        sr = json.loads((ROOT / j["search_result_path"]).read_text())
                        row = {"dataset": ds, "model": model, "feature_set": fs, "seed": seed, "fold": fold,
                               "outer_fold_runtime_s": j["runtime_seconds"],
                               "search_fit_s": sr["search_fit_seconds"], "search_physical_fits": sr["physical_search_fits"],
                               "final_model_fit_s": fit_src["final_model_fit_seconds"],
                               "threshold_model_fit_s": fit_src["threshold_model_fit_seconds"],
                               "tuned_reuses_anchor": bool(tr["physical_reuse_anchor"]),
                               "selected_rounds": tr["selected_rounds"], "inference_s": np.nan, "n_test": np.nan}
                        if seed == 0:  # measure inference on seed-0 folds only
                            est = joblib.load(ROOT / tr["model_path"])
                            test = np.load(TUNED / ds / f"splits/seed{seed}_fold{fold}.npz")["outer_test"]
                            Xt = np.ascontiguousarray(X[test][:, :ncol])
                            if model == "lightgbm":
                                t0 = time.perf_counter()
                                est.booster_.predict(Xt, num_threads=2)
                            elif model == "xgboost":
                                est.set_params(n_jobs=2)
                                t0 = time.perf_counter()
                                est.predict_proba(Xt)
                            else:
                                est.set_params(n_jobs=2)
                                t0 = time.perf_counter()
                                est.predict_proba(Xt)
                            row["inference_s"], row["n_test"] = time.perf_counter() - t0, len(test)
                            print(f"timing {model} {ds} {fs} fold{fold} {row['inference_s']:.2f}s", flush=True)
                        rows.append(row)
        pd.DataFrame(rows).to_csv(part, index=False)
    parts = [OUT / f"timing_fold_values_{m}.csv" for m in MODELS]
    if not all(p.exists() for p in parts):
        return {"partial": True, "parts_done": sum(p.exists() for p in parts)}
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    df.to_csv(OUT / "timing_fold_values.csv", index=False)
    df.groupby(["dataset", "model", "feature_set"]).agg(
        outer_fold_runtime_median_s=("outer_fold_runtime_s", "median"), search_fit_s=("search_fit_s", "mean"),
        search_physical_fits=("search_physical_fits", "mean"), final_model_fit_s=("final_model_fit_s", "mean"),
        tuned_reuses_anchor=("tuned_reuses_anchor", "sum"), selected_rounds=("selected_rounds", "mean"),
        inference_s=("inference_s", "mean"), n_test=("n_test", "mean")).reset_index().to_csv(OUT / "timing_summary.csv", index=False)
    return {"rows": len(df), "note": "pure fit times (search_fit_seconds, final_model_fit_seconds; anchor values when the tuned model reuses it) from frozen fold records (6 concurrent workers x 2 threads); outer-fold wall clock as median because two random-forest searches were resumed; inference measured with 2 threads on seed-0 folds, one model family per process"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    parts = {"stats": run_stats, "budget": run_budget, "patterns": run_patterns, "shap": run_shap, "timing": run_timing}
    todo = list(parts) if sys.argv[1] == "all" else sys.argv[1:]
    log_path = OUT / "run_log.json"
    for name in todo:
        t0 = time.time()
        info = parts[name]()
        info["seconds"] = round(time.time() - t0, 1)
        info["script_sha256"] = sha(__file__)
        if info.get("partial"):
            print(name, json.dumps(info), flush=True)
            continue
        log = json.loads(log_path.read_text()) if log_path.exists() else {}
        log[name] = info
        log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False, default=float))
        print(name, json.dumps(info, ensure_ascii=False, default=float)[:600], flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    main()
