"""Audit packaged files and reaggregate the manuscript's saved results (stdlib only).

Run from any working directory. No raw data, network, model fitting, or inference
is used. Outputs are written to a new directory specified by --out.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "Altman_experiments/results"
EXPECTED_FOLDS = {(s, f) for s in range(5) for f in range(5)}
METRICS = ("ap", "auc", "f1", "precision", "recall", "balanced_accuracy")
CORE_ARMS = ("base16", "base_g32", "gmke44", "gmkep56")
TREE_MODELS = ("lightgbm", "xgboost", "random_forest")
PACKS = (
    "account_gmkep56_tuned_20261008",
    "account_eswa_supplement_20261013/validation/tree/complete_settings",
    "account_eswa_remote_completion_20261020/final/complete_settings",
    "account_eswa_remaining_20261029/final/complete_settings",
    "account_mlp_cuda_paired_20261022/finalization_20261023/validation/complete_settings",
    "account_graphsage_cuda_optimized_20261018/final_results_20261020/validation/complete_settings",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def check(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b, message):
    check(math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-12), message)


def write_csv(path, records):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def integrity():
    manifest = json.loads((ROOT / "MANIFEST.json").read_text())
    for entry in manifest["files"]:
        path = ROOT / entry["path"]
        check(path.is_file() and path.stat().st_size == entry["bytes"], "Missing/size mismatch: " + entry["path"])
        check(sha(path) == entry["sha256"], "SHA mismatch: " + entry["path"])
    # Parse scripts without importing heavyweight libraries or executing legacy jobs.
    scripts = list(ROOT.rglob("*.py"))
    for path in scripts:
        ast.parse(path.read_text(), filename=str(path.relative_to(ROOT)))
    return dict(files_checked=len(manifest["files"]), python_sources_parsed=len(scripts))


def model_metrics(out):
    groups = defaultdict(dict)
    reference = {}
    for name in PACKS:
        folder = RESULTS / name
        prefix = "" if name == PACKS[0] else "verified_"
        for r in rows(folder / (prefix + "summary.csv")):
            if r["strategy"] == "tuned" and r["threshold_policy"] == "inner_max_f1":
                key = tuple(r[k] for k in ("dataset", "model", "feature_set"))
                if key in reference:
                    for metric in METRICS:
                        close(reference[key][metric + "_mean"], r[metric + "_mean"], "Duplicate summary changed")
                reference[key] = r
        for r in rows(folder / (prefix + "fold_metrics.csv")):
            if r["strategy"] != "tuned" or r["threshold_policy"] != "inner_max_f1":
                continue
            key = tuple(r[k] for k in ("dataset", "model", "feature_set"))
            fold = int(r["seed"]), int(r["fold"])
            if fold in groups[key]:
                for metric in METRICS:
                    close(groups[key][fold][metric], r[metric], "Duplicate fold changed")
            groups[key][fold] = r
    records = []
    for key, folds in sorted(groups.items()):
        check(set(folds) == EXPECTED_FOLDS, "Incomplete setting: " + str(key))
        row = dict(zip(("dataset", "model", "feature_set"), key), n_folds=25)
        for metric in METRICS:
            values = [float(r[metric]) for r in folds.values()]
            row[metric + "_mean"] = statistics.mean(values)
            row[metric + "_std"] = statistics.stdev(values)
            for suffix in ("mean", "std"):
                close(row[metric + "_" + suffix], reference[key][metric + "_" + suffix], str(key) + metric + suffix)
        records.append(row)
    expected = {(d, m, a) for d in ("hi_small", "li_small") for m in (*TREE_MODELS, "mlp", "graphsage_weak") for a in CORE_ARMS}
    expected |= {(d, m, "base_gfp_account_stream") for d in ("hi_small", "li_small") for m in TREE_MODELS}
    check(len(expected) == 46 and expected <= set(groups), "Missing manuscript method/input setting")
    write_csv(out / "all_tuned_model_summary.csv", records)
    write_csv(out / "manuscript_46_setting_summary.csv", [r for r in records if tuple(r[k] for k in ("dataset", "model", "feature_set")) in expected])
    return dict(complete_accepted_settings=len(groups), manuscript_settings=46, matched_outer_folds=sum(len(x) for x in groups.values()))


def budgets(out):
    original = RESULTS / "account_unified_alert_budget_20261028/final"
    extension = RESULTS / "account_gfp_alert_budget_extension_20261106"
    folded = rows(original / "budget_fold_values.csv") + rows(extension / "budget_fold_values.csv")
    groups = defaultdict(list)
    for r in folded:
        key = tuple(r[k] for k in ("dataset", "model", "feature_set")) + (float(r["budget"]),)
        groups[key].append(r)
        close(r["precision_at_k"], int(r["tp"]) / int(r["k"]), "Precision/count mismatch")
        close(r["recall_at_k"], int(r["tp"]) / int(r["n_positive"]), "Recall/count mismatch")
    accepted = {tuple(r[k] for k in ("dataset", "model", "feature_set")) + (float(r["budget"]),): r
                for r in rows(extension / "combined_budget_summary.csv")}
    check(len(groups) == len(accepted) == 138 and len(folded) == 3450, "Expected 46 settings x three budgets x 25 folds")
    records = []
    for key, values in sorted(groups.items()):
        check({(int(r["seed"]), int(r["fold"])) for r in values} == EXPECTED_FOLDS and len(values) == 25, "Incomplete budget group")
        row = dict(zip(("dataset", "model", "feature_set", "budget"), key))
        for metric in ("precision", "recall"):
            x = [float(r[metric + "_at_k"]) for r in values]
            row[metric + "_mean"] = statistics.mean(x)
            row[metric + "_sd"] = statistics.stdev(x)
            for suffix in ("mean", "sd"):
                close(row[metric + "_" + suffix], accepted[key][metric + "_" + suffix], "Budget aggregation mismatch")
        records.append(row)
    write_csv(out / "budget_summary.csv", records)
    return dict(budget_setting_rows=138, budget_fold_rows=3450)


def paper_coverage():
    inventory = json.loads((ROOT / "paper/table_inventory.json").read_text())
    check(len(inventory) == 22, "Expected current 22 tables")
    labels = {r["label"]: (ROOT / r["path"]).read_text() for r in inventory}
    threshold = [line for line in labels["tab:threshold"].splitlines() if line.startswith(("HI-Small &", "LI-Small &"))]
    check(len(threshold) == 24 and sum("& Base+G &" in line for line in threshold) == 6, "C.1 lacks Base+G update")
    check("Add N &" in labels["tab:controls"] and "Add Q &" in labels["tab:controls"], "Table 5B lacks controls")
    timing = labels["tab:timing"]
    check(all("GFP+" + model in timing for model in ("LightGBM", "XGBoost", "Random forest")), "Missing GFP timings")
    return dict(tables=22, threshold_rows=24, new_ph_control_rows=2, gfp_timing_classifiers=3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="new output directory")
    args = parser.parse_args()
    check(not args.out.exists(), "Choose a new output directory; existing results are never overwritten")
    checks = integrity()
    args.out.mkdir(parents=True)
    checks.update(model_metrics(args.out))
    checks.update(budgets(args.out))
    checks.update(paper_coverage())
    checks["passed"] = True
    checks["scope"] = "Integrity and saved-result reaggregation; not independent raw-score/model verification"
    (args.out / "audit.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
