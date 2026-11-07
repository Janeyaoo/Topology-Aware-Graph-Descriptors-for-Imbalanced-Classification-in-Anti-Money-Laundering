"""Versioned independent verification of matched IBM Small tuning.

Analysis revision 20261009 qualifies physical fit identity by dataset. The
registered training protocol and all original eight analysis scripts remain
unchanged. This verifier binds the versioned recovery sources and snapshots.

Search audit ranks are decoded without importing the trainer. Every candidate's
AP and AUROC are independently reconstructed; only explicitly retained search
models and calibration/final models are reloaded and scored. Discarded search
models are not described as independently rescored.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime
import gc
import hashlib
import itertools
import json
from pathlib import Path
import traceback
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score,
    balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parents[2]
ALT = ROOT / "Altman_experiments"
DEFAULT_OUT = ALT / "results/account_gmkep56_tuned_20261008"
OLD_FIXED = ALT / "results/account_gmkep56_multiclassifier_20261008"
FEATURE_SOURCE = ALT / "results/account_s10_increment_20261008"
TRAINER = ALT / "code/run_account_gmkep_tuning.py"
TZ = ZoneInfo("Asia/Shanghai")
TOL = 1e-12
DATASETS = ("hi_small", "li_small")
MODELS = ("lightgbm", "xgboost", "random_forest")
POLICIES = ("fixed_0.5", "inner_max_f1")
BASE = ["n_sent", "vol_sent", "avg_sent", "std_sent", "max_sent", "n_recv_banks",
    "n_recv", "vol_recv", "avg_recv", "std_recv", "max_recv", "n_sent_banks",
    "n_currencies", "n_formats", "sent_recv_ratio", "degree"]
G = ["A1", "A2", "D1", "D2", "nv", "ev", "density", "cc_minus", "bridge_frag",
    "tri", "clust", "beta1", "beta1_norm", "core", "core_mean", "core_max"]
GMKE = BASE + G + ["rec_out_count", "rec_in_count", "rec_out_ratio", "rec_in_ratio", "peel_score"] + \
    ["kappa", "kappa_mean", "kappa_min"] + ["asym", "asym2", "exp_out", "d_beta1"]
P12 = [f"ph_{hop}_{q}_{stat}" for hop in ("2hop", "3hop") for q in ("h0", "h1")
       for stat in ("max", "total", "entropy")]
ARMS = {"base16": BASE, "gmke44": GMKE, "gmkep56": GMKE + P12}
METRICS = ["precision", "recall", "f1", "auc", "ap", "accuracy", "balanced_accuracy",
           "specificity", "fpr", "fnr", "tn", "fp", "fn", "tp"]
HASH_CACHE = {}
STRATEGIES = ("same_protocol_anchor", "tuned")
FOLD_KEYS = ["dataset", "model", "feature_set", "strategy", "seed", "fold", "threshold_policy"]
SEED_KEYS = ["dataset", "model", "feature_set", "strategy", "seed", "threshold_policy"]
SUMMARY_KEYS = ["dataset", "model", "feature_set", "strategy", "threshold_policy"]
FEATURE_GROUP = ["dataset", "model", "strategy", "contrast_id", "threshold_policy"]
STRATEGY_GROUP = ["dataset", "model", "feature_set", "contrast_id", "threshold_policy"]
FEATURE_CONTRASTS = (
    ("gmke_vs_base", "gmke44", "base16", "structural"),
    ("ph_vs_gmke", "gmkep56", "gmke44", "barcode"),
    ("full_vs_base", "gmkep56", "base16", "full_dictionary"),
)
STRATEGY_CONTRASTS = (
    ("tuned_vs_anchor", "tuned", "same_protocol_anchor"),
    ("tuned_vs_old", "tuned", "original_fixed"),
    ("anchor_vs_old", "same_protocol_anchor", "original_fixed"),
)


def sha256(path):
    path = Path(path).resolve()
    stat = path.stat()
    signature = (stat.st_size, stat.st_mtime_ns)
    cached = HASH_CACHE.get(path)
    if cached and cached[0] == signature:
        return cached[1]
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    result = digest.hexdigest()
    HASH_CACHE[path] = signature, result
    return result


def array_sha256(value, dtype):
    return hashlib.sha256(np.ascontiguousarray(value, dtype=dtype).tobytes()).hexdigest()


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def check(report, name, condition):
    report["checks"][name] = bool(condition)
    if not condition:
        raise ValueError(f"Independent tuning verification failed: {name}")


def close(report, name, actual, expected, tolerance=TOL):
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    check(report, name, a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
          and np.allclose(a, b, atol=tolerance, rtol=0))
    return float(np.max(np.abs(a - b), initial=0))


def contained(base, name):
    path = (Path(base) / name).resolve()
    if not path.is_relative_to(Path(base).resolve()) or not path.is_file():
        raise ValueError(f"Missing or uncontained registered artifact: {name}")
    return path


def bind(report, out, path):
    key = Path(path).resolve().relative_to(Path(out).resolve()).as_posix()
    digest = sha256(path)
    check(report, "unchanged_" + key, report["result_hashes"].get(key, digest) == digest)
    report["result_hashes"][key] = digest


def source_bind(report, path, expected=None):
    key, digest = rel(path), sha256(path)
    check(report, "source_unchanged_" + key, report["source_hashes"].get(key, digest) == digest
          and (expected is None or expected == digest))
    report["source_hashes"][key] = digest


def threshold_from_pr_curve(y, score):
    y, score = np.asarray(y), np.asarray(score, dtype=float)
    if len(y) != len(score) or not len(y) or not np.isfinite(score).all() or set(np.unique(y)) != {0, 1}:
        raise ValueError("Invalid threshold truth or probabilities")
    minimum_gap = 2. / float(len(y) + np.count_nonzero(y)) ** 2
    if minimum_gap <= TOL + 16 * np.finfo(float).eps:
        raise ValueError("Threshold sample size exceeds the certified F1 tie tolerance")
    precision, recall, thresholds = precision_recall_curve(y, score, drop_intermediate=False)
    p, r = precision[:-1], recall[:-1]
    f = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=p + r > 0)
    index = int(np.flatnonzero(np.abs(f - f.max()) <= TOL)[0])
    return float(thresholds[index]), float(f[index])


def metric_values(y, score, predicted):
    y, score, predicted = np.asarray(y), np.asarray(score, dtype=float), np.asarray(predicted)
    if set(np.unique(y)) != {0, 1} or not np.isfinite(score).all() or not np.isin(predicted, [0, 1]).all():
        raise ValueError("Invalid scored labels, probabilities or predictions")
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return dict(precision=float(precision_score(y, predicted, zero_division=0)),
        recall=float(recall_score(y, predicted, zero_division=0)),
        f1=float(f1_score(y, predicted, zero_division=0)), auc=float(roc_auc_score(y, score)),
        ap=float(average_precision_score(y, score)), accuracy=float(accuracy_score(y, predicted)),
        balanced_accuracy=float(balanced_accuracy_score(y, predicted)), specificity=float(tn / (tn + fp)),
        fpr=float(fp / (tn + fp)), fnr=float(fn / (tp + fn)),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def unpack_verified(packed, length):
    """Decode little-endian bits while rejecting nonzero padding and bad widths."""
    value = np.asarray(packed)
    if value.dtype != np.uint8 or value.ndim != 1 or len(value) != (length + 7) // 8:
        raise ValueError("Invalid packed bit array shape or dtype")
    bits = np.unpackbits(value, bitorder="little")
    if bits[length:].any():
        raise ValueError("Nonzero packed audit padding")
    return bits[:length]


def rank_audit_values(audit, expected_y):
    """Rebuild exact tie-aware ranking metrics from an explicitly verified order.

    The saved candidate probability magnitudes are not recoverable from this
    audit. This independently verifies ranking metrics and truth/row binding,
    without claiming independent prediction of discarded candidate models.
    """
    n = int(np.asarray(audit["n"]).item())
    n_positive = int(np.asarray(audit["n_positive"]).item())
    width = int(np.asarray(audit["width"]).item())
    expected_y = np.asarray(expected_y, dtype=np.int8)
    if n != len(expected_y) or n <= 1 or n_positive != int(expected_y.sum()) or not 0 < n_positive < n:
        raise ValueError("Rank audit sample/positive count mismatch")
    if width != max(1, int(np.ceil(np.log2(n)))):
        raise ValueError("Invalid permutation bit width")
    bits = unpack_verified(audit["packed_order"], n * width).reshape(n, width)
    order = np.sum(bits.astype(np.uint64) << np.arange(width, dtype=np.uint64), axis=1).astype(np.int64)
    if not np.array_equal(np.sort(order), np.arange(n)):
        raise ValueError("Rank audit is not a full permutation")
    labels = unpack_verified(audit["packed_labels"], n)
    ends = unpack_verified(audit["packed_tie_end"], n)
    if not np.array_equal(labels, expected_y[order]) or ends[-1] != 1:
        raise ValueError("Rank audit sorted truth or tie boundary mismatch")
    if np.any((ends[:-1] == 0) & (order[:-1] >= order[1:])):
        raise ValueError("Rank audit violates stable ordering inside score ties")
    at = np.flatnonzero(ends)
    tp = np.cumsum(labels, dtype=np.int64)[at]
    group_p = np.diff(np.r_[0, tp])
    total = at + 1
    group_n = np.diff(np.r_[0, total]) - group_p
    ap = float(np.sum(group_p * (tp / total)) / n_positive)
    # Descending tie blocks: each positive outranks all negative observations
    # below the block and receives half credit for negatives inside its block.
    total_negative = n - n_positive
    negative_above = np.cumsum(group_n) - group_n
    negatives_below = total_negative - negative_above - group_n
    auc = float(np.sum(group_p * (negatives_below + .5 * group_n)) / (n_positive * total_negative))
    return dict(ap=ap, auc=auc, order=order, sorted_labels=labels, tie_end=ends)


def compare_table(report, out, filename, expected, keys):
    path = out / filename
    actual = pd.read_csv(path)
    check(report, filename + "_schema_and_unique_keys", set(actual) == set(expected)
          and not actual.duplicated(keys).any() and not expected.duplicated(keys).any())
    a, b = actual.set_index(keys).sort_index(), expected.set_index(keys).sort_index()
    check(report, filename + "_complete_keys", a.index.equals(b.index))
    maximum = 0.
    for name in b:
        if pd.api.types.is_numeric_dtype(b[name]):
            maximum = max(maximum, close(report, filename + "_" + name, a[name], b[name]))
        else:
            check(report, filename + "_" + name, a[name].fillna("<NULL>").equals(b[name].fillna("<NULL>")))
    report["table_max_abs_diff"][filename] = maximum
    bind(report, out, path)


def skeleton(dataset, model=None):
    return {"dataset": dataset, "model": model, "passed": False, "status": "failed",
        "validated_at": datetime.now(TZ).isoformat(timespec="seconds"), "checks": {},
        "validator": rel(__file__), "validator_sha256": sha256(__file__),
        "tolerance": TOL, "source_hashes": {rel(__file__): sha256(__file__)}, "result_hashes": {},
        "model_probability_max_abs_diff": {}, "metric_max_abs_diff": 0.,
        "threshold_max_abs_diff": 0., "table_max_abs_diff": {},
        "candidate_rank_audits_verified": 0, "retained_search_models_rescored": 0,
        "calibration_models_rescored": 0, "final_models_rescored": 0,
        "discarded_search_model_rescoring_claimed": False}


def independent_summary(frame, keys, paired=False):
    rows = []
    for key, group in frame.groupby(keys, sort=True):
        if len(group) != 25:
            raise ValueError("Incomplete 25-fold summary group")
        row = dict(zip(keys, key))
        row["n_pairs" if paired else "n_folds"] = len(group)
        for metric in METRICS:
            column = "delta_" + metric if paired else metric
            values = group[column]
            row[column + "_mean"] = float(values.mean())
            row[column + "_std"] = float(values.std(ddof=1))
            if paired:
                row[metric + "_positive_pairs"] = int((values > 0).sum())
        if not paired:
            row["threshold_mean"] = float(group.threshold.mean())
            row["threshold_std"] = float(group.threshold.std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def paired_values(left, right, row_keys, metadata):
    left = left.set_index(row_keys).sort_index()
    right = right.set_index(row_keys).sort_index()
    if not left.index.equals(right.index):
        raise ValueError("Incomplete or unpaired empirical contrast")
    output = left.reset_index()[row_keys].copy()
    for name, value in metadata.items():
        output[name] = value
    for metric in METRICS:
        a, b = left[metric].to_numpy(), right[metric].to_numpy()
        output["lhs_" + metric] = a
        output["rhs_" + metric] = b
        output["delta_" + metric] = a - b
    return output


def feature_pairs(frame, row_keys):
    rows = []
    for dataset, model, strategy, policy in itertools.product(DATASETS, MODELS, STRATEGIES, POLICIES):
        selected = frame[(frame.dataset == dataset) & (frame.model == model)
            & (frame.strategy == strategy) & (frame.threshold_policy == policy)]
        for cid, lhs, rhs, kind in FEATURE_CONTRASTS:
            rows.append(paired_values(selected[selected.feature_set == lhs], selected[selected.feature_set == rhs],
                row_keys, dict(dataset=dataset, model=model, strategy=strategy, threshold_policy=policy,
                    contrast_id=cid, lhs_arm=lhs, rhs_arm=rhs, contrast_type=kind)))
    return pd.concat(rows, ignore_index=True)


def strategy_pairs(frame, old, row_keys):
    rows = []
    for dataset, model, arm, policy in itertools.product(DATASETS, MODELS, ARMS, POLICIES):
        masks = lambda table: ((table.dataset == dataset) & (table.model == model)
            & (table.feature_set == arm) & (table.threshold_policy == policy))
        selected, previous = frame[masks(frame)], old[masks(old)]
        choices = {strategy: selected[selected.strategy == strategy] for strategy in STRATEGIES}
        choices["original_fixed"] = previous
        for cid, lhs, rhs in STRATEGY_CONTRASTS:
            rows.append(paired_values(choices[lhs], choices[rhs], row_keys,
                dict(dataset=dataset, model=model, feature_set=arm, threshold_policy=policy,
                    contrast_id=cid, lhs_strategy=lhs, rhs_strategy=rhs)))
    return pd.concat(rows, ignore_index=True)


def aggregate_scientific_tables(report, out, folds, seeds, old_folds, old_seeds):
    expected_fold_keys = set(itertools.product(DATASETS, MODELS, ARMS, STRATEGIES, range(5), range(5), POLICIES))
    expected_seed_keys = set(itertools.product(DATASETS, MODELS, ARMS, STRATEGIES, range(5), POLICIES))
    check(report, "complete_new_fold_keys_1800", len(folds) == 1800 and not folds.duplicated(FOLD_KEYS).any()
        and set(folds[FOLD_KEYS].itertuples(index=False, name=None)) == expected_fold_keys)
    check(report, "complete_new_seed_keys_360", len(seeds) == 360 and not seeds.duplicated(SEED_KEYS).any()
        and set(seeds[SEED_KEYS].itertuples(index=False, name=None)) == expected_seed_keys)
    f_pairs = feature_pairs(folds, ["seed", "fold"])
    s_pairs = feature_pairs(seeds, ["seed"])
    strategy_folds = strategy_pairs(folds, old_folds, ["seed", "fold"])
    strategy_seeds = strategy_pairs(seeds, old_seeds, ["seed"])
    tables = {
        "summary.csv": (independent_summary(folds, SUMMARY_KEYS), SUMMARY_KEYS),
        "paired_feature_fold_differences.csv": (f_pairs, FEATURE_GROUP + ["seed", "fold"]),
        "paired_feature_summary.csv": (independent_summary(f_pairs, FEATURE_GROUP, True), FEATURE_GROUP),
        "paired_feature_seed_oof_differences.csv": (s_pairs, FEATURE_GROUP + ["seed"]),
        "paired_strategy_fold_differences.csv": (strategy_folds, STRATEGY_GROUP + ["seed", "fold"]),
        "paired_strategy_summary.csv": (independent_summary(strategy_folds, STRATEGY_GROUP, True), STRATEGY_GROUP),
        "paired_strategy_seed_oof_differences.csv": (strategy_seeds, STRATEGY_GROUP + ["seed"]),
    }
    check(report, "seven_independent_derived_table_counts", [len(value[0]) for value in tables.values()]
        == [72, 1800, 72, 360, 2700, 108, 540])
    for name, (expected, keys) in tables.items():
        compare_table(report, out, name, expected, keys)
    for table, extras in ((folds, ["seed", "fold"]), (seeds, ["seed"])):
        index = ["dataset", "model", "feature_set", "strategy", *extras]
        for metric in ("ap", "auc"):
            values = table.pivot(index=index, columns="threshold_policy", values=metric)
            close(report, str(len(table)) + "_threshold_independent_" + metric,
                values[POLICIES[0]], values[POLICIES[1]], 0)


def independent_candidates(old):
    """Regenerate the registered finite search without importing its trainer."""
    spaces = {
        "lightgbm": {"learning_rate": [.03, .05, .1], "num_leaves": [15, 31, 63, 127],
            "min_child_samples": [10, 20, 50, 100], "reg_alpha": [0., .1, 1.],
            "reg_lambda": [1., 5., 10.], "colsample_bytree": [.6, .8, 1.],
            "subsample": [.7, .8, 1.], "subsample_freq": [1], "n_estimators": [300, 500, 700]},
        "xgboost": {"learning_rate": [.03, .05, .1], "max_depth": [3, 4, 6, 8],
            "min_child_weight": [1, 5, 10], "reg_alpha": [0., .1, 1.],
            "reg_lambda": [1., 5., 10.], "colsample_bytree": [.6, .8, 1.],
            "subsample": [.7, .8, 1.], "n_estimators": [300, 500, 700]},
    }
    answer = {}
    for offset, name in enumerate(MODELS[:2]):
        baseline = {**old["model_params"][name], "n_jobs": 2}
        if name == "lightgbm":
            baseline.update(subsample_freq=0, deterministic=True, force_col_wise=True)
        values = [baseline]
        seen = {json.dumps(baseline, sort_keys=True)}
        rng = np.random.default_rng(20261008 + offset)
        while len(values) < 12:
            parameters = dict(baseline)
            for parameter, candidates in spaces[name].items():
                parameters[parameter] = candidates[int(rng.integers(len(candidates)))]
            signature = json.dumps(parameters, sort_keys=True)
            if signature not in seen:
                values.append(parameters)
                seen.add(signature)
        answer[name] = [{"candidate_id": index, "params": parameters}
                        for index, parameters in enumerate(values)]
    profiles = [{}, {"max_depth": 10, "min_samples_leaf": 10, "max_features": .5},
        {"max_depth": 20, "min_samples_leaf": 2}, {"max_depth": None, "min_samples_leaf": 10},
        {"max_depth": 14, "min_samples_leaf": 5, "max_features": .5, "max_samples": .6},
        {"max_depth": 20, "min_samples_leaf": 5, "max_features": .25, "max_samples": 1.}]
    values = []
    for index, update in enumerate(profiles):
        for count in ((200, 100) if index == 0 else (100, 200)):
            values.append(dict(candidate_id=len(values), profile_id=index,
                params={**old["model_params"]["random_forest"], "n_jobs": 2, **update, "n_estimators": count}))
    answer["random_forest"] = values
    return answer, spaces, profiles


def analysis_revision_audit(report, out):
    """Bind the immutable v2 recovery provenance without rebinding mutable logs."""
    directory = out / "analysis_revision_20261009"
    manifest_path = directory / "manifest.json"
    manifest = read(manifest_path)
    original_path = out / "analysis_code_manifest.json"
    original = read(original_path)
    expected_scripts = {
        "finalize_account_gmkep_tuning_v2.py",
        "verify_account_gmkep_tuning_v2.py",
        "complete_account_gmkep_tuning_v2.py",
    }
    check(report, "analysis_v2_revision_same_protocol_original_registration", bool(manifest["revision_id"])
        and bool(manifest["scope"]) and manifest["protocol_sha256"] == sha256(out / "protocol.json")
        and original["protocol_sha256"] == manifest["protocol_sha256"]
        and manifest["original_analysis_code_manifest_sha256"] == sha256(original_path)
        and manifest["original_scripts_sha256"] == original["scripts_sha256"]
        and len(original["scripts_sha256"]) == 8
        and set(manifest["scripts_sha256"]) == expected_scripts)
    source_bind(report, manifest_path)
    source_bind(report, original_path, manifest["original_analysis_code_manifest_sha256"])
    bind(report, out, manifest_path); bind(report, out, original_path)
    for name, digest in original["scripts_sha256"].items():
        check(report, "analysis_v2_keeps_original_source_and_snapshot_" + name,
            sha256(ALT / "code" / name) == digest == sha256(out / "analysis_code_snapshots" / name))
        source_bind(report, ALT / "code" / name, digest)
        source_bind(report, out / "analysis_code_snapshots" / name, digest)
    expected_snapshots = {rel(directory / "snapshots" / name): digest
                          for name, digest in manifest["scripts_sha256"].items()}
    audit_source = contained(ROOT, manifest["audit_source_path"])
    audit_snapshot = contained(ROOT, manifest["namespace_audit_snapshot_path"])
    audit_original = contained(ROOT, manifest["audit_original_path"])
    check(report, "analysis_v2_namespace_audit_source_and_snapshot", audit_source == (out / "namespace_audit.json").resolve()
        and audit_snapshot == (directory / "snapshots/namespace_audit.json").resolve()
        and audit_original == (out / "qa/physical_fit_namespace_audit_20261009.json").resolve()
        and sha256(audit_source) == manifest["namespace_audit_source_sha256"]
            == manifest["namespace_audit_snapshot_sha256"] == sha256(audit_snapshot)
            == manifest["namespace_audit_original_sha256"] == sha256(audit_original))
    expected_snapshots[rel(audit_snapshot)] = manifest["namespace_audit_snapshot_sha256"]
    check(report, "analysis_v2_exact_three_script_and_audit_snapshot_paths", manifest["snapshots_sha256"] == expected_snapshots)
    source_bind(report, audit_source, manifest["namespace_audit_source_sha256"])
    source_bind(report, audit_snapshot, manifest["namespace_audit_snapshot_sha256"])
    source_bind(report, audit_original, manifest["namespace_audit_original_sha256"])
    bind(report, out, audit_source); bind(report, out, audit_snapshot); bind(report, out, audit_original)
    for name, digest in manifest["scripts_sha256"].items():
        path = ALT / "code" / name
        snapshot = directory / "snapshots" / name
        check(report, "analysis_v2_source_and_snapshot_" + name, sha256(path) == digest == sha256(snapshot))
        source_bind(report, path, digest); source_bind(report, snapshot, digest)
        bind(report, out, snapshot)
    check(report, "analysis_v2_failed_attempt_snapshot_present", bool(manifest["failed_attempt_sha256"]))
    for name, digest in manifest["failed_attempt_sha256"].items():
        path = contained(ROOT, name)
        check(report, "analysis_v2_failed_attempt_snapshot_contained_" + name,
            path.is_relative_to((directory / "attempt_before").resolve()) and sha256(path) == digest)
        source_bind(report, path, digest); bind(report, out, path)
    # Original failure logs and run manifests may legitimately change as the
    # new completion run advances; only their immutable archived copies bind.
    report.update(analysis_revision_id=manifest["revision_id"],
        analysis_revision_manifest=rel(manifest_path), analysis_revision_manifest_sha256=sha256(manifest_path),
        original_analysis_code_manifest_sha256=sha256(original_path),
        physical_fit_identity_definition="(dataset, physical_fit_id); RF100/200 share their existing within-dataset profile identity",
        original_training_and_analysis_sources_preserved=True)


def protocol_audit(report, out):
    analysis_revision_audit(report, out)
    protocol = read(out / "protocol.json")
    expected = {
        "experiment": "account_gmkep56_nested_tuning", "datasets": list(DATASETS), "models": list(MODELS),
        "feature_sets": {name: len(columns) for name, columns in ARMS.items()}, "feature_columns": ARMS,
        "strategies": list(STRATEGIES), "seeds": list(range(5)), "outer_n_splits": 5,
        "inner_n_splits": 3, "candidates_per_model": 12,
        "selection_metric": "unweighted mean of three inner-fold AP values", "primary_metrics": ["ap"],
        "candidate_tie_break": "exact tie: lowest candidate_id", "threshold_policies": list(POLICIES),
        "prediction_rule": "y_score >= threshold", "summary_std_ddof": 1,
        "threshold_selection": "max positive-class F1 on independent threshold_validation; ties lowest threshold",
        "rounds_aggregation": "integer median of the selected candidate's three inner early-stopping rounds",
        "early_stopping": {"metric": "unweighted AUROC", "patience": 30, "validation_fraction": .2},
        "development_threshold_split": "reuse old inner_fit and inner_calibration, 80/20 of outer_train",
        "inner_random_state": "20000 + 100 * seed + fold",
        "early_random_state": "30000 + 1000 * seed + 10 * fold + inner_fold",
        "search_model_random_state": "40000 + 1000 * seed + 10 * fold + inner_fold; independent of candidate and arm",
        "stage_model_random_state": "100 * seed + fold",
        "class_weighting": "balanced from actual fitting labels; XGB sample_weight; eval unweighted",
        "threads_per_worker": 2, "max_concurrent_workers": 6, "native_library_isolation": True,
        "rf_tree_candidates": [100, 200], "expected_outer_settings": 450,
        "logical_candidate_evaluations": 16200, "expected_search_physical_fits": 13500,
        "expected_representative_refits": 450, "expected_strategy_final_models_logical": 900,
        "expected_strategy_threshold_models_logical": 900, "old_fixed_source": rel(OLD_FIXED),
        "audit_encoding": "little-endian packbits: order elements represented least-significant-bit first; width=max(1,ceil(log2(n)))",
    }
    check(report, "complete_prespecified_matched_search_protocol", all(protocol.get(key) == value for key, value in expected.items()))
    old_protocol = read(OLD_FIXED / "protocol.json")
    prior = read(OLD_FIXED / "validation_models.json")
    check(report, "old_fixed_independent_validation_passed", prior["passed"] is True
        and prior["checks"] and all(prior["checks"].values())
        and prior["protocol_sha256"] == protocol["old_protocol_sha256"] == sha256(OLD_FIXED / "protocol.json"))
    check(report, "old_fixed_same_dictionary_and_outer_seed_protocol", old_protocol["feature_columns"] == ARMS
        and old_protocol["seeds"] == list(range(5)) and old_protocol["outer_n_splits"] == 5)
    for name in ("protocol.json", "validation_models.json", "summary.csv", "fold_metrics.csv", "seed_oof_metrics.csv"):
        path = OLD_FIXED / name
        if name not in ("protocol.json", "validation_models.json"):
            check(report, "old_fixed_independent_binding_" + name, prior["result_hashes"][name] == sha256(path))
        source_bind(report, path)
    source_bind(report, contained(ROOT, prior["validator"]), prior["validator_sha256"])
    for name, digest in prior["source_hashes"].items():
        source_bind(report, contained(ROOT, name), digest)
    report["old_fixed_validation_sha256"] = sha256(OLD_FIXED / "validation_models.json")
    candidates, spaces, profiles = independent_candidates(old_protocol)
    check(report, "all_36_candidates_independently_reproduced", read(out / "candidates.json") == candidates
        and protocol["boost_search_spaces"] == spaces and protocol["rf_profiles"] == profiles
        and all(len({json.dumps(item["params"], sort_keys=True) for item in candidates[name]}) == 12 for name in MODELS))
    for filename in ("candidates.json", "dataset_inputs.json", "split_manifest.json"):
        check(report, "registered_" + filename, protocol[filename.removesuffix(".json") + "_sha256"] == sha256(out / filename))
        bind(report, out, out / filename)
    helper = ALT / "code/run_account_feature_mechanism.py"
    check(report, "frozen_trainer_and_helper", protocol["trainer_sha256"] == sha256(TRAINER)
        == sha256(out / "snapshots" / TRAINER.name) and protocol["helper_sha256"] == sha256(helper))
    source_bind(report, TRAINER); source_bind(report, helper)
    for name, digest in read(out / "snapshot_manifest.json").items():
        path = contained(ROOT, name)
        check(report, "snapshot_contained_" + name, path.is_relative_to((out / "snapshots").resolve()))
        check(report, "snapshot_registered_" + name, sha256(path) == digest)
        bind(report, out, path)
    bind(report, out, out / "snapshot_manifest.json")
    for name in MODELS:
        probe_path = out / "snapshots" / f"gmkep_tuning_probe_{name}.json"
        probe = read(probe_path)
        check(report, name + "_independent_small_data_roundtrip_probe", probe["passed"] is True
            and probe["trainer_sha256"] == protocol["trainer_sha256"] and probe["exact_repeat_scores"] is True
            and probe.get("model_serialization_roundtrip", False) is True
            and (name != "random_forest" or probe["rf_prefix_direct_equivalence"] is True))
        bind(report, out, probe_path)
    tree = ast.parse(TRAINER.read_text())
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    fitting = functions["fit"]
    fits = [node for node in ast.walk(fitting) if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute) and node.func.attr == "fit"]
    check(report, "source_fit_only_registered_rows_and_labels", len(fits) == 3
        and all([ast.unparse(arg) for arg in call.args] == ["X[rows]", "y[rows]"] for call in fits)
        and "compute_sample_weight('balanced', y[rows])" in ast.unparse(fitting)
        and "X[early]" in ast.unparse(fitting) and "y[early]" in ast.unparse(fitting)
        and "sample_weight_eval_set" not in ast.unparse(fitting)
        and "eval_sample_weight" not in ast.unparse(fitting))
    stages = [node for node in ast.walk(functions["strategy_fit"]) if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name) and node.func.id == "fit"]
    check(report, "source_threshold_fit_D_final_fit_outer_train_no_eval", len(stages) == 2
        and {tuple(ast.unparse(arg) for arg in call.args) for call in stages} == {
            ("name", "X", "y", "D", "None", "params", "random_state", "rounds"),
            ("name", "X", "y", "train", "None", "params", "random_state", "rounds")})
    source = ast.unparse(functions["search"])
    check(report, "source_search_no_outer_test_or_threshold_labels", "outer_test" not in source
        and "threshold_validation" not in source
        and "arrays[f'inner_train_{j}'] if name == 'random_forest' else arrays[f'early_fit_{j}']" in source
        and "arrays[f'inner_score_{j}']" in source
        and "arrays[f'early_validation_{j}']" in source)
    report.update(protocol_sha256=sha256(out / "protocol.json"), trainer_sha256=sha256(TRAINER),
        helper_sha256=sha256(helper), source_fit_scope_audit="Frozen AST and disjoint registered splits; balanced fit-only weights; unweighted search/early validation; threshold T; final outer_train without evaluation set")
    bind(report, out, out / "protocol.json")
    return protocol, candidates


def load_data(report, out, dataset):
    metadata = read(out / "dataset_inputs.json")[dataset]
    check(report, dataset + "_source_cache_path", metadata["source_features"] == rel(FEATURE_SOURCE / dataset / "features.parquet"))
    for name, digest in metadata["source_sha256"].items():
        source_bind(report, contained(ROOT, name), digest)
    for name, digest in metadata["cache_sha256"].items():
        path = contained(ROOT, name)
        check(report, dataset + "_cache_contained_" + name, path.is_relative_to((out / dataset).resolve()) and sha256(path) == digest)
        bind(report, out, path)
    matrix = np.load(out / dataset / "arrays/features.npy", mmap_mode="r")
    y = np.load(out / dataset / "arrays/labels.npy", mmap_mode="r")
    identity = pd.read_parquet(out / dataset / "accounts.parquet")
    table = pd.read_parquet(FEATURE_SOURCE / dataset / "features.parquet", columns=["account_id", "label", *ARMS["gmkep56"]])
    expected = (515088, 6357) if dataset == "hi_small" else (705907, 5304)
    check(report, dataset + "_raw_cache_full_counts_order_values", matrix.shape == (expected[0], 56)
        and matrix.dtype == np.float32 and np.isfinite(matrix).all() and y.dtype == np.int8
        and (len(y), int(y.sum())) == expected and metadata["n_samples"] == len(y)
        and metadata["n_positive"] == int(y.sum()) and set(np.unique(y)) == {0, 1}
        and np.array_equal(matrix, table[ARMS["gmkep56"]].to_numpy(np.float32))
        and np.array_equal(y, table.label.to_numpy(np.int8))
        and np.array_equal(identity.row_index, np.arange(len(y)))
        and np.array_equal(identity.account_id, table.account_id) and np.array_equal(identity.label, y))
    original = pd.read_parquet(ALT / "results/account_original_multiclassifier_20261007_isolated" / dataset / "accounts.parquet")
    check(report, dataset + "_old_fixed_identity_exact", identity.equals(original))
    split_manifest = read(out / "split_manifest.json")
    splits = {}; coverage = {seed: np.zeros(len(y), dtype=np.uint8) for seed in range(5)}
    for seed in range(5):
        for fold, (train, test) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)):
            a, b = next(StratifiedShuffleSplit(1, test_size=.2, random_state=10000 + 100 * seed + fold).split(np.zeros(len(train)), y[train]))
            arrays = dict(outer_train=train.astype(np.int32), outer_test=test.astype(np.int32),
                development=train[a].astype(np.int32), threshold_validation=train[b].astype(np.int32))
            D = arrays["development"]
            for j, (inner_train, inner_score) in enumerate(StratifiedKFold(3, shuffle=True,
                    random_state=20000 + 100 * seed + fold).split(np.zeros(len(D)), y[D])):
                fit, early = next(StratifiedShuffleSplit(1, test_size=.2,
                    random_state=30000 + 1000 * seed + 10 * fold + j).split(np.zeros(len(inner_train)), y[D[inner_train]]))
                arrays[f"inner_train_{j}"] = D[inner_train]
                arrays[f"inner_score_{j}"] = D[inner_score]
                arrays[f"early_fit_{j}"] = D[inner_train][fit]
                arrays[f"early_validation_{j}"] = D[inner_train][early]
            path = out / dataset / "splits" / f"seed{seed}_fold{fold}.npz"
            prior = OLD_FIXED / dataset / "splits" / path.name
            entry = split_manifest[rel(path)]
            check(report, f"{dataset}_split{seed}_{fold}_source_binding", entry["sha256"] == sha256(path)
                and entry["old_outer_split"] == rel(prior) and entry["old_outer_split_sha256"] == sha256(prior))
            with np.load(prior) as old:
                check(report, f"{dataset}_split{seed}_{fold}_old_D_T_preserved", np.array_equal(old["outer_train"], arrays["outer_train"])
                    and np.array_equal(old["outer_test"], arrays["outer_test"])
                    and np.array_equal(old["inner_fit"], arrays["development"])
                    and np.array_equal(old["inner_calibration"], arrays["threshold_validation"]))
            with np.load(path) as actual:
                check(report, f"{dataset}_split{seed}_{fold}_all_arrays_rebuilt", set(actual.files) == set(arrays)
                    and all(actual[key].dtype == np.int32 and np.array_equal(actual[key], value) for key, value in arrays.items()))
            check(report, f"{dataset}_split{seed}_{fold}_all_array_hashes", entry["arrays_sha256"]
                == {key: array_sha256(value, np.int32) for key, value in arrays.items()})
            train, test, D, T = [arrays[key] for key in ("outer_train", "outer_test", "development", "threshold_validation")]
            check(report, f"{dataset}_split{seed}_{fold}_outer_D_T_disjoint", not np.intersect1d(train, test).size
                and np.array_equal(np.sort(np.r_[train, test]), np.arange(len(y)))
                and not np.intersect1d(D, T).size and np.array_equal(np.sort(np.r_[D, T]), train))
            inner_coverage = np.zeros(len(y), dtype=np.uint8)
            for j in range(3):
                tr, sc, ft, ev = [arrays[f"{key}_{j}"] for key in ("inner_train", "inner_score", "early_fit", "early_validation")]
                check(report, f"{dataset}_split{seed}_{fold}_inner{j}_disjoint_complete", not np.intersect1d(tr, sc).size
                    and np.array_equal(np.sort(np.r_[tr, sc]), np.sort(D))
                    and not np.intersect1d(ft, ev).size and np.array_equal(np.sort(np.r_[ft, ev]), np.sort(tr))
                    and not np.intersect1d(sc, T).size and not np.intersect1d(sc, test).size
                    and not np.intersect1d(ft, test).size and not np.intersect1d(ev, test).size)
                inner_coverage[sc] += 1
            check(report, f"{dataset}_split{seed}_{fold}_inner_score_D_once", (inner_coverage[D] == 1).all()
                and inner_coverage.sum() == len(D))
            coverage[seed][test] += 1
            splits[seed, fold] = arrays
            bind(report, out, path); source_bind(report, prior, entry["old_outer_split_sha256"])
    check(report, dataset + "_each_seed_outer_test_once", all((counts == 1).all() for counts in coverage.values()))
    del table, original
    gc.collect()
    return matrix, y, identity, splits


def model_fingerprint(name, model):
    digest = hashlib.sha256()
    digest.update(json.dumps(model.get_params(), sort_keys=True, default=str).encode())
    if name == "lightgbm":
        digest.update(model.booster_.model_to_string(num_iteration=-1).encode())
    elif name == "xgboost":
        digest.update(bytes(model.get_booster().save_raw(raw_format="ubj")))
    else:
        for tree in model.estimators_:
            state = tree.tree_.__getstate__()
            for field in state["nodes"].dtype.names:
                digest.update(field.encode())
                digest.update(np.ascontiguousarray(state["nodes"][field]).tobytes())
            digest.update(np.ascontiguousarray(state["values"]).tobytes())
    return digest.hexdigest()


def rescore_model(report, out, name, path, X, rows, expected, parameters, state, rounds, stage,
                  expected_fingerprint, early=False, expected_raw_hash=None):
    key = path.relative_to(out).as_posix()
    if key in report.get("physical_model_hashes", {}):
        # Only selected candidate 0 can legitimately alias an anchor model.
        check(report, key + "_aliased_model_unchanged", report["physical_model_hashes"][key] == sha256(path))
        return
    model = joblib.load(path)
    params = model.get_params()
    effective = {**parameters, "random_state": state}
    if not early:
        effective["n_estimators"] = rounds
    if name == "xgboost" and early:
        effective["early_stopping_rounds"] = 30
    types = {"lightgbm": "LGBMClassifier", "xgboost": "XGBClassifier", "random_forest": "RandomForestClassifier"}
    check(report, key + "_class_all_effective_params_and_inputs", type(model).__name__ == types[name]
        and all(params.get(parameter) == value for parameter, value in effective.items())
        and model.n_features_in_ == X.shape[1] and np.array_equal(model.classes_, [0, 1]))
    if name == "lightgbm":
        actual_rounds = int(model.best_iteration_ or model.n_estimators)
        check(report, key + "_selected_rounds", actual_rounds == rounds
            and ((1 <= model.booster_.num_trees() <= parameters["n_estimators"]) if early else model.booster_.num_trees() == rounds))
    elif name == "xgboost":
        count = model.get_booster().num_boosted_rounds()
        check(report, key + "_selected_rounds", (int(model.best_iteration + 1) == rounds
            and rounds <= count <= parameters["n_estimators"]) if early else count == rounds)
        check(report, key + "_early_only_search", params.get("early_stopping_rounds") == (30 if early else None))
    else:
        check(report, key + "_forest_tree_count", len(model.estimators_) == model.n_estimators == rounds)
        rng = np.random.RandomState(state)
        tree_seeds = rng.randint(np.iinfo(np.int32).max, size=rounds)
        check(report, key + "_forest_bootstrap_tree_seeds", np.array_equal(tree_seeds,
            np.asarray([tree.random_state for tree in model.estimators_])))
    check(report, key + "_semantic_model_digest", model_fingerprint(name, model) == expected_fingerprint)
    jobs = model.n_jobs
    if name == "random_forest":
        model.n_jobs = 1
    score = np.ascontiguousarray(model.predict_proba(X[rows])[:, 1], dtype=np.float64)
    model.n_jobs = jobs
    report["model_probability_max_abs_diff"][key] = close(report, key + "_all_saved_probabilities", score, expected)
    if expected_raw_hash is not None:
        check(report, key + "_exact_raw_score_digest", array_sha256(score, np.float64) == expected_raw_hash)
    report.setdefault("physical_model_hashes", {})[key] = sha256(path)
    report[stage + "_models_rescored"] += 1
    bind(report, out, path)
    del model, score
    gc.collect()


def artifact_bindings(report, out, mapping, expected_paths=None):
    if expected_paths is not None:
        check(report, "exact_artifacts_" + str(len(report["checks"])), set(mapping) == {rel(path) for path in expected_paths})
    for name, digest in mapping.items():
        path = contained(ROOT, name)
        check(report, "artifact_contained_" + name, path.is_relative_to(out.resolve()) and sha256(path) == digest)
        bind(report, out, path)


def audit_search(report, out, dataset, name, arm, seed, fold, X, y, arrays, candidates):
    stem = f"{name}_{arm}_seed{seed}_fold{fold}"
    directory = out / dataset / "search" / stem
    path = directory / "search_result.json"
    saved = read(path)
    label = dataset + "_" + stem
    keys = ("dataset", "model", "feature_set", "seed", "fold")
    check(report, label + "_search_identity_complete_records", [saved[key] for key in keys] == [dataset, name, arm, seed, fold]
        and saved["protocol_sha256"] == report["protocol_sha256"]
        and len(saved["candidate_records"]) == 36
        and {(row["candidate_id"], row["inner_fold"]) for row in saved["candidate_records"]}
            == set(itertools.product(range(12), range(3))))
    records = []
    for row in saved["candidate_records"]:
        cid, j = row["candidate_id"], row["inner_fold"]
        candidate = candidates[name][cid]
        parameters = candidate["params"]
        fit_rows = arrays[f"inner_train_{j}"] if name == "random_forest" else arrays[f"early_fit_{j}"]
        early_rows = None if name == "random_forest" else arrays[f"early_validation_{j}"]
        score_rows = arrays[f"inner_score_{j}"]
        state = 40000 + 1000 * seed + 10 * fold + j
        audit_path = directory / f"candidate{cid}_inner{j}_rank.npz"
        prefix = label + f"_candidate{cid}_inner{j}"
        check(report, prefix + "_candidate_scope_params_state", [row[key] for key in keys] == [dataset, name, arm, seed, fold]
            and row["effective_params"] == {**parameters, "random_state": state}
            and row["model_random_state"] == state
            and row["fit_n"] == len(fit_rows) and row["fit_positive"] == int(y[fit_rows].sum())
            and row["score_n"] == len(score_rows) and row["score_positive"] == int(y[score_rows].sum())
            and row["fit_rows_sha256"] == array_sha256(fit_rows, np.int32)
            and row["early_rows_sha256"] == (None if early_rows is None else array_sha256(early_rows, np.int32))
            and row["score_rows_sha256"] == array_sha256(score_rows, np.int32)
            and row["audit_path"] == rel(audit_path) and row["audit_sha256"] == sha256(audit_path)
            and isinstance(row["selected_rounds"], int) and 1 <= row["selected_rounds"] <= parameters["n_estimators"]
            and (name != "random_forest" or row["selected_rounds"] == parameters["n_estimators"]))
        with np.load(audit_path, allow_pickle=False) as audit:
            check(report, prefix + "_rank_schema_and_hash_bindings", set(audit.files) == {
                "n", "n_positive", "width", "packed_order", "packed_labels", "packed_tie_end", "raw_score_sha256", "score_rows_sha256"}
                and str(audit["raw_score_sha256"].item()) == row["raw_score_sha256"]
                and str(audit["score_rows_sha256"].item()) == row["score_rows_sha256"])
            metrics = rank_audit_values(audit, y[score_rows])
        report["metric_max_abs_diff"] = max(report["metric_max_abs_diff"], close(report,
            prefix + "_independent_full_rank_AP_AUC", [row["ap"], row["auc"]], [metrics["ap"], metrics["auc"]]))
        group_id = candidate.get("profile_id", cid)
        charged = name != "random_forest" or parameters["n_estimators"] == 200
        check(report, prefix + "_physical_fit_identity_cost", row["physical_fit_id"] == f"{stem}_group{group_id}_inner{j}"
            and row["physical_fit_charged"] == charged
            and all(np.isfinite(row[field]) and row[field] >= 0 for field in ("fit_seconds", "shared_fit_seconds", "predict_seconds", "audit_seconds"))
            and row["fit_seconds"] == (row["shared_fit_seconds"] if charged else 0.))
        bind(report, out, audit_path)
        report["candidate_rank_audits_verified"] += 1
        records.append({**row, "best_iteration": row["selected_rounds"]})
    summaries = []
    for cid in range(12):
        rows = [row for row in records if row["candidate_id"] == cid]
        summaries.append(dict(candidate_id=cid, mean_ap=float(np.mean([row["ap"] for row in rows])),
            std_ap=float(np.std([row["ap"] for row in rows], ddof=1)), rounds=int(np.median([row["selected_rounds"] for row in rows]))))
    check(report, label + "_all_twelve_candidate_summary_ids", [row["candidate_id"] for row in saved["candidate_summary"]] == list(range(12)))
    for actual, expected in zip(saved["candidate_summary"], summaries):
        check(report, label + "_summary_candidate" + str(expected["candidate_id"]), actual["candidate_id"] == expected["candidate_id"])
        close(report, label + "_mean_std_rounds_candidate" + str(expected["candidate_id"]),
            [actual[key] for key in ("mean_ap", "std_ap", "rounds")], [expected[key] for key in ("mean_ap", "std_ap", "rounds")])
    selected = min(summaries, key=lambda row: (-row["mean_ap"], row["candidate_id"]))
    check(report, label + "_selection_mean_AP_then_lowest_id", saved["selected_candidate_id"] == selected["candidate_id"]
        and saved["selected_rounds"] == selected["rounds"] and saved["anchor_rounds"] == summaries[0]["rounds"]
        and saved["representative_inner_fold"] == 0)
    close(report, label + "_selected_mean_AP", saved["selected_mean_ap"], selected["mean_ap"])
    check(report, label + "_representative_and_invocation_time", all(np.isfinite(saved[field]) and saved[field] >= 0
        for field in ("representative_refit_seconds", "search_elapsed_seconds_this_invocation")))
    physical = 18 if name == "random_forest" else 36
    check(report, label + "_physical_fit_count", saved["physical_search_fits"] == physical
        and sum(row["physical_fit_charged"] for row in records) == physical)
    for saved_field, source_field in (("search_fit_seconds", "fit_seconds"),
            ("search_predict_seconds", "predict_seconds"), ("search_audit_seconds", "audit_seconds")):
        close(report, label + "_" + saved_field, saved[saved_field], sum(row[source_field] for row in records), tolerance=1e-8)
    cid = selected["candidate_id"]
    representative = next(row for row in records if row["candidate_id"] == cid and row["inner_fold"] == 0)
    model_path = out / dataset / "selected_search_models" / f"{stem}.joblib"
    score_path = out / dataset / "selected_search_scores" / f"{stem}.npz"
    check(report, label + "_representative_paths", saved["representative_model_path"] == rel(model_path)
        and saved["representative_score_path"] == rel(score_path))
    score_rows = arrays["inner_score_0"]
    with np.load(score_path, allow_pickle=False) as scores:
        check(report, label + "_representative_full_score_truth_order", set(scores.files) == {"row_index", "y_true", "y_score"}
            and np.array_equal(scores["row_index"], score_rows) and np.array_equal(scores["y_true"], y[score_rows])
            and scores["y_score"].dtype == np.float64 and np.isfinite(scores["y_score"]).all()
            and ((scores["y_score"] >= 0) & (scores["y_score"] <= 1)).all()
            and array_sha256(scores["y_score"], np.float64) == representative["raw_score_sha256"])
        probability = scores["y_score"].copy()
    with np.load(contained(ROOT, representative["audit_path"])) as audit:
        ranks = rank_audit_values(audit, y[score_rows])
        order = np.argsort(-probability, kind="mergesort")
        check(report, label + "_representative_rank_ties_exact", np.array_equal(order, ranks["order"])
            and np.array_equal(np.r_[probability[order][:-1] != probability[order][1:], True], ranks["tie_end"]))
    rescore_model(report, out, name, model_path, X, score_rows, probability, candidates[name][cid]["params"],
        40000 + 1000 * seed + 10 * fold, representative["selected_rounds"], "retained_search",
        representative["model_digest"], early=name != "random_forest", expected_raw_hash=representative["raw_score_sha256"])
    artifacts = [contained(ROOT, row["audit_path"]) for row in records] + [model_path, score_path]
    artifact_bindings(report, out, saved["artifact_sha256"], artifacts)
    for group in sorted({candidate.get("profile_id", candidate["candidate_id"]) for candidate in candidates[name]}):
        for j in range(3):
            node = directory / f"group{group}_inner{j}.json"
            value = read(node)
            expected_records = [row for row in saved["candidate_records"] if row["inner_fold"] == j
                and candidates[name][row["candidate_id"]].get("profile_id", row["candidate_id"]) == group]
            check(report, label + f"_group{group}_inner{j}_exact_resume_record", value["protocol_sha256"] == report["protocol_sha256"]
                and value["records"] == expected_records)
            bind(report, out, node)
    bind(report, out, path)
    return saved, records


def verify_fold(report, out, dataset, name, arm, seed, fold, X, y, identity, arrays, candidates):
    stem = f"{name}_{arm}_seed{seed}_fold{fold}"
    path = out / dataset / "folds" / f"{stem}.json"
    done = read(path)
    label = dataset + "_" + stem
    keys = ("dataset", "model", "feature_set", "seed", "fold")
    check(report, label + "_fold_identity_dictionary", [done[key] for key in keys] == [dataset, name, arm, seed, fold]
        and done["protocol_sha256"] == report["protocol_sha256"] and done["feature_columns"] == ARMS[arm]
        and done["n_features"] == len(ARMS[arm]) and set(done["strategies"]) == set(STRATEGIES)
        and done["logical_candidate_evaluations"] == 36 and done["logical_strategy_models"] == 4)
    search, candidate_rows = audit_search(report, out, dataset, name, arm, seed, fold, X, y, arrays, candidates)
    search_path = out / dataset / "search" / stem / "search_result.json"
    split_path = out / dataset / "splits" / f"seed{seed}_fold{fold}.npz"
    selected = search["selected_candidate_id"]
    check(report, label + "_fold_search_split_paths_and_counts", done["search_result_path"] == rel(search_path)
        and done["split_path"] == rel(split_path) and done["physical_search_fits"] == search["physical_search_fits"]
        and done["physical_strategy_models"] == (2 if selected == 0 else 4))
    train, test, D, T = [arrays[key] for key in ("outer_train", "outer_test", "development", "threshold_validation")]
    all_rows = []; artifact_paths = [split_path, search_path]
    selection = dict(dataset=dataset, model=name, feature_set=arm, seed=seed, fold=fold,
        candidate_id=selected, selected_rounds=search["selected_rounds"], anchor_rounds=search["anchor_rounds"],
        selected_mean_ap=search["selected_mean_ap"],
        selected_params_json=json.dumps(candidates[name][selected]["params"], sort_keys=True),
        anchor_params_json=json.dumps(candidates[name][0]["params"], sort_keys=True),
        physical_reuse_anchor=selected == 0, logical_candidate_evaluations=36,
        physical_search_fits=search["physical_search_fits"], physical_strategy_models=done["physical_strategy_models"],
        search_fit_seconds=search["search_fit_seconds"], search_predict_seconds=search["search_predict_seconds"],
        search_audit_seconds=search["search_audit_seconds"], representative_stage_seconds=search["representative_refit_seconds"],
        completion_invocation_seconds=done["runtime_seconds"], search_result_path=rel(search_path),
        fold_result_path=rel(path), split_path=rel(split_path))
    for strategy in STRATEGIES:
        saved = done["strategies"][strategy]
        cid = 0 if strategy == "same_protocol_anchor" else selected
        rounds = search["anchor_rounds"] if strategy == "same_protocol_anchor" else search["selected_rounds"]
        params = candidates[name][cid]["params"]
        state = 100 * seed + fold
        effective = {**params, "random_state": state, "n_estimators": rounds}
        actual_strategy = "same_protocol_anchor" if strategy == "tuned" and selected == 0 else strategy
        physical_stem = stem + "_" + actual_strategy
        cal_model = out / dataset / "calibration_models" / f"{physical_stem}.joblib"
        cal_path = out / dataset / "calibration" / f"{physical_stem}.parquet"
        final_model = out / dataset / "models" / f"{physical_stem}.joblib"
        pred_path = out / dataset / "predictions" / f"{physical_stem}.parquet"
        prefix = label + "_" + strategy
        check(report, prefix + "_selected_params_rounds_scope_and_paths", saved["strategy"] == strategy
            and saved["candidate_id"] == cid and saved["selected_rounds"] == rounds and saved["params"] == params
            and saved["model_random_state"] == state and saved["effective_params"] == effective
            and saved["final_fit_has_eval_set"] is False
            and saved["fit_rows_sha256"] == array_sha256(train, np.int32)
            and saved["calibration_fit_rows_sha256"] == array_sha256(D, np.int32)
            and saved["threshold_rows_sha256"] == array_sha256(T, np.int32)
            and saved["outer_test_rows_sha256"] == array_sha256(test, np.int32)
            and saved["calibration_model_path"] == rel(cal_model) and saved["calibration_path"] == rel(cal_path)
            and saved["model_path"] == rel(final_model) and saved["prediction_path"] == rel(pred_path)
            and saved["physical_reuse_anchor"] == (strategy == "tuned" and selected == 0))
        if strategy == "tuned" and selected == 0:
            anchor = done["strategies"]["same_protocol_anchor"]
            expected = dict(anchor)
            expected["strategy"] = "tuned"
            expected["physical_reuse_anchor"] = True
            for field in ("threshold_fit_seconds", "final_fit_seconds", "threshold_model_fit_seconds", "final_model_fit_seconds"):
                expected[field] = 0.
            expected["metrics"] = [{**row, "strategy": "tuned"} for row in anchor["metrics"]]
            check(report, prefix + "_exact_anchor_reference_and_zero_extra_cost", saved == expected)
        calibration, prediction = pd.read_parquet(cal_path), pd.read_parquet(pred_path)
        for frame, rows, role in ((calibration, T, "threshold_validation"), (prediction, test, "outer_test")):
            columns = {"row_index", "account_id", "y_true", "y_score"}
            if role == "outer_test":
                columns |= {"y_pred_fixed", "y_pred_inner"}
            check(report, prefix + "_" + role + "_full_rows_identity_truth_probability", set(frame) == columns
                and frame.row_index.is_unique and np.array_equal(frame.row_index, rows)
                and np.array_equal(frame.account_id, identity.account_id.to_numpy()[rows])
                and np.array_equal(frame.y_true, y[rows]) and frame.y_score.dtype == np.float64
                and np.isfinite(frame.y_score).all() and frame.y_score.between(0, 1).all())
        threshold, maximum_f1 = threshold_from_pr_curve(calibration.y_true, calibration.y_score)
        report["threshold_max_abs_diff"] = max(report["threshold_max_abs_diff"], close(report,
            prefix + "_independent_threshold_and_F1", [saved["threshold"], saved["threshold_validation_f1"]], [threshold, maximum_f1]))
        close(report, prefix + "_threshold_AP_AUC", [saved["threshold_validation_ap"], saved["threshold_validation_auc"]],
            [average_precision_score(calibration.y_true, calibration.y_score), roc_auc_score(calibration.y_true, calibration.y_score)])
        rescore_model(report, out, name, cal_model, X, T, calibration.y_score.to_numpy(), params, state, rounds,
            "calibration", saved["calibration_model_digest"])
        rescore_model(report, out, name, final_model, X, test, prediction.y_score.to_numpy(), params, state, rounds,
            "final", saved["final_model_digest"])
        check(report, prefix + "_two_complete_metric_policies", len(saved["metrics"]) == 2
            and {row["threshold_policy"] for row in saved["metrics"]} == set(POLICIES))
        for policy, cutoff, column in ((POLICIES[0], .5, "y_pred_fixed"), (POLICIES[1], threshold, "y_pred_inner")):
            predicted = (prediction.y_score.to_numpy() >= cutoff).astype(np.int8)
            check(report, prefix + "_" + policy + "_predictions", np.array_equal(predicted, prediction[column]))
            values = metric_values(prediction.y_true, prediction.y_score, predicted)
            expected = dict(dataset=dataset, model=name, feature_set=arm, seed=seed, fold=fold, strategy=strategy,
                threshold_policy=policy, threshold=cutoff, n_features=len(ARMS[arm]), selected_candidate_id=cid,
                selected_rounds=rounds, **values, n_test=len(test), n_positive=int(y[test].sum()))
            actual = next(row for row in saved["metrics"] if row["threshold_policy"] == policy)
            check(report, prefix + "_" + policy + "_metric_schema_metadata", set(actual) == set(expected)
                and all(actual[key] == expected[key] for key in FOLD_KEYS))
            numeric = [key for key in expected if key not in FOLD_KEYS]
            report["metric_max_abs_diff"] = max(report["metric_max_abs_diff"], close(report,
                prefix + "_" + policy + "_all_outer_metrics", [actual[key] for key in numeric], [expected[key] for key in numeric]))
            all_rows.append(expected)
        artifacts = [cal_model, cal_path, final_model, pred_path]
        artifact_bindings(report, out, saved["artifact_sha256"], artifacts)
        artifact_paths.extend(artifacts)
        selection[("anchor" if strategy == "same_protocol_anchor" else "selected") + "_effective_params_json"] = json.dumps(effective, sort_keys=True)
        for destination, source in (("threshold_stage_seconds_", "threshold_fit_seconds"),
                ("final_stage_seconds_", "final_fit_seconds"), ("threshold_model_fit_seconds_", "threshold_model_fit_seconds"),
                ("final_model_fit_seconds_", "final_model_fit_seconds")):
            check(report, prefix + "_nonnegative_cost_" + source, np.isfinite(saved[source]) and saved[source] >= 0)
            selection[destination + strategy] = saved[source]
        check(report, prefix + "_fit_time_within_stage_time", saved["threshold_model_fit_seconds"] <= saved["threshold_fit_seconds"]
            and saved["final_model_fit_seconds"] <= saved["final_fit_seconds"])
        selection["threshold_" + strategy] = threshold
    artifact_bindings(report, out, done["artifact_sha256"], artifact_paths)
    bind(report, out, path)
    return all_rows, candidate_rows, selection


def pooled_seed_rows(report, out, dataset, name, y, identity):
    result = []
    for arm, strategy, seed in itertools.product(ARMS, STRATEGIES, range(5)):
        score = np.empty(len(y), dtype=np.float64)
        predictions = {policy: np.empty(len(y), dtype=np.int8) for policy in POLICIES}
        seen = np.zeros(len(y), dtype=np.uint8)
        for fold in range(5):
            stem = f"{name}_{arm}_seed{seed}_fold{fold}"
            checkpoint = read(out / dataset / "folds" / f"{stem}.json")
            saved = checkpoint["strategies"][strategy]
            table = pd.read_parquet(contained(ROOT, saved["prediction_path"]))
            rows = table.row_index.to_numpy(np.int64)
            check(report, f"{dataset}_{stem}_{strategy}_pooled_rows_disjoint", not seen[rows].any()
                and np.array_equal(table.y_true, y[rows]) and np.array_equal(table.account_id, identity.account_id.to_numpy()[rows]))
            score[rows] = table.y_score
            seen[rows] = 1
            predictions[POLICIES[0]][rows] = table.y_pred_fixed
            predictions[POLICIES[1]][rows] = table.y_pred_inner
        check(report, f"{dataset}_{name}_{arm}_{strategy}_seed{seed}_OOF_all_accounts_once", seen.all())
        for policy in POLICIES:
            result.append(dict(dataset=dataset, model=name, feature_set=arm, strategy=strategy, seed=seed,
                threshold_policy=policy, **metric_values(y, score, predictions[policy]), n_samples=len(y), n_positive=int(y.sum())))
        del score, predictions, seen
    return pd.DataFrame(result)


def verify_worker(out, dataset, name):
    report = skeleton(dataset, name)
    try:
        protocol, candidates = protocol_audit(report, out)
        worker_path = out / dataset / f"run_manifest_{name}.json"
        worker = read(worker_path)
        check(report, dataset + "_" + name + "_complete_worker", worker["status"] == "complete"
            and worker["dataset"] == dataset and worker["model"] == name
            and worker["protocol_sha256"] == report["protocol_sha256"] and worker["completed_outer_settings"] == 75
            and datetime.fromisoformat(protocol["registered_at"]) <= datetime.fromisoformat(worker["started_at"]))
        matrix, y, identity, splits = load_data(report, out, dataset)
        fold_rows = []; candidate_rows = []; selections = []
        for arm in ARMS:
            X = np.ascontiguousarray(matrix[:, :len(ARMS[arm])])
            for seed, fold in itertools.product(range(5), range(5)):
                rows, search_rows, selection = verify_fold(report, out, dataset, name, arm, seed, fold,
                    X, y, identity, splits[seed, fold], candidates)
                fold_rows.extend(rows); selections.append(selection)
                for row in search_rows:
                    item = dict(row)
                    item["effective_params_json"] = json.dumps(item.pop("effective_params"), sort_keys=True)
                    candidate_rows.append(item)
                print(f"Independent {dataset}/{name}/{arm} seed{seed} fold{fold}: 36 rank audits; retained models rescored", flush=True)
            del X
        folds = pd.DataFrame(fold_rows)
        candidate_scores = pd.DataFrame(candidate_rows)
        selected = pd.DataFrame(selections)
        seeds = pooled_seed_rows(report, out, dataset, name, y, identity)
        check(report, "complete_worker_rows_and_models", len(folds) == 300 and len(candidate_scores) == 2700
            and len(selected) == 75 and len(seeds) == 60 and report["candidate_rank_audits_verified"] == 2700
            and report["retained_search_models_rescored"] == 75
            and report["calibration_models_rescored"] == report["final_models_rescored"] == 150 - int(selected.physical_reuse_anchor.sum()))
        for filename, frame in ((f"verification_fold_metrics_{name}.csv", folds),
                (f"verification_candidate_scores_{name}.csv", candidate_scores),
                (f"verification_selection_records_{name}.csv", selected),
                (f"verification_seed_metrics_{name}.csv", seeds)):
            path = out / dataset / filename
            frame.to_csv(path, index=False)
            bind(report, out, path)
        bind(report, out, worker_path)
        for source, digest in list(report["source_hashes"].items()):
            check(report, "final_source_unchanged_" + source, sha256(contained(ROOT, source)) == digest)
        report.update(passed=True, status="passed", complete_worker=True,
            outer_settings_verified=75, logical_final_models_verified=150, logical_threshold_models_verified=150,
            logical_candidate_scores_verified=2700, physical_search_fit_records_verified=int(candidate_scores.physical_fit_charged.sum()),
            tuned_candidate0_reuse_settings=int(selected.physical_reuse_anchor.sum()),
            retained_physical_models_rescored=len(report.get("physical_model_hashes", {})),
            checks_passed_total=len(report["checks"]), original_inputs_sha256_verified=True,
            interpretation_scope="Full-order candidate ranking audits independently reconstructed; one selected inner-fold0 model per setting plus all physical threshold/final models independently rescored; discarded candidate models are not independently rescored")
    except Exception as error:
        report.update(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    write(out / dataset / f"validation_{name}.json", report)
    print(json.dumps({key: report.get(key) for key in ("dataset", "model", "passed", "outer_settings_verified", "error")}), flush=True)
    return report


def absorb(report, out, worker, dataset, name):
    check(report, dataset + "_" + name + "_complete_independent_worker_report", worker["passed"] is True
        and worker["status"] == "passed" and worker["complete_worker"] is True
        and worker["dataset"] == dataset and worker["model"] == name
        and worker["validator_sha256"] == sha256(__file__) and worker["protocol_sha256"] == report["protocol_sha256"]
        and worker["trainer_sha256"] == report["trainer_sha256"] and worker["helper_sha256"] == report["helper_sha256"]
        and worker["checks"] and all(worker["checks"].values()) and worker["original_inputs_sha256_verified"] is True
        and worker["outer_settings_verified"] == 75 and worker["logical_final_models_verified"] == 150
        and worker["logical_threshold_models_verified"] == 150 and worker["logical_candidate_scores_verified"] == 2700
        and worker["candidate_rank_audits_verified"] == 2700 and worker["retained_search_models_rescored"] == 75
        and worker["discarded_search_model_rescoring_claimed"] is False)
    for name, digest in worker["source_hashes"].items():
        source_bind(report, contained(ROOT, name), digest)
    for name, digest in worker["result_hashes"].items():
        path = contained(out, name)
        check(report, dataset + "_worker_result_unchanged_" + name, sha256(path) == digest)
        bind(report, out, path)
    for name, digest in worker["physical_model_hashes"].items():
        check(report, dataset + "_distinct_physical_model_" + name, name not in report.setdefault("physical_model_hashes", {}))
        report["physical_model_hashes"][name] = digest
    report["model_probability_max_abs_diff"].update(worker["model_probability_max_abs_diff"])
    for key in ("candidate_rank_audits_verified", "retained_search_models_rescored", "calibration_models_rescored", "final_models_rescored"):
        report[key] += worker[key]
    for key in ("metric_max_abs_diff", "threshold_max_abs_diff"):
        report[key] = max(report[key], worker[key])


def audit_execution_schedule(report, out):
    """Bind optional scheduling provenance only after it has become final.

    Model workers never bind this mutable log. The preserved registered trainer
    and complete scientific setting keys remain the authoritative experiment.
    """
    schedule_path = out / "execution_schedule.json"
    if not schedule_path.exists():
        report["execution_rescheduled"] = False
        return
    schedule = read(schedule_path)
    script = ALT / "code/schedule_account_gmkep_tuning_rf.py"
    check(report, "scheduler_code_protocol_hashes_and_scope", schedule["schema_version"] == 1
        and schedule["scheduler"] == rel(script) and schedule["scheduler_sha256"] == sha256(script)
        and schedule["trainer"] == rel(TRAINER) and schedule["trainer_sha256"] == report["trainer_sha256"]
        and schedule["protocol_sha256"] == report["protocol_sha256"] and schedule["statistical_protocol_changed"] is False
        and schedule["fit_threads_per_worker"] == 2 and schedule["max_concurrent_compute_workers"] == 6)
    snapshot = contained(ROOT, schedule["scheduler_snapshot"])
    check(report, "scheduler_snapshot_hash_and_copy_check", snapshot.is_relative_to((out / "execution_code").resolve())
        and schedule["scheduler_snapshot_sha256"] == schedule["scheduler_sha256"] == sha256(snapshot)
        and schedule["fold_copy_check_passed"] is True)
    bind(report, out, snapshot)
    original_tree = ast.parse(TRAINER.read_text())
    scheduler_tree = ast.parse(script.read_text())
    original = next(node for node in original_tree.body if isinstance(node, ast.FunctionDef) and node.name == "run")
    copied = next(node for node in scheduler_tree.body if isinstance(node, ast.FunctionDef) and node.name == "_complete_fold")
    def science_if(function):
        matches = [node for node in ast.walk(function) if isinstance(node, ast.If)
            and isinstance(node.test, ast.Call) and isinstance(node.test.func, ast.Attribute)
            and isinstance(node.test.func.value, ast.Name) and node.test.func.value.id == "path"
            and node.test.func.attr == "exists"]
        if len(matches) != 1:
            raise ValueError("Scheduler/original fold science block not unique")
        return ast.dump(matches[0], include_attributes=False)
    a, b = science_if(original), science_if(copied)
    check(report, "scheduler_fold_block_AST_identical", a == b and hashlib.sha256(a.encode()).hexdigest()
        == schedule["fold_if_else_ast_sha256"])
    check(report, "scheduler_has_final_complete_session", bool(schedule["sessions"]) and "completed_at" in schedule
        and schedule["sessions"][-1]["status"] == "complete")
    session = schedule["sessions"][-1]
    old_workers = session["original_rf_workers"]
    replacements = session["replacement_workers"]
    check(report, "original_RF_stopped_before_all_six_replacements", session["original_rf_all_stopped"] is True
        and len(old_workers) == 2 and {worker["dataset"] for worker in old_workers} == set(DATASETS)
        and all(worker["original_process_no_longer_live"] is True for worker in old_workers)
        and len(replacements) == 6 and {(job["dataset"], job["feature_set"]) for job in replacements}
            == set(itertools.product(DATASETS, ARMS))
        and all(job["status"] == "complete" and job["returncode"] == 0
            and job["fit_threads"] == 2 and job["expected_outer_settings"] == 25 for job in replacements))
    for job in replacements:
        manifest_path = out / job["dataset"] / "scheduler_manifests" / f"random_forest_{job['feature_set']}.json"
        manifest = read(manifest_path)
        check(report, "complete_replacement_" + job["dataset"] + "_" + job["feature_set"], manifest["status"] == "complete"
            and manifest["dataset"] == job["dataset"] and manifest["model"] == "random_forest"
            and manifest["feature_set"] == job["feature_set"] and manifest["completed_outer_settings"] == 25
            and manifest["expected_outer_settings"] == 25 and manifest["scheduler_sha256"] == schedule["scheduler_sha256"]
            and manifest["protocol_sha256"] == report["protocol_sha256"] and manifest["schedule_id"] == session["schedule_id"]
            and manifest["pid"] == job["pid"])
        bind(report, out, manifest_path)
    for dataset in DATASETS:
        checkpoint = session["checkpoint_after"][dataset]
        check(report, dataset + "_scheduler_committed_75_and_unknown_retry_cost", checkpoint["complete_outer_settings"] == 75
            and checkpoint["saved_physical_search_fit_records"] == 1350
            and checkpoint["uncommitted_interrupted_compute_seconds"] is None
            and checkpoint["uncommitted_interrupted_compute_is_assumed_zero"] is False)
    source_bind(report, script, schedule["scheduler_sha256"])
    bind(report, out, schedule_path)
    report.update(execution_rescheduled=True, execution_schedule_sha256=sha256(schedule_path),
        scheduler_sha256=sha256(script), scheduler_fold_science_AST_sha256=schedule["fold_if_else_ast_sha256"],
        interrupted_retry_compute_in_total_fit_count=False,
        fit_count_interpretation="13500 committed useful search-fit records plus 450 representative refits and retained strategy fits; unobserved interrupted/repeated computation is recorded as unknown and is not counted as zero",
        execution_cost_accounting=schedule["cost_accounting"])


def aggregate(out):
    report = skeleton("all")
    try:
        protocol, _ = protocol_audit(report, out)
        fold_frames = []; seed_frames = []; candidate_frames = []; selection_frames = []; workers = []
        for dataset, model in itertools.product(DATASETS, MODELS):
            path = out / dataset / f"validation_{model}.json"
            worker = read(path)
            absorb(report, out, worker, dataset, model); bind(report, out, path)
            workers.append(worker)
            for destination, filename in ((fold_frames, f"verification_fold_metrics_{model}.csv"),
                    (seed_frames, f"verification_seed_metrics_{model}.csv"),
                    (candidate_frames, f"verification_candidate_scores_{model}.csv"),
                    (selection_frames, f"verification_selection_records_{model}.csv")):
                destination.append(pd.read_csv(out / dataset / filename))
        folds = pd.concat(fold_frames, ignore_index=True)
        seeds = pd.concat(seed_frames, ignore_index=True)
        candidates = pd.concat(candidate_frames, ignore_index=True)
        selections = pd.concat(selection_frames, ignore_index=True)
        compare_table(report, out, "fold_metrics.csv", folds, FOLD_KEYS)
        compare_table(report, out, "seed_oof_metrics.csv", seeds, SEED_KEYS)
        keys = ["dataset", "model", "feature_set", "seed", "fold"]
        compare_table(report, out, "candidate_fold_scores.csv", candidates, keys + ["candidate_id", "inner_fold"])
        compare_table(report, out, "selection_records.csv", selections, keys)
        check(report, "all_450_settings_and_16200_candidate_keys", len(selections) == 450 and len(candidates) == 16200
            and not selections.duplicated(keys).any()
            and set(selections[keys].itertuples(index=False, name=None))
                == set(itertools.product(DATASETS, MODELS, ARMS, range(5), range(5)))
            and report["candidate_rank_audits_verified"] == 16200 and report["retained_search_models_rescored"] == 450
            and candidates[["dataset", "physical_fit_id"]].drop_duplicates().shape[0]
                == int(candidates.physical_fit_charged.sum()) == 13500)
        old_folds = pd.read_csv(OLD_FIXED / "fold_metrics.csv")
        old_seeds = pd.read_csv(OLD_FIXED / "seed_oof_metrics.csv")
        aggregate_scientific_tables(report, out, folds, seeds, old_folds, old_seeds)
        physical = 900 - int(selections.physical_reuse_anchor.sum())
        check(report, "all_retained_physical_models_independently_rescored", report["calibration_models_rescored"]
            == report["final_models_rescored"] == physical
            and len(report["physical_model_hashes"]) == 450 + 2 * physical
            and len(report["model_probability_max_abs_diff"]) == 450 + 2 * physical)
        manifest = read(out / "run_manifest.json")
        check(report, "complete_root_manifest_not_a_partial_search", manifest["status"] in ("trained_aggregated", "complete")
            and manifest["protocol_sha256"] == report["protocol_sha256"] and manifest["completed_outer_settings"] == 450
            and manifest["logical_candidate_evaluations_completed"] == 16200
            and manifest["physical_search_fits"] == 13500 and manifest["representative_refits"] == 450
            and manifest["physical_final_models"] == manifest["physical_threshold_models"] == physical
            and manifest["logical_final_models"] == manifest["logical_threshold_models"] == 900
            and manifest["tuned_candidate0_reuse_settings"] == int(selections.physical_reuse_anchor.sum())
            and manifest["retained_physical_model_count"] == 450 + 2 * physical)
        audit_execution_schedule(report, out)
        for source, digest in list(report["source_hashes"].items()):
            check(report, "final_source_unchanged_" + source, sha256(contained(ROOT, source)) == digest)
        report.update(passed=True, status="passed", complete_experiment=True, outer_settings_verified=450,
            logical_final_models_verified=900, logical_threshold_models_verified=900,
            physical_search_fit_records_verified=13500, representative_refits_verified=450,
            retained_physical_models_rescored=450 + 2 * physical,
            candidate_rank_audits_verified=16200, summary_rows_verified=72, seed_oof_rows_verified=360,
            tuned_candidate0_reuse_settings=int(selections.physical_reuse_anchor.sum()),
            original_inputs_sha256_verified=True,
            checks_passed_total=len(report["checks"]) + sum(worker["checks_passed_total"] for worker in workers),
            root_manifest_hash_omitted_for_final_status_transition=True,
            interpretation_scope="Complete finite-budget nested tuning; all candidate ranking metrics/truth permutations independently reconstructed; selected representative and every retained threshold/final model independently rescored; discarded search models not independently rescored; correlated static-account development splits")
    except Exception as error:
        report.update(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    write(out / "validation_models.json", report)
    print(json.dumps({key: report.get(key) for key in ("passed", "outer_settings_verified", "candidate_rank_audits_verified", "retained_physical_models_rescored", "error")}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    if args.preflight_only:
        report = skeleton(args.dataset or "preflight", args.model)
        protocol_audit(report, out)
        if args.dataset:
            load_data(report, out, args.dataset)
        print(json.dumps({"passed": True, "preflight_only": True, "checks": len(report["checks"])}))
        return
    if args.aggregate_only:
        if args.dataset or args.model:
            parser.error("Aggregate does not accept dataset/model")
        result = aggregate(out)
    else:
        if not args.dataset or not args.model:
            parser.error("Select both --dataset and --model, or --aggregate-only")
        result = verify_worker(out, args.dataset, args.model)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
