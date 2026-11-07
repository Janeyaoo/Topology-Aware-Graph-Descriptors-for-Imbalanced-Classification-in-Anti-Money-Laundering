"""Independently verify the three-classifier BASE/GMKE/GMKEP comparison.

Only newly fitted Random Forest inner/final models are loaded and rescored.
Reused models remain untouched: immutable prior validation/artifact hashes bind
their provenance, and all saved calibration/test predictions are re-evaluated.
No training module or its threshold/metric functions are imported.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime
import gc
import hashlib
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
DEFAULT_OUT = ALT / "results/account_gmkep56_multiclassifier_20261008"
ORIGINAL = ALT / "results/account_original_multiclassifier_20261007_isolated"
S10 = ALT / "results/account_s10_increment_20261008"
TRAINER = ALT / "code/run_account_gmkep_multiclassifier.py"
HELPER = ALT / "code/run_account_feature_mechanism.py"
DATASETS = ("hi_small", "li_small")
MODELS = ("lightgbm", "xgboost", "random_forest")
POLICIES = ("fixed_0.5", "inner_max_f1")
TZ = ZoneInfo("Asia/Shanghai")
TOL = 1e-12
PREDICTION_THREADS = 2
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
CONTRASTS = [
    dict(contrast_id="gmke_vs_base", contrast_type="structural", lhs_arm="gmke44", rhs_arm="base16"),
    dict(contrast_id="ph_vs_gmke", contrast_type="barcode", lhs_arm="gmkep56", rhs_arm="gmke44"),
    dict(contrast_id="full_vs_base", contrast_type="full_dictionary", lhs_arm="gmkep56", rhs_arm="base16")]
METRICS = ["precision", "recall", "f1", "auc", "ap", "accuracy", "balanced_accuracy",
           "specificity", "fpr", "fnr", "tn", "fp", "fn", "tp"]
FOLD_KEYS = ["dataset", "model", "feature_set", "seed", "fold", "threshold_policy"]
SUMMARY_KEYS = ["dataset", "model", "feature_set", "threshold_policy"]
PAIR_KEYS = ["dataset", "model", "contrast_id", "threshold_policy", "contrast_type", "lhs_arm", "rhs_arm"]
PRESETS = {
    "lightgbm": dict(objective="binary", metric="auc", n_estimators=500, learning_rate=.05,
        num_leaves=63, min_child_samples=20, subsample=.8, colsample_bytree=.8,
        reg_alpha=.1, reg_lambda=1., class_weight="balanced", n_jobs=6, verbose=-1),
    "xgboost": dict(objective="binary:logistic", eval_metric="auc", n_estimators=500,
        learning_rate=.05, max_depth=6, min_child_weight=5, subsample=.8, colsample_bytree=.8,
        reg_alpha=.1, reg_lambda=1., tree_method="hist", max_bin=256, n_jobs=6, verbosity=0),
    "random_forest": dict(n_estimators=200, max_depth=14, min_samples_leaf=5,
        max_features="sqrt", max_samples=.8, bootstrap=True, class_weight="balanced", n_jobs=6),
}
_HASH_CACHE = {}


def sha256(path):
    path = Path(path).resolve()
    stat = path.stat()
    signature = (stat.st_size, stat.st_mtime_ns)
    cached = _HASH_CACHE.get(path)
    if cached and cached[0] == signature:
        return cached[1]
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""): value.update(block)
    result = value.hexdigest()
    _HASH_CACHE[path] = signature, result
    return result


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def contained(base, name):
    path = (Path(base) / name).resolve()
    if not path.is_relative_to(Path(base).resolve()) or not path.is_file():
        raise ValueError(f"Missing or uncontained registered input: {name}")
    return path


def check(report, name, condition):
    report["checks"][name] = bool(condition)
    if not condition:
        raise ValueError(f"Independent GMKEP validation failed: {name}")


def close(report, name, actual, expected, tolerance=TOL):
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    check(report, name, a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
          and np.allclose(a, b, atol=tolerance, rtol=0))
    return float(np.max(np.abs(a - b), initial=0))


def skeleton(dataset):
    return {"dataset": dataset, "passed": False, "status": "failed",
        "validated_at": datetime.now(TZ).isoformat(timespec="seconds"), "checks": {},
        "validator": rel(__file__), "validator_sha256": sha256(__file__),
        "tolerance": TOL, "prediction_threads": PREDICTION_THREADS,
        "source_hashes": {rel(__file__): sha256(__file__)}, "result_hashes": {},
        "new_models_sha256": {}, "reused_models_sha256": {},
        "model_probability_max_abs_diff": {}, "metric_max_abs_diff": 0.,
        "threshold_max_abs_diff": 0., "table_max_abs_diff": {},
        "new_final_models_verified": 0, "new_inner_models_verified": 0,
        "reused_final_models_hash_verified": 0, "reused_inner_models_hash_verified": 0}


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
        raise ValueError("Invalid calibration truth or probabilities")
    # Every candidate F1 is 2*TP/(n_predicted_positive+n_positive). Distinct
    # rational values therefore differ by at least 2/(n+n_positive)**2.
    # This run's calibration sets are safely below that resolution limit;
    # refuse larger inputs rather than treating genuinely distinct F1s as ties.
    minimum_distinct_f1_gap = 2. / float(len(y) + np.count_nonzero(y)) ** 2
    if minimum_distinct_f1_gap <= TOL + 16 * np.finfo(float).eps:
        raise ValueError("Calibration size exceeds the certified F1 tie tolerance")
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


def independent_summary(frame):
    rows = []
    for keys, group in frame.groupby(SUMMARY_KEYS, sort=True):
        row = dict(zip(SUMMARY_KEYS, keys), n_folds=len(group),
            n_features=int(group.n_features.iloc[0]), source_origin=group.source_origin.iloc[0],
            source_run=group.source_run.iloc[0])
        if len(group) != 25: raise ValueError("Incomplete 25-fold feature setting")
        for metric in METRICS + ["threshold"]:
            row[metric + "_mean"] = float(group[metric].mean())
            row[metric + "_std"] = float(group[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def independent_pairs(frame, keys):
    output = []
    for spec in CONTRASTS:
        left = frame[frame.feature_set == spec["lhs_arm"]].set_index(keys).sort_index()
        right = frame[frame.feature_set == spec["rhs_arm"]].set_index(keys).sort_index()
        if not left.index.equals(right.index): raise ValueError("Unmatched paired keys")
        paired = left.reset_index()[keys].copy()
        for name, value in spec.items(): paired[name] = value
        for metric in METRICS:
            paired["lhs_" + metric] = left[metric].to_numpy()
            paired["rhs_" + metric] = right[metric].to_numpy()
            paired["delta_" + metric] = left[metric].to_numpy() - right[metric].to_numpy()
        output.append(paired)
    return pd.concat(output, ignore_index=True)


def independent_pair_summary(frame):
    rows = []
    for keys, group in frame.groupby(PAIR_KEYS, sort=True):
        row = dict(zip(PAIR_KEYS, keys), n_pairs=len(group))
        if len(group) != 25: raise ValueError("Incomplete 25 paired folds")
        for metric in METRICS:
            values = group["delta_" + metric]
            row["delta_" + metric + "_mean"] = float(values.mean())
            row["delta_" + metric + "_std"] = float(values.std(ddof=1))
            row[metric + "_positive_pairs"] = int((values > 0).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def expected_settings(out):
    settings = []
    for dataset in DATASETS:
        for model in MODELS:
            for arm in ARMS:
                if arm == "base16":
                    source, source_arm, origin, validation = ORIGINAL, "base", "reused", "validation.json"
                elif model != "random_forest":
                    source, source_arm, origin, validation = S10, arm, "reused", "validation_models.json"
                else:
                    source, source_arm, origin, validation = out, arm, "newly_fitted", None
                settings.append(dict(dataset=dataset, model=model, feature_set=arm, n_features=len(ARMS[arm]),
                    origin=origin, source_run=rel(source), source_feature_set=source_arm, source_validation=validation))
    return settings


def prior_digest(prior_report, source, path):
    """Original model keys are ROOT-relative; S10 keys are output-relative."""
    name = Path(path).resolve().relative_to(source.resolve()).as_posix()
    candidates = [prior_report.get("result_hashes", {}).get(name),
                  prior_report.get("models_sha256", {}).get(name),
                  prior_report.get("models_sha256", {}).get(rel(path))]
    available = [value for value in candidates if value is not None]
    if not available or len(set(available)) != 1:
        raise ValueError(f"Absent or inconsistent prior independent binding: {rel(path)}")
    return available[0]


def protocol_audit(report, out):
    protocol = read(out / "protocol.json")
    check(report, "fixed_full_matrix_18_settings", protocol["datasets"] == list(DATASETS)
        and protocol["models"] == list(MODELS) and protocol["feature_columns"] == ARMS
        and protocol["feature_sets"] == {arm: len(names) for arm, names in ARMS.items()}
        and protocol["new_model"] == "random_forest" and protocol["new_arms"] == ["gmke44", "gmkep56"]
        and protocol["contrasts"] == CONTRASTS and protocol["all_metrics"] == METRICS)
    check(report, "fixed_presets_and_source_paths", protocol["model_params"] == PRESETS
        and protocol["source_original_run"] == rel(ORIGINAL) and protocol["source_feature_run"] == rel(S10)
        and protocol["rf_sqrt_candidates"] == dict(base16=4, gmke44=6, gmkep56=7))
    check(report, "fixed_expected_new_reused_reported_counts", all(protocol[key] == value for key, value in {
        "expected_new_final_models": 100, "expected_new_inner_models": 100,
        "expected_reused_final_models": 350, "expected_reused_inner_models": 350,
        "expected_reported_final_models": 450, "expected_reported_inner_models": 450}.items())
        and protocol["table_counts"] == dict(fold=900, summary=36, paired_fold=900, paired_summary=36, seed=180, paired_seed=180))
    check(report, "frozen_trainer_and_helper", protocol["trainer_sha256"] == sha256(TRAINER)
        == sha256(out / "trainer_snapshot.py") and protocol["helper_sha256"] == sha256(HELPER))
    report.update(protocol_sha256=sha256(out / "protocol.json"), trainer_sha256=sha256(TRAINER), helper_sha256=sha256(HELPER))
    source_bind(report, TRAINER); source_bind(report, HELPER)
    bind(report, out, out / "protocol.json"); bind(report, out, out / "trainer_snapshot.py")
    clarification = read(out / "metadata_clarification.json")
    actual_blocks = dict(base16=BASE, G16=G, M5=GMKE[32:37], K3=GMKE[37:40], E4=GMKE[40:44], P12u=P12)
    unused = ["S_graph4", "S_hodge4", "S_PL2"]
    check(report, "catalog_clarification_matches_actual_56_input_columns", clarification["protocol_sha256"] == report["protocol_sha256"]
        and clarification["authoritative_input_field"] == "protocol.json/feature_columns"
        and clarification["actual_blocks"] == actual_blocks and clarification["unused_upstream_catalog_blocks"] == unused
        and clarification["actual_feature_counts"] == {arm: len(names) for arm, names in ARMS.items()}
        and clarification["contains_Q_or_S_in_actual_inputs"] is False
        and sum(actual_blocks.values(), []) == ARMS["gmkep56"]
        and all(protocol["feature_blocks"][name] == values for name, values in actual_blocks.items())
        and not any(name.startswith(("q_", "s_")) for columns in ARMS.values() for name in columns)
        and not set(sum((protocol["feature_blocks"][name] for name in unused), [])).intersection(ARMS["gmkep56"]))
    bind(report, out, out / "metadata_clarification.json")
    report["catalog_clarification_verified"] = clarification
    for name in ("dataset_inputs", "source_settings", "source_split_hashes", "reused_artifact_hashes"):
        path = out / (name + ".json")
        check(report, name + "_registered_hash", protocol[name + "_sha256"] == sha256(path))
        bind(report, out, path)
    settings = read(out / "source_settings.json")
    check(report, "all_18_setting_sources_independently_rebuilt", settings == expected_settings(out))
    prior_protocols = {ORIGINAL: read(ORIGINAL / "protocol.json"), S10: read(S10 / "protocol.json")}
    scientific = ("seeds", "outer_n_splits", "outer_split", "inner_calibration_fraction", "inner_random_state",
        "model_random_state", "parameter_candidates_per_model", "early_stopping", "class_weighting",
        "threshold_policies", "threshold_selection", "prediction_rule", "summary_std_ddof")
    check(report, "all_scientific_selection_fields_same_as_two_sources", all(
        protocol[name] == prior_protocols[ORIGINAL][name] == prior_protocols[S10][name] for name in scientific))
    check(report, "independent_fixed_splits_and_selection", protocol["seeds"] == list(range(5))
        and protocol["outer_n_splits"] == 5 and protocol["outer_split"] == "StratifiedKFold(shuffle=True, random_state=seed)"
        and protocol["inner_calibration_fraction"] == .2 and protocol["inner_random_state"] == "10000 + 100 * seed + fold"
        and protocol["model_random_state"] == "100 * seed + fold" and protocol["parameter_candidates_per_model"] == 1
        and protocol["threshold_policies"] == list(POLICIES) and protocol["prediction_rule"] == "y_score >= threshold"
        and protocol["summary_std_ddof"] == 1)
    check(report, "prior_model_presets_identical", prior_protocols[ORIGINAL]["model_params"] == PRESETS
        and all(prior_protocols[S10]["model_params"][name] == PRESETS[name] for name in MODELS[:2])
        and prior_protocols[S10]["feature_columns"]["gmke44"] == GMKE
        and prior_protocols[S10]["feature_columns"]["gmkep56"] == GMKE + P12)
    registry = read(out / "reused_artifact_hashes.json")
    prior_reports = {}
    for source, filename, count, hash_field in ((ORIGINAL, "validation.json", 600, "original_protocol_sha256"),
                                               (S10, "validation_models.json", 2200, "feature_protocol_sha256")):
        validation = read(source / filename)
        check(report, rel(source) + "_independent_model_validation_passed", validation["passed"] is True
            and validation["n_models_verified"] == count and validation["original_inputs_sha256_verified"] is True
            and validation["protocol_sha256"] == protocol[hash_field] == sha256(source / "protocol.json")
            and all(validation["checks"].values()) and bool(validation["checks"])
            and sha256(contained(ROOT, validation["validator"])) == validation["validator_sha256"])
        source_manifest = read(source / "run_manifest.json")
        check(report, rel(source) + "_source_complete_same_protocol", source_manifest["status"] == "complete"
            and source_manifest["protocol_sha256"] == sha256(source / "protocol.json"))
        for name, digest in validation.get("source_hashes", {}).items():
            source_bind(report, contained(ROOT, name), digest)
        for name in ("protocol.json", filename, "run_manifest.json"):
            path = source / name
            check(report, "prior_report_registered_" + rel(path), registry[rel(path)] == sha256(path))
            source_bind(report, path, registry[rel(path)])
        prior_reports[source] = validation
    expected_registry = {rel(source / name) for source, filename in ((ORIGINAL, "validation.json"), (S10, "validation_models.json"))
                         for name in ("protocol.json", filename, "run_manifest.json")}
    for name in ("feature_protocol.json", "validation_math.json", "validation_features.json", "design_alias_validation.json"):
        path = S10 / name
        expected_registry.add(rel(path))
        check(report, "prior_geometry_binding_" + name, prior_digest(prior_reports[S10], S10, path) == registry[rel(path)] == sha256(path))
        if name != "feature_protocol.json": check(report, "prior_geometry_passed_" + name, read(path)["passed"] is True)
    for dataset in DATASETS:
        for seed in range(5):
            for fold in range(5):
                for source in (ORIGINAL, S10):
                    expected_registry.add(rel(source / dataset / "splits" / f"seed{seed}_fold{fold}.npz"))
    for item in settings:
        if item["origin"] != "reused": continue
        source = contained(ROOT, item["source_run"] + "/protocol.json").parent
        for seed in range(5):
            for fold in range(5):
                stem = f"{item['model']}_{item['source_feature_set']}_seed{seed}_fold{fold}"
                checkpoint = source / item["dataset"] / "folds" / (stem + ".json")
                expected_registry.add(rel(checkpoint))
                record = read(checkpoint)
                expected_registry.update(record["artifact_sha256"])
    check(report, "reused_registry_exact_selected_artifacts", set(registry) == expected_registry)
    roots = {rel(source / name) for source, filename in ((ORIGINAL, "validation.json"), (S10, "validation_models.json"))
             for name in ("protocol.json", filename, "run_manifest.json")}
    for name, digest in registry.items():
        path = contained(ROOT, name)
        source = ORIGINAL if path.is_relative_to(ORIGINAL) else S10
        check(report, "selected_artifact_prior_binding_" + name,
            name in roots or prior_digest(prior_reports[source], source, path) == digest)
        source_bind(report, path, digest)
    tree = ast.parse(TRAINER.read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and node.func.attr == "fit_model"]
    args = [tuple(ast.unparse(arg) for arg in call.args) for call in calls]
    check(report, "trainer_inner_fit_cal_only_final_full_train_no_eval", len(calls) == 2 and set(args) == {
        ("'random_forest'", "X", "y", "fit", "cal", "protocol", "random_state"),
        ("'random_forest'", "X", "y", "train", "None", "protocol", "random_state", "200")})
    helper = ast.parse(HELPER.read_text())
    function = next(node for node in helper.body if isinstance(node, ast.FunctionDef) and node.name == "fit_model")
    fits = [node for node in ast.walk(function) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "fit"]
    check(report, "helper_uses_only_fit_labels_and_rf_no_extra_sampling_or_eval", all(
        [ast.unparse(arg) for arg in node.args[:2]] == ["X[fit_index]", "y[fit_index]"] for node in fits)
        and any(not node.keywords and [ast.unparse(arg) for arg in node.args] == ["X[fit_index]", "y[fit_index]"] for node in fits)
        and "RandomForestClassifier(**params)" in ast.unparse(function)
        and "params = dict(registered['model_params'][name])" in ast.unparse(function))
    report["source_fit_scope_audit"] = "Registered balanced RF; fit labels only; inner thresholds only; final outer_train without eval_set"
    report["source_control_hashes"] = registry
    return protocol, settings, prior_reports


def load_data(report, out, dataset):
    metadata = read(out / "dataset_inputs.json")[dataset]
    check(report, "dataset_feature_source_and_56_column_order", metadata["feature_source"] == rel(S10 / dataset / "features.parquet")
        and metadata["feature_columns"] == ARMS["gmkep56"])
    for name, digest in metadata["input_sha256"].items(): source_bind(report, contained(ROOT, name), digest)
    table = pd.read_parquet(contained(ROOT, metadata["feature_source"]), columns=["account_id", "label", *ARMS["gmkep56"]])
    identity = pd.read_parquet(out / dataset / "accounts.parquet")
    original = pd.read_parquet(ORIGINAL / dataset / "accounts.parquet")
    labels = pd.read_parquet(ALT / "data" / dataset / ("account_labels.parquet" if dataset == "hi_small" else "li_small_labels.parquet"))
    check(report, "account_identity_labels_order_exact", set(identity) == {"row_index", "account_id", "label"}
        and identity.equals(original) and table.account_id.is_unique and not table.isna().any().any()
        and np.array_equal(identity.row_index, np.arange(len(table)))
        and np.array_equal(table.account_id, identity.account_id) and np.array_equal(table.label, identity.label)
        and table[["account_id", "label"]].equals(labels))
    y = table.label.to_numpy(np.int8)
    check(report, "full_sample_positive_counts", (len(y), int(y.sum())) ==
        ((515088, 6357) if dataset == "hi_small" else (705907, 5304))
        and metadata["n_samples"] == len(y) and metadata["n_positive"] == int(y.sum()) and set(np.unique(y)) == {0, 1})
    for tag, columns in (("base", BASE), ("ftd", GMKE[16:])):
        previous = pd.read_parquet(ALT / "results" / dataset / "features" / f"{tag}_features.parquet")
        check(report, "original_" + tag + "_values_preserved", np.array_equal(previous.account_id, table.account_id)
            and np.array_equal(previous[columns].to_numpy(np.float32), table[columns].to_numpy(np.float32)))
    X = np.ascontiguousarray(table[ARMS["gmkep56"]].to_numpy(np.float32))
    check(report, "matrix_finite_float32_56", X.dtype == np.float32 and np.isfinite(X).all() and X.shape == (len(y), 56))
    split_hashes = read(out / "source_split_hashes.json")
    coverage = {seed: np.zeros(len(y), np.uint8) for seed in range(5)}
    splits = {}
    for seed in range(5):
        for fold, (train, test) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)):
            fit_local, cal_local = next(StratifiedShuffleSplit(1, test_size=.2,
                random_state=10000 + 100 * seed + fold).split(np.zeros(len(train)), y[train]))
            arrays = dict(outer_train=train, outer_test=test, inner_fit=train[fit_local], inner_calibration=train[cal_local])
            path = out / dataset / "splits" / f"seed{seed}_fold{fold}.npz"
            previous = ORIGINAL / dataset / "splits" / path.name
            check(report, f"split{seed}_{fold}_bytes_same_two_sources", sha256(path) == sha256(previous)
                == sha256(S10 / dataset / "splits" / path.name) == split_hashes[rel(previous)])
            with np.load(path) as saved:
                check(report, f"split{seed}_{fold}_arrays_reconstructed", set(saved.files) == set(arrays)
                    and all(saved[name].dtype == np.int32 and np.array_equal(saved[name], values) for name, values in arrays.items()))
            fit, cal = arrays["inner_fit"], arrays["inner_calibration"]
            check(report, f"split{seed}_{fold}_disjoint_cover", not np.intersect1d(train, test).size
                and np.array_equal(np.sort(np.r_[train, test]), np.arange(len(y)))
                and not np.intersect1d(fit, cal).size and not np.intersect1d(fit, test).size and not np.intersect1d(cal, test).size
                and np.array_equal(np.sort(np.r_[fit, cal]), train))
            coverage[seed][test] += 1
            splits[seed, fold] = arrays
            bind(report, out, path)
    check(report, "all_seed_test_accounts_once", all(np.all(value == 1) for value in coverage.values()))
    bind(report, out, out / dataset / "accounts.parquet")
    del table, previous, labels, original
    gc.collect()
    return X, y, identity, splits


def rescore_rf(report, out, path, X, rows, expected, state, stage):
    key = path.relative_to(out).as_posix()
    model = joblib.load(path)
    params = model.get_params()
    check(report, key + "_class_params_tree_budget_features_and_labels", type(model).__name__ == "RandomForestClassifier"
        and all(params.get(name) == value for name, value in {**PRESETS["random_forest"], "random_state": state}.items())
        and len(model.estimators_) == model.n_estimators == 200 and model.n_features_in_ == X.shape[1]
        and model.n_outputs_ == 1 and np.array_equal(model.classes_, [0, 1])
        and all(tree.max_features_ == int(np.sqrt(X.shape[1])) for tree in model.estimators_))
    # Changing prediction parallelism does not alter the registered fit budget.
    model.set_params(n_jobs=PREDICTION_THREADS)
    score = model.predict_proba(X[rows])[:, 1]
    report["model_probability_max_abs_diff"][key] = close(report, key + "_full_saved_probabilities", score, expected)
    report["new_models_sha256"][key] = sha256(path)
    report["new_" + stage + "_models_verified"] += 1
    bind(report, out, path)
    del model, score
    gc.collect()


def setting_paths(item, seed, fold):
    source = ROOT / item["source_run"]
    folder = source / item["dataset"]
    stem = f"{item['model']}_{item['source_feature_set']}_seed{seed}_fold{fold}"
    return source, stem, folder / "folds" / (stem + ".json"), [
        folder / "splits" / f"seed{seed}_fold{fold}.npz",
        folder / "inner_models" / (stem + ".joblib"), folder / "models" / (stem + ".joblib"),
        folder / "calibration" / (stem + ".parquet"), folder / "predictions" / (stem + ".parquet")]


def verify_fold(report, out, item, seed, fold, X, y, identity, splits, prior_reports):
    source, stem, checkpoint, artifacts = setting_paths(item, seed, fold)
    record = read(checkpoint)
    label = f"{item['dataset']}_{stem}"
    arrays = splits[seed, fold]
    train, test, fit, cal = [arrays[key] for key in ("outer_train", "outer_test", "inner_fit", "inner_calibration")]
    state = 100 * seed + fold
    columns = ARMS[item["feature_set"]]
    check(report, label + "_checkpoint_meta_and_selection_counts",
        [record[key] for key in ("dataset", "model", "feature_set", "seed", "fold", "n_features")] ==
        [item["dataset"], item["model"], item["source_feature_set"], seed, fold, len(columns)]
        and record.get("model_random_state", record.get("random_state")) == state
        and record["inner_random_state"] == 10000 + 100 * seed + fold
        and record["final_fit_has_eval_set"] is False
        and record["protocol_sha256"] == sha256(source / "protocol.json")
        and all(record[key] == value for key, value in dict(n_outer_train=len(train), n_outer_test=len(test),
            n_inner_fit=len(fit), n_inner_calibration=len(cal), outer_train_positive=int(y[train].sum()),
            outer_test_positive=int(y[test].sum()), inner_fit_positive=int(y[fit].sum()),
            inner_calibration_positive=int(y[cal].sum())).items()))
    if "feature_columns" in record:
        check(report, label + "_exact_column_order", record["feature_columns"] == columns)
    else:
        manifest_path = source / item["dataset"] / f"run_manifest_{item['model']}.json"
        manifest = read(manifest_path)
        check(report, label + "_legacy_column_order", source == ORIGINAL
            and manifest["feature_columns"][item["source_feature_set"]] == columns)
        source_bind(report, manifest_path, prior_digest(prior_reports[source], source, manifest_path))
    rounds = record["selected_rounds"]
    check(report, label + "_fixed_model_rounds", isinstance(rounds, int) and
        (rounds == 200 if item["model"] == "random_forest" else 1 <= rounds <= 500))
    check(report, label + "_exact_five_artifact_paths", set(record["artifact_sha256"]) == {rel(path) for path in artifacts})
    for path in artifacts:
        digest = record["artifact_sha256"][rel(path)]
        check(report, label + "_artifact_" + path.parent.name, sha256(path) == digest)
        if item["origin"] == "reused":
            check(report, label + "_prior_" + path.parent.name,
                prior_digest(prior_reports[source], source, path) == digest == report["source_control_hashes"][rel(path)])
            source_bind(report, path, digest)
        else:
            bind(report, out, path)
    if item["origin"] == "reused":
        source_bind(report, checkpoint, prior_digest(prior_reports[source], source, checkpoint))
        for stage, path in (("inner", artifacts[1]), ("final", artifacts[2])):
            report["reused_models_sha256"][rel(path)] = sha256(path)
            report["reused_" + stage + "_models_hash_verified"] += 1
    else:
        check(report, label + "_new_trainer_binding", record["trainer_sha256"] == report["trainer_sha256"])
        bind(report, out, checkpoint)
    calibration = pd.read_parquet(artifacts[3])
    prediction = pd.read_parquet(artifacts[4])
    for frame, rows, role in ((calibration, cal, "calibration"), (prediction, test, "outer_test")):
        allowed = {"row_index", "account_id", "y_true", "y_score"}
        if role == "outer_test": allowed |= {"y_pred_fixed", "y_pred_inner"}
        expected_columns = allowed if "account_id" in frame else allowed - {"account_id"}
        check(report, label + "_" + role + "_truth_identity_rows_and_probability_domain", set(frame) == expected_columns
            and ("account_id" in frame or source == ORIGINAL)
            and np.array_equal(frame.row_index, rows) and frame.row_index.is_unique
            and np.array_equal(frame.y_true, y[rows]) and np.isfinite(frame.y_score).all()
            and frame.y_score.between(0, 1).all()
            and ("account_id" not in frame or np.array_equal(frame.account_id, identity.account_id.to_numpy()[rows])))
    cutoff, inner_f1 = threshold_from_pr_curve(calibration.y_true, calibration.y_score)
    report["threshold_max_abs_diff"] = max(report["threshold_max_abs_diff"], close(report,
        label + "_independent_inner_f1_and_lowest_threshold", [record["inner_threshold"], record["inner_f1"]], [cutoff, inner_f1]))
    for key, value in (("inner_ap", average_precision_score(calibration.y_true, calibration.y_score)),
                       ("inner_auc", roc_auc_score(calibration.y_true, calibration.y_score))):
        if item["origin"] == "newly_fitted" or key in record:
            close(report, label + "_" + key, record[key], value)
    if item["origin"] == "newly_fitted":
        rescore_rf(report, out, artifacts[1], X, cal, calibration.y_score.to_numpy(), state, "inner")
        rescore_rf(report, out, artifacts[2], X, test, prediction.y_score.to_numpy(), state, "final")
    check(report, label + "_two_policies", len(record["metrics"]) == 2
        and {row["threshold_policy"] for row in record["metrics"]} == set(POLICIES))
    rows = []
    for policy, threshold, pred_column in ((POLICIES[0], .5, "y_pred_fixed"), (POLICIES[1], cutoff, "y_pred_inner")):
        predicted = (prediction.y_score.to_numpy() >= threshold).astype(np.int8)
        check(report, label + "_" + policy + "_binary_predictions", np.array_equal(predicted, prediction[pred_column]))
        values = metric_values(prediction.y_true, prediction.y_score, predicted)
        row = dict(dataset=item["dataset"], model=item["model"], feature_set=item["feature_set"], seed=seed, fold=fold,
            threshold_policy=policy, threshold=threshold, n_features=len(columns), selected_rounds=rounds,
            source_origin=item["origin"], source_run=item["source_run"], **values,
            n_test=len(prediction), n_positive=int(prediction.y_true.sum()))
        saved = next(value for value in record["metrics"] if value["threshold_policy"] == policy)
        expected_saved = {**row, "feature_set": item["source_feature_set"]}
        check(report, label + "_" + policy + "_saved_metric_metadata", set(saved).issubset(expected_saved)
            and all(saved[key] == expected_saved[key] for key in FOLD_KEYS)
            and set(METRICS[:8] + ["tn", "fp", "fn", "tp", "threshold", "n_test", "n_positive"]).issubset(saved))
        names = [name for name in saved if name not in FOLD_KEYS and name not in ("source_origin", "source_run")]
        report["metric_max_abs_diff"] = max(report["metric_max_abs_diff"], close(report,
            label + "_" + policy + "_all_saved_metrics", [saved[name] for name in names], [expected_saved[name] for name in names]))
        if "source_origin" in saved:
            check(report, label + "_new_source_provenance", saved["source_origin"] == item["origin"] and saved["source_run"] == item["source_run"])
        rows.append(row)
    return rows, len(prediction), len(calibration)


def verify_dataset(out, dataset):
    report = skeleton(dataset)
    try:
        protocol, settings, prior_reports = protocol_audit(report, out)
        worker_path = out / dataset / "run_manifest_random_forest.json"
        worker = read(worker_path)
        check(report, "new_rf_worker_complete_registered_50_each", worker["status"] == "complete"
            and worker["dataset"] == dataset and worker["model"] == "random_forest"
            and worker["protocol_sha256"] == report["protocol_sha256"]
            and worker["models_expected"] == worker["inner_models_expected"] == 50
            and worker["models_completed"] == worker["inner_models_completed"] == 50
            and worker["metric_rows"] == 100 and worker["input_hashes_unchanged"] is True
            and datetime.fromisoformat(protocol["registered_at"]) <= datetime.fromisoformat(worker["started_at"]))
        check(report, "worker_readonly_input_register", worker["inputs_sha256"] == read(out / "dataset_inputs.json")[dataset]["input_sha256"])
        matrix, y, identity, splits = load_data(report, out, dataset)
        rows = []; pred_count = 0; cal_count = 0
        for item in (value for value in settings if value["dataset"] == dataset):
            X = np.ascontiguousarray(matrix[:, :item["n_features"]]) if item["origin"] == "newly_fitted" else None
            for seed in range(5):
                for fold in range(5):
                    values, n_pred, n_cal = verify_fold(report, out, item, seed, fold, X, y, identity, splits, prior_reports)
                    rows.extend(values); pred_count += n_pred; cal_count += n_cal
            print(f"Independent {dataset} {item['model']} {item['feature_set']}: {item['origin']}, 25 folds", flush=True)
            del X
        frame = pd.DataFrame(rows)
        check(report, "all_dataset_setting_and_new_reused_counts", len(frame) == 450 and not frame.duplicated(FOLD_KEYS).any()
            and report["new_final_models_verified"] == report["new_inner_models_verified"] == 50
            and report["reused_final_models_hash_verified"] == report["reused_inner_models_hash_verified"] == 175
            and len(report["new_models_sha256"]) == len(report["model_probability_max_abs_diff"]) == 100
            and len(report["reused_models_sha256"]) == 350)
        new_rows = frame[(frame.model == "random_forest") & frame.feature_set.isin(["gmke44", "gmkep56"])]
        compare_table(report, out, f"{dataset}/fold_metrics_random_forest.csv", new_rows, FOLD_KEYS)
        if (out / dataset / "fold_metrics.csv").exists():
            compare_table(report, out, f"{dataset}/fold_metrics.csv", frame, FOLD_KEYS)
        path = out / dataset / "verification_fold_metrics.csv"
        frame.to_csv(path, index=False)
        bind(report, out, path); bind(report, out, worker_path)
        for name, digest in report["source_hashes"].items(): check(report, "final_source_" + name, sha256(contained(ROOT, name)) == digest)
        report.update(passed=True, status="passed", n_models_verified=100, n_reused_models_hash_verified=350,
            total_models_checked=450, reported_final_models_verified=225, reported_inner_models_verified=225,
            metric_rows_verified=450, saved_fold_predictions_recomputed=225, reused_saved_fold_predictions_recomputed=175,
            n_prediction_rows=pred_count, n_calibration_rows=cal_count, original_inputs_sha256_verified=True,
            checks_passed_total=len(report["checks"]))
    except Exception as error:
        report.update(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    write(out / dataset / "validation.json", report)
    print(json.dumps({key: report.get(key) for key in ("dataset", "passed", "n_models_verified", "n_reused_models_hash_verified", "error")}), flush=True)
    return report


def absorb(report, out, item, dataset):
    check(report, dataset + "_passed_same_frozen_verifier_and_protocol", item["passed"] is True and item["status"] == "passed"
        and item["validator_sha256"] == sha256(__file__) and item["protocol_sha256"] == report["protocol_sha256"]
        and item["trainer_sha256"] == report["trainer_sha256"] and item["helper_sha256"] == report["helper_sha256"]
        and item["source_control_hashes"] == report["source_control_hashes"]
        and item["original_inputs_sha256_verified"] is True and all(item["checks"].values()) and bool(item["checks"])
        and item["new_final_models_verified"] == item["new_inner_models_verified"] == 50
        and item["reused_final_models_hash_verified"] == item["reused_inner_models_hash_verified"] == 175
        and item["n_models_verified"] == 100 and item["n_reused_models_hash_verified"] == 350
        and item["total_models_checked"] == 450 and item["reported_final_models_verified"] == item["reported_inner_models_verified"] == 225
        and len(item["new_models_sha256"]) == len(item["model_probability_max_abs_diff"]) == 100
        and len(item["reused_models_sha256"]) == 350)
    for key, digest in item["source_hashes"].items(): source_bind(report, contained(ROOT, key), digest)
    for key, digest in item["result_hashes"].items():
        path = contained(out, key)
        check(report, dataset + "_verified_result_" + key, sha256(path) == digest)
        bind(report, out, path)
    for key in ("new_models_sha256", "reused_models_sha256", "model_probability_max_abs_diff"):
        check(report, dataset + "_distinct_" + key, not set(report[key]).intersection(item[key]))
        report[key].update(item[key])
    for key in ("new_final_models_verified", "new_inner_models_verified", "reused_final_models_hash_verified", "reused_inner_models_hash_verified"):
        report[key] += item[key]
    for key in ("metric_max_abs_diff", "threshold_max_abs_diff"):
        report[key] = max(report[key], item[key])


def aggregate(out):
    report = skeleton("all")
    try:
        protocol, settings, prior_reports = protocol_audit(report, out)
        workers = []; frames = []; pred_count = 0; cal_count = 0
        for dataset in DATASETS:
            validation_path = out / dataset / "validation.json"
            worker = read(validation_path)
            absorb(report, out, worker, dataset); bind(report, out, validation_path)
            frame = pd.read_csv(out / dataset / "verification_fold_metrics.csv")
            compare_table(report, out, f"{dataset}/fold_metrics.csv", frame, FOLD_KEYS)
            manifest_path = out / dataset / "run_manifest.json"
            manifest = read(manifest_path)
            check(report, dataset + "_training_and_reference_counts", manifest["status"] in ("trained_pending_independent_validation", "complete")
                and manifest["dataset"] == dataset and manifest["protocol_sha256"] == report["protocol_sha256"]
                and all(manifest[key] == value for key, value in dict(new_final_models=50, new_inner_models=50,
                    reused_final_models=175, reused_inner_models=175, reported_final_models=225, reported_inner_models=225, metric_rows=450).items())
                and manifest["worker_manifest_sha256"] == sha256(out / dataset / "run_manifest_random_forest.json"))
            bind(report, out, manifest_path)
            workers.append(worker); frames.append(frame)
            pred_count += worker["n_prediction_rows"]; cal_count += worker["n_calibration_rows"]
        folds = pd.concat(frames, ignore_index=True)
        compare_table(report, out, "fold_metrics.csv", folds, FOLD_KEYS)
        for metric in ("auc", "ap"):
            pivot = folds.pivot(index=FOLD_KEYS[:-1], columns="threshold_policy", values=metric)
            close(report, metric + "_same_score_across_policies", pivot[POLICIES[0]], pivot[POLICIES[1]], 0)
        summary = independent_summary(folds)
        compare_table(report, out, "summary.csv", summary, SUMMARY_KEYS)
        pairs = independent_pairs(folds, ["dataset", "model", "seed", "fold", "threshold_policy"])
        compare_table(report, out, "paired_fold_differences.csv", pairs, ["dataset", "model", "seed", "fold", "threshold_policy", "contrast_id"])
        paired_summary = independent_pair_summary(pairs)
        compare_table(report, out, "paired_summary.csv", paired_summary, PAIR_KEYS)
        seed_rows = []; pooled_count = 0
        for item in settings:
            identity = pd.read_parquet(out / item["dataset"] / "accounts.parquet")
            for seed in range(5):
                pieces = []
                for fold in range(5):
                    source, stem, checkpoint, artifacts = setting_paths(item, seed, fold)
                    prediction = pd.read_parquet(artifacts[4])
                    if "account_id" not in prediction:
                        check(report, stem + "_legacy_identity_mapping", source == ORIGINAL)
                        prediction["account_id"] = identity.account_id.to_numpy()[prediction.row_index.to_numpy()]
                    pieces.append(prediction)
                pooled = pd.concat(pieces, ignore_index=True).sort_values("row_index")
                label = f"{item['dataset']}_{item['model']}_{item['feature_set']}_seed{seed}"
                check(report, label + "_OOF_all_accounts_once", len(pooled) == len(identity) and pooled.row_index.is_unique
                    and np.array_equal(pooled.row_index, identity.row_index) and np.array_equal(pooled.account_id, identity.account_id)
                    and np.array_equal(pooled.y_true, identity.label))
                for policy, column in ((POLICIES[0], "y_pred_fixed"), (POLICIES[1], "y_pred_inner")):
                    seed_rows.append(dict(dataset=item["dataset"], model=item["model"], feature_set=item["feature_set"], seed=seed,
                        threshold_policy=policy, **metric_values(pooled.y_true, pooled.y_score, pooled[column]), n_samples=len(pooled),
                        n_positive=int(pooled.y_true.sum()), source_origin=item["origin"], source_run=item["source_run"]))
                pooled_count += len(pooled)
            print(f"Independent pooled OOF {item['dataset']} {item['model']} {item['feature_set']}", flush=True)
        seeds = pd.DataFrame(seed_rows)
        seed_keys = ["dataset", "model", "feature_set", "seed", "threshold_policy"]
        compare_table(report, out, "seed_oof_metrics.csv", seeds, seed_keys)
        seed_pairs = independent_pairs(seeds, ["dataset", "model", "seed", "threshold_policy"])
        compare_table(report, out, "paired_seed_oof_differences.csv", seed_pairs,
            ["dataset", "model", "seed", "threshold_policy", "contrast_id"])
        check(report, "all_expected_six_table_rows", [len(folds), len(summary), len(pairs), len(paired_summary), len(seeds), len(seed_pairs)]
            == [900, 36, 900, 36, 180, 180])
        manifest = read(out / "run_manifest.json")
        check(report, "root_counts_and_protocol_without_manifest_hash_cycle", manifest["status"] in ("trained_pending_independent_validation", "complete")
            and manifest["protocol_sha256"] == report["protocol_sha256"] and manifest["trainer_sha256"] == report["trainer_sha256"]
            and manifest["helper_sha256"] == report["helper_sha256"] and manifest["all_settings_newly_fitted"] is False
            and all(manifest[key] == value for key, value in dict(new_final_models=100, new_inner_models=100,
                reused_final_models=350, reused_inner_models=350, reported_final_models=450, reported_inner_models=450,
                settings=18, metric_rows=900, summary_rows=36, paired_fold_rows=900, paired_summary_rows=36,
                seed_oof_rows=180, paired_seed_oof_rows=180).items()))
        check(report, "new_models_rescored_reused_models_hash_checked_exact_counts", len(report["new_models_sha256"])
            == len(report["model_probability_max_abs_diff"]) == 200 and len(report["reused_models_sha256"]) == 700
            and report["new_final_models_verified"] == report["new_inner_models_verified"] == 100
            and report["reused_final_models_hash_verified"] == report["reused_inner_models_hash_verified"] == 350)
        report.update(passed=True, status="passed", n_models_verified=200, n_reused_models_hash_verified=700, total_models_checked=900,
            reported_final_models_verified=450, reported_inner_models_verified=450, metric_rows_verified=900, summary_rows_verified=36,
            paired_fold_rows_verified=900, paired_summary_rows_verified=36, seed_oof_metric_rows_verified=180, paired_seed_oof_rows_verified=180,
            saved_fold_predictions_recomputed=450, reused_saved_fold_predictions_recomputed=350,
            n_prediction_rows=pred_count, n_calibration_rows=cal_count, n_pooled_oof_rows=pooled_count,
            original_inputs_sha256_verified=True, checks_passed_total=len(report["checks"]) + sum(worker["checks_passed_total"] for worker in workers),
            root_manifest_hash_omitted_for_final_status_transition=True,
            interpretation_scope="Static whole-period, correlated account splits; newly rescored RF and previously independently validated controls; no independent significance or temporal-generalization claim")
    except Exception as error:
        report.update(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    write(out / "validation_models.json", report)
    print(json.dumps({key: report.get(key) for key in ("passed", "n_models_verified", "n_reused_models_hash_verified", "error")}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--preflight-only", action="store_true", help="Read-only protocol/source audit; no models loaded or report written")
    args = parser.parse_args(); out = args.out.resolve()
    if args.preflight_only:
        report = skeleton("preflight")
        protocol_audit(report, out)
        if args.dataset: load_data(report, out, args.dataset)
        print(json.dumps({"passed": True, "preflight_only": True, "checks": len(report["checks"]), "validator_sha256": sha256(__file__)}))
        return
    if args.aggregate_only:
        if args.dataset: parser.error("Aggregate does not accept --dataset")
        result = aggregate(out)
    else:
        if not args.dataset: parser.error("Select --dataset or --aggregate-only")
        result = verify_dataset(out, args.dataset)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
