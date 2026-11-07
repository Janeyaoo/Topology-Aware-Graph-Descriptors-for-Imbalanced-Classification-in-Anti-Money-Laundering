"""Read-only PH size diagnostics for complete account-level experiments.

No fitting, threshold selection, graph construction, or feature extraction occurs.
Original validated nested-tuning results and independently validated complete new
tree settings can be stratified. H1 block permutation uses only frozen original
56-column final models. Original evidence is never overwritten.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
MAIN = ROOT / "Altman_experiments/results/account_gmkep56_tuned_20261008"
DESIGN = ROOT / "Altman_experiments/results/eswa_supplement_design_20261013"
DEFAULT_OUT = ROOT / "Altman_experiments/results/account_eswa_supplement_20261013"
DATASETS = ("hi_small", "li_small")
MODELS = ("lightgbm", "xgboost", "random_forest")
MAIN_ARMS = ("base16", "gmke44", "gmkep56")
H1 = np.array([47, 48, 49, 53, 54, 55], dtype=np.int32)
METRICS = ("ap", "auc", "f1", "precision", "recall", "accuracy", "balanced_accuracy",
           "recall_at_0.5pct", "recall_at_1pct", "recall_at_2pct")
PERM_SEEDS = list(range(10))
COHORT_COUNTS = {"hi_small": (84007, 2267), "li_small": (112154, 1890)}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def rel(path):
    return str(Path(path).resolve().relative_to(ROOT))


def path(name):
    result = (ROOT / name).resolve()
    require(result.is_relative_to(ROOT) and result.is_file(), f"Missing project source: {name}")
    return result


def read(file):
    return json.loads(Path(file).read_text())


def write(file, value):
    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    temp = file.with_suffix(file.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(file)


def csv(file, values):
    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    temp = file.with_suffix(file.suffix + ".tmp")
    pd.DataFrame(values).to_csv(temp, index=False)
    temp.replace(file)


def npz(file, **values):
    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    temp = file.with_suffix(file.suffix + ".tmp")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, **values)
    temp.replace(file)


def check_bindings(mapping):
    for name, digest in mapping.items():
        require(sha(path(name)) == digest, f"Source changed: {name}")


def group_indices(geometry):
    """Geometry-only exact pairs; stable source account order within each layer."""
    require(geometry.ndim == 2 and geometry.shape[1] == 2, "Geometry shape")
    key = geometry[:, 0].astype(np.int64) * 41 + geometry[:, 1]
    order = np.argsort(key, kind="stable")
    cuts = np.r_[0, np.flatnonzero(np.diff(key[order])) + 1, len(order)]
    return [(int(key[order[a]] // 41), int(key[order[a]] % 41), order[a:b])
            for a, b in zip(cuts[:-1], cuts[1:])]


def donors(geometry, permutation_seed):
    """Joint H1 donor positions, using no label, model, score, or outcome."""
    rng = np.random.Generator(np.random.PCG64(permutation_seed))
    result = np.arange(len(geometry), dtype=np.uint32)
    for _, _, positions in group_indices(geometry):
        result[positions] = rng.permutation(positions).astype(np.uint32)
    return result


def metric_values(y, score, predicted, rows):
    y, score, predicted, rows = map(np.asarray, (y, score, predicted, rows))
    require(len(y) > 0 and len(y) == len(score) == len(predicted) == len(rows), "Metric identity")
    require(np.isfinite(score).all() and np.isin(y, [0, 1]).all(), "Metric inputs")
    positive = int(y.sum()); negative = len(y) - positive
    tp = int(np.sum((y == 1) & (predicted == 1)))
    fp = int(np.sum((y == 0) & (predicted == 1)))
    fn, tn = positive - tp, negative - fp
    precision = tp / (tp + fp) if tp + fp else 0.
    recall = tp / positive if positive else None
    specificity = tn / negative if negative else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.
    result = dict(n_samples=len(y), n_positive=positive, n_negative=negative,
                  positive_fraction=positive / len(y), tn=tn, fp=fp, fn=fn, tp=tp,
                  ap=float(average_precision_score(y, score)) if positive else None,
                  auc=float(roc_auc_score(y, score)) if positive and negative else None,
                  f1=f1, precision=precision, recall=recall,
                  accuracy=(tp + tn) / len(y),
                  balanced_accuracy=(recall + specificity) / 2 if positive and negative else None,
                  status="both_classes" if positive and negative else "no_positive" if not positive else "no_negative")
    order = np.lexsort((rows, -score))
    for budget, label in ((.005, "0.5pct"), (.01, "1pct"), (.02, "2pct")):
        k = int(np.ceil(budget * len(y)))
        result["recall_at_" + label] = int(y[order[:k]].sum()) / positive if positive else None
    return result


def weighted_ap(y, score, geometry, train_geometry):
    """An average of per-test-layer AP, weighted by outer TRAIN geometry only.

    AP is defined when a test layer has at least one positive. All-positive
    layers have AP=1 and are retained; AUROC's stricter condition is irrelevant
    here. No-positive and absent test layers have undefined AP and are excluded
    with both training-mass and test-account coverage explicitly reported.
    """
    training = {(a, b): len(pos) for a, b, pos in group_indices(train_geometry)}
    total = len(train_geometry)
    numerator, effective = 0., 0
    covered, positive_covered = 0, 0
    details = []
    seen = set()
    for a, b, positions in group_indices(geometry):
        seen.add((a, b))
        weight_count = training.get((a, b), 0)
        positives = int(y[positions].sum())
        ap = float(average_precision_score(y[positions], score[positions])) if positives else None
        valid = positives > 0 and weight_count > 0
        if valid:
            numerator += weight_count * ap
            effective += weight_count
            covered += len(positions)
            positive_covered += positives
        details.append(dict(n2=a, n3=b, n_test=len(positions), n_positive=positives,
                            n_outer_train=weight_count, outer_train_frequency=weight_count / total,
                            ap=ap, included_in_weighted_ap=valid,
                            definition_status="defined_positive_containing" if positives else "undefined_no_positive"))
    for (a, b), weight_count in sorted(training.items()):
        if (a, b) not in seen:
            details.append(dict(n2=a, n3=b, n_test=0, n_positive=0, n_outer_train=weight_count,
                                outer_train_frequency=weight_count / total, ap=None,
                                included_in_weighted_ap=False, definition_status="undefined_absent_test_stratum"))
    return dict(stratified_weighted_ap=numerator / effective if effective else None,
                defined_train_frequency=effective / total,
                defined_test_account_fraction=covered / len(y),
                defined_test_positive_fraction=positive_covered / int(y.sum()) if y.sum() else None,
                n_defined_strata=sum(d["included_in_weighted_ap"] for d in details),
                n_test_strata=len(seen), n_train_only_strata=len(details) - len(seen),
                weight_source="outer_train_geometry_only",
                average_before_outer_fold_aggregation=True), details


class Diagnostics:
    def __init__(self, out):
        self.out = Path(out).resolve()
        self.dest = self.out / "ph_diagnostics"
        require(self.out.is_relative_to(ROOT), "Output remains in project")
        self.protocol = self.dest / "protocol.json"

    def prepare(self):
        if self.protocol.exists():
            return self.registered()
        main_validation = read(MAIN / "validation_models.json")
        require(main_validation["passed"] is True and read(MAIN / "run_manifest.json")["status"] == "complete",
                "Original main results complete and independently validated")
        sources = [DESIGN / "design_manifest.json", DESIGN / "protocol_design.json", DESIGN / "feature_registry.json",
                   DESIGN / "ph_algebra_and_size_audit.json", DESIGN / "ph_ablation_audit.md",
                   ROOT / "ESWA_SUPPLEMENT_EXPERIMENT_DESIGN_20261013.md",
                   MAIN / "protocol.json", MAIN / "dataset_inputs.json", MAIN / "validation_models.json",
                   ROOT / "Altman_experiments/code/eswa_supplementary_analysis_20261012.py"]
        inputs = read(MAIN / "dataset_inputs.json")
        for ds in DATASETS:
            check_bindings(inputs[ds]["cache_sha256"])
            sources.extend(path(name) for name in inputs[ds]["cache_sha256"])
            qp = self.out / "inputs" / ds
            qmanifest = read(qp / "q10_input_manifest.json")
            require(read(qp / "q10_validation.json")["status"] == "passed", "Independent Q10 acceptance")
            check_bindings(qmanifest["output_sha256"])
            sources.extend([qp / "q10_input_manifest.json", qp / "q10_validation.json", qp / "q10.parquet"])
            sources.extend(MAIN / ds / "splits" / f"seed{s}_fold{f}.npz" for s, f in itertools.product(range(5), repeat=2))
        protocol = dict(status="registered_before_diagnostic_scores", created_at=now(),
                        script_sha256=sha(SCRIPT), source_sha256={rel(p): sha(p) for p in sources},
                        datasets=list(DATASETS), models=list(MODELS), seeds=list(range(5)), folds=list(range(5)),
                        strategy="tuned", original_arms=list(MAIN_ARMS),
                        allowed_tree_arms=[a["arm_id"] for a in read(DESIGN / "feature_registry.json")["tree_arms"]
                                           if a["status"] in ("planned", "reuse_completed_verified")],
                        existing_reference_contrasts=[dict(contrast_id="existing_struct44_vs_base16", lhs="gmke44", rhs="base16"),
                                                      dict(contrast_id="existing_full56_vs_base16", lhs="gmkep56", rhs="base16")],
                        primary_cohort=[40, 40], exact_strata="all exact (n2,n3), including undefined and sparse layers",
                        weighted_ap="each outer test fold separately; geometry frequencies of that fold's outer_train; renormalize defined positive-containing test layers; mean/SD of 25 values",
                        permutation=dict(count=10, seeds=PERM_SEEDS, generator="numpy.random.Generator(PCG64)",
                                         seed_integer_freeze="execution detail, fixed before any diagnostic scores; design froze count only",
                                         strata_order="ascending exact (n2,n3)", within_stratum_order="ascending source account row",
                                         block_columns=H1.tolist(), original_arm="gmkep56",
                                         scope="within each outer test fold; joint six-column block; no refit/no new threshold",
                                         limitations="model dependence only; breaks H1 correlations with GMKE; not causal and not independent significance"),
                        threshold_policies=["inner_max_f1", "fixed_0.5"],
                        alarm_budgets=[.005, .01, .02], alarm_rounding="ceil", alarm_ties="score descending then source account row ascending",
                        score_verification="original rescoring must be byte-exact before permutation", max_cpu_threads=2,
                        model_storage="read old model only, no model copy; compressed scores and deterministic donor rows retained",
                        incomplete_setting_rule="no aggregate until all 25 folds valid; complete five-fold OOF per seed",
                        conclusion_language="average increment and descriptive consistency; synthetic Small and repeated same-data exploration limitations")
        self.dest.mkdir(parents=True, exist_ok=True)
        snapshot = self.dest / "code_snapshot" / SCRIPT.name
        snapshot.parent.mkdir(exist_ok=True)
        snapshot.write_bytes(SCRIPT.read_bytes())
        write(self.protocol, protocol)
        self.probe()
        return protocol

    def registered(self):
        protocol = read(self.protocol)
        require(protocol["script_sha256"] == sha(SCRIPT), "Diagnostic code changed after registration")
        check_bindings(protocol["source_sha256"])
        return protocol

    def dataset(self, ds):
        folder = MAIN / ds
        X = np.load(folder / "arrays/features.npy", mmap_mode="r")
        y = np.load(folder / "arrays/labels.npy", mmap_mode="r")
        ids = pd.read_parquet(folder / "accounts.parquet", columns=["row_index", "account_id"])
        q = pd.read_parquet(self.out / "inputs" / ds / "q10.parquet", columns=["account_id", "q2_n", "q3_n"])
        require(np.array_equal(ids.row_index, np.arange(len(y))) and np.array_equal(ids.account_id, q.account_id), "Geometry/account alignment")
        geometry = q[["q2_n", "q3_n"]].to_numpy(np.int16)
        require(np.all((geometry >= 1) & (geometry <= 40)) and np.all(geometry[:, 0] <= geometry[:, 1]), "Frozen cap/path sizes")
        cohort = np.all(geometry == 40, axis=1)
        require((int(cohort.sum()), int(y[cohort].sum())) == COHORT_COUNTS[ds], "Prespecified fixed-size cohort")
        # H0 total is exactly n-1 in original float32 encoding. All six H0
        # summaries in the prespecified 40/40 cohort must be constant.
        require(np.array_equal(X[:, 45], geometry[:, 0] - 1) and np.array_equal(X[:, 51], geometry[:, 1] - 1), "H0/geometry algebra")
        for col in [44, 45, 46, 50, 51, 52]:
            require(np.unique(X[cohort, col]).size == 1, "H0 constant in fixed-size cohort")
        del q
        return X, y, geometry

    def fold(self, ds, name, arm, seed, fold):
        original = arm in MAIN_ARMS
        folder = MAIN if original else self.out / "trees"
        file = folder / ds / "folds" / f"{name}_{arm}_seed{seed}_fold{fold}.json"
        data = read(file)
        bindings = {rel(file): sha(file)}
        if not original:
            receipt = self.out / "validation/tree" / ds / file.name
            validation = read(receipt)
            require(validation["passed"] is True and validation["fold_done_sha256"] == sha(file), "Supplementary fold independently verified")
            bindings[rel(receipt)] = sha(receipt)
        stage = data["strategies"]["tuned"]
        prediction = path(stage["prediction_path"])
        require(sha(prediction) == stage["artifact_sha256"][rel(prediction)], "Original prediction hash")
        bindings[rel(prediction)] = sha(prediction)
        split = path(data["split_path"])
        if not original:
            require(sha(split) == sha(MAIN / ds / "splits" / f"seed{seed}_fold{fold}.npz"), "Same original split bytes")
        bindings[rel(split)] = sha(split)
        with np.load(split) as z:
            train, test = z["outer_train"].copy(), z["outer_test"].copy()
        values = pd.read_parquet(prediction).sort_values("row_index")
        require(np.array_equal(values.row_index.to_numpy(), np.sort(test)), "Prediction covers original outer test")
        return data, stage, values, train, bindings

    def stratify(self, ds, name, arm):
        protocol = self.registered()
        require(arm in protocol["allowed_tree_arms"], "Only frozen core tree settings are eligible")
        destination = self.dest / "stratification" / ds / f"{name}_{arm}"
        complete = destination / "validation.json"
        if complete.exists():
            result = read(complete); check_bindings(result["source_sha256"]); check_bindings(result["output_sha256"])
            require(result["protocol_sha256"] == sha(self.protocol), "Cached diagnostic registration")
            return result
        X, y, geometry = self.dataset(ds)
        fold_rows, fold_details, seed_rows, strata_rows, bindings = [], [], [], [], {}
        for seed in range(5):
            scores = np.full(len(y), np.nan, dtype=np.float64)
            predicted = {p: np.full(len(y), -1, dtype=np.int8) for p in ["inner_max_f1", "fixed_0.5"]}
            visits = np.zeros(len(y), dtype=np.uint8)
            for f in range(5):
                _, stage, values, train, bound = self.fold(ds, name, arm, seed, f)
                bindings.update(bound)
                rows = values.row_index.to_numpy(np.int32)
                score = values.y_score.to_numpy(np.float64)
                require(np.array_equal(values.y_true, y[rows]), "Prediction truth identity")
                require(not visits[rows].any(), "OOF no repeated test account")
                visits[rows] += 1; scores[rows] = score
                summary, details = weighted_ap(y[rows], score, geometry[rows], geometry[train])
                key = dict(dataset=ds, model=name, feature_set=arm, strategy="tuned", seed=seed, fold=f)
                fold_rows.append(dict(**key, **summary))
                fold_details.extend(dict(**key, **item) for item in details)
                for policy, cutoff, field in (("inner_max_f1", stage["threshold"], "y_pred_inner"), ("fixed_0.5", .5, "y_pred_fixed")):
                    require(np.array_equal(values[field], score >= cutoff), "Inherited threshold predictions")
                    predicted[policy][rows] = values[field].to_numpy(np.int8)
            require(np.all(visits == 1) and np.isfinite(scores).all(), "Full five-fold OOF")
            cohort = np.flatnonzero(np.all(geometry == 40, axis=1))
            for policy in predicted:
                key = dict(dataset=ds, model=name, feature_set=arm, strategy="tuned", seed=seed, threshold_policy=policy)
                seed_rows.append(dict(**key, cohort="n2_eq_n3_eq40", **metric_values(y[cohort], scores[cohort], predicted[policy][cohort], cohort)))
                for n2, n3, rows in group_indices(geometry):
                    strata_rows.append(dict(**key, n2=n2, n3=n3, sparse_lt20=len(rows) < 20,
                                            **metric_values(y[rows], scores[rows], predicted[policy][rows], rows)))
        destination.mkdir(parents=True, exist_ok=True)
        tables = {"outer_fold_weighted_ap.csv": fold_rows, "outer_fold_strata_details.csv": fold_details,
                  "seed_oof_cohort40.csv": seed_rows, "seed_oof_exact_strata.csv": strata_rows}
        for filename, rows in tables.items():
            csv(destination / filename, rows)
        result = dict(passed=True, status="complete_stratification_25fold_5oof", completed_at=now(),
                      dataset=ds, model=name, feature_set=arm, strategy="tuned", protocol_sha256=sha(self.protocol),
                      source_sha256=bindings, output_sha256={rel(destination / name): sha(destination / name) for name in tables},
                      n_outer_folds=25, n_complete_seed_oof=5, n_accounts=len(y),
                      cohort40_accounts=COHORT_COUNTS[ds][0], cohort40_positives=COHORT_COUNTS[ds][1],
                      no_model_fits=True, no_new_thresholds=True, no_p_values=True)
        write(complete, result)
        del X, y, geometry; gc.collect()
        return result

    def aggregate_stratification(self):
        """Report complete settings only, with five paired OOF cohort values."""
        protocol = self.registered()
        sources, cohorts, weighted = {}, [], []
        for receipt_file in sorted((self.dest / "stratification").glob("*/*/validation.json")):
            receipt = read(receipt_file)
            require(receipt["passed"] and receipt["n_outer_folds"] == 25 and receipt["n_complete_seed_oof"] == 5,
                    "Stratification summary uses complete settings only")
            require(receipt["protocol_sha256"] == sha(self.protocol), "Stratification protocol unchanged")
            check_bindings(receipt["source_sha256"]); check_bindings(receipt["output_sha256"])
            sources[rel(receipt_file)] = sha(receipt_file)
            cohorts.extend(pd.read_csv(receipt_file.parent / "seed_oof_cohort40.csv", float_precision="round_trip").to_dict("records"))
            weighted.extend(pd.read_csv(receipt_file.parent / "outer_fold_weighted_ap.csv", float_precision="round_trip").to_dict("records"))
        require(len(cohorts) > 0, "No complete stratification settings available")
        summary_dir = self.dest / "stratification_summary"
        cf, wf = pd.DataFrame(cohorts), pd.DataFrame(weighted)
        ckeys = ["dataset", "model", "feature_set", "strategy", "threshold_policy", "cohort"]
        csummary = cf.groupby(ckeys)[list(METRICS)].agg(["mean", "std"])
        csummary.columns = ["_".join(c) for c in csummary.columns]
        csummary = csummary.reset_index()
        csummary["n_complete_oof_seeds"] = 5
        wkeys = ["dataset", "model", "feature_set", "strategy"]
        wcols = ["stratified_weighted_ap", "defined_train_frequency", "defined_test_account_fraction", "defined_test_positive_fraction"]
        wsummary = wf.groupby(wkeys)[wcols].agg(["mean", "std"])
        wsummary.columns = ["_".join(c) for c in wsummary.columns]
        wsummary = wsummary.reset_index(); wsummary["n_outer_folds"] = 25
        registry = read(DESIGN / "feature_registry.json")
        contrasts = sum((registry[k] for k in ["module_contrasts", "ph_primary_contrasts", "ph_secondary_contrasts", "external_tree_contrasts"]), []) + protocol["existing_reference_contrasts"]
        paired_cohorts, paired_weights = [], []
        complete = set(cf[["dataset", "model", "feature_set"]].itertuples(index=False, name=None))
        skipped = []
        for ds, model, contrast in itertools.product(DATASETS, MODELS, contrasts):
            lhs, rhs = contrast["lhs"], contrast["rhs"]
            if (ds, model, lhs) not in complete or (ds, model, rhs) not in complete:
                skipped.append(dict(dataset=ds, model=model, contrast_id=contrast["contrast_id"], reason="both complete settings required"))
                continue
            a = cf[(cf.dataset == ds) & (cf.model == model) & (cf.feature_set == lhs)]
            b = cf[(cf.dataset == ds) & (cf.model == model) & (cf.feature_set == rhs)]
            joined = a.merge(b, on=["dataset", "model", "seed", "threshold_policy", "cohort"], suffixes=("_lhs", "_rhs"), validate="one_to_one")
            require(len(joined) == 10, "Five paired cohort OOF values per policy")
            for _, record in joined.iterrows():
                item = dict(dataset=ds, model=model, contrast_id=contrast["contrast_id"], lhs=lhs, rhs=rhs,
                            seed=int(record.seed), threshold_policy=record.threshold_policy, cohort=record.cohort)
                for metric in METRICS:
                    item["delta_" + metric] = float(record[metric + "_lhs"] - record[metric + "_rhs"])
                paired_cohorts.append(item)
            aa = wf[(wf.dataset == ds) & (wf.model == model) & (wf.feature_set == lhs)]
            bb = wf[(wf.dataset == ds) & (wf.model == model) & (wf.feature_set == rhs)]
            joined = aa.merge(bb, on=["dataset", "model", "seed", "fold"], suffixes=("_lhs", "_rhs"), validate="one_to_one")
            require(len(joined) == 25, "Twenty-five paired stratified AP values")
            for _, record in joined.iterrows():
                require(abs(record.defined_train_frequency_lhs - record.defined_train_frequency_rhs) < 1e-15, "Pair has common layer weights")
                paired_weights.append(dict(dataset=ds, model=model, contrast_id=contrast["contrast_id"], lhs=lhs, rhs=rhs,
                                           seed=int(record.seed), fold=int(record.fold),
                                           delta_stratified_weighted_ap=float(record.stratified_weighted_ap_lhs - record.stratified_weighted_ap_rhs)))
        tables = {"cohort40_seed_values.csv": cf, "cohort40_summary.csv": csummary,
                  "stratified_weighted_ap_fold_values.csv": wf, "stratified_weighted_ap_summary.csv": wsummary,
                  "cohort40_paired_seed_deltas.csv": paired_cohorts, "stratified_weighted_ap_paired_fold_deltas.csv": paired_weights}
        if paired_cohorts:
            frame = pd.DataFrame(paired_cohorts)
            keys = ["dataset", "model", "contrast_id", "lhs", "rhs", "threshold_policy", "cohort"]
            columns = ["delta_" + m for m in METRICS]
            paired = frame.groupby(keys)[columns].agg(["mean", "std"])
            paired.columns = ["_".join(c) for c in paired.columns]
            paired = paired.reset_index()
            direction = frame.groupby(keys).delta_ap.agg(ap_positive_oof_seeds=lambda v: int((v > 1e-12).sum())).reset_index()
            tables["cohort40_paired_summary.csv"] = paired.merge(direction, on=keys)
        if paired_weights:
            frame = pd.DataFrame(paired_weights)
            keys = ["dataset", "model", "contrast_id", "lhs", "rhs"]
            paired = frame.groupby(keys).delta_stratified_weighted_ap.agg(mean="mean", sd="std", positive_folds=lambda v: int((v > 1e-12).sum())).reset_index()
            tables["stratified_weighted_ap_paired_summary.csv"] = paired
        for filename, table in tables.items():
            csv(summary_dir / filename, table)
        result = dict(passed=True, status="complete_settings_only", created_at=now(),
                      protocol_sha256=sha(self.protocol), complete_settings=len(complete),
                      expected_core_tree_settings=2 * 3 * len(protocol["allowed_tree_arms"]),
                      source_sha256=sources, output_sha256={rel(summary_dir / name): sha(summary_dir / name) for name in tables},
                      skipped_incomplete_comparisons=skipped, cohort_oof_repetitions=5, weighted_ap_outer_folds=25,
                      comparison_scope="Within the same cohort or exact-size weighted fold; never compare cohort AP directly to overall AP",
                      no_incomplete_setting_means=True, no_p_values=True)
        write(summary_dir / "validation.json", result)
        return result

    def permutation(self, ds, name, seed, f, probe_only=False):
        self.registered()
        destination = self.dest / "permutation" / ds / name
        stem = f"seed{seed}_fold{f}"
        done = destination / (stem + ".json")
        if done.exists() and not probe_only:
            record = read(done); check_bindings(record["source_sha256"]); check_bindings(record["output_sha256"])
            require(record["protocol_sha256"] == sha(self.protocol), "Permutation checkpoint protocol")
            return record
        X, y, geometry = self.dataset(ds)
        data, stage, values, _, bindings = self.fold(ds, name, "gmkep56", seed, f)
        model_path = path(stage["model_path"])
        require(sha(model_path) == stage["artifact_sha256"][rel(model_path)], "Frozen final model hash")
        bindings[rel(model_path)] = sha(model_path)
        model = joblib.load(model_path)
        require(model.n_features_in_ == 56, "Frozen 56-column model")
        model.n_jobs = 1 if name == "random_forest" else 2
        rows = values.row_index.to_numpy(np.int32)
        base = np.ascontiguousarray(X[rows])
        original = np.ascontiguousarray(model.predict_proba(base)[:, 1], dtype=np.float64)
        expected = values.y_score.to_numpy(np.float64)
        require(np.array_equal(original, expected), f"Original rescore not byte-exact: {name}, {np.max(np.abs(original - expected))}")
        require(np.array_equal(values.y_true, y[rows]), "Permutation label identity")
        if probe_only:
            record = dict(passed=True, status="original_model_rescore_probe_only", dataset=ds, model=name,
                          seed=seed, fold=f, n_test=len(rows), original_score_byte_exact=True,
                          source_sha256=bindings, protocol_sha256=sha(self.protocol),
                          score_sha256=array_sha(original), n_permutation_scores=0)
            write(self.dest / "qa" / f"model_probe_{ds}_{name}_seed{seed}_fold{f}.json", record)
            del model, X, base; gc.collect()
            return record
        scores = np.empty((10, len(rows)), dtype=np.float64)
        donor_rows = np.empty((10, len(rows)), dtype=np.uint32)
        metric_rows, coverage = [], []
        layers = group_indices(geometry[rows])
        variable = np.zeros(len(rows), dtype=bool)
        singleton = same_value = 0
        for _, _, positions in layers:
            if len(positions) == 1:
                singleton += 1
            if np.unique(base[positions][:, H1], axis=0).shape[0] == 1:
                same_value += 1
            else:
                variable[positions] = True
        coverage.append(dict(n_strata=len(layers), n_singleton_strata=singleton, n_same_h1_strata=same_value,
                             variable_strata_accounts=int(variable.sum()), variable_strata_positives=int(y[rows][variable].sum()),
                             variable_strata_account_fraction=float(variable.mean()),
                             variable_strata_positive_fraction=float(y[rows][variable].sum() / y[rows].sum())))
        original_metrics = {}
        for policy, cutoff in (("inner_max_f1", stage["threshold"]), ("fixed_0.5", .5)):
            original_metrics[policy] = metric_values(y[rows], original, original >= cutoff, rows)
        for permutation_index, permutation_seed in enumerate(PERM_SEEDS):
            donor = donors(geometry[rows], permutation_seed)
            require(np.array_equal(geometry[rows], geometry[rows][donor]), "Permutation exact pair preserved")
            require(np.array_equal(np.sort(donor), np.arange(len(rows))), "Joint donor bijection")
            modified = base.copy(); modified[:, H1] = base[donor][:, H1]
            keep = np.setdiff1d(np.arange(56), H1)
            require(np.array_equal(modified[:, keep], base[:, keep]), "GMKE and H0 unchanged")
            score = np.ascontiguousarray(model.predict_proba(modified)[:, 1], dtype=np.float64)
            require(np.isfinite(score).all() and np.all((score >= 0) & (score <= 1)), "Permutation finite probabilities")
            scores[permutation_index] = score; donor_rows[permutation_index] = rows[donor]
            changed = np.any(modified[:, H1] != base[:, H1], axis=1)
            for policy, cutoff in (("inner_max_f1", stage["threshold"]), ("fixed_0.5", .5)):
                values_permuted = metric_values(y[rows], score, score >= cutoff, rows)
                result = dict(dataset=ds, model=name, feature_set="gmkep56", strategy="tuned", seed=seed, fold=f,
                              permutation_seed=permutation_seed, threshold_policy=policy, threshold=float(cutoff),
                              changed_accounts=int(changed.sum()), changed_positives=int(y[rows][changed].sum()),
                              changed_account_fraction=float(changed.mean()),
                              changed_positive_fraction=float(y[rows][changed].sum() / y[rows].sum()), **values_permuted)
                for key in METRICS:
                    result["original_" + key] = original_metrics[policy][key]
                    result["delta_" + key] = original_metrics[policy][key] - values_permuted[key]
                metric_rows.append(result)
            print(json.dumps(dict(event="permutation_score", dataset=ds, model=name, seed=seed, fold=f, permutation_seed=permutation_seed)), flush=True)
            del modified
        score_file = destination / (stem + ".npz")
        metrics_file = destination / (stem + "_metrics.csv")
        npz(score_file, row_index=rows, y_true=y[rows], original_score=original,
            permutation_seeds=np.asarray(PERM_SEEDS, np.int16), permutation_score=scores,
            donor_row_index=donor_rows, original_threshold=np.float64(stage["threshold"]),
            n2n3=geometry[rows], h1_block_columns=H1)
        csv(metrics_file, metric_rows)
        record = dict(passed=True, status="complete_single_fold_10permutations_not_setting", completed_at=now(),
                      dataset=ds, model=name, feature_set="gmkep56", strategy="tuned", seed=seed, fold=f,
                      n_test=len(rows), n_positive=int(y[rows].sum()), permutation_seeds=PERM_SEEDS,
                      original_score_byte_exact=True, no_training=True, threshold_unchanged=True,
                      geometry_and_h0_unchanged=True, labelblind_donors=True, coverage=coverage[0],
                      protocol_sha256=sha(self.protocol), source_sha256=bindings,
                      output_sha256={rel(p): sha(p) for p in [score_file, metrics_file]})
        write(done, record)
        del X, model, base, scores, donor_rows; gc.collect()
        return record

    def aggregate_permutation(self, ds, name):
        self.registered()
        destination = self.dest / "permutation" / ds / name
        _, y, geometry = self.dataset(ds)
        fold_records, source_bindings, oof_rows = [], {}, []
        for seed in range(5):
            scores = np.full((10, len(y)), np.nan, dtype=np.float64)
            original = np.full(len(y), np.nan, dtype=np.float64)
            cuts = np.full(len(y), np.nan, dtype=np.float64)
            visits = np.zeros(len(y), dtype=np.uint8)
            for f in range(5):
                stem = f"seed{seed}_fold{f}"
                receipt_file = destination / (stem + ".json")
                receipt = read(receipt_file)
                require(receipt["passed"] and receipt["permutation_seeds"] == PERM_SEEDS and receipt["protocol_sha256"] == sha(self.protocol), "Complete registered permutation fold")
                check_bindings(receipt["source_sha256"]); check_bindings(receipt["output_sha256"])
                source_bindings[rel(receipt_file)] = sha(receipt_file)
                fold_records.extend(pd.read_csv(destination / (stem + "_metrics.csv"), float_precision="round_trip").to_dict("records"))
                with np.load(destination / (stem + ".npz")) as z:
                    rows = z["row_index"]
                    require(not visits[rows].any() and np.array_equal(z["y_true"], y[rows]), "Permutation OOF identities")
                    require(np.array_equal(z["n2n3"], geometry[rows]), "OOF geometry source")
                    original[rows] = z["original_score"]; scores[:, rows] = z["permutation_score"]
                    cuts[rows] = z["original_threshold"]; visits[rows] += 1
                    for p, pseed in enumerate(PERM_SEEDS):
                        require(np.array_equal(z["donor_row_index"][p], rows[donors(geometry[rows], pseed)]), "Independently reproduced donor identities")
            require(np.all(visits == 1) and np.isfinite(scores).all(), "Five complete permutation OOF folds")
            for cohort_name, mask in (("all_accounts", np.ones(len(y), bool)), ("n2_eq_n3_eq40", np.all(geometry == 40, axis=1))):
                rows = np.flatnonzero(mask)
                for policy, cutoff in (("inner_max_f1", cuts[rows]), ("fixed_0.5", .5)):
                    origin = metric_values(y[rows], original[rows], original[rows] >= cutoff, rows)
                    for p, pseed in enumerate(PERM_SEEDS):
                        metrics = metric_values(y[rows], scores[p, rows], scores[p, rows] >= cutoff, rows)
                        result = dict(dataset=ds, model=name, feature_set="gmkep56", strategy="tuned", seed=seed,
                                      permutation_seed=pseed, threshold_policy=policy, cohort=cohort_name, **metrics)
                        for key in METRICS:
                            result["original_" + key] = origin[key]; result["delta_" + key] = origin[key] - metrics[key]
                        oof_rows.append(result)
        csv(destination / "complete_fold_permutation_metrics.csv", fold_records)
        csv(destination / "complete_seed_oof_metrics.csv", oof_rows)
        folds = pd.DataFrame(fold_records)
        # First average over the 10 diagnostic permutations of the same fitted
        # model, then summarize the 25 outer folds; 250 scores are not 250 folds.
        columns = ["delta_" + k for k in METRICS]
        averaged = folds.groupby(["dataset", "model", "threshold_policy", "seed", "fold"])[columns].mean().reset_index()
        csv(destination / "complete_fold_mean_deltas.csv", averaged)
        summaries = []
        for policy, group in averaged.groupby("threshold_policy"):
            result = dict(dataset=ds, model=name, strategy="tuned", threshold_policy=policy, n_outer_folds=25, permutations_per_fold=10)
            for key in columns:
                result[key + "_mean"] = float(group[key].mean()); result[key + "_sd"] = float(group[key].std(ddof=1))
                result[key + "_positive_folds"] = int((group[key] > 1e-12).sum())
            summaries.append(result)
        csv(destination / "complete_summary.csv", summaries)
        oof = pd.DataFrame(oof_rows)
        oofkeys = ["dataset", "model", "threshold_policy", "cohort", "seed"]
        mean_oof = oof.groupby(oofkeys)[columns].mean().reset_index()
        csv(destination / "complete_seed_oof_mean_deltas.csv", mean_oof)
        summary_keys = ["dataset", "model", "threshold_policy", "cohort"]
        oof_summary = mean_oof.groupby(summary_keys)[columns].agg(["mean", "std"])
        oof_summary.columns = ["_".join(c) for c in oof_summary.columns]
        direction = mean_oof.groupby(summary_keys).delta_ap.agg(ap_positive_oof_seeds=lambda v: int((v > 1e-12).sum())).reset_index()
        oof_summary = oof_summary.reset_index().merge(direction, on=summary_keys)
        csv(destination / "complete_oof_summary.csv", oof_summary)
        outputs = [destination / name for name in ["complete_fold_permutation_metrics.csv", "complete_seed_oof_metrics.csv", "complete_fold_mean_deltas.csv", "complete_summary.csv", "complete_seed_oof_mean_deltas.csv", "complete_oof_summary.csv"]]
        result = dict(passed=True, status="complete_permutation_25fold_5oof", dataset=ds, model=name,
                      n_outer_folds=25, n_complete_oof_seeds=5, n_permutations_each_fold=10,
                      completed_at=now(), protocol_sha256=sha(self.protocol), source_sha256=source_bindings,
                      output_sha256={rel(p): sha(p) for p in outputs},
                      interpretation="Descriptive model dependence within exact size pairs; no causal attribution or independent significance")
        write(destination / "complete_validation.json", result)
        return result

    def probe(self):
        geometry = np.array([[2, 3], [2, 3], [3, 4], [3, 4], [3, 4], [1, 1]], dtype=np.int16)
        donor = donors(geometry, 0)
        require(np.array_equal(geometry, geometry[donor]) and np.array_equal(np.sort(donor), np.arange(6)), "Synthetic donor geometry/bijection")
        require(np.array_equal(donor, donors(geometry.copy(), 0)), "Deterministic labelblind donor")
        y = np.array([1, 0, 0, 1, 0, 0], np.int8)
        score = np.array([.9, .8, .7, .6, .5, .4])
        metrics = metric_values(y, score, score >= .6, np.arange(6))
        # AP = (1 + 2/4)/2; AUROC = 6/8; confusion=(2,2,0,2).
        require(metrics["ap"] == .75 and metrics["auc"] == .75 and metrics["tn"] == 2 and metrics["fp"] == 2 and metrics["fn"] == 0 and metrics["tp"] == 2, "Hand-calculated metric probe")
        train = np.array([[2, 3]] * 4 + [[3, 4]] * 2 + [[1, 1]] * 2 + [[5, 5]] * 2)
        weighted, details = weighted_ap(y, score, geometry, train)
        # Defined layers have training mass .4+.2; AP 1 and 1/2.
        require(abs(weighted["stratified_weighted_ap"] - 5 / 6) < 1e-15 and weighted["defined_train_frequency"] == .6, "Hand-calculated train-frequency weighted AP")
        require(metric_values(np.zeros(2, np.int8), np.array([.1, .2]), np.zeros(2, np.int8), np.array([0, 1]))["ap"] is None, "Undefined positive-free AP explicit")
        require(metric_values(np.ones(2, np.int8), np.array([.1, .2]), np.zeros(2, np.int8), np.array([0, 1]))["auc"] is None, "Undefined single-class AUROC explicit")
        # A tied alarm score chooses source row 3 over source row 9.
        tie = metric_values(np.array([0, 1]), np.array([.5, .5]), np.zeros(2), np.array([9, 3]))
        require(tie["recall_at_1pct"] == 1., "Alarm ties use original source account row")
        result = dict(passed=True, status="synthetic_preflight_only", created_at=now(), script_sha256=sha(SCRIPT),
                      checks=dict(donor_geometry_bijection=True, donors_deterministic_and_labelblind=True,
                                  manual_metrics=True, manual_outer_train_weighted_ap=True,
                                  undefined_layers_explicit=True, source_row_alarm_ties=True),
                      real_model_scores_evaluated=False)
        write(self.dest / "qa/preflight.json", result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "stratify", "aggregate-stratification", "probe-model", "permute", "aggregate-permutation"])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--arm", default="gmkep56")
    parser.add_argument("--seed", type=int, choices=range(5))
    parser.add_argument("--fold", type=int, choices=range(5))
    args = parser.parse_args()
    runner = Diagnostics(args.out)
    with threadpool_limits(limits=2):
        if args.action == "prepare":
            result = runner.prepare()
        elif args.action == "aggregate-stratification":
            result = runner.aggregate_stratification()
        else:
            require(args.dataset and args.model, "Dataset and model required")
            if args.action == "stratify":
                result = runner.stratify(args.dataset, args.model, args.arm)
            elif args.action == "aggregate-permutation":
                result = runner.aggregate_permutation(args.dataset, args.model)
            else:
                require(args.seed is not None and args.fold is not None, "Seed and fold required")
                result = runner.permutation(args.dataset, args.model, args.seed, args.fold, args.action == "probe-model")
    print(json.dumps({k: result[k] for k in ["status", "passed", "dataset", "model", "feature_set"] if k in result}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
