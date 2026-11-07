"""Frozen 16/44/56-column IBM Small comparison; only missing RF arms fit anew.

Historical verified models are referenced, never overwritten or represented as
new fits. Both datasets share the established scientific selection protocol.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import run_account_feature_mechanism as common

ROOT = Path(__file__).resolve().parents[2]
ALT = ROOT / "Altman_experiments"
ORIGINAL = ALT / "results/account_original_multiclassifier_20261007_isolated"
FEATURE_SOURCE = ALT / "results/account_s10_increment_20261008"
DEFAULT_OUT = ALT / "results/account_gmkep56_multiclassifier_20261008"
DATASETS = ("hi_small", "li_small")
MODELS = ("lightgbm", "xgboost", "random_forest")
ARMS = ("base16", "gmke44", "gmkep56")
NEW_ARMS = ("gmke44", "gmkep56")
POLICIES = ("fixed_0.5", "inner_max_f1")
METRICS = (*common.METRICS, "fpr", "fnr", "tn", "fp", "fn", "tp")
sha256, rel, write_json, now = common.sha256, common.rel, common.write_json, common.now
KEYS = ["dataset", "model", "feature_set", "seed", "fold", "threshold_policy"]


def read_json(path):
    return json.loads(Path(path).read_text())


def contrasts():
    return [
        dict(contrast_id="gmke_vs_base", contrast_type="structural", lhs_arm="gmke44", rhs_arm="base16"),
        dict(contrast_id="ph_vs_gmke", contrast_type="barcode", lhs_arm="gmkep56", rhs_arm="gmke44"),
        dict(contrast_id="full_vs_base", contrast_type="full_dictionary", lhs_arm="gmkep56", rhs_arm="base16"),
    ]


def bound_digest(report, source, path):
    relative = str(Path(path).relative_to(source))
    return report.get("result_hashes", {}).get(relative) or report.get("models_sha256", {}).get(relative) or report.get("models_sha256", {}).get(rel(path))


def setting_source(out, dataset, model, arm):
    if arm == "base16":
        return ORIGINAL, "base", "reused", "validation.json"
    if model != "random_forest":
        return FEATURE_SOURCE, arm, "reused", "validation_models.json"
    return out, arm, "newly_fitted", None


def load_inputs(dataset, columns):
    path = FEATURE_SOURCE / dataset / "features.parquet"
    table = pd.read_parquet(path, columns=["account_id", "label", *columns["gmkep56"]])
    identity = pd.read_parquet(ORIGINAL / dataset / "accounts.parquet")
    assert table.account_id.is_unique and not table.isna().any().any()
    np.testing.assert_array_equal(identity.row_index, np.arange(len(table)))
    np.testing.assert_array_equal(table.account_id, identity.account_id)
    np.testing.assert_array_equal(table.label, identity.label)
    label_path = ALT / "data" / dataset / ("account_labels.parquet" if dataset == "hi_small" else "li_small_labels.parquet")
    labels = pd.read_parquet(label_path)
    pd.testing.assert_frame_equal(table[["account_id", "label"]], labels)
    y = table.label.to_numpy(dtype=np.int8)
    expected = (515088, 6357) if dataset == "hi_small" else (705907, 5304)
    assert (len(y), int(y.sum())) == expected and set(np.unique(y)) == {0, 1}
    for tag, names in (("base", columns["base16"]), ("ftd", columns["gmke44"][16:])):
        old = pd.read_parquet(ALT / "results" / dataset / "features" / f"{tag}_features.parquet")
        np.testing.assert_array_equal(old.account_id, table.account_id)
        np.testing.assert_array_equal(old[names].to_numpy(dtype=np.float32), table[names].to_numpy(dtype=np.float32))
        del old
    matrix = np.ascontiguousarray(table[columns["gmkep56"]].to_numpy(dtype=np.float32))
    assert np.isfinite(matrix).all() and matrix.shape == (expected[0], 56)
    del table, labels
    gc.collect()
    paths = [path, label_path, ORIGINAL / dataset / "accounts.parquet", FEATURE_SOURCE / dataset / "feature_manifest.json",
             FEATURE_SOURCE / dataset / "feature_qc.json", ALT / "results" / dataset / "features/base_features.parquet",
             ALT / "results" / dataset / "features/ftd_features.parquet"]
    return matrix, y, identity, paths


def prepare(out):
    assert not out.exists(), "Register in a new output directory"
    prior = read_json(ORIGINAL / "protocol.json")
    feature_protocol = read_json(FEATURE_SOURCE / "protocol.json")
    reports = {ORIGINAL: read_json(ORIGINAL / "validation.json"), FEATURE_SOURCE: read_json(FEATURE_SOURCE / "validation_models.json")}
    assert all(report["passed"] is True for report in reports.values())
    assert all(read_json(source / "run_manifest.json")["status"] == "complete" for source in reports)
    scientific = ["seeds", "outer_n_splits", "outer_split", "inner_calibration_fraction", "inner_random_state", "model_random_state",
                  "parameter_candidates_per_model", "early_stopping", "class_weighting", "threshold_policies", "threshold_selection", "prediction_rule", "summary_std_ddof"]
    assert all(prior[name] == feature_protocol[name] for name in scientific)
    assert all(prior["model_params"][name] == feature_protocol["model_params"][name] for name in ("lightgbm", "xgboost"))
    assert prior["seeds"] == list(range(5)) and prior["outer_n_splits"] == 5
    assert all(prior["model_params"][name]["n_jobs"] == 6 for name in MODELS)
    names56 = feature_protocol["feature_columns"]["gmkep56"]
    columns = {"base16": names56[:16], "gmke44": feature_protocol["feature_columns"]["gmke44"], "gmkep56": names56}
    assert [len(columns[arm]) for arm in ARMS] == [16, 44, 56]
    assert len(set(names56)) == 56 and names56[:44] == columns["gmke44"]
    registered = {name: prior[name] for name in scientific}
    registered.update(experiment="account_gmkep56_multiclassifier", run_date="2026-10-08", registered_at=now(),
        datasets=list(DATASETS), models=list(MODELS), feature_columns=columns, feature_sets={arm: len(columns[arm]) for arm in ARMS},
        feature_blocks={name: values for name, values in feature_protocol["feature_blocks"].items() if name not in ("Q6", "S10", "S9")},
        model_params=prior["model_params"], new_model="random_forest", new_arms=list(NEW_ARMS), contrasts=contrasts(),
        source_original_run=rel(ORIGINAL), source_feature_run=rel(FEATURE_SOURCE), primary_metrics=["ap"], all_metrics=list(METRICS),
        trainer_sha256=sha256(__file__), helper_sha256=sha256(common.__file__),
        original_protocol_sha256=sha256(ORIGINAL / "protocol.json"), feature_protocol_sha256=sha256(FEATURE_SOURCE / "protocol.json"),
        environment={"python": platform.python_version(), "python_executable": sys.executable, "logical_cpus": os.cpu_count(),
                     **{name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn", "joblib", "pyarrow", "lightgbm", "xgboost")}},
        expected_new_final_models=100, expected_new_inner_models=100, expected_reused_final_models=350, expected_reused_inner_models=350,
        expected_reported_final_models=450, expected_reported_inner_models=450,
        table_counts={"fold": 900, "summary": 36, "paired_fold": 900, "paired_summary": 36, "seed": 180, "paired_seed": 180},
        source_reuse="14 verified settings referenced by path and SHA256; only four missing RF44/56 settings newly fitted",
        evaluation_scope="Whole-period graph and random stratified accounts: static transductive development recognition, not temporal prediction or independent final confirmation",
        interpretation="Fixed presets, not equal computation budgets or individually optimized classifiers; 56 columns are not 56 independent information dimensions",
        rf_sqrt_candidates={"base16": 4, "gmke44": 6, "gmkep56": 7},
        ph_definition="Current uniformly sampled cap40 P12u; finite F2 VR2/3-hop H0/H1 maximum/total lifetime and Shannon lifetime entropy; unchanged geometry",
        ph_scope="No Q6 in main dictionary; PH conditional prediction increment cannot establish scale-independent topology; 50/62 controls remain separately archived",
        decision_rule="Report all six scenarios and paired seed OOF directions descriptively; no post-hoc filter, p-value, or new confirmation/optimality claim")
    reused = {}; inputs = {}; split_hashes = {}; settings = []
    cache = {}
    def digest(path):
        key = rel(path)
        if key not in cache: cache[key] = sha256(path)
        return cache[key]
    for source, report in reports.items():
        assert report["protocol_sha256"] == digest(source / "protocol.json")
        for name in ("protocol.json", "validation.json" if source == ORIGINAL else "validation_models.json", "run_manifest.json"):
            reused[rel(source / name)] = digest(source / name)
    for name in ("feature_protocol.json", "validation_math.json", "validation_features.json", "design_alias_validation.json"):
        path = FEATURE_SOURCE / name
        assert bound_digest(reports[FEATURE_SOURCE], FEATURE_SOURCE, path) == digest(path)
        if name.startswith("validation_"): assert read_json(path)["passed"] is True
        reused[rel(path)] = digest(path)
    out.mkdir()
    for filename in ("EXPERIMENT_RESULTS_SUMMARY.md", "EXPERIMENT_RESULTS_SUMMARY_gen.py"):
        if (ROOT / filename).is_file():
            shutil.copy2(ROOT / filename, out / ("before_" + filename))
    shutil.copy2(__file__, out / "trainer_snapshot.py")
    for dataset in DATASETS:
        folder = out / dataset
        folder.mkdir()
        (folder / "splits").mkdir()
        matrix, y, identity, input_paths = load_inputs(dataset, columns)
        source_cache = FEATURE_SOURCE / dataset / "features.parquet"
        assert bound_digest(reports[FEATURE_SOURCE], FEATURE_SOURCE, source_cache) == digest(source_cache)
        assert bound_digest(reports[ORIGINAL], ORIGINAL, ORIGINAL / dataset / "accounts.parquet") == digest(ORIGINAL / dataset / "accounts.parquet")
        inputs[dataset] = {"input_sha256": {rel(path): digest(path) for path in input_paths}, "feature_source": rel(source_cache),
                           "feature_columns": names56, "n_samples": len(y), "n_positive": int(y.sum())}
        identity.to_parquet(folder / "accounts.parquet", index=False, compression="zstd")
        outer_splits = {}
        for seed in range(5):
            outer = StratifiedKFold(5, shuffle=True, random_state=seed)
            for fold, (train, test) in enumerate(outer.split(np.zeros(len(y)), y)):
                inner = StratifiedShuffleSplit(1, test_size=0.2, random_state=10000 + 100 * seed + fold)
                fit, cal = next(inner.split(np.zeros(len(train)), y[train]))
                expected = {"outer_train": train, "outer_test": test, "inner_fit": train[fit], "inner_calibration": train[cal]}
                source = ORIGINAL / dataset / "splits" / f"seed{seed}_fold{fold}.npz"
                assert bound_digest(reports[ORIGINAL], ORIGINAL, source) == digest(source)
                s10 = FEATURE_SOURCE / dataset / "splits" / source.name
                assert digest(s10) == digest(source)
                with np.load(source) as saved:
                    assert set(saved.files) == set(expected)
                    for key in expected: np.testing.assert_array_equal(saved[key], expected[key])
                shutil.copy2(source, folder / "splits" / source.name)
                split_hashes[rel(source)] = digest(source)
                reused[rel(source)] = digest(source)
                reused[rel(s10)] = digest(s10)
                outer_splits[(seed, fold)] = source
        for model in MODELS:
            for arm in ARMS:
                source, source_arm, origin, validator = setting_source(out, dataset, model, arm)
                settings.append({"dataset": dataset, "model": model, "feature_set": arm, "n_features": len(columns[arm]),
                                 "origin": origin, "source_run": rel(source), "source_feature_set": source_arm, "source_validation": validator})
                if origin != "reused": continue
                report = reports[source]
                for seed in range(5):
                    for fold in range(5):
                        stem = f"{model}_{source_arm}_seed{seed}_fold{fold}"
                        checkpoint = source / dataset / "folds" / f"{stem}.json"
                        assert bound_digest(report, source, checkpoint) == digest(checkpoint)
                        saved = read_json(checkpoint)
                        assert saved["protocol_sha256"] == digest(source / "protocol.json")
                        assert saved["n_features"] == len(columns[arm])
                        assert saved["dataset"] == dataset and saved["model"] == model and saved["feature_set"] == source_arm
                        assert saved["seed"] == seed and saved["fold"] == fold and len(saved["metrics"]) == 2
                        if "feature_columns" in saved: assert saved["feature_columns"] == columns[arm]
                        reused[rel(checkpoint)] = digest(checkpoint)
                        for name, value in saved["artifact_sha256"].items():
                            path = ROOT / name
                            assert bound_digest(report, source, path) == value == digest(path), f"Source binding mismatch: {name}"
                            reused[name] = value
        del matrix, y, identity
        gc.collect()
    write_json(out / "dataset_inputs.json", inputs)
    write_json(out / "source_settings.json", settings)
    write_json(out / "source_split_hashes.json", split_hashes)
    write_json(out / "reused_artifact_hashes.json", reused)
    for filename in ("dataset_inputs.json", "source_settings.json", "source_split_hashes.json", "reused_artifact_hashes.json"):
        registered[filename.removesuffix(".json") + "_sha256"] = sha256(out / filename)
    write_json(out / "protocol.json", registered)
    write_json(out / "run_manifest.json", {"experiment": registered["experiment"], "status": "registered", "registered_at": now(),
        "protocol_sha256": sha256(out / "protocol.json"), "new_final_models": 0, "new_inner_models": 0,
        "reused_final_models": 350, "reused_inner_models": 350, "reported_final_models": 350, "reported_inner_models": 350})
    print(json.dumps({"prepared": rel(out), "settings": len(settings), "source_artifacts_bound": len(reused), "new_fits": 200}), flush=True)


def check_registered(out):
    protocol = read_json(out / "protocol.json")
    assert protocol["trainer_sha256"] == sha256(__file__) and protocol["helper_sha256"] == sha256(common.__file__)
    for filename in ("dataset_inputs.json", "source_settings.json", "source_split_hashes.json", "reused_artifact_hashes.json"):
        assert protocol[filename.removesuffix(".json") + "_sha256"] == sha256(out / filename)
    return protocol


def metric_record(y, score, cutoff):
    predicted, values = common.metrics(y, score, cutoff)
    values.update(fpr=1 - values["specificity"], fnr=1 - values["recall"])
    return predicted, values


def run(out, dataset):
    protocol = check_registered(out)
    folder = out / dataset
    for name in ("models", "inner_models", "predictions", "calibration", "folds"):
        (folder / name).mkdir(exist_ok=True)
    matrix, y, identity, paths = load_inputs(dataset, protocol["feature_columns"])
    before = {rel(path): sha256(path) for path in paths}
    assert before == read_json(out / "dataset_inputs.json")[dataset]["input_sha256"]
    manifest_path = folder / "run_manifest_random_forest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {"dataset": dataset, "model": "random_forest",
        "protocol_sha256": sha256(out / "protocol.json"), "started_at": now(), "inputs_sha256": before}
    assert manifest["protocol_sha256"] == sha256(out / "protocol.json") and manifest["inputs_sha256"] == before
    manifest.update(status="running", pid=os.getpid(), command=[sys.executable, *sys.argv], models_expected=50, inner_models_expected=50)
    write_json(manifest_path, manifest)
    records = []; started = time.monotonic()
    with (folder / "run_random_forest.log").open("a") as log, contextlib.redirect_stdout(common.Tee(sys.stdout, log)), contextlib.redirect_stderr(common.Tee(sys.stderr, log)):
        try:
            for arm in NEW_ARMS:
                names = protocol["feature_columns"][arm]
                X = np.ascontiguousarray(matrix[:, :len(names)])
                for seed in range(5):
                    for fold in range(5):
                        split = folder / "splits" / f"seed{seed}_fold{fold}.npz"
                        assert sha256(split) == read_json(out / "source_split_hashes.json")[rel(ORIGINAL / dataset / "splits" / split.name)]
                        with np.load(split) as saved: arrays = {key: saved[key] for key in saved.files}
                        train, test, fit, cal = (arrays[key] for key in ("outer_train", "outer_test", "inner_fit", "inner_calibration"))
                        stem = f"random_forest_{arm}_seed{seed}_fold{fold}"
                        checkpoint = folder / "folds" / f"{stem}.json"
                        if checkpoint.exists():
                            saved = read_json(checkpoint)
                            assert saved["protocol_sha256"] == sha256(out / "protocol.json") and saved["feature_columns"] == names
                            assert (saved["dataset"], saved["model"], saved["feature_set"], saved["seed"], saved["fold"]) == (dataset, "random_forest", arm, seed, fold)
                            for path, digest in saved["artifact_sha256"].items(): assert sha256(ROOT / path) == digest
                            assert len(saved["metrics"]) == 2
                            records.extend(saved["metrics"])
                        else:
                            tick = time.monotonic(); random_state = 100 * seed + fold
                            print(f"{now()} START {dataset} {stem}", flush=True)
                            inner = common.fit_model("random_forest", X, y, fit, cal, protocol, random_state)
                            cal_score = inner.predict_proba(X[cal])[:, 1].astype(np.float64)
                            assert np.isfinite(cal_score).all() and ((cal_score >= 0) & (cal_score <= 1)).all()
                            cutoff, inner_f1 = common.best_f1_threshold(y[cal], cal_score)
                            inner_path = folder / "inner_models" / f"{stem}.joblib"
                            joblib.dump(inner, inner_path, compress=3)
                            calibration_path = folder / "calibration" / f"{stem}.parquet"
                            calibration = identity.iloc[cal][["row_index", "account_id"]].copy()
                            calibration["y_true"] = y[cal]; calibration["y_score"] = cal_score
                            calibration.to_parquet(calibration_path, index=False, compression="zstd")
                            del inner; gc.collect()
                            final = common.fit_model("random_forest", X, y, train, None, protocol, random_state, 200)
                            score = final.predict_proba(X[test])[:, 1].astype(np.float64)
                            assert np.isfinite(score).all() and ((score >= 0) & (score <= 1)).all()
                            final_path = folder / "models" / f"{stem}.joblib"
                            joblib.dump(final, final_path, compress=3)
                            del final; gc.collect()
                            prediction_path = folder / "predictions" / f"{stem}.parquet"
                            predictions = identity.iloc[test][["row_index", "account_id"]].copy()
                            predictions["y_true"] = y[test]; predictions["y_score"] = score
                            rows = []
                            for policy, threshold in (("fixed_0.5", 0.5), ("inner_max_f1", cutoff)):
                                predicted, values = metric_record(y[test], score, threshold)
                                predictions["y_pred_fixed" if policy == "fixed_0.5" else "y_pred_inner"] = predicted
                                rows.append(dict(dataset=dataset, model="random_forest", feature_set=arm, seed=seed, fold=fold,
                                    threshold_policy=policy, threshold=threshold, n_features=len(names), selected_rounds=200,
                                    source_origin="newly_fitted", source_run=rel(out), **values))
                            predictions.to_parquet(prediction_path, index=False, compression="zstd")
                            artifacts = [split, inner_path, final_path, calibration_path, prediction_path]
                            completed = dict(dataset=dataset, model="random_forest", feature_set=arm, seed=seed, fold=fold,
                                feature_columns=names, n_features=len(names), protocol_sha256=sha256(out / "protocol.json"), trainer_sha256=sha256(__file__),
                                model_random_state=random_state, inner_random_state=10000 + 100 * seed + fold,
                                n_outer_train=len(train), n_outer_test=len(test), n_inner_fit=len(fit), n_inner_calibration=len(cal),
                                outer_train_positive=int(y[train].sum()), outer_test_positive=int(y[test].sum()), inner_fit_positive=int(y[fit].sum()),
                                inner_calibration_positive=int(y[cal].sum()), selected_rounds=200, inner_threshold=cutoff, inner_f1=inner_f1,
                                inner_ap=float(average_precision_score(y[cal], cal_score)), inner_auc=float(roc_auc_score(y[cal], cal_score)),
                                final_fit_has_eval_set=False, artifact_sha256={rel(path): sha256(path) for path in artifacts}, runtime_seconds=time.monotonic() - tick, metrics=rows)
                            write_json(checkpoint, completed)
                            records.extend(rows)
                            print(f"{now()} DONE {stem} AP={rows[0]['ap']:.6f} elapsed={completed['runtime_seconds']:.1f}s ({len(records)//2}/50)", flush=True)
                        pd.DataFrame(records).to_csv(folder / "fold_metrics_random_forest.csv", index=False)
                        manifest.update(models_completed=len(records)//2, inner_models_completed=len(records)//2,
                                        metric_rows=len(records), updated_at=now(), runtime_seconds=time.monotonic()-started)
                        write_json(manifest_path, manifest)
                del X; gc.collect()
            assert len(records) == 100 and not pd.DataFrame(records).duplicated(KEYS).any()
            assert before == {rel(path): sha256(path) for path in paths}
            manifest.update(status="complete", completed_at=now(), input_hashes_unchanged=True, runtime_seconds=time.monotonic()-started)
            write_json(manifest_path, manifest)
        except BaseException:
            manifest.update(status="failed", failed_at=now(), traceback=traceback.format_exc())
            write_json(manifest_path, manifest)
            raise


def differences(frame, keys):
    parts = []
    for spec in contrasts():
        lhs = frame[frame.feature_set.eq(spec["lhs_arm"])].set_index(keys).sort_index()
        rhs = frame[frame.feature_set.eq(spec["rhs_arm"])].set_index(keys).sort_index()
        assert lhs.index.is_unique and lhs.index.equals(rhs.index)
        part = lhs.index.to_frame(index=False)
        for key, value in spec.items(): part[key] = value
        for metric in METRICS:
            part[f"lhs_{metric}"] = lhs[metric].to_numpy()
            part[f"rhs_{metric}"] = rhs[metric].to_numpy()
            part[f"delta_{metric}"] = lhs[metric].to_numpy() - rhs[metric].to_numpy()
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def summarize(out):
    protocol = check_registered(out)
    for path, digest in read_json(out / "reused_artifact_hashes.json").items(): assert sha256(ROOT / path) == digest, path
    settings = read_json(out / "source_settings.json")
    fold_rows = []; seed_rows = []
    for dataset in DATASETS:
        worker = read_json(out / dataset / "run_manifest_random_forest.json")
        assert worker["status"] == "complete" and worker["models_completed"] == worker["inner_models_completed"] == 50
        identity = pd.read_parquet(out / dataset / "accounts.parquet")
        dataset_rows = []
        for setting in [item for item in settings if item["dataset"] == dataset]:
            source = ROOT / setting["source_run"]
            model, arm, source_arm = setting["model"], setting["feature_set"], setting["source_feature_set"]
            for seed in range(5):
                pieces = []
                for fold in range(5):
                    stem = f"{model}_{source_arm}_seed{seed}_fold{fold}"
                    saved = read_json(source / dataset / "folds" / f"{stem}.json")
                    for path, digest in saved["artifact_sha256"].items(): assert sha256(ROOT / path) == digest
                    for record in saved["metrics"]:
                        values = dict(record)
                        values.update(feature_set=arm, source_origin=setting["origin"], source_run=setting["source_run"], n_features=setting["n_features"],
                                      selected_rounds=saved["selected_rounds"], fpr=1-values["specificity"], fnr=1-values["recall"])
                        dataset_rows.append(values)
                    prediction = pd.read_parquet(source / dataset / "predictions" / f"{stem}.parquet")
                    if "account_id" not in prediction:
                        prediction["account_id"] = identity.account_id.to_numpy()[prediction.row_index.to_numpy()]
                    pieces.append(prediction)
                pooled = pd.concat(pieces, ignore_index=True).sort_values("row_index")
                assert pooled.row_index.is_unique
                np.testing.assert_array_equal(pooled.row_index, identity.row_index)
                np.testing.assert_array_equal(pooled.account_id, identity.account_id)
                np.testing.assert_array_equal(pooled.y_true, identity.label)
                truth = pooled.y_true.to_numpy(); score = pooled.y_score.to_numpy()
                for policy, name in (("fixed_0.5", "y_pred_fixed"), ("inner_max_f1", "y_pred_inner")):
                    pred = pooled[name].to_numpy(dtype=np.int8)
                    tp = int(np.sum((truth == 1) & (pred == 1))); fp = int(np.sum((truth == 0) & (pred == 1)))
                    fn = int(np.sum((truth == 1) & (pred == 0))); tn = int(np.sum((truth == 0) & (pred == 0)))
                    precision = tp/(tp+fp) if tp+fp else 0.; recall = tp/(tp+fn); specificity = tn/(tn+fp)
                    seed_rows.append(dict(dataset=dataset, model=model, feature_set=arm, seed=seed, threshold_policy=policy,
                        precision=precision, recall=recall, f1=2*tp/(2*tp+fp+fn), auc=float(roc_auc_score(truth, score)),
                        ap=float(average_precision_score(truth, score)), accuracy=(tp+tn)/len(truth), balanced_accuracy=(recall+specificity)/2,
                        specificity=specificity, fpr=1-specificity, fnr=1-recall, tp=tp, fp=fp, fn=fn, tn=tn,
                        n_samples=len(truth), n_positive=int(truth.sum()), source_origin=setting["origin"], source_run=setting["source_run"]))
        dataset_frame = pd.DataFrame(dataset_rows)
        assert len(dataset_frame) == 450 and not dataset_frame.duplicated(KEYS).any()
        dataset_frame.to_csv(out / dataset / "fold_metrics.csv", index=False)
        fold_rows.extend(dataset_rows)
        write_json(out / dataset / "run_manifest.json", dict(dataset=dataset, status="trained_pending_independent_validation",
            protocol_sha256=sha256(out / "protocol.json"), new_final_models=50, new_inner_models=50, reused_final_models=175,
            reused_inner_models=175, reported_final_models=225, reported_inner_models=225, metric_rows=450,
            worker_manifest_sha256=sha256(out / dataset / "run_manifest_random_forest.json")))
    folds = pd.DataFrame(fold_rows); seeds = pd.DataFrame(seed_rows)
    pairs = differences(folds, ["dataset", "model", "seed", "fold", "threshold_policy"])
    seed_pairs = differences(seeds, ["dataset", "model", "seed", "threshold_policy"])
    summaries = []
    group_keys = ["dataset", "model", "feature_set", "threshold_policy"]
    for key, group in folds.groupby(group_keys, sort=True):
        row = dict(zip(group_keys, key), n_folds=len(group), n_features=int(group.n_features.iloc[0]), source_origin=group.source_origin.iloc[0], source_run=group.source_run.iloc[0])
        assert len(group) == 25
        for metric in (*METRICS, "threshold"):
            row[f"{metric}_mean"] = float(group[metric].mean()); row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        summaries.append(row)
    paired = []
    group_keys = ["dataset", "model", "contrast_id", "threshold_policy", "contrast_type", "lhs_arm", "rhs_arm"]
    for key, group in pairs.groupby(group_keys, sort=True):
        row = dict(zip(group_keys, key), n_pairs=len(group))
        assert len(group) == 25
        for metric in METRICS:
            delta = group[f"delta_{metric}"]
            row[f"delta_{metric}_mean"] = float(delta.mean()); row[f"delta_{metric}_std"] = float(delta.std(ddof=1))
            row[f"{metric}_positive_pairs"] = int((delta > 0).sum())
        paired.append(row)
    tables = {"fold_metrics.csv": folds, "summary.csv": pd.DataFrame(summaries), "paired_fold_differences.csv": pairs,
              "paired_summary.csv": pd.DataFrame(paired), "seed_oof_metrics.csv": seeds, "paired_seed_oof_differences.csv": seed_pairs}
    assert [len(frame) for frame in tables.values()] == [900, 36, 900, 36, 180, 180]
    for filename, frame in tables.items(): frame.to_csv(out / filename, index=False)
    write_json(out / "run_manifest.json", dict(experiment=protocol["experiment"], status="trained_pending_independent_validation", completed_at=now(),
        protocol_sha256=sha256(out / "protocol.json"), trainer_sha256=sha256(__file__), helper_sha256=sha256(common.__file__),
        new_final_models=100, new_inner_models=100, reused_final_models=350, reused_inner_models=350, reported_final_models=450,
        reported_inner_models=450, all_settings_newly_fitted=False, settings=18, metric_rows=900, summary_rows=36,
        paired_fold_rows=900, paired_summary_rows=36, seed_oof_rows=180, paired_seed_oof_rows=180,
        result_hashes={filename: sha256(out / filename) for filename in tables},
        worker_manifests_sha256={rel(out / dataset / "run_manifest_random_forest.json"): sha256(out / dataset / "run_manifest_random_forest.json") for dataset in DATASETS}))
    print(json.dumps({"summarized": rel(out), "settings": 18, "new_final": 100, "new_inner": 100, "reused_final": 350, "reused_inner": 350}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("prepare", "run", "summarize"), required=True)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out.resolve()
    assert out.is_relative_to(ROOT)
    if args.phase == "prepare": prepare(out)
    elif args.phase == "run":
        if not args.dataset: parser.error("run requires --dataset")
        run(out, args.dataset)
    else: summarize(out)


if __name__ == "__main__":
    main()
