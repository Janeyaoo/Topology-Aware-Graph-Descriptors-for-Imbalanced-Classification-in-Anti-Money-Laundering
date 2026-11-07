"""Complete matched Graph32 controls without altering any earlier experiment.

Fixed RF uses the original fixed fitting implementation. Nested Graph32 uses
the frozen 16/44/56 search and strategy functions with identical candidates,
splits, random states, weighting, early stopping, and threshold selection.
"""
from __future__ import annotations

import argparse
import copy
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
import run_account_feature_mechanism as common
import run_account_gmkep_tuning as frozen

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "Altman_experiments/code"
RESULTS = ROOT / "Altman_experiments/results"
FIXED = RESULTS / "account_gmkep56_multiclassifier_20261008"
TUNED = RESULTS / "account_gmkep56_tuned_20261008"
MECHANISM = RESULTS / "account_feature_mechanism_20261007"
REUSE = RESULTS / "account_graph32_reuse_20261009"
OUT = RESULTS / "account_graph32_completion_20261009"
DATASETS = ("hi_small", "li_small")
MODELS = ("lightgbm", "xgboost", "random_forest")
ARM = "graph32"
sha, rel, jsave, now = common.sha256, common.rel, common.write_json, common.now
read = frozen.read


def source_gate():
    protected = {}
    for source in (FIXED, TUNED, MECHANISM):
        validation_path = source / ("validation_models.json" if source == TUNED else "validation.json")
        check = read(validation_path)
        assert check["passed"] is True
        assert read(source / "run_manifest.json")["status"] == "complete"
        for name in ("protocol.json", "fold_metrics.csv", "summary.csv", "seed_oof_metrics.csv",
                     validation_path.name, "run_manifest.json"):
            path = source / name
            expected = check.get("result_hashes", {}).get(name)
            if expected: assert sha(path) == expected, name
            protected[rel(path)] = sha(path)
        if (source / "validation_models.json").exists():
            assert read(source / "validation_models.json")["passed"] is True
            protected[rel(source / "validation_models.json")] = sha(source / "validation_models.json")
    audit = read(REUSE / "compatibility_audit.json")
    assert audit["passed"] is True
    for path, expected in audit["source_sha256"].items():
        assert sha(ROOT / path) == expected, path
        protected[path] = expected
    for path in (TUNED / "candidates.json", TUNED / "dataset_inputs.json", TUNED / "split_manifest.json",
                 REUSE / "compatibility_audit.json", Path(frozen.__file__), Path(common.__file__)):
        protected[rel(path)] = sha(path)
    tp = read(TUNED / "protocol.json")
    assert tp["trainer_sha256"] == sha(frozen.__file__)
    assert tp["helper_sha256"] == sha(common.__file__)
    assert tp["candidates_sha256"] == sha(TUNED / "candidates.json")
    versions = {k: importlib.metadata.version(k) for k in
                ("numpy", "pandas", "scikit-learn", "lightgbm", "xgboost", "joblib", "pyarrow")}
    assert all(tp["environment"][k] == v for k, v in versions.items())
    return protected, versions


def prepare(out):
    assert out.is_relative_to(RESULTS) and not out.exists(), "Use a new result directory"
    protected, versions = source_gate()
    fixed_prior, tuned_prior = read(FIXED / "protocol.json"), read(TUNED / "protocol.json")
    names = fixed_prior["feature_columns"]["gmke44"][:32]
    assert names == fixed_prior["feature_columns"]["base16"] + fixed_prior["feature_blocks"]["G16"]
    assert names == read(MECHANISM / "protocol.json")["feature_columns"]["base_graph_stats"]
    assert names == tuned_prior["feature_columns"]["gmke44"][:32]
    out.mkdir(); (out / "snapshots").mkdir()
    for path in (Path(__file__), Path(frozen.__file__), Path(common.__file__),
                 ROOT / "PAPER_FRAMEWORK_IBM_AML_SMALL_GMKEP56_20261008.md",
                 CODE / "paper_framework_gmkep56_template.md", ROOT / "EXPERIMENT_RESULTS_SUMMARY.md"):
        shutil.copy2(path, out / "snapshots" / path.name)
    for kind in ("fixed", "tuning"):
        (out / kind).mkdir()
    shutil.copy2(TUNED / "candidates.json", out / "tuning/candidates.json")
    inputs, nested_hashes, fixed_hashes = {}, {}, {}
    for ds in DATASETS:
        new = out / "tuning" / ds
        new.mkdir()
        for sub in ("splits", "arrays", "search", "folds", "models", "calibration_models", "predictions",
                    "calibration", "selected_search_models", "selected_search_scores", "logs"):
            (new / sub).mkdir()
        original = TUNED / ds
        bindings = read(TUNED / "dataset_inputs.json")[ds]["cache_sha256"]
        for path, expected in bindings.items():
            assert sha(ROOT / path) == expected
            protected[path] = expected
        matrix = np.load(original / "arrays/features.npy", mmap_mode="r")
        X = np.ascontiguousarray(matrix[:, :32])
        y = np.load(original / "arrays/labels.npy", mmap_mode="r")
        identity = pd.read_parquet(original / "accounts.parquet")
        expected = (515088, 6357) if ds == "hi_small" else (705907, 5304)
        assert X.shape == (expected[0], 32) and X.dtype == np.float32 and np.isfinite(X).all()
        assert (len(y), int(y.sum())) == expected
        np.testing.assert_array_equal(identity.row_index, np.arange(len(y)))
        np.testing.assert_array_equal(identity.label, y)
        np.save(new / "arrays/features.npy", X)
        shutil.copy2(original / "arrays/labels.npy", new / "arrays/labels.npy")
        shutil.copy2(original / "accounts.parquet", new / "accounts.parquet")
        fixed_ds = out / "fixed" / ds; fixed_ds.mkdir()
        for sub in ("splits", "folds", "models", "inner_models", "predictions", "calibration"):
            (fixed_ds / sub).mkdir()
        shutil.copy2(original / "accounts.parquet", fixed_ds / "accounts.parquet")
        for seed in range(5):
            for fold in range(5):
                name = f"seed{seed}_fold{fold}.npz"
                src, dst = original / "splits" / name, new / "splits" / name
                shutil.copy2(src, dst)
                assert sha(src) == sha(dst)
                protected[rel(src)] = sha(src)
                with np.load(src) as f: arrays = {k: f[k] for k in f.files}
                nested_hashes[rel(dst)] = {"sha256": sha(dst), "source_path": rel(src),
                                         "arrays_sha256": {k: frozen.array_sha(v) for k, v in arrays.items()}}
                fsrc, fdst = FIXED / ds / "splits" / name, fixed_ds / "splits" / name
                shutil.copy2(fsrc, fdst)
                assert sha(fsrc) == sha(fdst)
                protected[rel(fsrc)] = sha(fsrc)
                with np.load(fsrc) as f: farrays = {k: f[k] for k in f.files}
                for k in ("outer_train", "outer_test"):
                    np.testing.assert_array_equal(farrays[k], arrays[k])
                np.testing.assert_array_equal(farrays["inner_fit"], arrays["development"])
                np.testing.assert_array_equal(farrays["inner_calibration"], arrays["threshold_validation"])
                fixed_hashes[rel(fdst)] = {"sha256": sha(fdst), "source_path": rel(fsrc),
                                         "arrays_sha256": {k: frozen.array_sha(v) for k, v in farrays.items()}}
        inputs[ds] = {"n_samples": len(y), "n_positive": int(y.sum()), "feature_columns": names,
                      "cache_sha256": {rel(p): sha(p) for p in
                       (new / "arrays/features.npy", new / "arrays/labels.npy", new / "accounts.parquet")},
                      "original_cache_sha256": bindings,
                      "features_values_sha256": frozen.array_sha(X), "labels_values_sha256": frozen.array_sha(y)}
        del X, y, matrix, identity; gc.collect()
    jsave(out / "source_bindings.json", protected)
    jsave(out / "dataset_inputs.json", inputs)
    jsave(out / "tuning/dataset_inputs.json", inputs)
    jsave(out / "tuning/split_manifest.json", nested_hashes)
    jsave(out / "fixed/split_manifest.json", fixed_hashes)
    fixed_p = copy.deepcopy(fixed_prior)
    fixed_p.update(experiment="account_graph32_fixed_rf_completion", registered_at=now(),
                   feature_columns={ARM: names}, feature_sets={ARM: 32}, models=["random_forest"],
                   trainer_sha256=sha(__file__), helper_sha256=sha(common.__file__),
                   original_protocol_sha256=sha(FIXED / "protocol.json"),
                   expected_new_final_models=50, expected_new_inner_models=50,
                   prediction_order="Original fixed RF predict_proba n_jobs=6; raw score bytes retained")
    jsave(out / "fixed/protocol.json", fixed_p)
    tuned_p = copy.deepcopy(tuned_prior)
    tuned_p.update(experiment="account_graph32_matched_nested_tuning", registered_at=now(),
                   feature_columns={ARM: names}, feature_sets={ARM: 32},
                   trainer_sha256=sha(__file__), frozen_trainer_sha256=sha(frozen.__file__),
                   source_tuning_protocol_sha256=sha(TUNED / "protocol.json"),
                   candidates_sha256=sha(out / "tuning/candidates.json"),
                   dataset_inputs_sha256=sha(out / "tuning/dataset_inputs.json"),
                   split_manifest_sha256=sha(out / "tuning/split_manifest.json"),
                   expected_outer_settings=150, logical_candidate_evaluations=5400,
                   expected_search_physical_fits=4500, expected_representative_refits=150,
                   expected_strategy_final_models_logical=300, expected_strategy_threshold_models_logical=300)
    jsave(out / "tuning/protocol.json", tuned_p)
    design = {"experiment": "account_graph32_completion_20261009", "registered_at": now(),
              "question": "Does the M5+K3+E4 block improve classification beyond BASE16+G16 ordinary graph statistics?",
              "contrast": "GMKE44 minus Graph32; paired by dataset/model/strategy/seed/fold/policy",
              "primary_metric": "ap", "auxiliary_metrics": ["auc", "f1", "precision", "recall", "accuracy", "balanced_accuracy"],
              "strategies": ["original_fixed", "same_protocol_anchor", "tuned"],
              "new_fixed_outer_settings": 50, "new_tuning_outer_settings": 150,
              "reused_fixed_graph32_settings": 4, "reused_gmke44_settings_per_strategy": 6,
              "feature_columns": names, "feature_count_is_not_independent_information_dimension": True,
              "scope": tuned_prior["evaluation_scope"],
              "inference": "Paired mean/SD, exact direction counts, and complete seed OOF are descriptive, not independent significance tests",
              "search_fairness": "Byte-identical 12 candidates/classifier, nested splits and selection rules; no outer-test selection",
              "environment": {"python": platform.python_version(), **versions},
              "source_bindings_sha256": sha(out / "source_bindings.json"),
              "trainer_sha256": sha(__file__), "frozen_trainer_sha256": sha(frozen.__file__),
              "fixed_protocol_sha256": sha(out / "fixed/protocol.json"),
              "tuning_protocol_sha256": sha(out / "tuning/protocol.json"),
              "initial_free_GiB": shutil.disk_usage(out).free / 2**30}
    jsave(out / "design.json", design)
    jsave(out / "run_manifest.json", {"status": "registered", "registered_at": now(), "design_sha256": sha(out / "design.json")})
    print(json.dumps({"prepared": rel(out), "fixed_RF_folds": 50, "tuning_folds": 150,
                      "logical_search_scores": 5400, "physical_search_fits": 4500}), flush=True)


def registered(out):
    d = read(out / "design.json")
    assert d["trainer_sha256"] == sha(__file__)
    assert d["frozen_trainer_sha256"] == sha(frozen.__file__)
    assert d["source_bindings_sha256"] == sha(out / "source_bindings.json")
    assert d["fixed_protocol_sha256"] == sha(out / "fixed/protocol.json")
    assert d["tuning_protocol_sha256"] == sha(out / "tuning/protocol.json")
    p = read(out / "tuning/protocol.json")
    assert p["helper_sha256"] == sha(common.__file__)
    for name in ("candidates", "dataset_inputs", "split_manifest"):
        assert p[name + "_sha256"] == sha(out / "tuning" / (name + ".json"))
    return d


def inputs(out, ds):
    folder = out / "tuning" / ds
    entry = read(out / "dataset_inputs.json")[ds]
    for name, expected in entry["cache_sha256"].items(): assert sha(ROOT / name) == expected
    X = np.load(folder / "arrays/features.npy", mmap_mode="r")
    y = np.load(folder / "arrays/labels.npy", mmap_mode="r")
    return X, y, pd.read_parquet(folder / "accounts.parquet")


def dump_fixed(model, path):
    frozen.guard_disk(path.parent)
    temporary = path.with_suffix(".joblib.tmp")
    joblib.dump(model, temporary, compress=3)
    temporary.replace(path)


def fixed_worker(out, ds):
    registered(out)
    folder = out / "fixed" / ds
    p = read(out / "fixed/protocol.json")
    X, y, identity = inputs(out, ds)
    manifest = {"dataset": ds, "model": "random_forest", "status": "running", "pid": os.getpid(),
                "started_at": now(), "protocol_sha256": sha(out / "fixed/protocol.json"), "models_completed": 0}
    mp = folder / "run_manifest_random_forest.json"; jsave(mp, manifest)
    try:
        for seed in range(5):
            for fold in range(5):
                stem = f"random_forest_{ARM}_seed{seed}_fold{fold}"
                cp = folder / "folds" / (stem + ".json")
                if cp.exists():
                    done = read(cp)
                    assert done["protocol_sha256"] == sha(out / "fixed/protocol.json")
                    for path, expected in done["artifact_sha256"].items(): assert sha(ROOT / path) == expected
                else:
                    frozen.guard_disk(folder)
                    started = time.monotonic()
                    split = folder / "splits" / f"seed{seed}_fold{fold}.npz"
                    assert sha(split) == read(out / "fixed/split_manifest.json")[rel(split)]["sha256"]
                    with np.load(split) as f: arrays = {k: f[k] for k in f.files}
                    train, test, fit, cal = (arrays[k] for k in ("outer_train", "outer_test", "inner_fit", "inner_calibration"))
                    state = 100 * seed + fold
                    tick = time.monotonic()
                    inner = common.fit_model("random_forest", X, y, fit, cal, p, state)
                    inner_fit_seconds = time.monotonic() - tick
                    cs = inner.predict_proba(X[cal])[:, 1].astype(np.float64)
                    threshold, cal_f1 = common.best_f1_threshold(y[cal], cs)
                    ip = folder / "inner_models" / (stem + ".joblib"); dump_fixed(inner, ip)
                    calpath = folder / "calibration" / (stem + ".parquet")
                    table = identity.iloc[cal][["row_index", "account_id"]].copy()
                    table["y_true"] = y[cal]; table["y_score"] = cs
                    table.to_parquet(calpath, index=False, compression="zstd")
                    del inner, table; gc.collect()
                    tick = time.monotonic()
                    final = common.fit_model("random_forest", X, y, train, None, p, state, 200)
                    final_fit_seconds = time.monotonic() - tick
                    score = final.predict_proba(X[test])[:, 1].astype(np.float64)
                    fp = folder / "models" / (stem + ".joblib"); dump_fixed(final, fp)
                    del final; gc.collect()
                    predpath = folder / "predictions" / (stem + ".parquet")
                    table = identity.iloc[test][["row_index", "account_id"]].copy()
                    table["y_true"] = y[test]; table["y_score"] = score
                    rows = []
                    for policy, cut in (("fixed_0.5", .5), ("inner_max_f1", threshold)):
                        pred, values = frozen.metric_record(y[test], score, cut)
                        table["y_pred_fixed" if policy == "fixed_0.5" else "y_pred_inner"] = pred
                        rows.append(dict(dataset=ds, model="random_forest", feature_set=ARM, seed=seed, fold=fold,
                                         threshold_policy=policy, threshold=cut, n_features=32, selected_rounds=200, **values))
                    table.to_parquet(predpath, index=False, compression="zstd")
                    done = dict(dataset=ds, model="random_forest", feature_set=ARM, seed=seed, fold=fold,
                                n_features=32, feature_columns=p["feature_columns"][ARM],
                                protocol_sha256=sha(out / "fixed/protocol.json"), trainer_sha256=sha(__file__),
                                model_random_state=state, inner_random_state=10000 + 100 * seed + fold,
                                n_outer_train=len(train), n_outer_test=len(test), n_inner_fit=len(fit), n_inner_calibration=len(cal),
                                selected_rounds=200, inner_threshold=threshold, inner_f1=cal_f1,
                                inner_ap=float(average_precision_score(y[cal], cs)), inner_auc=float(roc_auc_score(y[cal], cs)),
                                final_fit_has_eval_set=False, inner_fit_seconds=inner_fit_seconds, final_fit_seconds=final_fit_seconds,
                                runtime_seconds=time.monotonic() - started, metrics=rows,
                                artifact_sha256={rel(path): sha(path) for path in (split, ip, fp, calpath, predpath)})
                    jsave(cp, done)
                    print(f"{now()} FIXED {ds} {stem} AP={rows[0]['ap']:.6f} elapsed={done['runtime_seconds']:.1f}s", flush=True)
                manifest["models_completed"] += 1
                manifest["inner_models_completed"] = manifest["models_completed"]
                manifest["updated_at"] = now(); jsave(mp, manifest)
        assert manifest["models_completed"] == 25
        manifest.update(status="complete", completed_at=now()); jsave(mp, manifest)
    except BaseException:
        manifest.update(status="failed", traceback=traceback.format_exc()); jsave(mp, manifest); raise


def tuning_fold(out, ds, name, seed, fold):
    registered(out)
    target = out / "tuning"; folder = target / ds
    p = read(target / "protocol.json"); design = read(target / "candidates.json")
    stem = f"{name}_{ARM}_seed{seed}_fold{fold}"
    path = folder / "folds" / (stem + ".json")
    if path.exists():
        done = read(path)
        assert done["protocol_sha256"] == sha(target / "protocol.json")
        for f, expected in done["artifact_sha256"].items(): assert sha(ROOT / f) == expected
        return
    started = time.monotonic(); X, y, identity = inputs(out, ds)
    split = folder / "splits" / f"seed{seed}_fold{fold}.npz"
    assert sha(split) == read(target / "split_manifest.json")[rel(split)]["sha256"]
    with np.load(split) as f: arrays = {k: f[k] for k in f.files}
    sr = frozen.search(target, folder, name, ARM, seed, fold, X, y, arrays, p, design)
    anchor = frozen.strategy_fit(target, folder, name, ARM, seed, fold, "same_protocol_anchor", 0,
                                sr["anchor_rounds"], X, y, identity, arrays, design[name][0]["params"])
    cid = sr["selected_candidate_id"]
    if cid == 0:
        tuned = copy.deepcopy(anchor)
        tuned["strategy"] = "tuned"; tuned["physical_reuse_anchor"] = True
        tuned["threshold_fit_seconds"] = tuned["final_fit_seconds"] = 0.
        tuned["threshold_model_fit_seconds"] = tuned["final_model_fit_seconds"] = 0.
        for row in tuned["metrics"]: row["strategy"] = "tuned"
    else:
        tuned = frozen.strategy_fit(target, folder, name, ARM, seed, fold, "tuned", cid,
                                    sr["selected_rounds"], X, y, identity, arrays, design[name][cid]["params"])
    strategies = {"same_protocol_anchor": anchor, "tuned": tuned}
    srpath = folder / "search" / stem / "search_result.json"
    artifacts = {rel(split): sha(split), rel(srpath): sha(srpath)}
    for st in strategies.values(): artifacts.update(st["artifact_sha256"])
    done = dict(dataset=ds, model=name, feature_set=ARM, seed=seed, fold=fold, n_features=32,
                feature_columns=p["feature_columns"][ARM], protocol_sha256=sha(target / "protocol.json"),
                search_result_path=rel(srpath), split_path=rel(split), strategies=strategies,
                logical_candidate_evaluations=36, physical_search_fits=sr["physical_search_fits"],
                logical_strategy_models=4, physical_strategy_models=2 if cid == 0 else 4,
                runtime_seconds=time.monotonic() - started, artifact_sha256=artifacts)
    jsave(path, done)
    print(f"{now()} TUNED {ds} {stem} candidate={cid} AP={tuned['metrics'][0]['ap']:.6f} "
          f"anchor={anchor['metrics'][0]['ap']:.6f} elapsed={done['runtime_seconds']:.1f}s", flush=True)


def execute(out):
    registered(out)
    manifest_path = out / "run_manifest.json"
    manifest = read(manifest_path)
    manifest.update(status="running", pid=os.getpid(), started_at=now()); jsave(manifest_path, manifest)
    schedule = {"started_at": now(), "fixed_stage_max_workers": 2, "fixed_threads_per_worker": 6,
                "tuning_stage_max_workers": 6, "tuning_threads_per_worker": 2,
                "fresh_process_per_tuning_fold": True, "jobs": []}
    try:
        fixed_jobs = []
        for ds in DATASETS:
            log = (out / "fixed" / ds / "execution_random_forest.log").open("a")
            cmd = [sys.executable, __file__, "fixed", "--out", str(out), "--dataset", ds]
            process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            fixed_jobs.append((ds, process, log))
        while any(p.poll() is None for _, p, _ in fixed_jobs):
            counts = {ds: len(list((out / "fixed" / ds / "folds").glob("*.json"))) for ds in DATASETS}
            print(now(), "fixed", counts, flush=True); time.sleep(20)
        for ds, process, log in fixed_jobs:
            log.close(); assert process.returncode == 0, f"Fixed failed: {ds}"
        # Longest jobs first avoids the former serial RF-worker tail. This is
        # execution scheduling only: each fold has the frozen random states.
        jobs = [(ds, name, seed, fold) for name in ("random_forest", "lightgbm", "xgboost")
                for seed in range(5) for fold in range(5) for ds in DATASETS]
        pending = [j for j in jobs if not (out / "tuning" / j[0] / "folds" /
                   f"{j[1]}_{ARM}_seed{j[2]}_fold{j[3]}.json").exists()]
        active = []; finished = []
        while pending or active:
            while pending and len(active) < 6:
                ds, name, seed, fold = pending.pop(0)
                frozen.guard_disk(out)
                logpath = out / "tuning" / ds / "logs" / f"{name}_{ARM}_seed{seed}_fold{fold}.log"
                log = logpath.open("a")
                cmd = [sys.executable, __file__, "tuning-fold", "--out", str(out), "--dataset", ds,
                       "--model", name, "--seed", str(seed), "--fold", str(fold)]
                process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
                record = {"dataset": ds, "model": name, "seed": seed, "fold": fold, "pid": process.pid,
                          "started_at": now(), "log_path": rel(logpath), "command": cmd}
                schedule["jobs"].append(record)
                active.append((process, log, record))
            remaining = []
            for process, log, record in active:
                if process.poll() is None: remaining.append((process, log, record)); continue
                log.close(); record.update(exit_code=process.returncode, completed_at=now())
                assert process.returncode == 0, f"Tuning failed: {record}"
                finished.append(record)
            active = remaining
            jsave(out / "execution_schedule.json", schedule)
            counts = {ds: len(list((out / "tuning" / ds / "folds").glob("*.json"))) for ds in DATASETS}
            manifest.update(fixed_completed=50, tuning_completed=sum(counts.values()), updated_at=now())
            jsave(manifest_path, manifest)
            print(now(), "tuning", counts, "active", len(active), "free_GiB", round(shutil.disk_usage(out).free / 2**30, 2), flush=True)
            if pending or active: time.sleep(20)
        assert sum(len(list((out / "tuning" / ds / "folds").glob("*.json"))) for ds in DATASETS) == 150
        for ds in DATASETS:
            for name in MODELS:
                assert len(list((out / "tuning" / ds / "folds").glob(f"{name}_*.json"))) == 25
                jsave(out / "tuning" / ds / f"run_manifest_{name}.json",
                      {"dataset": ds, "model": name, "status": "complete", "completed_outer_settings": 25,
                       "protocol_sha256": sha(out / "tuning/protocol.json"), "completed_at": now()})
        manifest.update(status="trained_pending_independent_validation", completed_at=now()); jsave(manifest_path, manifest)
        print("ALL NEW GRAPH32 FITS COMPLETE", flush=True)
    except BaseException:
        manifest.update(status="failed", failed_at=now(), traceback=traceback.format_exc()); jsave(manifest_path, manifest)
        raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("prepare", "fixed", "tuning-fold", "execute"))
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--dataset", choices=DATASETS)
    p.add_argument("--model", choices=MODELS)
    p.add_argument("--seed", type=int, choices=range(5))
    p.add_argument("--fold", type=int, choices=range(5))
    args = p.parse_args(); out = args.out.resolve()
    assert out.is_relative_to(RESULTS)
    if args.action == "prepare": prepare(out)
    elif args.action == "execute": execute(out)
    elif args.action == "fixed": fixed_worker(out, args.dataset)
    else: tuning_fold(out, args.dataset, args.model, args.seed, args.fold)


if __name__ == "__main__": main()
