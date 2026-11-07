"""Aggregate a fully trained IBM Small tuning batch without fitting models.

Incomplete batches are rejected before any science CSV is written.  The saved
fold predictions are checked and reassembled by account row into seed-level
OOF evaluations.  Independent model validation must run after this program;
``trained_aggregated`` does not mean independently validated.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score,
    balanced_accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score)

import report_account_gmkep_tuning as report

ROOT = report.ROOT
OUT = report.OUT
DATASETS, MODELS, ARMS = tuple(report.DATASETS), tuple(report.MODELS), tuple(report.ARMS)
STRATEGIES, POLICIES = tuple(report.STRATEGIES), report.POLICIES
KEYS = ["dataset", "model", "feature_set", "seed", "fold"]


def sha(path: Path) -> str:
    return report.sha(path)


def rel(path: Path) -> str:
    return str(Path(path).resolve().relative_to(ROOT))


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def write(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def inside(out: Path, name: str) -> Path:
    path = (ROOT / name).resolve()
    assert path.is_relative_to(out.resolve()) and path.is_file(), f"Missing/uncontained artifact: {name}"
    return path


def values(y: np.ndarray, score: np.ndarray, predicted: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "auc": float(roc_auc_score(y, score)), "ap": float(average_precision_score(y, score)),
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "specificity": float(tn / (tn + fp)), "fpr": float(fp / (tn + fp)),
        "fnr": float(fn / (tp + fn)), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "n_samples": len(y), "n_positive": int(y.sum()),
    }


def _artifact_hashes(out: Path, mapping: dict, checked: dict) -> None:
    for name, digest in mapping.items():
        path = inside(out, name)
        if name not in checked:
            checked[name] = sha(path)
        assert checked[name] == digest, f"Changed completed training artifact: {name}"


def collect(out: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict]:
    """Read all 450 completed fold/search JSONs; write nothing."""
    protocol_path = out / "protocol.json"
    protocol = read(protocol_path)
    protocol_sha = sha(protocol_path)
    assert protocol["seeds"] == list(range(5)) and protocol["outer_n_splits"] == 5
    assert protocol["inner_n_splits"] == 3 and protocol["candidates_per_model"] == 12
    assert protocol["primary_metrics"] == ["ap"]
    assert set(protocol["strategies"]) == set(STRATEGIES)
    assert protocol["candidates_sha256"] == sha(out / "candidates.json")
    assert protocol["dataset_inputs_sha256"] == sha(out / "dataset_inputs.json")
    assert protocol["split_manifest_sha256"] == sha(out / "split_manifest.json")
    design = read(out / "candidates.json")
    trainer = ROOT / "Altman_experiments/code/run_account_gmkep_tuning.py"
    assert protocol["trainer_sha256"] == sha(trainer)
    workers = []
    for dataset, model in itertools.product(DATASETS, MODELS):
        worker = read(out / dataset / f"run_manifest_{model}.json")
        assert worker["status"] == "complete" and worker["completed_outer_settings"] == 75
        assert worker["protocol_sha256"] == protocol_sha
        workers.append(worker)
    expected = set(itertools.product(DATASETS, MODELS, ARMS, range(5), range(5)))
    found = set()
    for dataset in DATASETS:
        for path in (out / dataset / "folds").glob("*.json"):
            row = read(path)
            key = tuple(row[name] for name in KEYS)
            assert key in expected and key not in found, f"Duplicate/foreign completed fold: {path}"
            found.add(key)
    assert found == expected, f"Incomplete trained batch: {len(found)}/450"
    fold_records, candidate_records, selections = [], [], []
    completed, checked = {}, {}
    final_models, threshold_models, representative_models = set(), set(), set()
    scientific_sources = {}
    for dataset, model, arm, seed, fold in sorted(expected):
        stem = f"{model}_{arm}_seed{seed}_fold{fold}"
        path = out / dataset / "folds" / f"{stem}.json"
        done = read(path)
        assert done["protocol_sha256"] == protocol_sha
        assert done["n_features"] == len(protocol["feature_columns"][arm])
        assert done["feature_columns"] == protocol["feature_columns"][arm]
        assert done["logical_candidate_evaluations"] == 36
        _artifact_hashes(out, done["artifact_sha256"], checked)
        scientific_sources[rel(path)] = sha(path)
        search_path = inside(out, done["search_result_path"])
        search = read(search_path)
        assert search["protocol_sha256"] == protocol_sha
        assert tuple(search[name] for name in KEYS) == (dataset, model, arm, seed, fold)
        _artifact_hashes(out, search["artifact_sha256"], checked)
        assert len(search["candidate_records"]) == 36
        scientific_sources[rel(search_path)] = sha(search_path)
        for record in search["candidate_records"]:
            item = dict(record)
            item["best_iteration"] = item["selected_rounds"]
            item["effective_params_json"] = json.dumps(item.pop("effective_params"), sort_keys=True)
            candidate_records.append(item)
        anchor, tuned = (done["strategies"][name] for name in ("same_protocol_anchor", "tuned"))
        chosen = int(search["selected_candidate_id"])
        assert anchor["candidate_id"] == 0 and tuned["candidate_id"] == chosen
        assert anchor["selected_rounds"] == search["anchor_rounds"]
        assert tuned["selected_rounds"] == search["selected_rounds"]
        assert bool(tuned["physical_reuse_anchor"]) == (chosen == 0)
        assert chosen != 0 or (tuned["model_path"] == anchor["model_path"]
                               and tuned["calibration_model_path"] == anchor["calibration_model_path"])
        assert done["physical_strategy_models"] == (2 if chosen == 0 else 4)
        base = dict(zip(KEYS, (dataset, model, arm, seed, fold)))
        selection = {
            **base, "candidate_id": chosen, "selected_rounds": search["selected_rounds"],
            "anchor_rounds": search["anchor_rounds"], "selected_mean_ap": search["selected_mean_ap"],
            "selected_params_json": json.dumps(design[model][chosen]["params"], sort_keys=True),
            "anchor_params_json": json.dumps(design[model][0]["params"], sort_keys=True),
            "selected_effective_params_json": json.dumps(tuned["effective_params"], sort_keys=True),
            "anchor_effective_params_json": json.dumps(anchor["effective_params"], sort_keys=True),
            "physical_reuse_anchor": tuned["physical_reuse_anchor"],
            "logical_candidate_evaluations": done["logical_candidate_evaluations"],
            "physical_search_fits": search["physical_search_fits"],
            "physical_strategy_models": done["physical_strategy_models"],
            "search_fit_seconds": search["search_fit_seconds"],
            "search_predict_seconds": search["search_predict_seconds"],
            "search_audit_seconds": search["search_audit_seconds"],
            "representative_stage_seconds": search["representative_refit_seconds"],
            "completion_invocation_seconds": done["runtime_seconds"],
            "search_result_path": done["search_result_path"], "fold_result_path": rel(path),
            "split_path": done["split_path"],
        }
        for strategy, item in done["strategies"].items():
            assert strategy == item["strategy"] and strategy in STRATEGIES
            assert item["final_fit_has_eval_set"] is False
            assert item["params"] == design[model][item["candidate_id"]]["params"]
            assert item["model_random_state"] == 100 * seed + fold
            expected_params = {**item["params"], "random_state": 100 * seed + fold,
                               "n_estimators": item["selected_rounds"]}
            assert item["effective_params"] == expected_params
            _artifact_hashes(out, item["artifact_sha256"], checked)
            selection["threshold_stage_seconds_" + strategy] = item["threshold_fit_seconds"]
            selection["final_stage_seconds_" + strategy] = item["final_fit_seconds"]
            selection["threshold_model_fit_seconds_" + strategy] = item["threshold_model_fit_seconds"]
            selection["final_model_fit_seconds_" + strategy] = item["final_model_fit_seconds"]
            selection["threshold_" + strategy] = item["threshold"]
            final_models.add(item["model_path"])
            threshold_models.add(item["calibration_model_path"])
            assert len(item["metrics"]) == 2
            for metric in item["metrics"]:
                assert tuple(metric[name] for name in KEYS) == (dataset, model, arm, seed, fold)
                assert metric["strategy"] == strategy
                fold_records.append(metric)
        representative_models.add(search["representative_model_path"])
        selections.append(selection)
        completed[(dataset, model, arm, seed, fold)] = done
    frame, search_frame, selected = (pd.DataFrame(items) for items in (fold_records, candidate_records, selections))
    assert (len(frame), len(search_frame), len(selected)) == (1800, 16200, 450)
    report._validate_search(selected, search_frame)
    assert search_frame.physical_fit_charged.sum() == 13500
    assert search_frame[["dataset", "physical_fit_id"]].drop_duplicates().shape[0] == 13500
    assert len(representative_models) == 450
    assert len(final_models) == len(threshold_models)
    meta = {
        "workers": workers, "physical_search_fits": int(search_frame.physical_fit_charged.sum()),
        "physical_fit_id_scope": "dataset/physical_fit_id",
        "representative_refits": len(representative_models), "physical_final_models": len(final_models),
        "physical_threshold_models": len(threshold_models),
        "logical_final_models": 900, "logical_threshold_models": 900,
        "tuned_candidate0_reuse_settings": int(selected.physical_reuse_anchor.sum()),
        "retained_physical_model_count": len(final_models) + len(threshold_models) + len(representative_models),
        "source_fold_and_search_sha256": scientific_sources,
        "registered_artifact_sha256": checked,
    }
    return frame, search_frame, selected, completed, meta


def seed_oof(out: Path, completed: dict, folds: pd.DataFrame) -> pd.DataFrame:
    records = []
    indexed = folds.set_index(KEYS + ["strategy", "threshold_policy"])
    for dataset in DATASETS:
        folder = out / dataset
        accounts = pd.read_parquet(folder / "accounts.parquet")
        y = np.load(folder / "arrays/labels.npy", mmap_mode="r")
        assert np.array_equal(accounts.row_index, np.arange(len(y)))
        assert np.array_equal(accounts.label, y)
        for model, arm, strategy, seed in itertools.product(MODELS, ARMS, STRATEGIES, range(5)):
            score = np.empty(len(y), dtype=np.float64)
            predictions = {policy: np.empty(len(y), dtype=np.int8) for policy in POLICIES}
            seen = np.zeros(len(y), dtype=np.uint8)
            for fold in range(5):
                key = dataset, model, arm, seed, fold
                done = completed[key]
                item = done["strategies"][strategy]
                path = inside(out, item["prediction_path"])
                table = pd.read_parquet(path)
                index = table.row_index.to_numpy(np.int64)
                assert len(index) and np.unique(index).size == len(index)
                assert index.min() >= 0 and index.max() < len(y)
                assert not seen[index].any(), f"Overlapping outer predictions: {key}"
                with np.load(inside(out, done["split_path"])) as split:
                    assert np.array_equal(index, split["outer_test"])
                assert np.array_equal(table.account_id.to_numpy(), accounts.account_id.to_numpy()[index])
                assert np.array_equal(table.y_true.to_numpy(), y[index])
                probability = table.y_score.to_numpy(np.float64)
                assert np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all()
                score[index] = probability
                seen[index] = 1
                for policy, column in (("fixed_0.5", "y_pred_fixed"), ("inner_max_f1", "y_pred_inner")):
                    stored = indexed.loc[(*key, strategy, policy)]
                    expected = (probability >= stored.threshold).astype(np.int8)
                    predicted = table[column].to_numpy(np.int8)
                    assert np.array_equal(predicted, expected), f"Stored prediction threshold differs: {key}"
                    recomputed = values(y[index], probability, predicted)
                    for metric in report.METRICS:
                        report._close(stored[metric], recomputed[metric], f"Fold prediction metric {key}/{metric}")
                    predictions[policy][index] = predicted
                del table
            assert seen.all(), f"Incomplete OOF account coverage: {dataset}/{model}/{arm}/{strategy}/{seed}"
            for policy in POLICIES:
                records.append({"dataset": dataset, "model": model, "feature_set": arm,
                                "strategy": strategy, "seed": seed, "threshold_policy": policy,
                                **values(y, score, predictions[policy])})
            del score, predictions, seen
        del accounts, y
        gc.collect()
    result = pd.DataFrame(records)
    assert len(result) == 360
    return result


def finalize(out: Path = OUT) -> None:
    out = out.resolve()
    assert out.is_relative_to((ROOT / "Altman_experiments/results").resolve())
    assert not (out / "validation_models.json").exists(), "Do not rewrite an independently sealed batch"
    protected = {name: sha(out / name) for name in
                 ("protocol.json", "candidates.json", "dataset_inputs.json", "split_manifest.json")}
    folds, candidates, selections, completed, meta = collect(out)
    seeds = seed_oof(out, completed, folds)
    old = report.fixed.load()
    derived = report.aggregate_tables(folds, seeds, old)
    tables = {"fold_metrics.csv": folds, "seed_oof_metrics.csv": seeds,
              "candidate_fold_scores.csv": candidates, "selection_records.csv": selections, **derived}
    assert len(tables) == 11
    for name, frame in tables.items():
        csv(out / name, frame)
    manifest = read(out / "run_manifest.json")
    manifest.update({
        "status": "trained_aggregated", "aggregated_at": datetime.now().astimezone().isoformat(),
        "independent_validation_completed": False, "completed_outer_settings": 450,
        "logical_candidate_evaluations_completed": 16200,
        "physical_search_fits": meta["physical_search_fits"], "representative_refits": meta["representative_refits"],
        "physical_fit_id_scope": meta["physical_fit_id_scope"],
        "physical_final_models": meta["physical_final_models"],
        "physical_threshold_models": meta["physical_threshold_models"],
        "logical_final_models": 900, "logical_threshold_models": 900,
        "tuned_candidate0_reuse_settings": meta["tuned_candidate0_reuse_settings"],
        "retained_physical_model_count": meta["retained_physical_model_count"],
        "total_physical_fits_including_representative_and_strategy": meta["physical_search_fits"]
            + meta["representative_refits"] + meta["physical_final_models"] + meta["physical_threshold_models"],
        "search_fit_seconds": float(candidates.fit_seconds.sum()),
        "search_prediction_seconds": float(candidates.predict_seconds.sum()),
        "search_audit_seconds": float(candidates.audit_seconds.sum()),
        "representative_stage_seconds": float(selections.representative_stage_seconds.sum()),
        "strategy_model_fit_seconds": float(selections.filter(regex="^(threshold|final)_model_fit_seconds_").sum().sum()),
        "strategy_stage_seconds": float(selections.filter(regex="^(threshold|final)_stage_seconds_").sum().sum()),
        "science_csv_rows": {name: len(frame) for name, frame in tables.items()},
        "science_csv_sha256": {name: sha(out / name) for name in tables},
    })
    write(out / "run_manifest.json", manifest)
    assert protected == {name: sha(out / name) for name in protected}
    audit = {
        "aggregated_at": datetime.now().astimezone().isoformat(),
        "scope": "Complete trained batch aggregation only; independent model validation still required",
        "aggregator": rel(Path(__file__)), "aggregator_sha256": sha(Path(__file__)),
        "report_aggregation_helper": rel(Path(report.__file__)), "report_aggregation_helper_sha256": sha(Path(report.__file__)),
        "protected_science_inputs_sha256": protected,
        "old_fixed_validation_sha256": sha(report.fixed.OUT / "validation_models.json"),
        "source_fold_and_search_sha256": meta["source_fold_and_search_sha256"],
        "registered_artifact_sha256": meta["registered_artifact_sha256"],
        "checks": {"all_450_outer_settings_complete": True, "all_16200_candidate_records_present": True,
                   "candidate_ap_argmax_and_rounds_recomputed": True,
                   "saved_fold_predictions_and_metrics_recomputed": True,
                   "complete_oof_coverage_and_account_identity": True,
                   "source_outer_sample_counts_match_old": True, "protocol_and_inputs_unchanged": True},
        "science_csv_sha256": manifest["science_csv_sha256"],
        "physical_fit_id_scope": meta["physical_fit_id_scope"],
        "notes": "Stage seconds include fitting, prediction, hashing and saving; model_fit_seconds are fit-only. "
                 "RF shared search fits are charged once to the 200-tree parent candidate; candidate100 fit_seconds=0. "
                 "Logical scoring count does not equal physical model fitting count. "
                 "Raw physical_fit_id strings are local to a dataset; physical-fit uniqueness uses (dataset, physical_fit_id).",
    }
    write(out / "aggregation_validation.json", audit)
    print(json.dumps({"out": str(out), "status": manifest["status"],
                      "complete_outer_settings": 450, "science_csv_rows": manifest["science_csv_rows"]}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    finalize(parser.parse_args().out)
