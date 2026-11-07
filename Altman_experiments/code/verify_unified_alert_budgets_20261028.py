"""Independent acceptance of the saved-score, 42-setting alert-budget analysis.

No runner or training module is imported. Top-k is reconstructed by a numeric
partition boundary and original-index tie resolution, not the runner's sort.
AP is recomputed by score-tie blocks without sklearn. Seed results combine the
five fold-local alert sets; cross-fold scores are never ranked together.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np


BUDGETS = (0.005, 0.01, 0.02)
DENOMINATORS = {0.005: 200, 0.01: 100, 0.02: 50}
IDENTITY = ("dataset", "model", "feature_set", "seed", "fold")
FOLD_KEY = IDENTITY + ("budget",)
SEED_KEY = ("dataset", "model", "feature_set", "seed", "budget")
SETTING_BUDGET_KEY = ("dataset", "model", "feature_set", "budget")
PAIR_FOLD_KEY = ("dataset", "pair_id", "seed", "fold", "budget")
PAIR_SEED_KEY = ("dataset", "pair_id", "seed", "budget")
PAIR_SUMMARY_KEY = ("dataset", "pair_id", "budget")
INTEGER_COLUMNS = {
    "seed", "fold", "k", "n_test", "n_positive", "tp", "fp", "n_folds", "n_seeds",
    "tp_left", "tp_right", "delta_tp", "precision_positive_folds", "precision_zero_folds",
    "precision_negative_folds", "recall_positive_folds", "recall_zero_folds",
    "recall_negative_folds", "precision_positive_seeds", "precision_zero_seeds",
    "precision_negative_seeds", "recall_positive_seeds", "recall_zero_seeds", "recall_negative_seeds",
}
FORBIDDEN_COLUMNS = {"p", "p_value", "pvalue", "p_holm", "holm", "significant", "significance"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def independent_alert_indices(row_index, scores, budget):
    """Exactly k: all above kth score, then lowest original indices at the cut."""
    row_index = np.asarray(row_index, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    require(n > 0 and len(row_index) == n, "Empty or mismatched prediction arrays")
    require(budget in DENOMINATORS, "Unregistered budget")
    require(np.isfinite(scores).all(), "Nonfinite scores")
    denominator = DENOMINATORS[budget]
    k = (n + denominator - 1) // denominator
    cut = np.partition(scores, n - k)[n - k]
    above = np.flatnonzero(scores > cut)
    tied = np.flatnonzero(scores == cut)
    tied = tied[np.argsort(row_index[tied], kind="stable")]
    selected = np.concatenate((above, tied[: k - len(above)]))
    require(len(selected) == k, "Independent exact-k reconstruction failed")
    return selected, k


def independent_ap(labels, scores):
    """Noninterpolated AP, summing precision at complete score-tie blocks."""
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    positive = int(labels.sum())
    require(positive > 0, "AP requires a positive test account")
    order = np.argsort(-scores, kind="quicksort")
    ordered_labels, ordered_scores = labels[order], scores[order]
    block_ends = np.r_[np.flatnonzero(ordered_scores[:-1] != ordered_scores[1:]), len(labels) - 1]
    cumulative = np.cumsum(ordered_labels, dtype=np.int64)[block_ends]
    new_positive = np.diff(np.r_[0, cumulative])
    return math.fsum((new_positive * cumulative / (block_ends + 1)).tolist()) / positive


def typed_key(row, columns):
    return tuple(int(row[c]) if c in INTEGER_COLUMNS else float(row[c]) if c == "budget" else str(row[c])
                 for c in columns)


def moment(values):
    values = [float(x) for x in values]
    require(len(values) > 1 and all(math.isfinite(x) for x in values), "Invalid summary values")
    return statistics.fmean(values), statistics.stdev(values)


def expected_settings():
    answer = set()
    for dataset in ("hi_small", "li_small"):
        for model in ("lightgbm", "xgboost", "random_forest", "mlp", "graphsage_weak"):
            for feature in ("base16", "base_g32", "gmke44", "gmkep56"):
                answer.add((dataset, model, feature))
        answer.add((dataset, "xgboost", "base_gfp_account_stream"))
    return answer


class Verifier:
    def __init__(self, manifest_path, results_dir, allow_incomplete=False):
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest = read_json(self.manifest_path)
        self.initial_manifest_sha256 = digest(self.manifest_path)
        self.root = self.manifest_path.parent
        self.results = Path(results_dir).resolve()
        self.allow_incomplete = allow_incomplete
        self.missing_entries = []
        self.processed_entries = 0
        self.account_ids = {}
        self.bound_hashes = {}
        self.checked_output_hashes = {}
        self.canonical = {}
        self.test_rows = {}
        self.max_ap_difference = 0.0
        self.max_metric_difference = 0.0
        self.fold_values = {}
        self.seed_values = {}
        self.selection_values = {}
        self.ap_checks = []
        self.formal_tables = {}

    def bind_protocol(self):
        require("protocol_path" in self.manifest and "protocol_sha256" in self.manifest,
                "Frozen analysis protocol path/SHA required")
        path = self.bind({"path": self.manifest["protocol_path"], "sha256": self.manifest["protocol_sha256"]})
        protocol = read_json(path)
        require(protocol["expected_settings"] == 42 and protocol["expected_entries"] == 1050,
                "Protocol full-scope mismatch")
        require(tuple(protocol["budgets"]) == BUDGETS, "Protocol budgets changed")
        require(tuple(self.manifest.get("budgets", BUDGETS)) == BUDGETS, "Manifest budgets changed")

    def published_ap(self, entry, identity_key, n_test, n_positive):
        require("formal_metrics_path" in entry and "formal_metrics_sha256" in entry,
                f"Missing direct formal metric source binding: {identity_key}")
        path = self.bind({"path": entry["formal_metrics_path"], "sha256": entry["formal_metrics_sha256"]})
        if str(path) not in self.formal_tables:
            with path.open(newline="") as stream:
                reader = csv.DictReader(stream)
                index = {}
                for record in reader:
                    if record.get("strategy", "tuned") != "tuned" or record.get("threshold_policy", "inner_max_f1") != "inner_max_f1":
                        continue
                    key = typed_key(record, IDENTITY)
                    require(key not in index, f"Formal metrics duplicate identity: {path}/{key}")
                    index[key] = record
            self.formal_tables[str(path)] = index
        require(identity_key in self.formal_tables[str(path)], f"Formal AP identity missing: {path}/{identity_key}")
        record = self.formal_tables[str(path)][identity_key]
        require(int(record["n_test"]) == n_test and int(record["n_positive"]) == n_positive,
                f"Formal versus canonical test counts: {identity_key}")
        ap = float(record[entry.get("expected_ap_field", "ap")])
        require(abs(ap - float(entry["expected_ap"])) <= 1e-15, f"Manifest expected AP differs from formal table: {identity_key}")
        return ap

    def resolve(self, value):
        path = Path(value)
        return path.resolve() if path.is_absolute() else (self.root / path).resolve()

    def bind(self, spec):
        if isinstance(spec, str):
            raise ValueError("An input must include both path and SHA-256")
        path = self.resolve(spec.get("path", spec.get("prediction_path")))
        expected = spec.get("sha256", spec.get("prediction_sha256"))
        require(expected and len(expected) == 64, f"Missing SHA binding: {path}")
        if str(path) not in self.bound_hashes:
            actual = digest(path)
            require(actual == expected, f"Source SHA mismatch: {path}")
            self.bound_hashes[str(path)] = actual
        else:
            require(self.bound_hashes[str(path)] == expected, f"Conflicting source binding: {path}")
        return path

    @staticmethod
    def table(path, columns=None):
        import pandas as pd
        path = Path(path)
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path, columns=columns)
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path, usecols=columns, float_precision="round_trip")
        raise ValueError(f"Unsupported table format: {path}")

    def load_canonical(self):
        canonical = self.manifest.get("canonical", {})
        require(set(canonical) == {"hi_small", "li_small"}, "Both canonical datasets required")
        for dataset, config in canonical.items():
            accounts = config["accounts"]
            frame = self.table(self.bind(accounts))
            index_column = accounts.get("row_index_col", "row_index")
            require(index_column in frame and np.issubdtype(frame[index_column].dtype, np.integer),
                    f"Canonical index must be stored as integer: {dataset}")
            indices = (frame[index_column].to_numpy(dtype=np.int64) if index_column in frame
                       else np.arange(len(frame), dtype=np.int64))
            require(np.array_equal(np.sort(indices), np.arange(len(frame))), f"Canonical index range: {dataset}")
            account_id_column = accounts.get("account_id_col")
            if account_id_column:
                require(frame[account_id_column].is_unique, f"Duplicate canonical account: {dataset}")
            if "labels" in config:
                spec = config["labels"]
                label_path = self.bind(spec)
                if label_path.suffix == ".npy":
                    labels = np.load(label_path, allow_pickle=False)
                else:
                    label_frame = self.table(label_path)
                    labels = label_frame[spec.get("label_col", "y_true")].to_numpy()
            else:
                column = accounts.get("label_col", config.get("label_col", "label"))
                labels = frame[column].to_numpy()
            require(len(labels) == len(indices) and np.isin(labels, [0, 1]).all(), f"Canonical binary labels: {dataset}")
            order = np.argsort(indices)
            labels = np.asarray(labels, dtype=np.int8)[order]
            self.canonical[dataset] = labels
            self.account_ids[dataset] = frame[account_id_column].to_numpy()[order] if account_id_column else None
            splits = config["splits"]
            if isinstance(splits, dict):
                if "path" in splits:
                    value = read_json(self.bind(splits))
                    splits = value.get("outer_splits", value.get("splits", value)) if isinstance(value, dict) else value
                else:
                    records = []
                    for key, value in splits.items():
                        match = re.fullmatch(r"seed([0-4])_fold([0-4])", key)
                        require(match is not None, "Invalid canonical split mapping key")
                        records.append({**value, "seed": int(match[1]), "fold": int(match[2])})
                    splits = records
            for spec in splits:
                seed, fold = int(spec["seed"]), int(spec["fold"])
                key = (dataset, seed, fold)
                if "path" in spec:
                    split_path = self.bind(spec)
                    if split_path.suffix == ".npz":
                        with np.load(split_path, allow_pickle=False) as archive:
                            rows = archive[spec.get("outer_test_key", "outer_test")].copy()
                    else:
                        value = read_json(split_path)
                        rows = value[spec.get("outer_test_key", "outer_test")]
                else:
                    rows = spec["outer_test"]
                rows = np.asarray(rows, dtype=np.int64)
                require(key not in self.test_rows, f"Duplicate canonical split: {key}")
                require(len(np.unique(rows)) == len(rows) and len(rows) > 0, f"Invalid test index: {key}")
                require(rows.min() >= 0 and rows.max() < len(labels), f"Out of range split: {key}")
                self.test_rows[key] = np.sort(rows)
            for seed in range(5):
                pieces = [self.test_rows[(dataset, seed, fold)] for fold in range(5)]
                require(np.array_equal(np.sort(np.concatenate(pieces)), np.arange(len(labels))),
                        f"Canonical five folds must cover every account exactly once: {dataset}/{seed}")
        require(len(self.test_rows) == 50, "Canonical splits must contain 2 datasets x5 seeds x5 folds")

    def recompute_folds(self):
        entries = self.manifest["entries"]
        require(len(entries) == 1050, "Expected 1050 unique saved-score entries")
        identities = set()
        for number, entry in enumerate(entries, 1):
            key = typed_key(entry, IDENTITY)
            require(key not in identities, f"Duplicate prediction entry: {key}")
            identities.add(key)
            require(key[:3] in expected_settings() and key[3] in range(5) and key[4] in range(5), f"Unregistered entry: {key}")
            require(entry.get("strategy", "tuned") == "tuned", f"Non-tuned entry: {key}")
            declared_path = self.resolve(entry.get("path", entry.get("prediction_path")))
            if not declared_path.is_file():
                require(self.allow_incomplete, f"Missing planned prediction: {key}: {declared_path}")
                self.missing_entries.append({**dict(zip(IDENTITY, key)), "path": str(declared_path)})
                continue
            path = self.bind(entry)
            index_column = entry.get("row_index_col", "row_index")
            label_column = entry.get("label_col", entry.get("y_col", "y_true"))
            score_column = entry.get("score_col", "y_score")
            frame = self.table(path)
            require(np.issubdtype(frame[index_column].dtype, np.integer), f"Prediction row_index stored as noninteger: {key}")
            rows = frame[index_column].to_numpy(dtype=np.int64)
            labels = frame[label_column].to_numpy()
            scores = frame[score_column].to_numpy(dtype=np.float64)
            require(np.isin(labels, [0, 1]).all(), f"Nonbinary labels: {key}")
            require(np.isfinite(scores).all(), f"Nonfinite scores: {key}")
            sort = np.argsort(rows)
            rows, labels, scores = rows[sort], np.asarray(labels, dtype=np.int8)[sort], scores[sort]
            expected_rows = self.test_rows[(key[0], key[3], key[4])]
            require(np.array_equal(rows, expected_rows), f"Prediction accounts differ from canonical test fold: {key}")
            require(np.array_equal(labels, self.canonical[key[0]][rows]), f"Prediction labels differ from canonical: {key}")
            account_column = entry.get("account_id_col", "account_id")
            if account_column in frame and self.account_ids[key[0]] is not None:
                require(np.array_equal(frame[account_column].to_numpy()[sort], self.account_ids[key[0]][rows]), f"Prediction account IDs differ from canonical: {key}")
            require("expected_ap" in entry, f"Missing published fold AP: {key}")
            published_ap = self.published_ap(entry, key, len(rows), int(labels.sum()))
            ap = independent_ap(labels, scores)
            difference = abs(ap - published_ap)
            self.max_ap_difference = max(self.max_ap_difference, difference)
            require(difference <= 1e-10, f"Formal AP reconstruction mismatch: {key}: {difference}")
            self.ap_checks.append({**dict(zip(IDENTITY, key)), "ap": ap, "expected_ap": float(entry["expected_ap"]), "absolute_difference": difference})
            for budget in BUDGETS:
                positions, k = independent_alert_indices(rows, scores, budget)
                selected = np.sort(rows[positions])
                tp = int(labels[positions].sum())
                record = {**dict(zip(IDENTITY, key)), "budget": budget, "k": k,
                          "n_test": len(rows), "n_positive": int(labels.sum()), "tp": tp,
                          "precision_at_k": tp / k, "recall_at_k": tp / int(labels.sum())}
                record_key = key + (budget,)
                self.fold_values[record_key] = record
                # Arrays are checked while this source is loaded, then released.
                self.check_selected(record_key, selected, entry)
            self.processed_entries += 1
            if self.processed_entries % 50 == 0:
                print(json.dumps({"phase": "independent_saved_score_recheck", "verified_entries": self.processed_entries,
                                  "inspected_entries": number, "expected_entries": 1050}), flush=True)
        expected_keys = {setting + (seed, fold) for setting in expected_settings() for seed in range(5) for fold in range(5)}
        require(identities == expected_keys, "Exact 42x25 prediction coverage mismatch")
        require(len(self.fold_values) == 3 * self.processed_entries, "Three budgets per available prediction required")
        if not self.allow_incomplete:
            require(len(self.fold_values) == 3150, "Expected 3150 independently recomputed budget rows")

    def read_csv_index(self, filename, key_columns):
        path = self.results / filename
        require(path.is_file(), f"Missing published output: {filename}")
        self.checked_output_hashes[str(path)] = digest(path)
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            require(reader.fieldnames and not (set(reader.fieldnames) & FORBIDDEN_COLUMNS), f"Forbidden inference columns: {filename}")
            rows = {}
            for row in reader:
                key = typed_key(row, key_columns)
                require(key not in rows, f"Duplicate output key: {filename}/{key}")
                rows[key] = row
        return rows

    def check_selected(self, record_key, independently_selected, entry):
        if not self.selection_values:
            self.selection_values = self.read_csv_index("selection_manifest.csv", FOLD_KEY)
            if not self.allow_incomplete:
                require(len(self.selection_values) == 3150, "Expected 3150 selection references")
        spec = self.selection_values[record_key]
        selected_path = Path(spec["selected_path"])
        if not selected_path.is_absolute():
            selected_path = self.results / selected_path
        selected_path = selected_path.resolve()
        expected_hash = spec["selected_sha256"]
        if str(selected_path) not in self.checked_output_hashes:
            require(digest(selected_path) == expected_hash, f"Selected account SHA mismatch: {selected_path}")
            self.checked_output_hashes[str(selected_path)] = expected_hash
        else:
            require(self.checked_output_hashes[str(selected_path)] == expected_hash, "Conflicting selected account binding")
        with np.load(selected_path, allow_pickle=False) as archive:
            selected = np.asarray(archive[spec["selected_array_key"]], dtype=np.int64)
            require(len(selected) == len(independently_selected) == int(spec["k"]), f"Alert cardinality mismatch: {record_key}")
            require(np.array_equal(np.sort(selected), independently_selected), f"Wrong selected accounts or tie resolution: {record_key}")
            require("budgets" in archive and np.array_equal(archive["budgets"], np.asarray(BUDGETS)), "Selected file budget identity")
            expected_id = entry.get("entry_id", "/".join(str(entry[c]) for c in IDENTITY))
            require("entry_id" in archive and str(archive["entry_id"].item()) == str(expected_id), "Selected file entry identity")

    def check_table(self, filename, expected, key_columns):
        actual = self.read_csv_index(filename, key_columns)
        require(set(actual) == set(expected), f"Exact row coverage mismatch: {filename}: {len(actual)} vs {len(expected)}")
        for key, row in actual.items():
            target = expected[key]
            require(set(row) == set(target), f"Schema mismatch: {filename}: {set(row) ^ set(target)}")
            for column, value in row.items():
                if column in key_columns and column not in INTEGER_COLUMNS and column != "budget":
                    continue
                if column in INTEGER_COLUMNS:
                    require(int(value) == int(target[column]) and float(value) == int(value), f"Integer mismatch: {filename}/{key}/{column}")
                elif isinstance(target[column], (int, float, np.integer, np.floating)):
                    difference = abs(float(value) - float(target[column]))
                    self.max_metric_difference = max(self.max_metric_difference, difference)
                    require(math.isfinite(float(value)) and difference <= 1e-15, f"Arithmetic mismatch: {filename}/{key}/{column}: {difference}")
                else:
                    require(str(value) == str(target[column]), f"Metadata mismatch: {filename}/{key}/{column}")
        return len(actual)

    def summarize(self):
        summary = {}
        for setting in sorted(expected_settings()):
            for budget in BUDGETS:
                key = setting + (budget,)
                folds = [self.fold_values[setting + (seed, fold, budget)] for seed in range(5) for fold in range(5)
                         if setting + (seed, fold, budget) in self.fold_values]
                if len(folds) == 25:
                    precision_mean, precision_sd = moment(row["precision_at_k"] for row in folds)
                    recall_mean, recall_sd = moment(row["recall_at_k"] for row in folds)
                    summary[key] = {**dict(zip(SETTING_BUDGET_KEY, key)), "precision_mean": precision_mean,
                                    "precision_sd": precision_sd, "recall_mean": recall_mean, "recall_sd": recall_sd}
                for seed in range(5):
                    records = [self.fold_values[setting + (seed, fold, budget)] for fold in range(5)
                               if setting + (seed, fold, budget) in self.fold_values]
                    if len(records) != 5:
                        continue
                    sums = {column: sum(row[column] for row in records) for column in ("k", "n_test", "n_positive", "tp")}
                    require(sums["n_test"] == len(self.canonical[setting[0]]) and sums["n_positive"] == int(self.canonical[setting[0]].sum()), "Seed must cover full canonical dataset")
                    seed_key = setting + (seed, budget)
                    self.seed_values[seed_key] = {**dict(zip(SEED_KEY, seed_key)), **sums,
                                                  "precision_at_k": sums["tp"] / sums["k"],
                                                  "recall_at_k": sums["tp"] / sums["n_positive"], "n_folds": 5}
        if not self.allow_incomplete:
            require(len(summary) == 126 and len(self.seed_values) == 630, "Summary/seed completeness")
        return summary

    def old1350(self, independent_summary):
        spec = self.manifest.get("legacy_budget_fold_values", self.manifest.get("legacy_budget"))
        require(spec is not None, "SHA-bound old 1350-row budget evidence required")
        path = self.bind(spec)
        with path.open(newline="") as stream:
            old = list(csv.DictReader(stream))
        require(len(old) == 1350, "Legacy file must have exactly 1350 rows")
        seen = set()
        max_difference = 0.0
        for row in old:
            key = typed_key(row, FOLD_KEY)
            require(key not in seen, "Duplicate old budget row")
            seen.add(key)
            target = self.fold_values[key]
            require(set(row) == set(target), "Legacy twelve-column schema changed")
            for column in row:
                if column in INTEGER_COLUMNS:
                    require(int(row[column]) == target[column], f"Legacy exact count differs: {key}/{column}")
                elif column in ("precision_at_k", "recall_at_k", "budget"):
                    difference = abs(float(row[column]) - float(target[column]))
                    max_difference = max(max_difference, difference)
                    require(difference <= 1e-15, f"Legacy floating ratio differs: {key}/{column}")
                else:
                    require(row[column] == target[column], f"Legacy key differs: {key}/{column}")
        expected = {(d, m, f, s, k, b) for d in ("hi_small", "li_small")
                    for m in ("lightgbm", "xgboost", "random_forest")
                    for f in ("base16", "gmke44", "gmkep56") for s in range(5) for k in range(5) for b in BUDGETS}
        require(seen == expected, "Legacy exact 18x25x3 coverage differs")
        require("legacy_budget_summary" in self.manifest, "SHA-bound old 54-row summary required")
        summary_path = self.bind(self.manifest["legacy_budget_summary"])
        with summary_path.open(newline="") as stream:
            old_summary = list(csv.DictReader(stream))
        require(len(old_summary) == 54, "Legacy budget summary must have exactly 54 rows")
        summary_seen = set()
        for record in old_summary:
            key = typed_key(record, SETTING_BUDGET_KEY)
            require(key not in summary_seen and key in independent_summary, "Legacy summary identity mismatch")
            summary_seen.add(key)
            target = independent_summary[key]
            require(set(record) == set(target), "Legacy eight-column summary schema changed")
            for column in ("precision_mean", "precision_sd", "recall_mean", "recall_sd"):
                require(abs(float(record[column]) - target[column]) <= 1e-15, f"Legacy summary arithmetic mismatch: {key}/{column}")
        require(summary_seen == {(k[0], k[1], k[2], k[5]) for k in seen}, "Legacy summary coverage differs")
        return {"rows": len(old), "summary_rows": len(old_summary), "all_counts_exact": True, "max_absolute_ratio_difference": max_difference,
                "scope": "Old per-fold budget values only; old p-value/ Holm analyses are not reused."}

    def verify_pairs(self):
        definitions = self.manifest["pairs"]
        expanded = []
        for pair in definitions:
            for dataset in pair.get("datasets", [pair["dataset"]] if "dataset" in pair else ["hi_small", "li_small"]):
                left, right = pair["left"], pair["right"]
                expanded.append((dataset, str(pair.get("id", pair.get("pair_id"))), str(left["model"]), str(left["feature_set"]),
                                 str(right["model"]), str(right["feature_set"])))
        require(len(expanded) == len(set((p[0], p[1]) for p in expanded)) == 74, "Expected 74 unique dataset-pairs")
        present = {(p[0], p[2], p[3], p[4], p[5]) for p in expanded}
        for dataset in ("hi_small", "li_small"):
            for model in ("lightgbm", "xgboost", "random_forest", "mlp", "graphsage_weak"):
                arms = ("base16", "base_g32", "gmke44", "gmkep56")
                for higher in range(1, 4):
                    for lower in range(higher):
                        require((dataset, model, arms[higher], model, arms[lower]) in present, "Missing within-model six-input-pair contrast")
            require((dataset, "xgboost", "base_gfp_account_stream", "xgboost", "gmkep56") in present, "Missing GFP-minus-XGB56")
            for arm in ("base16", "base_g32", "gmke44", "gmkep56"):
                require((dataset, "graphsage_weak", arm, "mlp", arm) in present, "Missing SAGE-minus-MLP same-input contrast")
            require((dataset, "graphsage_weak", "base16", "xgboost", "gmkep56") in present,
                    "Missing SAGE-Base-minus-XGB56 context contrast")
            require((dataset, "mlp", "gmkep56", "xgboost", "gmkep56") in present,
                    "Missing MLP56-minus-XGB56 context contrast")
        pair_fold, pair_seed, pair_summary = {}, {}, {}
        for dataset, pair_id, lm, lf, rm, rf in expanded:
            for budget in BUDGETS:
                metadata = {"dataset": dataset, "pair_id": pair_id, "left_model": lm, "left_feature_set": lf,
                            "right_model": rm, "right_feature_set": rf, "budget": budget}
                folded, seeded = [], []
                for seed in range(5):
                    for fold in range(5):
                        if ((dataset, lm, lf, seed, fold, budget) not in self.fold_values
                            or (dataset, rm, rf, seed, fold, budget) not in self.fold_values):
                            continue
                        left = self.fold_values[(dataset, lm, lf, seed, fold, budget)]
                        right = self.fold_values[(dataset, rm, rf, seed, fold, budget)]
                        row = self.paired_row(metadata, left, right, seed, fold)
                        pair_fold[(dataset, pair_id, seed, fold, budget)] = row
                        folded.append(row)
                    if ((dataset, lm, lf, seed, budget) in self.seed_values
                        and (dataset, rm, rf, seed, budget) in self.seed_values):
                        left = self.seed_values[(dataset, lm, lf, seed, budget)]
                        right = self.seed_values[(dataset, rm, rf, seed, budget)]
                        row = self.paired_row(metadata, left, right, seed, None)
                        pair_seed[(dataset, pair_id, seed, budget)] = row
                        seeded.append(row)
                if len(folded) != 25 or len(seeded) != 5:
                    continue
                pm, ps = moment(row["delta_precision_at_k"] for row in folded)
                rm_, rs = moment(row["delta_recall_at_k"] for row in folded)
                result = {**metadata, "n_folds": 25, "n_seeds": 5, "precision_delta_mean": pm,
                          "precision_delta_sd": ps, "recall_delta_mean": rm_, "recall_delta_sd": rs}
                for metric in ("precision", "recall"):
                    field = "delta_" + metric + "_at_k"
                    for suffix, records in (("folds", folded), ("seeds", seeded)):
                        differences = [row[field] for row in records]
                        result[f"{metric}_positive_{suffix}"] = sum(value > 0 for value in differences)
                        result[f"{metric}_zero_{suffix}"] = sum(value == 0 for value in differences)
                        result[f"{metric}_negative_{suffix}"] = sum(value < 0 for value in differences)
                pair_summary[(dataset, pair_id, budget)] = result
        if not self.allow_incomplete:
            require(len(pair_fold) == 5550 and len(pair_seed) == 1110 and len(pair_summary) == 222, "Pair row completeness")
        return pair_fold, pair_seed, pair_summary

    @staticmethod
    def paired_row(metadata, left, right, seed, fold):
        for column in ("k", "n_test", "n_positive"):
            require(left[column] == right[column], f"Paired evaluation denominators differ: {metadata}")
        row = {**metadata, "seed": seed, "k": left["k"], "n_test": left["n_test"],
               "n_positive": left["n_positive"], "tp_left": left["tp"], "tp_right": right["tp"],
               "delta_tp": left["tp"] - right["tp"],
               "delta_precision_at_k": left["precision_at_k"] - right["precision_at_k"],
               "delta_recall_at_k": left["recall_at_k"] - right["recall_at_k"]}
        if fold is None:
            row["n_folds"] = 5
        else:
            row["fold"] = fold
        return row

    def run(self):
        self.bind_protocol()
        self.load_canonical()
        self.recompute_folds()
        require(set(self.selection_values) == set(self.fold_values), "Unexpected selection references")
        summary = self.summarize()
        pair_fold, pair_seed, pair_summary = self.verify_pairs()
        rows = {
            "fold": self.check_table("budget_fold_values.csv", self.fold_values, FOLD_KEY),
            "summary": self.check_table("budget_summary.csv", summary, SETTING_BUDGET_KEY),
            "seed": self.check_table("budget_seed_values.csv", self.seed_values, SEED_KEY),
            "pair_fold": self.check_table("budget_paired_fold_values.csv", pair_fold, PAIR_FOLD_KEY),
            "pair_seed": self.check_table("budget_paired_seed_values.csv", pair_seed, PAIR_SEED_KEY),
            "pair_summary": self.check_table("budget_paired_summary.csv", pair_summary, PAIR_SUMMARY_KEY),
        }
        operating = {key: {**dict(zip(FOLD_KEY, key)), "fp": row["k"] - row["tp"],
                           "actual_fraction": row["k"] / row["n_test"]} for key, row in self.fold_values.items()}
        rows["operating_counts"] = self.check_table("budget_operating_counts.csv", operating, FOLD_KEY)
        legacy = self.old1350(summary)
        # Acceptance pins each input twice, rejecting an input changed during analysis.
        for path, expected_hash in self.bound_hashes.items():
            require(digest(path) == expected_hash, f"Source changed during independent acceptance: {path}")
        for path, expected_hash in self.checked_output_hashes.items():
            require(digest(path) == expected_hash, f"Published output changed during acceptance: {path}")
        require(digest(self.manifest_path) == self.initial_manifest_sha256, "Manifest changed during independent acceptance")
        complete = self.processed_entries == 1050
        return {"passed": complete, "available_scope_passed": True, "full_analysis_complete": complete,
                "status": "independent_saved_score_budget_acceptance" if complete else "PARTIAL_INDEPENDENTLY_CHECKED_NOT_COMPLETE",
                "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                "manifest_path": str(self.manifest_path), "manifest_sha256": digest(self.manifest_path),
                "verifier_sha256": digest(Path(__file__)), "rows": rows, "legacy_reproduction": legacy,
                "processed_entries": self.processed_entries, "missing_entries": self.missing_entries,
                "complete_settings": len(summary) // 3, "expected_settings": 42,
                "independent_ap_checks": len(self.ap_checks), "max_absolute_ap_difference": self.max_ap_difference,
                "fold_ap_checks": self.ap_checks,
                "max_absolute_output_metric_difference": self.max_metric_difference,
                "input_sha256": self.bound_hashes, "output_sha256": self.checked_output_hashes,
                "checks": {"canonical_accounts_labels_and_outer_test_indices": True,
                           "formal_fold_ap_reconstructed_by_tie_blocks": True,
                           "formal_ap_csv_identity_strategy_policy_counts_and_sha_bound": True,
                           "exact_k_sets_partition_boundary_and_shared_original_index_ties": True,
                           "all42_settings_and25_folds_complete": complete,
                           "old1350_fold_counts_reproduced": True,
                           "seed_counts_union_fold_selected_alerts_no_cross_fold_ranking": True,
                           "all_within_model_pairs_and_registered_cross_method_pairs": True,
                           "only_tuned_continuous_scores_no_duplicate_thresholds": True,
                           "no_training_model_inference_or_calibration": True,
                           "descriptive_only_no_p_values": True},
                "scope": "Saved outer-test scores only; no new fits, model inference, calibration, hyperparameter search or typology analysis. Mean/SD and paired direction counts are descriptive; repeats and graph accounts are dependent."}


def self_test():
    # Deliberately scrambled row order and ties exactly at the cutoff.
    rows = np.asarray([11, 3, 6, 2, 9], dtype=np.int64)
    scores = np.asarray([0.2, 0.8, 0.8, 0.8, 0.1])
    picked, k = independent_alert_indices(rows, scores, 0.02)
    require(k == 1 and rows[picked].tolist() == [2], "Tie must use original account index")
    for n in (1, 49, 50, 51, 99, 100, 101, 199, 200, 201):
        for budget in BUDGETS:
            ids = np.arange(n)[::-1]
            score = np.ones(n)
            selected, count = independent_alert_indices(ids, score, budget)
            expected = math.ceil(n * budget)
            require(count == expected and np.array_equal(np.sort(ids[selected]), np.arange(count)), "Ceil/tie boundary")
    require(abs(independent_ap([1, 0, 1], [0.8, 0.8, 0.1]) - 7 / 12) < 1e-15, "Independent tied AP")
    require(abs(independent_ap([1, 0], [0.2, 0.2]) - 0.5) < 1e-15, "All-tie AP")
    require(independent_ap([1, 1], [0.1, 0.2]) == 1, "All-positive AP")
    require(len(expected_settings()) == 42, "Setting inventory")
    require(moment([1, 2, 3]) == (2, 1), "Sample SD must use ddof1")
    rng = np.random.default_rng(20261028)
    for _ in range(100):
        n = int(rng.integers(2, 502))
        ids = rng.permutation(n) + 1000
        score = rng.integers(0, 7, size=n).astype(np.float64)
        labels = rng.integers(0, 2, size=n, dtype=np.int8)
        labels[0] = 1
        for budget in BUDGETS:
            selected, count = independent_alert_indices(ids, score, budget)
            brute = sorted(range(n), key=lambda i: (-float(score[i]), int(ids[i])))[:count]
            require(set(selected.tolist()) == set(brute), "Partition top-k differs from Python oracle")
        ap_reference = 0.0
        prior_tp = 0
        for cut in sorted(set(score.tolist()), reverse=True):
            flagged = score >= cut
            tp = int(labels[flagged].sum())
            ap_reference += (tp - prior_tp) / int(labels.sum()) * tp / int(flagged.sum())
            prior_tp = tp
        require(abs(independent_ap(labels, score) - ap_reference) < 1e-14, "Tie-block AP differs from direct threshold oracle")
    return {"passed": True, "tests": ["scrambled-index cutoff tie", "exact ceil at budget boundaries",
                                    "all-tie original-index order", "tie-block AP", "all-tie AP",
                                    "all-positive AP", "42-setting closure", "sample SD ddof1",
                                    "300 randomized top-k checks against Python ordering",
                                    "100 randomized AP checks against direct threshold counts"],
            "no_experiment_data_read": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest")
    parser.add_argument("--results-dir")
    parser.add_argument("--output")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), ensure_ascii=False))
        return
    require(args.manifest and args.results_dir and args.output, "--manifest, --results-dir and --output required")
    try:
        report = Verifier(args.manifest, args.results_dir, args.allow_incomplete).run()
    except Exception as error:
        atomic_json(args.output, {"passed": False, "status": "rejected",
                                 "error_type": type(error).__name__, "error": str(error),
                                 "verifier_sha256": digest(Path(__file__))})
        raise
    atomic_json(args.output, report)
    print(json.dumps({"passed": report["passed"], "available_scope_passed": report["available_scope_passed"],
                      "full_analysis_complete": report["full_analysis_complete"], "status": report["status"], "rows": report["rows"],
                      "legacy_reproduction": report["legacy_reproduction"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
