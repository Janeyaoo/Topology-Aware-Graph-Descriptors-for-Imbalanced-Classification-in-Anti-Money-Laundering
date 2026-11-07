"""Independently validate the paired CUDA MLP experiment without fitting models.

The validator never imports a training runner. It reconstructs fit-only
preprocessing and the two affine layers from retained CPU state tensors,
recomputes selection, thresholds and classification metrics, and admits only
complete 25-fold settings to aggregate tables. Repeated folds and seed OOF
direction counts are descriptive, not independent replications.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import os
from pathlib import Path
import resource
import sys
import traceback
from datetime import datetime, timezone

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


SCRIPT = Path(__file__).resolve()
ROOT = Path(".").resolve()
DATASETS = ("hi_small", "li_small")
ARMS = ("base16", "base_g32", "gmke44", "gmkep56")
WIDTHS = dict(zip(ARMS, (16, 32, 44, 56)))
STRATEGIES = ("same_protocol_anchor", "tuned")
POLICIES = ("fixed_0.5", "inner_max_f1")
MAIN_METRICS = ("ap", "auc", "precision", "recall", "f1", "accuracy", "balanced_accuracy")
METRICS = MAIN_METRICS + ("specificity", "fpr", "fnr", "tn", "fp", "fn", "tp", "n_test", "n_positive")
CONTRASTS = tuple((ARMS[j], ARMS[i]) for i in range(4) for j in range(i + 1, 4))
TOL = 1e-12
HASH_CACHE = {}
REFERENCE_MLP_SHA256 = "87577bfc6337c4e8e612cc7a16b9287d321f4a54633ac38e30024ce544c3e8e4"


def now():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(filename):
    return json.loads(Path(filename).read_text(encoding="utf-8"))


def write(filename, value):
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    temporary = filename.with_name(filename.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, filename)


def sha(filename):
    filename = Path(filename).resolve()
    stat = filename.stat()
    # No persisted mtime-only trust: every new verifier process hashes bytes.
    key = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    if filename in HASH_CACHE and HASH_CACHE[filename][0] == key:
        return HASH_CACHE[filename][1]
    result = hashlib.sha256()
    with filename.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    value = result.hexdigest()
    HASH_CACHE[filename] = (key, value)
    return value


def ash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def absolute(name):
    item = Path(name)
    return (ROOT / item if not item.is_absolute() else item).resolve()


def path(name):
    item = absolute(name)
    require(item.is_relative_to(ROOT) and item.is_file(), "Missing or escaping result artifact: " + str(name))
    return item


def rel(name):
    return Path(name).resolve().relative_to(ROOT).as_posix()


def source_hashes(mapping):
    require(isinstance(mapping, dict) and bool(mapping), "Nonempty source SHA mapping required")
    result = {}
    for name, digest in mapping.items():
        require(isinstance(digest, str) and len(digest) == 64, "SHA256 source digest required")
        item = absolute(name)
        require(item.is_file() and sha(item) == digest, "Source/input SHA differs: " + str(item))
        result[str(item)] = digest
    return result


def binding(mapping, receipt=None):
    require(isinstance(mapping, dict) and bool(mapping), "Nonempty result SHA mapping required")
    for name, expected in mapping.items():
        item = path(name)
        require(sha(item) == expected, "Result artifact SHA differs: " + str(item))
        if receipt is not None:
            receipt["verified_artifact_sha256"][str(item)] = expected


def close(actual, expected, label, atol=TOL, rtol=0):
    actual, expected = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    require(actual.shape == expected.shape and np.isfinite(actual).all() and np.isfinite(expected).all()
            and np.allclose(actual, expected, atol=atol, rtol=rtol), label)
    return float(np.max(np.abs(actual - expected), initial=0))


def peak_rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def setup_cpu():
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        require(torch.get_num_interop_threads() == 1, "Torch interop threads must equal one")


def ranking_metrics(y, score):
    """Independent complete-score AP and AUROC using exact score tie groups."""
    y, score = np.asarray(y, np.int8), np.asarray(score, np.float64)
    require(y.ndim == score.ndim == 1 and len(y) == len(score) and set(np.unique(y)) == {0, 1}
            and np.isfinite(score).all(), "Finite complete binary ranking population")
    order = np.argsort(-score, kind="stable")
    labels, values = y[order], score[order]
    ends = np.r_[np.flatnonzero(values[:-1] != values[1:]), len(y) - 1]
    tp = np.cumsum(labels, dtype=np.int64)[ends]
    delta_p = np.diff(np.r_[0, tp])
    delta_n = np.diff(np.r_[0, ends + 1]) - delta_p
    positive, negative = int(y.sum()), len(y) - int(y.sum())
    ap = float(np.sum(delta_p * (tp / (ends + 1))) / positive)
    auc = float(np.sum(delta_p * (negative - np.cumsum(delta_n) + .5 * delta_n)) / (positive * negative))
    return ap, auc


def metrics(y, score, predicted):
    y, predicted = np.asarray(y, np.int8), np.asarray(predicted, np.int8)
    require(y.shape == predicted.shape and set(np.unique(predicted)) <= {0, 1}, "Binary threshold outputs")
    tn = int(np.sum((y == 0) & (predicted == 0)))
    fp = int(np.sum((y == 0) & (predicted == 1)))
    fn = int(np.sum((y == 1) & (predicted == 0)))
    tp = int(np.sum((y == 1) & (predicted == 1)))
    ap, auc = ranking_metrics(y, score)
    precision = tp / (tp + fp) if tp + fp else 0.
    recall, specificity = tp / (tp + fn), tn / (tn + fp)
    return dict(ap=ap, auc=auc, precision=precision, recall=recall,
                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
                accuracy=(tp + tn) / len(y), balanced_accuracy=(recall + specificity) / 2,
                specificity=specificity, fpr=1 - specificity, fnr=1 - recall,
                tn=tn, fp=fp, fn=fn, tp=tp, n_test=len(y), n_positive=int(y.sum()))


def threshold(y, score):
    """Exact rational F1 comparison, with the smallest score threshold on ties."""
    y, score = np.asarray(y, np.int8), np.asarray(score, np.float64)
    order = np.argsort(score, kind="stable")
    values, labels = score[order], y[order]
    starts = np.r_[0, np.flatnonzero(values[:-1] != values[1:]) + 1]
    suffix_p = np.cumsum(labels[::-1], dtype=np.int64)[::-1]
    positive = int(y.sum())
    best_index = 0
    best_num = -1
    best_den = 1
    for j, position in enumerate(starts):
        numerator, denominator = 2 * int(suffix_p[position]), len(y) - int(position) + positive
        if numerator * best_den > best_num * denominator:
            best_index, best_num, best_den = j, numerator, denominator
    return float(values[starts[best_index]]), float(best_num / best_den)


def state_forward(state, X, rows, chunk=16384):
    """Two affine layers, first ReLU, no dropout in evaluation; CPU float32."""
    rows = np.asarray(rows)
    require(X.ndim == 2 and X.dtype == np.float32 and chunk == 16384, "Frozen CPU float32 inference contract")
    result = np.empty(len(rows), np.float64)
    with torch.no_grad():
        for start in range(0, len(rows), chunk):
            index = rows[start:start + chunk]
            x = torch.from_numpy(np.ascontiguousarray(X[index]))
            hidden = torch.relu(torch.nn.functional.linear(x, state["layers.0.weight"], state["layers.0.bias"]))
            logits = torch.nn.functional.linear(hidden, state["layers.1.weight"], state["layers.1.bias"]).squeeze(-1)
            result[start:start + len(index)] = torch.sigmoid(logits).double().numpy()
    return result


class Validator:
    def __init__(self, out):
        global ROOT
        ROOT = Path(out).resolve()
        self.root, self.dest = ROOT, ROOT / "validation"
        self.cache, self.split_cache, self.raw_digests = {}, {}, {}
        setup_cpu()
        self.p = read(ROOT / "protocol.json")
        self.psha = sha(ROOT / "protocol.json")
        self.c = read(ROOT / "candidates.json")["candidates"]
        self.sources = {}
        self.source_gate()

    def source_gate(self):
        expected = [{"candidate_id": i, "params": {"hidden_dim": h, "learning_rate": lr, "dropout": d}}
                    for i, (h, lr, d) in enumerate(itertools.product((64, 128, 256), (.001, .003), (0., .2)))]
        require(self.c == expected, "Original ordered twelve neural candidates must remain unchanged")
        require(self.p.get("model") == "mlp" and str(self.p.get("device", "")).startswith("cuda"), "One CUDA MLP backend; historical MPS excluded")
        require(set(self.p["datasets"]) == set(DATASETS) and self.p["feature_sets"] == list(ARMS), "Two datasets and all four representations registered")
        require(self.p["budget"] == {"arms_per_dataset": 4, "outer_folds_per_setting": 25, "outer_jobs": 200,
                                      "inner_candidate_scores_per_outer": 36, "stages_per_outer": 4}, "Full 200-outer / 8000-fit budget")
        execution = self.p["execution_settings"]
        expected_execution = dict(cpu_threads=2, interop_threads=1, prediction_chunk_size=16384,
                fit_dtype="float32", prediction_device="cpu", prediction_dtype="float32->float64",
                autocast=False, tf32=False, deterministic_algorithms=True, preloaded_tensors=True)
        require(all(execution.get(k) == v for k, v in expected_execution.items()), "Registered deterministic float32 CUDA / CPU2 inference contract")
        require(self.p.get("historical_mps_scores_reused") is False and self.p.get("graph_read_by_mlp") is False,
                "No historical MPS score migration or neighbor aggregation by MLP")
        sourcefile = ROOT / self.p.get("source_binding_path", "source_binding.json")
        require(sha(sourcefile) == self.p["source_binding_sha256"], "Protocol-bound immutable source manifest")
        raw_binding = read(sourcefile)
        mapping = raw_binding.get("source_sha256", raw_binding)
        self.sources.update(source_hashes(mapping))
        self.sources.update({str(ROOT / "protocol.json"): self.psha,
                             str(ROOT / "candidates.json"): sha(ROOT / "candidates.json"), str(sourcefile): sha(sourcefile)})
        snapshot = ROOT / "runner_snapshot.py"
        require(snapshot.is_file() and sha(snapshot) == self.p["runner_sha256"], "Immutable runner snapshot agrees with registered source")
        self.sources[str(snapshot)] = sha(snapshot)
        references = [name for name, digest in self.sources.items()
                      if Path(name).name == "run_account_eswa_supplement_neural_20261013.py" and digest == REFERENCE_MLP_SHA256]
        require(bool(references), "Unchanged original MLP training recipe is retained and hash bound")
        probe = ROOT / "probe_cuda.json"
        value = read(probe)
        require(value.get("passed") is True and str(value.get("device", "")).startswith("cuda")
                and value.get("runner_sha256") == self.p["runner_sha256"], "Target CUDA probe passed for the exact registered runner")
        reports = value.get("reports", [])
        require(value.get("reference_sha256") == REFERENCE_MLP_SHA256 and value.get("synthetic_only") is True
                and value.get("outer_test_evaluated") is False
                and value.get("cpu16_reference_model_scaler_seed_loss_prediction_bitwise_equal") is True,
                "Original MLP model/scaler/seed loss/inference recipe equivalence and synthetic-only CUDA QA")
        require(len(reports) == 4 and {x["input_dimension"] for x in reports} == set(WIDTHS.values())
                and all(x.get("interrupted_minibatch_resume_bitwise_equal") is True and x.get("scores_bitwise_equal") is True
                        and x.get("history_bitwise_equal_excluding_time") is True for x in reports), "All four widths have bitwise CUDA interruption recovery checks")
        if "protocol_sha256" in value:
            require(value["protocol_sha256"] == self.psha, "CUDA probe binds current protocol")
        self.sources[str(probe)] = sha(probe)
        package = Path(self.p["input_package"]).resolve()
        self.input_folder = package / "inputs" if (package / "inputs").is_dir() else package
        for ds in DATASETS:
            record = self.p["datasets"][ds]
            manifest = ROOT / f"input_manifest_{ds}.json"
            require(sha(manifest) == record["input_binding_sha256"]
                    and read(manifest) == {k: v for k, v in record.items() if k != "input_binding_sha256"}, "Dataset manifest file exactly binds prepared input registry")
            self.sources[str(manifest)] = sha(manifest)
            bound = source_hashes(record["input_files"])
            must = [self.input_folder / ds / x for x in ("features.npy", "names.json", "labels.npy", "accounts.parquet")]
            must += [self.input_folder / ds / "splits" / f"seed{s}_fold{f}.npz" for s, f in itertools.product(range(5), repeat=2)]
            require(all(str(item.resolve()) in bound for item in must), "All original feature, identity, label and 25 split files bound: " + ds)
            self.sources.update(bound)
        self.source_manifest_sha = sha(sourcefile)

    def data(self, ds, arm):
        key = ds, arm
        if key in self.cache:
            return self.cache[key]
        self.cache.clear()
        gc.collect()
        folder = self.input_folder / ds
        names = read(folder / "names.json")
        allcols, base = names["columns"], names["base16"]
        columns = names["feature_sets"][arm]
        require(len(allcols) == len(set(allcols)) == 56 and len(base) == 16 and allcols[:16] == base,
                "Original 56-column dictionary and BASE16 prefix")
        require(len(columns) == len(set(columns)) == WIDTHS[arm] and columns[:16] == base
                and all(c in allcols for c in columns), "Registered representation width and common BASE16")
        record = self.p["datasets"][ds]
        require(record["feature_sets"][arm] == columns and record["base16"] == base, "Protocol feature dictionary matches inputs")
        original = np.load(folder / "features.npy", mmap_mode="r", allow_pickle=False)
        y = np.load(folder / "labels.npy", mmap_mode="r", allow_pickle=False)
        ids = pd.read_parquet(folder / "accounts.parquet")
        require(original.shape == (len(ids), 56) and original.dtype == np.float64
                and y.shape == (len(ids),) and y.dtype == np.int8, "Original float64 matrix and int8 labels")
        require(set(np.unique(y)) == {0, 1} and np.array_equal(ids.row_index, np.arange(len(y)))
                and ids.account_id.is_unique, "All account rows retained exactly once")
        if "label" in ids:
            require(np.array_equal(ids.label, y), "Identity label agrees with immutable label array")
        require(record["n_samples"] == len(y) and record["n_positive"] == int(y.sum()), "Original account and positive counts")
        if "labels_array_sha256" in record:
            require(record["labels_array_sha256"] == ash(y), "Full label bytes agree with protocol")
        if "account_ids_hash_sha256" in record:
            require(record["account_ids_hash_sha256"] == ash(pd.util.hash_pandas_object(ids.account_id, index=False).to_numpy()), "Full account identity digest")
        index = [allcols.index(c) for c in columns]
        raw = np.empty((len(y), len(columns)), np.float32)
        for start in range(0, len(y), 65536):
            stop = min(start + 65536, len(y))
            block = np.array(original[start:stop, index], np.float64, copy=True)
            require(np.isfinite(block).all() and (block[:, :16] >= 0).all(), "Finite original features and nonnegative BASE16")
            block[:, :16] = np.log1p(block[:, :16])
            raw[start:stop] = block.astype(np.float32)
        require(np.isfinite(raw).all(), "Finite independently transformed feature matrix")
        self.raw_digests[key] = ash(raw), ash(y)
        self.cache[key] = raw, y, ids, columns
        return self.cache[key]

    def split(self, ds, seed, fold):
        key = ds, seed, fold
        if key in self.split_cache:
            return self.split_cache[key]
        filename = self.input_folder / ds / "splits" / f"seed{seed}_fold{fold}.npz"
        require(str(filename) in self.sources and sha(filename) == self.sources[str(filename)], "Original split file bytes")
        with np.load(filename, allow_pickle=False) as archive:
            arrays = {k: archive[k] for k in archive.files}
        expected = {"outer_train", "outer_test", "development", "threshold_validation"}
        expected |= {f"{prefix}_{j}" for prefix in ("inner_train", "inner_score", "early_fit", "early_validation") for j in range(3)}
        require(set(arrays) == expected, "Full original sixteen split arrays")
        n = self.p["datasets"][ds]["n_samples"]
        for name, value in arrays.items():
            require(value.ndim == 1 and np.issubdtype(value.dtype, np.integer) and len(np.unique(value)) == len(value)
                    and (value >= 0).all() and (value < n).all(), "Unique legal split rows: " + name)
        def partition(a, b, parent, label):
            require(not np.intersect1d(a, b).size and np.array_equal(np.sort(np.r_[a, b]), np.sort(parent)), label)
        partition(arrays["outer_train"], arrays["outer_test"], np.arange(n), "Full outer partition")
        partition(arrays["development"], arrays["threshold_validation"], arrays["outer_train"], "D/T partition within outer training")
        for j in range(3):
            partition(arrays[f"inner_train_{j}"], arrays[f"inner_score_{j}"], arrays["development"], "Inner fit/score partition")
            partition(arrays[f"early_fit_{j}"], arrays[f"early_validation_{j}"], arrays[f"inner_train_{j}"], "Early fit/validation partition")
        require(np.array_equal(np.sort(np.concatenate([arrays[f"inner_score_{j}"] for j in range(3)])), np.sort(arrays["development"])), "Three inner score folds cover D exactly once")
        self.split_cache[key] = arrays
        return arrays

    def checkpoint(self, filename, ds, arm, fit_rows, early_rows, params, seed, rows, saved_score, epochs, receipt):
        filename = path(filename)
        payload = torch.load(filename, map_location="cpu", weights_only=True)
        metadata, state = payload["metadata"], payload["state_dict"]
        raw, y, _, columns = self.data(ds, arm)
        require(metadata == read(filename.with_suffix(".json")), "Model PT/JSON metadata identity")
        require(metadata["model"] == "mlp" and metadata.get("feature_set", arm) == arm and metadata["params"] == params
                and metadata["random_state"] == seed and metadata["device"] == self.p["device"], "MLP arm, parameters, seed and CUDA backend")
        dimension = metadata.get("input_dimension", metadata.get("input_dim"))
        require(dimension == len(columns) and metadata["protocol_sha256"] == self.psha
                and metadata["input_binding_sha256"] == self.p["datasets"][ds]["input_binding_sha256"], "Model dimension and immutable protocol/input identity")
        require(metadata.get("prediction_threads") == 2 and metadata.get("prediction_chunk_size") == 16384, "CPU2 / chunk16384 prediction contract")
        expected_fit_binding = dict(protocol_sha256=self.psha, input_binding_sha256=self.p["datasets"][ds]["input_binding_sha256"],
                raw_sha256=self.raw_digests[ds, arm][0], labels_sha256=self.raw_digests[ds, arm][1], fit_rows_sha256=ash(fit_rows),
                early_rows_sha256=None if early_rows is None else ash(early_rows), params=params, seed=seed,
                input_dimension=len(columns), device=self.p["device"], epochs=None if early_rows is not None else epochs,
                probe_epoch_limit=None)
        require(metadata["fit_binding"] == expected_fit_binding and metadata["fit_binding_sha256"] == canonical_sha(expected_fit_binding),
                "Full retained fit context, including transformed feature bytes, labels, epochs and seed")
        require(metadata.get("preloaded_tensors") is True and metadata.get("autocast") is False and metadata.get("tf32") is False,
                "Actual checkpoint belongs to registered float32 execution")
        completionfile = filename.with_suffix(".fit_done.json")
        completion = read(completionfile)
        require(completion["fit_binding_sha256"] == canonical_sha(expected_fit_binding), "Fit completion seal binds full fit context")
        binding(completion["artifact_sha256"], receipt)
        require(absolute(str(filename)) in {absolute(x) for x in completion["artifact_sha256"]}
                and filename.with_suffix(".json") in {absolute(x) for x in completion["artifact_sha256"]}, "Completion seal includes retained model and metadata")
        receipt["verified_artifact_sha256"][str(completionfile)] = sha(completionfile)
        require(metadata["fit_rows_sha256"] == ash(fit_rows) and metadata["scaler_fit_rows_sha256"] == ash(fit_rows)
                and metadata["early_rows_sha256"] == (None if early_rows is None else ash(early_rows))
                and metadata["scaler_n_samples_seen"] == len(fit_rows)
                and (metadata["fit_n"], metadata["fit_positive"]) == (len(fit_rows), int(y[fit_rows].sum()))
                and metadata["seed_only_loss"] is True, "Fit-only scaler, class weights and loss scope")
        fitraw = np.asarray(raw[fit_rows], np.float64)
        mean = fitraw.mean(axis=0)
        centered = fitraw - mean
        variance = (np.square(centered).sum(axis=0) - np.square(centered.sum(axis=0)) / len(fitraw)) / len(fitraw)
        eps = np.finfo(np.float64).eps
        constant = variance <= len(fitraw) * eps * variance + (len(fitraw) * mean * eps) ** 2
        scale = np.sqrt(variance)
        scale[constant] = 1.
        for name, expected in (("scaler_mean", mean), ("scaler_var", variance), ("scaler_scale", scale)):
            close(metadata[name], expected, "Independent fit-only " + name, rtol=1e-11)
        n, positive = len(fit_rows), int(y[fit_rows].sum())
        weights = np.array([n / (2 * (n - positive)), n / (2 * positive)], np.float32)
        require(np.array_equal(weights, np.asarray(metadata["class_weights"], np.float32)), "Fit-only balanced BCE weights")
        del fitraw, centered
        hidden = int(params["hidden_dim"])
        shapes = {"layers.0.weight": (hidden, len(columns)), "layers.0.bias": (hidden,),
                  "layers.1.weight": (1, hidden), "layers.1.bias": (1,)}
        require(set(state) == set(shapes) and all(tuple(state[k].shape) == shape and state[k].dtype == torch.float32
                and state[k].device.type == "cpu" and torch.isfinite(state[k]).all() for k, shape in shapes.items()), "Exactly two float32 affine layers; no identity embedding")
        digest = hashlib.sha256()
        for name, value in state.items():
            digest.update(name.encode())
            digest.update(value.contiguous().numpy().tobytes())
        require(digest.hexdigest() == metadata["model_digest"], "Raw state tensor digest")
        history = metadata["history"]
        require(history and [v["epoch"] for v in history] == list(range(1, len(history) + 1))
                and metadata["epochs_run"] == len(history) and metadata["selected_epochs"] == epochs,
                "Complete epoch trajectory and selected epoch")
        require(all(np.isfinite(v["weighted_seed_loss"]) and v["weighted_seed_loss"] >= 0 for v in history), "Finite weighted fit loss history")
        if early_rows is not None:
            aucs = np.array([v["early_auroc"] for v in history])
            require(np.isfinite(aucs).all() and np.argmax(aucs) + 1 == epochs and len(history) <= 200
                    and (len(history) == 200 or len(history) - epochs == 30), "Strict first-best early AUROC, patience30/max200")
            close(metadata["best_early_auroc"], aucs.max(), "Retained best early AUROC")
        else:
            require(metadata["best_early_auroc"] is None and len(history) == epochs, "Fixed selected-epoch refit")
        X = np.array(raw, np.float32, copy=True)
        np.subtract(X, np.asarray(metadata["scaler_mean"]), out=X, casting="unsafe")
        np.divide(X, np.asarray(metadata["scaler_scale"]), out=X, casting="unsafe")
        score = state_forward(state, X, rows)
        require(np.isfinite(score).all() and ((score >= 0) & (score <= 1)).all(), "Finite independently computed probabilities")
        difference = close(score, saved_score, "Independent CPU affine/ReLU/sigmoid full-score equality")
        require(ash(score) == ash(saved_score), "Exact CPU2 / chunk16384 saved probability bytes")
        if early_rows is not None:
            close(ranking_metrics(y[early_rows], state_forward(state, X, early_rows))[1], metadata["best_early_auroc"], "Best state independently reproduces early AUROC")
        receipt["physical_models_rescored"][str(filename)] = dict(sha256=sha(filename), n_features=len(columns),
                score_rows=len(rows), fit_rows=len(fit_rows), exact_score_bytes=True, max_abs_diff=difference,
                model_digest=metadata["model_digest"])
        del X, state, payload, score
        gc.collect()
        return metadata

    def locations(self, ds, arm, seed, fold):
        folder = ROOT / "runs" / ds / arm
        stem = f"mlp_seed{seed}_fold{fold}"
        return folder, stem, folder / "folds" / (stem + ".json"), folder / "folds" / (stem + "_DONE.json"), self.dest / "receipts" / ds / arm / f"seed{seed}_fold{fold}.json"

    def candidate(self, ds, arm, seed, fold, cid, inner, receipt):
        folder, stem, _, _, _ = self.locations(ds, arm, seed, fold)
        filename = folder / "search" / stem / f"candidate{cid}_inner{inner}_record.json"
        value = read(filename)
        binding(value["artifact_sha256"], receipt)
        receipt["verified_artifact_sha256"][str(filename)] = sha(filename)
        arrays = self.split(ds, seed, fold)
        _, y, _, _ = self.data(ds, arm)
        require([value[k] for k in ("dataset", "model", "feature_set", "seed", "fold", "candidate_id", "inner_fold")]
                == [ds, "mlp", arm, seed, fold, cid, inner] and value["protocol_sha256"] == self.psha
                and value["params"] == self.c[cid]["params"], "Candidate identity, parameters and protocol")
        require(value["input_dimension"] == WIDTHS[arm] and value["input_binding_sha256"] == self.p["datasets"][ds]["input_binding_sha256"], "Candidate arm dimension and immutable inputs")
        fit, early, rows = (arrays[f"{key}_{inner}"] for key in ("early_fit", "early_validation", "inner_score"))
        require(value["fit_rows_sha256"] == ash(fit) and value["early_rows_sha256"] == ash(early)
                and value["score_rows_sha256"] == ash(rows), "Candidate fit/early/score scopes")
        with np.load(path(value["score_path"]), allow_pickle=False) as scores:
            require(set(scores.files) == {"row_index", "y_true", "y_score"} and np.array_equal(scores["row_index"], rows)
                    and np.array_equal(scores["y_true"], y[rows]) and scores["y_score"].dtype == np.float64,
                    "Full retained candidate score identity, truth and dtype")
            score = scores["y_score"].copy()
        require(np.isfinite(score).all() and ((score >= 0) & (score <= 1)).all()
                and ash(score) == value["raw_score_sha256"], "Finite candidate probability bytes")
        close(ranking_metrics(y[rows], score), [value["ap"], value["auc"]], "Independent candidate AP/AUROC")
        require(value["ap"] == float(average_precision_score(y[rows], score))
                and value["auc"] == float(roc_auc_score(y[rows], score)), "Canonical exact selection scalars, including ties")
        metadata = self.checkpoint(value["model_path"], ds, arm, fit, early, value["params"],
                40000 + 1000 * seed + 10 * fold + inner, rows, score, value["selected_epochs"], receipt)
        require(metadata["model_digest"] == value["model_digest"], "Candidate model digest agrees with record")
        if "fit_metadata" in value:
            require(value["fit_metadata"] == metadata, "Candidate embedded fit metadata")
        return value

    def current_receipt(self, ds, arm, seed, fold):
        _, _, result, done, receipt_path = self.locations(ds, arm, seed, fold)
        if not (result.is_file() and done.is_file() and receipt_path.is_file()):
            return None
        receipt = read(receipt_path)
        if not (receipt.get("passed") is True and receipt.get("verifier_sha256") == sha(SCRIPT)
                and receipt.get("protocol_sha256") == self.psha and receipt.get("source_binding_sha256") == self.source_manifest_sha
                and receipt.get("fold_result_sha256") == sha(result) and receipt.get("fold_done_sha256") == sha(done)):
            return None
        source_hashes(receipt["verified_source_sha256"])
        binding(receipt["verified_artifact_sha256"])
        require(len(receipt["physical_models_rescored"]) == 40, "Complete forty-model fold receipt")
        return receipt

    def fold(self, ds, arm, seed, fold, force=False):
        if not force:
            previous = self.current_receipt(ds, arm, seed, fold)
            if previous is not None:
                return previous
        folder, stem, resultfile, donefile, receiptfile = self.locations(ds, arm, seed, fold)
        receipt = dict(passed=False, status="validating_single_mlp_cuda_outer", dataset=ds, model="mlp", feature_set=arm,
                seed=seed, fold=fold, started_at=now(), verifier_sha256=sha(SCRIPT), protocol_sha256=self.psha,
                source_binding_sha256=self.source_manifest_sha, verified_source_sha256=self.sources.copy(),
                verified_artifact_sha256={}, physical_models_rescored={})
        try:
            done, value = read(donefile), read(resultfile)
            identity = [ds, "mlp", arm, seed, fold]
            require([value[k] for k in ("dataset", "model", "feature_set", "seed", "fold")] == identity
                    and value["protocol_sha256"] == self.psha, "Fold result identity and protocol")
            require([done[k] for k in ("dataset", "model", "feature_set", "seed", "fold")] == identity
                    and value["logical_candidate_evaluations"] == 36 and value["logical_strategy_models"] == 4
                    and value["input_dimension"] == WIDTHS[arm], "Complete DONE identity, dimensionality and forty-model budget")
            require(done.get("passed") is True and done.get("status") == "complete"
                    and done["protocol_sha256"] == self.psha and done["source_binding_sha256"] == self.source_manifest_sha
                    and done["input_binding_sha256"] == self.p["datasets"][ds]["input_binding_sha256"]
                    and path(done["result_path"]) == resultfile and done["result_sha256"] == sha(resultfile), "Last-written complete DONE marker binds exact fold result")
            for obj in (value, done):
                binding(obj["artifact_sha256"], receipt)
            require(value.get("source_binding_sha256") == self.source_manifest_sha
                    and value.get("input_binding_sha256") == self.p["datasets"][ds]["input_binding_sha256"], "Fold source/input bindings")
            search = value["search"]
            searchfile = folder / "search" / stem / "search_result.json"
            require(search == read(searchfile), "Embedded and saved complete search identity")
            binding(search["artifact_sha256"], receipt)
            receipt["verified_artifact_sha256"][str(searchfile)] = sha(searchfile)
            require(len(search["candidate_records"]) == 36 and {(x["candidate_id"], x["inner_fold"]) for x in search["candidate_records"]}
                    == set(itertools.product(range(12), range(3))), "All 12-by-3 inner candidate records retained")
            for item in search["candidate_records"]:
                require(self.candidate(ds, arm, seed, fold, item["candidate_id"], item["inner_fold"], receipt) == item,
                        "Embedded candidate record agrees with independently verified saved candidate")
            summaries = []
            for cid in range(12):
                group = [x for x in search["candidate_records"] if x["candidate_id"] == cid]
                summaries.append(dict(candidate_id=cid, mean_ap=float(np.mean([x["ap"] for x in group])),
                        std_ap=float(np.std([x["ap"] for x in group], ddof=1)), epochs=int(np.median([x["selected_epochs"] for x in group]))))
            require(search["candidate_summary"] == summaries, "Exact mean AP, sample SD and integer epoch median for all candidates")
            best = min(summaries, key=lambda x: (-x["mean_ap"], x["candidate_id"]))
            require(search["selected_candidate_id"] == best["candidate_id"] and search["selected_epochs"] == best["epochs"]
                    and search["anchor_epochs"] == summaries[0]["epochs"] and search["selected_mean_ap"] == best["mean_ap"], "Highest mean inner AP, lowest-ID exact tie, three-epoch median")
            arrays = self.split(ds, seed, fold)
            _, y, ids, _ = self.data(ds, arm)
            train, test, development, calrows = (arrays[k] for k in ("outer_train", "outer_test", "development", "threshold_validation"))
            require(len(value["strategies"]) == 2 and {x["strategy"] for x in value["strategies"]} == set(STRATEGIES), "Both anchor and tuned strategies completed")
            for stage in value["strategies"]:
                partialfile = folder / "folds" / f"{stem}_{stage['strategy']}_partial.json"
                require(read(partialfile) == stage, "Saved stage completion record agrees with embedded result")
                cid = 0 if stage["strategy"] == "same_protocol_anchor" else best["candidate_id"]
                epochs, params = summaries[cid]["epochs"], self.c[cid]["params"]
                require(stage["protocol_sha256"] == self.psha and stage["candidate_id"] == cid and stage["selected_epochs"] == epochs
                        and stage["params"] == params and stage["model_random_state"] == 100 * seed + fold, "Selected stage parameters, epochs and seed")
                require(all(stage[k] == ash(rows) for k, rows in (("fit_rows_sha256", train), ("calibration_fit_rows_sha256", development),
                        ("threshold_rows_sha256", calrows), ("outer_test_rows_sha256", test))), "Stage D/T/outer train/test scope hashes")
                binding(stage["artifact_sha256"], receipt)
                calibration, prediction = pd.read_parquet(path(stage["calibration_path"])), pd.read_parquet(path(stage["prediction_path"]))
                for table, rows in ((calibration, calrows), (prediction, test)):
                    require(np.array_equal(table.row_index, rows) and np.array_equal(table.account_id, ids.iloc[rows].account_id)
                            and np.array_equal(table.y_true, y[rows]) and table.y_score.dtype == np.float64
                            and table.y_score.between(0, 1).all(), "Complete prediction account order, labels and probabilities")
                cutoff, calf1 = threshold(y[calrows], calibration.y_score.to_numpy())
                require(stage["threshold"] == cutoff, "Independent T maximum F1 threshold, smallest threshold on exact ties")
                close(stage["threshold_validation_f1"], calf1, "Independent calibration F1")
                close(stage["threshold_validation_ap"], ranking_metrics(y[calrows], calibration.y_score.to_numpy())[0], "Independent calibration AP")
                for role, rows, score_rows, table, modelname in (("calibration", development, calrows, calibration, "calibration_model_path"),
                        ("final", train, test, prediction, "model_path")):
                    metadata = self.checkpoint(stage[modelname], ds, arm, rows, None, params, 100 * seed + fold,
                            score_rows, table.y_score.to_numpy(), epochs, receipt)
                    require(metadata == stage[role + "_metadata"], "Stage embedded model metadata")
                require(ash(prediction.y_score.to_numpy()) == stage["raw_score_sha256"], "Final prediction raw probability bytes")
                require(len(stage["metrics"]) == 2 and {x["threshold_policy"] for x in stage["metrics"]} == set(POLICIES), "Both threshold policies retained")
                for policy, column, cut in (("fixed_0.5", "y_pred_fixed", .5), ("inner_max_f1", "y_pred_inner", cutoff)):
                    predicted = (prediction.y_score.to_numpy() >= cut).astype(np.int8)
                    require(np.array_equal(prediction[column], predicted), "Saved binary decisions use registered score threshold")
                    actual = next(x for x in stage["metrics"] if x["threshold_policy"] == policy)
                    expected = metrics(y[test], prediction.y_score.to_numpy(), predicted)
                    require([actual[k] for k in ("dataset", "model", "feature_set", "strategy", "seed", "fold", "selected_candidate_id", "selected_epochs")]
                            == [ds, "mlp", arm, stage["strategy"], seed, fold, cid, epochs], "Metric row strategy and selected configuration")
                    require(all(k in actual for k in MAIN_METRICS + ("specificity", "tn", "fp", "fn", "tp", "n_test", "n_positive")), "Seven metrics and complete confusion counts retained")
                    close([actual[k] for k in expected if k in actual], [expected[k] for k in expected if k in actual], "Independent seven metrics and confusion counts")
                    require(actual["threshold"] == cut, "Metric threshold equals registered cutoff")
            require(len(receipt["physical_models_rescored"]) == 40, "All 36 search and four strategy physical models rescored")
            receipt["verified_artifact_sha256"].update({str(resultfile): sha(resultfile), str(donefile): sha(donefile)})
            receipt.update(passed=True, status="passed_single_cuda_outer_not_complete_setting", completed_at=now(),
                    fold_result_path=str(resultfile), fold_result_sha256=sha(resultfile), fold_done_path=str(donefile),
                    fold_done_sha256=sha(donefile), done_sha256=sha(donefile), candidate_models_and_scores_verified=36, strategy_models_verified=4,
                    score_byte_equality=True, validator_peak_rss_bytes=peak_rss_bytes(),
                    scope="One outer fold only; a setting requires 25 verified folds. No trainer import and no fitting.")
            write(receiptfile, receipt)
            return receipt
        except Exception:
            receipt.update(status="failed_not_accepted", failed_at=now(), error=traceback.format_exc())
            write(receiptfile.with_suffix(".failed.json"), receipt)
            raise

    def overview(self):
        settings = []
        for ds, arm in itertools.product(DATASETS, ARMS):
            trained = verified = 0
            errors = []
            for seed, fold in itertools.product(range(5), repeat=2):
                _, _, result, done, _ = self.locations(ds, arm, seed, fold)
                trained += int(result.is_file() and done.is_file())
                try:
                    verified += int(self.current_receipt(ds, arm, seed, fold) is not None)
                except Exception as exc:
                    errors.append(dict(seed=seed, fold=fold, error=str(exc)))
            settings.append(dict(dataset=ds, model="mlp", feature_set=arm, training_done_folds=trained,
                    independently_verified_folds=verified, expected_folds=25,
                    status="complete_independently_verified" if verified == 25 else "partial_not_complete", invalid_receipts=errors))
        total = sum(x["independently_verified_folds"] for x in settings)
        result = dict(status="complete" if total == 200 else "partial", updated_at=now(), protocol_sha256=self.psha,
                verifier_sha256=sha(SCRIPT), planned_settings=8, planned_outer_folds=200,
                training_done_outer_folds=sum(x["training_done_folds"] for x in settings), independently_verified_outer_folds=total,
                complete_independently_verified_settings=sum(x["independently_verified_folds"] == 25 for x in settings),
                settings=settings, no_partial_means_reported=True)
        write(self.dest / "overview.json", result)
        return result

    def scan(self, ds=None, arm=None, max_folds=1, force=False):
        checked = []
        count = 0
        for dataset, feature in itertools.product((ds,) if ds else DATASETS, (arm,) if arm else ARMS):
            for seed, fold in itertools.product(range(5), repeat=2):
                _, _, result, done, receiptfile = self.locations(dataset, feature, seed, fold)
                if not (result.is_file() and done.is_file()):
                    continue
                previous = None if force else self.current_receipt(dataset, feature, seed, fold)
                if previous is not None:
                    continue
                if max_folds is not None and count >= max_folds:
                    continue
                receipt = self.fold(dataset, feature, seed, fold, force=force)
                checked.append(str(receiptfile))
                count += 1
                print(json.dumps(dict(fold_passed=[dataset, feature, seed, fold], models_rescored=len(receipt["physical_models_rescored"]))), flush=True)
        write(self.dest / "last_scan.json", dict(checked_at=now(), verifier_sha256=sha(SCRIPT), new_folds_verified=count, receipt_paths=checked))
        return self.overview()

    def predictions(self, ds, arm, seed, fold, strategy):
        require(self.current_receipt(ds, arm, seed, fold) is not None, "Aggregation admits only a current strong-SHA verified fold")
        _, _, result, done, receiptfile = self.locations(ds, arm, seed, fold)
        value = read(result)
        stage = next(x for x in value["strategies"] if x["strategy"] == strategy)
        filename = path(stage["prediction_path"])
        table = pd.read_parquet(filename)
        _, y, ids, _ = self.data(ds, arm)
        rows = self.split(ds, seed, fold)["outer_test"]
        require(np.array_equal(table.row_index, rows) and np.array_equal(table.account_id, ids.iloc[rows].account_id)
                and np.array_equal(table.y_true, y[rows]), "Aggregation preserves original held-out rows, accounts and truth")
        values = []
        for policy, column in (("fixed_0.5", "y_pred_fixed"), ("inner_max_f1", "y_pred_inner")):
            cutoff = .5 if policy == "fixed_0.5" else stage["threshold"]
            values.append(dict(dataset=ds, model="mlp", feature_set=arm, strategy=strategy, seed=seed, fold=fold,
                    threshold_policy=policy, threshold=cutoff, selected_candidate_id=stage["candidate_id"],
                    selected_epochs=stage["selected_epochs"], **metrics(y[rows], table.y_score.to_numpy(), table[column].to_numpy())))
        return table, values, {str(x): sha(x) for x in (result, done, receiptfile, filename)}

    def aggregate(self, allow_partial=False):
        status = self.overview()
        complete = {(x["dataset"], x["feature_set"]) for x in status["settings"] if x["status"] == "complete_independently_verified"}
        require(allow_partial or len(complete) == 8, "Formal aggregation requires eight complete 25-fold settings; use --allow-partial for explicitly incomplete monitoring tables")
        foldrows, seedrows, pairrows, pairseeds, skipped, bindings = [], [], [], [], [], {}
        for ds, arm in sorted(complete):
            for strategy in STRATEGIES:
                for seed in range(5):
                    pieces = []
                    for fold in range(5):
                        table, values, hashes = self.predictions(ds, arm, seed, fold, strategy)
                        pieces.append(table)
                        foldrows.extend(values)
                        bindings.update(hashes)
                    oof = pd.concat(pieces).sort_values("row_index")
                    _, y, ids, _ = self.data(ds, arm)
                    require(len(oof) == len(y) and oof.row_index.is_unique and np.array_equal(oof.row_index, np.arange(len(y)))
                            and np.array_equal(oof.account_id, ids.account_id) and np.array_equal(oof.y_true, y), "Each seed OOF covers the original account population exactly once")
                    for policy, column in (("fixed_0.5", "y_pred_fixed"), ("inner_max_f1", "y_pred_inner")):
                        seedrows.append(dict(dataset=ds, model="mlp", feature_set=arm, strategy=strategy, seed=seed,
                                threshold_policy=policy, **metrics(y, oof.y_score.to_numpy(), oof[column].to_numpy())))
                    del oof, pieces
        foldkeys = ["dataset", "model", "feature_set", "strategy", "threshold_policy", "seed", "fold"]
        seedkeys = [x for x in foldkeys if x != "fold"]
        frame = pd.DataFrame(foldrows, columns=foldkeys + ["threshold", "selected_candidate_id", "selected_epochs"] + list(METRICS))
        seedframe = pd.DataFrame(seedrows, columns=seedkeys + list(METRICS))
        for ds, (left, right) in itertools.product(DATASETS, CONTRASTS):
            if (ds, left) not in complete or (ds, right) not in complete:
                skipped.append(dict(dataset=ds, lhs_arm=left, rhs_arm=right, reason="Both complete 25-fold settings required"))
                continue
            for strategy, policy in itertools.product(STRATEGIES, POLICIES):
                for source, keys, destination in ((frame, ["seed", "fold"], pairrows), (seedframe, ["seed"], pairseeds)):
                    subset = source[(source.dataset == ds) & (source.strategy == strategy) & (source.threshold_policy == policy)]
                    lhs = subset[subset.feature_set == left].set_index(keys).sort_index()
                    rhs = subset[subset.feature_set == right].set_index(keys).sort_index()
                    require(lhs.index.equals(rhs.index) and len(lhs) == (25 if "fold" in keys else 5), "All paired folds/seeds share matching original split keys")
                    for index, row in lhs.iterrows():
                        index = index if isinstance(index, tuple) else (index,)
                        other = rhs.loc[index if len(keys) > 1 else index[0]]
                        destination.append(dict(dataset=ds, model="mlp", lhs_arm=left, rhs_arm=right, strategy=strategy,
                                threshold_policy=policy, **dict(zip(keys, map(int, index))),
                                **{"delta_" + metric: float(row[metric] - other[metric]) for metric in METRICS}))
        folder = self.dest / "complete_settings"
        folder.mkdir(parents=True, exist_ok=True)
        groupkeys = ["dataset", "model", "feature_set", "strategy", "threshold_policy"]
        if len(frame):
            summary = frame.groupby(groupkeys)[list(MAIN_METRICS)].agg(["mean", "std"])
            summary.columns = ["_".join(x) for x in summary.columns]
            summary = summary.reset_index()
        else:
            summary = pd.DataFrame(columns=groupkeys + [f"{m}_{s}" for m in MAIN_METRICS for s in ("mean", "std")])
        pairkeys = ["dataset", "model", "lhs_arm", "rhs_arm", "strategy", "threshold_policy"]
        pairframe = pd.DataFrame(pairrows, columns=pairkeys + ["seed", "fold"] + ["delta_" + m for m in METRICS])
        pairseedframe = pd.DataFrame(pairseeds, columns=pairkeys + ["seed"] + ["delta_" + m for m in METRICS])
        if len(pairframe):
            pairsummary = pairframe.groupby(pairkeys)[["delta_" + m for m in MAIN_METRICS]].agg(["mean", "std"])
            pairsummary.columns = ["_".join(x) for x in pairsummary.columns]
            pairsummary = pairsummary.reset_index()
            for metric in MAIN_METRICS:
                for source, suffix in ((pairframe, "folds"), (pairseedframe, "oof_seeds")):
                    directions = source.groupby(pairkeys)["delta_" + metric].agg(**{
                        f"{metric}_positive_{suffix}": lambda values: int((values > 0).sum()),
                        f"{metric}_negative_{suffix}": lambda values: int((values < 0).sum()),
                        f"{metric}_zero_{suffix}": lambda values: int((values == 0).sum())}).reset_index()
                    pairsummary = pairsummary.merge(directions, on=pairkeys, validate="one_to_one")
        else:
            pairsummary = pd.DataFrame(columns=pairkeys + [f"delta_{m}_{s}" for m in MAIN_METRICS for s in ("mean", "std")])
        outputs = {"verified_fold_metrics.csv": frame, "verified_seed_oof_metrics.csv": seedframe,
                   "verified_summary.csv": summary, "verified_paired_fold_differences.csv": pairframe,
                   "verified_paired_seed_oof_differences.csv": pairseedframe, "verified_paired_summary.csv": pairsummary}
        output_hashes = {}
        for name, table in outputs.items():
            filename = folder / name
            temporary = filename.with_name(filename.name + f".{os.getpid()}.tmp")
            table.to_csv(temporary, index=False)
            os.replace(temporary, filename)
            output_hashes[str(filename)] = sha(filename)
        if len(complete) == 8:
            require((len(frame), len(seedframe), len(pairframe), len(pairseedframe)) == (800, 160, 1200, 240), "All eight settings and all six representation contrasts, both strategies and thresholds")
        result = dict(passed=True, status="complete" if len(complete) == 8 else "partial_complete_settings_only", created_at=now(),
                verifier_sha256=sha(SCRIPT), protocol_sha256=self.psha, source_binding_sha256=self.source_manifest_sha,
                complete_settings=len(complete), expected_settings=8, fold_metric_rows=len(frame), seed_oof_rows=len(seedframe),
                paired_fold_rows=len(pairframe), paired_seed_oof_rows=len(pairseedframe),
                skipped_incomplete_comparisons=skipped, source_prediction_sha256=bindings, output_sha256=output_hashes,
                reporting="Mean paired increments; sample SD ddof=1; descriptive fold/seed direction counts. Repeated folds and OOF seeds are dependent. No independent-confirmation or significance claim.",
                scope="Only complete 25-fold groups under one CUDA protocol; historical MPS BASE16 is not mixed with new CUDA results.")
        write(folder / "validation.json", result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", "--root", dest="out", type=Path, required=True)
    parser.add_argument("action", choices=("job", "fold", "candidate", "scan", "overview", "aggregate"))
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--arm", "--feature-set", dest="arm", choices=ARMS)
    parser.add_argument("--seed", type=int, choices=range(5))
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--candidate", type=int, choices=range(12))
    parser.add_argument("--inner-fold", type=int, choices=range(3))
    parser.add_argument("--max-folds", type=int, default=1, help="New folds per scan; zero means no limit")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-partial", action="store_true", help="Aggregate completed settings only; incomplete groups remain excluded")
    args = parser.parse_args()
    validator = Validator(args.out)
    if args.action in ("job", "fold", "candidate"):
        require(all(v is not None for v in (args.dataset, args.arm, args.seed, args.fold)), "Dataset, arm, seed and fold required")
        if args.action in ("job", "fold"):
            result = validator.fold(args.dataset, args.arm, args.seed, args.fold, args.force)
            validator.overview()
        else:
            require(args.candidate is not None and args.inner_fold is not None, "Candidate ID and inner fold required")
            result = dict(passed=False, status="validating_single_candidate", dataset=args.dataset, model="mlp", feature_set=args.arm,
                    seed=args.seed, fold=args.fold, candidate_id=args.candidate, inner_fold=args.inner_fold, started_at=now(),
                    verifier_sha256=sha(SCRIPT), protocol_sha256=validator.psha, verified_artifact_sha256={}, physical_models_rescored={})
            validator.candidate(args.dataset, args.arm, args.seed, args.fold, args.candidate, args.inner_fold, result)
            result.update(passed=True, status="passed_single_candidate_not_outer_fold", completed_at=now())
            write(validator.dest / "candidate_receipts" / args.dataset / args.arm /
                    f"seed{args.seed}_fold{args.fold}_candidate{args.candidate}_inner{args.inner_fold}.json", result)
    elif args.action == "scan":
        require(args.max_folds >= 0, "Nonnegative max-folds required")
        result = validator.scan(args.dataset, args.arm, args.max_folds or None, args.force)
    elif args.action == "aggregate":
        result = validator.aggregate(args.allow_partial)
    else:
        result = validator.overview()
    print(json.dumps({k: result[k] for k in ("passed", "status", "dataset", "feature_set", "seed", "fold", "training_done_outer_folds",
            "independently_verified_outer_folds", "complete_independently_verified_settings", "complete_settings") if k in result}), flush=True)


if __name__ == "__main__":
    main()
