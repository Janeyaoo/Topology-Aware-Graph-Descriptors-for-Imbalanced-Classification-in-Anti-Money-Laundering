"""Official Snap ML GFP streaming account adaptation; no labels or fitting.

Production requires a passed native small-graph receipt from the exact wheel,
configuration and interpreter. Six statistics per engineered edge column are
aggregated out/in with disk-backed float64 sums/maxima, never a full edge-feature
matrix. Raw IDs/time/amounts are excluded from the final account features.
"""
from __future__ import annotations

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import resource
import sys
import time
import traceback

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
ALT = ROOT / "Altman_experiments"
SOURCE = ALT / "results/account_s10_increment_20261008"
DEFAULT_OUT = ALT / "results/account_eswa_supplement_20261013"
AMENDMENT = DEFAULT_OUT / "GFP_PROTOCOL_AMENDMENT_20261013.json"
WINDOW = 604800
RAW_COLS = ["timestamp", "from_id", "to_id", "amount_paid", "amount_recv"]
ROLES = ("out", "in")
AGGREGATES = ("sum", "mean", "max")
PATTERNS = (("fan-in", "fan", [2, 4, 8, 16, 32, 64]),
            ("fan-out", "fan", [2, 4, 8, 16, 32, 64]),
            ("degree-in", "degree", [2, 4, 8, 16, 32, 64]),
            ("degree-out", "degree", [2, 4, 8, 16, 32, 64]),
            ("scatter-gather", "scatter-gather", [2, 4, 8, 16, 32, 64]),
            ("temp-cycle", "temp-cycle", [2, 3, 4, 5, 6]),
            ("lc-cycle", "lc-cycle", [2, 3, 4, 5, 6]))
STAT_NAMES = ["fan", "degree", "ratio", "average", "sum", "minimum", "maximum",
              "median", "var", "skew", "kurtosis"]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (np.ndarray, list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def jsave(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def peak_rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def intended_parameters():
    params = {"num_threads": 1, "time_window": WINDOW, "max_no_edges": -1,
              "vertex_stats": True, "vertex_stats_tw": WINDOW,
              "vertex_stats_cols": [4, 5], "vertex_stats_feats": list(range(11)),
              "lc-cycle_len": 6}
    for pattern in ("fan", "degree", "scatter-gather", "temp-cycle", "lc-cycle"):
        params[pattern] = True
        params[pattern + "_tw"] = WINDOW
        params[pattern + "_bins"] = [2, 3, 4, 5, 6] if "cycle" in pattern else [2, 4, 8, 16, 32, 64]
    return params


def create_native():
    # Lazy import prevents architecture-specific native loading during code review.
    from snapml import GraphFeaturePreprocessor
    native = GraphFeaturePreprocessor()
    defaults = json_safe(native.get_params())
    requested = intended_parameters()
    assert set(requested) <= set(defaults), f"Unsupported required GFP keys: {set(requested) - set(defaults)}"
    native.set_params(requested)
    actual = json_safe(native.get_params())
    for key, value in requested.items():
        assert actual[key] == value, f"Native parameter mismatch: {key}: {actual[key]} != {value}"
    return native, defaults, actual


def native_environment(wheel):
    import snapml
    wheel = Path(wheel).resolve()
    assert wheel.is_file(), "Exact native wheel is required for reproducible package binding."
    distribution = importlib.metadata.distribution("snapml")
    binaries = {}
    for name in distribution.files or []:
        if str(name).endswith((".so", ".dylib", ".dll")):
            path = Path(distribution.locate_file(name)).resolve()
            binaries[str(path)] = sha(path)
    library_paths = {name: os.environ.get(name, "") for name in
                     ("DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH", "LD_LIBRARY_PATH")}
    extra_libraries = {}
    for value in library_paths.values():
        for directory in value.split(os.pathsep):
            if not directory:
                continue
            for name in ("libomp.dylib", "libiomp5.dylib", "libomp.so", "libgomp.so.1"):
                path = Path(directory) / name
                if path.is_file():
                    extra_libraries[str(path.resolve())] = sha(path)
    return {"python_executable": str(Path(sys.executable).resolve()),
            "python_version": platform.python_version(), "platform": platform.platform(),
            "machine": platform.machine(), "snapml_version": importlib.metadata.version("snapml"),
            "snapml_module_path": str(Path(snapml.__file__).resolve()),
            "snapml_module_sha256": sha(Path(snapml.__file__)),
            "wheel_path": str(wheel), "wheel_sha256": sha(wheel),
            "installed_native_binary_sha256": binaries,
            "dynamic_library_paths": library_paths,
            "extra_runtime_library_sha256": extra_libraries,
            "dependencies": {name: importlib.metadata.version(name)
                             for name in ("numpy", "pandas", "pyarrow", "scikit-learn")}}


def proposed_schema(actual):
    """Official documented output order; accepted only after native QA matches q."""
    schema = []
    for name, key, bins in PATTERNS:
        if not actual[key]:
            continue
        bins = list(actual[key + "_bins"])
        for index, lower in enumerate(bins):
            schema.append({"id": len(schema), "name": f"pattern_{name}_bin{index}",
                           "type": "pattern_histogram", "pattern": name,
                           "lower_inclusive": lower,
                           "upper_exclusive": bins[index + 1] if index + 1 < len(bins) else None})
    if actual["vertex_stats"]:
        selected = list(actual["vertex_stats_feats"])
        for endpoint in ("source", "target"):
            for role in ROLES:
                for key in (0, 1, 2):
                    if key in selected:
                        schema.append({"id": len(schema), "name": f"vertex_{endpoint}_{role}_{STAT_NAMES[key]}",
                                       "type": "vertex_statistic", "endpoint": endpoint,
                                       "edge_role": role, "statistic": STAT_NAMES[key], "raw_column": None})
                for column in actual["vertex_stats_cols"]:
                    for key in range(3, 11):
                        if key in selected:
                            schema.append({"id": len(schema),
                                           "name": f"vertex_{endpoint}_{role}_{STAT_NAMES[key]}_raw{column}",
                                           "type": "vertex_statistic", "endpoint": endpoint,
                                           "edge_role": role, "statistic": STAT_NAMES[key], "raw_column": column})
    return schema


def canonical_engineered(native, raw, q):
    raw = np.ascontiguousarray(raw, dtype=np.float64)
    assert len(raw) and len(np.unique(raw[:, 3])) == 1, "Native call must contain one complete timestamp group."
    assert len(np.unique(raw[:, 0])) == len(raw), "Every source transaction ID is unique."
    ordering = np.argsort(raw[:, 0], kind="stable")
    canonical = np.ascontiguousarray(raw[ordering])
    output = np.asarray(native.transform(canonical))
    assert output.shape == (len(raw), 6 + q), f"Unexpected native schema: {output.shape}"
    assert np.array_equal(output[:, :6], canonical), "Native transform changed indexing or raw columns."
    inverse = np.argsort(ordering)
    engineered = np.ascontiguousarray(output[inverse, 6:], dtype=np.float64)
    undefined = ~np.isfinite(engineered)
    counts = undefined.sum(axis=0).astype(np.int64)
    engineered[undefined] = 0.0
    return engineered, counts


def source_paths(ds):
    stem = "HI-Small" if ds == "hi_small" else "LI-Small"
    return [ALT / "data" / ds / f"{ds}_trans.parquet",
            ALT / "data" / ds / f"{stem}_Trans.csv",
            ALT / "data" / ds / f"{stem}_accounts.csv",
            SOURCE / ds / "features.parquet", ALT / "code/ibm_pipeline.py",
            ALT / "results/eswa_supplement_design_20261013/protocol_design.json",
            ALT / "results/eswa_supplement_design_20261013/source_manifest.json", AMENDMENT,
            AMENDMENT.with_suffix(".md")]


def prepare_raw(folder, ds, accounts):
    """Batch-only normalized reads; full raw input lives on disk, not in pandas."""
    parquet = ALT / "data" / ds / f"{ds}_trans.parquet"
    pf = pq.ParquetFile(parquet)
    n = pf.metadata.num_rows
    assert n < 2**53 and len(accounts) < 2**53
    native_path = folder / "gfp_scratch/native_raw.npy"
    raw = np.lib.format.open_memmap(native_path, mode="w+", dtype=np.float64, shape=(n, 6))
    index = pd.Index(accounts)
    assert index.is_unique
    row = 0
    minimum = None
    maximum = None
    selfloops = 0
    for batch in pf.iter_batches(batch_size=32768, columns=RAW_COLS):
        frame = batch.to_pandas()
        assert not frame.isna().any().any()
        count = len(frame)
        times = pd.to_datetime(frame.timestamp, format="%Y/%m/%d %H:%M", errors="raise")
        seconds = times.to_numpy(dtype="datetime64[s]").astype(np.int64)
        src = index.get_indexer(frame.from_id)
        dst = index.get_indexer(frame.to_id)
        assert np.all(src >= 0) and np.all(dst >= 0)
        amounts = frame[["amount_paid", "amount_recv"]].to_numpy(np.float64)
        assert np.isfinite(amounts).all()
        raw[row:row + count, 0] = np.arange(row, row + count, dtype=np.int64)
        raw[row:row + count, 1] = src
        raw[row:row + count, 2] = dst
        raw[row:row + count, 3] = seconds
        raw[row:row + count, 4:] = amounts
        minimum = int(seconds.min()) if minimum is None else min(minimum, int(seconds.min()))
        maximum = int(seconds.max()) if maximum is None else max(maximum, int(seconds.max()))
        selfloops += int((src == dst).sum())
        row += count
    assert row == n
    raw[:, 3] -= minimum
    assert np.isfinite(raw).all()
    raw.flush()
    # Stable sort preserves source row ID within equal timestamps.
    order = np.argsort(raw[:, 3], kind="stable")
    np.save(folder / "gfp_scratch/time_order.npy", order)
    del raw, order
    gc.collect()
    return {"n_transactions": n, "n_self_transfers": selfloops,
            "timestamp_format": "%Y/%m/%d %H:%M", "timestamp_unit": "seconds",
            "timestamp_origin": str(np.datetime64(minimum, "s")),
            "timestamp_naive_timezone": "no source timezone; only differences used; constant epoch shift has no pattern effect",
            "max_relative_seconds": maximum - minimum,
            "edge_id": "zero-based normalized/source transaction row; every row retained including parallel and self edges",
            "account_index": "exact original account-feature row order",
            "raw_native_sha256": sha(native_path),
            "time_order_sha256": sha(folder / "gfp_scratch/time_order.npy")}


class AccountAccumulator:
    def __init__(self, folder, n, q, create=False):
        self.folder, self.n, self.q = folder, n, q
        self.sums = self.maxima = self.counts = None
        if create:
            sums = np.lib.format.open_memmap(folder / "sums.npy", mode="w+", dtype=np.float64, shape=(2, n, q))
            maxima = np.lib.format.open_memmap(folder / "maxima.npy", mode="w+", dtype=np.float64, shape=(2, n, q))
            counts = np.lib.format.open_memmap(folder / "counts.npy", mode="w+", dtype=np.int64, shape=(2, n))
            # Newly extended files are zero-filled: sums/counts need no full-RAM
            # initialization. Initialize negative-infinite maxima in bounded tiles.
            for value in (sums, maxima, counts):
                value.flush()
                value._mmap.close()
            del sums, maxima, counts
            for role in range(2):
                for start in range(0, n, 8192):
                    value = np.load(folder / "maxima.npy", mmap_mode="r+")
                    value[role, start:min(start + 8192, n)] = -np.inf
                    value.flush()
                    value._mmap.close()
                    del value
        self.open()

    def open(self):
        self.sums = np.load(self.folder / "sums.npy", mmap_mode="r+")
        self.maxima = np.load(self.folder / "maxima.npy", mmap_mode="r+")
        self.counts = np.load(self.folder / "counts.npy", mmap_mode="r+")

    def add(self, raw, features):
        for role, source_column in enumerate((1, 2)):
            ids = raw[:, source_column].astype(np.int64)
            np.add.at(self.counts[role], ids, 1)
            np.add.at(self.sums[role], ids, features)
            np.maximum.at(self.maxima[role], ids, features)

    def release_pages(self):
        # Flush/close/reopen bounds mapped resident pages without discarding sums.
        for value in (self.sums, self.maxima, self.counts):
            value.flush()
            value._mmap.close()
        self.sums = self.maxima = self.counts = None
        gc.collect()
        self.open()

    def export(self, destination, accounts, columns, batch_size=8192):
        self.release_pages()
        temporary = destination.with_suffix(".tmp.parquet")
        writer = None
        try:
            for start in range(0, self.n, batch_size):
                stop = min(start + batch_size, self.n)
                result = {"account_id": pa.array(accounts[start:stop])}
                for feature in range(self.q):
                    for role_index, role in enumerate(ROLES):
                        count = self.counts[role_index, start:stop]
                        total = np.array(self.sums[role_index, start:stop, feature])
                        mean = np.divide(total, count, out=np.zeros(len(total)), where=count > 0)
                        maximum = np.array(self.maxima[role_index, start:stop, feature])
                        maximum[count == 0] = 0.0
                        for aggregation, value in zip(AGGREGATES, (total, mean, maximum)):
                            assert np.isfinite(value).all(), (feature, role, aggregation)
                            assert np.isfinite(value.astype(np.float32)).all(), "Feature exceeds training float32 range."
                            result[f"gfp_{feature:03d}_{role}_{aggregation}"] = pa.array(value)
                table = pa.table(result)
                assert table.column_names == ["account_id", *columns]
                if writer is None:
                    writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
                writer.write_table(table)
                del table, result
                self.release_pages()
        finally:
            if writer is not None:
                writer.close()
        temporary.replace(destination)


def build(out, ds, evidence, wheel, memory_budget_gb=3.0, aggregation_batch_size=2048):
    started = time.monotonic()
    folder = Path(out) / "inputs" / ds
    folder.mkdir(parents=True, exist_ok=True)
    assert not (folder / "gfp_account.parquet").exists(), "Immutable complete GFP input already exists."
    scratch = folder / "gfp_scratch"
    assert not scratch.exists(), "Existing partial scratch requires reviewed restart; never silently double-aggregate."
    receipt = json.loads(Path(evidence).read_text())
    assert receipt.get("passed") is True and receipt["status"] == "passed"
    assert receipt["builder_sha256"] == sha(Path(__file__))
    assert receipt["protocol_amendment_sha256"] == sha(AMENDMENT)
    environment = native_environment(wheel)
    assert environment == receipt["native_environment"], "Native environment differs from accepted QA."
    native, defaults, actual = create_native()
    assert actual == receipt["get_params_actual"]
    q = receipt["native_engineered_dimension"]
    schema = proposed_schema(actual)
    assert len(schema) == q and schema == receipt["engineered_schema"]
    source_hashes = {relative(path): sha(path) for path in source_paths(ds)}
    frozen_sources = json.loads((ALT / "results/eswa_supplement_design_20261013/source_manifest.json").read_text())["source_sha256"]
    for key, expected in source_hashes.items():
        if key in frozen_sources:
            assert expected == frozen_sources[key], f"Frozen design source changed: {key}"
    accounts = pd.read_parquet(SOURCE / ds / "features.parquet", columns=["account_id"]).account_id.to_numpy()
    assert pd.Index(accounts).is_unique
    scratch.mkdir()
    state = {"status": "building", "dataset": ds, "pid": os.getpid(),
             "method": "Official Snap ML GraphFeaturePreprocessor, streaming account adaptation",
             "official_api_source": "https://snapml.readthedocs.io/en/latest/graph_preprocessor.html",
             "original_method_source": "https://doi.org/10.1145/3677052.3698674",
             "n_accounts": len(accounts), "native_engineered_dimension": q,
             "planned_aggregation_scratch_bytes": int(4 * len(accounts) * q * 8 + 2 * len(accounts) * 8),
             "memory_budget_gb": memory_budget_gb, "native_environment": environment,
             "get_params_actual": actual, "source_sha256": source_hashes,
             "native_acceptance_path": str(Path(evidence).resolve()),
             "native_acceptance_sha256": sha(Path(evidence)),
             "builder_sha256": sha(Path(__file__)), "created_utc": datetime.now(timezone.utc).isoformat()}
    jsave(folder / "gfp_run_status.json", state)
    jsave(folder / "gfp_engineered_schema.json", {"native_dimension": q, "columns": schema,
                                                   "raw_columns_excluded": ["edge_id", "source_id", "target_id", "timestamp", "amount_paid", "amount_received"],
                                                   "schema_source": "official output order, native count and statistic-position small-graph QA"})
    raw_metadata = prepare_raw(folder, ds, accounts)
    raw = np.load(scratch / "native_raw.npy", mmap_mode="r")
    order = np.load(scratch / "time_order.npy", mmap_mode="r")
    accumulator = AccountAccumulator(scratch, len(accounts), q, create=True)
    undefined_counts = np.zeros(q, dtype=np.int64)
    sample_account_rows = np.sort(np.random.default_rng(20261013).choice(
        len(accounts), size=min(64, len(accounts)), replace=False))
    sample_path = folder / "gfp_audit_edge_samples.parquet"
    sample_temporary = sample_path.with_suffix(".tmp.parquet")
    sample_writer = None
    n_sample_edges = 0
    group_count = processed = max_group = 0
    minimum_ts = last_ts = None
    while processed < len(order):
        begin = processed
        timestamp = raw[order[begin], 3]
        stop = begin + 1
        while stop < len(order) and raw[order[stop], 3] == timestamp:
            stop += 1
        # Exactly one native transform for a complete time group. Native repeat-ID
        # queries are NOT idempotent for vertex statistics in official 1.17.2.
        group_rows = order[begin:stop]
        group = np.ascontiguousarray(raw[group_rows], dtype=np.float64)
        group_features, undefined = canonical_engineered(native, group, q)
        undefined_counts += undefined
        for a in range(0, len(group_rows), aggregation_batch_size):
            query = group[a:a + aggregation_batch_size]
            features = group_features[a:a + aggregation_batch_size]
            accumulator.add(query, features)
            # Retain only incidents of 64 preselected, label-blind accounts for
            # independent aggregation acceptance; no full native edge matrix.
            selected = np.isin(query[:, 1].astype(np.int64), sample_account_rows) | np.isin(
                query[:, 2].astype(np.int64), sample_account_rows)
            if selected.any():
                sampled = {name: pa.array(query[selected, j]) for j, name in enumerate(
                    ("edge_id", "source_account_row", "target_account_row", "timestamp_seconds", "amount_paid", "amount_received"))}
                sampled.update({f"native_{j:03d}": pa.array(features[selected, j]) for j in range(q)})
                table = pa.table(sampled)
                if sample_writer is None:
                    sample_writer = pq.ParquetWriter(sample_temporary, table.schema, compression="zstd")
                sample_writer.write_table(table)
                n_sample_edges += int(selected.sum())
        processed = stop
        group_count += 1
        max_group = max(max_group, stop - begin)
        minimum_ts = int(timestamp) if minimum_ts is None else minimum_ts
        last_ts = int(timestamp)
        if group_count % 32 == 0:
            accumulator.release_pages()
        if group_count % 100 == 0 or processed == len(order):
            progress = {**state, "processed_transactions": processed, "timestamp_groups": group_count,
                        "last_relative_timestamp_seconds": last_ts, "max_timestamp_group_rows": max_group,
                        "runtime_seconds": time.monotonic() - started,
                        "peak_rss_bytes": peak_rss_bytes(), "undefined_values_total": int(undefined_counts.sum())}
            jsave(folder / "gfp_run_status.json", progress)
            print("GFP_PROGRESS", ds, json.dumps({key: progress[key] for key in
                  ("processed_transactions", "timestamp_groups", "runtime_seconds", "peak_rss_bytes")}), flush=True)
            if peak_rss_bytes() > memory_budget_gb * (1024**3):
                jsave(folder / "gfp_run_status.json", {**progress, "status": "stopped_resource_budget",
                                                       "pid": None, "note": "No classifier result; uncommitted partial input. Restart requires a reviewed resource budget."})
                raise MemoryError("Native extraction exceeded registered peak-RSS budget; partial input not admitted.")
    assert int(accumulator.counts[0].sum()) == int(accumulator.counts[1].sum()) == len(order)
    assert sample_writer is not None, "Preselected accounts unexpectedly had no incident transactions."
    sample_writer.close()
    sample_temporary.replace(sample_path)
    columns = [f"gfp_{feature:03d}_{role}_{aggregation}" for feature in range(q)
               for role in ROLES for aggregation in AGGREGATES]
    destination = folder / "gfp_account.parquet"
    accumulator.export(destination, accounts, columns)
    counts = pd.DataFrame({"account_id": accounts,
                           "gfp_out_transaction_count": np.array(accumulator.counts[0]),
                           "gfp_in_transaction_count": np.array(accumulator.counts[1])})
    counts.to_parquet(folder / "gfp_incident_counts.parquet", index=False, compression="zstd")
    for key, expected in source_hashes.items():
        assert sha(ROOT / key) == expected, f"Scientific source changed: {key}"
    manifest = {**state, "status": "complete", "pid": None,
                "raw_input": raw_metadata, "n_transactions": raw_metadata["n_transactions"],
                "n_timestamp_groups": group_count, "max_timestamp_group_rows": max_group,
                "first_last_relative_seconds": [minimum_ts, last_ts],
                "feature_columns": columns, "gfp_account_dimension": len(columns),
                "full_model_dimension": 16 + len(columns), "storage_dtype": "float64",
                "training_conversion": "float32, checked finite before export",
                "undefined_native_values_by_schema_id": undefined_counts.tolist(),
                "undefined_native_values_total": int(undefined_counts.sum()),
                "native_query_batch_policy": "one complete timestamp group once, canonical immutable source txID order",
                "aggregation_batch_size": aggregation_batch_size, "aggregation_map_release_every_timestamp_groups": 32,
                "aggregation": "per native engineered column: out sum/mean/max then in sum/mean/max; self-transfers in both roles; empty role=0",
                "meaning": "transaction participation summaries, not unique account subgraph counts",
                "data_view": "official temporal directed multigraph, configured window604800; native expiry lags one call in measured1.17.2; differs from whole-observation static FlowTopo simple graph",
                "amount_semantics": "unchanged paid/received raw amounts; no invented currency conversion",
                "time_ties": "immutable source txID canonical order, full group exactly once; wrapper reproducibility, not native LC-cycle permutation invariance",
                "native_known_deviations": receipt["native_known_deviations"],
                "native_default_params": defaults, "peak_rss_bytes": peak_rss_bytes(),
                "audit_sample_account_rows": sample_account_rows.tolist(),
                "audit_sample_edges": n_sample_edges,
                "runtime_seconds": time.monotonic() - started,
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "output_sha256": {relative(path): sha(path) for path in
                                  (destination, folder / "gfp_incident_counts.parquet", folder / "gfp_engineered_schema.json", sample_path)}}
    jsave(folder / "gfp_input_manifest.json", manifest)
    jsave(folder / "gfp_run_status.json", {"status": "complete", "dataset": ds, "pid": None,
                                           "n_accounts": len(accounts), "n_transactions": len(order),
                                           "runtime_seconds": manifest["runtime_seconds"],
                                           "peak_rss_bytes": manifest["peak_rss_bytes"]})
    print("GFP_COMPLETE", ds, len(accounts), q, manifest["runtime_seconds"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("hi_small", "li_small"), required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--native-evidence", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--memory-budget-gb", type=float, default=3.0)
    parser.add_argument("--aggregation-batch-size", type=int, default=2048)
    args = parser.parse_args()
    assert 0 < args.memory_budget_gb <= 3.0
    assert 1 <= args.aggregation_batch_size <= 2048
    try:
        build(args.out, args.dataset, args.native_evidence, args.wheel,
              args.memory_budget_gb, args.aggregation_batch_size)
    except BaseException as error:
        status_path = args.out / "inputs" / args.dataset / "gfp_run_status.json"
        previous = json.loads(status_path.read_text()) if status_path.exists() else {}
        previous.update(status="failed_uncommitted_input", pid=None,
                        exception_type=type(error).__name__, exception_message=str(error),
                        traceback=traceback.format_exc(), completed_utc=datetime.now(timezone.utc).isoformat())
        if "resource budget" in str(error).lower() or isinstance(error, MemoryError):
            previous["status"] = "stopped_resource_budget"
        status_path.parent.mkdir(parents=True, exist_ok=True)
        jsave(status_path, previous)
        raise
