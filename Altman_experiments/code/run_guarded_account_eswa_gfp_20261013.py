"""Explicit production-only safety wrapper for the frozen official GFP builder.

Does not modify the builder, native algorithm, feature values or arithmetic
order. Checks every source histogram after the original cleaning policy and
releases accumulator mappings after each aggregation chunk. Separate wrapper
preflight is required; original native QA is not claimed to cover this wrapper.
"""
from __future__ import annotations
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import build_account_eswa_gfp_20261013 as builder

FROZEN_BUILDER_SHA = "802a9a1e7ed2661253f825b84264dce217176cb75de08152ba1a261f3a82a43d"
AMENDMENT = builder.DEFAULT_OUT / "GFP_PRODUCTION_GUARD_AMENDMENT_20261013.json"
PREFLIGHT = builder.DEFAULT_OUT / "gfp_production_guard_preflight_20261013/preflight_receipt.json"


class HistogramStreamGuard:
    def __init__(self, n_transactions, q, histogram_ids):
        self.seen = np.zeros(n_transactions, dtype=bool)
        self.histogram_ids = np.array(histogram_ids, dtype=np.int64)
        self.q = q
        self.n_checked = 0
        self.n_groups = 0
        self.nonfinite_counts = np.zeros(q, dtype=np.int64)

    def check_ids_before_call(self, raw):
        ids = raw[:, 0].astype(np.int64)
        assert np.array_equal(ids, raw[:, 0])
        assert np.all((ids >= 0) & (ids < len(self.seen)))
        assert len(np.unique(ids)) == len(ids) and not self.seen[ids].any(), "Duplicate source txID forbidden."

    def accept(self, raw, features, undefined):
        self.check_ids_before_call(raw)
        assert features.shape == (len(raw), self.q) and features.dtype == np.float64
        assert np.isfinite(features).all()
        histograms = features[:, self.histogram_ids]
        bad = (histograms < 0) | (np.abs(histograms - np.rint(histograms)) > 1e-9)
        if bad.any():
            rows, columns = np.where(bad)
            examples = [{"source_txID": int(raw[row, 0]),
                         "native_schema_id": int(self.histogram_ids[column]),
                         "value": float(histograms[row, column])}
                        for row, column in zip(rows[:20], columns[:20])]
            raise ValueError("Negative/noninteger source native histogram: " + json.dumps(examples))
        self.seen[raw[:, 0].astype(np.int64)] = True
        self.n_checked += len(raw)
        self.n_groups += 1
        self.nonfinite_counts += undefined

    def report(self):
        return {"n_checked_source_transactions": self.n_checked,
                "n_complete_timestamp_groups": self.n_groups,
                "n_histogram_columns": len(self.histogram_ids),
                "source_ids_unique_once": bool(self.seen.sum() == self.n_checked),
                "every_source_id_checked": bool(self.seen.all()),
                "native_histogram_negative_or_noninteger_detected": False,
                "native_nonfinite_replacements_per_schema_id": self.nonfinite_counts.tolist(),
                "pattern_nonfinite_replacements_per_schema_id": {
                    str(i): int(self.nonfinite_counts[i]) for i in self.histogram_ids},
                "pattern_nonfinite_replacement_flag": bool(self.nonfinite_counts[self.histogram_ids].sum())}


def preflight():
    assert builder.sha(Path(builder.__file__)) == FROZEN_BUILDER_SHA
    assert not PREFLIGHT.exists(), "Wrapper preflight is immutable."
    folder = PREFLIGHT.parent
    folder.mkdir(parents=True)
    schema = builder.proposed_schema(builder.intended_parameters())
    histogram_ids = [item["id"] for item in schema if item["type"] == "pattern_histogram"]
    assert len(schema) == 116 and len(histogram_ids) == 40
    n = 25
    rng = np.random.default_rng(20261013)
    raw = np.zeros((n, 6), dtype=np.float64)
    raw[:, 0] = np.arange(n)
    raw[:, 1:3] = rng.integers(0, 5, (n, 2))
    raw[0, 1:3] = [0, 0]  # Explicit self-transfer.
    features = rng.normal(size=(n, len(schema)))
    features[:, histogram_ids] = rng.integers(0, 50, size=(n, len(histogram_ids)))
    features = np.ascontiguousarray(features)
    original_bytes = features.tobytes()
    guard = HistogramStreamGuard(n, len(schema), histogram_ids)
    guard.accept(raw, features, np.zeros(len(schema), dtype=np.int64))
    assert features.tobytes() == original_bytes and guard.report()["every_source_id_checked"]
    failures = []
    for name, value in (("negative", -1), ("noninteger", 1.5)):
        bad = features[:1].copy()
        bad[0, histogram_ids[0]] = value
        candidate = HistogramStreamGuard(n, len(schema), histogram_ids)
        try:
            candidate.accept(raw[:1], bad, np.zeros(len(schema), dtype=np.int64))
        except ValueError:
            failures.append(name)
        else:
            raise AssertionError("Injected histogram anomaly was not rejected: " + name)
    duplicate = HistogramStreamGuard(n, len(schema), histogram_ids)
    duplicate.accept(raw[:1], features[:1], np.zeros(len(schema), dtype=np.int64))
    try:
        duplicate.check_ids_before_call(raw[:1])
    except AssertionError:
        failures.append("duplicate_source_txID")
    else:
        raise AssertionError("Duplicate txID guard failed.")
    accumulators = []
    for name in ("base_release", "per_chunk_release"):
        path = folder / name
        path.mkdir()
        accumulator = builder.AccountAccumulator(path, 8, len(schema), create=True)
        for a in range(0, n, 4):
            accumulator.add(raw[a:a + 4], features[a:a + 4])
            if name == "per_chunk_release":
                accumulator.release_pages()
        accumulator.release_pages()
        columns = [f"gfp_{j:03d}_{role}_{aggregation}" for j in range(len(schema))
                   for role in builder.ROLES for aggregation in builder.AGGREGATES]
        accumulator.export(path / "account_features.parquet", np.array([f"a{i}" for i in range(8)]), columns, batch_size=3)
        accumulators.append((np.array(accumulator.sums), np.array(accumulator.maxima),
                             np.array(accumulator.counts), pd.read_parquet(path / "account_features.parquet")))
    for first, second in zip(accumulators[0][:3], accumulators[1][:3]):
        assert first.tobytes() == second.tobytes(), "Accumulator bits differ after extra page releases."
    np.testing.assert_array_equal(accumulators[0][3].to_numpy(), accumulators[1][3].to_numpy())
    receipt = {"passed": True, "status": "passed_wrapper_preflight", "date_client": "2026-10-13",
               "launcher_sha256": builder.sha(Path(__file__)), "frozen_builder_sha256": FROZEN_BUILDER_SHA,
               "checks": {"all_source_histogram_checks_read_only": True,
                          "negative_and_noninteger_injections_rejected": True,
                          "duplicate_source_ID_rejected_before_native_call": True,
                          "per_chunk_release_sum_max_count_bitwise_identical": True,
                          "all_6q_exported_statistics_identical": True,
                          "signed_vertex_statistics_preserved": True,
                          "self_transfers_and_empty_accounts_in_fixture": True},
               "rejected_injections": failures, "n_test_transactions": n,
               "n_native_histogram_columns": len(histogram_ids), "q": len(schema),
               "scope": "Separate implementation wrapper QA; no native algorithm or feature-value modification; not classifier evidence.",
               "completed_utc": datetime.now(timezone.utc).isoformat()}
    shutil.copy2(Path(__file__), folder / Path(__file__).name)
    builder.jsave(PREFLIGHT, receipt)
    print("GFP_GUARD_PREFLIGHT_PASSED", flush=True)


def guarded_build(args):
    started = time.monotonic()
    assert builder.sha(Path(builder.__file__)) == FROZEN_BUILDER_SHA
    preflight_receipt = json.loads(PREFLIGHT.read_text())
    assert preflight_receipt.get("passed") is True
    assert preflight_receipt["launcher_sha256"] == builder.sha(Path(__file__))
    amendment = json.loads(AMENDMENT.read_text())
    assert amendment["launcher_sha256"] == builder.sha(Path(__file__))
    native_receipt = json.loads(args.native_evidence.read_text())
    assert native_receipt.get("passed") is True and native_receipt["builder_sha256"] == FROZEN_BUILDER_SHA
    schema = native_receipt["engineered_schema"]
    histogram_ids = [item["id"] for item in schema if item["type"] == "pattern_histogram"]
    assert len(histogram_ids) == 40
    n_transactions = pq.ParquetFile(builder.ALT / "data" / args.dataset / f"{args.dataset}_trans.parquet").metadata.num_rows
    guard = HistogramStreamGuard(n_transactions, native_receipt["native_engineered_dimension"], histogram_ids)
    folder = args.out / "inputs" / args.dataset
    folder.mkdir(parents=True, exist_ok=True)
    original_kernel = builder.canonical_engineered
    original_add = builder.AccountAccumulator.add
    extra_releases = 0

    def checked_kernel(native, raw, q):
        guard.check_ids_before_call(raw)
        features, undefined = original_kernel(native, raw, q)
        guard.accept(raw, features, undefined)
        return features, undefined

    def release_after_add(self, raw, features):
        nonlocal extra_releases
        assert len(raw) <= args.aggregation_batch_size <= 2048
        original_add(self, raw, features)
        self.release_pages()
        extra_releases += 1
        if builder.peak_rss_bytes() > args.memory_budget_gb * 1024**3:
            raise MemoryError("Guarded production exceeded registered peak-RSS budget.")

    binding = {"status": "guarded_production_running", "pid": os.getpid(), "dataset": args.dataset,
               "launcher_sha256": builder.sha(Path(__file__)), "frozen_builder_sha256": FROZEN_BUILDER_SHA,
               "native_qa_receipt_sha256": builder.sha(args.native_evidence),
               "native_qa_scope": "Original realized-semantics native gate, separately supplemented by wrapper QA.",
               "wrapper_preflight_sha256": builder.sha(PREFLIGHT),
               "production_guard_amendment_sha256": builder.sha(AMENDMENT),
               "api_amendment_sha256": builder.sha(builder.AMENDMENT),
               "api_addendum_sha256": builder.sha(builder.DEFAULT_OUT / "GFP_PROTOCOL_AMENDMENT_ADDENDUM_20261013.json"),
               "extra_page_release": "after every aggregation chunk<=2048rows, in addition to unchanged builder every32groups",
               "features_and_arithmetic_unchanged": True, "created_utc": datetime.now(timezone.utc).isoformat()}
    builder.jsave(folder / "gfp_production_guard_binding.json", binding)
    builder.canonical_engineered = checked_kernel
    builder.AccountAccumulator.add = release_after_add
    try:
        builder.build(args.out, args.dataset, args.native_evidence, args.wheel,
                      args.memory_budget_gb, args.aggregation_batch_size)
        assert guard.n_checked == n_transactions and guard.seen.all()
        base_manifest = folder / "gfp_input_manifest.json"
        supplement = {**binding, "status": "complete_guarded_input_pending_independent_validation", "pid": None,
                      **guard.report(), "extra_page_release_calls": extra_releases,
                      "base_input_manifest_sha256": builder.sha(base_manifest),
                      "q10_or_classifier_results_created": False,
                      "runtime_seconds": time.monotonic() - started,
                      "completed_utc": datetime.now(timezone.utc).isoformat()}
        builder.jsave(folder / "gfp_input_manifest_supplement.json", supplement)
        builder.jsave(folder / "gfp_production_guard_binding.json", supplement)
        print("GFP_GUARDED_INPUT_COMPLETE", args.dataset, n_transactions, extra_releases, flush=True)
    except BaseException as error:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        failed = {**binding, "status": "failed_uncommitted_guarded_input", "pid": None,
                  "n_checked_source_transactions": guard.n_checked,
                  "extra_page_release_calls": extra_releases,
                  "exception_type": type(error).__name__, "exception_message": str(error),
                  "traceback": traceback.format_exc(), "completed_utc": datetime.now(timezone.utc).isoformat()}
        builder.jsave(folder / f"gfp_stream_guard_failed_{stamp}.json", failed)
        builder.jsave(folder / "gfp_production_guard_binding.json", failed)
        status_path = folder / "gfp_run_status.json"
        state = json.loads(status_path.read_text()) if status_path.exists() else {}
        builder.jsave(status_path, {**state, "status": "failed_uncommitted_guarded_input", "pid": None,
                                  "exception_type": type(error).__name__, "exception_message": str(error)})
        raise
    finally:
        builder.canonical_engineered = original_kernel
        builder.AccountAccumulator.add = original_add


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "build"), required=True)
    parser.add_argument("--dataset", choices=("hi_small", "li_small"))
    parser.add_argument("--out", type=Path, default=builder.DEFAULT_OUT)
    parser.add_argument("--native-evidence", type=Path)
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--memory-budget-gb", type=float, default=3.0)
    parser.add_argument("--aggregation-batch-size", type=int, default=2048)
    args = parser.parse_args()
    assert 0 < args.memory_budget_gb <= 3.0 and 1 <= args.aggregation_batch_size <= 2048
    if args.phase == "preflight":
        preflight()
    else:
        assert args.dataset and args.native_evidence and args.wheel
        guarded_build(args)
