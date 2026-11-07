"""Native GFP semantic gate and independent account-aggregation acceptance.

Native tests must run on the supported, exact frozen wheel before extraction.
Input validation uses batch-only reads, source transaction counts, independent
aggregation of all incidents of preselected accounts and complete source hashes.
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
import sys
import time
import traceback

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import build_account_eswa_gfp_20261013 as builder

ROOT, ALT, SOURCE = builder.ROOT, builder.ALT, builder.SOURCE
DEFAULT_OUT = builder.DEFAULT_OUT
RTOL, ATOL = 1e-10, 1e-9


def frame(edges):
    return np.ascontiguousarray(edges, dtype=np.float64).reshape((-1, 6))


def cleaned(output):
    result = np.array(output[:, 6:], dtype=np.float64)
    result[~np.isfinite(result)] = 0
    return result


def compare_by_id(first_raw, first_features, second_raw, second_features):
    first = np.argsort(first_raw[:, 0], kind="stable")
    second = np.argsort(second_raw[:, 0], kind="stable")
    np.testing.assert_array_equal(first_raw[first, 0], second_raw[second, 0])
    np.testing.assert_allclose(first_features[first], second_features[second], rtol=RTOL, atol=ATOL)


def direct_group_query(raw, ingestion_batch, permute=False, canonical=True):
    native, _, _ = builder.create_native()
    rng = np.random.default_rng(20261013)
    # Simulated input-reader chunks are reassembled before a whole-time-group
    # native call. They must never become separate transform/query calls.
    raw = np.concatenate([raw[a:a + ingestion_batch] for a in range(0, len(raw), ingestion_batch)])
    pieces = []
    for timestamp in np.unique(raw[:, 3]):
        group = raw[raw[:, 3] == timestamp]
        group = group[rng.permutation(len(group))] if permute else group
        if canonical:
            features, undefined = builder.canonical_engineered(native, group, len(builder.proposed_schema(builder.intended_parameters())))
            pieces.append(np.column_stack([group, features]))
        else:
            pieces.append(np.asarray(native.transform(group.copy())))
    return np.concatenate(pieces)


def native_gate(out, wheel):
    started = time.monotonic()
    folder = Path(out) / "inputs"
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / "gfp_native_validation.json"
    assert not destination.exists(), "Native gate receipt is immutable; use a new diagnostic output root for reruns."
    checks = []
    result = {"passed": False, "status": "failed_native_gate",
              "builder_sha256": builder.sha(Path(builder.__file__)),
              "validator_sha256": builder.sha(Path(__file__)),
              "protocol_amendment_sha256": builder.sha(builder.AMENDMENT),
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "pid": os.getpid(), "check_tolerance": {"rtol": RTOL, "atol": ATOL},
              "checks": checks, "scope": "Native API semantics, not classification performance."}
    try:
        environment = builder.native_environment(wheel)
        native, defaults, actual = builder.create_native()
        result.update(native_environment=environment, get_params_default=defaults, get_params_actual=actual)
        raw = frame([[0, 0, 1, 0, 10, 11], [1, 0, 2, 0, 20, 21]])
        first = np.asarray(native.transform(raw.copy()))
        assert first.ndim == 2 and first.shape[0] == 2 and first.shape[1] > 6
        assert np.array_equal(first[:, :6], raw)
        q = first.shape[1] - 6
        schema = builder.proposed_schema(actual)
        assert len(schema) == q, f"Official documented schema mismatch: expected {len(schema)}, native {q}"
        position = {column["name"]: column["id"] for column in schema}
        result.update(native_engineered_dimension=q, engineered_schema=schema)
        checks.append({"name": "native_api_parameters_and_output_dimension", "passed": True,
                       "n_engineered_columns": q})
        z = cleaned(first)
        expected = {"vertex_source_out_fan": 2, "vertex_source_out_degree": 2,
                    "vertex_source_out_ratio": 1, "vertex_source_in_degree": 0,
                    "vertex_target_in_degree": 1, "vertex_target_out_degree": 0,
                    "vertex_source_out_average_raw4": 15, "vertex_source_out_sum_raw4": 30,
                    "vertex_source_out_minimum_raw4": 10, "vertex_source_out_maximum_raw4": 20,
                    "vertex_source_out_median_raw4": 15, "vertex_source_out_average_raw5": 16,
                    "vertex_source_out_sum_raw5": 32}
        for name, value in expected.items():
            np.testing.assert_allclose(z[:, position[name]], value, rtol=RTOL, atol=ATOL, err_msg=name)
        # Fan-out pattern exists in a two-edge outgoing star; fan-in does not.
        fan_in = [column["id"] for column in schema if column.get("pattern") == "fan-in"]
        fan_out = [column["id"] for column in schema if column.get("pattern") == "fan-out"]
        assert z[:, fan_out].sum() > 0 and z[:, fan_in].sum() == 0
        checks.append({"name": "direction_and_vertex_statistic_column_semantics", "passed": True,
                       "expected_positions_checked": expected,
                       "fan_in_sum": float(z[:, fan_in].sum()), "fan_out_sum": float(z[:, fan_out].sum())})
        second = np.asarray(native.transform(raw.copy()))
        third = np.asarray(native.transform(raw.copy()))
        np.testing.assert_allclose(cleaned(second)[:, position["vertex_source_out_degree"]], 4)
        np.testing.assert_allclose(cleaned(third)[:, position["vertex_source_out_degree"]], 6)
        pattern_positions = [column["id"] for column in schema if column["type"] == "pattern_histogram"]
        changed_patterns = int((~np.isclose(cleaned(first)[:, pattern_positions],
                                          cleaned(second)[:, pattern_positions], rtol=RTOL, atol=ATOL)).sum())
        assert changed_patterns > 0, "Record known native repeat-ID non-idempotence, including pattern outputs."
        checks.append({"name": "repeat_id_negative_control_native_stats_not_idempotent", "passed": True,
                       "native_degree_sequence": [2, 4, 6],
                       "changed_native_pattern_cells": changed_patterns,
                       "documented_topology_deduplication_not_independently_established": True,
                       "production_requirement": "immutable unique txIDs, one transform per complete time group, no preinsert/same-ID query"})
        reversed_raw = raw.copy()
        reversed_raw[:, [1, 2]] = reversed_raw[:, [2, 1]]
        reversed_output = direct_group_query(reversed_raw, 2)
        reversed_z = cleaned(reversed_output)
        for first_name, second_name in (("fan-in", "fan-out"), ("degree-in", "degree-out")):
            first_indices = [column["id"] for column in schema if column.get("pattern") == first_name]
            second_indices = [column["id"] for column in schema if column.get("pattern") == second_name]
            np.testing.assert_allclose(z[:, first_indices], reversed_z[:, second_indices], rtol=RTOL, atol=ATOL)
            np.testing.assert_allclose(z[:, second_indices], reversed_z[:, first_indices], rtol=RTOL, atol=ATOL)
        checks.append({"name": "edge_reversal_swaps_fan_degree_in_out", "passed": True})
        parallel = frame([[0, 0, 1, 0, 10, 11], [1, 0, 1, 0, 20, 21]])
        parallel_z = cleaned(direct_group_query(parallel, 2))
        np.testing.assert_allclose(parallel_z[:, position["vertex_source_out_fan"]], 1)
        np.testing.assert_allclose(parallel_z[:, position["vertex_source_out_degree"]], 2)
        np.testing.assert_allclose(parallel_z[:, position["vertex_source_out_ratio"]], 2)
        checks.append({"name": "distinct_parallel_ids_remain_two_edges_and_native_ratio_is_degree_over_fan", "passed": True})
        self_raw = frame([[0, 0, 0, 0, 10, 11], [1, 0, 1, 0, 20, 21]])
        self_z = cleaned(direct_group_query(self_raw, 2))
        np.testing.assert_allclose(self_z[:, position["vertex_source_out_degree"]], 2)
        np.testing.assert_allclose(self_z[:, position["vertex_source_in_degree"]], 1)
        self_pattern = [column["id"] for column in schema if column["type"] == "pattern_histogram"]
        checks.append({"name": "self_transfer_kept_and_counted_in_out_roles", "passed": True,
                       "self_edge_native_pattern_values": self_z[0, self_pattern].tolist()})
        times = [0, builder.WINDOW - 1, builder.WINDOW, builder.WINDOW + 1,
                 2 * builder.WINDOW + 1, 2 * builder.WINDOW + 2]
        window_raw = frame([[i, 0, i + 1, timestamp, 10 * (i + 1), 11 * (i + 1)]
                            for i, timestamp in enumerate(times)])
        window_z = cleaned(direct_group_query(window_raw, 1))
        np.testing.assert_allclose(window_z[:, position["vertex_source_out_degree"]], [1, 2, 3, 4, 4, 3])
        np.testing.assert_allclose(window_z[:, position["vertex_source_out_sum_raw4"]], [10, 30, 60, 100, 140, 150])
        shifted = window_raw.copy()
        shifted[:, 3] += 1661990400
        shifted_z = cleaned(direct_group_query(shifted, 2))
        np.testing.assert_allclose(window_z, shifted_z, rtol=RTOL, atol=ATOL)
        checks.append({"name": "native_realized_window_expiry_lag_and_timestamp_shift_consistency", "passed": True,
                       "times": times, "native_degree_sequence": [1, 2, 3, 4, 4, 3],
                       "scope": "Measured native one-call cleanup lag, NOT exact current-time seven-day expiry."})
        tied = frame([[0, 0, 1, 0, 1, 2], [1, 1, 2, 0, 3, 4], [2, 2, 0, 0, 5, 6],
                      [3, 0, 2, 0, 7, 8], [4, 0, 1, 0, 9, 10], [5, 2, 2, 0, 11, 12],
                      [6, 2, 3, 60, 13, 14], [7, 3, 0, 60, 15, 16], [8, 1, 3, 60, 17, 18]])
        all_at_once = direct_group_query(tied, len(tied))
        single_at_once = direct_group_query(tied, 1)
        permuted = direct_group_query(tied, 2, permute=True)
        compare_by_id(all_at_once[:, :6], cleaned(all_at_once), single_at_once[:, :6], cleaned(single_at_once))
        compare_by_id(all_at_once[:, :6], cleaned(all_at_once), permuted[:, :6], cleaned(permuted))
        checks.append({"name": "canonical_wrapper_input_chunk_and_row_permutation_reproducibility", "passed": True,
                       "tested_ingestion_chunks": [1, 2, 9],
                       "native_batch_policy": "Each full timestamp group once; IDs canonicalized before transform."})
        direct = direct_group_query(tied, len(tied), canonical=False)
        direct_permuted = direct_group_query(tied, 2, permute=True, canonical=False)
        sorted_direct = np.argsort(direct[:, 0])
        sorted_permuted = np.argsort(direct_permuted[:, 0])
        first_direct = cleaned(direct)[sorted_direct]
        second_direct = cleaned(direct_permuted)[sorted_permuted]
        differences = np.where(~np.isclose(first_direct, second_direct, rtol=RTOL, atol=ATOL))
        differing_columns = sorted(set(int(column) for column in differences[1]))
        assert differing_columns == [36] and schema[36]["name"] == "pattern_lc-cycle_bin1"
        checks.append({"name": "native_unordered_tie_negative_control_lc_cycle_sensitivity", "passed": True,
                       "differing_schema_ids": differing_columns, "different_cells": len(differences[0]),
                       "scope": "Canonical wrapper reproducibility does not establish native tie-order invariance."})
        # Disk-backed aggregator itself is exercised on signed engineered values,
        # duplicate role rows, self-transfers and empty in/out roles.
        scratch = folder / "gfp_native_aggregation_test"
        assert not scratch.exists()
        scratch.mkdir()
        accumulator = builder.AccountAccumulator(scratch, 4, 2, create=True)
        audit_raw = frame([[0, 0, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0], [2, 2, 1, 0, 0, 0]])
        audit_features = np.array([[1, -3], [2, -5], [3, -7]], dtype=np.float64)
        accumulator.add(audit_raw[:1], audit_features[:1])
        accumulator.release_pages()
        accumulator.add(audit_raw[1:], audit_features[1:])
        accumulator.release_pages()
        expected_counts = np.array([[2, 0, 1, 0], [1, 2, 0, 0]])
        np.testing.assert_array_equal(accumulator.counts, expected_counts)
        np.testing.assert_array_equal(accumulator.sums[0, 0], [3, -8])
        np.testing.assert_array_equal(accumulator.maxima[0, 0], [2, -3])
        np.testing.assert_array_equal(accumulator.sums[1, 1], [4, -10])
        columns = [f"gfp_{j:03d}_{role}_{aggregation}" for j in range(2)
                   for role in builder.ROLES for aggregation in builder.AGGREGATES]
        path = scratch / "aggregated.parquet"
        accumulator.export(path, np.array(["a", "b", "c", "d"]), columns, batch_size=2)
        aggregated = pd.read_parquet(path)
        np.testing.assert_array_equal(aggregated.iloc[3, 1:].to_numpy(np.float64), np.zeros(12))
        assert aggregated.loc[0, "gfp_001_out_max"] == -3
        assert aggregated.loc[0, "gfp_001_out_mean"] == -4
        checks.append({"name": "disk_accumulator_signed_values_self_roles_and_empty_zero", "passed": True})
        result.update(passed=True, status="passed", pid=None,
                      native_known_deviations={
                          "repeat_ids": "Native statistics AND pattern outputs change on repeat-ID calls; documented topology deduplication was not established; repeats forbidden in production.",
                          "time_ties": "LC-cycle length3 bin can depend on same-time order; wrapper fixes immutable source txID order, not native invariance.",
                          "window": "Configured604800 unchanged; measured cleanup lags one native call; not exact current-time seven-day interval.",
                          "ratio": "Native1.17.2 measured degree/fan; documentation's fan/degree prose does not match parallel-edge probe."},
                      runtime_seconds=time.monotonic() - started,
                      completed_utc=datetime.now(timezone.utc).isoformat(),
                      peak_rss_bytes=builder.peak_rss_bytes(),
                      undefined_test_values_total=int((~np.isfinite(all_at_once[:, 6:])).sum()))
    except Exception as error:
        result.update(passed=False, status="failed_native_gate", pid=None,
                      exception_type=type(error).__name__, exception_message=str(error),
                      traceback=traceback.format_exc(), runtime_seconds=time.monotonic() - started,
                      completed_utc=datetime.now(timezone.utc).isoformat())
    builder.jsave(destination, result)
    print("GFP_NATIVE_GATE", result["status"], len(checks), flush=True)
    if not result["passed"]:
        raise SystemExit(2)


def validate_input(out, ds):
    started = time.monotonic()
    folder = Path(out) / "inputs" / ds
    manifest_path = folder / "gfp_input_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "complete"
    assert manifest["builder_sha256"] == builder.sha(Path(builder.__file__))
    assert builder.sha(Path(manifest["native_acceptance_path"])) == manifest["native_acceptance_sha256"]
    native_receipt = json.loads(Path(manifest["native_acceptance_path"]).read_text())
    assert native_receipt.get("passed") is True and native_receipt["status"] == "passed"
    assert native_receipt["native_environment"] == manifest["native_environment"]
    assert native_receipt["get_params_actual"] == manifest["get_params_actual"]
    supplement_path = folder / "gfp_input_manifest_supplement.json"
    supplement = json.loads(supplement_path.read_text())
    assert supplement["status"] == "complete_guarded_input_pending_independent_validation"
    assert supplement["base_input_manifest_sha256"] == builder.sha(manifest_path)
    assert supplement["n_checked_source_transactions"] == manifest["n_transactions"]
    assert supplement["every_source_id_checked"] is True and supplement["source_ids_unique_once"] is True
    assert supplement["n_histogram_columns"] == 40
    assert supplement["native_histogram_negative_or_noninteger_detected"] is False
    assert supplement["native_nonfinite_replacements_per_schema_id"] == manifest["undefined_native_values_by_schema_id"]
    assert supplement["launcher_sha256"] == builder.sha(ALT / "code/run_guarded_account_eswa_gfp_20261013.py")
    assert supplement["wrapper_preflight_sha256"] == builder.sha(builder.DEFAULT_OUT / "gfp_production_guard_preflight_20261013/preflight_receipt.json")
    assert supplement["production_guard_amendment_sha256"] == builder.sha(builder.DEFAULT_OUT / "GFP_PRODUCTION_GUARD_AMENDMENT_20261013.json")
    for key, expected in manifest["source_sha256"].items():
        assert builder.sha(ROOT / key) == expected, f"Source mismatch: {key}"
    for key, expected in manifest["output_sha256"].items():
        assert builder.sha(ROOT / key) == expected, f"Output mismatch: {key}"
    accounts = pd.read_parquet(SOURCE / ds / "features.parquet", columns=["account_id"]).account_id.to_numpy()
    index = pd.Index(accounts)
    assert index.is_unique and len(accounts) == manifest["n_accounts"]
    incident = pd.read_parquet(folder / "gfp_incident_counts.parquet")
    np.testing.assert_array_equal(incident.account_id.to_numpy(), accounts)
    counts = np.zeros((2, len(accounts)), dtype=np.int64)
    trans = pq.ParquetFile(ALT / "data" / ds / f"{ds}_trans.parquet")
    for batch in trans.iter_batches(batch_size=65536, columns=["from_id", "to_id"]):
        frame = batch.to_pandas()
        for role, column in enumerate(("from_id", "to_id")):
            rows = index.get_indexer(frame[column])
            assert np.all(rows >= 0)
            counts[role] += np.bincount(rows, minlength=len(accounts))
    np.testing.assert_array_equal(counts[0], incident.gfp_out_transaction_count.to_numpy())
    np.testing.assert_array_equal(counts[1], incident.gfp_in_transaction_count.to_numpy())
    assert int(counts[0].sum()) == int(counts[1].sum()) == manifest["n_transactions"]
    q = manifest["native_engineered_dimension"]
    native_schema = json.loads((folder / "gfp_engineered_schema.json").read_text())["columns"]
    histogram_ids = [item["id"] for item in native_schema if item["type"] == "pattern_histogram"]
    assert len(histogram_ids) == 40, "Frozen native schema requires 40 pattern histogram columns."
    pattern_undefined = {str(i): int(manifest["undefined_native_values_by_schema_id"][i]) for i in histogram_ids}
    columns = [f"gfp_{j:03d}_{role}_{aggregation}" for j in range(q)
               for role in builder.ROLES for aggregation in builder.AGGREGATES]
    assert columns == manifest["feature_columns"]
    account_file = pq.ParquetFile(folder / "gfp_account.parquet")
    assert account_file.schema_arrow.names == ["account_id", *columns]
    assert account_file.metadata.num_rows == len(accounts)
    sample_rows = np.array(manifest["audit_sample_account_rows"], dtype=np.int64)
    sample_index = {int(row): j for j, row in enumerate(sample_rows)}
    sampled_sums = np.zeros((2, len(sample_rows), q))
    sampled_maxima = np.full((2, len(sample_rows), q), -np.inf)
    sampled_counts = np.zeros((2, len(sample_rows)), dtype=np.int64)
    seen_edge_ids = set()
    sample_file = pq.ParquetFile(folder / "gfp_audit_edge_samples.parquet")
    for batch in sample_file.iter_batches(batch_size=8192):
        frame = batch.to_pandas()
        features = frame[[f"native_{j:03d}" for j in range(q)]].to_numpy(np.float64)
        assert np.isfinite(features).all()
        native_histograms = features[:, histogram_ids]
        assert np.all(native_histograms >= 0), "Negative native pattern count in independently sampled edges."
        np.testing.assert_allclose(native_histograms, np.rint(native_histograms), rtol=0, atol=1e-9,
                                   err_msg="Noninteger native pattern count in sampled edges.")
        for i, edge in enumerate(frame.itertuples(index=False)):
            edge_id = int(edge.edge_id)
            assert edge_id not in seen_edge_ids
            seen_edge_ids.add(edge_id)
            for role, row in enumerate((int(edge.source_account_row), int(edge.target_account_row))):
                if row in sample_index:
                    j = sample_index[row]
                    sampled_counts[role, j] += 1
                    # Independent scalar-row aggregation, without np.add.at or
                    # reading the builder's large accumulator files.
                    sampled_sums[role, j] = sampled_sums[role, j] + features[i]
                    sampled_maxima[role, j] = np.maximum(sampled_maxima[role, j], features[i])
    assert len(seen_edge_ids) == manifest["audit_sample_edges"]
    np.testing.assert_array_equal(sampled_counts, counts[:, sample_rows])
    row = 0
    n_checked_samples = 0
    for batch in account_file.iter_batches(batch_size=8192):
        frame = batch.to_pandas()
        stop = row + len(frame)
        np.testing.assert_array_equal(frame.account_id.to_numpy(), accounts[row:stop])
        values = frame[columns].to_numpy(np.float64).reshape(len(frame), q, 2, 3)
        assert np.isfinite(values).all() and np.isfinite(values.astype(np.float32)).all()
        histograms = values[:, histogram_ids]
        assert np.all(histograms >= 0), "Negative pattern count aggregate; input not admitted."
        # Means can be fractional; only sums and maxima retain count integrality.
        histogram_count_aggregates = histograms[:, :, :, [0, 2]]
        np.testing.assert_allclose(histogram_count_aggregates,
                                   np.rint(histogram_count_aggregates), rtol=0, atol=1e-9,
                                   err_msg="Noninteger histogram sum/max aggregate; input not admitted.")
        for role in range(2):
            count = counts[role, row:stop]
            total, mean, maximum = values[:, :, role, 0], values[:, :, role, 1], values[:, :, role, 2]
            expected_mean = np.divide(total, count[:, None], out=np.zeros_like(total), where=count[:, None] > 0)
            np.testing.assert_allclose(mean, expected_mean, rtol=1e-12, atol=1e-12)
            np.testing.assert_array_equal(values[count == 0, :, role, :], np.zeros((int((count == 0).sum()), q, 3)))
            assert np.all(maximum[count > 0] + 1e-9 * (1 + np.abs(maximum[count > 0])) >= mean[count > 0])
        for global_row in sample_rows[(sample_rows >= row) & (sample_rows < stop)]:
            sample_id = sample_index[int(global_row)]
            for role in range(2):
                count = sampled_counts[role, sample_id]
                total = sampled_sums[role, sample_id]
                mean = total / count if count else np.zeros(q)
                maximum = sampled_maxima[role, sample_id] if count else np.zeros(q)
                expected = np.column_stack([total, mean, maximum])
                np.testing.assert_allclose(values[int(global_row) - row, :, role, :], expected, rtol=1e-12, atol=1e-10)
            n_checked_samples += 1
        row = stop
    assert row == len(accounts) and n_checked_samples == len(sample_rows)
    for key, expected in manifest["source_sha256"].items():
        assert builder.sha(ROOT / key) == expected
    for key, expected in manifest["output_sha256"].items():
        assert builder.sha(ROOT / key) == expected
    result = {"passed": True, "status": "passed", "dataset": ds,
              "n_accounts": len(accounts), "n_transactions": manifest["n_transactions"],
              "n_features": len(columns), "native_engineered_dimension": q,
              "n_independently_aggregated_accounts": n_checked_samples,
              "n_audit_native_edges": len(seen_edge_ids),
              "all_row_checks": {"strict_original_account_order": True,
                                 "no_labels_or_indexing_fields_in_features": True,
                                 "float64_and_training_float32_finite": True,
                                 "native_transaction_role_counts_from_full_original_data": True,
                                 "means_equal_sums_divided_by_counts": True,
                                 "empty_roles_exact_zero": True,
                                 "maxima_at_least_mean": True,
                                 "histogram_all_rows_nonnegative_and_sum_max_integer": True,
                                 "every_source_native_histogram_passed_bound_read_only_stream_guard": True,
                                 "all_scientific_sources_unchanged": True},
              "sampled_raw_native_histogram_checks": {"n_columns": 40, "finite_nonnegative_integer": True,
                                                       "scope": "All retained native edges of64 prespecified accounts; full other edge matrix not retained."},
              "native_pattern_nonfinite_replacements_by_schema_id": pattern_undefined,
              "native_pattern_nonfinite_replacement_flag": any(pattern_undefined.values()),
              "native_negative_or_noninteger_histogram_detected": False,
              "input_manifest_sha256": builder.sha(manifest_path),
              "input_manifest_supplement_sha256": builder.sha(supplement_path),
              "gfp_account_sha256": builder.sha(folder / "gfp_account.parquet"),
              "builder_sha256": builder.sha(Path(builder.__file__)),
              "validator_sha256": builder.sha(Path(__file__)),
              "runtime_seconds": time.monotonic() - started,
              "completed_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Official native input and account aggregation acceptance; no classifier results or future prediction evidence."}
    builder.jsave(folder / "gfp_validation.json", result)
    print("GFP_INPUT_VALIDATION_PASSED", ds, len(accounts), n_checked_samples, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("native", "input"), required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dataset", choices=("hi_small", "li_small"))
    parser.add_argument("--wheel", type=Path)
    args = parser.parse_args()
    if args.phase == "native":
        assert args.wheel is not None
        native_gate(args.out, args.wheel)
    else:
        assert args.dataset is not None
        try:
            validate_input(args.out, args.dataset)
        except BaseException as error:
            folder = args.out / "inputs" / args.dataset
            folder.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            failed = {"passed": False, "status": "failed_input_guard", "dataset": args.dataset,
                      "exception_type": type(error).__name__, "exception_message": str(error),
                      "traceback": traceback.format_exc(),
                      "histogram_guard_exception": "histogram" in str(error).lower() or "pattern count" in str(error).lower(),
                      "validator_sha256": builder.sha(Path(__file__)),
                      "completed_utc": datetime.now(timezone.utc).isoformat(),
                      "scope": "Input not admitted to classifiers; retain diagnosis, never silently accept anomalous patterns."}
            builder.jsave(folder / f"gfp_validation_failed_{stamp}.json", failed)
            raise
