"""Lossless-to-training float32 cache for an independently accepted GFP input.

The cache equals the frozen runner's original float64.astype(float32) conversion
bit for bit, validated by a separate complete readback pass. Never registers or
trains a classifier; the original float64 scientific feature input is preserved.
"""
from __future__ import annotations
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import build_account_eswa_gfp_20261013 as builder


def build_and_verify(out, ds):
    started = time.monotonic()
    folder = Path(out) / "inputs" / ds
    original = folder / "gfp_account.parquet"
    original_validation = folder / "gfp_validation.json"
    evidence = json.loads(original_validation.read_text())
    assert evidence.get("passed") is True and evidence["status"] == "passed"
    original_sha = builder.sha(original)
    assert evidence["gfp_account_sha256"] == original_sha
    manifest_path = folder / "gfp_input_manifest.json"
    supplement_path = folder / "gfp_input_manifest_supplement.json"
    assert evidence["input_manifest_sha256"] == builder.sha(manifest_path)
    assert evidence["input_manifest_supplement_sha256"] == builder.sha(supplement_path)
    columns = pq.ParquetFile(original).schema_arrow.names
    assert columns[0] == "account_id" and "label" not in columns
    assert len(columns) == 697
    destination = folder / "gfp_account_train_float32.parquet"
    assert not destination.exists(), "Completed cache is immutable."
    temporary = destination.with_suffix(".tmp.parquet")
    writer = None
    n_written = 0
    try:
        for batch in pq.ParquetFile(original).iter_batches(batch_size=4096):
            raw = batch.to_pandas()
            numeric = np.ascontiguousarray(raw[columns[1:]].to_numpy(np.float64).astype(np.float32))
            assert np.isfinite(numeric).all()
            arrays = {"account_id": pa.array(raw.account_id)}
            arrays.update({column: pa.array(numeric[:, j], type=pa.float32())
                           for j, column in enumerate(columns[1:])})
            table = pa.table(arrays)
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
            writer.write_table(table)
            n_written += len(raw)
        assert writer is not None
    finally:
        if writer is not None:
            writer.close()
    temporary.replace(destination)
    # Independent full readback, not an in-memory comparison before writing.
    original_batches = pq.ParquetFile(original).iter_batches(batch_size=4096)
    cache_file = pq.ParquetFile(destination)
    assert cache_file.schema_arrow.names == columns
    assert all(pa.types.is_float32(field.type) for field in list(cache_file.schema_arrow)[1:])
    cache_batches = cache_file.iter_batches(batch_size=4096)
    rows_checked = 0
    for source, cached in itertools.zip_longest(original_batches, cache_batches):
        assert source is not None and cached is not None
        first, second = source.to_pandas(), cached.to_pandas()
        np.testing.assert_array_equal(first.account_id.to_numpy(), second.account_id.to_numpy())
        expected = np.ascontiguousarray(first[columns[1:]].to_numpy(np.float64).astype(np.float32))
        actual = np.ascontiguousarray(second[columns[1:]].to_numpy(np.float32))
        assert expected.shape == actual.shape and expected.tobytes() == actual.tobytes(), "Cache conversion differs from frozen runner."
        rows_checked += len(first)
    assert rows_checked == n_written == evidence["n_accounts"]
    assert builder.sha(original) == original_sha
    cache_sha = builder.sha(destination)
    result = {"passed": True, "status": "passed", "dataset": ds,
              "n_accounts": rows_checked, "n_features": len(columns) - 1,
              "feature_columns": columns[1:], "storage_dtype": "float32",
              "conversion": "exact original float64.astype(float32) on each cell; all rows read back and bytes compared",
              "origin_float64_path": builder.relative(original), "origin_float64_sha256": original_sha,
              "origin_input_manifest_sha256": builder.sha(manifest_path),
              "origin_input_manifest_supplement_sha256": builder.sha(supplement_path),
              "origin_independent_validation_sha256": builder.sha(original_validation),
              "cache_path": builder.relative(destination), "cache_sha256": cache_sha,
              "converter_sha256": builder.sha(Path(__file__)),
              "n_numeric_cells_bitwise_checked": rows_checked * (len(columns) - 1),
              "all_row_checks": {"source_account_order_identity": True,
                                 "column_order_identity": True, "no_label_or_indexing_feature_columns": True,
                                 "float32_finite": True, "bitwise_original_training_conversion": True,
                                 "original_float64_input_unchanged": True},
              "tree_admitted": False, "classifier_fits": 0,
              "runtime_seconds": time.monotonic() - started,
              "completed_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Training memory cache only; root must review one-worker high-dimensional training/admission separately."}
    builder.jsave(folder / "gfp_float32_cache_validation.json", result)
    builder.jsave(folder / "gfp_float32_cache_manifest.json", result)
    print("GFP_FLOAT32_CACHE_ACCEPTED", ds, rows_checked, cache_sha, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("hi_small", "li_small"), required=True)
    parser.add_argument("--out", type=Path, default=builder.DEFAULT_OUT)
    args = parser.parse_args()
    build_and_verify(args.out, args.dataset)
