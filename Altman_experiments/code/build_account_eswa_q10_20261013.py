"""Label-blind Q10 controls from frozen cap-40 samples, without PH or spectra.

The six existing Q6 counts are reused exactly. Only the four missing radius-2
edge/triangle counts are computed, using a uint64 adjacency representation.
Old geometry, source chunks, account order and feature files are read-only.
"""
from __future__ import annotations

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time

from numba import njit
import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]
ALT = ROOT / "Altman_experiments"
SOURCE = ALT / "results/account_s10_increment_20261008"
DEFAULT_OUT = ALT / "results/account_eswa_supplement_20261013"
DESIGN = ALT / "results/eswa_supplement_design_20261013/source_manifest.json"
Q10 = [f"q{h}_{key}" for h in (2, 3) for key in ("n", "e1", "t1", "e2", "t2")]
Q6 = ["q_n2", "q_n", "q_e1", "q_t1", "q_e2", "q_t2"]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def jsave(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


@njit(cache=True)
def popcount(value):
    """SWAR population count; no platform-specific intrinsic required."""
    value = value - ((value >> np.uint64(1)) & np.uint64(0x5555555555555555))
    value = (value & np.uint64(0x3333333333333333)) + ((value >> np.uint64(2)) & np.uint64(0x3333333333333333))
    value = (value + (value >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return np.int64((value * np.uint64(0x0101010101010101)) >> np.uint64(56))


@njit(cache=True)
def simplex_counts(masks, n):
    edges = np.int64(0)
    triangles = np.int64(0)
    all_bits = (np.uint64(1) << np.uint64(n)) - np.uint64(1)
    for i in range(n):
        for j in range(i + 1, n):
            if masks[i] & (np.uint64(1) << np.uint64(j)):
                edges += 1
                higher = all_bits ^ ((np.uint64(1) << np.uint64(j + 1)) - np.uint64(1))
                triangles += popcount(masks[i] & masks[j] & higher)
    return edges, triangles


@njit(cache=True)
def counts_from_vertices(vertices, n, indptr, indices):
    """Counts of flag-complex edges/triangles at shortest-path scales 1,2."""
    masks = np.zeros(n, dtype=np.uint64)
    for i in range(n):
        lo0 = indptr[vertices[i]]
        hi0 = indptr[vertices[i] + 1]
        for j in range(i + 1, n):
            value = vertices[j]
            lo = lo0
            hi = hi0
            while lo < hi:
                mid = (lo + hi) // 2
                if indices[mid] < value:
                    lo = mid + 1
                else:
                    hi = mid
            if lo < hi0 and indices[lo] == value:
                masks[i] |= np.uint64(1) << np.uint64(j)
                masks[j] |= np.uint64(1) << np.uint64(i)
    e1, t1 = simplex_counts(masks, n)
    scale2 = masks.copy()
    for i in range(n):
        for j in range(n):
            if masks[i] & (np.uint64(1) << np.uint64(j)):
                scale2[i] |= masks[j]
        scale2[i] &= ~(np.uint64(1) << np.uint64(i))
    e2, t2 = simplex_counts(scale2, n)
    return e1, t1, e2, t2


@njit(cache=True)
def batch_twohop(vertices, depths, indptr, indices):
    values = np.zeros((len(vertices), 4), dtype=np.int64)
    for row in range(len(vertices)):
        n = 0
        for j in range(vertices.shape[1]):
            if vertices[row, j] >= 0 and depths[row, j] <= 2:
                n += 1
        values[row] = counts_from_vertices(vertices[row], n, indptr, indices)
    return values


_GRAPH = None
_OUT = None
_DS = None


def init_worker(out, ds):
    global _GRAPH, _OUT, _DS
    _GRAPH = sparse.load_npz(SOURCE / ds / "graph_csr.npz")
    _OUT = Path(out)
    _DS = ds


def chunk_worker(path):
    path = Path(path)
    a, b = [int(x) for x in path.stem.split("_")]
    destination = _OUT / "inputs" / _DS / "q10_chunks" / path.name
    receipt = destination.with_suffix(".json")
    if destination.exists():
        old = json.loads(receipt.read_text())
        assert old["source_sha256"] == sha(path)
        assert old["builder_sha256"] == sha(Path(__file__))
        assert old["output_sha256"] == sha(destination)
        return {"start": a, "stop": b, "reused": True}
    start = time.monotonic()
    with np.load(path, allow_pickle=False) as source:
        vertices = source["vertices"]
        depths = source["depths"]
        q6 = source["values"][:, 12:18]
        assert vertices.shape == depths.shape == (b - a, 40)
        assert q6.shape == (b - a, 6)
        assert np.isfinite(q6).all() and np.array_equal(q6, q6.astype(np.int64))
        n3 = (vertices >= 0).sum(axis=1)
        n2 = ((vertices >= 0) & (depths <= 2)).sum(axis=1)
        assert np.array_equal(n2, q6[:, 0]) and np.array_equal(n3, q6[:, 1])
        assert np.all(vertices[:, 0] >= 0) and np.all(depths[:, 0] == 0)
        assert np.all((vertices < 0) == (depths < 0))
        assert np.all(np.diff(depths, axis=1)[(depths[:, 1:] >= 0)] >= 0)
        new = batch_twohop(vertices, depths, _GRAPH.indptr, _GRAPH.indices)
        counts = np.column_stack([q6[:, 0], new, q6[:, 1:]]).astype(np.int64)
    temporary = destination.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, counts=counts)
    temporary.replace(destination)
    report = {"start": a, "stop": b, "status": "complete",
              "runtime_seconds": time.monotonic() - start,
              "source_sha256": sha(path), "builder_sha256": sha(Path(__file__)),
              "output_sha256": sha(destination), "n_accounts": b - a}
    jsave(receipt, report)
    return report


def selected_sources(ds):
    folder = SOURCE / ds
    paths = [SOURCE / "feature_protocol.json", folder / "features.parquet",
             folder / "graph_csr.npz", folder / "graph_accounts.parquet",
             folder / "account_to_graph.npy", folder / "graph_manifest.json"]
    paths.extend(sorted((folder / "chunks").glob("*.npz")))
    paths.extend(sorted((folder / "chunks").glob("*.json")))
    return paths


def build(out, ds, workers):
    out = Path(out)
    folder = out / "inputs" / ds
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "q10_chunks").mkdir(exist_ok=True)
    started = time.monotonic()
    paths = selected_sources(ds)
    frozen = json.loads(DESIGN.read_text())["source_sha256"]
    hashes = {}
    for path in paths:
        key = relative(path)
        digest = sha(path)
        assert frozen[key] == digest, f"Frozen design source changed: {key}"
        hashes[key] = digest
    chunks = sorted((SOURCE / ds / "chunks").glob("*.npz"))
    order = np.load(SOURCE / ds / "account_to_graph.npy", allow_pickle=False)
    expected = [(a, min(a + 10000, len(order))) for a in range(0, len(order), 10000)]
    assert [tuple(map(int, p.stem.split("_"))) for p in chunks] == expected
    jsave(folder / "q10_run_status.json", {
        "status": "building", "dataset": ds, "workers": workers,
        "n_accounts": len(order), "n_chunks": len(chunks),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(), "source_sha256": hashes,
        "builder_sha256": sha(Path(__file__))})
    # Compile before spawning; each worker keeps all numerical libraries single-threaded.
    tiny = sparse.csr_matrix(np.array([[0, 1], [1, 0]], dtype=np.int8))
    assert counts_from_vertices(np.arange(2), 2, tiny.indptr, tiny.indices) == (1, 0, 1, 0)
    if workers == 1:
        init_worker(out, ds)
        for path in chunks:
            print(json.dumps(chunk_worker(path)), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=init_worker,
                                 initargs=(str(out), ds)) as pool:
            for report in pool.map(chunk_worker, chunks):
                print(json.dumps(report), flush=True)
    values = []
    for path in chunks:
        a, b = [int(x) for x in path.stem.split("_")]
        with np.load(folder / "q10_chunks" / path.name, allow_pickle=False) as chunk:
            value = chunk["counts"]
            assert value.shape == (b - a, 10) and value.dtype == np.int64
            values.append(value)
    values = np.concatenate(values)
    ids = pd.read_parquet(SOURCE / ds / "features.parquet", columns=["account_id", *Q6])
    assert len(ids) == len(order) == len(values) and ids.account_id.is_unique
    graph_ids = pd.read_parquet(SOURCE / ds / "graph_accounts.parquet", columns=["account_id"])
    assert np.array_equal(ids.account_id.to_numpy(), graph_ids.account_id.to_numpy()[order])
    assert np.array_equal(values[:, [0, 5, 6, 7, 8, 9]], ids[Q6].to_numpy())
    result = pd.DataFrame(values, columns=Q10)
    result.insert(0, "account_id", ids.account_id.to_numpy())
    destination = folder / "q10.parquet"
    temporary = destination.with_suffix(".tmp.parquet")
    result.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(destination)
    for path in paths:
        assert sha(path) == hashes[relative(path)], f"Source modified during build: {path}"
    manifest = {"status": "complete", "dataset": ds, "n_accounts": len(ids),
                "feature_columns": Q10, "storage_dtype": "int64",
                "model_conversion": "float32 only at training input construction",
                "identity": "original feature account order; no labels read by builder",
                "sampling": "exact frozen vertices/depths; no neighborhood resampling",
                "metric": "unweighted shortest paths inside each induced sampled subgraph",
                "simplex_counts": "flag edges and triangles at epsilon 1 and 2; no spectra/PH extraction",
                "reused_q6_columns": {"q2_n": "q_n2", "q3_n": "q_n", "q3_e1": "q_e1",
                                      "q3_t1": "q_t1", "q3_e2": "q_e2", "q3_t2": "q_t2"},
                "source_sha256": hashes, "design_manifest_sha256": sha(DESIGN),
                "builder_sha256": sha(Path(__file__)),
                "output_sha256": {relative(destination): sha(destination)},
                "n_chunks": len(chunks), "runtime_seconds": time.monotonic() - started,
                "completed_utc": datetime.now(timezone.utc).isoformat()}
    jsave(folder / "q10_input_manifest.json", manifest)
    jsave(folder / "q10_run_status.json", {"status": "complete", "dataset": ds,
                                           "n_accounts": len(ids), "n_chunks": len(chunks),
                                           "pid": None, "runtime_seconds": manifest["runtime_seconds"]})
    print("Q10_COMPLETE", ds, len(ids), manifest["runtime_seconds"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("hi_small", "li_small"), required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, default=2)
    arguments = parser.parse_args()
    assert 1 <= arguments.workers <= 2, "Coordinate any larger resource budget with the root runner."
    build(arguments.out, arguments.dataset, arguments.workers)
