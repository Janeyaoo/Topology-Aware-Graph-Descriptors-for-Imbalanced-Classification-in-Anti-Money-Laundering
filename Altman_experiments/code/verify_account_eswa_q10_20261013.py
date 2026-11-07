"""Independent Q10 acceptance: all-row identity checks and two count oracles.

The production bitset kernel is checked against (1) queue all-pairs BFS with
explicit triangle enumeration and (2) matrix-walk closure/trace counts. Neither
oracle invokes PH, persistent Laplacians, labels, model scores or training.
Only the new run's q10_validation.json is written; scientific inputs are read.
"""
from __future__ import annotations

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]
ALT = ROOT / "Altman_experiments"
SOURCE = ALT / "results/account_s10_increment_20261008"
DEFAULT_OUT = ALT / "results/account_eswa_supplement_20261013"
BUILDER = ALT / "code/build_account_eswa_q10_20261013.py"
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


def explicit_counts(adjacency):
    """Queue-BFS shortest-path metric and explicit unordered simplices."""
    n = len(adjacency)
    metric = np.full((n, n), n + 1, dtype=np.int64)
    for start in range(n):
        metric[start, start] = 0
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for neighbor in np.flatnonzero(adjacency[current]):
                if metric[start, neighbor] > metric[start, current] + 1:
                    metric[start, neighbor] = metric[start, current] + 1
                    queue.append(int(neighbor))
    result = []
    for epsilon in (1, 2):
        edges = sum(metric[i, j] <= epsilon for i, j in combinations(range(n), 2))
        triangles = sum(metric[i, j] <= epsilon and metric[i, k] <= epsilon
                        and metric[j, k] <= epsilon for i, j, k in combinations(range(n), 3))
        result.extend([int(edges), int(triangles)])
    return np.array(result, dtype=np.int64)


def matrix_counts(adjacency):
    """Independent walk-matrix closure and trace(A^3)/6."""
    a = adjacency.astype(np.int64)
    b = ((a + a @ a) > 0).astype(np.int64)
    np.fill_diagonal(b, 0)
    result = []
    for graph in (a, b):
        result.extend([int(graph.sum() // 2), int(np.trace(graph @ graph @ graph) // 6)])
    return np.array(result, dtype=np.int64)


def manual_graphs():
    examples = []
    def add(name, n, edges):
        a = np.zeros((n, n), dtype=bool)
        for i, j in edges:
            a[i, j] = a[j, i] = True
        examples.append((name, a))
    add("singleton", 1, [])
    add("single_edge", 2, [(0, 1)])
    add("path4", 4, [(0, 1), (1, 2), (2, 3)])
    add("triangle", 3, [(0, 1), (0, 2), (1, 2)])
    add("cycle4", 4, [(0, 1), (1, 2), (2, 3), (3, 0)])
    add("cycle5", 5, [(i, (i + 1) % 5) for i in range(5)])
    add("star40", 40, [(0, i) for i in range(1, 40)])
    add("path40", 40, [(i, i + 1) for i in range(39)])
    add("complete40", 40, list(combinations(range(40), 2)))
    add("disconnected_triangle_edge", 5, [(0, 1), (0, 2), (1, 2), (3, 4)])
    rng = np.random.default_rng(20261013)
    for n in (3, 7, 16, 40):
        for density in (0.05, 0.25, 0.65):
            a = np.triu(rng.random((n, n)) < density, 1)
            examples.append((f"random_n{n}_p{density}", a | a.T))
    return examples


def induced_reference(graph, vertices):
    """SciPy index slicing, independent from production CSR binary searches."""
    return graph[vertices][:, vertices].toarray().astype(bool)


def validate(out, ds):
    import build_account_eswa_q10_20261013 as production
    started = time.monotonic()
    folder = Path(out) / "inputs" / ds
    manifest_path = folder / "q10_input_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "complete"
    assert manifest["builder_sha256"] == sha(BUILDER)
    for key, expected in manifest["source_sha256"].items():
        assert sha(ROOT / key) == expected, f"Source mismatch: {key}"
    for key, expected in manifest["output_sha256"].items():
        assert sha(ROOT / key) == expected, f"Output mismatch: {key}"
    q10 = pd.read_parquet(folder / "q10.parquet")
    old = pd.read_parquet(SOURCE / ds / "features.parquet", columns=["account_id", *Q6])
    order = np.load(SOURCE / ds / "account_to_graph.npy", allow_pickle=False)
    graph_ids = pd.read_parquet(SOURCE / ds / "graph_accounts.parquet", columns=["account_id"])
    graph = sparse.load_npz(SOURCE / ds / "graph_csr.npz")
    assert list(q10.columns) == ["account_id", *Q10]
    assert len(q10) == len(old) == len(order) == manifest["n_accounts"]
    assert q10.account_id.is_unique and not q10.isna().any().any()
    assert np.array_equal(q10.account_id.to_numpy(), old.account_id.to_numpy())
    assert np.array_equal(q10.account_id.to_numpy(), graph_ids.account_id.to_numpy()[order])
    values = q10[Q10].to_numpy()
    assert values.dtype == np.int64 and np.isfinite(values).all()
    assert np.array_equal(values[:, [0, 5, 6, 7, 8, 9]], old[Q6].to_numpy())
    assert np.all(values[:, 0] <= values[:, 5])
    for offset in (0, 5):
        n, e1, t1, e2, t2 = values[:, offset:offset + 5].T
        assert np.all((n >= 1) & (n <= 40))
        assert np.all((e1 >= n - 1) & (e1 <= n * (n - 1) // 2))
        assert np.all((e2 >= e1) & (e2 <= n * (n - 1) // 2))
        assert np.all((t1 >= 0) & (t2 >= t1) & (t2 <= n * (n - 1) * (n - 2) // 6))
    same_size = values[:, 0] == values[:, 5]
    assert np.array_equal(values[same_size, :5], values[same_size, 5:])
    manual_results = []
    expected_hand = {"singleton": [0, 0, 0, 0], "single_edge": [1, 0, 1, 0],
                     "path4": [3, 0, 5, 2], "triangle": [3, 1, 3, 1],
                     "cycle4": [4, 0, 6, 4], "cycle5": [5, 0, 10, 10],
                     "star40": [39, 0, 780, 9880], "complete40": [780, 9880, 780, 9880]}
    for name, adjacency in manual_graphs():
        first = explicit_counts(adjacency)
        second = matrix_counts(adjacency)
        csr = sparse.csr_matrix(adjacency.astype(np.int8))
        fast = np.array(production.counts_from_vertices(np.arange(len(adjacency)),
                                                       len(adjacency), csr.indptr, csr.indices))
        assert np.array_equal(first, second) and np.array_equal(first, fast), name
        if name in expected_hand:
            assert first.tolist() == expected_hand[name], name
        manual_results.append({"name": name, "n": len(adjacency), "counts": first.tolist(), "passed": True})
    chunks = sorted((SOURCE / ds / "chunks").glob("*.npz"))
    rng = np.random.default_rng(20261013 + (0 if ds == "hi_small" else 1))
    sample_rows = set()
    for path in chunks:
        a, b = map(int, path.stem.split("_"))
        sample_rows.update([a, b - 1, int(rng.integers(a, b))])
    # Add diverse sizes/extreme simplex counts, independent of labels or scores.
    for column in range(10):
        for quantile in np.linspace(0, 1, 5):
            target = np.quantile(values[:, column], quantile)
            sample_rows.add(int(np.argmin(np.abs(values[:, column] - target))))
    sample_results = []
    verified_all_rows = 0
    for path in chunks:
        a, b = map(int, path.stem.split("_"))
        with np.load(path, allow_pickle=False) as source:
            vertices = source["vertices"]
            depths = source["depths"]
            source_q6 = source["values"][:, 12:18]
            assert np.array_equal(vertices[:, 0], order[a:b])
            assert np.array_equal(values[a:b, [0, 5, 6, 7, 8, 9]], source_q6)
            assert np.array_equal((vertices >= 0).sum(axis=1), values[a:b, 5])
            assert np.array_equal(((vertices >= 0) & (depths <= 2)).sum(axis=1), values[a:b, 0])
            assert np.all((vertices < 0) == (depths < 0))
            assert np.all((depths[vertices >= 0] >= 0) & (depths[vertices >= 0] <= 3))
            sorted_vertices = np.sort(vertices, axis=1)
            assert np.all((np.diff(sorted_vertices, axis=1) != 0) | (sorted_vertices[:, 1:] < 0))
            with np.load(folder / "q10_chunks" / path.name, allow_pickle=False) as output:
                assert np.array_equal(output["counts"], values[a:b])
            receipt = json.loads((folder / "q10_chunks" / path.with_suffix(".json").name).read_text())
            assert receipt["source_sha256"] == sha(path)
            assert receipt["builder_sha256"] == manifest["builder_sha256"]
            assert receipt["output_sha256"] == sha(folder / "q10_chunks" / path.name)
            verified_all_rows += b - a
            for row in sorted(sample_rows.intersection(range(a, b))):
                i = row - a
                for h, offset in ((2, 0), (3, 5)):
                    n = int(values[row, offset])
                    sampled = vertices[i, :n]
                    adjacency = induced_reference(graph, sampled)
                    first = explicit_counts(adjacency)
                    second = matrix_counts(adjacency)
                    actual = values[row, offset + 1:offset + 5]
                    assert np.array_equal(first, second) and np.array_equal(first, actual), (ds, row, h)
                    sample_results.append({"source_row": row, "radius": h, "n": n,
                                           "counts": actual.tolist(), "passed": True})
        print("Q10_VERIFY_CHUNK", ds, a, b, flush=True)
    assert verified_all_rows == len(values)
    # Repeat all source/output hashes after the complete validation pass.
    for key, expected in manifest["source_sha256"].items():
        assert sha(ROOT / key) == expected
    for key, expected in manifest["output_sha256"].items():
        assert sha(ROOT / key) == expected
    result = {"status": "passed", "dataset": ds, "n_accounts": len(values),
              "all_row_checks": {"account_id_exact_source_order": True,
                                 "original_q6_exact_equality": True,
                                 "storage_integer_finite_nonnegative": True,
                                 "connected_simplex_count_bounds": True,
                                 "frozen_sample_sizes_centers_and_unique_vertices": True,
                                 "same_vertex_set_same_counts": True,
                                 "all_chunks_and_receipts_bound": True,
                                 "old_sources_unchanged": True},
              "n_verified_full_rows": verified_all_rows, "n_chunks": len(chunks),
              "n_sampled_accounts": len(sample_rows), "n_sampled_radius_checks": len(sample_results),
              "reference_algorithms": ["queue_all_pairs_BFS_plus_explicit_combinations",
                                        "matrix_walk_closure_plus_trace_cube"],
              "manual_graph_results": manual_results,
              "sampled_count_checks": sample_results,
              "same_two_three_hop_vertex_set_fraction": float(same_size.mean()),
              "feature_unique_counts": {key: int(q10[key].nunique()) for key in Q10},
              "input_manifest_sha256": sha(manifest_path),
              "builder_sha256": sha(BUILDER), "validator_sha256": sha(Path(__file__)),
              "q10_sha256": sha(folder / "q10.parquet"),
              "runtime_seconds": time.monotonic() - started,
              "completed_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Independent acceptance of label-blind count inputs; no classifier result or independent new dataset evidence."}
    jsave(folder / "q10_validation.json", result)
    print("Q10_VALIDATION_PASSED", ds, len(values), len(sample_results), result["runtime_seconds"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("hi_small", "li_small"), required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    arguments = parser.parse_args()
    validate(arguments.out, arguments.dataset)
