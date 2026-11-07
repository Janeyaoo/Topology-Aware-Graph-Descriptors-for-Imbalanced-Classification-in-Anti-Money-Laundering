"""Independent PH, geometry and complete-spectrum checks for the new S10 cache.

The historical VR pipeline is never imported: four pure functions are extracted
with AST and executed without its top-level data loading or writes.  The oracle
uses explicit F2 barcode reduction, whereas the builder uses boundary ranks.
Signed dense boundary matrices independently check the production spectra.
Only this run's validation_features.json is written; scientific caches are read.
"""
from __future__ import annotations

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import ast
from collections import deque
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
from itertools import combinations
import json
from pathlib import Path
import re
import sys
import time

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]
ALT = ROOT / "Altman_experiments"
DEFAULT_OUT = ALT / "results/account_s10_increment_20261008"
BUILDER = ALT / "code/build_account_s10_features.py"
CORE = ALT / "code/account_s10_math.py"
ORACLE = ALT / "code/02_build_features_vr_persistence.py"
DATASETS = ("hi_small", "li_small")
GMKE = ["A1", "A2", "D1", "D2", "nv", "ev", "density", "cc_minus",
        "bridge_frag", "tri", "clust", "beta1", "beta1_norm", "core",
        "core_mean", "core_max", "rec_out_count", "rec_in_count",
        "rec_out_ratio", "rec_in_ratio", "peel_score", "kappa", "kappa_mean",
        "kappa_min", "asym", "asym2", "exp_out", "d_beta1"]
P_NAMES = [f"ph_{h}hop_h{q}_{s}" for h in (2, 3) for q in (0, 1)
           for s in ("max", "total", "entropy")]
Q_NAMES = ["q_n2", "q_n", "q_e1", "q_t1", "q_e2", "q_t2"]
OPS = ("g1", "g2", "h1", "h2", "pl12")
S_NAMES = [f"s_{op}_{s}" for op in OPS for s in ("min", "entropy")]
QC_NAMES = ["cap_hit", "effective_radius", "ph2_beta1_f2", "ph2_survive2_f2",
            "ph3_beta1_f2", "ph3_survive2_f2"] + [f"{op}_{k}" for op in OPS
                                                        for k in ("rank", "nullity")]
ZERO_TOL = 1e-9
NEGATIVE_TOL = 1e-7
PH_ATOL = 1e-9  # historical entropy contains log(p + 1e-12).
SPECTRUM_ATOL = 2e-7
FEATURE_ATOL = 2e-8
CAP = 40


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_functions(path, names, environment):
    """Execute only the enumerated pure function definitions from a source AST."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=str(path))
    found = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert set(names) <= set(found), f"Missing AST functions in {path}"
    selected = ast.Module(body=[found[name] for name in names], type_ignores=[])
    context = dict(environment)
    exec(compile(ast.fix_missing_locations(selected), str(path), "exec"), context)
    return context


def load_references():
    old = load_functions(ORACLE,
        ("shortest_path_metric", "persistence_entropy", "summarize_persistence",
         "vr_persistence_h0_h1"), {"np": np, "deque": deque,
                                   "combinations": combinations})
    fast = load_functions(BUILDER, ("triangle_bits", "gf2_rank", "ph_summaries"),
                          {"np": np})
    spec = importlib.util.spec_from_file_location("s10_core_under_test", CORE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclass annotation introspection.
    spec.loader.exec_module(module)
    assert list(module.FEATURE_NAMES) == S_NAMES
    assert module.ZERO_TOL == ZERO_TOL and module.NEGATIVE_TOL == NEGATIVE_TOL
    return old, fast, module


def adjacency(n, edges):
    out = np.zeros((n, n), dtype=bool)
    for a, b in edges:
        assert a != b
        out[a, b] = out[b, a] = True
    return out


def distances(a):
    """Independent all-pairs queue BFS, without production matrix squaring."""
    n = len(a)
    d = np.full((n, n), n + 1, dtype=np.int32)
    for start in range(n):
        d[start, start] = 0
        todo = deque([start])
        while todo:
            u = todo.popleft()
            for w in np.flatnonzero(a[u]):
                if d[start, w] > n:
                    d[start, w] = d[start, u] + 1
                    todo.append(int(w))
    assert (d <= n).all(), "Graph is not connected"
    return d


def epsilon2(a):
    b = distances(a) <= 2
    np.fill_diagonal(b, False)
    return b


def explicit_ph(a, old):
    edges = set(zip(*np.nonzero(np.triu(a, 1))))
    diagram = old["vr_persistence_h0_h1"](set(range(len(a))), edges)
    vector = np.asarray([value for dim in (0, 1)
                         for value in old["summarize_persistence"](diagram, dim)])
    bars = [(int(q), float(b), float(d)) for q, b, d in diagram
            if q <= 1 and np.isfinite(d) and d > b]
    beta = sum(q == 1 and b <= 1 < d for q, b, d in bars)
    survive = sum(q == 1 and b <= 1 < 2 < d for q, b, d in bars)
    assert all(b == 1 and d in (2, 3) for q, b, d in bars if q == 1)
    return vector, (beta, survive), bars


def signed_boundaries(a):
    """Explicit oriented simplex columns; no production Gram assembly helpers."""
    n = len(a)
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if a[i, j]]
    index = {edge: k for k, edge in enumerate(edges)}
    triangles = [(i, j, k) for i, j, k in combinations(range(n), 3)
                 if a[i, j] and a[i, k] and a[j, k]]
    b1 = np.zeros((n, len(edges)), dtype=float)
    b2 = np.zeros((len(edges), len(triangles)), dtype=float)
    for k, (i, j) in enumerate(edges):
        b1[i, k], b1[j, k] = -1., 1.
    for k, (i, j, l) in enumerate(triangles):
        b2[index[(j, l)], k] = 1.
        b2[index[(i, l)], k] = -1.
        b2[index[(i, j)], k] = 1.
    np.testing.assert_allclose(b1 @ b2, 0, atol=1e-12, rtol=0)
    return b1, b2, edges, triangles


def clean_eigenvalues(matrix):
    assert matrix.shape[0] == matrix.shape[1]
    if not len(matrix):
        return np.empty(0)
    np.testing.assert_allclose(matrix, matrix.T, atol=1e-8, rtol=0)
    values = np.linalg.eigvalsh((matrix + matrix.T) / 2)
    assert float(values.min()) >= -NEGATIVE_TOL, "Independent PSD violation"
    values[values <= ZERO_TOL] = 0.
    return values


def independent_spectra(a):
    """Dense L0/L1 plus persistent Schur operator, without fast-case identities."""
    target = epsilon2(a)
    b1s, b2s, old_edges, old_triangles = signed_boundaries(a)
    b1t, b2t, target_edges, target_triangles = signed_boundaries(target)
    old_index = {edge: i for i, edge in enumerate(old_edges)}
    old_rows = [target_edges.index(edge) for edge in old_edges]
    new_rows = [i for i, edge in enumerate(target_edges) if edge not in old_index]
    x = b2t[old_rows, :]
    y = b2t[new_rows, :]
    upper = x @ x.T
    if len(new_rows):
        yy = y @ y.T
        vals, vecs = np.linalg.eigh(yy)
        assert np.min(vals, initial=0) >= -NEGATIVE_TOL
        keep = vals > ZERO_TOL
        cross = x @ y.T
        if (~keep).any():
            assert np.max(np.abs(cross @ vecs[:, ~keep]), initial=0) < 1e-6
        if keep.any():
            inverse = (vecs[:, keep] / vals[keep]) @ vecs[:, keep].T
            upper = upper - cross @ inverse @ cross.T
    upper = (upper + upper.T) / 2
    assert np.max(np.abs(b1s @ upper), initial=0) < 1e-6
    mats = {"g1": b1s @ b1s.T, "g2": b1t @ b1t.T,
            "h1": b1s.T @ b1s + b2s @ b2s.T,
            "h2": b1t.T @ b1t + b2t @ b2t.T,
            "pl12": b1s.T @ b1s + upper}
    spectra = {op: clean_eigenvalues(mat) for op, mat in mats.items()}
    counts = [len(old_edges), len(old_triangles), len(target_edges), len(target_triangles)]
    return spectra, counts


def summarize_spectra(spectra, n):
    values, qc = [], {}
    for op in OPS:
        whole = spectra[op]
        positive = whole[whole > ZERO_TOL]
        rank = len(positive)
        minimum = float(positive.min()) if rank else 0.
        entropy = 0.
        if rank > 1:
            probability = positive / positive.sum()
            entropy = float(-sum(p * np.log(p) for p in probability) / np.log(rank))
        values += [minimum / n, entropy]
        qc[op] = {"dimension": len(whole), "rank": rank,
                  "nullity": len(whole) - rank, "trace": float(positive.sum()),
                  "min_positive": minimum}
    return np.asarray(values), qc


def compare_spectra(a, core):
    reference, counts = independent_spectra(a)
    vector, qc = summarize_spectra(reference, len(a))
    got = core.compute_s10(a, return_spectra=True)
    worst = 0.
    for op in OPS:
        np.testing.assert_allclose(got["spectra"][op], reference[op],
                                   atol=SPECTRUM_ATOL, rtol=1e-8)
        worst = max(worst, float(np.max(np.abs(got["spectra"][op] - reference[op]), initial=0)))
        actual = got["qc"]["operators"][op]
        for key in ("dimension", "rank", "nullity"):
            assert actual[key] == qc[op][key], (op, key, actual[key], qc[op][key])
        for key in ("trace", "min_positive"):
            np.testing.assert_allclose(actual[key], qc[op][key], atol=SPECTRUM_ATOL, rtol=1e-8)
    np.testing.assert_allclose(got["values"], vector, atol=FEATURE_ATOL, rtol=1e-8)
    return vector, qc, counts, worst


def rooted_order(a):
    depths = distances(a)[0]
    order = sorted(range(len(a)), key=lambda i: (int(depths[i]), i))
    return a[np.ix_(order, order)], depths[order]


def example_graphs(random_cases, seed):
    graphs = [("isolated", np.zeros((1, 1), bool)),
              ("edge", adjacency(2, [(0, 1)])),
              ("complete5", np.ones((5, 5), bool) ^ np.eye(5, dtype=bool))]
    for n in (4, 5, 6, 7):
        graphs.append((f"cycle{n}", adjacency(n, [(i, (i + 1) % n) for i in range(n)])))
    # Two cycles share only the root: C4 dies at 2, C7 dies at 3.
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (4, 5),
             (5, 6), (6, 7), (7, 8), (8, 9), (9, 0)]
    graphs.append(("C4_C7_shared_root", adjacency(10, edges)))
    graphs.append(("three_layer_tree", adjacency(10,
        [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (3, 6), (3, 7), (5, 8), (5, 9)])))
    rng = np.random.default_rng(seed)
    for trial in range(random_cases):
        n = int(rng.integers(2, 19))
        a = np.zeros((n, n), bool)
        depth = [0]
        for v in range(1, n):
            parents = [i for i, d in enumerate(depth) if d < 3]
            u = int(rng.choice(parents))
            a[v, u] = a[u, v] = True
            depth.append(depth[u] + 1)
        probability = float(rng.uniform(0, .55))
        for i, j in combinations(range(n), 2):
            if rng.random() < probability:
                a[i, j] = a[j, i] = True
        graphs.append((f"random_{trial:04d}", a))
    return graphs


def verify_math(random_cases, seed, old, fast, core):
    assert random_cases >= 100, "At least 100 randomized graphs are required"
    start = time.monotonic()
    ph_cases = spectrum_cases = 0
    max_ph_error = max_spectrum_error = 0.
    named = []
    for name, initial in example_graphs(random_cases, seed):
        a, depth = rooted_order(initial)
        assert int(depth.max()) <= 3
        small = a[:int((depth <= 2).sum()), :int((depth <= 2).sum())]
        for radius, value in ((3, a), (2, small)):
            reference, ranks, bars = explicit_ph(value, old)
            result, shortcut_ranks = fast["ph_summaries"](value, epsilon2(value), radius)
            np.testing.assert_allclose(result, reference, atol=PH_ATOL, rtol=0)
            assert tuple(shortcut_ranks) == ranks
            if radius == 2:
                assert ranks[1] == 0
            error = float(np.max(np.abs(result - reference), initial=0))
            max_ph_error = max(max_ph_error, error)
            ph_cases += 1
            if not name.startswith("random_"):
                named.append({"name": name, "radius": radius, "n": len(value),
                              "finite_bars": bars, "beta1": ranks[0],
                              "survive_at_two": ranks[1], "max_abs_error": error})
        _, _, _, error = compare_spectra(a, core)
        max_spectrum_error = max(max_spectrum_error, error)
        spectrum_cases += 1
    mixed = next(row for row in named if row["name"] == "C4_C7_shared_root" and row["radius"] == 3)
    assert mixed["beta1"] == 2 and mixed["survive_at_two"] == 1
    return {"passed": True, "random_seed": seed, "random_graphs": random_cases,
            "explicit_barcode_comparisons": ph_cases,
            "independent_complete_spectrum_comparisons": spectrum_cases,
            "max_abs_ph_summary_error": max_ph_error,
            "max_abs_spectrum_error": max_spectrum_error,
            "named_cases": named, "runtime_seconds": time.monotonic() - start,
            "oracle_method": "Historical pure functions loaded by AST: full simplex enumeration and F2 column reduction",
            "spectral_oracle": "Explicit signed dense B1/B2; full eigvalsh and direct generalized Schur operator; no production fast cases",
            "coefficient_fields": {"barcodes": "F2", "spectra": "R"}}


def verify_bound_hashes(records, root):
    checked = {}
    for name, digest in records.items():
        path = root / name
        assert path.is_file() and sha(path) == digest, f"Hash mismatch: {path}"
        checked[str(path.resolve())] = digest
    return checked


def independent_prefix(root, graph):
    """Layer-set BFS selection, independently of the builder's heap merge."""
    seen = {int(root)}
    selected, depth = [int(root)], [0]
    frontier = {int(root)}
    hit = 0
    for radius in range(1, 4):
        candidates = set()
        for u in frontier:
            candidates.update(map(int, graph.indices[graph.indptr[u]:graph.indptr[u + 1]]))
        candidates -= seen
        ordered = sorted(candidates)
        remaining = CAP - len(selected)
        chosen = ordered[:remaining]
        selected.extend(chosen)
        depth.extend([radius] * len(chosen))
        seen.update(chosen)
        if len(ordered) > remaining:
            hit = 1
            break
        if len(selected) == CAP:
            if radius < 3:
                hit = int(any(int(w) not in seen for u in chosen
                              for w in graph.indices[graph.indptr[u]:graph.indptr[u + 1]]))
            break
        frontier = set(chosen)
        if not frontier:
            break
    return np.asarray(selected, np.int32), np.asarray(depth, np.int8), hit


def verify_geometry_inputs(out, dataset):
    folder = out / dataset
    graph_manifest = json.loads((folder / "graph_manifest.json").read_text())
    feature_manifest = json.loads((folder / "feature_manifest.json").read_text())
    assert feature_manifest["status"] == "complete"
    verify_bound_hashes(graph_manifest["input_sha256"], ROOT)
    verify_bound_hashes(graph_manifest["artifact_sha256"], folder)
    verify_bound_hashes(feature_manifest["input_sha256"], ROOT)
    verify_bound_hashes(feature_manifest["artifacts_sha256"], folder)
    source_dir = ALT / "results" / dataset / "features"
    full = pd.read_parquet(source_dir / "full_features.parquet")
    labels_path = ALT / "data" / dataset / ("account_labels.parquet" if dataset == "hi_small" else "li_small_labels.parquet")
    labels = pd.read_parquet(labels_path)
    np.testing.assert_array_equal(full.account_id, labels.account_id)
    np.testing.assert_array_equal(full.label, labels.label)
    expected = (515088, 6357) if dataset == "hi_small" else (705907, 5304)
    assert (len(full), int(full.label.sum())) == expected and full.account_id.is_unique
    ids = pd.Index(sorted(full.account_id))
    assert list(ids) == sorted(ids, key=str), "Graph indices do not implement lexicographic IDs"
    registry = pd.read_parquet(folder / "graph_accounts.parquet")
    np.testing.assert_array_equal(registry.graph_index, np.arange(len(ids)))
    np.testing.assert_array_equal(registry.account_id, ids)
    mapping = np.load(folder / "account_to_graph.npy")
    np.testing.assert_array_equal(mapping, ids.get_indexer(full.account_id))
    graph = sparse.load_npz(folder / "graph_csr.npz").tocsr()
    assert graph.shape == (len(ids), len(ids)) and graph.has_sorted_indices
    assert np.all(graph.data == 1) and not graph.diagonal().any()
    assert (graph != graph.T).nnz == 0
    trans_path = ALT / "data" / dataset / f"{dataset}_trans.parquet"
    trans = pd.read_parquet(trans_path, columns=["from_id", "to_id", "is_laundering"])
    left, right = ids.get_indexer(trans.from_id), ids.get_indexer(trans.to_id)
    assert (left >= 0).all() and (right >= 0).all()
    # Boolean COO reconstruction avoids the builder's integer duplicate sum.
    keep = left != right
    edges_left, edges_right = left[keep], right[keep]
    rebuilt = sparse.coo_matrix((np.ones(2 * keep.sum(), dtype=bool),
        (np.r_[edges_left, edges_right], np.r_[edges_right, edges_left])),
        shape=graph.shape).tocsr()
    rebuilt.sort_indices()
    np.testing.assert_array_equal(graph.indptr, rebuilt.indptr)
    np.testing.assert_array_equal(graph.indices, rebuilt.indices)
    assert graph.nnz // 2 == graph_manifest["n_undirected_edges"]
    laundering = trans.is_laundering.to_numpy() == 1
    raw_label = np.zeros(len(ids), np.int8)
    raw_label[np.r_[left[laundering], right[laundering]]] = 1
    np.testing.assert_array_equal(raw_label[mapping], full.label.to_numpy())
    del rebuilt, trans, left, right, edges_left, edges_right, raw_label
    gc.collect()
    features = pd.read_parquet(folder / "features.parquet")
    base = pd.read_parquet(source_dir / "base_features.parquet")
    base_names = [c for c in base.columns if c != "account_id"]
    assert len(base_names) == 16
    expected_names = ["account_id", "label", *base_names, *GMKE, *P_NAMES, *Q_NAMES, *S_NAMES]
    assert list(features.columns) == expected_names and features.columns.is_unique
    assert feature_manifest["feature_columns"] == expected_names[2:]
    assert feature_manifest["n_features"] == 72
    assert feature_manifest["n_accounts"] == len(features) == len(full)
    assert feature_manifest["n_positive"] == int(full.label.sum())
    np.testing.assert_array_equal(features.account_id, full.account_id)
    np.testing.assert_array_equal(features.label, full.label)
    np.testing.assert_array_equal(base.account_id, full.account_id)
    for col in base_names + GMKE:
        np.testing.assert_array_equal(features[col], full[col])
    for col in base_names:
        np.testing.assert_array_equal(features[col], base[col])
    for col in features.columns[1:]:
        assert np.isfinite(features[col].to_numpy()).all(), f"Nonfinite column {col}"
    qc = pd.read_parquet(folder / "geometry_qc.parquet")
    assert list(qc.columns) == ["account_id", *QC_NAMES]
    np.testing.assert_array_equal(qc.account_id, full.account_id)
    paths = [source_dir / "full_features.parquet", source_dir / "base_features.parquet",
             source_dir / "ftd_features.parquet", trans_path, labels_path]
    del full, base
    gc.collect()
    return features, qc, graph, mapping, paths


def chunks_inventory(folder, n):
    entries = []
    next_row = 0
    for path in sorted((folder / "chunks").glob("*.npz")):
        match = re.fullmatch(r"(\d+)_(\d+)\.npz", path.name)
        assert match, f"Unexpected chunk filename: {path.name}"
        start, stop = map(int, match.groups())
        assert start == next_row and start < stop <= n
        with np.load(path) as data:
            assert set(data.files) == {"values", "qc", "vertices", "depths"}
            assert data["values"].shape == (stop - start, 28)
            assert data["qc"].shape == (stop - start, 16)
            assert data["vertices"].shape == data["depths"].shape == (stop - start, CAP)
            assert np.isfinite(data["values"]).all()
        report_path = path.with_suffix(".json")
        report = json.loads(report_path.read_text())
        assert (report["start"], report["stop"]) == (start, stop)
        assert report["sha256"] == sha(path)
        entries.append((start, stop, path))
        next_row = stop
    assert next_row == n, "Incomplete or overlapping archived chunks"
    return entries


def verify_dataset(out, dataset, sample_count, seed, old, fast, core):
    assert sample_count >= 50, "At least 50 uniform rows per dataset are required"
    started = time.monotonic()
    features, qc, graph, mapping, input_paths = verify_geometry_inputs(out, dataset)
    n = len(features)
    folder = out / dataset
    inventory = chunks_inventory(folder, n)
    selected = np.sort(np.random.default_rng(seed).choice(n, sample_count, replace=False))
    sample_rows = []
    maximum_ph = maximum_feature = maximum_spectrum = 0.
    sampled_compute_seconds = 0.
    archive_columns = P_NAMES + Q_NAMES + S_NAMES
    for start, stop, path in inventory:
        # Whole-array archive-to-final binding, independently of assembly metadata.
        with np.load(path) as chunk:
            np.testing.assert_array_equal(chunk["values"], features.iloc[start:stop][archive_columns].to_numpy())
            np.testing.assert_array_equal(chunk["qc"], qc.iloc[start:stop][QC_NAMES].to_numpy())
            for row in selected[(selected >= start) & (selected < stop)]:
                tick = time.monotonic()
                position = int(row - start)
                vertices, depth, hit = independent_prefix(mapping[row], graph)
                stored = chunk["vertices"][position]
                stored_depth = chunk["depths"][position]
                np.testing.assert_array_equal(stored[:len(vertices)], vertices)
                np.testing.assert_array_equal(stored_depth[:len(vertices)], depth)
                assert np.all(stored[len(vertices):] == -1)
                assert np.all(stored_depth[len(vertices):] == -1)
                assert vertices[0] == mapping[row] and len(set(vertices)) == len(vertices)
                a = graph[vertices, :][:, vertices].toarray().astype(bool)
                local_distance = distances(a)[0]
                np.testing.assert_array_equal(local_distance, depth)
                n2 = int((depth <= 2).sum())
                small = a[:n2, :n2]
                ph2, ranks2, _ = explicit_ph(small, old)
                ph3, ranks3, _ = explicit_ph(a, old)
                assert ranks2[1] == 0 and int(depth.max()) <= 3
                np.testing.assert_allclose(fast["ph_summaries"](small, epsilon2(small), 2)[0], ph2,
                                           atol=PH_ATOL, rtol=0)
                np.testing.assert_allclose(fast["ph_summaries"](a, epsilon2(a), 3)[0], ph3,
                                           atol=PH_ATOL, rtol=0)
                spectral, operator_qc, counts, spectrum_error = compare_spectra(a, core)
                wanted = np.r_[ph2, ph3, n2, len(a), counts, spectral]
                actual = features.iloc[row][archive_columns].to_numpy(dtype=float)
                np.testing.assert_allclose(actual[:12], wanted[:12], atol=PH_ATOL, rtol=0)
                np.testing.assert_array_equal(actual[12:18], wanted[12:18])
                np.testing.assert_allclose(actual[18:], wanted[18:], atol=FEATURE_ATOL, rtol=1e-8)
                desired_qc = [hit, int(depth.max()), *ranks2, *ranks3]
                for op in OPS:
                    desired_qc += [operator_qc[op]["rank"], operator_qc[op]["nullity"]]
                np.testing.assert_array_equal(qc.iloc[row][QC_NAMES].to_numpy(dtype=int), desired_qc)
                ph_error = float(np.max(np.abs(actual[:12] - wanted[:12]), initial=0))
                feature_error = float(np.max(np.abs(actual[18:] - wanted[18:]), initial=0))
                maximum_ph = max(maximum_ph, ph_error)
                maximum_feature = max(maximum_feature, feature_error)
                maximum_spectrum = max(maximum_spectrum, spectrum_error)
                sampled_compute_seconds += time.monotonic() - tick
                sample_rows.append({"row_index": int(row), "account_id": str(features.iloc[row].account_id),
                    "n": len(vertices), "n2": n2, "cap_hit": hit,
                    "effective_radius": int(depth.max()), "ph2_beta1_f2": ranks2[0],
                    "ph3_beta1_f2": ranks3[0], "ph3_survive2_f2": ranks3[1],
                    "max_abs_ph_error": ph_error, "max_abs_feature_error": feature_error,
                    "max_abs_spectrum_error": spectrum_error})
        print(f"CHECKED {dataset} chunk {start}:{stop}; sampled {len(sample_rows)}/{sample_count}", flush=True)
    assert len(sample_rows) == sample_count
    feature_qc = json.loads((folder / "feature_qc.json").read_text())
    assert feature_qc["n_accounts"] == n
    np.testing.assert_allclose(feature_qc["cap_hit_fraction"], qc.cap_hit.mean(), atol=1e-15, rtol=0)
    assert feature_qc["radius_counts"] == {str(int(k)): int(v) for k, v in qc.effective_radius.value_counts().sort_index().items()}
    source_discrepancy = int((qc.ph3_beta1_f2 != qc.h1_nullity).sum())
    persistent_discrepancy = int((qc.ph3_survive2_f2 != qc.pl12_nullity).sum())
    assert feature_qc["field_discrepancy_source_count"] == source_discrepancy
    assert feature_qc["field_discrepancy_persistent_count"] == persistent_discrepancy
    for col in archive_columns:
        assert feature_qc["feature_unique_counts"][col] == int(features[col].nunique())
    equality = float(np.all(features[P_NAMES[:6]].to_numpy() == features[P_NAMES[6:]].to_numpy(), axis=1).mean())
    np.testing.assert_allclose(feature_qc["ph2_equals_ph3_fraction"], equality, atol=1e-15, rtol=0)
    all_duplicates = []
    arrays = {col: features[col].to_numpy() for col in archive_columns}
    for i, col in enumerate(archive_columns):
        for earlier in archive_columns[:i]:
            if np.allclose(arrays[col], arrays[earlier], atol=1e-10, rtol=0):
                all_duplicates.append([earlier, col])
    assert feature_qc["full_column_equalities_atol1e_10"] == all_duplicates
    output_paths = [folder / name for name in ("features.parquet", "geometry_qc.parquet",
        "feature_manifest.json", "feature_qc.json", "graph_manifest.json", "graph_csr.npz",
        "graph_accounts.parquet", "account_to_graph.npy")]
    for _, _, path in inventory:
        output_paths += [path, path.with_suffix(".json")]
    result = {"passed": True, "n_accounts": n, "n_sampled_accounts_verified": sample_count,
              "sampling": "Uniform rows without replacement; independent fixed RNG; no label stratification",
              "random_seed": seed, "all_graph_edges_rebuilt_from_raw_transactions": True,
              "all_labels_rebuilt_as_any_incident_laundering_transaction": True,
              "all_base_gmke_values_compared": True, "all_chunk_values_and_qc_compared": True,
              "n_archived_chunks": len(inventory), "sample_rows": sample_rows,
              "max_abs_ph_error": maximum_ph, "max_abs_feature_error": maximum_feature,
              "max_abs_spectrum_error": maximum_spectrum,
              "field_discrepancy_source_count": source_discrepancy,
              "field_discrepancy_persistent_count": persistent_discrepancy,
              "runtime_seconds": time.monotonic() - started,
              "sampled_oracle_compute_seconds": sampled_compute_seconds,
              "estimated_full_dataset_oracle_seconds": sampled_compute_seconds / sample_count * n,
              "cost_estimate_scope": "Linear extrapolation of independent PH+full-spectrum oracle cost from uniform rows; excludes geometry rebuild, I/O and production cache reuse; not a production feature-builder walltime estimate"}
    del features, qc, graph, mapping, arrays
    gc.collect()
    return result, input_paths, output_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("math", "data", "all"), required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--random-cases", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20261008)
    args = parser.parse_args()
    out = args.out.resolve()
    assert out.is_relative_to(ALT / "results"), "Validation writes must stay in the new experiment results tree"
    assert out != ALT / "results" and (out / "feature_protocol.json").is_file()
    target = out / "validation_features.json"
    previous = json.loads(target.read_text()) if target.exists() else {}
    bindings = {relative(path): sha(path) for path in (Path(__file__), BUILDER, CORE, ORACLE)}
    result = {"passed": False, "status": "running", "phase": args.phase,
              "validator_sha256": sha(__file__), "source_hashes": bindings,
              "result_hashes": {"feature_protocol.json": sha(out / "feature_protocol.json")},
              "created_at": datetime.now(timezone.utc).isoformat(),
              "tolerances": {"zero": ZERO_TOL, "negative": NEGATIVE_TOL,
                  "ph_atol": PH_ATOL, "complete_spectrum_atol": SPECTRUM_ATOL,
                  "spectral_scalar_atol": FEATURE_ATOL}, "datasets": {}}
    try:
        protocol = json.loads((out / "feature_protocol.json").read_text())
        assert protocol["version"] == "S10_matched_cap40_v1"
        assert protocol["node_cap"] == 40 and protocol["radius"] == 3
        assert protocol["geometry_uses_labels"] is False and protocol["controls"] == Q_NAMES
        old, fast, core = load_references()
        if args.phase in ("math", "all"):
            result["math"] = verify_math(args.random_cases, args.seed, old, fast, core)
        else:
            assert previous.get("math", {}).get("passed") is True, "Run --phase math first"
            for key, digest in bindings.items():
                assert previous.get("source_hashes", {}).get(key) == digest, f"Math validation binding changed: {key}"
            assert previous["result_hashes"]["feature_protocol.json"] == result["result_hashes"]["feature_protocol.json"]
            result["math"] = previous["math"]
            for dataset, section in previous.get("datasets", {}).items():
                if section.get("passed"):
                    for name, digest in previous["result_hashes"].items():
                        if name.startswith(dataset + "/"):
                            assert sha(out / name) == digest, f"Previously checked artifact changed: {name}"
                            result["result_hashes"][name] = digest
                    result["datasets"][dataset] = section
            for name, digest in previous.get("source_hashes", {}).items():
                assert sha(ROOT / name) == digest, f"Previously checked input changed: {name}"
                result["source_hashes"][name] = digest
        if args.phase in ("data", "all"):
            datasets = DATASETS if args.dataset == "all" else (args.dataset,)
            for dataset in datasets:
                section, inputs, outputs = verify_dataset(out, dataset, args.samples,
                    args.seed + DATASETS.index(dataset), old, fast, core)
                result["datasets"][dataset] = section
                result["source_hashes"].update({relative(path): sha(path) for path in inputs})
                result["result_hashes"].update({str(path.relative_to(out)): sha(path) for path in outputs})
        result["n_sampled_accounts_verified"] = sum(item["n_sampled_accounts_verified"]
                                                    for item in result["datasets"].values())
        result["passed"] = result["math"]["passed"] and all(
            result["datasets"].get(ds, {}).get("passed") is True for ds in DATASETS)
        result["status"] = "passed" if result["passed"] else "math_passed_data_pending"
        for path, digest in bindings.items():
            assert sha(ROOT / path) == digest, f"Validation source changed during execution: {path}"
        result["completed_at"] = datetime.now(timezone.utc).isoformat()
        write_json(target, result)
        print(json.dumps({"status": result["status"], "passed": result["passed"],
            "math_barcode_comparisons": result["math"]["explicit_barcode_comparisons"],
            "n_sampled_accounts_verified": result["n_sampled_accounts_verified"],
            "validation_path": relative(target)}, ensure_ascii=False), flush=True)
    except BaseException as error:
        result.update(passed=False, status="failed", error=f"{type(error).__name__}: {error}",
                      failed_at=datetime.now(timezone.utc).isoformat())
        write_json(target, result)
        raise


if __name__ == "__main__":
    main()
