"""Independent meaningful mathematical checks for account_s10_math.

No project data, model training, external packages beyond NumPy, or file writes.
Direct full matrices and an independently restricted-chain SVD construction are
used to check the production Schur and spectral-union algorithms.  --benchmark
also times connected n=40 sparse examples with all positive spectra retained.
"""
from __future__ import annotations

import argparse
import hashlib
from itertools import combinations
import json
from pathlib import Path
import time

import numpy as np

import account_s10_math as s10


def adjacency(n, edges):
    a = np.zeros((n, n), dtype=bool)
    for u, v in edges:
        a[u, v] = a[v, u] = True
    return a


def boundaries(a):
    """Independent oriented chain matrices, intentionally not using core helpers."""
    n = len(a)
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if a[i, j]]
    edge_idx = {e: i for i, e in enumerate(edges)}
    triangles = [(i, j, k) for i, j, k in combinations(range(n), 3)
                 if a[i, j] and a[j, k] and a[i, k]]
    b1 = np.zeros((n, len(edges)))
    b2 = np.zeros((len(edges), len(triangles)))
    for column, (i, j) in enumerate(edges):
        b1[i, column] = -1; b1[j, column] = 1
    for column, (i, j, k) in enumerate(triangles):
        for e, sign in (((i, j), 1), ((j, k), 1), ((i, k), -1)):
            b2[edge_idx[e], column] = sign
    return edges, b1, b2


def distance_two(a):
    """Independent BFS distance<=2 construction."""
    result = a.copy()
    for u in range(len(a)):
        for v in np.flatnonzero(a[u]):
            result[u] |= a[v]
    np.fill_diagonal(result, False)
    return result


def clean(values):
    out = np.sort(values)
    if len(out):
        assert out.min() > -1e-7
    out[np.abs(out) < 1e-8] = 0
    return out


def reference(a, target):
    old_edges, old_b1, old_b2 = boundaries(a)
    new_edges, target_b1, target_b2 = boundaries(target)
    old_rows = [new_edges.index(e) for e in old_edges]
    new_rows = [i for i, e in enumerate(new_edges) if e not in old_edges]
    # Restrict C2(target) to chains whose new-edge boundary coordinates vanish.
    # This can contain combinations of individually ineligible triangles.
    forbidden_boundary = target_b2[new_rows, :]
    if len(new_rows) and target_b2.shape[1]:
        _, sigma, vt = np.linalg.svd(forbidden_boundary, full_matrices=True)
        rank = int((sigma > 1e-8).sum())
        z = vt[rank:, :].T
    else:
        z = np.eye(target_b2.shape[1])
    allowed_boundary = target_b2[old_rows, :] @ z
    persistent = old_b1.T @ old_b1 + allowed_boundary @ allowed_boundary.T
    return {
        "g1": clean(np.linalg.eigvalsh(old_b1 @ old_b1.T)),
        "g2": clean(np.linalg.eigvalsh(target_b1 @ target_b1.T)),
        "h1": clean(np.linalg.eigvalsh(old_b1.T @ old_b1 + old_b2 @ old_b2.T)),
        "h2": clean(np.linalg.eigvalsh(target_b1.T @ target_b1 + target_b2 @ target_b2.T)),
        "pl12": clean(np.linalg.eigvalsh(persistent)),
        "allowed_boundary": allowed_boundary,
    }


def normalized_features(eigenvalues, n):
    positive = eigenvalues[eigenvalues > 1e-8]
    minimum = float(positive.min() / n) if len(positive) else 0.0
    if len(positive) < 2:
        entropy = 0.0
    else:
        p = positive / positive.sum()
        entropy = float(-(p * np.log(p)).sum() / np.log(len(p)))
    return minimum, entropy


def verify():
    cases = {
        "single_vertex": adjacency(1, []),
        "single_edge": adjacency(2, [(0, 1)]),
        "path4": adjacency(4, [(0, 1), (1, 2), (2, 3)]),
        "star5": adjacency(5, [(0, i) for i in range(1, 5)]),
        "cycle4": adjacency(4, [(0, 1), (1, 2), (2, 3), (3, 0)]),
        "cycle5": adjacency(5, [(i, (i + 1) % 5) for i in range(5)]),
        "cycle6": adjacency(6, [(i, (i + 1) % 6) for i in range(6)]),
        "triangle_pendant": adjacency(4, [(0, 1), (1, 2), (0, 2), (2, 3)]),
        "complete4": adjacency(4, combinations(range(4), 2)),
        "radius2_noncomplete_target": adjacency(7, [(0, 1), (0, 2), (0, 3),
            (1, 4), (2, 4), (2, 5), (3, 5), (1, 6)]),
    }
    rng = np.random.default_rng(8112026)
    for trial in range(24):
        n = 6
        a = adjacency(n, [(i, i + 1) for i in range(n - 1)])
        for i, j in combinations(range(n), 2):
            if rng.random() < (0.08 + 0.025 * trial):
                a[i, j] = a[j, i] = True
        cases[f"random6_{trial}"] = a
    checks = 0
    max_spectral_difference = 0.0
    max_feature_difference = 0.0
    saw_nonshortcut_schur = False
    for name, a in cases.items():
        frozen = a.copy()
        target = distance_two(a)
        assert np.array_equal(target, s10.vr_epsilon2(a)); checks += 1
        actual = s10.compute_s10(a, return_spectra=True)
        assert np.array_equal(frozen, a); checks += 1
        expected = reference(a, target)
        for op in s10.S10_OPERATORS:
            left, right = actual["spectra"][op], expected[op]
            assert left.shape == right.shape, (name, op, left, right)
            difference = float(np.max(np.abs(left - right), initial=0))
            max_spectral_difference = max(max_spectral_difference, difference)
            assert np.allclose(left, right, atol=1e-8, rtol=1e-8), (name, op, left, right)
            assert np.all(left >= 0) and np.all(left <= len(a) + 1e-8)
            qc = actual["qc"]["operators"][op]
            assert qc["dimension"] == len(left)
            assert qc["rank"] + qc["nullity"] == len(left)
            minimum, entropy = normalized_features(right, len(a))
            for feature, value in ((f"s_{op}_min", minimum), (f"s_{op}_entropy", entropy)):
                diff = abs(actual["features"][feature] - value)
                max_feature_difference = max(max_feature_difference, diff)
                assert diff < 1e-8, (name, feature, diff)
            checks += 7
        for matrix_a in (a, target):
            core_b = s10.oriented_boundaries(matrix_a)
            _, b1, b2 = boundaries(matrix_a)
            assert np.array_equal(core_b["b1"], b1)
            assert np.array_equal(core_b["b2"], b2)
            assert np.max(np.abs(b1 @ b2), initial=0) == 0
            # Consistent reorientation changes matrices by orthogonal similarity,
            # leaving both Laplacian spectra invariant.
            signs1 = rng.choice([-1., 1.], size=b1.shape[1])
            signs2 = rng.choice([-1., 1.], size=b2.shape[1])
            ob1, ob2 = b1 * signs1, b2 * signs1[:, None] * signs2
            left = clean(np.linalg.eigvalsh(b1.T @ b1 + b2 @ b2.T))
            right = clean(np.linalg.eigvalsh(ob1.T @ ob1 + ob2 @ ob2.T))
            assert np.allclose(left, right, atol=1e-9); checks += 4
            ordinary = s10.ordinary_spectra(matrix_a)
            assert np.allclose(ordinary["q0"], np.linalg.eigvalsh(b1 @ b1.T), atol=1e-8)
            assert np.allclose(ordinary["q1"], left, atol=1e-8)
            assert abs(ordinary["q0"].sum() - 2 * b1.shape[1]) < 1e-8
            assert abs(ordinary["q1"].sum() - (2 * b1.shape[1] + 3 * b2.shape[1])) < 1e-8
            checks += 4
        assert np.allclose(s10.persistent_spectrum(a, target), expected["pl12"], atol=1e-8)
        checks += 1
        permutation = rng.permutation(len(a))
        permuted = s10.compute_s10(a[np.ix_(permutation, permutation)])
        assert np.allclose(actual["values"], permuted["values"], atol=1e-8)
        checks += 1
        saw_nonshortcut_schur |= actual["qc"]["operators"]["pl12"]["method"] == \
            "exact_upper_schur_complete_spectrum"
    assert saw_nonshortcut_schur; checks += 1
    c4 = s10.compute_s10(cases["cycle4"], return_spectra=True)
    assert np.allclose(c4["spectra"]["h1"], [0, 2, 2, 4]); checks += 1
    assert np.allclose(c4["spectra"]["pl12"], [2, 2, 4, 4]); checks += 1
    assert np.allclose(c4["spectra"]["h2"], np.full(6, 4)); checks += 1
    # A naive individual-triangle restriction leaves C4 unfilled; the true
    # persistent operator instead has nullity zero. This catches a substantive bug.
    assert c4["qc"]["operators"]["h1"]["nullity"] == 1
    assert c4["qc"]["operators"]["pl12"]["nullity"] == 0; checks += 2
    for name in ("single_vertex", "single_edge", "path4", "star5"):
        result = s10.compute_s10(cases[name], return_spectra=True)
        assert result["qc"]["operators"]["pl12"]["nullity"] == 0; checks += 1
    assert s10.compute_s10(cases["single_vertex"])["values"] == [0.] * 10; checks += 1
    assert s10.compute_s10(cases["single_vertex"], return_spectra=True)["spectra"]["pl12"].size == 0
    checks += 1
    # Input failures cannot silently convert weighted/directed/disconnected data.
    bad = [np.ones((3, 3)), np.zeros((2, 2)), np.zeros((41, 41)), np.array([[0, 1], [0, 0]]),
           np.array([[0, .5], [.5, 0]]), np.zeros((0, 0))]
    for a in bad:
        try:
            s10.compute_s10(a)
        except ValueError:
            checks += 1
        else:
            raise AssertionError("invalid input accepted")
    return {"passed": True, "cases": len(cases), "checks": checks,
            "max_spectral_abs_diff": max_spectral_difference,
            "max_feature_abs_diff": max_feature_difference,
            "reference_method": "independent full matrices + restricted-chain SVD",
            "complete_positive_spectrum": True}


def benchmark():
    results = []
    rng = np.random.default_rng(41008)
    for probability in (0.00, 0.025, 0.06, 0.12):
        a = adjacency(40, [(i, i + 1) for i in range(39)])
        for i, j in combinations(range(40), 2):
            if rng.random() < probability:
                a[i, j] = a[j, i] = True
        begin = time.perf_counter()
        output = s10.compute_s10(a)
        duration = time.perf_counter() - begin
        qc = output["qc"]
        results.append({"n": 40, "added_edge_probability": probability,
            "seconds": duration, "edges1": qc["n_edges_eps1"],
            "edges2": qc["n_edges_eps2"], "triangles2": qc["n_triangles_eps2"],
            "persistent_method": qc["operators"]["pl12"]["method"]})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--out", type=Path,
                        help="also write the bound validation_math.json in this run directory")
    args = parser.parse_args()
    result = verify()
    if args.benchmark:
        result["benchmark"] = benchmark()
    root = Path(__file__).resolve().parents[2]
    paths = [Path(s10.__file__).resolve(), Path(__file__).resolve()]
    hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in paths}
    result.update(validator_sha256=hashes[str(paths[1].relative_to(root))],
                  source_hashes=hashes, result_hashes={}, source_coefficient_field="R")
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / "validation_math.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
