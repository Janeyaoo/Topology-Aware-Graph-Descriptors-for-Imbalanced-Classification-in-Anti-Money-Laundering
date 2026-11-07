"""Exact, real-coefficient S10 spectra on one fixed local point set.

The input is a connected, simple, undirected, unweighted adjacency matrix with
1 <= n <= 40.  K1 is its clique complex, and K2 is the clique complex of the
distance-at-most-two graph.  Only the 2-skeleton is needed for q=0,1.

All simplex inner products are the standard Euclidean inner products.  The
persistent operator is the genuine L1^(K1,K2), not a restriction to triangles
whose individual boundaries already lie in K1.  Its upper part is computed by
the generalized Schur complement of B2(K2) B2(K2)^T.

Every positive eigenvalue is used.  No sparse truncated eigenspectrum or entropy
approximation is used.  Eigenvalues at/below ZERO_TOL are numerical zeros; a
diagnostic flags values close enough to this boundary to merit inspection.
Hodge nullities use real coefficients, not the F2 coefficient field of PH.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import warnings

import numpy as np

MAX_VERTICES = 40
ZERO_TOL = 1e-9
NEGATIVE_TOL = 1e-7
FEATURE_NAMES = (
    "s_g1_min", "s_g1_entropy", "s_g2_min", "s_g2_entropy",
    "s_h1_min", "s_h1_entropy", "s_h2_min", "s_h2_entropy",
    "s_pl12_min", "s_pl12_entropy",
)
S10_OPERATORS = ("g1", "g2", "h1", "h2", "pl12")


class NumericalSpectrumWarning(RuntimeWarning):
    """The complete computed spectrum has eigenvalues near the zero cutoff."""


class NumericalSpectrumError(ArithmeticError):
    """The computed operator violates a mathematical spectral invariant."""


def _adjacency(value: np.ndarray) -> np.ndarray:
    a = np.asarray(value)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("adjacency must be square")
    n = len(a)
    if not 1 <= n <= MAX_VERTICES:
        raise ValueError(f"adjacency needs 1 <= n <= {MAX_VERTICES}")
    if a.dtype.kind not in "buif" or not np.isfinite(a).all():
        raise ValueError("adjacency must contain finite real 0/1 entries")
    if not np.logical_or(a == 0, a == 1).all():
        raise ValueError("weighted adjacency is not supported")
    a = np.array(a, dtype=bool, copy=True)
    if np.diag(a).any() or not np.array_equal(a, a.T):
        raise ValueError("adjacency must be symmetric with zero diagonal")
    seen = {0}
    todo = [0]
    while todo:
        u = todo.pop()
        for v in np.flatnonzero(a[u]):
            if int(v) not in seen:
                seen.add(int(v))
                todo.append(int(v))
    if len(seen) != n:
        raise ValueError("adjacency must be connected")
    return a


def vr_epsilon2(adjacency: np.ndarray) -> np.ndarray:
    """The VR epsilon=2 edge set in the input graph's shortest-path metric."""
    a = _adjacency(adjacency)
    out = a | ((a.astype(np.uint16) @ a.astype(np.uint16)) > 0)
    np.fill_diagonal(out, False)
    return out


@dataclass
class _Complex:
    adjacency: np.ndarray
    edges: np.ndarray
    triangles: np.ndarray
    triangle_edges: np.ndarray
    b1: np.ndarray
    q0: np.ndarray
    q0_raw: np.ndarray
    complete: bool

    @property
    def n(self) -> int:
        return len(self.adjacency)

    @property
    def m(self) -> int:
        return len(self.edges)

    def b2(self) -> np.ndarray:
        b = np.zeros((self.m, len(self.triangles)), dtype=np.float64)
        if len(self.triangles):
            columns = np.arange(len(self.triangles))
            # Boundary[a,b,c] = [b,c] - [a,c] + [a,b].
            b[self.triangle_edges[:, 0], columns] = 1.0
            b[self.triangle_edges[:, 1], columns] = -1.0
            b[self.triangle_edges[:, 2], columns] = 1.0
        return b

    def upper(self) -> np.ndarray:
        """Dense B2 B2^T, assembled exactly with nine entries per triangle.

        This avoids allocating a potentially m-by-C(n,3) dense B2 in production;
        it is the standard signed Gram matrix, not an approximation.
        """
        u = np.zeros((self.m, self.m), dtype=np.float64)
        if len(self.triangle_edges):
            signs = (1.0, -1.0, 1.0)
            for j in range(3):
                for k in range(3):
                    np.add.at(u, (self.triangle_edges[:, j],
                                  self.triangle_edges[:, k]), signs[j] * signs[k])
        return u


def _clean(raw: np.ndarray, *, name: str) -> np.ndarray:
    vals = np.asarray(raw, dtype=np.float64).reshape(-1)
    if not np.isfinite(vals).all():
        raise NumericalSpectrumError(f"{name}: nonfinite eigenvalues")
    if len(vals) and vals.min() < -NEGATIVE_TOL:
        raise NumericalSpectrumError(f"{name}: PSD violation {vals.min():.6g}")
    ambiguous = (np.abs(vals) > ZERO_TOL / 10) & (np.abs(vals) < 10 * ZERO_TOL)
    if ambiguous.any():
        warnings.warn(f"{name}: {int(ambiguous.sum())} eigenvalues near numerical "
                      f"zero cutoff {ZERO_TOL}", NumericalSpectrumWarning, stacklevel=2)
    out = np.sort(vals.copy())
    out[out <= ZERO_TOL] = 0.0
    return out


def _complex(a: np.ndarray) -> _Complex:
    n = len(a)
    edges = np.asarray(list(zip(*np.nonzero(np.triu(a, 1)))), dtype=np.int64).reshape(-1, 2)
    edge_index = {tuple(e): i for i, e in enumerate(edges)}
    triangles = np.asarray([(x, y, z) for x, y, z in combinations(range(n), 3)
                            if a[x, y] and a[x, z] and a[y, z]],
                           dtype=np.int64).reshape(-1, 3)
    triangle_edges = np.asarray([(edge_index[(y, z)], edge_index[(x, z)],
                                  edge_index[(x, y)]) for x, y, z in triangles],
                                 dtype=np.int64).reshape(-1, 3)
    b1 = np.zeros((n, len(edges)), dtype=np.float64)
    if len(edges):
        b1[edges[:, 0], np.arange(len(edges))] = -1.0
        b1[edges[:, 1], np.arange(len(edges))] = 1.0
    complete = len(edges) == n * (n - 1) // 2
    if complete:
        raw = np.concatenate(([0.0], np.full(n - 1, float(n))))
    else:
        raw = np.linalg.eigvalsh(b1 @ b1.T)
    q0 = _clean(raw, name="q0")
    if np.count_nonzero(q0 > 0) != n - 1:
        raise NumericalSpectrumError("connected q0 does not have rank n-1")
    return _Complex(a, edges, triangles, triangle_edges, b1, q0, raw, complete)


def oriented_boundaries(adjacency: np.ndarray) -> dict:
    """Return lexicographically oriented B1/B2 and simplex lists for inspection."""
    c = _complex(_adjacency(adjacency))
    return {"edges": c.edges.copy(), "triangles": c.triangles.copy(),
            "b1": c.b1.copy(), "b2": c.b2()}


def _merge_spectra(c: _Complex, upper_raw: np.ndarray, *, name: str) -> tuple[np.ndarray, dict]:
    upper = _clean(upper_raw, name=f"{name}.upper")
    down_pos = c.q0[c.q0 > 0]
    up_pos = upper[upper > 0]
    pos = np.concatenate((down_pos, up_pos))
    if len(pos) > c.m:
        raise NumericalSpectrumError(f"{name}: upper/down rank exceeds edge dimension")
    # im(B1^T) and im(B2) are orthogonal: the complete q1 positive spectrum is
    # their union, with the remaining edge-space dimensions equal to zero.
    out = np.sort(np.concatenate((np.zeros(c.m - len(pos)), pos)))
    if len(out) and out.max() > c.n + NEGATIVE_TOL:
        raise NumericalSpectrumError(f"{name}: unweighted eigenvalue exceeds n")
    diagnostics = {
        "raw_min_eigenvalue": min(float(c.q0_raw.min(initial=0)),
                                  float(np.min(upper_raw, initial=0))),
        "near_zero_eigenvalues": int(np.count_nonzero(
            (np.abs(upper_raw) > ZERO_TOL / 10) &
            (np.abs(upper_raw) < 10 * ZERO_TOL))),
        "upper_rank": int(len(up_pos)),
    }
    return out, diagnostics


def _ordinary_q1(c: _Complex) -> tuple[np.ndarray, dict]:
    if c.complete:
        return np.full(c.m, float(c.n)), {"method": "complete_n_identity",
            "raw_min_eigenvalue": 0.0, "near_zero_eigenvalues": 0,
            "upper_rank": c.m - (c.n - 1)}
    if not len(c.triangles):
        out, diag = _merge_spectra(c, np.empty(0), name="ordinary_q1")
        diag["method"] = "triangle_free_down_spectrum"
        return out, diag
    if len(c.triangles) < c.m:
        b2 = c.b2()
        gram = b2.T @ b2
        method = "complete_triangle_gram_spectrum"
    else:
        gram = c.upper()
        method = "complete_edge_gram_spectrum"
    out, diag = _merge_spectra(c, np.linalg.eigvalsh(gram), name="ordinary_q1")
    diag["method"] = method
    return out, diag


def ordinary_spectra(adjacency: np.ndarray) -> dict:
    """Return complete ordinary q0/q1 spectra, including every zero eigenvalue."""
    c = _complex(_adjacency(adjacency))
    q1, diag = _ordinary_q1(c)
    return {"q0": c.q0.copy(), "q1": q1, "q1_diagnostics": diag,
            "n": c.n, "n_edges": c.m, "n_triangles": len(c.triangles)}


def _persistent_q1(source: _Complex, target: _Complex) -> tuple[np.ndarray, dict]:
    # On a forest, C1 has no cycles, so the persistent upper part is exactly zero.
    if source.m == source.n - 1:
        return source.q0[source.q0 > 0].copy(), {"method": "source_forest_down_spectrum",
            "raw_min_eigenvalue": 0.0, "near_zero_eigenvalues": 0,
            "upper_rank": 0, "schur_eliminated_rank": 0,
            "schur_kernel_coupling_max": 0.0}
    if target.complete:
        # The Schur complement of the complete upper Laplacian is n times the
        # orthogonal projector onto source graph cycles.  Combine with old down.
        cycle_rank = source.m - source.n + 1
        out = np.sort(np.concatenate((source.q0[source.q0 > 0],
                                      np.full(cycle_rank, float(source.n)))))
        return out, {"method": "complete_target_cycle_projector",
            "raw_min_eigenvalue": 0.0, "near_zero_eigenvalues": 0,
            "upper_rank": cycle_rank, "schur_eliminated_rank": None,
            "schur_kernel_coupling_max": 0.0}
    if np.array_equal(source.adjacency, target.adjacency):
        out, diag = _ordinary_q1(source)
        diag.update(method="equal_complexes_ordinary_q1", schur_eliminated_rank=0,
                    schur_kernel_coupling_max=0.0)
        return out, diag
    upper = target.upper()
    old_mask = source.adjacency[target.edges[:, 0], target.edges[:, 1]]
    old = np.flatnonzero(old_mask)
    new = np.flatnonzero(~old_mask)
    a = upper[np.ix_(old, old)]
    cross = upper[np.ix_(old, new)]
    d = upper[np.ix_(new, new)]
    d_raw, d_vec = np.linalg.eigh(d)
    d_vals = _clean(d_raw, name="persistent_schur_D")
    keep = d_vals > 0
    d_rank = int(keep.sum())
    if (~keep).any():
        kernel_coupling = float(np.max(np.abs(cross @ d_vec[:, ~keep]), initial=0.0))
        if kernel_coupling > 1e-6:
            raise NumericalSpectrumError("Schur cross block does not annihilate D kernel")
    else:
        kernel_coupling = 0.0
    if d_rank:
        reduced_cross = cross @ d_vec[:, keep]
        a = a - (reduced_cross / d_vals[keep]) @ reduced_cross.T
    a = (a + a.T) / 2
    # New-boundary cancellation should leave only source cycles in the image.
    annihilation = float(np.max(np.abs(source.b1 @ a), initial=0.0))
    if annihilation > 1e-6:
        raise NumericalSpectrumError("persistent upper fails B1_s upper=0")
    out, diag = _merge_spectra(source, np.linalg.eigvalsh(a), name="persistent_q1")
    diag.update(method="exact_upper_schur_complete_spectrum", schur_eliminated_rank=d_rank,
                schur_kernel_coupling_max=kernel_coupling,
                boundary_annihilation_max=annihilation,
                schur_D_near_zero_eigenvalues=int(np.count_nonzero(
                    (np.abs(d_raw) > ZERO_TOL / 10) & (np.abs(d_raw) < 10 * ZERO_TOL))))
    return out, diag


def persistent_spectrum(adjacency_source: np.ndarray,
                        adjacency_target: np.ndarray) -> np.ndarray:
    """Complete genuine persistent q1 spectrum for nested same-vertex cliques."""
    a, b = _adjacency(adjacency_source), _adjacency(adjacency_target)
    if a.shape != b.shape or np.any(a & ~b):
        raise ValueError("source must be a same-vertex subgraph of target")
    out, _ = _persistent_q1(_complex(a), _complex(b))
    return out


def _summary(spectrum: np.ndarray, n: int, diagnostics: dict | None = None) -> tuple[float, float, dict]:
    spectrum = np.asarray(spectrum, dtype=np.float64)
    positive = spectrum[spectrum > ZERO_TOL]
    r = len(positive)
    trace = float(positive.sum())
    min_n = float(positive.min() / n) if r else 0.0
    entropy = 0.0
    if r > 1:
        p = positive / trace
        entropy = float(-np.sum(p * np.log(p)) / np.log(r))
        if not -1e-12 <= entropy <= 1 + 1e-12:
            raise NumericalSpectrumError("positive rank-normalized entropy out of bounds")
        entropy = float(np.clip(entropy, 0.0, 1.0))
    qc = {"dimension": int(len(spectrum)), "rank": int(r),
          "nullity": int(len(spectrum) - r), "trace": trace,
          "min_positive": float(positive.min()) if r else 0.0,
          "no_positive_spectrum": not bool(r), "entropy_rank_le_one": r <= 1}
    if diagnostics:
        qc.update(diagnostics)
    return min_n, entropy, qc


def compute_s10(adjacency: np.ndarray, *, return_spectra: bool = False) -> dict:
    """Compute all ten scalar features and real-coefficient operator QC.

    Return {features: name->float, values: list of ten floats in FEATURE_NAMES
    order, qc: JSON-compatible diagnostics}.  With return_spectra=True, add a
    dict of complete NumPy eigenvalue arrays keyed by S10_OPERATORS.
    The input adjacency is never mutated.
    """
    a1 = _adjacency(adjacency)
    a2 = a1 | ((a1.astype(np.uint16) @ a1.astype(np.uint16)) > 0)
    np.fill_diagonal(a2, False)
    c1, c2 = _complex(a1), _complex(a2)
    h1, h1_diag = _ordinary_q1(c1)
    h2, h2_diag = _ordinary_q1(c2)
    pl12, pl_diag = _persistent_q1(c1, c2)
    spectra = {"g1": c1.q0, "g2": c2.q0, "h1": h1, "h2": h2, "pl12": pl12}
    diagnostics = {"h1": h1_diag, "h2": h2_diag, "pl12": pl_diag}
    features, operator_qc = {}, {}
    for op in S10_OPERATORS:
        minimum, entropy, qc = _summary(spectra[op], c1.n, diagnostics.get(op))
        features[f"s_{op}_min"] = minimum
        features[f"s_{op}_entropy"] = entropy
        operator_qc[op] = qc
    assert tuple(features) == FEATURE_NAMES
    qc = {"n": c1.n, "n_edges_eps1": c1.m, "n_edges_eps2": c2.m,
          "n_triangles_eps1": len(c1.triangles), "n_triangles_eps2": len(c2.triangles),
          "eps1_complete": c1.complete, "eps2_complete": c2.complete,
          "source_forest": c1.m == c1.n - 1, "coefficient_field": "R",
          "inner_product": "standard_unweighted", "zero_tolerance": ZERO_TOL,
          "negative_tolerance": NEGATIVE_TOL, "uses_complete_positive_spectrum": True,
          "operators": operator_qc}
    result = {"features": features, "values": [features[k] for k in FEATURE_NAMES], "qc": qc}
    if return_spectra:
        result["spectra"] = {key: value.copy() for key, value in spectra.items()}
    return result
