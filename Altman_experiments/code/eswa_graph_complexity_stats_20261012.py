"""Descriptive graph statistics and empirical complexity quantities (2026-10-12).

Read-only: uses the transaction parquet files, the frozen feature cache of
account_s10_increment_20261008 (size columns nv, ev, q_*), its geometry QC and
the saved BFS samples in its chunk files. Writes JSON and Markdown to the
supplementary analysis directory. No feature is recomputed.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path("Altman_experiments")
S10 = ROOT / "results/account_s10_increment_20261008"
OUT = ROOT / "results/paper_eswa_supplementary_analysis_20261012"
CAP = 40


def q(x, p):
    return float(np.quantile(x, p))


def describe(x):
    x = np.asarray(x, float)
    return {"mean": float(x.mean()), "median": q(x, .5), "p90": q(x, .9), "p99": q(x, .99), "max": float(x.max()), "sum": float(x.sum())}


def run(ds):
    tr = pd.read_parquet(ROOT / f"data/{ds}/{ds}_trans.parquet", columns=["timestamp", "from_id", "to_id"])
    feat = pd.read_parquet(S10 / ds / "features.parquet",
                           columns=["account_id", "label", "A1", "A2", "D1", "D2", "nv", "ev", "core",
                                    "q_n2", "q_n", "q_e1", "q_t1", "q_e2", "q_t2"])
    qc = pd.read_parquet(S10 / ds / "geometry_qc.parquet", columns=["account_id", "cap_hit", "effective_radius"])
    assert (qc.account_id.to_numpy() == feat.account_id.to_numpy()).all()
    ids = pd.Index(sorted(feat.account_id))
    a, b = ids.get_indexer(tr.from_id), ids.get_indexer(tr.to_id)
    assert (a >= 0).all() and (b >= 0).all()
    self_loops = int((a == b).sum())
    m = a != b
    n = len(ids)
    D = sparse.coo_matrix((np.ones(m.sum(), np.int8), (a[m], b[m])), shape=(n, n)).tocsr()
    D.data[:] = 1
    U = ((D + D.T) > 0).astype(np.int8).tocsr()
    out_deg = np.asarray(D.sum(1)).ravel()
    in_deg = np.asarray(D.sum(0)).ravel()
    deg = np.asarray(U.sum(1)).ravel()
    recip = int(D.multiply(D.T).nnz // 2)
    # two-step same-direction path counts P_v^+ = sum_{u in N1+(v)} d+(u), P_v^- analogous
    p2_out = D @ out_deg
    p2_in = D.T @ in_deg
    # BFS work of the persistent-homology layer: sum of degrees of expanded vertices (depth <= 2) in each sample
    order = np.load(S10 / ds / "account_to_graph.npy")
    graph_ids = pd.read_parquet(S10 / ds / "graph_accounts.parquet").account_id
    assert (graph_ids.to_numpy() == ids.to_numpy()).all()
    bfs = np.zeros(n, np.int64)
    unique_graphs = 0
    chunks = sorted((S10 / ds / "chunks").glob("*.npz"))
    for path in chunks:
        start, stop = map(int, path.stem.split("_"))
        with np.load(path) as f:
            v, d = f["vertices"], f["depths"]
        deg_s = np.where(v >= 0, deg[np.clip(v, 0, None)], 0)
        bfs[start:stop] = (deg_s * ((d >= 0) & (d <= 2))).sum(1)
        unique_graphs += json.loads(path.with_suffix(".json").read_text())["unique_s10_graphs"]
    # rows of the feature cache follow feat order; chunk rows follow the same order (account_to_graph maps row -> graph index)
    assert len(order) == len(feat)
    lab = feat.label.to_numpy() == 1
    deg_rows = deg[order]
    t = pd.to_datetime(tr.timestamp)
    stats = {
        "accounts": n, "transactions": len(tr), "self_transfers": self_loops,
        "directed_simple_edges": int(D.nnz), "undirected_simple_edges": int(U.nnz // 2), "reciprocal_pairs": recip,
        "isolated_accounts": int((deg == 0).sum()), "max_core": int(feat.core.max()),
        "period_start": str(t.min()), "period_end": str(t.max()), "period_days": float((t.max() - t.min()).total_seconds() / 86400),
        "degree": describe(deg), "out_degree_max": int(out_deg.max()), "in_degree_max": int(in_deg.max()),
        "sum_deg_squared": float((deg.astype(float) ** 2).sum()),
        "degree_median_pos": q(deg_rows[lab], .5), "degree_median_neg": q(deg_rows[~lab], .5),
        "degree_mean_pos": float(deg_rows[lab].mean()), "degree_mean_neg": float(deg_rows[~lab].mean()),
        "struct_nv": describe(feat.nv), "struct_ev": describe(feat.ev),
        "struct_D1D2_A1A2": describe(feat.D1 * feat.D2 + feat.A1 * feat.A2),
        "two_step_paths": describe(p2_out + p2_in),
        "ph_cap_hit_rate": float(qc.cap_hit.mean()), "ph_n3": describe(feat.q_n), "ph_n2": describe(feat.q_n2),
        "ph_e1": describe(feat.q_e1), "ph_t1": describe(feat.q_t1), "ph_e2": describe(feat.q_e2), "ph_t2": describe(feat.q_t2),
        "ph_bfs_degree_scan": describe(bfs),
        "ph_unique_subgraph_computations": unique_graphs, "ph_unique_share": unique_graphs / n,
        "worst_case_triangles_c40": int(CAP * (CAP - 1) * (CAP - 2) / 6),
    }
    return stats


def main():
    res = {ds: run(ds) for ds in ["hi_small", "li_small"]}
    (OUT / "graph_complexity_stats.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1)[:6000])


if __name__ == "__main__":
    main()
