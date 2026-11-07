"""Graph statistics by class and by laundering typology (2026-10-12).

Read-only: transactions, the frozen feature cache and geometry QC of
account_s10_increment_20261008, and the IBM pattern files. Labels are used
only to group accounts for description; nothing is fitted.
Outputs graph_label_stats.json and graph_label_tables.md in the
supplementary analysis directory.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components

ROOT = Path("Altman_experiments")
S10 = ROOT / "results/account_s10_increment_20261008"
OUT = ROOT / "results/paper_eswa_supplementary_analysis_20261012"
DS = {"hi_small": "HI-Small", "li_small": "LI-Small"}
COLS = ["account_id", "label", "A1", "A2", "D1", "D2", "nv", "core", "rec_out_count", "rec_in_count",
        "beta1", "ph_2hop_h1_total", "ph_3hop_h1_max"]

spec = importlib.util.spec_from_file_location("supp", ROOT / "code/eswa_supplementary_analysis_20261012.py")
supp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(supp)


def account_frame(ds):
    tr = pd.read_parquet(ROOT / f"data/{ds}/{ds}_trans.parquet", columns=["from_id", "to_id"])
    f = pd.read_parquet(S10 / ds / "features.parquet", columns=COLS)
    qc = pd.read_parquet(S10 / ds / "geometry_qc.parquet", columns=["account_id", "cap_hit"])
    assert (qc.account_id.to_numpy() == f.account_id.to_numpy()).all()
    f["cap_hit"] = qc.cap_hit.to_numpy()
    ids = pd.Index(sorted(f.account_id))
    a, b = ids.get_indexer(tr.from_id), ids.get_indexer(tr.to_id)
    m = a != b
    n = len(ids)
    D = sparse.coo_matrix((np.ones(m.sum(), np.int8), (a[m], b[m])), shape=(n, n)).tocsr()
    D.data[:] = 1
    U = ((D + D.T) > 0).astype(np.int8).tocsr()
    row = ids.get_indexer(f.account_id)
    deg = np.asarray(U.sum(1)).ravel()
    n_comp, comp = connected_components(U, directed=False)
    sizes = np.bincount(comp)
    giant = sizes.argmax()
    pos_g = np.zeros(n, np.int64)  # int64: an int8 product would overflow for hubs with >127 positive neighbours
    pos_g[row[f.label.to_numpy() == 1]] = 1
    n_pos_nb = U @ pos_g
    f["degree"] = deg[row]
    f["in_giant"] = comp[row] == giant
    f["isolated"] = deg[row] == 0
    f["pos_neighbors"] = n_pos_nb[row]
    # edge composition
    ui, uj = sparse.triu(U, 1).nonzero()
    both = int(((pos_g[ui] == 1) & (pos_g[uj] == 1)).sum())
    one = int(((pos_g[ui] + pos_g[uj]) == 1).sum())
    pos_deg_share = deg[pos_g == 1].sum() / deg.sum()
    graph = {"components": int(n_comp), "nontrivial_components": int((sizes > 1).sum()),
             "giant_size": int(sizes[giant]), "giant_share": float(sizes[giant] / n),
             "edges": int(len(ui)), "edges_pos_pos": both, "edges_pos_neg": one,
             "share_pos_pos": both / len(ui), "expected_share_pos_pos_degree_preserving": float(pos_deg_share ** 2),
             "homophily_ratio": float((both / len(ui)) / pos_deg_share ** 2)}
    return f, graph


def profile(g):
    return {"n": int(len(g)),
            "isolated": float(g.isolated.mean()), "in_giant": float(g.in_giant.mean()),
            "degree_median": float(g.degree.median()), "degree_mean": float(g.degree.mean()),
            "D1_median": float(g.D1.median()), "A1_median": float(g.A1.median()),
            "D1_ge5": float((g.D1 >= 5).mean()), "A1_ge5": float((g.A1 >= 5).mean()),
            "core_ge2": float((g.core >= 2).mean()), "nv_median": float(g.nv.median()),
            "rec_out_pos": float((g.rec_out_count > 0).mean()), "rec_in_pos": float((g.rec_in_count > 0).mean()),
            "beta1_pos": float((g.beta1 > 0).mean()), "h1_2hop_pos": float((g.ph_2hop_h1_total > 0).mean()),
            "h1_long": float((g.ph_3hop_h1_max >= 2).mean()), "cap_hit": float(g.cap_hit.mean()),
            "has_pos_neighbor": float((g.pos_neighbors > 0).mean())}


def pct(x):
    return f"{100 * x:.1f}%"


def main():
    res, md = {}, []
    frames = {}
    for ds in DS:
        f, graph = account_frame(ds)
        frames[ds] = f
        pos, neg = f[f.label == 1], f[f.label == 0]
        pat, _ = supp.parse_patterns(ds)
        groups = {"负类账户": neg, "全部正类": pos,
                  "未归入模式的正类": pos[~pos.account_id.isin(pat.account_id)]}
        for p in ["FAN-OUT", "FAN-IN", "GATHER-SCATTER", "SCATTER-GATHER", "CYCLE", "BIPARTITE", "STACK", "RANDOM"]:
            groups[p] = f[f.account_id.isin(pat.account_id[pat.pattern == p])]
        res[ds] = {"graph": graph, "groups": {k: profile(v) for k, v in groups.items()}}
    (OUT / "graph_label_stats.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))

    # Table: positive vs negative accounts
    rows = [("孤立账户（只有自转账）", "isolated", pct), ("位于最大连通分量", "in_giant", pct),
            ("度：中位数（均值）", None, None), ("出度D1≥5", "D1_ge5", pct), ("入度A1≥5", "A1_ge5", pct),
            ("k-core≥2", "core_ge2", pct), ("局部集合\\(n_v\\)中位数", "nv_median", lambda x: f"{x:.0f}"),
            ("存在下游再汇合（M）", "rec_out_pos", pct), ("存在上游再汇合（M）", "rec_in_pos", pct),
            ("局部循环秩\\(\\beta_1>0\\)", "beta1_pos", pct), ("二跳采样子图有H1条码", "h1_2hop_pos", pct),
            ("三跳采样有寿命为2的H1条码", "h1_long", pct), ("三跳采样触发顶点上限", "cap_hit", pct),
            ("至少有一个正类邻居", "has_pos_neighbor", pct)]
    md.append("## table: positive vs negative\n")
    md.append("| 统计量 | HI-Small 正类 | HI-Small 负类 | LI-Small 正类 | LI-Small 负类 |\n| --- | --- | --- | --- | --- |")
    for lab, key, fmt in rows:
        cells = []
        for ds in DS:
            for grp in ["全部正类", "负类账户"]:
                p = res[ds]["groups"][grp]
                cells.append(f"{p['degree_median']:.0f}（{p['degree_mean']:.1f}）" if key is None else fmt(p[key]))
        md.append(f"| {lab} | " + " | ".join(cells) + " |")
    md.append("\n## graph-level\n")
    for ds in DS:
        md.append(f"- {DS[ds]}: " + json.dumps(res[ds]["graph"], ensure_ascii=False))
    # Table: typology profiles
    names = {"FAN-OUT": "扇出", "FAN-IN": "扇入", "GATHER-SCATTER": "gather-scatter", "SCATTER-GATHER": "scatter-gather",
             "CYCLE": "环路", "BIPARTITE": "二部", "STACK": "堆叠", "RANDOM": "随机"}
    md.append("\n## table: typology profiles\n")
    md.append("| 数据集 | 账户组 | 账户数 | 度中位数 | D1≥5 | A1≥5 | 下游再汇合 | 上游再汇合 | \\(\\beta_1>0\\) | 二跳H1 | 寿命2的H1 |")
    md.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for ds in DS:
        for grp, p in res[ds]["groups"].items():
            md.append(f"| {DS[ds]} | {names.get(grp, grp)} | {p['n']:,} | {p['degree_median']:.0f} | {pct(p['D1_ge5'])} | {pct(p['A1_ge5'])} | "
                      f"{pct(p['rec_out_pos'])} | {pct(p['rec_in_pos'])} | {pct(p['beta1_pos'])} | {pct(p['h1_2hop_pos'])} | {pct(p['h1_long'])} |")
    (OUT / "graph_label_tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
