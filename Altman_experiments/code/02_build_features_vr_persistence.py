"""
02_build_features_vr_persistence.py
===================================
Compute two feature sets for each account in the IBM AML HI-Small dataset.

BASE features (15 dims) — raw transaction statistics, no graph topology:
  Volume, count, timing, currency diversity — truly structurally agnostic.

FTD-VR features (36 dims) — FlowTopo flow-topological descriptors computed
  on the directed 2-hop account neighborhood. Family D is replaced by
  Vietoris-Rips persistent-homology descriptors extracted from H0/H1
  persistence barcodes: max persistence, total persistence, and
  persistence entropy for each homology dimension.

VR design
---------
  * Point cloud: accounts in the 2-hop local neighborhood of account v.
  * Metric: unweighted shortest-path distance on the undirected local graph.
  * Filtration: Vietoris-Rips complex up to dimension 2.
  * Betti0/Betti1: number of H0/H1 bars alive at epsilon=1.0, which matches
    the original one-edge local graph scale while still being obtained from
    the VR filtration.
  * Persistence summaries: computed from finite bars only. Infinite H0 bars
    are excluded from length-based summaries to avoid arbitrary truncation.

Memory strategy
---------------
  * Adjacency lists: defaultdict(set) — O(|E|) total, no matrix needed.
  * K-core: NetworkX on the undirected projection (1-time cost, ~2 min).
  * FTD computation: node-by-node with adjacency lookups; no large temp objects.
  * VR guardrail: very large local neighborhoods are deterministically capped
    by graph distance from v before VR expansion, preventing cubic blow-ups.
  * Checkpointing: save after base and after FTD so a crash doesn't lose work.

Outputs (results/features/)
---------------------------
  base_features.parquet   — (n_accounts, 15)
  ftd_vr_features.parquet — (n_accounts, 36)
  full_vr_features.parquet — (n_accounts, 51) = base + FTD-VR
  kcore.parquet           — (n_accounts,)  core numbers
"""

import os, gc, time
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict, deque
from itertools import combinations

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA    = f"{BASE}/data"
if os.path.exists(f"{DATA}/hi_small/hi_small_trans.parquet"):
    DATA = f"{DATA}/hi_small"
OUT     = f"{BASE}/results/features"
os.makedirs(OUT, exist_ok=True)

print("=" * 60)
print("  IBM AML — Feature Computation with VR Persistence")
print("=" * 60)

# ── 1. Load data ─────────────────────────────────────────────
print("\n[1/6] Loading transactions …")
t0 = time.time()
trans  = pd.read_parquet(f"{DATA}/hi_small_trans.parquet")
labels = pd.read_parquet(f"{DATA}/account_labels.parquet")
accounts = labels["account_id"].values
acct_set = set(accounts)
print(f"  {len(trans):,} transactions | {len(accounts):,} accounts  ({time.time()-t0:.1f}s)")

# ── 2. Base features (raw transaction stats) ──────────────────
print("\n[2/6] Computing BASE features (raw stats, no topology) …")
t0 = time.time()

# Sender perspective
sent = trans.groupby("from_id").agg(
    n_sent       = ("amount_paid",  "count"),
    vol_sent     = ("amount_paid",  "sum"),
    avg_sent     = ("amount_paid",  "mean"),
    std_sent     = ("amount_paid",  "std"),
    max_sent     = ("amount_paid",  "max"),
    n_recv_banks = ("to_id",        "nunique"),
).fillna(0)

# Receiver perspective
recv = trans.groupby("to_id").agg(
    n_recv       = ("amount_recv", "count"),
    vol_recv     = ("amount_recv", "sum"),
    avg_recv     = ("amount_recv", "mean"),
    std_recv     = ("amount_recv", "std"),
    max_recv     = ("amount_recv", "max"),
    n_sent_banks = ("from_id",     "nunique"),
).fillna(0)

# Diversity features (currency, payment format)
diversity = pd.concat([
    trans[["from_id","recv_currency","pay_format"]].rename(columns={"from_id":"account_id"}),
    trans[["to_id",  "recv_currency","pay_format"]].rename(columns={"to_id":  "account_id"}),
], ignore_index=True).groupby("account_id").agg(
    n_currencies = ("recv_currency", "nunique"),
    n_formats    = ("pay_format",    "nunique"),
).fillna(0)

# Merge
base = (labels[["account_id"]]
        .merge(sent,      left_on="account_id", right_index=True, how="left")
        .merge(recv,      left_on="account_id", right_index=True, how="left")
        .merge(diversity, on="account_id",                         how="left")
        .fillna(0))

# Derived ratios
base["sent_recv_ratio"] = base["vol_sent"] / (base["vol_recv"] + 1e-9)
base["degree"]          = base["n_sent"] + base["n_recv"]

feature_cols_base = [c for c in base.columns if c != "account_id"]
print(f"  BASE features: {len(feature_cols_base)} dims  ({time.time()-t0:.1f}s)")
base.to_parquet(f"{OUT}/base_features.parquet", index=False)

del sent, recv, diversity; gc.collect()

# ── 3. Build adjacency lists ──────────────────────────────────
print("\n[3/6] Building adjacency lists …")
t0 = time.time()

# Directed adjacency (account graph)
out_adj: dict[str, set] = defaultdict(set)   # out_adj[v] = successors
in_adj:  dict[str, set] = defaultdict(set)   # in_adj[v]  = predecessors

edges_df = trans[["from_id","to_id"]].copy()
edges_df = edges_df[edges_df["from_id"] != edges_df["to_id"]]
for f, t in zip(edges_df["from_id"], edges_df["to_id"]):
    out_adj[f].add(t)
    in_adj[t].add(f)

print(f"  Adjacency lists built  ({time.time()-t0:.1f}s)")

# ── 4. Compute k-core on undirected projection ────────────────
print("\n[4/6] Computing k-core (NetworkX undirected) …")
t0 = time.time()

G_und = nx.Graph()
# Only need edges involving labeled accounts and their 1-hop neighbors
# (avoids loading truly isolated nodes that don't affect labeled accounts)
edges_und = set()
for v in accounts:
    for u in out_adj[v]: edges_und.add((min(v,u), max(v,u)))
    for u in in_adj[v]:  edges_und.add((min(v,u), max(v,u)))

G_und.add_edges_from(edges_und)
G_und.remove_edges_from(nx.selfloop_edges(G_und))
del edges_und; gc.collect()

core_dict = nx.core_number(G_und)
del G_und; gc.collect()
print(f"  k-core computed for {len(core_dict):,} nodes  ({time.time()-t0:.1f}s)")

kcore_df = pd.DataFrame({
    "account_id": list(core_dict.keys()),
    "core":       list(core_dict.values()),
})
kcore_df.to_parquet(f"{OUT}/kcore.parquet", index=False)

# ── 5. Compute FTD features ──────────────────────────────────
print("\n[5/6] Computing FTD-VR features (36 dims) …")
t0 = time.time()

VR_MAX_POINTS = 80
BETTI_EPSILON = 1.0


def cap_local_vertices(v, Vv, out_adj, in_adj, max_points=VR_MAX_POINTS):
    """Deterministically cap a local neighborhood by undirected BFS distance."""
    if len(Vv) <= max_points:
        return set(Vv)

    seen = {v}
    queue = deque([(v, 0)])
    ranked = [(0, str(v), v)]

    while queue:
        u, dist = queue.popleft()
        nbrs = (out_adj.get(u, set()) | in_adj.get(u, set())) & Vv
        for w in sorted(nbrs, key=str):
            if w in seen:
                continue
            seen.add(w)
            ranked.append((dist + 1, str(w), w))
            queue.append((w, dist + 1))

    for u in Vv - seen:
        ranked.append((999, str(u), u))

    ranked.sort(key=lambda item: (item[0], item[1]))
    return {u for _, _, u in ranked[:max_points]}


def build_local_undirected_edges(Vv, out_adj):
    """Return undirected edges induced by Vv from the global directed graph."""
    und_edges: set[tuple] = set()
    for u in Vv:
        for w in out_adj.get(u, set()):
            if w in Vv:
                und_edges.add((min(u, w), max(u, w)))
    return und_edges


def shortest_path_metric(Vv, und_edges):
    """Compute finite shortest-path distances in an induced undirected graph."""
    nodes = sorted(Vv, key=str)
    idx = {u: i for i, u in enumerate(nodes)}
    adj = {u: set() for u in nodes}
    for u, w in und_edges:
        adj[u].add(w)
        adj[w].add(u)

    dist = {}
    for src in nodes:
        src_i = idx[src]
        seen = {src: 0}
        queue = deque([src])
        while queue:
            u = queue.popleft()
            for w in adj[u]:
                if w in seen:
                    continue
                seen[w] = seen[u] + 1
                queue.append(w)
        for dst, d in seen.items():
            dst_i = idx[dst]
            if src_i < dst_i:
                dist[(src_i, dst_i)] = float(d)

    return nodes, dist


def persistence_entropy(lengths):
    """Shannon entropy of normalized finite persistence lengths."""
    vals = np.array([x for x in lengths if np.isfinite(x) and x > 0], dtype=float)
    total = vals.sum()
    if total <= 0:
        return 0.0
    probs = vals / total
    return max(0.0, float(-(probs * np.log(probs + 1e-12)).sum()))


def summarize_persistence(diagram, dim):
    """Summarize finite persistence bars for one homology dimension."""
    lengths = [
        death - birth
        for d, birth, death in diagram
        if d == dim and np.isfinite(death) and death > birth
    ]
    total = float(np.sum(lengths)) if lengths else 0.0
    max_p = float(np.max(lengths)) if lengths else 0.0
    entropy = persistence_entropy(lengths)
    return max_p, total, entropy


def count_alive_bars(diagram, dim, epsilon=BETTI_EPSILON):
    """Betti number at a filtration scale: bars born by epsilon and not dead yet."""
    return sum(
        1
        for d, birth, death in diagram
        if d == dim and birth <= epsilon and (not np.isfinite(death) or death > epsilon)
    )


def vr_persistence_h0_h1(Vv, und_edges):
    """Compute H0/H1 persistence of the VR complex up to 2-simplices over Z2."""
    nodes, pair_dist = shortest_path_metric(Vv, und_edges)
    n = len(nodes)
    if n == 0:
        return []

    simplices = []
    for i in range(n):
        simplices.append((0.0, 0, (i,)))

    for (i, j), dij in pair_dist.items():
        simplices.append((dij, 1, (i, j)))

    for i, j, k in combinations(range(n), 3):
        dij = pair_dist.get((min(i, j), max(i, j)))
        dik = pair_dist.get((min(i, k), max(i, k)))
        djk = pair_dist.get((min(j, k), max(j, k)))
        if dij is None or dik is None or djk is None:
            continue
        simplices.append((max(dij, dik, djk), 2, (i, j, k)))

    simplices.sort(key=lambda item: (item[0], item[1], item[2]))
    simplex_index = {simplex: i for i, (_, _, simplex) in enumerate(simplices)}
    reduced_columns = {}
    low_to_col = {}
    paired_births = set()
    diagram = []

    for col_idx, (filt, dim, simplex) in enumerate(simplices):
        if dim == 0:
            boundary = set()
        elif dim == 1:
            boundary = {
                simplex_index[(simplex[0],)],
                simplex_index[(simplex[1],)],
            }
        else:
            i, j, k = simplex
            boundary = {
                simplex_index[(i, j)],
                simplex_index[(i, k)],
                simplex_index[(j, k)],
            }

        while boundary:
            low = max(boundary)
            if low not in low_to_col:
                break
            boundary ^= reduced_columns[low_to_col[low]]

        reduced_columns[col_idx] = boundary
        if boundary:
            low = max(boundary)
            low_to_col[low] = col_idx
            paired_births.add(low)
            birth_filt, birth_dim, _ = simplices[low]
            if birth_dim <= 1:
                diagram.append((birth_dim, birth_filt, filt))

    for idx, (birth_filt, birth_dim, _) in enumerate(simplices):
        if birth_dim <= 1 and idx not in paired_births and not reduced_columns[idx]:
            diagram.append((birth_dim, birth_filt, np.inf))

    return diagram


def compute_ftd(v, out_adj, in_adj, core_dict) -> dict:
    """Compute 36 FlowTopo-VR descriptors for account v."""
    # --- 2-hop neighborhoods ---
    n1_out = out_adj.get(v, set())
    n1_in  = in_adj.get(v,  set())
    n2_out = set()
    for u in n1_out: n2_out.update(out_adj.get(u, set()))
    n2_out -= n1_out | {v}
    n2_in  = set()
    for u in n1_in:  n2_in.update(in_adj.get(u, set()))
    n2_in  -= n1_in  | {v}

    A1, A2 = len(n1_in),  len(n2_in)
    D1, D2 = len(n1_out), len(n2_out)

    # ── Family A: Directed flow (14) ──────────────────────────
    asym   = abs(D1 - A1) / (A1 + D1 + 1)
    asym2  = abs(D2 - A2) / (A2 + D2 + 1)
    exp_out = D2 / (D1 + 1)

    # Downstream reconvergence
    rec_out_count = sum(
        1 for w in n2_out
        if sum(1 for u in n1_out if w in out_adj.get(u, set())) >= 2
    )
    P2_out = sum(len(out_adj.get(u, set())) for u in n1_out)
    rec_out_ratio = 1 - D2 / (P2_out + 1)

    # Upstream reconvergence
    rec_in_count = sum(
        1 for x in n2_in
        if sum(1 for u in n1_in if x in in_adj.get(u, set())) >= 2
    )
    P2_in = sum(len(in_adj.get(u, set())) for u in n1_in)
    rec_in_ratio = 1 - A2 / (P2_in + 1)

    # Peeling-like score
    if D1 >= 2:
        cont  = sum(1 for u in n1_out if len(out_adj.get(u, set())) >= 1) / (D1 + 1)
        leaf  = sum(1 for u in n1_out if len(out_adj.get(u, set())) == 0) / (D1 + 1)
        peel  = cont * leaf
    else:
        cont = leaf = peel = 0.0

    # ── Family B: Undirected topology (10) ────────────────────
    Vv = {v} | n1_out | n1_in | n2_out | n2_in
    nv = len(Vv)

    # Undirected edges in Vv
    und_edges = build_local_undirected_edges(Vv, out_adj)
    ev = len(und_edges)
    density = 2 * ev / (nv * (nv - 1)) if nv > 1 else 0.0

    # Connectivity via Union-Find, used by bridge-fragmentation.
    parent = {u: u for u in Vv}
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(x, y):
        parent[find(x)] = find(y)
    for (u, w) in und_edges: union(u, w)
    beta0_graph = len({find(u) for u in Vv})

    # Bridge-like fragmentation: components after removing v
    Vv_minus = Vv - {v}
    parent2 = {u: u for u in Vv_minus}
    def find2(x):
        while parent2[x] != x: parent2[x] = parent2[parent2[x]]; x = parent2[x]
        return x
    for (u, w) in und_edges:
        if u != v and w != v: parent2[find2(u)] = find2(w)
    cc_minus = len({find2(u) for u in Vv_minus}) if Vv_minus else 0
    bridge_frag = cc_minus - beta0_graph

    # Triangles and clustering around v
    neighbors_v = {u for u in Vv if (min(v,u), max(v,u)) in und_edges}
    kv = len(neighbors_v)
    tri_v = sum(
        1 for u in neighbors_v for w in neighbors_v
        if u < w and (min(u,w), max(u,w)) in und_edges
    )
    clust_v = 2 * tri_v / (kv * (kv - 1)) if kv >= 2 else 0.0

    # VR persistent homology on the local shortest-path metric.
    # For very large neighborhoods we cap the point cloud deterministically.
    Vv_vr = cap_local_vertices(v, Vv, out_adj, in_adj)
    und_edges_vr = build_local_undirected_edges(Vv_vr, out_adj)
    vr_diagram = vr_persistence_h0_h1(Vv_vr, und_edges_vr)

    # Betti numbers at epsilon=1, recovered from the VR filtration.
    beta0     = count_alive_bars(vr_diagram, 0, BETTI_EPSILON)
    beta1     = count_alive_bars(vr_diagram, 1, BETTI_EPSILON)
    beta1norm = beta1 / (nv + 1) if nv > 0 else 0.0

    # ── Family C: Global position (6) ─────────────────────────
    core_v    = core_dict.get(v,    0)
    cores_Vv  = [core_dict.get(u, 0) for u in Vv]
    core_mean = float(np.mean(cores_Vv))  if cores_Vv else 0.0
    core_max  = float(np.max(cores_Vv))   if cores_Vv else 0.0

    # Forman-Ricci curvature
    deg_und = {}
    for u in Vv:
        deg_und[u] = sum(1 for w in out_adj.get(u, set()) if w in Vv) + \
                     sum(1 for w in in_adj.get(u, set())  if w in Vv)

    kappa_edges = {(u,w): 4 - deg_und.get(u,0) - deg_und.get(w,0)
                   for (u,w) in und_edges}
    v_edges = [(u,w) for (u,w) in und_edges if u == v or w == v]
    kappa_v    = np.mean([kappa_edges[e] for e in v_edges]) if v_edges else 0.0
    kappa_mean = np.mean(list(kappa_edges.values()))        if kappa_edges else 0.0
    kappa_min  = np.min(list(kappa_edges.values()))         if kappa_edges else 0.0

    # ── Family D: VR persistence barcode features (6) ─────────
    h0_max_pers, h0_total_pers, h0_entropy = summarize_persistence(vr_diagram, 0)
    h1_max_pers, h1_total_pers, h1_entropy = summarize_persistence(vr_diagram, 1)

    return {
        # A: Directed (14)
        "A1": A1, "A2": A2, "D1": D1, "D2": D2,
        "asym": asym, "asym2": asym2,
        "div1": D1, "div2": D2, "exp_out": exp_out,
        "rec_out_count": rec_out_count, "rec_out_ratio": rec_out_ratio,
        "rec_in_count": rec_in_count,   "rec_in_ratio": rec_in_ratio,
        "peel_score": peel,
        # B: Undirected (10)
        "nv": nv, "ev": ev, "density": density,
        "beta0": beta0, "cc_minus": cc_minus, "bridge_frag": bridge_frag,
        "tri": tri_v, "clust": clust_v,
        "beta1": beta1, "beta1_norm": beta1norm,
        # C: Global position (6)
        "core": core_v, "core_mean": core_mean, "core_max": core_max,
        "kappa": kappa_v, "kappa_mean": kappa_mean, "kappa_min": kappa_min,
        # D: VR persistence barcode summaries (6)
        "h0_max_persistence": h0_max_pers,
        "h0_total_persistence": h0_total_pers,
        "h0_persistence_entropy": h0_entropy,
        "h1_max_persistence": h1_max_pers,
        "h1_total_persistence": h1_total_pers,
        "h1_persistence_entropy": h1_entropy,
    }

BATCH = 5000
n     = len(accounts)
PART_DIR = f"{OUT}/ftd_vr_parts"
os.makedirs(PART_DIR, exist_ok=True)

for start in range(0, n, BATCH):
    end = min(start + BATCH, n)
    part_path = f"{PART_DIR}/part_{start:07d}_{end:07d}.parquet"
    if os.path.exists(part_path):
        elapsed = time.time() - t0
        print(f"  {end:>7,}/{n:,}  checkpoint exists ({elapsed:.0f}s elapsed)", flush=True)
        continue

    rows = []
    for i in range(start, end):
        row = compute_ftd(accounts[i], out_adj, in_adj, core_dict)
        row["account_id"] = accounts[i]
        rows.append(row)

    pd.DataFrame(rows).to_parquet(part_path, index=False)
    elapsed = time.time() - t0
    rate    = end / elapsed
    eta     = (n - end) / rate if rate > 0 else 0
    print(f"  {end:>7,}/{n:,}  ({elapsed:.0f}s elapsed, ETA {eta:.0f}s)", flush=True)

part_files = [
    f"{PART_DIR}/part_{start:07d}_{min(start + BATCH, n):07d}.parquet"
    for start in range(0, n, BATCH)
]
ftd = pd.concat((pd.read_parquet(p) for p in part_files), ignore_index=True)
cols = ["account_id"] + [c for c in ftd.columns if c != "account_id"]
ftd = ftd[cols]
print(f"  FTD shape: {ftd.shape}  ({time.time()-t0:.1f}s)")
ftd.to_parquet(f"{OUT}/ftd_vr_features.parquet", index=False)

del out_adj, in_adj, core_dict; gc.collect()

# ── 6. Merge and save full feature matrix ─────────────────────
print("\n[6/6] Merging and saving full feature matrix …")
full = (labels
        .merge(base, on="account_id", how="left")
        .merge(ftd,  on="account_id", how="left")
        .fillna(0))
full.to_parquet(f"{OUT}/full_vr_features.parquet", index=False)

base_cols = [c for c in base.columns if c != "account_id"]
ftd_cols  = [c for c in ftd.columns  if c != "account_id"]
print(f"  Full matrix: {full.shape}  |  base={len(base_cols)}  ftd_vr={len(ftd_cols)}")
print(f"  Saved: {OUT}/full_vr_features.parquet  ({os.path.getsize(f'{OUT}/full_vr_features.parquet')/1e6:.1f} MB)")

del base, ftd, full, trans; gc.collect()
print("\nDone. Run 03_classify_significance.py next.")
