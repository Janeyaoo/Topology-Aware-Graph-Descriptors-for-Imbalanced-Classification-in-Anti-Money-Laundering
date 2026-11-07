"""Matched, label-blind capped graph geometry for the S10 development experiment.

No legacy feature builders are imported (they execute whole pipelines on import).
The new PH dictionary uses a shared cap40 BFS prefix, not historical cap80/500.
"""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import argparse, hashlib, heapq, json, time, shutil
from pathlib import Path
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from scipy import sparse
from numba import njit

ROOT=Path(__file__).resolve().parents[2]
ALT=ROOT/'Altman_experiments'
OUT=ALT/'results/account_s10_increment_20261008'
GMKE=['A1','A2','D1','D2','nv','ev','density','cc_minus','bridge_frag','tri','clust','beta1','beta1_norm','core','core_mean','core_max',
      'rec_out_count','rec_in_count','rec_out_ratio','rec_in_ratio','peel_score','kappa','kappa_mean','kappa_min','asym','asym2','exp_out','d_beta1']
P_NAMES=[f'ph_{h}hop_h{q}_{s}' for h in (2,3) for q in (0,1) for s in ('max','total','entropy')]
Q_NAMES=['q_n2','q_n','q_e1','q_t1','q_e2','q_t2']
CAP=40

def sha(path):
    d=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): d.update(b)
    return d.hexdigest()

def jsave(path,value):
    p=Path(path); tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n'); tmp.replace(p)

def prepare_geometry(out):
    out.mkdir(parents=True,exist_ok=True)
    protocol={'version':'S10_matched_cap40_v1','sampling':'weak undirected radius3; center first; remaining vertices in increasing BFS layer then lexicographic account_id; retain at most40; paths retained',
      'two_hop_PH':'subset of the shared sampled radius3 prefix with global BFS distance<=2; recompute shortest paths in this induced subgraph; at most40',
      'three_hop_PH':'full sampled induced graph; recompute shortest paths in this induced subgraph; at most40',
      'metric':'unweighted shortest path, no edge multiplicity/amount/time weights; self loops removed',
      'simplicial_complexes':'flag/VR triangles at epsilon1 and2 on identical radius3 sampled vertices',
      'PH':'F2 finite H0/H1 bars; connected H0=[0,1) count n-1; H1 birth1, death2 or3 (radius2 death2); GF2 boundary ranks/inclusion ranks; checked independently by explicit barcode reduction',
      'S10':'real signed standard-inner-product combinatorial L0(K1),L0(K2),L1(K1),L1(K2),true L1^(1,2) via restricted chains/Schur; each min positive eigenvalue/n and complete positive-spectrum entropy/log(positive rank)',
      'zero_convention':'no positive eigenvalues: min=entropy=0; positive rank1 entropy0; QC rank/nullity recorded',
      'controls':Q_NAMES,'node_cap':CAP,'radius':3,'geometry_uses_labels':False,
      'interpretation':'static whole-period transductive account recognition; development comparison on previously studied accounts; no chronological/independent confirmation',
      'historical_protocol_change':'P12u is newly recomputed under unified cap40, not old HI cap80 2hop or cap500->40 3hop; old results cannot be substituted',
      'sources':['https://arxiv.org/abs/2012.02808','https://arxiv.org/pdf/2312.07563']}
    p=out/'feature_protocol.json'
    if p.exists(): assert json.loads(p.read_text())==protocol
    else:
        jsave(p,protocol)
        for name in ('EXPERIMENT_RESULTS_SUMMARY.md','EXPERIMENT_RESULTS_SUMMARY_gen.py'):
            if (ROOT/name).is_file():
                shutil.copy2(ROOT/name,out/('before_'+name))
    for ds in ('hi_small','li_small'):
        folder=out/ds; folder.mkdir(exist_ok=True)
        gp=folder/'graph_csr.npz'
        if gp.exists(): continue
        t=time.monotonic()
        fp=ALT/'results'/ds/'features/full_features.parquet'
        full=pd.read_parquet(fp,columns=['account_id','label'])
        ids=pd.Index(sorted(full.account_id)); assert ids.is_unique
        trans=ALT/'data'/ds/f'{ds}_trans.parquet'
        df=pd.read_parquet(trans,columns=['from_id','to_id'])
        a=ids.get_indexer(df.from_id); b=ids.get_indexer(df.to_id)
        assert (a>=0).all() and (b>=0).all()
        mask=a!=b; a=a[mask]; b=b[mask]
        graph=sparse.coo_matrix((np.ones(2*len(a),dtype=np.int32),(np.r_[a,b],np.r_[b,a])),shape=(len(ids),len(ids))).tocsr()
        graph.data[:]=1; graph=graph.astype(np.int8); graph.sort_indices()
        sparse.save_npz(gp,graph,compressed=True)
        np.save(folder/'account_to_graph.npy',ids.get_indexer(full.account_id).astype(np.int32))
        pd.DataFrame({'graph_index':np.arange(len(ids)), 'account_id':ids}).to_parquet(folder/'graph_accounts.parquet',index=False)
        jsave(folder/'graph_manifest.json',{'n_nodes':len(ids),'n_undirected_edges':int(graph.nnz//2),'input_sha256':{str(trans.relative_to(ROOT)):sha(trans),str(fp.relative_to(ROOT)):sha(fp)},
          'artifact_sha256':{x.name:sha(x) for x in (gp,folder/'account_to_graph.npy',folder/'graph_accounts.parquet')},'runtime_seconds':time.monotonic()-t})
        print('GRAPH',ds,len(ids),graph.nnz//2,time.monotonic()-t,flush=True)

def prefix_vertices(root,indptr,indices):
    seen={int(root)}; vertices=[int(root)]; distances=[0]; frontier=[int(root)]; hit=0
    for depth in range(1,4):
        streams=[iter(indices[indptr[u]:indptr[u+1]]) for u in frontier]
        nxt=[]
        for w in heapq.merge(*streams):
            w=int(w)
            if w in seen: continue
            if len(vertices)==CAP: hit=1; break
            seen.add(w); vertices.append(w); distances.append(depth); nxt.append(w)
        if hit: break
        if len(vertices)==CAP:
            if depth<3:
                hit=int(any(int(w) not in seen for u in nxt for w in indices[indptr[u]:indptr[u+1]]))
            break
        frontier=nxt
        if not frontier: break
    return np.asarray(vertices,np.int32),np.asarray(distances,np.int8),hit

@njit(cache=True)
def induced(vertices,indptr,indices):
    n=len(vertices); adj=np.zeros((n,n),np.bool_)
    for i in range(n):
        lo0=indptr[vertices[i]]; hi0=indptr[vertices[i]+1]
        for j in range(i+1,n):
            value=vertices[j]; lo=lo0; hi=hi0
            while lo<hi:
                mid=(lo+hi)//2
                if indices[mid]<value: lo=mid+1
                else: hi=mid
            if lo<hi0 and indices[lo]==value: adj[i,j]=True; adj[j,i]=True
    return adj

def graph_key(adj): return bytes([len(adj)])+np.packbits(adj[np.triu_indices(len(adj),1)]).tobytes()

def triangle_bits(adj,edge_index=None,source=None):
    n=len(adj)
    if edge_index is None: edge_index={(i,j):k for k,(i,j) in enumerate(zip(*np.where(np.triu(adj,1))))}
    for i in range(n):
        for j in np.flatnonzero(adj[i,i+1:])+i+1:
            for k in np.flatnonzero(adj[i,j+1:] & adj[j,j+1:])+j+1:
                pairs=((i,j),(i,k),(j,k))
                if source is None: yield sum(1<<edge_index[p] for p in pairs)
                else: yield sum(1<<edge_index[p] for p in pairs if not source[p])

def gf2_rank(vectors):
    basis={}
    for v in vectors:
        while v:
            pivot=v.bit_length()-1
            if pivot in basis: v^=basis[pivot]
            else: basis[pivot]=v; break
    return len(basis)

def ph_summaries(adj,adj2=None,radius=3):
    n=len(adj); e=int(adj.sum()//2)
    beta=e-n+1-gf2_rank(triangle_bits(adj))
    assert beta>=0
    survive=0
    if radius==3 and beta:
        if adj2 is None:
            from account_s10_math import vr_epsilon2
            adj2=vr_epsilon2(adj)
        if int(adj2.sum())<n*(n-1):
            rank_all=gf2_rank(triangle_bits(adj2))
            rank_new=gf2_rank(triangle_bits(adj2,source=adj))
            survive=e-n+1-(rank_all-rank_new)
    assert 0<=survive<=beta
    a=beta-survive; b=survive; total=a+2*b
    h1entropy=0. if total==0 else -(a/total*np.log(1/total) if a else 0)-(2*b/total*np.log(2/total) if b else 0)
    return np.array([float(n>1),float(n-1),float(np.log(n-1)) if n>1 else 0.,float(2 if b else (1 if a else 0)),float(total),h1entropy]),(beta,survive)

_GRAPH=None; _ORDER=None; _OUT=None; _DS=None
def init_worker(out,ds):
    global _GRAPH,_ORDER,_OUT,_DS
    _OUT=Path(out); _DS=ds
    _GRAPH=sparse.load_npz(_OUT/ds/'graph_csr.npz')
    _ORDER=np.load(_OUT/ds/'account_to_graph.npy')

def chunk_worker(bounds):
    from account_s10_math import compute_s10,vr_epsilon2,FEATURE_NAMES
    start,stop=bounds; path=_OUT/_DS/'chunks'/f'{start:07d}_{stop:07d}.npz'
    if path.exists(): return {'start':start,'stop':stop,'reused':True}
    t=time.monotonic(); cache={}; p2cache={}; output=[]; qc=[]; samples=np.full((stop-start,CAP),-1,np.int32); depths=np.full((stop-start,CAP),-1,np.int8)
    misses=0; diagnostic={'min_raw_eigenvalue':0.,'max_boundary_annihilation':0.,'max_schur_kernel_coupling':0.,'near_zero_count':0,'persistent_methods_unique':{}}
    for offset,row in enumerate(range(start,stop)):
        vertices,dists,hit=prefix_vertices(_ORDER[row],_GRAPH.indptr,_GRAPH.indices)
        samples[offset,:len(vertices)]=vertices; depths[offset,:len(vertices)]=dists
        adj=induced(vertices,_GRAPH.indptr,_GRAPH.indices); n2=int((dists<=2).sum()); key=graph_key(adj)
        if key not in cache:
            result=compute_s10(adj,return_spectra=False)
            adj2=vr_epsilon2(adj)
            ph3,phq3=ph_summaries(adj,adj2,3)
            e1=result['qc']['n_edges_eps1']; e2=result['qc']['n_edges_eps2']
            t1=result['qc']['n_triangles_eps1']; t2=result['qc']['n_triangles_eps2']
            # operator QC names/dimensions read generically, preserving math module details.
            ops=result['qc']['operators']; opvals=[]
            for op in ops.values(): opvals += [op['rank'],op['nullity']]
            for op in ops.values():
                diagnostic['min_raw_eigenvalue']=min(diagnostic['min_raw_eigenvalue'],op.get('raw_min_eigenvalue',0.))
                diagnostic['max_boundary_annihilation']=max(diagnostic['max_boundary_annihilation'],op.get('boundary_annihilation_max',0.))
                diagnostic['max_schur_kernel_coupling']=max(diagnostic['max_schur_kernel_coupling'],op.get('schur_kernel_coupling_max',0.))
                diagnostic['near_zero_count']+=op.get('near_zero_eigenvalues',0)+op.get('schur_D_near_zero_eigenvalues',0)
            method=ops['pl12']['method']; diagnostic['persistent_methods_unique'][method]=diagnostic['persistent_methods_unique'].get(method,0)+1
            cache[key]=(np.asarray(result['values']),ph3,phq3,[e1,t1,e2,t2],opvals)
            misses+=1
        sval,ph3,phq3,counts,opvals=cache[key]
        small=adj[:n2,:n2]; skey=graph_key(small)
        if skey not in p2cache: p2cache[skey]=ph_summaries(small,radius=2)
        ph2,phq2=p2cache[skey]
        output.append(np.r_[ph2,ph3,n2,len(adj),counts,sval])
        qc.append([hit,int(dists.max()),*phq2,*phq3,*opvals])
        if offset and offset%1000==0:
            print(_DS,start,offset,'miss',misses,'seconds',round(time.monotonic()-t,1),flush=True)
    temp=path.with_suffix('.tmp.npz')
    np.savez_compressed(temp,values=np.asarray(output,np.float64),qc=np.asarray(qc,np.int16),vertices=samples,depths=depths)
    temp.replace(path)
    report={'start':start,'stop':stop,'unique_s10_graphs':misses,'unique_ph2_graphs':len(p2cache),'runtime_seconds':time.monotonic()-t,'sha256':sha(path),'numerical_diagnostics':diagnostic}
    jsave(path.with_suffix('.json'),report)
    return report

def build(out,ds,workers=4,limit=None):
    from account_s10_math import FEATURE_NAMES
    folder=out/ds; (folder/'chunks').mkdir(exist_ok=True)
    n=len(np.load(folder/'account_to_graph.npy')); total=n if limit is None else min(n,limit)
    chunk_size=10000 if limit is None else total
    bounds=[(i,min(total,i+chunk_size)) for i in range(0,total,chunk_size)]
    start=time.monotonic()
    if workers==1:
        init_worker(out,ds)
        for b in bounds: print(chunk_worker(b),flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers,initializer=init_worker,initargs=(str(out),ds)) as pool:
            for report in pool.map(chunk_worker,bounds): print(report,flush=True)
    if limit is not None: return
    values=[]; qcs=[]
    for a,b in bounds:
        with np.load(folder/'chunks'/f'{a:07d}_{b:07d}.npz') as f:
            assert f['values'].shape==(b-a,28)
            values.append(f['values']); qcs.append(f['qc'])
    values=np.concatenate(values); qc=np.concatenate(qcs)
    full=pd.read_parquet(ALT/'results'/ds/'features/full_features.parquet')
    basecols=list(pd.read_parquet(ALT/'results'/ds/'features/base_features.parquet').columns[1:])
    feature=full[['account_id','label',*basecols,*GMKE]].copy()
    for j,col in enumerate(P_NAMES+Q_NAMES+list(FEATURE_NAMES)): feature[col]=values[:,j]
    assert not feature.isna().any().any() and np.isfinite(feature.drop(columns=['account_id']).to_numpy()).all()
    feature.to_parquet(folder/'features.parquet',index=False,compression='zstd')
    qc_names=['cap_hit','effective_radius','ph2_beta1_f2','ph2_survive2_f2','ph3_beta1_f2','ph3_survive2_f2']+[f'{op}_{k}' for op in ('g1','g2','h1','h2','pl12') for k in ('rank','nullity')]
    qframe=pd.DataFrame(qc,columns=qc_names); qframe.insert(0,'account_id',full.account_id)
    qframe.to_parquet(folder/'geometry_qc.parquet',index=False,compression='zstd')
    duplicates=[]; names=P_NAMES+Q_NAMES+list(FEATURE_NAMES)
    for i in range(len(names)):
        for j in range(i):
            if np.allclose(values[:,i],values[:,j],rtol=0,atol=1e-10): duplicates.append([names[j],names[i]])
    jsave(folder/'feature_qc.json',{'n_accounts':n,'cap_hit_fraction':float(qc[:,0].mean()),'radius_counts':qframe.effective_radius.value_counts().sort_index().to_dict(),
      'ph2_equals_ph3_fraction':float(np.all(values[:,:6]==values[:,6:12],axis=1).mean()),'full_column_equalities_atol1e_10':duplicates,
      'feature_unique_counts':{col:int(feature[col].nunique()) for col in names},
      'field_discrepancy_source_count':int((qframe.ph3_beta1_f2!=qframe.h1_nullity).sum()),'field_discrepancy_persistent_count':int((qframe.ph3_survive2_f2!=qframe.pl12_nullity).sum()),
      'runtime_seconds':time.monotonic()-start})
    sources=[Path(__file__),ALT/'code/account_s10_math.py',out/'feature_protocol.json',folder/'graph_manifest.json',ALT/'results'/ds/'features/full_features.parquet']
    jsave(folder/'feature_manifest.json',{'status':'complete','n_accounts':n,'n_positive':int(full.label.sum()),'n_features':len(feature.columns)-2,'feature_columns':list(feature.columns[2:]),
      'input_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sources},'artifacts_sha256':{p.name:sha(p) for p in [folder/'features.parquet',folder/'geometry_qc.parquet',folder/'feature_qc.json']},'runtime_seconds':time.monotonic()-start})
    print('FEATURES_COMPLETE',ds,n,time.monotonic()-start,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['prepare','build'],required=True);p.add_argument('--out',type=Path,default=OUT);p.add_argument('--dataset',choices=['hi_small','li_small']);p.add_argument('--workers',type=int,default=4);p.add_argument('--limit',type=int)
    a=p.parse_args()
    if a.phase=='prepare': prepare_geometry(a.out)
    else:
        assert a.dataset; build(a.out,a.dataset,a.workers,a.limit)
