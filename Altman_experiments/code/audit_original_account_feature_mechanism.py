import ast
import hashlib
import json
from pathlib import Path
from datetime import datetime
import gc
import numpy as np
import pandas as pd

ROOT=Path('.')
FAMILIES={
 'A_directed_flow':['A1','A2','D1','D2','asym','asym2','div1','div2','exp_out','rec_out_count','rec_out_ratio','rec_in_count','rec_in_ratio','peel_score'],
 'B_undirected_topology':['nv','ev','density','beta0','cc_minus','bridge_frag','tri','clust','beta1','beta1_norm'],
 'C_core_and_local_curvature':['core','core_mean','core_max','kappa','kappa_mean','kappa_min'],
 'D_one_two_hop_cycle_rank':['beta0_1','beta0_2','d_beta0','beta1_1','beta1_2','d_beta1']}
FTD=sum(FAMILIES.values(),[])
BASIC=['A1','A2','D1','D2','nv','ev','density','tri','clust','core']
BROAD=['A1','A2','D1','D2','nv','ev','density','cc_minus','bridge_frag','tri','clust','beta1','beta1_norm','core','core_mean','core_max']
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
 return h.hexdigest()
def rel(p):return str(p.relative_to(ROOT))
report={'created_at':datetime.now().astimezone().isoformat(), 'read_only_project_files':True,
 'families':FAMILIES,'family_dimensions':{k:len(v) for k,v in FAMILIES.items()},'cache_checks':{},'sources':{},
 'control_recommendation':{'basic10':BASIC,'basic_total_features':26,'basic10_train_in_new_protocol':False,'broad16':BROAD,'broad_total_features':32,'selected_main_control':'broad16',
 'reason':'Prespecify a broader conventional ego-network topology/core comparator including cycle rank, its normalization, fragmentation and neighbor core statistics; selected from the same original cached feature matrices without labels or rebuilding. The 10-column version is a selected basic-statistics control, not an exhaustive graph baseline.',
 'boundary':'No PageRank, betweenness, eigenvector centrality, graph embeddings or alternative graph estimators are tested. Keep curvature/reconvergence features out of the conventional control; retain the original 36-column family mapping for faithful ablation.'},
 'multiscale_definition':{'radii':[1,2], 'graph':'Undirected induced subgraphs on center plus directed upstream/downstream radius-1 or radius-2 reachable accounts',
 'beta0':'Union-find connected component count', 'beta1':'cycle rank E-V+beta0; no simplex filling',
 'd_beta0':'beta0_2-beta0_1','d_beta1':'beta1_2-beta1_1',
 'not_vr_persistent_homology':True,'not_temporal_filtration':True},
 'source_implementation_notes':['A2/D2 remove the center and existing one-hop neighbors; they count directed second-shell reachability.',
 'Adjacency sets collapse repeated transactions to account edges; self-loops remain in out/in adjacency and induced edges.',
 'k-core is computed once on a loop-removed undirected global graph.',
 'kappa uses 4-deg(u)-deg(w), with local deg_und computed as in-count plus out-count inside Vv; reciprocal edges may count twice. This should be described as the cached local degree-based curvature proxy, not an exact external standard Ricci estimator.',
 'Family C mixes global k-core values with local curvature summaries; do not call all six dimensions global position.']}
for name in ('02_build_features.py','ibm_pipeline.py'):
 p=ROOT/'Altman_experiments/code'/name
 t=ast.parse(p.read_text())
 fn=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='compute_ftd')
 ret=next(n for n in ast.walk(fn) if isinstance(n,ast.Return) and isinstance(n.value,ast.Dict))
 keys=[ast.literal_eval(k) for k in ret.value.keys]
 assert keys==FTD
 report['sources'][rel(p)]={'sha256':digest(p),'compute_ftd_line':fn.lineno,'return_line':ret.lineno,'column_order_matches':True}
for ds in ('hi_small','li_small'):
 directory=ROOT/'Altman_experiments/results'/ds/'features'
 paths={k:directory/f'{k}_features.parquet' for k in ('base','ftd','full')}
 base=pd.read_parquet(paths['base']);ftd=pd.read_parquet(paths['ftd']);full=pd.read_parquet(paths['full'])
 assert [c for c in ftd if c!='account_id']==FTD
 bcols=[c for c in base if c!='account_id']
 assert len(bcols)==16 and full.columns.tolist()==['account_id','label']+bcols+FTD
 assert base.account_id.is_unique and ftd.account_id.is_unique and full.account_id.is_unique
 pd.testing.assert_series_equal(base.account_id,ftd.account_id)
 pd.testing.assert_series_equal(base.account_id,full.account_id)
 pd.testing.assert_frame_equal(full[bcols],base[bcols]);pd.testing.assert_frame_equal(full[FTD],ftd[FTD])
 arr=ftd[FTD].to_numpy(dtype=np.float64)
 assert np.isfinite(arr).all()
 constants={c:float(ftd[c].iloc[0]) for c in FTD if ftd[c].nunique(dropna=False)==1}
 grouped={}
 for j,c in enumerate(FTD):
  h=hashlib.sha256(np.ascontiguousarray(arr[:,j]).tobytes()).hexdigest();grouped.setdefault(h,[]).append(c)
 exact=[x for x in grouped.values() if len(x)>1]
 nonconstant_duplicate=[x for x in exact if x[0] not in constants]
 affine=[]
 varying=[c for c in FTD if c not in constants]
 for i,a in enumerate(varying):
  for b in varying[i+1:]:
   difference=ftd[a].to_numpy(dtype=float)-ftd[b].to_numpy(dtype=float)
   if np.all(difference==difference[0]) and difference[0]!=0:
    affine.append({'left':a,'right':b,'constant_left_minus_right':float(difference[0])})
 identities={
  'div1==D1':np.array_equal(ftd.div1,ftd.D1), 'div2==D2':np.array_equal(ftd.div2,ftd.D2),
  'beta0_2==beta0':np.array_equal(ftd.beta0_2,ftd.beta0),'beta1_2==beta1':np.array_equal(ftd.beta1_2,ftd.beta1),
  'beta1_1==tri':np.array_equal(ftd.beta1_1,ftd.tri),
  'd_beta0==beta0_2-beta0_1':np.array_equal(ftd.d_beta0,ftd.beta0_2-ftd.beta0_1),
  'd_beta1==beta1_2-beta1_1':np.array_equal(ftd.d_beta1,ftd.beta1_2-ftd.beta1_1),
  'beta1==ev-nv+beta0':np.array_equal(ftd.beta1,ftd.ev-ftd.nv+ftd.beta0),
  'bridge_frag==cc_minus-beta0':np.array_equal(ftd.bridge_frag,ftd.cc_minus-ftd.beta0)}
 report['cache_checks'][ds]={'n_samples':len(full),'n_positive':int(full.label.sum()),'base_columns':bcols,'ftd_columns':FTD,
 'base_ftd_full_consistency':True,'account_order_and_uniqueness':True,'all_ftd_finite':True,'constant_columns':constants,
 'exact_duplicate_groups':exact,'nonconstant_exact_duplicate_groups':nonconstant_duplicate,
 'nonzero_affine_offset_duplicates':affine,'nonconstant_unique_vector_count':len(varying)-sum(len(x)-1 for x in nonconstant_duplicate),
 'family_varying_dimensions':{k:sum(c not in constants for c in cols) for k,cols in FAMILIES.items()},
 'identities':{k:bool(v) for k,v in identities.items()},
 'files':{rel(p):{'sha256':digest(p),'size_bytes':p.stat().st_size} for p in paths.values()}}
 if ds=='hi_small':
  mirror=ROOT/'Altman_experiments/results/features'
  mirrors={}
  for k in paths:
   old=mirror/f'{k}_features.parquet'
   original=pd.read_parquet(old)
   pd.testing.assert_frame_equal({'base':base,'ftd':ftd,'full':full}[k],original)
   mirrors[rel(old)]={'sha256':digest(old),'dataframe_equal_to_hi_small_cache':True,'bytewise_equal':digest(old)==digest(paths[k])}
   del original
  report['cache_checks'][ds]['legacy_feature_mirror_checks']=mirrors
 del base,ftd,full,arr;gc.collect()
report['both_datasets_same_column_order']=True
report['frozen_ablation_interpretation']={'preserve_original_nominal_dimensions':True,'deduplicate_or_reassign_original_families':False,'drop_family_effect':'Conditional marginal predictive performance of that cached column group under the frozen learner/protocol, not causal attribution or complete information removal','family_D_information':'In both complete caches beta0_1=beta0_2=1,d_beta0=0,beta1_1=tri,beta1_2=beta1,d_beta1=beta1-tri. Thus the entire six-column family D is determined by the existing family B columns; it does not add independent topology information. Drop-D can still alter representation, feature subsampling and fitted predictions.'}

out=Path('/private/tmp/ibm_original_ftd_feature_audit_20261007.json')
out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'report':str(out),'family_dimensions':report['family_dimensions'],'datasets':{k:{name:v[name] for name in ['n_samples','n_positive','constant_columns','nonconstant_exact_duplicate_groups','nonzero_affine_offset_duplicates','nonconstant_unique_vector_count','family_varying_dimensions']} for k,v in report['cache_checks'].items()}},ensure_ascii=False,indent=2))
