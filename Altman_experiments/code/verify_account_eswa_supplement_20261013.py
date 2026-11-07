"""Independent read-only validation of completed ESWA supplement account models.

Does not import any training runner or call fit. Reconstructs every candidate's
AP/AUROC from rank/tie audits, rescoring only retained selected/threshold/final
models. Incomplete batches produce single-fold receipts and partial manifests,
never completed-setting means. Neural tensors are evaluated with independent torch affine operations and CSR means.
"""
from __future__ import annotations
import argparse,ast,copy,gc,hashlib,itertools,json,os,time,traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','2')
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score,roc_auc_score,precision_recall_curve,
 accuracy_score,balanced_accuracy_score,precision_score,recall_score,f1_score,confusion_matrix)
ROOT=Path(__file__).resolve().parents[2]
ALT=ROOT/'Altman_experiments'
DEFAULT_OUT=ALT/'results/account_eswa_supplement_20261013'
MAIN=ALT/'results/account_gmkep56_tuned_20261008'
SCRIPT=Path(__file__).resolve()
MODELS=('lightgbm','xgboost','random_forest')
METRICS=('ap','auc','precision','recall','f1','accuracy','balanced_accuracy','specificity','fpr','fnr','tn','fp','fn','tp','n_test','n_positive')
POLICIES=('fixed_0.5','inner_max_f1')
TOL=1e-12
HASH_CACHE={}
def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def rel(p):return Path(p).resolve().relative_to(ROOT).as_posix()
def sha(p):
 p=Path(p);s=p.stat();key=(s.st_size,s.st_mtime_ns)
 if p in HASH_CACHE and HASH_CACHE[p][0]==key:return HASH_CACHE[p][1]
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 value=h.hexdigest();HASH_CACHE[p]=(key,value);return value
def ash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def require(ok,message):
 if not ok:raise ValueError(message)
def close(a,b,what,tol=TOL):
 a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
 require(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all() and np.allclose(a,b,rtol=0,atol=tol),what)
 return float(np.max(np.abs(a-b),initial=0))
def path(name):
 p=(ROOT/name).resolve();require(p.is_relative_to(ROOT) and p.is_file(),'Missing/uncontained artifact '+str(name));return p
def binding(mapping,report=None):
 for name,expected in mapping.items():
  p=path(name);require(sha(p)==expected,'Hash mismatch '+name)
  if report is not None:report['verified_artifact_sha256'][name]=expected

def unpack(value,n):
 require(value.dtype==np.uint8 and value.ndim==1 and len(value)==(n+7)//8,'Bad packed bit shape')
 b=np.unpackbits(value,bitorder='little');require(not b[n:].any(),'Nonzero audit padding');return b[:n]
def rank_metrics(z,y):
 n=int(z['n']);w=int(z['width']);pos=int(z['n_positive'])
 require(n==len(y) and n>1 and 0<pos<n and pos==int(y.sum()) and w==max(1,(n-1).bit_length()),'Rank counts/width')
 bits=unpack(z['packed_order'],n*w).reshape(n,w)
 order=np.sum(bits.astype(np.uint64)<<np.arange(w,dtype=np.uint64),axis=1).astype(np.int64)
 require(np.array_equal(np.sort(order),np.arange(n)),'Rank is not full permutation')
 labels=unpack(z['packed_labels'],n);ends=unpack(z['packed_tie_end'],n)
 require(np.array_equal(labels,y[order]) and ends[-1]==1,'Rank truth/ties')
 require(not np.any((ends[:-1]==0)&(order[:-1]>=order[1:])),'Tie order not stable')
 at=np.flatnonzero(ends);tp=np.cumsum(labels,dtype=np.int64)[at]
 dp=np.diff(np.r_[0,tp]);dn=np.diff(np.r_[0,at+1])-dp
 ap=float(np.sum(dp*(tp/(at+1)))/pos)
 below=n-pos-(np.cumsum(dn)-dn)-dn
 auc=float(np.sum(dp*(below+.5*dn))/(pos*(n-pos)))
 return ap,auc,order,ends

def threshold(y,s):
 # Independent sklearn PR formulation. Certify tolerance cannot merge two
 # distinct attainable rational F1 values at the registered sample size.
 require(2./float(len(y)+int(y.sum()))**2>TOL+16*np.finfo(float).eps,'Threshold tolerance uncertified')
 p,r,t=precision_recall_curve(y,s,drop_intermediate=False)
 f=np.divide(2*p[:-1]*r[:-1],p[:-1]+r[:-1],out=np.zeros_like(p[:-1]),where=p[:-1]+r[:-1]>0)
 best=int(np.flatnonzero(abs(f-f.max())<=TOL)[0]);return float(t[best]),float(f[best])
def metrics(y,s,pred):
 tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
 return dict(ap=float(average_precision_score(y,s)),auc=float(roc_auc_score(y,s)),
 precision=float(precision_score(y,pred,zero_division=0)),recall=float(recall_score(y,pred,zero_division=0)),
 f1=float(f1_score(y,pred,zero_division=0)),accuracy=float(accuracy_score(y,pred)),
 balanced_accuracy=float(balanced_accuracy_score(y,pred)),specificity=float(tn/(tn+fp)),
 fpr=float(fp/(tn+fp)),fnr=float(fn/(tp+fn)),tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),n_test=len(y),n_positive=int(y.sum()))
def fingerprint(name,m):
 h=hashlib.sha256();h.update(json.dumps(m.get_params(),sort_keys=True,default=str).encode())
 if name=='lightgbm':h.update(m.booster_.model_to_string(num_iteration=-1).encode())
 elif name=='xgboost':h.update(bytes(m.get_booster().save_raw(raw_format='ubj')))
 else:
  for t in m.estimators_:
   state=t.tree_.__getstate__()
   for k in state['nodes'].dtype.names:h.update(k.encode());h.update(np.ascontiguousarray(state['nodes'][k]).tobytes())
   h.update(np.ascontiguousarray(state['values']).tobytes())
 return h.hexdigest()

class Validator:
 def __init__(self,out):
  self.out=Path(out).resolve();self.tree=self.out/'trees';self.dest=self.out/'validation/tree'
  self.p=read(self.tree/'protocol.json');self.psha=sha(self.tree/'protocol.json')
  self.c=read(self.tree/'candidates.json');self.inputs=read(self.tree/'dataset_inputs.json')
  self.splits=read(self.tree/'split_manifest.json');self.cache={};self.matrices={}
  self.src={};self.source_gate()
 def source_gate(self):
  p=self.p
  for stem in ('candidates','dataset_inputs','split_manifest','source_manifest'):
   require(sha(self.tree/(stem+'.json'))==p[stem+'_sha256'],'Protocol registry '+stem)
  registered_sources=read(self.tree/'source_manifest.json')['source_sha256'];binding(registered_sources)
  self.src.update(registered_sources)
  require(sha(MAIN/'protocol.json')==p['old_experiment_protocol_sha256'],'Original protocol')
  inherited=path('Altman_experiments/code/run_account_gmkep_tuning.py')
  helper=path('Altman_experiments/code/run_account_feature_mechanism.py')
  require(sha(inherited)==p['inherited_trainer_sha256'] and sha(helper)==p['helper_sha256'],'Inherited source hashes')
  runner=path('Altman_experiments/code/run_account_eswa_supplement_trees_20261013.py')
  require(sha(runner)==p['trainer_sha256'],'Registered supplement runner')
  require(read(MAIN/'run_manifest.json')['status']=='complete' and read(MAIN/'validation_models.json')['passed'] is True,'Original completed evidence')
  require(self.c==read(MAIN/'candidates.json') and all([x['candidate_id'] for x in self.c[m]]==list(range(12)) for m in MODELS),'Shared 12 candidate IDs')
  registry=read(ALT/'results/eswa_supplement_design_20261013/feature_registry.json')
  planned={r['arm_id']:r for r in registry['tree_arms'] if r['status']=='planned'}
  require(set(p['feature_columns'])==set(planned) and all(p['feature_columns'][a]==r['columns'] and p['feature_sets'][a]==r['n_features'] for a,r in planned.items()),'All planned arm columns/dimensions')
  old=read(MAIN/'protocol.json')
  scientific=['seeds','outer_n_splits','inner_n_splits','candidates_per_model','selection_metric','candidate_tie_break','threshold_policies','prediction_rule','threshold_selection','rounds_aggregation','early_stopping','development_threshold_split','inner_random_state','early_random_state','search_model_random_state','stage_model_random_state','class_weighting','summary_std_ddof','rf_prefix_reuse','prediction_order']
  require(all(p[k]==old[k] for k in scientific),'Inherited scientific selection contract')
  require(len(self.splits)==50 and p['seeds']==list(range(5)) and p['outer_n_splits']==5 and p['inner_n_splits']==3,'Registered full split dimensions')
  # Frozen source AST audit, without importing or executing the fitting source.
  fun={x.name:x for x in ast.parse(inherited.read_text()).body if isinstance(x,ast.FunctionDef)}
  fitting=fun['fit'];calls=[x for x in ast.walk(fitting) if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr=='fit']
  require(len(calls)==3 and all([ast.unparse(a) for a in x.args]==['X[rows]','y[rows]'] for x in calls),'AST fit only fitting rows')
  text=ast.unparse(fitting)
  require("compute_sample_weight('balanced', y[rows])" in text and 'sample_weight_eval_set' not in text and 'eval_sample_weight' not in text,'AST fit-only weighting/eval unweighted')
  stages=[x for x in ast.walk(fun['strategy_fit']) if isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id=='fit']
  require({tuple(ast.unparse(a) for a in x.args) for x in stages}=={('name','X','y','D','None','params','random_state','rounds'),('name','X','y','train','None','params','random_state','rounds')},'AST D/T/train disjoint staging')
  text=ast.unparse(fun['search']);require('outer_test' not in text and 'threshold_validation' not in text,'AST search isolated')
  self.src.update({rel(inherited):sha(inherited),rel(helper):sha(helper),rel(runner):sha(runner)})
 def dataset(self,ds):
  if ds in self.cache:return self.cache[ds]
  inp=self.inputs[ds];binding(inp['cache_sha256']);binding(inp['source_sha256'])
  x=np.load(path(inp['feature_array_path']),mmap_mode='r');y=np.load(path(inp['labels_array_path']),mmap_mode='r')
  ids=pd.read_parquet(path(inp['accounts_path']));counts=np.load(path(inp['sample_counts_path']),mmap_mode='r')
  require(x.shape==(inp['n_samples'],56) and x.dtype==np.float32 and y.dtype==np.int8 and int(y.sum())==inp['n_positive'],'Dataset dimensions/truth')
  require(np.array_equal(ids.row_index,np.arange(len(y))) and ids.account_id.is_unique and np.array_equal(ids.label,y),'Account identity/labels')
  # Independent source parquet reconstruction, not runner's matrix selector.
  raw=pd.read_parquet(path(inp['source_features']),columns=['account_id','label',*inp['columns']])
  require(np.array_equal(raw.account_id,ids.account_id) and np.array_equal(raw.label,y),'Source account rows')
  require(np.array_equal(raw[inp['columns'][:56]].to_numpy(np.float32),x) and np.array_equal(raw[inp['columns'][56:]].to_numpy(np.float32),counts),'Main/count cache source equivalence')
  del raw;self.cache[ds]=(x,y,ids,counts);return self.cache[ds]
 def matrix(self,ds,arm):
  key=(ds,arm)
  if key in self.matrices:return self.matrices[key]
  # Keep one selected matrix rather than accumulate every arm during scans.
  self.matrices.clear();gc.collect()
  base,y,ids,counts=self.dataset(ds);names=self.inputs[ds]['columns'];columns=self.p['feature_columns'][arm];reg=None
  if arm in self.p['deferred_input_arms']:
   kind='gfp' if arm=='base_gfp_account_stream' else 'q10';rp=self.tree/ds/'input_registrations'/(kind+'.json');reg=read(rp)
   require(reg['passed'] and reg['dataset']==ds and reg['protocol_sha256']==self.psha and arm in reg['arms'],'Deferred registration scope')
   binding(reg['artifact_sha256']);require(read(path(reg['acceptance_path']))['passed'] is True,'Input acceptance')
   extra=np.load(path(reg['array_path']),mmap_mode='r')
   source=pd.read_parquet(path(reg['source_path']))
   require(np.array_equal(source.account_id,ids.account_id) and np.array_equal(source[reg['columns']].to_numpy(np.float32),extra),'Deferred source identity/matrix')
   require('label' not in source and extra.shape==(len(y),len(reg['columns'])),'Deferred schema')
   if kind=='gfp':columns=names[:16]+reg['columns'];pool=np.column_stack((base[:,:16],extra));pool_names=columns
   else:
    require(np.array_equal(extra[:,0],counts[:,0]) and np.array_equal(extra[:,5],counts[:,1]),'Q10 sampled count agreement')
    pool=np.column_stack((base,extra));pool_names=names[:56]+reg['columns']
   X=np.ascontiguousarray(pool[:,[pool_names.index(c) for c in columns]],dtype=np.float32);del pool,source
   reg={'path':rel(rp),'sha256':sha(rp)}
  else:
   pool=np.column_stack((base,counts));X=np.ascontiguousarray(pool[:,[names.index(c) for c in columns]],dtype=np.float32);del pool
  require(len(columns)==len(set(columns)) and np.isfinite(X).all(),'Finite unique feature columns')
  self.matrices[key]=(X,columns,reg);return self.matrices[key]
 def model(self,name,pname,X,rows,expected,params,state,rounds,digest,early,r):
  q=path(pname);key=rel(q)
  if key in r['physical_models_rescored']:return
  m=joblib.load(q);actual=m.get_params();effective={**params,'random_state':state}
  if not early:effective['n_estimators']=rounds
  if name=='xgboost' and early:effective['early_stopping_rounds']=30
  types=dict(lightgbm='LGBMClassifier',xgboost='XGBClassifier',random_forest='RandomForestClassifier')
  require(type(m).__name__==types[name] and m.n_features_in_==X.shape[1] and np.array_equal(m.classes_,[0,1]) and all(actual.get(k)==v for k,v in effective.items()),'Loaded model parameters/input scope '+key)
  if name=='lightgbm':
   require(int(m.best_iteration_ or m.n_estimators)==rounds and (1<=m.booster_.num_trees()<=params['n_estimators'] if early else m.booster_.num_trees()==rounds),'LGB rounds')
  elif name=='xgboost':
   n=m.get_booster().num_boosted_rounds();require((m.best_iteration+1==rounds and rounds<=n<=params['n_estimators']) if early else n==rounds,'XGB rounds')
   require(actual.get('early_stopping_rounds')==(30 if early else None),'XGB early only search')
  else:
   require(len(m.estimators_)==m.n_estimators==rounds,'RF trees')
   seeds=np.random.RandomState(state).randint(np.iinfo(np.int32).max,size=rounds)
   require(np.array_equal(seeds,[t.random_state for t in m.estimators_]),'RF prefix/bootstrap sequence')
  require(fingerprint(name,m)==digest,'Semantic model digest')
  jobs=m.n_jobs
  try:
   if name=='random_forest':m.n_jobs=1
   score=np.ascontiguousarray(m.predict_proba(X[rows])[:,1],dtype=np.float64)
  finally:m.n_jobs=jobs
  diff=close(score,expected,'Retained model full saved probabilities '+key)
  # Exact raw bytes expected by the frozen deterministic tree prediction contract.
  require(ash(score)==ash(expected),'Rescored probability bytes differ '+key)
  r['physical_models_rescored'][key]={'sha256':sha(q),'max_abs_diff':diff,'rows':len(rows)}
  del m,score;gc.collect()
 def fold(self,ds,name,arm,seed,fold,skip=True):
  stem=f'{name}_{arm}_seed{seed}_fold{fold}';fpath=self.tree/ds/'folds'/(stem+'.json');receipt=self.dest/ds/(stem+'.json')
  if skip and receipt.exists():
   old=read(receipt)
   if old.get('passed') and old.get('verifier_sha256')==sha(SCRIPT) and old.get('fold_done_sha256')==sha(fpath):
    binding(old['verified_artifact_sha256']);return old
  r=dict(passed=False,status='validating_single_fold',dataset=ds,model=name,feature_set=arm,seed=seed,fold=fold,
   started_at=now(),verifier_sha256=sha(SCRIPT),protocol_sha256=self.psha,verified_artifact_sha256={},physical_models_rescored={},candidate_rank_audits_reconstructed=0)
  try:
   require(name in MODELS and arm in self.p['feature_columns'] and seed in range(5) and fold in range(5),'Registered fold identity')
   d=read(fpath);require([d[k] for k in ['dataset','model','feature_set','seed','fold']]==[ds,name,arm,seed,fold] and d['protocol_sha256']==self.psha,'DONE identity/protocol')
   binding(d['artifact_sha256'],r);r['verified_artifact_sha256'][rel(fpath)]=sha(fpath)
   X,columns,extra=self.matrix(ds,arm);_,y,ids,_=self.dataset(ds)
   require(d['feature_columns']==columns and d['n_features']==len(columns) and d['input_registration']==extra,'Registered arm input')
   sp=path(d['split_path']);sb=self.splits[rel(sp)];require(sha(sp)==sb['sha256']==sb['original_split_sha256'] and sha(path(sb['original_split_path']))==sb['original_split_sha256'],'Split copied source identity')
   with np.load(sp,allow_pickle=False) as z:a={k:z[k] for k in z.files}
   require({k:ash(v) for k,v in a.items()}==sb['arrays_sha256'],'Split array hashes')
   train,test,D,T=(a[k] for k in ['outer_train','outer_test','development','threshold_validation'])
   require(np.array_equal(np.sort(np.r_[train,test]),np.arange(len(y))) and not np.intersect1d(train,test).size and np.array_equal(np.sort(np.r_[D,T]),np.sort(train)) and not np.intersect1d(D,T).size,'Split partitions')
   for j in range(3):
    it,sc,ft,ev=(a[f'{k}_{j}'] for k in ['inner_train','inner_score','early_fit','early_validation'])
    require(np.array_equal(np.sort(np.r_[it,sc]),np.sort(D)) and not np.intersect1d(it,sc).size and np.array_equal(np.sort(np.r_[ft,ev]),np.sort(it)) and not np.intersect1d(ft,ev).size,'Inner partitions')
   require(np.array_equal(np.sort(np.concatenate([a[f'inner_score_{j}'] for j in range(3)])),np.sort(D)),'Three inner scoring folds complete development coverage')
   sr=read(path(d['search_result_path']));binding(sr['artifact_sha256'],r)
   require(sr['protocol_sha256']==self.psha and [sr[k] for k in ['dataset','model','feature_set','seed','fold']]==[ds,name,arm,seed,fold],'Search identity')
   records=sr['candidate_records'];require(len(records)==36 and {(v['candidate_id'],v['inner_fold']) for v in records}==set(itertools.product(range(12),range(3))),'Complete candidate rank grid')
   for v in records:
    cid,j=v['candidate_id'],v['inner_fold'];cand=self.c[name][cid];params=cand['params'];fit=a[f'inner_train_{j}'] if name=='random_forest' else a[f'early_fit_{j}'];early=None if name=='random_forest' else a[f'early_validation_{j}'];sc=a[f'inner_score_{j}'];state=40000+1000*seed+10*fold+j
    require(v['effective_params']=={**params,'random_state':state} and v['model_random_state']==state,'Candidate effective params/seed')
    require(v['fit_rows_sha256']==ash(fit) and v['early_rows_sha256']==(None if early is None else ash(early)) and v['score_rows_sha256']==ash(sc),'Candidate fit/early/score rows')
    require((v['fit_n'],v['fit_positive'],v['score_n'],v['score_positive'])==(len(fit),int(y[fit].sum()),len(sc),int(y[sc].sum())),'Fit/score class counts')
    require(1<=v['selected_rounds']<=params['n_estimators'],'Candidate rounds cap')
    apath=path(v['audit_path']);require(sha(apath)==v['audit_sha256'],'Rank audit hash')
    with np.load(apath,allow_pickle=False) as z:
     ap,auc,_,_=rank_metrics(z,y[sc]);require(str(z['raw_score_sha256'].item())==v['raw_score_sha256'] and str(z['score_rows_sha256'].item())==ash(sc),'Rank raw/row digests')
    close([ap,auc],[v['ap'],v['auc']],'Independent candidate ranking metrics');r['candidate_rank_audits_reconstructed']+=1
    group=cand.get('profile_id',cid);peers=[c for c in self.c[name] if c.get('profile_id',c['candidate_id'])==group];parent=max(peers,key=lambda c:c['params']['n_estimators'])
    require(v['physical_fit_id']==f'{stem}_group{group}_inner{j}' and v['physical_fit_charged']==(cid==parent['candidate_id']),'RF shared fit identity')
    if name=='random_forest':require(v['selected_rounds']==params['n_estimators'] and all({k:v for k,v in c['params'].items() if k!='n_estimators'}=={k:v for k,v in params.items() if k!='n_estimators'} for c in peers),'RF prefix profile structure')
    groupfile=path(rel(self.tree/ds/'search'/stem/f'group{group}_inner{j}.json'));groupval=read(groupfile)
    expected=[x for x in records if x['inner_fold']==j and self.c[name][x['candidate_id']].get('profile_id',x['candidate_id'])==group]
    require(groupval['protocol_sha256']==self.psha and groupval['records']==expected,'Resume group exact records');r['verified_artifact_sha256'][rel(groupfile)]=sha(groupfile)
   summaries=[]
   for cid in range(12):
    rows=[v for v in records if v['candidate_id']==cid];summaries.append(dict(candidate_id=cid,mean_ap=float(np.mean([v['ap'] for v in rows])),std_ap=float(np.std([v['ap'] for v in rows],ddof=1)),rounds=int(np.median([v['selected_rounds'] for v in rows]))))
   require([v['candidate_id'] for v in sr['candidate_summary']]==list(range(12)),'Summary IDs')
   for actual,expected in zip(sr['candidate_summary'],summaries):close([actual[k] for k in ['mean_ap','std_ap','rounds']],[expected[k] for k in ['mean_ap','std_ap','rounds']],'Candidate mean/SD/median')
   selected=min(summaries,key=lambda v:(-v['mean_ap'],v['candidate_id']));cid=selected['candidate_id']
   require(sr['selected_candidate_id']==cid and sr['selected_rounds']==selected['rounds'] and sr['anchor_rounds']==summaries[0]['rounds'] and sr['representative_inner_fold']==0,'Selection AP/exact tie/median')
   close(sr['selected_mean_ap'],selected['mean_ap'],'Selected mean AP')
   require(sr['physical_search_fits']==(18 if name=='random_forest' else 36)==sum(v['physical_fit_charged'] for v in records),'Physical search counts')
   for dest,src in [('search_fit_seconds','fit_seconds'),('search_predict_seconds','predict_seconds'),('search_audit_seconds','audit_seconds')]:close(sr[dest],sum(v[src] for v in records),'Search time sum',1e-8)
   representative=next(v for v in records if v['candidate_id']==cid and v['inner_fold']==0);sc=a['inner_score_0']
   with np.load(path(sr['representative_score_path']),allow_pickle=False) as z:
    require(set(z.files)=={'row_index','y_true','y_score'} and np.array_equal(z['row_index'],sc) and np.array_equal(z['y_true'],y[sc]) and z['y_score'].dtype==np.float64,'Representative score identity');score=z['y_score'].copy()
   require(ash(score)==representative['raw_score_sha256'],'Representative exact score digest')
   with np.load(path(representative['audit_path']),allow_pickle=False) as z:
    _,_,order,ends=rank_metrics(z,y[sc]);expected_order=np.argsort(-score,kind='mergesort');require(np.array_equal(order,expected_order) and np.array_equal(ends,np.r_[score[order][:-1]!=score[order][1:],True]),'Representative full rank/ties')
   self.model(name,sr['representative_model_path'],X,sc,score,self.c[name][cid]['params'],40000+1000*seed+10*fold,representative['selected_rounds'],representative['model_digest'],name!='random_forest',r)
   require(set(d['strategies'])=={'same_protocol_anchor','tuned'},'Both complete strategies')
   for strategy in ['same_protocol_anchor','tuned']:
    st=d['strategies'][strategy];chosen=0 if strategy=='same_protocol_anchor' else cid;rounds=summaries[chosen]['rounds'];params=self.c[name][chosen]['params'];state=100*seed+fold
    require(st['strategy']==strategy and st['candidate_id']==chosen and st['selected_rounds']==rounds and st['params']==params and st['effective_params']=={**params,'random_state':state,'n_estimators':rounds} and st['model_random_state']==state and st['final_fit_has_eval_set'] is False,'Strategy params/fit staging')
    require(all(st[k]==ash(v) for k,v in [('fit_rows_sha256',train),('calibration_fit_rows_sha256',D),('threshold_rows_sha256',T),('outer_test_rows_sha256',test)]),'Strategy fitting row bindings')
    if strategy=='tuned' and cid==0:
     expect=copy.deepcopy(d['strategies']['same_protocol_anchor']);expect['strategy']='tuned';expect['physical_reuse_anchor']=True
     for k in ['threshold_fit_seconds','final_fit_seconds','threshold_model_fit_seconds','final_model_fit_seconds']:expect[k]=0.
     for v in expect['metrics']:v['strategy']='tuned'
     require(st==expect,'C0 physical alias exact copied record')
    else:require(st['physical_reuse_anchor'] is False,'No unintended physical alias')
    cal=pd.read_parquet(path(st['calibration_path']));pred=pd.read_parquet(path(st['prediction_path']))
    for table,rows in [(cal,T),(pred,test)]:
     require(np.array_equal(table.row_index,rows) and np.array_equal(table.account_id,ids.iloc[rows].account_id) and np.array_equal(table.y_true,y[rows]) and table.y_score.dtype==np.float64 and np.isfinite(table.y_score).all() and table.y_score.between(0,1).all(),'Prediction identity/full scores')
    require(set(cal)=={'row_index','account_id','y_true','y_score'} and set(pred)=={'row_index','account_id','y_true','y_score','y_pred_fixed','y_pred_inner'},'Prediction schemas')
    cutoff,best=threshold(y[T],cal.y_score.to_numpy());close([cutoff,best],[st['threshold'],st['threshold_validation_f1']],'Independent max F1 tie threshold')
    close([average_precision_score(y[T],cal.y_score),roc_auc_score(y[T],cal.y_score)],[st['threshold_validation_ap'],st['threshold_validation_auc']],'Threshold ranking metrics')
    self.model(name,st['calibration_model_path'],X,T,cal.y_score.to_numpy(),params,state,rounds,st['calibration_model_digest'],False,r)
    self.model(name,st['model_path'],X,test,pred.y_score.to_numpy(),params,state,rounds,st['final_model_digest'],False,r)
    require(len(st['metrics'])==2 and {v['threshold_policy'] for v in st['metrics']}==set(POLICIES),'Two threshold policies')
    for policy,column,c in [('fixed_0.5','y_pred_fixed',.5),('inner_max_f1','y_pred_inner',cutoff)]:
     predicted=(pred.y_score.to_numpy()>=c).astype(np.int8);require(np.array_equal(pred[column],predicted),'Saved binary predictions')
     value=metrics(y[test],pred.y_score.to_numpy(),predicted);actual=next(v for v in st['metrics'] if v['threshold_policy']==policy)
     require([actual[k] for k in ['dataset','model','feature_set','strategy','seed','fold','selected_candidate_id','selected_rounds','n_features']]==[ds,name,arm,strategy,seed,fold,chosen,rounds,len(columns)],'Metric row metadata')
     close(actual['threshold'],c,'Metric threshold');close([actual[k] for k in METRICS],[value[k] for k in METRICS],'All metrics/confusion counts')
   require(d['logical_candidate_evaluations']==36 and d['physical_search_fits']==sr['physical_search_fits'] and d['logical_strategy_models']==4 and d['physical_strategy_models']==(2 if cid==0 else 4),'DONE physical counts')
   r.update(passed=True,status='passed_single_fold_not_complete_setting',completed_at=now(),fold_done_path=rel(fpath),fold_done_sha256=sha(fpath),selected_candidate_id=cid,n_features=len(columns),
    assurance='All 36 candidate rank/tie metrics reconstructed; retained representative plus threshold/final models rescored. Discarded candidate models not rescored. Fit-only weighting audited from SHA-bound source AST, disjoint split scopes and saved parameters; actual fit cannot be re-observed without refitting.',
    source_registration_sha256={rel(self.tree/(stem+'.json')):sha(self.tree/(stem+'.json')) for stem in ['protocol','candidates','dataset_inputs','split_manifest','source_manifest']},
    scope='Single completed fold only. No full-setting mean/SD or comparative conclusion.')
   write(receipt,r);return r
  except Exception:
   r.update(passed=False,status='failed',failed_at=now(),error=traceback.format_exc());write(receipt,r);raise
 def scan(self,ds=None,name=None,arm=None,max_folds=None,force=False):
  done=[]
  for dataset in ([ds] if ds else self.inputs):
   for fp in sorted((self.tree/dataset/'folds').glob('*.json')):
    d=read(fp)
    if (name and d['model']!=name) or (arm and d['feature_set']!=arm):continue
    done.append((dataset,d['model'],d['feature_set'],d['seed'],d['fold']))
  processed=[];new=0
  for identity in done:
   ds1,m,a,s,f=identity;rp=self.dest/ds1/f'{m}_{a}_seed{s}_fold{f}.json'
   already=rp.exists() and read(rp).get('passed') and read(rp).get('verifier_sha256')==sha(SCRIPT)
   if max_folds is not None and not already and new>=max_folds:continue
   report=self.fold(*identity,skip=not force);processed.append(rel(rp))
   if not already:new+=1
   print(json.dumps({'single_fold_passed':list(identity),'retained_models_rescored':len(report['physical_models_rescored']),'candidate_rank_audits':report['candidate_rank_audits_reconstructed']}),flush=True)
  overview=self.overview()
  write(self.dest/'last_scan.json',dict(status='partial_validation_scan',created_at=now(),verifier_sha256=sha(SCRIPT),eligible_done_folds=len(done),receipts_checked=len(processed),new_folds_validated=new,receipt_paths=processed,overview=overview))
  return overview
 def overview(self):
  counts=[]
  for ds in self.inputs:
   for model in MODELS:
    for arm in self.p['feature_arm_priority']:
     done=0;passed=0
     for seed,fold in itertools.product(range(5),range(5)):
      stem=f'{model}_{arm}_seed{seed}_fold{fold}';fp=self.tree/ds/'folds'/(stem+'.json');rp=self.dest/ds/(stem+'.json')
      if fp.exists():
       done+=1
       if rp.exists():
        r=read(rp)
        if r.get('passed') and r.get('verifier_sha256')==sha(SCRIPT) and r.get('fold_done_sha256')==sha(fp):passed+=1
     counts.append(dict(dataset=ds,model=model,feature_set=arm,training_done_folds=done,independently_passed_folds=passed,expected_folds=25,status='complete_independently_verified' if passed==25 else 'partial_not_complete'))
  v=dict(status='complete' if sum(v['independently_passed_folds'] for v in counts)==2400 else 'partial',
    created_at=now(),verifier_sha256=sha(SCRIPT),expected_new_tree_folds=2400,
    training_done_folds=sum(v['training_done_folds'] for v in counts),independently_passed_folds=sum(v['independently_passed_folds'] for v in counts),settings=counts,
    no_partial_means_reported=True,neural_validation='Not evaluated by tree-only action; use separate neural CLI.')
  write(self.dest/'overview.json',v);return v
 def prediction(self,ds,model,arm,seed,fold,strategy='tuned'):
  reference=arm in ('base16','gmke44','gmkep56')
  base=MAIN if reference else self.tree
  fp=base/ds/'folds'/f'{model}_{arm}_seed{seed}_fold{fold}.json';d=read(fp)
  require(d['protocol_sha256']==sha(base/'protocol.json'),'Paired source protocol')
  if not reference:
   rp=self.dest/ds/f'{model}_{arm}_seed{seed}_fold{fold}.json';r=read(rp)
   require(r.get('passed') and r['verifier_sha256']==sha(SCRIPT) and r['fold_done_sha256']==sha(fp),'Complete validated pair fold')
  st=d['strategies'][strategy];pp=path(st['prediction_path'])
  require(sha(pp)==st['artifact_sha256'][rel(pp)],'Paired prediction digest')
  table=pd.read_parquet(pp);_,y,ids,_=self.dataset(ds)
  with np.load(base/ds/'splits'/f'seed{seed}_fold{fold}.npz',allow_pickle=False) as z:rows=z['outer_test']
  require(np.array_equal(table.row_index,rows) and np.array_equal(table.account_id,ids.iloc[rows].account_id) and np.array_equal(table.y_true,y[rows]),'Paired full identity/truth')
  return table,st['metrics'],sha(pp)
 def aggregate(self):
  overview=self.overview();complete={(r['dataset'],r['model'],r['feature_set']) for r in overview['settings'] if r['independently_passed_folds']==25}
  rows=[];seeds=[];bindings={}
  for ds,model,arm in sorted(complete):
   for strategy in ['same_protocol_anchor','tuned']:
    for seed in range(5):
     pieces=[]
     for fold in range(5):
      t,m,h=self.prediction(ds,model,arm,seed,fold,strategy);pieces.append(t)
      fp=self.tree/ds/'folds'/f'{model}_{arm}_seed{seed}_fold{fold}.json';bindings[rel(fp)]=sha(fp)
      for v in m:rows.append(v)
     oof=pd.concat(pieces).sort_values('row_index');_,y,ids,_=self.dataset(ds)
     require(len(oof)==len(y) and oof.row_index.is_unique and np.array_equal(oof.row_index,np.arange(len(y))) and np.array_equal(oof.account_id,ids.account_id) and np.array_equal(oof.y_true,y),'Complete OOF coverage/truth')
     for policy,column in [('fixed_0.5','y_pred_fixed'),('inner_max_f1','y_pred_inner')]:
      seeds.append(dict(dataset=ds,model=model,feature_set=arm,strategy=strategy,seed=seed,threshold_policy=policy,**metrics(y,oof.y_score.to_numpy(),oof[column].to_numpy())))
  output=self.dest/'complete_settings';output.mkdir(parents=True,exist_ok=True)
  if rows:
   df=pd.DataFrame(rows);df.to_csv(output/'verified_fold_metrics.csv',index=False)
   sd=pd.DataFrame(seeds);sd.to_csv(output/'verified_seed_oof_metrics.csv',index=False)
   keys=['dataset','model','feature_set','strategy','threshold_policy'];ag=df.groupby(keys)[list(METRICS)].agg(['mean','std']);ag.columns=['_'.join(c) for c in ag.columns];ag.reset_index().to_csv(output/'verified_summary.csv',index=False)
  registry=read(ALT/'results/eswa_supplement_design_20261013/feature_registry.json')
  contrasts=sum((registry[k] for k in ['module_contrasts','ph_primary_contrasts','ph_secondary_contrasts','external_tree_contrasts']),[])
  pairs=[];pairseeds=[];skipped=[]
  for ds,model,c in itertools.product(self.inputs,MODELS,contrasts):
   lhs,rhs=c['lhs'],c['rhs'];available=lambda a:a in ('base16','gmke44','gmkep56') or (ds,model,a) in complete
   if not available(lhs) or not available(rhs):
    skipped.append(dict(dataset=ds,model=model,contrast_id=c['contrast_id'],reason='Both full25fold arms not independently complete'));continue
   for seed in range(5):
    pieces={lhs:[],rhs:[]}
    for fold in range(5):
     ta,ma,ha=self.prediction(ds,model,lhs,seed,fold);tb,mb,hb=self.prediction(ds,model,rhs,seed,fold)
     require(np.array_equal(ta.row_index,tb.row_index) and np.array_equal(ta.y_true,tb.y_true),'Pair exact split/truth');pieces[lhs].append(ta);pieces[rhs].append(tb)
     for policy in POLICIES:
      va=next(v for v in ma if v['threshold_policy']==policy);vb=next(v for v in mb if v['threshold_policy']==policy)
      pairs.append(dict(dataset=ds,model=model,contrast_id=c['contrast_id'],family=c['family'],lhs_arm=lhs,rhs_arm=rhs,seed=seed,fold=fold,threshold_policy=policy,**{'delta_'+k:float(va[k]-vb[k]) for k in METRICS}))
    aa=pd.concat(pieces[lhs]).sort_values('row_index');bb=pd.concat(pieces[rhs]).sort_values('row_index');_,y,_,_=self.dataset(ds)
    require(np.array_equal(aa.row_index,np.arange(len(y))) and np.array_equal(bb.row_index,np.arange(len(y))),'Paired OOF population')
    for policy,column in [('fixed_0.5','y_pred_fixed'),('inner_max_f1','y_pred_inner')]:
     va=metrics(y,aa.y_score.to_numpy(),aa[column].to_numpy());vb=metrics(y,bb.y_score.to_numpy(),bb[column].to_numpy())
     pairseeds.append(dict(dataset=ds,model=model,contrast_id=c['contrast_id'],family=c['family'],lhs_arm=lhs,rhs_arm=rhs,seed=seed,threshold_policy=policy,**{'delta_'+k:float(va[k]-vb[k]) for k in METRICS}))
  if pairs:
   pf=pd.DataFrame(pairs);ps=pd.DataFrame(pairseeds);pf.to_csv(output/'verified_paired_fold_differences.csv',index=False);ps.to_csv(output/'verified_paired_seed_oof_differences.csv',index=False)
   k=['dataset','model','contrast_id','family','lhs_arm','rhs_arm','threshold_policy'];cols=['delta_'+x for x in METRICS]
   summary=pf.groupby(k)[cols].agg(['mean','std']);summary.columns=['_'.join(c) for c in summary.columns];summary=summary.reset_index()
   positive=pf.groupby(k).delta_ap.agg(ap_positive_folds=lambda x:int((x>1e-12).sum())).reset_index();summary=summary.merge(positive,on=k)
   positive=ps.groupby(k).delta_ap.agg(ap_positive_oof_seeds=lambda x:int((x>1e-12).sum())).reset_index();summary=summary.merge(positive,on=k)
   summary.to_csv(output/'verified_paired_summary.csv',index=False)
  report=dict(passed=True,status='complete_tree_batch' if len(complete)==96 else 'partial_complete_settings_only',created_at=now(),verifier_sha256=sha(SCRIPT),complete_tree_settings=len(complete),expected_tree_settings=96,
   fold_metric_rows=len(rows),seed_oof_rows=len(seeds),paired_fold_rows=len(pairs),skipped_incomplete_comparisons=skipped,full_oof_verified=bool(complete),source_done_sha256=bindings,
   scope='Only complete25fold independently verified arms are aggregated; source main 16/44/56 controls use previously completed validated artifacts. No partial setting conclusions.',
   output_sha256={rel(p):sha(p) for p in output.glob('*.csv')})
  write(output/'validation.json',report);return report

def tree_main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['tree-fold','tree-scan','tree-aggregate']);p.add_argument('--out',type=Path,default=DEFAULT_OUT);p.add_argument('--dataset',choices=['hi_small','li_small']);p.add_argument('--model',choices=MODELS);p.add_argument('--arm');p.add_argument('--seed',type=int);p.add_argument('--fold',type=int);p.add_argument('--max-folds',type=int);p.add_argument('--force',action='store_true');a=p.parse_args()
 v=Validator(a.out)
 if a.action=='tree-fold':
  require(all(x is not None for x in [a.dataset,a.model,a.arm,a.seed,a.fold]),'tree-fold requires dataset/model/arm/seed/fold');r=v.fold(a.dataset,a.model,a.arm,a.seed,a.fold,skip=not a.force)
  print(json.dumps({k:r[k] for k in ['passed','status','dataset','model','feature_set','seed','fold','candidate_rank_audits_reconstructed']}),flush=True);v.overview()
 elif a.action=='tree-scan':
  r=v.scan(a.dataset,a.model,a.arm,a.max_folds,a.force);print(json.dumps({k:r[k] for k in ['status','training_done_folds','independently_passed_folds']},ensure_ascii=False),flush=True)
 else:
  r=v.aggregate();print(json.dumps({k:r[k] for k in ['passed','status','complete_tree_settings','expected_tree_settings']},ensure_ascii=False),flush=True)


NNMODELS=('mlp','graphsage_weak','graphsage_directed')
class NeuralValidator:
 """Raw state_dict affine operations and independent CSR means; no model class."""
 def __init__(self,out):
  import torch
  torch.set_num_threads(2)
  self.torch=torch;self.out=Path(out).resolve();self.nn=self.out/'neural';self.dest=self.out/'validation/neural'
  self.p=read(self.nn/'protocol.json');self.psha=sha(self.nn/'protocol.json');self.c=read(self.nn/'candidates.json')['candidates'];self.inputs=read(self.nn/'dataset_inputs.json');self.cache={};self.graph_cache={}
  self.binding_sources={rel(self.nn/'protocol.json'):self.psha,rel(self.nn/'candidates.json'):sha(self.nn/'candidates.json'),rel(self.nn/'dataset_inputs.json'):sha(self.nn/'dataset_inputs.json')}
  binding(read(self.nn/'source_binding.json'));self.binding_sources.update(read(self.nn/'source_binding.json'))
  dependency=self.nn/'DEPENDENCY_SOURCE_BINDING.json';require(dependency.exists(),'Neural dependency provenance absent');dep=read(dependency);binding(dep['source_sha256']);self.binding_sources[rel(dependency)]=sha(dependency)
  globaldep=self.out/'DEPENDENCY_SOURCE_BINDING.json'
  if globaldep.exists():
   gd=read(globaldep);binding(gd['source_sha256']);self.binding_sources[rel(globaldep)]=sha(globaldep)
  trainer=path('Altman_experiments/code/run_account_eswa_supplement_neural_20261013.py')
  require(sha(trainer)==self.p['runner_sha256'],'Neural frozen source');self.binding_sources[rel(trainer)]=sha(trainer)
  require(read(MAIN/'validation_models.json')['passed'] is True and read(MAIN/'run_manifest.json')['status']=='complete','Neural main identity source complete')
  require([c['candidate_id'] for c in self.c]==list(range(12)) and self.c==read(ALT/'results/eswa_supplement_design_20261013/neural_candidates.json')['candidates'],'Neural shared12 candidates')
  require(self.p['seeds']==list(range(5)) and self.p['early_stopping']=={'metric':'unweighted AUROC','patience':30,'validation_fraction':.2},'Neural scientific protocol')
  f={n.name:n for n in ast.parse(trainer.read_text()).body if isinstance(n,ast.FunctionDef)}
  fittext=ast.unparse(f['fit']);searchtext=ast.unparse(f['search'])
  require('raw[fit_rows]' in fittext and 'y[roots]' in fittext and 'rng.permutation(fit_rows)' in fittext and "reduction='none'" in fittext and 'weights[y[roots]]' in fittext,'SHA frozen fit-only scaler and seed loss AST')
  require('outer_test' not in searchtext and 'threshold_validation' not in searchtext and "arrays[f'{key}_{j}']" in searchtext,'Neural search scope AST')
 def dataset(self,ds):
  if ds in self.cache:return self.cache[ds]
  self.cache.clear();self.graph_cache.clear();gc.collect()
  inp=self.inputs[ds];binding(inp['cache_sha256']);binding(inp['source_sha256']);binding(inp['split_sources'])
  raw=np.load(self.nn/ds/'arrays/log_base16.npy',mmap_mode='r');y=np.load(self.nn/ds/'arrays/labels.npy',mmap_mode='r');ids=pd.read_parquet(self.nn/ds/'accounts.parquet')
  require(raw.shape==(inp['n_samples'],16) and raw.dtype==np.float32 and y.dtype==np.int8 and int(y.sum())==inp['n_positive'],'Neural cache dimensions')
  source=next(n for n in inp['source_sha256'] if n.endswith('/features.parquet'))
  table=pd.read_parquet(path(source),columns=['account_id','label',*inp['base_columns']]);before=table[inp['base_columns']].to_numpy(np.float64)
  require(np.isfinite(before).all() and (before>=0).all(),'Nonnegative original BASE source')
  require(np.array_equal(np.asarray(np.log1p(before),dtype=np.float32),raw),'Neural logBASE source is float64 parquet log1p then float32, not log of main rounded array')
  require(np.array_equal(ids.row_index,np.arange(len(y))) and ids.account_id.is_unique and np.array_equal(ids.account_id,table.account_id) and np.array_equal(ids.label,y) and np.array_equal(table.label,y),'Neural source identity/labels')
  mainids=pd.read_parquet(MAIN/ds/'accounts.parquet');require(np.array_equal(ids.account_id,mainids.account_id) and np.array_equal(y,mainids.label),'Neural main account task')
  require(ash(y)==inp['label_sha256'] and ash(ids.row_index.to_numpy())==inp['account_order_sha256'] and ash(pd.util.hash_pandas_object(ids.account_id,index=False).to_numpy())==inp['account_id_sha256'],'Neural identity digests')
  del table,before,mainids;gc.collect();self.cache[ds]=(raw,y,ids);return self.cache[ds]
 def graphs(self,ds,name):
  if name=='mlp':return []
  key=(ds,name)
  if key in self.graph_cache:return self.graph_cache[key]
  from scipy import sparse
  raw,y,ids=self.dataset(ds);inp=self.inputs[ds];folder=self.nn/ds/'arrays'
  weak=sparse.load_npz(folder/'weak_csr.npz').tocsr();outgoing=sparse.load_npz(folder/'outgoing_csr.npz').tocsr();incoming=sparse.load_npz(folder/'incoming_csr.npz').tocsr()
  require(weak.shape==(len(y),len(y)) and (weak!=weak.T).nnz==0 and weak.diagonal().sum()==0 and (outgoing.T!=incoming).nnz==0 and (outgoing.maximum(incoming)!=weak).nnz==0,'All-account graph relations/no loops')
  require(weak.nnz//2==inp['weak_edges'] and outgoing.nnz==inp['directed_edges'] and np.all(weak.data==1) and np.all(outgoing.data==1),'Simple graph edge counts/values')
  source_csr=next(n for n in inp['source_sha256'] if n.endswith('/graph_csr.npz'));source_map=next(n for n in inp['source_sha256'] if n.endswith('/account_to_graph.npy'))
  original=sparse.load_npz(path(source_csr)).tocsr();mapping=np.load(path(source_map));require((original[mapping][:,mapping]!=weak).nnz==0,'Graph/account reindex equivalence')
  trans=next(n for n in inp['source_sha256'] if n.endswith('_trans.parquet'));edge=pd.read_parquet(path(trans),columns=['from_id','to_id']);index=pd.Index(ids.account_id);a=index.get_indexer(edge.from_id);b=index.get_indexer(edge.to_id);mask=a!=b
  require((a>=0).all() and (b>=0).all(),'All transaction endpoints represented')
  expected=sparse.coo_matrix((np.ones(mask.sum(),np.int32),(a[mask],b[mask])),shape=weak.shape).tocsr();expected.data[:]=1
  require((expected!=outgoing).nnz==0,'Directed graph label-blind transaction source')
  del original,mapping,edge,a,b,mask,expected;gc.collect()
  chosen=[weak] if name=='graphsage_weak' else [incoming,outgoing];normalized=[]
  for g in chosen:
   g=g.astype(np.float32);degree=np.diff(g.indptr);g.data[:]=np.repeat((1/np.maximum(degree,1)).astype(np.float32),degree);normalized.append(g)
  self.graph_cache.clear();self.graph_cache[key]=normalized;return normalized
 def state_forward(self,state,X,graphs,rows,name,chunk=16384):
  torch=self.torch;F=torch.nn.functional
  with torch.no_grad():
   if name=='mlp':
    result=np.empty(len(rows),np.float64)
    for start in range(0,len(rows),chunk):
     ii=rows[start:start+chunk];x=torch.from_numpy(np.ascontiguousarray(X[ii]));h=torch.relu(F.linear(x,state['layers.0.weight'],state['layers.0.bias']));z=F.linear(h,state['layers.1.weight'],state['layers.1.bias']).squeeze(-1);result[start:start+len(ii)]=torch.sigmoid(z).double().numpy()
    return result
   h=X
   for layer in range(2):
    width=state[f'root.{layer}.0.weight'].shape[0];out=np.empty((len(X),width),np.float32)
    for start in range(0,len(X),chunk):
     stop=min(start+chunk,len(X));root=torch.from_numpy(np.ascontiguousarray(h[start:stop]));values=[]
     for relation,g in enumerate(graphs):
      mean=torch.from_numpy(np.ascontiguousarray(g[start:stop].dot(h)))
      values.append(F.linear(root,state[f'root.{layer}.{relation}.weight'],state[f'root.{layer}.{relation}.bias'])+F.linear(mean,state[f'neighbor.{layer}.{relation}.weight']))
     z=torch.cat(values,-1)
     if name=='graphsage_directed':z=F.linear(z,state[f'fusion.{layer}.weight'],state[f'fusion.{layer}.bias'])
     if layer==0:z=torch.relu(z)
     out[start:stop]=z.numpy()
    h=out
   return np.ascontiguousarray(torch.sigmoid(torch.from_numpy(h[rows,0])).double().numpy(),dtype=np.float64)
 def model(self,ds,name,mp,fit,early,params,seed,rows,saved,expected_epochs,r):
  raw,y,ids=self.dataset(ds);graphs=self.graphs(ds,name);p=path(mp);payload=self.torch.load(p,map_location='cpu',weights_only=True);meta=payload['metadata'];state=payload['state_dict']
  require(meta==read(p.with_suffix('.json')) and meta['model']==name and meta['params']==params and meta['random_state']==seed and meta['device']==self.p['device'],'PT/JSON metadata')
  require(meta['fit_rows_sha256']==ash(fit) and meta['scaler_fit_rows_sha256']==ash(fit) and meta['early_rows_sha256']==(None if early is None else ash(early)) and meta['scaler_n_samples_seen']==len(fit) and (meta['fit_n'],meta['fit_positive'])==(len(fit),int(y[fit].sum())) and meta['seed_only_loss'] is True,'Neural fit/scaler/label scopes')
  fitraw=np.asarray(raw[fit],dtype=np.float64);mu=fitraw.mean(axis=0);var=fitraw.var(axis=0,ddof=0);eps=np.finfo(np.float64).eps
  constant=var<=len(fit)*eps*var+(len(fit)*mu*eps)**2;scale=np.sqrt(var);scale[constant]=1.
  close(meta['scaler_mean'],mu,'Fit-only scaler mean');close(meta['scaler_var'],var,'Fit-only scaler population variance');close(meta['scaler_scale'],scale,'Fit-only scaler scale')
  w=np.array([len(fit)/(2*(len(fit)-int(y[fit].sum()))),len(fit)/(2*int(y[fit].sum()))],dtype=np.float32);require(np.array_equal(w,np.array(meta['class_weights'],np.float32)),'Fit-only balanced BCE class weights')
  del fitraw;gc.collect()
  hidden=params['hidden_dim'];dims={}
  if name=='mlp':dims={'layers.0.weight':(hidden,16),'layers.0.bias':(hidden,),'layers.1.weight':(1,hidden),'layers.1.bias':(1,)}
  else:
   for layer,(d,o) in enumerate([(16,hidden),(hidden,1)]):
    for relid in range(1 if name=='graphsage_weak' else 2):
     dims[f'root.{layer}.{relid}.weight']=(o,d);dims[f'root.{layer}.{relid}.bias']=(o,);dims[f'neighbor.{layer}.{relid}.weight']=(o,d)
    if name=='graphsage_directed':dims[f'fusion.{layer}.weight']=(o,2*o);dims[f'fusion.{layer}.bias']=(o,)
  require(set(state)==set(dims) and all(tuple(state[k].shape)==v and state[k].dtype==self.torch.float32 and self.torch.isfinite(state[k]).all() for k,v in dims.items()),'Neural tensor architecture/no ID embedding')
  h=hashlib.sha256()
  for key,value in state.items():h.update(key.encode());h.update(value.contiguous().numpy().tobytes())
  require(h.hexdigest()==meta['model_digest'],'Neural state tensor digest')
  history=meta['history'];require([h['epoch'] for h in history]==list(range(1,len(history)+1)) and meta['epochs_run']==len(history) and meta['selected_epochs']==expected_epochs,'Epoch history/selected count')
  require(all(np.isfinite(v['weighted_seed_loss']) and v['weighted_seed_loss']>=0 and np.isfinite(v['seconds']) for v in history),'Finite loss/history')
  if early is not None:
   au=np.array([h['early_auroc'] for h in history]);best=int(np.argmax(au))+1
   require(best==expected_epochs and len(history)<=200 and (len(history)==200 or len(history)-best>=30),'Early AUROC first best/patience')
   close(meta['best_early_auroc'],au.max(),'Early best metadata')
  else:require(meta['best_early_auroc'] is None and len(history)==expected_epochs,'Refit without early validation')
  X=np.array(raw,dtype=np.float32,copy=True);np.subtract(X,np.asarray(meta['scaler_mean']),out=X,casting='unsafe');np.divide(X,np.asarray(meta['scaler_scale']),out=X,casting='unsafe')
  score=self.state_forward(state,X,graphs,rows,name);diff=close(score,saved,'Independent affine/CSR all saved scores')
  require(ash(score)==ash(saved),'Neural CPU inference exact bytes')
  if early is not None:close(roc_auc_score(y[early],self.state_forward(state,X,graphs,early,name)),meta['best_early_auroc'],'Selected checkpoint AUROC')
  r['physical_models_rescored'][rel(p)]={'sha256':sha(p),'rows':len(rows),'max_abs_diff':diff,'scaler_fit_count':len(fit),'source_state_digest':meta['model_digest']}
  del X,payload,state,score;gc.collect();return meta
 def candidate(self,ds,name,seed,fold,cid,j,r=None):
  stem=f'{name}_seed{seed}_fold{fold}';rp=self.nn/ds/'search'/stem/f'candidate{cid}_inner{j}_record.json';v=read(rp)
  if r is None:r=dict(passed=False,status='validating_single_candidate',dataset=ds,model=name,seed=seed,fold=fold,candidate_id=cid,inner_fold=j,started_at=now(),verifier_sha256=sha(SCRIPT),verified_artifact_sha256={},physical_models_rescored={})
  binding(v['artifact_sha256'],r);r['verified_artifact_sha256'][rel(rp)]=sha(rp)
  sp=MAIN/ds/'splits'/f'seed{seed}_fold{fold}.npz';require(sha(sp)==self.inputs[ds]['split_sources'][rel(sp)],'Neural candidate original split')
  with np.load(sp,allow_pickle=False) as z:a={k:z[k] for k in z.files}
  fit,early,sc=(a[f'{k}_{j}'] for k in ['early_fit','early_validation','inner_score']);_,y,_=self.dataset(ds)
  require([v[k] for k in ['dataset','model','seed','fold','candidate_id','inner_fold']]==[ds,name,seed,fold,cid,j] and v['protocol_sha256']==self.psha and v['params']==self.c[cid]['params'],'Candidate registration')
  require(v['fit_rows_sha256']==ash(fit) and v['early_rows_sha256']==ash(early) and v['score_rows_sha256']==ash(sc),'Candidate stage row scopes')
  require(not np.intersect1d(fit,early).size and not np.intersect1d(np.r_[fit,early],sc).size and not np.intersect1d(np.r_[fit,early,sc],np.r_[a['threshold_validation'],a['outer_test']]).size,'Neural disjoint selection/early/test scopes')
  with np.load(path(v['score_path']),allow_pickle=False) as z:
   require(set(z.files)=={'row_index','y_true','y_score'} and np.array_equal(z['row_index'],sc) and np.array_equal(z['y_true'],y[sc]) and z['y_score'].dtype==np.float64,'Candidate full-score identity');score=z['y_score'].copy()
  require(ash(score)==v['raw_score_sha256'],'Candidate raw-score bytes');close([average_precision_score(y[sc],score),roc_auc_score(y[sc],score)],[v['ap'],v['auc']],'Independent neural inner AP/AUC')
  meta=self.model(ds,name,v['model_path'],fit,early,v['params'],40000+1000*seed+10*fold+j,sc,score,v['selected_epochs'],r)
  require(meta['model_digest']==v['model_digest'],'Record state digest')
  return v,r
 def fold(self,ds,name,seed,fold,skip=True):
  stem=f'{name}_seed{seed}_fold{fold}';fp=self.nn/ds/'folds'/(stem+'.json');rp=self.dest/ds/(stem+'.json')
  if skip and rp.exists():
   old=read(rp)
   if old.get('passed') and old.get('verifier_sha256')==sha(SCRIPT) and old.get('fold_done_sha256')==sha(fp):
    binding(old['verified_artifact_sha256']);binding(old['source_binding_sha256']);return old
  r=dict(passed=False,status='validating_neural_outer_fold',dataset=ds,model=name,feature_set='base16',seed=seed,fold=fold,started_at=now(),verifier_sha256=sha(SCRIPT),protocol_sha256=self.psha,verified_artifact_sha256={},physical_models_rescored={})
  try:
   d=read(fp);require(d['protocol_sha256']==self.psha and [d[k] for k in ['dataset','model','seed','fold']]==[ds,name,seed,fold],'Neural DONE registration')
   sr=d['search'];require(sr==read(self.nn/ds/'search'/stem/'search_result.json'),'Embedded complete search identical');binding(sr['artifact_sha256'],r)
   require(len(sr['candidate_records'])==36 and {(v['candidate_id'],v['inner_fold']) for v in sr['candidate_records']}==set(itertools.product(range(12),range(3))),'Neural complete12x3 candidate grid')
   for saved in sr['candidate_records']:
    actual,r=self.candidate(ds,name,seed,fold,saved['candidate_id'],saved['inner_fold'],r);require(saved==actual,'Complete candidate record identity')
   summaries=[]
   for cid in range(12):
    rows=[v for v in sr['candidate_records'] if v['candidate_id']==cid];summaries.append(dict(candidate_id=cid,mean_ap=float(np.mean([v['ap'] for v in rows])),std_ap=float(np.std([v['ap'] for v in rows],ddof=1)),epochs=int(np.median([v['selected_epochs'] for v in rows]))))
   require([v['candidate_id'] for v in sr['candidate_summary']]==list(range(12)),'Neural candidate summary IDs')
   for actual,expected in zip(sr['candidate_summary'],summaries):close([actual[k] for k in ['mean_ap','std_ap','epochs']],[expected[k] for k in ['mean_ap','std_ap','epochs']],'Neural summaries/median')
   best=min(summaries,key=lambda x:(-x['mean_ap'],x['candidate_id']));require(sr['selected_candidate_id']==best['candidate_id'] and sr['selected_epochs']==best['epochs'] and sr['anchor_epochs']==summaries[0]['epochs'],'Neural AP exact ties/median selection')
   sp=MAIN/ds/'splits'/f'seed{seed}_fold{fold}.npz'
   with np.load(sp,allow_pickle=False) as z:a={k:z[k] for k in z.files}
   raw,y,ids=self.dataset(ds);train,test,D,T=(a[k] for k in ['outer_train','outer_test','development','threshold_validation'])
   require(len(d['strategies'])==2 and {v['strategy'] for v in d['strategies']}=={'same_protocol_anchor','tuned'},'Neural two strategies')
   for st in d['strategies']:
    cid=0 if st['strategy']=='same_protocol_anchor' else best['candidate_id'];epochs=summaries[cid]['epochs'];params=self.c[cid]['params'];binding(st['artifact_sha256'],r)
    require(st['protocol_sha256']==self.psha and st['candidate_id']==cid and st['selected_epochs']==epochs and st['params']==params and st['model_random_state']==100*seed+fold,'Neural selected stage configuration')
    require(all(st[k]==ash(v) for k,v in [('fit_rows_sha256',train),('calibration_fit_rows_sha256',D),('threshold_rows_sha256',T),('outer_test_rows_sha256',test)]),'Neural stage row binding')
    cal=pd.read_parquet(path(st['calibration_path']));pred=pd.read_parquet(path(st['prediction_path']))
    for table,rows in [(cal,T),(pred,test)]:require(np.array_equal(table.row_index,rows) and np.array_equal(table.account_id,ids.iloc[rows].account_id) and np.array_equal(table.y_true,y[rows]) and table.y_score.dtype==np.float64 and table.y_score.between(0,1).all(),'Neural stage full account predictions')
    cutoff,mf1=threshold(y[T],cal.y_score.to_numpy());close([cutoff,mf1],[st['threshold'],st['threshold_validation_f1']],'Neural independent T maxF1/tie threshold');close(average_precision_score(y[T],cal.y_score),st['threshold_validation_ap'],'Neural T AP')
    for role,fit,rows,table in [('calibration',D,T,cal),('final',train,test,pred)]:
     mp=st['calibration_model_path'] if role=='calibration' else st['model_path'];meta=self.model(ds,name,mp,fit,None,params,100*seed+fold,rows,table.y_score.to_numpy(),epochs,r);require(meta==st[role+'_metadata'],'Neural embedded stage metadata')
    require(ash(pred.y_score.to_numpy())==st['raw_score_sha256'],'Neural final raw score digest')
    require(len(st['metrics'])==2 and {v['threshold_policy'] for v in st['metrics']}==set(POLICIES),'Neural two threshold rows')
    for policy,column,c in [('fixed_0.5','y_pred_fixed',.5),('inner_max_f1','y_pred_inner',cutoff)]:
     predicted=(pred.y_score.to_numpy()>=c).astype(np.int8);require(np.array_equal(pred[column],predicted),'Neural binary outputs');actual=next(v for v in st['metrics'] if v['threshold_policy']==policy);values=metrics(y[test],pred.y_score.to_numpy(),predicted)
     require(all(k in actual for k in ['ap','auc','precision','recall','f1','accuracy','balanced_accuracy','specificity','tn','fp','fn','tp','n_test','n_positive']),'Neural complete metric fields')
     require([actual[k] for k in ['dataset','model','feature_set','strategy','seed','fold','selected_candidate_id','selected_epochs']]==[ds,name,'base16',st['strategy'],seed,fold,cid,epochs],'Neural metric row metadata')
     close([actual[k] for k in values if k in actual],[values[k] for k in values if k in actual],'Neural metrics/confusion matrix');close(actual['threshold'],c,'Neural metric cutoff')
   r['verified_artifact_sha256'][rel(fp)]=sha(fp);r.update(passed=True,status='passed_single_neural_fold_not_complete_setting',completed_at=now(),fold_done_path=rel(fp),fold_done_sha256=sha(fp),candidate_models_and_scores_independently_verified=36,source_binding_sha256=self.binding_sources,assurance='All36 retained neural inner models plus4 threshold/final models independently evaluated from state tensors and fit-only scaler; no imported trainer class/no fitting. Seed-only training loss audited from frozen source AST and saved fit-row/class-weight records.',scope='Single fold only; completed setting requires all25 folds.')
   write(rp,r);return r
  except Exception:r.update(status='failed',error=traceback.format_exc(),failed_at=now());write(rp,r);raise
 def overview(self):
  settings=[]
  for ds,name in itertools.product(self.inputs,NNMODELS):
   done=passed=0
   for seed,fold in itertools.product(range(5),range(5)):
    stem=f'{name}_seed{seed}_fold{fold}';fp=self.nn/ds/'folds'/(stem+'.json');rp=self.dest/ds/(stem+'.json')
    if fp.exists():
     done+=1
     if rp.exists():
      r=read(rp)
      if r.get('passed') and r.get('verifier_sha256')==sha(SCRIPT) and r.get('fold_done_sha256')==sha(fp):passed+=1
   settings.append(dict(dataset=ds,model=name,training_done_folds=done,independently_passed_folds=passed,expected_folds=25,status='complete_independently_verified' if passed==25 else 'partial_not_complete'))
  r=dict(status='complete' if sum(v['independently_passed_folds'] for v in settings)==150 else 'partial',created_at=now(),verifier_sha256=sha(SCRIPT),expected_neural_folds=150,training_done_folds=sum(v['training_done_folds'] for v in settings),independently_passed_folds=sum(v['independently_passed_folds'] for v in settings),settings=settings,no_partial_means_reported=True)
  write(self.dest/'overview.json',r);return r
 def scan(self,ds=None,name=None,max_folds=None,force=False):
  done=[]
  for dataset in ([ds] if ds else self.inputs):
   for fp in sorted((self.nn/dataset/'folds').glob('*_seed*_fold[0-4].json')):
    d=read(fp)
    if name and d['model']!=name:continue
    done.append((dataset,d['model'],d['seed'],d['fold']))
  checked=[];new=0
  for identity in done:
   ds1,m,s,f=identity;rp=self.dest/ds1/f'{m}_seed{s}_fold{f}.json';already=rp.exists() and read(rp).get('passed') and read(rp).get('verifier_sha256')==sha(SCRIPT)
   if max_folds is not None and not already and new>=max_folds:continue
   r=self.fold(*identity,skip=not force);checked.append(rel(rp));new+=int(not already)
   print(json.dumps({'neural_single_fold_passed':list(identity),'physical_models_rescored':len(r['physical_models_rescored'])}),flush=True)
  r=self.overview();write(self.dest/'last_scan.json',dict(status='partial_scan',created_at=now(),verifier_sha256=sha(SCRIPT),done_available=len(done),receipts_checked=checked,new_folds_verified=new));return r
 def predictions(self,ds,model,seed,fold,strategy='tuned',reference_arm=None):
  reference=reference_arm is not None
  base=MAIN if reference else self.nn;stem=f'{model}_{reference_arm}_seed{seed}_fold{fold}' if reference else f'{model}_seed{seed}_fold{fold}';fp=base/ds/'folds'/(stem+'.json');d=read(fp)
  require(d['protocol_sha256']==sha(base/'protocol.json'),'Neural paired source protocol')
  if reference:st=d['strategies'][strategy]
  else:
   r=read(self.dest/ds/(stem+'.json'));require(r.get('passed') and r['verifier_sha256']==sha(SCRIPT) and r['fold_done_sha256']==sha(fp),'Neural paired complete verified fold');st=next(v for v in d['strategies'] if v['strategy']==strategy)
  pp=path(st['prediction_path']);require(sha(pp)==st['artifact_sha256'][rel(pp)],'Neural paired score digest')
  t=pd.read_parquet(pp);_,y,ids=self.dataset(ds)
  with np.load(MAIN/ds/'splits'/f'seed{seed}_fold{fold}.npz',allow_pickle=False) as z:ii=z['outer_test']
  require(np.array_equal(t.row_index,ii) and np.array_equal(t.account_id,ids.iloc[ii].account_id) and np.array_equal(t.y_true,y[ii]),'Neural paired population')
  return t,st['metrics'],{rel(fp):sha(fp),rel(pp):sha(pp)}
 def aggregate(self):
  overview=self.overview();complete={(r['dataset'],r['model']) for r in overview['settings'] if r['independently_passed_folds']==25};rows=[];seedrows=[];bindings={}
  for ds,model in sorted(complete):
   for strategy in ('same_protocol_anchor','tuned'):
    for seed in range(5):
     pieces=[]
     for fold in range(5):
      t,m,b=self.predictions(ds,model,seed,fold,strategy);pieces.append(t);rows.extend(m);bindings.update(b)
     oof=pd.concat(pieces).sort_values('row_index');_,y,ids=self.dataset(ds)
     require(len(oof)==len(y) and oof.row_index.is_unique and np.array_equal(oof.row_index,np.arange(len(y))) and np.array_equal(oof.y_true,y),'Neural complete seed OOF all accounts')
     for policy,column in [('fixed_0.5','y_pred_fixed'),('inner_max_f1','y_pred_inner')]:seedrows.append(dict(dataset=ds,model=model,feature_set='base16',strategy=strategy,seed=seed,threshold_policy=policy,**metrics(y,oof.y_score.to_numpy(),oof[column].to_numpy())))
  output=self.dest/'complete_settings';output.mkdir(parents=True,exist_ok=True)
  if rows:
   df=pd.DataFrame(rows);df.to_csv(output/'verified_fold_metrics.csv',index=False);pd.DataFrame(seedrows).to_csv(output/'verified_seed_oof_metrics.csv',index=False)
   keys=['dataset','model','strategy','threshold_policy'];summary=df.groupby(keys)[['ap','auc','f1','precision','recall','accuracy','balanced_accuracy']].agg(['mean','std']);summary.columns=['_'.join(k) for k in summary.columns];summary.reset_index().to_csv(output/'verified_summary.csv',index=False)
  contrasts=[('weak_vs_mlp','graphsage_weak',None,'mlp',None),('directed_vs_mlp','graphsage_directed',None,'mlp',None),('directed_vs_weak','graphsage_directed',None,'graphsage_weak',None)]
  contrasts +=[(f'xgboost_{arm}_vs_{m}','xgboost',arm,m,None) for arm in ('gmke44','gmkep56') for m in ('graphsage_weak','graphsage_directed')]
  pairs=[];ps=[];skipped=[]
  for ds,(cid,lm,la,rm,ra) in itertools.product(self.inputs,contrasts):
   if (la is None and (ds,lm) not in complete) or (ra is None and (ds,rm) not in complete):skipped.append(dict(dataset=ds,contrast_id=cid,reason='Both25fold settings not independently complete'));continue
   for seed in range(5):
    left=[];right=[]
    for fold in range(5):
     aa,am,ab=self.predictions(ds,lm,seed,fold,reference_arm=la);bb,bm,bbb=self.predictions(ds,rm,seed,fold,reference_arm=ra);bindings.update(ab);bindings.update(bbb);require(np.array_equal(aa.row_index,bb.row_index),'Neural pair exact test rows');left.append(aa);right.append(bb)
     for policy,column in [('fixed_0.5','y_pred_fixed'),('inner_max_f1','y_pred_inner')]:
      va=metrics(aa.y_true.to_numpy(),aa.y_score.to_numpy(),aa[column].to_numpy());vb=metrics(bb.y_true.to_numpy(),bb.y_score.to_numpy(),bb[column].to_numpy());pairs.append(dict(dataset=ds,contrast_id=cid,lhs_model=lm,lhs_arm=la or 'base16',rhs_model=rm,rhs_arm=ra or 'base16',seed=seed,fold=fold,threshold_policy=policy,**{'delta_'+k:float(va[k]-vb[k]) for k in METRICS}))
    aa=pd.concat(left).sort_values('row_index');bb=pd.concat(right).sort_values('row_index');_,y,_=self.dataset(ds);require(np.array_equal(aa.row_index,np.arange(len(y))) and np.array_equal(bb.row_index,np.arange(len(y))),'Neural paired OOF completeness')
    for policy,column in [('fixed_0.5','y_pred_fixed'),('inner_max_f1','y_pred_inner')]:
     va=metrics(y,aa.y_score.to_numpy(),aa[column].to_numpy());vb=metrics(y,bb.y_score.to_numpy(),bb[column].to_numpy());ps.append(dict(dataset=ds,contrast_id=cid,lhs_model=lm,lhs_arm=la or 'base16',rhs_model=rm,rhs_arm=ra or 'base16',seed=seed,threshold_policy=policy,**{'delta_'+k:float(va[k]-vb[k]) for k in METRICS}))
  if pairs:
   df=pd.DataFrame(pairs);ss=pd.DataFrame(ps);df.to_csv(output/'verified_paired_fold_differences.csv',index=False);ss.to_csv(output/'verified_paired_seed_oof_differences.csv',index=False)
   keys=['dataset','contrast_id','lhs_model','lhs_arm','rhs_model','rhs_arm','threshold_policy'];s=df.groupby(keys)[['delta_'+k for k in METRICS]].agg(['mean','std']);s.columns=['_'.join(k) for k in s.columns];s=s.reset_index();s=s.merge(df.groupby(keys).delta_ap.agg(ap_positive_folds=lambda x:int((x>1e-12).sum())).reset_index(),on=keys);s=s.merge(ss.groupby(keys).delta_ap.agg(ap_positive_oof_seeds=lambda x:int((x>1e-12).sum())).reset_index(),on=keys);s.to_csv(output/'verified_paired_summary.csv',index=False)
  r=dict(passed=True,status='complete_neural_batch' if len(complete)==6 else 'partial_complete_settings_only',created_at=now(),verifier_sha256=sha(SCRIPT),complete_neural_settings=len(complete),expected_neural_settings=6,fold_rows=len(rows),seed_oof_rows=len(seedrows),paired_fold_rows=len(pairs),skipped_incomplete_comparisons=skipped,source_prediction_sha256=bindings,output_sha256={rel(p):sha(p) for p in output.glob('*.csv')},scope='Only25fold complete independently verified settings; no partial setting mean/SD or comparison conclusion.')
  write(output/'validation.json',r);return r

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['tree-fold','tree-scan','tree-aggregate','neural-candidate','neural-fold','neural-scan','neural-aggregate']);p.add_argument('--out',type=Path,default=DEFAULT_OUT);p.add_argument('--dataset',choices=['hi_small','li_small']);p.add_argument('--model',choices=MODELS+NNMODELS);p.add_argument('--arm');p.add_argument('--seed',type=int);p.add_argument('--fold',type=int);p.add_argument('--candidate',type=int);p.add_argument('--inner-fold',type=int);p.add_argument('--max-folds',type=int);p.add_argument('--force',action='store_true');a=p.parse_args()
 if a.action.startswith('tree-'):
  require(a.model is None or a.model in MODELS,'Tree model required');v=Validator(a.out)
  if a.action=='tree-fold':
   require(all(x is not None for x in [a.dataset,a.model,a.arm,a.seed,a.fold]),'tree-fold requires dataset/model/arm/seed/fold');r=v.fold(a.dataset,a.model,a.arm,a.seed,a.fold,skip=not a.force);print(json.dumps({k:r[k] for k in ['passed','status','dataset','model','feature_set','seed','fold','candidate_rank_audits_reconstructed']}),flush=True);v.overview()
  elif a.action=='tree-scan':
   r=v.scan(a.dataset,a.model,a.arm,a.max_folds,a.force);print(json.dumps({k:r[k] for k in ['status','training_done_folds','independently_passed_folds']},ensure_ascii=False),flush=True)
  else:r=v.aggregate();print(json.dumps({k:r[k] for k in ['passed','status','complete_tree_settings','expected_tree_settings']},ensure_ascii=False),flush=True)
 else:
  require(a.model is None or a.model in NNMODELS,'Neural model required');v=NeuralValidator(a.out)
  if a.action in ('neural-fold','neural-candidate'):
   require(all(x is not None for x in [a.dataset,a.model,a.seed,a.fold]),'Neural fold identity required')
   if a.action=='neural-fold':r=v.fold(a.dataset,a.model,a.seed,a.fold,skip=not a.force);v.overview()
   else:
    require(a.candidate in range(12) and a.inner_fold in range(3),'Candidate/inner-fold required');record,r=v.candidate(a.dataset,a.model,a.seed,a.fold,a.candidate,a.inner_fold);r.update(passed=True,status='passed_single_neural_candidate_not_outer_fold',completed_at=now(),protocol_sha256=v.psha,source_binding_sha256=v.binding_sources,scope='Only one retained inner candidate verified; no complete outerfold/selection or result means.')
    write(v.dest/'candidate_receipts'/a.dataset/f'{a.model}_seed{a.seed}_fold{a.fold}_candidate{a.candidate}_inner{a.inner_fold}.json',r)
   print(json.dumps({k:r[k] for k in ['passed','status','dataset','model','seed','fold']}),flush=True)
  elif a.action=='neural-scan':r=v.scan(a.dataset,a.model,a.max_folds,a.force);print(json.dumps({k:r[k] for k in ['status','training_done_folds','independently_passed_folds']}),flush=True)
  else:r=v.aggregate();print(json.dumps({k:r[k] for k in ['passed','status','complete_neural_settings','expected_neural_settings']}),flush=True)
if __name__=='__main__':main()
