"""Independent no-fit validation for the separately registered optimized CUDA run.

No trainer/model class is imported. State tensors are evaluated on CPU with two
threads using affine operations and independently normalized CSR means. A
single-fold receipt never denotes a complete25fold feature setting. All CUDA
feature arms must use the same registered backend; local MPS runs are references.
"""
from __future__ import annotations
import argparse, ast, gc, hashlib, itertools, json, os, resource, sys, time, traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np
import pandas as pd
from scipy import sparse
import torch
from sklearn.metrics import precision_recall_curve,average_precision_score,roc_auc_score

PROJECT=Path(__file__).resolve().parents[2]
ROOT=PROJECT
SCRIPT=Path(__file__).resolve()
DEFAULT_OUT=ROOT/'Altman_experiments/results/account_graphsage_cuda_optimized_20261018'
METRICS=('ap','auc','precision','recall','f1','accuracy','balanced_accuracy','specificity','fpr','fnr','tn','fp','fn','tp','n_test','n_positive')
POLICIES=('fixed_0.5','inner_max_f1')
TOL=1e-12
HASH_CACHE={}
# This new verifier never admits old-namespace fits or an arbitrary new runner.
EXPECTED_OPTIMIZED_RUNNER_SHA256='3830ebf59298f86e4cef4fa8c715e02fa37b8d75f35e6be0f7618e178f10930e'
REFERENCE_RUNNER_SHA256='e827b8422fcf25ea63df9c1033935f302f7f7546d32b24a5c5bcfd76f8f4ad99'
EQUIVALENCE_HARNESS_SHA256='ba2bf73c41c5dc9048eead5db3fdf46a86a4452f43d421dfbb7b241a6b68c3a3'
REFERENCE_VERIFIER_SHA256='c3556963872dc0356050c617baf0a9847c229ebf24865dedee37d756455b71af'
EXPECTED_EXECUTION_SETTINGS={'cache_first_mean':True,'cache_csr_blocks':True,'packed_h2d':True,'gather_mode':'row_sentinel'}

def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def require(ok,message):
 if not ok:raise ValueError(message)
def read(p):return json.loads(Path(p).read_text())
def write(p,value):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_name(p.name+f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)
def rel(p):return Path(p).resolve().relative_to(ROOT).as_posix()
def path(name):
 p=(ROOT/name).resolve();require(p.is_relative_to(ROOT) and p.is_file(),'Uncontained/missing artifact: '+str(name));return p
def sha(p):
 p=Path(p);s=p.stat();key=(s.st_size,s.st_mtime_ns)
 if p in HASH_CACHE and HASH_CACHE[p][0]==key:return HASH_CACHE[p][1]
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 value=h.hexdigest();HASH_CACHE[p]=(key,value);return value
def ash(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def binding(mapping,receipt=None):
 for name,expected in mapping.items():
  require(sha(path(name))==expected,'Artifact/source SHA differs: '+name)
  if receipt is not None:receipt['verified_artifact_sha256'][name]=expected
def close(a,b,label,atol=TOL,rtol=0):
 a,b=np.asarray(a,dtype=np.float64),np.asarray(b,dtype=np.float64)
 require(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all() and np.allclose(a,b,atol=atol,rtol=rtol),label)
 return float(np.max(np.abs(a-b),initial=0))
def peak_rss_bytes():
 """Darwin ru_maxrss is bytes; Linux ru_maxrss is KiB."""
 n=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
 return int(n if sys.platform=='darwin' else n*1024)
def setup_cpu():
 torch.set_num_threads(2)
 try:torch.set_num_interop_threads(1)
 except RuntimeError:require(torch.get_num_interop_threads()==1,'Existing torch interop thread contract')

def ranking_metrics(y,score):
 """Independent AP/AUROC from complete scores with stable exact tie groups."""
 y=np.asarray(y,dtype=np.int8);score=np.asarray(score,dtype=np.float64)
 require(y.ndim==score.ndim==1 and len(y)==len(score) and set(np.unique(y))=={0,1} and np.isfinite(score).all(),'Binary ranking population')
 order=np.argsort(-score,kind='stable');labels=y[order];s=score[order]
 ends=np.r_[np.flatnonzero(s[:-1]!=s[1:]),len(y)-1]
 tp=np.cumsum(labels,dtype=np.int64)[ends];dp=np.diff(np.r_[0,tp]);dn=np.diff(np.r_[0,ends+1])-dp
 positives=int(y.sum());negatives=len(y)-positives
 ap=float(np.sum(dp*(tp/(ends+1)))/positives)
 below=negatives-np.cumsum(dn)
 auc=float(np.sum(dp*(below+.5*dn))/(positives*negatives))
 return ap,auc
def metrics(y,score,pred):
 y=np.asarray(y,dtype=np.int8);pred=np.asarray(pred,dtype=np.int8)
 require(y.shape==pred.shape and set(np.unique(pred))<={0,1},'Binary metric outputs')
 tn=int(np.sum((y==0)&(pred==0)));fp=int(np.sum((y==0)&(pred==1)));fn=int(np.sum((y==1)&(pred==0)));tp=int(np.sum((y==1)&(pred==1)))
 ap,auc=ranking_metrics(y,score);precision=tp/(tp+fp) if tp+fp else 0.;recall=tp/(tp+fn);specificity=tn/(tn+fp)
 return dict(ap=ap,auc=auc,precision=precision,recall=recall,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,accuracy=(tp+tn)/len(y),balanced_accuracy=(recall+specificity)/2,specificity=specificity,fpr=1-specificity,fnr=1-recall,tn=tn,fp=fp,fn=fn,tp=tp,n_test=len(y),n_positive=int(y.sum()))
def threshold(y,score):
 require(2./float(len(y)+int(np.sum(y)))**2>TOL+16*np.finfo(float).eps,'Threshold tolerance cannot merge attainable rationalF1 values')
 p,r,t=precision_recall_curve(y,score,drop_intermediate=False)
 f=np.divide(2*p[:-1]*r[:-1],p[:-1]+r[:-1],out=np.zeros_like(p[:-1]),where=p[:-1]+r[:-1]>0)
 index=int(np.flatnonzero(abs(f-f.max())<=TOL)[0]);return float(t[index]),float(f[index])
def normalize_graph(g):
 g=g.tocsr().astype(np.float32);degree=np.diff(g.indptr)
 require(g.has_sorted_indices and np.all(g.data==1),'Sorted simple graph, binary edges')
 g.data[:]=np.repeat((1/np.maximum(degree,1)).astype(np.float32),degree)
 return g
def state_forward(state,X,graphs,rows,name='graphsage_weak',chunk=16384):
 """Raw affine+CSR equations; no training class, dropout disabled by omission."""
 F=torch.nn.functional;rows=np.asarray(rows,dtype=np.int64)
 require(X.ndim==2 and X.dtype==np.float32 and chunk>0,'CPU float32 feature matrix')
 with torch.no_grad():
  if name=='mlp':
   result=np.empty(len(rows),np.float64)
   for start in range(0,len(rows),chunk):
    index=rows[start:start+chunk];x=torch.from_numpy(np.ascontiguousarray(X[index]));h=torch.relu(F.linear(x,state['layers.0.weight'],state['layers.0.bias']));z=F.linear(h,state['layers.1.weight'],state['layers.1.bias']).squeeze(-1);result[start:start+len(index)]=torch.sigmoid(z).double().numpy()
   return result
  require(name in ('graphsage_weak','graphsage_directed') and len(graphs)==(1 if name=='graphsage_weak' else 2),'Graph relation count')
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

def audit_checkpoint(model_path,raw,y,graphs,fit_rows,early_rows,params,seed,score_rows,saved,expected_epochs,name,input_dim,receipt,device='cuda'):
 """Full retained checkpoint/scaler/scope/epoch/raw-score audit; never fit."""
 p=path(model_path);payload=torch.load(p,map_location='cpu',weights_only=True);meta=payload['metadata'];state=payload['state_dict']
 require(meta==read(p.with_suffix('.json')) and meta['model']==name and meta['params']==params and meta['random_state']==seed and meta['device']==device,'PT/JSON/device/candidate identity')
 require(meta.get('execution_optimizations')==EXPECTED_EXECUTION_SETTINGS,'Every checkpoint belongs to the registered optimized execution, without old-fit substitution')
 require(raw.shape[1]==input_dim and meta.get('input_dim',input_dim)==input_dim,'Registered generic input dimension')
 require(meta['fit_rows_sha256']==ash(fit_rows) and meta['scaler_fit_rows_sha256']==ash(fit_rows) and meta['early_rows_sha256']==(None if early_rows is None else ash(early_rows)) and meta['scaler_n_samples_seen']==len(fit_rows) and (meta['fit_n'],meta['fit_positive'])==(len(fit_rows),int(y[fit_rows].sum())) and meta['seed_only_loss'] is True,'Fit-only scaler/seed-only loss scopes')
 fitraw=np.asarray(raw[fit_rows],dtype=np.float64);mu=fitraw.mean(axis=0);centered=fitraw-mu;var=(np.square(centered).sum(axis=0)-np.square(centered.sum(axis=0))/len(fitraw))/len(fitraw)
 eps=np.finfo(np.float64).eps;constant=var<=len(fitraw)*eps*var+(len(fitraw)*mu*eps)**2;scale=np.sqrt(var);scale[constant]=1.
 # Corrected two-pass float64 statistics; relative tolerance covers summation
 # roundoff for unlogged structural magnitudes, never used for score/metric ties.
 for field,expected in [('scaler_mean',mu),('scaler_var',var),('scaler_scale',scale)]:close(meta[field],expected,'Independent fit-only '+field,rtol=1e-11)
 n=len(fit_rows);positive=int(y[fit_rows].sum());weights=np.array([n/(2*(n-positive)),n/(2*positive)],dtype=np.float32);require(np.array_equal(weights,np.asarray(meta['class_weights'],np.float32)),'Fit-only BCE balance weights')
 del fitraw,centered;gc.collect()
 hidden=int(params['hidden_dim']);dims={}
 if name=='mlp':dims={'layers.0.weight':(hidden,input_dim),'layers.0.bias':(hidden,),'layers.1.weight':(1,hidden),'layers.1.bias':(1,)}
 else:
  for layer,(d,o) in enumerate([(input_dim,hidden),(hidden,1)]):
   for relation in range(1 if name=='graphsage_weak' else 2):
    dims[f'root.{layer}.{relation}.weight']=(o,d);dims[f'root.{layer}.{relation}.bias']=(o,);dims[f'neighbor.{layer}.{relation}.weight']=(o,d)
   if name=='graphsage_directed':dims[f'fusion.{layer}.weight']=(o,2*o);dims[f'fusion.{layer}.bias']=(o,)
 require(set(state)==set(dims) and all(tuple(state[k].shape)==shape and state[k].dtype==torch.float32 and state[k].device.type=='cpu' and torch.isfinite(state[k]).all() for k,shape in dims.items()),'Generic affine tensor shapes; no identity embedding')
 digest=hashlib.sha256()
 for key,value in state.items():digest.update(key.encode());digest.update(value.contiguous().numpy().tobytes())
 require(digest.hexdigest()==meta['model_digest'],'Raw state tensor digest')
 history=meta['history'];require([x['epoch'] for x in history]==list(range(1,len(history)+1)) and meta['epochs_run']==len(history) and meta['selected_epochs']==expected_epochs,'Full epoch history/selected epoch')
 require(all(np.isfinite(x['weighted_seed_loss']) and x['weighted_seed_loss']>=0 and np.isfinite(x['seconds']) for x in history),'Finite saved loss andtimers')
 if early_rows is not None:
  aucs=np.asarray([x['early_auroc'] for x in history]);best=int(np.argmax(aucs))+1
  require(best==expected_epochs and len(history)<=200 and (len(history)==200 or len(history)-best>=30),'First best AUROC/patience30')
  close(meta['best_early_auroc'],aucs.max(),'Best earlyAUROC')
 else:require(meta['best_early_auroc'] is None and len(history)==expected_epochs,'Full fixedepoch refit')
 X=np.array(raw,dtype=np.float32,copy=True);np.subtract(X,np.asarray(meta['scaler_mean']),out=X,casting='unsafe');np.divide(X,np.asarray(meta['scaler_scale']),out=X,casting='unsafe')
 score=state_forward(state,X,graphs,score_rows,name);diff=close(score,saved,'Independent CPU affine/CSR allscores')
 require(ash(score)==ash(saved),'Exact CPU2 inference scorebytes')
 if early_rows is not None:close(ranking_metrics(y[early_rows],state_forward(state,X,graphs,early_rows,name))[1],meta['best_early_auroc'],'Best retainedcheckpoint earlyAUROC')
 receipt['physical_models_rescored'][rel(p)]=dict(sha256=sha(p),n_features=input_dim,rows=len(score_rows),max_abs_diff=diff,exact_score_bytes=True,fit_rows=len(fit_rows),model_digest=meta['model_digest'])
 del X,payload,state,score;gc.collect();return meta

class Validator:
 """One experiment-root context; no import of the CUDA runner."""
 def __init__(self,out):
  global ROOT
  ROOT=Path(out).resolve();self.root=ROOT;self.dest=ROOT/'validation';self.cache={};self.graph_cache={}
  setup_cpu();self.p=read(ROOT/'protocol.json');self.psha=sha(ROOT/'protocol.json');self.c=read(ROOT/'candidates.json')['candidates']
  self.sources={'protocol.json':self.psha,'candidates.json':sha(ROOT/'candidates.json')}
  self.source_gate()
 def source_gate(self):
  require([x['candidate_id'] for x in self.c]==list(range(12)),'Shared complete12 candidate order')
  require({(x['params']['hidden_dim'],x['params']['learning_rate'],x['params']['dropout']) for x in self.c}==set(itertools.product((64,128,256),(.001,.003),(0.,.2))),'Registered neural candidate space')
  require(self.p['device'].startswith('cuda'),'CUDA comparison backend, no MPS reference substitution')
  require(set(self.p['datasets'])=={'hi_small','li_small'} and self.p['feature_sets']==['base16','base_g32','gmke44','gmkep56'] and self.p['model']=='graphsage_weak','Complete two-dataset/four-arm registry')
  require(self.p['budget']=={'arms_per_dataset':4,'outer_folds_per_setting':25,'inner_candidate_scores_per_outer':36,'stages_per_outer':4},'Untrimmed200outer/7200inner budget')
  # A bound root manifest must enumerate every uploaded input and source file.
  sourcefile=ROOT/self.p.get('source_binding_path','source_binding.json')
  require(sourcefile.exists(),'Immutable CUDA source/input manifest absent')
  manifest=read(sourcefile);mapping=manifest.get('source_sha256',manifest)
  require(isinstance(mapping,dict) and mapping and all(isinstance(v,str) and len(v)==64 for v in mapping.values()),'Source binding SHA map')
  require(self.p['source_binding_sha256']==sha(sourcefile),'Protocol-bound source manifest')
  binding(mapping);self.sources.update(mapping);self.sources[rel(sourcefile)]=sha(sourcefile)
  runner=ROOT/'runner_snapshot.py';require(runner.exists() and self.p['runner_sha256']==sha(runner),'Registered frozenCUDA source snapshot')
  self.sources[rel(runner)]=sha(runner)
  require(len(EXPECTED_OPTIMIZED_RUNNER_SHA256)==64 and sha(runner)==EXPECTED_OPTIMIZED_RUNNER_SHA256,'Fixed independently audited optimized runner identity')
  require(self.p.get('execution_settings')==EXPECTED_EXECUTION_SETTINGS,'Exact registered execution-only optimization switches')
  self.optimization_gate()
  probe=ROOT/'probe.json';require(probe.exists(),'ActualCUDA primitive/protocol probe required')
  pr=read(probe);require(pr['passed'] and pr['device']=='cuda' and pr['runner_sha256']==sha(runner),'CUDA probe/source identity')
  require(len(pr['tests'])==4 and {x['input_dim'] for x in pr['tests']}=={16,32,44,56} and all(x['serial_prefetch_2epoch_weights_losses_sampling_bitwise'] and x['nonfit_label_invariance'] and x['isolates_retained'] and x['qa_epochs']==2 and x['qa_minibatches_total']==4 for x in pr['tests']),'Allwidth actualCUDA serial/prefetch/dropout/RNG/nonfitlabel invariant probe')
  self.sources[rel(probe)]=sha(probe)
  transfer=ROOT/'transfer_manifest.json';require(transfer.exists(),'Input transferprovenance required')
  tm=read(transfer);require(all(name in mapping and mapping[name]==h for name,h in tm['input_sha256'].items()),'All61 original transferred inputSHA match preparedsource binding')
  require(len(tm['input_sha256'])==61,'Alltwo datasets/full50splits/candidate transfer budget')
  self.sources[rel(transfer)]=sha(transfer)
  functions={n.name:n for n in ast.parse(runner.read_text()).body if isinstance(n,ast.FunctionDef)}
  fit=ast.unparse(functions['fit']);search=ast.unparse(functions['search']);configure=ast.unparse(functions['configure'])
  require('raw[fit_rows]' in fit and 'rng.permutation(fit_rows)' in fit and 'gpuY[root_t]' in fit and 'gpuW[labels.long()]' in fit and "reduction='none'" in fit,'Source fit-only scaler/seed-only weighted loss')
  require('outer_test' not in search and 'threshold_validation' not in search,'Search excluded T/test')
  require('set_num_threads(2)' in configure and 'set_num_interop_threads(1)' in configure and 'use_deterministic_algorithms(True)' in configure and 'allow_tf32 = False' in configure,'CPU2/CUDA deterministic float32 source contract')
 def optimization_gate(self):
  """A protocol-bound independent target-CUDA QA closure; never execute it."""
  acceptance=path('optimization_acceptance.json');value=read(acceptance)
  require(self.p.get('optimization_acceptance_sha256')==sha(acceptance),'Protocol-bound immutable optimization acceptance')
  require(value.get('schema_version')==1 and value.get('status')=='accepted_after_independent_cuda_equivalence' and value.get('experiment_namespace')=='graphsage_cuda_optimized_20261018','Explicit optimized-only namespace acceptance')
  require(value.get('no_old_fit_migration') is True and value.get('new_namespace_full_restart') is True and value.get('formal_budget_fit_count')==8000,'All8000 fits start in new namespace; retained old35 fits are references only')
  require(value.get('old_reference_runner_sha256')==REFERENCE_RUNNER_SHA256 and value.get('optimized_runner_sha256')==EXPECTED_OPTIMIZED_RUNNER_SHA256 and value.get('independent_regression_harness_sha256')==EQUIVALENCE_HARNESS_SHA256,'Accepted old/new/harness frozen identities')
  report=path(value['independent_cuda_report']);require(sha(report)==value['independent_cuda_report_sha256'],'Independent target-CUDA regression report identity')
  qa=read(report);expected_sources={'old':REFERENCE_RUNNER_SHA256,'optimized':EXPECTED_OPTIMIZED_RUNNER_SHA256,'verifier':REFERENCE_VERIFIER_SHA256,'harness':EQUIVALENCE_HARNESS_SHA256}
  require(qa.get('status')=='passed' and qa.get('device')=='cuda' and qa.get('cuda_equivalence_passed') is True and qa.get('full_registered_widths_tested') is True and qa.get('source_sha256')==expected_sources,'Actual independent allwidth CUDA byte-equivalence, never CPU fallback')
  require(qa.get('widths')==[16,32,44,56] and qa.get('gather_mode')=='row_sentinel' and qa.get('epochs')==4 and qa.get('batches_per_epoch')==5 and qa.get('formal_fits')==0 and qa.get('synthetic_only') is True and qa.get('CPU_threads')==2,'Complete registered independent synthetic regression budget')
  cases=qa.get('cases',[]);require(len(cases)==8 and {(x['width'],x['phase']) for x in cases}==set(itertools.product((16,32,44,56),('early','fixed_refit'))),'Fourwidth early/refit traced and untraced regression cases')
  flags=('passed','exact_gradients','exact_adam_state','exact_each_step_parameters','exact_epoch_state','exact_sampling','exact_batch_loss','exact_early_trajectory','exact_untraced_model','exact_raw_scores','independent_CPU_affine_CSR_exact')
  require(all(all(x.get(k) is True for k in flags) and x.get('optimizer_steps_per_source')==40 and all(x.get('coverage',{}).get(k,0)>0 for k in ('invalid','valid_node_zero','valid_position_zero')) for x in cases),'Byte-exact sampled steps/Adam/early/untraced/rawscores including valid index0+padding')
  gathers=qa.get('adversarial_gather',[]);require(len(gathers)==8 and {x['hidden'] for x in gathers}=={64,128} and {x['fanout'] for x in gathers}=={10,15} and all(x.get('passed') is True and x.get('target_count')==2048 and x.get('valid_node_zero',0)>0 for x in gathers) and any(x.get('invalid',0)>18000 for x in gathers),'Adversarial highpadding/isolates and valid-node0 gather gradient regression')
  caches=qa.get('cache_regression',[]);require(len(caches)==4 and {x['width'] for x in caches}=={16,32,44,56} and all(x.get('chunks')==2 and x.get('exact_CSR_and_mean') is True and x.get('cache_reused') is True and x.get('weakref_cleanup_passed') is True for x in caches),'Immutable-X twochunk CSR mean and cache-lifetime regression')
  expected_artifacts={rel(report):sha(report)}
  for filename,digest in [('reference_snapshot.py',REFERENCE_RUNNER_SHA256),('optimized_snapshot.py',EXPECTED_OPTIMIZED_RUNNER_SHA256),('independent_verifier_snapshot.py',REFERENCE_VERIFIER_SHA256),('regression_snapshot.py',EQUIVALENCE_HARNESS_SHA256)]:
   snapshot=report.parent/filename;require(snapshot.is_relative_to(ROOT) and sha(snapshot)==digest,'Independent regression frozen source closure: '+filename);expected_artifacts[rel(snapshot)]=digest
  require(value.get('source_sha256')==expected_artifacts,'Complete exact optimization-acceptance source/report SHA map')
  binding(expected_artifacts);self.sources.update(expected_artifacts);self.sources[rel(acceptance)]=sha(acceptance)
 def input_binding(self,ds):
  prefix=f'inputs/{ds}/';expected={n:h for n,h in self.sources.items() if n.startswith(prefix)}
  must=[prefix+n for n in ['features.npy','labels.npy','names.json','accounts.parquet','weak_csr.npz']]
  require(all(n in expected for n in must),'Full input identity/features/graph immutable bindings')
  binding(expected);return expected
 def dataset(self,ds,arm):
  key=(ds,arm)
  if key in self.cache:return self.cache[key]
  self.cache.clear();gc.collect();bound=self.input_binding(ds);folder=ROOT/'inputs'/ds
  names=read(folder/'names.json');allcols=names['columns'];base=names['base16'];cols=names['feature_sets'][arm]
  require(len(allcols)==56 and len(set(allcols))==56 and len(base)==16 and base==allcols[:16] and all(c in allcols for c in cols),'Frozen original56 feature dictionary')
  require(arm in ('base16','base_g32','gmke44','gmkep56') and len(cols)=={'base16':16,'base_g32':32,'gmke44':44,'gmkep56':56}[arm] and cols[:16]==base,'Registered arm dimensions/sharedBASE16 prefix')
  require(self.p['datasets'][ds]['feature_sets'][arm]==cols and self.p['datasets'][ds]['base16']==base,'Global protocolper-dataset feature dictionary')
  original=np.load(folder/'features.npy',mmap_mode='r');y=np.load(folder/'labels.npy',mmap_mode='r');ids=pd.read_parquet(folder/'accounts.parquet')
  require(original.shape==(len(ids),56) and original.dtype==np.float64 and y.shape==(len(ids),) and y.dtype==np.int8,'Original f64 source56 / binarylabels')
  require(set(np.unique(y))=={0,1} and np.array_equal(ids.row_index,np.arange(len(ids))) and ids.account_id.is_unique,'Complete original accountorder inclisolated')
  if 'label' in ids:require(np.array_equal(ids.label,y),'Optional accountlabel column agrees withimmutable labels array')
  registered=self.p['datasets'][ds]
  require(registered['n_samples']==len(y) and registered['n_positive']==int(y.sum()) and registered['labels_array_sha256']==ash(y) and registered['account_ids_hash_sha256']==ash(pd.util.hash_pandas_object(ids.account_id,index=False).to_numpy()),'Protocoloriginal population/label/identity digest')
  index=[allcols.index(c) for c in cols];raw=np.empty((len(y),len(cols)),np.float32)
  for start in range(0,len(y),65536):
   stop=min(start+65536,len(y));block=np.array(original[start:stop,index],dtype=np.float64,copy=True)
   require(np.isfinite(block).all() and (block[:,:16]>=0).all(),'Finite original fields and nonnegativeBASE')
   block[:,:16]=np.log1p(block[:,:16]);raw[start:stop]=block.astype(np.float32)
  require(np.isfinite(raw).all(),'Finite transformedfeature matrix')
  self.cache[key]=(raw,y,ids,cols)
  return self.cache[key]
 def graphs(self,ds):
  if ds in self.graph_cache:return self.graph_cache[ds]
  self.graph_cache.clear();gc.collect();self.input_binding(ds)
  g=sparse.load_npz(ROOT/'inputs'/ds/'weak_csr.npz').tocsr();n=len(np.load(ROOT/'inputs'/ds/'labels.npy',mmap_mode='r'))
  require(g.shape==(n,n) and g.has_canonical_format and g.has_sorted_indices and (g!=g.T).nnz==0 and g.diagonal().sum()==0 and np.all(g.data==1),'All-account simpleweak projection/no loops')
  self.graph_cache[ds]=[normalize_graph(g)];return self.graph_cache[ds]
 def split(self,ds,seed,fold):
  p=ROOT/'inputs'/ds/'splits'/f'seed{seed}_fold{fold}.npz';require(rel(p) in self.sources and sha(p)==self.sources[rel(p)],'Registered copiedsplit SHA')
  with np.load(p,allow_pickle=False) as f:a={k:f[k] for k in f.files}
  n=len(np.load(ROOT/'inputs'/ds/'labels.npy',mmap_mode='r'))
  for k,v in a.items():require(v.ndim==1 and np.issubdtype(v.dtype,np.integer) and len(np.unique(v))==len(v) and (v>=0).all() and (v<n).all(),'Valid unique splitrow indices: '+k)
  def partition(left,right,whole,label):
   require(not np.intersect1d(left,right).size and np.array_equal(np.sort(np.r_[left,right]),np.sort(whole)),label)
  partition(a['outer_train'],a['outer_test'],np.arange(n),'Outer partition full population')
  partition(a['development'],a['threshold_validation'],a['outer_train'],'D/T within outertraining')
  for j in range(3):
   partition(a[f'inner_train_{j}'],a[f'inner_score_{j}'],a['development'],'Innerfit/score partition')
   partition(a[f'early_fit_{j}'],a[f'early_validation_{j}'],a[f'inner_train_{j}'],'Earlyfit/eval partition')
  require(np.array_equal(np.sort(np.concatenate([a[f'inner_score_{j}'] for j in range(3)])),np.sort(a['development'])),'Threeinner score folds coverD exactly')
  return a,p
 def group_binding(self,ds,arm):
  _,y,ids,cols=self.dataset(ds,arm)
  return dict(dataset=ds,feature_set=arm,input_dim=len(cols),columns=cols,protocol_sha256=self.psha,source_binding_sha256=self.sources,account_count=len(y),row_index_digest=ash(ids.row_index.to_numpy()),label_digest=ash(y),preprocessing='Original56 float64; commonBASE16 log1p beforefloat32;extra columns identityfloat32;currentfit-only StandardScaler')
 def candidate(self,ds,arm,seed,fold,cid,j,receipt):
  folder=ROOT/'runs'/ds/arm;stem=f'graphsage_weak_seed{seed}_fold{fold}';rp=folder/'search'/stem/f'candidate{cid}_inner{j}_record.json';v=read(rp)
  binding(v['artifact_sha256'],receipt);receipt['verified_artifact_sha256'][rel(rp)]=sha(rp);a,_=self.split(ds,seed,fold);raw,y,ids,cols=self.dataset(ds,arm)
  require([v[k] for k in ['dataset','model','feature_set','seed','fold','candidate_id','inner_fold']]==[ds,'graphsage_weak',arm,seed,fold,cid,j] and v['protocol_sha256']==self.psha and v['params']==self.c[cid]['params'],'Candidate registeredidentity/params')
  fit,early,score_rows=(a[f'{k}_{j}'] for k in ['early_fit','early_validation','inner_score'])
  require(v['fit_rows_sha256']==ash(fit) and v['early_rows_sha256']==ash(early) and v['score_rows_sha256']==ash(score_rows),'Candidate row scope digests')
  with np.load(path(v['score_path']),allow_pickle=False) as z:
   require(set(z.files)=={'row_index','y_true','y_score'} and np.array_equal(z['row_index'],score_rows) and np.array_equal(z['y_true'],y[score_rows]) and z['y_score'].dtype==np.float64,'Full savedscore row/label identity');score=z['y_score'].copy()
  require(np.isfinite(score).all() and ((score>=0)&(score<=1)).all() and ash(score)==v['raw_score_sha256'],'Probabilityfinite/rawscore byte hash')
  close(ranking_metrics(y[score_rows],score),[v['ap'],v['auc']],'Independentcomplete innerAP/AUC')
  # Independent tie-group arithmetic above checks the statistical value. Exact
  # registered-library float values additionally certify sub-tolerance candidate
  # ties cannot be changed by a forged/rounded saved scalar.
  require(v['ap']==float(average_precision_score(y[score_rows],score)) and v['auc']==float(roc_auc_score(y[score_rows],score)),'Canonical innerAP/AUC exactfloat contract for selectionties')
  meta=audit_checkpoint(v['model_path'],raw,y,self.graphs(ds),fit,early,v['params'],40000+1000*seed+10*fold+j,score_rows,score,v['selected_epochs'],'graphsage_weak',len(cols),receipt,self.p['device'])
  require(meta['feature_set']==arm and meta['input_dim']==len(cols) and meta['model_digest']==v['model_digest'],'Candidate featurearm/tensor digest')
  return v
 def fold(self,ds,arm,seed,fold,force=False):
  folder=ROOT/'runs'/ds/arm;stem=f'graphsage_weak_seed{seed}_fold{fold}';fp=folder/'folds'/(stem+'.json');rp=self.dest/ds/arm/(stem+'.json')
  if not force and rp.exists():
   old=read(rp)
   if old.get('passed') and old.get('verifier_sha256')==sha(SCRIPT) and old.get('fold_done_sha256')==sha(fp):
    binding(old['verified_artifact_sha256']);binding(old['group_binding']['source_binding_sha256']);return old
  receipt=dict(passed=False,status='validating_single_cuda_outer',dataset=ds,feature_set=arm,model='graphsage_weak',seed=seed,fold=fold,started_at=now(),verifier_sha256=sha(SCRIPT),protocol_sha256=self.psha,verified_artifact_sha256={},physical_models_rescored={})
  try:
   d=read(fp);require([d[k] for k in ['dataset','model','feature_set','seed','fold']]==[ds,'graphsage_weak',arm,seed,fold] and d['protocol_sha256']==self.psha,'Canonical DONE identity')
   require(d['logical_candidate_evaluations']==36 and d['logical_strategy_models']==4,'Full outer40logical model budget');binding(d['artifact_sha256'],receipt)
   sr=d['search'];srp=folder/'search'/stem/'search_result.json';require(sr==read(srp),'Embeddedcomplete searchidentity');binding(sr['artifact_sha256'],receipt);receipt['verified_artifact_sha256'][rel(srp)]=sha(srp)
   require(len(sr['candidate_records'])==36 and {(x['candidate_id'],x['inner_fold']) for x in sr['candidate_records']}==set(itertools.product(range(12),range(3))),'Complete12x3 grid')
   for item in sr['candidate_records']:require(self.candidate(ds,arm,seed,fold,item['candidate_id'],item['inner_fold'],receipt)==item,'Candidate embeddedrecord identity')
   summaries=[]
   for cid in range(12):
    vals=[x for x in sr['candidate_records'] if x['candidate_id']==cid];summaries.append(dict(candidate_id=cid,mean_ap=float(np.mean([x['ap'] for x in vals])),std_ap=float(np.std([x['ap'] for x in vals],ddof=1)),epochs=int(np.median([x['selected_epochs'] for x in vals]))))
   require([x['candidate_id'] for x in sr['candidate_summary']]==list(range(12)),'Complete summarycandidate order')
   for actual,expected in zip(sr['candidate_summary'],summaries):require(actual==expected,'Independentexact mean AP/SD/epochmedian')
   best=min(summaries,key=lambda x:(-x['mean_ap'],x['candidate_id']));require(sr['selected_candidate_id']==best['candidate_id'] and sr['selected_epochs']==best['epochs'] and sr['anchor_epochs']==summaries[0]['epochs'],'ExactAP tie lowestid andinteger median epochs');close(sr['selected_mean_ap'],best['mean_ap'],'Selectedmean AP')
   a,sp=self.split(ds,seed,fold);receipt['verified_artifact_sha256'][rel(sp)]=sha(sp);raw,y,ids,cols=self.dataset(ds,arm);train,test,D,T=(a[k] for k in ['outer_train','outer_test','development','threshold_validation'])
   require(len(d['strategies'])==2 and {x['strategy'] for x in d['strategies']}=={'same_protocol_anchor','tuned'},'Anchor/tuned complete strategies')
   for st in d['strategies']:
    cid=0 if st['strategy']=='same_protocol_anchor' else best['candidate_id'];epochs=summaries[cid]['epochs'];params=self.c[cid]['params'];binding(st['artifact_sha256'],receipt)
    require(st['protocol_sha256']==self.psha and st['candidate_id']==cid and st['selected_epochs']==epochs and st['params']==params and st['model_random_state']==100*seed+fold,'Full strategyselection/config')
    require(all(st[k]==ash(v) for k,v in [('fit_rows_sha256',train),('calibration_fit_rows_sha256',D),('threshold_rows_sha256',T),('outer_test_rows_sha256',test)]),'D/T/test scope digest')
    cal=pd.read_parquet(path(st['calibration_path']));pred=pd.read_parquet(path(st['prediction_path']))
    for table,index in [(cal,T),(pred,test)]:require(np.array_equal(table.row_index,index) and np.array_equal(table.account_id,ids.iloc[index].account_id) and np.array_equal(table.y_true,y[index]) and table.y_score.dtype==np.float64 and table.y_score.between(0,1).all(),'Allsaved prediction identity/truth/score')
    cutoff,bestf1=threshold(y[T],cal.y_score.to_numpy());close([cutoff,bestf1],[st['threshold'],st['threshold_validation_f1']],'Independent T maxF1/lowestthreshold tie');close(ranking_metrics(y[T],cal.y_score.to_numpy())[0],st['threshold_validation_ap'],'Independent T AP')
    for role,fit,index,table in [('calibration',D,T,cal),('final',train,test,pred)]:
     modelpath=st['calibration_model_path'] if role=='calibration' else st['model_path'];meta=audit_checkpoint(modelpath,raw,y,self.graphs(ds),fit,None,params,100*seed+fold,index,table.y_score.to_numpy(),epochs,'graphsage_weak',len(cols),receipt,self.p['device']);require(meta==st[role+'_metadata'] and meta['feature_set']==arm and meta['input_dim']==len(cols),'Stage embeddedmetadata/featurearm')
    require(ash(pred.y_score.to_numpy())==st['raw_score_sha256'],'Final scorebytes')
    require(len(st['metrics'])==2 and {x['threshold_policy'] for x in st['metrics']}==set(POLICIES),'Complete two thresholdmetric policies')
    for policy,column,value in [('fixed_0.5','y_pred_fixed',.5),('inner_max_f1','y_pred_inner',cutoff)]:
     predicted=(pred.y_score.to_numpy()>=value).astype(np.int8);require(np.array_equal(pred[column],predicted),'Savedbinarythreshold outputs');actual=next(x for x in st['metrics'] if x['threshold_policy']==policy);computed=metrics(y[test],pred.y_score.to_numpy(),predicted)
     require([actual[k] for k in ['dataset','model','feature_set','strategy','seed','fold','selected_candidate_id','selected_epochs']]==[ds,'graphsage_weak',arm,st['strategy'],seed,fold,cid,epochs],'Metricrow identity')
     require(all(k in actual for k in ['ap','auc','precision','recall','f1','accuracy','balanced_accuracy','specificity','tn','fp','fn','tp','n_test','n_positive']),'Allseven metrics/confusion fields')
     close([actual[k] for k in computed if k in actual],[computed[k] for k in computed if k in actual],'Independentseven metrics/confusion');close(actual['threshold'],value,'Metricthreshold')
   receipt['verified_artifact_sha256'][rel(fp)]=sha(fp);require(len(receipt['physical_models_rescored'])==40,'All36inner+4strategy physicalmodels rescored')
   receipt.update(passed=True,status='passed_single_cuda_outer_not_complete_setting',completed_at=now(),fold_done_path=rel(fp),fold_done_sha256=sha(fp),candidate_models_and_scores_verified=36,strategy_models_verified=4,group_binding=self.group_binding(ds,arm),validator_peak_rss_bytes=peak_rss_bytes(),scaler_audit_tolerance={'atol':TOL,'rtol':1e-11},score_byte_equality=True,scope='Singleouterfold only; fullgrouprequires25verified folds. No trainer import/no fit; independentCPU2 affine+CSR andfullscores,fit-onlyscalers,selection/epochmedian,T thresholdandmetrics.')
   write(rp,receipt);return receipt
  except Exception:receipt.update(status='failed',failed_at=now(),error=traceback.format_exc());write(rp,receipt);raise
 def overview(self):
  settings=[]
  for ds,arm in itertools.product(('hi_small','li_small'),('base16','base_g32','gmke44','gmkep56')):
   done=verified=0
   for seed,fold in itertools.product(range(5),range(5)):
    stem=f'graphsage_weak_seed{seed}_fold{fold}';fp=ROOT/'runs'/ds/arm/'folds'/(stem+'.json');rp=self.dest/ds/arm/(stem+'.json')
    if fp.exists():
     done+=1
     if rp.exists():
      v=read(rp)
      if v.get('passed') and v.get('verifier_sha256')==sha(SCRIPT) and v.get('protocol_sha256')==self.psha and v.get('fold_done_sha256')==sha(fp):verified+=1
   settings.append(dict(dataset=ds,feature_set=arm,model='graphsage_weak',training_done_folds=done,independently_verified_folds=verified,expected_folds=25,status='complete_independently_verified' if verified==25 else 'partial_not_complete'))
  result=dict(status='complete' if sum(x['independently_verified_folds'] for x in settings)==200 else 'partial',updated_at=now(),protocol_sha256=self.psha,verifier_sha256=sha(SCRIPT),planned_settings=8,planned_outer_folds=200,training_done_outer_folds=sum(x['training_done_folds'] for x in settings),independently_verified_outer_folds=sum(x['independently_verified_folds'] for x in settings),complete_independently_verified_settings=sum(x['independently_verified_folds']==25 for x in settings),settings=settings,no_partial_means_reported=True)
  write(self.dest/'overview.json',result);return result
 def scan(self,ds=None,arm=None,max_folds=1,force=False):
  count=0;checked=[]
  for dataset,feature in itertools.product([ds] if ds else ('hi_small','li_small'),[arm] if arm else ('base16','base_g32','gmke44','gmkep56')):
   for seed,fold in itertools.product(range(5),range(5)):
    stem=f'graphsage_weak_seed{seed}_fold{fold}';fp=ROOT/'runs'/dataset/feature/'folds'/(stem+'.json');rp=self.dest/dataset/feature/(stem+'.json')
    if not fp.exists():continue
    previous=read(rp) if rp.exists() else {};already=previous.get('passed') and previous.get('verifier_sha256')==sha(SCRIPT) and previous.get('fold_done_sha256')==sha(fp)
    if max_folds is not None and count>=max_folds and not already:continue
    result=self.fold(dataset,feature,seed,fold,force=force);count+=int(not already or force);checked.append(rel(rp))
    print(json.dumps({'fold_passed':[dataset,feature,seed,fold],'models_rescored':len(result['physical_models_rescored'])}),flush=True)
  result=self.overview();write(self.dest/'last_scan.json',dict(checked_at=now(),verifier_sha256=sha(SCRIPT),new_folds_verified=count,receipt_paths=checked));return result
 def predictions(self,ds,arm,seed,fold,strategy='tuned'):
  stem=f'graphsage_weak_seed{seed}_fold{fold}';fp=ROOT/'runs'/ds/arm/'folds'/(stem+'.json');rp=self.dest/ds/arm/(stem+'.json');d=read(fp);receipt=read(rp)
  require(receipt.get('passed') and receipt['verifier_sha256']==sha(SCRIPT) and receipt['protocol_sha256']==self.psha and receipt['fold_done_sha256']==sha(fp),'Aggregation onlycurrent verifiedfold')
  st=next(x for x in d['strategies'] if x['strategy']==strategy);pp=path(st['prediction_path']);require(sha(pp)==st['artifact_sha256'][rel(pp)],'Aggregated predictionSHA')
  table=pd.read_parquet(pp);_,y,ids,cols=self.dataset(ds,arm);a,sp=self.split(ds,seed,fold);index=a['outer_test']
  require(np.array_equal(table.row_index,index) and np.array_equal(table.account_id,ids.iloc[index].account_id) and np.array_equal(table.y_true,y[index]),'Aggregation full accountidentity/truth')
  return table,st['metrics'],{rel(fp):sha(fp),rel(rp):sha(rp),rel(pp):sha(pp)}
 def aggregate(self):
  status=self.overview();complete={(x['dataset'],x['feature_set']) for x in status['settings'] if x['status']=='complete_independently_verified'};foldrows=[];seedrows=[];bindings={}
  for ds,arm in sorted(complete):
   for strategy in ('same_protocol_anchor','tuned'):
    for seed in range(5):
     pieces=[]
     for fold in range(5):
      table,values,b=self.predictions(ds,arm,seed,fold,strategy);pieces.append(table);foldrows.extend(values);bindings.update(b)
     oof=pd.concat(pieces).sort_values('row_index');_,y,ids,_=self.dataset(ds,arm)
     require(len(oof)==len(y) and oof.row_index.is_unique and np.array_equal(oof.row_index,np.arange(len(y))) and np.array_equal(oof.account_id,ids.account_id) and np.array_equal(oof.y_true,y),'Full25fold/5seed OOF coverage')
     for policy,column in [('fixed_0.5','y_pred_fixed'),('inner_max_f1','y_pred_inner')]:seedrows.append(dict(dataset=ds,model='graphsage_weak',feature_set=arm,strategy=strategy,seed=seed,threshold_policy=policy,**metrics(y,oof.y_score.to_numpy(),oof[column].to_numpy())))
  folder=self.dest/'complete_settings';folder.mkdir(parents=True,exist_ok=True)
  if foldrows:
   frame=pd.DataFrame(foldrows);frame.to_csv(folder/'verified_fold_metrics.csv',index=False);pd.DataFrame(seedrows).to_csv(folder/'verified_seed_oof_metrics.csv',index=False)
   keys=['dataset','model','feature_set','strategy','threshold_policy'];summary=frame.groupby(keys)[list(METRICS[:7])].agg(['mean','std']);summary.columns=['_'.join(x) for x in summary.columns];summary.reset_index().to_csv(folder/'verified_summary.csv',index=False)
  contrasts=[('gmkep56','gmke44'),('gmke44','base16'),('gmkep56','base16'),('base_g32','base16'),('gmke44','base_g32'),('gmkep56','base_g32')];pairs=[];pairseeds=[];skipped=[]
  for ds,(lhs,rhs) in itertools.product(('hi_small','li_small'),contrasts):
   if (ds,lhs) not in complete or (ds,rhs) not in complete:skipped.append(dict(dataset=ds,lhs=lhs,rhs=rhs,reason='Both25fold groups notcomplete'));continue
   for seed in range(5):
    pieces={lhs:[],rhs:[]}
    for fold in range(5):
     left,lm,lb=self.predictions(ds,lhs,seed,fold);right,rm,rb=self.predictions(ds,rhs,seed,fold);bindings.update(lb);bindings.update(rb);require(np.array_equal(left.row_index,right.row_index) and np.array_equal(left.y_true,right.y_true),'Pair sharedtest rows/truth');pieces[lhs].append(left);pieces[rhs].append(right)
     for policy,column in [('fixed_0.5','y_pred_fixed'),('inner_max_f1','y_pred_inner')]:
      a=metrics(left.y_true.to_numpy(),left.y_score.to_numpy(),left[column].to_numpy());b=metrics(right.y_true.to_numpy(),right.y_score.to_numpy(),right[column].to_numpy());pairs.append(dict(dataset=ds,lhs_arm=lhs,rhs_arm=rhs,seed=seed,fold=fold,threshold_policy=policy,**{'delta_'+k:a[k]-b[k] for k in METRICS}))
    left=pd.concat(pieces[lhs]).sort_values('row_index');right=pd.concat(pieces[rhs]).sort_values('row_index');_,y,_,_=self.dataset(ds,lhs)
    require(np.array_equal(left.row_index,np.arange(len(y))) and np.array_equal(right.row_index,np.arange(len(y))),'Paired OOF fullcoverage')
    for policy,column in [('fixed_0.5','y_pred_fixed'),('inner_max_f1','y_pred_inner')]:
     a=metrics(y,left.y_score.to_numpy(),left[column].to_numpy());b=metrics(y,right.y_score.to_numpy(),right[column].to_numpy());pairseeds.append(dict(dataset=ds,lhs_arm=lhs,rhs_arm=rhs,seed=seed,threshold_policy=policy,**{'delta_'+k:a[k]-b[k] for k in METRICS}))
  if pairs:
   frame=pd.DataFrame(pairs);sf=pd.DataFrame(pairseeds);frame.to_csv(folder/'verified_paired_fold_differences.csv',index=False);sf.to_csv(folder/'verified_paired_seed_oof_differences.csv',index=False);keys=['dataset','lhs_arm','rhs_arm','threshold_policy'];summary=frame.groupby(keys)[['delta_'+k for k in METRICS]].agg(['mean','std']);summary.columns=['_'.join(x) for x in summary.columns];summary=summary.reset_index().merge(frame.groupby(keys).delta_ap.agg(ap_positive_folds=lambda x:int((x>0).sum())).reset_index(),on=keys).merge(sf.groupby(keys).delta_ap.agg(ap_positive_oof_seeds=lambda x:int((x>0).sum())).reset_index(),on=keys);summary.to_csv(folder/'verified_paired_summary.csv',index=False)
  result=dict(passed=True,status='complete' if len(complete)==8 else 'partial_complete_settings_only',created_at=now(),verifier_sha256=sha(SCRIPT),protocol_sha256=self.psha,complete_settings=len(complete),expected_settings=8,fold_metric_rows=len(foldrows),seed_oof_rows=len(seedrows),paired_fold_rows=len(pairs),skipped_incomplete_comparisons=skipped,source_prediction_sha256=bindings,output_sha256={rel(p):sha(p) for p in folder.glob('*.csv')},scope='Onlycomplete25fold groups underoneCUDA protocol; meanincrement/sampleSD/ddof1anddescriptivepositives, noindependentfoldsignificance. LocalMPS16 notmixed withCUDA44/56.')
  write(folder/'validation.json',result);return result

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root','--out',dest='root',type=Path,default=DEFAULT_OUT);parser.add_argument('action',choices=['fold','candidate','scan','overview','aggregate']);parser.add_argument('--dataset',choices=['hi_small','li_small']);parser.add_argument('--arm',choices=['base16','base_g32','gmke44','gmkep56']);parser.add_argument('--seed',type=int,choices=range(5));parser.add_argument('--fold',type=int,choices=range(5));parser.add_argument('--candidate',type=int,choices=range(12));parser.add_argument('--inner-fold',type=int,choices=range(3));parser.add_argument('--max-folds',type=int,default=1);parser.add_argument('--force',action='store_true');args=parser.parse_args();v=Validator(args.root)
 if args.action in ('fold','candidate'):
  require(all(x is not None for x in [args.dataset,args.arm,args.seed,args.fold]),'Dataset/arm/seed/fold required')
  if args.action=='fold':result=v.fold(args.dataset,args.arm,args.seed,args.fold,args.force);v.overview()
  else:
   require(args.candidate is not None and args.inner_fold is not None,'Candidateid andinnerfold required');result=dict(passed=False,status='validating_single_candidate',dataset=args.dataset,feature_set=args.arm,model='graphsage_weak',seed=args.seed,fold=args.fold,candidate_id=args.candidate,inner_fold=args.inner_fold,started_at=now(),verifier_sha256=sha(SCRIPT),protocol_sha256=v.psha,verified_artifact_sha256={},physical_models_rescored={});v.candidate(args.dataset,args.arm,args.seed,args.fold,args.candidate,args.inner_fold,result);result.update(passed=True,status='passed_single_candidate_not_outer_fold',completed_at=now(),group_binding=v.group_binding(args.dataset,args.arm),scope='Onlyone retainedinner model; no completeouter selection/result claim.');write(v.dest/'candidate_receipts'/args.dataset/args.arm/f'seed{args.seed}_fold{args.fold}_candidate{args.candidate}_inner{args.inner_fold}.json',result)
 elif args.action=='scan':result=v.scan(args.dataset,args.arm,args.max_folds,args.force)
 elif args.action=='aggregate':result=v.aggregate()
 else:result=v.overview()
 print(json.dumps({k:result[k] for k in ['passed','status','dataset','feature_set','seed','fold','training_done_outer_folds','independently_verified_outer_folds','complete_independently_verified_settings','complete_settings'] if k in result}),flush=True)
if __name__=='__main__':main()
