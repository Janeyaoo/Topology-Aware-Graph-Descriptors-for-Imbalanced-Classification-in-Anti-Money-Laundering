"""One durable outer task for the authorised 20-setting remote completion.

Numerical fitting/search/threshold functions are imported unchanged from the
frozen historical core. The adapter adds explicit input registration, per-job
locking, durable commits and granular task dispatch, not a new search budget.
"""
from __future__ import annotations
import argparse, copy, fcntl, gc, hashlib, json, os, time
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import numpy as np
import pandas as pd
import run_account_gmkep_tuning as core

ROOT=Path(__file__).resolve().parents[2]
SCRIPT=Path(__file__).resolve()

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def read(p): return json.loads(Path(p).read_text())
def rel(p): return str(Path(p).resolve().relative_to(ROOT))
def sync(p):
    with Path(p).open('rb') as f: os.fsync(f.fileno())
def atomic(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    # Flush nested stage payloads before publishing a checkpoint receipt.
    def payloads(v):
        if isinstance(v,dict):
            for name in v.get('artifact_sha256',{}): yield name
            if 'audit_path' in v: yield v['audit_path']
            for k,item in v.items():
                if k!='artifact_sha256': yield from payloads(item)
        elif isinstance(v,list):
            for item in v: yield from payloads(item)
    for name in set(payloads(value)): sync(ROOT/name)
    tmp=p.with_name(p.name+f'.{os.getpid()}.tmp')
    with tmp.open('w') as f:
        json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
    fd=os.open(p.parent,os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)
def binding(mapping):
    for name,digest in mapping.items():
        if sha(ROOT/name)!=digest: raise RuntimeError('SHA mismatch: '+name)
def registered(run):
    reg=read(run/'remote_registration.json');tree=run/'trees';p=read(tree/'protocol.json')
    assert reg['run_dir']==rel(run)
    assert sha(tree/'protocol.json')==reg['protocol_sha256']
    assert p['trainer_sha256']==sha(SCRIPT) and p['inherited_trainer_sha256']==sha(core.__file__)
    assert p['helper_sha256']==sha(core.common.__file__)
    for stem in ['candidates','dataset_inputs','split_manifest','source_manifest']:
        assert sha(tree/(stem+'.json'))==p[stem+'_sha256'],stem
    # Full immutable source closure is checked by preflight/supervisor; each
    # worker also checks all code, task, input array and copied split bindings.
    binding(reg['code_sha256'])
    assert sha(ROOT/reg['task_matrix_path'])==reg['task_matrix_sha256']
    return reg,p
def matrix(tree,ds,arm,p):
    inp=read(tree/'dataset_inputs.json')[ds]
    binding(inp['cache_sha256'])
    base=np.load(ROOT/inp['feature_array_path'],mmap_mode='r')
    y=np.load(ROOT/inp['labels_array_path'],mmap_mode='r')
    counts=np.load(ROOT/inp['sample_counts_path'],mmap_mode='r')
    ids=pd.read_parquet(ROOT/inp['accounts_path']);names=inp['columns']
    columns=p['feature_columns'][arm];extra=None
    if arm in p['deferred_input_arms']:
        kind='gfp' if arm=='base_gfp_account_stream' else 'q10'
        rp=tree/ds/'input_registrations'/(kind+'.json');r=read(rp)
        assert r['passed'] and r['protocol_sha256']==sha(tree/'protocol.json')
        binding(r['artifact_sha256'])
        x=np.load(ROOT/r['array_path'],mmap_mode='r')
        if kind=='gfp':
            columns=names[:16]+r['columns'];X=np.ascontiguousarray(np.column_stack([base[:,:16],x]),dtype=np.float32)
        else:
            pool=np.column_stack([base,x]);pool_names=names[:56]+r['columns']
            X=np.ascontiguousarray(pool[:,[pool_names.index(c) for c in columns]],dtype=np.float32)
        extra={'path':rel(rp),'sha256':sha(rp)}
    else:
        pool=np.column_stack([base,counts]);X=np.ascontiguousarray(pool[:,[names.index(c) for c in columns]],dtype=np.float32)
    assert X.shape==(len(y),len(columns)) and np.isfinite(X).all()
    assert np.array_equal(ids.row_index,np.arange(len(y))) and np.array_equal(ids.label,y)
    return X,y,ids,columns,extra
def execute(run,ds,name,arm,seed,fold):
    reg,p=registered(run);tree=run/'trees';folder=tree/ds
    assert {'dataset':ds,'model':name,'arm':arm} in read(ROOT/reg['task_matrix_path'])
    assert seed in range(5) and fold in range(5)
    stem=f'{name}_{arm}_seed{seed}_fold{fold}'
    lock=(folder/'locks'/(stem+'.lock'));lock.parent.mkdir(exist_ok=True)
    with lock.open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        done_path=folder/'folds'/(stem+'.json')
        if done_path.exists():
            old=read(done_path);assert old['protocol_sha256']==sha(tree/'protocol.json');binding(old['artifact_sha256'])
            print(json.dumps({'status':'already_complete','task':stem}),flush=True);return
        tick=time.monotonic();X,y,ids,columns,extra=matrix(tree,ds,arm,p)
        split=folder/'splits'/f'seed{seed}_fold{fold}.npz'
        sb=read(tree/'split_manifest.json')[rel(split)];assert sha(split)==sb['sha256']
        with np.load(split,allow_pickle=False) as z: arrays={k:z[k] for k in z.files}
        assert {k:core.array_sha(v) for k,v in arrays.items()}==sb['arrays_sha256']
        design=read(tree/'candidates.json')
        core.jsave=atomic;core.common.write_json=atomic
        sr=core.search(tree,folder,name,arm,seed,fold,X,y,arrays,p,design)
        sp=folder/'stage_checkpoints'/(stem+'_stages.json');stages=read(sp) if sp.exists() else {}
        for st in stages.values(): binding(st['artifact_sha256'])
        if 'same_protocol_anchor' not in stages:
            stages['same_protocol_anchor']=core.strategy_fit(tree,folder,name,arm,seed,fold,'same_protocol_anchor',0,sr['anchor_rounds'],X,y,ids,arrays,design[name][0]['params'])
            atomic(sp,stages)
        cid=sr['selected_candidate_id']
        if 'tuned' not in stages:
            if cid==0:
                tuned=copy.deepcopy(stages['same_protocol_anchor']);tuned['strategy']='tuned';tuned['physical_reuse_anchor']=True
                for k in ['threshold_fit_seconds','final_fit_seconds','threshold_model_fit_seconds','final_model_fit_seconds']:tuned[k]=0.
                for r in tuned['metrics']:r['strategy']='tuned'
            else:tuned=core.strategy_fit(tree,folder,name,arm,seed,fold,'tuned',cid,sr['selected_rounds'],X,y,ids,arrays,design[name][cid]['params'])
            stages['tuned']=tuned;atomic(sp,stages)
        srpath=folder/'search'/stem/'search_result.json'
        artifacts={rel(split):sha(split),rel(srpath):sha(srpath),rel(sp):sha(sp)}
        if extra:artifacts[extra['path']]=extra['sha256']
        for st in stages.values():artifacts.update(st['artifact_sha256'])
        done=dict(dataset=ds,model=name,feature_set=arm,seed=seed,fold=fold,n_features=len(columns),feature_columns=columns,
                  protocol_sha256=sha(tree/'protocol.json'),input_registration=extra,search_result_path=rel(srpath),split_path=rel(split),
                  strategies=stages,logical_candidate_evaluations=36,physical_search_fits=sr['physical_search_fits'],logical_strategy_models=4,
                  physical_strategy_models=2 if cid==0 else 4,runtime_seconds=time.monotonic()-tick,artifact_sha256=artifacts,
                  execution_adapter_sha256=sha(SCRIPT),completed_at=core.now())
        atomic(done_path,done)
        print(json.dumps({'status':'complete','dataset':ds,'model':name,'arm':arm,'seed':seed,'fold':fold,'runtime_seconds':done['runtime_seconds']}),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--run',type=Path,required=True);a.add_argument('--dataset',required=True);a.add_argument('--model',required=True)
    a.add_argument('--arm',required=True);a.add_argument('--seed',type=int,required=True);a.add_argument('--fold',type=int,required=True)
    v=a.parse_args();execute(v.run.resolve(),v.dataset,v.model,v.arm,v.seed,v.fold)
