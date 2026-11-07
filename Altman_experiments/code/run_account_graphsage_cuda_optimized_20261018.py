"""Frozen-budget CUDA mean-GraphSAGE across account feature sets.

New independent protocol: full-graph label-blind sampling; seed-only weighted
loss; shared 12 x 3 search; unweighted early AUROC / mean AP selection.  CUDA
optimizes only fit execution. All inference remains CPU two-thread float32 CSR.
"""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse, copy, gc, hashlib, json, platform, shutil, sys, tempfile, time, traceback, resource, fcntl, weakref, signal
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy import sparse
from numba import njit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, balanced_accuracy_score
ROOT = Path('.')
OUT = ROOT
DATASETS=('hi_small','li_small')
ARMS=('base16','base_g32','gmke44','gmkep56')
INPUT_DIM=16
CACHE_FIRST_MEAN=True
CACHE_CSR_BLOCKS=True
PACKED_H2D=True
GATHER_MODE='row_sentinel'
STEP_OBSERVER=None  # Independent QA can inspect each completed optimizer step.
_CSR_BLOCK_CACHE={}
_FIRST_MEAN_CACHE={}
MODEL='graphsage_weak'
def now(): return datetime.now(timezone.utc).isoformat()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def read(path): return json.loads(Path(path).read_text())
def rel(path): return str(Path(path).resolve().relative_to(ROOT))
def jsave(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n'); tmp.replace(path)
def ah(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def atomic_npz(path, **arrays):
    tmp=Path(str(path)+'.tmp')
    with tmp.open('wb') as f: np.savez_compressed(f,**arrays)
    tmp.replace(path)
def atomic_torch(path, value):
    tmp=Path(str(path)+'.tmp'); torch.save(value,tmp); tmp.replace(path)
def memory():
    v=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    out={'peak_rss_bytes':int(v if sys.platform=='darwin' else v*1024)}
    if torch.cuda.is_available():
        out.update(cuda_peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),cuda_peak_reserved_bytes=int(torch.cuda.max_memory_reserved()),cuda_allocated_bytes=int(torch.cuda.memory_allocated()),cuda_reserved_bytes=int(torch.cuda.memory_reserved()))
    return out
def configure():
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    if hasattr(torch,'set_float32_matmul_precision'): torch.set_float32_matmul_precision('highest')
def metrics(y, probability, threshold):
    predicted=(probability>=threshold).astype(np.int8)
    tn,fp,fn,tp=confusion_matrix(y,predicted,labels=[0,1]).ravel()
    return predicted,dict(precision=float(precision_score(y,predicted,zero_division=0)),recall=float(recall_score(y,predicted,zero_division=0)),f1=float(f1_score(y,predicted,zero_division=0)),auc=float(roc_auc_score(y,probability)),ap=float(average_precision_score(y,probability)),accuracy=float(accuracy_score(y,predicted)),balanced_accuracy=float(balanced_accuracy_score(y,predicted)),specificity=float(tn/(tn+fp)),tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),n_test=len(y),n_positive=int(y.sum()))
def best_f1_threshold(y,probability):
    order=np.argsort(-probability,kind='mergesort'); scores=probability[order]
    positives=np.cumsum(y[order],dtype=np.int64)
    ends=np.r_[np.flatnonzero(scores[:-1]!=scores[1:]),len(scores)-1]
    thresholds=scores[ends][::-1]; values=(2*positives[ends])[::-1]/(ends+1+int(y.sum()))[::-1]
    position=int(np.argmax(values)); return float(thresholds[position]),float(values[position])

@njit(cache=True, nogil=True)
def sample_neighbors(indptr, indices, targets, fanout, seed):
    """Fixed-width without-replacement sample; padding never contributes."""
    np.random.seed(seed)
    result=np.full((len(targets),fanout),-1,dtype=np.int64)
    for i in range(len(targets)):
        left=indptr[targets[i]]; degree=indptr[targets[i]+1]-left
        if degree<=fanout:
            for j in range(degree): result[i,j]=indices[left+j]
        else:
            for j in range(fanout):
                while True:
                    picked=np.random.randint(0,degree)
                    duplicate=False
                    for k in range(j):
                        if result[i,k]==indices[left+picked]: duplicate=True; break
                    if not duplicate: break
                result[i,j]=indices[left+picked]
    return result

def block(graphs, targets, fanout, rng):
    sampled=[sample_neighbors(g.indptr,g.indices,targets,fanout,int(rng.integers(0,2**31-1))) for g in graphs]
    nodes=np.unique(np.concatenate([targets,*[a[a>=0] for a in sampled]]))
    maps=[]
    for a in sampled:
        valid=a>=0
        positions=np.searchsorted(nodes,np.maximum(a,0))
        positions[~valid]=0
        maps.append((positions,valid))
    return nodes,np.searchsorted(nodes,targets),maps

def neighbor_gather(h,position,valid):
    if GATHER_MODE=='reference':return h[position]
    if GATHER_MODE=='row_sentinel':
        if not h.requires_grad:return h[position]
        # Only post-ReLU/dropout h receives gradients; forward padding remains zero.
        n=h.shape[0];zeros=h.new_zeros((position.shape[0],h.shape[1]))
        padded=torch.cat((h,zeros),dim=0)
        sentinel=n+torch.arange(position.shape[0],dtype=torch.long,device=position.device).unsqueeze(1)
        mapped=torch.where(valid,position,sentinel)
        return padded[mapped]
    raise ValueError(GATHER_MODE)

class AccountNetwork(nn.Module):
    """Weak=SAGEConv mean, directed=independent relation SAGE + linear fusion."""
    def __init__(self,name,params):
        super().__init__(); self.name=name; self.dropout=float(params['dropout'])
        hidden=int(params['hidden_dim'])
        if name=='mlp':
            self.layers=nn.ModuleList([nn.Linear(INPUT_DIM,hidden),nn.Linear(hidden,1)])
        else:
            relations=1 if name=='graphsage_weak' else 2
            self.root=nn.ModuleList([nn.ModuleList([nn.Linear(d,o) for _ in range(relations)]) for d,o in ((INPUT_DIM,hidden),(hidden,1))])
            self.neighbor=nn.ModuleList([nn.ModuleList([nn.Linear(d,o,bias=False) for _ in range(relations)]) for d,o in ((INPUT_DIM,hidden),(hidden,1))])
            self.fusion=nn.ModuleList([nn.Identity() if relations==1 else nn.Linear(relations*o,o) for o in (hidden,1)])
    def conv(self,h,root_index,maps,layer):
        values=[]
        for r,(position,valid) in enumerate(maps):
            gathered=neighbor_gather(h,position,valid)*valid.unsqueeze(-1)
            mean=gathered.sum(1)/valid.sum(1).clamp_min(1).unsqueeze(-1)
            values.append(self.root[layer][r](h[root_index])+self.neighbor[layer][r](mean))
        return self.fusion[layer](torch.cat(values,-1))
    def forward_mlp(self,x):
        h=torch.relu(self.layers[0](x)); h=nn.functional.dropout(h,p=self.dropout,training=self.training)
        return self.layers[1](h).squeeze(-1)
    def forward_sampled(self,x,inner,outer):
        h=torch.relu(self.conv(x,*inner,0)); h=nn.functional.dropout(h,p=self.dropout,training=self.training)
        return self.conv(h,*outer,1).squeeze(-1)

def tensor_block(root,maps,device):
    return (torch.as_tensor(root,dtype=torch.long,device=device),
            [(torch.as_tensor(p,dtype=torch.long,device=device),torch.as_tensor(v,dtype=torch.bool,device=device)) for p,v in maps])

def normalize(graph):
    g=graph.tocsr().astype(np.float32)
    degree=np.diff(g.indptr)
    g.data*=np.repeat(1/np.maximum(degree,1),degree).astype(np.float32)
    return g

def clear_inference_caches():
    _FIRST_MEAN_CACHE.clear(); _CSR_BLOCK_CACHE.clear()

def graph_blocks(graph,chunk):
    if not CACHE_CSR_BLOCKS:
        return [graph[start:min(start+chunk,graph.shape[0])] for start in range(0,graph.shape[0],chunk)]
    key=(id(graph),chunk)
    entry=_CSR_BLOCK_CACHE.get(key)
    if entry is None or entry[0] is not graph:
        entry=(graph,[graph[start:min(start+chunk,graph.shape[0])] for start in range(0,graph.shape[0],chunk)])
        _CSR_BLOCK_CACHE[key]=entry
    return entry[1]

def first_neighbor_mean(X,graph,chunk):
    key=(id(X),id(graph),chunk);entry=_FIRST_MEAN_CACHE.get(key)
    if entry is None or entry[0]() is not X:
        mean=np.empty_like(X,dtype=np.float32);blocks=graph_blocks(graph,chunk)
        for index,start in enumerate(range(0,len(X),chunk)):
            stop=min(start+chunk,len(X))
            # Exactly the original CSR slice, dot shape, dtype and summation order.
            block=blocks[index]
            mean[start:stop]=np.ascontiguousarray(block.dot(X))
        def release(reference,key=key): _FIRST_MEAN_CACHE.pop(key,None)
        entry=(weakref.ref(X,release),mean);_FIRST_MEAN_CACHE[key]=entry
    return entry[1]

def predict(model,X,graphs,rows,chunk=16384):
    """Unchanged CPU equations; immutable graph blocks / fit-only A@X cached."""
    cpu=copy.deepcopy(model).to('cpu').eval()
    with torch.no_grad():
        if model.name=='mlp':
            result=np.empty(len(rows),dtype=np.float64)
            for start in range(0,len(rows),chunk):
                index=rows[start:start+chunk]
                logits=cpu.forward_mlp(torch.from_numpy(np.ascontiguousarray(X[index])))
                result[start:start+len(index)]=torch.sigmoid(logits).double().numpy()
            return result
        h=X
        for layer in range(2):
            width=int(cpu.root[layer][0].out_features)
            out=np.empty((len(X),width),dtype=np.float32)
            for block_index,start in enumerate(range(0,len(X),chunk)):
                stop=min(start+chunk,len(X)); root=torch.from_numpy(np.ascontiguousarray(h[start:stop]))
                values=[]
                for r,g in enumerate(graphs):
                    if layer==0 and CACHE_FIRST_MEAN:
                        mean_array=first_neighbor_mean(X,g,chunk)[start:stop]
                    else:
                        block=graph_blocks(g,chunk)[block_index] if CACHE_CSR_BLOCKS else g[start:stop]
                        mean_array=np.ascontiguousarray(block.dot(h))
                    mean=torch.from_numpy(mean_array)
                    values.append(cpu.root[layer][r](root)+cpu.neighbor[layer][r](mean))
                values=cpu.fusion[layer](torch.cat(values,-1))
                if layer==0: values=torch.relu(values)
                out[start:stop]=values.numpy()
            h=out
        score=torch.sigmoid(torch.from_numpy(h[rows,0])).double().numpy()
        assert np.isfinite(score).all() and ((score>=0)&(score<=1)).all()
        return np.ascontiguousarray(score,dtype=np.float64)

def packed_transfer(second,roots,r1,m1,r0,m0,device):
    """Two pinned host copies; same integer/bool values and CUDA compute stream."""
    longs=[np.asarray(second,np.int64),np.asarray(roots,np.int64),np.asarray(r1,np.int64),np.asarray(r0,np.int64)]
    bools=[]
    for maps in (m1,m0):
        for position,valid in maps:
            longs.append(np.asarray(position,np.int64));bools.append(np.asarray(valid,np.bool_))
    shapes=[a.shape for a in longs];sizes=[a.size for a in longs]
    host_long=torch.from_numpy(np.concatenate([a.ravel() for a in longs])).pin_memory()
    host_bool=torch.from_numpy(np.concatenate([a.ravel() for a in bools])).pin_memory()
    packed_long=host_long.to(device,non_blocking=True);packed_bool=host_bool.to(device,non_blocking=True)
    pieces=list(packed_long.split(sizes));pieces=[a.view(shape) for a,shape in zip(pieces,shapes)]
    masks=list(packed_bool.split([a.size for a in bools]));masks=[a.view(b.shape) for a,b in zip(masks,bools)]
    offset=4;mask_offset=0;relations=[]
    for maps in (m1,m0):
        relation=[]
        for _ in maps:
            relation.append((pieces[offset],masks[mask_offset]));offset+=1;mask_offset+=1
        relations.append(relation)
    # Nonblocking H2D uses PyTorch pinned caching allocator lifetime tracking.
    return pieces[0],pieces[1],(pieces[2],relations[0]),(pieces[3],relations[1])

def state_digest(model):
    h=hashlib.sha256()
    for key,value in model.state_dict().items():
        h.update(key.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def sampled_batch(graphs,roots,rng):
    start=time.perf_counter()
    first,r0,m0=block(graphs,roots,15,rng)
    second,r1,m1=block(graphs,first,10,rng)
    return roots,second,r1,m1,r0,m0,time.perf_counter()-start

def fit(name,raw,y,graphs,fit_rows,early_rows,params,seed,device,path,epochs=None,prefetch=True,trace=False,batch_size=2048):
    """Same data-dependent protocol; async one-batch CPU sampling, resident X/y."""
    started=time.perf_counter(); torch.manual_seed(seed)
    is_cuda=device.startswith('cuda')
    if is_cuda: torch.cuda.manual_seed_all(seed); torch.cuda.reset_peak_memory_stats()
    scaler=StandardScaler().fit(raw[fit_rows])
    X=np.ascontiguousarray(scaler.transform(raw),dtype=np.float32)
    assert np.isfinite(X).all()
    model=AccountNetwork(name,params).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=float(params['learning_rate']),weight_decay=1e-4)
    n=len(fit_rows); npos=int(y[fit_rows].sum()); assert 0<npos<n
    weights=np.array([n/(2*(n-npos)),n/(2*npos)],dtype=np.float32)
    # Cache node attributes and labels once on GPU; no repeated host attribute copies.
    gpuX=torch.from_numpy(X).to(device)
    gpuY=torch.from_numpy(np.ascontiguousarray(y,dtype=np.float32)).to(device)
    gpuW=torch.from_numpy(weights).to(device)
    rng=np.random.default_rng(seed+1000000)
    best_auc=-np.inf; best_epoch=0; best_state=None; history=[]
    totals={'sampling_cpu_seconds':0.,'sampling_wait_seconds':0.,'cuda_transfer_seconds':0.,'cuda_forward_backward_seconds':0.,'early_evaluation_seconds':0.,'checkpoint_io_seconds':0.}
    limit=int(epochs or 200)
    qa_trace={'sampling_sha256':[],'batch_loss':[]} if trace else None
    with ThreadPoolExecutor(max_workers=1,thread_name_prefix='label_blind_sampler') as sampler:
        for epoch in range(1,limit+1):
            epoch_start=time.perf_counter(); model.train(); total_loss=torch.zeros((),device=device)
            ordered=rng.permutation(fit_rows); cursor=0
            future=sampler.submit(sampled_batch,graphs,ordered[:batch_size],rng); cursor=batch_size
            events=[]; ep_sampling=ep_wait=0.
            while future is not None:
                tick=time.perf_counter(); roots,second,r1,m1,r0,m0,sampling_seconds=future.result()
                ep_wait+=time.perf_counter()-tick; ep_sampling+=sampling_seconds
                # The only RNG consumer is this single sampler; preserve exact batch order.
                has_next=cursor<n; next_roots=ordered[cursor:cursor+batch_size]; cursor+=batch_size
                future=sampler.submit(sampled_batch,graphs,next_roots,rng) if has_next and prefetch else None
                if trace:
                    digest=hashlib.sha256()
                    for value in (roots,second,r1,r0): digest.update(np.ascontiguousarray(value).tobytes())
                    for maps in (m1,m0):
                        for position,valid in maps:
                            digest.update(np.ascontiguousarray(position).tobytes()); digest.update(np.ascontiguousarray(valid).tobytes())
                    qa_trace['sampling_sha256'].append(digest.hexdigest())
                if is_cuda:
                    begin=torch.cuda.Event(enable_timing=True); transferred=torch.cuda.Event(enable_timing=True); trained=torch.cuda.Event(enable_timing=True)
                    begin.record()
                if is_cuda and PACKED_H2D:
                    second_t,root_t,inner,outer=packed_transfer(second,roots,r1,m1,r0,m0,device)
                else:
                    second_t=torch.as_tensor(second,dtype=torch.long,device=device)
                    root_t=torch.as_tensor(roots,dtype=torch.long,device=device)
                    inner=tensor_block(r1,m1,device); outer=tensor_block(r0,m0,device)
                if is_cuda: transferred.record()
                optimizer.zero_grad(set_to_none=True)
                logits=model.forward_sampled(gpuX[second_t],inner,outer)
                labels=gpuY[root_t]; w=gpuW[labels.long()]
                loss=(nn.functional.binary_cross_entropy_with_logits(logits,labels,reduction='none')*w).mean()
                loss.backward(); optimizer.step(); total_loss+=loss.detach()*len(roots)
                if STEP_OBSERVER is not None: STEP_OBSERVER(model,optimizer,epoch,roots)
                if is_cuda: trained.record(); events.append((begin,transferred,trained))
                if trace: qa_trace['batch_loss'].append(float(loss.detach().cpu()))
                if has_next and not prefetch: future=sampler.submit(sampled_batch,graphs,next_roots,rng)
            if is_cuda:
                torch.cuda.synchronize()
                transfer=sum(a.elapsed_time(b) for a,b,c in events)/1000
                gpu_train=sum(b.elapsed_time(c) for a,b,c in events)/1000
            else: transfer=0.; gpu_train=0.
            record={'epoch':epoch,'weighted_seed_loss':float(total_loss.cpu())/n,'seconds':time.perf_counter()-started,
                    'epoch_fit_seconds':time.perf_counter()-epoch_start,'sampling_cpu_seconds':ep_sampling,'sampling_wait_seconds':ep_wait,
                    'cuda_transfer_seconds':transfer,'cuda_forward_backward_seconds':gpu_train}
            totals['sampling_cpu_seconds']+=ep_sampling; totals['sampling_wait_seconds']+=ep_wait
            totals['cuda_transfer_seconds']+=transfer; totals['cuda_forward_backward_seconds']+=gpu_train
            if early_rows is not None:
                tick=time.perf_counter(); score=predict(model,X,graphs,early_rows)
                auc=float(roc_auc_score(y[early_rows],score)); record['early_auroc']=auc
                record['early_evaluation_seconds']=time.perf_counter()-tick
                totals['early_evaluation_seconds']+=record['early_evaluation_seconds']
                if auc>best_auc:
                    best_auc=auc; best_epoch=epoch
                    best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                if epoch-best_epoch>=30: history.append(record); break
            else: best_epoch=epoch
            history.append(record)
            if epoch%10==0 or epoch==1:
                print(json.dumps({'event':'fit_epoch','path':rel(path),'model':name,'epoch':epoch,**record,**memory()}),flush=True)
    if early_rows is not None: model.load_state_dict(best_state)
    if is_cuda: torch.cuda.synchronize()
    metadata={'model':name,'feature_set':ARM,'input_dim':INPUT_DIM,'params':params,'device':device,'random_state':seed,
       'scaler_fit_rows_sha256':ah(fit_rows),'scaler_mean':scaler.mean_.tolist(),'scaler_scale':scaler.scale_.tolist(),
       'scaler_var':scaler.var_.tolist(),'scaler_n_samples_seen':int(scaler.n_samples_seen_),
       'fit_rows_sha256':ah(fit_rows),'early_rows_sha256':None if early_rows is None else ah(early_rows),
       'fit_n':n,'fit_positive':npos,'class_weights':weights.tolist(),'seed_only_loss':True,
       'selected_epochs':best_epoch,'epochs_run':len(history),'best_early_auroc':None if early_rows is None else best_auc,
       'history':history,'model_digest':state_digest(model),'seconds':time.perf_counter()-started,'phase_timing':totals,
       'qa_trace':qa_trace,'sampling_prefetch':prefetch,'seed_batch_size':batch_size,
       'execution_optimizations':{'cache_first_mean':CACHE_FIRST_MEAN,'cache_csr_blocks':CACHE_CSR_BLOCKS,'packed_h2d':PACKED_H2D,'gather_mode':GATHER_MODE},
       'timing_note':'CPU sampling overlaps CUDA fit; phase times are not additive. CUDA events synchronized once per epoch.',
       **memory()}
    tick=time.perf_counter()
    atomic_torch(path,{'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'metadata':metadata})
    # Keep checkpoint and JSON metadata identical, including measured first-save cost.
    totals['checkpoint_io_seconds']=time.perf_counter()-tick
    metadata['seconds']=time.perf_counter()-started
    atomic_torch(path,{'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'metadata':metadata})
    jsave(path.with_suffix('.json'),metadata)
    del gpuX,gpuY,gpuW
    return model,X,metadata

def search(ds,name,seed,fold,raw,y,graphs,arrays,p):
    folder=OUT/ds/ARM; stem=f'{name}_seed{seed}_fold{fold}'; directory=folder/'search'/stem
    directory.mkdir(exist_ok=True); result_path=directory/'search_result.json'
    if result_path.exists():
        result=read(result_path)
        for path,digest in result['artifact_sha256'].items(): assert sha(ROOT/path)==digest
        return result
    candidates=read(ROOT/'candidates.json')['candidates']; rows=[]
    for c in candidates:
        for j in range(3):
            recpath=directory/f"candidate{c['candidate_id']}_inner{j}_record.json"
            if recpath.exists():
                record=read(recpath)
                assert record['protocol_sha256']==sha(ROOT/'protocol.json')
                for path,digest in record['artifact_sha256'].items(): assert sha(ROOT/path)==digest
                rows.append(record); continue
            modelpath=directory/f"candidate{c['candidate_id']}_inner{j}.pt"
            fit_rows,early,score_rows=(arrays[f'{key}_{j}'] for key in ('early_fit','early_validation','inner_score'))
            model,X,meta=fit(name,raw,y,graphs,fit_rows,early,c['params'],40000+1000*seed+10*fold+j,p['device'],modelpath)
            tick=time.monotonic(); score=predict(model,X,graphs,score_rows)
            scorepath=modelpath.with_suffix('.npz'); atomic_npz(scorepath,row_index=score_rows,y_true=y[score_rows],y_score=score)
            record={'dataset':ds,'model':name,'feature_set':ARM,'seed':seed,'fold':fold,
                'candidate_id':c['candidate_id'],'inner_fold':j,'params':c['params'],
                'ap':float(average_precision_score(y[score_rows],score)),'auc':float(roc_auc_score(y[score_rows],score)),
                'selected_epochs':meta['selected_epochs'],'fit_seconds':meta['seconds'],'predict_seconds':time.monotonic()-tick,
                'fit_rows_sha256':ah(fit_rows),'early_rows_sha256':ah(early),'score_rows_sha256':ah(score_rows),'raw_score_sha256':ah(score),
                'model_path':rel(modelpath),'score_path':rel(scorepath),'model_digest':meta['model_digest'],
                'protocol_sha256':sha(ROOT/'protocol.json'),'artifact_sha256':{rel(x):sha(x) for x in (modelpath,modelpath.with_suffix('.json'),scorepath)}}
            jsave(recpath,record); rows.append(record)
            print(json.dumps({'event':'candidate_complete',**{k:record[k] for k in ('dataset','model','seed','fold','candidate_id','inner_fold','ap','auc','selected_epochs','fit_seconds')}}),flush=True)
            del model,X,score; gc.collect()
            if p['device'].startswith('cuda'): torch.cuda.empty_cache()
    assert len(rows)==36
    summary=[]
    for c in candidates:
        records=[r for r in rows if r['candidate_id']==c['candidate_id']]
        summary.append({'candidate_id':c['candidate_id'],'mean_ap':float(np.mean([r['ap'] for r in records])),
                        'std_ap':float(np.std([r['ap'] for r in records],ddof=1)),
                        'epochs':int(np.median([r['selected_epochs'] for r in records]))})
    selected=min(summary,key=lambda r:(-r['mean_ap'],r['candidate_id']))
    result={'dataset':ds,'model':name,'seed':seed,'fold':fold,'candidate_records':rows,'candidate_summary':summary,
            'selected_candidate_id':selected['candidate_id'],'selected_mean_ap':selected['mean_ap'],
            'selected_epochs':selected['epochs'],'anchor_epochs':summary[0]['epochs'],
            'artifact_sha256':{path:digest for r in rows for path,digest in r['artifact_sha256'].items()},
            'protocol_sha256':sha(ROOT/'protocol.json')}
    jsave(result_path,result); return result

def stages(ds,name,seed,fold,strategy,cid,epochs,params,raw,y,graphs,arrays,identity,p):
    folder=OUT/ds/ARM; stem=f'{name}_seed{seed}_fold{fold}_{strategy}'
    D,T,train,test=(arrays[k] for k in ('development','threshold_validation','outer_train','outer_test'))
    modelpath=folder/'calibration_models'/f'{stem}.pt'
    model,X,calmeta=fit(name,raw,y,graphs,D,None,params,100*seed+fold,p['device'],modelpath,epochs)
    cal_score=predict(model,X,graphs,T); threshold,calf1=best_f1_threshold(y[T],cal_score)
    calpath=folder/'calibration'/f'{stem}.parquet'
    table=identity.iloc[T][['row_index','account_id']].copy(); table['y_true']=y[T]; table['y_score']=cal_score
    table.to_parquet(calpath,index=False,compression='zstd'); del model,X,table; gc.collect()
    finalpath=folder/'models'/f'{stem}.pt'
    model,X,finalmeta=fit(name,raw,y,graphs,train,None,params,100*seed+fold,p['device'],finalpath,epochs)
    score=predict(model,X,graphs,test); predpath=folder/'predictions'/f'{stem}.parquet'
    table=identity.iloc[test][['row_index','account_id']].copy(); table['y_true']=y[test]; table['y_score']=score
    records=[]
    for policy,cutoff in (('fixed_0.5',.5),('inner_max_f1',threshold)):
        pred,values=metrics(y[test],score,cutoff)
        table['y_pred_fixed' if policy=='fixed_0.5' else 'y_pred_inner']=pred
        records.append({'dataset':ds,'model':name,'feature_set':ARM,'seed':seed,'fold':fold,'strategy':strategy,
                        'threshold_policy':policy,'threshold':cutoff,'selected_candidate_id':cid,'selected_epochs':epochs,**values})
    table.to_parquet(predpath,index=False,compression='zstd')
    files=[modelpath,modelpath.with_suffix('.json'),calpath,finalpath,finalpath.with_suffix('.json'),predpath]
    result={'strategy':strategy,'candidate_id':cid,'selected_epochs':epochs,'params':params,'threshold':threshold,
            'threshold_validation_f1':calf1,'threshold_validation_ap':float(average_precision_score(y[T],cal_score)),
            'model_random_state':100*seed+fold,'fit_rows_sha256':ah(train),'calibration_fit_rows_sha256':ah(D),
            'threshold_rows_sha256':ah(T),'outer_test_rows_sha256':ah(test),
            'calibration_model_path':rel(modelpath),'model_path':rel(finalpath),'calibration_path':rel(calpath),'prediction_path':rel(predpath),
            'calibration_metadata':calmeta,'final_metadata':finalmeta,'raw_score_sha256':ah(score),
            'metrics':records,'artifact_sha256':{rel(f):sha(f) for f in files},'protocol_sha256':sha(ROOT/'protocol.json')}
    del model,X,score,cal_score,table; gc.collect()
    if p['device'].startswith('cuda'): torch.cuda.empty_cache()
    return result


def arm_columns(names,arm):
    values=names['feature_sets'][arm]
    return [names['columns'][v] if isinstance(v,int) else v for v in values]

def validate_splits(arrays,n):
    for key,value in arrays.items():
        assert value.ndim==1 and np.issubdtype(value.dtype,np.integer)
        assert len(value)==len(np.unique(value)) and (value>=0).all() and (value<n).all(),key
    def partition(parent,a,b):
        np.testing.assert_array_equal(np.sort(np.r_[a,b]),np.sort(parent))
    partition(np.arange(n),arrays['outer_train'],arrays['outer_test'])
    partition(arrays['outer_train'],arrays['development'],arrays['threshold_validation'])
    for j in range(3):
        partition(arrays['development'],arrays[f'inner_train_{j}'],arrays[f'inner_score_{j}'])
        partition(arrays[f'inner_train_{j}'],arrays[f'early_fit_{j}'],arrays[f'early_validation_{j}'])

def prepare():
    assert not (ROOT/'protocol.json').exists(),'Refuse to replace a registered protocol'
    candidates=read(ROOT/'candidates.json')['candidates']
    expected=[{'candidate_id':i,'params':{'hidden_dim':h,'learning_rate':lr,'dropout':drop}} for i,(h,lr,drop) in enumerate((h,lr,d) for h in (64,128,256) for lr in (.001,.003) for d in (0.,.2))]
    assert candidates==expected,'Candidates must preserve all 12 original choices and order'
    source={rel(ROOT/'candidates.json'):sha(ROOT/'candidates.json')}; datasets={}
    for ds in DATASETS:
        folder=ROOT/'inputs'/ds; names=read(folder/'names.json')
        assert len(names['columns'])==len(set(names['columns']))==56
        X=np.load(folder/'features.npy',mmap_mode='r'); y=np.load(folder/'labels.npy')
        identity=pd.read_parquet(folder/'accounts.parquet')
        assert X.shape==(len(y),56) and X.dtype==np.float64 and np.isfinite(X).all()
        assert y.ndim==1 and set(np.unique(y))=={0,1}
        np.testing.assert_array_equal(identity.row_index.to_numpy(),np.arange(len(y)))
        assert identity.account_id.is_unique
        graph=sparse.load_npz(folder/'weak_csr.npz').tocsr()
        assert graph.shape==(len(y),len(y)) and (graph!=graph.T).nnz==0 and graph.diagonal().sum()==0
        assert (graph.data==1).all() and graph.has_sorted_indices and graph.has_canonical_format
        base=names['base16']; assert len(base)==16
        for arm,width in zip(ARMS,(16,32,44,56)):
            columns=arm_columns(names,arm)
            assert len(columns)==len(set(columns))==width and set(base).issubset(columns)
            if arm=='base16': assert columns==base
            for sub in ('search','folds','models','calibration_models','calibration','predictions'):
                (OUT/ds/arm/sub).mkdir(parents=True,exist_ok=True)
        baseids=[names['columns'].index(c) for c in base]
        assert (X[:,baseids]>=0).all()
        splitfiles=[]
        for seed in range(5):
            outer=[]
            for fold in range(5):
                f=folder/'splits'/f'seed{seed}_fold{fold}.npz'
                with np.load(f) as z: arrays={k:z[k] for k in z.files}
                validate_splits(arrays,len(y)); outer.append(arrays['outer_test']); splitfiles.append(f)
            np.testing.assert_array_equal(np.sort(np.concatenate(outer)),np.arange(len(y)))
        files=[folder/x for x in ('features.npy','names.json','labels.npy','accounts.parquet','weak_csr.npz')]+splitfiles
        source.update({rel(x):sha(x) for x in files})
        datasets[ds]={'n_samples':len(y),'n_positive':int(y.sum()),'weak_edges':int(graph.nnz//2),'feature_sets':{arm:arm_columns(names,arm) for arm in ARMS},'base16':base,'labels_array_sha256':ah(y),'account_ids_hash_sha256':ah(pd.util.hash_pandas_object(identity.account_id,index=False).to_numpy())}
    protocol={'registered_at':now(),'runner_sha256':sha(__file__),'device':'cuda','model':MODEL,'datasets':datasets,
       'feature_sets':list(ARMS),'preprocessing':'BASE16: log1p original float64 then float32; other selected columns: float32 raw; all selected fields fit-only StandardScaler',
       'architecture':'two mean GraphSAGE; root affine with bias plus neighbor affine without bias; first ReLU/dropout; hidden 64/128/256',
       'graph':'same unweighted undirected projection, all accounts including isolates; no edge features; without-replacement fanouts [15,10]',
       'selection':'12 candidates x 3 exact original inner splits; early unweighted AUROC strict > patience30/max200; mean inner unweighted AP, lowest candidate ID tie; median3 best epochs',
       'threshold':'D fit then T maximum positive F1, lowest threshold on exact tie; final fit outer_train with same seed/rounds',
       'weights':'n/(2*n_class) actual fit-only labels; seed-only BCE; Adam lr candidate, weight_decay1e-4; batch2048',
       'numeric':'float32 CUDA, no autocast, TF32 off, deterministic algorithms; CPU2threads full-neighbor CSR inference and float64 score copy',
       'execution_optimizations':['resident node feature/label tensors','single-worker deterministic next-batch sampling overlap','one CUDA synchronization per epoch, phase events','exact fixed-chunk CPU first-neighbor-mean and CSR block cache','packed pinned asynchronous H2D on original compute stream','independent per-target zero padding sentinels only for differentiable post-ReLU neighbor gather'],
       'budget':{'arms_per_dataset':4,'outer_folds_per_setting':25,'inner_candidate_scores_per_outer':36,'stages_per_outer':4},
       'execution_settings':{'cache_first_mean':CACHE_FIRST_MEAN,'cache_csr_blocks':CACHE_CSR_BLOCKS,'packed_h2d':PACKED_H2D,'gather_mode':GATHER_MODE},
       'environment':{'torch':torch.__version__,'numpy':np.__version__,'platform':platform.platform(),'cuda_runtime':torch.version.cuda},
       'source_binding_sha256':None}
    jsave(ROOT/'source_binding.json',source); protocol['source_binding_sha256']=sha(ROOT/'source_binding.json')
    jsave(ROOT/'protocol.json',protocol); shutil.copy2(__file__,ROOT/'runner_snapshot.py')
    print(json.dumps({'event':'prepared','protocol_sha256':sha(ROOT/'protocol.json'),'datasets':{k:v['n_samples'] for k,v in datasets.items()}}),flush=True)

def registered(ds=None):
    p=read(ROOT/'protocol.json'); assert sha(__file__)==p['runner_sha256'],'Registered code changed'
    assert sha(ROOT/'source_binding.json')==p['source_binding_sha256']
    assert p['execution_settings']=={'cache_first_mean':CACHE_FIRST_MEAN,'cache_csr_blocks':CACHE_CSR_BLOCKS,'packed_h2d':PACKED_H2D,'gather_mode':GATHER_MODE},'Execution switches differ from registration'
    for path,digest in read(ROOT/'source_binding.json').items():
        if ds is None or not path.startswith('inputs/') or path.startswith(f'inputs/{ds}/'):
            assert sha(ROOT/path)==digest,f'Source changed: {path}'
    return p

def load_data(ds,arm,seed,fold):
    global INPUT_DIM,ARM
    ARM=arm; folder=ROOT/'inputs'/ds; names=read(folder/'names.json'); columns=arm_columns(names,arm)
    original=np.load(folder/'features.npy',mmap_mode='r'); indices=[names['columns'].index(c) for c in columns]
    raw=np.ascontiguousarray(original[:,indices],dtype=np.float64)
    for j,c in enumerate(columns):
        if c in names['base16']:
            assert (raw[:,j]>=0).all(); raw[:,j]=np.log1p(raw[:,j])
    raw=np.ascontiguousarray(raw,dtype=np.float32); INPUT_DIM=raw.shape[1]
    assert np.isfinite(raw).all()
    y=np.load(folder/'labels.npy'); identity=pd.read_parquet(folder/'accounts.parquet')
    graphs=[normalize(sparse.load_npz(folder/'weak_csr.npz').tocsr())]
    with np.load(folder/'splits'/f'seed{seed}_fold{fold}.npz') as z: arrays={k:z[k] for k in z.files}
    validate_splits(arrays,len(y))
    return raw,y,identity,graphs,arrays

def run_fold(ds,arm,seed,fold):
    assert torch.cuda.is_available(),'Formal fit requires CUDA; no silent CPU fallback'
    p=registered(ds); probe=read(ROOT/'probe.json')
    assert probe['passed'] and probe['runner_sha256']==sha(__file__) and probe['device']=='cuda'
    folder=OUT/ds/arm; stem=f'{MODEL}_seed{seed}_fold{fold}'
    lock=(folder/'folds'/f'{stem}.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=folder/'folds'/f'{stem}.json'; jobpath=folder/'folds'/f'{stem}_job.json'
    def check_saved(value):
        assert value['protocol_sha256']==sha(ROOT/'protocol.json')
        for strategy in value['strategies']:
            for item,digest in strategy['artifact_sha256'].items(): assert sha(ROOT/item)==digest
        for item,digest in value['search']['artifact_sha256'].items(): assert sha(ROOT/item)==digest
    if path.exists(): check_saved(read(path)); print(json.dumps({'event':'already_complete','path':rel(path)})); return
    manifest={'dataset':ds,'feature_set':arm,'model':MODEL,'seed':seed,'fold':fold,'status':'running','pid':os.getpid(),'started_at':now(),'protocol_sha256':sha(ROOT/'protocol.json')}
    jsave(jobpath,manifest)
    try:
        raw,y,identity,graphs,arrays=load_data(ds,arm,seed,fold)
        candidates=read(ROOT/'candidates.json')['candidates']
        result=search(ds,MODEL,seed,fold,raw,y,graphs,arrays,p)
        strategies=[]
        for strategy,cid,epochs in (('same_protocol_anchor',0,result['anchor_epochs']),('tuned',result['selected_candidate_id'],result['selected_epochs'])):
            partial=folder/'folds'/f'{stem}_{strategy}_partial.json'
            if partial.exists():
                value=read(partial); assert value['protocol_sha256']==sha(ROOT/'protocol.json')
                for item,digest in value['artifact_sha256'].items(): assert sha(ROOT/item)==digest
            else:
                value=stages(ds,MODEL,seed,fold,strategy,cid,epochs,candidates[cid]['params'],raw,y,graphs,arrays,identity,p)
                jsave(partial,value)
            strategies.append(value)
        saved={'dataset':ds,'model':MODEL,'feature_set':arm,'seed':seed,'fold':fold,'search':result,'strategies':strategies,'logical_candidate_evaluations':36,'logical_strategy_models':4,'artifact_sha256':{**result['artifact_sha256'],**{item:digest for strategy in strategies for item,digest in strategy['artifact_sha256'].items()}},'protocol_sha256':sha(ROOT/'protocol.json'),'completed_at':now()}
        jsave(path,saved); check_saved(saved)
        manifest.update(status='complete',pid=None,finished_at=now(),result_path=rel(path),result_sha256=sha(path),**memory()); jsave(jobpath,manifest)
        print(json.dumps({'event':'outer_complete','dataset':ds,'feature_set':arm,'model':MODEL,'seed':seed,'fold':fold,'path':rel(path)}),flush=True)
    except BaseException as exc:
        manifest.update(status='failed',pid=None,error=repr(exc),traceback=traceback.format_exc(),updated_at=now(),**memory()); jsave(jobpath,manifest); raise


def profile(ds,arm,seed,fold,epochs,tag):
    assert torch.cuda.is_available()
    assert epochs==5,'Resource QA is frozen to five epochs; no test evaluation or candidate selection'
    assert tag and Path(tag).name==tag and tag not in ('.','..')
    registered(ds)
    directory=ROOT/'qa'/'profiles'/tag; directory.mkdir(parents=True,exist_ok=False)
    raw,y,identity,graphs,arrays=load_data(ds,arm,seed,fold)
    params={'hidden_dim':256,'learning_rate':.001,'dropout':.2}
    start=time.perf_counter()
    model,X,metadata=fit(MODEL,raw,y,graphs,arrays['early_fit_0'],arrays['early_validation_0'],params,40000+1000*seed+10*fold,'cuda',directory/'resource_model.pt',epochs=5)
    result={'status':'complete','resource_qa_only':True,'outer_test_evaluated':False,'training_budget_counted':False,'dataset':ds,'feature_set':arm,'seed':seed,'fold':fold,'fixed_params':params,'epochs_requested':5,'elapsed_seconds':time.perf_counter()-start,'metadata':metadata,'protocol_sha256':sha(ROOT/'protocol.json'),'runner_sha256':sha(__file__),'finished_at':now()}
    jsave(directory/'measurement.json',result); print(json.dumps({'event':'profile_complete',**{k:v for k,v in result.items() if k!='metadata'},**memory()}),flush=True)

def probe(device):
    """Independent dense mean equations, isolation, sampling, deterministic fit."""
    global INPUT_DIM,ARM
    if device=='cuda': assert torch.cuda.is_available()
    ARM='probe'; reports=[]; rng=np.random.default_rng(123)
    edges=np.array([[0,1],[0,2],[2,3],[3,4]],np.int64)
    g=sparse.coo_matrix((np.ones(len(edges),np.float32),(edges[:,0],edges[:,1])),shape=(6,6)).tocsr()
    g=g.maximum(g.T); graphs=[normalize(g)]
    params={'hidden_dim':64,'dropout':0.,'learning_rate':.001}
    qa=ROOT/'qa'; qa.mkdir(parents=True,exist_ok=True)
    for width in (16,32,44,56):
        INPUT_DIM=width; torch.manual_seed(123)
        X=rng.normal(size=(6,width)).astype(np.float32); model=AccountNetwork(MODEL,params).eval()
        # Dense reference constructs each node's neighbor mean without CSR dot.
        state=model.state_dict(); h=X.copy()
        for layer in range(2):
            means=np.zeros_like(h)
            for node in range(6):
                neighbors=g.indices[g.indptr[node]:g.indptr[node+1]]
                if len(neighbors): means[node]=h[neighbors].mean(axis=0)
            with torch.no_grad():
                out=nn.functional.linear(torch.from_numpy(h),state[f'root.{layer}.0.weight'],state[f'root.{layer}.0.bias'])+nn.functional.linear(torch.from_numpy(means),state[f'neighbor.{layer}.0.weight'])
                h=(torch.relu(out) if layer==0 else out).numpy()
        expected=torch.sigmoid(torch.from_numpy(h[:,0])).double().numpy()
        actual=predict(model,X,graphs,np.arange(6))
        np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=1e-7)
        np.testing.assert_array_equal(predict(model,X,graphs,np.arange(6)),actual)
        first,r0,m0=block(graphs,np.arange(6),15,rng); second,r1,m1=block(graphs,first,10,rng)
        gpu=copy.deepcopy(model).to(device)
        with torch.no_grad():
            score=torch.sigmoid(gpu.forward_sampled(torch.from_numpy(X[second]).to(device),tensor_block(r1,m1,device),tensor_block(r0,m0,device))).cpu().double().numpy()
        np.testing.assert_allclose(score,actual,rtol=2e-5,atol=2e-6)
        with tempfile.TemporaryDirectory(dir=qa) as temporary:
            labels=np.array([0,1,0,1,0,1],np.int8); changed=labels.copy(); changed[4:]=1-changed[4:]
            fitparams={**params,'dropout':.2}; rows=np.arange(4)
            a,xa,ma=fit(MODEL,X,labels,graphs,rows,None,fitparams,987,device,Path(temporary)/'a.pt',epochs=2,prefetch=True,trace=True,batch_size=2)
            b,xb,mb=fit(MODEL,X,changed,graphs,rows,None,fitparams,987,device,Path(temporary)/'b.pt',epochs=2,prefetch=False,trace=True,batch_size=2)
            assert ma['model_digest']==mb['model_digest'],'Serial/prefetch seed repeat / nonfit-label invariance failed'
            assert ma['qa_trace']==mb['qa_trace'],'Serial/prefetch sampling indices or minibatch losses differ'
            np.testing.assert_array_equal(predict(a,xa,graphs,np.arange(6)),predict(b,xb,graphs,np.arange(6)))
            # Prefetch does not change sampled roots, maps, or per-batch RNG sequence.
            rnga=np.random.default_rng(987); rngb=np.random.default_rng(987)
            serial=sampled_batch(graphs,rows,rnga)
            with ThreadPoolExecutor(max_workers=1) as executor: threaded=executor.submit(sampled_batch,graphs,rows,rngb).result()
            for i in (0,1,2,4): np.testing.assert_array_equal(serial[i],threaded[i])
            for i in (3,5):
                for (pa,va),(pb,vb) in zip(serial[i],threaded[i]):
                    np.testing.assert_array_equal(pa,pb); np.testing.assert_array_equal(va,vb)
        reports.append({'input_dim':width,'dense_equation_max_abs_error':float(np.max(abs(actual-expected))),'device_sampled_max_abs_error':float(np.max(abs(score-actual))),'same_seed_exact_weights_and_scores':True,'nonfit_label_invariance':True,'isolates_retained':True,'prefetch_sample_equivalence':True,'serial_prefetch_2epoch_weights_losses_sampling_bitwise':True,'qa_epochs':2,'qa_seed_batch_size':2,'qa_minibatches_total':len(ma['qa_trace']['batch_loss'])})
    dense=sparse.csr_matrix(np.ones((25,25),dtype=np.int8)-np.eye(25,dtype=np.int8))
    sa=sample_neighbors(dense.indptr,dense.indices,np.arange(25),10,234)
    sb=sample_neighbors(dense.indptr,dense.indices,np.arange(25),10,234)
    np.testing.assert_array_equal(sa,sb)
    assert all(len(np.unique(row))==10 for row in sa)
    original=rng.normal(size=(6,16)).astype(np.float32); modified=original.copy(); modified[3:]+=10000
    np.testing.assert_array_equal(StandardScaler().fit(original[:3]).mean_,StandardScaler().fit(modified[:3]).mean_)
    result={'passed':True,'device':device,'runner_sha256':sha(__file__),'tested_at':now(),'torch_version':torch.__version__,'cuda_version':torch.version.cuda,'gpu_name':torch.cuda.get_device_name() if device=='cuda' else None,'tests':reports,'without_replacement_repeat':True,'fit_only_scaler':True,**memory()}
    jsave(ROOT/'probe.json',result); print(json.dumps(result,indent=2),flush=True)

_STOPPING=False
def cooperative_sigterm(signum,frame):
    """Drain only this process's queued CUDA work before graceful termination."""
    global _STOPPING
    if _STOPPING:return
    _STOPPING=True
    try:
        if torch.cuda.is_initialized():torch.cuda.synchronize()
    except RuntimeError as exc:
        print(json.dumps({'event':'sigterm_cuda_drain_error','error':str(exc)}),file=sys.stderr,flush=True)
    raise SystemExit(128+signum)

def main():
    global ROOT,OUT,ARM,CACHE_FIRST_MEAN,CACHE_CSR_BLOCKS,PACKED_H2D,GATHER_MODE
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('command',choices=('prepare','probe','profile','run-fold'))
    parser.add_argument('--dataset',choices=DATASETS); parser.add_argument('--arm',choices=ARMS)
    parser.add_argument('--seed',type=int,choices=range(5),default=0); parser.add_argument('--fold',type=int,choices=range(5),default=0)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--epochs',type=int,default=5); parser.add_argument('--tag')
    parser.add_argument('--gather-mode',choices=('reference','row_sentinel'),default='row_sentinel')
    parser.add_argument('--disable-first-mean-cache',action='store_true');parser.add_argument('--disable-csr-cache',action='store_true');parser.add_argument('--disable-packed-h2d',action='store_true')
    args=parser.parse_args(); ROOT=args.root.resolve(); ROOT.mkdir(parents=True,exist_ok=True); OUT=ROOT/'runs'; ARM=args.arm or 'probe'
    CACHE_FIRST_MEAN=not args.disable_first_mean_cache;CACHE_CSR_BLOCKS=not args.disable_csr_cache;PACKED_H2D=not args.disable_packed_h2d;GATHER_MODE=args.gather_mode
    configure();signal.signal(signal.SIGTERM,cooperative_sigterm)
    if args.command=='prepare': prepare()
    elif args.command=='probe': probe(args.device)
    else:
        assert args.dataset and args.arm,'dataset and arm required'
        if args.command=='profile': profile(args.dataset,args.arm,args.seed,args.fold,args.epochs,args.tag)
        else: run_fold(args.dataset,args.arm,args.seed,args.fold)

if __name__=='__main__': main()
