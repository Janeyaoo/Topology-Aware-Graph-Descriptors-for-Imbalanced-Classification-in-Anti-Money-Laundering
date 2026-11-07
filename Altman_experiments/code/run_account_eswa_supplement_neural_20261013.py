"""Account-only MLP / mean GraphSAGE under the frozen supplementary protocol.

No historical transaction classifier is imported. CSR sampling is label blind;
only current fit seeds enter loss, and all-neighbour inference is deterministic.
"""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','2')
import argparse, copy, gc, hashlib, json, platform, shutil, sys, tempfile, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy import sparse
from numba import njit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score
import run_account_feature_mechanism as common

ROOT=Path(__file__).resolve().parents[2]
DESIGN=ROOT/'Altman_experiments/results/eswa_supplement_design_20261013'
OLD=ROOT/'Altman_experiments/results/account_gmkep56_tuned_20261008'
SOURCE=ROOT/'Altman_experiments/results/account_s10_increment_20261008'
OUT=ROOT/'Altman_experiments/results/account_eswa_supplement_20261013/neural'
MODELS=('mlp','graphsage_weak','graphsage_directed')
DATASETS=('hi_small','li_small')
sha, rel, jsave, now=common.sha256, common.rel, common.write_json, common.now

def read(path): return json.loads(Path(path).read_text())
def ah(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def atomic_npz(path, **arrays):
    tmp=Path(str(path)+'.tmp')
    with tmp.open('wb') as f: np.savez_compressed(f,**arrays)
    tmp.replace(path)
def atomic_torch(path, value):
    tmp=Path(str(path)+'.tmp'); torch.save(value,tmp); tmp.replace(path)
def memory():
    import resource
    return {'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}

@njit(cache=True)
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

class AccountNetwork(nn.Module):
    """Weak=SAGEConv mean, directed=independent relation SAGE + linear fusion."""
    def __init__(self,name,params):
        super().__init__(); self.name=name; self.dropout=float(params['dropout'])
        hidden=int(params['hidden_dim'])
        if name=='mlp':
            self.layers=nn.ModuleList([nn.Linear(16,hidden),nn.Linear(hidden,1)])
        else:
            relations=1 if name=='graphsage_weak' else 2
            self.root=nn.ModuleList([nn.ModuleList([nn.Linear(d,o) for _ in range(relations)]) for d,o in ((16,hidden),(hidden,1))])
            self.neighbor=nn.ModuleList([nn.ModuleList([nn.Linear(d,o,bias=False) for _ in range(relations)]) for d,o in ((16,hidden),(hidden,1))])
            self.fusion=nn.ModuleList([nn.Identity() if relations==1 else nn.Linear(relations*o,o) for o in (hidden,1)])
    def conv(self,h,root_index,maps,layer):
        values=[]
        for r,(position,valid) in enumerate(maps):
            gathered=h[position]*valid.unsqueeze(-1)
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

def predict(model,X,graphs,rows,chunk=16384):
    """CPU, all-neighbour, layerwise; same inference for search and final."""
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
            for start in range(0,len(X),chunk):
                stop=min(start+chunk,len(X)); root=torch.from_numpy(np.ascontiguousarray(h[start:stop]))
                values=[]
                for r,g in enumerate(graphs):
                    mean=torch.from_numpy(np.ascontiguousarray(g[start:stop].dot(h)))
                    values.append(cpu.root[layer][r](root)+cpu.neighbor[layer][r](mean))
                values=cpu.fusion[layer](torch.cat(values,-1))
                if layer==0: values=torch.relu(values)
                out[start:stop]=values.numpy()
            h=out
        score=torch.sigmoid(torch.from_numpy(h[rows,0])).double().numpy()
        assert np.isfinite(score).all() and ((score>=0)&(score<=1)).all()
        return np.ascontiguousarray(score,dtype=np.float64)

def state_digest(model):
    h=hashlib.sha256()
    for key,value in model.state_dict().items():
        h.update(key.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def fit(name,raw,y,graphs,fit_rows,early_rows,params,seed,device,path,epochs=None):
    start=time.monotonic()
    torch.manual_seed(seed)
    if device=='mps': torch.mps.manual_seed(seed)
    scaler=StandardScaler().fit(raw[fit_rows])
    X=np.ascontiguousarray(scaler.transform(raw),dtype=np.float32)
    assert np.isfinite(X).all()
    model=AccountNetwork(name,params).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=float(params['learning_rate']),weight_decay=1e-4)
    n=len(fit_rows); npos=int(y[fit_rows].sum()); weights=np.array([n/(2*(n-npos)),n/(2*npos)],dtype=np.float32)
    rng=np.random.default_rng(seed+1000000)
    best_auc=-np.inf; best_epoch=0; best_state=None; history=[]
    batch_size=2048; fanouts=(15,10) if name=='graphsage_weak' else (10,5)
    limit=int(epochs or 200)
    for epoch in range(1,limit+1):
        model.train(); total_loss=0.; ordered=rng.permutation(fit_rows)
        for i in range(0,n,batch_size):
            roots=ordered[i:i+batch_size]
            optimizer.zero_grad(set_to_none=True)
            if name=='mlp':
                logits=model.forward_mlp(torch.from_numpy(np.ascontiguousarray(X[roots])).to(device))
            else:
                first,root0,maps0=block(graphs,roots,fanouts[0],rng)
                second,root1,maps1=block(graphs,first,fanouts[1],rng)
                logits=model.forward_sampled(torch.from_numpy(np.ascontiguousarray(X[second])).to(device),
                     tensor_block(root1,maps1,device),tensor_block(root0,maps0,device))
            labels=torch.from_numpy(np.ascontiguousarray(y[roots],dtype=np.float32)).to(device)
            w=torch.from_numpy(np.ascontiguousarray(weights[y[roots]])).to(device)
            loss=(nn.functional.binary_cross_entropy_with_logits(logits,labels,reduction='none')*w).mean()
            loss.backward(); optimizer.step(); total_loss+=float(loss.detach().cpu())*len(roots)
        record={'epoch':epoch,'weighted_seed_loss':total_loss/n,'seconds':time.monotonic()-start}
        if early_rows is not None:
            score=predict(model,X,graphs,early_rows)
            auc=float(roc_auc_score(y[early_rows],score)); record['early_auroc']=auc
            if auc>best_auc:
                best_auc=auc; best_epoch=epoch
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            if epoch-best_epoch>=30:
                history.append(record); break
        else: best_epoch=epoch
        history.append(record)
        if epoch%10==0 or epoch==1:
            print(json.dumps({'event':'fit_epoch','path':rel(path),'model':name,'epoch':epoch,**record}),flush=True)
    if early_rows is not None: model.load_state_dict(best_state)
    metadata={'model':name,'params':params,'device':device,'random_state':seed,
       'scaler_fit_rows_sha256':ah(fit_rows),'scaler_mean':scaler.mean_.tolist(),'scaler_scale':scaler.scale_.tolist(),
       'scaler_var':scaler.var_.tolist(),'scaler_n_samples_seen':int(scaler.n_samples_seen_),
       'fit_rows_sha256':ah(fit_rows),'early_rows_sha256':None if early_rows is None else ah(early_rows),
       'fit_n':n,'fit_positive':npos,'class_weights':weights.tolist(),'seed_only_loss':True,
       'selected_epochs':best_epoch,'epochs_run':len(history),'best_early_auroc':None if early_rows is None else best_auc,
       'history':history,'model_digest':state_digest(model),'seconds':time.monotonic()-start,**memory()}
    atomic_torch(path,{'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'metadata':metadata})
    jsave(path.with_suffix('.json'),metadata)
    return model,X,metadata

def prepare(device):
    assert not OUT.exists(),'Refuse to replace a registered neural experiment'
    OUT.mkdir(parents=True)
    candidates=read(DESIGN/'neural_candidates.json'); jsave(OUT/'candidates.json',candidates)
    proto=read(DESIGN/'protocol_design.json')
    proto.update(status='registered_for_execution',registered_at=now(),device=device,
       models=list(MODELS),runner_sha256=sha(__file__),torch_version=torch.__version__,
       preprocessing='all 16 nonnegative BASE fields log1p, fit-only StandardScaler; values float32',
       graph_architecture={'weak':'two mean SAGEConv equations with root affine plus bias-free neighbor affine',
          'directed':'independent incoming/outgoing mean SAGEConv then concat and linear fusion per layer',
          'nonlinearity':'ReLU and dropout only after first layer; binary second-layer logit'},
       inference='CPU float32 layerwise normalized CSR all-neighbour mean and torch linear, sigmoid then float64 copy',
       numeric_determinism='seeded torch/NumPy/Numba; CPU prediction fixed two threads; no autocast; exact score-repeat probe',
       threads=2,platform=platform.platform(),transaction_model_reused=False)
    jsave(OUT/'protocol.json',proto)
    binding={rel(DESIGN/'neural_candidates.json'):sha(DESIGN/'neural_candidates.json'),
             rel(DESIGN/'protocol_design.json'):sha(DESIGN/'protocol_design.json'),
             rel(OLD/'protocol.json'):sha(OLD/'protocol.json'),rel(OLD/'split_manifest.json'):sha(OLD/'split_manifest.json')}
    columns=read(OLD/'protocol.json')['feature_columns']['base16']
    inputs={}
    for ds in DATASETS:
        folder=OUT/ds
        for sub in ('arrays','search','folds','models','calibration_models','calibration','predictions'):
            (folder/sub).mkdir(parents=True)
        feature=SOURCE/ds/'features.parquet'
        table=pd.read_parquet(feature,columns=['account_id','label',*columns])
        identity=pd.read_parquet(OLD/ds/'accounts.parquet')
        np.testing.assert_array_equal(table.account_id,identity.account_id)
        y=table.label.to_numpy(np.int8)
        np.testing.assert_array_equal(y,np.load(OLD/ds/'arrays/labels.npy'))
        raw=np.ascontiguousarray(table[columns].to_numpy(np.float64))
        assert np.isfinite(raw).all() and (raw>=0).all()
        raw=np.ascontiguousarray(np.log1p(raw),dtype=np.float32)
        np.save(folder/'arrays/log_base16.npy',raw); np.save(folder/'arrays/labels.npy',y)
        identity.to_parquet(folder/'accounts.parquet',index=False,compression='zstd')
        # Cached weak graph is lexicographic, while models use frozen account rows.
        gp=SOURCE/ds/'graph_csr.npz'; mp=SOURCE/ds/'account_to_graph.npy'
        weak=sparse.load_npz(gp).tocsr(); mapping=np.load(mp)
        weak=weak[mapping][:,mapping].tocsr(); weak.sort_indices()
        assert weak.shape==(len(y),len(y)) and (weak!=weak.T).nnz==0 and weak.diagonal().sum()==0
        sparse.save_npz(folder/'arrays/weak_csr.npz',weak,compressed=True)
        trans=ROOT/f'Altman_experiments/data/{ds}/{ds}_trans.parquet'
        edges=pd.read_parquet(trans,columns=['from_id','to_id']); ids=pd.Index(identity.account_id)
        a=ids.get_indexer(edges.from_id); b=ids.get_indexer(edges.to_id)
        assert (a>=0).all() and (b>=0).all(); keep=a!=b
        directed=sparse.coo_matrix((np.ones(int(keep.sum()),np.int32),(a[keep],b[keep])),shape=weak.shape).tocsr()
        directed.data[:]=1; directed=directed.astype(np.int8); directed.sort_indices()
        assert (directed.maximum(directed.T)!=weak).nnz==0
        sparse.save_npz(folder/'arrays/outgoing_csr.npz',directed,compressed=True)
        sparse.save_npz(folder/'arrays/incoming_csr.npz',directed.T.tocsr(),compressed=True)
        files=list((folder/'arrays').iterdir())+[folder/'accounts.parquet']
        sources=[feature,gp,mp,trans,OLD/ds/'accounts.parquet',OLD/ds/'arrays/labels.npy']
        inputs[ds]={'n_samples':len(y),'n_positive':int(y.sum()),'base_columns':columns,
                    'account_order_sha256':ah(identity.row_index.to_numpy()),'account_id_sha256':ah(pd.util.hash_pandas_object(identity.account_id,index=False).to_numpy()),
                    'label_sha256':ah(y),'weak_edges':int(weak.nnz//2),'directed_edges':int(directed.nnz),
                    'cache_sha256':{rel(p):sha(p) for p in files},'source_sha256':{rel(p):sha(p) for p in sources},
                    'split_sources':{rel(OLD/ds/'splits'/f'seed{s}_fold{f}.npz'):sha(OLD/ds/'splits'/f'seed{s}_fold{f}.npz') for s in range(5) for f in range(5)}}
        del table,identity,raw,weak,directed,edges; gc.collect()
    jsave(OUT/'dataset_inputs.json',inputs); jsave(OUT/'source_binding.json',binding)
    jsave(OUT/'run_manifest.json',{'status':'registered','models':list(MODELS),'datasets':list(DATASETS),'planned_outer_settings':150,'planned_logical_candidate_scores':5400,'started_at':None})
    shutil.copy2(__file__,OUT/Path(__file__).name)

def registered():
    p=read(OUT/'protocol.json')
    assert sha(__file__)==p['runner_sha256'],'Registered code changed'
    for path,digest in read(OUT/'source_binding.json').items(): assert sha(ROOT/path)==digest
    return p

def search(ds,name,seed,fold,raw,y,graphs,arrays,p):
    folder=OUT/ds; stem=f'{name}_seed{seed}_fold{fold}'; directory=folder/'search'/stem
    directory.mkdir(exist_ok=True); result_path=directory/'search_result.json'
    if result_path.exists():
        result=read(result_path)
        for path,digest in result['artifact_sha256'].items(): assert sha(ROOT/path)==digest
        return result
    candidates=read(OUT/'candidates.json')['candidates']; rows=[]
    for c in candidates:
        for j in range(3):
            recpath=directory/f"candidate{c['candidate_id']}_inner{j}_record.json"
            if recpath.exists():
                record=read(recpath)
                assert record['protocol_sha256']==sha(OUT/'protocol.json')
                for path,digest in record['artifact_sha256'].items(): assert sha(ROOT/path)==digest
                rows.append(record); continue
            modelpath=directory/f"candidate{c['candidate_id']}_inner{j}.pt"
            fit_rows,early,score_rows=(arrays[f'{key}_{j}'] for key in ('early_fit','early_validation','inner_score'))
            model,X,meta=fit(name,raw,y,graphs,fit_rows,early,c['params'],40000+1000*seed+10*fold+j,p['device'],modelpath)
            tick=time.monotonic(); score=predict(model,X,graphs,score_rows)
            scorepath=modelpath.with_suffix('.npz'); atomic_npz(scorepath,row_index=score_rows,y_true=y[score_rows],y_score=score)
            record={'dataset':ds,'model':name,'feature_set':'base16','seed':seed,'fold':fold,
                'candidate_id':c['candidate_id'],'inner_fold':j,'params':c['params'],
                'ap':float(average_precision_score(y[score_rows],score)),'auc':float(roc_auc_score(y[score_rows],score)),
                'selected_epochs':meta['selected_epochs'],'fit_seconds':meta['seconds'],'predict_seconds':time.monotonic()-tick,
                'fit_rows_sha256':ah(fit_rows),'early_rows_sha256':ah(early),'score_rows_sha256':ah(score_rows),'raw_score_sha256':ah(score),
                'model_path':rel(modelpath),'score_path':rel(scorepath),'model_digest':meta['model_digest'],
                'protocol_sha256':sha(OUT/'protocol.json'),'artifact_sha256':{rel(x):sha(x) for x in (modelpath,modelpath.with_suffix('.json'),scorepath)}}
            jsave(recpath,record); rows.append(record)
            print(json.dumps({'event':'candidate_complete',**{k:record[k] for k in ('dataset','model','seed','fold','candidate_id','inner_fold','ap','auc','selected_epochs','fit_seconds')}}),flush=True)
            del model,X,score; gc.collect()
            if p['device']=='mps': torch.mps.empty_cache()
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
            'protocol_sha256':sha(OUT/'protocol.json')}
    jsave(result_path,result); return result

def stages(ds,name,seed,fold,strategy,cid,epochs,params,raw,y,graphs,arrays,identity,p):
    folder=OUT/ds; stem=f'{name}_seed{seed}_fold{fold}_{strategy}'
    D,T,train,test=(arrays[k] for k in ('development','threshold_validation','outer_train','outer_test'))
    modelpath=folder/'calibration_models'/f'{stem}.pt'
    model,X,calmeta=fit(name,raw,y,graphs,D,None,params,100*seed+fold,p['device'],modelpath,epochs)
    cal_score=predict(model,X,graphs,T); threshold,calf1=common.best_f1_threshold(y[T],cal_score)
    calpath=folder/'calibration'/f'{stem}.parquet'
    table=identity.iloc[T][['row_index','account_id']].copy(); table['y_true']=y[T]; table['y_score']=cal_score
    table.to_parquet(calpath,index=False,compression='zstd'); del model,X,table; gc.collect()
    finalpath=folder/'models'/f'{stem}.pt'
    model,X,finalmeta=fit(name,raw,y,graphs,train,None,params,100*seed+fold,p['device'],finalpath,epochs)
    score=predict(model,X,graphs,test); predpath=folder/'predictions'/f'{stem}.parquet'
    table=identity.iloc[test][['row_index','account_id']].copy(); table['y_true']=y[test]; table['y_score']=score
    records=[]
    for policy,cutoff in (('fixed_0.5',.5),('inner_max_f1',threshold)):
        pred,values=common.metrics(y[test],score,cutoff)
        table['y_pred_fixed' if policy=='fixed_0.5' else 'y_pred_inner']=pred
        records.append({'dataset':ds,'model':name,'feature_set':'base16','seed':seed,'fold':fold,'strategy':strategy,
                        'threshold_policy':policy,'threshold':cutoff,'selected_candidate_id':cid,'selected_epochs':epochs,**values})
    table.to_parquet(predpath,index=False,compression='zstd')
    files=[modelpath,modelpath.with_suffix('.json'),calpath,finalpath,finalpath.with_suffix('.json'),predpath]
    result={'strategy':strategy,'candidate_id':cid,'selected_epochs':epochs,'params':params,'threshold':threshold,
            'threshold_validation_f1':calf1,'threshold_validation_ap':float(average_precision_score(y[T],cal_score)),
            'model_random_state':100*seed+fold,'fit_rows_sha256':ah(train),'calibration_fit_rows_sha256':ah(D),
            'threshold_rows_sha256':ah(T),'outer_test_rows_sha256':ah(test),
            'calibration_model_path':rel(modelpath),'model_path':rel(finalpath),'calibration_path':rel(calpath),'prediction_path':rel(predpath),
            'calibration_metadata':calmeta,'final_metadata':finalmeta,'raw_score_sha256':ah(score),
            'metrics':records,'artifact_sha256':{rel(f):sha(f) for f in files},'protocol_sha256':sha(OUT/'protocol.json')}
    del model,X,score,cal_score,table; gc.collect()
    if p['device']=='mps': torch.mps.empty_cache()
    return result

def run(ds,name,only_seed=None,only_fold=None):
    p=registered(); folder=OUT/ds; inputs=read(OUT/'dataset_inputs.json')[ds]
    for path,digest in {**inputs['cache_sha256'],**inputs['source_sha256'],**inputs['split_sources']}.items(): assert sha(ROOT/path)==digest
    raw=np.load(folder/'arrays/log_base16.npy',mmap_mode='r'); y=np.load(folder/'arrays/labels.npy',mmap_mode='r')
    identity=pd.read_parquet(folder/'accounts.parquet')
    if name=='mlp': graphs=[]
    else:
        names=('weak',) if name=='graphsage_weak' else ('incoming','outgoing')
        graphs=[normalize(sparse.load_npz(folder/'arrays'/f'{n}_csr.npz').tocsr()) for n in names]
    candidates=read(OUT/'candidates.json')['candidates']
    manifest_path=folder/f'run_manifest_{name}.json'
    manifest={'dataset':ds,'model':name,'status':'running','pid':os.getpid(),'started_at':now(),
              'protocol_sha256':sha(OUT/'protocol.json'),'completed_outer_settings':0,'device':p['device']}
    jsave(manifest_path,manifest)
    try:
        for seed in range(5):
            if only_seed is not None and seed!=only_seed: continue
            for fold in range(5):
                if only_fold is not None and fold!=only_fold: continue
                path=folder/'folds'/f'{name}_seed{seed}_fold{fold}.json'
                if path.exists():
                    saved=read(path)
                    assert saved['protocol_sha256']==sha(OUT/'protocol.json')
                    for result in saved['strategies']:
                        for artifact,digest in result['artifact_sha256'].items(): assert sha(ROOT/artifact)==digest
                else:
                    with np.load(OLD/ds/'splits'/f'seed{seed}_fold{fold}.npz') as f: arrays={k:f[k] for k in f.files}
                    result=search(ds,name,seed,fold,raw,y,graphs,arrays,p)
                    strategy_results=[]
                    for strategy,cid,epochs in (('same_protocol_anchor',0,result['anchor_epochs']),('tuned',result['selected_candidate_id'],result['selected_epochs'])):
                        partial=folder/'folds'/f'{name}_seed{seed}_fold{fold}_{strategy}_partial.json'
                        if partial.exists():
                            value=read(partial)
                            assert value['protocol_sha256']==sha(OUT/'protocol.json')
                            for artifact,digest in value['artifact_sha256'].items(): assert sha(ROOT/artifact)==digest
                        else:
                            value=stages(ds,name,seed,fold,strategy,cid,epochs,candidates[cid]['params'],raw,y,graphs,arrays,identity,p)
                            jsave(partial,value)
                        strategy_results.append(value)
                    saved={'dataset':ds,'model':name,'feature_set':'base16','seed':seed,'fold':fold,'search':result,
                           'strategies':strategy_results,'protocol_sha256':sha(OUT/'protocol.json'),'completed_at':now()}
                    jsave(path,saved)
                manifest['completed_outer_settings']+=1; manifest['last_completed']={'seed':seed,'fold':fold}; manifest['updated_at']=now()
                jsave(manifest_path,manifest)
                print(json.dumps({'event':'outer_complete','dataset':ds,'model':name,'seed':seed,'fold':fold}),flush=True)
        completed=len(list((folder/'folds').glob(f'{name}_seed*_fold[0-4].json')))
        manifest.update(status='complete' if completed==25 else 'partial_scope_complete',completed_outer_settings=completed,finished_at=now(),pid=None)
        jsave(manifest_path,manifest); summarize()
    except BaseException as exc:
        manifest.update(status='failed',error=repr(exc),traceback=traceback.format_exc(),updated_at=now(),pid=None)
        jsave(manifest_path,manifest); raise

def summarize():
    rows=[]; candidate_rows=[]
    for ds in DATASETS:
        folder=OUT/ds
        for path in sorted((folder/'folds').glob('*_seed*_fold[0-4].json')):
            result=read(path)
            candidate_rows.extend(result['search']['candidate_records'])
            for stage in result['strategies']: rows.extend(stage['metrics'])
    if rows:
        frame=pd.DataFrame(rows); frame.to_csv(OUT/'fold_metrics.csv',index=False)
        metrics=['ap','auc','f1','precision','recall','balanced_accuracy','accuracy']
        summary=frame.groupby(['dataset','model','feature_set','strategy','threshold_policy'])[metrics].agg(['mean','std','count'])
        summary.columns=['_'.join(c) for c in summary.columns]; summary.reset_index().to_csv(OUT/'summary_partial.csv',index=False)
        pd.DataFrame([{k:v for k,v in r.items() if k not in ('params','artifact_sha256')} for r in candidate_rows]).to_csv(OUT/'candidate_fold_scores.csv',index=False)
    oof_rows=[]
    for ds in DATASETS:
        folder=OUT/ds; y=np.load(folder/'arrays/labels.npy')
        for name in MODELS:
            for seed in range(5):
                completed=[folder/'folds'/f'{name}_seed{seed}_fold{fold}.json' for fold in range(5)]
                if not all(x.exists() for x in completed): continue
                for strategy in ('same_protocol_anchor','tuned'):
                    score=np.full(len(y),np.nan); pred_fixed=np.zeros(len(y),np.int8); pred_inner=pred_fixed.copy(); seen=np.zeros(len(y),np.int8)
                    for path in completed:
                        fold=read(path); stage=next(s for s in fold['strategies'] if s['strategy']==strategy)
                        table=pd.read_parquet(ROOT/stage['prediction_path']); index=table.row_index.to_numpy(); seen[index]+=1
                        score[index]=table.y_score; pred_fixed[index]=table.y_pred_fixed; pred_inner[index]=table.y_pred_inner
                    assert (seen==1).all() and np.isfinite(score).all()
                    oofpath=folder/'predictions'/f'{name}_seed{seed}_{strategy}_oof.parquet'
                    pd.DataFrame({'row_index':np.arange(len(y)),'y_true':y,'y_score':score,'y_pred_fixed':pred_fixed,'y_pred_inner':pred_inner}).to_parquet(oofpath,index=False,compression='zstd')
                    # OOF threshold predictions use each fold's independent T cutoff.
                    from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score,accuracy_score,balanced_accuracy_score
                    for policy,pred in (('fixed_0.5',pred_fixed),('inner_max_f1',pred_inner)):
                        tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
                        oof_rows.append({'dataset':ds,'model':name,'feature_set':'base16','seed':seed,'strategy':strategy,'threshold_policy':policy,
                            'ap':float(average_precision_score(y,score)),'auc':float(roc_auc_score(y,score)),
                            'precision':float(precision_score(y,pred,zero_division=0)),'recall':float(recall_score(y,pred,zero_division=0)),
                            'f1':float(f1_score(y,pred,zero_division=0)),'accuracy':float(accuracy_score(y,pred)),
                            'balanced_accuracy':float(balanced_accuracy_score(y,pred)), 'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),
                            'n_test':len(y),'n_positive':int(y.sum()),'prediction_path':rel(oofpath),'prediction_sha256':sha(oofpath)})
    if oof_rows: pd.DataFrame(oof_rows).to_csv(OUT/'seed_oof_metrics_partial.csv',index=False)
    completed=sum(len(list((OUT/ds/'folds').glob('*_seed*_fold[0-4].json'))) for ds in DATASETS)
    records=sum(len(list((OUT/ds/'search').glob('*/candidate*_inner*_record.json'))) for ds in DATASETS)
    jsave(OUT/'progress.json',{'updated_at':now(),'completed_outer_settings':completed,'planned_outer_settings':150,
           'completed_candidate_scores':records,'planned_candidate_scores':5400,'is_complete':completed==150,'partial_statistics_not_final':completed!=150})

def probe(device):
    """Independent PyG-equation, empty-neighbour, scaler, seed-loss, repeat tests."""
    from torch_geometric.nn import SAGEConv
    torch.manual_seed(123); rng=np.random.default_rng(123)
    edges=np.array([[0,1],[0,2],[2,3],[3,4]],np.int64)
    directed=sparse.coo_matrix((np.ones(len(edges)),(edges[:,0],edges[:,1])),shape=(6,6)).tocsr()
    weak=directed.maximum(directed.T); graphs=[normalize(weak)]
    params={'hidden_dim':64,'dropout':0.,'learning_rate':.001}
    X=rng.normal(size=(6,16)).astype(np.float32)
    model=AccountNetwork('graphsage_weak',params)
    refs=[SAGEConv((16,16),64,aggr='mean'),SAGEConv((64,64),1,aggr='mean')]
    for layer,ref in enumerate(refs):
        with torch.no_grad():
            ref.lin_l.weight.copy_(model.neighbor[layer][0].weight); ref.lin_l.bias.copy_(model.root[layer][0].bias)
            ref.lin_r.weight.copy_(model.root[layer][0].weight)
    row,col=weak.nonzero(); edge_index=torch.tensor(np.vstack([col,row]),dtype=torch.long)
    with torch.no_grad():
        h=refs[1](torch.relu(refs[0](torch.from_numpy(X),edge_index)),edge_index).squeeze()
        reference=torch.sigmoid(h).double().numpy()
    actual=predict(model,X,graphs,np.arange(6)); np.testing.assert_allclose(actual,reference,rtol=1e-6,atol=1e-7)
    first,r0,m0=block(graphs,np.arange(6),15,rng); second,r1,m1=block(graphs,first,10,rng)
    model.eval()
    with torch.no_grad():
        sampled=torch.sigmoid(model.forward_sampled(torch.from_numpy(X[second]),tensor_block(r1,m1,'cpu'),tensor_block(r0,m0,'cpu'))).double().numpy()
    np.testing.assert_allclose(sampled,actual,rtol=1e-6,atol=1e-7)
    np.testing.assert_array_equal(predict(model,X,graphs,np.arange(6)),actual)
    dgraphs=[normalize(directed.T),normalize(directed)]
    for g in dgraphs:
        assert np.isfinite(g.data).all()
    dm=AccountNetwork('graphsage_directed',params).eval()
    h=torch.from_numpy(X)
    with torch.no_grad():
        for layer in range(2):
            value=[]
            for r,g in enumerate(dgraphs):
                # Independently use PyG's source->target mean aggregation.
                ref=SAGEConv((h.shape[1],h.shape[1]),dm.root[layer][r].out_features,aggr='mean')
                ref.lin_l.weight.copy_(dm.neighbor[layer][r].weight); ref.lin_l.bias.copy_(dm.root[layer][r].bias)
                ref.lin_r.weight.copy_(dm.root[layer][r].weight)
                row,col=g.nonzero(); relation_edges=torch.tensor(np.vstack([col,row]),dtype=torch.long)
                value.append(ref(h,relation_edges))
            h=dm.fusion[layer](torch.cat(value,-1))
            if layer==0: h=torch.relu(h)
        dref=torch.sigmoid(h[:,0]).double().numpy()
    dactual=predict(dm,X,dgraphs,np.arange(6)); np.testing.assert_allclose(dactual,dref,rtol=1e-6,atol=1e-7)
    df,r0,m0=block(dgraphs,np.arange(6),10,rng); ds,r1,m1=block(dgraphs,df,5,rng)
    with torch.no_grad():
        dsampled=torch.sigmoid(dm.forward_sampled(torch.from_numpy(X[ds]),tensor_block(r1,m1,'cpu'),tensor_block(r0,m0,'cpu'))).double().numpy()
    np.testing.assert_allclose(dsampled,dactual,rtol=1e-6,atol=1e-7)
    dense=sparse.csr_matrix(np.ones((25,25),dtype=np.int8)-np.eye(25,dtype=np.int8))
    a=sample_neighbors(dense.indptr,dense.indices,np.arange(25),10,234)
    b=sample_neighbors(dense.indptr,dense.indices,np.arange(25),10,234)
    np.testing.assert_array_equal(a,b)
    assert all(len(np.unique(row))==10 for row in a)
    # Targets alone produce labels: changing every non-seed label changes no loss.
    logits=torch.tensor([.1,-.3]); seed_labels=torch.tensor([1.,0.]); all_y=np.array([1,0,1,0,0,1]); other=all_y.copy(); other[2:]=1-other[2:]
    seed=np.array([0,1]); loss=nn.functional.binary_cross_entropy_with_logits(logits,torch.tensor(all_y[seed],dtype=torch.float32))
    loss2=nn.functional.binary_cross_entropy_with_logits(logits,torch.tensor(other[seed],dtype=torch.float32)); assert loss.item()==loss2.item()
    scaler=StandardScaler().fit(X[:3]); changed=X.copy(); changed[3:]+=10000
    scaler2=StandardScaler().fit(changed[:3]); np.testing.assert_array_equal(scaler.mean_,scaler2.mean_)
    backend={'requested_device':device,'mps_available':torch.backends.mps.is_available(),'cpu_prediction_repeat_exact':True,'pyg_mean_equation_max_abs_error':float(np.max(abs(actual-reference))),
             'sampled_all_neighbor_max_abs_error':float(np.max(abs(sampled-actual))),'directed_pyg_equation_max_abs_error':float(np.max(abs(dactual-dref))),
             'directed_sampled_all_neighbor_max_abs_error':float(np.max(abs(dsampled-dactual))),
             'without_replacement_seed_repeat_passed':True,'isolated_nodes_retained':True,'seed_loss_label_mask_passed':True,'fit_only_scaler_passed':True}
    if device=='mps':
        assert torch.backends.mps.is_available()
        mps=copy.deepcopy(model).to('mps')
        first,r0,m0=block(graphs,np.arange(6),15,rng)
        second,r1,m1=block(graphs,first,10,rng)
        with torch.no_grad():
            mscore=torch.sigmoid(mps.forward_sampled(torch.from_numpy(X[second]).to('mps'),tensor_block(r1,m1,'mps'),tensor_block(r0,m0,'mps'))).cpu().double().numpy()
        np.testing.assert_allclose(mscore,actual,rtol=2e-5,atol=2e-6)
        mps.train(); mlog=mps.forward_sampled(torch.from_numpy(X[second]).to('mps'),tensor_block(r1,m1,'mps'),tensor_block(r0,m0,'mps'))
        nn.functional.binary_cross_entropy_with_logits(mlog[:2],seed_labels.to('mps')).backward()
        backend.update(mps_forward_max_abs_error=float(np.max(abs(mscore-actual))),mps_backward_seed_loss_passed=True)
    with tempfile.TemporaryDirectory(dir=OUT.parent) as scratch:
        labels=np.array([0,1,0,1,0,1],np.int8); modified=labels.copy(); modified[4:]=1-modified[4:]
        for name,gs in (('mlp',[]),('graphsage_weak',graphs),('graphsage_directed',dgraphs)):
            fitparams={**params,'dropout':.2}; rows=np.arange(4)
            m1,x1,z1=fit(name,X,labels,gs,rows,None,fitparams,987,device,Path(scratch)/f'{name}_a.pt',epochs=2)
            m2,x2,z2=fit(name,X,modified,gs,rows,None,fitparams,987,device,Path(scratch)/f'{name}_b.pt',epochs=2)
            assert z1['model_digest']==z2['model_digest'],'Fit seeds or non-fit labels change trained weights'
            np.testing.assert_array_equal(predict(m1,x1,gs,np.arange(6)),predict(m2,x2,gs,np.arange(6)))
            backend[f'{name}_seeded_fit_repeat_and_nonfit_labels_passed']=True
    report={'passed':True,'runner_sha256':sha(__file__),'tested_at':now(),'backend':backend}
    dest=OUT/'probe.json' if OUT.exists() else OUT.parent/f'neural_probe_{device}.json'
    dest.parent.mkdir(parents=True,exist_ok=True); jsave(dest,report); print(json.dumps(report,indent=2))

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('command',choices=('probe','prepare','run','summarize'))
    parser.add_argument('--device',choices=('cpu','mps'),default='cpu'); parser.add_argument('--dataset',choices=DATASETS); parser.add_argument('--model',choices=MODELS)
    parser.add_argument('--seed',type=int); parser.add_argument('--fold',type=int); args=parser.parse_args()
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    if args.command=='probe': probe(args.device)
    elif args.command=='prepare':
        check=read(OUT.parent/f'neural_probe_{args.device}.json'); assert check['passed'] and check['runner_sha256']==sha(__file__)
        prepare(args.device)
    elif args.command=='run':
        assert args.dataset and args.model; run(args.dataset,args.model,args.seed,args.fold)
    else: summarize()

if __name__=='__main__': main()
