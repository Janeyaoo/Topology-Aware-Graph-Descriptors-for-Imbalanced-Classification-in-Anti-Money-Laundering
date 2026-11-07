"""Source-bound ESWA supplement tree training; historical experiments are read-only.

The unchanged historical trainer supplies fit/search/threshold pure functions.
This runner never calls its prepare/run/execute entry points and always writes
to the new trees subdirectory. Deferred inputs are admitted by immutable,
dataset-specific registrations, so the pre-registered search does not change.
"""
from __future__ import annotations
import argparse, copy, fcntl, gc, importlib.metadata, json, os, platform
import shutil, sys, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import run_account_gmkep_tuning as core

ROOT=Path(__file__).resolve().parents[2]
ALT=ROOT/'Altman_experiments'
MAIN=ALT/'results/account_gmkep56_tuned_20261008'
DESIGN=ALT/'results/eswa_supplement_design_20261013'
DEFAULT_OUT=ALT/'results/account_eswa_supplement_20261013/trees'
SCRIPT=Path(__file__).resolve()
DATASETS=core.DATASETS
MODELS=core.MODELS
sha,rel,read,save,now=core.sha,core.rel,core.read,core.jsave,core.now
PRIORITY=('struct_h0_50','struct_h1_50','full_drop_ge36','full_drop_m51',
          'full_drop_k53','full_drop_e52','base_g32','base_m21','base_k19',
          'base_e20','base_p28','struct_n46','struct_n_h1_52',
          'struct_q10_54','struct_q10_h1_60','base_gfp_account_stream')

def require(ok,message):
    if not ok: raise RuntimeError(message)

def check_artifacts(mapping):
    for path,digest in mapping.items():
        require(sha(ROOT/path)==digest,'Source/artifact changed: '+path)

def registered(out):
    p=read(out/'protocol.json')
    require(p['trainer_sha256']==sha(SCRIPT),'Supplement trainer changed after registration')
    require(p['inherited_trainer_sha256']==sha(core.__file__),'Historical pure-function source changed')
    require(p['helper_sha256']==sha(core.common.__file__),'Historical metrics helper changed')
    for stem in ('candidates','dataset_inputs','split_manifest','source_manifest'):
        require(p[stem+'_sha256']==sha(out/(stem+'.json')),'Registration changed: '+stem)
    return p

def prepare(out):
    require(not out.exists(),'New tree output directory required; existing results are not overwritten')
    dm=read(DESIGN/'design_manifest.json')
    require(read(DESIGN/'validation.json')['passed'] is True,'Design validation missing')
    require(sha(DESIGN/'source_manifest.json')==dm['source_manifest_sha256'],'Design source registry changed')
    check_artifacts(dm['output_sha256'])
    source=read(DESIGN/'source_manifest.json')['source_sha256']
    check_artifacts(source)
    prior=read(MAIN/'protocol.json');registry=read(DESIGN/'feature_registry.json')
    arms={a['arm_id']:a for a in registry['tree_arms'] if a['status']=='planned'}
    require(set(arms)==set(PRIORITY),'Unexpected core tree arms')
    candidates=read(DESIGN/'tree_candidates.json')
    require(candidates==read(MAIN/'candidates.json'),'Shared candidate lists changed')
    require(all(len(candidates[m])==12 for m in MODELS),'12 candidates per classifier required')
    out.mkdir(parents=True)
    (out/'snapshots').mkdir()
    for path in [SCRIPT,Path(core.__file__),Path(core.common.__file__),DESIGN/'feature_registry.json',
                 DESIGN/'protocol_design.json',DESIGN/'tree_candidates.json',DESIGN/'design_manifest.json']:
        shutil.copy2(path,out/'snapshots'/path.name)
    shutil.copy2(DESIGN/'tree_candidates.json',out/'candidates.json')
    stored_columns=prior['feature_columns']['gmkep56']+['q_n2','q_n']
    input_registry={};split_registry={}
    old_inputs=read(MAIN/'dataset_inputs.json');old_splits=read(MAIN/'split_manifest.json')
    for ds in DATASETS:
        folder=out/ds;folder.mkdir()
        for sub in ('splits','arrays','search','folds','models','calibration_models','predictions',
                    'calibration','selected_search_models','selected_search_scores','input_registrations','stage_checkpoints'):
            (folder/sub).mkdir()
        src=ROOT/old_inputs[ds]['source_features']
        require(sha(src)==old_inputs[ds]['source_sha256'][rel(src)],'Full feature source changed')
        table=pd.read_parquet(src,columns=['account_id','label',*stored_columns])
        identity=pd.read_parquet(MAIN/ds/'accounts.parquet')
        y=np.load(MAIN/ds/'arrays/labels.npy')
        np.testing.assert_array_equal(table.account_id.to_numpy(),identity.account_id.to_numpy())
        np.testing.assert_array_equal(table.label.to_numpy(),y)
        matrix=np.ascontiguousarray(table[stored_columns].to_numpy(np.float32))
        require(np.isfinite(matrix).all(),'Nonfinite cache')
        expected=np.load(MAIN/ds/'arrays/features.npy',mmap_mode='r')
        np.testing.assert_array_equal(matrix[:,:56],expected)
        # Keep the existing 56-column matrix, labels and identity read-only.
        # Only the two previously excluded geometry counts require a new cache.
        np.save(folder/'arrays/sample_counts.npy',matrix[:,56:])
        paths=(MAIN/ds/'arrays/features.npy',MAIN/ds/'arrays/labels.npy',MAIN/ds/'accounts.parquet',
               folder/'arrays/sample_counts.npy')
        cache={rel(path):sha(path) for path in paths}
        input_registry[ds]={'n_samples':len(y),'n_positive':int(y.sum()),'columns':stored_columns,
                           'source_features':rel(src),'source_sha256':{rel(src):sha(src)},'cache_sha256':cache,
                           'feature_array_path':rel(paths[0]),'labels_array_path':rel(paths[1]),
                           'accounts_path':rel(paths[2]),'sample_counts_path':rel(paths[3]),
                           'historical_arrays_are_referenced_not_copied':True}
        for seed in range(5):
            for fold in range(5):
                source_split=MAIN/ds/'splits'/f'seed{seed}_fold{fold}.npz'
                source_record=old_splits[rel(source_split)]
                require(sha(source_split)==source_record['sha256'],'Original split changed')
                target=folder/'splits'/source_split.name;shutil.copy2(source_split,target)
                with np.load(target) as f:
                    require({k:core.array_sha(f[k]) for k in f.files}==source_record['arrays_sha256'],
                            'Split array hashes changed')
                split_registry[rel(target)]={**source_record,'original_split_path':rel(source_split),
                                            'original_split_sha256':sha(source_split)}
        del table,matrix,y,identity,expected;gc.collect()
    save(out/'dataset_inputs.json',input_registry)
    save(out/'split_manifest.json',split_registry)
    protected={**source,rel(DESIGN/'feature_registry.json'):sha(DESIGN/'feature_registry.json'),
               rel(DESIGN/'tree_candidates.json'):sha(DESIGN/'tree_candidates.json'),
               rel(DESIGN/'protocol_design.json'):sha(DESIGN/'protocol_design.json'),
               rel(SCRIPT):sha(SCRIPT)}
    save(out/'source_manifest.json',{'source_sha256':protected,'old_sources_read_only':True})
    p=copy.deepcopy(prior)
    p.update(experiment='account_eswa_supplement_tree_nested_tuning_20261013',registered_at=now(),
             old_experiment_protocol_sha256=sha(MAIN/'protocol.json'),
             design_manifest_sha256=sha(DESIGN/'design_manifest.json'),
             feature_columns={a:arms[a]['columns'] for a in PRIORITY},
             feature_sets={a:arms[a]['n_features'] for a in PRIORITY},
             feature_arm_priority=list(PRIORITY),deferred_input_arms=['struct_q10_54','struct_q10_h1_60','base_gfp_account_stream'],
             trainer_sha256=sha(SCRIPT),inherited_trainer_sha256=sha(core.__file__),
             helper_sha256=sha(core.common.__file__),threads_per_worker=2,max_concurrent_workers=2,
             expected_outer_settings=2400,logical_candidate_evaluations=86400,
             expected_search_physical_fits=72000,expected_representative_refits=2400,
             expected_strategy_final_models_logical=4800,expected_strategy_threshold_models_logical=4800,
             expected_old_reference_settings_reused=450,
             deferred_input_registration='Immutable dataset/arm registration includes finite full-account aligned matrix, schema, acceptance evidence and source hashes; protocol and candidates remain fixed.',
             source_manifest_sha256=sha(out/'source_manifest.json'),
             partial_result_policy='Incomplete settings do not form completed mean/SD comparison claims.',
             environment={'python':platform.python_version(),'machine':platform.machine(),'cpu_count':os.cpu_count(),
               **{k:importlib.metadata.version(k) for k in ('numpy','pandas','scikit-learn','lightgbm','xgboost','joblib','pyarrow')}})
    for stem in ('candidates','dataset_inputs','split_manifest'):
        p[stem+'_sha256']=sha(out/(stem+'.json'))
    save(out/'protocol.json',p)
    save(out/'run_manifest.json',{'status':'registered','registered_at':now(),
        'protocol_sha256':sha(out/'protocol.json'),'completed_outer_settings':0,
        'expected_outer_settings':2400,'original_graph32_resumed':False})
    print(json.dumps({'prepared':rel(out),'cached_arms':list(PRIORITY[:13]),'new_tree_settings':96}),flush=True)

def admit_input(out,ds,kind,path,evidence):
    p=registered(out);folder=out/ds
    require(kind in ('q10','gfp'),'Unknown deferred input kind')
    path=path.resolve();evidence=evidence.resolve()
    acceptance=read(evidence)
    require(acceptance.get('passed') is True,'Deferred input must have independent passed acceptance')
    require(acceptance.get('dataset',ds)==ds,'Acceptance dataset mismatch')
    inputs=read(out/'dataset_inputs.json')[ds]
    identity=pd.read_parquet(ROOT/inputs['accounts_path'])
    table=pd.read_parquet(path)
    require(table.account_id.is_unique and len(table)==len(identity),'Input must contain every unique account')
    np.testing.assert_array_equal(table.account_id.to_numpy(),identity.account_id.to_numpy())
    require('label' not in table.columns,'Deferred features must not contain labels')
    numeric=[c for c in table.columns if c!='account_id']
    if kind=='q10':
        columns=[f'q{hop}_{item}' for hop in (2,3) for item in ('n','e1','t1','e2','t2')]
        require(numeric==columns,'Q10 schema/order must match registered dictionary')
        features=np.ascontiguousarray(table[columns].to_numpy(np.float32))
        counts=np.load(ROOT/inputs['sample_counts_path'],mmap_mode='r')
        np.testing.assert_array_equal(features[:,0],counts[:,0])
        np.testing.assert_array_equal(features[:,5],counts[:,1])
        arms=['struct_q10_54','struct_q10_h1_60']
    else:
        require(len(numeric)>0 and all(c not in ['label','row_index','transaction_id','timestamp'] for c in numeric),
                'GFP engineering columns required; no indexing fields')
        columns=numeric;features=np.ascontiguousarray(table[columns].to_numpy(np.float32))
        arms=['base_gfp_account_stream']
    require(np.isfinite(features).all(),'Deferred input contains nonfinite values')
    array_path=folder/'arrays'/(kind+'_features.npy')
    manifest_path=folder/'input_registrations'/(kind+'.json')
    require(not manifest_path.exists(),'Deferred input already registered; immutable registration required')
    np.save(array_path,features)
    save(manifest_path,{'kind':kind,'dataset':ds,'registered_at':now(),'passed':True,
         'protocol_sha256':sha(out/'protocol.json'),'columns':columns,'n_rows':len(features),'n_features':len(columns),
         'arms':arms,'source_path':rel(path),'acceptance_path':rel(evidence),'array_path':rel(array_path),
         'artifact_sha256':{rel(path):sha(path),rel(evidence):sha(evidence),rel(array_path):sha(array_path)},
         'account_id_array_sha256':core.array_sha(identity.row_index.to_numpy()),
         'schema':{c:str(table[c].dtype) for c in table.columns}})
    print(json.dumps({'admitted':rel(manifest_path),'arms':arms}),flush=True)

def matrix_for_arm(out,folder,p,base,counts,stored_columns,arm):
    columns=p['feature_columns'][arm]
    if arm in p['deferred_input_arms']:
        kind='gfp' if arm=='base_gfp_account_stream' else 'q10'
        regpath=folder/'input_registrations'/(kind+'.json')
        if not regpath.exists(): return None,None,None
        registration=read(regpath)
        require(registration['protocol_sha256']==sha(out/'protocol.json'),'Deferred registration binding changed')
        check_artifacts(registration['artifact_sha256'])
        extra=np.load(ROOT/registration['array_path'],mmap_mode='r')
        if kind=='gfp':
            columns=stored_columns[:16]+registration['columns']
            X=np.ascontiguousarray(np.column_stack((base[:,:16],extra)),dtype=np.float32)
        else:
            names=stored_columns[:56]+registration['columns']
            full=np.column_stack((base,extra))
            X=np.ascontiguousarray(full[:,[names.index(c) for c in columns]],dtype=np.float32)
            del full
        return X,columns,{'path':rel(regpath),'sha256':sha(regpath)}
    indices=[stored_columns.index(c) for c in columns]
    if max(indices)<56:
        return np.ascontiguousarray(base[:,indices]),columns,None
    combined=np.column_stack((base,counts))
    return np.ascontiguousarray(combined[:,indices]),columns,None

def verify_done(out,path):
    done=read(path)
    require(done['protocol_sha256']==sha(out/'protocol.json'),'Completed setting protocol changed')
    check_artifacts(done['artifact_sha256'])
    return done

def run(out,ds,name,arms=None,max_settings=None):
    p=registered(out);folder=out/ds;design=read(out/'candidates.json')
    inp=read(out/'dataset_inputs.json')[ds];check_artifacts(inp['cache_sha256'])
    base=np.load(ROOT/inp['feature_array_path'],mmap_mode='r')
    counts=np.load(ROOT/inp['sample_counts_path'],mmap_mode='r')
    y=np.load(ROOT/inp['labels_array_path'],mmap_mode='r')
    identity=pd.read_parquet(ROOT/inp['accounts_path'])
    selected=p['feature_arm_priority'] if not arms else arms
    require(set(selected)<=set(p['feature_arm_priority']),'Unregistered arm requested')
    lock=(folder/(name+'.worker.lock')).open('a')
    try: fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: raise RuntimeError('A worker for this dataset/model is already active')
    manifest_path=folder/f'run_manifest_{name}.json'
    previous=read(manifest_path) if manifest_path.exists() else None
    manifest={'dataset':ds,'model':name,'status':'running','pid':os.getpid(),'started_at':now(),
              'first_started_at':previous.get('first_started_at',previous['started_at']) if previous else now(),
              'protocol_sha256':sha(out/'protocol.json'),'requested_arms':selected,
              'expected_outer_settings':len(p['feature_arm_priority'])*25,'completed_outer_settings':0,
              'previous_attempts':(previous.get('previous_attempts',[])+[previous]) if previous else [],'deferred_arms':[]}
    save(manifest_path,manifest);new_settings=0
    split_registry=read(out/'split_manifest.json')
    try:
        for arm in selected:
            X,columns,extra_reg=matrix_for_arm(out,folder,p,base,counts,inp['columns'],arm)
            if X is None:
                manifest['deferred_arms'].append(arm)
                print(f'{now()} DEFERRED {ds}/{name}/{arm}: validated input not registered',flush=True)
                continue
            for seed in range(5):
                for fold in range(5):
                    stem=f'{name}_{arm}_seed{seed}_fold{fold}';path=folder/'folds'/(stem+'.json')
                    if path.exists(): verify_done(out,path)
                    else:
                        core.guard_disk(folder);start=time.monotonic()
                        split=folder/'splits'/f'seed{seed}_fold{fold}.npz'
                        require(sha(split)==split_registry[rel(split)]['sha256'],'Copied split changed')
                        with np.load(split) as f:arrays={k:f[k] for k in f.files}
                        sr=core.search(out,folder,name,arm,seed,fold,X,y,arrays,p,design)
                        stage_path=folder/'stage_checkpoints'/(stem+'_stages.json')
                        stages=read(stage_path) if stage_path.exists() else {}
                        for saved in stages.values():check_artifacts(saved['artifact_sha256'])
                        if 'same_protocol_anchor' not in stages:
                            stages['same_protocol_anchor']=core.strategy_fit(out,folder,name,arm,seed,fold,'same_protocol_anchor',
                                0,sr['anchor_rounds'],X,y,identity,arrays,design[name][0]['params'])
                            save(stage_path,stages)
                        anchor=stages['same_protocol_anchor'];cid=sr['selected_candidate_id']
                        if 'tuned' not in stages:
                            if cid==0:
                                tuned=copy.deepcopy(anchor);tuned['strategy']='tuned';tuned['physical_reuse_anchor']=True
                                tuned['threshold_fit_seconds']=tuned['final_fit_seconds']=0.
                                tuned['threshold_model_fit_seconds']=tuned['final_model_fit_seconds']=0.
                                for row in tuned['metrics']:row['strategy']='tuned'
                            else:
                                tuned=core.strategy_fit(out,folder,name,arm,seed,fold,'tuned',cid,sr['selected_rounds'],
                                    X,y,identity,arrays,design[name][cid]['params'])
                            stages['tuned']=tuned;save(stage_path,stages)
                        srpath=folder/'search'/stem/'search_result.json'
                        artifacts={rel(split):sha(split),rel(srpath):sha(srpath),rel(stage_path):sha(stage_path)}
                        if extra_reg:artifacts[extra_reg['path']]=extra_reg['sha256']
                        for stage in stages.values():artifacts.update(stage['artifact_sha256'])
                        done={'dataset':ds,'model':name,'feature_set':arm,'seed':seed,'fold':fold,
                              'n_features':len(columns),'feature_columns':columns,'protocol_sha256':sha(out/'protocol.json'),
                              'input_registration':extra_reg,'search_result_path':rel(srpath),'split_path':rel(split),
                              'strategies':stages,'logical_candidate_evaluations':36,'physical_search_fits':sr['physical_search_fits'],
                              'logical_strategy_models':4,'physical_strategy_models':2 if cid==0 else 4,
                              'runtime_seconds':time.monotonic()-start,'artifact_sha256':artifacts}
                        save(path,done);new_settings+=1
                        print(f"{now()} DONE {ds} {stem} candidate={cid} AP={stages['tuned']['metrics'][0]['ap']:.6f} "
                              f"anchor={anchor['metrics'][0]['ap']:.6f} elapsed={done['runtime_seconds']:.1f}s",flush=True)
                    manifest['completed_outer_settings']=sum(1 for a in p['feature_arm_priority']
                        for s in range(5) for f in range(5) if (folder/'folds'/f'{name}_{a}_seed{s}_fold{f}.json').exists())
                    manifest['updated_at']=now();save(manifest_path,manifest)
                    if max_settings is not None and new_settings>=max_settings:
                        manifest.update(status='requested_chunk_complete',completed_at=now(),pid=None)
                        save(manifest_path,manifest);return
            del X;gc.collect()
        completed=manifest['completed_outer_settings']
        manifest.update(status='complete' if completed==manifest['expected_outer_settings'] else 'ready_input_queue_complete',
                        completed_at=now(),pid=None)
        save(manifest_path,manifest)
    except BaseException:
        manifest.update(status='failed',failed_at=now(),pid=None,traceback=traceback.format_exc())
        save(manifest_path,manifest);raise
    finally:
        fcntl.flock(lock.fileno(),fcntl.LOCK_UN);lock.close()

def probe(out,name):
    folder=out/'qa';folder.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(31415);X=rng.normal(size=(1000,19)).astype(np.float32)
    y=(X[:,0]+.5*X[:,1]>1.8).astype(np.int8)
    tr=np.arange(700);ev=np.arange(700,850);sc=np.arange(850,1000)
    params=read(DESIGN/'tree_candidates.json')[name][0]['params'].copy()
    params['n_estimators']=200 if name=='random_forest' else 60
    a=core.fit(name,X,y,tr,None if name=='random_forest' else ev,params,19)
    b=core.fit(name,X,y,tr,None if name=='random_forest' else ev,params,19)
    score=core.predict(name,a,X,sc);np.testing.assert_array_equal(score,core.predict(name,b,X,sc))
    model_path=folder/('probe_'+name+'.joblib');core.dump_model(a,model_path)
    restored=core.joblib.load(model_path)
    require(core.model_digest(name,a)==core.model_digest(name,restored),'Model digest not stable')
    np.testing.assert_array_equal(score,core.predict(name,restored,X,sc))
    if name=='random_forest':
        direct=core.fit(name,X,y,tr,None,{**params,'n_estimators':100},19)
        np.testing.assert_array_equal(core.predict(name,core.prefix(a,100),X,sc),core.predict(name,direct,X,sc))
    save(folder/('probe_'+name+'.json'),{'passed':True,'model':name,'trainer_sha256':sha(SCRIPT),
          'exact_repeat_scores':True,'serialization_roundtrip':True,'rf_prefix_direct_equivalence':name=='random_forest',
          'historical_trainer_sha256':sha(core.__file__),'model_sha256':sha(model_path)})
    print(json.dumps(read(folder/('probe_'+name+'.json'))),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run','admit-input','probe','verify-registration'))
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--dataset',choices=DATASETS)
    parser.add_argument('--model',choices=MODELS)
    parser.add_argument('--arms',nargs='+')
    parser.add_argument('--max-settings',type=int)
    parser.add_argument('--kind',choices=('q10','gfp'))
    parser.add_argument('--input',type=Path)
    parser.add_argument('--acceptance',type=Path)
    args=parser.parse_args()
    if args.action=='prepare':prepare(args.out)
    elif args.action=='run':run(args.out,args.dataset,args.model,args.arms,args.max_settings)
    elif args.action=='admit-input':admit_input(args.out,args.dataset,args.kind,args.input,args.acceptance)
    elif args.action=='probe':probe(args.out,args.model)
    else:
        p=registered(args.out);check_artifacts(read(args.out/'source_manifest.json')['source_sha256'])
        print(json.dumps({'passed':True,'protocol_sha256':sha(args.out/'protocol.json'),'trainer_sha256':sha(SCRIPT)}))
