"""Prepare/validate an account-level experiment DESIGN. No training entry point."""
from pathlib import Path
import argparse,csv,hashlib,importlib.metadata as metadata,json,platform
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[2]
ALT=ROOT/'Altman_experiments'
OUT=ALT/'results/eswa_supplement_design_20261013'
MAIN=ALT/'results/account_gmkep56_tuned_20261008'
PAPER=ROOT/'PAPER_ESWA_FLOWTOPO_REVISED_20261011.md'
DESIGN=ROOT/'ESWA_SUPPLEMENT_EXPERIMENT_DESIGN_20261013.md'
DATASETS=('hi_small','li_small')
MODELS=('lightgbm','xgboost','random_forest')
SCRIPT=Path(__file__).resolve()

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def rel(path):return path.relative_to(ROOT).as_posix()
def read(path):return json.loads(path.read_text())
def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def require(ok,msg):
    if not ok:raise ValueError(msg)

def prepare():
    require(not (OUT/'design_manifest.json').exists(),'Design already sealed; use validate or a new version.')
    require(DESIGN.is_file(),'Write the reviewable design before freezing.')
    p=read(MAIN/'protocol.json');di=read(MAIN/'dataset_inputs.json');sm=read(MAIN/'split_manifest.json')
    completed=read(MAIN/'run_manifest.json');passed=read(MAIN/'validation_models.json')
    protocol_sha=sha(MAIN/'protocol.json')
    require(completed['status']=='complete' and completed['completed_outer_settings']==450
            and completed['independent_validation_completed'] is True,'Main experiment not complete.')
    require(passed['passed'] is True and passed['complete_experiment'] is True
            and passed['outer_settings_verified']==450,'Main independent validation not complete.')
    require(completed['protocol_sha256']==passed['protocol_sha256']==protocol_sha,'Main protocol binding changed.')
    require(completed['validation_models_sha256']==sha(MAIN/'validation_models.json'),'Main validation binding changed.')
    for key,name in [('candidates_sha256','candidates.json'),('dataset_inputs_sha256','dataset_inputs.json'),
                     ('split_manifest_sha256','split_manifest.json')]:
        require(p[key]==sha(MAIN/name),'Registered main source changed: '+name)
    require(p['trainer_sha256']==passed['trainer_sha256']==sha(ALT/'code/run_account_gmkep_tuning.py'),'Main trainer changed.')
    require(p['helper_sha256']==passed['helper_sha256']==sha(ALT/'code/run_account_feature_mechanism.py'),'Main helper changed.')
    full=p['feature_columns']['gmkep56'];base=full[:16];g=full[16:32];m=full[32:37];k=full[37:40];e=full[40:44];ph=full[44:]
    h0=[x for x in ph if '_h0_' in x];h1=[x for x in ph if '_h1_' in x]
    require([len(x) for x in [base,g,m,k,e,h0,h1]]==[16,16,5,3,4,6,6],'Unexpected modules.')
    q2=['q_n2','q_n']
    q10=[f'q{h}_{x}' for h in [2,3] for x in ['n','e1','t1','e2','t2']]
    def drop(columns):return [x for x in full if x not in columns]
    arms=[]
    def arm(name,cols,family,status='planned',gate=None):
        arms.append(dict(arm_id=name,columns=cols,n_features=None if cols is None else len(cols),
                         family=family,status=status,models=list(MODELS),datasets=list(DATASETS),
                         execution_gate=gate))
    for name in ['base16','gmke44','gmkep56']:arm(name,p['feature_columns'][name],'reference','reuse_completed_verified')
    for name,cols in [('base_g32',base+g),('base_m21',base+m),('base_k19',base+k),
                      ('base_e20',base+e),('base_p28',base+ph)]:
        arm(name,cols,'module_add','planned')
    for name,cols in [('full_drop_ge36',drop(g+e)),('full_drop_m51',drop(m)),
                      ('full_drop_k53',drop(k)),('full_drop_e52',drop(e))]:
        arm(name,cols,'module_delete_final56')
    struct=full[:44]
    for name,cols in [('struct_h0_50',struct+h0),('struct_h1_50',struct+h1),
                      ('struct_n46',struct+q2),('struct_n_h1_52',struct+q2+h1)]:
        arm(name,cols,'ph_size')
    for name,cols in [('struct_q10_54',struct+q10),('struct_q10_h1_60',struct+q10+h1)]:
        arm(name,cols,'ph_combinatorial_counts',gate='new_Q10_same_geometry_cache_validated')
    arm('base_gfp_account_stream',None,'external_gfp',gate='official_snapml_supported_platform_native_schema_and_stream_adaptation_validated')
    # These are distinct anchors, excluded from the headline core budget.
    optional=[('struct_drop_ge24',[x for x in struct if x not in g+e]),
              ('struct_drop_m39',[x for x in struct if x not in m]),
              ('struct_drop_k41',[x for x in struct if x not in k]),
              ('struct_drop_e40',[x for x in struct if x not in e]),
              ('struct_drop_g_keep_e28',[x for x in struct if x not in g]),
              ('full_drop_g_keep_e40',drop(g))]
    for name,cols in optional:arm(name,cols,'optional_structure_anchor','optional_not_scheduled')
    neural=[dict(arm_id=name,n_features=16,columns=base,family='external_neural',
                 status='planned',models=[model],datasets=list(DATASETS),
                 execution_gate=('seed_loss_scaler_validated' if model=='mlp' else 'seed_node_loss_scaler_and_neighborhood_backend_validated'))
            for name,model in [('mlp_base16','mlp'),('sage_weak_base16','graphsage_weak'),
                               ('sage_directed_base16','graphsage_directed')]]
    module_contrasts=[]
    for block,new in [('G','base_g32'),('M','base_m21'),('K','base_k19'),('E','base_e20'),('P','base_p28')]:
        module_contrasts.append(dict(contrast_id='add_'+block.lower(),lhs=new,rhs='base16',family='module',label=block+'单独加入T'))
    for block,new in [('GE','full_drop_ge36'),('M','full_drop_m51'),('K','full_drop_k53'),('E','full_drop_e52'),('P','gmke44')]:
        module_contrasts.append(dict(contrast_id='delete_'+block.lower()+'_from56',lhs='gmkep56',rhs=new,family='module',label='完整56中'+block+'闭包／列组增量'))
    ph_contrasts=[dict(contrast_id=cid,lhs=lhs,rhs=rhs,family='ph_size') for cid,lhs,rhs in [
        ('h0_vs_struct','struct_h0_50','gmke44'),('h1_vs_struct','struct_h1_50','gmke44'),
        ('h1_given_identical_h0','gmkep56','struct_h0_50'),
        ('h1_given_raw_sample_counts','struct_n_h1_52','struct_n46'),
        ('h1_given_twohop_and_threehop_simplex_counts','struct_q10_h1_60','struct_q10_54')]]
    ph_secondary=[dict(contrast_id=cid,lhs=lhs,rhs=rhs,family='ph_encoding_secondary') for cid,lhs,rhs in [
        ('raw_sample_counts_vs_struct','struct_n46','gmke44'),
        ('h0_encoding_vs_two_raw_counts','struct_h0_50','struct_n46'),
        ('q10_vs_struct','struct_q10_54','gmke44')]]
    external=[dict(contrast_id=cid,lhs=lhs,rhs=rhs,family='external_same_tree') for cid,lhs,rhs in [
        ('gfp_vs_base','base_gfp_account_stream','base16'),
        ('flowtopo44_vs_gfp','gmke44','base_gfp_account_stream'),
        ('flowtopo56_vs_gfp','gmkep56','base_gfp_account_stream')]]
    registry={'status':'design_only_not_executed','authoritative_core_scope':'root design; agent audits also describe alternative smaller staged scopes',
              'modules':dict(T=base,G=g,M=m,K=k,E=e,H0=h0,H1=h1),
              'tree_arms':arms,'neural_arms':neural,
              'module_contrasts':module_contrasts,'ph_primary_contrasts':ph_contrasts,
              'ph_secondary_contrasts':ph_secondary,'external_tree_contrasts':external,
              'external_neural_contrasts':[
                  {'lhs':'sage_weak_base16','rhs':'mlp_base16'},
                  {'lhs':'sage_directed_base16','rhs':'mlp_base16'},
                  {'lhs':'sage_directed_base16','rhs':'sage_weak_base16'},
                  *[{'lhs':lhs,'rhs':rhs,'lhs_model':'xgboost'} for lhs in ['gmke44','gmkep56']
                    for rhs in ['sage_weak_base16','sage_directed_base16']]],
              'Q10_column_source':{'q2_n':'existing q_n2','q3_n':'existing q_n',
                                   'q3_e1':'existing q_e1','q3_t1':'existing q_t1','q3_e2':'existing q_e2','q3_t2':'existing q_t2',
                                   'q2_e1':'new count on cached S2','q2_t1':'new count on cached S2','q2_e2':'new count on cached VR2(S2)','q2_t2':'new count on cached VR2(S2)'},
              'graph32_pause_policy':'Do not resume old batch or reuse partial metrics. base_g32 is a new independently registered proposed arm.'}
    for row in arms+neural:
        if row['columns'] is not None:
            require(len(set(row['columns']))==len(row['columns']),'Duplicate model columns: '+row['arm_id'])
            if row['arm_id'][-2:].isdigit():require(row['n_features']==int(row['arm_id'][-2:]),'Dimension mismatch.')
    save(OUT/'feature_registry.json',registry)
    # Actual existing tree candidates are copied unchanged, never redrawn.
    (OUT/'tree_candidates.json').write_bytes((MAIN/'candidates.json').read_bytes())
    nn=[]
    for width in [64,128,256]:
        for lr in [0.001,0.003]:
            for dropout in [0.0,0.2]:
                nn.append(dict(candidate_id=len(nn),params=dict(hidden_dim=width,learning_rate=lr,dropout=dropout)))
    save(OUT/'neural_candidates.json',{'status':'proposed_pre_execution','shared_by':['mlp','graphsage_weak','graphsage_directed'],'candidates':nn,
        'fixed':dict(layers=2,optimizer='Adam',weight_decay=1e-4,max_epochs=200,patience=30,seed_batch_size=2048,
                     weak_fanouts=[15,10],directed_fanouts_per_relation=[10,5],replace=False,
                     inference='deterministic layerwise all-neighbor; sampling fallback is a new protocol version, not a per-arm exception')})
    save(OUT/'protocol_design.json',{'status':'design_only_not_executed','client_date':'2026-10-13','inherited_protocol':rel(MAIN/'protocol.json'),
        'account_identity':'Exact account_order and labels, complete account population, isolated accounts retained.',
        'evaluation':'Static full-observation transductive account classification. No transaction-label graph sampling.',
        'seeds':p['seeds'],'outer_n_splits':5,'inner_n_splits':3,'candidates_per_model':12,
        'split_manifest_source':rel(MAIN/'split_manifest.json'),'tree_candidates_source_sha256':sha(MAIN/'candidates.json'),
        'selection':p['selection_metric'],'candidate_tie_break':p['candidate_tie_break'],
        'development_threshold_split':p['development_threshold_split'],'early_stopping':p['early_stopping'],
        'threshold_selection':p['threshold_selection'],'rounds_or_epochs':'integer median across selected three inner fits',
        'tree_model_state_formula':p['stage_model_random_state'],'class_weighting':p['class_weighting'],
        'gnn_weighting':'Global balanced weights computed from fit labels; seed-only BCE, minibatch mean weighted loss.',
        'gnn_preprocessing':'Nonnegative BASE magnitude log1p, then standardization; fit scaler on current fit subset only. Identity and labels are never model inputs.',
        'gnn_log1p_columns':['n_sent','vol_sent','avg_sent','std_sent','max_sent','n_recv_banks','n_recv','vol_recv','avg_recv','std_recv','max_recv','n_sent_banks','n_currencies','n_formats','sent_recv_ratio','degree'],
        'metrics_primary':['ap'],'metrics_auxiliary':['auc','f1','precision','recall','balanced_accuracy','accuracy'],
        'threshold_policies':['inner_max_f1','fixed_0.5'],'alarm_budgets':[0.005,0.01,0.02],
        'alarm_ties':'score descending then source account row ascending; same k rounding as existing supplementary code',
        'inference_primary':'mean paired increments, sample SD(ddof=1), positive folds and complete seed OOF consistency',
        'optional_p_values':'Approximate corrected-CV/Holm sensitivity only; freeze new families and acknowledge residual graph dependence; do not convert to independent confirmation.',
        'geometry':{'source':'account_s10_increment_20261008 existing samples','cap':40,'sampling':'path-preserving shared BFS prefix; lexicographic ID within layer',
                    'H0':'same exact original six encoded columns, not independent PH shape'},
        'gfp':{'method':'official Snap ML GFP with documented account adaptation','native_schema_dim':None,'account_dim_formula':'16+6q',
               'input':'float64 [unique transaction_id, source_id, target_id, seconds_timestamp, amount_paid, amount_received]',
               'window_seconds':604800,'pattern_bins':[2,4,8,16,32,64],'cycle_bins':[2,3,4,5,6],'cycle_max_length':6,
               'max_no_edges':-1,'vertex_stats_cols':[4,5],'vertex_stats_feats':list(range(11)),
               'account_aggregates':['out_sum','out_mean','out_max','in_sum','in_mean','in_max'],
               'undefined_engineered_statistics':'nonfinite native engineered values -> 0, count by column; never drop accounts or impute indexing fields',
               'time_ties':'timestamp group; stable original_row_id order; explicit partial_fit(group) then query same IDs only after native duplicate-ID and tie-batch tests pass',
               'freeze_before_training':['package wheel hash/version','example commit or permalink','get_params full dict','native engineered column schema','same-time and batch semantics','label-blind identity and finite values'],
               'not_equivalent':'Native temporal directed multigraph GFP vs static simple-graph FlowTopo; report this input-view difference.'},
        'existing_graph32_state':'remains paused_by_user; no old partial setting is promoted to complete.',
        'new_runners':'not implemented by this design generator; implementation and resource gates precede any training.'})
    source_paths={PAPER,DESIGN,SCRIPT,MAIN/'protocol.json',MAIN/'candidates.json',MAIN/'dataset_inputs.json',MAIN/'split_manifest.json',
        MAIN/'validation_models.json',MAIN/'summary.csv',MAIN/'fold_metrics.csv',MAIN/'seed_oof_metrics.csv',MAIN/'selection_records.csv',
                  MAIN/'run_manifest.json',
                  ALT/'results/account_graph32_completion_20261009/run_manifest.json',
                  ALT/'code/run_account_gmkep_tuning.py',ALT/'code/run_account_gmkep_multiclassifier.py',
                  ALT/'code/run_account_feature_mechanism.py',ALT/'code/build_account_s10_features.py',
                  ALT/'code/account_s10_math.py',ALT/'code/eswa_supplementary_analysis_20261012.py',
                  ALT/'results/account_s10_increment_20261008/feature_protocol.json'}
    schema_report={}
    for ds in DATASETS:
        item=di[ds]
        cache=MAIN/ds/'arrays/features.npy';y_path=MAIN/ds/'arrays/labels.npy';accounts=MAIN/ds/'accounts.parquet'
        for name,expected in item['cache_sha256'].items():
            path=ROOT/name;require(sha(path)==expected,'Main cache changed: '+name);source_paths.add(path)
        feat=ROOT/item['source_features'];require(sha(feat)==item['source_sha256'][rel(feat)],'PH cache changed.')
        source_paths.add(feat)
        x=np.load(cache,mmap_mode='r');y=np.load(y_path,mmap_mode='r')
        require(x.shape==(item['n_samples'],56) and x.dtype==np.float32,'Main matrix shape/dtype mismatch.')
        require(np.isfinite(x[:,:16]).all() and (x[:,:16]>=0).all(),'BASE16 log1p assumptions differ.')
        require(y.shape==(item['n_samples'],) and int(y.sum())==item['n_positive'],'Labels differ.')
        names=pq.read_schema(feat).names
        require(set(full+q2).issubset(names),'Required existing columns missing.')
        schema_report[ds]=dict(n_accounts=len(y),n_positive=int(y.sum()),model_dtype=str(x.dtype),label_dtype=str(y.dtype),
                               cached_columns=len(names),main_features=56,planned_Q10_new_columns=4,GFP_dimension='pending native schema')
        folder=ALT/'results/account_s10_increment_20261008'/ds
        source_paths.update(folder/name for name in ['graph_csr.npz','graph_accounts.parquet','account_to_graph.npy','graph_manifest.json','feature_manifest.json','feature_qc.json','geometry_qc.parquet'])
        graph_manifest=read(folder/'graph_manifest.json');feature_manifest=read(folder/'feature_manifest.json')
        for local,expected in graph_manifest['artifact_sha256'].items():
            require(sha(folder/local)==expected,'Cached graph changed: '+ds+'/'+local)
        for local,expected in feature_manifest['artifacts_sha256'].items():
            require(sha(folder/local)==expected,'Cached geometry/feature changed: '+ds+'/'+local)
        chunks=sorted((folder/'chunks').glob('*.npz'));next_row=0
        for chunk in chunks:
            receipt=chunk.with_suffix('.json');record=read(receipt)
            require(record['start']==next_row and record['stop']>next_row,'Sample chunk gap/overlap: '+rel(chunk))
            require(sha(chunk)==record['sha256'],'Sample chunk changed: '+rel(chunk))
            next_row=record['stop'];source_paths.update([chunk,receipt])
        require(next_row==item['n_samples'],'Sample chunks do not cover all accounts.')
    for name,binding in sm.items():
        path=ROOT/name;require(sha(path)==binding['sha256'],'Split changed: '+name);source_paths.add(path)
    require(len(sm)==50,'Need all 50 matched dataset/split files.')
    for name,expected in completed['science_csv_sha256'].items():
        path=MAIN/name;require(sha(path)==expected,'Completed scientific CSV changed: '+name);source_paths.add(path)
    source_paths.update(path for path in OUT.iterdir() if path.suffix in {'.md','.json'} and path.name not in {
        'feature_registry.json','tree_candidates.json','neural_candidates.json','protocol_design.json','source_manifest.json',
        'preflight_report.json','design_manifest.json','validation.json'})
    bindings={rel(path):sha(path) for path in sorted(source_paths)}
    save(OUT/'source_manifest.json',{'status':'read_only_design_binding','source_sha256':bindings,'model_probabilities_rescored':False})
    run=read(ALT/'results/account_graph32_completion_20261009/run_manifest.json')
    require(run['status']=='paused_by_user' and run['tuning_completed']==12 and run.get('live_pid') is None,'Old Graph32 pause differs.')
    env={}
    for key in ['numpy','pandas','pyarrow','lightgbm','xgboost','scikit-learn','torch','torch-geometric','pyg-lib','torch-sparse','snapml']:
        try:env[key]=metadata.version(key)
        except metadata.PackageNotFoundError:env[key]=None
    save(OUT/'preflight_report.json',{'passed_for_design':True,'training_ready':False,'training_run':False,
        'python':platform.python_version(),'platform':platform.platform(),'dependencies':env,'dataset_schema':schema_report,
        'original_paused_graph32':{'status':run['status'],'tuning_completed':run['tuning_completed'],'live_pid':run.get('live_pid')},
        'execution_blockers':['Official GFP wheel/native schema/account adapter unvalidated on supported platform.',
                              'GNN node adapters and accelerated neighborhood backend not yet validated.',
                              'Q10 extra 2-hop count cache not yet extracted or independently checked.',
                              'Generalized supplement training and independent result validator not implemented.'],
        'existing_inputs_verified':True,'cache_split_source_hashes_verified':True,
        'hardware_memory':'Not determined: sandbox sysctl read denied; no memory or completion-time promise.'})
    jobs=[]
    all_core=[a for a in arms if a['status']=='planned']+neural
    for a in all_core:
        for ds in DATASETS:
            for model in a['models']:
                for seed in p['seeds']:
                    for fold in range(5):
                        sp=MAIN/ds/'splits'/f'seed{seed}_fold{fold}.npz'
                        jobs.append(dict(dataset=ds,model=model,arm_id=a['arm_id'],seed=seed,fold=fold,
                                         split_path=rel(sp),split_sha256=sm[rel(sp)]['sha256'],
                                         status='planned_not_run',execution_gate=a['execution_gate']))
    require(len(jobs)==2550,'Core job count drift.')
    (OUT/'run_matrix.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in jobs))
    with (OUT/'experiment_matrix.csv').open('w',newline='') as stream:
        fields=['arm_id','family','n_features','status','n_dataset_model_settings','outer_jobs','logical_candidate_evaluations','execution_gate']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for a in arms+neural:
            settings=2*len(a['models']);n=25*settings if a['status']=='planned' else 0
            writer.writerow(dict(arm_id=a['arm_id'],family=a['family'],n_features=a['n_features'],status=a['status'],
                                 n_dataset_model_settings=settings,outer_jobs=n,logical_candidate_evaluations=n*36,execution_gate=a['execution_gate']))
    output_paths=[OUT/name for name in ['feature_registry.json','tree_candidates.json','neural_candidates.json','protocol_design.json',
                                       'source_manifest.json','preflight_report.json','run_matrix.jsonl','experiment_matrix.csv']]
    save(OUT/'design_manifest.json',{'status':'design_only_not_executed','client_date':'2026-10-13','prepared_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'main_design_document':rel(DESIGN),'main_design_sha256':sha(DESIGN),'builder_sha256':sha(SCRIPT),
        'source_manifest_sha256':sha(OUT/'source_manifest.json'),'output_sha256':{rel(path):sha(path) for path in output_paths},
        'core_new_tree_arms':16,'core_neural_arms':3,'new_dataset_model_settings':102,
        'new_outer_setting_jobs':2550,'logical_candidate_inner_evaluations':91800,'reused_dataset_model_settings':18,
        'counts_are':'proposed candidate scores and final-setting jobs; not completed models, physical fits or runtime',
        'optional_tree_arms_not_scheduled':6,'old_graph32_resumed':False,'training_run':False,'dependency_installation':False,
        'new_feature_extraction':False,'new_model_scoring':False})
    return validate()

def validate():
    manifest=read(OUT/'design_manifest.json');sources=read(OUT/'source_manifest.json')
    for name,expected in sources['source_sha256'].items():require(sha(ROOT/name)==expected,'Bound source changed: '+name)
    for name,expected in manifest['output_sha256'].items():require(sha(ROOT/name)==expected,'Design output changed: '+name)
    require(sha(SCRIPT)==manifest['builder_sha256'] and sha(DESIGN)==manifest['main_design_sha256'],'Builder/design changed.')
    jobs=[json.loads(row) for row in (OUT/'run_matrix.jsonl').read_text().splitlines()]
    keys={(r['dataset'],r['model'],r['arm_id'],r['seed'],r['fold']) for r in jobs}
    require(len(jobs)==len(keys)==2550 and all(r['status']=='planned_not_run' for r in jobs),'Invalid job registry.')
    require(sum(r['model'] in MODELS for r in jobs)==2400,'Wrong tree job count.')
    require(sum(r['model'] not in MODELS for r in jobs)==150,'Wrong neural job count.')
    require(sha(OUT/'tree_candidates.json')==sha(MAIN/'candidates.json'),'Tree candidates were altered.')
    registry=read(OUT/'feature_registry.json')
    trees=read(OUT/'tree_candidates.json');nc=read(OUT/'neural_candidates.json')['candidates']
    require(set(trees)==set(MODELS) and all(len(trees[name])==12 for name in MODELS),'Tree candidate count differs.')
    require(all([row['candidate_id'] for row in trees[name]]==list(range(12)) for name in MODELS),'Tree candidate IDs differ.')
    require(len(nc)==12 and [row['candidate_id'] for row in nc]==list(range(12)),'Neural candidates differ.')
    require(len({json.dumps(row['params'],sort_keys=True) for row in nc})==12,'Neural candidate duplicates.')
    all_arms=registry['tree_arms']+registry['neural_arms']
    require(len({a['arm_id'] for a in all_arms})==len(all_arms),'Arm IDs duplicate.')
    for a in all_arms:
        if a['columns'] is not None:
            require(a['n_features']==len(a['columns'])==len(set(a['columns'])),'Dimension/uniqueness differs: '+a['arm_id'])
            if a['arm_id'][-2:].isdigit():require(a['n_features']==int(a['arm_id'][-2:]),'Named dimension differs.')
    expected_keys={(ds,model,a['arm_id'],seed,fold) for a in all_arms if a['status']=='planned'
                   for ds in DATASETS for model in a['models'] for seed in range(5) for fold in range(5)}
    require(keys==expected_keys,'Run matrix differs from full registered Cartesian product.')
    columns={a['arm_id']:a['columns'] for a in registry['tree_arms']}
    require(not set(registry['modules']['G']+registry['modules']['E'])&set(columns['full_drop_ge36']),'G/E closure not removed.')
    require(columns['struct_h0_50']==columns['gmke44']+registry['modules']['H0'],'H0 matching encoding differs.')
    require(columns['struct_q10_h1_60']==columns['struct_q10_54']+registry['modules']['H1'],'Q10 context differs.')
    report={'passed':True,'status':'design_only_not_executed','source_count':len(sources['source_sha256']),
            'new_outer_jobs':len(jobs),'logical_candidate_inner_evaluations':len(jobs)*36,
            'candidate_lists_and_dimensions_verified':True,'source_hashes_verified':True,'original_graph32_still_paused':True,
            'main_paper_unchanged':True,'training_run':False,'new_feature_extraction':False,'new_model_scoring':False,
            'manifest_sha256':sha(OUT/'design_manifest.json')}
    save(OUT/'validation.json',report)
    print(json.dumps(report,ensure_ascii=False))
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['prepare','validate'])
    args=parser.parse_args()
    prepare() if args.action=='prepare' else validate()
