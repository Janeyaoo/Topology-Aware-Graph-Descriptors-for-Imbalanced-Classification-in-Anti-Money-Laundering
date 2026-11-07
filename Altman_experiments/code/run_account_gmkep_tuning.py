"""Source-bound nested tuning for all IBM Small 16/44/56-column settings.

Search scores are retained as exact rank/tie audits. Only selected representative,
threshold and final models are retained, within the local disk budget.
"""
from __future__ import annotations
import argparse, copy, gc, hashlib, importlib.metadata, itertools, json, math, os
import platform, shutil, subprocess, sys, time, traceback
import tempfile
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_sample_weight
import run_account_feature_mechanism as common

ROOT=Path(__file__).resolve().parents[2]
CODE=ROOT/"Altman_experiments/code"
OLD=ROOT/"Altman_experiments/results/account_gmkep56_multiclassifier_20261008"
SOURCE=ROOT/"Altman_experiments/results/account_s10_increment_20261008"
OUT=ROOT/"Altman_experiments/results/account_gmkep56_tuned_20261008"
DATASETS=("hi_small","li_small")
MODELS=("lightgbm","xgboost","random_forest")
ARMS=("base16","gmke44","gmkep56")
STRATEGIES=("same_protocol_anchor","tuned")
POLICIES=("fixed_0.5","inner_max_f1")
sha, rel, jsave, now=common.sha256,common.rel,common.write_json,common.now

def read(path): return json.loads(Path(path).read_text())
def array_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def guard_disk(path):
    free=shutil.disk_usage(path).free
    if free < 1.5*(1<<30):
        raise RuntimeError(f"Disk guard: {free/(1<<30):.2f} GiB free; no old artifacts deleted")

def dump_model(model,path):
    guard_disk(path.parent)
    temp=path.with_suffix(".joblib.tmp")
    joblib.dump(model,temp,compress=("xz",3))
    temp.replace(path)

def effective(params,random_state,rounds=None):
    p=dict(params); p["random_state"]=int(random_state)
    if rounds is not None: p["n_estimators"]=int(rounds)
    return p

def fit(name,X,y,rows,early,params,random_state,rounds=None):
    p=effective(params,random_state,rounds)
    if name=="lightgbm":
        import lightgbm as lgb
        model=lgb.LGBMClassifier(**p)
        kw={} if early is None else {
            "eval_set":[(X[early],y[early])],
            "callbacks":[lgb.early_stopping(30,verbose=False),lgb.log_evaluation(-1)]}
        model.fit(X[rows],y[rows],**kw)
    elif name=="xgboost":
        import xgboost as xgb
        if early is not None: p["early_stopping_rounds"]=30
        model=xgb.XGBClassifier(**p)
        kw={"sample_weight":compute_sample_weight("balanced",y[rows])}
        if early is not None: kw.update(eval_set=[(X[early],y[early])],verbose=False)
        model.fit(X[rows],y[rows],**kw)
    else:
        model=RandomForestClassifier(**p)
        model.fit(X[rows],y[rows])
    return model

def rounds_for(name,model,params):
    if name=="lightgbm": return int(model.best_iteration_ or params["n_estimators"])
    if name=="xgboost": return int(model.best_iteration+1)
    return int(model.n_estimators)

def predict(name,model,X,rows):
    # Fixed tree accumulation order keeps exact score/tie bytes reproducible.
    jobs=model.n_jobs
    if name=="random_forest": model.n_jobs=1
    try: score=np.ascontiguousarray(model.predict_proba(X[rows])[:,1],dtype=np.float64)
    finally: model.n_jobs=jobs
    assert np.isfinite(score).all() and ((score>=0)&(score<=1)).all()
    return score

def prefix(model,n):
    result=copy.copy(model)
    result.n_estimators=int(n)
    result.estimators_=model.estimators_[:n]
    return result

def model_digest(name,model):
    h=hashlib.sha256()
    h.update(json.dumps(model.get_params(),sort_keys=True,default=str).encode())
    if name=="lightgbm": h.update(model.booster_.model_to_string(num_iteration=-1).encode())
    elif name=="xgboost": h.update(bytes(model.get_booster().save_raw(raw_format="ubj")))
    else:
        for tree in model.estimators_:
            state=tree.tree_.__getstate__()
            # Structured records have uninitialised alignment padding; hash
            # actual fields so serialisation roundtrips preserve the digest.
            for field in state["nodes"].dtype.names:
                h.update(field.encode())
                h.update(np.ascontiguousarray(state["nodes"][field]).tobytes())
            h.update(np.ascontiguousarray(state["values"]).tobytes())
    return h.hexdigest()

def audit_save(path,y,score,rows):
    n=len(score); order=np.argsort(-score,kind="mergesort").astype(np.uint32)
    width=max(1,(n-1).bit_length())
    bits=((order[:,None]>>np.arange(width,dtype=np.uint32))&1).astype(np.uint8)
    ends=np.r_[score[order][:-1]!=score[order][1:],True]
    temp=path.with_suffix(".npz.tmp")
    with temp.open("wb") as f:
        np.savez_compressed(f,n=np.int64(n),n_positive=np.int64(y.sum()),width=np.int64(width),
            packed_order=np.packbits(bits.ravel(),bitorder="little"),
            packed_labels=np.packbits(y[order],bitorder="little"),
            packed_tie_end=np.packbits(ends,bitorder="little"),
            raw_score_sha256=np.array(array_sha(score)),score_rows_sha256=np.array(array_sha(rows)))
    temp.replace(path)

def candidates(old):
    answer={}
    spaces={
        "lightgbm":{"learning_rate":[.03,.05,.1],"num_leaves":[15,31,63,127],
            "min_child_samples":[10,20,50,100],"reg_alpha":[0.,.1,1.],
            "reg_lambda":[1.,5.,10.],"colsample_bytree":[.6,.8,1.],
            "subsample":[.7,.8,1.],"subsample_freq":[1],"n_estimators":[300,500,700]},
        "xgboost":{"learning_rate":[.03,.05,.1],"max_depth":[3,4,6,8],
            "min_child_weight":[1,5,10],"reg_alpha":[0.,.1,1.],
            "reg_lambda":[1.,5.,10.],"colsample_bytree":[.6,.8,1.],
            "subsample":[.7,.8,1.],"n_estimators":[300,500,700]},
    }
    for k,name in enumerate(("lightgbm","xgboost")):
        base=dict(old["model_params"][name]); base["n_jobs"]=2
        if name=="lightgbm":
            base.update(subsample_freq=0,deterministic=True,force_col_wise=True)
        rows=[base]; seen={json.dumps(base,sort_keys=True)}
        rng=np.random.default_rng(20261008+k)
        while len(rows)<12:
            p=dict(base)
            for key,values in spaces[name].items():
                value=values[int(rng.integers(len(values)))]
                p[key]=value
            signature=json.dumps(p,sort_keys=True)
            if signature not in seen: rows.append(p); seen.add(signature)
        answer[name]=[{"candidate_id":i,"params":p} for i,p in enumerate(rows)]
    rf=dict(old["model_params"]["random_forest"]); rf["n_jobs"]=2
    profiles=[
        {},
        {"max_depth":10,"min_samples_leaf":10,"max_features":.5},
        {"max_depth":20,"min_samples_leaf":2},
        {"max_depth":None,"min_samples_leaf":10},
        {"max_depth":14,"min_samples_leaf":5,"max_features":.5,"max_samples":.6},
        {"max_depth":20,"min_samples_leaf":5,"max_features":.25,"max_samples":1.},
    ]
    rfs=[]
    for profile_id,update in enumerate(profiles):
        for trees in ((200,100) if profile_id==0 else (100,200)):
            p={**rf,**update,"n_estimators":trees}
            rfs.append({"candidate_id":len(rfs),"profile_id":profile_id,"params":p})
    answer["random_forest"]=rfs
    return answer,spaces,profiles

def prepare(out):
    assert not out.exists(),"New experiment directory required"
    import report_account_gmkep_multiclassifier as previous
    old=previous.load()
    design,spaces,rfprofiles=candidates(old.protocol)
    out.mkdir()
    (out/"snapshots").mkdir()
    for name in ("PAPER_FRAMEWORK_IBM_AML_SMALL_GMKEP56_20261008.md","EXPERIMENT_RESULTS_SUMMARY.md",
                 "EXPERIMENT_RESULTS_SUMMARY_gen.py"):
        if (ROOT/name).is_file():
            shutil.copy2(ROOT/name,out/"snapshots"/("before_"+name))
    for path in (Path(__file__),CODE/"paper_framework_gmkep56_template.md",CODE/"build_ibm_small_paper_framework.py"):
        shutil.copy2(path,out/"snapshots"/path.name)
    for name in MODELS:
        probe_path=ROOT/"Altman_experiments/results"/f"gmkep_tuning_probe_{name}.json"
        check=read(probe_path)
        assert check["passed"] and check["trainer_sha256"]==sha(__file__)
        shutil.copy2(probe_path,out/"snapshots"/probe_path.name)
    jsave(out/"snapshot_manifest.json",{rel(path):sha(path) for path in (out/"snapshots").iterdir()})
    jsave(out/"candidates.json",design)
    inputs={}; split_hashes={}
    for ds in DATASETS:
        folder=out/ds; folder.mkdir()
        for sub in ("splits","arrays","search","folds","models","calibration_models","predictions",
                    "calibration","selected_search_models","selected_search_scores"):
            (folder/sub).mkdir()
        src=SOURCE/ds/"features.parquet"
        table=pd.read_parquet(src,columns=["account_id","label",*old.protocol["feature_columns"]["gmkep56"]])
        identity=pd.read_parquet(ROOT/"Altman_experiments/results/account_original_multiclassifier_20261007_isolated"/ds/"accounts.parquet")
        np.testing.assert_array_equal(table.account_id,identity.account_id)
        np.testing.assert_array_equal(table.label,identity.label)
        matrix=np.ascontiguousarray(table[old.protocol["feature_columns"]["gmkep56"]].to_numpy(np.float32))
        y=table.label.to_numpy(np.int8)
        assert np.isfinite(matrix).all()
        np.save(folder/"arrays/features.npy",matrix); np.save(folder/"arrays/labels.npy",y)
        identity.to_parquet(folder/"accounts.parquet",index=False,compression="zstd")
        paths=[src,OLD/"protocol.json",OLD/"validation_models.json",OLD/"summary.csv",OLD/"fold_metrics.csv",
               OLD/"seed_oof_metrics.csv",OLD/"source_settings.json",OLD/"dataset_inputs.json"]
        inputs[ds]={"n_samples":len(y),"n_positive":int(y.sum()),"source_features":rel(src),
                    "source_sha256":{rel(p):sha(p) for p in paths},
                    "cache_sha256":{rel(p):sha(p) for p in (folder/"arrays/features.npy",folder/"arrays/labels.npy",folder/"accounts.parquet")}}
        for seed in range(5):
            for fold in range(5):
                original=OLD/ds/"splits"/f"seed{seed}_fold{fold}.npz"
                with np.load(original) as f:
                    arrays={k:f[k] for k in f.files}
                arrays["development"]=arrays.pop("inner_fit")
                arrays["threshold_validation"]=arrays.pop("inner_calibration")
                D=arrays["development"]
                cv=StratifiedKFold(3,shuffle=True,random_state=20000+100*seed+fold)
                for j,(tr,sc) in enumerate(cv.split(np.zeros(len(D)),y[D])):
                    arrays[f"inner_train_{j}"]=D[tr]
                    arrays[f"inner_score_{j}"]=D[sc]
                    es=StratifiedShuffleSplit(1,test_size=.2,random_state=30000+1000*seed+10*fold+j)
                    ft,ev=next(es.split(np.zeros(len(tr)),y[D[tr]]))
                    arrays[f"early_fit_{j}"]=D[tr][ft]
                    arrays[f"early_validation_{j}"]=D[tr][ev]
                path=folder/"splits"/original.name
                np.savez_compressed(path,**arrays)
                split_hashes[rel(path)]={"sha256":sha(path),"old_outer_split":rel(original),
                    "old_outer_split_sha256":sha(original),"arrays_sha256":{k:array_sha(v) for k,v in arrays.items()}}
        del table,matrix,y,identity; gc.collect()
    jsave(out/"dataset_inputs.json",inputs); jsave(out/"split_manifest.json",split_hashes)
    p={"experiment":"account_gmkep56_nested_tuning","registered_at":now(),"datasets":list(DATASETS),
       "models":list(MODELS),"feature_sets":{"base16":16,"gmke44":44,"gmkep56":56},
       "feature_columns":old.protocol["feature_columns"],"strategies":list(STRATEGIES),
       "seeds":list(range(5)),"outer_n_splits":5,"inner_n_splits":3,"candidates_per_model":12,
       "selection_metric":"unweighted mean of three inner-fold AP values",
       "primary_metrics":["ap"],"candidate_tie_break":"exact tie: lowest candidate_id",
       "threshold_policies":list(POLICIES),"prediction_rule":"y_score >= threshold",
       "threshold_selection":"max positive-class F1 on independent threshold_validation; ties lowest threshold",
       "rounds_aggregation":"integer median of the selected candidate's three inner early-stopping rounds",
       "early_stopping":{"metric":"unweighted AUROC","patience":30,"validation_fraction":.2},
       "development_threshold_split":"reuse old inner_fit and inner_calibration, 80/20 of outer_train",
       "inner_random_state":"20000 + 100 * seed + fold",
       "early_random_state":"30000 + 1000 * seed + 10 * fold + inner_fold",
       "search_model_random_state":"40000 + 1000 * seed + 10 * fold + inner_fold; independent of candidate and arm",
       "stage_model_random_state":"100 * seed + fold",
       "class_weighting":"balanced from actual fitting labels; XGB sample_weight; eval unweighted",
       "threads_per_worker":2,"max_concurrent_workers":6,"native_library_isolation":True,
       "boost_search_spaces":spaces,"rf_profiles":rfprofiles,"rf_tree_candidates":[100,200],
       "rf_prefix_reuse":"fit 200 trees per profile/inner fold; candidate100 uses first100; tested independently",
       "prediction_order":"RF probability accumulation n_jobs=1; fitting n_jobs=2",
       "candidate0_runtime_changes":"n_jobs=2; LGB deterministic=True,force_col_wise=True,subsample_freq=0; structural hyperparameters unchanged",
       "search_storage":"all full-order rank/tie audits; retain selected inner_fold0 model/rawscore; unselected models not retained",
       "audit_encoding":"little-endian packbits: order elements represented least-significant-bit first; width=max(1,ceil(log2(n)))",
       "model_storage":"joblib xz level3; tuned selected0 references anchor physical artifacts",
       "expected_outer_settings":450,"logical_candidate_evaluations":16200,
       "expected_search_physical_fits":13500,"expected_representative_refits":450,
       "expected_strategy_final_models_logical":900,"expected_strategy_threshold_models_logical":900,
       "old_fixed_source":rel(OLD),"old_protocol_sha256":sha(OLD/"protocol.json"),
       "trainer_sha256":sha(__file__),"helper_sha256":sha(common.__file__),
       "candidates_sha256":sha(out/"candidates.json"),"dataset_inputs_sha256":sha(out/"dataset_inputs.json"),
       "split_manifest_sha256":sha(out/"split_manifest.json"),"summary_std_ddof":1,
       "evaluation_scope":old.protocol["evaluation_scope"],
       "interpretation":"finite-budget developmental search; no global optimum, independent confirmation or causal PH claim",
       "environment":{"python":platform.python_version(),"machine":platform.machine(),"cpu_count":os.cpu_count(),
          **{k:importlib.metadata.version(k) for k in ("numpy","pandas","scikit-learn","lightgbm","xgboost","joblib","pyarrow")}}}
    jsave(out/"protocol.json",p)
    jsave(out/"run_manifest.json",{"status":"registered","registered_at":now(),"protocol_sha256":sha(out/"protocol.json"),
          "completed_outer_settings":0,"logical_candidate_evaluations_completed":0})
    print(json.dumps({"prepared":rel(out),"candidate_evaluations":16200,"search_physical_fits":13500}),flush=True)

def registered(out):
    p=read(out/"protocol.json")
    assert p["trainer_sha256"]==sha(__file__) and p["helper_sha256"]==sha(common.__file__)
    for stem in ("candidates","dataset_inputs","split_manifest"):
        assert p[stem+"_sha256"]==sha(out/(stem+".json"))
    return p

def search(out,folder,name,arm,seed,fold,X,y,arrays,p,design):
    stem=f"{name}_{arm}_seed{seed}_fold{fold}"
    directory=folder/"search"/stem; directory.mkdir(exist_ok=True)
    result_path=directory/"search_result.json"
    if result_path.exists():
        result=read(result_path)
        assert result["protocol_sha256"]==sha(out/"protocol.json")
        for path,digest in result["artifact_sha256"].items(): assert sha(ROOT/path)==digest
        return result
    candidates_for_model=design[name]
    groups={}
    for candidate in candidates_for_model:
        key=candidate.get("profile_id",candidate["candidate_id"])
        groups.setdefault(key,[]).append(candidate)
    rows=[]; tick=time.monotonic()
    for group_id,cands in groups.items():
        for j in range(3):
            node=directory/f"group{group_id}_inner{j}.json"
            if node.exists():
                saved=read(node)
                assert saved["protocol_sha256"]==sha(out/"protocol.json")
                for r in saved["records"]: assert sha(ROOT/r["audit_path"])==r["audit_sha256"]
                rows.extend(saved["records"]); continue
            guard_disk(folder)
            fit_rows=arrays[f"inner_train_{j}"] if name=="random_forest" else arrays[f"early_fit_{j}"]
            early=None if name=="random_forest" else arrays[f"early_validation_{j}"]
            score_rows=arrays[f"inner_score_{j}"]
            random_state=40000+1000*seed+10*fold+j
            parent=max(cands,key=lambda c:c["params"]["n_estimators"])
            start=time.monotonic()
            model=fit(name,X,y,fit_rows,early,parent["params"],random_state)
            fit_seconds=time.monotonic()-start
            records=[]
            for c in cands:
                candidate_model=prefix(model,c["params"]["n_estimators"]) if name=="random_forest" else model
                t=time.monotonic(); score=predict(name,candidate_model,X,score_rows)
                predict_seconds=time.monotonic()-t
                audit=directory/f"candidate{c['candidate_id']}_inner{j}_rank.npz"
                t=time.monotonic(); audit_save(audit,y[score_rows],score,score_rows)
                rec={"dataset":folder.name,"model":name,"feature_set":arm,"seed":seed,"fold":fold,
                    "candidate_id":c["candidate_id"],"inner_fold":j,
                    "ap":float(average_precision_score(y[score_rows],score)),
                    "auc":float(roc_auc_score(y[score_rows],score)),
                    "selected_rounds":rounds_for(name,candidate_model,c["params"]),
                    "fit_seconds":fit_seconds if c["candidate_id"]==parent["candidate_id"] else 0.,
                    "shared_fit_seconds":fit_seconds,"predict_seconds":predict_seconds,
                    "audit_seconds":time.monotonic()-t,"physical_fit_id":f"{stem}_group{group_id}_inner{j}",
                    "physical_fit_charged":c["candidate_id"]==parent["candidate_id"],
                    "model_random_state":random_state,"effective_params":effective(c["params"],random_state),
                    "fit_n":len(fit_rows),"fit_positive":int(y[fit_rows].sum()),
                    "score_n":len(score_rows),"score_positive":int(y[score_rows].sum()),
                    "fit_rows_sha256":array_sha(fit_rows),"early_rows_sha256":None if early is None else array_sha(early),
                    "score_rows_sha256":array_sha(score_rows),"raw_score_sha256":array_sha(score),
                    "audit_path":rel(audit),"audit_sha256":sha(audit),"model_digest":model_digest(name,candidate_model)}
                records.append(rec)
            jsave(node,{"protocol_sha256":sha(out/"protocol.json"),"records":records})
            rows.extend(records)
            del model,score,candidate_model; gc.collect()
    assert len(rows)==36
    rows.sort(key=lambda r:(r["candidate_id"],r["inner_fold"]))
    summary=[]
    for c in candidates_for_model:
        records=[r for r in rows if r["candidate_id"]==c["candidate_id"]]
        assert len(records)==3
        summary.append({"candidate_id":c["candidate_id"],"mean_ap":float(np.mean([r["ap"] for r in records])),
                        "std_ap":float(np.std([r["ap"] for r in records],ddof=1)),
                        "rounds":int(np.median([r["selected_rounds"] for r in records]))})
    selected=min(summary,key=lambda r:(-r["mean_ap"],r["candidate_id"]))
    cid=selected["candidate_id"]; representative=[r for r in rows if r["candidate_id"]==cid and r["inner_fold"]==0][0]
    c=candidates_for_model[cid]; j=0
    fit_rows=arrays["inner_train_0"] if name=="random_forest" else arrays["early_fit_0"]
    early=None if name=="random_forest" else arrays["early_validation_0"]
    score_rows=arrays["inner_score_0"]
    t=time.monotonic()
    model=fit(name,X,y,fit_rows,early,c["params"],40000+1000*seed+10*fold)
    score=predict(name,model,X,score_rows)
    assert rounds_for(name,model,c["params"])==representative["selected_rounds"]
    assert array_sha(score)==representative["raw_score_sha256"],"Representative score not exactly reproducible"
    path=folder/"selected_search_models"/f"{stem}.joblib"
    scorepath=folder/"selected_search_scores"/f"{stem}.npz"
    dump_model(model,path)
    np.savez_compressed(scorepath,row_index=score_rows,y_true=y[score_rows],y_score=score)
    refit_seconds=time.monotonic()-t
    result={"dataset":folder.name,"model":name,"feature_set":arm,"seed":seed,"fold":fold,
        "protocol_sha256":sha(out/"protocol.json"),"selected_candidate_id":cid,
        "selected_mean_ap":selected["mean_ap"],"selected_rounds":selected["rounds"],
        "anchor_rounds":summary[0]["rounds"],"candidate_summary":summary,"candidate_records":rows,
        "representative_inner_fold":0,"representative_model_path":rel(path),"representative_score_path":rel(scorepath),
        "representative_refit_seconds":refit_seconds,
        "physical_search_fits":sum(r["physical_fit_charged"] for r in rows),
        "search_fit_seconds":sum(r["fit_seconds"] for r in rows),
        "search_predict_seconds":sum(r["predict_seconds"] for r in rows),
        "search_audit_seconds":sum(r["audit_seconds"] for r in rows),
        "search_elapsed_seconds_this_invocation":time.monotonic()-tick,
        "artifact_sha256":{**{r["audit_path"]:r["audit_sha256"] for r in rows},rel(path):sha(path),rel(scorepath):sha(scorepath)}}
    jsave(result_path,result); del model,score; gc.collect()
    return result

def metric_record(y,score,threshold):
    pred,values=common.metrics(y,score,threshold)
    values.update(fpr=1-values["specificity"],fnr=1-values["recall"])
    return pred,values

def strategy_fit(out,folder,name,arm,seed,fold,strategy,cid,rounds,X,y,identity,arrays,params):
    stem=f"{name}_{arm}_seed{seed}_fold{fold}_{strategy}"
    D,T,train,test=(arrays[k] for k in ("development","threshold_validation","outer_train","outer_test"))
    random_state=100*seed+fold
    t=time.monotonic(); model=fit(name,X,y,D,None,params,random_state,rounds)
    threshold_model_fit_seconds=time.monotonic()-t
    cal_score=predict(name,model,X,T)
    threshold,cal_f1=common.best_f1_threshold(y[T],cal_score)
    cal_model=folder/"calibration_models"/f"{stem}.joblib"
    dump_model(model,cal_model); cal_model_digest=model_digest(name,model)
    calpath=folder/"calibration"/f"{stem}.parquet"
    table=identity.iloc[T][["row_index","account_id"]].copy()
    table["y_true"]=y[T]; table["y_score"]=cal_score
    table.to_parquet(calpath,index=False,compression="zstd")
    cal_seconds=time.monotonic()-t; del model,table; gc.collect()
    t=time.monotonic(); model=fit(name,X,y,train,None,params,random_state,rounds)
    final_model_fit_seconds=time.monotonic()-t
    score=predict(name,model,X,test)
    finalpath=folder/"models"/f"{stem}.joblib"
    dump_model(model,finalpath); final_digest=model_digest(name,model)
    predpath=folder/"predictions"/f"{stem}.parquet"
    table=identity.iloc[test][["row_index","account_id"]].copy()
    table["y_true"]=y[test]; table["y_score"]=score
    records=[]
    for policy,cutoff in (("fixed_0.5",.5),("inner_max_f1",threshold)):
        pred,values=metric_record(y[test],score,cutoff)
        table["y_pred_fixed" if policy=="fixed_0.5" else "y_pred_inner"]=pred
        records.append({"dataset":folder.name,"model":name,"feature_set":arm,"seed":seed,"fold":fold,
            "strategy":strategy,"threshold_policy":policy,"threshold":cutoff,"n_features":X.shape[1],
            "selected_candidate_id":cid,"selected_rounds":rounds,**values})
    table.to_parquet(predpath,index=False,compression="zstd")
    final_seconds=time.monotonic()-t
    result={"strategy":strategy,"candidate_id":cid,"selected_rounds":rounds,"params":params,
        "model_random_state":random_state,"effective_params":effective(params,random_state,rounds),
        "threshold":threshold,"threshold_validation_f1":cal_f1,
        "threshold_validation_ap":float(average_precision_score(y[T],cal_score)),
        "threshold_validation_auc":float(roc_auc_score(y[T],cal_score)),
        "threshold_fit_seconds":cal_seconds,"final_fit_seconds":final_seconds,
        "threshold_model_fit_seconds":threshold_model_fit_seconds,"final_model_fit_seconds":final_model_fit_seconds,
        "calibration_model_path":rel(cal_model),"calibration_path":rel(calpath),
        "model_path":rel(finalpath),"prediction_path":rel(predpath),
        "calibration_model_digest":cal_model_digest,"final_model_digest":final_digest,
        "final_fit_has_eval_set":False,"fit_rows_sha256":array_sha(train),
        "calibration_fit_rows_sha256":array_sha(D),"threshold_rows_sha256":array_sha(T),
        "outer_test_rows_sha256":array_sha(test),"physical_reuse_anchor":False,
        "artifact_sha256":{rel(p):sha(p) for p in (cal_model,calpath,finalpath,predpath)},"metrics":records}
    del model,score,cal_score,table; gc.collect()
    return result

def run(out,ds,name):
    p=registered(out); folder=out/ds; design=read(out/"candidates.json")
    inputs=read(out/"dataset_inputs.json")[ds]
    assert all(sha(ROOT/path)==digest for path,digest in inputs["cache_sha256"].items())
    matrix=np.load(folder/"arrays/features.npy",mmap_mode="r")
    y=np.load(folder/"arrays/labels.npy",mmap_mode="r")
    identity=pd.read_parquet(folder/"accounts.parquet")
    manifest_path=folder/f"run_manifest_{name}.json"
    manifest={"dataset":ds,"model":name,"status":"running","pid":os.getpid(),"started_at":now(),
              "protocol_sha256":sha(out/"protocol.json"),"completed_outer_settings":0}
    if manifest_path.exists():
        previous_manifest=read(manifest_path)
        manifest["first_started_at"]=previous_manifest.get("first_started_at",previous_manifest["started_at"])
        manifest["previous_attempts"]=previous_manifest.get("previous_attempts",[])+[previous_manifest]
    else:
        manifest["first_started_at"]=manifest["started_at"]
    jsave(manifest_path,manifest)
    try:
        for arm in ARMS:
            names=p["feature_columns"][arm]
            X=np.ascontiguousarray(matrix[:,:len(names)])
            for seed in range(5):
                for fold in range(5):
                    stem=f"{name}_{arm}_seed{seed}_fold{fold}"
                    path=folder/"folds"/f"{stem}.json"
                    if path.exists():
                        done=read(path)
                        assert done["protocol_sha256"]==sha(out/"protocol.json")
                        for f,digest in done["artifact_sha256"].items(): assert sha(ROOT/f)==digest
                    else:
                        started=time.monotonic()
                        split=folder/"splits"/f"seed{seed}_fold{fold}.npz"
                        assert sha(split)==read(out/"split_manifest.json")[rel(split)]["sha256"]
                        with np.load(split) as f: arrays={k:f[k] for k in f.files}
                        sr=search(out,folder,name,arm,seed,fold,X,y,arrays,p,design)
                        anchor=strategy_fit(out,folder,name,arm,seed,fold,"same_protocol_anchor",0,
                            sr["anchor_rounds"],X,y,identity,arrays,design[name][0]["params"])
                        cid=sr["selected_candidate_id"]
                        if cid==0:
                            tuned=copy.deepcopy(anchor)
                            tuned["strategy"]="tuned"; tuned["physical_reuse_anchor"]=True
                            tuned["threshold_fit_seconds"]=tuned["final_fit_seconds"]=0.
                            tuned["threshold_model_fit_seconds"]=tuned["final_model_fit_seconds"]=0.
                            for row in tuned["metrics"]: row["strategy"]="tuned"
                        else:
                            tuned=strategy_fit(out,folder,name,arm,seed,fold,"tuned",cid,
                                sr["selected_rounds"],X,y,identity,arrays,design[name][cid]["params"])
                        strategies={"same_protocol_anchor":anchor,"tuned":tuned}
                        srpath=folder/"search"/stem/"search_result.json"
                        artifacts={rel(split):sha(split),rel(srpath):sha(srpath)}
                        for st in strategies.values(): artifacts.update(st["artifact_sha256"])
                        done={"dataset":ds,"model":name,"feature_set":arm,"seed":seed,"fold":fold,
                            "n_features":len(names),"feature_columns":names,"protocol_sha256":sha(out/"protocol.json"),
                            "search_result_path":rel(srpath),"split_path":rel(split),"strategies":strategies,
                            "logical_candidate_evaluations":36,"physical_search_fits":sr["physical_search_fits"],
                            "logical_strategy_models":4,"physical_strategy_models":2 if cid==0 else 4,
                            "runtime_seconds":time.monotonic()-started,"artifact_sha256":artifacts}
                        jsave(path,done)
                        print(f"{now()} DONE {ds} {stem} candidate={cid} AP={tuned['metrics'][0]['ap']:.6f} "
                              f"anchor={anchor['metrics'][0]['ap']:.6f} elapsed={done['runtime_seconds']:.1f}s",flush=True)
                    manifest["completed_outer_settings"]+=1
                    manifest["updated_at"]=now()
                    jsave(manifest_path,manifest)
            del X; gc.collect()
        assert manifest["completed_outer_settings"]==75
        manifest.update(status="complete",completed_at=now()); jsave(manifest_path,manifest)
    except BaseException:
        manifest.update(status="failed",failed_at=now(),traceback=traceback.format_exc())
        jsave(manifest_path,manifest); raise

def execute(out):
    registered(out)
    handles=[]; processes=[]
    for ds,name in itertools.product(DATASETS,MODELS):
        log=(out/ds/f"execution_{name}.log").open("a")
        handles.append(log)
        cmd=[sys.executable,__file__,"run","--out",str(out),"--dataset",ds,"--model",name]
        processes.append((ds,name,subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)))
    while any(p.poll() is None for _,_,p in processes):
        states=[]
        for ds,name,process in processes:
            path=out/ds/f"run_manifest_{name}.json"
            m=read(path) if path.exists() else {}
            states.append(f"{ds}/{name}:{m.get('completed_outer_settings',0)}/75({m.get('status','starting')})")
        print(now()+" "+" ".join(states),flush=True)
        time.sleep(30)
    for handle in handles: handle.close()
    failed=[(ds,name,p.returncode) for ds,name,p in processes if p.returncode]
    assert not failed,f"Worker failures: {failed}"
    print("ALL WORKERS COMPLETE",flush=True)

def probe(name):
    rng=np.random.default_rng(31415)
    X=rng.normal(size=(1000,16)).astype(np.float32)
    y=(X[:,0]+.5*X[:,1]>1.8).astype(np.int8)
    train=np.arange(700); early=np.arange(700,850); score_rows=np.arange(850,1000)
    old=read(OLD/"protocol.json"); design,_,_=candidates(old)
    p=design[name][0]["params"].copy()
    p["n_estimators"]=200 if name=="random_forest" else 60
    a=fit(name,X,y,train,None if name=="random_forest" else early,p,19)
    scores=predict(name,a,X,score_rows)
    b=fit(name,X,y,train,None if name=="random_forest" else early,p,19)
    np.testing.assert_array_equal(scores,predict(name,b,X,score_rows))
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/"probe.joblib"
        joblib.dump(a,path,compress=("xz",3))
        restored=joblib.load(path)
        assert model_digest(name,a)==model_digest(name,restored)
        np.testing.assert_array_equal(scores,predict(name,restored,X,score_rows))
    if name=="random_forest":
        direct=fit(name,X,y,train,None,{**p,"n_estimators":100},19)
        np.testing.assert_array_equal(predict(name,prefix(a,100),X,score_rows),predict(name,direct,X,score_rows))
    path=ROOT/"Altman_experiments/results"/f"gmkep_tuning_probe_{name}.json"
    jsave(path,{"passed":True,"model":name,"exact_repeat_scores":True,
                "model_serialization_roundtrip":True,
                "rf_prefix_direct_equivalence":name=="random_forest","trainer_sha256":sha(__file__)})
    print(json.dumps(read(path)),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=("prepare","run","execute","probe"))
    parser.add_argument("--out",type=Path,default=OUT)
    parser.add_argument("--dataset",choices=DATASETS)
    parser.add_argument("--model",choices=MODELS)
    args=parser.parse_args()
    if args.action=="prepare": prepare(args.out)
    elif args.action=="run": run(args.out,args.dataset,args.model)
    elif args.action=="execute": execute(args.out)
    else: probe(args.model)
