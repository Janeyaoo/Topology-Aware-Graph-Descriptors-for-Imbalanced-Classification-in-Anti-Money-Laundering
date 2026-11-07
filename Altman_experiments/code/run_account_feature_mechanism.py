"""IBM Small original FTD family mechanism experiments.

Frozen add/drop arms with the previously verified BASE/FULL controls reused
without refitting. All features remain static whole-graph account descriptors.
"""
from __future__ import annotations
import argparse, contextlib, gc, hashlib, importlib.metadata, json, os, platform
import shutil, subprocess, sys, time, traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_sample_weight
ROOT=Path(__file__).resolve().parents[2]
ALT=ROOT/"Altman_experiments"
SOURCE_RUN=ALT/"results/account_original_multiclassifier_20261007_isolated"
DEFAULT_OUT=ALT/"results/account_feature_mechanism_20261007"
DATASETS=("hi_small","li_small")
SUPPORTED_MODELS=("lightgbm","xgboost","random_forest")
DEFAULT_MODELS=("lightgbm","xgboost")
AUDIT_SOURCE=ALT/"config/account_feature_mechanism_audit_20261007.json"
AUDIT_SCRIPT=ALT/"code/audit_original_account_feature_mechanism.py"
POLICIES=("fixed_0.5","inner_max_f1")
METRICS=("precision","recall","f1","auc","ap","accuracy","balanced_accuracy","specificity")
TZ=ZoneInfo("Asia/Shanghai")
FAMILIES={
 "directed":["A1","A2","D1","D2","asym","asym2","div1","div2","exp_out","rec_out_count","rec_out_ratio","rec_in_count","rec_in_ratio","peel_score"],
 "local":["nv","ev","density","beta0","cc_minus","bridge_frag","tri","clust","beta1","beta1_norm"],
 "global":["core","core_mean","core_max","kappa","kappa_mean","kappa_min"],
 "multiscale":["beta0_1","beta0_2","d_beta0","beta1_1","beta1_2","d_beta1"],
}
GRAPH_STATS=["A1","A2","D1","D2","nv","ev","density","cc_minus","bridge_frag","tri","clust","beta1","beta1_norm","core","core_mean","core_max"]
NEW_FEATURES=tuple(["add_"+key for key in FAMILIES]+["drop_"+key for key in FAMILIES]+["ftd_only","base_graph_stats"])
ALL_FEATURES=("base","full",*NEW_FEATURES)


def now():
    return datetime.now(TZ).isoformat(timespec="seconds")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path):
    return str(Path(path).resolve().relative_to(ROOT))


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def load_original_inputs(dataset):
    # Do not import a legacy runner here: it imports LightGBM, which must not
    # share a fitting process with XGBoost on this local OpenMP installation.
    features = ALT / "results" / dataset / "features"
    paths = [features / f"{tag}_features.parquet" for tag in ("full", "base", "ftd")]
    full, base, ftd = [pd.read_parquet(path) for path in paths]
    labels_path = ALT / "data" / dataset / ("account_labels.parquet" if dataset == "hi_small" else "li_small_labels.parquet")
    labels = pd.read_parquet(labels_path)
    pd.testing.assert_frame_equal(full[["account_id", "label"]], labels)
    rebuilt = labels.merge(base, on="account_id", how="left").merge(ftd, on="account_id", how="left").fillna(0)
    pd.testing.assert_frame_equal(full, rebuilt)
    assert full.account_id.is_unique and not full.isna().any().any()
    np.testing.assert_array_equal(full.account_id, base.account_id)
    np.testing.assert_array_equal(full.account_id, ftd.account_id)
    columns = {"base": [c for c in base.columns if c != "account_id"]}
    columns["full"] = columns["base"] + [c for c in ftd.columns if c != "account_id"]
    assert [len(columns[tag]) for tag in ("base", "full")] == [16, 52]
    matrices = {tag: full[names].to_numpy(dtype=np.float32) for tag, names in columns.items()}
    assert all(np.isfinite(matrix).all() for matrix in matrices.values())
    y = full.label.to_numpy(dtype=np.int8)
    expected = (515088, 6357) if dataset == "hi_small" else (705907, 5304)
    assert (len(y), int(y.sum())) == expected and set(np.unique(y)) == {0, 1}
    identity = pd.DataFrame({"row_index": np.arange(len(full)), "account_id": full.account_id})
    original = ALT / "results" / dataset / "classification/cv_results.parquet"
    source = ALT / "code" / ("03_classify_significance.py" if dataset == "hi_small" else "ibm_pipeline.py")
    paths += [labels_path, original, source,
        ALT / "code" / f"rerun_{dataset}_original_full_metrics.py",
        ALT / "data" / dataset / "dataset_stats.txt",
        ALT / "data" / dataset / f"{dataset}_trans.parquet"]
    if dataset == "hi_small":
        paths += [ALT / "code/01_download_explore.py", ALT / "code/02_build_features.py"]
    else:
        copy = ROOT / "FlowTopo/results/ibm_aml/classification/cv_results.parquet"
        pd.testing.assert_frame_equal(pd.read_parquet(original), pd.read_parquet(copy))
        paths += [copy, ROOT / "FlowTopo/code/ibm_aml/ibm_pipeline.py"]
    return matrices, columns, y, identity, paths


def best_f1_threshold(y, probability):
    order = np.argsort(-probability, kind="mergesort")
    scores = probability[order]
    positives = np.cumsum(y[order], dtype=np.int64)
    ends = np.r_[np.flatnonzero(scores[:-1] != scores[1:]), len(scores) - 1]
    thresholds = scores[ends][::-1]
    numerator = (2 * positives[ends])[::-1]
    denominator = (ends + 1 + int(y.sum()))[::-1]
    f1 = numerator / denominator
    position = int(np.argmax(f1))
    return float(thresholds[position]), float(f1[position])


def metrics(y, probability, threshold):
    predicted = (probability >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return predicted, {
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "auc": float(roc_auc_score(y, probability)),
        "ap": float(average_precision_score(y, probability)),
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "specificity": float(tn / (tn + fp)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "n_test": len(y), "n_positive": int(y.sum()),
    }


def fit_model(name, X, y, fit_index, calibration_index, registered,
              random_state, rounds=None):
    params = dict(registered["model_params"][name])
    params["random_state"] = random_state
    if rounds is not None:
        params["n_estimators"] = int(rounds)
    if name == "lightgbm":
        import lightgbm as lgb
        model = lgb.LGBMClassifier(**params)
        if calibration_index is None:
            model.fit(X[fit_index], y[fit_index])
        else:
            model.fit(X[fit_index], y[fit_index],
                eval_set=[(X[calibration_index], y[calibration_index])],
                callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(period=-1)])
    elif name == "xgboost":
        import xgboost as xgb
        if calibration_index is not None:
            params["early_stopping_rounds"] = 30
        model = xgb.XGBClassifier(**params)
        fit_kwargs = {"sample_weight": compute_sample_weight("balanced", y[fit_index])}
        if calibration_index is not None:
            fit_kwargs.update(eval_set=[(X[calibration_index], y[calibration_index])], verbose=False)
        model.fit(X[fit_index], y[fit_index], **fit_kwargs)
    else:
        model = RandomForestClassifier(**params)
        model.fit(X[fit_index], y[fit_index])
    return model


def selected_rounds(name, model, registered):
    if name == "lightgbm":
        return int(model.best_iteration_ or registered["model_params"][name]["n_estimators"])
    if name == "xgboost":
        return int(model.best_iteration + 1)
    return int(registered["model_params"][name]["n_estimators"])


class Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, text):
        for stream in self.streams: stream.write(text); stream.flush()
    def flush(self):
        for stream in self.streams: stream.flush()



class LazyFeatureMatrices(dict):
    def __init__(self, originals, columns):
        super().__init__(originals)
        self.columns=columns
        self.index={key:[columns["full"].index(name) for name in names] for key,names in columns.items()}
    def __getitem__(self,key):
        if key in ("base","full"):
            return super().__getitem__(key)
        return super().__getitem__("full")[:,self.index[key]]

def mechanism_columns(base_columns,full_columns):
    ftd_columns=full_columns[len(base_columns):]
    assert ftd_columns==sum(FAMILIES.values(),[]) and [len(v) for v in FAMILIES.values()]==[14,10,6,6]
    columns={"base":base_columns,"full":full_columns}
    for family,names in FAMILIES.items():
        columns["add_"+family]=base_columns+names
        columns["drop_"+family]=[name for name in full_columns if name not in names]
    columns["ftd_only"]=ftd_columns
    columns["base_graph_stats"]=base_columns+GRAPH_STATS
    assert all(len(names)==len(set(names)) for names in columns.values())
    assert len(columns["base_graph_stats"])==32
    return columns

def load_inputs(dataset):
    matrices,original,y,identity,paths=load_original_inputs(dataset)
    columns=mechanism_columns(original["base"],original["full"])
    # No legacy classifiers are imported while constructing these subsets.
    paths += [SOURCE_RUN/"protocol.json",SOURCE_RUN/"validation.json",SOURCE_RUN/"run_manifest.json"]
    return LazyFeatureMatrices(matrices,columns),columns,y,identity,paths

def contrasts():
    rows=[dict(contrast_id="full_vs_base",contrast_type="whole_group",family=None,lhs_arm="full",rhs_arm="base")]
    for family in FAMILIES:
        rows.append(dict(contrast_id="add_"+family,contrast_type="addition",family=family,lhs_arm="add_"+family,rhs_arm="base"))
        rows.append(dict(contrast_id="remove_"+family,contrast_type="removal_contribution",family=family,lhs_arm="full",rhs_arm="drop_"+family))
    rows += [dict(contrast_id="ftd_only_vs_base",contrast_type="standalone_comparison",family=None,lhs_arm="ftd_only",rhs_arm="base"),
        dict(contrast_id="graph_vs_base",contrast_type="ordinary_graph",family=None,lhs_arm="base_graph_stats",rhs_arm="base"),
        dict(contrast_id="full_vs_graph",contrast_type="beyond_selected_graph_stats",family=None,lhs_arm="full",rhs_arm="base_graph_stats")]
    assert len(rows)==12
    return rows

def prepare(out,models):
    assert not out.exists(), "Use a new output directory when registering"
    prior=json.loads((SOURCE_RUN/"protocol.json").read_text())
    validation=json.loads((SOURCE_RUN/"validation.json").read_text())
    manifest=json.loads((SOURCE_RUN/"run_manifest.json").read_text())
    assert validation["passed"] and validation["n_models_verified"]==600 and manifest["status"]=="complete"
    for name,expected in validation["result_hashes"].items():
        assert sha256(SOURCE_RUN/name)==expected, name
    scientific=["seeds","outer_n_splits","outer_split","inner_calibration_fraction","inner_random_state","model_random_state",
        "parameter_candidates_per_model","early_stopping","class_weighting","threshold_policies","threshold_selection","prediction_rule","summary_std_ddof","primary_metrics","feature_setting","outer_labels"]
    registered={key:prior[key] for key in scientific}
    registered.update(experiment="original_account_feature_family_mechanism",registered_at=now(),run_date="2026-10-07",
        datasets=list(DATASETS),models=list(models),model_params={model:prior["model_params"][model] for model in models},
        families=FAMILIES,graph_stats_columns=GRAPH_STATS,new_feature_sets=list(NEW_FEATURES),reused_feature_sets=["base","full"],
        contrasts=contrasts(),contrast_definition="delta = lhs_arm - rhs_arm; removal contribution = FULL - FULL_without_family",
        source_control_run=rel(SOURCE_RUN),source_protocol_sha256=sha256(SOURCE_RUN/"protocol.json"),
        source_validation_sha256=sha256(SOURCE_RUN/"validation.json"),trainer_sha256=sha256(__file__),
        feature_audit_sha256=sha256(AUDIT_SOURCE),feature_audit_script_sha256=sha256(AUDIT_SCRIPT),
        expected_new_final_models=500*len(models),expected_new_inner_models=500*len(models),
        expected_reused_final_models=100*len(models),expected_reused_inner_models=100*len(models),
        expected_final_models_reported=600*len(models),expected_inner_models_reported=600*len(models),
        parameter_selection="Inherited fixed parameter set; no new search, feature selection or weight tuning",
        feature_construction="Frozen column subsets of original whole-graph float32 caches; no new graph construction or label-derived features",
        execution_model_isolation="Fresh process per classifier; HI and LI may run in parallel",
        family_interpretation="Column-family conditional performance effects, not causal or unique information contributions; constants and exact/algebraic cross-family overlaps remain",
        multiscale_interpretation="Original 1/2-hop cycle-rank expressions; entirely determined by local-family columns in both caches; not VR barcode persistence",
        graph_control_interpretation="Selected 16 conventional graph-statistic columns already present in the same cache; not an exhaustive comparison to all graph representations",
        environment={"python":platform.python_version(),"python_executable":sys.executable,"platform":platform.platform(),"cpu_count":os.cpu_count(),
            "lightgbm":importlib.metadata.version("lightgbm"),"xgboost":importlib.metadata.version("xgboost"),"scikit_learn":sklearn.__version__,
            "numpy":np.__version__,"pandas":pd.__version__,"pyarrow":pa.__version__,"joblib":joblib.__version__})
    datasets={}; control_hashes={}
    for dataset in DATASETS:
        _,columns,y,identity,paths=load_inputs(dataset)
        datasets[dataset]={"n_samples":len(y),"n_positive":int(y.sum()),"feature_columns":columns,"input_sha256":{rel(p):sha256(p) for p in paths}}
        prior_data=json.loads((SOURCE_RUN/dataset/"run_manifest.json").read_text())
        for path,digest in prior_data["inputs_sha256"].items():
            assert sha256(ROOT/path)==digest,path
    registered["feature_sets"]={key:len(value) for key,value in datasets["hi_small"]["feature_columns"].items()}
    assert datasets["hi_small"]["feature_columns"]==datasets["li_small"]["feature_columns"]
    registered["feature_columns"]=datasets["hi_small"]["feature_columns"]
    out.mkdir(parents=True)
    write_json(out/"protocol.json",registered)
    shutil.copy2(__file__,out/"trainer_snapshot.py")
    for name in ("EXPERIMENT_RESULTS_SUMMARY.md","EXPERIMENT_RESULTS_SUMMARY_gen.py"):
        if (ROOT/name).is_file():
            shutil.copy2(ROOT/name,out/("before_"+name))
    shutil.copy2(AUDIT_SOURCE,out/"feature_audit.json")
    shutil.copy2(AUDIT_SCRIPT,out/"feature_audit_snapshot.py")
    for dataset in DATASETS:
        folder=out/dataset
        (folder/"splits").mkdir(parents=True)
        (folder/"reused_controls").mkdir()
        shutil.copy2(SOURCE_RUN/dataset/"accounts.parquet",folder/"accounts.parquet")
        rows=[]
        for seed in registered["seeds"]:
            for fold in range(registered["outer_n_splits"]):
                split=SOURCE_RUN/dataset/"splits"/f"seed{seed}_fold{fold}.npz"
                shutil.copy2(split,folder/"splits"/split.name)
                control_hashes[rel(split)]=sha256(split)
                for model in models:
                    for feature in ("base","full"):
                        stem=f"{model}_{feature}_seed{seed}_fold{fold}"
                        source=SOURCE_RUN/dataset/"folds"/(stem+".json")
                        checkpoint=json.loads(source.read_text())
                        assert checkpoint["protocol_sha256"]==registered["source_protocol_sha256"]
                        control_hashes[rel(source)]=sha256(source)
                        for name,digest in checkpoint["artifact_sha256"].items():
                            assert sha256(ROOT/name)==digest,name
                            control_hashes[name]=digest
                        write_json(folder/"reused_controls"/(stem+".json"),{
                            "training_origin":"reused_control","trained_in_this_run":False,
                            "source_checkpoint":rel(source),"source_checkpoint_sha256":sha256(source),
                            "source_protocol_sha256":registered["source_protocol_sha256"],
                            "source_validation_sha256":registered["source_validation_sha256"],
                            "artifact_sha256":checkpoint["artifact_sha256"],"metrics":checkpoint["metrics"]})
                        rows.extend(checkpoint["metrics"])
        pd.DataFrame(rows).to_csv(folder/"control_fold_metrics.csv",index=False)
    for name in ("protocol.json","validation.json","run_manifest.json","fold_metrics.csv","summary.csv","paired_summary.csv"):
        control_hashes[rel(SOURCE_RUN/name)]=sha256(SOURCE_RUN/name)
    write_json(out/"dataset_inputs.json",datasets)
    write_json(out/"reused_control_hashes.json",control_hashes)
    (out/"PROTOCOL.md").write_text("# IBM AML Small 原始账户特征：机制消融\n\n"
        "四家族按原列定义冻结；每个家族新增 BASE+该家族与 FULL-该家族，共八组；另加 FTD-only 与 BASE+选定常规图统计16列（BASE32）。BASE16/FULL52复用上一已核验正式实验，不重复训练。\n\n"
        "保持相同5 seeds×5外层折、20%内层校准、固定参数、内层AUC早停与max-F1阈值，最终外层重训无eval_set，同时报告固定0.5；指标、误差条与配对方向均以protocol.json为准。\n\n"
        "原FTD含常量/重复/代数派生列，特别是多尺度D由局部B完全确定；消融解释为固定模型下显式列的条件性能变化，不作为独立信息或因果归因。仍为全图静态账户识别；25折有相关性，不作独立重复显著性检验。\n")
    print(json.dumps({"prepared":rel(out),"models":models,"new_final_models":registered["expected_new_final_models"],
        "new_inner_models":registered["expected_new_inner_models"],"reused_final_models":registered["expected_reused_final_models"],
        "protocol_sha256":sha256(out/"protocol.json")},ensure_ascii=False),flush=True)



def run_model_dataset(out, dataset, model_name):
    registered = json.loads((out / "protocol.json").read_text())
    assert registered["trainer_sha256"] == sha256(__file__), "Trainer changed after registration"
    folder = out / dataset
    folder.mkdir(exist_ok=True)
    for name in ("splits", "models", "inner_models", "predictions", "calibration", "folds"):
        (folder / name).mkdir(exist_ok=True)
    start = time.monotonic()
    with (folder / "run.log").open("a") as log, contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
        manifest_path = folder / f"run_manifest_{model_name}.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
            "dataset": dataset, "model": model_name, "pid": os.getpid(), "started_at": now(), "protocol_sha256": sha256(out / "protocol.json"),
            "script_sha256": sha256(__file__), "script": rel(__file__),
            "command": [sys.executable, *sys.argv], "status": "running"}
        manifest["status"] = "running"
        write_json(manifest_path, manifest)
        try:
            matrices, columns, y, identity, input_paths = load_inputs(dataset)
            input_paths = list(dict.fromkeys(input_paths + [Path(__file__), out / "protocol.json"]))
            before = {rel(path): sha256(path) for path in input_paths}
            if "inputs_sha256" in manifest:
                assert manifest["inputs_sha256"] == before, "Input changed during resume"
            manifest.update(inputs_sha256=before, feature_columns=columns,
                n_samples=len(y), n_positive=int(y.sum()), models_expected=len(NEW_FEATURES)*25,
                inner_models_expected=len(NEW_FEATURES)*25, feature_dtype="float32")
            write_json(manifest_path, manifest)
            identity["label"] = y
            identity.to_parquet(folder / "accounts.parquet", index=False, compression="zstd")
            records = []
            for seed in registered["seeds"]:
                splitter = StratifiedKFold(registered["outer_n_splits"], shuffle=True, random_state=seed)
                for fold, (outer_train, outer_test) in enumerate(splitter.split(matrices["base"], y)):
                    inner_seed = 10000 + 100 * seed + fold
                    inner = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=inner_seed)
                    fit_local, cal_local = next(inner.split(matrices["base"][outer_train], y[outer_train]))
                    fit_index, cal_index = outer_train[fit_local], outer_train[cal_local]
                    split_path = folder / "splits" / f"seed{seed}_fold{fold}.npz"
                    arrays = {"outer_train": outer_train.astype(np.int32), "outer_test": outer_test.astype(np.int32),
                        "inner_fit": fit_index.astype(np.int32), "inner_calibration": cal_index.astype(np.int32)}
                    if split_path.exists():
                        with np.load(split_path) as prior:
                            for key, value in arrays.items(): np.testing.assert_array_equal(prior[key], value)
                    else:
                        np.savez_compressed(split_path, **arrays)
                    for name in (model_name,):
                        for feature in NEW_FEATURES:
                            stem = f"{name}_{feature}_seed{seed}_fold{fold}"
                            record_path = folder / "folds" / f"{stem}.json"
                            if record_path.exists():
                                completed = json.loads(record_path.read_text())
                                assert completed["protocol_sha256"] == sha256(out / "protocol.json")
                                for path, expected in completed["artifact_sha256"].items():
                                    assert sha256(ROOT / path) == expected, f"Changed checkpoint {path}"
                                records.extend(completed["metrics"])
                                continue
                            model_start = time.monotonic()
                            X = matrices[feature]
                            random_state = 100 * seed + fold
                            print(f"{now()} START {dataset} {stem}", flush=True)
                            inner_model = fit_model(name, X, y, fit_index, cal_index, registered, random_state)
                            cal_score = inner_model.predict_proba(X[cal_index])[:, 1].astype(np.float64)
                            assert np.isfinite(cal_score).all() and ((cal_score >= 0) & (cal_score <= 1)).all()
                            threshold, inner_f1 = best_f1_threshold(y[cal_index], cal_score)
                            rounds = selected_rounds(name, inner_model, registered)
                            inner_path = folder / "inner_models" / f"{stem}.joblib"
                            joblib.dump(inner_model, inner_path, compress=3)
                            calibration_path = folder / "calibration" / f"{stem}.parquet"
                            pd.DataFrame({"row_index": cal_index.astype(np.int32), "y_true": y[cal_index],
                                "y_score": cal_score}).to_parquet(calibration_path, index=False, compression="zstd")
                            del inner_model; gc.collect()
                            # The final fit receives outer_train only; no outer eval_set.
                            final_model = fit_model(name, X, y, outer_train, None, registered, random_state, rounds)
                            final_path = folder / "models" / f"{stem}.joblib"
                            joblib.dump(final_model, final_path, compress=3)
                            score = final_model.predict_proba(X[outer_test])[:, 1].astype(np.float64)
                            assert np.isfinite(score).all() and ((score >= 0) & (score <= 1)).all()
                            test_records = []
                            predictions = {"row_index": outer_test.astype(np.int32), "y_true": y[outer_test], "y_score": score}
                            for policy, cutoff in (("fixed_0.5", 0.5), ("inner_max_f1", threshold)):
                                predicted, metric = metrics(y[outer_test], score, cutoff)
                                predictions["y_pred_fixed" if policy == "fixed_0.5" else "y_pred_inner"] = predicted
                                test_records.append({"dataset": dataset, "model": name, "feature_set": feature,
                                    "seed": seed, "fold": fold, "threshold_policy": policy, "threshold": cutoff,
                                    "n_features": X.shape[1], "selected_rounds": rounds, **metric})
                            prediction_path = folder / "predictions" / f"{stem}.parquet"
                            pd.DataFrame(predictions).to_parquet(prediction_path, index=False, compression="zstd")
                            artifacts = [split_path, inner_path, final_path, calibration_path, prediction_path]
                            completed = {"dataset": dataset, "model": name, "feature_set": feature,
                                "seed": seed, "fold": fold, "random_state": random_state,
                                "inner_random_state": inner_seed, "n_features": X.shape[1],
                                "n_outer_train": len(outer_train), "n_inner_fit": len(fit_index),
                                "n_inner_calibration": len(cal_index), "n_outer_test": len(outer_test),
                                "inner_fit_positive": int(y[fit_index].sum()),
                                "inner_calibration_positive": int(y[cal_index].sum()),
                                "outer_train_positive": int(y[outer_train].sum()),
                                "outer_test_positive": int(y[outer_test].sum()),
                                "selected_rounds": rounds, "inner_threshold": threshold,
                                "inner_f1": inner_f1, "inner_ap": float(average_precision_score(y[cal_index], cal_score)),
                                "inner_auc": float(roc_auc_score(y[cal_index], cal_score)),
                                "final_fit_has_eval_set": False,
                                "protocol_sha256": sha256(out / "protocol.json"),
                                "artifact_sha256": {rel(path): sha256(path) for path in artifacts},
                                "runtime_seconds": time.monotonic() - model_start, "metrics": test_records}
                            write_json(record_path, completed)
                            records.extend(test_records)
                            pd.DataFrame(records).to_csv(folder / f"fold_metrics_{model_name}.csv", index=False)
                            manifest.update(models_completed=len(records) // 2, inner_models_completed=len(records) // 2,
                                last_completed=stem, updated_at=now())
                            write_json(manifest_path, manifest)
                            print(f"{now()} DONE {stem} rounds={rounds} AP={test_records[0]['ap']:.6f} "
                                f"F1(inner)={test_records[1]['f1']:.6f} {completed['runtime_seconds']:.1f}s "
                                f"({len(records)//2}/{len(NEW_FEATURES)*25})", flush=True)
                            del final_model, score, cal_score; gc.collect()
            assert len(records) == len(NEW_FEATURES)*50
            frame = pd.DataFrame(records)
            frame.to_csv(folder / f"fold_metrics_{model_name}.csv", index=False)
            after = {rel(path): sha256(path) for path in input_paths}
            assert after == before, "Original inputs changed"
            manifest.update(status="complete", completed_at=now(), models_completed=len(NEW_FEATURES)*25,
                inner_models_completed=len(NEW_FEATURES)*25, metric_rows=len(records),
                inputs_unchanged_after_run=True, runtime_seconds=time.monotonic() - start)
            write_json(manifest_path, manifest)
        except BaseException:
            manifest.update(status="failed", failed_at=now(), traceback=traceback.format_exc())
            write_json(manifest_path, manifest)
            raise


def run_dataset(out,dataset):
    registered=json.loads((out/"protocol.json").read_text()); models=registered["models"]
    folder=out/dataset; start=time.monotonic()
    manifest={"dataset":dataset,"status":"running","started_at":now(),"pid":os.getpid(),
        "protocol_sha256":sha256(out/"protocol.json"),"models_expected":250*len(models),"inner_models_expected":250*len(models)}
    write_json(folder/"run_manifest.json",manifest)
    try:
        for model in models:
            proc=subprocess.run([sys.executable,str(Path(__file__).resolve()),"--phase","worker","--dataset",dataset,"--model",model,"--out",str(out.resolve())])
            assert proc.returncode==0,f"{dataset}/{model} worker exit {proc.returncode}"
        workers=[json.loads((folder/f"run_manifest_{model}.json").read_text()) for model in models]
        assert all(worker["status"]=="complete" for worker in workers)
        assert all(worker["inputs_sha256"]==workers[0]["inputs_sha256"] for worker in workers)
        assert all(worker["feature_columns"]==workers[0]["feature_columns"] for worker in workers)
        frames=[pd.read_csv(folder/f"fold_metrics_{model}.csv") for model in models]
        new=pd.concat(frames,ignore_index=True); assert len(new)==500*len(models)
        new.to_csv(folder/"new_fold_metrics.csv",index=False)
        controls=pd.read_csv(folder/"control_fold_metrics.csv"); assert len(controls)==100*len(models)
        combined=pd.concat([controls,new],ignore_index=True); combined.to_csv(folder/"fold_metrics.csv",index=False)
        manifest.update(status="complete",completed_at=now(),models_completed=250*len(models),inner_models_completed=250*len(models),
            reused_final_models=50*len(models),reused_inner_models=50*len(models),final_models_reported=300*len(models),
            metric_rows=len(combined),new_metric_rows=len(new),inputs_sha256=workers[0]["inputs_sha256"],
            inputs_unchanged_after_run=True,feature_columns=workers[0]["feature_columns"],n_samples=workers[0]["n_samples"],
            n_positive=workers[0]["n_positive"],runtime_seconds=time.monotonic()-start)
        write_json(folder/"run_manifest.json",manifest)
    except BaseException:
        manifest.update(status="failed",traceback=traceback.format_exc()); write_json(folder/"run_manifest.json",manifest); raise

def prediction_file(out,dataset,model,feature,seed,fold):
    parent=SOURCE_RUN if feature in ("base","full") else out
    return parent/dataset/"predictions"/f"{model}_{feature}_seed{seed}_fold{fold}.parquet"

def summarize(out):
    registered=json.loads((out/"protocol.json").read_text()); models=registered["models"]
    frames=[]; seed_rows=[]
    for dataset in DATASETS:
        manifest=json.loads((out/dataset/"run_manifest.json").read_text())
        assert manifest["status"]=="complete" and manifest["models_completed"]==250*len(models)
        frame=pd.read_csv(out/dataset/"fold_metrics.csv"); assert len(frame)==600*len(models); frames.append(frame)
        for model in models:
            for feature in ALL_FEATURES:
                for seed in registered["seeds"]:
                    pred=pd.concat([pd.read_parquet(prediction_file(out,dataset,model,feature,seed,fold)) for fold in range(5)],ignore_index=True).sort_values("row_index")
                    assert pred.row_index.nunique()==len(pred)==manifest["n_samples"]
                    for policy,column in (("fixed_0.5","y_pred_fixed"),("inner_max_f1","y_pred_inner")):
                        truth=pred.y_true.to_numpy(); score=pred.y_score.to_numpy(); labels=pred[column].to_numpy()
                        tn,fp,fn,tp=confusion_matrix(truth,labels,labels=[0,1]).ravel()
                        seed_rows.append({"dataset":dataset,"model":model,"feature_set":feature,"seed":seed,"threshold_policy":policy,
                            "precision":precision_score(truth,labels,zero_division=0),"recall":recall_score(truth,labels,zero_division=0),
                            "f1":f1_score(truth,labels,zero_division=0),"auc":roc_auc_score(truth,score),"ap":average_precision_score(truth,score),
                            "accuracy":accuracy_score(truth,labels),"balanced_accuracy":balanced_accuracy_score(truth,labels),"specificity":tn/(tn+fp),
                            "tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp)})
    all_folds=pd.concat(frames,ignore_index=True); all_folds.to_csv(out/"fold_metrics.csv",index=False)
    rows=[]; keys=["dataset","model","feature_set","threshold_policy"]
    for key,group in all_folds.groupby(keys,sort=False):
        assert len(group)==25
        row=dict(zip(keys,key),n_folds=25)
        for metric in (*METRICS,"threshold"):
            row[metric+"_mean"]=group[metric].mean(); row[metric+"_std"]=group[metric].std(ddof=1)
        rows.append(row)
    pd.DataFrame(rows).to_csv(out/"summary.csv",index=False)
    seed_frame=pd.DataFrame(seed_rows); seed_frame.to_csv(out/"seed_oof_metrics.csv",index=False)
    paired=[]; seed_paired=[]
    for spec in registered["contrasts"]:
        for original,join_keys,destination in ((all_folds,["dataset","model","seed","fold","threshold_policy"],paired),
            (seed_frame,["dataset","model","seed","threshold_policy"],seed_paired)):
            lhs=original[original.feature_set==spec["lhs_arm"]]; rhs=original[original.feature_set==spec["rhs_arm"]]
            pair=lhs.merge(rhs,on=join_keys,suffixes=("_lhs","_rhs"),validate="one_to_one")
            difference=pair[join_keys].copy()
            for name,value in spec.items(): difference[name]=value
            for metric in METRICS: difference["delta_"+metric]=pair[metric+"_lhs"]-pair[metric+"_rhs"]
            destination.append(difference)
    pairs=pd.concat(paired,ignore_index=True); pairs.to_csv(out/"paired_fold_differences.csv",index=False)
    pd.concat(seed_paired,ignore_index=True).to_csv(out/"paired_seed_oof_differences.csv",index=False)
    rows=[]; keys=["dataset","model","contrast_id","threshold_policy","contrast_type","lhs_arm","rhs_arm"]
    for key,group in pairs.groupby(keys,sort=False):
        assert len(group)==25
        row=dict(zip(keys,key),n_pairs=25)
        for metric in METRICS:
            v=group["delta_"+metric]; row["delta_"+metric+"_mean"]=v.mean(); row["delta_"+metric+"_std"]=v.std(ddof=1); row[metric+"_positive_pairs"]=int((v>0).sum())
        rows.append(row)
    pd.DataFrame(rows).to_csv(out/"paired_summary.csv",index=False)
    source_hashes=json.loads((out/"reused_control_hashes.json").read_text())
    for name,digest in source_hashes.items(): assert sha256(ROOT/name)==digest,name
    write_json(out/"run_manifest.json",{"status":"trained_pending_independent_validation","completed_at":now(),"protocol_sha256":sha256(out/"protocol.json"),
        "new_final_models_completed":500*len(models),"new_inner_models_completed":500*len(models),
        "reused_final_models":100*len(models),"reused_inner_models":100*len(models),
        "final_models_reported":600*len(models),"inner_models_reported":600*len(models),
        "metric_rows":len(all_folds),"summary_rows":48*len(models),"paired_summary_rows":48*len(models),
        "old_inputs_unchanged":True,"reused_controls_unchanged":True,
        "summary_aggregation":"Outer-fold mean and sample SD ddof=1; seed pooled OOF supplied separately"})
    print(f"Summarized {len(all_folds)} fold rows; {len(rows)} contrasts",flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--phase",choices=("prepare","run","worker","summarize"),required=True)
    parser.add_argument("--dataset",choices=DATASETS)
    parser.add_argument("--model",choices=SUPPORTED_MODELS)
    parser.add_argument("--models",nargs="+",choices=SUPPORTED_MODELS,default=list(DEFAULT_MODELS))
    parser.add_argument("--out",type=Path,default=DEFAULT_OUT)
    args=parser.parse_args()
    if args.phase=="prepare":
        assert len(args.models)==len(set(args.models)); prepare(args.out,args.models)
    elif args.phase=="summarize": summarize(args.out)
    else:
        if args.dataset is None: parser.error("dataset required")
        registered=json.loads((args.out/"protocol.json").read_text())
        if args.phase=="run": run_dataset(args.out,args.dataset)
        else:
            if args.model not in registered["models"]: parser.error("model must be registered")
            run_model_dataset(args.out,args.dataset,args.model)

if __name__=="__main__": main()
