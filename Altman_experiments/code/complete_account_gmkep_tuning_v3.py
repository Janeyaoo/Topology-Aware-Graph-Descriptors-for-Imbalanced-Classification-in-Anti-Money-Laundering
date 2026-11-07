"""Complete post-validation CSV recovery without rerunning trained models.

The six v2 model-verification reports have passed. This new revision preserves
those reports, their 24 verification CSVs and all 11 scientific CSVs. It changes
only post-validation CSV parsing to round-trip floats, then aggregates validation,
renders the report and builds the paper. Original eight scripts, three v2 scripts
and the v2 revision remain immutable. Registration never overwrites a failed
attempt; execution uses the existing pipeline status file and the same lock.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import traceback

import complete_account_gmkep_tuning_v2 as legacy

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "Altman_experiments/code"
OUT = ROOT / "Altman_experiments/results/account_gmkep56_tuned_20261008"
DATASETS = ("hi_small", "li_small")
MODELS = ("lightgbm", "xgboost", "random_forest")
REVISION_SCRIPTS = (
    "complete_account_gmkep_tuning_v3.py", "verify_account_gmkep_tuning_v3.py",
    "report_account_gmkep_tuning_v3.py", "build_tuned_paper_revision_v3.py",
)
REVISION_DIRECTORY = "analysis_revision_roundtrip_20261009"
REVISION_ID = "post_validation_csv_roundtrip_20261009"
DIAGNOSIS_FILE = "qa/csv_roundtrip_aggregate_diagnosis_20261009.json"
FAILED_ATTEMPT_FILES = (
    "completion_pipeline.json", "validation_models.json",
    "completion_v2_verify_account_gmkep_tuning_v2.log", "run_manifest.json", "aggregation_validation.json",
)
SCIENCE = (
    "fold_metrics.csv", "seed_oof_metrics.csv", "summary.csv", "candidate_fold_scores.csv", "selection_records.csv",
    "paired_feature_fold_differences.csv", "paired_feature_summary.csv", "paired_feature_seed_oof_differences.csv",
    "paired_strategy_fold_differences.csv", "paired_strategy_summary.csv", "paired_strategy_seed_oof_differences.csv",
)
VERIFICATION_FILES = (
    "verification_fold_metrics_{model}.csv", "verification_candidate_scores_{model}.csv",
    "verification_selection_records_{model}.csv", "verification_seed_metrics_{model}.csv",
)


def now(): return datetime.now().astimezone().isoformat()
def read(path): return json.loads(path.read_text())
def rel(path): return str(path.resolve().relative_to(ROOT))
def sha(path):
    with path.open("rb") as stream: return hashlib.file_digest(stream, "sha256").hexdigest()
def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)
def inventory(directory):
    return {rel(path): sha(path) for path in sorted(directory.rglob("*")) if path.is_file()}


def passed_worker_evidence(out, v2_manifest):
    """Read already completed verification; never fit or predict a model."""
    reports, tables = {}, {}
    validator_hash = v2_manifest["scripts_sha256"]["verify_account_gmkep_tuning_v2.py"]
    v2_hash = sha(out / legacy.REVISION_DIRECTORY / "manifest.json")
    for dataset, model in itertools.product(DATASETS, MODELS):
        path = out / dataset / f"validation_{model}.json"
        worker = read(path)
        assert worker["passed"] is True and worker["status"] == "passed" and worker["complete_worker"] is True
        assert worker["dataset"] == dataset and worker["model"] == model
        assert worker["validator"] == rel(CODE / "verify_account_gmkep_tuning_v2.py")
        assert worker["validator_sha256"] == validator_hash and worker["analysis_revision_manifest_sha256"] == v2_hash
        assert worker["protocol_sha256"] == sha(out / "protocol.json")
        assert worker["checks"] and all(worker["checks"].values())
        assert worker["original_inputs_sha256_verified"] is True
        assert worker["outer_settings_verified"] == 75 and worker["logical_candidate_scores_verified"] == 2700
        assert worker["logical_final_models_verified"] == worker["logical_threshold_models_verified"] == 150
        assert worker["candidate_rank_audits_verified"] == 2700 and worker["retained_search_models_rescored"] == 75
        reports[rel(path)] = sha(path)
        for pattern in VERIFICATION_FILES:
            table = out / dataset / pattern.format(model=model)
            digest = sha(table)
            assert worker["result_hashes"][str(table.relative_to(out))] == digest
            tables[rel(table)] = digest
    assert len(reports) == 6 and len(tables) == 24
    return reports, tables


def register(out):
    """Seal a new post-validation revision without replacing any prior state."""
    v2 = legacy.frozen(out)
    revision = out / REVISION_DIRECTORY
    assert not revision.exists(), "Post-validation revision already exists; registration never overwrites it"
    failure = read(out / "completion_pipeline.json")
    assert failure["status"] == "failed" and failure["stage"] == "verify_account_gmkep_tuning_v2.py"
    assert failure["analysis_revision_manifest_sha256"] == sha(out / legacy.REVISION_DIRECTORY / "manifest.json")
    failed_validation = read(out / "validation_models.json")
    assert failed_validation["passed"] is False and failed_validation["validator_sha256"] == v2["scripts_sha256"]["verify_account_gmkep_tuning_v2.py"]
    assert "paired_strategy_summary.csv_precision_positive_pairs" in failed_validation["error"]
    assert read(out / "run_manifest.json")["status"] == "trained_aggregated"
    reports, verification = passed_worker_evidence(out, v2)
    science = {rel(out / name): sha(out / name) for name in SCIENCE}
    old_revision_files = inventory(out / legacy.REVISION_DIRECTORY)
    old_analysis = read(out / "analysis_code_manifest.json")
    original_failure = {rel(out / name): sha(out / name) for name in FAILED_ATTEMPT_FILES}
    sources = {name: sha(CODE / name) for name in REVISION_SCRIPTS}
    diagnosis = out / DIAGNOSIS_FILE
    assert diagnosis.is_file(), "Save the round-trip parser diagnosis before registration"
    read(diagnosis)
    diagnosis_hash = sha(diagnosis)
    staging = Path(tempfile.mkdtemp(prefix=".analysis_revision_roundtrip_20261009.register.", dir=out))
    try:
        snapshots, attempt = staging / "snapshots", staging / "attempt_before"
        snapshots.mkdir(); attempt.mkdir()
        for name, digest in sources.items():
            shutil.copy2(CODE / name, snapshots / name)
            assert sha(snapshots / name) == digest
        shutil.copy2(diagnosis, snapshots / "roundtrip_diagnosis.json")
        for name in FAILED_ATTEMPT_FILES:
            shutil.copy2(out / name, attempt / name)
            assert sha(attempt / name) == original_failure[rel(out / name)]
        snapshots_sha = {rel(revision / "snapshots" / name): digest for name, digest in sources.items()}
        snapshots_sha[rel(revision / "snapshots/roundtrip_diagnosis.json")] = diagnosis_hash
        manifest = {
            "schema_version": 1, "revision_id": REVISION_ID, "registered_at": now(),
            "scope": "Post-validation CSV float round-trip recovery; reuse six fully passed v2 model reports; scientific CSV bytes unchanged",
            "training_protocol_changed": False, "training_rerun": False, "model_validation_rerun": False,
            "scientific_csv_rewritten": False, "new_scientific_csv_float_parser": "round_trip",
            "old_fixed_csv_parser_preserved": True,
            "protocol_sha256": sha(out / "protocol.json"),
            "original_analysis_code_manifest_sha256": sha(out / "analysis_code_manifest.json"),
            "original_scripts_sha256": old_analysis["scripts_sha256"],
            "v2_scripts_sha256": v2["scripts_sha256"],
            "v2_analysis_revision_manifest_sha256": sha(out / legacy.REVISION_DIRECTORY / "manifest.json"),
            "v2_analysis_revision_files_sha256": old_revision_files,
            "scripts_sha256": sources, "snapshots_sha256": snapshots_sha,
            "roundtrip_diagnosis_source_path": rel(diagnosis), "roundtrip_diagnosis_source_sha256": diagnosis_hash,
            "roundtrip_diagnosis_snapshot_path": rel(revision / "snapshots/roundtrip_diagnosis.json"),
            "roundtrip_diagnosis_snapshot_sha256": diagnosis_hash,
            "passed_worker_reports_sha256": reports, "verification_csv_sha256": verification,
            "scientific_csv_sha256": science,
            "failed_attempt_sha256": {rel(revision / "attempt_before" / name): original_failure[rel(out / name)]
                                      for name in FAILED_ATTEMPT_FILES},
            "failed_attempt_original_paths": {rel(revision / "attempt_before" / name): rel(out / name)
                                              for name in FAILED_ATTEMPT_FILES},
            "failed_attempt": {key: failure.get(key) for key in ("status", "stage", "started_at", "failed_at")},
        }
        write(staging / "manifest.json", manifest)
        (staging / "README.md").write_text(
            "# 六份模型核验完成后的CSV浮点往返解析恢复\n\n"
            "六份v2独立模型核验均通过。原全局聚合在严格方向计数时失败：默认CSV解析产生约1e−17的浮点符号差异。"
            "新的聚合与报告使用round_trip读取新实验CSV，保留旧固定实验的历史解析方式。"
            "本次不重新finalize、不拟合／预测模型、不重新执行六份模型核验，也不改写原11份科学CSV。\n\n"
            "snapshots保存四份v3分析脚本与只读诊断。attempt_before完整保留第二次失败的管道状态、"
            "全局验证报告、聚合日志、trained根manifest与aggregation_validation。"
            "六份passed v2报告、24份verification CSV和11份科学CSV以原路径SHA绑定；不复制模型。"
            "原八脚本、三份v2脚本与整个v2修订目录保持原样。\n\n"
            "pipeline保留analysis_revision_manifest_sha256指向v2来源，另用post_validation_revision_manifest_sha256绑定本修订。"
            "只有新全局聚合通过、报告核验和论文构建完成后才更新整批完成状态。\n")
        for mapping in (original_failure, reports, verification, science, old_revision_files):
            assert all(sha(ROOT / name) == digest for name, digest in mapping.items())
        assert sources == {name: sha(CODE / name) for name in REVISION_SCRIPTS}
        assert sha(diagnosis) == diagnosis_hash
        assert not revision.exists()
        staging.rename(revision)
    except BaseException:
        if staging.exists(): shutil.rmtree(staging)
        raise
    frozen(out)
    print(json.dumps({"post_validation_revision_registered": rel(revision / "manifest.json"),
                      "post_validation_revision_manifest_sha256": sha(revision / "manifest.json"),
                      "passed_worker_reports": 6, "verification_csvs": 24, "scientific_csvs_unchanged": 11,
                      "training_and_model_validation_rerun": False}), flush=True)


def frozen(out):
    v2 = legacy.frozen(out)
    revision = out / REVISION_DIRECTORY
    manifest = read(revision / "manifest.json")
    assert manifest["schema_version"] == 1 and manifest["revision_id"] == REVISION_ID
    assert manifest["protocol_sha256"] == sha(out / "protocol.json")
    assert manifest["original_analysis_code_manifest_sha256"] == sha(out / "analysis_code_manifest.json")
    assert manifest["original_scripts_sha256"] == read(out / "analysis_code_manifest.json")["scripts_sha256"]
    assert manifest["v2_scripts_sha256"] == v2["scripts_sha256"]
    assert manifest["v2_analysis_revision_manifest_sha256"] == sha(out / legacy.REVISION_DIRECTORY / "manifest.json")
    assert manifest["v2_analysis_revision_files_sha256"] == inventory(out / legacy.REVISION_DIRECTORY)
    for flag in ("training_protocol_changed", "training_rerun", "model_validation_rerun", "scientific_csv_rewritten"):
        assert manifest[flag] is False
    assert manifest["new_scientific_csv_float_parser"] == "round_trip" and manifest["old_fixed_csv_parser_preserved"] is True
    assert set(manifest["scripts_sha256"]) == set(REVISION_SCRIPTS)
    snapshots = {rel(revision / "snapshots" / name): digest for name, digest in manifest["scripts_sha256"].items()}
    snapshots[rel(revision / "snapshots/roundtrip_diagnosis.json")] = manifest["roundtrip_diagnosis_snapshot_sha256"]
    assert manifest["snapshots_sha256"] == snapshots
    for name, digest in manifest["scripts_sha256"].items():
        assert sha(CODE / name) == digest == sha(revision / "snapshots" / name), name
    assert manifest["roundtrip_diagnosis_source_path"] == rel(out / DIAGNOSIS_FILE)
    assert manifest["roundtrip_diagnosis_snapshot_path"] == rel(revision / "snapshots/roundtrip_diagnosis.json")
    assert sha(ROOT / manifest["roundtrip_diagnosis_source_path"]) == manifest["roundtrip_diagnosis_source_sha256"] \
        == manifest["roundtrip_diagnosis_snapshot_sha256"] == sha(revision / "snapshots/roundtrip_diagnosis.json")
    reports, verification = passed_worker_evidence(out, v2)
    assert manifest["passed_worker_reports_sha256"] == reports
    assert manifest["verification_csv_sha256"] == verification
    assert manifest["scientific_csv_sha256"] == {rel(out / name): sha(out / name) for name in SCIENCE}
    paths = {rel(revision / "attempt_before" / name): rel(out / name) for name in FAILED_ATTEMPT_FILES}
    assert manifest["failed_attempt_original_paths"] == paths and set(manifest["failed_attempt_sha256"]) == set(paths)
    for name, digest in manifest["failed_attempt_sha256"].items():
        assert sha(ROOT / name) == digest, name
    return manifest


def subprocess_stage(out, status, script, args=(), *, builder=False):
    frozen(out)
    status.update(stage=script, stage_started_at=now())
    write(out / "completion_pipeline.json", status)
    log = out / ("completion_v3_" + Path(script).stem + ".log")
    command = [sys.executable, str(CODE / script), *([] if builder else ["--out", str(out)]), *args]
    status.setdefault("commands", []).append({"command": command, "log": rel(log), "started_at": now()})
    with log.open("a") as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    status["commands"][-1].update(returncode=result.returncode, ended_at=now())
    write(out / "completion_pipeline.json", status)
    assert result.returncode == 0, f"{script} failed; see {log}"
    frozen(out)


def run(out):
    revision = frozen(out)
    assert not (out / "paper_revision_20261008/validation.json").exists(), "Completed paper revision already exists"
    for archived, original in revision["failed_attempt_original_paths"].items():
        assert sha(ROOT / original) == revision["failed_attempt_sha256"][archived], "Current failed attempt changed after registration"
    status = {"status": "running", "started_at": now(), "stage": "post_validation_aggregate_v3",
              "analysis_code_manifest_sha256": sha(out / "analysis_code_manifest.json"),
              "analysis_revision_manifest": rel(out / legacy.REVISION_DIRECTORY / "manifest.json"),
              "analysis_revision_manifest_sha256": revision["v2_analysis_revision_manifest_sha256"],
              "analysis_revision_id": legacy.REVISION_ID,
              "post_validation_revision_manifest": rel(out / REVISION_DIRECTORY / "manifest.json"),
              "post_validation_revision_manifest_sha256": sha(out / REVISION_DIRECTORY / "manifest.json"),
              "post_validation_revision_id": REVISION_ID, "training_rerun": False, "model_validation_rerun": False,
              "reused_passed_worker_reports": 6, "scientific_csv_rewritten": False,
              "failed_attempt_archive": rel(out / REVISION_DIRECTORY / "attempt_before"),
              "failed_attempt_sha256": revision["failed_attempt_sha256"], "failed_attempt": revision["failed_attempt"]}
    path = out / "completion_pipeline.json"
    write(path, status)
    try:
        subprocess_stage(out, status, "verify_account_gmkep_tuning_v3.py", ("--aggregate-only",))
        validation = read(out / "validation_models.json")
        assert validation["passed"] is True and validation["complete_experiment"] is True
        assert validation["analysis_revision_manifest_sha256"] == status["analysis_revision_manifest_sha256"]
        assert validation["post_validation_revision_manifest_sha256"] == status["post_validation_revision_manifest_sha256"]
        assert validation["validator_sha256"] == revision["scripts_sha256"]["verify_account_gmkep_tuning_v3.py"]
        manifest = read(out / "run_manifest.json")
        manifest.update(status="complete", independent_validation_completed=True, completed_at=now(),
                        validation_models_sha256=sha(out / "validation_models.json"),
                        analysis_revision_manifest_sha256=status["analysis_revision_manifest_sha256"],
                        post_validation_revision_manifest_sha256=status["post_validation_revision_manifest_sha256"])
        write(out / "run_manifest.json", manifest)
        subprocess_stage(out, status, "report_account_gmkep_tuning_v3.py")
        assert out == OUT
        subprocess_stage(out, status, "build_tuned_paper_revision_v3.py", builder=True)
        check = read(out / "paper_revision_20261008/validation.json")
        assert check["passed"] is True
        frozen(out)
        status.update(status="complete", stage="complete", completed_at=now(),
                      validation_models_sha256=sha(out / "validation_models.json"),
                      paper_revision_validation_sha256=sha(out / "paper_revision_20261008/validation.json"),
                      paper_sha256=sha(ROOT / "PAPER_FRAMEWORK_IBM_AML_SMALL_GMKEP56_20261008.md"))
        write(path, status)
        print(now() + " COMPLETE: reused model verification, round-trip aggregate and paper revision", flush=True)
    except BaseException:
        status.update(status="failed", failed_at=now(), traceback=traceback.format_exc())
        write(path, status)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "run"))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    target = args.out.resolve()
    assert target == OUT, "This recovery targets the single post-validation failed batch"
    with (target / "completion_pipeline.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == "register": register(target)
        else: run(target)
