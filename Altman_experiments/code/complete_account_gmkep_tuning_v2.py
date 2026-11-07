"""Recover the completed IBM Small batch after a dataset-namespace assertion.

This new analysis revision leaves the original eight registered scripts, trainer
protocol and trained artifacts unchanged. ``register`` only creates a separate
revision archive, sealing the failed attempt and three new analysis scripts.
``run`` uses the corrected finalizer/validator and the original report/builder.
It does not start training or alter candidate parameters, predictions or splits.
"""
from __future__ import annotations

import argparse
import fcntl
from datetime import datetime
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "Altman_experiments/code"
OUT = ROOT / "Altman_experiments/results/account_gmkep56_tuned_20261008"
DATASETS = ("hi_small", "li_small")
MODELS = ("lightgbm", "xgboost", "random_forest")
SCRIPTS = (
    "complete_account_gmkep_tuning.py", "run_account_gmkep_tuning.py",
    "schedule_account_gmkep_tuning_rf.py", "finalize_account_gmkep_tuning.py",
    "verify_account_gmkep_tuning.py", "report_account_gmkep_tuning.py",
    "build_tuned_paper_revision.py", "build_ibm_small_paper_framework.py",
)
REVISION_SCRIPTS = (
    "complete_account_gmkep_tuning_v2.py", "finalize_account_gmkep_tuning_v2.py",
    "verify_account_gmkep_tuning_v2.py",
)
REVISION_DIRECTORY = "analysis_revision_20261009"
REVISION_ID = "dataset_scoped_physical_fit_id_20261009"
FAILED_ATTEMPT_FILES = (
    "completion_pipeline.json", "completion_finalize_account_gmkep_tuning.log", "run_manifest.json",
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


def frozen_original(out):
    manifest = read(out / "analysis_code_manifest.json")
    assert set(manifest["scripts_sha256"]) == set(SCRIPTS), "Unexpected original registration"
    assert manifest["protocol_sha256"] == sha(out / "protocol.json")
    for name, digest in manifest["scripts_sha256"].items():
        assert sha(CODE / name) == digest and sha(out / "analysis_code_snapshots" / name) == digest, name
    return manifest


def register(out):
    """Create a new revision; never overwrite original registration/failure files."""
    original = frozen_original(out)
    revision = out / REVISION_DIRECTORY
    assert not revision.exists(), "Analysis revision already exists; registration never overwrites it"
    failure = read(out / "completion_pipeline.json")
    assert failure["status"] == "failed" and failure["stage"] == "finalize_account_gmkep_tuning.py"
    assert failure["analysis_code_manifest_sha256"] == sha(out / "analysis_code_manifest.json")
    assert read(out / "run_manifest.json")["status"] == "registered", "Only pre-aggregation recovery is supported"
    originals = [out / "analysis_code_manifest.json", out / "protocol.json"] + [out / name for name in FAILED_ATTEMPT_FILES]
    original_hashes = {rel(path): sha(path) for path in originals}
    audit_original = out / "qa/physical_fit_namespace_audit_20261009.json"
    audit = out / "namespace_audit.json"
    assert audit_original.is_file() and audit.is_file(), "Save the diagnosis and its identical root copy before registration"
    read(audit)  # Require a parseable diagnostic record, without claiming new model verification.
    source_hashes = {name: sha(CODE / name) for name in REVISION_SCRIPTS}
    audit_hash = sha(audit)
    assert sha(audit_original) == audit_hash, "Namespace diagnosis and root copy differ"
    staging = Path(tempfile.mkdtemp(prefix=".analysis_revision_20261009.register.", dir=out))
    try:
        snapshots, attempt = staging / "snapshots", staging / "attempt_before"
        snapshots.mkdir(); attempt.mkdir()
        for name in REVISION_SCRIPTS:
            shutil.copy2(CODE / name, snapshots / name)
            assert sha(snapshots / name) == source_hashes[name]
        shutil.copy2(audit, snapshots / "namespace_audit.json")
        assert sha(snapshots / "namespace_audit.json") == audit_hash
        for name in FAILED_ATTEMPT_FILES:
            shutil.copy2(out / name, attempt / name)
            assert sha(attempt / name) == original_hashes[rel(out / name)]
        snapshots_sha256 = {rel(revision / "snapshots" / name): source_hashes[name] for name in REVISION_SCRIPTS}
        snapshots_sha256[rel(revision / "snapshots" / "namespace_audit.json")] = audit_hash
        failed_sha256 = {rel(revision / "attempt_before" / name): sha(attempt / name) for name in FAILED_ATTEMPT_FILES}
        manifest = {
            "schema_version": 1, "revision_id": REVISION_ID, "registered_at": now(),
            "scope": "Analysis-only correction: physical fit IDs are unique within dataset; no training rerun or statistical change",
            "training_protocol_changed": False, "training_rerun": False,
            "physical_fit_id_scope": "dataset/physical_fit_id",
            "protocol_sha256": sha(out / "protocol.json"),
            "original_analysis_code_manifest_sha256": sha(out / "analysis_code_manifest.json"),
            "original_scripts_sha256": original["scripts_sha256"],
            "scripts_sha256": source_hashes, "snapshots_sha256": snapshots_sha256,
            "audit_source_path": rel(audit), "namespace_audit_source_sha256": audit_hash,
            "audit_original_path": rel(audit_original), "namespace_audit_original_sha256": audit_hash,
            "namespace_audit_snapshot_path": rel(revision / "snapshots" / "namespace_audit.json"),
            "namespace_audit_snapshot_sha256": audit_hash,
            "failed_attempt_sha256": failed_sha256,
            "failed_attempt_original_paths": {rel(revision / "attempt_before" / name): rel(out / name)
                                              for name in FAILED_ATTEMPT_FILES},
            "failed_attempt": {key: failure.get(key) for key in ("status", "stage", "started_at", "failed_at")},
            "failed_root_manifest_status": "registered",
        }
        write(staging / "manifest.json", manifest)
        (staging / "README.md").write_text(
            "# IBM Small调优：分析计数命名空间修订\n\n"
            "原finalizer在写科学CSV前检查physical_fit_id全局唯一性时失败。实际ID在数据集内唯一，"
            "HI与LI使用相同局部命名；修订仅按dataset与physical_fit_id共同计数。"
            "训练已完成，本次不重跑模型，不更改候选、划分、概率、统计评价或成本。\n\n"
            "原八脚本、原analysis_code_manifest、protocol及训练结果保持原样。"
            "snapshots保存三个v2脚本与只读namespace_audit；attempt_before保存原失败管道状态、"
            "失败finalizer日志及registered根manifest。注册不覆盖这些原始文件。"
            "运行v2后completion_pipeline.json和根run_manifest作为当前运行状态继续更新；"
            "manifest中的失败SHA只绑定封存副本，不要求当前状态仍等于失败时的内容。\n\n"
            "v2 finalizer/validator完成后仍使用原report与论文builder。"
            "所有验证通过之前不宣称最终实验或论文修订完成。\n")
        assert original_hashes == {name: sha(ROOT / name) for name in original_hashes}, "Original failure inputs changed during registration"
        assert source_hashes == {name: sha(CODE / name) for name in REVISION_SCRIPTS}
        assert sha(audit) == audit_hash
        assert sha(audit_original) == audit_hash
        assert not revision.exists()
        staging.rename(revision)
    except BaseException:
        if staging.exists(): shutil.rmtree(staging)
        raise
    frozen(out)
    print(json.dumps({"analysis_revision_registered": rel(revision / "manifest.json"),
                      "analysis_revision_manifest_sha256": sha(revision / "manifest.json"),
                      "original_registration_and_failure_files_unchanged": True,
                      "new_scripts": len(REVISION_SCRIPTS), "training_rerun": False}), flush=True)


def frozen(out):
    original = frozen_original(out)
    revision = out / REVISION_DIRECTORY
    manifest = read(revision / "manifest.json")
    assert manifest["schema_version"] == 1 and manifest["revision_id"] == REVISION_ID
    assert manifest["protocol_sha256"] == sha(out / "protocol.json")
    assert manifest["original_analysis_code_manifest_sha256"] == sha(out / "analysis_code_manifest.json")
    assert manifest["original_scripts_sha256"] == original["scripts_sha256"]
    assert manifest["training_protocol_changed"] is False and manifest["training_rerun"] is False
    assert set(manifest["scripts_sha256"]) == set(REVISION_SCRIPTS)
    expected_snapshots = {rel(revision / "snapshots" / name) for name in (*REVISION_SCRIPTS, "namespace_audit.json")}
    assert set(manifest["snapshots_sha256"]) == expected_snapshots
    for name, digest in manifest["scripts_sha256"].items():
        assert sha(CODE / name) == digest and sha(revision / "snapshots" / name) == digest, name
    for name, digest in manifest["snapshots_sha256"].items():
        assert sha(ROOT / name) == digest, name
    assert sha(ROOT / manifest["audit_source_path"]) == manifest["namespace_audit_source_sha256"]
    assert sha(ROOT / manifest["audit_original_path"]) == manifest["namespace_audit_original_sha256"]
    assert sha(ROOT / manifest["namespace_audit_snapshot_path"]) == manifest["namespace_audit_snapshot_sha256"]
    assert manifest["namespace_audit_original_sha256"] == manifest["namespace_audit_source_sha256"] == manifest["namespace_audit_snapshot_sha256"]
    expected_attempt = {rel(revision / "attempt_before" / name) for name in FAILED_ATTEMPT_FILES}
    assert set(manifest["failed_attempt_sha256"]) == expected_attempt
    for name, digest in manifest["failed_attempt_sha256"].items():
        assert sha(ROOT / name) == digest, name
    # Current completion_pipeline.json and run_manifest.json are mutable execution
    # state. Only their failed-attempt copies are bound here.
    return manifest


def subprocess_stage(out, status, script, args=()):
    frozen(out)
    status.update(stage=script, stage_started_at=now())
    write(out / "completion_pipeline.json", status)
    log = out / ("completion_v2_" + Path(script).stem + ".log")
    command = [sys.executable, str(CODE / script), "--out", str(out), *args]
    status.setdefault("commands", []).append({"command": command, "log": rel(log), "started_at": now()})
    with log.open("a") as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    status["commands"][-1].update(returncode=result.returncode, ended_at=now())
    write(out / "completion_pipeline.json", status)
    assert result.returncode == 0, f"{script} failed; see {log}"


def run(out):
    revision = frozen(out)
    assert not (out / "paper_revision_20261008/validation.json").exists(), "Completed paper revision already exists"
    # A single recovery attempt starts from exactly the registered failed state.
    # Further failed analysis attempts require a separately reviewed revision.
    for archived, original in revision["failed_attempt_original_paths"].items():
        assert sha(ROOT / original) == revision["failed_attempt_sha256"][archived], \
            "Current failure state changed since registration; do not overwrite another attempt"
    status = {"status": "running", "started_at": now(), "stage": "waiting_for_training",
              "analysis_code_manifest_sha256": sha(out / "analysis_code_manifest.json"),
              "analysis_revision_manifest": rel(out / REVISION_DIRECTORY / "manifest.json"),
              "analysis_revision_manifest_sha256": sha(out / REVISION_DIRECTORY / "manifest.json"),
              "analysis_revision_id": REVISION_ID, "training_rerun": False,
              "failed_attempt_archive": rel(out / REVISION_DIRECTORY / "attempt_before"),
              "failed_attempt_sha256": revision["failed_attempt_sha256"],
              "failed_attempt": revision["failed_attempt"]}
    path = out / "completion_pipeline.json"
    write(path, status)
    try:
        while True:
            states = {}
            for dataset, model in itertools.product(DATASETS, MODELS):
                file = out / dataset / f"run_manifest_{model}.json"
                m = read(file) if file.exists() else {}
                states[f"{dataset}/{model}"] = {"status": m.get("status"), "completed": m.get("completed_outer_settings", 0)}
                if model != "random_forest":
                    assert m.get("status") != "failed", f"Training failed: {dataset}/{model}"
            schedule = out / "execution_schedule.json"
            schedule_complete = True
            if schedule.exists():
                schedule_status = read(schedule)["sessions"][-1]["status"]
                assert schedule_status != "failed", "RF execution scheduler failed"
                schedule_complete = schedule_status == "complete"
            status.update(training_states=states, updated_at=now())
            write(path, status)
            if schedule_complete and all(value["status"] == "complete" and value["completed"] == 75 for value in states.values()):
                break
            print(now() + " WAIT training " + str(sum(value["completed"] for value in states.values())) + "/450", flush=True)
            time.sleep(30)
        subprocess_stage(out, status, "finalize_account_gmkep_tuning_v2.py")
        frozen(out)
        status.update(stage="independent_model_validation_v2", stage_started_at=now())
        write(path, status)
        jobs, handles = [], []
        for dataset, model in itertools.product(DATASETS, MODELS):
            log = out / dataset / f"execution_validation_v2_{model}.log"
            stream = log.open("a"); handles.append(stream)
            command = [sys.executable, str(CODE / "verify_account_gmkep_tuning_v2.py"), "--out", str(out),
                       "--dataset", dataset, "--model", model]
            jobs.append((dataset, model, subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)))
        while any(job.poll() is None for _, _, job in jobs):
            print(now() + " VALIDATING retained models and candidate rank audits (analysis v2)", flush=True)
            time.sleep(30)
        for stream in handles: stream.close()
        failed = [(dataset, model, job.returncode) for dataset, model, job in jobs if job.returncode]
        assert not failed, f"Independent worker validation failed: {failed}"
        subprocess_stage(out, status, "verify_account_gmkep_tuning_v2.py", ("--aggregate-only",))
        validation = read(out / "validation_models.json")
        assert validation["passed"] is True and validation["complete_experiment"] is True
        manifest = read(out / "run_manifest.json")
        manifest.update(status="complete", independent_validation_completed=True, completed_at=now(),
                        validation_models_sha256=sha(out / "validation_models.json"),
                        analysis_revision_manifest_sha256=sha(out / REVISION_DIRECTORY / "manifest.json"))
        write(out / "run_manifest.json", manifest)
        subprocess_stage(out, status, "report_account_gmkep_tuning.py")
        frozen(out)
        assert out == OUT
        status.update(stage="build_tuned_paper_revision.py", stage_started_at=now())
        write(path, status)
        log = out / "completion_v2_build_tuned_paper_revision.log"
        with log.open("a") as stream:
            result = subprocess.run([sys.executable, str(CODE / "build_tuned_paper_revision.py")], cwd=ROOT,
                                    stdout=stream, stderr=subprocess.STDOUT)
        assert result.returncode == 0, f"Paper revision failed; see {log}"
        check = read(out / "paper_revision_20261008/validation.json")
        assert check.get("passed") is True
        frozen(out)
        status.update(status="complete", stage="complete", completed_at=now(),
                      validation_models_sha256=sha(out / "validation_models.json"),
                      paper_revision_validation_sha256=sha(out / "paper_revision_20261008/validation.json"),
                      paper_sha256=sha(ROOT / "PAPER_FRAMEWORK_IBM_AML_SMALL_GMKEP56_20261008.md"))
        write(path, status)
        print(now() + " COMPLETE: preserved training, analysis v2 verification and paper revision", flush=True)
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
    assert target == OUT, "This recovery targets the single registered failed batch"
    # The same lock excludes the original pipeline and another recovery process.
    with (target / "completion_pipeline.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == "register": register(target)
        else: run(target)
