"""Complete and document a fully trained, frozen IBM Small tuning batch.

Waits for the training manifests, then aggregates, validates models in isolated
processes, builds the verified report and installs the paper revision. A failed
stage stops this pipeline; it never substitutes partial results for completion.
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


def now(): return datetime.now().astimezone().isoformat()
def read(path): return json.loads(path.read_text())
def sha(path):
    with path.open("rb") as stream: return hashlib.file_digest(stream, "sha256").hexdigest()
def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def register(out):
    path = out / "analysis_code_manifest.json"
    assert not path.exists(), "Analysis registration already exists"
    directory = out / "analysis_code_snapshots"
    directory.mkdir()
    hashes = {}
    for name in SCRIPTS:
        source = CODE / name
        shutil.copy2(source, directory / name)
        hashes[name] = sha(source)
        assert sha(directory / name) == hashes[name]
    write(path, {"registered_at": now(), "scripts_sha256": hashes,
                 "protocol_sha256": sha(out / "protocol.json"),
                 "scope": "Analysis and execution code snapshots; statistical trainer protocol unchanged"})
    print(json.dumps({"analysis_registered": str(path), "scripts": len(hashes)}), flush=True)


def frozen(out):
    manifest = read(out / "analysis_code_manifest.json")
    assert manifest["protocol_sha256"] == sha(out / "protocol.json")
    for name, digest in manifest["scripts_sha256"].items():
        assert sha(CODE / name) == digest and sha(out / "analysis_code_snapshots" / name) == digest, name


def subprocess_stage(out, status, script, args=()):
    frozen(out)
    status.update(stage=script, stage_started_at=now())
    write(out / "completion_pipeline.json", status)
    log = out / ("completion_" + Path(script).stem + ".log")
    command = [sys.executable, str(CODE / script), "--out", str(out), *args]
    status.setdefault("commands", []).append({"command": command, "log": str(log.relative_to(ROOT)), "started_at": now()})
    with log.open("a") as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    status["commands"][-1].update(returncode=result.returncode, ended_at=now())
    write(out / "completion_pipeline.json", status)
    assert result.returncode == 0, f"{script} failed; see {log}"


def run(out):
    frozen(out)
    assert not (out / "paper_revision_20261008/validation.json").exists(), "Completed paper revision already exists"
    status = {"status": "running", "started_at": now(), "stage": "waiting_for_training",
              "analysis_code_manifest_sha256": sha(out / "analysis_code_manifest.json")}
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
        subprocess_stage(out, status, "finalize_account_gmkep_tuning.py")
        frozen(out)
        status.update(stage="independent_model_validation", stage_started_at=now())
        write(path, status)
        jobs, handles = [], []
        for dataset, model in itertools.product(DATASETS, MODELS):
            log = out / dataset / f"execution_validation_{model}.log"
            stream = log.open("a"); handles.append(stream)
            command = [sys.executable, str(CODE / "verify_account_gmkep_tuning.py"), "--out", str(out),
                       "--dataset", dataset, "--model", model]
            jobs.append((dataset, model, subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)))
        while any(job.poll() is None for _, _, job in jobs):
            print(now() + " VALIDATING retained models and candidate rank audits", flush=True)
            time.sleep(30)
        for stream in handles: stream.close()
        failed = [(dataset, model, job.returncode) for dataset, model, job in jobs if job.returncode]
        assert not failed, f"Independent worker validation failed: {failed}"
        subprocess_stage(out, status, "verify_account_gmkep_tuning.py", ("--aggregate-only",))
        validation = read(out / "validation_models.json")
        assert validation["passed"] is True and validation["complete_experiment"] is True
        manifest = read(out / "run_manifest.json")
        manifest.update(status="complete", independent_validation_completed=True, completed_at=now(),
                        validation_models_sha256=sha(out / "validation_models.json"))
        write(out / "run_manifest.json", manifest)
        subprocess_stage(out, status, "report_account_gmkep_tuning.py")
        # The paper builder intentionally uses the single registered experiment.
        frozen(out)
        assert out == OUT
        status.update(stage="build_tuned_paper_revision.py", stage_started_at=now())
        write(path, status)
        log = out / "completion_build_tuned_paper_revision.log"
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
        print(now() + " COMPLETE: tuning, independent verification and paper revision", flush=True)
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
    assert target.is_relative_to((ROOT / "Altman_experiments/results").resolve())
    if args.action == "register":
        register(target)
    else:
        with (target / "completion_pipeline.lock").open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            run(target)
