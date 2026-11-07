"""Reallocate finished boosting-worker slots to matched RF feature arms.

Inspection and copy verification are read-only.  ``supervise`` is the explicit
execution action: only after all four boosting workers have completed and
exited, it identity-checks and stops the two original RF workers, preserving
every checkpoint, and starts at most six RF workers with two fit threads each.
The registered trainer/protocol and all statistical choices remain unchanged.
"""
from __future__ import annotations

import argparse
import ast
import copy
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import uuid

import numpy as np
import pandas as pd
import psutil

import run_account_gmkep_tuning as trainer

ROOT, OUT = trainer.ROOT, trainer.OUT
DATASETS, ARMS = trainer.DATASETS, trainer.ARMS
TRAINER = Path(trainer.__file__).resolve()
SCRIPT = Path(__file__).resolve()
read, sha, rel, jsave, now = trainer.read, trainer.sha, trainer.rel, trainer.jsave, trainer.now
search, strategy_fit = trainer.search, trainer.strategy_fit


def _complete_fold(out, ds, name, arm, seed, fold, X, y, identity, p, design, folder, names):
    """The if/else science block is AST-identical to the registered trainer."""
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
    return done


def verify_copy(out: Path) -> dict:
    """Check copied operations and the original registered code without running fits."""
    protocol = trainer.registered(out)
    original = ast.parse(TRAINER.read_text())
    copied = ast.parse(SCRIPT.read_text())
    original_run = next(node for node in original.body if isinstance(node, ast.FunctionDef) and node.name == "run")
    copied_run = next(node for node in copied.body if isinstance(node, ast.FunctionDef) and node.name == "_complete_fold")
    def core(function):
        found = [node for node in ast.walk(function) if isinstance(node, ast.If)
                 and ast.dump(node.test, include_attributes=False) == "Call(func=Attribute(value=Name(id='path', ctx=Load()), attr='exists', ctx=Load()), args=[], keywords=[])"]
        assert len(found) == 1
        return ast.dump(found[0], include_attributes=False)
    a, b = core(original_run), core(copied_run)
    assert a == b, "Scheduler copied fold operations differ from the registered trainer"
    assert protocol["threads_per_worker"] == 2 and protocol["max_concurrent_workers"] == 6
    return {"passed": True, "trainer_sha256": sha(TRAINER), "scheduler_sha256": sha(SCRIPT),
            "protocol_sha256": sha(out / "protocol.json"),
            "fold_if_else_ast_sha256": hashlib.sha256(a.encode()).hexdigest(),
            "fit_threads_per_worker": 2, "max_concurrent_compute_workers": 6}


def process_info(pid: int) -> dict | None:
    try:
        process = psutil.Process(int(pid))
        return {"pid": process.pid, "ppid": process.ppid(), "create_time": process.create_time(),
                "status": process.status(), "cmdline": process.cmdline(), "cwd": process.cwd()}
    except psutil.NoSuchProcess:
        return None
    except psutil.ZombieProcess:
        return {"pid": int(pid), "status": "zombie"}
    except psutil.AccessDenied as error:
        raise RuntimeError(f"Cannot safely identify PID {pid}; no signal is permitted without identity") from error


def live(info: dict | None) -> bool:
    return info is not None and info.get("status") not in ("zombie", "dead")


def runner_arguments(info: dict, action: str, out: Path, dataset: str | None = None,
                     model: str | None = None) -> bool:
    argv = info.get("cmdline", [])
    if len(argv) < 3:
        return False
    script = Path(argv[1])
    if not script.is_absolute():
        script = Path(info["cwd"]) / script
    if script.resolve() != TRAINER or argv[2] != action:
        return False
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--out", type=Path, default=trainer.OUT)
    parser.add_argument("--dataset")
    parser.add_argument("--model")
    try:
        parsed, unknown = parser.parse_known_args(argv[3:])
    except (SystemExit, ValueError):
        return False
    target = parsed.out
    if not target.is_absolute():
        target = Path(info["cwd"]) / target
    return (not unknown and target.resolve() == out.resolve()
            and (dataset is None or parsed.dataset == dataset)
            and (model is None or parsed.model == model))


def _worker_info(out: Path, dataset: str, model: str) -> tuple[dict, dict | None]:
    manifest = read(out / dataset / f"run_manifest_{model}.json")
    assert manifest["protocol_sha256"] == sha(out / "protocol.json")
    info = process_info(int(manifest["pid"]))
    if live(info):
        assert runner_arguments(info, "run", out, dataset, model), f"PID identity mismatch: {dataset}/{model}"
    return manifest, info


def inspect(out: Path) -> dict:
    check = verify_copy(out)
    workers = {}
    ready = True
    for dataset in DATASETS:
        for model in ("lightgbm", "xgboost", "random_forest"):
            manifest, info = _worker_info(out, dataset, model)
            workers[f"{dataset}/{model}"] = {"status": manifest["status"],
                "completed_outer_settings": manifest["completed_outer_settings"], "process": info}
            if model != "random_forest":
                ready &= manifest["status"] == "complete" and manifest["completed_outer_settings"] == 75 and not live(info)
    return {"copy_check": check, "all_four_boosting_workers_complete_and_exited": bool(ready),
            "workers": workers, "read_only": True}


def checkpoint_snapshot(out: Path, dataset: str) -> dict:
    folder = out / dataset
    saved = {}
    for node in (folder / "search").glob("random_forest_*/group*_inner*.json"):
        content = read(node)
        assert content["protocol_sha256"] == sha(out / "protocol.json")
        for record in content["records"]:
            if record["physical_fit_charged"]:
                saved[record["physical_fit_id"]] = float(record["fit_seconds"])
    complete = sorted(rel(path) for path in (folder / "folds").glob("random_forest_*.json"))
    partial = [{"path": rel(path), "bytes": path.stat().st_size}
               for sub in ("search", "models", "calibration_models", "selected_search_models", "selected_search_scores")
               for path in (folder / sub).rglob("*.tmp")]
    return {"dataset": dataset, "snapshot_at": now(), "complete_outer_fold_jsons": complete,
            "complete_outer_settings": len(complete), "saved_physical_search_fit_records": len(saved),
            "saved_search_fit_seconds": sum(saved.values()), "partial_temporary_files": partial,
            "saved_fit_seconds_include_only_committed_group_records": True,
            "uncommitted_interrupted_compute_seconds": None,
            "uncommitted_interrupted_compute_is_assumed_zero": False,
            "note": "The scheduler deletes no committed checkpoint. Partial temporary files are inventoried; "
                    "the frozen trainer may overwrite them during its ordinary atomic retry. Work after the last "
                    "committed node may be repeated; its duration is not observable from these records."}


def _same_process(bound: dict, current: dict | None) -> bool:
    return live(current) and current["pid"] == bound["pid"] and current["create_time"] == bound["create_time"]


def _wait_stopped(bound: dict, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _same_process(bound, process_info(bound["pid"])):
            return True
        time.sleep(.5)
    return not _same_process(bound, process_info(bound["pid"]))


def stop_original_rf(out: Path, dataset: str, session: dict, schedule: dict,
                     schedule_path: Path, timeout: float) -> None:
    manifest, info = _worker_info(out, dataset, "random_forest")
    record = {"dataset": dataset, "manifest_before": manifest, "process_before": info,
              "checkpoint_before": checkpoint_snapshot(out, dataset), "signals": []}
    session["original_rf_workers"].append(record)
    jsave(schedule_path, schedule)
    if live(info):
        assert runner_arguments(info, "run", out, dataset, "random_forest")
        parent = process_info(info["ppid"])
        assert not live(parent) or runner_arguments(parent, "execute", out), "Live parent is not the original execute supervisor"
        record["original_execute_parent"] = parent
        current = process_info(info["pid"])
        if _same_process(info, current):
            assert runner_arguments(current, "run", out, dataset, "random_forest")
            target = psutil.Process(info["pid"])
            assert target.create_time() == info["create_time"]
            record["signals"].append({"signal": "SIGTERM", "requested_at": now(), "identity": current})
            jsave(schedule_path, schedule)
            try:
                # A cached psutil.Process checks creation time before signalling,
                # protecting against PID reuse between inspection and delivery.
                target.send_signal(signal.SIGTERM)
                record["signals"][-1]["delivery"] = "sent"
            except psutil.NoSuchProcess:
                record["signals"][-1]["delivery"] = "original_process_already_exited"
            except (psutil.AccessDenied, PermissionError) as error:
                record["signals"][-1]["delivery"] = "denied"
                record["stop_error"] = "Signal denied by process permissions; no replacement worker was started"
                jsave(schedule_path, schedule)
                raise RuntimeError(record["stop_error"]) from error
            if not _wait_stopped(info, timeout):
                current = process_info(info["pid"])
                assert _same_process(info, current) and runner_arguments(current, "run", out, dataset, "random_forest")
                record["signals"].append({"signal": "SIGKILL", "requested_at": now(), "identity": current})
                jsave(schedule_path, schedule)
                try:
                    target.send_signal(signal.SIGKILL)
                    record["signals"][-1]["delivery"] = "sent"
                except psutil.NoSuchProcess:
                    record["signals"][-1]["delivery"] = "original_process_already_exited"
                assert _wait_stopped(info, timeout), "Original RF process has not exited; replacements are forbidden"
        else:
            record["original_process_exited_before_signal"] = True
    record.update(stopped_confirmed_at=now(), original_process_no_longer_live=True,
                  checkpoint_after=checkpoint_snapshot(out, dataset))
    jsave(schedule_path, schedule)


def worker(out: Path, dataset: str, arm: str, schedule_id: str) -> None:
    check = verify_copy(out)
    schedule = read(out / "execution_schedule.json")
    session = schedule["sessions"][-1]
    assert session["schedule_id"] == schedule_id and session["original_rf_all_stopped"] is True
    assert schedule["scheduler_sha256"] == check["scheduler_sha256"]
    assert any(job["dataset"] == dataset and job["feature_set"] == arm for job in session["replacement_workers"])
    lockdir = out / "scheduler_locks"
    lockdir.mkdir(exist_ok=True)
    with (lockdir / f"{dataset}_random_forest_{arm}.lock").open("a") as arm_lock:
        fcntl.flock(arm_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        p = trainer.registered(out)
        folder = out / dataset
        input_metadata = read(out / "dataset_inputs.json")[dataset]
        assert all(sha(ROOT / name) == digest for name, digest in input_metadata["cache_sha256"].items())
        matrix = np.load(folder / "arrays/features.npy", mmap_mode="r")
        y = np.load(folder / "arrays/labels.npy", mmap_mode="r")
        identity = pd.read_parquet(folder / "accounts.parquet")
        names = p["feature_columns"][arm]
        X = np.ascontiguousarray(matrix[:, :len(names)])
        design = read(out / "candidates.json")
        manifest_dir = folder / "scheduler_manifests"
        manifest_dir.mkdir(exist_ok=True)
        manifest_path = manifest_dir / f"random_forest_{arm}.json"
        manifest = {"dataset": dataset, "model": "random_forest", "feature_set": arm,
                    "pid": os.getpid(), "status": "running", "started_at": now(),
                    "protocol_sha256": check["protocol_sha256"], "scheduler_sha256": check["scheduler_sha256"],
                    "schedule_id": schedule_id, "completed_outer_settings": 0, "expected_outer_settings": 25}
        jsave(manifest_path, manifest)
        try:
            for seed in range(5):
                for fold in range(5):
                    stem = f"random_forest_{arm}_seed{seed}_fold{fold}"
                    with (lockdir / f"{dataset}_{stem}.lock").open("a") as fold_lock:
                        fcntl.flock(fold_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        _complete_fold(out,dataset,"random_forest",arm,seed,fold,X,y,identity,p,design,folder,names)
                    manifest.update(completed_outer_settings=manifest["completed_outer_settings"] + 1,
                                    updated_at=now(), last_completed=stem)
                    jsave(manifest_path, manifest)
            assert manifest["completed_outer_settings"] == 25
            manifest.update(status="complete", completed_at=now())
            jsave(manifest_path, manifest)
        except BaseException:
            manifest.update(status="failed", failed_at=now(), traceback=traceback.format_exc())
            jsave(manifest_path, manifest)
            raise
        finally:
            del X, matrix, y, identity
            gc.collect()


def _completed_rf(out: Path, dataset: str, verify_hashes: bool = False) -> int:
    count = 0
    for arm in ARMS:
        for seed in range(5):
            for fold in range(5):
                path = out / dataset / "folds" / f"random_forest_{arm}_seed{seed}_fold{fold}.json"
                if not path.exists():
                    continue
                done = read(path)
                assert done["protocol_sha256"] == sha(out / "protocol.json")
                assert done["dataset"] == dataset and done["model"] == "random_forest" and done["feature_set"] == arm
                if verify_hashes:
                    for name, digest in done["artifact_sha256"].items():
                        assert sha(ROOT / name) == digest
                count += 1
    return count


def _mark_original_manifest(out: Path, dataset: str, status: str, session: dict) -> None:
    path = out / dataset / "run_manifest_random_forest.json"
    manifest = read(path)
    count = _completed_rf(out, dataset, verify_hashes=status == "complete")
    if status == "complete":
        assert count == 75
    manifest.update(status=status, completed_outer_settings=count, updated_at=now(),
                    execution_scheduler=rel(SCRIPT), execution_schedule_id=session["schedule_id"],
                    scheduler_pid=os.getpid(), original_worker_stopped_for_reschedule=True)
    if status == "complete":
        manifest["completed_at"] = now()
    jsave(path, manifest)


def supervise(out: Path, poll_seconds: float, stop_timeout: float) -> None:
    check = verify_copy(out)
    assert not (out / "validation_models.json").exists(), "Do not reschedule an independently sealed experiment"
    lockdir = out / "scheduler_locks"
    lockdir.mkdir(exist_ok=True)
    with (lockdir / "supervisor.lock").open("a") as supervisor_lock:
        fcntl.flock(supervisor_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        schedule_path = out / "execution_schedule.json"
        if schedule_path.exists():
            schedule = read(schedule_path)
            assert schedule["scheduler_sha256"] == check["scheduler_sha256"]
            assert schedule["trainer_sha256"] == check["trainer_sha256"]
            assert schedule["protocol_sha256"] == check["protocol_sha256"]
            for old_session in schedule["sessions"]:
                for job in old_session.get("replacement_workers", []):
                    if job.get("pid") and live(process_info(job["pid"])):
                        raise RuntimeError("A previous replacement worker is still live; no duplicate worker is launched")
        else:
            code_dir = out / "execution_code"
            code_dir.mkdir(exist_ok=True)
            snapshot = code_dir / f"rf_scheduler_{check['scheduler_sha256'][:12]}.py"
            if snapshot.exists():
                assert sha(snapshot) == check["scheduler_sha256"]
            else:
                snapshot.write_bytes(SCRIPT.read_bytes())
                assert sha(snapshot) == check["scheduler_sha256"]
            schedule = {"schema_version": 1, "created_at": now(), "scheduler": rel(SCRIPT),
                        "fold_copy_check_passed": check["passed"],
                        **{name: value for name, value in check.items() if name != "passed"},
                        "scheduler_snapshot": rel(snapshot), "scheduler_snapshot_sha256": sha(snapshot),
                        "trainer": rel(TRAINER), "sessions": [],
                        "statistical_protocol_changed": False,
                        "cost_accounting": "Science records count committed useful fits, not unmeasured interrupted "
                            "work. Execution history retains interruptions and partial files; total compute is not claimed."}
        session = {"schedule_id": uuid.uuid4().hex, "supervisor_pid": os.getpid(), "started_at": now(),
                   "status": "waiting_for_boosting", "events": [], "original_rf_workers": [],
                   "replacement_workers": [], "original_rf_all_stopped": False}
        schedule["sessions"].append(session)
        jsave(schedule_path, schedule)
        children, handles = [], []
        try:
            while True:
                state = inspect(out)
                session["last_read_only_inspection"] = state
                session["updated_at"] = now()
                jsave(schedule_path, schedule)
                failed_boosting = [name for name, item in state["workers"].items()
                                   if not name.endswith("/random_forest") and item["status"] == "failed"]
                assert not failed_boosting, f"Boosting workers failed; RF reallocation is withheld: {failed_boosting}"
                if state["all_four_boosting_workers_complete_and_exited"]:
                    break
                print(now() + " WAIT four boosting workers must complete and exit", flush=True)
                time.sleep(poll_seconds)
            session["status"] = "stopping_original_rf"
            session["events"].append({"event": "boosting_complete_and_exited", "at": now()})
            jsave(schedule_path, schedule)
            for dataset in DATASETS:
                stop_original_rf(out, dataset, session, schedule, schedule_path, stop_timeout)
            session["original_rf_all_stopped"] = True
            session["status"] = "rf_parallel_running"
            session["events"].append({"event": "all_original_rf_stopped_before_any_replacement", "at": now()})
            for dataset in DATASETS:
                for arm in ARMS:
                    session["replacement_workers"].append({"dataset": dataset, "model": "random_forest",
                        "feature_set": arm, "fit_threads": 2, "expected_outer_settings": 25,
                        "status": "planned", "log": rel(out / dataset / f"execution_random_forest_{arm}_rescheduled.log")})
            jsave(schedule_path, schedule)
            assert len(session["replacement_workers"]) == 6
            for job in session["replacement_workers"]:
                log = (ROOT / job["log"]).open("a")
                handles.append(log)
                command = [sys.executable, str(SCRIPT), "worker", "--out", str(out), "--dataset", job["dataset"],
                           "--feature-set", job["feature_set"], "--schedule-id", session["schedule_id"]]
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
                children.append((job, child))
                job.update(pid=child.pid, command=command, started_at=now(), status="running")
                jsave(schedule_path, schedule)
            while any(child.poll() is None for _, child in children):
                progress = []
                for job, child in children:
                    path = out / job["dataset"] / "scheduler_manifests" / f"random_forest_{job['feature_set']}.json"
                    manifest = read(path) if path.exists() else {}
                    progress.append(f"{job['dataset']}/{job['feature_set']}:{manifest.get('completed_outer_settings',0)}/25")
                    job["last_manifest"] = manifest
                    if child.poll() is not None:
                        job.update(returncode=child.returncode, status="complete" if child.returncode == 0 else "failed")
                for dataset in DATASETS:
                    _mark_original_manifest(out, dataset, "rescheduled_running", session)
                session["updated_at"] = now()
                jsave(schedule_path, schedule)
                print(now() + " " + " ".join(progress), flush=True)
                time.sleep(poll_seconds)
            failed = []
            for job, child in children:
                code = child.wait()
                job.update(returncode=code, completed_at=now(), status="complete" if code == 0 else "failed")
                manifest_path = out / job["dataset"] / "scheduler_manifests" / f"random_forest_{job['feature_set']}.json"
                if manifest_path.exists():
                    job["last_manifest"] = read(manifest_path)
                if code == 0:
                    assert job["last_manifest"]["status"] == "complete"
                    assert job["last_manifest"]["completed_outer_settings"] == 25
                if code:
                    failed.append(f"{job['dataset']}/{job['feature_set']}:{code}")
            assert not failed, f"Replacement RF worker failures: {failed}"
            for dataset in DATASETS:
                _mark_original_manifest(out, dataset, "complete", session)
            session.update(status="complete", completed_at=now(),
                           checkpoint_after={dataset: checkpoint_snapshot(out,dataset) for dataset in DATASETS})
            schedule["completed_at"] = now()
            jsave(schedule_path, schedule)
            assert verify_copy(out) == check, "Registered code or protocol changed during scheduling"
            print(now() + " RF RESCHEDULE COMPLETE: 150/150 outer settings", flush=True)
        except BaseException:
            session.update(status="failed", failed_at=now(), traceback=traceback.format_exc(),
                           still_live_replacement_pids=[child.pid for _,child in children if child.poll() is None])
            jsave(schedule_path, schedule)
            raise
        finally:
            for handle in handles:
                handle.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inspect", "verify-copy", "supervise", "worker"))
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--poll-seconds", type=float, default=30.)
    parser.add_argument("--stop-timeout", type=float, default=30.)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--feature-set", choices=ARMS)
    parser.add_argument("--schedule-id")
    args = parser.parse_args()
    out = args.out.resolve()
    assert out.is_relative_to((ROOT / "Altman_experiments/results").resolve())
    assert 1 <= args.poll_seconds <= 60 and 1 <= args.stop_timeout <= 60
    if args.action == "inspect":
        print(json.dumps(inspect(out), ensure_ascii=False, indent=2))
    elif args.action == "verify-copy":
        print(json.dumps(verify_copy(out), ensure_ascii=False, indent=2))
    elif args.action == "worker":
        assert args.dataset and args.feature_set and args.schedule_id
        worker(out,args.dataset,args.feature_set,args.schedule_id)
    else:
        supervise(out,args.poll_seconds,args.stop_timeout)


if __name__ == "__main__":
    main()
