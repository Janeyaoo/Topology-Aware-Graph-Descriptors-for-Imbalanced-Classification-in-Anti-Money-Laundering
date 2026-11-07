"""Versioned independent acceptance for the 20261029 remaining tree settings.

Loads the SHA-pinned 20261013 independent verifier, never a training module.
Its retained-model inference, rank/tie reconstruction, split checks and fold
arithmetic are reused unchanged. The explicitly new source gate checks the new
scientific closure, not the old manuscript/reporting closure. Legacy receipts
are neither edited nor promoted under a new protocol. Only acceptance outputs
in this run's validation directory are written; no fit or restart is performed.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import tempfile
from datetime import datetime, timezone

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "2")

FROZEN_VERIFIER_SHA = "66a84982c5569e6e7db803c2b6fbb8671beef9c67aa307c0d287b4e648abb03f"
HISTORICAL_PROTOCOL_SHA = "82fa9f190f944c11d10a243482720f6a0d263d5ce98009160e395117e92b73b4"
SCRIPT = Path(__file__).resolve()
DEFAULT_RUN = "Altman_experiments/results/account_eswa_remaining_20261029/run"
DEFAULT_ORIGINAL = "Altman_experiments/code/verify_account_eswa_supplement_20261013.py"
IDENTITY_FIELDS = ("dataset", "model", "arm", "seed", "fold")
STRATEGIES = ("same_protocol_anchor", "tuned")
POLICIES = ("fixed_0.5", "inner_max_f1")
# This is a new, separately registered batch. Counts used in acceptance and
# reporting are read from its budget, after checking this scope contract.
REMAINING_BUDGET = {
    "settings": 46, "planned_outer": 1150, "reused_outer": 0,
    "new_outer": 1150, "candidates": 12, "inner_folds": 3,
}
# Only execution/provenance fields of the old protocol can differ. In
# particular threads, candidate parameters, storage, feature columns, seeds,
# weighting, early stopping, selection, and inference are not exempted.
EXECUTION_FIELDS = {
    "experiment", "registered_at", "environment", "trainer_sha256",
    "source_manifest_sha256", "dataset_inputs_sha256", "split_manifest_sha256",
    "expected_outer_settings", "logical_candidate_evaluations",
    "expected_search_physical_fits", "expected_representative_refits",
    "expected_strategy_final_models_logical", "expected_strategy_threshold_models_logical",
    "max_concurrent_workers",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def checked_budget(registration):
    budget = registration.get("budget")
    require(isinstance(budget, dict), "Missing registered budget")
    for key, expected in REMAINING_BUDGET.items():
        require(type(budget.get(key)) is int and budget[key] == expected,
                "Remaining batch budget mismatch: " + key)
    require(budget["planned_outer"] == budget["settings"] * 25,
            "Setting/fold budget mismatch")
    require(budget["new_outer"] + budget["reused_outer"] == budget["planned_outer"],
            "New/imported fold budget mismatch")
    return budget.copy()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def atomic_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        frame.to_csv(temp, index=False)
        with open(temp, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temp, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def load_original(workspace, source):
    source = Path(source)
    require(digest(source) == FROZEN_VERIFIER_SHA, "Independent legacy verifier SHA changed")
    spec = importlib.util.spec_from_file_location("_flowtopo_frozen_independent_20261013", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = workspace
    module.ALT = workspace / "Altman_experiments"
    module.MAIN = module.ALT / "results/account_gmkep56_tuned_20261008"
    # This causes new receipts to identify this wrapper, never the old verifier.
    module.SCRIPT = SCRIPT
    module.HASH_CACHE.clear()
    module._remote_receipt_context = None

    def write_new(path, value):
        context = module._remote_receipt_context
        if context and isinstance(value, dict) and "protocol_sha256" in value and "physical_models_rescored" in value:
            value.update(context)
        atomic_json(path, value)

    module.write = write_new
    return module


class Acceptance:
    def __init__(self, workspace, out, registration):
        self.workspace = Path(workspace).resolve()
        self.out = Path(out).resolve()
        require(self.out.is_relative_to(self.workspace), "Run is outside workspace")
        self.registration_path = Path(registration).resolve()
        require(self.registration_path.is_relative_to(self.workspace), "Registration outside workspace")
        self.registration = read(self.registration_path)
        r = self.registration
        require(r.get("schema_version") == 1, "Unsupported registration schema")
        require(self.resolve(r["run_dir"], must_exist=False) == self.out, "Registration run directory mismatch")
        require(digest(SCRIPT) == r.get("wrapper_sha256"), "Wrapper changed after registration")
        self.registration_sha = digest(self.registration_path)
        self.protocol_sha = digest(self.out / "trees/protocol.json")
        require(self.protocol_sha == r["protocol_sha256"], "New scientific protocol changed after registration")
        frozen = self.checked(r["frozen_verifier_path"], r["frozen_verifier_sha256"])
        require(r["frozen_verifier_sha256"] == FROZEN_VERIFIER_SHA, "Unexpected numerical verifier")
        self.base = load_original(self.workspace, frozen)
        old_path = self.checked(r["historical_protocol_path"], r["historical_protocol_sha256"])
        require(r["historical_protocol_sha256"] == HISTORICAL_PROTOCOL_SHA, "Unexpected original scientific protocol")
        self.historical_protocol = read(old_path)
        manifest_path = self.checked(r["source_manifest_path"], r["source_manifest_sha256"])
        require(manifest_path == self.out / "trees/source_manifest.json", "New scientific manifest location")
        self.sources = read(manifest_path)["source_sha256"]
        require(isinstance(self.sources, dict) and bool(self.sources), "Empty scientific source closure")
        for path, sha in self.sources.items():
            self.checked(path, sha)
        require(self.relative(frozen) in self.sources and self.sources[self.relative(frozen)] == FROZEN_VERIFIER_SHA,
                "Frozen verifier absent from scientific closure")
        self.budget = checked_budget(r)
        self.expected_settings = self.budget["settings"]
        self.expected_outer_folds = self.budget["planned_outer"]
        self.reused_outer_folds = self.budget["reused_outer"]
        self.new_outer_folds = self.budget["new_outer"]
        code = r.get("code_sha256")
        require(isinstance(code, dict) and bool(code), "Missing registered code closure")
        require(self.relative(SCRIPT) in code and r["trainer_path"] in code,
                "Wrapper/trainer absent from registered code closure")
        for path, sha in code.items():
            require(self.sources.get(path) == sha, "Registered code absent from scientific closure: " + path)
            self.checked(path, sha)
        tasks_path = self.checked(r["task_matrix_path"], r["task_matrix_sha256"])
        require(self.sources.get(self.relative(tasks_path)) == r["task_matrix_sha256"],
                "Task matrix absent from scientific closure")
        self.tasks = read(tasks_path)
        require(isinstance(self.tasks, list) and len(self.tasks) == self.expected_settings, "Registered setting count differs from budget")
        self.settings = set()
        for row in self.tasks:
            key = (row["dataset"], row["model"], row["arm"])
            require(row["dataset"] in ("hi_small", "li_small") and row["model"] in self.base.MODELS, "Task scope")
            require(row["arm"] in self.historical_protocol["feature_arm_priority"], "Unregistered feature arm")
            require(key not in self.settings, "Duplicate setting")
            self.settings.add(key)
        self.imports = {}
        require(isinstance(r.get("legacy_imports"), list), "Legacy import list must be explicit")
        for row in r["legacy_imports"]:
            key = tuple(row[k] for k in IDENTITY_FIELDS)
            require(key[:3] in self.settings and key[3] in range(5) and key[4] in range(5), "Legacy task outside scope")
            require(key not in self.imports, "Duplicate imported fold")
            self.imports[key] = row
        require(len(self.imports) == self.reused_outer_folds, "Imported fold count differs from budget")
        self.validators = {}
        self.verifier_sha = digest(SCRIPT)

    def relative(self, path):
        return Path(path).resolve().relative_to(self.workspace).as_posix()

    def resolve(self, path, must_exist=True):
        require(not Path(path).is_absolute(), "Bound paths must be workspace relative")
        value = (self.workspace / path).resolve()
        require(value.is_relative_to(self.workspace), "Artifact escapes workspace: " + str(path))
        if must_exist:
            require(value.is_file(), "Missing bound file: " + str(path))
        return value

    def checked(self, path, sha):
        value = self.resolve(path)
        require(digest(value) == sha, "Source/artifact SHA mismatch: " + str(path))
        return value

    def source_required(self, path):
        value = self.resolve(path)
        key = self.relative(value)
        require(key in self.sources, "Required scientific source not in new closure: " + key)
        self.checked(key, self.sources[key])
        return value

    def stamp(self, legacy=False):
        return dict(remote_acceptance_schema=1, verifier_sha256=self.verifier_sha,
                    frozen_numerical_verifier_sha256=FROZEN_VERIFIER_SHA,
                    remote_registration_path=self.relative(self.registration_path),
                    remote_registration_sha256=self.registration_sha,
                    remote_protocol_sha256=self.protocol_sha,
                    scientific_source_manifest_sha256=self.registration["source_manifest_sha256"],
                    acceptance_mode="imported_legacy_protocol" if legacy else "new_remote_protocol",
                    report_provenance=self.registration.get("report_provenance", {}),
                    reporting_exception_policy="None. New scientific sources are all strictly SHA checked. Historical report drift is disclosure only.")

    def valid_receipt(self, receipt, done, legacy=False):
        if not receipt.exists() or not done.exists():
            return False
        value = read(receipt)
        expected_protocol = HISTORICAL_PROTOCOL_SHA if legacy else self.protocol_sha
        return (value.get("passed") is True and value.get("protocol_sha256") == expected_protocol
                and read(done).get("protocol_sha256") == expected_protocol
                and value.get("fold_done_sha256") == digest(done)
                and all(value.get(k) == v for k, v in self.stamp(legacy).items()))

    def validator(self, legacy=False):
        require(not legacy or bool(self.imports), "No legacy folds are registered for this batch")
        if legacy in self.validators:
            return self.validators[legacy]
        owner = self
        base = self.base
        out = self.out if not legacy else self.resolve(next(iter(self.imports.values()))["out_dir"], must_exist=False)

        class ScopedValidator(base.Validator):
            def source_gate(v):
                p = v.p
                for stem in ("candidates", "dataset_inputs", "split_manifest", "source_manifest"):
                    base.require(base.sha(v.tree / (stem + ".json")) == p[stem + "_sha256"], "Protocol registry " + stem)
                for key, value in owner.historical_protocol.items():
                    if key not in EXECUTION_FIELDS:
                        base.require(p.get(key) == value, "Scientific protocol changed: " + key)
                if legacy:
                    entry = next(iter(owner.imports.values()))
                    base.require(v.psha == entry["protocol_sha256"] == HISTORICAL_PROTOCOL_SHA, "Imported original protocol")
                    base.require(base.sha(v.tree / "source_manifest.json") == entry["source_manifest_sha256"], "Imported historical manifest bytes")
                    base.require(owner.relative(v.tree / "source_manifest.json") in owner.sources, "Historical manifest absent from new provenance closure")
                else:
                    base.require(p["expected_outer_settings"] == owner.expected_outer_folds and p["task_matrix_sha256"] == owner.registration["task_matrix_sha256"], "Scoped task budget")
                    trainer = owner.source_required(owner.registration["trainer_path"])
                    base.require(base.sha(trainer) == p["trainer_sha256"], "New execution adapter source")
                v.src.update(owner.sources)
                inherited = owner.source_required("Altman_experiments/code/run_account_gmkep_tuning.py")
                helper = owner.source_required("Altman_experiments/code/run_account_feature_mechanism.py")
                base.require(base.sha(inherited) == p["inherited_trainer_sha256"] and base.sha(helper) == p["helper_sha256"], "Frozen fit/search/metrics sources")
                old_protocol = owner.source_required("Altman_experiments/results/account_gmkep56_tuned_20261008/protocol.json")
                base.require(base.sha(old_protocol) == p["old_experiment_protocol_sha256"], "Original main scientific protocol")
                original_candidates = owner.source_required("Altman_experiments/results/account_gmkep56_tuned_20261008/candidates.json")
                base.require(v.c == base.read(original_candidates) and all([x["candidate_id"] for x in v.c[m]] == list(range(12)) for m in base.MODELS), "Shared twelve candidates")
                base.require(len(v.splits) == 50 and p["seeds"] == list(range(5)) and p["outer_n_splits"] == 5 and p["inner_n_splits"] == 3, "Full inherited split dimensions")
                # Reproduce the original independent AST audit without importing
                # or executing either fitting source.
                fun = {x.name: x for x in ast.parse(inherited.read_text()).body if isinstance(x, ast.FunctionDef)}
                fitting = fun["fit"]
                calls = [x for x in ast.walk(fitting) if isinstance(x, ast.Call) and isinstance(x.func, ast.Attribute) and x.func.attr == "fit"]
                base.require(len(calls) == 3 and all([ast.unparse(a) for a in x.args] == ["X[rows]", "y[rows]"] for x in calls), "AST fit-only rows")
                text = ast.unparse(fitting)
                base.require("compute_sample_weight('balanced', y[rows])" in text and "sample_weight_eval_set" not in text and "eval_sample_weight" not in text, "AST weighting/eval scope")
                stages = [x for x in ast.walk(fun["strategy_fit"]) if isinstance(x, ast.Call) and isinstance(x.func, ast.Name) and x.func.id == "fit"]
                base.require({tuple(ast.unparse(a) for a in x.args) for x in stages} == {("name", "X", "y", "D", "None", "params", "random_state", "rounds"), ("name", "X", "y", "train", "None", "params", "random_state", "rounds")}, "AST disjoint staging")
                text = ast.unparse(fun["search"])
                base.require("outer_test" not in text and "threshold_validation" not in text, "AST search isolation")

            def matrix(v, ds, arm):
                if arm == "base_gfp_account_stream":
                    # GFP is a registered deferred input. Its large payload need
                    # not be in the initial transfer or scientific manifest;
                    # the already SHA-bound original cache manifest commits its
                    # exact path, bytes and all 696 column names in advance.
                    manifest = owner.source_required(
                        f"Altman_experiments/results/account_eswa_supplement_20261013/inputs/{ds}/gfp_float32_cache_manifest.json")
                    cache = read(manifest)
                    reg = read(v.tree / ds / "input_registrations/gfp.json")
                    base.require(cache.get("passed") is True and cache.get("dataset") == ds
                                 and cache.get("n_features") == 696 and cache.get("storage_dtype") == "float32",
                                 "Original GFP cache contract")
                    base.require(reg["source_path"] == cache["cache_path"]
                                 and reg["columns"] == cache["feature_columns"], "GFP original path and full schema")
                    base.require(base.sha(owner.resolve(reg["source_path"])) == cache["cache_sha256"], "GFP original cache bytes")
                    base.require(reg["artifact_sha256"].get(reg["source_path"]) == cache["cache_sha256"]
                                 and all(reg[k] in reg["artifact_sha256"] for k in ("acceptance_path", "array_path")),
                                 "GFP source, admission and array must all be hash bound")
                    base.require(cache["n_accounts"] == v.inputs[ds]["n_samples"], "GFP complete account population")
                # Frozen implementation additionally checks deferred protocol,
                # all artifact hashes, passing acceptance, full account IDs,
                # and every float32 source-parquet/array cell, then selects X.
                return super().matrix(ds, arm)

            def fold(v, ds, name, arm, seed, fold, skip=True):
                identity = (ds, name, arm, seed, fold)
                base.require(identity[:3] in owner.settings, "Fold setting outside registered remaining settings")
                base.require(type(seed) is int and seed in range(5)
                             and type(fold) is int and fold in range(5), "Fold index outside registered splits")
                base.require((identity in owner.imports) == legacy, "New/legacy fold provenance mismatch")
                fp = v.tree / ds / "folds" / f"{name}_{arm}_seed{seed}_fold{fold}.json"
                rp = v.dest / ds / fp.name
                if legacy:
                    entry = owner.imports[identity]
                    base.require(fp == owner.resolve(entry["fold_done_path"]) and base.sha(fp) == entry["fold_done_sha256"], "Imported fold seal")
                if skip and not owner.valid_receipt(rp, fp, legacy):
                    skip = False
                before = base._remote_receipt_context
                base._remote_receipt_context = owner.stamp(legacy)
                try:
                    return super().fold(ds, name, arm, seed, fold, skip=skip)
                finally:
                    base._remote_receipt_context = before

        value = ScopedValidator(out)
        value.dest = self.out / "validation" / ("imported" if legacy else "tree")
        self.validators[legacy] = value
        return value

    def locations(self, identity):
        ds, model, arm, seed, fold = identity
        legacy = identity in self.imports
        out = self.out if not legacy else self.resolve(self.imports[identity]["out_dir"], must_exist=False)
        name = f"{model}_{arm}_seed{seed}_fold{fold}.json"
        return out / "trees" / ds / "folds" / name, self.out / "validation" / ("imported" if legacy else "tree") / ds / name, legacy

    def overview(self, write=True):
        settings = []
        for ds, model, arm in sorted(self.settings):
            done_count = passed_count = imported_count = 0
            for seed, fold in itertools.product(range(5), range(5)):
                fp, rp, legacy = self.locations((ds, model, arm, seed, fold))
                done_count += int(fp.exists())
                good = self.valid_receipt(rp, fp, legacy)
                passed_count += int(good)
                imported_count += int(good and legacy)
            settings.append(dict(dataset=ds, model=model, feature_set=arm, training_done_folds=done_count,
                                 independently_passed_folds=passed_count, imported_verified_folds=imported_count,
                                 expected_folds=25, status="complete_independently_verified" if passed_count == 25 else "partial_not_complete"))
        result = dict(status="complete" if sum(x["independently_passed_folds"] for x in settings) == self.expected_outer_folds else "partial",
                      created_at=datetime.now(timezone.utc).isoformat(), expected_settings=self.expected_settings, expected_outer_folds=self.expected_outer_folds,
                      expected_new_outer_folds=self.new_outer_folds, expected_imported_outer_folds=self.reused_outer_folds,
                      training_done_folds=sum(x["training_done_folds"] for x in settings),
                      independently_passed_folds=sum(x["independently_passed_folds"] for x in settings),
                      complete_settings=sum(x["independently_passed_folds"] == 25 for x in settings),
                      settings=settings, no_partial_means_reported=True, **self.stamp())
        if write:
            atomic_json(self.out / "validation/overview.json", result)
        return result

    def scan(self, max_folds=None, force=False):
        processed = []
        for setting in sorted(self.settings):
            for seed, fold in itertools.product(range(5), range(5)):
                identity = (*setting, seed, fold)
                fp, rp, legacy = self.locations(identity)
                if not fp.exists() or (not force and self.valid_receipt(rp, fp, legacy)):
                    continue
                if max_folds is not None and len(processed) >= max_folds:
                    break
                self.validator(legacy).fold(*identity, skip=not force)
                processed.append(dict(identity=list(identity), receipt_path=self.relative(rp), imported=legacy))
                print(json.dumps(dict(event="fold_independently_verified", **processed[-1])), flush=True)
        result = self.overview()
        atomic_json(self.out / "validation/last_scan.json", dict(processed=processed, overview=result, **self.stamp()))
        return result

    def aggregate(self):
        import numpy as np
        import pandas as pd
        overview = self.overview()
        complete = [r for r in overview["settings"] if r["independently_passed_folds"] == 25]
        rows, seeds, source_seals = [], [], {}
        for setting in complete:
            ds, model, arm = (setting[k] for k in ("dataset", "model", "feature_set"))
            for strategy, seed in itertools.product(STRATEGIES, range(5)):
                pieces = []
                for fold in range(5):
                    identity = (ds, model, arm, seed, fold)
                    fp, rp, legacy = self.locations(identity)
                    require(self.valid_receipt(rp, fp, legacy), "Missing current receipt during aggregation")
                    receipt = read(rp)
                    self.base.binding(receipt["verified_artifact_sha256"])
                    source_seals[self.relative(fp)] = digest(fp)
                    source_seals[self.relative(rp)] = digest(rp)
                    d = read(fp)
                    st = d["strategies"][strategy]
                    prediction = self.resolve(st["prediction_path"])
                    require(digest(prediction) == st["artifact_sha256"][self.relative(prediction)], "Aggregate prediction hash")
                    table = pd.read_parquet(prediction)
                    pieces.append(table)
                    rows.extend(st["metrics"])
                oof = pd.concat(pieces).sort_values("row_index")
                validator = self.validator(False)
                _, y, ids, _ = validator.dataset(ds)
                require(len(oof) == len(y) and oof.row_index.is_unique and np.array_equal(oof.row_index, np.arange(len(y)))
                        and np.array_equal(oof.account_id, ids.account_id) and np.array_equal(oof.y_true, y), "Full OOF identity/truth")
                for policy, column in (("fixed_0.5", "y_pred_fixed"), ("inner_max_f1", "y_pred_inner")):
                    seeds.append(dict(dataset=ds, model=model, feature_set=arm, strategy=strategy, seed=seed,
                                      threshold_policy=policy, **self.base.metrics(y, oof.y_score.to_numpy(), oof[column].to_numpy())))
        output = self.out / "validation/complete_settings"
        output.mkdir(parents=True, exist_ok=True)
        df, sd = pd.DataFrame(rows), pd.DataFrame(seeds)
        if len(df):
            atomic_csv(df, output / "verified_fold_metrics.csv")
            atomic_csv(sd, output / "verified_seed_oof_metrics.csv")
            keys = ["dataset", "model", "feature_set", "strategy", "threshold_policy"]
            summary = df.groupby(keys)[list(self.base.METRICS)].agg(["mean", "std"])
            summary.columns = ["_".join(x) for x in summary.columns]
            atomic_csv(summary.reset_index(), output / "verified_summary.csv")
        pair_report = self.aggregate_pairs(df, sd, output)
        result = dict(passed=True, status="complete" if len(complete) == self.expected_settings else "partial_complete_settings_only",
                      created_at=datetime.now(timezone.utc).isoformat(), complete_settings=len(complete), expected_settings=self.expected_settings,
                      expected_outer_folds=self.expected_outer_folds, expected_new_outer_folds=self.new_outer_folds,
                      expected_imported_outer_folds=self.reused_outer_folds,
                      fold_rows=len(rows), seed_oof_rows=len(seeds), source_done_and_receipt_sha256=source_seals,
                      historical_metrics_policy="Reference metrics are reused from SHA-bound previously accepted CSVs; their original models are not newly rescored here.",
                      no_partial_means_reported=True, **pair_report, **self.stamp(),
                      output_sha256={self.relative(p): digest(p) for p in output.glob("*.csv")})
        atomic_json(output / "validation.json", result)
        return result

    def aggregate_pairs(self, fresh, fresh_oof, output):
        import pandas as pd
        folds, oofs = [fresh], [fresh_oof]
        references = []
        for entry in self.registration.get("reference_metrics", []):
            f = self.checked(entry["fold_path"], entry["fold_sha256"])
            s = self.checked(entry["oof_path"], entry["oof_sha256"])
            require(self.relative(f) in self.sources and self.relative(s) in self.sources, "Reference CSV absent from scientific closure")
            ff, ss = pd.read_csv(f), pd.read_csv(s)
            if "n_test" not in ss and "n_samples" in ss:
                ss = ss.rename(columns={"n_samples": "n_test"})
            folds.append(ff)
            oofs.append(ss)
            references.append(dict(label=entry["label"], fold_path=self.relative(f), fold_sha256=digest(f), oof_path=self.relative(s), oof_sha256=digest(s)))
        frames = pd.concat(folds, ignore_index=True) if any(len(x) for x in folds) else pd.DataFrame()
        oo = pd.concat(oofs, ignore_index=True) if any(len(x) for x in oofs) else pd.DataFrame()
        registry_path = self.registration.get("feature_registry_path", "Altman_experiments/results/eswa_supplement_design_20261013/feature_registry.json")
        registry = read(self.source_required(registry_path))
        contrasts = sum((registry[k] for k in ("module_contrasts", "ph_primary_contrasts", "ph_secondary_contrasts", "external_tree_contrasts")), [])
        contrasts += [dict(contrast_id="struct44_vs_g32", family="graph32_control", lhs="gmke44", rhs="base_g32"),
                      dict(contrast_id="full56_vs_g32", family="graph32_control", lhs="gmkep56", rhs="base_g32")]
        pairs, seedpairs, skipped = [], [], []
        new_settings = {(r.dataset, r.model, r.feature_set) for r in fresh.itertuples()} if len(fresh) else set()
        metric_keys = list(self.base.METRICS)

        def selected(frame, ds, model, arm, policy):
            if not len(frame):
                return frame
            return frame[(frame.dataset == ds) & (frame.model == model) & (frame.feature_set == arm)
                         & (frame.strategy == "tuned") & (frame.threshold_policy == policy)].copy()

        for ds, model in sorted({(a, b) for a, b, _ in self.settings}):
            for contrast in contrasts:
                lhs, rhs = contrast["lhs"], contrast["rhs"]
                if (ds, model, lhs) not in new_settings and (ds, model, rhs) not in new_settings:
                    continue
                for policy in POLICIES:
                    a, b = (selected(frames, ds, model, arm, policy) for arm in (lhs, rhs))
                    aa, bb = (selected(oo, ds, model, arm, policy) for arm in (lhs, rhs))
                    unique = lambda f, keys, count: len(f) == count and not f.duplicated(keys).any()
                    if not (unique(a, ["seed", "fold"], 25) and unique(b, ["seed", "fold"], 25)
                            and unique(aa, ["seed"], 5) and unique(bb, ["seed"], 5)):
                        skipped.append(dict(dataset=ds, model=model, contrast_id=contrast["contrast_id"], threshold_policy=policy,
                                            reason="Both full 25-fold and five complete OOF metric sets are required; missing or duplicate sources are not pooled."))
                        continue
                    a, b = a.sort_values(["seed", "fold"]), b.sort_values(["seed", "fold"])
                    aa, bb = aa.sort_values("seed"), bb.sort_values("seed")
                    require(list(zip(a.seed, a.fold)) == list(itertools.product(range(5), range(5))) == list(zip(b.seed, b.fold)), "Paired fold population")
                    require(list(aa.seed) == list(range(5)) == list(bb.seed), "Paired OOF seed population")
                    common = dict(dataset=ds, model=model, contrast_id=contrast["contrast_id"], family=contrast["family"],
                                  lhs_arm=lhs, rhs_arm=rhs, threshold_policy=policy)
                    for left, right in zip(a.to_dict("records"), b.to_dict("records")):
                        require(left["n_test"] == right["n_test"] and left["n_positive"] == right["n_positive"], "Paired population counts")
                        pairs.append(dict(**common, seed=int(left["seed"]), fold=int(left["fold"]),
                                          **{"delta_" + k: float(left[k] - right[k]) for k in metric_keys}))
                    for left, right in zip(aa.to_dict("records"), bb.to_dict("records")):
                        require(left["n_test"] == right["n_test"] and left["n_positive"] == right["n_positive"], "Paired OOF counts")
                        seedpairs.append(dict(**common, seed=int(left["seed"]), **{"delta_" + k: float(left[k] - right[k]) for k in metric_keys}))
        if pairs:
            pf, ps = pd.DataFrame(pairs), pd.DataFrame(seedpairs)
            atomic_csv(pf, output / "verified_paired_fold_differences.csv")
            atomic_csv(ps, output / "verified_paired_seed_oof_differences.csv")
            keys = ["dataset", "model", "contrast_id", "family", "lhs_arm", "rhs_arm", "threshold_policy"]
            summary = pf.groupby(keys)[["delta_" + k for k in metric_keys]].agg(["mean", "std"])
            summary.columns = ["_".join(x) for x in summary.columns]
            summary = summary.reset_index()
            for data, label in ((pf, "ap_positive_folds"), (ps, "ap_positive_oof_seeds")):
                directions = data.groupby(keys).delta_ap.apply(lambda x: int((x > 1e-12).sum())).reset_index(name=label)
                summary = summary.merge(directions, on=keys)
            atomic_csv(summary, output / "verified_paired_summary.csv")
        return dict(paired_fold_rows=len(pairs), paired_oof_rows=len(seedpairs), skipped_incomplete_comparisons=skipped, reference_metrics=references)


def toy(workspace, original):
    import numpy as np
    module = load_original(workspace, original)
    y = np.array([0, 1, 1, 0, 1, 0], dtype=np.int8)
    scores = np.array([.9, .9, .5, .5, .1, .1])
    order = np.argsort(-scores, kind="mergesort")
    width = (len(y) - 1).bit_length()
    bits = ((order[:, None].astype(np.uint64) >> np.arange(width, dtype=np.uint64)) & 1).astype(np.uint8)
    z = dict(n=np.array(len(y)), width=np.array(width), n_positive=np.array(y.sum()),
             packed_order=np.packbits(bits.ravel(), bitorder="little"),
             packed_labels=np.packbits(y[order], bitorder="little"),
             packed_tie_end=np.packbits(np.r_[scores[order][:-1] != scores[order][1:], True], bitorder="little"))
    ap, auc, _, _ = module.rank_metrics(z, y)
    module.close([ap, auc], [module.average_precision_score(y, scores), module.roc_auc_score(y, scores)], "Toy tied rankings")
    rejected = []
    for kind in ("padding", "labels"):
        bad = {k: v.copy() for k, v in z.items()}
        bad["packed_order" if kind == "padding" else "packed_labels"][-1 if kind == "padding" else 0] ^= np.uint8(128 if kind == "padding" else 1)
        try:
            module.rank_metrics(bad, y)
        except ValueError:
            rejected.append(kind)
    require(rejected == ["padding", "labels"], "Toy corruption was not rejected")
    cutoff, _ = module.threshold(np.array([1, 0, 0, 1]), np.array([.9, .8, .7, .6]))
    require(cutoff == .6, "Lowest threshold on exact F1 tie")
    parsed = ast.parse(SCRIPT.read_text())
    calls = [node for node in ast.walk(parsed) if isinstance(node, ast.Call)]
    require(not any(isinstance(node.func, ast.Attribute) and node.func.attr in ("fit", "partial_fit") for node in calls), "Verifier must not invoke fitting")
    require(not any(isinstance(node, (ast.Import, ast.ImportFrom)) and "run_account" in ast.unparse(node) for node in ast.walk(parsed)), "Training module import prohibited")
    provenance_checks = []
    with tempfile.TemporaryDirectory(prefix="flowtopo_acceptance_toy_") as temp:
        root = Path(temp).resolve()
        run = root / "run"
        atomic_json(run / "trees/protocol.json", {"toy": True})
        registration = run / "remote_registration.json"
        atomic_json(registration, {"schema_version": 1, "run_dir": "run", "wrapper_sha256": digest(SCRIPT),
                                   "protocol_sha256": "0" * 64})
        try:
            Acceptance(root, run, registration)
        except ValueError as error:
            require(str(error) == "New scientific protocol changed after registration", "Unexpected protocol guard failure")
            provenance_checks.append("registration_protocol_SHA_rejected")
        require(provenance_checks, "Changed registered protocol was accepted")
        owner = Acceptance.__new__(Acceptance)
        owner.workspace, owner.out, owner.registration_path = root, run, registration
        owner.registration = {"source_manifest_sha256": "scientific_closure", "report_provenance": {"historical_drift": True}}
        owner.registration_sha = digest(registration)
        owner.protocol_sha = digest(run / "trees/protocol.json")
        owner.verifier_sha = digest(SCRIPT)
        done, receipt = run / "toy_done.json", run / "toy_receipt.json"
        atomic_json(done, {"protocol_sha256": owner.protocol_sha})
        seal = dict(passed=True, protocol_sha256=owner.protocol_sha, fold_done_sha256=digest(done), **owner.stamp())
        atomic_json(receipt, seal)
        require(owner.valid_receipt(receipt, done), "Valid toy receipt rejected")
        for field in ("remote_registration_sha256", "remote_protocol_sha256", "protocol_sha256"):
            changed = {**seal, field: "changed"}
            atomic_json(receipt, changed)
            require(not owner.valid_receipt(receipt, done), "Tampered receipt accepted: " + field)
            provenance_checks.append(field + "_rejected")
        atomic_json(receipt, seal)
        atomic_json(done, {"protocol_sha256": "changed"})
        require(not owner.valid_receipt(receipt, done), "Changed canonical DONE accepted")
        provenance_checks.append("changed_DONE_rejected")
        try:
            owner.resolve("../outside")
        except ValueError:
            provenance_checks.append("path_escape_rejected")
        require("path_escape_rejected" in provenance_checks, "Path escape was accepted")
    return dict(passed=True, wrapper_sha256=digest(SCRIPT), frozen_numerical_verifier_sha256=FROZEN_VERIFIER_SHA,
                tied_rank_metrics=True, corruption_rejections=rejected, lowest_f1_tie=True,
                provenance_checks=provenance_checks,
                no_estimator_fits=True, no_training_module_imports=True,
                limitation="Pure arithmetic and static checks only; no production model or remote environment qualification.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "fold", "legacy-fold", "scan", "aggregate", "overview", "toy"))
    parser.add_argument("--workspace-root", type=Path, default=SCRIPT.parents[2])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--registration", type=Path)
    parser.add_argument("--frozen-verifier", type=Path)
    parser.add_argument("--dataset", choices=("hi_small", "li_small"))
    parser.add_argument("--model", choices=("lightgbm", "xgboost", "random_forest"))
    parser.add_argument("--arm")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--max-folds", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--qa-out", type=Path)
    args = parser.parse_args()
    workspace = args.workspace_root.resolve()
    if args.action == "toy":
        result = toy(workspace, args.frozen_verifier or workspace / DEFAULT_ORIGINAL)
        if args.qa_out:
            atomic_json(args.qa_out, result)
    else:
        out = (args.out or workspace / DEFAULT_RUN).resolve()
        registration = args.registration or out / "remote_registration.json"
        acceptance = Acceptance(workspace, out, registration)
        if args.action == "preflight":
            acceptance.validator(False)
            if acceptance.imports:
                acceptance.validator(True)
            result = dict(passed=True, settings=acceptance.expected_settings, expected_outer_folds=acceptance.expected_outer_folds,
                          new_outer_folds=acceptance.new_outer_folds, imports=acceptance.reused_outer_folds,
                          no_fits=True, **acceptance.stamp())
        elif args.action in ("fold", "legacy-fold"):
            require(all(x is not None for x in (args.dataset, args.model, args.arm, args.seed, args.fold)), "Complete fold identity required")
            result = acceptance.validator(args.action == "legacy-fold").fold(args.dataset, args.model, args.arm, args.seed, args.fold, skip=not args.force)
        elif args.action == "scan":
            require(args.max_folds is None or args.max_folds > 0, "Positive max folds required")
            result = acceptance.scan(args.max_folds, args.force)
        elif args.action == "aggregate":
            result = acceptance.aggregate()
        else:
            result = acceptance.overview()
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
