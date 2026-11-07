"""Source-bound report for limited-budget, per-representation tuning.

``aggregate_tables`` is also used before independent validation to produce
scientific CSVs.  Final rendering never mutates those CSVs and requires a
successful independently produced validation_models.json with hash bindings.
The prior fixed-parameter experiment and document archives remain read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import report_account_gmkep_multiclassifier as fixed

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Altman_experiments/results/account_gmkep56_tuned_20261008"
DATASETS = fixed.DATASETS
MODELS = fixed.MODELS
ARMS = {"base16": "BASE16", "gmke44": "GMKE44", "gmkep56": "GMKEP56"}
STRATEGIES = {"tuned": "逐组调优", "same_protocol_anchor": "同流程固定锚点"}
POLICIES = ("fixed_0.5", "inner_max_f1")
SEEDS = tuple(range(5))
FOLDS = tuple(range(5))
METRICS = tuple(fixed.METRICS)
GROUP = ["dataset", "model", "feature_set", "strategy", "threshold_policy"]
FEATURE_GROUP = ["dataset", "model", "strategy", "contrast_id", "threshold_policy"]
STRATEGY_GROUP = ["dataset", "model", "feature_set", "contrast_id", "threshold_policy"]
STRATEGY_CONTRASTS = {
    "tuned_vs_anchor": ("tuned", "same_protocol_anchor", "调优−同流程固定锚点"),
    "tuned_vs_old": ("tuned", "original_fixed", "调优−原固定流程"),
    "anchor_vs_old": ("same_protocol_anchor", "original_fixed", "同流程固定锚点−原固定流程"),
}
CORE_FILES = (
    "fold_metrics.csv", "seed_oof_metrics.csv", "summary.csv",
    "paired_feature_fold_differences.csv", "paired_feature_summary.csv",
    "paired_feature_seed_oof_differences.csv", "paired_strategy_fold_differences.csv",
    "paired_strategy_summary.csv", "paired_strategy_seed_oof_differences.csv",
)
SCIENTIFIC_FILES = (*CORE_FILES, "candidate_fold_scores.csv", "selection_records.csv")


def sha(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _close(actual, expected, message: str) -> None:
    assert np.allclose(actual, expected, atol=1e-12, rtol=0, equal_nan=False), message


def _keys(frame: pd.DataFrame, columns: list[str]) -> set[tuple]:
    assert not frame.duplicated(columns).any(), f"Duplicate rows for {columns}"
    return set(frame[columns].itertuples(index=False, name=None))


def _ordered(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return frame.sort_values(columns).reset_index(drop=True)


def _validate_inputs(folds: pd.DataFrame, seeds: pd.DataFrame, old: fixed.Results) -> None:
    settings = set(itertools.product(DATASETS, MODELS, ARMS, STRATEGIES, POLICIES))
    assert _keys(folds, GROUP + ["seed", "fold"]) == {
        item + (seed, fold) for item in settings for seed in SEEDS for fold in FOLDS
    }, "Incomplete outer fold table"
    assert _keys(seeds, GROUP + ["seed"]) == {
        item + (seed,) for item in settings for seed in SEEDS
    }, "Incomplete OOF seed table"
    assert len(folds) == 1800 and len(seeds) == 360
    assert np.isfinite(folds[list(METRICS) + ["threshold"]].to_numpy()).all()
    assert np.isfinite(seeds[list(METRICS)].to_numpy()).all()
    for frame in (folds, seeds):
        for metric in ("precision", "recall", "f1", "auc", "ap", "accuracy",
                       "balanced_accuracy", "specificity", "fpr", "fnr"):
            assert frame[metric].between(0, 1).all(), f"Out-of-range metric {metric}"
        assert (frame[["tp", "fp", "fn", "tn"]] >= 0).all().all()
        assert (frame[["tp", "fp", "fn", "tn"]] % 1 == 0).all().all()
        _close(frame.tp + frame.fn, frame.n_positive, "Positive account count")
        count_col = "n_test" if frame is folds else "n_samples"
        _close(frame.tp + frame.fp + frame.fn + frame.tn, frame[count_col], "Account count")
        _close(frame.fpr, 1 - frame.specificity, "FPR identity")
        _close(frame.fnr, 1 - frame.recall, "FNR identity")
        _close(frame.balanced_accuracy, (frame.recall + frame.specificity) / 2, "BAcc identity")
        extras = ["seed", "fold"] if frame is folds else ["seed"]
        compare_cols = ["dataset", "model", "feature_set", *extras]
        old_frame = old.folds if frame is folds else old.seeds
        old_frame = old_frame[old_frame.threshold_policy.eq("inner_max_f1")]
        expected = old_frame.set_index(compare_cols)
        for strategy in STRATEGIES:
            current = frame[(frame.strategy == strategy) &
                            (frame.threshold_policy == "inner_max_f1")].set_index(compare_cols)
            current = current.sort_index()
            previous = expected.sort_index()
            assert current.index.equals(previous.index)
            _close(current.n_positive, previous.n_positive, "Prior positive counts")
            _close(current[count_col], previous[count_col], "Prior account counts")
        index_cols = ["dataset", "model", "feature_set", "strategy", *extras]
        by_policy = frame.set_index(index_cols + ["threshold_policy"])
        for metric in ("ap", "auc"):
            left = by_policy.xs("fixed_0.5", level="threshold_policy")[metric].sort_index()
            right = by_policy.xs("inner_max_f1", level="threshold_policy")[metric].sort_index()
            assert left.index.equals(right.index)
            _close(left, right, f"Threshold-independent {metric}")


def _summary(frame: pd.DataFrame, columns: list[str], *, paired: bool = False) -> pd.DataFrame:
    rows = []
    for key, group in frame.groupby(columns, sort=True):
        row = dict(zip(columns, key))
        row["n_pairs" if paired else "n_folds"] = len(group)
        for metric in METRICS:
            name = "delta_" + metric if paired else metric
            row[name + "_mean"] = group[name].mean()
            row[name + "_std"] = group[name].std(ddof=1)
            if paired:
                row[metric + "_positive_pairs"] = int(group[name].gt(0).sum())
        if not paired and "threshold" in group:
            row["threshold_mean"] = group.threshold.mean()
            row["threshold_std"] = group.threshold.std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def _difference(left: pd.DataFrame, right: pd.DataFrame, extras: list[str],
                metadata: dict) -> list[dict]:
    left = left.set_index(extras).sort_index()
    right = right.set_index(extras).sort_index()
    assert left.index.equals(right.index), f"Unpaired rows {metadata}"
    rows = []
    for key in left.index:
        values = key if isinstance(key, tuple) else (key,)
        lrow, rrow = left.loc[key], right.loc[key]
        row = {**metadata, **dict(zip(extras, values))}
        for metric in METRICS:
            row["lhs_" + metric] = lrow[metric]
            row["rhs_" + metric] = rrow[metric]
            row["delta_" + metric] = lrow[metric] - rrow[metric]
        rows.append(row)
    return rows


def _feature_pairs(frame: pd.DataFrame, extras: list[str]) -> pd.DataFrame:
    rows = []
    for dataset, model, strategy, policy in itertools.product(DATASETS, MODELS, STRATEGIES, POLICIES):
        group = frame[(frame.dataset == dataset) & (frame.model == model) &
                      (frame.strategy == strategy) & (frame.threshold_policy == policy)]
        for cid, (lhs, rhs, kind, _) in fixed.CONTRASTS.items():
            rows.extend(_difference(group[group.feature_set.eq(lhs)], group[group.feature_set.eq(rhs)],
                                    extras, {"dataset": dataset, "model": model, "strategy": strategy,
                                             "threshold_policy": policy, "contrast_id": cid,
                                             "lhs_arm": lhs, "rhs_arm": rhs, "contrast_type": kind}))
    return pd.DataFrame(rows)


def _strategy_pairs(frame: pd.DataFrame, old_frame: pd.DataFrame, extras: list[str]) -> pd.DataFrame:
    rows = []
    for dataset, model, arm, policy in itertools.product(DATASETS, MODELS, ARMS, POLICIES):
        subset = {"dataset": dataset, "model": model, "feature_set": arm, "threshold_policy": policy}
        select = lambda f: f[(f.dataset == dataset) & (f.model == model) &
                            (f.feature_set == arm) & (f.threshold_policy == policy)]
        current, previous = select(frame), select(old_frame)
        strategies = {strategy: current[current.strategy.eq(strategy)] for strategy in STRATEGIES}
        strategies["original_fixed"] = previous
        for cid, (lhs, rhs, _) in STRATEGY_CONTRASTS.items():
            rows.extend(_difference(strategies[lhs], strategies[rhs], extras,
                                    {**subset, "contrast_id": cid, "lhs_strategy": lhs, "rhs_strategy": rhs}))
    return pd.DataFrame(rows)


def aggregate_tables(folds: pd.DataFrame, seeds: pd.DataFrame,
                     old: fixed.Results | None = None) -> dict[str, pd.DataFrame]:
    """Return exact scientific aggregates before they are hash-sealed.

    This function writes nothing.  Its caller is responsible for storing the
    returned files before independent validation binds the science outputs.
    """
    old = fixed.load() if old is None else old
    _validate_inputs(folds, seeds, old)
    feature_folds = _feature_pairs(folds, ["seed", "fold"])
    feature_seeds = _feature_pairs(seeds, ["seed"])
    strategy_folds = _strategy_pairs(folds, old.folds, ["seed", "fold"])
    strategy_seeds = _strategy_pairs(seeds, old.seeds, ["seed"])
    tables = {
        "summary.csv": _summary(folds, GROUP),
        "paired_feature_fold_differences.csv": feature_folds,
        "paired_feature_summary.csv": _summary(feature_folds, FEATURE_GROUP, paired=True),
        "paired_feature_seed_oof_differences.csv": feature_seeds,
        "paired_strategy_fold_differences.csv": strategy_folds,
        "paired_strategy_summary.csv": _summary(strategy_folds, STRATEGY_GROUP, paired=True),
        "paired_strategy_seed_oof_differences.csv": strategy_seeds,
    }
    assert [len(tables[name]) for name in tables] == [72, 1800, 72, 360, 2700, 108, 540]
    return tables


@dataclass
class Results:
    out: Path
    protocol: dict
    validation: dict
    manifest: dict
    tables: dict[str, pd.DataFrame]
    selections: pd.DataFrame
    candidates: pd.DataFrame
    old: fixed.Results


def _hash_bindings(out: Path, mapping: dict, *, source: bool = False) -> None:
    for name, digest in mapping.items():
        path = Path(name)
        if not path.is_absolute():
            path = ROOT / path if source or str(path).startswith("Altman_experiments/") else out / path
        assert path.is_file() and sha(path) == digest, f"Validated artifact changed: {name}"


def _validate_search(selections: pd.DataFrame, candidates: pd.DataFrame) -> None:
    keys = ["dataset", "model", "feature_set", "seed", "fold"]
    expected = set(itertools.product(DATASETS, MODELS, ARMS, SEEDS, FOLDS))
    assert _keys(selections, keys) == expected and len(selections) == 450
    assert _keys(candidates, keys + ["candidate_id", "inner_fold"]) == {
        key + (candidate, inner_fold) for key in expected for candidate in range(12) for inner_fold in range(3)
    }
    assert len(candidates) == 16200
    assert np.isfinite(candidates[["ap", "best_iteration", "fit_seconds"]].to_numpy()).all()
    assert candidates.ap.between(0, 1).all() and candidates.fit_seconds.ge(0).all()
    assert selections.candidate_id.between(0, 11).all()
    averages = candidates.groupby(keys + ["candidate_id"], sort=True).ap.mean()
    rounds = candidates.groupby(keys + ["candidate_id"], sort=True).best_iteration.median()
    indexed = selections.set_index(keys)
    for key in expected:
        scores = averages.loc[key]
        # Predefined candidate order resolves exact AP ties without using test outcomes.
        candidate_id = int(scores[scores.eq(scores.max())].index.min())
        assert int(indexed.loc[key].candidate_id) == candidate_id, f"Inner AP argmax differs {key}"
        if "selected_rounds" in indexed and key[1] != "random_forest":
            assert int(indexed.loc[key].selected_rounds) == int(rounds.loc[(*key, candidate_id)])
    for name in selections.columns:
        if name.endswith("_seconds"):
            assert np.isfinite(selections[name]).all() and selections[name].ge(0).all()


def load(out: Path = OUT) -> Results:
    out = Path(out)
    protocol = json.loads((out / "protocol.json").read_text())
    manifest = json.loads((out / "run_manifest.json").read_text())
    validation = json.loads((out / "validation_models.json").read_text())
    assert validation.get("passed") is True, "Independent tuning/model validation has not passed"
    assert validation.get("checks") and all(validation["checks"].values())
    assert validation["protocol_sha256"] == sha(out / "protocol.json")
    required = {"protocol.json", *SCIENTIFIC_FILES}
    assert required.issubset(validation["result_hashes"]), "Independent validation omits scientific CSV bindings"
    _hash_bindings(out, validation["result_hashes"])
    assert validation.get("source_hashes"), "Independent validation omits source bindings"
    _hash_bindings(out, validation["source_hashes"], source=True)
    if "validator" in validation:
        validator = Path(validation["validator"])
        if not validator.is_absolute():
            validator = ROOT / validator
        assert sha(validator) == validation["validator_sha256"]
    old = fixed.load()
    tables = {name: pd.read_csv(out / name, float_precision="round_trip") for name in CORE_FILES}
    recalculated = aggregate_tables(tables["fold_metrics.csv"], tables["seed_oof_metrics.csv"], old)
    for name, expected in recalculated.items():
        actual = tables[name]
        assert set(actual.columns).issuperset(expected.columns), f"Missing columns {name}"
        key_cols = {
            "summary.csv": GROUP,
            "paired_feature_fold_differences.csv": FEATURE_GROUP + ["seed", "fold"],
            "paired_feature_summary.csv": FEATURE_GROUP,
            "paired_feature_seed_oof_differences.csv": FEATURE_GROUP + ["seed"],
            "paired_strategy_fold_differences.csv": STRATEGY_GROUP + ["seed", "fold"],
            "paired_strategy_summary.csv": STRATEGY_GROUP,
            "paired_strategy_seed_oof_differences.csv": STRATEGY_GROUP + ["seed"],
        }[name]
        assert _keys(actual, key_cols) == _keys(expected, key_cols), f"Keys differ {name}"
        actual, expected = _ordered(actual, key_cols), _ordered(expected, key_cols)
        for column in expected:
            if pd.api.types.is_numeric_dtype(expected[column]):
                _close(actual[column], expected[column], f"Derived numeric table differs {name}/{column}")
            else:
                assert actual[column].equals(expected[column]), f"Derived metadata differs {name}/{column}"
    selections = pd.read_csv(out / "selection_records.csv", float_precision="round_trip")
    candidates = pd.read_csv(out / "candidate_fold_scores.csv", float_precision="round_trip")
    _validate_search(selections, candidates)
    return Results(out, protocol, validation, manifest, tables, selections, candidates, old)


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    assert all(len(row) == len(headers) for row in rows)
    assert all("|" not in str(cell) and "\n" not in str(cell) for row in rows for cell in row)
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join(["---"] * len(headers)) + " |",
                      *["| " + " | ".join(map(str, row)) + " |" for row in rows]])


def _stat(row: pd.Series, metric: str, *, delta: bool = False) -> str:
    prefix = "delta_" if delta else ""
    number, sd = row[prefix + metric + "_mean"], row[prefix + metric + "_std"]
    return f"{number:+.6f} ± {sd:.6f}" if delta else f"{number:.6f} ± {sd:.6f}"


def _one(frame: pd.DataFrame, **conditions) -> pd.Series:
    match = np.ones(len(frame), dtype=bool)
    for name, value in conditions.items():
        match &= frame[name].eq(value).to_numpy()
    result = frame[match]
    assert len(result) == 1, f"Expected one report row: {conditions}"
    return result.iloc[0]


def _main_table(results: Results, strategy: str) -> str:
    summary = results.tables["summary.csv"]
    rows = []
    for dataset, model, arm in itertools.product(DATASETS, MODELS, ARMS):
        row = _one(summary, dataset=dataset, model=model, feature_set=arm,
                   strategy=strategy, threshold_policy="inner_max_f1")
        rows.append([DATASETS[dataset], MODELS[model], ARMS[arm], _stat(row, "ap"),
                     _stat(row, "auc"), _stat(row, "f1"),
                     *[f"{row[metric + '_mean']:.6f}" for metric in
                       ("precision", "recall", "accuracy", "balanced_accuracy")]])
    return md_table(["数据集", "分类器", "特征组", "AP（主要）", "AUROC", "F1",
                     "P均值", "R均值", "Acc均值", "BAcc均值"], rows)


def _strategy_table(results: Results, contrast: str) -> str:
    pairs = results.tables["paired_strategy_summary.csv"]
    seeds = results.tables["paired_strategy_seed_oof_differences.csv"]
    rows = []
    for dataset, model, arm in itertools.product(DATASETS, MODELS, ARMS):
        row = _one(pairs, dataset=dataset, model=model, feature_set=arm,
                   contrast_id=contrast, threshold_policy="inner_max_f1")
        seed = seeds[(seeds.dataset == dataset) & (seeds.model == model) &
                     (seeds.feature_set == arm) & (seeds.contrast_id == contrast) &
                     (seeds.threshold_policy == "inner_max_f1")]
        assert len(seed) == 5
        rows.append([DATASETS[dataset], MODELS[model], ARMS[arm], _stat(row, "ap", delta=True),
                     _stat(row, "f1", delta=True), f"{row.delta_recall_mean:+.6f}",
                     f"{row.delta_balanced_accuracy_mean:+.6f}",
                     f"{int(row.ap_positive_pairs)}/25", f"{int(seed.delta_ap.gt(0).sum())}/5"])
    return md_table(["数据集", "分类器", "特征组", "ΔAP", "ΔF1", "ΔR均值", "ΔBAcc均值",
                     "AP正向折", "AP正向seed"], rows)


def _feature_table(results: Results, strategy: str, contrast: str) -> str:
    pairs = results.tables["paired_feature_summary.csv"]
    seeds = results.tables["paired_feature_seed_oof_differences.csv"]
    rows = []
    for dataset, model in itertools.product(DATASETS, MODELS):
        row = _one(pairs, dataset=dataset, model=model, strategy=strategy,
                   contrast_id=contrast, threshold_policy="inner_max_f1")
        seed = seeds[(seeds.dataset == dataset) & (seeds.model == model) &
                     (seeds.strategy == strategy) & (seeds.contrast_id == contrast) &
                     (seeds.threshold_policy == "inner_max_f1")]
        assert len(seed) == 5
        rows.append([DATASETS[dataset], MODELS[model], _stat(row, "ap", delta=True),
                     _stat(row, "f1", delta=True), f"{row.delta_recall_mean:+.6f}",
                     f"{row.delta_balanced_accuracy_mean:+.6f}",
                     f"{int(row.ap_positive_pairs)}/25", f"{int(seed.delta_ap.gt(0).sum())}/5"])
    return md_table(["数据集", "分类器", "ΔAP", "ΔF1", "ΔR均值", "ΔBAcc均值",
                     "AP正向折", "AP正向seed"], rows)


def _seed_table(results: Results) -> str:
    seeds = results.tables["seed_oof_metrics.csv"]
    rows = []
    for dataset, model, arm in itertools.product(DATASETS, MODELS, ARMS):
        first = [DATASETS[dataset], MODELS[model], ARMS[arm]]
        for strategy in STRATEGIES:
            group = seeds[(seeds.dataset == dataset) & (seeds.model == model) &
                          (seeds.feature_set == arm) & (seeds.strategy == strategy) &
                          (seeds.threshold_policy == "inner_max_f1")]
            assert len(group) == 5
            first.extend(f"{group[metric].mean():.6f} ± {group[metric].std(ddof=1):.6f}"
                         for metric in ("ap", "f1"))
        rows.append(first)
    return md_table(["数据集", "分类器", "特征组", "调优OOF AP", "调优OOF F1",
                     "锚点OOF AP", "锚点OOF F1"], rows)


def _selection_table(results: Results) -> str:
    rows = []
    for dataset, model, arm in itertools.product(DATASETS, MODELS, ARMS):
        group = results.selections[(results.selections.dataset == dataset) &
                                   (results.selections.model == model) &
                                   (results.selections.feature_set == arm)]
        assert len(group) == 25
        counts = group.candidate_id.value_counts().sort_index()
        frequency = "；".join(f"C{int(candidate)}：{int(count)}/25" for candidate, count in counts.items())
        rounds = "不适用" if model == "random_forest" else \
            f"{int(group.selected_rounds.min())}—{int(group.selected_rounds.max())}（中位数{group.selected_rounds.median():.0f}）"
        rows.append([DATASETS[dataset], MODELS[model], ARMS[arm], frequency, rounds])
    return md_table(["数据集", "分类器", "特征组", "内层AP选择候选分布", "选择轮数范围"], rows)


def _cost_table(results: Results) -> str:
    rows = []
    for dataset, model, arm in itertools.product(DATASETS, MODELS, ARMS):
        group = results.selections[(results.selections.dataset == dataset) &
                                   (results.selections.model == model) &
                                   (results.selections.feature_set == arm)]
        candidate = results.candidates[(results.candidates.dataset == dataset) &
                                       (results.candidates.model == model) &
                                       (results.candidates.feature_set == arm)]
        search_seconds = float(candidate.fit_seconds.sum())
        tune_seconds = sum(float(group[name].sum()) for name in
                           ("threshold_stage_seconds_tuned", "final_stage_seconds_tuned"))
        anchor_seconds = sum(float(group[name].sum()) for name in
                             ("threshold_stage_seconds_same_protocol_anchor", "final_stage_seconds_same_protocol_anchor"))
        tune_fit = sum(float(group[name].sum()) for name in
                       ("threshold_model_fit_seconds_tuned", "final_model_fit_seconds_tuned"))
        anchor_fit = sum(float(group[name].sum()) for name in
                         ("threshold_model_fit_seconds_same_protocol_anchor", "final_model_fit_seconds_same_protocol_anchor"))
        representative = float(group.representative_stage_seconds.sum())
        reuse_flags = group.physical_reuse_anchor.astype(str).str.lower()
        assert reuse_flags.isin(("true", "false", "1", "0")).all()
        reused = int(reuse_flags.isin(("true", "1")).sum())
        rows.append([DATASETS[dataset], MODELS[model], ARMS[arm], str(len(candidate)),
                     f"{search_seconds/3600:.3f}", f"{representative/3600:.3f}",
                     f"{tune_fit/3600:.3f} / {tune_seconds/3600:.3f}",
                     f"{anchor_fit/3600:.3f} / {anchor_seconds/3600:.3f}", f"{reused}/25"])
    return md_table(["数据集", "分类器", "特征组", "逻辑候选评分数", "搜索拟合小时",
                     "搜索代表重拟合阶段小时", "调优：纯拟合 / 全阶段小时",
                     "锚点：纯拟合 / 全阶段小时", "C0物理复用折"], rows)


def _space_table(results: Results) -> str:
    protocol = results.protocol
    rows = []
    for model in ("lightgbm", "xgboost"):
        space = protocol["boost_search_spaces"][model]
        description = "；".join(f"`{name}`∈{json.dumps(values, ensure_ascii=False)}" for name, values in space.items())
        rows.append([MODELS[model], description,
                     "原固定C0＋预先随机抽取11个不同配置；固定候选列表共享于HI／LI与三组特征"])
    profiles = "；".join(f"结构{i}：{json.dumps(value, ensure_ascii=False, sort_keys=True)}"
                         for i, value in enumerate(protocol["rf_profiles"]))
    rows.append([MODELS["random_forest"], profiles + "；每种结构100／200棵树",
                 "六种固定结构×两种树数＝12候选，C0是原固定200树配置"])
    return md_table(["分类器", "预先固定的搜索空间／结构配置", "实际候选生成"], rows)


def _conclusions(results: Results) -> str:
    rows = results.tables["paired_strategy_summary.csv"]
    feature = results.tables["paired_feature_summary.csv"]
    counts = []
    for cid, (_, _, label) in STRATEGY_CONTRASTS.items():
        group = rows[(rows.contrast_id == cid) & (rows.threshold_policy == "inner_max_f1")]
        assert len(group) == 18
        counts.append(f"{label}：18设置中平均ΔAP正向{int(group.delta_ap_mean.gt(0).sum())}个、"
                      f"负向{int(group.delta_ap_mean.lt(0).sum())}个、完全相同{int(group.delta_ap_mean.eq(0).sum())}个。")
    for cid, (_, _, _, label) in fixed.CONTRASTS.items():
        group = feature[(feature.strategy == "tuned") & (feature.contrast_id == cid) &
                        (feature.threshold_policy == "inner_max_f1")]
        assert len(group) == 6
        counts.append(f"逐组调优后的{label}：6个数据集—分类器组合中平均ΔAP正向"
                      f"{int(group.delta_ap_mean.gt(0).sum())}个、负向{int(group.delta_ap_mean.lt(0).sum())}个。")
    return "\n\n".join(counts) + (
        "\n\n这些计数描述当前有限候选搜索流程的外层结果，不等同于显著性检验。"
        "调优−同流程锚点更直接评价候选搜索的条件效果；调优−原固定流程还包含内层划分、"
        "早停与轮数汇总变化；锚点−原固定流程用于观察这些流程变化。三者均不能解释为找到了"
        "模型的全局最佳性能。三组分别选参后的特征增量包含表示与优化的交互，须与原固定结果并列。"
    )


def render(results: Results) -> dict[str, str]:
    """Create a standalone report and insertion fragments, without editing paper."""
    out = results.out
    links = lambda names: "、".join(f"[{name}]({out / name})" for name in names)
    protocol_text = (
        "### 3.7 已完成的有限预算逐组调优设计\n\n"
        "在保留原18设置固定配置结果的前提下，HI／LI、三种分类器和BASE16／GMKE44／GMKEP56"
        "分别选择参数。沿用原五个seed、每seed五折外层账户划分，共18×25个外层设置折；"
        "同一分类器共享冻结的12个候选配置，候选C0是原固定参数。三个特征组与两个数据集"
        "使用相同候选列表；同一数据集同一外层折使用完全相同的内层账户划分与无权重AP评价。\n\n"
        + _space_table(results) + "\n\n"
        "两个提升分类器的空间值用于生成固定的11个新增候选，不穷举其笛卡尔积；"
        "实际12候选的完整参数与候选ID见"
        + links(["candidates.json"]) + "。C0主要结构参数保持原设置，但本批运行线程改为2；"
        "LightGBM额外采用deterministic=True、force_col_wise=True，并明确subsample_freq=0。"
        "因此锚点−原固定流程还包含这些运行配置变化，不能将其全部解释为划分或轮数变化。\n\n"
        "每个外层训练折沿用原内层80%拟合集D和20%阈值集T。D内进行三折分层评分；"
        "提升树在每个内层训练部分再分出20%早停集，内层评分折不用于早停。"
        "早停使用无权重AUROC、耐心30；AP用于候选选择，两者作用不同。"
        "按三个评分折的无权重AP平均值选择候选，完全并列按预先固定候选顺序取较小编号；"
        "提升树采用该候选三个评分折最佳轮数的中位数。随后用D拟合，在独立T上选择max-F1阈值，"
        "并以所选参数／轮数在完整外层训练集重训后评价外层测试。所有权重仅按实际拟合标签计算。\n\n"
        "新增同流程固定锚点：C0也通过相同三折轮数汇总、D→T阈值选择和完整外层重训。"
        "当所选候选就是C0且参数、轮数和随机状态相同，可物理复用同一阈值模型、最终模型及分数，"
        "结果中明确记录复用数量；两策略仍分别报告。外层测试标签不用于参数、轮数或阈值选择。\n\n"
        "候选评分共18×25×12×3=16,200条逻辑候选评估记录；其中RF每个结构先拟合200树，"
        "100树候选采用相同训练序列的前100树，经小样本独立核对等价后复用。"
        "已保存的有效搜索记录对应13,500次物理拟合，另有450次所选候选inner-fold0代表模型重拟合，"
        "便于保存模型与原分数核验；这些计数不包括后续阈值模型及最终重训。"
        "物理复用不重复当作新增拟合。若运行调度造成中断或重试，其额外计算另见执行记录，"
        "不能把有效拟合数解释为包含中断重试的总拟合数。相同候选数保证搜索机会一致，"
        "并不保证三种算法运行时间或容量一致。搜索空间、早停规则、随机状态、文件哈希和"
        f"物理模型计数见{links(['protocol.json', 'run_manifest.json'])}。\n\n"
        "所有候选保留完整排序和并列信息的rank-audit，可独立重算AP／AUROC；"
        "只长期保留每折所选候选inner-fold0代表搜索模型、原分数、阈值模型和最终模型。"
        "落选搜索模型不长期保存，不能声称全部候选均经模型重新预测核验。\n\n"
        "本批是有限候选搜索，不是穷举全部参数组合，也不保证外层性能高于固定配置。"
        "所有18设置、负向增量、每折所选参数与计算成本均保留。原§3.4和§4.1—4.6继续描述原固定批次。"
    )
    sections = [
        "### 4.7 逐组调优：全部18设置结果\n\n" + _main_table(results, "tuned"),
        "### 4.8 同流程固定锚点：全部18设置结果\n\n" + _main_table(results, "same_protocol_anchor"),
        "### 4.9 参数搜索与内层流程变化的分开比较\n\n"
        "下列差值均按同一外层seed／fold配对；ΔAP和ΔF1为25折均值±样本标准差，"
        "seed计数来自五次完整OOF的配对差值。\n\n" + "\n\n".join(
            f"**{label}。**\n\n" + _strategy_table(results, cid)
            for cid, (_, _, label) in STRATEGY_CONTRASTS.items()),
        "### 4.10 新流程内的三组特征配对\n\n" + "\n\n".join(
            f"**{STRATEGIES[strategy]}：{label}。**\n\n" + _feature_table(results, strategy, cid)
            for strategy in STRATEGIES for cid, (_, _, _, label) in fixed.CONTRASTS.items()),
        "### 4.11 完整OOF、选参分布与计算成本\n\n"
        "每个seed将五个互斥外层测试折拼接为完整OOF分数后重新计算指标；下表为五个seed"
        "的均值±样本标准差。完整OOF AP一般不等于折AP均值，五次重复也不是独立数据集。\n\n"
        + _seed_table(results) + "\n\n候选选择分布如下；参数并非依据外层测试排序选择。\n\n"
        + _selection_table(results) + "\n\n搜索拟合小时和模型纯拟合小时只计fit调用；全阶段小时还包含预测、"
        "阈值计算、哈希和文件压缩保存。代表模型重拟合阶段也包含预测与保存。"
        "累计计时在并行执行时不等于整批墙钟时间；RF共享搜索拟合只向200树parent候选计一次，"
        "100树候选的新增fit时间为零。C0获选时，调优策略引用锚点的物理模型，新增调优阶段时间为零。\n\n"
        + _cost_table(results)
        + "\n\n逐折所选参数、轮数、候选评分和全部运行记录见"
        + links(["selection_records.csv", "candidate_fold_scores.csv", "run_manifest.json"]) + "。",
        "### 4.12 调优结果的结论范围\n\n" + _conclusions(results),
    ]
    results_text = "\n\n".join(sections)
    files = (
        "# IBM AML Small：有限预算逐组调优结果\n\n"
        "独立核验已通过，原固定参数结果和封存档案保持原样。"
        f"本报告读取{links(['validation_models.json', 'protocol.json'])}绑定的科学产物，"
        "重算全部汇总和配对表后生成。指标的主要／辅助层级沿用原论文框架：AP为主要，"
        "F1和P／R补充阈值分类质量，AUROC和BAcc为辅助，Accuracy用于完整报告。\n\n"
        + protocol_text + "\n\n" + results_text + "\n\n"
        "## 文件索引\n\n" + links(SCIENTIFIC_FILES) + "。\n\n"
        "本报告只生成新目录内的说明和待插入正文片段，不直接覆写原框架、模板或旧生成器。"
    )
    fragments = (
        "<!-- 论文修订插入片段：3.7放于第3节末尾，4.7—4.12放于第4节末尾。 -->\n\n"
        + protocol_text + "\n\n" + results_text + "\n"
    )
    return {"README.md": files + "\n", "paper_tuning_sections.md": fragments}


def _markdown_checks(text: str) -> dict:
    table_width = None
    table_count = 0
    for line in text.splitlines():
        if line.startswith("|"):
            width = len(line.split("|")) - 2
            if table_width is None:
                table_width = width
                table_count += 1
            assert width == table_width, f"Malformed Markdown table: {line}"
        else:
            table_width = None
    assert "{{" not in text
    return {"markdown_tables": table_count, "table_cell_counts_valid": True, "no_unfilled_tokens": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    results = load(args.out)
    science_hashes = {name: sha(args.out / name) for name in ("protocol.json", "run_manifest.json",
                                                             "validation_models.json", *SCIENTIFIC_FILES)}
    rendered = render(results)
    checks = {name: _markdown_checks(text) for name, text in rendered.items()}
    for name, text in rendered.items():
        (args.out / name).write_text(text)
    assert science_hashes == {name: sha(args.out / name) for name in science_hashes}, "Science changed during report"
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "scope": "Source-bound report and paper fragments only; no model fitting or scientific CSV mutation",
        "generator": str(Path(__file__).relative_to(ROOT)), "generator_sha256": sha(Path(__file__)),
        "independent_validation_passed": True, "all_science_hashes_unchanged": True,
        "source_sha256": science_hashes, "rendered_sha256": {name: sha(args.out / name) for name in rendered},
        "checks": checks, "settings_per_strategy": 18, "outer_folds_per_setting": 25,
        "logical_candidate_scoring_records": len(results.candidates), "selection_rows": len(results.selections),
    }
    (args.out / "report_generation.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "passed": True,
                      "report_files": list(rendered)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
