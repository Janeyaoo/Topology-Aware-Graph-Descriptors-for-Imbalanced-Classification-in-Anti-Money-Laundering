"""Render the source-bound BASE16 / GMKE44 / GMKEP56 comparison.

This report does not fit models, alter manifests, or infer verification status.
The default mode requires the independently verified model report.  Pending
mode writes only an explicitly marked preview, never the final README or plots.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Altman_experiments/results/account_gmkep56_multiclassifier_20261008"
DATASETS = {"hi_small": "HI-Small", "li_small": "LI-Small"}
MODELS = {"lightgbm": "LightGBM", "xgboost": "XGBoost", "random_forest": "Random Forest"}
ARMS = {"base16": 16, "gmke44": 44, "gmkep56": 56}
LABELS = {"base16": "BASE16", "gmke44": "BASE16＋GMKE28（44）", "gmkep56": "BASE16＋GMKE28＋P12u（56）"}
POLICIES = {"inner_max_f1": "内层 max-F1 阈值", "fixed_0.5": "固定 0.5 阈值"}
CONTRASTS = {
    "gmke_vs_base": ("gmke44", "base16", "structural", "GMKE44−BASE16（44−16）"),
    "ph_vs_gmke": ("gmkep56", "gmke44", "barcode", "GMKEP56−GMKE44（56−44）"),
    "full_vs_base": ("gmkep56", "base16", "full_dictionary", "GMKEP56−BASE16（56−16）"),
}
SCORES = [
    ("precision", "P"), ("recall", "R"), ("f1", "F1"),
    ("auc", "AUROC"), ("ap", "AP"), ("accuracy", "Acc"),
    ("balanced_accuracy", "BAcc"), ("specificity", "Specificity"),
]
COUNTS = [("fpr", "FPR"), ("fnr", "FNR"), ("tn", "TN"), ("fp", "FP"), ("fn", "FN"), ("tp", "TP")]
METRICS = [name for name, _ in SCORES + COUNTS]
GROUP = ["dataset", "model", "feature_set", "threshold_policy"]
PAIR_GROUP = ["dataset", "model", "contrast_id", "threshold_policy"]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def link(name: str, out: Path = OUT) -> str:
    return f"[{name}]({out / name})"


def source_link(name: str) -> str:
    return f"[{name}]({ROOT / name})"


def _hash_bindings(out: Path, mapping: dict, *, sources: bool = False) -> None:
    for name, digest in mapping.items():
        path = ROOT / name if sources or name.startswith("Altman_experiments/") else out / name
        assert path.is_file() and sha(path) == digest, f"Validated artifact changed: {name}"


def _close(actual, expected, message: str) -> None:
    assert np.allclose(actual, expected, atol=1e-12, rtol=0, equal_nan=False), message


def _keys(frame: pd.DataFrame, columns: list[str]) -> set[tuple]:
    assert not frame.duplicated(columns).any(), f"Duplicate rows for {columns}"
    return set(frame[columns].itertuples(index=False, name=None))


@dataclass
class Results:
    out: Path
    protocol: dict
    manifest: dict
    validation: dict | None
    sources: list[dict]
    folds: pd.DataFrame
    summary: pd.DataFrame
    pairs: pd.DataFrame
    pair_folds: pd.DataFrame
    seeds: pd.DataFrame
    seed_pairs: pd.DataFrame


@lru_cache(maxsize=4)
def _load(out_name: str, allow_pending: bool) -> Results:
    out = Path(out_name)
    assert out.is_dir(), f"Experiment directory has not been prepared: {out}"
    protocol = json.loads((out / "protocol.json").read_text())
    manifest = json.loads((out / "run_manifest.json").read_text())
    sources = json.loads((out / "source_settings.json").read_text())
    assert isinstance(sources, list) and len(sources) == 18
    expected_settings = set(itertools.product(DATASETS, MODELS, ARMS))
    assert {(row["dataset"], row["model"], row["feature_set"]) for row in sources} == expected_settings
    assert len({(row["dataset"], row["model"], row["feature_set"]) for row in sources}) == 18
    for row in sources:
        assert row["n_features"] == ARMS[row["feature_set"]]
        expected_new = row["model"] == "random_forest" and row["feature_set"] != "base16"
        assert row["origin"] == ("newly_fitted" if expected_new else "reused")
    assert sum(row["origin"] == "newly_fitted" for row in sources) == 4
    assert dict(protocol["feature_sets"]) == ARMS
    assert set(protocol["models"]) == set(MODELS)
    assert sha(out / "protocol.json") == manifest["protocol_sha256"]
    assert sha(out / "source_settings.json") == protocol["source_settings_sha256"]
    assert sha(out / "reused_artifact_hashes.json") == protocol["reused_artifact_hashes_sha256"]
    clarification = json.loads((out / "metadata_clarification.json").read_text())
    assert clarification["protocol_sha256"] == sha(out / "protocol.json")
    assert clarification["actual_feature_counts"] == ARMS
    assert clarification["contains_Q_or_S_in_actual_inputs"] is False
    assert [column for columns in clarification["actual_blocks"].values() for column in columns] == protocol["feature_columns"]["gmkep56"]
    validation_path = out / "validation_models.json"
    validation = json.loads(validation_path.read_text()) if validation_path.exists() else None
    if not allow_pending:
        assert validation is not None and validation.get("passed") is True, "Independent model validation has not passed"
        assert validation["checks"] and all(validation["checks"].values())
        assert validation["protocol_sha256"] == sha(out / "protocol.json")
        assert sha(ROOT / validation["validator"]) == validation["validator_sha256"]
        required_hashes = {"protocol.json", "source_settings.json", "reused_artifact_hashes.json", "metadata_clarification.json",
            "fold_metrics.csv", "summary.csv", "paired_fold_differences.csv", "paired_summary.csv",
            "seed_oof_metrics.csv", "paired_seed_oof_differences.csv"}
        assert required_hashes.issubset(validation["result_hashes"]), "Independent report lacks scientific artifact bindings"
        assert validation["source_hashes"]
        _hash_bindings(out, validation.get("result_hashes", {}))
        _hash_bindings(out, validation.get("source_hashes", {}), sources=True)
        for name in ("new_final_models", "new_inner_models", "reused_final_models", "reused_inner_models", "reported_final_models", "reported_inner_models"):
            assert isinstance(manifest[name], int) and manifest[name] >= 0
        assert (manifest["new_final_models"], manifest["new_inner_models"]) == (100, 100)
        assert (manifest["reused_final_models"], manifest["reused_inner_models"]) == (350, 350)
        assert (manifest["reported_final_models"], manifest["reported_inner_models"]) == (450, 450)
        for name, count in {
            "new_final_models_verified": 100, "new_inner_models_verified": 100,
            "reused_final_models_hash_verified": 350, "reused_inner_models_hash_verified": 350,
            "n_models_verified": 200, "n_reused_models_hash_verified": 700,
            "total_models_checked": 900,
            "reported_final_models_verified": 450, "reported_inner_models_verified": 450,
        }.items():
            assert validation[name] == count, f"Independent validation count mismatch: {name}"
    tables = {name: pd.read_csv(out / name) for name in (
        "fold_metrics.csv", "summary.csv", "paired_fold_differences.csv", "paired_summary.csv",
        "seed_oof_metrics.csv", "paired_seed_oof_differences.csv",
    )}
    folds, summary = tables["fold_metrics.csv"], tables["summary.csv"]
    pair_folds, pairs = tables["paired_fold_differences.csv"], tables["paired_summary.csv"]
    seeds, seed_pairs = tables["seed_oof_metrics.csv"], tables["paired_seed_oof_differences.csv"]
    assert (len(folds), len(summary), len(pair_folds), len(pairs), len(seeds), len(seed_pairs)) == (900, 36, 900, 36, 180, 180)
    expected_groups = set(itertools.product(DATASETS, MODELS, ARMS, POLICIES))
    expected_pair_groups = set(itertools.product(DATASETS, MODELS, CONTRASTS, POLICIES))
    assert _keys(summary, GROUP) == expected_groups and summary.n_folds.eq(25).all()
    assert _keys(pairs, PAIR_GROUP) == expected_pair_groups and pairs.n_pairs.eq(25).all()
    assert _keys(folds, GROUP + ["seed", "fold"]) == {group + (seed, fold) for group in expected_groups for seed in range(5) for fold in range(5)}
    assert _keys(seeds, GROUP + ["seed"]) == {group + (seed,) for group in expected_groups for seed in range(5)}
    assert _keys(pair_folds, PAIR_GROUP + ["seed", "fold"]) == {group + (seed, fold) for group in expected_pair_groups for seed in range(5) for fold in range(5)}
    assert _keys(seed_pairs, PAIR_GROUP + ["seed"]) == {group + (seed,) for group in expected_pair_groups for seed in range(5)}
    assert np.isfinite(folds[METRICS + ["threshold"]].to_numpy()).all()
    assert np.isfinite(seeds[METRICS].to_numpy()).all()
    for source in sources:
        for table in (folds, summary, seeds):
            rows = table[(table.dataset == source["dataset"]) & (table.model == source["model"]) & (table.feature_set == source["feature_set"])]
            assert rows.source_origin.eq(source["origin"]).all() and rows.source_run.eq(source["source_run"]).all()
            if "n_features" in rows:
                assert rows.n_features.eq(source["n_features"]).all()
    indexed_summary = summary.set_index(GROUP)
    for key, group in folds.groupby(GROUP, sort=False):
        row = indexed_summary.loc[key]
        for metric in METRICS + ["threshold"]:
            _close(row[metric + "_mean"], group[metric].mean(), f"Mean mismatch {key}/{metric}")
            _close(row[metric + "_std"], group[metric].std(ddof=1), f"SD mismatch {key}/{metric}")
    for table, extras in ((pair_folds, ["seed", "fold"]), (seed_pairs, ["seed"])):
        original = folds if "fold" in extras else seeds
        by_setting = original.set_index(["dataset", "model", "feature_set", "threshold_policy", *extras])
        for key, group in table.groupby(PAIR_GROUP, sort=False):
            dataset, model, cid, policy = key
            lhs, rhs, kind, _ = CONTRASTS[cid]
            assert group.lhs_arm.eq(lhs).all() and group.rhs_arm.eq(rhs).all()
            assert group.contrast_type.eq(kind).all()
            group = group.sort_values(extras)
            for metric in METRICS:
                left = by_setting.loc[(dataset, model, lhs, policy)].sort_index()[metric].to_numpy()
                right = by_setting.loc[(dataset, model, rhs, policy)].sort_index()[metric].to_numpy()
                _close(group["delta_" + metric].to_numpy(), left - right, f"Paired mismatch {key}/{metric}")
                if "lhs_" + metric in group:
                    _close(group["lhs_" + metric].to_numpy(), left, f"LHS mismatch {key}/{metric}")
                    _close(group["rhs_" + metric].to_numpy(), right, f"RHS mismatch {key}/{metric}")
    indexed_pairs = pairs.set_index(PAIR_GROUP)
    for key, group in pair_folds.groupby(PAIR_GROUP, sort=False):
        row = indexed_pairs.loc[key]
        lhs, rhs, kind, _ = CONTRASTS[key[2]]
        assert (row.lhs_arm, row.rhs_arm, row.contrast_type) == (lhs, rhs, kind)
        for metric in METRICS:
            delta = group["delta_" + metric]
            _close(row["delta_" + metric + "_mean"], delta.mean(), f"Pair mean mismatch {key}/{metric}")
            _close(row["delta_" + metric + "_std"], delta.std(ddof=1), f"Pair SD mismatch {key}/{metric}")
            assert int(row[metric + "_positive_pairs"]) == int((delta > 0).sum())
    for table, extras in ((folds, ["seed", "fold"]), (seeds, ["seed"])):
        index = ["dataset", "model", "feature_set", *extras]
        fixed = table[table.threshold_policy.eq("fixed_0.5")].set_index(index).sort_index()
        chosen = table[table.threshold_policy.eq("inner_max_f1")].set_index(index).sort_index()
        assert fixed.index.equals(chosen.index)
        _close(fixed[["auc", "ap"]].to_numpy(), chosen[["auc", "ap"]].to_numpy(), "Ranking metrics changed with threshold")
    return Results(out, protocol, manifest, validation, sources, folds, summary, pairs, pair_folds, seeds, seed_pairs)


def load(out: Path = OUT, *, allow_pending: bool = False) -> Results:
    return _load(str(Path(out).resolve()), allow_pending)


def _mean_sd(row, metric: str, *, signed: bool = False, digits: int = 5) -> str:
    value, sd = row[metric + "_mean"], row[metric + "_std"]
    return f"{value:+.{digits}f}±{sd:.{digits}f}" if signed else f"{value:.{digits}f}±{sd:.{digits}f}"


def _table(lines: list[str], headers: list[str]) -> None:
    lines += ["| " + " | ".join(headers) + " |", "|:--|" + "--:|" * (len(headers) - 1)]


def _result_tables(data: Results, dataset: str) -> list[str]:
    lines = []
    summary = data.summary.set_index(GROUP)
    for policy, label in POLICIES.items():
        lines += [f"**{label}：全部设置**", "", "各值为25个外层折的均值±样本SD（ddof=1）；AP为 average_precision，不是梯形积分PR-AUC。", ""]
        _table(lines, ["模型／特征", *[label for _, label in SCORES], "Threshold"])
        for model in MODELS:
            for arm in ARMS:
                row = summary.loc[(dataset, model, arm, policy)]
                cells = [_mean_sd(row, metric) for metric, _ in SCORES] + [_mean_sd(row, "threshold")]
                lines.append("| " + " | ".join([MODELS[model] + " — " + LABELS[arm], *cells]) + " |")
        lines += ["", "混淆矩阵项为每折计数的均值±SD；其平均不是单一全数据混淆矩阵。", ""]
        _table(lines, ["模型／特征", *[label for _, label in COUNTS]])
        for model in MODELS:
            for arm in ARMS:
                row = summary.loc[(dataset, model, arm, policy)]
                cells = [_mean_sd(row, metric, digits=5 if metric in ("fpr", "fnr") else 2) for metric, _ in COUNTS]
                lines.append("| " + " | ".join([MODELS[model] + " — " + LABELS[arm], *cells]) + " |")
        lines += [""]
    return lines


def _paired_tables(data: Results, dataset: str) -> list[str]:
    lines = []
    pairs = data.pairs.set_index(PAIR_GROUP)
    for policy, label in POLICIES.items():
        lines += [f"**同折配对增量：{label}**", "", "Δ均为左组−右组；负ΔFP/FN表示误报/漏报减少。AP/AUROC差不随阈值策略改变。", ""]
        _table(lines, ["模型／比较", "ΔAP（均值±SD）", "正AP折／25", "正OOF seed／5", "ΔAUROC", "ΔF1", "ΔR", "ΔBAcc", "ΔFP／折", "ΔFN／折"])
        for model in MODELS:
            for cid, (_, _, _, label) in CONTRASTS.items():
                row = pairs.loc[(dataset, model, cid, policy)]
                seed = data.seed_pairs[(data.seed_pairs.dataset == dataset) & (data.seed_pairs.model == model) & (data.seed_pairs.contrast_id == cid) & (data.seed_pairs.threshold_policy == policy)]
                cells = [_mean_sd(row, "delta_ap", signed=True, digits=6), str(int(row.ap_positive_pairs)), str(int((seed.delta_ap > 0).sum()))]
                cells += [_mean_sd(row, "delta_" + metric, signed=True) for metric in ("auc", "f1", "recall", "balanced_accuracy")]
                cells += [_mean_sd(row, "delta_" + metric, signed=True, digits=2) for metric in ("fp", "fn")]
                lines.append("| " + " | ".join([MODELS[model] + " — " + label, *cells]) + " |")
        lines += [""]
    lines += ["**每seed完整OOF的AP趋势**", "", "每个seed将五个互斥test折拼接成全量账户OOF概率后重新计算AP；该估计不同于五折AP均值。五个seed复用同一数据，不是独立重复。", ""]
    _table(lines, ["模型／seed", "BASE16 AP", "GMKE44 AP", "GMKEP56 AP", "Δ44−16", "Δ56−44", "Δ56−16"])
    chosen = data.seeds[data.seeds.threshold_policy.eq("inner_max_f1")].set_index(["dataset", "model", "seed", "feature_set"])
    for model in MODELS:
        for seed in range(5):
            values = [chosen.loc[(dataset, model, seed, arm)].ap for arm in ARMS]
            cells = [f"{value:.6f}" for value in values] + [f"{values[1] - values[0]:+.8f}", f"{values[2] - values[1]:+.8f}", f"{values[2] - values[0]:+.8f}"]
            lines.append("| " + " | ".join([f"{MODELS[model]} / {seed}", *cells]) + " |")
    return lines + [""]


def _interpretation(data: Results, dataset: str) -> list[str]:
    rows = data.pairs[(data.pairs.dataset == dataset) & data.pairs.contrast_id.eq("ph_vs_gmke") & data.pairs.threshold_policy.eq("inner_max_f1")]
    positive = int((rows.delta_ap_mean > 0).sum())
    directions = "；".join(f"{MODELS[row.model]} ΔAP={row.delta_ap_mean:+.6f}" for row in rows.itertuples())
    lines = [f"本数据集56−44的AP均值差在{positive}/3个分类器为正：{directions}。这是当前固定学习器及协议下加入P12u显式列的条件比较；正向差、胜折数和seed同号均为描述统计，不能据此把PH模块判为已确认有效或将56列判为普遍最优。", ""]
    ph = rows[rows.model.eq("random_forest")].iloc[0]
    ph_seeds = data.seed_pairs[(data.seed_pairs.dataset == dataset) & data.seed_pairs.model.eq("random_forest") & data.seed_pairs.contrast_id.eq("ph_vs_gmke") & data.seed_pairs.threshold_policy.eq("inner_max_f1")]
    full = data.pairs[(data.pairs.dataset == dataset) & data.pairs.model.eq("random_forest") & data.pairs.contrast_id.eq("full_vs_base") & data.pairs.threshold_policy.eq("inner_max_f1")].iloc[0]
    full_seeds = data.seed_pairs[(data.seed_pairs.dataset == dataset) & data.seed_pairs.model.eq("random_forest") & data.seed_pairs.contrast_id.eq("full_vs_base") & data.seed_pairs.threshold_policy.eq("inner_max_f1")]
    ph_positive, full_positive = int((ph_seeds.delta_ap > 0).sum()), int((full_seeds.delta_ap > 0).sum())
    lines += [f"**RF的条码添加（56−44）限制：**ΔAP={ph.delta_ap_mean:+.6f}，{int(ph.ap_positive_pairs)}/25折AP正向、仅{ph_positive}/5完整OOF seed正向；ΔF1={ph.delta_f1_mean:+.6f}、ΔRecall={ph.delta_recall_mean:+.6f}、ΔBAcc={ph.delta_balanced_accuracy_mean:+.6f}（内层max-F1阈值）。", ""]
    if (ph_seeds.delta_ap < 0).all():
        lines += ["RF的5个完整OOF seed的条码AP增量全部为负；本数据集RF的P12u添加没有呈现稳定正收益，不能用boosting模型的正差替代此结果。", ""]
    elif ph.delta_ap_mean > 0 and ph_positive < 3:
        lines += ["RF条码添加的折均AP微小正差伴随多数seed的OOF AP差非正；均值正差不能解释为一致改善。阈值指标仍按其实际方向报告。", ""]
    lines += [f"**RF的整字典比较（56−16）：**ΔAP={full.delta_ap_mean:+.6f}，{int(full.ap_positive_pairs)}/25折正向、{full_positive}/5完整OOF seed正向；ΔRecall={full.delta_recall_mean:+.6f}、ΔBAcc={full.delta_balanced_accuracy_mean:+.6f}。", ""]
    if full.delta_ap_mean > 0 and full_positive < 3 and full.delta_balanced_accuracy_mean < 0:
        lines += ["RF的整字典AP均值虽有微小正差，只有少数seed正向且BAcc下降，因此不称为稳定提升或所有识别指标同时改善；56列也不是本设置已确认的最优字典。", ""]
    return lines


def _overall_interpretation(data: Results) -> list[str]:
    chosen = data.pairs[data.pairs.threshold_policy.eq("inner_max_f1")]
    full = chosen[chosen.contrast_id.eq("full_vs_base")]
    ph = chosen[chosen.contrast_id.eq("ph_vs_gmke")]
    assert len(full) == len(ph) == 6
    lines = [f"**六场景统一观察：**56−BASE16的折均AP差在{int((full.delta_ap_mean > 0).sum())}/6个数据集/分类器场景为正；单独的P12u添加（56−44）在{int((ph.delta_ap_mean > 0).sum())}/6个场景为正。两种比较不能互相替代，整字典均值胜BASE不表示PH普适、每个seed稳定或56列最优。", ""]
    rf_li = full[(full.dataset == "li_small") & full.model.eq("random_forest")].iloc[0]
    rf_li_seeds = data.seed_pairs[(data.seed_pairs.dataset == "li_small") & data.seed_pairs.model.eq("random_forest") & data.seed_pairs.contrast_id.eq("full_vs_base") & data.seed_pairs.threshold_policy.eq("inner_max_f1")]
    rf_li_ph = data.seed_pairs[(data.seed_pairs.dataset == "li_small") & data.seed_pairs.model.eq("random_forest") & data.seed_pairs.contrast_id.eq("ph_vs_gmke") & data.seed_pairs.threshold_policy.eq("inner_max_f1")]
    lines += [f"RF/LI整字典ΔAP={rf_li.delta_ap_mean:+.6f}，仅{int((rf_li_seeds.delta_ap > 0).sum())}/5完整OOF seed正向，ΔBAcc={rf_li.delta_balanced_accuracy_mean:+.6f}；其条码添加有{int((rf_li_ph.delta_ap < 0).sum())}/5个完整OOF seed为负。该限制与boosting模型的收益同时保留，详见下方配对表与seed表。", ""]
    return lines


def section(dataset: str, loaded: Results | None = None, *, title: str | None = None) -> str:
    assert dataset in DATASETS
    data = loaded or load()
    title = title or ("5.6" if dataset == "hi_small" else "6.5")
    lines = [f"### {title} BASE16 / GMKE44 / GMKEP56 三分类器统一比较（2026-10-08）", "",
        "在相同账户身份、标签、外层/内层划分与固定选择规则下统一比较三组实际输入列。BASE16来自已核验的原始多分类器新协议，LightGBM/XGBoost的44/56列来自已核验S10实验；本批只新训练RF的44/56列。来源复用不计作新拟合或再次独立确认，历史结果表保持原有含义。", "",
        "GMKE44=BASE16＋G16/M5/K3/E4，共44列；GMKEP56再加本轮同采样P12u，实际56列。P12u是F2域上弱无向最短路径VR的2/3hop H0/H1有限条码 max/total/entropy 摘要，使用中心和路径保留、cap40的共同采样协议；本表未加入Q6规模控制或谱特征。", "",
        "G16中的beta1是弱无向诱导图的循环秩E−V＋组件数，不是PH条码；M5是上下游汇合/再汇合代理，不是互惠边；K3是代码中的度数构造曲率代理，不等同于外部标准Ricci估计器；E4为方向不平衡及循环秩派生列。各块并非正交信息分解。", "",
        f"注册JSON的feature_blocks沿用上游模块目录，含未用S_graph4/S_hodge4/S_PL2；实际输入严格以feature_columns为准，只包含BASE/G/M/K/E/P，无Q/S。独立元数据澄清见{link('metadata_clarification.json', data.out)}，冻结协议及训练数值未改。", "",
        "每个设置5 seeds×5外层分层折；内层20%校准集只选择max-F1阈值与boosting早停轮数，最终模型在完整外层训练集重新拟合。所有分类器各有一套固定参数，无本批超参搜索；固定参数不等于等计算预算或等模型容量。RF固定200树、depth14、leaf5、max_features=sqrt，16→44→56列的每节点名义候选数为4→6→7；44→56的差异因此同时包含新增列及列采样/拟合表达变化。", "",
        "25折互相重叠，多个seed共用账户；以下是配对描述统计，不将25折作为独立样本做显著性或因果归因。训练图、账户标签与图特征覆盖同一全观察期，属于静态传导式账户识别；BASE也含计数/对手方统计，不称严格graph-free。", ""]
    if data.validation is None or data.validation.get("passed") is not True:
        lines += ["**预览状态：独立模型核验尚未通过；本段不得作为正式结果写入根汇总文档。**", ""]
    else:
        lines += ["新模型独立重评分及复用产物来源/哈希核对已通过。两阈值、全部评分与混淆矩阵指标如下。", ""]
    if dataset == "hi_small":
        lines += _overall_interpretation(data)
    lines += _result_tables(data, dataset)
    lines += _paired_tables(data, dataset)
    lines += _interpretation(data, dataset)
    lines += ["P12u包含已核对重复与规模依赖：2/3hop H0最大有限寿命列同值；连通单位距离样本在n>1时H0 total=n−1、entropy约为log(n−1)，n=1没有有限H0条且三项摘要均为0，H0 max主要反映是否有非中心顶点。56−44没有匹配Q6规模控制，不能将收益全部归为独立的持续同调信息；同规模PH控制另见已完成S10实验的62−50比较。G/M/K/E也保留代数派生列，实际列数不等于独立信息维数。", "",
        f"完整来源与归档：{link('README.md', data.out)}、{link('protocol.json', data.out)}、{link('source_settings.json', data.out)}、{link('reused_artifact_hashes.json', data.out)}、{link('summary.csv', data.out)}、{link('paired_summary.csv', data.out)}、{link('seed_oof_metrics.csv', data.out)}、{link('paired_seed_oof_differences.csv', data.out)}、{link('validation_models.json', data.out)}。", ""]
    return "\n".join(lines)


def gap_row() -> str:
    return "| IBM HI/LI-Small 账户级：BASE16/GMKE44/GMKEP56 × LightGBM/XGBoost/RF | 三分类器三字典的统一完整指标与配对增量 | 无 | ✅ 来源复用与新增RF实验均已独立核验 | §5.6、§6.5：18设置、两阈值、44−16/56−44/56−16同折配对与5seed完整OOF；仅RF44/56共100最终＋100内层为本批新拟合，其余350最终＋350内层为来源复用；" + link("source_settings.json") + "、" + link("validation_models.json") + "；描述性比较不自动确认PH模块有效 |"


def _source_table(data: Results) -> list[str]:
    lines = ["## 每个设置的来源", "", "来源引用指向原验证归档；复用原模型/概率/折次元数据，当前目录的统一CSV由这些产物生成。新RF的内层与最终模型保存在当前目录。", "", "| 数据集 | 模型 | 字典 | 本批状态 | 来源设置 | 原验证 |", "|:--|:--|:--|:--|:--|:--|"]
    by_key = {(row["dataset"], row["model"], row["feature_set"]): row for row in data.sources}
    for dataset, model, arm in itertools.product(DATASETS, MODELS, ARMS):
        row = by_key[(dataset, model, arm)]
        run = row.get("source_run")
        source = f"{source_link(run)} / {row['source_feature_set']}" if run else "当前新目录 / " + arm
        validation = source_link(str(Path(run) / row["source_validation"])) if run and row.get("source_validation") else "本批validation_models.json"
        lines.append("| " + " | ".join([DATASETS[dataset], MODELS[model], LABELS[arm], "复用" if row["origin"] == "reused" else "新拟合", source, validation]) + " |")
    return lines + [""]


def write_figures(data: Results) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/flowtopo_gmkep_multiclassifier_mpl")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure_dir = data.out / "figures"
    figure_dir.mkdir(exist_ok=True)
    colors = ["#6b7280", "#277da8", "#d17b2a"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.3), sharey=True)
    selected = data.summary[data.summary.threshold_policy.eq("inner_max_f1")].set_index(GROUP)
    x = np.arange(len(MODELS))
    for ax, dataset in zip(axes, DATASETS):
        for index, arm in enumerate(ARMS):
            values = [selected.loc[(dataset, model, arm, "inner_max_f1")].ap_mean for model in MODELS]
            errors = [selected.loc[(dataset, model, arm, "inner_max_f1")].ap_std for model in MODELS]
            bars = ax.bar(x + (index - 1) * .25, values, .23, yerr=errors, capsize=3, color=colors[index], label=arm.upper())
            ax.bar_label(bars, labels=[f"{value:.4f}" for value in values], fontsize=7, padding=3)
        ax.set_title(DATASETS[dataset])
        ax.set_xticks(x, list(MODELS.values()))
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        ax.margins(y=.18)
    axes[0].set_ylabel("Average precision (25-fold mean ± fold SD)")
    axes[1].legend(frameon=False, fontsize=8)
    fig.text(.5, .022, "Paired, overlapping account folds; error bars are descriptive SD, not confidence intervals. Reused and new models follow the same split protocol.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .06, 1, 1))
    for suffix in ("png", "pdf"):
        fig.savefig(figure_dir / f"gmkep_multiclassifier_ap.{suffix}", dpi=240, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.9), sharey=True)
    for ax, dataset in zip(axes, DATASETS):
        for position, model in enumerate(MODELS):
            seed = data.seed_pairs[(data.seed_pairs.dataset == dataset) & (data.seed_pairs.model == model) & data.seed_pairs.contrast_id.eq("ph_vs_gmke") & data.seed_pairs.threshold_policy.eq("inner_max_f1")].sort_values("seed")
            ax.scatter(position + np.linspace(-.09, .09, 5), seed.delta_ap, color="#277da8", s=32, alpha=.85)
            ax.plot([position - .18, position + .18], [seed.delta_ap.mean()] * 2, color="#d17b2a", lw=2.3)
        ax.axhline(0, color="#555555", lw=.8)
        ax.set_title(DATASETS[dataset] + " | GMKEP56 − GMKE44")
        ax.set_xticks(range(3), list(MODELS.values()))
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Delta AP of complete seed OOF predictions")
    fig.text(.5, .037, "Five correlated seeds; orange line = mean seed OOF delta. This differs from mean paired fold AP delta. No independent-repeat significance claim.", ha="center", fontsize=8)
    fig.text(.5, .012, "P12 includes repeated / graph-size-dependent summaries; RF sqrt feature sampling changes from 6 to 7 candidate columns per node.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .085, 1, 1))
    for suffix in ("png", "pdf"):
        fig.savefig(figure_dir / f"ph12_paired_seed_oof_delta_ap.{suffix}", dpi=240, bbox_inches="tight")
    plt.close(fig)


def write_readme(data: Results) -> None:
    manifest = data.manifest
    lines = ["# IBM AML Small：BASE16 / GMKE44 / GMKEP56 三分类器统一结果", "",
        "本档案统一18个设置的指标、来源与配对增量；仅补训练此前未运行的RF44/56，其他已核验结果按来源哈希复用。完整账户参与，无类别下采样。", "",
        "## 运行与核验账本", "",
        f"- 新拟合：{manifest['new_final_models']}个最终模型＋{manifest['new_inner_models']}个内层模型；对应HI/LI的RF44/56四个设置。",
        f"- 来源复用：{manifest['reused_final_models']}个最终模型＋{manifest['reused_inner_models']}个内层模型；BASE16六个设置来自原多分类器档案，boosted44/56八个设置来自S10档案。来源引用、原验证报告与相关产物SHA分开记录；不称为本批新训练。",
        f"- 统一报告：{manifest['reported_final_models']}个最终模型＋{manifest['reported_inner_models']}个内层模型；900行逐折两阈值指标、36行设置汇总、900行逐折配对、36行配对汇总，180行seed OOF指标和180行seed配对。",
        f"- 独立模型核验：validation_models.json的passed={data.validation['passed']}；报告SHA256 `{sha(data.out / 'validation_models.json')}`。核验状态从实际报告读取，本程序不设置passed或改写manifest。", "",
        f"- 本批独立重评分{data.validation['n_models_verified']}个新RF模型，原来源哈希核对{data.validation['n_reused_models_hash_verified']}个复用模型，共检查{data.validation['total_models_checked']}个模型；复用模型的全量重评分依据保留在其原验证报告，不冒称本批重评分全部900个模型。", "",
        "## 协议与解释边界", "",
        "- HI-Small：515,088账户／6,357正类；LI-Small：705,907账户／5,304正类；全期any-positive账户标签。",
        "- 同一数据集全部设置复用相同5 seeds×5-fold StratifiedKFold及20%分层内层校准划分；先在inner_fit拟合，在calibration选max-F1阈值和boosting轮数，再以对应最佳轮数在outer_train完整重训。RF固定树数，不以test结果选参数。",
        "- P/R/F1/Acc/BAcc/Specificity、FPR/FNR及CM同时按内层max-F1和固定0.5评分；预测规则≥阈值。AP是sklearn average_precision_score；AUROC/AP均来自连续概率。",
        "- 每个学习器一套固定参数；LGB/XGB最多500轮、校准AUC早停30轮，RF200树、depth14、min_leaf5、balanced class weight、bootstrap=True、max_samples=0.8、max_features=sqrt。模型类别、树结构及内层选择轮数不同，不能宣称等训练预算、等容量或公平搜索预算。",
        "- RF sqrt使用16/44/56列时每节点名义候选列数为4/6/7；44→56同时改变列采样机会。LGB/XGB也有colsample_bytree0.8；条件性能差不是PH独有信息的因果效应。",
        "- 全图与全期账户标签用于随机账户CV，属于静态传导式开发比较；没有时间留出、未来交易验证或训练图独占评估。BASE含对手方/次数统计，其self-transaction处理与FTD图不同。",
        "- P12u使用本轮共同cap40的2/3hop最短路径VR条码摘要，未替换成旧HI cap80/500缓存。P12u含H0重复及样本规模依赖；本批56−44无Q6规模匹配，不能直接作PH保留确认。该来源S10实验另包含62−50同规模比较。",
        "- 25个fold及5个seed互相依赖；SD和同号计数只描述稳定性，未进行独立重复显著性检验。没有为本批新增结果事后制定PH有效性阈值，也不将56列宣称为最终最优字典。", "",
        "G16/M5/K3/E4来源于原FTD36去除四常量和四同值重复；仍保留代数派生。K3为缓存degree-based局部曲率代理，非外部标准Ricci估计器；实际列数不等于独立信息维数。", ""]
    clarification = json.loads((data.out / "metadata_clarification.json").read_text())
    lines += ["## 实际输入字典与元数据澄清", "",
        "冻结protocol的feature_blocks是沿用上游的模块目录，包含未使用的S_graph4/S_hodge4/S_PL2。实际模型矩阵以feature_columns为唯一依据，本批16/44/56列均无Q/S；不是对feature_blocks各模块列数求和。独立metadata_clarification.json绑定原注册协议SHA，并记录以下六个实际块；该澄清只修正文档理解，不改冻结协议、输入、模型、指标或划分。", "",
        "| 块 | 实际列数 | 精确列序 |", "|:--|--:|:--|"]
    for block, columns in clarification["actual_blocks"].items():
        lines.append(f"| {block} | {len(columns)} | " + ", ".join(f"`{name}`" for name in columns) + " |")
    lines += ["", "BASE16取首块；44列取BASE/G/M/K/E；56列再取P12u。G中的beta1为图循环秩，不是PH条码；M5为上下游汇合/再汇合代理，而非互惠；K3为代码缓存的曲率代理；E4包含可由其他列推导的方向与循环秩摘要。H0重复和规模依赖按注册列保留；本批保持同一基线，不同时精简其他模块。", ""]
    lines += _source_table(data)
    for dataset in DATASETS:
        lines += [section(dataset, data, title=DATASETS[dataset]), ""]
    lines += ["## 图表", "",
        "![AP comparison](figures/gmkep_multiclassifier_ap.png)", "",
        "图中误差线为相关外层折的SD，不是CI。四位数标注为折均值。可导出PDF：`figures/gmkep_multiclassifier_ap.pdf`。", "",
        "![PH paired OOF deltas](figures/ph12_paired_seed_oof_delta_ap.png)", "",
        "点为5个相关seed的完整OOF AP差，橙线为其均值；不与折均值混用。可导出PDF：`figures/ph12_paired_seed_oof_delta_ap.pdf`。", "",
        "## 归档与文件管理", "",
        "- `protocol.json`：运行前冻结的字典、参数、选择流程、来源hash与模型计数；`source_settings.json`：18个设置逐一来源；`reused_artifact_hashes.json`：来源复用产物绑定。",
        "- `hi_small/`与`li_small/`中的`models/`、`inner_models/`、`predictions/`、`calibration/`、`folds/`保存本批新RF44/56产物；每数据集50个最终模型＋50个内层模型。复用模型和概率保留原来源目录，不再次复制或算作新拟合。",
        "- `fold_metrics.csv`、`summary.csv`：统一逐折与均值/样本SD；`paired_fold_differences.csv`、`paired_summary.csv`：三个预定差值方向；`seed_oof_metrics.csv`、`paired_seed_oof_differences.csv`：完整seed OOF评分与配对。",
        "- `validation_models.json`：新RF全量独立重评分与复用来源核对；`validation.json`：交付阶段总体核验（由finalize生成）；README与图形不改写这些核验报告。",
        "- 冻结代码快照、根MD更新前后快照、根生成器副本、文档hash与最终产物inventory由交付归档步骤保存；以实际清单为准。原BASE/FULL、FTD机制与S10历史档案保持原路径和指标。", "",
        "## 重现", "",
        "先对不存在的新输出路径执行冻结runner的prepare，仅训练缺失的RF44/56；来源归档必须保留并通过hash绑定。每个worker使用fresh process。以下命令中的<新目录>不可替换为本次已完成目录；两个数据集run可在独立进程并行，summarize在两者complete后执行。", "", "```bash",
        "/opt/anaconda3/envs/topo_clean_env/bin/python Altman_experiments/code/run_account_gmkep_multiclassifier.py --phase prepare --out <新目录>",
        "/opt/anaconda3/envs/topo_clean_env/bin/python Altman_experiments/code/clarify_account_gmkep_metadata.py --out <新目录>",
        "/opt/anaconda3/envs/topo_clean_env/bin/python Altman_experiments/code/run_account_gmkep_multiclassifier.py --phase run --dataset hi_small --out <新目录>",
        "/opt/anaconda3/envs/topo_clean_env/bin/python Altman_experiments/code/run_account_gmkep_multiclassifier.py --phase run --dataset li_small --out <新目录>",
        "/opt/anaconda3/envs/topo_clean_env/bin/python Altman_experiments/code/run_account_gmkep_multiclassifier.py --phase summarize --out <新目录>",
        "/opt/anaconda3/envs/topo_clean_env/bin/python Altman_experiments/code/verify_account_gmkep_multiclassifier.py --dataset hi_small --out <新目录>",
        "/opt/anaconda3/envs/topo_clean_env/bin/python Altman_experiments/code/verify_account_gmkep_multiclassifier.py --dataset li_small --out <新目录>",
        "/opt/anaconda3/envs/topo_clean_env/bin/python Altman_experiments/code/verify_account_gmkep_multiclassifier.py --aggregate-only --out <新目录>",
        "/opt/anaconda3/envs/topo_clean_env/bin/python Altman_experiments/code/report_account_gmkep_multiclassifier.py --out <独立核验通过的目录>",
        "```", "",
        "prepare不会产生元数据澄清文件；独立clarify辅助入口在prepare后、run前读取新protocol，构造实际六块并绑定该新protocol的SHA。不能复制本次旧metadata_clarification.json到新输出目录；该辅助入口不修改冻结protocol或训练数值。", "",
        f"当前报告源码SHA256 `{sha(Path(__file__))}`；元数据辅助入口SHA256 `{sha(ROOT / 'Altman_experiments/code/clarify_account_gmkep_metadata.py')}`。完整来源/运行参数见protocol，结果与来源hash见核验报告与归档清单。", ""]
    (data.out / "README.md").write_text("\n".join(lines) + "\n")
    notes = ["# BASE16 / GMKE44 / GMKEP56：统一比较结果笔记", "",
        "以下结果由通过独立核验的统一CSV生成；本批仅新增RF44/56，其余结果按来源hash复用。", ""]
    for dataset in DATASETS:
        notes += [section(dataset, data), ""]
    (data.out / "EXPERIMENT_NOTES.md").write_text("\n".join(notes) + "\n")


def write_protocol_doc(data: Results) -> None:
    protocol = data.protocol
    clarification = json.loads((data.out / "metadata_clarification.json").read_text())
    lines = ["# BASE16 / GMKE44 / GMKEP56 统一比较：注册协议说明", "",
        "本说明在结果交付时由已冻结JSON生成；运行前注册的依据为protocol.json及其注册时间和SHA，而非本Markdown的生成时间。", "",
        f"注册时间：{protocol['registered_at']}；protocol SHA256 `{sha(data.out / 'protocol.json')}`。", "",
        "## 数据、字典与来源", "",
        "HI/LI-Small全量账户，原全期any-positive标签。18设置＝2数据集×3学习器×3实际字典；四个RF44/56设置为本批新训练，十四个来源设置按path/SHA复用。原模型/结果不覆盖。", "",
        "实际输入严格依protocol.json/feature_columns；feature_blocks含上游目录中的未用S模块。metadata_clarification.json只解释模块目录，实际输入无Q/S。", "",
        "| 实际块 | 列数 | 列名（顺序固定） |", "|:--|--:|:--|"]
    for block, columns in clarification["actual_blocks"].items():
        lines.append(f"| {block} | {len(columns)} | " + ", ".join(f"`{column}`" for column in columns) + " |")
    lines += ["", "BASE16；BASE16＋G16＋M5＋K3＋E4＝44；再加P12u＝56。G中的beta1为图循环秩而非PH条码；M5为上下游汇合/再汇合代理，非互惠；K3为代码中的曲率代理。P12u使用源S10的matched cap40缓存，2hop为同3hop样本的前缀，在诱导图重算距离；F2有限H0/H1条码max/total/entropy。重复与代数依赖保留，报告不将列数解释为独立信息维数。", "",
        "## 划分与拟合/选择", "",
        "外层StratifiedKFold(shuffle=True, random_state=seed)，seeds0–4，各5folds；全部设置共用账户行号和原split。内层从outer_train取20%分层calibration，random_state=10000+100×seed+fold。模型random_state=100×seed+fold。", "",
        "只在inner_fit拟合，用calibration选择阈值，boosting同时选早停轮数；最终模型在整个outer_train重训。阈值最大化正类F1，exact tie取最低阈值；每组最终概率按该阈值及固定0.5评分，规则≥。LGB/RF使用class_weight=balanced；XGB以compute_sample_weight('balanced', fit_labels)得到sample_weight并传入fit，本协议未使用scale_pos_weight。权重依对应fit标签产生，最终refit基于完整outer_train重新计算。test不参与选择。", "",
        "固定参数，未进行搜索；参数相同不保证等计算预算或等学习器容量。RF的max_features=sqrt在16/44/56列对应每节点名义4/6/7个候选列。", "", "```json", json.dumps(protocol["model_params"], ensure_ascii=False, indent=2), "```", "",
        "## 预定比较与账本", "",
        "三个差值方向：44−16结构列条件增量；56−44条码显式列条件增量；56−16整字典增量。全部25折按相同seed/fold配对，完整seed OOF另计算。AP主报告，但本批不新建事后有效性门槛、p-value筛选或最优字典确认。", "",
        "新100 final＋100 inner；来源复用350 final＋350 inner；报告450 final＋450 inner。独立核验本批200个新模型的所有calibration/test概率，并核对700个复用模型与其来源已通过的原验证报告/产物hash；不得把复用称为本批再次拟合或全部900模型重评分。", "",
        "统一表：900fold rows、36summary rows、900paired-fold rows、36paired-summary rows、180seed OOF rows、180paired-seed rows。评分：P/R/F1/AUROC/AP/Acc/BAcc/Specificity/FPR/FNR/TN/FP/FN/TP，含两阈值；汇总SD使用ddof=1。", "",
        "全期图随机账户CV是静态传导式开发识别，不是未来时间或训练图独占验证。相关fold/seed不能作独立实验；P12u含H0同值重复及规模依赖，无Q6的56−44不证明scale-independent PH信息。K3为缓存代理，非精确标准Ricci估计器。", "",
        f"冻结训练源码SHA `{protocol['trainer_sha256']}`；helper SHA `{protocol['helper_sha256']}`；source_settings SHA `{protocol['source_settings_sha256']}`；reused_artifact_hashes SHA `{protocol['reused_artifact_hashes_sha256']}`。完整输入、源划分/验证hash与环境保留于JSON。", ""]
    (data.out / "PROTOCOL.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--allow-pending", action="store_true", help="Write only PENDING_REPORT.md from complete CSV tables before independent validation")
    args = parser.parse_args()
    data = load(args.out, allow_pending=args.allow_pending)
    if args.allow_pending:
        text = "# Pending comparison preview — not a formal report\n\nDo not copy this preview to the root summary.\n\n"
        text += "\n\n".join(section(dataset, data) for dataset in DATASETS)
        (data.out / "PENDING_REPORT.md").write_text(text + "\n")
        print(json.dumps({"status": "pending_preview_only", "out": str(data.out), "summary_rows": len(data.summary)}))
        return
    write_figures(data)
    write_readme(data)
    write_protocol_doc(data)
    print(json.dumps({"status": "verified_report_generated", "out": str(data.out), "settings": 18, "summary_rows": len(data.summary), "new_models": data.manifest["new_final_models"] + data.manifest["new_inner_models"], "reused_models": data.manifest["reused_final_models"] + data.manifest["reused_inner_models"]}))


if __name__ == "__main__":
    main()
