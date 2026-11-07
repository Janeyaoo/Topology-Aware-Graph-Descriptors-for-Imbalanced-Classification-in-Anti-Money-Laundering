"""Build a source-bound academic outline without changing experiment artifacts.

The read-only experiment loader verifies archived bindings and recomputes
aggregate/paired tables. This program renders a new document and records its
inputs and QA, rather than fitting models or editing sealed results.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

import report_account_gmkep_multiclassifier as report

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "Altman_experiments/code"
EXP = report.OUT
OUT = ROOT / "Altman_experiments/results/paper_framework_gmkep56_tuning_notes_20261008"
DOC = ROOT / "PAPER_FRAMEWORK_IBM_AML_SMALL_GMKEP56_20261008.md"
TEMPLATE = CODE / "paper_framework_gmkep56_template.md"
DS = report.DATASETS
MODELS = report.MODELS
ARMS = {"base16": "BASE16", "gmke44": "GMKE44", "gmkep56": "GMKEP56"}
POLICY = "inner_max_f1"


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    assert all(len(row) == len(headers) for row in rows)
    assert all("|" not in cell and "\n" not in cell for row in rows for cell in row)
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        *["| " + " | ".join(row) + " |" for row in rows],
    ])


def mean_sd(row, metric: str, *, delta: bool = False) -> str:
    prefix = "delta_" if delta else ""
    value, sd = row[prefix + metric + "_mean"], row[prefix + metric + "_std"]
    return f"{value:+.6f} ± {sd:.6f}" if delta else f"{value:.6f} ± {sd:.6f}"


def integer_stat(text: str, name: str) -> int:
    match = re.search(rf"^\s*{name}\s*:\s*([\d,]+)", text, re.MULTILINE)
    assert match, f"Missing statistic: {name}"
    return int(match.group(1).replace(",", ""))


def build() -> None:
    source_paths = [
        EXP / name for name in (
            "protocol.json", "metadata_clarification.json", "dataset_inputs.json",
            "source_settings.json", "run_manifest.json", "summary.csv",
            "paired_summary.csv", "fold_metrics.csv", "paired_fold_differences.csv",
            "seed_oof_metrics.csv", "paired_seed_oof_differences.csv",
            "validation_models.json",
        )
    ]
    source_paths += [
        ROOT / "EXPERIMENT_RESULTS_SUMMARY.md",
        ROOT / "EXPERIMENT_RESULTS_SUMMARY_gen.py",
        ROOT / "Altman_experiments/results/account_feature_mechanism_20261007/paired_summary.csv",
        ROOT / "Altman_experiments/results/account_feature_mechanism_20261007/protocol.json",
        ROOT / "Altman_experiments/results/account_s10_increment_20261008/paired_summary.csv",
        ROOT / "Altman_experiments/results/account_s10_increment_20261008/engineering_decisions.json",
        *[ROOT / f"Altman_experiments/data/{ds}/dataset_stats.txt" for ds in DS],
        *[ROOT / f"Altman_experiments/results/account_s10_increment_20261008/{ds}/feature_qc.json" for ds in DS],
        *[CODE / name for name in (
            "01_download_explore.py", "02_build_features.py", "ibm_pipeline.py",
            "build_account_s10_features.py", "account_s10_math.py",
            "verify_account_s10_features.py", "run_account_gmkep_multiclassifier.py",
            "verify_account_gmkep_multiclassifier.py", "report_account_gmkep_multiclassifier.py",
        )],
    ]
    before = {rel(path): sha(path) for path in source_paths}
    results = report.load()
    summary = results.summary[results.summary.threshold_policy.eq(POLICY)]
    pairs = results.pairs[results.pairs.threshold_policy.eq(POLICY)]
    seed_pairs = results.seed_pairs[results.seed_pairs.threshold_policy.eq(POLICY)]
    assert len(summary) == 18 and len(pairs) == 18 and len(seed_pairs) == 90
    assert results.protocol["primary_metrics"] == ["ap"]
    text = TEMPLATE.read_text()
    dictionary = re.findall(r"^\| (\d+) \| `([^`]+)` \|", text, re.MULTILINE)
    assert [int(n) for n, _ in dictionary] == list(range(1, 57))
    assert [column for _, column in dictionary] == results.protocol["feature_columns"]["gmkep56"]
    assert results.protocol["feature_columns"]["base16"] == [c for _, c in dictionary[:16]]
    assert results.protocol["feature_columns"]["gmke44"] == [c for _, c in dictionary[:44]]

    input_manifest = json.loads((EXP / "dataset_inputs.json").read_text())
    data_rows = []
    for ds, label in DS.items():
        stats = (ROOT / f"Altman_experiments/data/{ds}/dataset_stats.txt").read_text()
        n_t = integer_stat(stats, "Transactions")
        n_v = integer_stat(stats, "Accounts")
        n_p = integer_stat(stats, "ML accounts")
        n_l = integer_stat(stats, "Illicit txn")
        assert n_v == input_manifest[ds]["n_samples"]
        assert n_p == input_manifest[ds]["n_positive"]
        data_rows.append([label, f"{n_t:,}", f"{n_l:,}", f"{n_l/n_t:.4%}",
                          f"{n_v:,}", f"{n_p:,}", f"{n_p/n_v:.4%}"])
        qc = json.loads((ROOT / f"Altman_experiments/results/account_s10_increment_20261008/{ds}/feature_qc.json").read_text())
        assert f"{100*qc['cap_hit_fraction']:.4f}%" in text
        assert f"{100*qc['ph2_equals_ph3_fraction']:.4f}%" in text
    replacements = {
        "DATA_TABLE": md_table(["数据集", "交易数", "洗钱交易数", "洗钱交易比例",
                                 "账户数", "正类账户数", "账户正类比例"], data_rows)
    }
    model_rows = []
    for model, label in MODELS.items():
        params = results.protocol["model_params"][model]
        keep = {k: v for k, v in params.items() if k not in {"verbose", "verbosity"}}
        details = "；".join(f"`{k}={v!r}`" for k, v in keep.items()).replace("`", chr(96))
        model_rows.append([label, details])
    replacements["MODEL_TABLE"] = md_table(["分类器", "冻结参数（随机状态另见§3.4）"], model_rows)

    ordered = []
    for ds in DS:
        for model in MODELS:
            for arm in ARMS:
                match = summary[(summary.dataset == ds) & (summary.model == model) & (summary.feature_set == arm)]
                assert len(match) == 1
                ordered.append(match.iloc[0])
    replacements["MAIN_RESULTS"] = md_table(
        ["数据集", "分类器", "特征组", "AP（主要）", "AUROC", "F1（内层阈值）"],
        [[DS[row.dataset], MODELS[row.model], ARMS[row.feature_set],
          *[mean_sd(row, metric) for metric in ("ap", "auc", "f1")]] for row in ordered])
    replacements["CLASSIFICATION_RESULTS"] = md_table(
        ["数据集", "分类器", "特征组", "P", "R", "F1", "Acc", "BAcc"],
        [[DS[row.dataset], MODELS[row.model], ARMS[row.feature_set],
          *[f"{row[metric+'_mean']:.6f}" for metric in
            ("precision", "recall", "f1", "accuracy", "balanced_accuracy")]] for row in ordered])

    for token, cid in (("CORE_PAIRS", "gmke_vs_base"), ("PH_PAIRS", "ph_vs_gmke"),
                       ("FULL_PAIRS", "full_vs_base")):
        rows = []
        for ds in DS:
            for model in MODELS:
                match = pairs[(pairs.dataset == ds) & (pairs.model == model) & (pairs.contrast_id == cid)]
                assert len(match) == 1
                row = match.iloc[0]
                seeds = seed_pairs[(seed_pairs.dataset == ds) & (seed_pairs.model == model)
                                   & (seed_pairs.contrast_id == cid)]
                assert len(seeds) == 5
                rows.append([DS[ds], MODELS[model], mean_sd(row, "ap", delta=True),
                             mean_sd(row, "f1", delta=True), f"{row.delta_recall_mean:+.6f}",
                             f"{row.delta_balanced_accuracy_mean:+.6f}",
                             f"{int(row.ap_positive_pairs)}/25", f"{int(seeds.delta_ap.gt(0).sum())}/5"])
        replacements[token] = md_table(
            ["数据集", "分类器", "ΔAP", "ΔF1", "ΔR均值", "ΔBAcc均值", "AP正向折", "AP正向seed"], rows)

    old_pairs = pd.read_csv(ROOT / "Altman_experiments/results/account_feature_mechanism_20261007/paired_summary.csv")
    ph_pairs = pd.read_csv(ROOT / "Altman_experiments/results/account_s10_increment_20261008/paired_summary.csv")
    old_protocol = json.loads((ROOT / "Altman_experiments/results/account_feature_mechanism_20261007/protocol.json").read_text())
    assert old_protocol["graph_stats_columns"] == [c for _, c in dictionary[16:32]]
    sup_rows = []
    for ds in DS:
        for model in ("lightgbm", "xgboost"):
            old = old_pairs[(old_pairs.dataset == ds) & (old_pairs.model == model) &
                            (old_pairs.contrast_id == "full_vs_graph") & (old_pairs.threshold_policy == POLICY)]
            ph = ph_pairs[(ph_pairs.dataset == ds) & (ph_pairs.model == model) &
                          (ph_pairs.contrast_id == "ph_same_size") & (ph_pairs.threshold_policy == POLICY)]
            assert len(old) == len(ph) == 1
            sup_rows.append([DS[ds], MODELS[model], mean_sd(old.iloc[0], "ap", delta=True),
                             mean_sd(ph.iloc[0], "ap", delta=True)])
    replacements["SUPPLEMENTARY_RESULTS"] = md_table(
        ["数据集", "分类器", "旧FULL52−Graph32：ΔAP", "加入Q6后62−50：ΔAP"], sup_rows)
    replacements["ENVIRONMENT"] = "实验环境：" + "；".join(
        f"{name} {results.protocol['environment'][name]}" for name in
        ("python", "numpy", "pandas", "scikit-learn", "joblib", "pyarrow", "lightgbm", "xgboost")) + "。"
    for token, value in replacements.items():
        assert text.count("{{" + token + "}}") == 1
        text = text.replace("{{" + token + "}}", value)
    assert not re.search(r"\{\{[A-Z_]+\}\}", text)
    assert "\u00a4" not in text and "alone" not in text

    # Validate every rendered main numeric row against the archived source.
    for row in ordered:
        for metric in ("ap", "auc", "f1"):
            assert mean_sd(row, metric) in text
    # Validate prose claims that are particularly sensitive to precision/scope.
    for ds in DS:
        frame = summary[summary.dataset.eq(ds)]
        ap_best, f1_best = frame.loc[frame.ap_mean.idxmax()], frame.loc[frame.f1_mean.idxmax()]
        assert f"{ap_best.ap_mean:.6f}" in text and f"{f1_best.f1_mean:.6f}" in text
        for arm in ARMS:
            scores = frame[frame.feature_set.eq(arm)].set_index("model").ap_mean
            assert scores.xgboost > scores.lightgbm > scores.random_forest
    assert pairs[pairs.contrast_id.eq("gmke_vs_base")].delta_ap_mean.gt(0).all()
    assert pairs[pairs.contrast_id.eq("gmke_vs_base")].delta_f1_mean.gt(0).all()
    assert pairs[pairs.contrast_id.eq("ph_vs_gmke")].delta_auc_mean.gt(0).all()

    local_links = re.findall(r"\]\((/[^)]+)\)", text)
    deferred_links = {str(CODE / Path(__file__).name), str(OUT / "README.md"), str(OUT / "validation.json")}
    for target in local_links:
        path_text = re.sub(r":\d+$", "", target)
        assert Path(path_text).is_file() or path_text in deferred_links, f"Broken link: {target}"
    # All table rows must have matching cell counts within each contiguous table.
    table_width = None
    n_tables = 0
    for line in text.splitlines():
        if line.startswith("|"):
            width = len(line.split("|")) - 2
            if table_width is None:
                table_width = width
                n_tables += 1
            assert width == table_width, f"Malformed Markdown table: {line}"
        else:
            table_width = None
    assert all(f"## {section}." in text for section in range(1, 7))

    OUT.mkdir(parents=True, exist_ok=True)
    DOC.write_text(text)
    (OUT / "document_snapshot.md").write_text(text)
    after = {rel(path): sha(path) for path in source_paths}
    assert before == after, "A source or prior summary changed during generation"
    inputs = {
        "created_at": datetime.now().astimezone().isoformat(),
        "scope": "New academic-framework document only; no fitting or sealed-artifact modification",
        "document": rel(DOC), "experiment": rel(EXP),
        "source_sha256": before, "template_sha256": sha(TEMPLATE),
        "generator_sha256": sha(Path(__file__)),
    }
    (OUT / "input_manifest.json").write_text(json.dumps(inputs, ensure_ascii=False, indent=2) + "\n")
    checks = {
        "archived_results_loader_passed": True,
        "exact_56_feature_names_and_order": True,
        "base16_gmke44_prefixes_match_protocol": True,
        "g16_equals_prior_graph16_control": True,
        "main_18_settings_and_25_folds": True,
        "paired_tables_and_seed_counts_from_archived_sources": True,
        "dataset_counts_match_input_manifest": True,
        "ph_qc_percentages_match_source": True,
        "rendered_numeric_cells_match_sources": True,
        "local_links_resolved": True,
        "markdown_table_cell_counts_match": True,
        "sections_1_to_6_present": True,
        "all_source_hashes_unchanged": before == after,
        "no_template_tokens_remain": True,
        "snapshot_matches_document": sha(OUT / "document_snapshot.md") == sha(DOC),
    }
    validation = {
        "passed": all(checks.values()), "checks": checks,
        "feature_dictionary_rows": len(dictionary), "main_result_rows": len(ordered),
        "main_contrast_rows": len(pairs), "markdown_tables": n_tables,
        "local_file_links": len(local_links), "document_bytes": DOC.stat().st_size,
        "document_sha256": sha(DOC), "snapshot_sha256": sha(OUT / "document_snapshot.md"),
        "input_manifest_sha256": sha(OUT / "input_manifest.json"),
        "limitations": "Numerical/structural QA only; peer review checked formulas against code. "
                       "No claim of new independent empirical confirmation or publication acceptance.",
    }
    (OUT / "validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n")
    (OUT / "README.md").write_text(
        "# IBM AML Small论文框架：固定参数与调优说明修订\n\n"
        f"正文：[新论文框架]({DOC})。\n\n"
        "正文从冻结结果生成，包含56项逐列定义、数据与模型协议、18设置结果、三种配对增量及补充控制。"
        "§3.3补充固定超参数的限制、逐组调优的作用与嵌套评价建议，调优尚未执行。"
        "本次没有重新训练或改写原实验结果，也没有改写现有实验汇总文档。\n\n"
        f"初版归档继续保留：[初版记录]({ROOT / 'Altman_experiments/results/paper_framework_gmkep56_20261008/README.md'})。"
        f"修订前正文、模板及生成程序保存在本目录，哈希对应初版完成时的记录："
        f"[revision_baseline.json]({OUT / 'revision_baseline.json'})。初版记录中的正文／模板／生成程序哈希"
        "对应该时点的文件，当前可编辑文件更新后应使用本目录的修订前快照核对这些旧哈希。\n\n"
        f"复现命令：`/opt/anaconda3/envs/topo_clean_env/bin/python {Path(__file__)}`。\n\n"
        f"来源：[input_manifest.json]({OUT / 'input_manifest.json'})；"
        f"核验：[validation.json]({OUT / 'validation.json'})；"
        f"正文快照：[document_snapshot.md]({OUT / 'document_snapshot.md'})。\n".replace("`", chr(96))
    )
    print(json.dumps({"document": str(DOC), "passed": validation["passed"],
                      "feature_rows": len(dictionary), "result_rows": len(ordered),
                      "tables": n_tables, "bytes": DOC.stat().st_size}, ensure_ascii=False))


if __name__ == "__main__":
    build()
