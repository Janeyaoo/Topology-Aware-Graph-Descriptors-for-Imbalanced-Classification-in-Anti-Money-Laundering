"""Install or reproduce the independently verified tuning paper revision.

No model is trained here.  The original result block and original builder stay
unchanged.  The new canonical template/builder read the registered pre-tuning
snapshots and the independently verified tuning report.  Rebuilding an already
archived revision restores the editable root documents without modifying its
sealed snapshots, manifests, or verification record.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

import build_ibm_small_paper_framework as original
import report_account_gmkep_tuning_v3 as tuned

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "Altman_experiments/code"
EXP = tuned.OUT
OUT = EXP / "paper_revision_20261008"
DOC = ROOT / "PAPER_FRAMEWORK_IBM_AML_SMALL_GMKEP56_20261008.md"
TEMPLATE = CODE / "paper_framework_gmkep56_template.md"
BUILDER = Path(__file__).resolve()
BASELINE = EXP / "snapshots"
OLD_DOC = BASELINE / ("before_" + DOC.name)
OLD_TEMPLATE = BASELINE / TEMPLATE.name
OLD_BUILDER = BASELINE / "build_ibm_small_paper_framework.py"
OLD_RESULTS = tuned.fixed.OUT
POLICY = "inner_max_f1"
ARMS = tuned.ARMS


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def text_sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def section(text: str, start: str, end: str) -> str:
    assert text.count(start) == text.count(end) == 1
    return text[text.index(start):text.index(end)]


def replace_section(text: str, start: str, end: str, replacement: str) -> str:
    old = section(text, start, end)
    return text.replace(old, replacement.rstrip() + "\n\n", 1)


def baseline_tokens() -> dict[str, str]:
    """Recover the original rendered token values from matched frozen files."""
    template, document = OLD_TEMPLATE.read_text(), OLD_DOC.read_text()
    tokens = re.findall(r"\{\{([A-Z_]+)\}\}", template)
    assert len(tokens) == len(set(tokens))
    pattern = re.escape(template)
    for name in tokens:
        pattern = pattern.replace(re.escape("{{" + name + "}}"), f"(?P<{name}>.*?)")
    match = re.fullmatch(pattern, document, re.DOTALL)
    assert match, "Registered pre-tuning paper does not match its template"
    return match.groupdict()


def verify_fixed_tokens(tokens: dict[str, str], old) -> None:
    """Regenerate all fixed main/paired/supplementary numerical result tables."""
    summary = old.summary[old.summary.threshold_policy.eq(POLICY)]
    pairs = old.pairs[old.pairs.threshold_policy.eq(POLICY)]
    seed_pairs = old.seed_pairs[old.seed_pairs.threshold_policy.eq(POLICY)]
    assert len(summary) == 18 and len(pairs) == 18 and len(seed_pairs) == 90
    ordered = []
    for ds in tuned.DATASETS:
        for model in tuned.MODELS:
            for arm in ARMS:
                row = summary[(summary.dataset == ds) & (summary.model == model)
                              & (summary.feature_set == arm)]
                assert len(row) == 1
                ordered.append(row.iloc[0])
    expected = {}
    expected["MAIN_RESULTS"] = original.md_table(
        ["数据集", "分类器", "特征组", "AP（主要）", "AUROC", "F1（内层阈值）"],
        [[tuned.DATASETS[r.dataset], tuned.MODELS[r.model], ARMS[r.feature_set],
          *[original.mean_sd(r, metric) for metric in ("ap", "auc", "f1")]]
         for r in ordered])
    expected["CLASSIFICATION_RESULTS"] = original.md_table(
        ["数据集", "分类器", "特征组", "P", "R", "F1", "Acc", "BAcc"],
        [[tuned.DATASETS[r.dataset], tuned.MODELS[r.model], ARMS[r.feature_set],
          *[f"{r[metric+'_mean']:.6f}" for metric in
            ("precision", "recall", "f1", "accuracy", "balanced_accuracy")]]
         for r in ordered])
    for token, cid in (("CORE_PAIRS", "gmke_vs_base"), ("PH_PAIRS", "ph_vs_gmke"),
                       ("FULL_PAIRS", "full_vs_base")):
        rows = []
        for ds in tuned.DATASETS:
            for model in tuned.MODELS:
                row = pairs[(pairs.dataset == ds) & (pairs.model == model)
                            & (pairs.contrast_id == cid)]
                seed = seed_pairs[(seed_pairs.dataset == ds) & (seed_pairs.model == model)
                                  & (seed_pairs.contrast_id == cid)]
                assert len(row) == 1 and len(seed) == 5
                r = row.iloc[0]
                rows.append([tuned.DATASETS[ds], tuned.MODELS[model],
                             original.mean_sd(r, "ap", delta=True),
                             original.mean_sd(r, "f1", delta=True),
                             f"{r.delta_recall_mean:+.6f}",
                             f"{r.delta_balanced_accuracy_mean:+.6f}",
                             f"{int(r.ap_positive_pairs)}/25", f"{int(seed.delta_ap.gt(0).sum())}/5"])
        expected[token] = original.md_table(
            ["数据集", "分类器", "ΔAP", "ΔF1", "ΔR均值", "ΔBAcc均值", "AP正向折", "AP正向seed"], rows)
    mechanism = pd.read_csv(ROOT / "Altman_experiments/results/account_feature_mechanism_20261007/paired_summary.csv")
    ph = pd.read_csv(ROOT / "Altman_experiments/results/account_s10_increment_20261008/paired_summary.csv")
    rows = []
    for ds in tuned.DATASETS:
        for model in ("lightgbm", "xgboost"):
            left = mechanism[(mechanism.dataset == ds) & (mechanism.model == model)
                             & (mechanism.contrast_id == "full_vs_graph")
                             & (mechanism.threshold_policy == POLICY)]
            right = ph[(ph.dataset == ds) & (ph.model == model)
                       & (ph.contrast_id == "ph_same_size") & (ph.threshold_policy == POLICY)]
            assert len(left) == len(right) == 1
            rows.append([tuned.DATASETS[ds], tuned.MODELS[model],
                         original.mean_sd(left.iloc[0], "ap", delta=True),
                         original.mean_sd(right.iloc[0], "ap", delta=True)])
    expected["SUPPLEMENTARY_RESULTS"] = original.md_table(
        ["数据集", "分类器", "旧FULL52−Graph32：ΔAP", "加入Q6后62−50：ΔAP"], rows)
    for name, value in expected.items():
        assert tokens[name] == value, f"Pre-tuning numerical table differs from its archive: {name}"


def completed_section_33() -> str:
    return r"""### 3.3 原固定配置、同流程固定锚点与已完成的逐组调优

下表完整保留原固定参数，继续对应§3.4和§4.1—4.6的原18设置结果。随后完成的有限预算调优与固定锚点另列于§3.7、§4.7—4.12，不覆盖原批次参数或结果。

{{MODEL_TABLE}}

#### 3.3.1 固定配置的作用与表现边界

原批次每个分类器只有一套主要结构参数，未开展网格搜索或逐特征组单独调优；LightGBM／XGBoost最多500轮，由内层验证选择实际轮数，RF固定200树，分类阈值由内层max-F1确定。因此原比较具有统一设置下的可复现性，却不能代表各模型或各表示已经达到最优表现。同一套树复杂度、正则化与列采样配置可能适应不同表示的程度不同，固定配置既可能欠拟合，也可能过拟合。

本研究已进一步完成三个表示分别选参的统一预算实验。它直接检验有限候选搜索后的外层效果，而非先验认定调优必然改善；完整正向、零增量与负向结果都进入§4.7—4.12。调优获选原配置或外层表现下降同样属于有效结果，不能只记录最高一次seed或最高内层分数。

#### 3.3.2 三个实验层次回答不同问题

| 实验层次 | 参数与内层安排 | 所回答的问题 | 解释范围 |
|---|---|---|---|
| 原固定配置 | 共同主要结构参数；原20%内层验证同时用于早停与阈值选择 | 在原学习器配置下，16→44→56的条件特征增量怎样？ | 原§4.1—4.6完整保留，不能解释为最优容量或等计算成本 |
| 同流程固定锚点 | 原结构参数C0；新三折评分／独立早停／独立阈值流程 | 改变内层安排与轮数汇总本身造成怎样的变化？ | 锚点−原固定反映整个流程与明确记录的运行设置变化 |
| 逐组调优 | 同模型共享12候选、账户划分和无权重AP；各特征组独立选择配置 | 相同搜索机会下，各表示经有限适配后的表现怎样？ | 调优−锚点分离多候选搜索的贡献；调优−原固定包含搜索与流程变化 |

相同搜索机会允许三个特征组选择不同参数，不要求它们共享一个最优配置。同一分类器的候选列表同时共享于HI／LI与16／44／56列；每个数据集的同一外层折共享全部内层账户划分。相同候选数量不是相同算法容量、运行时间或全面最优性。有限随机候选与固定结构列表的依据参见[scikit-learn参数搜索说明](https://scikit-learn.org/stable/modules/grid_search.html)，正式搜索空间及候选生成规则见§3.7。

#### 3.3.3 已完成的嵌套选择与公平性控制

全部18个设置沿用五seed×五折外层划分；每个外层训练集保留20%独立阈值集T，其余80%开发集D内开展三折评分。提升树从每个内层训练部分再划20%早停集，以无权重AUROC早停，耐心值30；评分折不参与早停。候选按三个评分折的无权重AP均值选择，精确并列取预先固定的较低候选编号；所选提升轮数取三个最佳轮数的整数中位数。RF无需早停，使用完整内层训练部分。选择参数后，用D拟合、在T上选择max-F1阈值，再以相同参数／轮数重训完整外层训练集。外层测试标签不参与候选、轮数或阈值选择；嵌套评价的依据见[scikit-learn嵌套交叉验证示例](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)。

有效候选评分共\(18\times25\times12\times3=16{,}200\)条。RF六套结构各共享200树训练结果，其100树候选使用经独立核验的前100树，因此有效物理搜索拟合为13,500次；另有450次获选候选代表模型重拟合，用于独立重评分核验。阈值模型与最终外层模型另行计数；当调优获选C0时，相同锚点模型和分数被明确物理复用。以上是有效产物对应的算法工作量，运行重试与并行墙钟时间在执行记录中另行保留。

类别权重仍由每次实际拟合标签计算：
\[
w_c=\frac{N_{\mathrm{fit}}}{2N_{\mathrm{fit},c}},\qquad c\in\{0,1\}.
\]
LightGBM、RF使用`class_weight='balanced'`；XGBoost使用`compute_sample_weight('balanced', fit_labels)`传入训练，不采用`scale_pos_weight`。每次开发集／外层重训均重新计算权重，评价与早停集不加权。没有使用SMOTE或类别欠采样；RF的bootstrap属于其训练机制。

原LightGBM的`subsample=0.8`配合默认`subsample_freq=0`并未开启行bagging；C0保留这一语义，其他提升候选的实际行采样规则公开记录。新批次将每个拟合进程设为两线程，并为LightGBM启用确定性列方式；这些运行差异被明示，不能将锚点−原固定全部归因于超参数。原RF的`sqrt`在16／44／56列下名义候选列数为4／6／7，调优中还比较固定比例的候选列数；特征竞争的变化不等同于新增独立机制。

所有候选分数、参数、轮数、阈值、模型、测试分数、配对变化与成本均按冻结协议管理。相关25折的样本标准差和五次完整OOF方向仅为描述性稳定性证据。新增调优仍使用此前参与特征探索的数据体系，属于静态传导式开发证据，不构成独立最终确认，也不保证普遍最优配置。
"""


def revised_template(fragment: str) -> str:
    text = OLD_TEMPLATE.read_text()
    methods = section(fragment, "### 3.7 ", "### 4.7 ").strip()
    results = fragment[fragment.index("### 4.7 "):].strip()
    assert not re.search(r"^## ", methods + "\n" + results, re.MULTILINE)
    assert "{{" not in fragment
    old_fixed_block = section(text, "### 4.1 ", "## 5. ")
    text = replace_section(text, "### 3.3 ", "### 3.4 ", completed_section_33())
    text = text.replace("## 4. 实证结果与解释", methods + "\n\n## 4. 实证结果与解释", 1)
    text = text.replace("## 5. 讨论、研究边界与论文组织", results + "\n\n## 5. 讨论、研究边界与论文组织", 1)
    assert section(text, "### 4.1 ", "### 4.7 ") == old_fixed_block
    text = text.replace("评价当前固定参数下的模型依赖性，不作普遍模型优劣定理",
                        "并列评价原固定配置与有限预算调优的模型依赖性，不作普遍优劣定理")
    text = text.replace("结果表明，GMKE44相对于BASE16", "原固定配置结果表明，GMKE44相对于BASE16", 1)
    text = text.replace("当前最充分的证据是", "原固定配置下的证据是", 1)
    text = text.replace("| 固定一套分类器参数 | 比较不等于等预算、充分调优或所有模型的最佳性能 |",
                        "| 原固定比较＋统一12候选补充调优 | 搜索机会共享，但有限候选仍不等于全面最优、等容量或等计算时间 |")
    text = text.replace("当前框架未新增训练或曲线实验。",
                        "新增调优训练汇总于§4.7—4.12，本文未新增PR／ROC曲线实验。")
    conclusion_start = "## 6. 结论草稿"
    conclusion_end = "## 附录A：代码、实验与文件管理索引"
    conclusion = section(text, conclusion_start, conclusion_end)
    conclusion = conclusion.replace("结果表明，从BASE16", "原固定配置结果表明，从BASE16", 1)
    conclusion = conclusion.replace("结论限定于当前合成Small数据、固定学习器配置及完整观测图任务。",
                                    "结论限定于当前合成Small数据、原固定配置与当前有限候选调优，以及完整观测图任务。")
    tuning_conclusion = results[results.index("### 4.12 "):].split("\n\n", 1)[1].strip()
    conclusion += "**补充调优的结论。**\n\n" + tuning_conclusion + "\n\n"
    text = replace_section(text, conclusion_start, conclusion_end, conclusion)
    text = text.replace(
        "[build_ibm_small_paper_framework.py](Altman_experiments/code/build_ibm_small_paper_framework.py)",
        f"[{BUILDER.name}]({BUILDER})")
    prior = ROOT / "Altman_experiments/results/paper_framework_gmkep56_tuning_notes_20261008"
    text = text.replace(f"[当前文档修订记录]({prior / 'README.md'})",
                        f"[完成调优后的文档修订记录]({OUT / 'README.md'})")
    text = text.replace(f"[validation.json]({prior / 'validation.json'})",
                        f"[validation.json]({OUT / 'validation.json'})；[此前调优建议修订]({prior / 'README.md'})")
    text = text.replace("生成程序从独立核验通过的主实验表提取结果，并验证字典列顺序、表格数值和来源哈希。",
                        "新生成程序保留原固定表并核对原实验来源，读取独立核验通过的调优表与正文片段，验证56列字典、全部数值与来源哈希。旧生成程序与历次封存文档保持原样；当前版本由上述新生成程序重建。")
    text = text.replace("表内数值由封存实验结果自动提取。",
                        "原固定表保留原数值，新增有限预算逐组调优及同流程固定锚点表由独立核验通过的实验结果自动提取。", 1)
    assert "尚未执行" not in text and "现阶段没有新增调优结果" not in text
    return text.rstrip() + "\n"


def scientific_sources() -> list[Path]:
    paths = [EXP / name for name in ("protocol.json", "run_manifest.json", "validation_models.json",
                                    "report_generation.json", "paper_tuning_sections.md", *tuned.SCIENTIFIC_FILES)]
    paths += [OLD_RESULTS / name for name in (
        "protocol.json", "run_manifest.json", "metadata_clarification.json", "validation_models.json",
        "summary.csv", "paired_summary.csv", "fold_metrics.csv", "paired_fold_differences.csv",
        "seed_oof_metrics.csv", "paired_seed_oof_differences.csv")]
    paths += [OLD_DOC, OLD_TEMPLATE, OLD_BUILDER, EXP / "snapshot_manifest.json",
              CODE / "build_ibm_small_paper_framework.py", Path(tuned.__file__).resolve(),
              ROOT / "EXPERIMENT_RESULTS_SUMMARY.md", ROOT / "EXPERIMENT_RESULTS_SUMMARY_gen.py",
              ROOT / "Altman_experiments/results/account_feature_mechanism_20261007/paired_summary.csv",
              ROOT / "Altman_experiments/results/account_s10_increment_20261008/paired_summary.csv"]
    return list(dict.fromkeys(paths))


def qa(document: str, template: str, baseline: str, old) -> dict:
    assert section(document, "### 4.1 ", "### 4.7 ") == section(baseline, "### 4.1 ", "## 5. ")
    names = re.findall(r"^\| (\d+) \| `([^`]+)` \|", document, re.MULTILINE)
    assert [int(n) for n, _ in names] == list(range(1, 57))
    assert [name for _, name in names] == old.protocol["feature_columns"]["gmkep56"]
    assert not re.search(r"\{\{[A-Z_]+\}\}", document)
    assert all(document.count(f"### {heading} ") == 1 for heading in
               ("3.3", "3.7", "4.7", "4.8", "4.9", "4.10", "4.11", "4.12"))
    assert all(f"## {number}." in document for number in range(1, 7))
    assert "16,200" in document and "13,500" in document and "450" in document
    deferred = {str(OUT / "README.md"), str(OUT / "validation.json")}
    for target in re.findall(r"\]\((/[^)]+)\)", document):
        path = re.sub(r":\d+$", "", target)
        assert Path(path).is_file() or path in deferred, f"Broken local link: {target}"
    width = None
    tables = 0
    for line in document.splitlines():
        if line.startswith("|"):
            size = len(line.split("|")) - 2
            if width is None:
                width = size
                tables += 1
            assert size == width, f"Malformed Markdown table: {line}"
        else:
            width = None
    return {"independent_tuning_loader_passed": True,
            "original_sections_4_1_to_4_6_byte_identical": True,
            "original_numerical_tables_recomputed_from_archives": True,
            "exact_56_dictionary_names_and_order": True,
            "completed_methods_and_tuning_sections_present": True,
            "no_unfilled_template_tokens_in_document": True,
            "all_local_links_resolved": True,
            "markdown_table_cell_counts_match": True,
            "new_template_sha256": text_sha(template), "markdown_tables": tables}


def build(*, rebuild: bool = False) -> None:
    # This gate is checked before any editable document or archive is written.
    results = tuned.load(EXP)
    assert results.validation["passed"] is True
    assert len(results.candidates) == 16200 and len(results.selections) == 450
    assert results.manifest["physical_search_fits"] == 13500
    assert results.manifest["representative_refits"] == 450
    assert int(results.candidates.physical_fit_charged.sum()) == 13500
    for name in (OLD_DOC, OLD_TEMPLATE, OLD_BUILDER):
        manifest = json.loads((EXP / "snapshot_manifest.json").read_text())
        assert sha(name) == manifest[rel(name)]
    assert sha(CODE / "build_ibm_small_paper_framework.py") == sha(OLD_BUILDER)
    record = json.loads((EXP / "report_generation.json").read_text())
    fragment_path = EXP / "paper_tuning_sections.md"
    assert sha(fragment_path) == record["rendered_sha256"][fragment_path.name]
    assert tuned.render(results)[fragment_path.name] == fragment_path.read_text()
    source_hashes = {rel(path): sha(path) for path in scientific_sources()}
    tokens = baseline_tokens()
    verify_fixed_tokens(tokens, results.old)
    template = revised_template(fragment_path.read_text())
    document = template
    for name, value in tokens.items():
        assert document.count("{{" + name + "}}") == 1
        document = document.replace("{{" + name + "}}", value)
    checks = qa(document, template, OLD_DOC.read_text(), results.old)
    assert source_hashes == {rel(path): sha(path) for path in scientific_sources()}
    if rebuild:
        assert OUT.is_dir(), "No completed paper revision archive to rebuild"
        archived = json.loads((OUT / "revision_manifest.json").read_text())
        validation = json.loads((OUT / "validation.json").read_text())
        assert validation["passed"] and validation["revision_manifest_sha256"] == sha(OUT / "revision_manifest.json")
        inventory = json.loads((OUT / "artifact_manifest.json").read_text())
        assert all(sha(ROOT / name) == digest for name, digest in inventory.items())
        assert archived["builder_sha256"] == sha(BUILDER)
        assert archived["source_sha256"] == source_hashes
        assert sha(OUT / "document_snapshot.md") == text_sha(document) == archived["document_sha256"]
        assert sha(OUT / "template_snapshot.md") == text_sha(template) == archived["template_sha256"]
        archive_before = {rel(p): sha(p) for p in OUT.rglob("*") if p.is_file()}
        TEMPLATE.write_text(template)
        DOC.write_text(document)
        assert archive_before == {rel(p): sha(p) for p in OUT.rglob("*") if p.is_file()}
        print(json.dumps({"reproduced": rel(DOC), "document_sha256": sha(DOC),
                          "sealed_archive_unchanged": True}, ensure_ascii=False))
        return
    assert not OUT.exists(), "Revision already exists; use --rebuild to reproduce without changing its archive"
    assert sha(DOC) == sha(OLD_DOC), "Editable paper changed after tuning registration; review before replacing"
    assert sha(TEMPLATE) == sha(OLD_TEMPLATE), "Editable template changed after tuning registration; review before replacing"
    OUT.mkdir()
    shutil.copy2(BUILDER, OUT / "builder_snapshot.py")
    TEMPLATE.write_text(template)
    DOC.write_text(document)
    (OUT / "document_snapshot.md").write_text(document)
    (OUT / "template_snapshot.md").write_text(template)
    write_json(OUT / "frozen_fixed_tokens.json", tokens)
    created = datetime.now().astimezone().isoformat()
    revision = {"created_at": created, "document": rel(DOC), "template": rel(TEMPLATE),
                "builder": rel(BUILDER), "builder_sha256": sha(BUILDER),
                "original_builder_retained": rel(CODE / "build_ibm_small_paper_framework.py"),
                "original_builder_sha256": sha(OLD_BUILDER),
                "scope": "Completed tuning paper revision only; original experiment and original result sections retained",
                "source_sha256": source_hashes, "document_sha256": sha(DOC),
                "template_sha256": sha(TEMPLATE), "before_document_sha256": sha(OLD_DOC),
                "before_template_sha256": sha(OLD_TEMPLATE),
                "fixed_result_block_sha256": text_sha(section(document, "### 4.1 ", "### 4.7 ")),
                "tuning_fragment_sha256": sha(fragment_path), "independent_tuning_validation_passed": True}
    write_json(OUT / "revision_manifest.json", revision)
    (OUT / "README.md").write_text(
        "# IBM AML Small论文框架：已完成逐组调优修订\n\n"
        f"正文：[{DOC.name}]({DOC})。保留原§4.1—4.6全部文字与数值，"
        "§3.3改为完成后的三个实验层次，新增§3.7及§4.7—4.12。\n\n"
        "本程序只有在独立调优核验通过、结果哈希一致且报告可以从科学表重算时才更新正文。"
        "原固定实验、旧文档档案、旧生成程序与实验汇总均保持原样；当前版本使用新生成程序。\n\n"
        f"新生成程序：[{BUILDER.name}]({BUILDER})。首次生成命令："
        f"`/opt/anaconda3/envs/topo_clean_env/bin/python {BUILDER}`；"
        f"重现已归档版本：`/opt/anaconda3/envs/topo_clean_env/bin/python {BUILDER} --rebuild`。"
        "重现仅恢复可编辑正文与模板，不改写本修订档案。\n\n"
        f"修订基线：[{OLD_DOC.name}]({OLD_DOC})、[{OLD_TEMPLATE.name}]({OLD_TEMPLATE})、"
        f"[{OLD_BUILDER.name}]({OLD_BUILDER})；新快照：[document_snapshot.md]({OUT / 'document_snapshot.md'})、"
        f"[template_snapshot.md]({OUT / 'template_snapshot.md'})；来源：[revision_manifest.json]({OUT / 'revision_manifest.json'})；"
        f"核验：[validation.json]({OUT / 'validation.json'})。\n\n"
        "25折和五seed均相关，均值、样本标准差与方向计数用于描述性比较。"
        "全部负结果保留；有限候选搜索不宣称全面最优，不消除同一数据体系上的开发性边界。\n")
    after = {rel(path): sha(path) for path in scientific_sources()}
    assert source_hashes == after, "Scientific input or prior document archive changed during revision"
    write_json(OUT / "validation.json", {"passed": True, "created_at": created, "checks": checks,
               "all_scientific_sources_unchanged": source_hashes == after,
               "document_sha256": sha(DOC), "document_snapshot_sha256": sha(OUT / "document_snapshot.md"),
               "template_sha256": sha(TEMPLATE), "revision_manifest_sha256": sha(OUT / "revision_manifest.json"),
               "limitations": "Document arithmetic/structure QA; the model/ranking verification is separately recorded in the tuning archive"})
    write_json(OUT / "artifact_manifest.json", {rel(p): sha(p) for p in OUT.rglob("*") if p.is_file()})
    print(json.dumps({"document": rel(DOC), "revision": rel(OUT), "passed": True,
                      "original_result_sections_unchanged": True}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true",
                        help="Reproduce editable paper/template from verified sources without changing sealed revision records")
    build(rebuild=parser.parse_args().rebuild)
