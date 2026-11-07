"""Generate manuscript-ready tables from independently accepted budget counts."""
from pathlib import Path
import hashlib
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'Altman_experiments/results/account_unified_alert_budget_20261028'
MODELS = ['lightgbm', 'xgboost', 'random_forest', 'mlp', 'graphsage_weak']
ARMS = ['base16', 'base_g32', 'gmke44', 'gmkep56']
MODEL_NAMES = {'lightgbm': 'LightGBM', 'xgboost': 'XGBoost', 'random_forest': 'Random Forest', 'mlp': 'MLP', 'graphsage_weak': 'GraphSAGE'}
ARM_NAMES = {'base16': 'Base16', 'base_g32': 'G32', 'gmke44': 'Struct44', 'gmkep56': 'Full56'}

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def write(name, value):
    p = OUT / 'paper_tables' / name
    p.parent.mkdir(exist_ok=True)
    p.write_text(value + '\n')
    return value

def main():
    accepted = json.loads((OUT / 'validation/independent_acceptance.json').read_text())
    assert accepted['passed'] and accepted['full_analysis_complete']
    final_manifest = json.loads((OUT / 'final/FILE_MANIFEST.json').read_text())
    for name, item in final_manifest['files'].items():
        assert sha(OUT / 'final' / name) == item['sha256']
    summary = pd.read_csv(OUT / 'final/budget_summary.csv', float_precision='round_trip')
    seeds = pd.read_csv(OUT / 'final/budget_seed_values.csv', float_precision='round_trip')
    pairs = pd.read_csv(OUT / 'final/budget_paired_summary.csv', float_precision='round_trip')
    assert (len(summary), len(seeds), len(pairs)) == (126, 630, 222)

    def row(ds, model, arm, budget=.01):
        z = summary[(summary.dataset == ds) & (summary.model == model) & (summary.feature_set == arm) & (summary.budget == budget)]
        assert len(z) == 1
        return z.iloc[0]

    def seed(ds, model, arm, budget=.01):
        z = seeds[(seeds.dataset == ds) & (seeds.model == model) & (seeds.feature_set == arm) & (seeds.budget == budget)]
        assert len(z) == 5 and z.k.nunique() == z.n_positive.nunique() == 1
        return dict(alerts=int(z.k.iloc[0]), positives=int(z.n_positive.iloc[0]), tp_mean=float(z.tp.mean()), tp_sd=float(z.tp.std(ddof=1)))

    methods = [(m, a, f'{MODEL_NAMES[m]}＋{ARM_NAMES[a]}') for m in MODELS for a in ARMS]
    methods.append(('xgboost', 'base_gfp_account_stream', 'GFP712＋XGBoost'))
    body = ['| 方法与输入 | HI Recall (%) | HI Precision (%) | HI 平均命中数 | LI Recall (%) | LI Precision (%) | LI 平均命中数 |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    bindings = []
    for m, a, label in methods:
        hi, li = row('hi_small', m, a), row('li_small', m, a)
        hc, lc = seed('hi_small', m, a), seed('li_small', m, a)
        body.append(f'| {label} | {100*hi.recall_mean:.2f} | {100*hi.precision_mean:.2f} | {hc["tp_mean"]:.1f} | {100*li.recall_mean:.2f} | {100*li.precision_mean:.2f} | {lc["tp_mean"]:.1f} |')
        bindings.append(dict(model=m, feature_set=a, hi_recall_mean=float(hi.recall_mean), hi_precision_mean=float(hi.precision_mean), li_recall_mean=float(li.recall_mean), li_precision_mean=float(li.precision_mean), hi_seed=hc, li_seed=lc))
    main_table = write('chinese_budget_1pct.md', '\n'.join(body))
    note = 'Recall／Precision为25折均值；平均命中数来自每个种子合并五折已选告警后的TP，再对五种子取均值，故可含小数。HI／LI每种子分别固定选中5155／7060个账户，正类总数为6357／5304。两类汇总权重略有差异，不把折均Recall乘总正类数当作精确命中数。'
    paired = ['| 分类器 | HI ΔRecall (百分点) | HI 正向（折；种子） | LI ΔRecall (百分点) | LI 正向（折；种子） |', '| --- | ---: | --- | ---: | --- |']
    for m in MODELS:
        values = []
        for ds in ['hi_small', 'li_small']:
            z = pairs[(pairs.dataset == ds) & (pairs.left_model == m) & (pairs.right_model == m) & (pairs.left_feature_set == 'gmkep56') & (pairs.right_feature_set == 'base_g32') & (pairs.budget == .01)]
            assert len(z) == 1
            z = z.iloc[0]
            values.extend([f'{100*z.recall_delta_mean:+.3f} ± {100*z.recall_delta_sd:.3f}', f'{int(z.recall_positive_folds)}/25；{int(z.recall_positive_seeds)}/5'])
        paired.append('| ' + ' | '.join([MODEL_NAMES[m], *values]) + ' |')
    paired_table = write('chinese_budget_full_minus_g32.md', '\n'.join(paired))
    endpoints = ['| 方法与输入 | HI Recall@0.5% | HI Recall@2% | LI Recall@0.5% | LI Recall@2% |', '| --- | ---: | ---: | ---: | ---: |']
    for m, a, label in methods:
        values = [f'{100*row(ds,m,a,b).recall_mean:.2f}' for ds in ['hi_small', 'li_small'] for b in [.005,.02]]
        endpoints.append('| ' + ' | '.join([label, *values]) + ' |')
    endpoint_table = write('chinese_budget_endpoints.md', '\n'.join(endpoints))
    appendix_note = '表内单位为百分数，均为25折均值。完整的三个预算Precision／Recall、样本SD、计数和全部配对见final/中的CSV。'
    design = ('比较范围为两数据集的三树四输入24设置、MLP四输入8设置、GraphSAGE四输入8设置和GFP＋XGBoost两设置，共42个设置、1050个外层测试折结果。读取冻结的tuned测试分数，按共同row_index先对齐，再按分数降序及共同账户顺序处理并列。预算为0.5%、1%、2%，每折准确选择ceil(b×n_test)个账户。计算TP、FP、Precision=TP/k与Recall=TP/n_positive，不重训、不重选参数、不校准，也不使用max-F1标签。每种子只汇总五折已经选定的告警与命中数，不跨折混排分数。主结果为25折均值与样本SD，比较按数据集、种子、折、预算配对，只作平均增量与描述性一致性解释。')
    findings = ('1%预算下，GraphSAGE＋G32的平均Recall在全部已评估方法中最高，HI／LI为37.55%／20.08%，平均覆盖2387.2／1065.0个正类账户。XGBoost＋Full56对应29.06%／15.89%，平均覆盖1847.4／843.0个；GFP＋XGBoost为31.40%／15.53%，平均覆盖1996.2／823.6个。三种树的Full56在HI／LI覆盖29.06%–30.29%／15.28%–15.89%。以上人数为五个种子的整图计数均值，不是25折命中数均值。')
    conditional = ('在1%预算下，三树Full−G32的平均Recall增量在HI为+0.535至+1.026个百分点，LI为+0.045至+0.569个百分点；MLP为+0.790／+0.324个百分点，而GraphSAGE为−0.088／−0.222个百分点。MLP的44列覆盖率又略高于56列（HI 27.38%对27.08%，LI 15.42%对15.35%），说明AP提升不保证同容量下更高覆盖。')
    sensitivity = ('预算改变时增量方向会变化：LI–LightGBM的Full−G32在0.5%预算为−0.358个百分点，在2%为+0.596；HI–GraphSAGE在0.5%／1%为−0.120／−0.088，在2%为+0.355。结果不支持56列在各预算下普遍最好，普通图统计与明确的工作容量仍应作为方法选择的参照。')
    report = f'''# 统一告警预算分析：设计、结果与复现记录

日期：2026-10-28。状态：42/42设置、1050/1050预测已通过独立验收；本轮没有训练模型或重新推断。

## 研究设计

{design}

独立验收重新计算每份预测的AP、预算边界与实际选中账户集合，并核对全部汇总与配对。原1350条逐折预算记录的计数和比例完全复现，原54条汇总最大差为浮点误差量级。正式输出为3150条逐折、126条设置汇总、630条整种子记录、5550条折配对、1110条种子配对与222条配对汇总。

## 主要结果：只能复核1%的账户

{findings}

{main_table}

{note}

## 完整字典相对普通图统计

{conditional}

{paired_table}

配对Δ为Full56−G32，单位为百分点；±为25个配对差的样本SD，正向次数仅作描述性一致性。固定同一预算和同一测试集时，Precision和Recall都由相同TP决定，不是两项独立验证。

## 预算敏感性

{sensitivity}

{endpoint_table}

{appendix_note}

## 解释边界

本分析回答合成测试标签下，相同告警容量能覆盖多少正类账户。GraphSAGE、GFP与表格模型仍具有输入机制和计算差异，结果不识别因果机制，也未测量调查结局、团队生产率或真实业务损失。保留同一生成器Small、静态完整观测图、账户与重复折依赖及同数据多轮探索限制。

## 本地材料

- `planning/analysis_protocol.json`：计算前冻结的范围、预算、配对和验收规则。
- `planning/input_manifest.json`：1050条来源路径、SHA、正式AP来源及canonical账户与划分。
- `inputs/download_verification_receipt.json`：补取150份预测的本地验收收据。
- `final/`：冻结的CSV、独立验收和FILE_MANIFEST。
- `run/cache/`：逐折提交记录；`run/`中的selection_manifest定位每折三个预算选中账户的NPZ。
- `validation/independent_acceptance.json`：全部1050份分数、告警集合和汇总的独立验收。
- `paper_tables/`：直接由正式CSV生成的中英文表格和数值绑定。
- `recovery/storage/`：空间不足的中断记录、无损归档索引与恢复工具。最终特征、模型和测试预测未删除。

分析脚本为 `Altman_experiments/code/analyze_unified_alert_budgets_20261028.py`；独立验收为 `verify_unified_alert_budgets_20261028.py`；本报告与中文表由 `summarize_unified_alert_budgets_20261028.py`生成。计算输出与论文文字分开保存，科学CSV保持冻结。
'''
    (OUT / 'RESULTS_UNIFIED_ALERT_BUDGET_20261028.md').write_text(report)
    write('chinese_budget_narrative.json', json.dumps(dict(design=design, findings=findings, conditional=conditional, sensitivity=sensitivity, table_note=note, appendix_note=appendix_note), ensure_ascii=False, indent=2))
    write('chinese_budget_numeric_bindings.json', json.dumps(bindings, ensure_ascii=False, indent=2))
    print('Generated accepted report and Chinese manuscript tables.')

if __name__ == '__main__':
    main()
