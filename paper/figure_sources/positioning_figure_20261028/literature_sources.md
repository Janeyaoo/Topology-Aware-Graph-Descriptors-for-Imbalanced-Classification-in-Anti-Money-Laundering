# 研究定位图：代表性文献及放置依据

核对日期：2026-10-28。以下为图中字母的对应关系。采用正式期刊／会议版本年份；Weber 使用当前稿件引用的 2019 年作者预印本，K 为本文 2026 年工作稿。选择服务于方法定位，不构成系统综述或跨论文性能排名。

| 标记 | 文献及版本 | 放置依据 | 原始来源 |
|---|---|---|---|
| A | Chang et al. (2008), *Scalable and Interactive Visual Analysis of Financial Wire Transactions for Fraud Detection*, Information Visualization 7(1):63–76 | 与 Bank of America AML 部门合作；协调视图与 search-by-example 支持可疑账户、交易调查。放在显式分析区，主要显示账户对象，范围也包含交易。属于调查支持，不是账户监督二分类器。 | [共同作者发表页](https://eagereyes.org/publications/Chang-InfoVis-2008)；[作者原文](https://illusioneering.cs.umn.edu/papers/chang-ivs2008.pdf)；[DOI](https://doi.org/10.1057/palgrave.ivs.9500172) |
| B | A. Fronzetti Colladon & E. Remondi (2017), *Using social network analysis to prevent money laundering*, Expert Systems with Applications 67:49–58 | 保理公司客户风险和显式网络指标；分析对象为客户／社会行动者。完整复姓为 Fronzetti Colladon。 | [作者接受稿](https://andreafc.com/articles/ESWA1.pdf)；[DOI](https://doi.org/10.1016/j.eswa.2016.09.029) |
| C | Weber et al. (2019), *Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics* | Elliptic 的节点表示交易。图中明确仅标 GCN 分支；原论文还比较 RF 等模型，不能把整篇论文概括为 GCN 优越。 | [作者预印本](https://arxiv.org/abs/1908.02591) |
| D | Akcora et al. (2020), *BitcoinHeist: Topological Data Analysis for Ransomware Prediction on the Bitcoin Blockchain*, IJCAI 2020:4439–4445 | 地址级勒索软件风险，显式路径／流和混淆模式特征，属于相关金融犯罪。没有将这些特征标成 PH 条码。 | [IJCAI 原文](https://ijcai.org/proceedings/2020/612) |
| E | Hofer et al. (2020), *Graph Filtration Learning*, ICML / PMLR 119:4314–4323 | 学习滤过函数和 PH readout，任务为整图分类。放在独立的通用方法灰区。 | [PMLR 原文](https://proceedings.mlr.press/v119/hofer20b.html) |
| F | Cardoso et al. (2022), *LaundroGraph: Self-Supervised Graph Representation Learning for Anti-Money Laundering*, ICAIF 2022 | 客户—交易二部图，自监督链接预测。节点卡片跨账户／交易边界，避免将其误认为与本文相同的账户分类协议。 | [作者论文](https://arxiv.org/abs/2210.14360)；[DOI](https://doi.org/10.1145/3533271.3561727) |
| G | Horn et al. (2022), *Topological Graph Neural Networks*, ICLR 2022 | 将 PH 融入 GNN 的通用节点／整图方法。放在灰色方法来源区，未声称其已在银行 AML 下验证。 | [正式论文](https://openreview.net/forum?id=oxxUMeFwEHd)；[作者机构记录](https://www.research-collection.ethz.ch/entities/publication/dc9ef042-7099-4c29-a949-eff1d9884e20) |
| H | Lo et al. (2023), *Inspection-L: self-supervised GNN node embeddings for money laundering detection in bitcoin*, Applied Intelligence 53:19406–19417 | DGI/GIN 表示接 RF，Elliptic 交易分类。标题中的 node 不等于账户或地址。 | [期刊原文](https://link.springer.com/article/10.1007/s10489-023-04504-9) |
| I | Egressy et al. (2024), *Provably Powerful Graph Neural Networks for Directed Multigraphs*, AAAI 38(10):11838–11846 | 端口编号、ego IDs 和逆向消息传递；图中放置其 AML 交易检测应用。 | [作者机构原文记录](https://research.ibm.com/publications/provably-powerful-graph-neural-networks-for-directed-multigraphs)；[DOI](https://doi.org/10.1609/aaai.v38i10.29069) |
| J | Blanuša et al. (2024), *Graph Feature Preprocessor: Real-time Subgraph-based Feature Extraction for Financial Crime Detection*, ICAIF 2024:222–230 | 从交易流提取子图模式特征，原论文的交易级任务。图中没有把本研究的账户级适配冒充原始实验。 | [IBM Research 原文记录](https://research.ibm.com/publications/graph-feature-preprocessor-real-time-subgraph-based-feature-extraction-for-financial-crime-detection)；[DOI](https://doi.org/10.1145/3677052.3698674) |
| K | TopoAML，本研究（2026 工作稿） | 静态、传导式的合成 IBM AML Small 账户分类；显式有向结构描述符与固定 VR 采样的 PH 汇总。图中的位置描述研究路线，不表示性能最高或首次采用 PH。 | 本项目 `../FlowTopo_manuscript.tex`；本轮未修改该稿件 |

## 选择与边界

- 采用 2008–2026，约十八年；没有为凑足二十年填入关联较弱的工作。
- 仅使用代表性节点，未声称覆盖截至 2026 年的全部研究。
- 账户／客户／地址与交易是分析对象分类，部分工作同时涉及多个对象。
- 泛称 topological features 的工作与持续同调 PH 不等同；BitcoinHeist 在图中明确使用 flow/path 标签。
- ChainNet 未加入：其 PH 研究涉及 Bitcoin 价格动态，加入会增加篇幅，并不能直接补强 AML 账户分类的横向定位。
- GFL 与 TOGL 是方法来源，不与金融 AML 论文共享任务协议。
- A 为当前稿件尚未列出的文献，未来若采用此图，需要在正文参考文献中新增；其余节点对应已有引用。原稿目前没有新增参考文献或替换 Table 1。

## 后续整合时可用的英文图注

> Representative graph-based research and the positioning of TopoAML. (a) Selected milestones from 2008 to 2026. (b) Categorical positioning by representation construction and analysed objects. General PH graph-learning methods are shown separately from financial applications. The map describes methodological scope; the studies use different tasks and evaluation protocols.

## A 的新增文献（仅供后续整合）

Chang, R., Lee, A., Ghoniem, M., Kosara, R., Ribarsky, W., Yang, J., Suma, E., Ziemkiewicz, C., Kern, D., & Sudjianto, A. (2008). Scalable and interactive visual analysis of financial wire transactions for fraud detection. *Information Visualization, 7*(1), 63–76. https://doi.org/10.1057/palgrave.ivs.9500172
