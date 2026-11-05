# 论文第 4 章复现（4.1 信号质量、4.2 频率识别、4.3 状态分类）

只复现这一份数据集，不做工程化封装：一个参数文件、一个数据文件、每个小节一个主脚本，
主脚本按数据流从上到下线性执行。

## 每节复现什么

| 小节 | 论文内容 | 图 | 本仓库脚本 |
|---|---|---|---|
| 4.1 | 12 Hz CS 信号的平均波形、幅度谱、谐波头皮图、式(3) SNR | 图 4 | `reproduce_4_1.py` |
| 4.2 | 8 分类频率识别：FBCCA、TRCA、eTRCA | 图 5、图 6 | `reproduce_4_2_fbcca.py`、`reproduce_4_2_trca_etrca.py` |
| 4.3 | CS/NS 状态分类：IFB-OCN（二分类）、FS-eTRCA 与 FS-eTRCA&FBCCA（九分类） | 图 7 | `reproduce_4_3_ifb_ocn.py`、`reproduce_4_3_fs.py` |

## 文件

| 文件 | 内容 |
|---|---|
| `config.py` | 全部固定参数：路径、采样率、频率、子带、窗口、原文参考值 |
| `data.py` | 读 MAT 与预处理：12 Hz 平均响应、全部 CS 试次、NS 原始记录 |
| `reproduce_4_1.py` | 24 人平均 → 幅度谱 → 式(3) SNR → 波形/谱/头皮图 |
| `reproduce_4_2_fbcca.py` | FBCCA：参考信号 + CCA → 12 折 → 准确率/ITR |
| `reproduce_4_2_trca_etrca.py` | TRCA/eTRCA：空间滤波器 + 模板相关 → 12 折 |
| `reproduce_4_3_ifb_ocn.py` | IFB-OCN 重建版：Fp1 → 递归网络 → 训练内选带 → 线性 SVM（二分类） |
| `reproduce_4_3_fs.py` | FS-eTRCA / FS-eTRCA&FBCCA：eTRCA 选候选频率 + 频率特定 LDA（九分类） |
| `plot_paper_figures.py` | 只读已有结果 CSV，把 4.2、4.3 的结果画成论文图 5/6/7 |

## 运行

```powershell
conda activate SSVEP
python reproduce_4_1.py            # 约 2.5 分钟
python reproduce_4_2_fbcca.py      # 约 2.6 分钟
python reproduce_4_2_trca_etrca.py # 约 3.5 分钟
python reproduce_4_3_fs.py         # 约 5.7 分钟
python reproduce_4_3_ifb_ocn.py    # 约 16.6 分钟（逐片段构图最费时）
python plot_paper_figures.py       # 约 3 秒（只读 CSV，不重新计算）
```

数据目录为大项目根目录下的 `../EEG_data/`（S01~S24 与 `Channel Loc.xlsx`），四个版本直接读取同一份数据，无须创建目录联接。

## 输出

### 4.1 → `results/section4_1/`

`waveform_oz.csv`、`spectrum_snr_oz.csv`、`harmonics_all_channels.csv`、
`comparison_with_paper.csv`、`figure4.png`（四面板图 4）。

### 4.2 → `results/section4_2_fbcca/`、`section4_2_trca/`、`section4_2_etrca/`

三个目录结构相同：`predictions.csv`（9216 条逐试次预测）、`subject_metrics.csv`（96 行）、
`summary.csv`、`comparison_with_paper.csv`、`confusion_matrices.csv`（256 行）、
`*_accuracy_itr.png`。

### 4.3 → `results/section4_3_ifb_ocn/`、`section4_3_fs/`

| 文件 | IFB-OCN | FS |
|---|---|---|
| `predictions.csv` | 13824 条（192 段 × 3 状态 × 24 人） | 27648 条（× 2 方法） |
| `subject_metrics.csv` | 72 行 | 144 行 |
| `summary.csv` | 3 行 | 6 行 |
| `comparison_with_paper.csv` | 3 行 | 6 行 |
| `confusion_matrices.csv` | 12 行（2×2 × 3） | 486 行（9×9 × 6） |
| `selected_bands.csv` | 2592 行（训练内选带记录） | — |
| 图 | `ifb_ocn_accuracy.png` | `fs_accuracy.png` |

### 论文图 5/6/7 → `results/figures/`

| 图 | 内容 |
|---|---|
| `figure5.png` | 三种算法的准确率随 1~4 秒窗长变化：均值 ± SEM 曲线 + 各算法 24 人箱线图 |
| `figure6.png` | 三种算法的 ITR 随窗长变化：同样的四面板布局 |
| `figure7.png` | CS 与 NS1/NS2/NS3 分类：三种方法的均值柱状图 + 每种状态的个体分布 |

图上空心菱形是原文均值；图 7 底部注明了“IFB-OCN 是二分类、FS 是九分类，准确率不可直接比较”。
这三张图由 `plot_paper_figures.py` 直接读取上面各节的 `subject_metrics.csv` 生成，不重新计算实验。

## 计算要点

**4.1**
- 每人先对 12 Hz 的 12 次重复平均，再预处理；24 人再等权平均（每人 1/24）。
- 幅度谱只取刺激后 0~4 秒（1000 点），不补零不加窗，分辨率 0.25 Hz。
- SNR 按原文式 (3) 在**跨人平均后的幅度谱**上计算：`(2/Δf+1)·Amp(f)/Σ邻域Amp`。
- 头皮图用标准 10–20 坐标（数据无实测位置），图中已注明。

**4.2（三种方法共用）**
- 9 个顶枕电极、4 个子带（通带 `[m*8-2, 80]` Hz）、12 折交叉验证。
- 打分都是 `Σ_m w_m·ρ_m²`（`w_m = m^-1.25+0.25`），取最大。
- ITR 先按每个人的准确率计算，再跨 24 人平均；决策时长 = 窗长 + 1.5 s 提示 + 0.14 s 延迟。
- FBCCA 用每个频率的正弦/余弦参考（5 次谐波）求 CCA；TRCA/eTRCA 用每折训练的空间滤波器
  （广义特征值问题 `S w = λ Q w`）和平均模板。

**4.3（两种任务）**
- 样本：CS 取 0.14~1.14 秒共 96 段；NS 每条记录切不重叠 1 秒段共 96 段；
  每个任务 192 段，12 折，每折 8 CS + 8 NS，同一条 NS 记录的所有片段同折。
- IFB-OCN：只用 Fp1，5 个候选频带、8 组（延迟、嵌入维度、半径）参数；
  每折在训练段内部做 3 折验证，为每个频带选参数、再选前 3 个频带拼成 6 维特征，
  最后线性 SVM 做 CS/NS 二分类（0=NS，1=CS）。
- FS：用 eTRCA 相关平方加权选出候选频率，再用该频率的 LDA 判断接受（输出 0~7）或拒绝（输出 8）；
  融合版在 LDA 输入里再加 4 维 FBCCA 相关。CS 正样本特征用内部留一，避免自己出现在自己的模板里。

## 复现结果（24 人平均）

| 项目 | 复现值 | 原文值 |
|---|---:|---:|
| 4.1 12 Hz 幅度 | 1.6596 | 1.9006 |
| 4.1 12 Hz SNR | 5.0607 | 6.4196 |
| 4.2 FBCCA 4 s 准确率 | 92.97% | 93.49% |
| 4.2 TRCA 4 s 准确率 | 92.80% | 73.00% |
| 4.2 eTRCA 4 s 准确率 | 93.75% | 79.47% |
| 4.2 FBCCA 1 s ITR | 42.2603 | 43.8325 |
| 4.2 TRCA 1 s ITR | 32.3027 | 29.1990 |
| 4.2 eTRCA 1 s ITR | 37.2068 | 33.3243 |
| 4.3 IFB-OCN，NS1/NS2/NS3 | 58.22 / 61.96 / 68.19% | 70.51 / 72.35 / 69.34% |
| 4.3 FS-eTRCA，NS1/NS2/NS3 | 74.37 / 73.16 / 73.44% | 67.64 / 66.47 / 65.34% |
| 4.3 FS-eTRCA&FBCCA，NS1/NS2/NS3 | 82.03 / 81.21 / 80.49% | 76.39 / 75.35 / 75.02% |

**验证**：与参考实现逐位一致——四个小节共 30 个 CSV、6 张 PNG，数值最大差均为 0，
PNG 的 SHA256 相同。

**关于与原文的差距**：
- 4.1 的幅度/SNR 复现值比原文低一些，参考实现也是同样的数字。
- 4.2 的 TRCA/eTRCA 准确率明显高于原文均值（原文里 FBCCA 反而最高）。
- 4.3 的 IFB-OCN 是明确标注的重建版（原论文完整参数不可得），比原文低；
  两种 FS 方法则高于原文均值。二分类（IFB-OCN）与九分类（FS）准确率不可直接比较。
- 本仓库以参考实现的口径为准，没有根据测试结果反向调整任何参数，差距如实列出。
