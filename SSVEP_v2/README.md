# SSVEP 异步数据集复现（易读版）

这是对 `asyn-SSVEP` 原复现代码（`SSVEP_v1/` 目录）的**可读性重写**：算法、分折规则、参数和结果数值保持一致，
但代码按“数据怎么流动”重新组织，删掉了让初学者卡住的缩写、多层转置和工程化样板。

一句话概括整个项目：

> 给一段脑电，判断受试者看的是 8 个闪烁频率中的哪一个（图 5、图 6）；
> 不给提示时，判断他此刻在看哪个频率或者处于空闲状态（图 7）。

---

## 1. 先跑起来

```powershell
conda activate SSVEP                        # Python 3.11 环境
cd SSVEP_v2                          # 本项目根目录

python quick_start.py                       # 单文件教学版，约 30 秒，打印每一步的形状
python -m unittest discover -s tests -v     # 11 个单元测试，不依赖原始数据
python run.py --subjects S01                # 跑一个人的全流程，结果在 results/runs/
```

全量运行（24 人，需要数小时）：

```powershell
python run.py                        # 数据检查 → 质量分析 → 频率识别 → 异步分类 → 汇总
python run.py --stage verify         # 独立核验已有结果
```

**数据准备**：从原文数据集（figshare，见文末）下载后放成如下结构。原始 MAT 文件不会被修改，缓存和结果都是派生文件。

```
asyn-SSVEP/
├── EEG_data/                     ← 四个版本共用的原始数据
│   ├── S01/sub_01_data_CS.mat, sub_01_data_NS1.mat, sub_01_data_NS2.mat, sub_01_data_NS3.mat
│   ├── ... S24/
│   ├── Channel Loc.xlsx          ← 官方电极顺序，程序按它选通道
│   └── Subjects' Info.xlsx
└── SSVEP_v2/
    ├── cache/                    ← 本地预处理缓存（联接到 v1/cache，不提交 Git）
    └── results/                  ← 本版本实验结果
```

---

## 2. 整个流程一张图

```
../EEG_data/Sxx/*.mat
      │  data.load_state()：选 9 个后部电极 → 去均值 → 5000→250 Hz 抗混叠降采样 → 50 Hz 陷波
      ▼
CS  [8 频率, 12 重复, 9 通道, 1125 点]      NS [原始试次, 9 通道, 1000/500 点]
      │ crop_cs()：从 0.14 s 视觉延迟后截窗         │ segment_ns()：切成连续 1 秒片段
      ▼                                              ▼
algorithms.filter_bank()：同一段信号变成 4 个子带
      │
      ├── FBCCA ─── 与 8 个频率的正余弦参考基比典型相关，4 子带加权求和   → 八分类（图 5、图 6）
      ├── TRCA ──── 训练空间滤波器 + 模板（单类）                          → 八分类
      ├── eTRCA ─── 训练空间滤波器 + 模板（八类联合）                      → 八分类
      │
      └── 异步任务：eTRCA 选出候选频率 → 每个频率一个二分类 LDA 判断 CS/NS → 九分类（图 7）
            IFB-OCN：Fp1 单通道 → 递归网络 → 训练内选频带/参数 → 线性 SVM → 二分类（图 7）
      ▼
results/：逐试次预测 CSV、准确率/ITR 表、图 5/6/7、质量图、REPORT.md、verification.json
```

---

## 3. 建议的阅读顺序

| 顺序 | 文件 | 一句话职责 | 阅读重点 |
|---:|---|---|---|
| 1 | `quick_start.py` | 单文件教学版 | 看“一段脑电怎么变成 8 个分数” |
| 2 | `ssvep/config.py` | 读配置 | 三个目录与常用参数 |
| 3 | `ssvep/data.py` | 读 MAT、选通道、预处理、切窗、缓存 | 数组形状约定、缓存键 |
| 4 | `ssvep/algorithms.py` | 滤波器组、FBCCA、TRCA/eTRCA、ITR | `fit_trca` 的广义特征值问题 |
| 5 | `ssvep/frequency.py` | 八分类交叉验证 | 12 折怎么分、指标怎么算 |
| 6 | `ssvep/asynchronous.py` | 异步九分类 + 二分类 | 样本怎么来、分折为什么按原始试次 |
| 7 | `ssvep/ocn.py` | IFB-OCN 特征与训练内选择 | 延迟嵌入 → 网络统计量 |
| 8 | `ssvep/quality.py` | 63 通道质量分析 | SNR 公式、地形图 |
| 9 | `ssvep/report.py` | 汇总、画图、写报告 | 只做聚合，不参与算法 |
| 10 | `ssvep/verify.py` | 独立核验 | 从预测 CSV 重算指标 |
| 11 | `run.py` | 命令行入口 | 阶段与输出目录 |

模块之间是**单向依赖**：`data → algorithms → frequency/asynchronous/quality → report/verify`，
不会出现互相调用，所以可以按上面顺序逐层读。更细的逐模块讲解见 `GUIDE.md`。

---

## 4. 论文方法 ↔ 代码对照

| 论文/协议内容 | 代码位置 | 关键参数（config.json） |
|---|---|---|
| 降采样到 250 Hz、50 Hz 陷波 | `data.preprocess` | `eeg.original_fs`、`eeg.fs`、`eeg.notch_hz`、`eeg.notch_q` |
| 后部 9 通道、Fp1 | `data.channel_indices` | `paradigm.posterior_channels`、`paradigm.frontal_channel` |
| 视觉延迟 0.14 s、提示 1.5 s | `data.crop_cs`、`algorithms.decision_seconds` | `paradigm.delay_seconds`、`paradigm.cue_seconds` |
| 四个子带 6–80/14–80/22–80/30–80 Hz | `algorithms.filter_sections` | `filter_bank.passbands`、`stopbands` |
| 子带权重 `m^-1.25+0.25` | `algorithms.band_weights` | `filter_bank.weight_exponent`、`weight_offset` |
| 参考信号：基频 + 3 次谐波 | `algorithms.reference_bases` | `paradigm.harmonics` |
| FBCCA 相关系数与加权融合 | `algorithms.fbcca_correlations`、`fbcca_scores` | `filter_bank.score_power` |
| TRCA/eTRCA 空间滤波与模板 | `algorithms.fit_trca`、`trca_correlations` | `trca.ridge_relative` |
| 12 折交叉验证 | `frequency.run_frequency` | `frequency.folds` |
| ITR 公式 | `algorithms.itr` | `frequency.windows_seconds` |
| 异步 CS/NS 切窗与分折 | `asynchronous._build_task`、`_evaluate_task` | `asynchronous.*` |
| 频率特定 LDA | `asynchronous._train_lda_models` | `asynchronous.lda_shrinkage` |
| IFB-OCN 五频带、延迟嵌入、网络统计量 | `ocn.extract_features`、`recurrence_features` | `ocn.*` |
| 训练内选参数/选频带、线性 SVM | `ocn.evaluate_ocn_fold` | `ocn.inner_folds`、`top_bands`、`svm_c` |
| 12 Hz 响应、SNR 公式 (2/Δf+1)、地形图 | `quality.run_quality`、`snr_spectra` | `paradigm.frequencies` |

---

## 5. 参数怎么改（config.json）

配置文件按用途分组，改完直接重跑即可：

| 分组 | 内容 | 什么时候改 |
|---|---|---|
| `subjects` | 24 个受试者编号 | 只分析部分人时 |
| `paths` | `data` / `cache` / `output` 三个目录 | 数据放在别处时 |
| `eeg` | 原始/目标采样率、陷波频率与 Q | 换数据集或预处理时 |
| `paradigm` | 8 个频率、9 通道、Fp1、延迟、提示、谐波数 | 换实验范式时 |
| `filter_bank` | 4 个子带的通带/阻带、权重指数、`score_power` | 改滤波器组时 |
| `trca` | 岭正则（相对迹的 1e-8） | 数值不稳定时 |
| `frequency` | 窗长列表、折数 | 只跑 1–2 秒窗时 |
| `asynchronous` | 窗长、折数、`fs_score_power`、LDA 收缩 | 改异步协议时 |
| `ocn` | 5 个候选频带、延迟/维度/半径网格、内层折数、SVM 的 C | 改网络法时 |
| `random_seed` | 分层分组内层折的随机种子 | 需要完全复现时不要动 |

示例：

```powershell
python run.py --windows 1 2          # 只跑 1 秒和 2 秒窗长
python run.py --subjects S01 S02 --stage async
```

**缓存会自动失效**：`cache/` 里的文件名由“源文件大小/修改时间 + 预处理参数 + 通道”哈希得到，
改了参数就会生成新文件，旧缓存不会被误用。想强制重算，删掉 `cache/` 即可。

---

## 6. 输出文件

```
results/
├── data_inventory.csv                 96 个 MAT 的变量名、维度、字节数
├── manifest_<stage>.json              本次运行的参数、软件版本、源码哈希
├── frequency/
│   ├── Sxx_predictions.csv            每人 1152 条：窗长 × 算法 × 折 × 真假类别
│   └── Sxx_metrics.csv                每人 12 行：准确率、ITR、决策时间
├── asynchronous/
│   ├── Sxx_predictions.csv            每人 1728 条：状态 × 算法 × 样本
│   ├── Sxx_metrics.csv                每人 9 行：九分类/二分类指标
│   └── Sxx_band_selection.json        每折选中的频带、网络参数、内层准确率
├── quality/
│   ├── figure4_1_waveform_spectrum.png
│   ├── figure4_2_harmonic_topographies.png
│   ├── harmonics_all_channels.csv     24×63×4 个谐波数值
│   ├── spectra_by_subject.csv         每人 Oz 全频谱
│   ├── waveform.csv / grand_average_all_channels.csv / harmonics.csv
├── frequency_summary.csv              三算法 × 四窗长的跨人均值/SD
├── asynchronous_summary.csv           三状态 × 三算法的跨人汇总
├── frequency_paper_comparison.csv     4 秒准确率与原文的差值
├── asynchronous_paper_comparison.csv  1 秒三状态与原文的差值
├── figure5_accuracy.png / figure6_itr.png / frequency_confusion.png / figure7_asynchronous.png
├── REPORT.md / validation.txt / verification.json
└── runs/<标签>/                       指定 --subjects 或 --windows 时的独立输出
```

---

## 7. 与原复现代码的差异

**保留（结果可与原版逐条对照）**

- 算法公式、滤波器参数、分折规则、切窗起点、ITR 定义、IFB-OCN 的重建规则完全一致。
- 输出 CSV 的列名、图文件名、报告结构保持一致。
- 24 人全量实测（70,488 行逐试次预测）与原版**逐条相同**，汇总表数值差为 0，
  质量分析 7 个输出文件与原版 SHA256 完全相同（见第 9 节）。

**简化（为了好读）**

- 10 个平铺的 `.py` → `ssvep/` 包内 9 个按流程排列的模块；入口 `run.py`。
- 去掉“完成签名 + `--resume`”的断点续算机制（改由 `--skip-existing` 按文件是否存在跳过）。
- 去掉缩写（`bank`、`skt`、`fbo` 等）和多层 `reshape/transpose` 的隐式约定，统一说明形状。
- 配置从 35 个平铺键改为按用途分组的嵌套结构。
- 缓存键加入版本号，与旧版缓存互不干扰。

**没有改变的边界（务必先读）**

- IFB-OCN 仍是 **reconstructed** 版本：原论文的完整参数优化细节不可得，这里的网格搜索是显式替代规则，不能当成逐参数复刻。
- 异步的九分类准确率与 OCN 的二分类准确率**不能直接排名比较**。
- S11 存在疑似类别循环偏移，主实验保留原始标签未重标注（见 `DATA_AUDIT.md`）。
- 地形图使用标准 10–20 坐标，不是实测电极位置。
- 完整 epoch 的零相位滤波是离线处理，不能宣称为实时因果处理。

---

## 8. 常见问题

**为什么从 0.14 秒开始截窗？**
刺激出现后视觉通路需要约 0.14 秒才把信号送到皮层。异步部分论文明确使用了该延迟，频率识别部分沿用同一假设。

**为什么频率识别按“重复编号”分折、NS 按“原始试次”分折？**
CS 的 12 次重复是相互独立的原始试次，可以按编号轮换；NS 的 1 秒片段来自同一个 4 秒试次，如果按片段随机分折，训练和测试会共享同一次试次，属于数据泄漏。

**“内部留一”是什么？**
异步任务训练 CS 特征时，如果这次试次参与了自己的模板，特征会虚高。所以每算一个训练样本的特征，都把它从模板里剔除一次。它只用外层训练数据，不额外消耗测试数据。

**为什么先平均再算质量分析的频谱？**
预处理是线性运算，先对 12 次重复平均再滤波，结果相同但省内存和时间。

**一次全量运行要多久？**
首次运行要读 96 个 MAT（约 20 GB）并预处理，是主要耗时；之后有缓存。频率识别每位受试者约数十秒，异步分类约 1 分钟（含网络特征）。

**能不能只跑图？**
可以：`python run.py --stage report` 只读取已有个体结果重新汇总画图。

---

## 9. 复现验证

- 单元测试 11 个，覆盖 ITR 边界、秩亏 CCA、合成频率识别、训练/测试隔离、NS 分组切窗、网络统计量、SNR 边界、OCN 选择不看测试标签。
- 24 人全量运行结果与原版逐条比对（`tools/compare_with_baseline.py`）：

```powershell
# 在 SSVEP_v2 目录执行，与同级 SSVEP_v1 的结果对照。
python tools/compare_with_baseline.py --new results --baseline ../SSVEP_v1/results_full
```

  结果：120 个文件、70,488 行数据全部一致（predictions 逐条相同，metrics 与选带日志在 1e-12 内）。
- 汇总与作图同样一致：`frequency_summary.csv`、`asynchronous_summary.csv`、两张原文对照表的数值差为 0，
  `figure5/6/7`、混淆矩阵与基线图片字节数相同；质量分析 7 个输出文件 SHA256 完全相同。
- 独立核验（`ssvep/verify.py`）：27,648 条频率预测 / 41,472 条异步预测的数量、分折、标签与重算指标全部 `passed`，
  结论写入 `results/verification.json`。

| 复现指标（24 人均值） | 复现值 | 原文值 |
|---|---:|---:|
| FBCCA 4 s 准确率 | 92.27% | 93.49% |
| TRCA 4 s 准确率 | 88.63% | 73.00% |
| eTRCA 4 s 准确率 | 89.84% | 79.47% |
| IFB-OCN 1 s，NS1/2/3 | 59.22 / 61.70 / 68.60% | 70.51 / 72.35 / 69.34% |
| FS-eTRCA 1 s，NS1/2/3 | 73.94 / 72.14 / 72.48% | 67.64 / 66.47 / 65.34% |
| FS-eTRCA&FBCCA 1 s，NS1/2/3 | 81.86 / 80.34 / 79.84% | 76.39 / 75.35 / 75.02% |

> 数值接近不代表方法已精确复刻；IFB-OCN 仍是重建版，且二分类（OCN）与九分类（FS）的准确率不可直接比较。

---

## 10. 目录结构

```
SSVEP/
├── config.json            全部参数
├── run.py                 命令行入口
├── quick_start.py         单文件教学版（看这个最快）
├── ssvep/
│   ├── config.py
│   ├── data.py
│   ├── algorithms.py
│   ├── frequency.py
│   ├── asynchronous.py
│   ├── ocn.py
│   ├── quality.py
│   ├── report.py
│   └── verify.py
├── tests/
│   ├── test_core.py
│   └── test_ocn_quality.py
├── tools/
│   └── compare_with_baseline.py
├── README.md              本文件
├── GUIDE.md               逐模块代码导读
├── REPRODUCTION.md        方法依据、实现假设与原文数值
├── DATA_AUDIT.md          S11 数据异常说明
├── environment.yml        Conda 环境
└── .gitignore
```

---

## 引用

1. 数据集论文：<https://doi.org/10.1080/27706710.2024.2418650>
2. 官方数据：<https://doi.org/10.6084/m9.figshare.24906300.v3>
3. FBCCA：<https://doi.org/10.1088/1741-2560/12/4/046008>
4. TRCA：<https://doi.org/10.1109/TBME.2017.2694818>
5. FS 框架：<https://doi.org/10.1109/TNSRE.2023.3246359>
6. OCN 网络定义：<https://doi.org/10.1155/2019/2167871>
7. IFB-OCN：<https://doi.org/10.1016/j.jneumeth.2020.108776>
