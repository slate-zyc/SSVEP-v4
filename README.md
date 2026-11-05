# asyn-SSVEP：异步 SSVEP 论文复现

本项目复现 Zhao 等人于 2024 年发表的 **An EEG dataset for studying asynchronous steady-state visual evoked potential (SSVEP) based brain computer interfaces**，保留四种代码组织形式及其已有实验结果，便于学习、核对算法和比较实现。

- **目标论文**：[期刊论文 / DOI](https://doi.org/10.1080/27706710.2024.2418650)
- **官方数据集**：[YSU asynchronous SSVEP-BCI，Figshare v3](https://doi.org/10.6084/m9.figshare.24906300.v3)
- **论文原文、阅读笔记与中英对照译文**：[relevant_work_for_thesis](https://github.com/slate-zyc/relevant_work_for_thesis)

## 项目结构

```text
asyn-SSVEP/
├── README.md
├── EEG_data/                # 四个版本共用的唯一原始数据目录
├── SSVEP_v1/                # 完整模块化复现与独立结果核验
├── SSVEP_v2/                # 易读版：ssvep 包、运行入口与说明
├── SSVEP_v3/                # 按论文小节拆分的独立脚本版
└── SSVEP_v4/                # 统一 config.py / data.py 的第 4 章复现
```

四个版本直接位于大项目根目录；原 worktree 包装层已合并。每个版本独立保存结果，读取数据时都根据脚本位置定位 `../EEG_data`，不依赖启动终端的工作目录。

| 版本 | 主要入口 | 已有结果 | 进一步说明 |
| --- | --- | --- | --- |
| v1 | `main.py`、`verify_results.py` | `SSVEP_v1/results_full/` | [README](SSVEP_v1/README.md)、[复现协议](SSVEP_v1/REPRODUCTION.md)、[参数指南](SSVEP_v1/CONFIG_GUIDE.md) |
| v2 | `run.py`、`quick_start.py` | `SSVEP_v2/results/` | [README](SSVEP_v2/README.md)、[学习指南](SSVEP_v2/GUIDE.md) |
| v3 | `reproduce_4_1.py` 及 `reproduce_4_2_*.py`、`reproduce_4_3_*.py` | `SSVEP_v3/results_full/` | [README](SSVEP_v3/README.md) |
| v4 | 各节 `reproduce_*.py`、`plot_paper_figures.py` | `SSVEP_v4/results/` | [README](SSVEP_v4/README.md) |

## 复现内容

1. **论文 4.1 节，图 4**：12 Hz 控制态的时域波形、幅度谱、谐波头皮图与 SNR。
2. **论文 4.2 节，图 5、图 6**：FBCCA、TRCA、eTRCA 的八分类频率识别、准确率与信息传输率（ITR）。
3. **论文 4.3 节，图 7**：CS 与 NS1/NS2/NS3 的异步分类，包括 FS-eTRCA、FS-eTRCA&FBCCA 和 IFB-OCN 重建版。

## 数据与环境

数据集包含 24 名受试者、63 通道、5000 Hz 原始采样率，以及 CS、NS1、NS2、NS3 四种状态。下载后保持官方文件名，并按以下方式放入根目录：

```text
EEG_data/
├── Channel Loc.xlsx
├── Subjects' Info.xlsx
├── S01/
│   ├── sub_01_data_CS.mat
│   ├── sub_01_data_NS1.mat
│   ├── sub_01_data_NS2.mat
│   └── sub_01_data_NS3.mat
└── ... S24/
```

本地保留全部 96 个 MAT 文件和两个 Excel 文件，约 18.7 GiB。**原始数据和预处理缓存不提交到 Git**；远端保留源码、说明、现有实验结果及数据下载指引。数据使用应遵循官方发布页的许可和引用要求。

推荐使用 Conda / Python 3.11。已有环境为 `SSVEP`；新机器可在项目根目录执行：

```powershell
conda env create -f SSVEP_v1/environment.yml
conda activate SSVEP
```

本机 PyCharm 可选择已有解释器 `E:\conda-envs\SSVEP\python.exe`。四个版本共用该环境，分别选择版本目录作为运行配置的工作目录。

## 运行示例

以下命令从项目根目录启动；完整实验耗时较长，可先检查数据和现有结果。

```powershell
conda activate SSVEP
python SSVEP_v1/main.py --stage audit
python SSVEP_v1/main.py
python SSVEP_v1/verify_results.py
python SSVEP_v2/run.py --stage audit
python SSVEP_v3/reproduce_4_1.py
python SSVEP_v4/reproduce_4_1.py
python SSVEP_v4/plot_paper_figures.py
```

v3、v4 其余实验的运行顺序见对应 README。各版本保留自身的参数口径；版本编号表示代码组织演进，不表示它们的结果必然完全一致。

## 复现边界

- 这是基于公开论文和算法参考资料的独立复现。均值接近原文，或本地验证通过，都不能证明逐参数精确复刻。
- **IFB-OCN-reconstructed 为重建实现**：原始方法的完整参数优化细节未全部核实，不能省略该限定。
- IFB-OCN 输出 CS/NS 二分类，FS 框架输出八个频率加 NS 的九分类；两类准确率不能直接排名。
- v1/v2 中的数据审计保留了 S11 疑似标签循环偏移的观察；主实验使用原始标签，未擅自重标注。详见 [数据审计](SSVEP_v1/DATA_AUDIT.md)。
- 原始数据保持不变；已有结果和运行记录按原样保留，记录中的旧绝对路径反映当时的运行位置。

## 参考文献与算法项目

| 用途 | 来源 |
| --- | --- |
| 目标数据集论文 | Zhao et al., 2024，[DOI: 10.1080/27706710.2024.2418650](https://doi.org/10.1080/27706710.2024.2418650) |
| 官方数据 | [Figshare: 10.6084/m9.figshare.24906300.v3](https://doi.org/10.6084/m9.figshare.24906300.v3) |
| FBCCA | [DOI: 10.1088/1741-2560/12/4/046008](https://doi.org/10.1088/1741-2560/12/4/046008) |
| TRCA / eTRCA | Nakanishi et al., [DOI: 10.1109/TBME.2017.2694818](https://doi.org/10.1109/TBME.2017.2694818) |
| 公开算法参考 | [mnakanishi/TRCA-SSVEP](https://github.com/mnakanishi/TRCA-SSVEP)，包含 TRCA、FBCCA、滤波器组与 ITR 的 MATLAB 实现 |
| FS 框架 | [DOI: 10.1109/TNSRE.2023.3246359](https://doi.org/10.1109/TNSRE.2023.3246359) |
| OCN 定义 | [DOI: 10.1155/2019/2167871](https://doi.org/10.1155/2019/2167871) |
| IFB-OCN 原始方法 | [DOI: 10.1016/j.jneumeth.2020.108776](https://doi.org/10.1016/j.jneumeth.2020.108776) |

详细预处理、分折规则、参数假设和验证方法见各版本说明。配套文献资料统一存放在 [relevant_work_for_thesis](https://github.com/slate-zyc/relevant_work_for_thesis)。

## 各版本运行耗时

下表按版本记录本次执行的实验脚本、测试及核验命令，单位为秒。v1 至 v3 采用运行日志记录的实际耗时。v4 标为「约」的数值，根据 PyCharm 的启动记录与最后一个结果文件的写入时间估算。

| 文件或命令 | 运行时间（秒） |
| --- | ---: |
| `SSVEP_v1` · `python -m unittest -v test_algorithms` | 3.3 |
| `SSVEP_v1/main.py` | 1338.5 |
| `SSVEP_v1/verify_results.py` | 7.5 |
| `SSVEP_v2` · `python -m unittest discover -s tests -v` | 4.0 |
| `SSVEP_v2/quick_start.py` | 2.5 |
| `SSVEP_v2/run.py` | 439.5 |
| `SSVEP_v2/run.py --stage verify` | 6.4 |
| `SSVEP_v2/tools/compare_with_baseline.py` | 1.6 |
| `SSVEP_v3/reproduce_4_1.py` | 110.3 |
| `SSVEP_v3/reproduce_4_2_fbcca.py` | 139.1 |
| `SSVEP_v3/reproduce_4_2_trca.py` | 168.8 |
| `SSVEP_v3/reproduce_4_2_etrca.py` | 182.1 |
| `SSVEP_v3/reproduce_4_3_fs.py` | 294.9 |
| `SSVEP_v3/reproduce_4_3_ifb_ocn.py` | 777.6 |
| `SSVEP_v4/reproduce_4_1.py` | 约 115 |
| `SSVEP_v4/reproduce_4_2_fbcca.py` | 约 144 |
| `SSVEP_v4/reproduce_4_2_trca_etrca.py` | 约 144 |
| `SSVEP_v4/reproduce_4_3_fs.py` | 约 277 |
| `SSVEP_v4/reproduce_4_3_ifb_ocn.py` | 约 765 |
