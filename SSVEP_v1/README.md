# SSVEP 论文复现项目

项目路径：`E:\Codex\asyn-SSVEP\SSVEP_v1`

## PyCharm 配置

1. 使用 PyCharm 打开本项目目录。
2. 在 Settings → Project → Python Interpreter 中选择 Add Interpreter → Add Local Interpreter → Conda → Existing environment（不同版本菜单名称可能略有差异）。
3. 选择环境 `SSVEP`，解释器路径为 `E:\conda-envs\SSVEP\python.exe`。若需指定 Conda 程序，使用 `D:\ProgramData\miniconda3\Scripts\conda.exe`。
4. Python 运行配置的 Working directory 使用本项目目录。

Conda 环境名为 `SSVEP`，使用 Python 3.11。该环境隔离 Python 包依赖，不提供操作系统级安全隔离。

## 命令行验证

```powershell
conda env list
conda activate SSVEP
python --version
python -m pip --version
```

## 运行复现

在 PyCharm 打开 `main.py`，使用项目解释器运行，无须填写参数。默认执行24人的数据检查、63通道质量分析、FBCCA/TRCA/eTRCA的1–4秒八分类，以及三种NS下的异步分类。

数据目录应为 `../EEG_data/S01` 至 `../EEG_data/S24`，每人含四种状态的 MAT 文件；`../EEG_data/Channel Loc.xlsx` 提供官方通道顺序。代码使用脚本所在目录定位数据，不依赖终端当前目录。

```powershell
conda activate SSVEP
python main.py
```

单独检查数据或快速调试：

```powershell
python main.py --stage audit
python main.py --subjects S01 --windows 1
python main.py --stage async
python main.py --resume
python main.py --stage report
python -m unittest -v test_algorithms
python verify_results.py
```

默认结果写入 `results_full/`，包括 `REPORT.md`、两种任务的汇总CSV、逐试次预测、混淆矩阵、图5/6/7及质量图。指定受试者或时间窗运行时，输出写入独立的 `results_full/runs/` 子目录。

`--resume` 可跳过签名一致且输出完整的受试者阶段；原始文件属性、通道表、配置或源码变化时会重新计算。原始数据保持原样，缓存和结果均为派生文件。逐人处理不会同时载入全部20GB数据。

所有Python源文件均以三引号模块说明开头，函数和关键计算步骤有中文说明。JSON/YAML保留各自合法语法，其参数说明见 `REPRODUCTION.md`。

## 方法状态

频率识别和两种FS异步框架已实现。IFB-OCN实现了网络特征、训练内选带和线性SVM，但原始完整参数优化规则未取得，故明确命名为 **IFB-OCN-reconstructed**。它不是已经验证的精确原作者复刻，不能省略这一限定。

`config.json` 可调整受试者、时间窗和参数；全部方法假设、来源、分折规则及输出解释见 `REPRODUCTION.md` 与 `FILES.md`。
