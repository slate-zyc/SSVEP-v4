# 文件与运行流程说明

## 阅读顺序

建议先读 `main.py` 的模块说明，再依次读 `data_io.py`、`algorithms.py`、`evaluation.py`、`asynchronous.py`、`ocn.py`，最后读绘图与核验模块。所有Python文件都以三引号说明用途，主要函数有文档字符串，关键数组操作和防泄漏步骤有中文注释。

## Python程序

|文件|职责|主要输入|主要输出|
|---|---|---|---|
|main.py|解析命令、运行各阶段、保存环境与代码快照、支持断点续算|config.json、命令行参数|各阶段输出、完成签名、manifest|
|data_io.py|核验MAT结构、匹配通道、预处理、缓存、CS/NS切窗|MAT、通道Excel|NumPy数组、原始试次分组|
|algorithms.py|四子带滤波、CCA、TRCA/eTRCA、ITR数学计算|信号数组、训练数组|相关特征、滤波器、模板、得分|
|evaluation.py|每人1–4秒八分类12折测试|九通道CS|每人频率预测和指标|
|asynchronous.py|三组CS–NS任务，训练频率特定LDA和网络法分类|CS、NS、Fp1与后部九通道|二/九分类预测、指标、选带日志|
|ocn.py|延迟嵌入、递归网络、平均度和聚类系数、内层参数与频带选择|Fp1一秒窗|网络候选特征、外层SVM预测|
|quality_analysis.py|63通道12Hz响应、频谱、原文SNR及邻域对照、标准坐标地形图|各人12Hz平均响应|质量图、全部通道谐波表|
|reporting.py|聚合全体结果、重算准确率、绘制图5/6/7与混淆矩阵|每人CSV|总体CSV、原文差值表、REPORT.md|
|test_algorithms.py|使用已知答案的合成信号/图验证算法行为|构造的数组|unittest结果|
|verify_results.py|独立核验全部真实结果、标签、分折和各指标|全体实验CSV|verification.json|

## 关键数组

原始CS：`[8频率, 63通道, 22500采样点, 12重复]`。

九通道预处理CS：`[8频率, 12重复, 9通道, 1125采样点]`。

一秒滤波器组CS：`[4子带, 8频率, 12重复, 9通道, 250采样点]`。

NS预处理：`[原始试次, 通道, 时间]`，切成一秒后为`[96片段, 通道, 250采样点]`。每片段同时保存原始试次编号。

FS候选特征：`[片段, 8个候选频率, 4个相关系数]`；融合CCA后最后一维变为8。

OCN候选特征：`[片段, 5个频带, 8组网络参数, 2个网络统计量]`。每个外层训练折选每带最佳参数及最佳三带，最终SVM输入六维特征。

## 外层与内层验证

频率识别外层12折：每折每频率一个重复测试、其余11个重复训练。

异步外层12折：每折8个CS和8个NS测试。同一NS原始试次的全部相邻片段保持同折。

FS内部留一：产生某个训练CS的模板相关特征时，再从外层训练集排除这个CS重复，避免“自己与包含自己的模板比较”。内部留一不额外使用外层测试数据。

OCN内层3折：只在外层训练样本中选择网络参数和频带；每个内层训练管线独立拟合StandardScaler与SVM，不能用外层测试准确率选带。

## 配置和说明文件

- `config.json`：24人名单、目录、采样率、频率/通道、时间窗、滤波参数、FS参数与OCN候选网格。必须保持合法JSON，不加入Python三引号或注释。
- `environment.yml`：Conda环境安装规格；实际包版本另保存在每次运行结果中的`requirements-lock.txt`。
- `README.md`：PyCharm解释器选择和常用运行命令。
- `REPRODUCTION.md`：方法依据、实现假设、重建版边界、原文均值和验证标准。
- `.gitignore`：设置原始数据、缓存、实验结果和本地配置的版本管理忽略规则。

## 输入和缓存

- `../EEG_data/S01`至`S24`：用户下载的96个原始MAT，不改动。
- `../EEG_data/Channel Loc.xlsx`：电极索引和名称，参与所有通道选择。
- `../EEG_data/Subjects' Info.xlsx`：受试者信息，目前不作为预测特征，也不按人口学信息筛选样本。
- `cache/*_CS_*.npy`与`*_NS*_*.npy`：选定通道预处理缓存。
- `cache/*_quality_*.npy`：63通道12Hz平均响应缓存。
- `cache/*_ocn_*.npy`：无需标签的网络候选统计缓存；不存储基于测试标签选出的“最佳参数”。
- `__pycache__`：Python自动生成的字节码，不是EEG缓存。

## 全体结果目录 results_full

`results_full`保存全体24人的实验结果。

|文件或模式|内容|
|---|---|
|data_inventory.csv|96个MAT的变量名、形状、类型与字节数|
|manifest_all.json|实验参数、时间、Python及顶层包版本、全部Python源码SHA256|
|requirements-lock.txt|运行时全部已安装包的版本快照|
|frequency/Sxx_predictions.csv|每人1152条八分类预测，保留窗长、算法、折号、真实及预测类|
|frequency/Sxx_metrics.csv|每人12行准确率/ITR，共288行|
|frequency/Sxx_complete.json|该人该阶段成功标记及来源/参数/代码签名|
|asynchronous/Sxx_predictions.csv|每人1728条异步预测，保留状态、算法、样本与原始分组|
|asynchronous/Sxx_metrics.csv|每人9行异步指标，共216行|
|asynchronous/Sxx_band_selection.json|每人三状态12折的网络参数、前三频带与内层成绩|
|asynchronous/Sxx_complete.json|异步阶段成功标记|
|frequency_metrics.csv|合并的288行频率指标|
|frequency_summary.csv|三算法×四窗长的24人均值/SD，共12行|
|asynchronous_metrics.csv|合并的216行异步指标|
|asynchronous_summary.csv|三NS×三算法的跨人汇总，共9行|
|frequency_paper_comparison.csv|4秒准确率与原文的百分点差值|
|asynchronous_paper_comparison.csv|1秒三状态结果与原文差值，明确非精确方法匹配|
|figure5_accuracy.png|准确率曲线及各算法的逐人箱线图|
|figure6_itr.png|ITR曲线及箱线图|
|frequency_confusion.png|最长窗三算法的八分类混淆矩阵|
|figure7_asynchronous.png|三组异步任务的均值/SD和逐人箱线图，注明二/九分类区别|
|quality/waveform_spectrum.png|Oz时域波形、幅值谱及邻域SNR曲线|
|quality/harmonic_topographies.png|12/24/36/48Hz标准坐标头皮图|
|quality/harmonics_all_channels.csv|24人×63通道×4谐波的6048行数值|
|quality/harmonics.csv|从上表提取Oz的96行谐波指标|
|quality/spectra_by_subject.csv|每人Oz全频谱；snr_paper_formula为原文(2/Δf+1)公式，snr_neighbor_ratio排除中心频点|
|quality/waveform.csv|平均Oz响应与显示滤波波形|
|quality/grand_average_all_channels.csv|63通道跨人平均响应|
|REPORT.md|自动生成的全体实验报告和方法限制|
|validation.txt|汇总时的预测数量与分组核验记录|
|verification.json|独立核验脚本生成的计数及验证结论|
|runs/（按需生成）|指定--subjects或--windows时的独立运行输出|

类别编号：频率类0–7依次为8/9/10.5/11/12/13/14/15Hz；FS类8为NS。IFB-OCN重建版单独使用0=NS、1=CS。两套标签语义不可混用。

## 常用操作

PyCharm直接运行`main.py`即可执行全部；程序参数填`--resume`可复用签名匹配的已完成阶段。

只重建报告使用`--stage report`，不重新训练；运行`verify_results.py`可独立核验全体结果。

不要在运行过程中修改配置或源码；确需修改时先停止当前进程，改好后重新运行，避免一次运行跨越两版实现。
