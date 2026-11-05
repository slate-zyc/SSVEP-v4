"""复现论文第 4.1 节：12 Hz CS 信号的平均波形、幅度谱、头皮图和 SNR。

使用大项目根目录下共享的 EEG_data；按本脚本位置定位，直接运行本文件即可。
只使用本文件，不导入原项目模块，不读取或写入预处理缓存。
依赖：numpy、scipy、pandas、openpyxl、matplotlib、mne。
"""

# ==================== 导入本文件实际使用的库 ====================
# import 负责导入模块；from ... import ... 只取出所需名称；as 为较长的模块名设置简写。
from pathlib import Path  # Path 把路径表示成对象，后面可用 / 拼接目录，用 .parent 取父目录，用 .mkdir() 创建结果目录。
from time import perf_counter  # 导入高精度、单调递增的计时器；结束值减开始值即可得到本次计算经过的秒数。

import matplotlib  # 先导入绘图库本体，以便在加载 pyplot、创建任何图窗之前选择绘图后端。
matplotlib.use("Agg")  # 使用 Agg 非交互式后端：直接将图保存为 PNG，不弹出窗口，也不需要手动关闭窗口才能继续执行。
import matplotlib.pyplot as plt  # pyplot 提供创建画布、设置绘图样式和关闭画布的接口；plt 是常用简写。
import mne  # MNE 用于读取内置标准电极坐标、创建电极信息，并根据 63 个通道的幅度绘制头皮地形图。
import numpy as np  # NumPy 负责多维数组、求均值、广播运算、快速傅里叶变换，以及频率下标等数值计算。
import pandas as pd  # pandas 用于读取通道 Excel 表、整理对照结果，并将计算结果保存为可用 Excel 打开的 CSV 表格。
from scipy.io import loadmat  # loadmat 读取 MATLAB 的 .mat 数据文件；返回字典，通过变量名 data_CS 取出脑电数组。
from scipy.signal import butter, filtfilt, iirnotch, resample_poly, sosfiltfilt  # 依次导入 Butterworth 滤波器设计、双向滤波、陷波器设计、多相重采样及二阶节双向滤波。


# ==================== 1. 固定的数据路径和实验参数 ====================
# 本文件专门处理这份数据集：固定 24 人、63 通道、8 种频率、每种频率 12 次重复，不编写其他数据集的适配分支。
# Path 中的 / 是路径拼接运算，不是普通除法；例如 Path("a") / "b" 会得到 a 目录下的 b。
DATA_DIR = Path(__file__).resolve().parent.parent / "EEG_data"  # 四个版本共用根目录数据；路径不依赖当前工作目录。
OUTPUT_DIR = Path(__file__).resolve().parent / "results_full" / "section4_1"  # __file__ 是本脚本路径；resolve() 得到绝对路径，parent 取脚本目录，因此结果始终写入新工作树。
SUBJECTS = [f"S{number:02d}" for number in range(1, 25)]  # 列表推导式生成 S01 到 S24；range 不包含右端 25，02d 表示将整数补零为两位。
ORIGINAL_FS = 5000  # 原始采样率，单位 Hz：每秒记录 5000 个时间点，4.5 秒 CS 试次共有 22500 个时间点。
FS = 250  # 降采样后的采样率：每秒 250 点，时间间隔为 1/250=0.004 秒，能够表示的最高频率为 125 Hz。
TARGET_INDEX = 4  # 频率类别依次为 8、9、10.5、11、12、13、14、15 Hz；Python 从 0 编号，所以第五类 12 Hz 的下标是 4。
FFT_SAMPLES = 4 * FS  # 频谱只使用刺激开始后的 4 秒，不含随后 0.5 秒休息；250 点/秒乘 4 秒得到 1000 点。
HARMONICS = [12, 24, 36, 48]  # 要观察的四个频率：12 Hz 基频，以及它的第二、第三、第四次谐波；列表顺序也是表格和头皮图的顺序。

started = perf_counter()  # 记录主体计算开始时刻；计时包含后面的数据读取、处理及绘图，但不包含前面导入各库所花的时间。
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # 创建结果目录；parents=True 会连同上级目录一起创建，exist_ok=True 允许重复运行时使用已存在的目录。
channels = pd.read_excel(DATA_DIR / "Channel Loc.xlsx").sort_values("Channel Index")  # 读取官方通道表，再按一基通道编号排序，保证名称顺序与 MAT 数组的通道轴一致。
channel_names = channels["Channel Name"].tolist()  # 取通道名称列并转为 Python 列表，共 63 个名称；列表下标对应脑电数组的零基通道下标。
oz_index = channel_names.index("Oz")  # 找到枕中线电极 Oz 在名称列表中的位置；本数据表中 Oz 是第 16 个通道，因此这里得到零基下标 15。

# 50 Hz 是电源工频；陷波器主要抑制这一频率附近的成分，避免工频干扰影响信号分析。
# b、a 是数字滤波器传递函数的分子、分母系数数组，并非两个可以随意调节的标量。
notch_b, notch_a = iirnotch(50, Q=30, fs=FS)  # 设计中心频率 50 Hz、品质因数 Q=30 的陷波器；fs=250 指明频率单位为 Hz，单向设计的带宽约为 50/30 Hz。

# 7-70 Hz 带通用于观察波形；这里保留原实现的选择，不把该显示滤波器用于后面的幅度谱和 SNR。
# butter 的 4 是原型阶数参数；转换成带通后传递函数总阶数为 8，sos 表示把它拆成四个二阶滤波节以改善数值稳定性。
display_filter = butter(4, [7, 70], btype="bandpass", fs=FS, output="sos")  # 设计保留 7-70 Hz 的 Butterworth 带通，返回形状为 (4, 6) 的二阶节系数数组。
responses = []  # 依次保存每人经过预处理的 63 通道平均响应；每个元素的形状为 (63, 1125)，没有写入磁盘缓存。
waveforms = []  # 依次保存每人额外做过 7-70 Hz 显示带通的 Oz 波形；每个元素有 1125 点，供跨人平均后绘图。


# ==================== 2. 逐人读取并处理 12 Hz 的平均响应 ====================
# data_CS 的四个轴依次是 [频率类别, 电极通道, 时间采样点, 重复试次]，完整形状为 (8, 63, 22500, 12)。
# 这里先将 12 次重复平均，再做去均值、重采样及固定系数滤波；这些处理是线性的，和逐次处理后再平均等价。
# 每一轮只载入一人的完整 CS 文件，提取所需平均响应后就解除对大数组 raw 的引用。
for subject in SUBJECTS:  # 按 S01、S02、……、S24 的顺序循环，subject 是当前受试者的文件夹名称。
    print(f"[{subject}] 读取原始 CS 数据并计算 12 Hz 平均响应", flush=True)  # 输出当前进度；f 字符串将 subject 替换成实际编号，flush=True 立即刷新输出。
    mat_path = DATA_DIR / subject / f"sub_{subject[1:]}_data_CS.mat"  # 拼出当前 MAT 路径；例如 S01 的 [1:] 去掉首字母 S，生成文件名 sub_01_data_CS.mat。
    raw = loadmat(mat_path, variable_names=["data_CS"])["data_CS"]  # 只要求读取名为 data_CS 的 MAT 变量，再从返回字典取出数组；形状为 (8, 63, 22500, 12)。
    response = raw[TARGET_INDEX].mean(axis=-1)  # 先取第五个频率类别 12 Hz，得到 (63, 22500, 12)，再沿最后的试次轴平均，得到 (63, 22500)。
    del raw  # 删除对完整原始数组的引用，使这块大数组的内存可以被回收；不会修改或删除磁盘中的 MAT 文件。

    # 去均值使用整个 4.5 秒记录的均值，不是刺激前基线校正，因为公开的 CS 记录没有提供刺激前数据。
    # mean(..., keepdims=True) 保留长度为 1 的时间轴，产生 (63, 1) 数组，从而可广播到 (63, 22500)。
    response -= response.mean(axis=-1, keepdims=True)  # 每个通道减去自身时间均值，消除直流偏移

    # 不能简单每隔 20 点取一个点，否则较高频率可能混叠到低频；resample_poly 会结合 FIR 抗混叠滤波进行多相重采样。
    response = resample_poly(response, up=1, down=ORIGINAL_FS // FS, axis=-1)  # // 为整数除法，这里 down=20；沿时间轴按 1/20 重采样，形状从 (63, 22500) 变为 (63, 1125)。
    response = filtfilt(notch_b, notch_a, response, axis=-1)  # 沿时间轴正向、反向各应用一次 50 Hz 陷波器，以消除相位延迟；双向处理也会使单向幅频响应的模被平方。
    responses.append(response)  # 将当前人的预处理结果加入列表；这一份数据还没有做 7-70 Hz 显示带通，后面用它计算总体幅度谱。

    # sosfiltfilt 与 filtfilt 都做双向零相位滤波，区别是前者接收二阶节系数，后者接收分子、分母系数。
    # 双向滤波使用记录两侧的样本，因此这里属于离线分析，不代表实时、只使用过去样本的因果处理。
    waveforms.append(sosfiltfilt(display_filter, response[oz_index]))  # 先取 Oz 的一维 1125 点波形，再应用 7-70 Hz 二阶节双向带通，并把显示波形加入列表。


# ==================== 3. 跨受试者平均并计算单边幅度谱 ====================
# 每人已经平均了自己的 12 次重复，下面再对 24 人等权平均，每人权重均为 1/24。
grand_response = np.mean(responses, axis=0)  # 列表在求均值时相当于 (24, 63, 1125) 数组；沿第 0 轴平均后得到 (63, 1125) 的总体平均响应。
grand_waveform = np.mean(waveforms, axis=0)  # 对相当于 (24, 1125) 的显示波形沿受试者轴平均，得到图 (a) 使用的一维 1125 点 Oz 波形。
times = np.arange(grand_response.shape[-1]) / FS  # shape[-1]=1125；arange 生成 0 到 1124 的样本下标，除以 250 后得到 0、0.004、……、4.496 秒。

# 下面先对波形做跨人平均，再做 FFT（傅里叶） 并取复数模；这与每人先取幅度、再将幅度平均是不同的统计量。
# FFT 使用恰好 1000 点，不补零、不额外加窗；有限长度截取等价于使用矩形窗，频率分辨率为 FS/N=250/1000=0.25 Hz。
frequencies = np.fft.rfftfreq(FFT_SAMPLES, d=1 / FS)  # 生成实数信号单边 FFT 对应的频率轴；N=1000、采样间隔 d=0.004 秒，得到 0-125 Hz 共 501 个频点。
amplitude = np.abs(np.fft.rfft(grand_response[:, :FFT_SAMPLES], axis=-1)) / FFT_SAMPLES  # :1000 取前 4 秒；rfft 得到 (63, 501) 复数谱，abs 取模，再除以 N 消除点数带来的幅度缩放。
amplitude[:, 1:-1] *= 2  # 对除直流和 Nyquist 频点外的 499 个频点乘 2，补回未保留的负频率贡献；得到幅度谱，而非功率谱或功率谱密度。


# ==================== 4. 只按原文式 (3) 计算 SNR ====================
# SNR(f) = (2/Δf + 1) * Amp(f) / sum[Amp(x), x 属于 [f-1, f+1]]。
# Δf=0.25 Hz 时，邻域是 f-1、f-0.75、f-0.5、f-0.25、f、f+0.25、f+0.5、f+0.75、f+1，共 9 点。
# 因此原式等价于中心频点幅度除以这 9 点的平均幅度；中心点也计入分母，结果是幅度比，不是 dB。
# 本实现先形成总体平均响应的幅度谱，再在该幅度谱上应用此式；没有另外计算其他 SNR 定义。
delta_f = frequencies[1] - frequencies[0]  # 由相邻频率作差得到 Δf=0.25 Hz；它取决于分析时长，不能与时间采样间隔 0.004 秒混淆。
half_width = round(1 / delta_f)  # ±1 Hz 的单侧宽度对应 1/0.25=4 个频率间隔；round 将该计算结果转为整数下标宽度。
neighbor_count = 2 * half_width + 1  # 左侧 4 点加右侧 4 点，再加中心 1 点，共 9 点；这个数也就是下面公式中的系数 2/Δf+1。
snr = np.full_like(amplitude, np.nan)  # 创建与幅度谱相同的 (63, 501) 浮点数组，初值为 NaN；频谱两端没有完整 ±1 Hz 邻域的点将保持缺失值。
for index in range(half_width, len(frequencies) - half_width):  # index 依次取 4 到 496，对应 1-124 Hz；range 不包含右端 497，确保每个中心左右都各有 4 点。
    local_sum = amplitude[:, index - half_width:index + half_width + 1].sum(axis=-1)  # 所有通道取 index-4 到 index+4 的 9 点，右端 +1 是因为切片右端不包含；求和后形状为 (63,)。
    snr[:, index] = (2 / delta_f + 1) * amplitude[:, index] / local_sum  # 对 63 个通道同时计算 9*中心幅度/邻域幅度和；这里保留原文括号 (2/Δf+1) 的运算顺序。

harmonic_indices = np.array(HARMONICS) * round(1 / delta_f)  # 将频率转换成频谱下标：12、24、36、48 Hz 分别乘 4，得到下标 48、96、144、192。
oz_amplitude = amplitude[oz_index]  # 从 (63, 501) 幅度谱中取 Oz 通道，得到长度为 501 的一维数组，供图 (b) 及对照表使用。
oz_snr = snr[oz_index]  # 同样提取 Oz 的 501 点 SNR；最前、最后各 4 点为 NaN，对应无法组成完整邻域的频率。
comparison = pd.DataFrame({  # 将四个谐波的复现结果和原文数值整理成 4 行表格；下面每个字典键都会成为一个列名。
    "frequency_hz": HARMONICS,  # 第一列列出四个目标频率 12、24、36、48 Hz，作为每行数值的对应关系。
    "amplitude_reproduced": oz_amplitude[harmonic_indices],  # 用整数下标数组一次取出 Oz 在四个谐波上的幅度；保留原始数据的数值单位，不人为缩放。
    "amplitude_paper": [1.9006, 0.2071, 0.1413, 0.0656],  # 原文第 4.1 节给出的四个幅度参考值，只用于对照，不参与滤波、平均或参数选择。
    "snr_reproduced": oz_snr[harmonic_indices],  # 取出同一组四个谐波下标处、已经按原文式 (3) 算出的 Oz SNR。
    "snr_paper": [6.4196, 3.4815, 3.4932, 3.2040],  # 原文给出的四个 SNR 参考值；与上一列按相同频率顺序逐一比较。
})  # 结束字典及 DataFrame 构造；至此有频率、复现幅度、原文幅度、复现 SNR、原文 SNR 五列。
comparison["amplitude_difference"] = comparison["amplitude_reproduced"] - comparison["amplitude_paper"]  # 新增幅度差列，定义为复现值减原文值；负数表示复现值较小，不是相对误差或百分比。
comparison["snr_difference"] = comparison["snr_reproduced"] - comparison["snr_paper"]  # 同样保存 SNR 的有符号绝对差；这些差异只被报告，不会反过来改变计算参数。


# ==================== 5. 将最终计算结果写成 CSV (可省)====================
# CSV 是最终结果表，不是供程序下次读取的缓存；每次运行都会重新读取所有原始 MAT 并覆盖同名结果。
# index=False 表示不额外写出 pandas 自动生成的行号列；数值列本身保留默认输出精度，结尾的四位小数格式只作用于屏幕显示。
pd.DataFrame({  # 开始整理图 (a) 的时间点及两种 Oz 波形，三列长度均为 1125。
    "time_from_stimulus_onset_s": times,  # 时间以刺激开始为 0 秒，单位为秒；最后一点是 4.496 秒，而不是再额外加入一个 4.500 秒采样点。
    "oz_average": grand_response[oz_index],  # 保存仅做去均值、降采样和陷波的总体平均 Oz 响应，便于与显示带通后的波形对应比较。
    "oz_display_7_70_hz": grand_waveform,  # 保存图 (a) 实际绘制的、经过 7-70 Hz 显示带通后的跨人平均波形。
}).to_csv(OUTPUT_DIR / "waveform_oz.csv", index=False)  # 结束表格构造并写出波形 CSV；文件含 1125 行数据及一行列名。
pd.DataFrame({  # 开始整理图 (b)、图 (d) 共用的 501 个频点，使幅度与 SNR 共用同一条频率轴。
    "frequency_hz": frequencies,  # 保存从 0 到 125 Hz、间隔为 0.25 Hz 的完整频率列表，作图时只显示其中 0-70 Hz。
    "amplitude": oz_amplitude,  # 保存 Oz 的单边幅度谱；它不是幅度平方，因此不应按功率谱或功率单位解释。
    "snr_equation_3": oz_snr,  # 保存原文式 (3) 的 SNR；两端的 NaN 在 CSV 中写成空单元格，表示这些频点没有计算 SNR。
}).to_csv(OUTPUT_DIR / "spectrum_snr_oz.csv", index=False)  # 将频率、幅度、SNR 三列写入同一个 CSV，共 501 行数据。
pd.DataFrame({  # 开始整理所有通道在四个谐波处的结果，共 63*4=252 行，便于查看头皮图背后的实际数值。
    "channel": np.repeat(channel_names, len(HARMONICS)),  # 每个通道名连续重复 4 次，例如 Fp1、Fp1、Fp1、Fp1、F3、F3、F3、F3，与下面逐通道展开的数值对齐。
    "frequency_hz": np.tile(HARMONICS, len(channel_names)),  # 将完整的 [12,24,36,48] 列表重复 63 遍，让每个通道名都依次对应四个频率；tile 和上一行 repeat 的作用不同。
    "amplitude": amplitude[:, harmonic_indices].ravel(),  # 先取出形状 (63, 4) 的谐波幅度，再按默认行优先顺序展开为 252 个数，即先排完一个通道的四个频率再排下一个通道。
    "snr_equation_3": snr[:, harmonic_indices].ravel(),  # 以完全相同的通道、频率顺序展开谐波 SNR，确保每一行的幅度和 SNR 对应同一个电极及频率。
}).to_csv(OUTPUT_DIR / "harmonics_all_channels.csv", index=False)  # 写出全通道谐波表；四列分别是通道、频率、幅度和 SNR。
comparison.to_csv(OUTPUT_DIR / "comparison_with_paper.csv", index=False)  # 将前面得到的四个谐波及两种有符号差值保存为原文对照 CSV，共 4 行、7 列。


# ==================== 6. 为头皮图准备电极坐标 ====================
# 官方 Excel 只有电极名称和编号，不能据此得到每名受试者实测的三维坐标，因此这里使用 MNE 的标准模板。
# 名称顺序始终跟随官方通道编号；这里只统一名称大小写，不更改任何通道的脑电数值或数据排列。
montage = mne.channels.make_standard_montage("colin27_1020")  # 载入 MNE 内置的 Colin27 标准 10-20 电极坐标模板，用于根据名称定位 63 个电极。
standard_names = {name.casefold(): name for name in montage.ch_names}  # 建立小写名称到标准名称的映射，例如 poz -> POz；casefold 使官方表里的 Poz 能与模板的 POz 对应。
plot_names = [standard_names[name.casefold()] for name in channel_names]  # 按官方通道顺序查出每个电极在模板中的规范名称，得到长度为 63 的新列表。
info = mne.create_info(plot_names, sfreq=FS, ch_types="eeg")  # 创建含通道名、250 Hz 采样率和 EEG 通道类型的元数据对象；此对象描述通道，不存放脑电样本本身。
info.set_montage(montage)  # 将模板电极位置按名称写入 info，后面 plot_topomap 据此确定各通道在头皮平面上的位置。


# ==================== 7. 将图 4 的四个部分排在同一张图中 ====================
# 画布共四行：第 1 行波形，第 2 行幅度谱，第 3 行四张谐波头皮图，第 4 行 SNR。
# 公开 CS 只含刺激开始后的 4.5 秒，图 (a) 因而画实际提供的记录，不补造原文图中刺激前的提示阶段。
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})  # 将默认字号设为 10 磅，并隐藏普通坐标轴的上、右边框，使图形更清楚。
figure = plt.figure(figsize=(12, 13), layout="constrained")  # 创建宽 12 英寸、高 13 英寸的画布，启用自动约束排版，为标题、刻度、标签和色条分配空间。
grid = figure.add_gridspec(4, 4, height_ratios=[1.2, 1, 1.5, 1])  # 将画布划成 4 行 4 列；四行高度按 1.2:1:1.5:1 分配，为头皮图留出更高的一行。
waveform_axis = figure.add_subplot(grid[0, :])  # 图 (a) 占第 0 行的全部四列；冒号表示该行所有列，返回供后续绘图使用的坐标轴对象。
spectrum_axis = figure.add_subplot(grid[1, :])  # 图 (b) 占第 1 行的全部四列，横向显示 0-70 Hz 幅度谱。
topography_axes = [figure.add_subplot(grid[2, column]) for column in range(4)]  # 在第 2 行逐列建立四个独立坐标轴，组成按 12、24、36、48 Hz 排列的列表。
snr_axis = figure.add_subplot(grid[3, :])  # 图 (d) 占第 3 行的全部四列，用于显示 SNR 随频率的变化。
figure.suptitle("Section 4.1 | 12 Hz CS signal quality | 24 subjects, 12 trials each", fontsize=15)  # 设置总标题，说明本图使用 24 人、每人 12 个 12 Hz 试次，标题字号为 15 磅。

# ---------- 图 (a)：Oz 平均时间波形和刺激、延迟参考线 ----------
waveform_axis.plot(times, grand_waveform, color="#2074a5", linewidth=0.8)  # 以秒为横轴、显示带通后的平均波形为纵轴，画蓝色折线，线宽为 0.8 磅。
waveform_axis.axvline(0, color="#d94b46", linestyle="--", label="Stimulus onset / offset")  # 在 0 秒画红色竖虚线表示闪烁开始，并给红线设置图例名称。
waveform_axis.axvline(4, color="#d94b46", linestyle="--")  # 在 4 秒画同样的红色虚线表示闪烁结束；不再设置 label，避免图例出现重复条目。
waveform_axis.axvline(0.14, color="#2c9d65", linestyle="--", label="140 ms reference")  # 在开始后 140 ms 画绿色虚线；这是原文提供的视觉延迟参考，不是本脚本从数据估计的延迟。
waveform_axis.axvline(4.14, color="#2c9d65", linestyle="--")  # 在结束后 140 ms 画第二条绿色参考线，对应 4+0.14 秒。
waveform_axis.set(xlim=(-0.02, 4.5), xlabel="Time from stimulus onset (s)",  # 设置横轴范围及秒单位标签；左侧 -0.02 秒仅为显示红线留白，并未产生任何刺激前样本。
                  ylabel="Amplitude (source units)", title="(a) Average Oz waveform, 7-70 Hz display filter")  # 设置纵轴及子图标题，明确幅度保留源数据单位，当前波形做过显示带通。
waveform_axis.legend(loc="lower right", fontsize=9)  # 将两种颜色参考线的图例放到坐标轴右下角，字号为 9 磅。

# ---------- 图 (b)：总体平均响应的 Oz 单边幅度谱 ----------
spectrum_axis.plot(frequencies, oz_amplitude, color="#2074a5", linewidth=0.9)  # 将所有 501 个频点与 Oz 幅度配对画线；下一行通过横轴范围只显示感兴趣的部分。
spectrum_axis.set(xlim=(0, 70), xlabel="Frequency (Hz)", ylabel="Amplitude (source units)",  # 显示 0-70 Hz；横轴为 Hz，纵轴仍为幅度的源数据单位，不是 dB。
                  title="(b) Spectrum of the grand-average Oz response, 0-4 s")  # 标题说明该谱来自 0-4 秒的跨人平均 Oz 响应，区别于逐人幅度谱的平均。

# ---------- 图 (c)：四个谐波在 63 个通道上的幅度分布 ----------
# 四张图分别自动设置色标范围；相同颜色不代表四张图上的幅度相等，比较绝对幅度时必须看各自色条上的数值。
for axis, frequency, index in zip(topography_axes, HARMONICS, harmonic_indices):  # zip 同时取当前坐标轴、目标频率和频谱下标，四轮循环依次画四个谐波。
    plotted, _ = mne.viz.plot_topomap(amplitude[:, index], info, axes=axis, show=False, cmap="viridis")  # 取当前频率的 63 个幅度并按电极坐标插值绘图；plotted 是供色条使用的图像对象，_ 接收无需使用的等高线对象。
    figure.colorbar(plotted, ax=axis, orientation="horizontal", shrink=0.9, pad=0.03)  # 为本幅头皮图添加水平色条；shrink=0.9 缩短色条，pad=0.03 设置色条与图之间的相对间隔。
    axis.set_title(f"{frequency} Hz")  # 将当前图的标题设为对应谐波频率，例如 12 Hz，避免只凭从左到右的位置判断频率。
topography_axes[0].text(0, 1.2, "(c) Harmonic scalp maps", transform=topography_axes[0].transAxes)  # 在第一张头皮图上方标出整个图 (c) 的名称；transAxes 使用轴内相对坐标，(0,1.2) 位于左边界、顶部以上。

# ---------- 图 (d)：与图 (b) 同一幅度谱对应的原文 SNR ----------
snr_axis.plot(frequencies, oz_snr, color="#2074a5", linewidth=0.9)  # 以频率为横轴画 Oz SNR；NaN 点不会连线，因此缺少完整邻域的两端频点自然不显示。
snr_axis.set(xlim=(0, 70), ylim=(0, 9), xlabel="Frequency (Hz)", ylabel="SNR (amplitude ratio)",  # 显示 0-70 Hz、SNR 0-9；因为分母含中心点且幅度非负，该 9 点定义的有效比值不会超过 9。
             title="(d) SNR of the grand-average Oz response, paper equation (3)")  # 标题明确 SNR 来自总体平均响应，使用的是原文式 (3)。
for axis in [spectrum_axis, snr_axis]:  # 分别对幅度谱与 SNR 两个坐标轴添加同一组谐波位置辅助线，不改变波形图和头皮图。
    for frequency in HARMONICS:  # 在每个坐标轴内依次处理 12、24、36、48 Hz 四个频率。
        axis.axvline(frequency, color="gray", linestyle=":", linewidth=0.7, alpha=0.5)  # 在当前频率画灰色点状竖线，线宽为 0.7 磅、透明度为 0.5，方便对照谱峰位置。

# constrained layout 的 rect 按 (左边界, 下边界, 宽度, 高度) 给出，四个数都是相对整张画布的比例，并非 (左,下,右,上)。
figure.get_layout_engine().set(rect=(0, 0.035, 1, 0.925))  # 将主体排版限制在下边界 3.5%、高度 92.5% 的区域，上边界为 96%，为上下文字留出空间。
figure.text(0.5, 0.01, "Scalp maps: standard 10-20 coordinates; amplitudes retain the original data units.",  # 在整张画布的水平中点、底部 1% 高度处添加说明，交代头皮坐标为标准模板且幅度保留原单位。
            ha="center", fontsize=9, color="#555555")  # 将这段底部文字按水平中心对齐，字号 9 磅，使用深灰色以区别于主要坐标标签。
figure.savefig(OUTPUT_DIR / "figure4.png", dpi=200)  # 将整张图保存为 200 dpi 的 PNG；结合 12*13 英寸尺寸，输出图像为 2400*2600 像素，重复运行覆盖同名文件。
plt.close(figure)  # 关闭并释放当前画布的绘图资源；已经写入磁盘的 PNG 不受影响。


# ==================== 8. 在控制台显示数值对照和运行耗时 ====================
elapsed = perf_counter() - started  # 用当前计时器读数减去第 1 节记录的开始读数，得到数据处理、CSV 输出及图像保存的总耗时，单位为秒。
print(comparison.to_string(index=False, float_format="%.4f"), flush=True)  # 将四个谐波的对照表打印为不含行号的文本，浮点数显示四位小数；只影响显示，不修改表格或已保存 CSV 的精度。
print(f"完成：{OUTPUT_DIR}\n用时：{elapsed:.1f} 秒", flush=True)  # 输出结果目录，再用 \n 换行显示保留一位小数的耗时；flush=True 确保最后的完成信息立即可见。
