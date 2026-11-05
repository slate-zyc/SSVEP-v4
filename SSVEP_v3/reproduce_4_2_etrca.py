"""单文件复现第 4.2 节的 eTRCA：24 人、8 类、12 折、1-4 秒。

直接用 SSVEP Conda 环境运行本文件；不导入其他项目脚本，不使用缓存。
每折只用 11 次训练重复构造模板和空间滤波器，留出的第 12 次只用于测试。
"""

# ==================== 方法来源与本文件固定采用的复现口径 ====================
# 数据集论文：https://doi.org/10.1080/27706710.2024.2418650，第 4.2 节规定九电极、四子带、八分类和每人 12 折评估。
# eTRCA 原始论文即目标论文参考 [8]：https://doi.org/10.1109/TBME.2017.2694818，全文 https://pmc.ncbi.nlm.nih.gov/articles/PMC5783827/ 。
# 原始论文式 (9)-(11) 最大化跨试次协方差与总协方差之比；式 (13)-(14) 把八类各自训练的空间滤波器按列拼成 (9,8) 矩阵，用二维相关匹配测试信号和训练平均模板。
# 子带融合遵循原始论文式 (1)：score_n=sum_m((m**(-1.25)+0.25)*r_mn**2)，只使用这一种融合公式。
# 公开仓库的 test_trca.m 使用线性相关系数融合，与原始论文式 (1) 不同；这里以原始论文公式为准，不并行计算其他版本。
# S 的构造与作者 trca.m 等价：https://raw.githubusercontent.com/mnakanishi/TRCA-SSVEP/master/src/trca.m；逐试次去均值后累加所有不同试次的双向交叉乘积。
# Q 则先沿时间拼接 11 个原始训练窗，再按每个通道的拼接均值中心化；它与先对每个训练窗分别中心化再求和的矩阵有区别。
# 滤波器细节沿用前一个 FBCCA 文件：四个通带 [m*8-2,80] Hz，低侧阻带沿用作者 filterbank.m，上侧阻带设为 90 Hz。
# 第 4.2 节没有单列截窗起点；本文件与已完成 FBCCA 统一取 0.14 秒延迟，这是实现选择，不宣称目标论文已明确给出。
# 四秒窗覆盖 0.14-4.14 秒，包含刺激结束后的 0.14 秒；公开 CS 的每次记录有 4.5 秒，足以支持这一截窗。
# ITR 按每人每窗的 96 个测试预测计算，再跨 24 人平均；总决策时长为分析窗+1.5 秒提示+0.14 秒延迟。
# 保留全部 24 人及其原始类别顺序，包括 S11；不删除低表现者，不根据识别结果改写标签，也不使用测试结果调参。

from pathlib import Path  # Path 表示文件路径，支持用 / 拼接目录，并将输出位置固定到本脚本所在的工作树。
from time import perf_counter  # perf_counter 是单调计时器，结束读数减开始读数得到本次读取、计算及绘图耗时。

import matplotlib  # 先加载绘图库本体，以便在 pyplot 创建画布之前选定不弹窗的绘图后端。
matplotlib.use("Agg")  # 使用 Agg 后端直接保存 PNG；脚本计算时不弹出图窗，也不需要手动关闭窗口。
import matplotlib.pyplot as plt  # pyplot 提供画布、坐标轴、误差线、箱线图及保存图片所需接口，plt 是简写。
import numpy as np  # NumPy 负责脑电多维数组、训练协方差、空间投影、Pearson 相关及矩阵乘法。
import pandas as pd  # pandas 负责读取官方通道表，以及整理并输出预测、指标、汇总和原文对照表。
from scipy.io import loadmat  # loadmat 读取 MATLAB 文件并返回以变量名为键的字典；这里仅取 data_CS。
from scipy.linalg import eigh  # eigh 求实对称广义特征值问题 S*w=lambda*Q*w；Q 为训练数据总协方差，最大特征值对应 eTRCA 目标的最大值。
from scipy.signal import cheb1ord, cheby1, filtfilt, iirnotch, resample_poly, sosfiltfilt  # 依次导入 Chebyshev 求阶数、构造滤波器、双向滤波、陷波器设计、多相降采样及二阶节双向滤波。
from scipy.special import xlogy  # xlogy(x,y) 计算 x*ln(y)，并按数学极限将 0*ln(0) 定义为 0，使 ITR 在 100% 准确率时也可直接使用同一公式。


# ==================== 1. 固定参数：只处理当前这份数据 ====================
DATA_DIR = Path(__file__).resolve().parent.parent / "EEG_data"  # 四个版本共用根目录数据；路径不依赖当前工作目录。
OUTPUT_DIR = Path(__file__).resolve().parent / "results_full" / "section4_2_etrca"  # __file__ 为本文件，parent 为其所在目录；所有本方法结果写到当前工作树下的独立结果目录。
SUBJECTS = [f"S{number:02d}" for number in range(1, 25)]  # 生成 S01 到 S24；range 的右端 25 不包含在内，02d 表示把整数补零为两位。
FREQUENCIES = np.array([8, 9, 10.5, 11, 12, 13, 14, 15])  # 八类刺激频率，顺序严格对应原始数组的类别轴；数据含 10.5 Hz，不能写成 10 Hz。
CHANNELS = ["Pz", "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2"]  # 按目标论文选取九个顶枕区电极；不是第 4.1 节使用的全部 63 通道。
WINDOWS = [1, 2, 3, 4]  # 四种独立分析窗长，单位秒；都从同一时刻开始，不是将试次先后切成四段。
ORIGINAL_FS = 5000  # 原始采样率为每秒 5000 点，每个 CS 试次 4.5 秒，共 22500 个采样点。
FS = 250  # 分析采样率为 250 Hz，降采样因子为 20；四种窗长分别包含 250、500、750、1000 点。
DELAY_SECONDS = 0.14  # 截窗时跳过刺激开始后的 140 ms，同时将该延迟计入 ITR 决策总时长；这是上方已说明的复现选择。
CUE_SECONDS = 1.5  # 数据集试次在闪烁前有 1.5 秒提示时间；它只进入决策时长，不从已按刺激开始对齐的 CS 中再次扣除。
CLASS_COUNT = 8  # 八分类，只识别 CS 的八种刺激频率，不加入 NS，不训练第 4.3 节的状态分类器。
TRIAL_COUNT = 12  # 每名受试者每种频率有 12 次重复；第 fold 折测试各频率相同重复编号的一次，共 8 个测试试次。
BAND_COUNT = 4  # 原文规定四个子带；子带编号在公式中从 1 开始，在 Python 数组中从 0 开始。
PASSBANDS = [[6, 80], [14, 80], [22, 80], [30, 80]]  # 通带为 [m*8-2,80] Hz，m=1、2、3、4；四带相互重叠，逐渐去除低频部分。
STOPBANDS = [[4, 90], [10, 90], [16, 90], [24, 90]]  # 求滤波器阶数所用的阻带；低侧沿用公开代码，高侧随 80 Hz 通带设置为 90 Hz，属于补充设计细节。
weights = np.arange(1, BAND_COUNT + 1, dtype=float) ** (-1.25) + 0.25  # 按原始 eTRCA 的 w(m)=m^(-1.25)+0.25 生成四个正权重，第一子带权重最大。

started = perf_counter()  # 记录主体计算开始时刻；后面的耗时包括读取原始 MAT、滤波、eTRCA、CSV 输出和绘图。
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # 创建输出目录及缺失的父目录；已有目录可以直接使用，重新运行会重新计算并覆盖同名结果。
channel_table = pd.read_excel(DATA_DIR / "Channel Loc.xlsx")  # 读取官方电极名称与原始通道编号的对应关系，读取 Excel 所需 openpyxl 已在现有环境中安装。
channel_lookup = dict(zip(channel_table["Channel Name"].str.casefold(), channel_table["Channel Index"] - 1))  # 建立小写电极名到零基下标的映射；官方表从 1 编号，减 1 后才能索引 NumPy。
channel_indices = [channel_lookup[name.casefold()] for name in CHANNELS]  # 按指定的九电极顺序查询原始数据下标；casefold 使 POz 能匹配表中的 Poz。
notch_b, notch_a = iirnotch(50, Q=30, fs=FS)  # 设计 50 Hz 陷波器，Q=30；b、a 分别是传递函数的分子和分母系数数组，不是两个标量参数。
filter_sections = []  # 保存四个子带各自的二阶节系数；这些系数由固定参数确定，不通过任何测试标签来选择。
for passband, stopband in zip(PASSBANDS, STOPBANDS):  # zip 将每个子带的通带和阻带配对，四轮循环分别设计四个滤波器。
    order, critical = cheb1ord(passband, stopband, gpass=3, gstop=40, fs=FS)  # 用 3 dB 通带、40 dB 阻带规格求最低原型阶数及临界频率；这组求阶数参数参考公开 MATLAB 实现。
    sections = cheby1(order, rp=0.5, Wn=critical, btype="bandpass", fs=FS, output="sos")  # 按公开实现用 0.5 dB 纹波构造 Chebyshev I 带通；rp 与上行求阶数的 gpass 是两个不同调用中的参数。
    filter_sections.append(sections)  # 将当前子带滤波器加入列表；sos 是二阶节形式，比直接使用高阶分子分母系数更适合数值计算。

prediction_rows = []  # 累积逐试次测试结果和八个候选频率得分；最终有 24*4*12*8=9216 行，不是训练集上的预测。
metric_rows = []  # 累积每人每窗的准确率和 ITR，最终有 24*4=96 行，每一行均基于该人的 96 个测试试次。
confusion_matrices = np.zeros((len(WINDOWS), CLASS_COUNT, CLASS_COUNT), dtype=int)  # 为四种窗长分别创建 8*8 混淆矩阵；行是真实类别，列是预测类别，初值为 0。


# ==================== 2. 每人独立读取、去均值、抗混叠降采样及陷波 ====================
# 本任务必须保留各个试次供测试，不能像第 4.1 节那样先平均同一频率的 12 次重复。
# 原始 data_CS 形状为 [8 类,63 通道,22500 点,12 次重复]；选电极、转轴后统一为 [类别,重复,通道,时间]。
for subject in SUBJECTS:  # 按固定编号依次处理全部 24 人，每人只读取一次原始 CS 文件，然后依次完成四种窗长。
    print(f"[{subject}] 读取原始 CS，计算 eTRCA 的 1-4 秒识别", flush=True)  # 输出当前受试者进度；flush=True 立即显示，避免长时间读取数据时信息被缓冲。
    mat_path = DATA_DIR / subject / f"sub_{subject[1:]}_data_CS.mat"  # 例如 subject=S01 时得到 EEG_data/S01/sub_01_data_CS.mat，subject[1:] 用于去掉字母 S。
    raw = loadmat(mat_path, variable_names=["data_CS"])["data_CS"]  # 从 MAT 字典取出完整 data_CS 数组，形状为 (8,63,22500,12)，不读取任何预处理文件。
    data = raw[:, channel_indices, :, :].transpose(0, 3, 1, 2).copy()  # 取九通道得到 (8,9,22500,12)，再转成 (8,12,9,22500)，copy 生成便于后续沿时间处理的独立数组。
    del raw  # 解除对完整 63 通道大数组的引用，使其内存可以被回收；不删除或修改磁盘中的原始 MAT 文件。
    data -= data.mean(axis=-1, keepdims=True)  # 每个类别、每次重复、每个通道分别减去自身 4.5 秒均值；保留长度为 1 的时间轴，以广播到全部采样点。
    data = resample_poly(data, up=1, down=ORIGINAL_FS // FS, axis=-1)  # 使用多相 FIR 抗混叠滤波将时间轴按 1/20 重采样，形状变成 (8,12,9,1125)，不是简单隔点抽样。
    data = filtfilt(notch_b, notch_a, data, axis=-1)  # 对每个试次的时间轴做正向、反向 50 Hz 陷波，消除相位延迟；这是使用整个试次的离线预处理。

    # ==================== 3. 对 1、2、3、4 秒分别截窗并构造滤波器组 ====================
    for window_index, seconds in enumerate(WINDOWS):  # enumerate 同时给出窗长位置 0-3 和实际时长 1-4 秒，前者用于索引四张混淆矩阵。
        samples = round(seconds * FS)  # 将秒换成采样点数，例如 1 秒为 250 点、4 秒为 1000 点；round 得到可以用于切片的整数。
        start = round(DELAY_SECONDS * FS)  # 0.14*250=35，因此从降采样后下标 35 处开始取 EEG 分析窗。
        cropped = data[..., start:start + samples]  # 省略号保留类别、重复、通道三轴，只截时间轴；一秒窗为 35:285，四秒窗为 35:1035。
        bank = np.stack([sosfiltfilt(sections, cropped, axis=-1) for sections in filter_sections])  # 对同一窗口分别做四个子带的零相位双向滤波并新增子带轴，形状为 (4,8,12,9,samples)。
        # ==================== 4. 每折仅用其余 11 次重复训练模板和空间滤波器 ====================
        # 记逐试次中心化的训练窗为 Z_r，S=sum_{r!=s}(Z_r@Z_s.T)。
        # 利用恒等式 S=(sum_r Z_r)@(sum_r Z_r).T-sum_r(Z_r@Z_r.T)，避免重复计算 110 个交叉乘积；这与原始定义完全等价。
        # Q 采用全部训练窗按时间拼接、再整体中心化后的乘积；省略协方差公共比例因子不会改变空间滤波方向。
        # 训练仅使用当前受试者、当前窗长、当前子带的 11 次重复；测试试次不进入模板、S、Q 或空间滤波器。
        window_predictions = []  # 按“先折号、后真实类别”保存 96 个预测，每折八个测试类别，后面用于准确率与混淆矩阵。
        for fold in range(TRIAL_COUNT):  # fold 为留出的重复下标 0-11；每次将八类的相同重复编号作为该折测试集。
            training_indices = np.arange(TRIAL_COUNT) != fold  # 生成 12 个布尔索引，仅当前测试重复为 False，其余 11 个为 True；这不是防错分支。
            templates = bank[:, :, training_indices].mean(axis=2)  # 仅平均训练重复轴，得到 (4,8,9,samples) 的个体模板，不对类别或通道求平均。
            spatial_filters = np.empty((BAND_COUNT, CLASS_COUNT, len(CHANNELS)))  # 保存每个子带、每个类别各一个九维 TRCA 空间滤波器，共 4*8 个。
            for band_index in range(BAND_COUNT):  # 四个子带分别训练，不能将子带信号先混合后再训练一个公共滤波器。
                for candidate_index in range(CLASS_COUNT):  # 八个频率各自使用本类的 11 次训练重复，分别求能最大化本类重复一致性的滤波器。
                    training = bank[band_index, candidate_index, training_indices]  # 当前训练数据形状为 (11,9,samples)，没有选取当前 fold 的测试信号。
                    centered_trials = training - training.mean(axis=-1, keepdims=True)  # 每次重复、每个通道分别减去当前分析窗的时间均值，满足跨试次协方差的定义。
                    summed_trials = centered_trials.sum(axis=0)  # 沿训练重复轴求和，得到 (9,samples)，供 S 的所有试次交叉乘积之和使用。
                    centered_concatenated = centered_trials.transpose(1, 0, 2).reshape(len(CHANNELS), -1)  # 将已分别中心化的训练窗按时间拼成 (9,11*samples)，用于扣除同一试次的自身乘积。
                    inter_trial = summed_trials @ summed_trials.T - centered_concatenated @ centered_concatenated.T  # 计算 S；前一项包含全部 r,s 组合，后一项恰好扣除 r=s，只留下不同试次的双向协方差。
                    concatenated = training.transpose(1, 0, 2).reshape(len(CHANNELS), -1)  # 按通道拼接未在本窗口逐试次中心化的训练信号，形状为 (9,11*samples)，对应原作者的 UX。
                    concatenated = concatenated - concatenated.mean(axis=-1, keepdims=True)  # 对每个通道在拼接后的全部 11*samples 点上减一次总均值；保留各训练窗均值之间的真实差异。
                    total_covariance = concatenated @ concatenated.T  # 计算 Q；仅含训练信号，不加入正则项或测试数据，比例常数不影响最优特征向量方向。
                    _, eigenvectors = eigh(inter_trial, total_covariance, subset_by_index=[len(CHANNELS) - 1, len(CHANNELS) - 1])  # eigh 按特征值升序返回；只取下标 8 对应的最大特征值及九维特征向量，不求逆矩阵。
                    spatial_filters[band_index, candidate_index] = eigenvectors[:, 0] / np.linalg.norm(eigenvectors[:, 0])  # 把 Q 归一化的特征向量转为欧氏单位长度，与作者 eigs 返回的滤波器尺度一致；eTRCA 同时使用八列，必须消除 eigh 默认 Q 归一化造成的任意列尺度差异。

            # ==================== 5. 用训练好的 eTRCA 识别本折八个测试试次 ====================
            for target_index, true_frequency in enumerate(FREQUENCIES):  # 只在训练完成后依次取八个真实类别的留出试次，真实类别仅用于取测试信号和最后评价。
                test = bank[:, target_index, fold]  # 当前留出试次的四子带数据，形状为 (4,9,samples)，不用于重新拟合任何空间滤波器。
                correlations = np.empty((BAND_COUNT, CLASS_COUNT))  # 存放四子带对八个候选模板的 Pearson 相关系数 r，允许负相关。
                for band_index in range(BAND_COUNT):  # 四个子带分别做模板匹配，再按原始论文式 (1) 融合为八个候选总得分。
                    ensemble_filter = spatial_filters[band_index].T  # 将本子带八个类别的单位滤波器按列排成 (9,8) 的 W；同一个 W 用于全部八个候选模板。
                    projected_test = (ensemble_filter.T @ test[band_index]).ravel()  # 先得到 (8,samples) 的八条投影，再按一致顺序展开为一条向量；等价于对原文 (samples,8) 矩阵计算二维相关。
                    projected_test -= projected_test.mean()  # 二维 Pearson 相关使用整块投影矩阵所有元素的总均值，不是先计算八个一维相关再平均。
                    for candidate_index in range(CLASS_COUNT):  # 逐个将当前测试试次与八类训练模板匹配；每次均使用整个 W，不仅使用该候选自己的那一列。
                        projected_template = (ensemble_filter.T @ templates[band_index, candidate_index]).ravel()  # 用同一组八个空间滤波器投影当前候选模板，并按与测试侧相同的次序展开。
                        projected_template -= projected_template.mean()  # 对候选模板的整块八投影数组中心化，保证与原始论文式 (14) 的二维相关定义一致。
                        correlations[band_index, candidate_index] = np.dot(projected_test, projected_template) / (np.linalg.norm(projected_test) * np.linalg.norm(projected_template))  # 整块投影的中心化内积除以两侧范数，等价于 MATLAB corrcoef(A,B) 对 A(:)、B(:) 的相关系数。
                scores = weights @ (correlations ** 2)  # 原始 TRCA 论文式 (1) 的四子带相关系数平方加权；得到八个候选类别的总得分，不同时实现线性融合版本。
                predicted_index = int(np.argmax(scores))  # 选择总得分最大的候选类别；argmax 返回 0-7 的下标，这也是后面预测频率在 FREQUENCIES 中的位置。
                window_predictions.append(predicted_index)  # 将当前测试试次的预测类别追加到当前窗的列表，循环结束后长度恰为 12*8=96。
                prediction_rows.append({  # 保存当前测试试次的完整识别记录，便于不重新计算 eTRCA 也能独立核对准确率和混淆矩阵。
                    "subject": subject,  # 记录当前受试者编号，避免跨人的试次在汇总时混淆。
                    "window_seconds": seconds,  # 记录分析时长 1-4 秒，四种时长各自对应一次独立的频率识别。
                    "fold": fold + 1,  # 表格中的折号从 1 到 12，方便对应第几次重复；内部数组下标仍从 0 开始。
                    "true_class": target_index,  # 保存原始真实类别下标 0-7，不用 eTRCA 预测结果改写真实标签。
                    "predicted_class": predicted_index,  # 保存预测类别下标，后续直接与 true_class 比较判断识别是否正确。
                    "true_frequency_hz": true_frequency,  # 将真实类别同时写成 Hz，便于识别 10.5 Hz 等具体频率的错误模式。
                    "predicted_frequency_hz": FREQUENCIES[predicted_index],  # 根据最大得分下标查出预测刺激频率，单位 Hz。
                    **{f"score_{frequency:g}_hz": score for frequency, score in zip(FREQUENCIES, scores)},  # 字典推导式生成八个候选频率得分列，再用 ** 展开到本行记录；g 格式去掉无意义的 .0。
                })  # 结束当前试次记录；这里只保存本文件 eTRCA 的测试预测，不混入其他方法或 NS 分类结果。

        # ==================== 6. 每人每窗的准确率、ITR 与混淆矩阵 ====================
        predicted = np.array(window_predictions).reshape(TRIAL_COUNT, CLASS_COUNT)  # 将 96 个预测按测试顺序排成 (12,8)，每行是一折，每列对应一个真实类别。
        accuracy = np.mean(predicted == np.arange(CLASS_COUNT)[None, :])  # 将每折八个预测与 0-7 的真实类别广播比较，96 个布尔值的均值就是整体正确比例，取值为 0-1。
        decision_seconds = seconds + CUE_SECONDS + DELAY_SECONDS  # 一次选择的总时间是分析窗加提示加视觉延迟；一秒窗使用 1+1.5+0.14=2.64 秒。
        bits_per_selection = (np.log(CLASS_COUNT) + xlogy(accuracy, accuracy) + xlogy(1 - accuracy, (1 - accuracy) / (CLASS_COUNT - 1))) / np.log(2)  # 用自然对数表达原文 ITR 式 (2) 的括号项，再除 ln(2) 转成以 2 为底的比特数；xlogy 正确处理 0*ln(0)。
        itr = float((accuracy > 1 / CLASS_COUNT) * bits_per_selection * 60 / decision_seconds)  # 乘每分钟的选择次数 60/T；按公开 ITR 实现约定，机会水平 1/8 及以下计为 0，这个布尔乘法不添加 if 分支。
        np.add.at(confusion_matrices[window_index], (np.tile(np.arange(CLASS_COUNT), TRIAL_COUNT), predicted.ravel()), 1)  # 将当前人的 96 次“真实类别、预测类别”配对累加到对应窗长矩阵；add.at 会正确累加重复索引。
        metric_rows.append({  # 保存当前人当前窗的一行指标，所有受试者计算结束后共 96 行。
            "subject": subject,  # 标明这行指标属于哪名受试者，便于绘制跨人的箱线图。
            "window_seconds": seconds,  # 标明这行指标的分析时长，避免把不同窗长的结果平均到一起。
            "accuracy": accuracy,  # 保存 0-1 比例而非百分数；绘图及与原文百分数比较时才乘 100。
            "itr_bits_min": itr,  # 保存单位为 bits/min 的 ITR，是先按当前人的准确率计算所得，并非把所有人的平均准确率直接代入公式。
            "n_trials": predicted.size,  # 当前指标实际包含的测试试次数，固定为 96；没有将训练试次重复计入评价。
            "decision_seconds": decision_seconds,  # 保存 ITR 实际使用的总时间，便于独立复算并区分分析窗时长与完整选择时长。
        })  # 结束本行指标字典，并将其加入全体受试者的指标列表。
        print(f"  {seconds} s: accuracy={accuracy:.2%}, ITR={itr:.4f} bits/min", flush=True)  # 立即输出当前人的当前窗结果；.2% 将比例显示为百分数，CSV 中的原始数值不受显示格式影响。


# ==================== 7. 整理并保存全体结果 ====================
predictions = pd.DataFrame(prediction_rows)  # 把 9216 条字典记录转为预测表，每行对应某个人、某个窗、某折的一个测试试次。
metrics = pd.DataFrame(metric_rows)  # 把 96 条每人每窗的指标记录转成表格，为跨人等权平均和箱线图提供数据。
summary = metrics.groupby("window_seconds", as_index=False).agg(  # 按窗长分成四组，不合并不同长度；as_index=False 将窗长保留为普通列。
    accuracy_mean=("accuracy", "mean"),  # 对当前窗长的 24 个受试者准确率等权求平均；每人权重都是 1/24。
    accuracy_sd=("accuracy", "std"),  # pandas 的 std 默认采用 ddof=1，给出 24 人准确率的样本标准差，反映个体差异。
    itr_mean=("itr_bits_min", "mean"),  # 先得到每人 ITR 再取 24 人平均；ITR 对准确率非线性，不能用平均准确率代替这一过程。
    itr_sd=("itr_bits_min", "std"),  # 保存跨人的 ITR 样本标准差，与后面标准误的含义不同。
)  # 结束具名聚合，summary 有四行，每行对应一种分析时长。
summary["accuracy_sem"] = summary["accuracy_sd"] / np.sqrt(len(SUBJECTS))  # 用样本标准差除以 sqrt(24) 得到跨人均值的标准误 SEM，供误差线使用；不是 95% 置信区间。
summary["itr_sem"] = summary["itr_sd"] / np.sqrt(len(SUBJECTS))  # 同样计算平均 ITR 的标准误，不把同一人的不同试次当作 24 人之外的独立受试者。
comparison = pd.DataFrame({  # 目标论文正文只给出两个 eTRCA 明确数值，因此这里只比较这两个，不从图片中猜测其他窗长的均值。
    "metric": ["accuracy_4s_percent", "itr_1s_bits_min"],  # 第一行单位是四秒准确率百分数，第二行单位是一秒 ITR 的 bits/min，二者不能直接互相比较大小。
    "reproduced": [summary.loc[summary.window_seconds == 4, "accuracy_mean"].iloc[0] * 100, summary.loc[summary.window_seconds == 1, "itr_mean"].iloc[0]],  # 按指定窗长取复现跨人均值，准确率乘 100 转成与原文一致的百分数。
    "paper": [79.47, 33.3243],  # 第 4.2 节正文报告的 eTRCA 四秒平均准确率和一秒平均 ITR，只用于最终对照，不用于选取算法参数。
})  # 结束两行原文对照表的构造。
comparison["difference"] = comparison["reproduced"] - comparison["paper"]  # 保存有符号差值；准确率差的单位是百分点，ITR 差的单位是 bits/min。
confusion_table = pd.DataFrame({  # 将四个 8*8 混淆矩阵展开为 4*8*8=256 行的长表，便于按真实频率和预测频率查看识别错误。
    "window_seconds": np.repeat(WINDOWS, CLASS_COUNT * CLASS_COUNT),  # 每个窗长连续重复 64 次，使四个矩阵的行依次对应 1、2、3、4 秒。
    "true_frequency_hz": np.tile(np.repeat(FREQUENCIES, CLASS_COUNT), len(WINDOWS)),  # 每个真实频率先重复八次以对应八个预测列，再将该顺序重复四个窗长。
    "predicted_frequency_hz": np.tile(FREQUENCIES, len(WINDOWS) * CLASS_COUNT),  # 每个真实频率对应一次完整八频率列表，顺序与混淆矩阵按行展开一致。
    "count": confusion_matrices.ravel(),  # 按“窗长、真实类别、预测类别”的顺序展开计数；每个窗长计数之和为 24*96=2304。
})  # 结束混淆矩阵长表构造，保持行是真实类别、列是预测类别这一统计口径。
predictions.to_csv(OUTPUT_DIR / "predictions.csv", index=False)  # 保存每次测试的真实标签、预测标签及八频率得分；index=False 不额外写入 pandas 自动行号。
metrics.to_csv(OUTPUT_DIR / "subject_metrics.csv", index=False)  # 保存 24 人各四种窗长的准确率、ITR、试次数和决策时长，共 96 行。
summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)  # 保存四种窗长的跨人均值、标准差和标准误，不把这些最终表格作为下次运行的输入。
comparison.to_csv(OUTPUT_DIR / "comparison_with_paper.csv", index=False)  # 保存与正文两个已知数值的差异，不因为差异存在就回头用测试集调整参数。
confusion_table.to_csv(OUTPUT_DIR / "confusion_matrices.csv", index=False)  # 保存所有窗长、所有频率对的混淆计数；再次运行会从原始 MAT 重新生成并覆盖文件。


# ==================== 8. 绘制 eTRCA 的准确率、ITR 均值曲线和个体分布 ====================
accuracy_by_subject = metrics.pivot(index="subject", columns="window_seconds", values="accuracy") * 100  # 将长表变成 24 行受试者、4 列窗长的百分数表，每一列是一个箱线图的数据。
itr_by_subject = metrics.pivot(index="subject", columns="window_seconds", values="itr_bits_min")  # 同样整理 ITR 为 24*4 表格，保留 bits/min 单位，不进行百分数转换。
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})  # 默认字体设为 10 磅，隐藏普通坐标轴的上、右边框，使四个子图更易读。
figure, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")  # 创建 12*9 英寸的四子图画布，自动为总标题、轴标签和图例分配空间。
figure.suptitle("Section 4.2 | eTRCA | 24 subjects | 12-fold evaluation", fontsize=15)  # 总标题明确只有 eTRCA；这张图仅展示当前方法的测试结果及对应的原文参考点。
axes[0, 0].errorbar(WINDOWS, summary.accuracy_mean * 100, yerr=summary.accuracy_sem * 100, fmt="o-", capsize=4, color="#2074a5", label="eTRCA mean +/- SEM")  # 左上画四种窗长的平均准确率及标准误，比例和误差都乘 100 转成百分数。
axes[0, 0].scatter([4], [79.47], marker="D", facecolors="none", edgecolors="#d94b46", s=65, label="Paper: 79.47% at 4 s")  # 在四秒位置标出正文给出的参考准确率，用空心菱形与复现均值区分。
axes[0, 0].set(xlabel="Data length (s)", ylabel="Accuracy (%)", xticks=WINDOWS, ylim=(0, 105), title="(a) Mean accuracy across subjects")  # 设置横轴四个窗长、百分数纵轴及子图名称；略高于 100 的上界为误差线和标记留白。
axes[0, 0].legend(fontsize=9, loc="lower right")  # 在左上子图右下方显示复现均值与原文数值的图例，字号为 9 磅。
axes[0, 1].boxplot([accuracy_by_subject[seconds].to_numpy() for seconds in WINDOWS], tick_labels=WINDOWS, showmeans=True)  # 右上分别画每种窗长的 24 人准确率分布；箱中横线为中位数，额外标记为均值。
axes[0, 1].set(xlabel="Data length (s)", ylabel="Accuracy (%)", ylim=(0, 105), title="(b) Accuracy distribution: 24 subjects")  # 给箱线图设置秒、百分数标签及人数说明；低表现者仍保留在图中。
axes[1, 0].errorbar(WINDOWS, summary.itr_mean, yerr=summary.itr_sem, fmt="o-", capsize=4, color="#2074a5", label="eTRCA mean +/- SEM")  # 左下画平均 ITR 和跨人标准误，单位为 bits/min，不能用每秒窗长简单替代完整决策时间。
axes[1, 0].scatter([1], [33.3243], marker="D", facecolors="none", edgecolors="#d94b46", s=65, label="Paper: 33.3243 at 1 s")  # 在一秒处标出正文给出的 ITR 参考均值，不绘制正文未给出的其他参考数值。
axes[1, 0].set(xlabel="Data length (s)", ylabel="ITR (bits/min)", xticks=WINDOWS, ylim=(0, None), title="(c) Mean ITR across subjects")  # 将横轴设为窗长、纵轴设为 ITR；下界取 0，上界由实际结果自动确定。
axes[1, 0].legend(fontsize=9)  # 显示 ITR 复现曲线与原文参考点的含义，图例位置由 Matplotlib 根据绘图区选择。
axes[1, 1].boxplot([itr_by_subject[seconds].to_numpy() for seconds in WINDOWS], tick_labels=WINDOWS, showmeans=True)  # 右下画四个窗长的 24 人 ITR 箱线图，显示个体差异，而不是把四个均值再次做箱线图。
axes[1, 1].set(xlabel="Data length (s)", ylabel="ITR (bits/min)", ylim=(0, None), title="(d) ITR distribution: 24 subjects")  # 设置 ITR 分布图的标签和非负纵轴范围，保留低于机会水平而计为 0 的受试者结果。
figure.savefig(OUTPUT_DIR / "etrca_accuracy_itr.png", dpi=200)  # 以 200 dpi 保存四子图，结合画布尺寸输出 2400*1800 像素 PNG，重复运行覆盖同名图片。
plt.close(figure)  # 关闭画布并释放绘图资源；已经保存的 PNG 不会被删除。


# ==================== 9. 打印最终数值与耗时 ====================
elapsed = perf_counter() - started  # 计算本次从读取通道表、设计滤波器到处理 24 人并输出所有图表的总耗时，单位秒。
print(summary.to_string(index=False, float_format="%.4f"), flush=True)  # 打印四个窗长的跨人汇总，浮点数显示四位小数，显示格式不影响 CSV 中保存的精度。
print(comparison.to_string(index=False, float_format="%.4f"), flush=True)  # 打印与正文两个参考数值的差异，方便直接看到复现结果而非只确认程序运行结束。
print(f"完成：{OUTPUT_DIR}\n用时：{elapsed:.1f} 秒", flush=True)  # 输出最终结果目录和一位小数的秒数；\n 是换行符，flush=True 使结束信息立即显示。
