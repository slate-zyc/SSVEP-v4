"""第 4.3 节：同一文件完成 FS-eTRCA 和 FS-eTRCA&FBCCA 的九分类评估。

两种方法共享原始数据、每折 eTRCA 模型及候选频率，分别训练各自的八个 LDA。
逐行中文注释、顺序执行，不导入其他项目脚本，不使用磁盘特征或模型缓存。
"""

# ==================== 方法来源、外层评估与补充实现选择 ====================
# 目标论文：https://doi.org/10.1080/27706710.2024.2418650，第 4.3 节：CS 取 0.14-1.14 秒，NS 切不重叠一秒段，分别与 NS1/NS2/NS3 进行 12 折九分类。
# FS 原始论文：https://doi.org/10.1109/TNSRE.2023.3246359，先用 eTRCA 选候选频率，再用对应频率的二分类 LDA 决定接受该频率或拒绝为 NS。
# eTRCA 原始论文：https://doi.org/10.1109/TBME.2017.2694818，八类滤波器按列拼成空间投影矩阵，以二维相关计算每个子带的匹配特征。
# FBCCA 原始论文：https://doi.org/10.1088/1741-2560/12/4/046008，每个频率取五阶正弦/余弦参考，第一典型相关作为每个子带的附加特征。
# 四个子带沿用已完成的 4.2：通带 [m*8-2,80] Hz，m=1、2、3、4；阻带和滤波设计亦与已有文件相同。
# 候选频率按原始方法的相关系数平方加权选取；两个 FS 版本共用候选频率，融合版仅在 LDA 特征中增加对应频率的四个 FBCCA 相关系数。
# FS-eTRCA 的 LDA 输入为本候选的四维有符号 eTRCA 相关；融合版为四维 eTRCA 加四维 FBCCA，共八维，不把八类候选的特征混进同一个分类器。
# 外层每折用每类其余 11 次 CS 训练模板和滤波器；LDA 的 11 个本类 CS 正样本采用内部留一模型提特征，每个内部模型只用其余 10 次 CS。
# NS 原始记录分组及 LDA 先验、内部留一属于目标论文未详细说明的实现选择；这里使用全部 88 个训练 NS、11 个本频率 CS，LDA 两类先验均为 0.5。
# 每种 NS 各有 96 个片段，来自 NS1/NS2 的 24 条四秒记录或 NS3 的 48 条两秒记录；同一原始记录的所有片段保持同一外层折。
# 主准确率为全部 192 段的正确比例，其中 NS 占一半；同时输出九类召回率等权均值，避免把样本准确率与九类均衡准确率混为一谈。
# 输出 0-7 为八种频率，8 为 NS；所有受试者保留原始标签，包括 S11，不以接近原文数值为目标调整测试参数。

from pathlib import Path  # Path 定位官方原始数据及当前脚本所在的工作树，运行时不依赖另一个脚本。
from time import perf_counter  # 单调时钟用于记录原始数据读取、特征计算、交叉验证和绘图的总耗时。
import matplotlib  # 先导入绘图库本体，以便在 pyplot 加载之前设置不弹窗的绘图后端。
matplotlib.use("Agg")  # Agg 直接保存图片，使长时间运行不需要图形界面交互。
import matplotlib.pyplot as plt  # pyplot 用于跨人均值误差线、原文参考点以及个体分布图。
import numpy as np  # NumPy 提供脑电多维数组、协方差、空间投影、QR/SVD 和分类指标计算。
import pandas as pd  # pandas 读取电极表，并整理保存预测、每人指标、汇总和原文对照。
from scipy.io import loadmat  # 仅从官方 MAT 中取所需状态变量，不读取此前生成的结果表或缓存。
from scipy.linalg import eigh  # 求解实对称广义特征值问题 S*w=lambda*Q*w，最大特征值方向用于训练 TRCA 滤波器。
from scipy.signal import cheb1ord, cheby1, filtfilt, iirnotch, resample_poly, sosfiltfilt  # 分别用于子带求阶数、构造滤波器、双向陷波、陷波器设计、抗混叠降采样及二阶节双向滤波。
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # 每种方法每个频率分别训练普通线性判别分析，以判断候选是 CS 还是 NS。
from threadpoolctl import threadpool_limits  # 限制 BLAS 线程数，避免大量九维特征值问题与小型矩阵运算产生过多线程开销。

DATA_DIR = Path(__file__).resolve().parent.parent / "EEG_data"  # 四个版本共用根目录数据；路径不依赖当前工作目录。
OUTPUT_DIR = Path(__file__).resolve().parent / "results_full" / "section4_3_fs"  # 两个 FS 版本的最终预测、统计表和图表保存在同一个结果目录中，以 method 列区分。
SUBJECTS = [f"S{number:02d}" for number in range(1, 25)]  # 顺序处理全部 S01-S24，不删受试者或重新编号。
STATES = ["NS1", "NS2", "NS3"]  # 每种 NS 分别构成一个独立的 CS+NS 训练测试任务。
METHODS = ["FS-eTRCA", "FS-eTRCA&FBCCA"]  # 固定两个版本的顺序，前者四维特征，后者八维特征。
FREQUENCIES = np.array([8, 9, 10.5, 11, 12, 13, 14, 15])  # 类别 0-7 对应的刺激频率，顺序与原始 CS 数组一致。
CHANNELS = ["Pz", "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2"]  # 使用目标论文规定的九个后部通道，IFB-OCN 的 Fp1 不加入本文件。
ORIGINAL_FS = 5000  # 官方原始采样率为 5000 Hz，每个 CS 记录有 22500 点。
FS = 250  # 分析采样率 250 Hz；固定一秒窗口，因此每个分析片段为 250 点。
FOLDS = 12  # 外层十二折，当前折留出每频率相同编号的一次重复，余下十一重复用于训练。
START = round(0.14 * FS)  # CS 的起始点为 35，时间区间为 0.14-1.14 秒，与 4.3 明确规定一致。
PASSBANDS = [[6, 80], [14, 80], [22, 80], [30, 80]]  # 目标论文规定的四子带通带，逐渐去掉低频部分以利用谐波信息。
STOPBANDS = [[4, 90], [10, 90], [16, 90], [24, 90]]  # 沿用 4.2 的滤波器补充设计细节：下阻带取作者代码边界，上阻带取 90 Hz。
weights = np.arange(1, 5, dtype=float) ** (-1.25) + 0.25  # 原始方法四子带权重 w(m)=m^(-1.25)+0.25，用于候选频率的相关平方融合。
PAPER_ACCURACY = {("FS-eTRCA", "NS1"): 67.64, ("FS-eTRCA", "NS2"): 66.47, ("FS-eTRCA", "NS3"): 65.34, ("FS-eTRCA&FBCCA", "NS1"): 76.39, ("FS-eTRCA&FBCCA", "NS2"): 75.35, ("FS-eTRCA&FBCCA", "NS3"): 75.02}  # 原文第 4.3 节给出的六个平均准确率，仅用于最终对照。

threadpool_limits(limits=1, user_api="blas")  # 固定单线程 BLAS，使大量小型矩阵乘法和特征值求解的耗时更稳定。
started = perf_counter()  # 记录运行起点，最后统计本次从原始 MAT 到全部输出的总耗时。
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # 创建最终输出目录，已有目录可覆盖；不据此跳过任何计算。
channel_table = pd.read_excel(DATA_DIR / "Channel Loc.xlsx")  # 读取官方通道编号与电极名称的对应表。
channel_lookup = dict(zip(channel_table["Channel Name"].str.casefold(), channel_table["Channel Index"] - 1))  # 官方通道编号减一变成零基下标，并统一大小写使 POz 匹配表中的 Poz。
channel_indices = [channel_lookup[name.casefold()] for name in CHANNELS]  # 按固定九电极顺序取得原始通道下标。
notch_b, notch_a = iirnotch(50, Q=30, fs=FS)  # 使用与已有 4.2 一致的 50 Hz、Q=30 陷波器。
filter_sections = []  # 保存四个固定子带的二阶节系数，系数不通过任何类别标签拟合。
for passband, stopband in zip(PASSBANDS, STOPBANDS):  # 按四个通带与阻带的配对顺序构造滤波器。
    order, critical = cheb1ord(passband, stopband, gpass=3, gstop=40, fs=FS)  # 按作者公开设计使用 3 dB/40 dB 规格求最低原型阶数和临界频率。
    filter_sections.append(cheby1(order, rp=0.5, Wn=critical, btype="bandpass", fs=FS, output="sos"))  # 构造通带纹波 0.5 dB 的 Chebyshev I 带通；求阶数和构造滤波器的纹波参数沿用公开代码两次调用。
reference_bases = []  # 保存八个候选频率的正交正弦余弦参考，每个参考有五阶谐波、十列。
times = np.arange(1, FS + 1) / FS  # 一秒参考时间取 1/250 到 250/250 秒，正弦余弦共同张成的子空间不受统一时间平移影响。
for frequency in FREQUENCIES:  # 数学参考不依赖受试者或训练折，因此在读取数据前按固定频率构造。
    phase = 2 * np.pi * frequency * np.arange(1, 6)[:, None] * times[None, :]  # 广播生成五阶谐波的相位矩阵，形状为 (5,250)。
    reference = np.stack([np.sin(phase), np.cos(phase)], axis=1).reshape(10, FS)  # 排列 sin1、cos1、直到 sin5、cos5，共十条参考。
    reference -= reference.mean(axis=-1, keepdims=True)  # 按每条参考的时间均值中心化，满足典型相关对变量中心化的要求。
    basis, _ = np.linalg.qr(reference.T, mode="reduced")  # 对 (250,10) 参考取列正交基，后续与 EEG 正交基交叉矩阵的最大奇异值即第一典型相关。
    reference_bases.append(basis)  # 按八种刺激频率的原始顺序保存参考基。
prediction_rows = []  # 累积 24 人*3 状态*2 方法*192 段=27648 条外层测试预测。
metric_rows = []  # 保存每人每任务每方法一行，总计 24*3*2=144 行。
confusion = np.zeros((2, 3, 9, 9), dtype=int)  # 分别为两方法、三种 NS 构造九分类混淆矩阵，真实类别为行，预测类别为列。

for subject in SUBJECTS:  # 每个人独立训练和测试，受试者之间不共享模板、空间滤波器或 LDA。
    print(f"[{subject}] 读取九通道，计算两个 FS 方法", flush=True)  # 显示当前人，后面按外层折报告进度。
    raw = loadmat(DATA_DIR / subject / f"sub_{subject[1:]}_data_CS.mat", variable_names=["data_CS"])["data_CS"]  # 读取 (8,63,22500,12) 的官方 CS 数组。
    cs = raw[:, channel_indices, :, :].transpose(0, 3, 1, 2).copy()  # 选择九通道，转为 (8,12,9,22500)，最后一轴为时间。
    del raw  # 释放完整 63 通道数组，后续只处理九通道数据。
    cs -= cs.mean(axis=-1, keepdims=True)  # 对每类每次重复的各通道分别减去整条 4.5 秒记录的时间均值。
    cs = resample_poly(cs, up=1, down=ORIGINAL_FS // FS, axis=-1)  # 经过多相 FIR 抗混叠滤波降到 250 Hz，时间长度变为 1125。
    cs = filtfilt(notch_b, notch_a, cs, axis=-1)  # 完整试次内双向 50 Hz 陷波，不跨试次滤波。
    cs = cs[..., START:START + FS]  # 只取目标论文规定的 0.14-1.14 秒一秒 CS 窗。
    cs_bank = np.stack([sosfiltfilt(sections, cs, axis=-1) for sections in filter_sections])  # 四个子带分别滤波，得到 (4,8,12,9,250)。
    banks_by_state = [cs_bank.reshape(4, 96, len(CHANNELS), FS)]  # 将 CS 样本排成类别优先、重复次之的 96 行，后续接上三种 NS。
    ns_groups_by_state = []  # 保存每种 NS 的 96 片段分别属于哪条原始记录，用于外层分组。
    ns_segments_by_state = []  # 保存每段在原始记录中的起始秒位置，供结果追溯。
    for state in STATES:  # NS1、NS2、NS3 分别读取与预处理，不对三种状态标签进行混合训练。
        variable = f"data_{state}"  # 根据状态构造 MAT 变量名 data_NS1、data_NS2 或 data_NS3。
        raw = loadmat(DATA_DIR / subject / f"sub_{subject[1:]}_data_{state}.mat", variable_names=[variable])[variable]  # NS1/NS2 原始形状为 (63,20000,24)，NS3 为 (63,10000,48)。
        ns = raw[channel_indices].transpose(2, 0, 1).copy()  # 取九通道并转为 (原始记录数,9,时间点)，每条原始记录保持独立。
        del raw  # 释放未使用的其他通道数据，避免同时保留四个完整 MAT 大数组。
        ns -= ns.mean(axis=-1, keepdims=True)  # 每条原始记录各通道分别中心化，不混合不同记录。
        ns = resample_poly(ns, up=1, down=ORIGINAL_FS // FS, axis=-1)  # 降采样至 250 Hz，NS1/NS2 每条有 1000 点，NS3 每条有 500 点。
        ns = filtfilt(notch_b, notch_a, ns, axis=-1)  # 在完整原始 NS 记录内部做工频陷波，该记录的所有片段之后始终保持同折。
        segments_per_trial = ns.shape[-1] // FS  # 固定官方数据每记录恰好分为四个或两个一秒片段。
        ns_groups_by_state.append(np.repeat(np.arange(len(ns)), segments_per_trial))  # 为同一原始记录的所有片段写相同组号，避免相邻窗口跨训练和测试。
        ns_segments_by_state.append(np.tile(np.arange(segments_per_trial), len(ns)))  # 各记录内部片段位置依次为 0-3 或 0-1，直接对应一秒段的起始秒。
        ns = ns.reshape(len(ns), len(CHANNELS), segments_per_trial, FS).transpose(0, 2, 1, 3).reshape(96, len(CHANNELS), FS)  # 先拆时间为片段和采样点，再把片段轴移到通道前，保证每段保留对应九通道而不串乱通道。
        banks_by_state.append(np.stack([sosfiltfilt(sections, ns, axis=-1) for sections in filter_sections]))  # 对 96 个 NS 片段分别做四子带滤波，形状为 (4,96,9,250)。
    all_bank = np.concatenate(banks_by_state, axis=1)  # 按 CS、NS1、NS2、NS3 拼成 (4,384,9,250)，保留固定样本来源位置。
    cca_features = np.empty((384, 8, 4))  # 存放每片段、每候选频率、每子带的第一典型相关，数学参考法不依赖训练折。
    for sample_index in range(384):  # 每种状态的全部 96 个样本都提取 FBCCA 特征，两个 FS 版本共享这一无监督计算。
        for band_index in range(4):  # 各子带分别计算 CCA，不能先将四子带或九个通道平均。
            centered = all_bank[band_index, sample_index] - all_bank[band_index, sample_index].mean(axis=-1, keepdims=True)  # 将当前一秒九通道 EEG 按通道中心化。
            eeg_basis, _ = np.linalg.qr(centered.T, mode="reduced")  # 对 (250,9) EEG 取正交列基，避免迭代求解 CCA 权重。
            for candidate_index, reference_basis in enumerate(reference_bases):  # 对八个候选频率分别计算第一典型相关系数。
                cca_features[sample_index, candidate_index, band_index] = np.linalg.svd(eeg_basis.T @ reference_basis, compute_uv=False)[0]  # 两正交基交叉矩阵的最大奇异值即第一典型相关，结果非负。
    subject_predictions = np.empty((2, 3, 192), dtype=int)  # 保存两方法、三状态对每个样本的唯一一次外层测试预测。
    for fold in range(FOLDS):  # 每折留出 CS 每类一重复和 NS 对应原始记录组，模型仅由当前受试者训练部分获得。
        keep = np.arange(FOLDS) != fold  # 标记本折 11 个训练重复，留出的原始重复始终为 False。
        training_repeats = np.flatnonzero(keep)  # 依升序得到 11 个训练重复编号，内部留一特征与对应 FBCCA 特征必须保持同一顺序。
        fit_masks = [keep] + [keep & (np.arange(FOLDS) != repeat) for repeat in training_repeats]  # 第 0 组用于外层测试，其后 11 组各排除一个训练重复，用于无自身模板的 CS 训练特征。
        templates = np.empty((12, 4, 8, len(CHANNELS), FS))  # 保存当前外层折的一个十一重复模板和十一个十重复模板，均只在内存中使用。
        spatial_filters = np.empty((12, 4, 8, len(CHANNELS)))  # 对每组训练掩码、每子带、每频率保存一个九维 TRCA 滤波器。
        for fit_index, mask in enumerate(fit_masks):  # 统一执行相同的 TRCA 训练计算，首组 11 次训练，其余每组 10 次训练，不写重复训练函数。
            templates[fit_index] = cs_bank[:, :, mask].mean(axis=2)  # 只平均当前训练掩码包含的重复，得到 (4,8,9,250) 的类模板。
            for band_index in range(4):  # 四子带各自训练其八类滤波器，不用任何 NS 数据拟合 SSVEP 模板。
                for candidate_index in range(8):  # 每个频率只使用本频率的训练 CS 求最大跨试次一致性方向。
                    training = cs_bank[band_index, candidate_index, mask]  # 形状为 (11 或 10,9,250)，外层测试重复和对应内部留出重复均未进入。
                    centered_trials = training - training.mean(axis=-1, keepdims=True)  # 逐训练试次、逐通道去当前窗均值，用于跨试次协方差 S。
                    summed = centered_trials.sum(axis=0)  # 当前训练试次沿重复轴求和，得到 (9,250)。
                    centered_flat = centered_trials.transpose(1, 0, 2).reshape(len(CHANNELS), -1)  # 将逐试次中心化数据按通道拼接，用于扣除同一试次自身乘积。
                    inter_trial = summed @ summed.T - centered_flat @ centered_flat.T  # S 等于所有试次两两乘积之和减去各试次自身乘积，恰好保留 r!=s 的双向交叉协方差。
                    concatenated = training.transpose(1, 0, 2).reshape(len(CHANNELS), -1)  # 原始训练窗先按时间拼接，再进行一次通道总均值中心化以构造 Q。
                    concatenated = concatenated - concatenated.mean(axis=-1, keepdims=True)  # 保留不同训练窗均值之间的真实差异，与作者 trca.m 的 UX 定义一致。
                    total_covariance = concatenated @ concatenated.T  # Q 为全部拼接训练点的总协方差乘积，不加入测试信号或人为正则项。
                    _, eigenvectors = eigh(inter_trial, total_covariance, subset_by_index=[len(CHANNELS) - 1, len(CHANNELS) - 1])  # 只取最大广义特征值对应向量，它最大化 w.T*S*w/(w.T*Q*w)。
                    spatial_filters[fit_index, band_index, candidate_index] = eigenvectors[:, 0] / np.linalg.norm(eigenvectors[:, 0])  # 转为欧氏单位长度，防止八个 eTRCA 投影列因 Q 归一化带入不同任意尺度。
        template_vectors = np.einsum("bkc,bfct->bfkt", spatial_filters[0], templates[0]).reshape(4, 8, -1)  # 外层模型用八类滤波器投影每类平均模板，并将八条投影展开，得到 (4,8,2000)。
        template_vectors -= template_vectors.mean(axis=-1, keepdims=True)  # 二维相关对整块八投影矩阵减一个总均值，不先平均八个一维相关系数。
        template_vectors /= np.linalg.norm(template_vectors, axis=-1, keepdims=True)  # 将中心化模板向量归一化，后续内积即 Pearson 相关系数。
        projected = np.einsum("bkc,bnct->bnkt", spatial_filters[0], all_bank).reshape(4, 384, -1)  # 用同一个外层 eTRCA 空间矩阵投影全部片段，得到四子带各 384 个长度 2000 的向量。
        projected -= projected.mean(axis=-1, keepdims=True)  # 每片段每子带对整块投影矩阵中心化，匹配原始二维相关定义。
        projected /= np.linalg.norm(projected, axis=-1, keepdims=True)  # 各投影向量归一化为单位范数，便于直接矩阵乘法计算八类相关。
        etrca_features = np.stack([projected[band_index] @ template_vectors[band_index].T for band_index in range(4)], axis=-1)  # 得到 (384,8,4) 的有符号相关；每个子带的八候选都使用同一个八列空间矩阵。
        candidate_classes = np.argmax(np.einsum("nfb,b->nf", etrca_features ** 2, weights), axis=1)  # 先按四子带相关平方加权，再取八个总得分的最大者作为候选频率，不能使用真实类别。
        positive = np.empty((11, 8, 4))  # LDA 正类训练特征：11 个训练重复、八个真实频率、每频率四个子带相关。
        for position, repeat in enumerate(training_repeats):  # 对每个外层训练重复，选用完全排除该重复的内部模型提取正类特征。
            inner_test = np.einsum("bkc,bfct->bfkt", spatial_filters[position + 1], cs_bank[:, :, repeat]).reshape(4, 8, -1)  # 用其余十重复训练的滤波器投影当前八类 CS，不让当前样本参与自己的空间滤波器训练。
            inner_template = np.einsum("bkc,bfct->bfkt", spatial_filters[position + 1], templates[position + 1]).reshape(4, 8, -1)  # 同样用排除了当前重复的十重复模板，确保两侧来自同一内部模型。
            inner_test -= inner_test.mean(axis=-1, keepdims=True)  # 对每个频率的整块八投影测试矩阵中心化。
            inner_template -= inner_template.mean(axis=-1, keepdims=True)  # 对对应的训练模板投影矩阵中心化。
            positive[position] = (np.sum(inner_test * inner_template, axis=-1) / (np.linalg.norm(inner_test, axis=-1) * np.linalg.norm(inner_template, axis=-1))).T  # 只提取真实频率对应的四维相关作为该频率 LDA 的 CS 正样本，转置成 (8,4)。
        positive_cca = np.stack([cca_features[candidate_index * 12 + training_repeats, candidate_index] for candidate_index in range(8)], axis=1)  # 按相同训练重复次序取本类的四维 FBCCA 特征，形状为 (11,8,4)。
        for state_index, state in enumerate(STATES):  # 当前外层折依次训练三种 NS 的状态分类器，eTRCA 的 CS 模型共用，LDA 互不混合。
            state_indices = np.r_[np.arange(96), np.arange((state_index + 1) * 96, (state_index + 2) * 96)]  # 在 384 行总数据中选当前 CS 和这一种 NS，共 192 行。
            state_features = etrca_features[state_indices]  # 当前任务的 eTRCA 特征形状为 (192,8,4)，前 96 行是 CS。
            state_cca = cca_features[state_indices]  # 取同样样本顺序的 FBCCA 特征，只在融合版 LDA 中使用。
            labels = np.r_[np.repeat(np.arange(8), 12), np.full(96, 8)]  # 前 96 个 CS 的真实类别为 0-7，每类 12 次；后 96 个 NS 的真实类别固定为 8。
            original_trials = np.r_[np.arange(96), 96 + ns_groups_by_state[state_index]]  # CS 试次各自独立，NS 的多个片段共用原始记录号，并与 CS 组号隔开。
            segment_indices = np.r_[np.zeros(96, dtype=int), ns_segments_by_state[state_index]]  # CS 只有一个一秒段，NS 记录其原始记录内的分段位置。
            assignments = np.r_[np.tile(np.arange(FOLDS), 8), ns_groups_by_state[state_index] % FOLDS]  # 每折测试每种频率各一次 CS 和同组的八个 NS，所有相邻 NS 段保持同折。
            ns_train = assignments[96:] != fold  # 当前种 NS 的 88 个训练片段，不包含本折测试原始记录的任何片段。
            test = np.flatnonzero(assignments == fold)  # 选当前任务的 16 个外层测试位置，八 CS 加八 NS。
            candidates = candidate_classes[state_indices[test]]  # 候选类别来自外层 eTRCA 的最大得分，与两个 LDA 版本的选择完全一致。
            positives = [positive, np.concatenate([positive, positive_cca], axis=-1)]  # 两版本的正类特征分别为四维和八维，样本与频率轴保持相同。
            negatives = [state_features[96:][ns_train], np.concatenate([state_features[96:][ns_train], state_cca[96:][ns_train]], axis=-1)]  # 两版本都使用同一批 88 个训练 NS，融合版增加相同候选的四维 CCA。
            tests = [state_features[test], np.concatenate([state_features[test], state_cca[test]], axis=-1)]  # 当前折测试特征分别为 (16,8,4) 和 (16,8,8)，每候选独立送入对应 LDA。
            for method_index, (method, positive_features, negative_features, test_features) in enumerate(zip(METHODS, positives, negatives, tests)):  # 共用数据与候选，仅切换两种明确的特征输入，不加入额外方法。
                models = []  # 当前方法为八个频率各训练一个二分类 LDA，不能用一个统一分类器替代频率特定框架。
                for candidate_index in range(8):  # 每个分类器只接收该频率的特征和该频率的 CS 正样本。
                    training_features = np.concatenate([positive_features[:, candidate_index], negative_features[:, candidate_index]], axis=0)  # 拼接本频率 11 个 CS 和全部 88 个 NS，得到 99 个训练样本。
                    training_labels = np.r_[np.ones(11, dtype=int), np.zeros(88, dtype=int)]  # LDA 的内部二分类标签为 CS=1、NS=0，与最终九分类的频率标签分开。
                    model = LinearDiscriminantAnalysis(priors=[0.5, 0.5])  # 普通 LDA 使用相等状态先验，避免 11:88 样本数差直接决定先验；不加入未指定的协方差收缩。
                    model.fit(training_features, training_labels)  # 只用当前外层折的训练特征拟合，外层测试既不拟合模板也不拟合 LDA。
                    models.append(model)  # 将当前频率 LDA 按 0-7 原始频率顺序保存，后续按候选下标选择。
                decision_matrix = np.column_stack([model.decision_function(test_features[:, candidate_index]) for candidate_index, model in enumerate(models)])  # 计算八个 LDA 对当前测试片段对应频率特征的有符号判别值，形状为 (16,8)。
                decisions = decision_matrix[np.arange(len(test)), candidates]  # 每个测试片段只采用其 eTRCA 候选频率对应 LDA 的结果，不能按八个 LDA 的最大值另选频率。
                predicted = np.where(decisions > 0, candidates, 8)  # 对应 LDA 接受 CS 时输出候选频率 0-7，否则输出 NS=8；零判别值按类别排序判为 NS。
                subject_predictions[method_index, state_index, test] = predicted  # 写入当前方法当前任务的唯一一次外层测试预测，十二折填满 192 个样本。
                for position, sample_index in enumerate(test):  # 逐测试片段保存来源和判别结果，供独立复算及分组审查。
                    prediction_rows.append({"subject": subject, "state": state, "method": method, "fold": fold + 1, "sample": sample_index, "original_trial": original_trials[sample_index], "segment": segment_indices[sample_index], "true_class": labels[sample_index], "candidate_class": candidates[position], "predicted_class": predicted[position], "lda_decision": decisions[position]})  # 记录候选及其 LDA 决策值，能核查每个接受或拒绝是否遵循 FS 流程。
        print(f"  外层折 {fold + 1}/12：两个 FS 版本、三种 NS 已完成", flush=True)  # 每完成一个外层折显示进度，避免内部留一训练阶段长期无输出。
    labels = np.r_[np.repeat(np.arange(8), 12), np.full(96, 8)]  # 三种 NS 任务均采用相同的八频率+NS 真值顺序，供当前人整体指标统计。
    for method_index, method in enumerate(METHODS):  # 两个方法分别统计，不平均它们的预测。
        for state_index, state in enumerate(STATES):  # 每种 NS 独立得到一行 192 段测试结果指标。
            predicted = subject_predictions[method_index, state_index]  # 取该人当前方法当前 NS 的全部十二折预测。
            accuracy = np.mean(predicted == labels)  # 九分类样本准确率：所有 192 段等权，NS 因 96 段而占一半权重。
            recalls = np.array([np.mean(predicted[labels == target] == target) for target in range(9)])  # 每类分别算召回率，前八类各有 12 段，NS 有 96 段。
            balanced_accuracy = recalls.mean()  # 对九类召回率等权平均，单独输出，不能用它替换原文报告的样本准确率。
            binary_accuracy = np.mean((predicted != 8) == (labels != 8))  # 只判断控制/非控制状态时的二分类正确比例，CS 频率错但仍接受时算状态判断正确。
            cs_detection = np.mean(predicted[:96] != 8)  # 真正 CS 被接受为某个频率的比例，独立于候选频率是否正确。
            ns_false_positive = np.mean(predicted[96:] != 8)  # 真正 NS 被错误接受为任意频率的比例，即非控制状态误触发率。
            np.add.at(confusion[method_index, state_index], (labels, predicted), 1)  # 累加九分类混淆矩阵，真实与预测类别始终保持同一方向。
            metric_rows.append({"subject": subject, "state": state, "method": method, "accuracy": accuracy, "balanced_accuracy": balanced_accuracy, "binary_accuracy": binary_accuracy, "cs_detection_rate": cs_detection, "ns_false_positive_rate": ns_false_positive, "n_classes": 9, "n_samples": len(labels)})  # 保存 144 行每人指标，明确九分类、二状态和类均衡指标的不同含义。
            print(f"  {method} CS-{state}: accuracy={accuracy:.2%}, balanced={balanced_accuracy:.2%}", flush=True)  # 打印当前人的九分类准确率与九类均衡准确率。

predictions = pd.DataFrame(prediction_rows)  # 汇总全部 27648 条逐片段外层测试预测，方法之间保留相同样本来源。
metrics = pd.DataFrame(metric_rows)  # 汇总 24 人*3 状态*2 方法=144 行指标，按受试者等权统计。
summary = metrics.groupby(["method", "state"], as_index=False).agg(accuracy_mean=("accuracy", "mean"), accuracy_sd=("accuracy", "std"), balanced_accuracy_mean=("balanced_accuracy", "mean"), binary_accuracy_mean=("binary_accuracy", "mean"), cs_detection_mean=("cs_detection_rate", "mean"), ns_false_positive_mean=("ns_false_positive_rate", "mean"))  # 六种方法/状态组合各统计 24 人均值和样本标准差。
summary["accuracy_sem"] = summary.accuracy_sd / np.sqrt(len(SUBJECTS))  # 对跨人准确率均值计算 SEM，图中误差线使用该标准误。
comparison = summary[["method", "state", "accuracy_mean"]].rename(columns={"accuracy_mean": "reproduced"})  # 按论文报告的样本准确率生成六行对照，不混入其他指标。
comparison["reproduced"] *= 100  # 将比例转为百分数，与原文参考值单位统一。
comparison["paper"] = [PAPER_ACCURACY[(row.method, row.state)] for row in comparison.itertuples()]  # 按明确的方法和状态键查原文值，不依靠偶然的 DataFrame 行顺序。
comparison["difference_percentage_points"] = comparison.reproduced - comparison.paper  # 保存有符号百分点差值，计算输出不会反过来影响训练参数。
confusion_table = pd.DataFrame({"method": np.repeat(METHODS, 3 * 81), "state": np.tile(np.repeat(STATES, 81), 2), "true_class": np.tile(np.repeat(np.arange(9), 9), 6), "predicted_class": np.tile(np.arange(9), 6 * 9), "count": confusion.ravel()})  # 将六个 9*9 矩阵展平为 486 行，类别 8 明确代表 NS。
predictions.to_csv(OUTPUT_DIR / "predictions.csv", index=False)  # 保存每个样本的真值、候选、最终预测、折号、来源及 LDA 判别值。
metrics.to_csv(OUTPUT_DIR / "subject_metrics.csv", index=False)  # 保存每人每任务每方法的 144 行指标，方便独立复算。
summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)  # 保存六行跨人汇总，下一次运行仍从原始 MAT 重新计算。
comparison.to_csv(OUTPUT_DIR / "comparison_with_paper.csv", index=False)  # 保存六行原文对照及差值，不覆盖已有 4.1 或 4.2 结果。
confusion_table.to_csv(OUTPUT_DIR / "confusion_matrices.csv", index=False)  # 保存九分类混淆计数，每方法每 NS 的计数总和均为 24*192=4608。

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})  # 统一字体并隐藏顶部和右侧边框，改善最终图表可读性。
figure, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")  # 一张四面板图同时展示六个均值和三种 NS 各自的个体分布。
figure.suptitle("Section 4.3 | Frequency-specific methods | 9-class CS/NS", fontsize=14)  # 总标题强调两种 FS 方法与九分类任务，区别于 IFB-OCN 的二分类。
positions = np.arange(3)  # 三种 NS 任务的横轴位置固定为 0、1、2。
colors = ["#2074a5", "#e29b35"]  # 两个方法使用固定的蓝色和橙色，保持均值图与图例一致。
for method_index, method in enumerate(METHODS):  # 分别画两种方法的跨人均值与对应原文点。
    current = summary.loc[summary.method == method].set_index("state").loc[STATES]  # 显式按 NS1、NS2、NS3 顺序排统计量，避免横轴与数据错位。
    locations = positions + (method_index - 0.5) * 0.34  # 两个方法在每种 NS 周围左右错开，便于直接比较。
    axes[0, 0].bar(locations, current.accuracy_mean * 100, width=0.32, yerr=current.accuracy_sem * 100, capsize=3, color=colors[method_index], label=method)  # 画 24 人均值及跨人 SEM，均转为百分数。
    axes[0, 0].scatter(locations, [PAPER_ACCURACY[(method, state)] for state in STATES], marker="D", facecolors="white", edgecolors=colors[method_index], s=60, zorder=4, label=f"Paper {method}")  # 白心菱形配同方法边框颜色标出原文参考值，zorder=4 使标记显示在柱子上方，避免与柱子颜色融为一体。
axes[0, 0].set(xticks=positions, xticklabels=STATES, ylabel="9-class accuracy (%)", ylim=(0, 105), title="(a) Mean accuracy +/- SEM: 24 subjects")  # 样本准确率纵轴和 24 人均值标题明确图中的统计含义。
axes[0, 0].legend(fontsize=8, loc="lower right")  # 显示两个计算方法与两组原文参考点的图例。
for axis, state in zip(axes.ravel()[1:], STATES):  # 剩余三个面板依次展示 NS1、NS2、NS3 的个体分布。
    axis.boxplot([metrics.loc[(metrics.method == method) & (metrics.state == state), "accuracy"].to_numpy() * 100 for method in METHODS], tick_labels=METHODS, showmeans=True)  # 每个箱线图包含一个方法的全部 24 人准确率，并额外标记均值。
    axis.set(ylabel="9-class accuracy (%)", ylim=(0, 105), title=f"CS versus {state}: individual accuracy")  # 纵轴与均值图使用同一九分类样本准确率口径。
figure.savefig(OUTPUT_DIR / "fs_accuracy.png", dpi=200)  # 保存 2400*1800 像素的四面板图，重复运行覆盖本文件的最终图表。
plt.close(figure)  # 保存完成后关闭画布并释放绘图资源。
print(comparison.to_string(index=False, float_format="%.4f"), flush=True)  # 打印两方法三任务的实际均值、原文值和差值。
print(f"完成：{OUTPUT_DIR}\n用时：{perf_counter() - started:.1f} 秒", flush=True)  # 输出最终结果目录和完整运行耗时，不生成用于续跑的缓存。
