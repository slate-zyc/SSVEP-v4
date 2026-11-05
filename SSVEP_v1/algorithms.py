"""实现滤波器组、FBCCA、TRCA/eTRCA 以及 ITR 计算。

本模块只接收数组，不读取原始数据或进行交叉验证。相关系数按子带保留，
既可加权生成频率评分，也可直接作为频率特定 LDA 的输入特征。

滤波器组（filter bank）
为什么要把信号切成多个频带？

SSVEP 响应不是纯正弦，能量分布在基频和若干谐波上。不同频带的信息质量不同：

低子带（6–80 Hz）：保留基频（信号最强），但也保留大量低频噪声（眼动、肌电、漂移）；
高子带（30–80 Hz）：只剩谐波（信号弱），但噪声也少、波形更“干净”。
单独用任何一个都不理想。滤波器组的思路是：把同一段信号过 N 个不同通带的滤波器，得到 N 个版本，各自算相关，再加权融合。
低子带权重大、高子带权重小（w = m^−1.25 + 0.25），相当于“主要听低子带的，高子带提供辅助证据”。论文 4.2 节：四子带、通带 m×8−2 至 80 Hz。
"""

import numpy as np
from scipy.linalg import eigh       # 解广义特征值问题
from scipy.signal import cheb1ord, cheby1, sosfiltfilt  # 用于零相位滤波


# 四个子带零相位滤波，让滤波后的信号在时间上不移位——滤掉不需要的频率，但每一时刻的波形仍然对应原来的时刻
# 前向滤一遍，后向再滤一遍，延迟抵消
def filterbank(config, data):
    """沿最后一维进行四子带零相位滤波，最前面新增子带轴。"""
    filtered = []
    for passband, stopband in zip(config["filter_passbands"], config["filter_stopbands"]):
        # 通带范围、阻带范围、通带允许衰减、阻带要求衰减; 返回“满足要求的最小阶数”和“实际截止频率”
        order, critical = cheb1ord(passband, stopband, config["cheb_order_ripple_db"],
                                  config["stop_attenuation_db"], fs=config["fs"])
        sections = cheby1(order, config["cheb_ripple_db"], critical, btype="bandpass",
                          output="sos", fs=config["fs"])
        filtered.append(sosfiltfilt(sections, data, axis=-1))
    return np.stack(filtered)   # eg (8 类别, 12 试次, 9 通道, 250 点)


# 每个子带分配的权重
def weights(config):
    """返回 m^-1.25+0.25 型子带权重，具体指数和偏移由配置控制。低子带权重高、高子带权重低"""
    return np.arange(1, len(config["filter_passbands"]) + 1) ** config["weight_exponent"] + config["weight_offset"]


# CCA 系数的计算可以转为奇异值分解
def orthogonal_basis(data):
    """对中心化信号做 SVD，只保留数值非零方向，兼容秩亏信号。"""
    centered = data - data.mean(axis=-1, keepdims=True)
    left, singular, _ = np.linalg.svd(centered.T, full_matrices=False)
    if not len(singular) or singular[0] == 0:
        raise ValueError("CCA input has zero variance")
    tolerance = np.finfo(float).eps * max(centered.shape) * singular[0]
    return left[:, singular > tolerance]


# 生成参考信号：构造正余弦，包含基频和若干次的谐波，这里选到三次谐波
def reference_bases(config, samples):
    """生成各刺激频率及谐波的正余弦参考基；相位改变不改变完整参考子空间。"""
    times = np.arange(1, samples + 1) / config["fs"]    # 时间轴：1/250, 2/250, ..., samples/250 秒
    bases = []
    for frequency in config["frequencies"]:
        rows = []
        for harmonic in range(1, config["harmonics"] + 1):
            if frequency * harmonic >= config["fs"] / 2:
                raise ValueError("Reference harmonic exceeds Nyquist")
            phase = 2 * np.pi * frequency * harmonic * times
            rows.extend([np.sin(phase), np.cos(phase)])
        bases.append(orthogonal_basis(np.stack(rows)))
    return bases

# 生成特定参考信号，每个子带与他们计算 CCA ，子带加权融合
def fbcca_correlations(config, bank, bases=None):
    """返回 [子带, 候选频率] 的第一典型相关系数，不提前合并子带。"""
    bases = reference_bases(config, bank.shape[-1]) if bases is None else bases
    correlations = np.empty((len(bank), len(bases)))
    for band_index, data in enumerate(bank):
        basis = orthogonal_basis(data)
        for class_index, reference in enumerate(bases):
            correlations[band_index, class_index] = np.linalg.svd(
                basis.T @ reference, compute_uv=False
            )[0]
    return correlations


# 计算得分，选出最大的即可
def fbcca_scores(config, bank, bases=None):
    """将各子带的 CCA 相关系数融合成八个候选频率得分。"""
    return weights(config) @ fbcca_correlations(config, bank, bases) ** config["score_power"]

# 为每个刺激频率寻找一个空间滤波器 w，使滤波后信号 y = w^T x 在不同试次间的协方差最大化。
# 广义特征值问题
# 这个空间滤波器是训练来的，上一个方法的是构建来的
def fit_trca(config, training):
    """用训练集学习模板和空间滤波器；输入为 [子带, 类别, 重复, 通道, 时间]。"""
    # 训练数据 (n_bands, n_classes, n_repeats, n_channels, n_samples)
    # eg (4, 8, 11, 9, 250) # 250 表示 1 s
    band_count, class_count, _, channel_count, _ = training.shape
    templates = training.mean(axis=2)       # 训练数据11试次，平均
    filters = np.empty((band_count, class_count, channel_count))    # 每个子带、每个类别一个 9 维空间滤波器。
    # 双重循环：4 子带 × 8 类别 = 32 个滤波器，逐个训练。trials 形状 (11 重复, 9 通道, 250 点)。
    for band_index in range(band_count):
        for class_index in range(class_count):
            trials = training[band_index, class_index]
            trials = trials - trials.mean(axis=-1, keepdims=True)   # 逐试次去均值
            # Q 是单试次协方差之和；S 用总信号外积减 Q，等价于跨试次协方差和。
            within = np.einsum("rct,rdt->cd", trials, trials)   # 表示对 r 和 t 求和
            summed = trials.sum(axis=0)
            between = summed @ summed.T - within
            ridge = config["trca_ridge_relative"] * np.trace(within) / channel_count
            if ridge <= 0:
                raise ValueError("TRCA input has zero variance")
            # 解 S w = lambda Q w，最大特征值方向最大化跨试次可重复成分。
            eigenvalues, eigenvectors = eigh(between, within + ridge * np.eye(channel_count))
            spatial_filter = eigenvectors[:, np.argmax(eigenvalues)]
            filters[band_index, class_index] = spatial_filter / np.linalg.norm(spatial_filter)
    return templates, filters


def pearson_flat(first, second):
    """将投影信号整体展平后计算相关；零方差时返回 0，避免 NaN 传播。"""
    first = first.ravel() - first.mean()
    second = second.ravel() - second.mean()
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    return float(first @ second / denominator) if denominator > 0 else 0.0

# 计算相关系数，(n_bands, n_classes) (4, 8)
def trca_correlations(config, bank, model, ensemble=False):
    """返回未融合的子带相关；ensemble=True 时联合各类别的空间滤波器（eTRCA）。"""
    templates, filters = model
    correlations = np.empty((len(bank), templates.shape[1]))
    for band_index, data in enumerate(bank):
        for class_index, template in enumerate(templates[band_index]):
            projection = filters[band_index] if ensemble else filters[band_index, class_index][None, :]
            correlations[band_index, class_index] = pearson_flat(projection @ data, projection @ template)
    return correlations
'''
eTRCA 的相关为什么是“整体”算的？ 因为 pearson_flat 会把 (8, 250) 展平成 2000 维向量，相当于把八个类别的投影首尾拼接成一条长信号，再与同样拼接的模板比。
这样相关衡量的是“八个类别的投影模式是否同时吻合”，而不是某一类的单独吻合——这就是 "ensembled"（集成）的含义：用全部类别的信息共同判断，比单类更稳。
缺点是要求测试信号在所有类别上都与模板一致，容错性低。
'''


# 加权求和
def trca_scores(config, bank, model, ensemble=False):
    """将 TRCA/eTRCA 的子带相关系数融合为候选类别得分。"""
    return weights(config) @ trca_correlations(config, bank, model, ensemble) ** config["score_power"]


# 套公式计算 ITR = [log2 N + P·log2 P + (1-P)·log2((1-P)/(N-1))] × 60 / T
def information_rate(accuracy, class_count, duration):
    """计算标准对称信道 ITR；机会水平以下按 0 处理，不解释为有效信息传输。"""
    if duration <= 0 or class_count < 2 or not 0 <= accuracy <= 1:
        raise ValueError("Invalid ITR arguments")
    if accuracy <= 1 / class_count:
        return 0.0
    entropy = np.log2(class_count) + accuracy * np.log2(accuracy)
    if accuracy < 1:
        entropy += (1 - accuracy) * np.log2((1 - accuracy) / (class_count - 1))
    return float(entropy * 60 / duration)
