"""四种数学工具：滤波器组、FBCCA、TRCA/eTRCA、信息传输率 ITR。

这一层只做数组运算，不读文件、不做交叉验证。所有函数的时间轴都是最后一维，
所以同一份代码既能处理单个试次 [通道, 时间]，也能一次处理整批数据
[子带, 类别, 重复, 通道, 时间]。
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh
from scipy.signal import cheb1ord, cheby1, sosfiltfilt

from .config import n_bands, sampling_rate


# ---------------------------------------------------------------------------
# 1. 滤波器组（filter bank）
# ---------------------------------------------------------------------------

def filter_sections(config: dict) -> list[np.ndarray]:
    """为每个子带设计一个 Chebyshev I 带通滤波器，返回二阶节(sos)列表。

    这里用“阶数由指标反推”的写法：给定通带、阻带和衰减要求，
    cheb1ord 算出满足要求的最小阶数，cheby1 再设计出滤波器。
    """
    bank = config["filter_bank"]
    fs = sampling_rate(config)
    sections = []
    for passband, stopband in zip(bank["passbands"], bank["stopbands"]):
        order, critical = cheb1ord(passband, stopband,
                                   bank["order_ripple_db"],
                                   bank["stop_attenuation_db"], fs=fs)
        sections.append(cheby1(order, bank["ripple_db"], critical,
                               btype="bandpass", output="sos", fs=fs))
    return sections


def filter_bank(config: dict, signals: np.ndarray) -> np.ndarray:
    """沿时间轴同时做 4 个子带的零相位滤波，返回在轴 0 新增子带维。

    signals 的最后两维必须是 [通道, 时间]，形状可以是
        [通道, 时间]                 → [4, 通道, 时间]
        [类别, 重复, 通道, 时间]      → [4, 类别, 重复, 通道, 时间]

    为什么要分 4 个子带：SSVEP 响应不只是基频，还包含谐波；
    低频子带信噪比高但噪声多，高频子带干净但信号弱，融合起来更稳。
    """
    return np.stack([sosfiltfilt(section, signals, axis=-1)
                     for section in filter_sections(config)])


def band_weights(config: dict) -> np.ndarray:
    """子带权重 w_m = m^(-1.25) + 0.25（m 从 1 开始）。

    越低的子带权重越大：低频携带的基频信息最多，高频只做补充。
    """
    bank = config["filter_bank"]
    m = np.arange(1, n_bands(config) + 1)
    return m ** bank["weight_exponent"] + bank["weight_offset"]


# ---------------------------------------------------------------------------
# 2. CCA / FBCCA（不需要训练，只和参考信号比相关）
# ---------------------------------------------------------------------------

def orthonormal_basis(signals: np.ndarray) -> np.ndarray:
    """把 [通道, 时间] 信号变成 [时间, 秩] 的正交基。

    做法是对中心化后的信号做 SVD，只保留奇异值明显大于 0 的方向。
    这样即使输入是秩亏的（比如几路信号完全相同），也不会把数值噪声
    当成“新的独立方向”，CCA 的结果才可靠。
    """
    centered = signals - signals.mean(axis=-1, keepdims=True)
    left, singular, _ = np.linalg.svd(centered.T, full_matrices=False)
    if singular.size == 0 or singular[0] == 0:
        raise ValueError("CCA 输入没有变化（方差为 0）")

    tolerance = np.finfo(float).eps * max(centered.shape) * singular[0]
    return left[:, singular > tolerance]


def reference_bases(config: dict, n_samples: int) -> list[np.ndarray]:
    """为每个刺激频率生成正余弦参考基（基频 + 3 次谐波）。

    对频率 f，参考信号是
        sin(2πf t), cos(2πf t), sin(4πf t), cos(4πf t), ...
    它们合起来构成一个子空间；把 8 个频率各自的子空间存成 8 个基矩阵。

    注意时间轴从 1/fs 开始（和 MATLAB 实现一致）。
    """
    fs = sampling_rate(config)
    n_harmonics = config["paradigm"]["harmonics"]
    times = np.arange(1, n_samples + 1) / fs

    bases = []
    for frequency in config["paradigm"]["frequencies"]:
        rows = []
        for harmonic in range(1, n_harmonics + 1):
            if frequency * harmonic >= fs / 2:
                raise ValueError("参考信号的谐波超过了奈奎斯特频率")
            phase = 2 * np.pi * frequency * harmonic * times
            rows.extend([np.sin(phase), np.cos(phase)])
        bases.append(orthonormal_basis(np.stack(rows)))
    return bases


def cca_correlations(bases: list[np.ndarray], signals: np.ndarray) -> np.ndarray:
    """一个试次 [通道, 时间] 与 8 个参考基的一阶典型相关系数，返回 [8]。

    典型相关的几何含义：把信号子空间和参考子空间各看成一堆方向，
    在两个子空间之间找一对“最像”的方向，它们的夹角余弦就是相关系数。
    数学上就是两个正交基矩阵乘积的最大奇异值。
    """
    signal_basis = orthonormal_basis(signals)
    return np.array([np.linalg.svd(signal_basis.T @ reference, compute_uv=False)[0]
                     for reference in bases])


def fbcca_correlations(config: dict, bank: np.ndarray,
                       bases: list[np.ndarray] | None = None) -> np.ndarray:
    """bank: [子带, 通道, 时间] → [子带, 候选频率] 的相关系数。

    保留“每个子带各自的相关系数”而不是直接加权，是因为异步分类里的
    频率特定 LDA 需要把这 4 个子带相关当作 4 维特征使用。
    """
    if bases is None:
        bases = reference_bases(config, bank.shape[-1])
    return np.stack([cca_correlations(bases, band) for band in bank])


def fbcca_scores(config: dict, bank: np.ndarray,
                 bases: list[np.ndarray] | None = None) -> np.ndarray:
    """FBCCA 的 8 个候选频率得分：Σ_m w_m · r_m^p。

    r_m 是第 m 个子带的相关系数，w_m 是子带权重，p 由 score_power 控制
    （本实现用 p=1，即线性加权，与参考 MATLAB 实现一致）。
    """
    correlations = fbcca_correlations(config, bank, bases)
    return band_weights(config) @ correlations ** config["filter_bank"]["score_power"]


# ---------------------------------------------------------------------------
# 3. TRCA / eTRCA（需要训练：从训练试次里学空间滤波器和模板）
# ---------------------------------------------------------------------------

def trca_spatial_filter(config: dict, trials: np.ndarray) -> np.ndarray:
    """从一个类别的训练试次学一个空间滤波器 w（长度 = 通道数）。

    输入 trials: [重复, 通道, 时间]（已经按时间中心化）。

    记第 r 个试次为 X_r，则
        Q = Σ_r X_r X_rᵀ            （同一试次内的空间协方差）
        S = Σ_{r≠s} X_r X_sᵀ        （不同试次之间重复出现的成分）
    我们要的 w 让“可重复成分”最大、同时压制“试次内噪声”，
    于是解广义特征值问题  S w = λ Q w，取 λ 最大的特征向量。

    S 不直接二重循环算，而是用 (ΣX_r)(ΣX_r)ᵀ - Q = S 这个恒等式，
    与逐对求和完全等价，但只用一次矩阵乘法。
    """
    trials = trials - trials.mean(axis=-1, keepdims=True)
    n_channels = trials.shape[1]

    within = np.einsum("rct,rdt->cd", trials, trials)       # Q
    summed = trials.sum(axis=0)
    between = summed @ summed.T - within                    # S
    if np.trace(within) <= 0:
        raise ValueError("TRCA 输入没有变化（方差为 0）")

    # 极小的岭正则项，保证数值上一定可解（相对迹的 1e-8）。
    ridge = config["trca"]["ridge_relative"] * np.trace(within) / n_channels
    eigenvalues, eigenvectors = eigh(between, within + ridge * np.eye(n_channels))
    w = eigenvectors[:, np.argmax(eigenvalues)]
    return w / np.linalg.norm(w)


def fit_trca(config: dict, training: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """用训练试次学习模板和空间滤波器。

    输入 training: [子带, 类别, 训练重复, 通道, 时间]
    返回
        templates: [子带, 类别, 通道, 时间]   （该类别的平均波形）
        filters  : [子带, 类别, 通道]         （每个类别一个空间滤波器）
    """
    n_bands_, n_classes = training.shape[0], training.shape[1]
    n_channels = training.shape[3]

    templates = training.mean(axis=2)
    filters = np.empty((n_bands_, n_classes, n_channels))
    for band in range(n_bands_):
        for target in range(n_classes):
            filters[band, target] = trca_spatial_filter(config, training[band, target])
    return templates, filters


def pearson_on_flat(first: np.ndarray, second: np.ndarray) -> float:
    """把两个数组展平后算 Pearson 相关；任一为零方差时返回 0。"""
    first = first.ravel() - first.mean()
    second = second.ravel() - second.mean()
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    return float(first @ second / denominator) if denominator > 0 else 0.0


def trca_correlations(config: dict, bank: np.ndarray,
                      model: tuple[np.ndarray, np.ndarray],
                      ensemble: bool = False) -> np.ndarray:
    """测试信号与各类别模板的相关系数，返回 [子带, 类别]。

    ensemble=False（TRCA）：每个类别用自己训练出的滤波器 w_c 投影后比相关。
    ensemble=True （eTRCA）：把 8 个类别的滤波器拼在一起（8×通道）同时投影，
        于是相关系数衡量的是“8 个类别的投影模式是否同时吻合”，比单类更稳。
    """
    templates, filters = model
    correlations = np.empty((len(bank), templates.shape[1]))
    for band, data in enumerate(bank):
        for target, template in enumerate(templates[band]):
            if ensemble:
                projection = filters[band]      # [8, 通道] 一次性投影
            else:
                projection = filters[band, target][None, :]
            correlations[band, target] = pearson_on_flat(projection @ data,
                                                          projection @ template)
    return correlations


def trca_scores(config: dict, bank: np.ndarray, model, ensemble: bool = False) -> np.ndarray:
    """把 [子带, 类别] 相关按子带权重融合成 8 个类别得分。"""
    correlations = trca_correlations(config, bank, model, ensemble)
    return band_weights(config) @ correlations ** config["filter_bank"]["score_power"]


# ---------------------------------------------------------------------------
# 4. ITR（信息传输率）
# ---------------------------------------------------------------------------

def itr(accuracy: float, n_classes: int, duration_seconds: float) -> float:
    """标准 ITR，单位 bits/min。

    公式（Wolpaw 等）： ITR = (log2 N + P log2 P + (1-P) log2((1-P)/(N-1))) × 60 / T
    P 是准确率，N 是类别数，T 是做出一次决策所需的总时间（秒）。
    P 不高于机会水平 1/N 时按 0 处理，避免算出负的“信息率”。
    """
    if duration_seconds <= 0 or n_classes < 2 or not 0 <= accuracy <= 1:
        raise ValueError("ITR 参数不合法")
    if accuracy <= 1 / n_classes:
        return 0.0

    bits = np.log2(n_classes) + accuracy * np.log2(accuracy)
    if accuracy < 1:
        bits += (1 - accuracy) * np.log2((1 - accuracy) / (n_classes - 1))
    return float(bits * 60 / duration_seconds)


def decision_seconds(config: dict, window_seconds: float) -> float:
    """一次决策的总耗时 = 观察窗 + 提示时间 + 视觉延迟。"""
    paradigm = config["paradigm"]
    return window_seconds + paradigm["cue_seconds"] + paradigm["delay_seconds"]
