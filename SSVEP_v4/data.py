"""数据读取与预处理：把原始 MAT 文件变成第 4 章各节要用的数组。

这个文件是三个小节共用的数据入口，按用途分成三组：

4.1（12 Hz 信号质量）
    load_channel_names()        读官方电极表 → 63 个电极名 + Oz 的下标
    load_twelve_hz_response()   读 sub_XX_data_CS.mat，取 12 Hz 类别、把 12 次重复
                                平均，再预处理 → [63 通道, 1125 点]
    display_waveform()          7-70 Hz 带通，只用于图 (a) 的显示波形

4.2 / 4.3（需要保留每一次重复，做 12 折交叉验证）
    channel_indices()           按电极名查出原始数组里的列号
    load_cs_trials()            读一个人的全部 CS 试次并预处理
                                → [8 频率, 12 重复, 通道, 1125 点]
                                （4.3 的 IFB-OCN 传 channels=["Fp1"] 只取单通道）

4.3（NS 状态，读的是完整原始记录，之后再切成 1 秒片段）
    load_ns_trials()            读一个人的一种 NS 状态并预处理
                                → [原始记录数, 通道, 1000 或 500 点]

三个小节的预处理完全相同：
    逐试次/逐记录去均值 → 5000→250 Hz 抗混叠降采样 → 50 Hz 零相位陷波。
"""

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, iirnotch, resample_poly, sosfiltfilt

from config import (DATA_DIR, DISPLAY_BAND, FS, NOTCH_HZ, NOTCH_Q,
                    ORIGINAL_FS, POSTERIOR_CHANNELS, TARGET_INDEX)


def load_channel_names():
    """读取官方通道表，返回 (63 个电极名, Oz 在数组里的下标)。

    Excel 第 1 列是通道编号（1~63），第 2 列是电极名。按编号排序后，
    列表下标就和 MAT 数组通道轴的顺序一一对应。Oz 是第 16 个通道，
    所以在 Python 里的下标是 15。
    """
    table = pd.read_excel(DATA_DIR / "Channel Loc.xlsx").sort_values("Channel Index")
    names = table["Channel Name"].tolist()
    return names, names.index("Oz")


def load_twelve_hz_response(subject):
    """读取一个人的 12 Hz 平均响应，返回 [63, 1125] 的预处理结果。

    参数 subject 形如 "S01"；文件里 data_CS 的形状是
    [8 频率, 63 通道, 22500 点, 12 重复]。
    """
    # 拼出文件路径，例如 EEG_data/S01/sub_01_data_CS.mat。
    # subject[1:] 把 "S01" 变成 "01"。
    path = DATA_DIR / subject / f"sub_{subject[1:]}_data_CS.mat"
    raw = loadmat(path, variable_names=["data_CS"])["data_CS"]

    # 第 1 步：取 12 Hz 的频率块（下标 4），沿重复轴平均 → [63, 22500]。
    # 预处理是线性运算，"先对 12 次重复平均再滤波" 和 "逐次滤波再平均"
    # 结果相同，但先平均能省掉 12 倍的计算量和内存。
    response = raw[TARGET_INDEX].mean(axis=-1)
    del raw          # 原始数组占 1 GB，平均完立即释放

    # 第 2 步：去直流。每个通道减去自己的时间均值。
    # 这份数据没有刺激前基线，所以只能对整段 4.5 秒取均值。
    response = response - response.mean(axis=-1, keepdims=True)

    # 第 3 步：降采样 5000 Hz → 250 Hz。
    # resample_poly 会先做抗混叠低通再抽点，不能直接每隔 20 个点取一个。
    response = resample_poly(response, up=1, down=ORIGINAL_FS // FS, axis=-1)

    # 第 4 步：50 Hz 工频陷波，零相位（正向、反向各滤一次，相位互相抵消）。
    b, a = iirnotch(NOTCH_HZ, NOTCH_Q, fs=FS)
    return filtfilt(b, a, response, axis=-1)


def display_waveform(signal):
    """对一维波形做 7-70 Hz 带通，只用于图 (a) 的显示。

    幅度谱和 SNR 用的是没有做这一步的原始响应，因为显示滤波器会改变
    各频点的幅度，论文的方法里也没有把它用于 4.1 的谱分析。
    """
    sections = butter(4, DISPLAY_BAND, btype="bandpass", fs=FS, output="sos")
    return sosfiltfilt(sections, signal)    # 零相位滤波


# ===========================================================================
# 4.2 / 4.3 节：读取每个试次（不平均重复）
# ===========================================================================

def channel_indices(names):
    """按电极名从官方通道表查出零基下标。

    官方 Excel 里通道编号从 1 开始，直接用 numpy 索引要减 1。
    """
    table = pd.read_excel(DATA_DIR / "Channel Loc.xlsx")
    lookup = dict(zip(table["Channel Name"].str.casefold(), table["Channel Index"] - 1))
    return [lookup[name.casefold()] for name in names]


def load_cs_trials(subject, channels=None):
    """读取一个人的全部 CS 试次并预处理。

    4.2 / 4.3 节要用 12 折交叉验证，必须保留每一次重复，不能像 4.1 节那样
    先把 12 次重复平均掉。channels 默认是 9 个后部电极；4.3 的 IFB-OCN
    只需要 Fp1 时传入 ["Fp1"]。

    返回 [8 频率, 12 重复, 通道数, 1125 点]。
    """
    channels = channels or POSTERIOR_CHANNELS
    path = DATA_DIR / subject / f"sub_{subject[1:]}_data_CS.mat"
    raw = loadmat(path, variable_names=["data_CS"])["data_CS"]       # (8, 63, 22500, 12)

    # 先只取需要的电极，再把轴顺序统一成 [频率, 重复, 通道, 时间]。
    indices = channel_indices(channels)
    data = raw[:, indices, :, :].transpose(0, 3, 1, 2).copy()        # (8, 12, 通道, 22500)
    del raw

    # 预处理和 4.1 节完全相同，只是这里对每个试次单独做，而不是先平均。
    data -= data.mean(axis=-1, keepdims=True)                        # 去直流
    data = resample_poly(data, up=1, down=ORIGINAL_FS // FS, axis=-1)  # 5000 → 250 Hz
    b, a = iirnotch(NOTCH_HZ, NOTCH_Q, fs=FS)
    return filtfilt(b, a, data, axis=-1)                             # 50 Hz 陷波


# ===========================================================================
# 4.3 节：读取 NS 状态的每个原始记录（每条记录稍后切成 1 秒片段）
# ===========================================================================

def load_ns_trials(subject, state, channels=None):
    """读取一个人的一种 NS 状态并预处理。

    NS1/NS2 的原始文件是 (63, 20000, 24)，NS3 是 (63, 10000, 48)；
    统一转成 [原始记录数, 通道数, 时间点]，其中时间点分别为 1000 或 500。

    预处理与 CS 相同：去均值 → 降采样 → 50 Hz 陷波，只是这里对每条
    完整记录单独做，随后再切成不重叠的 1 秒片段。
    """
    channels = channels or POSTERIOR_CHANNELS
    path = DATA_DIR / subject / f"sub_{subject[1:]}_data_{state}.mat"
    raw = loadmat(path, variable_names=[f"data_{state}"])[f"data_{state}"]

    indices = channel_indices(channels)
    data = raw[indices, :, :].transpose(2, 0, 1).copy()               # (原始记录, 通道, 时间)
    del raw

    data -= data.mean(axis=-1, keepdims=True)                         # 每条记录各自去直流
    data = resample_poly(data, up=1, down=ORIGINAL_FS // FS, axis=-1)  # 5000 → 250 Hz
    b, a = iirnotch(NOTCH_HZ, NOTCH_Q, fs=FS)
    return filtfilt(b, a, data, axis=-1)                              # 50 Hz 陷波
