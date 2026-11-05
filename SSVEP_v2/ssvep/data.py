"""原始数据 → 干净信号：读 MAT、选通道、预处理、切窗、缓存。

先把全项目统一的数组约定说清楚（后面每个模块都遵守它）：

    最后一维  = 时间
    倒数第二维 = 通道

    CS（连续闪烁，4.5 s）: [频率 8, 重复 12, 通道, 时间 1125]
    NS（空闲状态）       : [原始试次, 通道, 时间]  20000 点(4 s) 或 10000 点(2 s)

预处理只做三件事，顺序不能换：

    1. 每个原始试次自己去均值（数据没有刺激前基线，只能这样去直流）
    2. 多相抗混叠降采样 5000 Hz → 250 Hz（先低通再抽点，避免混叠）
    3. 50 Hz 零相位陷波（滤掉工频干扰且不引入时间延迟）

预处理结果存进 cache/ 目录，文件名由“源文件 + 参数 + 通道”的哈希决定：
任何一项变了，文件名就变，旧缓存自然不会被误用。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat, whosmat
from scipy.signal import filtfilt, iirnotch, resample_poly

from .config import sampling_rate

CACHE_VERSION = 2                    # 缓存格式版本；改预处理逻辑时 +1
CHANNEL_TABLE_FILE = "Channel Loc.xlsx"


# ---------------------------------------------------------------------------
# 电极表：官方 Excel 决定了“第几列是哪个电极”，不能靠猜。
# ---------------------------------------------------------------------------

def channel_table(config: dict) -> dict[str, int]:
    """读取官方电极表，返回 {电极名(小写): 0 起始的列号}。

    Excel 第一列是 MATLAB 的 1 起始编号，第二列是电极名。
    这里做三件事：去掉空行、把 1 起始编号减 1、名字统一小写。
    同时校验 63 个编号各出现一次，防止表被改坏后静默出错。
    """
    path = Path(config["paths"]["data"]) / CHANNEL_TABLE_FILE
    table = pd.read_excel(path).iloc[:, :2].dropna()

    indices = pd.to_numeric(table.iloc[:, 0], errors="raise").astype(int)
    names = table.iloc[:, 1].astype(str).str.strip()

    if len(table) != 63 or sorted(indices) != list(range(1, 64)):
        raise ValueError("通道表必须恰好包含 1..63 每个编号一次")
    if names.str.casefold().duplicated().any():
        raise ValueError("通道表里有重名的电极")

    return dict(zip(names.str.casefold(), indices - 1))     # 强制转为小写


def channel_indices(config: dict, names: list[str]) -> list[int]:
    """把电极名列表翻译成原始数组里的列号，顺序完全按 names 给的顺序。"""
    table = channel_table(config)
    return [int(table[name.casefold()]) for name in names]


def posterior_channels(config: dict) -> list[str]:
    """频率识别/异步分类用的后部 9 通道。"""
    return list(config["paradigm"]["posterior_channels"])


# ---------------------------------------------------------------------------
# 读原始文件
# ---------------------------------------------------------------------------

def mat_path(config: dict, subject: str, state: str) -> Path:
    """一个人的一种状态对应的 MAT 文件路径，例如 EEG_data/S01/sub_01_data_CS.mat。

    subject 形如 "S01"，subject[1:] 就是 "01"。
    """
    path = Path(config["paths"]["data"]) / subject / f"sub_{subject[1:]}_data_{state}.mat"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def read_mat(config: dict, subject: str, state: str) -> np.ndarray:
    """只读取需要的那个变量（data_CS / data_NS1 ...），其余变量不加载。"""
    path = mat_path(config, subject, state)
    variable = "data_" + state
    return loadmat(path, variable_names=[variable])[variable]   # loadmat() 返回的是字典，[variable] 是原文里面还有其他键
    # eg: reshape: (8, 63, 22500, 12) [频率, 通道, 时间, 试次]

def audit(config: dict, subjects: list[str]) -> pd.DataFrame:
    """检查每个 MAT 的变量名和维度是否与论文一致，返回一张清单表。

    .mat 文件像一个“变量仓库”，每个变量是有名字的条目；loadmat 把它读成一个字典，变量名就是字典的键，变量值就是字典的值。

    只读 MAT 的目录信息（whosmat），不加载数据本体，所以很快。
    """
    rows = []
    for subject in subjects:
        for state in ("CS", "NS1", "NS2", "NS3"):
            path = mat_path(config, subject, state)
            variables = whosmat(path)   # # 只读 MAT 文件的变量目录，不加载数据本体。返回一个列表，每个元素是一个三元组，格式为 (变量名, 维度, 数据类型)
            expected_name = "data_" + state
            found = [item for item in variables if item[0] == expected_name]
            if len(found) != 1:
                raise ValueError(f"{path} 里找不到唯一的 {expected_name}，实际有 {variables}")

            _, shape, dtype = found[0]
            if state == "CS":
                expected_shape = (8, 63, 22500, 12)     # [频率, 通道, 时间, 试次]
            elif state == "NS3":
                expected_shape = (63, 10000, 48)    # [通道, 时间, 试次]
            else:
                expected_shape = (63, 20000, 24)
            if shape != expected_shape:
                raise ValueError(f"{path} 的维度是 {shape}，应为 {expected_shape}")

            rows.append(dict(subject=subject, state=state, variable=expected_name,
                             shape=str(shape), dtype=dtype, bytes=path.stat().st_size))

    channel_indices(config, posterior_channels(config))   # 顺便校验电极表可读
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 预处理
# ---------------------------------------------------------------------------

def preprocess(config: dict, signals: np.ndarray) -> np.ndarray:
    """去均值 → 抗混叠降采样 → 50 Hz 零相位陷波。输入不会被修改。"""
    eeg = config["eeg"]
    original_fs, fs = int(eeg["original_fs"]), int(eeg["fs"])   # 5000, 250
    x = np.array(signals, dtype=np.float64, copy=True)
    if not np.isfinite(x).all():
        raise ValueError("信号里有 NaN 或 Inf")

    # 1. 每个试次自己的均值当直流分量去掉。
    x -= x.mean(axis=-1, keepdims=True)

    # 2. 降采样必须是整数倍；resample_poly 内部先抗混叠低通再抽点。
    if original_fs % fs:
        raise ValueError("原始采样率必须能被目标采样率整除")
    x = resample_poly(x, up=1, down=original_fs // fs, axis=-1)

    # 3. 陷波并做零相位滤波（前后各滤一遍，相位正好抵消）。
    b, a = iirnotch(eeg["notch_hz"], eeg["notch_q"], fs=fs)
    return filtfilt(b, a, x, axis=-1)   # 零相位滤波


# ---------------------------------------------------------------------------
# 带缓存的读取：日常流程都走这里
# ---------------------------------------------------------------------------

def cache_file(config: dict, subject: str, state: str, source: Path,
               channels: list[str], tag: str = "", extra: dict | None = None) -> Path:
    """算出某个预处理结果的缓存文件路径。

    文件名里的哈希由“源文件信息 + 预处理参数 + 通道（+ extra）”组成，
    所以换参数、换通道、重下数据都会自动换成新文件，不会读到过期结果。
    extra 用来把额外参数（例如 OCN 的候选网格）也纳入哈希。
    """
    eeg = config["eeg"]
    settings = {
        "version": CACHE_VERSION,
        "original_fs": eeg["original_fs"],
        "fs": eeg["fs"],
        "notch_hz": eeg["notch_hz"],
        "notch_q": eeg["notch_q"],
        "channels": list(channels),
        "source": [source.name, source.stat().st_size, source.stat().st_mtime_ns],
        "extra": extra or {},
    }
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:16]
    prefix = f"{subject}_{tag or state}"
    return Path(config["paths"]["cache"]) / f"{prefix}_{digest}.npy"


def load_state(config: dict, subject: str, state: str,
               channels: list[str] | None = None) -> np.ndarray:
    """读取一个人的一种状态并预处理，结果会缓存。

    channels 给 None 时用后部 9 通道（频率识别/异步分类的默认选择）。

    返回
    ----
    CS : [8, 12, 通道, 1125]
    NS : [原始试次, 通道, 1000]（NS3 是 [48, 通道, 500]）
    """
    channels = channels or posterior_channels(config)
    source = mat_path(config, subject, state)
    cached = cache_file(config, subject, state, source, channels)
    if cached.is_file():
        return np.load(cached, allow_pickle=False)

    print(f"  读取并预处理 {subject} {state}（通道 {len(channels)} 个，"
          f"{source.stat().st_size / 1e6:.0f} MB）...", flush=True)
    raw = read_mat(config, subject, state)
    indices = channel_indices(config, channels)

    if state == "CS":
        # 原始 [频率, 通道, 时间, 重复] → 统一成 [频率, 重复, 通道, 时间]
        signals = raw[:, indices, :, :].transpose(0, 3, 1, 2)
    else:
        # 原始 [通道, 时间, 试次] → 统一成 [试次, 通道, 时间]
        signals = raw[indices, :, :].transpose(2, 0, 1)
    del raw

    result = preprocess(config, signals)
    cached.parent.mkdir(parents=True, exist_ok=True)
    np.save(cached, result, allow_pickle=False)
    return result


def load_quality_response(config: dict, subject: str) -> np.ndarray:
    """63 通道对 12 Hz 刺激的平均响应，用于信号质量分析。

    预处理是线性的，所以先把 12 次重复平均，再预处理，省内存也省时间。
    返回 [63, 1125]。
    """
    source = mat_path(config, subject, "CS")
    # 用电极表原始顺序（即列号 0..62 的顺序），数组的行就是这个顺序。
    channels = list(channel_table(config))
    cached = cache_file(config, subject, "CS", source, channels, tag="quality")
    if cached.is_file():
        return np.load(cached, allow_pickle=False)

    raw = read_mat(config, subject, "CS")
    twelve_hz_position = config["paradigm"]["frequencies"].index(12)
    response = raw[twelve_hz_position].mean(axis=-1)      # [63, 22500]
    del raw

    result = preprocess(config, response)
    cached.parent.mkdir(parents=True, exist_ok=True)
    np.save(cached, result, allow_pickle=False)
    return result


# ---------------------------------------------------------------------------
# 切窗：CS 按时间截取，NS 切成连续的 1 秒片段
# ---------------------------------------------------------------------------

def crop_cs(config: dict, cs: np.ndarray, seconds: float) -> np.ndarray:
    """从视觉延迟（0.14 s）开始，截取 seconds 秒的 CS 数据。

    为什么从 0.14 s 开始：刺激出现后视觉通路需要约 0.14 s 才把信号送到皮层，
    论文的异步部分明确用了这个延迟，频率识别部分沿用同一假设。
    """
    fs = sampling_rate(config)
    start = round(config["paradigm"]["delay_seconds"] * fs)
    stop = start + round(seconds * fs)
    if seconds <= 0 or seconds > 4 or stop > cs.shape[-1]:
        raise ValueError(f"请求的窗长 {seconds}s 超出可用区间")
    return cs[..., start:stop]


def segment_ns(config: dict, ns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """把 NS 原始试次切成不重叠的 1 秒片段。

    NS1/NS2 每个试次 4 s → 4 段；NS3 每个试次 2 s → 2 段。
    返回值第二项 groups 记录“第 i 段来自第几个原始试次”，
    交叉验证时同一个原始试次的片段必须留在同一折，否则相邻片段泄漏。
    """
    fs = sampling_rate(config)
    n_trials, n_channels, n_samples = ns.shape
    if n_samples % fs:
        raise ValueError("NS 时长必须是整秒")

    n_segments = n_samples // fs
    # [试次, 通道, 段数, 每段点数] → [试次, 段数, 通道, 点数] → [总段数, 通道, 点数]
    segments = ns.reshape(n_trials, n_channels, n_segments, fs)
    segments = segments.transpose(0, 2, 1, 3).reshape(-1, n_channels, fs)

    groups = np.repeat(np.arange(n_trials), n_segments)
    return segments, groups
