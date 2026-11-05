"""
读取和核验 YSU 数据，完成通道选择、预处理、缓存和状态切窗。

CS 返回 [类别, 原始试次, 通道, 时间]，NS 返回 [原始试次, 通道, 时间]。
eg (类别=8, 试次=12, 通道=9, 时间=22500)    (试次=24, 通道=9, 时间=20000)

所有切分保留原始试次编号，供后续分组交叉验证，避免相邻片段跨训练/测试折。
"""

import hashlib                  # 计算哈希
import json                     # 读写 JSON 格式文本
from pathlib import Path        # 面向对象的文件路径

import numpy as np
import pandas as pd             # 这里仅用于读取通道表 excel
from scipy.io import loadmat, whosmat       # 读写各种科学数据格式，本文件用于读取 MATLAB 格式数据
from scipy.signal import filtfilt, iirnotch, resample_poly      # 信号处理，零相位滤波、陷波滤波、抗混叠降采样


ROOT = Path(__file__).resolve().parent      # 取得"当前这个 .py 文件所在的文件夹"
# __file__：Python 的内置变量，值是当前 data_io.py 文件的路径字符串。
# Path(...)：把路径字符串包装成 Path 对象，获得各种路径操作能力。
# .resolve()：根据脚本的实际位置得到规范化的绝对路径，不绑定盘符或启动目录。
# .parent：取脚本所在的 SSVEP_v1 目录；配置中的 ../EEG_data 指向大项目共享数据。

def read_config(path=None):
    """读取实验配置，三个目录始终以项目根目录为基准。"""
    config_path = Path(path) if path else ROOT / "config.json"      # Path 对象重载了 / 运算符，用除号拼路径，pathlib 的招牌写法
    config = json.loads(config_path.read_text(encoding="utf-8"))
    # 一次性读出整个文件的全部文本，返回一个字符串
    # json.loads(文本)：把 JSON 格式的文本解析成 Python 字典
    for name in ("data_dir", "output_dir", "cache_dir"):
        config[name] = str((ROOT / config[name]).resolve())
    return config       # 返回字典，记录绝对路径


def channel_table(config):
    """校验官方通道表，将 MATLAB 一基编号转换为 Python 零基索引。"""
    table = pd.read_excel(Path(config["data_dir"]) / "Channel Loc.xlsx")
    table = table.iloc[:, :2].dropna()      # 取子表 + 删除含空值的行
    indices = pd.to_numeric(table.iloc[:, 0], errors="raise").astype(int)
    # 强制转为数字类型，errors="raise" 表示“遇到转不了的值就报错” + 数字类型统一为整数类型
    names = table.iloc[:, 1].astype(str).str.strip()    # .str.strip() 去掉每个名称首尾的空格
    if len(table) != 63 or sorted(indices) != list(range(1, 64)):
        raise ValueError("Channel table must contain each one-based index 1..63 exactly once")
    mapping = dict(zip(names.str.casefold(), indices - 1))
    # 全部名称转小写（casefold 比 lower 更彻底，处理特殊字符）
    if len(mapping) != 63:
        raise ValueError("Duplicate channel names")
    return mapping      # 字典 {通道名：从0开始的索引}


def channel_indices(config, names=None):
    """按传入的电极顺序返回索引，忽略大小写但不猜测电极位置。"""
    mapping = channel_table(config)
    return [int(mapping[name.casefold()]) for name in (names or config["channels"])]


def mat_path(config, subject, state):
    """构建单名受试者某种状态的原始文件路径。"""
    path = Path(config["data_dir"]) / subject / f"sub_{subject[1:]}_data_{state}.mat"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path

# 不重要，防御性编程
def audit(config, subjects):
    """读取 MAT 变量元数据并核对论文维度；这不是逐字节完整性校验。"""
    rows = []
    for subject in subjects:
        for state in ("CS", "NS1", "NS2", "NS3"):
            path = mat_path(config, subject, state)
            variables = whosmat(path)   # 只读 MAT 文件的变量目录，不加载数据本体。返回一个列表，每个元素是一个三元组，格式为 (变量名, 维度, 数据类型)
            expected = "data_" + state
            selected = [item for item in variables if item[0] == expected]
            if len(selected) != 1:
                raise ValueError(f"{path}: expected {expected}, found {variables}")
            variable, shape, dtype = selected[0]
            expected_shape = (8, 63, 22500, 12) if state == "CS" else (
                63, 10000 if state == "NS3" else 20000, 48 if state == "NS3" else 24
            )
            if shape != expected_shape:
                raise ValueError(f"Unexpected shape in {path}: {shape}, expected {expected_shape}")
            rows.append(dict(subject=subject, state=state, variable=variable,
                             shape=str(shape), dtype=dtype, bytes=path.stat().st_size))
    channel_indices(config)
    return pd.DataFrame(rows)

# 预处理很慢（一个 MAT 几百 MB，滤波降采样要几十秒），所以算完存成 .npy 缓存下次直接用。但缓存有个致命问题：过期。
# 如果你改了采样率参数、换了通道组合、甚至原始数据被重新下载过，旧缓存就是错的，而程序可能傻乎乎地继续用，结果全错还不报错。
# 解决办法：给每组缓存生成一个由所有影响因素决定的文件名。因素一变，文件名就变，旧缓存自然“找不到”，从而被迫重算。这就是“缓存键（cache key）”。
def cache_key(config, path, names=None):
    """由源文件属性、通道表和预处理参数生成缓存键，防止使用过期缓存。"""
    settings = {key: config[key] for key in ("original_fs", "fs", "notch_q")}
    settings["channels"] = names or config["channels"]
    settings["source"] = [str(path), path.stat().st_size, path.stat().st_mtime_ns]
    table = Path(config["data_dir"]) / "Channel Loc.xlsx"
    settings["channel_table_sha256"] = hashlib.sha256(table.read_bytes()).hexdigest()
    settings["preprocessing_version"] = 1
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:16]

# todo 三种预处理
def preprocess(config, selected):
    """沿最后一维去均值、抗混叠降采样和零相位陷波，不改变输入数组。"""
    selected = np.array(selected, dtype=np.float64, copy=True)
    if not np.isfinite(selected).all():     # 保证都是有限值
        raise ValueError("Nonfinite EEG samples")
    # 去均值
    # 原始文件没有刺激前基线，因此这里采用片段去均值，而非刺激前校正。
    selected -= selected.mean(axis=-1, keepdims=True)   # axis 指定计算维度，keepdims=True 表示“保留计算均值的维度，但长度变成 1”
    # 保留最后的维度是为了利用广播机制

    # 降采样
    if config["original_fs"] % config["fs"]:    # 降采样要是倍数关系
        raise ValueError("Original sampling rate must be divisible by target rate")
    # 采样定理（奈奎斯特-香农定理）：采样率只要大于信号最高频率的 2 倍，就能完整还原信号。250 Hz 对应可表示 125 Hz 以下的成分，远超需要。
    processed = resample_poly(selected, 1, config["original_fs"] // config["fs"], axis=-1)
    # resample_poly(x, up, down, axis=-1)：多相重采样。数学含义是“先把信号上采样 up 倍，再下采样 down 倍”，净效果是采样率乘以 up/down。
    # 这里 up=1, down=20，即 5000 × (1/20) = 250。
    '''
    为什么不用“每 20 个点取 1 个”这种简单做法？ 那会引发混叠（aliasing）。
    举个直观例子：电影里车轮高速转动时看起来在倒转——这就是时间采样太慢导致的频率错觉。
    信号里也一样：如果原始信号里有 200 Hz 的成分，每 20 个点抽 1 个（等效 250 Hz 采样），
    这个 200 Hz 会被伪装成 50 Hz 出现在结果里，而 50 Hz 正好是市电频率，你会误以为是工频干扰，实际上是被折叠的假信号。
    resample_poly 内部会先做一个抗混叠低通滤波器（把高于新采样率一半的频率先砍掉），再抽点。
    '''

    # 50 Hz 陷波
    numerator, denominator = iirnotch(50, config["notch_q"], fs=config["fs"])   # 可以认为返回的是滤波器

    return filtfilt(numerator, denominator, processed, axis=-1)     # 零相位滤波


def load_preprocessed(config, subject, state, names=None):
    """优先读取缓存，否则每次只加载一名受试者的一种状态，减少峰值内存。"""
    path = mat_path(config, subject, state)
    destination = Path(config["cache_dir"]) / f"{subject}_{state}_{cache_key(config, path, names)}.npy"
    if destination.is_file():
        return np.load(destination, allow_pickle=False)
    print(f"Loading {subject} {state}: {path.stat().st_size / 1e6:.1f} MB", flush=True)
    raw = loadmat(path, variable_names=["data_" + state])["data_" + state]
    selection = channel_indices(config, names)
    # 转置统一最后一维为时间，后续滤波函数便可复用。
    if state == "CS":
        selected = raw[:, selection, :, :].transpose(0, 3, 1, 2).copy()     # (8, 9, 22500, 12) -> (0, 12, 9, 22500)
    else:
        selected = raw[selection, :, :].transpose(2, 0, 1).copy()   # (9, 20000, 24) -> (24, 9, 20000)
    del raw     # 删除变量引用，显式释放内存
    processed = preprocess(config, selected)    # 使用前面预处理函数
    del selected
    destination.parent.mkdir(parents=True, exist_ok=True)   # 建目录，已存在不报错
    np.save(destination, processed, allow_pickle=False)     # 存储预处理后的数据：路径，数组， .npy后缀不重复
    return processed    # 返回预处理后的数据


def crop_cs(config, data, seconds):
    """从配置中的视觉延迟起截取 CS 窗，禁止静默补零或越过可用片段。"""
    start = round(config["delay_seconds"] * config["fs"])   # 去除视觉延迟部分，0.14 秒 × 250 点/秒 = 35 个采样点
    stop = start + round(seconds * config["fs"])        # eg 要 1 秒：stop = 35 + 250 = 285
    if stop > data.shape[-1] or seconds > 4 or seconds <= 0:
        raise ValueError("Requested CS window exceeds the response interval")
    return data[..., start:stop]    # 等价于 data[:, :, :, start:stop]   取下标 35 到 284，共 250 个点 = 恰好 1 秒


def segment_ns(config, data):
    """NS 原始试次切成不重叠 1 秒窗，并返回每窗所属的原始试次编号。"""
    trial_count, channel_count, samples = data.shape    # 降采样后 data.shape: (24, 9, 1000)
    if samples % config["fs"]:
        raise ValueError("NS duration must contain an integer number of 1 s windows")
    segment_count = samples // config["fs"]  # 4 秒分 4 段
    segments = data.reshape(trial_count, channel_count, segment_count, config["fs"])
    # 原:  (24 试次, 9 通道, 1000 时间点)
    # 新:  (24 试次, 9 通道, 4 段, 250 点)
    segments = segments.transpose(0, 2, 1, 3).reshape(-1, channel_count, config["fs"])      # (96 段, 9 通道, 250 点)
    # 同一原始试次的所有片段必须留在同一折，而不能按片段随机拆分。
    groups = np.repeat(np.arange(trial_count), segment_count)
    # np.arange(24)：生成 [0, 1, 2, ..., 23]（numpy 版的 range）。
    # np.repeat(数组, 次数)：把每个元素连续重复指定次数：[0,0,0,0, 1,1,1,1, 2,2,2,2, ...]。
    # 合起来 groups 长度 96，第 i 个元素 = “第 i 段来自哪个原始试次”。
    return segments, groups

# 用于信号质量分析
# todo 没有做 7-70 HZ 滤波吗
def quality_response(config, subject):
    """读取 12 Hz 目标的 63 通道平均响应，用于频谱和标准坐标地形图。"""
    path = mat_path(config, subject, "CS")
    names = list(channel_table(config))     # list(字典) 是 Python 的一个特性：对字典取 list 得到它的所有键
    destination = Path(config["cache_dir"]) / f"{subject}_quality_{cache_key(config, path, names)}.npy"
    # # 使用缓存数据
    if destination.is_file():
        return np.load(destination, allow_pickle=False)
    raw = loadmat(path, variable_names=["data_CS"])["data_CS"]
    # 预处理是线性运算，所以先在试次轴平均可显著减少内存与滤波计算。
    mean = raw[config["frequencies"].index(12)].mean(axis=-1)   # (63, 22500, 12) → (63, 22500) # 每个实验对象12次取平均
    del raw
    result = preprocess(config, mean)   # 预处理
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.save(destination, result, allow_pickle=False)
    return result
