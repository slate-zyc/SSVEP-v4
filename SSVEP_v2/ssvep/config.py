"""读取并解析 config.json。

整个项目只有一个参数文件；这里只做三件事：
1. 读出 JSON；
2. 把 data / cache / output 三个相对路径换算成绝对路径；
3. 提供几个常用参数的小函数，避免每个模块都写 config["..."] 的长串。
"""

from __future__ import annotations

import json
from pathlib import Path

# ROOT 是 SSVEP_v2/ 目录；数据位于上一级的 EEG_data/。
ROOT = Path(__file__).resolve().parent.parent


def load_config(path: Path | str | None = None) -> dict:
    """读取配置，并把三个目录解析为绝对路径。

    参数
    ----
    path : 可选，指定别的配置文件；默认用项目根目录的 config.json。

    返回
    ----
    dict : 与 config.json 结构相同的字典，其中 config["paths"] 下的值是绝对路径。
    """
    config_path = Path(path) if path else ROOT / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    # 相对路径都相对“项目根目录”（也就是 config.json 所在目录），
    # 这样无论从哪个目录启动 Python 都能找到数据。
    for name, value in config["paths"].items():
        config["paths"][name] = str((ROOT / value).resolve())
    return config


def sampling_rate(config: dict) -> int:
    """降采样后的采样率（Hz）。"""
    return int(config["eeg"]["fs"])


def n_frequencies(config: dict) -> int:
    """刺激频率个数，也就是频率识别任务的类别数（8）。"""
    return len(config["paradigm"]["frequencies"])


def n_bands(config: dict) -> int:
    """滤波器组的子带个数（4）。"""
    return len(config["filter_bank"]["passbands"])
