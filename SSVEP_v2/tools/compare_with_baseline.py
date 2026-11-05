"""逐条比对“新版结果”和“基线结果”，证明易读版没有改动数值。

为什么需要这个工具：新版代码被重写得很直白，但目标仍是复现同一批结果。
只靠“看起来差不多”不够，必须逐行证明：

    frequency/{Sxx}_predictions.csv        逐试次预测，要求完全相等
    frequency/{Sxx}_metrics.csv            准确率 / ITR 等，浮点允许 1e-12 绝对误差
    asynchronous/{Sxx}_predictions.csv     逐样本预测，要求完全相等
    asynchronous/{Sxx}_metrics.csv         准确率 / 平衡准确率等，浮点同上
    asynchronous/{Sxx}_band_selection.json 选带日志，浮点同上

用法（在项目根目录运行）：

    python tools/compare_with_baseline.py --new results --baseline <基线目录> --subjects S01

不给 --subjects 时，自动比对两边 .csv 文件名里出现过的全部受试者。
退出码：0 = 全部一致；1 = 有差异或有文件缺失。差异最多打印 --max-diffs 条。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 各结果文件的“主键列”和“数值列”。主键用于把两边同一行对齐，
# 数值列用于逐条比较；这份清单就是新旧格式共同遵守的列约定。
FILE_SPECS = [
    dict(
        folder="frequency", suffix="predictions.csv",
        keys=["subject", "window_seconds", "algorithm", "fold", "true_class"],
        values=["predicted_class"],
    ),
    dict(
        folder="frequency", suffix="metrics.csv",
        keys=["subject", "window_seconds", "algorithm"],
        values=["accuracy", "itr_bits_min", "n_trials", "decision_seconds"],
    ),
    dict(
        folder="asynchronous", suffix="predictions.csv",
        keys=["subject", "state", "algorithm", "sample"],
        values=["fold", "original_group", "true_class", "predicted_class"],
    ),
    dict(
        folder="asynchronous", suffix="metrics.csv",
        keys=["subject", "state", "algorithm"],
        values=["accuracy", "balanced_accuracy", "cs_detection_rate",
                "ns_false_positive_rate", "n_classes", "n_samples"],
    ),
]


# ---------------------------------------------------------------------------
# 表格比较
# ---------------------------------------------------------------------------

def _row_label(row: pd.Series, keys: list[str]) -> str:
    """把一行数据的主键拼成便于定位的文字，例如 window_seconds=4, algorithm=FBCCA。"""
    return ", ".join(f"{key}={row[key]}" for key in keys)


def _values_equal(series_new: pd.Series, series_base: pd.Series, tolerance: float) -> np.ndarray:
    """逐元素判断两列是否相等；数值列按绝对误差 tolerance，其余按字面值。"""
    if pd.api.types.is_numeric_dtype(series_new) and pd.api.types.is_numeric_dtype(series_base):
        return np.isclose(series_new.to_numpy(float), series_base.to_numpy(float),
                          rtol=0, atol=tolerance, equal_nan=True)
    new_text = series_new.astype(str).to_numpy()
    base_text = series_base.astype(str).to_numpy()
    return np.array([a == b for a, b in zip(new_text, base_text)], dtype=bool)


def _describe(value) -> str:
    """差异信息里打印数值：浮点保留足够位数，避免看起来相等其实不等。"""
    if isinstance(value, np.integer):
        return repr(int(value))
    if isinstance(value, (float, np.floating)):
        return repr(float(value))
    return repr(value)


def compare_table(path_new: Path, path_base: Path, keys: list[str], values: list[str],
                  tolerance: float, differences: list[str], stats: dict) -> bool:
    """比较一份 CSV 表，返回是否完全一致。

    做法：先检查列约定，再按主键排序后逐行、逐列比较。
    额外保留每行在原文件中的行号，报错时能直接定位到 CSV 的某一行。
    """
    name = f"{path_new.parent.name}/{path_new.name}"
    if not path_new.is_file() or not path_base.is_file():
        differences.append(f"{name}: 文件缺失（新版存在={path_new.is_file()}，"
                           f"基线存在={path_base.is_file()}）")
        return False

    new = pd.read_csv(path_new)
    base = pd.read_csv(path_base)
    expected = list(keys) + list(values)

    # 列多了少了都说明输出格式变了，先记一条差异；后续比较只针对共同列。
    missing_new = [column for column in expected if column not in new.columns]
    missing_base = [column for column in expected if column not in base.columns]
    if missing_new or missing_base:
        differences.append(f"{name}: 缺少列（新版缺 {missing_new}，基线缺 {missing_base}）")
        return False
    extra_new = [column for column in new.columns if column not in expected]
    extra_base = [column for column in base.columns if column not in expected]
    if extra_new or extra_base:
        differences.append(f"{name}: 列集合不同（新版多 {extra_new}，基线多 {extra_base}）")
        return False

    # 行号（含表头，所以 +2）会在排序后跟着行走，用于指出具体是第几行。
    new = new.reset_index(drop=True).assign(_line_number=lambda frame: frame.index + 2)

    if len(new) != len(base):
        differences.append(f"{name}: 行数不同（新版 {len(new)} 行，基线 {len(base)} 行）")
    if new.duplicated(keys).any():
        differences.append(f"{name}: 主键 {keys} 有重复行，无法逐条对齐")
        return False
    if base.duplicated(keys).any():
        differences.append(f"{name}: 基线主键 {keys} 有重复行，无法逐条对齐")
        return False

    new = new.sort_values(keys, kind="stable").reset_index(drop=True)
    base = base.sort_values(keys, kind="stable").reset_index(drop=True)

    matched_rows = min(len(new), len(base))
    for column in values:
        equal = _values_equal(new[column].iloc[:matched_rows], base[column].iloc[:matched_rows], tolerance)
        for position in np.flatnonzero(~equal):
            line_number = int(new.loc[position, "_line_number"])
            differences.append(
                f"{name} 第 {line_number} 行 [{_row_label(new.loc[position], keys)}]: "
                f"{column} 新版={_describe(new.loc[position, column])} "
                f"基线={_describe(base.loc[position, column])}")
    stats["rows"] += matched_rows
    return len(differences) == 0


# ---------------------------------------------------------------------------
# 选带日志比较
# ---------------------------------------------------------------------------

def _is_number(value) -> bool:
    """JSON 里的数字（排除 True/False，它们在 Python 里是 int 的子类）。"""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _compare_json(new_value, base_value, label: str, tolerance: float,
                  differences: list[str]) -> None:
    """递归比较一个 JSON 值；数字按容差，字典按字段，列表按位置。"""
    if _is_number(new_value) and _is_number(base_value):
        if not np.isclose(float(new_value), float(base_value), rtol=0, atol=tolerance):
            differences.append(f"{label}: 新版={_describe(new_value)} 基线={_describe(base_value)}")
        return
    if isinstance(new_value, dict) and isinstance(base_value, dict):
        missing = sorted(set(base_value) - set(new_value))
        extra = sorted(set(new_value) - set(base_value))
        if missing or extra:
            differences.append(f"{label}: 字段不同（新版缺 {missing}，基线缺 {extra}）")
        for key in sorted(set(new_value) & set(base_value)):
            _compare_json(new_value[key], base_value[key], f"{label}.{key}", tolerance, differences)
        return
    if isinstance(new_value, list) and isinstance(base_value, list):
        if len(new_value) != len(base_value):
            differences.append(f"{label}: 列表长度不同（新版 {len(new_value)}，基线 {len(base_value)}）")
        for index in range(min(len(new_value), len(base_value))):
            _compare_json(new_value[index], base_value[index], f"{label}[{index}]", tolerance, differences)
        return
    if new_value != base_value:
        differences.append(f"{label}: 新版={_describe(new_value)} 基线={_describe(base_value)}")


def compare_band_selection(path_new: Path, path_base: Path, tolerance: float,
                           differences: list[str], stats: dict) -> bool:
    """比较选带日志：先按 (state, fold) 对齐，再逐字段比较。"""
    name = f"{path_new.parent.name}/{path_new.name}"
    if not path_new.is_file() or not path_base.is_file():
        differences.append(f"{name}: 文件缺失（新版存在={path_new.is_file()}，"
                           f"基线存在={path_base.is_file()}）")
        return False

    new_entries = json.loads(path_new.read_text(encoding="utf-8"))
    base_entries = json.loads(path_base.read_text(encoding="utf-8"))
    if not isinstance(new_entries, list) or not isinstance(base_entries, list):
        differences.append(f"{name}: 顶层不是列表，格式与约定不符")
        return False

    def key_of(entry):
        return (str(entry.get("state")), int(entry.get("fold", -1)))

    new_by_key = {key_of(entry): entry for entry in new_entries}
    base_by_key = {key_of(entry): entry for entry in base_entries}
    missing = sorted(set(base_by_key) - set(new_by_key))
    extra = sorted(set(new_by_key) - set(base_by_key))
    if missing or extra:
        differences.append(f"{name}: 选带记录缺失或多余（新版缺 {missing}，新版多 {extra}）")
    for key in sorted(set(new_by_key) & set(base_by_key)):
        label = f"{name} {key[0]} fold={key[1]}"
        _compare_json(new_by_key[key], base_by_key[key], label, tolerance, differences)
    stats["rows"] += len(set(new_by_key) & set(base_by_key))
    return len(differences) == 0


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def discover_subjects(new_root: Path, baseline_root: Path) -> list[str]:
    """从两边结果目录的文件名里收集受试者编号，如 S01_metrics.csv → S01。"""
    subjects = set()
    for root in (new_root, baseline_root):
        for folder in ("frequency", "asynchronous"):
            directory = root / folder
            if not directory.is_dir():
                continue
            for path in directory.glob("S*_*.csv"):
                subjects.add(path.name[:3])
    return sorted(subjects)


def compare_one_subject(new_root: Path, baseline_root: Path, subject: str,
                        tolerance: float, differences: list[str], stats: dict) -> None:
    """比对一个人的全部结果文件；所有差异追加到 differences 里。"""
    for spec in FILE_SPECS:
        relative = f"{spec['folder']}/{subject}_{spec['suffix']}"
        compare_table(new_root / relative, baseline_root / relative, spec["keys"], spec["values"],
                      tolerance, differences, stats)
        stats["files"] += 1
    relative = f"asynchronous/{subject}_band_selection.json"
    compare_band_selection(new_root / relative, baseline_root / relative,
                           tolerance, differences, stats)
    stats["files"] += 1


def main(argv: list[str] | None = None) -> int:
    """解析参数、逐个受试者比对、打印差异摘要，返回进程退出码。"""
    parser = argparse.ArgumentParser(description="逐条比对新版 SSVEP 结果与基线结果")
    parser.add_argument("--new", required=True, help="新版结果根目录（含 frequency/ 与 asynchronous/）")
    parser.add_argument("--baseline", required=True, help="基线结果根目录")
    parser.add_argument("--subjects", nargs="+", help="只比对指定受试者，例如 S01 S02；默认自动发现")
    parser.add_argument("--tolerance", type=float, default=1e-12,
                        help="浮点指标的绝对误差上限（默认 1e-12）")
    parser.add_argument("--max-diffs", type=int, default=20, help="最多打印多少条差异（默认 20）")
    arguments = parser.parse_args(argv)

    new_root = Path(arguments.new)
    baseline_root = Path(arguments.baseline)
    if not new_root.is_dir() or not baseline_root.is_dir():
        print(f"目录不存在：新版={new_root} 基线={baseline_root}", file=sys.stderr)
        return 1

    subjects = arguments.subjects or discover_subjects(new_root, baseline_root)
    if not subjects:
        print("在两个目录里都没有找到 Sxx_*.csv，无法比对", file=sys.stderr)
        return 1

    differences: list[str] = []
    stats = dict(files=0, rows=0)
    for subject in subjects:
        compare_one_subject(new_root, baseline_root, subject, arguments.tolerance, differences, stats)

    print(f"比对受试者：{', '.join(subjects)}")
    print(f"比对文件：{stats['files']} 个，数据行：{stats['rows']} 行，容差：{arguments.tolerance:g}")
    if not differences:
        print("结果一致：所有 predictions 逐条相同，metrics 与选带日志均在容差内")
        return 0

    print(f"发现 {len(differences)} 条差异，最多显示 {arguments.max_diffs} 条：")
    for message in differences[:arguments.max_diffs]:
        print("  " + message)
    if len(differences) > arguments.max_diffs:
        print(f"  ... 其余 {len(differences) - arguments.max_diffs} 条差异省略")
    return 1


if __name__ == "__main__":
    sys.exit(main())
