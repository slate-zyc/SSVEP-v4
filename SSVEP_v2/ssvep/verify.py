"""独立核验：不直接相信 metrics 文件，而是从逐试次预测重新算一遍指标。

为什么要有这个文件：实验代码自己写的准确率可能因为笔误、错位、漏样本而失真。
核验只读 predictions / metrics，不重新训练、不改任何数据，逐层检查：

1. 数量与格式：每个人每个任务的预测条数、列名、主键唯一性；
2. 协议约束：频率任务每折每类恰好一条；异步任务同一 NS 原始试次的片段不跨折；
3. 数值一致：用 predictions 重算准确率、平衡准确率、ITR、CS 检出率、误触发率，
   必须和 metrics 文件里记录的值一致（浮点按 1e-12 绝对误差）。

任何一层不通过都立即抛错，不会默默写出“验证通过”。最后把结论写进
output/verification.json，供报告和人工复查引用。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from .algorithms import itr

# 固定协议常量：这些数字来自论文实验设计，写死是有意的——
# 如果输出条数变了，说明协议被改动，必须让人看到，而不是自动适应。
FREQUENCY_ALGORITHMS = ("FBCCA", "TRCA", "eTRCA")
ASYNCHRONOUS_ALGORITHMS = ("FS-eTRCA", "FS-eTRCA&FBCCA", "IFB-OCN-reconstructed")
FREQUENCY_PREDICTIONS_PER_GROUP = 96    # 12 折 × 8 个刺激频率
ASYNCHRONOUS_SAMPLES_PER_GROUP = 192    # 96 个 CS 片段 + 96 个 NS 片段
TOLERANCE = 1e-12

FREQUENCY_KEY_COLUMNS = ["subject", "window_seconds", "algorithm", "fold", "true_class"]
ASYNCHRONOUS_KEY_COLUMNS = ["subject", "state", "algorithm", "sample"]


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------

def require(condition, message: str) -> None:
    """条件不成立就报错；用函数代替 assert，避免 Python -O 模式把检查去掉。"""
    if not condition:
        raise AssertionError(message)


def format_key(key) -> str:
    """把分组主键转成文字，避免错误信息里出现 np.int64 这类内部类型名。"""
    parts = [str(int(value)) if isinstance(value, np.integer) else str(value) for value in key]
    return "(" + ", ".join(parts) + ")"


def format_value(value) -> str:
    """把数值转成便于阅读的文字；浮点用 repr 保留全部有效位数。"""
    if isinstance(value, np.integer):
        return repr(int(value))
    if isinstance(value, np.floating):
        return repr(float(value))
    return repr(value)


def output_dir(config: dict) -> Path:
    """结果根目录（config.json 里 paths.output 的绝对路径）。"""
    return Path(config["paths"]["output"])


def read_subject_tables(output: Path, folder: str, suffix: str, subjects: list[str]) -> pd.DataFrame:
    """把多个人的同一类表拼成一张大表。

    例如 folder="frequency", suffix="predictions" 时读取
    frequency/S01_predictions.csv ... 并上下拼接。
    缺文件时立刻报错，避免拿“部分人”的结果冒充全量。
    """
    paths = [output / folder / f"{subject}_{suffix}.csv" for subject in subjects]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("缺少结果文件：" + ", ".join(missing[:5])
                                + (" ..." if len(missing) > 5 else ""))
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def require_columns(table: pd.DataFrame, columns: list[str], name: str) -> None:
    """检查必需的列都在；缺列说明输出格式变了。"""
    missing = [column for column in columns if column not in table.columns]
    require(not missing, f"{name} 缺列：{missing}")


def require_unique(table: pd.DataFrame, keys: list[str], name: str) -> None:
    """主键重复会让“逐条对齐”失去意义，必须在核验阶段报错。"""
    require(not table.duplicated(keys).any(), f"{name} 的主键 {keys} 有重复行")


def group_keys(table: pd.DataFrame, keys: list[str]) -> list[tuple]:
    """把主键列转成集合，用来检查“预测和 metrics 覆盖的是不是同一批任务”。"""
    return list(table[keys].drop_duplicates().itertuples(index=False, name=None))


def require_same_groups(measured: pd.DataFrame, recorded: pd.DataFrame, label: str) -> None:
    """检查重算表和 metrics 表覆盖的任务完全一致（不多不少）。"""
    measured_keys, recorded_keys = set(measured.index), set(recorded.index)
    missing = sorted(recorded_keys - measured_keys)[:3]
    extra = sorted(measured_keys - recorded_keys)[:3]
    require(not missing and not extra,
            f"{label} 的任务集合不一致（预测里缺 {[format_key(key) for key in missing]}，"
            f"metrics 里多 {[format_key(key) for key in extra]}）")


def require_same_values(measured: pd.DataFrame, recorded: pd.DataFrame, label: str,
                        columns: list[str], tolerance: float = TOLERANCE) -> None:
    """逐列比较重算值与记录值；报错时给出第一个不一致的任务和两边数值。"""
    require_same_groups(measured, recorded, label)
    recorded = recorded.loc[measured.index]
    for column in columns:
        new_values = measured[column].to_numpy(dtype=float)
        old_values = recorded[column].to_numpy(dtype=float)
        close = np.isclose(new_values, old_values, rtol=0, atol=tolerance)
        if not close.all():
            first = np.flatnonzero(~close)[0]
            raise AssertionError(
                f"{label} 的 {column} 不一致：任务 {format_key(measured.index[first])} "
                f"重算={format_value(new_values[first])} 记录={format_value(old_values[first])}")


def require_integer_column(table: pd.DataFrame, column: str, expected: int, label: str) -> None:
    """某一列必须是固定的整数值（比如 n_trials 必须等于 96）。"""
    values = sorted({int(value) for value in table[column]})
    require(values == [expected], f"{label} 的 {column} 应恒为 {expected}，实际为 {values}")


# ---------------------------------------------------------------------------
# 频率识别
# ---------------------------------------------------------------------------

def check_frequency_predictions(predictions: pd.DataFrame, config: dict) -> None:
    """检查频率预测的数量、折号、类别范围和每折结构。

    协议：12 折，每折把 8 个类别的同一次重复留作测试，所以
    每个人 × 每个窗长 × 每个算法 → 12 × 8 = 96 条预测。
    """
    require_columns(predictions, FREQUENCY_KEY_COLUMNS + ["predicted_class"], "频率预测")
    require_unique(predictions, FREQUENCY_KEY_COLUMNS, "频率预测")

    n_classes = len(config["paradigm"]["frequencies"])
    folds = int(config["frequency"]["folds"])
    windows = set(config["frequency"]["windows_seconds"])

    require(set(predictions.algorithm) == set(FREQUENCY_ALGORITHMS),
            f"频率算法名应为 {FREQUENCY_ALGORITHMS}，实际为 {sorted(set(predictions.algorithm))}")
    require(set(predictions.window_seconds) == windows,
            f"频率窗长应为 {sorted(windows)}，实际为 {sorted(set(predictions.window_seconds))}")

    for key, group in predictions.groupby(["subject", "window_seconds", "algorithm"], sort=True):
        require(len(group) == FREQUENCY_PREDICTIONS_PER_GROUP,
                f"频率预测条数不对：任务 {format_key(key)} 有 {len(group)} 条，应为 {FREQUENCY_PREDICTIONS_PER_GROUP}")
        require(sorted(group.fold.unique()) == list(range(1, folds + 1)),
                f"频率折号不对：任务 {format_key(key)} 的折号为 {sorted(group.fold.unique())}")
        # 每折里每个真实类别恰好出现一次，是“按重复编号分折”的直接证据。
        counts = group.groupby(["fold", "true_class"]).size()
        require(len(counts) == folds * n_classes and (counts == 1).all(),
                f"频率每折每类应恰好 1 条：任务 {format_key(key)} 的分折结构不符合协议")
        require(set(group.true_class) == set(range(n_classes)),
                f"频率真实类别应覆盖 0..{n_classes - 1}：任务 {format_key(key)}")
        require(set(group.predicted_class) <= set(range(n_classes)),
                f"频率预测类别越界：任务 {format_key(key)} 出现 {sorted(set(group.predicted_class))}")


def measure_frequency_metrics(predictions: pd.DataFrame, config: dict) -> pd.DataFrame:
    """从逐试次预测重算准确率、ITR、试次数和决策时间。

    决策时间按协议公式算：观察窗 + 提示 1.5 s + 视觉延迟 0.14 s；
    不抄 metrics 文件里的 decision_seconds，这样两边才对得上“独立计算”。
    """
    paradigm = config["paradigm"]
    n_classes = len(paradigm["frequencies"])
    keys = ["subject", "window_seconds", "algorithm"]

    accuracy = (predictions.assign(correct=predictions.true_class == predictions.predicted_class)
                .groupby(keys).correct.mean())
    n_trials = predictions.groupby(keys).size()
    decision = (predictions.groupby(keys).window_seconds.first()
                + float(paradigm["cue_seconds"]) + float(paradigm["delay_seconds"]))

    measured = pd.DataFrame(dict(accuracy=accuracy, n_trials=n_trials, decision_seconds=decision))
    measured["itr_bits_min"] = [itr(row.accuracy, n_classes, row.decision_seconds)
                                for row in measured.itertuples()]
    return measured


def check_frequency_metrics(predictions: pd.DataFrame, metrics: pd.DataFrame, config: dict) -> None:
    """检查 metrics 是否齐全，并把每个数值和重算值对齐。"""
    keys = ["subject", "window_seconds", "algorithm"]
    columns = ["accuracy", "itr_bits_min", "n_trials", "decision_seconds"]
    require_columns(metrics, keys + columns, "频率指标")
    require_unique(metrics, keys, "频率指标")
    require_integer_column(metrics, "n_trials", FREQUENCY_PREDICTIONS_PER_GROUP, "频率指标")

    measured = measure_frequency_metrics(predictions, config)
    recorded = metrics.set_index(keys)
    require_same_values(measured, recorded, "频率指标", columns)


# ---------------------------------------------------------------------------
# 异步分类
# ---------------------------------------------------------------------------

def _segments_per_trial(ns: pd.DataFrame) -> int:
    """每个 NS 原始试次被切成几段：从数据里推出来，而不是写死 NS1/NS2/NS3。

    NS1/NS2 每试次 4 秒 → 4 段；NS3 每试次 2 秒 → 2 段。
    这样即使以后换了数据长度，核验逻辑也不用改。
    """
    sizes = ns.groupby("original_group").size()
    require(sizes.nunique() == 1,
            f"同一状态里各原始试次的片段数应相同，实际为 {sorted(set(sizes))}")
    segments = int(sizes.iloc[0])
    require(segments in (2, 4), f"原始试次片段数应为 2 或 4，实际为 {segments}")
    require(len(sizes) == 96 // segments,
            f"NS 原始试次数应为 {96 // segments}，实际为 {len(sizes)}")
    return segments


def check_asynchronous_predictions(predictions: pd.DataFrame, config: dict) -> None:
    """检查异步预测的数量、标签语义、折号与原始试次分组。

    协议：每次（某状态）共 192 个样本 = 96 个 CS 片段（8 频率 × 12 重复）
    + 96 个 NS 片段；12 折，每折 8 个 CS + 8 个 NS。
    九分类用 0..7 表示频率、8 表示 NS；IFB-OCN 二分类用 1=CS、0=NS。
    """
    require_columns(predictions, ASYNCHRONOUS_KEY_COLUMNS + ["fold", "original_group",
                                                             "true_class", "predicted_class"],
                    "异步预测")
    require_unique(predictions, ASYNCHRONOUS_KEY_COLUMNS, "异步预测")

    n_classes = len(config["paradigm"]["frequencies"])
    folds = int(config["asynchronous"]["folds"])
    states = set(predictions.state)
    require(states == {"NS1", "NS2", "NS3"}, f"异步状态应为 NS1/NS2/NS3，实际为 {sorted(states)}")
    require(set(predictions.algorithm) == set(ASYNCHRONOUS_ALGORITHMS),
            f"异步算法名应为 {ASYNCHRONOUS_ALGORITHMS}，实际为 {sorted(set(predictions.algorithm))}")

    for key, group in predictions.groupby(["subject", "state", "algorithm"], sort=True):
        subject, state, algorithm = key
        require(len(group) == ASYNCHRONOUS_SAMPLES_PER_GROUP,
                f"异步预测条数不对：任务 {format_key(key)} 有 {len(group)} 条，应为 {ASYNCHRONOUS_SAMPLES_PER_GROUP}")
        group = group.sort_values("sample")
        require(np.array_equal(group["sample"].to_numpy(), np.arange(ASYNCHRONOUS_SAMPLES_PER_GROUP)),
                f"异步样本编号应恰为 0..191：任务 {format_key(key)}")

        # 前 96 个样本是 CS，后 96 个是 NS；这个顺序是输出格式的一部分。
        is_cs = group["sample"].to_numpy() < 96
        sample = group["sample"].to_numpy()

        if algorithm.startswith("IFB"):
            # 二分类：CS=1、NS=0，与九分类的 0..8 语义不同，不能混用。
            require(set(group.true_class) == {0, 1},
                    f"IFB 真实标签应为 0/1：任务 {format_key(key)} 出现 {sorted(set(group.true_class))}")
            require(np.array_equal(group.true_class.to_numpy(), is_cs.astype(int)),
                    f"IFB 真实标签与样本位置不符：任务 {format_key(key)}")
            require(set(group.predicted_class) <= {0, 1},
                    f"IFB 预测标签越界：任务 {format_key(key)}")
        else:
            expected_truth = np.where(is_cs, sample // 12, n_classes)
            require(np.array_equal(group.true_class.to_numpy(), expected_truth),
                    f"九分类真实标签与样本位置不符：任务 {format_key(key)}")
            require(set(group.predicted_class) <= set(range(n_classes + 1)),
                    f"九分类预测标签越界：任务 {format_key(key)}")
            cs_counts = group[is_cs].groupby("true_class").size()
            require(len(cs_counts) == n_classes and (cs_counts == 12).all(),
                    f"每个频率应有 12 个 CS 样本：任务 {format_key(key)}")

        # 折号：每折 8 个 CS + 8 个 NS，且 CS 每折覆盖全部类别。
        for fold, fold_group in group.groupby("fold"):
            fold_cs = fold_group[fold_group["sample"] < 96]
            fold_ns = fold_group[fold_group["sample"] >= 96]
            require(len(fold_cs) == 8 and len(fold_ns) == 8,
                    f"每折应有 8 CS + 8 NS：任务 {format_key(key)} 第 {fold} 折为 "
                    f"{len(fold_cs)} CS + {len(fold_ns)} NS")
            if not algorithm.startswith("IFB"):
                require(set(fold_cs.true_class) == set(range(n_classes)),
                        f"每折 CS 应覆盖全部 {n_classes} 个频率：任务 {format_key(key)} 第 {fold} 折")
        require(sorted(group.fold.unique()) == list(range(1, folds + 1)),
                f"异步折号不对：任务 {format_key(key)} 的折号为 {sorted(group.fold.unique())}")

        # 分组约束：同一个 NS 原始试次的所有 1 秒片段必须留在同一折，
        # 否则相邻片段会同时出现在训练集和测试集，造成信息泄漏。
        ns = group[~is_cs]
        require((ns.groupby("original_group").fold.nunique() == 1).all(),
                f"同一 NS 原始试次的片段跨折了：任务 {format_key(key)}")
        segments = _segments_per_trial(ns)
        require((ns.groupby(["fold", "original_group"]).size() == segments).all(),
                f"NS 分折应按原始试次整组进行：任务 {format_key(key)}")
        require((ns.groupby("fold").original_group.nunique() == 8 // segments).all(),
                f"每折 NS 原始试次数应为 {8 // segments}：任务 {format_key(key)}")


def measure_asynchronous_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """从逐样本预测重算异步指标。

    - accuracy：只看整体对错；
    - balanced_accuracy：九分类里 NS 有 96 个样本、每个频率只有 12 个，
      普通准确率会被 NS 主导，所以还要看每类召回率的平均；
    - cs_detection_rate：真实为 CS 的样本里被判成“有刺激”的比例；
    - ns_false_positive_rate：真实为 NS 的样本里被误判成“有刺激”的比例。
    """
    keys = ["subject", "state", "algorithm"]
    rows = []
    for key, group in predictions.groupby(keys, sort=True):
        algorithm = key[2]
        is_cs = group["sample"].to_numpy() < 96
        truth = group.true_class.to_numpy()
        predicted = group.predicted_class.to_numpy()
        binary = algorithm.startswith("IFB")
        # 判成“有刺激”：九分类是“不是 NS=8”，二分类是“预测为 1”。
        control = predicted == 1 if binary else predicted != 8
        rows.append(dict(
            subject=key[0], state=key[1], algorithm=algorithm,
            accuracy=float((truth == predicted).mean()),
            balanced_accuracy=float(balanced_accuracy_score(truth, predicted)),
            cs_detection_rate=float(control[is_cs].mean()),
            ns_false_positive_rate=float(control[~is_cs].mean()),
            n_classes=2 if binary else 9,
            n_samples=len(group),
        ))
    return pd.DataFrame(rows).set_index(keys)


def check_asynchronous_metrics(predictions: pd.DataFrame, metrics: pd.DataFrame) -> None:
    """检查异步 metrics 是否齐全，并把每个数值和重算值对齐。"""
    keys = ["subject", "state", "algorithm"]
    float_columns = ["accuracy", "balanced_accuracy", "cs_detection_rate", "ns_false_positive_rate"]
    require_columns(metrics, keys + float_columns + ["n_classes", "n_samples"], "异步指标")
    require_unique(metrics, keys, "异步指标")

    measured = measure_asynchronous_metrics(predictions)
    recorded = metrics.set_index(keys)
    require_same_values(measured, recorded, "异步指标", float_columns)
    require_integer_column(metrics, "n_samples", ASYNCHRONOUS_SAMPLES_PER_GROUP, "异步指标")
    require(set(metrics.n_classes) == {2, 9}, f"异步 n_classes 应为 2/9，实际为 {sorted(set(metrics.n_classes))}")


# ---------------------------------------------------------------------------
# 选带日志
# ---------------------------------------------------------------------------

def check_band_selection(output: Path, subjects: list[str], config: dict) -> None:
    """检查每个人异步阶段写出的选带日志。

    日志应覆盖 3 种状态 × 12 折 = 36 条；每条记录里选中的频带数量应等于
    config 里的 top_bands（本实验为 3），并且只引用合法的频带编号。
    """
    folds = int(config["asynchronous"]["folds"])
    bands = config["ocn"]["bands"]
    top_bands = int(config["ocn"]["top_bands"])
    for subject in subjects:
        path = output / "asynchronous" / f"{subject}_band_selection.json"
        require(path.is_file(), f"缺少选带日志：{path}")
        entries = json.loads(path.read_text(encoding="utf-8"))
        require(isinstance(entries, list), f"{path} 顶层应是列表")
        require(len(entries) == 3 * folds,
                f"{path} 应有 3 状态 × {folds} 折 = {3 * folds} 条，实际 {len(entries)} 条")
        seen = set()
        for entry in entries:
            key = (entry.get("state"), int(entry.get("fold", -1)))
            require(key not in seen, f"{path} 里 {key} 重复出现")
            seen.add(key)
            selected = entry.get("selected")
            require(isinstance(selected, list) and len(selected) == top_bands,
                    f"{path} {key} 应选中 {top_bands} 个频带，实际 {selected!r}")
            for choice in selected:
                require(0 <= int(choice["band_index"]) < len(bands),
                        f"{path} {key} 的频带编号 {choice['band_index']} 越界")
                require(0.0 <= float(choice["inner_accuracy"]) <= 1.0,
                        f"{path} {key} 的内层准确率 {choice['inner_accuracy']!r} 不在 0..1")


# ---------------------------------------------------------------------------
# 总入口
# ---------------------------------------------------------------------------

def verify_all(config: dict, subjects: list[str], include_frequency: bool = True,
               include_async: bool = True) -> dict:
    """核验指定受试者的全部结果，写出 output/verification.json 并返回结论字典。

    参数
    ----
    config : 由 ssvep.config.load_config() 读取的配置（结果目录用 paths.output）。
    subjects : 要核验的受试者，例如 ["S01"]；必须与已完成的结果一一对应。
    include_frequency / include_async : 只核验已完成的那部分任务，方便分阶段测试。

    返回
    ----
    dict : 直接写入 verification.json 的核验结论（数量 + 每项检查的 passed/skipped）。
    """
    require(subjects, "受试者列表不能为空")
    output = output_dir(config)
    result = dict(checked_subjects=list(subjects), subject_count=len(subjects),
                  tolerance=TOLERANCE)
    if include_frequency:
        predictions = read_subject_tables(output, "frequency", "predictions", subjects)
        metrics = read_subject_tables(output, "frequency", "metrics", subjects)
        check_frequency_predictions(predictions, config)
        check_frequency_metrics(predictions, metrics, config)
        result.update(
            frequency_predictions=int(len(predictions)),
            frequency_metrics=int(len(metrics)),
            frequency_prediction_counts="passed",
            frequency_protocol="passed",
            frequency_metric_recalculation="passed",
        )
    else:
        result["frequency_prediction_counts"] = "skipped"
    if include_async:
        predictions = read_subject_tables(output, "asynchronous", "predictions", subjects)
        metrics = read_subject_tables(output, "asynchronous", "metrics", subjects)
        check_asynchronous_predictions(predictions, config)
        check_asynchronous_metrics(predictions, metrics)
        check_band_selection(output, subjects, config)
        result.update(
            asynchronous_predictions=int(len(predictions)),
            asynchronous_metrics=int(len(metrics)),
            asynchronous_prediction_counts="passed",
            asynchronous_protocol="passed",
            raw_trial_grouping="passed",
            asynchronous_metric_recalculation="passed",
            band_selection="passed",
        )
    else:
        result["asynchronous_prediction_counts"] = "skipped"

    # 明确写出来：本复现不是论文的逐参数精确复刻，尤其是 IFB-OCN。
    result["exact_paper_reproduction"] = False
    result["limitation"] = ("IFB-OCN 是按已公开信息重建的版本；"
                            "二分类与九分类准确率不可直接比较")
    text = json.dumps(result, indent=2, ensure_ascii=False)
    (output / "verification.json").write_text(text + "\n", encoding="utf-8")
    print(text, flush=True)
    return result
