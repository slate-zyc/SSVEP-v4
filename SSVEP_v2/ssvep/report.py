"""把每个人的结果汇总成论文图表和一份可读的报告。

这个文件只做“汇总与画图”，不重新训练任何模型：

    读取 results/frequency/ 与 results/asynchronous/ 下每个人的 CSV
    → 独立核验（复用 ssvep.verify 的检查，确认数字没有被抄错）
    → 合并成全体指标表、跨人汇总表、与原文均值对照表
    → 画 figure5 / figure6 / frequency_confusion / figure7
    → 写 REPORT.md 与 validation.txt

文件名和列名都刻意与基线 results_full 保持一致，方便直接对照。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")                       # 无窗口运行，只把图写成文件
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from .verify import (
    FREQUENCY_ALGORITHMS,
    check_asynchronous_metrics,
    check_asynchronous_predictions,
    check_frequency_metrics,
    check_frequency_predictions,
    output_dir,
    read_subject_tables,
)

# 原文报告的均值，用来算“复现值 - 原文值”的百分点差。
PAPER_FREQUENCY_ACCURACY = {"FBCCA": 0.9349, "TRCA": 0.7300, "eTRCA": 0.7947}
PAPER_ASYNCHRONOUS_ACCURACY = {
    "IFB-OCN-reconstructed": [0.7051, 0.7235, 0.6934],
    "FS-eTRCA": [0.6764, 0.6647, 0.6534],
    "FS-eTRCA&FBCCA": [0.7639, 0.7535, 0.7502],
}
STATES = ("NS1", "NS2", "NS3")


# ---------------------------------------------------------------------------
# 读取 + 核验
# ---------------------------------------------------------------------------

def frequency_tables(config: dict, subjects: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """读频率任务的 metrics 和 predictions，并在汇总前先核验一遍。

    为什么先核验：报告是给人看的，如果输入本身有重复、漏样本或指标抄错，
    后面的均值、图和结论都会跟着错，所以宁可在这里直接报错。
    """
    output = output_dir(config)
    predictions = read_subject_tables(output, "frequency", "predictions", subjects)
    metrics = read_subject_tables(output, "frequency", "metrics", subjects)
    check_frequency_predictions(predictions, config)
    check_frequency_metrics(predictions, metrics, config)
    return metrics, predictions


def asynchronous_tables(config: dict, subjects: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """读异步任务的 metrics 和 predictions，并在汇总前先核验一遍。"""
    output = output_dir(config)
    predictions = read_subject_tables(output, "asynchronous", "predictions", subjects)
    metrics = read_subject_tables(output, "asynchronous", "metrics", subjects)
    check_asynchronous_predictions(predictions, config)
    check_asynchronous_metrics(predictions, metrics)
    return metrics, predictions


# ---------------------------------------------------------------------------
# 汇总表
# ---------------------------------------------------------------------------

def summarize_frequency(metrics: pd.DataFrame) -> pd.DataFrame:
    """跨人汇总频率任务：每个算法 × 每个窗长的准确率和 ITR 均值 ± 标准差。"""
    return metrics.groupby(["algorithm", "window_seconds"], as_index=False).agg(
        subjects=("subject", "nunique"),
        mean_accuracy=("accuracy", "mean"), sd_accuracy=("accuracy", "std"),
        mean_itr=("itr_bits_min", "mean"), sd_itr=("itr_bits_min", "std"),
    )


def summarize_asynchronous(metrics: pd.DataFrame) -> pd.DataFrame:
    """跨人汇总异步任务：三种 NS 状态 × 三种算法的各项指标均值。

    九分类和 IFB-OCN 二分类都放在同一张表里，但用 n_classes 列明确区分，
    避免读者把二分类准确率直接和九分类准确率排名。
    """
    return metrics.groupby(["state", "algorithm", "n_classes"], as_index=False).agg(
        subjects=("subject", "nunique"),
        mean_accuracy=("accuracy", "mean"), sd_accuracy=("accuracy", "std"),
        mean_balanced_accuracy=("balanced_accuracy", "mean"),
        mean_cs_detection_rate=("cs_detection_rate", "mean"),
        mean_ns_false_positive_rate=("ns_false_positive_rate", "mean"),
    )


def write_frequency_paper_comparison(summary: pd.DataFrame, output: Path) -> pd.DataFrame:
    """取最长窗长（本实验为 4 s）的结果，和原文均值算百分点差。"""
    longest = summary.window_seconds.max()
    comparison = summary[summary.window_seconds == longest].copy()
    comparison["paper_accuracy"] = comparison.algorithm.map(PAPER_FREQUENCY_ACCURACY)
    comparison["difference_percentage_points"] = (
        comparison.mean_accuracy - comparison.paper_accuracy) * 100
    comparison.to_csv(output / "frequency_paper_comparison.csv", index=False)
    return comparison


def write_asynchronous_paper_comparison(summary: pd.DataFrame, output: Path) -> pd.DataFrame:
    """把三种状态的结果和原文均值对照，并明确标注“不是精确方法匹配”。"""
    comparison = summary.copy()
    comparison["paper_accuracy"] = [
        PAPER_ASYNCHRONOUS_ACCURACY[row.algorithm][int(row.state[-1]) - 1]
        for row in comparison.itertuples()]
    comparison["difference_percentage_points"] = (
        comparison.mean_accuracy - comparison.paper_accuracy) * 100
    comparison["exact_method_match"] = False      # IFB-OCN 是重建版，不能宣称精确匹配
    comparison.to_csv(output / "asynchronous_paper_comparison.csv", index=False)
    return comparison


# ---------------------------------------------------------------------------
# 画图
# ---------------------------------------------------------------------------

def plot_frequency_curves(metrics: pd.DataFrame, output: Path) -> None:
    """画 figure5_accuracy.png / figure6_itr.png。

    每张图左上角是“跨人均值 ± 标准差随窗长变化”的折线，
    其余三格是每个算法自己的逐人箱线图；这样既能看总体趋势，也能看个体差异。
    """
    windows = sorted(metrics.window_seconds.unique())
    for metric, filename, ylabel, multiplier in (
            ("accuracy", "figure5_accuracy.png", "Accuracy (%)", 100),
            ("itr_bits_min", "figure6_itr.png", "ITR (bits/min)", 1)):
        figure, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
        for algorithm in FREQUENCY_ALGORITHMS:
            values = metrics[metrics.algorithm == algorithm].groupby("window_seconds")[metric]
            mean = values.mean().reindex(windows)
            # 只有 1 个人时标准差无定义（NaN）；画图时当 0 处理，避免曲线消失。
            spread = values.std().reindex(windows).fillna(0.0)
            axes[0, 0].errorbar(mean.index, mean.values * multiplier,
                                yerr=spread.values * multiplier,
                                marker="o", capsize=3, label=algorithm)
        axes[0, 0].set(title=f"Mean +/- between-subject SD; n={metrics.subject.nunique()}",
                       xlabel="Window (s)", ylabel=ylabel, xticks=windows)
        axes[0, 0].legend()
        for axis, algorithm in zip(axes.flat[1:], FREQUENCY_ALGORITHMS):
            per_subject = [
                metrics[(metrics.algorithm == algorithm)
                        & (metrics.window_seconds == window)][metric].values * multiplier
                for window in windows]
            axis.boxplot(per_subject, tick_labels=[f"{window:g}" for window in windows])
            axis.set(title=algorithm, xlabel="Window (s)", ylabel=ylabel)
        figure.savefig(output / filename, dpi=180)
        plt.close(figure)


def plot_frequency_confusion(config: dict, metrics: pd.DataFrame,
                             predictions: pd.DataFrame, output: Path) -> None:
    """画 frequency_confusion.png：最长窗长下三个算法的八分类混淆矩阵。

    混淆矩阵能看出错法：如果错误集中在相邻频率，通常是频率分辨率问题；
    如果整行偏移，则更可能是标签错位（S11 的疑似问题就是这一类）。
    """
    frequencies = config["paradigm"]["frequencies"]
    longest = metrics.window_seconds.max()
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for axis, algorithm in zip(axes, FREQUENCY_ALGORITHMS):
        selected = predictions[(predictions.algorithm == algorithm)
                               & (predictions.window_seconds == longest)]
        counts = confusion_matrix(selected.true_class, selected.predicted_class,
                                  labels=np.arange(len(frequencies)))
        axis.imshow(counts, cmap="Blues")
        axis.set(title=f"{algorithm}, {longest:g} s",
                 xlabel="Predicted frequency (Hz)", ylabel="True frequency (Hz)",
                 xticks=range(len(frequencies)), yticks=range(len(frequencies)),
                 xticklabels=frequencies, yticklabels=frequencies)
        for (row, column), value in np.ndenumerate(counts):
            axis.text(column, row, str(value), ha="center", va="center", fontsize=7,
                      color="white" if value > counts.max() / 2 else "black")
    figure.savefig(output / "frequency_confusion.png", dpi=180)
    plt.close(figure)


def plot_asynchronous(metrics: pd.DataFrame, output: Path) -> None:
    """画 figure7_asynchronous.png：三种 NS 状态下的准确率对比。

    图例和标题都写明“二分类 / 九分类”，因为这两类任务的难度和机会水平不同，
    放在同一张图里只是为了排版，绝不代表可以直接比较排名。
    """
    figure_order = ["IFB-OCN-reconstructed", "FS-eTRCA", "FS-eTRCA&FBCCA"]
    labels = ["IFB-OCN rebuilt\n2 classes", "FS-eTRCA\n9 classes", "FS-eTRCA+FBCCA\n9 classes"]
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    for index, algorithm in enumerate(figure_order):
        values = metrics[metrics.algorithm == algorithm].groupby("state").accuracy
        mean = values.mean().reindex(STATES)
        # 单个人时 SD 为 NaN，按 0 处理，否则柱状图会画不出来。
        spread = values.std().reindex(STATES).fillna(0.0)
        axes[0, 0].bar(np.arange(3) + (index - 1) * 0.25, mean.values * 100, width=0.25,
                       yerr=spread.values * 100, capsize=3, label=labels[index])
    axes[0, 0].set(xticks=range(3), xticklabels=STATES, ylabel="Accuracy (%)",
                   title="1 s windows; mean +/- SD")
    axes[0, 0].set_ylim(0, 120)              # 给顶部图例留出空间，避免遮挡误差棒
    axes[0, 0].legend(fontsize=8, loc="upper center", ncol=3)
    for axis, state in zip(axes.flat[1:], STATES):
        axis.boxplot(
            [metrics[(metrics.state == state) & (metrics.algorithm == algorithm)]
             .accuracy.values * 100 for algorithm in figure_order],
            tick_labels=labels)
        axis.set(title=f"CS vs {state}", ylabel="Accuracy (%)")
    figure.suptitle("Figure 7 reconstruction: IFB-OCN is a documented implementation variant")
    figure.savefig(output / "figure7_asynchronous.png", dpi=180)
    plt.close(figure)


# ---------------------------------------------------------------------------
# 报告文字
# ---------------------------------------------------------------------------

def _spread_text(value: float) -> str:
    """把标准差写成百分比；只有 1 个人时标准差无定义（NaN），显示成破折号更易读。"""
    return "—" if pd.isna(value) else f"{value * 100:.2f}%"


def frequency_report_lines(summary: pd.DataFrame) -> list[str]:
    """把频率汇总表写成 Markdown 表格行。"""
    lines = ["## 频率识别", "", "|算法|窗长(s)|人数|准确率均值±SD|ITR均值|",
             "|---|---:|---:|---:|---:|"]
    for row in summary.itertuples():
        lines.append(f"|{row.algorithm}|{row.window_seconds:g}|{row.subjects}|"
                     f"{row.mean_accuracy * 100:.2f}% ± {_spread_text(row.sd_accuracy)}|"
                     f"{row.mean_itr:.2f}|")
    return lines


def asynchronous_report_lines(summary: pd.DataFrame) -> list[str]:
    """把异步汇总表写成 Markdown 表格行。"""
    lines = ["", "## 1秒异步分类", "",
             "|状态|算法|类别数|准确率均值±SD|平衡准确率|NS误触发率|",
             "|---|---|---:|---:|---:|---:|"]
    for row in summary.itertuples():
        lines.append(f"|{row.state}|{row.algorithm}|{row.n_classes}|"
                     f"{row.mean_accuracy * 100:.2f}% ± {_spread_text(row.sd_accuracy)}|"
                     f"{row.mean_balanced_accuracy * 100:.2f}%|"
                     f"{row.mean_ns_false_positive_rate * 100:.2f}%|")
    return lines


LIMITATIONS = [
    "1. 每个 NS 原始试次的所有片段保持同一外层折；全部参数/频带选择只使用外层训练数据。",
    "2. FS 的 CS 训练特征采用内部留一模板；LDA 采用等先验、自动收缩。原文未完整交代这些选择，故需谨慎对照数值。",
    "3. IFB-OCN 重建版：Fp1、欧氏距离网络、平均度/聚类系数、训练内选三个频带和线性 SVM。五个频带取自同团队2023年方法论文；逐段标准化和嵌入网格为显式实现选择，并非原始 C-C/Cao 优化的精确实现。",
    "4. 频率识别从0.14秒起截窗；ITR计入1.5秒提示与0.14秒延迟；频率识别采用子带线性加权，FS候选频率使用平方加权。",
    "5. 地形图使用标准10–20坐标；信号单位保留原始单位。SNR主图按原文第9页式(3)的(2/Δf+1)公式计算，另保留排除中心点的邻域比；原有译文中的2/(Δf+1)括号错误已纠正。",
    "6. 九分类中NS有96个样本、每个频率只有12个样本，故普通准确率与九分类平衡准确率不同；二分类与九分类准确率不可直接排名。",
    "7. S11出现疑似类别循环偏移：FBCCA的86/96条4秒预测符合+1类别模式，主实验没有重标注。详见项目DATA_AUDIT.md；这仍是未经作者确认的数据异常推断。",
    "8. 质量分析在分类之外单独运行；分类算法、切窗、标签及预测不受质量分析影响。",
]

SOURCES = [
    "- 目标论文：https://doi.org/10.1080/27706710.2024.2418650",
    "- FS原始方法：https://doi.org/10.1109/TNSRE.2023.3246359",
    "- OCN网络定义：https://doi.org/10.1155/2019/2167871",
    "- IFB-OCN原始论文：https://doi.org/10.1016/j.jneumeth.2020.108776",
]


def report_header(subjects: list[str]) -> list[str]:
    """报告开头：受试者、实现了什么、必须强调的方法边界。"""
    return [
        f"# YSU 全流程复现实验（{len(subjects)} 名受试者）", "",
        "受试者：" + ", ".join(subjects), "",
        "实现了频率识别、两种 FS 异步框架、IFB-OCN 重建版与信号质量分析。",
        "**全流程运行不等同于逐参数精确复刻。IFB-OCN 原始参数优化细节未取得，"
        "其结果必须标作重建版。**", "",
    ]


# ---------------------------------------------------------------------------
# 总入口
# ---------------------------------------------------------------------------

def summarize_all(config: dict, subjects: list[str], include_frequency: bool = True,
                  include_async: bool = True) -> None:
    """汇总指定受试者的结果，写出报告所需的所有文件和四张图。

    参数
    ----
    config : 由 ssvep.config.load_config() 读取的配置（结果目录用 paths.output）。
    subjects : 要汇总的受试者，例如 ["S01"]；必须都已有完整结果。
    include_frequency / include_async : 只汇总已完成的任务，便于分阶段运行。

    输出（都写在 config["paths"]["output"] 下）
    ----
    frequency_metrics.csv、frequency_summary.csv、figure5_accuracy.png、figure6_itr.png、
    frequency_confusion.png、frequency_paper_comparison.csv、
    asynchronous_metrics.csv、asynchronous_summary.csv、figure7_asynchronous.png、
    asynchronous_paper_comparison.csv、REPORT.md、validation.txt
    """
    output = output_dir(config)
    output.mkdir(parents=True, exist_ok=True)
    report_lines = report_header(subjects)
    validation = []

    if include_frequency:
        metrics, predictions = frequency_tables(config, subjects)
        validation.append(f"Frequency predictions: {len(predictions)}")
        metrics.to_csv(output / "frequency_metrics.csv", index=False)
        summary = summarize_frequency(metrics)
        summary.to_csv(output / "frequency_summary.csv", index=False)
        write_frequency_paper_comparison(summary, output)
        plot_frequency_curves(metrics, output)
        plot_frequency_confusion(config, metrics, predictions, output)
        report_lines.extend(frequency_report_lines(summary))

    if include_async:
        metrics, predictions = asynchronous_tables(config, subjects)
        validation.append(f"Asynchronous predictions: {len(predictions)}")
        metrics.to_csv(output / "asynchronous_metrics.csv", index=False)
        summary = summarize_asynchronous(metrics)
        summary.to_csv(output / "asynchronous_summary.csv", index=False)
        write_asynchronous_paper_comparison(summary, output)
        plot_asynchronous(metrics, output)
        report_lines.extend(asynchronous_report_lines(summary))

    report_lines.extend(["", "## 解释与限制", ""] + LIMITATIONS
                        + ["", "## 来源", ""] + SOURCES + [""])
    (output / "REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")

    validation.append("Accuracy independently recomputed; group separation checked.")
    (output / "validation.txt").write_text("\n".join(validation), encoding="utf-8")
    print(f"报告已写入 {output}", flush=True)
