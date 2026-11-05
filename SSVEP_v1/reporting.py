"""汇总全体受试者结果、核验预测完整性，并生成对应论文图表的复现报告。

频率识别与异步二/九分类分别汇总。任何缺失受试者、重复预测或跨折原始试次
都视为失败；不会只用成功的一部分数据悄悄生成“24 人完整结果”。
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def read_subject_tables(output, folder, suffix, subjects):
    """必须逐个找到请求的受试者文件；路径缺失时直接抛出 FileNotFoundError。"""
    return pd.concat([pd.read_csv(output / folder / f"{subject}_{suffix}.csv") for subject in subjects], ignore_index=True)


def validate_predictions(predictions, asynchronous=False):
    """检查唯一预测、组内单折与类别范围；返回可写入验证记录的行数。"""
    keys = ["subject", "state", "algorithm", "sample"] if asynchronous else [
        "subject", "window_seconds", "algorithm", "fold", "true_class"]
    if predictions.duplicated(keys).any():
        raise AssertionError("Duplicate predictions")
    if asynchronous:
        group_folds = predictions.groupby(["subject", "state", "algorithm", "original_group"]).fold.nunique()
        if (group_folds != 1).any():
            raise AssertionError("A raw NS trial has segments in different folds")
        counts = predictions.groupby(["subject", "state", "algorithm"]).size()
        if (counts != 192).any():
            raise AssertionError("Expected 192 CS+NS predictions in each asynchronous task")
    else:
        counts = predictions.groupby(["subject", "window_seconds", "algorithm"]).size()
        if (counts != 96).any():
            raise AssertionError("Expected 96 frequency predictions per subject/window/algorithm")
    return len(predictions)


def frequency_figures(config, metrics, predictions, summary):
    """绘制平均曲线及逐人箱线图，并输出最长时间窗的混淆矩阵。"""
    output = Path(config["output_dir"])
    algorithms = ["FBCCA", "TRCA", "eTRCA"]
    windows = sorted(metrics.window_seconds.unique())
    for metric, filename, ylabel, multiplier in (
        ("accuracy", "figure5_accuracy.png", "Accuracy (%)", 100),
        ("itr_bits_min", "figure6_itr.png", "ITR (bits/min)", 1),
    ):
        figure, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
        for algorithm in algorithms:
            group = metrics[metrics.algorithm == algorithm].groupby("window_seconds")[metric]
            mean, standard = group.mean(), group.std()
            axes[0, 0].errorbar(mean.index, mean.values * multiplier, yerr=standard.values * multiplier,
                                marker="o", capsize=3, label=algorithm)
        axes[0, 0].set(title=f"Mean +/- between-subject SD; n={metrics.subject.nunique()}",
                       xlabel="Window (s)", ylabel=ylabel, xticks=windows)
        axes[0, 0].legend()
        for axis, algorithm in zip(axes.flat[1:], algorithms):
            values = [metrics[(metrics.algorithm == algorithm) & (metrics.window_seconds == window)][metric].values * multiplier
                      for window in windows]
            axis.boxplot(values, tick_labels=[f"{window:g}" for window in windows])
            axis.set(title=algorithm, xlabel="Window (s)", ylabel=ylabel)
        figure.savefig(output / filename, dpi=180)
        plt.close(figure)
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for axis, algorithm in zip(axes, algorithms):
        selected = predictions[(predictions.algorithm == algorithm) & (predictions.window_seconds == max(windows))]
        counts = confusion_matrix(selected.true_class, selected.predicted_class, labels=np.arange(8))
        axis.imshow(counts, cmap="Blues")
        axis.set(title=f"{algorithm}, {max(windows):g} s", xlabel="Predicted frequency (Hz)", ylabel="True frequency (Hz)",
                 xticks=range(8), yticks=range(8), xticklabels=config["frequencies"], yticklabels=config["frequencies"])
        for (row, column), value in np.ndenumerate(counts):
            axis.text(column, row, str(value), ha="center", va="center", fontsize=7,
                      color="white" if value > counts.max() / 2 else "black")
    figure.savefig(output / "frequency_confusion.png", dpi=180)
    plt.close(figure)


def asynchronous_figures(config, metrics):
    """分别标注二分类与九分类，避免将不同任务的准确率误读为公平的算法排名。"""
    output = Path(config["output_dir"])
    algorithms = ["IFB-OCN-reconstructed", "FS-eTRCA", "FS-eTRCA&FBCCA"]
    labels = ["IFB-OCN rebuilt\n2 classes", "FS-eTRCA\n9 classes", "FS-eTRCA+FBCCA\n9 classes"]
    states = ["NS1", "NS2", "NS3"]
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    for algorithm_index, algorithm in enumerate(algorithms):
        selected = metrics[metrics.algorithm == algorithm].groupby("state").accuracy
        mean, standard = selected.mean().reindex(states), selected.std().reindex(states)
        axes[0, 0].bar(np.arange(3) + (algorithm_index - 1) * 0.25, mean.values * 100, width=0.25,
                      yerr=standard.values * 100, capsize=3, label=labels[algorithm_index])
    axes[0, 0].set(xticks=range(3), xticklabels=states, ylabel="Accuracy (%)", title="1 s windows; mean +/- SD")
    # 横排图例放在预留的上部空间，避免遮挡柱状图及误差棒。
    axes[0, 0].set_ylim(0, 120)
    axes[0, 0].legend(fontsize=8, loc="upper center", ncol=3)
    for axis, state in zip(axes.flat[1:], states):
        axis.boxplot([metrics[(metrics.state == state) & (metrics.algorithm == algorithm)].accuracy.values * 100
                      for algorithm in algorithms], tick_labels=labels)
        axis.set(title=f"CS vs {state}", ylabel="Accuracy (%)")
    figure.suptitle("Figure 7 reconstruction: IFB-OCN is a documented implementation variant")
    figure.savefig(output / "figure7_asynchronous.png", dpi=180)
    plt.close(figure)


def summarize_all(config, subjects, include_frequency=True, include_async=True):
    """聚合全部请求受试者并核验每个指标与逐试次预测一致，再写报告。"""
    output = Path(config["output_dir"])
    report = [f"# YSU 全流程复现实验（{len(subjects)} 名受试者）", "",
              "受试者：" + ", ".join(subjects), "",
              "实现了频率识别、两种 FS 异步框架、IFB-OCN 重建版与信号质量分析。",
              "**全流程运行不等同于逐参数精确复刻。IFB-OCN 原始参数优化细节未取得，其结果必须标作重建版。**", ""]
    validation = []
    if include_frequency:
        metrics = read_subject_tables(output, "frequency", "metrics", subjects)
        predictions = read_subject_tables(output, "frequency", "predictions", subjects)
        validation.append(f"Frequency predictions: {validate_predictions(predictions)}")
        measured = predictions.assign(correct=predictions.true_class == predictions.predicted_class).groupby(
            ["subject", "window_seconds", "algorithm"]).correct.mean()
        recorded = metrics.set_index(["subject", "window_seconds", "algorithm"]).accuracy
        np.testing.assert_allclose(measured.sort_index(), recorded.sort_index())
        metrics.to_csv(output / "frequency_metrics.csv", index=False)
        summary = metrics.groupby(["algorithm", "window_seconds"], as_index=False).agg(
            subjects=("subject", "nunique"), mean_accuracy=("accuracy", "mean"), sd_accuracy=("accuracy", "std"),
            mean_itr=("itr_bits_min", "mean"), sd_itr=("itr_bits_min", "std"))
        summary.to_csv(output / "frequency_summary.csv", index=False)
        frequency_figures(config, metrics, predictions, summary)
        report.extend(["## 频率识别", "", "|算法|窗长(s)|人数|准确率均值±SD|ITR均值|",
                       "|---|---:|---:|---:|---:|"])
        for row in summary.itertuples():
            report.append(f"|{row.algorithm}|{row.window_seconds:g}|{row.subjects}|{row.mean_accuracy*100:.2f}% ± {row.sd_accuracy*100:.2f}%|{row.mean_itr:.2f}|")
        references = {"FBCCA": 0.9349, "TRCA": 0.7300, "eTRCA": 0.7947}
        comparison = summary[summary.window_seconds == 4].copy()
        comparison["paper_accuracy"] = comparison.algorithm.map(references)
        comparison["difference_percentage_points"] = (comparison.mean_accuracy - comparison.paper_accuracy) * 100
        comparison.to_csv(output / "frequency_paper_comparison.csv", index=False)
    if include_async:
        metrics = read_subject_tables(output, "asynchronous", "metrics", subjects)
        predictions = read_subject_tables(output, "asynchronous", "predictions", subjects)
        validation.append(f"Asynchronous predictions: {validate_predictions(predictions, asynchronous=True)}")
        measured = predictions.assign(correct=predictions.true_class == predictions.predicted_class).groupby(
            ["subject", "state", "algorithm"]).correct.mean()
        recorded = metrics.set_index(["subject", "state", "algorithm"]).accuracy
        np.testing.assert_allclose(measured.sort_index(), recorded.sort_index())
        metrics.to_csv(output / "asynchronous_metrics.csv", index=False)
        summary = metrics.groupby(["state", "algorithm", "n_classes"], as_index=False).agg(
            subjects=("subject", "nunique"), mean_accuracy=("accuracy", "mean"), sd_accuracy=("accuracy", "std"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"), mean_cs_detection_rate=("cs_detection_rate", "mean"),
            mean_ns_false_positive_rate=("ns_false_positive_rate", "mean"))
        summary.to_csv(output / "asynchronous_summary.csv", index=False)
        asynchronous_figures(config, metrics)
        report.extend(["", "## 1秒异步分类", "", "|状态|算法|类别数|准确率均值±SD|平衡准确率|NS误触发率|",
                       "|---|---|---:|---:|---:|---:|"])
        for row in summary.itertuples():
            report.append(f"|{row.state}|{row.algorithm}|{row.n_classes}|{row.mean_accuracy*100:.2f}% ± {row.sd_accuracy*100:.2f}%|{row.mean_balanced_accuracy*100:.2f}%|{row.mean_ns_false_positive_rate*100:.2f}%|")
        paper = {"IFB-OCN-reconstructed": [0.7051, 0.7235, 0.6934],
                 "FS-eTRCA": [0.6764, 0.6647, 0.6534], "FS-eTRCA&FBCCA": [0.7639, 0.7535, 0.7502]}
        comparison = summary.copy()
        comparison["paper_accuracy"] = [paper[row.algorithm][int(row.state[-1]) - 1] for row in comparison.itertuples()]
        comparison["difference_percentage_points"] = (comparison.mean_accuracy - comparison.paper_accuracy) * 100
        comparison["exact_method_match"] = False
        comparison.to_csv(output / "asynchronous_paper_comparison.csv", index=False)
    report.extend(["", "## 解释与限制", "",
        "1. 每个 NS 原始试次的所有片段保持同一外层折；全部参数/频带选择只使用外层训练数据。",
        "2. FS 的 CS 训练特征采用内部留一模板；LDA 采用等先验、自动收缩。原文未完整交代这些选择，故需谨慎对照数值。",
        "3. IFB-OCN 重建版：Fp1、欧氏距离网络、平均度/聚类系数、训练内选三个频带和线性 SVM。五个频带取自同团队2023年方法论文；逐段标准化和嵌入网格为显式实现选择，并非原始 C-C/Cao 优化的精确实现。",
        "4. 频率识别从0.14秒起截窗；ITR计入1.5秒提示与0.14秒延迟；频率识别采用子带线性加权，FS候选频率使用平方加权。",
        "5. 地形图使用标准10–20坐标；信号单位保留原始单位。SNR主图按原文第9页式(3)的(2/Δf+1)公式计算，另保留排除中心点的邻域比；原有译文中的2/(Δf+1)括号错误已纠正。",
        "6. 九分类中NS有96个样本、每个频率只有12个样本，故普通准确率与九类平衡准确率不同；二分类与九分类准确率不可直接排名。",
        "7. S11出现疑似类别循环偏移：FBCCA的86/96条4秒预测符合+1类别模式，主实验没有重标注。详见项目DATA_AUDIT.md；这仍是未经作者确认的数据异常推断。",
        "8. 质量分析在全部分类结束后依据原文PDF校正SNR公式并重算；分类算法、切窗、标签及预测未因此改变。初始分类来源见manifest_all.json，最终质量/报告来源见manifest_quality.json与manifest_report.json。",
        "", "## 来源", "",
        "- 目标论文：https://doi.org/10.1080/27706710.2024.2418650",
        "- FS原始方法：https://doi.org/10.1109/TNSRE.2023.3246359",
        "- OCN网络定义：https://doi.org/10.1155/2019/2167871",
        "- IFB-OCN原始论文：https://doi.org/10.1016/j.jneumeth.2020.108776", ""])
    (output / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    (output / "validation.txt").write_text("\n".join(validation + ["Accuracy independently recomputed; group separation checked."]), encoding="utf-8")
