"""把第 4.2、4.3 节的结果画成论文的图 5、图 6、图 7。

本文件只读取 results/ 下已经算好的逐人指标表，不重新训练、不重新测试：

    results/section4_2_fbcca/subject_metrics.csv
    results/section4_2_trca/subject_metrics.csv
    results/section4_2_etrca/subject_metrics.csv
    results/section4_3_ifb_ocn/subject_metrics.csv
    results/section4_3_fs/subject_metrics.csv

输出到 results/figures/：

    figure5.png  图 5：三种算法的准确率随数据长度（1~4 秒）变化 + 24 人分布
    figure6.png  图 6：三种算法的 ITR 随数据长度变化 + 24 人分布
    figure7.png  图 7：CS 与三种 NS 的分类（IFB-OCN 二分类 + 两种 FS 九分类）

运行：python plot_paper_figures.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")               # 只把图存成文件，不弹窗口
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from config import (PAPER_ETRCA_ACCURACY_4S, PAPER_ETRCA_ITR_1S,
                    PAPER_FBCCA_ACCURACY_4S, PAPER_FBCCA_ITR_1S,
                    PAPER_FIGURES_DIR, PAPER_FS_ACCURACY, PAPER_OCN_ACCURACY,
                    PAPER_TRCA_ACCURACY_4S, PAPER_TRCA_ITR_1S,
                    SECTION4_2_DIR, SECTION4_2_ETRCA_DIR, SECTION4_2_TRCA_DIR,
                    SECTION4_3_FS_DIR, SECTION4_3_IFB_OCN_DIR)

PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# 三种频率识别算法的固定颜色，图 5 和图 6 保持一致。
ALGORITHM_COLORS = {"FBCCA": "#2074a5", "TRCA": "#e29b35", "eTRCA": "#2c9d65"}
STATES = ["NS1", "NS2", "NS3"]


# ===========================================================================
# 1. 读入 4.2 三种算法的逐人指标（每行：某人、某窗长、准确率、ITR）
# ===========================================================================
method_sections = [("FBCCA", SECTION4_2_DIR),
                   ("TRCA", SECTION4_2_TRCA_DIR),
                   ("eTRCA", SECTION4_2_ETRCA_DIR)]

frequency_tables = []
for method, section in method_sections:
    table = pd.read_csv(section / "subject_metrics.csv")
    table["method"] = method
    frequency_tables.append(table)
metrics_42 = pd.concat(frequency_tables, ignore_index=True)

windows = sorted(metrics_42["window_seconds"].unique())
methods_42 = list(ALGORITHM_COLORS)


# ===========================================================================
# 2. 图 5 / 图 6 的通用画法：左上均值曲线，其余三格是各算法的 24 人分布
# ===========================================================================
def plot_frequency_metric(value, ylabel, scale, paper_values, paper_window,
                          title, filename):
    """画一张 2×2 图。

    value        : 指标列名，accuracy 或 itr_bits_min
    ylabel       : 纵轴文字
    scale        : 准确率乘 100 转百分数，ITR 保持 1
    paper_values : {算法: 原文在该窗长下的均值}，只用于画参考点
    """
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
    figure.suptitle(title, fontsize=15)

    # (a) 左上：三种算法的跨人均值 ± 标准误随窗长变化。
    for method in methods_42:
        stats = (metrics_42[metrics_42["method"] == method]
                 .groupby("window_seconds")[value]
                 .agg(["mean", "std", "count"]))
        sem = stats["std"] / np.sqrt(stats["count"])
        axes[0, 0].errorbar(stats.index, stats["mean"] * scale, yerr=sem * scale,
                            fmt="o-", capsize=4, color=ALGORITHM_COLORS[method],
                            label=method)
        # 原文均值用同色空心菱形标出（只标正文给出的那个窗长）。
        # 原文数值已经是绘图单位（准确率是 %，ITR 是 bits/min），不再乘 scale。
        axes[0, 0].scatter([paper_window], [paper_values[method]],
                           marker="D", facecolors="none",
                           edgecolors=ALGORITHM_COLORS[method], s=65, zorder=4)

    axes[0, 0].set(xlabel="Data length (s)", ylabel=ylabel, xticks=windows,
                   title="(a) Mean ± SEM across 24 subjects")
    axes[0, 0].legend(fontsize=9, loc="best")
    axes[0, 0].text(0.02, 0.98, "◇ = paper mean", transform=axes[0, 0].transAxes,
                    va="top", fontsize=8, color="#555555")

    # (b)(c)(d) 右上与下方：每种算法在四个窗长下的 24 人分布。
    for axis, method in zip(axes.flat[1:], methods_42):
        values = [metrics_42[(metrics_42["method"] == method)
                             & (metrics_42["window_seconds"] == seconds)][value].to_numpy()
                  * scale for seconds in windows]
        axis.boxplot(values, tick_labels=[f"{seconds:g}" for seconds in windows],
                     showmeans=True)
        axis.set(xlabel="Data length (s)", ylabel=ylabel, title=method)

    figure.savefig(PAPER_FIGURES_DIR / filename, dpi=200)
    plt.close(figure)
    print(f"已输出 {PAPER_FIGURES_DIR / filename}")


# 图 5：准确率（原文只给 4 秒的三个均值）。
plot_frequency_metric(
    value="accuracy", ylabel="Accuracy (%)", scale=100,
    paper_values={"FBCCA": PAPER_FBCCA_ACCURACY_4S,
                  "TRCA": PAPER_TRCA_ACCURACY_4S,
                  "eTRCA": PAPER_ETRCA_ACCURACY_4S},
    paper_window=4,
    title="Figure 5 | SSVEP frequency recognition accuracy | 24 subjects, 12-fold",
    filename="figure5.png")

# 图 6：ITR（原文只给 1 秒的三个均值）。
plot_frequency_metric(
    value="itr_bits_min", ylabel="ITR (bits/min)", scale=1,
    paper_values={"FBCCA": PAPER_FBCCA_ITR_1S,
                  "TRCA": PAPER_TRCA_ITR_1S,
                  "eTRCA": PAPER_ETRCA_ITR_1S},
    paper_window=1,
    title="Figure 6 | Information transfer rate | 24 subjects, 12-fold",
    filename="figure6.png")


# ===========================================================================
# 3. 读入 4.3 的两种任务：IFB-OCN 二分类 + 两种 FS 九分类
# ===========================================================================
# IFB-OCN 的 metrics 里 method 是 "IFB-OCN-reconstructed"，这里统一叫 IFB-OCN。
ocn = pd.read_csv(SECTION4_3_IFB_OCN_DIR / "subject_metrics.csv")
ocn["method"] = "IFB-OCN"
fs = pd.read_csv(SECTION4_3_FS_DIR / "subject_metrics.csv")

metrics_43 = pd.concat([ocn[["subject", "state", "method", "accuracy"]],
                        fs[["subject", "state", "method", "accuracy"]]],
                       ignore_index=True)

# 显示顺序：先二分类，再两种九分类。
methods_43 = ["IFB-OCN", "FS-eTRCA", "FS-eTRCA&FBCCA"]
labels_43 = ["IFB-OCN\n(2 classes)", "FS-eTRCA\n(9 classes)", "FS-eTRCA&FBCCA\n(9 classes)"]
colors_43 = ["#7b5ea7", "#2074a5", "#e29b35"]

# 原文均值：IFB-OCN 按 NS1/NS2/NS3 给三个数，FS 两种方法用 (方法, 状态) 查表。
paper_43 = {
    "IFB-OCN": list(zip(STATES, PAPER_OCN_ACCURACY)),
    "FS-eTRCA": [(state, PAPER_FS_ACCURACY[("FS-eTRCA", state)]) for state in STATES],
    "FS-eTRCA&FBCCA": [(state, PAPER_FS_ACCURACY[("FS-eTRCA&FBCCA", state)])
                       for state in STATES],
}


# ===========================================================================
# 4. 图 7：三状态 × 三方法的均值柱状图 + 每个状态的 24 人分布
# ===========================================================================
figure, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
figure.suptitle("Figure 7 | Classification between CS and NS states | 24 subjects",
                fontsize=15)

positions = np.arange(len(STATES))
bar_width = 0.26

# (a) 均值 ± SEM 柱状图；同色的空心菱形是原文均值。
for index, method in enumerate(methods_43):
    stats = (metrics_43[metrics_43["method"] == method]
             .groupby("state")["accuracy"].agg(["mean", "std", "count"]).reindex(STATES))
    sem = stats["std"] / np.sqrt(stats["count"])
    offset = (index - 1) * bar_width
    axes[0, 0].bar(positions + offset, stats["mean"] * 100, width=bar_width - 0.02,
                   yerr=sem * 100, capsize=3, color=colors_43[index],
                   label=labels_43[index].replace("\n", " "))
    axes[0, 0].scatter(positions + offset, [value for _, value in paper_43[method]],
                       marker="D", facecolors="white", edgecolors=colors_43[index],
                       s=55, zorder=4)

axes[0, 0].set(xticks=positions, xticklabels=STATES, ylabel="Accuracy (%)",
               ylim=(0, 120), title="(a) Mean ± SEM across 24 subjects")
# 图例里除了三种方法，再补一个“原文均值”的空心菱形代理，避免在旁边另写字。
handles, legend_labels = axes[0, 0].get_legend_handles_labels()
handles.append(Line2D([], [], marker="D", linestyle="none", markerfacecolor="white",
                      markeredgecolor="#777777", markersize=7))
legend_labels.append("paper mean")
axes[0, 0].legend(handles, legend_labels, fontsize=8, loc="upper center",
                  ncol=2, framealpha=0.9)

# (b)(c)(d) 每种 NS 状态下，三种方法的 24 人准确率分布。
short_labels = ["IFB-OCN\n2 cl.", "FS-eTRCA\n9 cl.", "FS&FBCCA\n9 cl."]
for axis, state in zip(axes.flat[1:], STATES):
    values = [metrics_43[(metrics_43["method"] == method)
                         & (metrics_43["state"] == state)]["accuracy"].to_numpy() * 100
              for method in methods_43]
    axis.boxplot(values, tick_labels=short_labels, showmeans=True)
    axis.set(ylabel="Accuracy (%)", ylim=(0, 105),
             title=f"CS versus {state}: individual accuracy")

# 二分类和九分类混在同一张图里，必须注明这个口径差异。
# 先给底部留出说明文字的空间，再在留白处写字，避免压住横轴刻度。
figure.get_layout_engine().set(rect=(0, 0.05, 1, 0.92))
figure.text(0.5, 0.012,
            "IFB-OCN is binary (CS vs NS); FS methods are 9-class (8 frequencies + NS); "
            "their accuracies are not directly comparable.",
            ha="center", fontsize=9, color="#555555")

figure.savefig(PAPER_FIGURES_DIR / "figure7.png", dpi=200)
plt.close(figure)
print(f"已输出 {PAPER_FIGURES_DIR / 'figure7.png'}")
