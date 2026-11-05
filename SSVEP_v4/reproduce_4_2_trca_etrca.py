"""复现论文第 4.2 节的 TRCA 与 eTRCA 频率识别（论文图 5、图 6）。

4.2 节用三种算法识别 8 个刺激频率：FBCCA（已单独复现在 reproduce_4_2_fbcca.py）、
TRCA 和 eTRCA。TRCA 与 eTRCA 的训练过程完全相同，只是“测试信号怎么和模板比”
不同，所以本文件一次算完两种方法，分别写出各自的结果目录。

两种方法共用的设置：
    - 电极：Pz, PO7, PO3, POz, PO4, PO8, O1, Oz, O2 共 9 个
    - 4 个子带：通带 [m*8-2, 80] Hz（m = 1,2,3,4）
    - 12 折：第 f 折拿每个频率的第 f 次重复当测试，其余 11 次作为训练
    - 训练：每个子带、每个类别用 11 次训练重复求一个空间滤波器 w 和一个平均模板
    - 打分：score(c) = Σ_m w_m · r_m(c)^2，其中 w_m = m^(-1.25) + 0.25

两种方法的区别：
    TRCA ：测试试次和候选模板都用“该候选类别自己的”空间滤波器投影，
            得到两条一维序列后算 Pearson 相关。
    eTRCA：把 8 个类别的空间滤波器按列拼成 (9,8) 的矩阵，一次投影出 8 条序列，
            拉直后算“二维”Pearson 相关；它衡量的是 8 个类别的投影模式
            是否同时吻合，比单类更稳。

空间滤波器怎么训练（TRCA 的目标）：
    对每个类别，用训练重复求两个 9×9 矩阵：
        S = 不同试次之间的交叉协方差之和（希望保留：跨试次重复出现的成分）
        Q = 全部训练数据拼接后的总协方差（希望压制：所有成分的总能量）
    解广义特征值问题 S w = λ Q w，取最大特征值对应的 w，再归一化。
    这样就把 9 个通道线性组合成一条最能体现该类别重复性的信号。

数据流（代码从上到下就是执行顺序）：
    MAT (8,63,22500,12)
      → 取 9 电极、预处理 → (8,12,9,1125)
      → 按 1~4 秒截窗 → 4 子带滤波 → (4,8,12,9,samples)
      → 每折用 11 次重复训练模板和空间滤波器
      → 对 8 个测试试次：TRCA 相关、eTRCA 相关 → 平方加权 → 预测
    24 人 × 4 窗长 × 96 个测试试次 = 每种方法 9216 条预测

运行：python reproduce_4_2_trca_etrca.py
输出：results/section4_2_trca/ 与 results/section4_2_etrca/ 各 5 个 CSV + 1 张图
"""

from time import perf_counter

import matplotlib
matplotlib.use("Agg")               # 只把图存成文件，不弹窗口
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.signal import cheb1ord, cheby1, sosfiltfilt
from scipy.special import xlogy

from config import (BAND_COUNT, CLASS_COUNT, CUE_SECONDS, DELAY_SECONDS,
                    FREQUENCIES, FS, PAPER_ETRCA_ACCURACY_4S, PAPER_ETRCA_ITR_1S,
                    PAPER_TRCA_ACCURACY_4S, PAPER_TRCA_ITR_1S, PASSBANDS,
                    POSTERIOR_CHANNELS, SECTION4_2_ETRCA_DIR, SECTION4_2_TRCA_DIR,
                    STOPBANDS, SUBJECTS, TRIAL_COUNT, WINDOWS)
from data import load_cs_trials


# ===========================================================================
# 1. 固定参数：子带权重、四个子带滤波器、两种方法各自的攒结果列表
# ===========================================================================
started = perf_counter()
SECTION4_2_TRCA_DIR.mkdir(parents=True, exist_ok=True)
SECTION4_2_ETRCA_DIR.mkdir(parents=True, exist_ok=True)

# 子带权重 w_m = m^(-1.25) + 0.25，第 1 子带权重最大。
weights = np.arange(1, BAND_COUNT + 1, dtype=float) ** (-1.25) + 0.25

# 四个子带的 Chebyshev I 带通：先由通带/阻带指标求最低阶数，再按 0.5 dB 纹波设计。
filter_sections = []
for passband, stopband in zip(PASSBANDS, STOPBANDS):
    order, critical = cheb1ord(passband, stopband, gpass=3, gstop=40, fs=FS)
    filter_sections.append(cheby1(order, rp=0.5, Wn=critical,
                                  btype="bandpass", fs=FS, output="sos"))

n_channels = len(POSTERIOR_CHANNELS)     # 9

# 两种方法各自要攒：9216 条预测、96 条指标、4 个 8×8 混淆矩阵。
trca_prediction_rows, trca_metric_rows = [], []
trca_confusion = np.zeros((len(WINDOWS), CLASS_COUNT, CLASS_COUNT), dtype=int)
etrca_prediction_rows, etrca_metric_rows = [], []
etrca_confusion = np.zeros((len(WINDOWS), CLASS_COUNT, CLASS_COUNT), dtype=int)


# ===========================================================================
# 2. 逐人处理：读一次数据，依次跑四种窗长
# ===========================================================================
for subject in SUBJECTS:
    print(f"[{subject}] 读取 CS 试次，计算 TRCA / eTRCA 的 1~4 秒识别", flush=True)
    data = load_cs_trials(subject)                       # [8, 12, 9, 1125]

    for window_index, seconds in enumerate(WINDOWS):
        samples = round(seconds * FS)                    # 1 秒 = 250 点，4 秒 = 1000 点
        start = round(DELAY_SECONDS * FS)                # 0.14 秒 = 35 点
        cropped = data[..., start:start + samples]       # [8, 12, 9, samples]

        # 四个子带分别零相位滤波，最前面新增子带轴 → [4, 8, 12, 9, samples]。
        bank = np.stack([sosfiltfilt(sections, cropped, axis=-1)
                         for sections in filter_sections])

        window_predictions_trca = []
        window_predictions_etrca = []

        # ---- 12 折交叉验证 ----
        for fold in range(TRIAL_COUNT):
            # 第 fold 折只测试每个类别的第 fold 次重复，其余 11 次参与训练。
            training_indices = np.arange(TRIAL_COUNT) != fold

            # 模板 = 11 次训练重复的平均，形状 [4 子带, 8 类别, 9 通道, samples]。
            templates = bank[:, :, training_indices].mean(axis=2)

            # 训练每个子带、每个类别的一个空间滤波器，形状 [4, 8, 9]。
            spatial_filters = np.empty((BAND_COUNT, CLASS_COUNT, n_channels))
            for band_index in range(BAND_COUNT):
                for candidate_index in range(CLASS_COUNT):
                    training = bank[band_index, candidate_index, training_indices]  # (11,9,samples)

                    # 逐试次去均值后，用恒等式算 S = Σ_{r≠s} Z_r Z_sᵀ，
                    # 避免写二重循环计算 110 个交叉乘积。
                    centered_trials = training - training.mean(axis=-1, keepdims=True)
                    summed_trials = centered_trials.sum(axis=0)
                    centered_concat = centered_trials.transpose(1, 0, 2).reshape(n_channels, -1)
                    inter_trial = (summed_trials @ summed_trials.T
                                   - centered_concat @ centered_concat.T)

                    # Q 用“未逐试次中心化、按通道拼起来后再整体去均值”的总协方差，
                    concatenated = training.transpose(1, 0, 2).reshape(n_channels, -1)
                    concatenated = concatenated - concatenated.mean(axis=-1, keepdims=True)
                    total_covariance = concatenated @ concatenated.T

                    # 解广义特征值问题，只取最大特征值对应的特征向量。
                    _, eigenvectors = eigh(inter_trial, total_covariance,
                                           subset_by_index=[n_channels - 1, n_channels - 1])
                    filter_vector = eigenvectors[:, 0]
                    spatial_filters[band_index, candidate_index] = filter_vector / np.linalg.norm(filter_vector)

            # ---- 用训练好的滤波器识别本折的 8 个测试试次 ----
            for target_index, true_frequency in enumerate(FREQUENCIES):
                test = bank[:, target_index, fold]        # [4, 9, samples]

                trca_correlations = np.empty((BAND_COUNT, CLASS_COUNT))
                etrca_correlations = np.empty((BAND_COUNT, CLASS_COUNT))

                for band_index in range(BAND_COUNT):
                    # (a) TRCA：每个候选类别用自己的滤波器分别投影测试和模板。
                    for candidate_index in range(CLASS_COUNT):
                        spatial_filter = spatial_filters[band_index, candidate_index]
                        projected_test = spatial_filter @ test[band_index]
                        projected_template = spatial_filter @ templates[band_index, candidate_index]
                        projected_test -= projected_test.mean()
                        projected_template -= projected_template.mean()
                        trca_correlations[band_index, candidate_index] = (
                            projected_test @ projected_template
                            / (np.linalg.norm(projected_test) * np.linalg.norm(projected_template)))

                    # (b) eTRCA：8 个滤波器拼成 (9,8) 矩阵，一次投影出 8 条序列。
                    ensemble_filter = spatial_filters[band_index].T          # (9, 8)
                    projected_test = (ensemble_filter.T @ test[band_index]).ravel()
                    projected_test -= projected_test.mean()
                    for candidate_index in range(CLASS_COUNT):
                        projected_template = (ensemble_filter.T
                                              @ templates[band_index, candidate_index]).ravel()
                        projected_template -= projected_template.mean()
                        etrca_correlations[band_index, candidate_index] = (
                            projected_test @ projected_template
                            / (np.linalg.norm(projected_test) * np.linalg.norm(projected_template)))

                # 论文式 (1)：四个子带的相关系数平方后，按 w_m 加权求和。
                trca_scores = weights @ (trca_correlations ** 2)
                etrca_scores = weights @ (etrca_correlations ** 2)
                trca_prediction = int(np.argmax(trca_scores))
                etrca_prediction = int(np.argmax(etrca_scores))
                window_predictions_trca.append(trca_prediction)
                window_predictions_etrca.append(etrca_prediction)

                # 两种方法各存一行预测，列与 FBCCA 那份完全一致。
                for rows, predicted_index, scores in (
                        (trca_prediction_rows, trca_prediction, trca_scores),
                        (etrca_prediction_rows, etrca_prediction, etrca_scores)):
                    rows.append({
                        "subject": subject,
                        "window_seconds": seconds,
                        "fold": fold + 1,
                        "true_class": target_index,
                        "predicted_class": predicted_index,
                        "true_frequency_hz": true_frequency,
                        "predicted_frequency_hz": FREQUENCIES[predicted_index],
                        **{f"score_{frequency:g}_hz": score
                           for frequency, score in zip(FREQUENCIES, scores)},
                    })

        # ---- 当前人当前窗长：两种方法各算准确率、ITR、混淆矩阵 ----
        for window_predictions, metric_rows, confusion in (
                (window_predictions_trca, trca_metric_rows, trca_confusion),
                (window_predictions_etrca, etrca_metric_rows, etrca_confusion)):
            predicted = np.array(window_predictions).reshape(TRIAL_COUNT, CLASS_COUNT)
            accuracy = np.mean(predicted == np.arange(CLASS_COUNT)[None, :])

            # ITR 式 (2)：log2(N) + P·log2(P) + (1-P)·log2((1-P)/(N-1))，再乘 60/T。
            decision_seconds = seconds + CUE_SECONDS + DELAY_SECONDS
            bits = (np.log(CLASS_COUNT) + xlogy(accuracy, accuracy)
                    + xlogy(1 - accuracy, (1 - accuracy) / (CLASS_COUNT - 1))) / np.log(2)
            itr = float((accuracy > 1 / CLASS_COUNT) * bits * 60 / decision_seconds)

            np.add.at(confusion[window_index],
                      (np.tile(np.arange(CLASS_COUNT), TRIAL_COUNT), predicted.ravel()), 1)
            metric_rows.append({
                "subject": subject,
                "window_seconds": seconds,
                "accuracy": accuracy,
                "itr_bits_min": itr,
                "n_trials": predicted.size,
                "decision_seconds": decision_seconds,
            })

        print(f"  {seconds} s: TRCA={trca_metric_rows[-1]['accuracy']:.2%}, "
              f"eTRCA={etrca_metric_rows[-1]['accuracy']:.2%}", flush=True)


# ===========================================================================
# 3. 两种方法各自：汇总、写 CSV、画图
# ===========================================================================
# 两种方法除了输出目录、预测表、原文参考值不同，其余处理完全一样，
# 所以放在一个循环里跑，避免复制两遍代码。
methods = [
    dict(name="TRCA", predictions=trca_prediction_rows, metrics=trca_metric_rows,
         confusion=trca_confusion, out_dir=SECTION4_2_TRCA_DIR,
         paper_accuracy=PAPER_TRCA_ACCURACY_4S, paper_itr=PAPER_TRCA_ITR_1S,
         figure_name="trca_accuracy_itr.png"),
    dict(name="eTRCA", predictions=etrca_prediction_rows, metrics=etrca_metric_rows,
         confusion=etrca_confusion, out_dir=SECTION4_2_ETRCA_DIR,
         paper_accuracy=PAPER_ETRCA_ACCURACY_4S, paper_itr=PAPER_ETRCA_ITR_1S,
         figure_name="etrca_accuracy_itr.png"),
]

for method in methods:
    predictions = pd.DataFrame(method["predictions"])
    metrics = pd.DataFrame(method["metrics"])

    # 跨 24 人汇总：先算每人的准确率/ITR，再取均值和标准差；
    # 标准误 SEM = 标准差 / sqrt(24)，只用于图上的误差线。
    summary = metrics.groupby("window_seconds", as_index=False).agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_sd=("accuracy", "std"),
        itr_mean=("itr_bits_min", "mean"),
        itr_sd=("itr_bits_min", "std"),
    )
    summary["accuracy_sem"] = summary["accuracy_sd"] / np.sqrt(len(SUBJECTS))
    summary["itr_sem"] = summary["itr_sd"] / np.sqrt(len(SUBJECTS))

    # 原文正文只给出两个数：4 秒准确率、1 秒 ITR，只比较这两个。
    comparison = pd.DataFrame({
        "metric": ["accuracy_4s_percent", "itr_1s_bits_min"],
        "reproduced": [summary.loc[summary.window_seconds == 4, "accuracy_mean"].iloc[0] * 100,
                       summary.loc[summary.window_seconds == 1, "itr_mean"].iloc[0]],
        "paper": [method["paper_accuracy"], method["paper_itr"]],
    })
    comparison["difference"] = comparison.reproduced - comparison.paper

    # 四个 8×8 混淆矩阵展开成 256 行长表。
    confusion_table = pd.DataFrame({
        "window_seconds": np.repeat(WINDOWS, CLASS_COUNT * CLASS_COUNT),
        "true_frequency_hz": np.tile(np.repeat(FREQUENCIES, CLASS_COUNT), len(WINDOWS)),
        "predicted_frequency_hz": np.tile(FREQUENCIES, len(WINDOWS) * CLASS_COUNT),
        "count": method["confusion"].ravel(),
    })

    out_dir = method["out_dir"]
    predictions.to_csv(out_dir / "predictions.csv", index=False)
    metrics.to_csv(out_dir / "subject_metrics.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    comparison.to_csv(out_dir / "comparison_with_paper.csv", index=False)
    confusion_table.to_csv(out_dir / "confusion_matrices.csv", index=False)

    # ---- 图：准确率 / ITR 的跨人均值曲线 + 24 人分布 ----
    accuracy_by_subject = metrics.pivot(index="subject", columns="window_seconds",
                                        values="accuracy") * 100
    itr_by_subject = metrics.pivot(index="subject", columns="window_seconds",
                                   values="itr_bits_min")

    plt.rcParams.update({"font.size": 10,
                         "axes.spines.top": False,
                         "axes.spines.right": False})
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
    figure.suptitle(f"Section 4.2 | {method['name']} | 24 subjects | 12-fold evaluation",
                    fontsize=15)

    axes[0, 0].errorbar(WINDOWS, summary.accuracy_mean * 100,
                        yerr=summary.accuracy_sem * 100, fmt="o-", capsize=4,
                        color="#2074a5", label=f"{method['name']} mean +/- SEM")
    axes[0, 0].scatter([4], [method["paper_accuracy"]], marker="D", facecolors="none",
                       edgecolors="#d94b46", s=65,
                       label=f"Paper: {method['paper_accuracy']:.2f}% at 4 s")
    axes[0, 0].set(xlabel="Data length (s)", ylabel="Accuracy (%)", xticks=WINDOWS,
                   ylim=(0, 105), title="(a) Mean accuracy across subjects")
    axes[0, 0].legend(fontsize=9, loc="lower right")

    axes[0, 1].boxplot([accuracy_by_subject[seconds].to_numpy() for seconds in WINDOWS],
                       tick_labels=WINDOWS, showmeans=True)
    axes[0, 1].set(xlabel="Data length (s)", ylabel="Accuracy (%)", ylim=(0, 105),
                   title="(b) Accuracy distribution: 24 subjects")

    axes[1, 0].errorbar(WINDOWS, summary.itr_mean, yerr=summary.itr_sem, fmt="o-",
                        capsize=4, color="#2074a5", label=f"{method['name']} mean +/- SEM")
    axes[1, 0].scatter([1], [method["paper_itr"]], marker="D", facecolors="none",
                       edgecolors="#d94b46", s=65,
                       label=f"Paper: {method['paper_itr']:.4f} at 1 s")
    axes[1, 0].set(xlabel="Data length (s)", ylabel="ITR (bits/min)", xticks=WINDOWS,
                   ylim=(0, None), title="(c) Mean ITR across subjects")
    axes[1, 0].legend(fontsize=9)

    axes[1, 1].boxplot([itr_by_subject[seconds].to_numpy() for seconds in WINDOWS],
                       tick_labels=WINDOWS, showmeans=True)
    axes[1, 1].set(xlabel="Data length (s)", ylabel="ITR (bits/min)", ylim=(0, None),
                   title="(d) ITR distribution: 24 subjects")

    figure.savefig(out_dir / method["figure_name"], dpi=200)
    plt.close(figure)

    print()
    print(f"===== {method['name']} =====")
    print(summary.to_string(index=False, float_format="%.4f"), flush=True)
    print(comparison.to_string(index=False, float_format="%.4f"), flush=True)


# ===========================================================================
# 4. 打印总耗时
# ===========================================================================
elapsed = perf_counter() - started
print(f"\n完成：{SECTION4_2_TRCA_DIR.parent}\n用时：{elapsed:.1f} 秒")
