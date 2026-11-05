"""复现论文第 4.2 节的 FBCCA 频率识别：24 人、8 类别、12 折、1~4 秒（图 5、图 6）。

4.2 节用三种算法识别 8 个刺激频率，本文件先复现其中的 FBCCA：

    - 电极：Pz, PO7, PO3, POz, PO4, PO8, O1, Oz, O2 共 9 个
    - 子带：4 个，通带 [m*8-2, 80] Hz（m = 1,2,3,4）
    - 参考信号：每个频率的正弦 + 余弦，含基频到第 5 次谐波，共 10 条
    - 打分：score(c) = Σ_m w_m · rho_m(c)^2，其中 w_m = m^(-1.25) + 0.25
    - 12 折：第 f 折拿每个频率的第 f 次重复当测试，其余 11 次是训练部分
      （FBCCA 用的是数学参考，不需要用训练数据拟合；训练折只起“不参与测试”的作用）

数据流（代码从上到下就是执行顺序）：

    MAT (8, 63, 22500, 12)
      → 取 9 个电极、转轴 → (8, 12, 9, 22500)
      → 去均值 / 降采样 5000→250 Hz / 50 Hz 陷波 → (8, 12, 9, 1125)
      → 按 1~4 秒截窗 → 4 子带滤波 → (4, 8, 12, 9, samples)
      → 每个测试试次和 8 个参考基做 CCA → (4 子带, 8 频率) 相关矩阵
      → 平方加权 → 取最大 → 预测频率
    24 人 × 4 窗长 × 96 个测试试次 = 9216 条预测

运行：python reproduce_4_2_fbcca.py
输出：results/section4_2_fbcca/ 下 5 个 CSV 和 1 张图
"""

from time import perf_counter

import matplotlib
matplotlib.use("Agg")               # 只把图存成文件，不弹窗口
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import cheb1ord, cheby1, sosfiltfilt
from scipy.special import xlogy

from config import (BAND_COUNT, CLASS_COUNT, CUE_SECONDS, DELAY_SECONDS,
                    FREQUENCIES, FS, HARMONIC_COUNT, PAPER_FBCCA_ACCURACY_4S,
                    PAPER_FBCCA_ITR_1S, PASSBANDS, SECTION4_2_DIR, STOPBANDS,
                    SUBJECTS, TRIAL_COUNT, WINDOWS)
from data import load_cs_trials


# ===========================================================================
# 1. 固定参数：子带权重、四个子带滤波器、用于攒结果的列表
# ===========================================================================
started = perf_counter()
SECTION4_2_DIR.mkdir(parents=True, exist_ok=True)

# 子带权重 w_m = m^(-1.25) + 0.25：第 1 子带权重最大，越高的子带越像补充证据。
weights = np.arange(1, BAND_COUNT + 1, dtype=float) ** (-1.25) + 0.25

# 四个子带的 Chebyshev I 带通：先由通带/阻带指标求最低阶数，再按 0.5 dB 纹波设计。
filter_sections = []    # 四条 子带的滤波器
for passband, stopband in zip(PASSBANDS, STOPBANDS):
    order, critical = cheb1ord(passband, stopband, gpass=3, gstop=40, fs=FS)    # 返回“满足要求的最小阶数”和“实际截止频率”
    filter_sections.append(cheby1(order, rp=0.5, Wn=critical,
                                  btype="bandpass", fs=FS, output="sos"))

prediction_rows = []     # 24 人 × 4 窗长 × 96 个测试试次 = 9216 条  累积逐试次测试结果和八个候选频率得分
metric_rows = []         # 24 人 × 4 窗长 = 96 条   累积每人每窗的准确率和 ITR
# 四种窗长各一个 8×8 混淆矩阵：行是真实类别，列是预测类别。
confusion_matrices = np.zeros((len(WINDOWS), CLASS_COUNT, CLASS_COUNT), dtype=int)


# ===========================================================================
# 2. 逐人处理：读一次数据，依次跑四种窗长
# ===========================================================================
for subject in SUBJECTS:
    print(f"[{subject}] 读取 CS 试次，计算 FBCCA 的 1~4 秒识别", flush=True)
    data = load_cs_trials(subject)                    # [8, 12, 9, 1125]

    for window_index, seconds in enumerate(WINDOWS):
        samples = round(seconds * FS)                 # 1 秒 = 250 点，4 秒 = 1000 点
        start = round(DELAY_SECONDS * FS)             # 0.14 秒 = 35 点
        cropped = data[..., start:start + samples]    # 只截时间轴：[8, 12, 9, samples]

        # 四个子带分别做零相位滤波，最前面新增子带轴 → [4, 8, 12, 9, samples]。
        bank = np.stack([sosfiltfilt(sections, cropped, axis=-1)
                         for sections in filter_sections])

        # ---- 构造 8 个频率的参考基 ----
        # 时间轴从 1/250 秒开始，和原始 MATLAB 实现一致。
        times = np.arange(1, samples + 1) / FS
        reference_bases = []    # 保存八个候选频率的正交参考基，每个基的形状为 (samples,10)，用于后续计算 CCA。
        for frequency in FREQUENCIES:   # 逐频率构造
            # 第 h 次谐波的相位 2πfht，堆成 (5, samples)：5 次谐波。
            phase = (2 * np.pi * frequency
                     * np.arange(1, HARMONIC_COUNT + 1)[:, None] * times[None, :])
            # 排成 sin1, cos1, sin2, cos2, ..., sin5, cos5 共 10 条参考信号。
            reference = np.stack([np.sin(phase), np.cos(phase)], axis=1).reshape(
                2 * HARMONIC_COUNT, samples)
            reference -= reference.mean(axis=-1, keepdims=True)   # CCA 要求中心化
            # 对 (samples, 10) 做 QR，得到列正交基 Q。相位平移不改变这个子空间，所以不需要估计参考信号的初相位。
            reference_basis, _ = np.linalg.qr(reference.T, mode="reduced")
            reference_bases.append(reference_basis)

        # ---- 12 折交叉验证 ----
        # 第 fold 折只测试每个类别的第 fold 次重复，共 8 个测试试次。
        # X、Y 中心化后，令 X.T=Qx*Rx、Y.T=Qy*Ry；Qx.T@Qy 的最大奇异值就是第一典型相关系数 rho。
        window_predictions = []     # 按“先折号、后真实类别”顺序保存当前人的 96 个预测类别，用于整体准确率与混淆矩阵统计。
        for fold in range(TRIAL_COUNT):
            for target_index, true_frequency in enumerate(FREQUENCIES):
                test = bank[:, target_index, fold]        # [4 子带, 9 通道, samples]
                correlations = np.empty((BAND_COUNT, CLASS_COUNT))  # # 准备存放四个子带分别对八个候选频率的第一典型相关系数，形状为 (4,8)。

                for band_index in range(BAND_COUNT):
                    # 每个子带单独中心化、单独取正交基；四个子带不能先平均。
                    centered = test[band_index] - test[band_index].mean(axis=-1, keepdims=True)
                    eeg_basis, _ = np.linalg.qr(centered.T, mode="reduced")   # (samples, 9)
                    for candidate_index, reference_basis in enumerate(reference_bases):
                        # 两个正交基交叉矩阵的最大奇异值，就是第一典型相关系数。
                        # 这等价于 CCA 求最大相关，但不用迭代优化空间滤波器。
                        correlations[band_index, candidate_index] = np.linalg.svd(
                            eeg_basis.T @ reference_basis, compute_uv=False)[0]

                # 论文式 (6)：四个子带的相关系数先平方、再按 w_m 加权求和。
                scores = weights @ (correlations ** 2)
                predicted_index = int(np.argmax(scores))
                window_predictions.append(predicted_index)

                # 每个测试试次存一行：真实/预测类别、真实/预测频率、8 个候选得分。
                prediction_rows.append({
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

        # ---- 当前人当前窗长的准确率、ITR、混淆矩阵 ----
        # 96 个预测按“先折号、后类别”排列，reshape 成 (12, 8) 后每行是一折。
        predicted = np.array(window_predictions).reshape(TRIAL_COUNT, CLASS_COUNT)
        accuracy = np.mean(predicted == np.arange(CLASS_COUNT)[None, :])

        # ITR 式 (2)：log2(N) + P·log2(P) + (1-P)·log2((1-P)/(N-1))，再乘 60/T。
        # 决策时间 T = 观察窗 + 1.5 秒提示 + 0.14 秒视觉延迟。
        decision_seconds = seconds + CUE_SECONDS + DELAY_SECONDS
        bits = (np.log(CLASS_COUNT) + xlogy(accuracy, accuracy)
                + xlogy(1 - accuracy, (1 - accuracy) / (CLASS_COUNT - 1))) / np.log(2)
        # 准确率不高于机会水平 1/8 时 ITR 记 0（负的“信息率”没有意义）。
        itr = float((accuracy > 1 / CLASS_COUNT) * bits * 60 / decision_seconds)

        np.add.at(confusion_matrices[window_index],
                  (np.tile(np.arange(CLASS_COUNT), TRIAL_COUNT), predicted.ravel()), 1)
        metric_rows.append({
            "subject": subject,
            "window_seconds": seconds,
            "accuracy": accuracy,
            "itr_bits_min": itr,
            "n_trials": predicted.size,
            "decision_seconds": decision_seconds,
        })
        print(f"  {seconds} s: accuracy={accuracy:.2%}, ITR={itr:.4f} bits/min", flush=True)


# ===========================================================================
# 3. 汇总：24 人每个窗长的均值、标准差、标准误，以及与原文的差值
# ===========================================================================
predictions = pd.DataFrame(prediction_rows)     # 把 9216 条字典记录转为预测表，每行对应某个人、某个窗、某折的一个测试试次
metrics = pd.DataFrame(metric_rows)     # 把 96 条每人每窗的指标字典记录转成表格，为跨人等权平均和箱线图提供数据

# 先算每个人的准确率/ITR，再跨 24 人平均；不能拿平均准确率代回 ITR 公式。
summary = metrics.groupby("window_seconds", as_index=False).agg(
    accuracy_mean=("accuracy", "mean"),
    accuracy_sd=("accuracy", "std"),
    itr_mean=("itr_bits_min", "mean"),
    itr_sd=("itr_bits_min", "std"),
)
# 标准误 = 样本标准差 / sqrt(n)，只用于图 5/6 的误差线。
summary["accuracy_sem"] = summary["accuracy_sd"] / np.sqrt(len(SUBJECTS))
summary["itr_sem"] = summary["itr_sd"] / np.sqrt(len(SUBJECTS))

# 原文正文只给出 FBCCA 的两个数：4 秒准确率、1 秒 ITR。只比这两个。
comparison = pd.DataFrame({
    "metric": ["accuracy_4s_percent", "itr_1s_bits_min"],
    "reproduced": [summary.loc[summary.window_seconds == 4, "accuracy_mean"].iloc[0] * 100,
                   summary.loc[summary.window_seconds == 1, "itr_mean"].iloc[0]],
    "paper": [PAPER_FBCCA_ACCURACY_4S, PAPER_FBCCA_ITR_1S],
})
comparison["difference"] = comparison.reproduced - comparison.paper

# 四个 8×8 混淆矩阵展开成 256 行长表，方便按真实/预测频率查错误模式。
confusion_table = pd.DataFrame({
    "window_seconds": np.repeat(WINDOWS, CLASS_COUNT * CLASS_COUNT),
    "true_frequency_hz": np.tile(np.repeat(FREQUENCIES, CLASS_COUNT), len(WINDOWS)),
    "predicted_frequency_hz": np.tile(FREQUENCIES, len(WINDOWS) * CLASS_COUNT),
    "count": confusion_matrices.ravel(),
})


# ===========================================================================
# 4. 写 CSV
# ===========================================================================
# 保存每次测试的真实标签、预测标签及八频率得分；index=False 不额外写入 pandas 自动行号。
predictions.to_csv(SECTION4_2_DIR / "predictions.csv", index=False)
# 存 24 人各四种窗长的准确率、ITR、试次数和决策时长，共 96 行
metrics.to_csv(SECTION4_2_DIR / "subject_metrics.csv", index=False)
# 保存四种窗长的跨人均值、标准差和标准误
summary.to_csv(SECTION4_2_DIR / "summary.csv", index=False)
# 保存与正文两个已知数值的差异
comparison.to_csv(SECTION4_2_DIR / "comparison_with_paper.csv", index=False)
# 保存所有窗长、所有频率对的混淆计数
confusion_table.to_csv(SECTION4_2_DIR / "confusion_matrices.csv", index=False)


# ===========================================================================
# 5. 画图：准确率和 ITR 的跨人均值 + 24 人分布（对应论文图 5、图 6）
# ===========================================================================
accuracy_by_subject = metrics.pivot(index="subject", columns="window_seconds",
                                    values="accuracy") * 100
itr_by_subject = metrics.pivot(index="subject", columns="window_seconds",
                               values="itr_bits_min")

plt.rcParams.update({"font.size": 10,
                     "axes.spines.top": False,
                     "axes.spines.right": False})
figure, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
figure.suptitle("Section 4.2 | FBCCA | 24 subjects | 12-fold evaluation", fontsize=15)

# (a) 平均准确率随窗长的变化，误差线是跨人标准误
axes[0, 0].errorbar(WINDOWS, summary.accuracy_mean * 100, yerr=summary.accuracy_sem * 100,
                    fmt="o-", capsize=4, color="#2074a5", label="FBCCA mean +/- SEM")
axes[0, 0].scatter([4], [PAPER_FBCCA_ACCURACY_4S], marker="D", facecolors="none",
                   edgecolors="#d94b46", s=65, label="Paper: 93.49% at 4 s")
axes[0, 0].set(xlabel="Data length (s)", ylabel="Accuracy (%)", xticks=WINDOWS,
               ylim=(0, 105), title="(a) Mean accuracy across subjects")
axes[0, 0].legend(fontsize=9, loc="lower right")

# (b) 每个窗长下 24 个人的准确率分布
axes[0, 1].boxplot([accuracy_by_subject[seconds].to_numpy() for seconds in WINDOWS],
                   tick_labels=WINDOWS, showmeans=True)
axes[0, 1].set(xlabel="Data length (s)", ylabel="Accuracy (%)", ylim=(0, 105),
               title="(b) Accuracy distribution: 24 subjects")

# (c) 平均 ITR 随窗长的变化
axes[1, 0].errorbar(WINDOWS, summary.itr_mean, yerr=summary.itr_sem,
                    fmt="o-", capsize=4, color="#2074a5", label="FBCCA mean +/- SEM")
axes[1, 0].scatter([1], [PAPER_FBCCA_ITR_1S], marker="D", facecolors="none",
                   edgecolors="#d94b46", s=65, label="Paper: 43.8325 at 1 s")
axes[1, 0].set(xlabel="Data length (s)", ylabel="ITR (bits/min)", xticks=WINDOWS,
               ylim=(0, None), title="(c) Mean ITR across subjects")
axes[1, 0].legend(fontsize=9)

# (d) 每个窗长下 24 个人的 ITR 分布
axes[1, 1].boxplot([itr_by_subject[seconds].to_numpy() for seconds in WINDOWS],
                   tick_labels=WINDOWS, showmeans=True)
axes[1, 1].set(xlabel="Data length (s)", ylabel="ITR (bits/min)", ylim=(0, None),
               title="(d) ITR distribution: 24 subjects")

figure.savefig(SECTION4_2_DIR / "fbcca_accuracy_itr.png", dpi=200)
plt.close(figure)


# ===========================================================================
# 6. 在控制台打印汇总、原文对照和耗时
# ===========================================================================
elapsed = perf_counter() - started
print()
print(summary.to_string(index=False, float_format="%.4f"), flush=True)
print(comparison.to_string(index=False, float_format="%.4f"), flush=True)
print(f"\n完成：{SECTION4_2_DIR}\n用时：{elapsed:.1f} 秒")
