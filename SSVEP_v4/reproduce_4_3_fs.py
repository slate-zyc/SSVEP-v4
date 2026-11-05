"""复现论文第 4.3 节的 FS-eTRCA 与 FS-eTRCA&FBCCA：九分类（8 频率 + 空闲）。

4.3 节的任务：不给同步提示，直接判断受试者此刻看的是 8 个频率中的哪一个，
还是处于空闲状态（NS）。三种空闲状态 NS1/NS2/NS3 分别与 CS 组成独立任务。

频率特定框架（FS）的思路：
    1. 用 eTRCA 给测试片段打分，选出得分最高的候选频率；
    2. 用这个候选频率对应的一个二分类 LDA 判断“是 CS 还是 NS”；
    3. LDA 接受 → 输出该频率（0~7）；拒绝 → 输出 8（NS）。
两个版本共用候选频率，区别只在 LDA 输入特征：
    FS-eTRCA       ：该候选频率的 4 维 eTRCA 相关（4 个子带各一个相关系数）
    FS-eTRCA&FBCCA ：再加 4 维 FBCCA 相关（4 个子带各一个第一典型相关系），共 8 维

样本与分折：
    CS：8 频率 × 12 重复，每段取 0.14~1.14 秒 → 96 段
    NS：4 秒记录切 4 段 / 2 秒记录切 2 段 → 96 段
    每个任务 = 96 CS + 96 NS = 192 段；12 折，每折 8 CS + 8 NS，
    同一条 NS 原始记录的所有片段必须在同一折。

训练（每折只用外层训练部分）：
    - eTRCA 空间滤波器：用 11 个训练重复求 S w = λ Q w 的最大特征向量；
    - CS 正样本特征用“内部留一”：计算某训练重复的特征时，用排除它的
      10 重复模型，避免自己出现在自己的模板里；
    - 每个候选频率一个 LDA，正类 11 个 CS、负类 88 个 NS，先验都是 0.5。

数据流（代码从上到下就是执行顺序）：
    MAT → 9 电极 / 预处理 → CS 1 秒窗、NS 切 1 秒段
      → 4 子带滤波 → 384 段 = 96 CS + 3×96 NS
      → FBCCA 特征（384×8×4）
      → 每折：训练 eTRCA 模型 → 384 段 eTRCA 特征 → 选候选频率
      → 三种 NS × 两种方法各训练 8 个 LDA → 192 段预测

运行：python reproduce_4_3_fs.py
输出：results/section4_3_fs/ 下 5 个 CSV 和 1 张图
"""

from time import perf_counter

import matplotlib
matplotlib.use("Agg")               # 只把图存成文件，不弹窗口
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.signal import cheb1ord, cheby1, sosfiltfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from threadpoolctl import threadpool_limits

from config import (BAND_COUNT, CLASS_COUNT, DELAY_SECONDS, FREQUENCIES, FS,
                    FS_METHODS, HARMONIC_COUNT, NS_STATES, PAPER_FS_ACCURACY,
                    PASSBANDS, POSTERIOR_CHANNELS, SECTION4_3_FS_DIR,
                    STOPBANDS, SUBJECTS)
from data import load_cs_trials, load_ns_trials


# ===========================================================================
# 1. 固定参数、滤波器、FBCCA 参考基
# ===========================================================================
threadpool_limits(limits=1, user_api="blas")
started = perf_counter()
SECTION4_3_FS_DIR.mkdir(parents=True, exist_ok=True)

FOLDS = 12
START = round(DELAY_SECONDS * FS)        # 0.14 秒 = 35 点
n_channels = len(POSTERIOR_CHANNELS)     # 9

# 四个子带的 Chebyshev I 带通（与 4.2 相同）。
filter_sections = []
for passband, stopband in zip(PASSBANDS, STOPBANDS):
    order, critical = cheb1ord(passband, stopband, gpass=3, gstop=40, fs=FS)
    filter_sections.append(cheby1(order, rp=0.5, Wn=critical,
                                  btype="bandpass", fs=FS, output="sos"))

# 子带权重和 FBCCA 的参考基（8 个频率 × 5 阶谐波 × sin/cos），与 4.2 相同。
weights = np.arange(1, BAND_COUNT + 1, dtype=float) ** (-1.25) + 0.25
times = np.arange(1, FS + 1) / FS
reference_bases = []
for frequency in FREQUENCIES:
    phase = 2 * np.pi * frequency * np.arange(1, HARMONIC_COUNT + 1)[:, None] * times[None, :]
    reference = np.stack([np.sin(phase), np.cos(phase)], axis=1).reshape(
        2 * HARMONIC_COUNT, FS)
    reference -= reference.mean(axis=-1, keepdims=True)
    reference_basis, _ = np.linalg.qr(reference.T, mode="reduced")
    reference_bases.append(reference_basis)

prediction_rows = []    # 24 人 × 3 状态 × 2 方法 × 192 段 = 27648 条
metric_rows = []        # 24 人 × 3 状态 × 2 方法 = 144 行
confusion = np.zeros((2, 3, 9, 9), dtype=int)   # [方法, 状态, 真实, 预测]


# ===========================================================================
# 2. 逐人处理
# ===========================================================================
for subject in SUBJECTS:
    print(f"[{subject}] 读取九通道，计算两个 FS 方法", flush=True)

    # ---- CS：取 0.14~1.14 秒，再做四个子带滤波 ----
    cs = load_cs_trials(subject)                      # [8, 12, 9, 1125]
    cs = cs[..., START:START + FS]                    # [8, 12, 9, 250]
    cs_bank = np.stack([sosfiltfilt(sections, cs, axis=-1)
                        for sections in filter_sections])   # [4, 8, 12, 9, 250]

    # ---- NS：切成 1 秒段，再做四个子带滤波 ----
    banks_by_state = [cs_bank.reshape(BAND_COUNT, 96, n_channels, FS)]
    ns_groups_by_state = []       # 每段属于哪条原始记录
    ns_segments_by_state = []     # 每段在原始记录内的位置
    for state in NS_STATES:
        ns = load_ns_trials(subject, state)           # (记录数, 9, 1000 或 500)
        segments_per_trial = ns.shape[-1] // FS
        ns_groups_by_state.append(np.repeat(np.arange(len(ns)), segments_per_trial))
        ns_segments_by_state.append(np.tile(np.arange(segments_per_trial), len(ns)))
        # [记录, 通道, 段数, 250] → [96, 通道, 250]
        ns = ns.reshape(len(ns), n_channels, segments_per_trial, FS)
        ns = ns.transpose(0, 2, 1, 3).reshape(96, n_channels, FS)
        banks_by_state.append(np.stack([sosfiltfilt(sections, ns, axis=-1)
                                        for sections in filter_sections]))

    all_bank = np.concatenate(banks_by_state, axis=1)   # [4, 384, 9, 250]

    # ---- FBCCA 特征：每个片段、每个候选频率、每个子带的第一典型相关 ----
    # 参考信号是数学构造，不需要训练，所以这一项可以在折循环外一次算完。
    cca_features = np.empty((384, 8, BAND_COUNT))
    for sample_index in range(384):
        for band_index in range(BAND_COUNT):
            centered = (all_bank[band_index, sample_index]
                        - all_bank[band_index, sample_index].mean(axis=-1, keepdims=True))
            eeg_basis, _ = np.linalg.qr(centered.T, mode="reduced")   # (250, 9)
            for candidate_index, reference_basis in enumerate(reference_bases):
                cca_features[sample_index, candidate_index, band_index] = np.linalg.svd(
                    eeg_basis.T @ reference_basis, compute_uv=False)[0]

    subject_predictions = np.empty((2, 3, 192), dtype=int)   # [方法, 状态, 样本]

    # ---- 12 折外层交叉验证 ----
    for fold in range(FOLDS):
        keep = np.arange(FOLDS) != fold              # 本折的 11 个训练重复
        training_repeats = np.flatnonzero(keep)

        # 第 0 组是外层模型（11 重复）；后面 11 组各排除一个训练重复（10 重复），
        # 用于给 11 个训练 CS 生成“内部留一”的正类特征。
        fit_masks = [keep] + [keep & (np.arange(FOLDS) != repeat)
                              for repeat in training_repeats]

        # 对 12 组训练掩码统一训练模板和空间滤波器。
        templates = np.empty((12, BAND_COUNT, 8, n_channels, FS))
        spatial_filters = np.empty((12, BAND_COUNT, 8, n_channels))
        for fit_index, mask in enumerate(fit_masks):
            templates[fit_index] = cs_bank[:, :, mask].mean(axis=2)
            for band_index in range(BAND_COUNT):
                for candidate_index in range(CLASS_COUNT):
                    training = cs_bank[band_index, candidate_index, mask]  # (R, 9, 250)

                    # S：跨试次协方差（逐试次中心化后，不同试次的交叉乘积之和）。
                    centered_trials = training - training.mean(axis=-1, keepdims=True)
                    summed = centered_trials.sum(axis=0)
                    centered_flat = centered_trials.transpose(1, 0, 2).reshape(n_channels, -1)
                    inter_trial = summed @ summed.T - centered_flat @ centered_flat.T

                    # Q：全部训练数据按时间拼接后的总协方差（与作者 trca.m 一致）。
                    concatenated = training.transpose(1, 0, 2).reshape(n_channels, -1)
                    concatenated = concatenated - concatenated.mean(axis=-1, keepdims=True)
                    total_covariance = concatenated @ concatenated.T

                    _, eigenvectors = eigh(inter_trial, total_covariance,
                                           subset_by_index=[n_channels - 1, n_channels - 1])
                    filter_vector = eigenvectors[:, 0]
                    spatial_filters[fit_index, band_index, candidate_index] = (
                        filter_vector / np.linalg.norm(filter_vector))

        # ---- 用外层模型（fit_masks[0]）给 384 个片段算 eTRCA 特征 ----
        # 每个频带的 8 个滤波器拼成 (9,8) 矩阵，投影出 8 条序列后拉直，
        # 再和同样拉直、归一化的模板向量内积，得到二维 Pearson 相关。
        template_vectors = np.einsum("bkc,bfct->bfkt",
                                     spatial_filters[0], templates[0]).reshape(BAND_COUNT, 8, -1)
        template_vectors -= template_vectors.mean(axis=-1, keepdims=True)
        template_vectors /= np.linalg.norm(template_vectors, axis=-1, keepdims=True)

        projected = np.einsum("bkc,bnct->bnkt",
                              spatial_filters[0], all_bank).reshape(BAND_COUNT, 384, -1)
        projected -= projected.mean(axis=-1, keepdims=True)
        projected /= np.linalg.norm(projected, axis=-1, keepdims=True)

        etrca_features = np.stack(
            [projected[band_index] @ template_vectors[band_index].T
             for band_index in range(BAND_COUNT)], axis=-1)    # (384, 8, 4)

        # 用平方加权的相关选每个片段的候选频率（8 个总得分取最大）。
        candidate_classes = np.argmax(
            np.einsum("nfb,b->nf", etrca_features ** 2, weights), axis=1)

        # ---- 训练 CS 正样本特征（内部留一）----
        positive = np.empty((11, 8, BAND_COUNT))
        for position, repeat in enumerate(training_repeats):
            inner_test = np.einsum("bkc,bfct->bfkt",
                                   spatial_filters[position + 1],
                                   cs_bank[:, :, repeat]).reshape(BAND_COUNT, 8, -1)
            inner_template = np.einsum("bkc,bfct->bfkt",
                                       spatial_filters[position + 1],
                                       templates[position + 1]).reshape(BAND_COUNT, 8, -1)
            inner_test -= inner_test.mean(axis=-1, keepdims=True)
            inner_template -= inner_template.mean(axis=-1, keepdims=True)
            # 只取该试次真实频率那一列的四维相关，转成 (8 频率, 4 子带)。
            positive[position] = (np.sum(inner_test * inner_template, axis=-1)
                                  / (np.linalg.norm(inner_test, axis=-1)
                                     * np.linalg.norm(inner_template, axis=-1))).T

        positive_cca = np.stack(
            [cca_features[candidate_index * 12 + training_repeats, candidate_index]
             for candidate_index in range(8)], axis=1)          # (11, 8, 4)

        # ---- 三种 NS 分别训练两个 FS 版本 ----
        for state_index, state in enumerate(NS_STATES):
            state_indices = np.r_[np.arange(96),
                                  np.arange((state_index + 1) * 96,
                                            (state_index + 2) * 96)]
            state_features = etrca_features[state_indices]      # (192, 8, 4)
            state_cca = cca_features[state_indices]             # (192, 8, 4)

            labels = np.r_[np.repeat(np.arange(8), 12), np.full(96, 8)]   # 8 = NS
            original_trials = np.r_[np.arange(96), 96 + ns_groups_by_state[state_index]]
            segment_indices = np.r_[np.zeros(96, dtype=int),
                                    ns_segments_by_state[state_index]]
            assignments = np.r_[np.tile(np.arange(FOLDS), 8),
                                ns_groups_by_state[state_index] % FOLDS]

            ns_train = assignments[96:] != fold                 # 88 个训练 NS
            test = np.flatnonzero(assignments == fold)          # 16 个测试段
            candidates = candidate_classes[state_indices[test]] # 候选频率

            # 两种方法的特征只差“是否拼接 FBCCA 相关”。
            positives = [positive,
                         np.concatenate([positive, positive_cca], axis=-1)]
            negatives = [state_features[96:][ns_train],
                         np.concatenate([state_features[96:][ns_train],
                                         state_cca[96:][ns_train]], axis=-1)]
            tests = [state_features[test],
                     np.concatenate([state_features[test], state_cca[test]], axis=-1)]

            for method_index, (method, pos, neg, tst) in enumerate(
                    zip(FS_METHODS, positives, negatives, tests)):
                # 每个候选频率训练一个 LDA：11 个本频率 CS 正类 + 88 个 NS 负类。
                models = []
                for candidate_index in range(8):
                    training_features = np.concatenate(
                        [pos[:, candidate_index], neg[:, candidate_index]], axis=0)
                    training_labels = np.r_[np.ones(11, dtype=int),
                                            np.zeros(88, dtype=int)]
                    model = LinearDiscriminantAnalysis(priors=[0.5, 0.5])
                    model.fit(training_features, training_labels)
                    models.append(model)

                # 每个测试片段只看它自己候选频率对应的 LDA：接受 → 该频率，拒绝 → 8。
                decision_matrix = np.column_stack(
                    [model.decision_function(tst[:, candidate_index])
                     for candidate_index, model in enumerate(models)])   # (16, 8)
                decisions = decision_matrix[np.arange(len(test)), candidates]
                predicted = np.where(decisions > 0, candidates, 8)
                subject_predictions[method_index, state_index, test] = predicted

                for position, sample_index in enumerate(test):
                    prediction_rows.append({
                        "subject": subject, "state": state, "method": method,
                        "fold": fold + 1, "sample": sample_index,
                        "original_trial": original_trials[sample_index],
                        "segment": segment_indices[sample_index],
                        "true_class": labels[sample_index],
                        "candidate_class": candidates[position],
                        "predicted_class": predicted[position],
                        "lda_decision": decisions[position]})

        print(f"  外层折 {fold + 1}/12：两个 FS 版本、三种 NS 完成", flush=True)

    # ---- 当前人的指标：样本准确率、九类均衡准确率、状态检出率/误触发率 ----
    labels = np.r_[np.repeat(np.arange(8), 12), np.full(96, 8)]
    for method_index, method in enumerate(FS_METHODS):
        for state_index, state in enumerate(NS_STATES):
            predicted = subject_predictions[method_index, state_index]
            accuracy = np.mean(predicted == labels)
            recalls = np.array([np.mean(predicted[labels == target] == target)
                                for target in range(9)])
            balanced_accuracy = recalls.mean()
            binary_accuracy = np.mean((predicted != 8) == (labels != 8))
            cs_detection = np.mean(predicted[:96] != 8)
            ns_false_positive = np.mean(predicted[96:] != 8)
            np.add.at(confusion[method_index, state_index], (labels, predicted), 1)
            metric_rows.append({
                "subject": subject, "state": state, "method": method,
                "accuracy": accuracy, "balanced_accuracy": balanced_accuracy,
                "binary_accuracy": binary_accuracy,
                "cs_detection_rate": cs_detection,
                "ns_false_positive_rate": ns_false_positive,
                "n_classes": 9, "n_samples": len(labels)})
            print(f"  {method} CS-{state}: accuracy={accuracy:.2%}, "
                  f"balanced={balanced_accuracy:.2%}", flush=True)


# ===========================================================================
# 3. 汇总、写 CSV、画图
# ===========================================================================
predictions = pd.DataFrame(prediction_rows)
metrics = pd.DataFrame(metric_rows)

summary = metrics.groupby(["method", "state"], as_index=False).agg(
    accuracy_mean=("accuracy", "mean"),
    accuracy_sd=("accuracy", "std"),
    balanced_accuracy_mean=("balanced_accuracy", "mean"),
    binary_accuracy_mean=("binary_accuracy", "mean"),
    cs_detection_mean=("cs_detection_rate", "mean"),
    ns_false_positive_mean=("ns_false_positive_rate", "mean"),
)
summary["accuracy_sem"] = summary.accuracy_sd / np.sqrt(len(SUBJECTS))

# 原文只报告样本准确率，所以对照表只比这一个指标。
comparison = summary[["method", "state", "accuracy_mean"]].rename(
    columns={"accuracy_mean": "reproduced"})
comparison["reproduced"] *= 100
comparison["paper"] = [PAPER_FS_ACCURACY[(row.method, row.state)]
                       for row in comparison.itertuples()]
comparison["difference_percentage_points"] = comparison.reproduced - comparison.paper

# 六个 9×9 混淆矩阵展平成 486 行：0~7 是频率，8 是 NS。
confusion_table = pd.DataFrame({
    "method": np.repeat(FS_METHODS, 3 * 81),
    "state": np.tile(np.repeat(NS_STATES, 81), 2),
    "true_class": np.tile(np.repeat(np.arange(9), 9), 6),
    "predicted_class": np.tile(np.arange(9), 6 * 9),
    "count": confusion.ravel(),
})

predictions.to_csv(SECTION4_3_FS_DIR / "predictions.csv", index=False)
metrics.to_csv(SECTION4_3_FS_DIR / "subject_metrics.csv", index=False)
summary.to_csv(SECTION4_3_FS_DIR / "summary.csv", index=False)
comparison.to_csv(SECTION4_3_FS_DIR / "comparison_with_paper.csv", index=False)
confusion_table.to_csv(SECTION4_3_FS_DIR / "confusion_matrices.csv", index=False)

# 图：左上跨人均值 + 原文点，其余三格是三种 NS 的个体分布。
plt.rcParams.update({"font.size": 10,
                     "axes.spines.top": False,
                     "axes.spines.right": False})
figure, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
figure.suptitle("Section 4.3 | Frequency-specific methods | 9-class CS/NS", fontsize=14)
positions = np.arange(3)
colors = ["#2074a5", "#e29b35"]
for method_index, method in enumerate(FS_METHODS):
    current = summary.loc[summary.method == method].set_index("state").loc[NS_STATES]
    locations = positions + (method_index - 0.5) * 0.34
    axes[0, 0].bar(locations, current.accuracy_mean * 100, width=0.32,
                   yerr=current.accuracy_sem * 100, capsize=3,
                   color=colors[method_index], label=method)
    axes[0, 0].scatter(locations,
                       [PAPER_FS_ACCURACY[(method, state)] for state in NS_STATES],
                       marker="D", facecolors="white", edgecolors=colors[method_index],
                       s=60, zorder=4, label=f"Paper {method}")
axes[0, 0].set(xticks=positions, xticklabels=NS_STATES, ylabel="9-class accuracy (%)",
               ylim=(0, 105), title="(a) Mean accuracy +/- SEM: 24 subjects")
axes[0, 0].legend(fontsize=8, loc="lower right")

for axis, state in zip(axes.ravel()[1:], NS_STATES):
    axis.boxplot([metrics.loc[(metrics.method == method) & (metrics.state == state),
                              "accuracy"].to_numpy() * 100 for method in FS_METHODS],
                 tick_labels=FS_METHODS, showmeans=True)
    axis.set(ylabel="9-class accuracy (%)", ylim=(0, 105),
             title=f"CS versus {state}: individual accuracy")

figure.savefig(SECTION4_3_FS_DIR / "fs_accuracy.png", dpi=200)
plt.close(figure)

print(comparison.to_string(index=False, float_format="%.4f"), flush=True)
print(f"完成：{SECTION4_3_FS_DIR}\n用时：{perf_counter() - started:.1f} 秒")
