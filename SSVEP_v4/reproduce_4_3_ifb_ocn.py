"""复现论文第 4.3 节的 IFB-OCN（重建版）：CS 与 NS 的二分类。

4.3 节的任务：判断受试者此刻是“连续闪烁（CS）”还是“空闲（NS）”，
并分别对 NS1、NS2、NS3 三种空闲状态各做一次二分类。

IFB-OCN 的做法：
    单通道 Fp1 → 按候选频带滤波 → 延迟嵌入 → 距离阈值建图
    → 每个网络算平均度和平均聚类系数 → 训练内选 3 个频带 → 线性 SVM

“重建版”的边界：原始 2020 论文的完整参数优化细节不可得，所以这里的
5 个候选频带、8 组网络参数、逐片段标准化和内部三折验证都是本实现
明确采用的替代规则；输出方法和文件名都带 reconstructed 标识。

数据流（代码从上到下就是执行顺序）：
    CS : 取 Fp1，0.14~1.14 秒 → 96 段（8 频率 × 12 重复）
    NS : 每条原始记录切成不重叠 1 秒段 → 96 段（24×4 或 48×2）
    每个状态 = 96 CS + 96 NS = 192 段
    → 384 段统一提取 [5 频带, 8 参数, 2 统计量] 候选特征
    → 12 折外层：每折在训练段里做 3 折内层选参数/选频带，训练线性 SVM

运行：python reproduce_4_3_ifb_ocn.py
输出：results/section4_3_ifb_ocn/ 下 6 个 CSV 和 1 张图
"""

from itertools import product
from time import perf_counter

import matplotlib
matplotlib.use("Agg")               # 只把图存成文件，不弹窗口
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from scipy.spatial.distance import cdist
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from config import (DELAY_SECONDS, FRONTAL_CHANNEL, FS, NS_STATES,
                    OCN_BANDS, OCN_DELAYS, OCN_DIMENSIONS, OCN_INNER_FOLDS,
                    OCN_INNER_SEED, OCN_METHOD, OCN_RADII, OCN_SVM_C,
                    PAPER_OCN_ACCURACY, SECTION4_3_IFB_OCN_DIR, SUBJECTS)
from data import load_cs_trials, load_ns_trials


# ===========================================================================
# 1. 固定参数与结果容器
# ===========================================================================
threadpool_limits(limits=1, user_api="blas")   # 网络只有约 240 个节点，单线程更快也更稳定
started = perf_counter()
SECTION4_3_IFB_OCN_DIR.mkdir(parents=True, exist_ok=True)

FOLDS = 12                                   # 外层 12 折
START = round(DELAY_SECONDS * FS)            # 0.14 秒 = 35 点：CS 截窗起点

# 8 组网络候选参数：4 组(延迟, 嵌入维度) × 2 个距离半径。
EMBEDDINGS = list(product(OCN_DELAYS, OCN_DIMENSIONS))
PARAMETERS = [(delay, dimension, radius)
              for delay, dimension in EMBEDDINGS
              for radius in OCN_RADII]

prediction_rows = []    # 24 人 × 3 NS状态 × 192 段 = 13824 条外层测试预测
metric_rows = []        # 24 人 × 3 NS状态 = 72 行指标
selection_rows = []     # 每折选中的 3 个频带及其参数，共 24×3×12×3 = 2592 行
confusion = np.zeros((3, 2, 2), dtype=int)   # 三种状态各一个 2×2 混淆矩阵（行真、列预测）


# ===========================================================================
# 2. 逐人处理
# ===========================================================================
for subject in SUBJECTS:
    print(f"[{subject}] 读取 Fp1，计算 IFB-OCN 重建版", flush=True)

    # ---- CS：只取 Fp1，截 0.14~1.14 秒，展平成 96 段 ----
    cs = load_cs_trials(subject, [FRONTAL_CHANNEL])     # [8 频率, 12 重复, 1 通道, 1125]
    cs = cs[..., START:START + FS].reshape(96, FS)      # [96 段, 250 点]，顺序为类别优先

    signals_by_state = [cs]          # 第 0 组是 CS，后三组依次是 NS1/NS2/NS3 保存当前人的一秒片段
    ns_groups_by_state = []          # 每段属于第几条原始记录（分组用）
    ns_segments_by_state = []        # 每段是原始记录内的第几个 1 秒段（追溯用）

    for state in NS_STATES:
        ns = load_ns_trials(subject, state, [FRONTAL_CHANNEL])[:, 0, :]   # (记录数, 1000 或 500)
        segments_per_trial = ns.shape[-1] // FS      # 4 秒记录切 4 段，2 秒记录切 2 段

        # 同一条原始记录的所有片段共享组号；segment 记录段内位置。
        ns_groups_by_state.append(np.repeat(np.arange(len(ns)), segments_per_trial))
        ns_segments_by_state.append(np.tile(np.arange(segments_per_trial), len(ns)))
        signals_by_state.append(ns.reshape(96, FS))     # 直接按每条记录的时间顺序切成 96 段，不重叠、不丢弃最后一秒

    signals = np.concatenate(signals_by_state, axis=0)   # [384 段, 250 点]   96*4=384

    # ---- 384 段统一提特征：[片段, 5 频带, 8 参数, 2 统计量] ----
    features = np.empty((384, len(OCN_BANDS), len(PARAMETERS), 2))
    for band_index, band in enumerate(OCN_BANDS):
        sections = butter(4, band, btype="bandpass", fs=FS, output="sos")
        filtered = sosfiltfilt(sections, signals, axis=-1)   # 零相位带通

        for sample_index, signal in enumerate(filtered):
            # 逐片段标准化：只用当前片段的均值和标准差，不看标签。
            standardized = (signal - signal.mean()) / signal.std()

            # 同一嵌入下两个半径共用一次距离计算，所以 distances 放在半径循环外。
            for embedding_index, (delay, dimension) in enumerate(EMBEDDINGS):
                node_count = FS - (dimension - 1) * delay
                # 每个节点是一个延迟状态向量 (x_i, x_{i+tau}, ..., x_{i+(m-1)tau})。
                embedded = standardized[
                    np.arange(node_count)[:, None] + np.arange(dimension)[None, :] * delay]
                distances = cdist(embedded, embedded, metric="euclidean")

                for radius_index, radius in enumerate(OCN_RADII):
                    adjacency = (distances <= radius).astype(float)   # 距离≤半径则连边
                    np.fill_diagonal(adjacency, 0)                    # 无自环
                    degrees = adjacency.sum(axis=1)                   # 每个节点的度

                    # (A²·A)_ii 等于节点 i 参与的三角形数的两倍；
                    # 除以 k(k-1) 就是局部聚类系数，度<2 的节点记 0。
                    closed_paths = ((adjacency @ adjacency) * adjacency).sum(axis=1)
                    clustering = np.divide(closed_paths, degrees * (degrees - 1),
                                           out=np.zeros_like(degrees), where=degrees > 1)

                    parameter_index = embedding_index * len(OCN_RADII) + radius_index
                    features[sample_index, band_index, parameter_index] = [
                        degrees.mean(), clustering.mean()]
        print(f"  网络特征 {band[0]:g}-{band[1]:g} Hz 完成", flush=True)

    # ---- 对三种 NS 分别做二分类 ----
    for state_index, state in enumerate(NS_STATES):
        # 当前任务 = CS 的 96 段 + 这一种 NS 的 96 段。
        inputs = np.concatenate([features[:96],
                                 features[(state_index + 1) * 96:(state_index + 2) * 96]])
        labels = np.r_[np.ones(96, dtype=int), np.zeros(96, dtype=int)]   # 1=CS，0=NS
        original_trials = np.r_[np.arange(96), 96 + ns_groups_by_state[state_index]]
        segment_indices = np.r_[np.zeros(96, dtype=int), ns_segments_by_state[state_index]]

        # 分折：CS 按重复编号（0~11 循环），NS 按原始记录编号 mod 12；
        # 每折恰好 8 个 CS + 8 个 NS，且同一条 NS 记录的所有片段在同一折。
        assignments = np.r_[np.tile(np.arange(FOLDS), 8),
                            ns_groups_by_state[state_index] % FOLDS]
        predicted = np.empty(192, dtype=int)

        for fold in range(FOLDS):
            train = np.flatnonzero(assignments != fold)   # 176 个训练段
            test = np.flatnonzero(assignments == fold)    # 16 个测试段

            # 内层 3 折只在 176 个训练段里划分，并且按原始记录分组。
            splitter = StratifiedGroupKFold(n_splits=OCN_INNER_FOLDS, shuffle=True,
                                            random_state=OCN_INNER_SEED)
            inner_splits = list(splitter.split(inputs[train], labels[train],
                                               original_trials[train]))

            # 每个候选频带选一组网络参数（内层验证准确率最高的一组）。
            best_parameters = np.empty(len(OCN_BANDS), dtype=int)
            band_accuracies = np.empty(len(OCN_BANDS))
            for band_index in range(len(OCN_BANDS)):
                parameter_accuracies = []
                for parameter_index in range(len(PARAMETERS)):
                    candidate = inputs[train, band_index, parameter_index]   # [176, 2]
                    correct = 0
                    for fit_rows, validation_rows in inner_splits:
                        model = make_pipeline(StandardScaler(),
                                              SVC(kernel="linear", C=OCN_SVM_C))
                        model.fit(candidate[fit_rows], labels[train][fit_rows])
                        correct += np.count_nonzero(
                            model.predict(candidate[validation_rows])
                            == labels[train][validation_rows])
                    parameter_accuracies.append(correct / len(train))
                best_parameters[band_index] = np.argmax(parameter_accuracies)
                band_accuracies[band_index] = parameter_accuracies[best_parameters[band_index]]

            # 按内层准确率选前 3 个频带，拼成 6 维特征。
            selected = np.argsort(-band_accuracies, kind="stable")[:3]
            combined = np.concatenate(
                [inputs[:, band_index, best_parameters[band_index]]
                 for band_index in selected], axis=1)

            model = make_pipeline(StandardScaler(), SVC(kernel="linear", C=OCN_SVM_C))
            model.fit(combined[train], labels[train])
            predicted[test] = model.predict(combined[test])
            decisions = model.decision_function(combined[test])

            # 记录每折选中的 3 个频带、参数和内层准确率，便于审查。
            for rank, band_index in enumerate(selected):
                delay, dimension, radius = PARAMETERS[best_parameters[band_index]]
                selection_rows.append({
                    "subject": subject, "state": state, "fold": fold + 1,
                    "rank": rank + 1,
                    "low_hz": OCN_BANDS[band_index][0], "high_hz": OCN_BANDS[band_index][1],
                    "delay_samples": delay, "dimension": dimension, "radius": radius,
                    "inner_accuracy": band_accuracies[band_index]})

            # 每条预测都保存来源（原始记录、段内位置）和 SVM 决策值。
            for sample_index, decision in zip(test, decisions):
                prediction_rows.append({
                    "subject": subject, "state": state, "method": OCN_METHOD,
                    "fold": fold + 1, "sample": sample_index,
                    "original_trial": original_trials[sample_index],
                    "segment": segment_indices[sample_index],
                    "true_class": labels[sample_index],
                    "predicted_class": predicted[sample_index],
                    "decision_value": decision})

        # 当前人当前状态的指标：准确率、CS 检出率（TPR）、NS 误触发率（FPR）。
        accuracy = np.mean(predicted == labels)
        cs_detection = np.mean(predicted[:96] == 1)
        ns_false_positive = np.mean(predicted[96:] == 1)
        np.add.at(confusion[state_index], (labels, predicted), 1)
        metric_rows.append({
            "subject": subject, "state": state, "method": OCN_METHOD,
            "accuracy": accuracy, "cs_detection_rate": cs_detection,
            "ns_false_positive_rate": ns_false_positive,
            "n_classes": 2, "n_samples": len(labels)})
        print(f"  CS-{state}: accuracy={accuracy:.2%}, TPR={cs_detection:.2%}, "
              f"FPR={ns_false_positive:.2%}", flush=True)


# ===========================================================================
# 3. 汇总、写 CSV、画图
# ===========================================================================
predictions = pd.DataFrame(prediction_rows)
metrics = pd.DataFrame(metric_rows)

summary = metrics.groupby("state", as_index=False).agg(
    accuracy_mean=("accuracy", "mean"),
    accuracy_sd=("accuracy", "std"),
    cs_detection_mean=("cs_detection_rate", "mean"),
    ns_false_positive_mean=("ns_false_positive_rate", "mean"),
)
summary["accuracy_sem"] = summary.accuracy_sd / np.sqrt(len(SUBJECTS))

# 原文只给出平均准确率，所以对照表只比这一个指标。
comparison = summary[["state", "accuracy_mean"]].rename(
    columns={"accuracy_mean": "reproduced"})
comparison["method"] = OCN_METHOD
comparison["reproduced"] *= 100
comparison["paper"] = PAPER_OCN_ACCURACY
comparison["difference_percentage_points"] = comparison.reproduced - comparison.paper

# 三个 2×2 混淆矩阵展平成 12 行：0=NS，1=CS。
confusion_table = pd.DataFrame({
    "state": np.repeat(NS_STATES, 4),
    "true_class": np.tile(np.repeat([0, 1], 2), 3),
    "predicted_class": np.tile([0, 1], 6),
    "count": confusion.ravel(),
})

predictions.to_csv(SECTION4_3_IFB_OCN_DIR / "predictions.csv", index=False)
metrics.to_csv(SECTION4_3_IFB_OCN_DIR / "subject_metrics.csv", index=False)
summary.to_csv(SECTION4_3_IFB_OCN_DIR / "summary.csv", index=False)
comparison.to_csv(SECTION4_3_IFB_OCN_DIR / "comparison_with_paper.csv", index=False)
pd.DataFrame(selection_rows).to_csv(SECTION4_3_IFB_OCN_DIR / "selected_bands.csv", index=False)
confusion_table.to_csv(SECTION4_3_IFB_OCN_DIR / "confusion_matrices.csv", index=False)

# 图：左图跨人均值 + 原文参考点，右图 24 人个体分布。
plt.rcParams.update({"font.size": 10,
                     "axes.spines.top": False,
                     "axes.spines.right": False})
figure, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
figure.suptitle("Section 4.3 | IFB-OCN reconstructed | Binary CS/NS", fontsize=14)
positions = np.arange(3)
axes[0].bar(positions, summary.accuracy_mean * 100, yerr=summary.accuracy_sem * 100,
            capsize=4, color="#2074a5", label="Reproduced mean +/- SEM")
axes[0].scatter(positions, PAPER_OCN_ACCURACY, marker="D", facecolors="none",
                edgecolors="#d94b46", s=65, label="Paper mean")
axes[0].set(xticks=positions, xticklabels=NS_STATES, ylabel="Binary accuracy (%)",
            ylim=(0, 105), title="(a) CS versus each NS state")
axes[0].legend(fontsize=9)
axes[1].boxplot([metrics.loc[metrics.state == state, "accuracy"].to_numpy() * 100
                 for state in NS_STATES], tick_labels=NS_STATES, showmeans=True)
axes[1].set(ylabel="Binary accuracy (%)", ylim=(0, 105),
            title="(b) Accuracy distribution: 24 subjects")
figure.savefig(SECTION4_3_IFB_OCN_DIR / "ifb_ocn_accuracy.png", dpi=200)
plt.close(figure)

print(comparison.to_string(index=False, float_format="%.4f"), flush=True)
print(f"完成：{SECTION4_3_IFB_OCN_DIR}\n用时：{perf_counter() - started:.1f} 秒")
