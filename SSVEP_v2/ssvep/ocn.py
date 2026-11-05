"""IFB-OCN 特征：把一条单通道 EEG 变成“递归网络”的两个统计量，再用线性 SVM 分类。

整体流程（原论文的 IFB-OCN 思路）：

    单通道信号
      → 按候选频带滤波
      → 延迟嵌入：把一维时间序列铺成高维状态点
      → 距离阈值建图：状态点之间距离足够近就连一条边
      → 每个网络算两个统计量：平均度、平均聚类系数
      → 每个（频带, 延迟, 维度, 半径）组合得到 2 维特征
      → 在训练折内部用交叉验证挑出最好的 3 个频带和各自的网络参数
      → 把这些特征拼起来训练线性 SVM

为什么这样做：单通道也能构造“网络”，节点是延迟状态向量而不是电极；
不同频带对应不同节律，不同延迟/维度/半径对应不同的时间尺度，
用训练数据挑组合比人工指定更稳。

说明：本复现明确采用训练内网格搜索和逐片段标准化；原论文的完整优化细节
未取得，因此结果应标为 IFB-OCN-reconstructed，测试折标签绝不参与选择。
"""

from __future__ import annotations

from itertools import product

import numpy as np
from scipy.spatial.distance import cdist
from scipy.signal import butter, sosfiltfilt
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# ---------------------------------------------------------------------------
# 1. 网络统计量：平均度与平均聚类系数
# ---------------------------------------------------------------------------

def network_statistics(adjacency: np.ndarray) -> np.ndarray:
    """输入二值邻接矩阵 [节点, 节点]，返回 [平均度, 平均聚类系数]。

    这是无向、无自环的简单图。聚类系数的含义是“邻居之间也互相连接的比例”：
        节点 i 的局部聚类系数 = (A²·A)_ii / (k_i·(k_i-1))
    其中 (A²·A)_ii 是节点 i 参与的三角形数的两倍，k_i 是它的度。
    度小于 2 的节点没有“邻居对”，局部聚类系数记 0，但仍计入平均。
    """
    adjacency = np.asarray(adjacency, dtype=np.float64).copy()
    np.fill_diagonal(adjacency, 0)

    degree = adjacency.sum(axis=1)
    triangles_twice = ((adjacency @ adjacency) * adjacency).sum(axis=1)
    denominator = degree * (degree - 1)
    clustering = np.divide(triangles_twice, denominator,
                           out=np.zeros_like(degree), where=denominator > 0)
    return np.array([degree.mean(), clustering.mean()])


# ---------------------------------------------------------------------------
# 2. 递归网络特征：延迟嵌入 → 距离阈值图
# ---------------------------------------------------------------------------

def recurrence_features(signal: np.ndarray, delay: int,
                        dimension: int, radius: float) -> np.ndarray:
    """输入单通道信号 [时间点]，返回 [平均度, 平均聚类系数] 两个统计量。

    做法（递归网络 / recurrence network）：
        1. 先对片段做单位标准差归一化，这样阈值 radius 与信号幅度无关；
        2. 延迟嵌入：第 i 个状态点是 (x[i], x[i+delay], ..., x[i+(dimension-1)·delay])；
        3. 若两个状态点的欧氏距离 ≤ radius，就在它们之间连一条边；
        4. 对得到的图调用 network_statistics。

    radius 越小图越稀疏（高阈值更严格），平均度低；radius 越大越接近全连接。
    """
    signal = np.asarray(signal, dtype=float)
    scale = signal.std()
    if scale <= np.finfo(float).eps:
        return np.zeros(2)          # 常数信号（无变化）无法建图
    standardized = (signal - signal.mean()) / scale

    node_count = len(signal) - (dimension - 1) * delay
    if node_count < 3:
        raise ValueError("延迟嵌入后节点太少，无法计算聚类系数")

    offsets = np.arange(dimension) * delay
    embedded = standardized[np.arange(node_count)[:, None] + offsets]   # [节点, dimension]
    distances = cdist(embedded, embedded, metric="euclidean")
    return network_statistics(distances <= radius)


# ---------------------------------------------------------------------------
# 3. 候选参数与特征提取
# ---------------------------------------------------------------------------

def candidate_parameters(config: dict) -> list[tuple[int, int, float]]:
    """返回所有 (延迟, 嵌入维度, 距离阈值) 组合。

    用 itertools.product 按“延迟 → 维度 → 半径”的顺序取笛卡尔积；
    顺序固定后，内层验证出现平局时 np.argmax 取第一个，结果可复现。
    """
    settings = config["ocn"]
    return list(product(settings["delays"], settings["dimensions"], settings["radii"]))


def band_filter(config: dict, band: list[float]) -> np.ndarray:
    """为一个候选频带设计 Butterworth 带通滤波器，返回二阶节（sos）。

    用 sos 形式是因为它数值稳定，sosfiltfilt 再做零相位滤波（正反各滤一次）。
    """
    settings = config["ocn"]
    return butter(settings["filter_order"], band, fs=config["eeg"]["fs"],
                  btype="bandpass", output="sos")


def extract_features(config: dict, signals: np.ndarray) -> np.ndarray:
    """输入连续信号片段 [片段, 时间]，输出 [片段, 频带, 参数候选, 2]。

    每个片段的归一化和建图都不看标签，所以可以一次性提前算好，不泄漏信息。
    需要标签的部分（选频带、选网络参数、训练分类器）全部留在 evaluate_ocn_fold。
    """
    parameters = candidate_parameters(config)
    bands = config["ocn"]["bands"]
    features = np.empty((len(signals), len(bands), len(parameters), 2))

    for band_index, band in enumerate(bands):
        filtered = sosfiltfilt(band_filter(config, band), signals, axis=-1)
        for sample_index, signal in enumerate(filtered):
            for parameter_index, (delay, dimension, radius) in enumerate(parameters):
                features[sample_index, band_index, parameter_index] = recurrence_features(
                    signal, delay, dimension, radius)
    return features


# ---------------------------------------------------------------------------
# 4. 训练折内选参数 + 线性 SVM
# ---------------------------------------------------------------------------

def make_svm(config: dict):
    """返回 StandardScaler + 线性 SVC 的管线。

    为什么缩放和 SVM 放在同一管线：缩放参数（均值/标准差）只能用训练折估计，
    否则验证/测试数据的信息会漏进训练过程。
    """
    return make_pipeline(StandardScaler(), SVC(kernel="linear", C=config["ocn"]["svm_c"]))


def inner_accuracy(config: dict, candidate: np.ndarray, inner_labels: np.ndarray,
                   inner_folds: list) -> float:
    """在训练折内部做交叉验证，返回一组特征的平均准确率。

    candidate    : [训练样本, 2]  一个（频带, 参数）组合下的网络特征
    inner_labels : [训练样本]     对应的类别标签
    inner_folds  : StratifiedGroupKFold 给出的 (训练行号, 验证行号) 对
    """
    correct, total = 0, 0
    for fit_rows, validation_rows in inner_folds:
        model = make_svm(config)
        model.fit(candidate[fit_rows], inner_labels[fit_rows])
        predicted = model.predict(candidate[validation_rows])
        correct += int(np.sum(predicted == inner_labels[validation_rows]))
        total += len(validation_rows)
    return correct / total


def evaluate_ocn_fold(config: dict, features: np.ndarray, labels: np.ndarray,
                      groups: np.ndarray, train: np.ndarray,
                      test: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    """在外层训练数据内部选参数和最好的若干频带，返回 (预测, 选择记录)。

    参数和频带只在一组固定的内层折上比较，测试折不参与任何选择。
    details 每项：band_index、band（频率区间）、parameters（延迟, 维度, 半径）、
    inner_accuracy（该组合在内层验证上的准确率）。
    """
    settings = config["ocn"]
    splitter = StratifiedGroupKFold(n_splits=settings["inner_folds"], shuffle=True,
                                    random_state=config["random_seed"])
    inner_folds = list(splitter.split(features[train], labels[train], groups[train]))
    inner_labels = labels[train]

    best_parameters, best_accuracies = [], []
    for band_index in range(features.shape[1]):
        accuracies = [
            inner_accuracy(config, features[train, band_index, parameter_index],
                           inner_labels, inner_folds)
            for parameter_index in range(features.shape[2])
        ]
        # 平分时 argmax 取第一个，也就是候选顺序里靠前的那个。
        best = int(np.argmax(accuracies))
        best_parameters.append(best)
        best_accuracies.append(accuracies[best])

    # 按内层准确率从高到低取前 top_bands 个频带；stable 保证平分时保持原顺序。
    selected = np.argsort(-np.array(best_accuracies), kind="stable")[:settings["top_bands"]]
    combined = np.concatenate(
        [features[:, band_index, best_parameters[band_index]] for band_index in selected],
        axis=1)

    model = make_svm(config)
    model.fit(combined[train], labels[train])
    details = [dict(band_index=int(band_index),
                    band=settings["bands"][band_index],
                    parameters=candidate_parameters(config)[best_parameters[band_index]],
                    inner_accuracy=float(best_accuracies[band_index]))
               for band_index in selected]
    return model.predict(combined[test]), details
