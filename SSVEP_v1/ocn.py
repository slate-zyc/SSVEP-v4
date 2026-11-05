"""提取复现版 IFB-OCN 的递归网络特征，并在训练折内选择个体化频带。

已核实的结构：Fp1 单通道 -> 延迟嵌入 -> 距离阈值网络 -> 平均度和聚类系数
-> 最佳三个频带 -> 线性 SVM。原始 2020 方法的完整优化细节尚未取得，
因此这里明确采用训练内网格搜索和逐片段标准化，结果标为 IFB-OCN-reconstructed，
不能当成原作者实现的逐参数复刻。测试折标签绝不参与参数或频带选择。
"""

from itertools import product

import numpy as np
from scipy.spatial.distance import cdist
from scipy.signal import butter, sosfiltfilt
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def network_statistics(adjacency):
    """计算无向、无自环二值图的平均度与平均局部聚类系数。

    每个节点的 (A² * A) 行和等于其参与的三角形数的两倍；除以 k(k-1)
    即局部聚类系数。度小于 2 的节点贡献设为 0，仍包含在全节点平均中。
    """
    adjacency = np.asarray(adjacency, dtype=np.float64).copy()
    np.fill_diagonal(adjacency, 0)
    degree = adjacency.sum(axis=1)
    closed_paths = ((adjacency @ adjacency) * adjacency).sum(axis=1)
    denominator = degree * (degree - 1)
    clustering = np.divide(closed_paths, denominator, out=np.zeros_like(degree), where=denominator > 0)
    return np.array([degree.mean(), clustering.mean()])


def recurrence_features(signal, delay, dimension, radius):
    """用逐片段单位标准差信号构建欧氏距离递归网络，输出两个网络统计量。"""
    signal = np.asarray(signal, dtype=float)
    scale = signal.std()
    if scale <= np.finfo(float).eps:
        return np.zeros(2)
    standardized = (signal - signal.mean()) / scale
    node_count = len(signal) - (dimension - 1) * delay
    if node_count < 3:
        raise ValueError("Embedding leaves too few recurrence-network nodes")
    offsets = np.arange(dimension) * delay
    embedded = standardized[np.arange(node_count)[:, None] + offsets]
    # 每个网络节点是一条延迟状态向量，不是一个 EEG 电极。
    distances = cdist(embedded, embedded, metric="euclidean")
    return network_statistics(distances <= radius)


def candidate_parameters(config):
    """返回事先固定的参数候选，后续仅用内层训练验证挑选。"""
    return list(product(config["ocn_delays"], config["ocn_dimensions"], config["ocn_radii"]))


def extract_candidates(config, signals):
    """对 [片段, 时间] 输入提取 [片段, 频带, 参数候选, 两个特征]。

    这里的逐片段归一化和网络构图都不使用任何标签，故可在交叉验证前预计算。
    监督的参数选择和频带排序必须留在 evaluate_ocn_fold 内。
    """
    parameters = candidate_parameters(config)
    features = np.empty((len(signals), len(config["ocn_bands"]), len(parameters), 2))
    # 五个候选频带依据同一研究团队2023年混合方法论文的 IFBOCN 部分；
    # 网络参数网格与逐片段标准化仍是本复现版的显式选择。
    for band_index, band in enumerate(config["ocn_bands"]):
        sections = butter(4, band, fs=config["fs"], btype="bandpass", output="sos")
        filtered = sosfiltfilt(sections, signals, axis=-1)
        for sample_index, signal in enumerate(filtered):
            for parameter_index, (delay, dimension, radius) in enumerate(parameters):
                features[sample_index, band_index, parameter_index] = recurrence_features(
                    signal, delay, dimension, radius
                )
    return features


def svm(config):
    """标准化与线性 SVM 放在同一管线，避免用验证或测试样本拟合缩放参数。"""
    return make_pipeline(StandardScaler(), SVC(kernel="linear", C=config["ocn_svm_c"]))


def evaluate_ocn_fold(config, features, labels, groups, train, test):
    """在外层训练数据内部选参数和三个频带，返回外层预测及选择日志。"""
    splitter = StratifiedGroupKFold(n_splits=config["ocn_inner_folds"], shuffle=True,
                                   random_state=config["seed"])
    inner_folds = list(splitter.split(features[train], labels[train], groups[train]))
    best_parameters, best_accuracies = [], []
    for band_index in range(features.shape[1]):
        accuracies = []
        for parameter_index in range(features.shape[2]):
            candidate = features[train, band_index, parameter_index]
            correct, total = 0, 0
            for fit_indices, validation_indices in inner_folds:
                model = svm(config)
                model.fit(candidate[fit_indices], labels[train][fit_indices])
                predicted = model.predict(candidate[validation_indices])
                correct += int(np.sum(predicted == labels[train][validation_indices]))
                total += len(validation_indices)
            accuracies.append(correct / total)
        # 平分时按预先固定的候选顺序选择，避免结果受不稳定排序影响。
        best = int(np.argmax(accuracies))
        best_parameters.append(best)
        best_accuracies.append(accuracies[best])
    selected = np.argsort(-np.array(best_accuracies), kind="stable")[:config["ocn_top_bands"]]
    combined = np.concatenate([features[:, band, best_parameters[band]] for band in selected], axis=1)
    model = svm(config)
    model.fit(combined[train], labels[train])
    details = [dict(band_index=int(band), band=config["ocn_bands"][band],
                    parameters=candidate_parameters(config)[best_parameters[band]],
                    inner_accuracy=float(best_accuracies[band])) for band in selected]
    return model.predict(combined[test]), details
