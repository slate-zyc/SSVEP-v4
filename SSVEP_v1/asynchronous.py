"""复现 CS 与 NS1/NS2/NS3 的离线状态分类，保存原始试次可追溯预测。

FS-eTRCA、FS-eTRCA&FBCCA 输出 0..7 频率类别或 8=NS；IFB-OCN 重建版 输出二分类。
采用 12 折分组验证，同一 NS 原始试次的所有 1 秒窗保持同折。
训练 CS 的模板相关特征由内部留一生成，避免样本同时作为自身模板导致虚高。
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis    # LDA 分类器
from sklearn.metrics import accuracy_score, balanced_accuracy_score     # 准确率，平衡准确率 = 每个类别的召回率的平均

from algorithms import (fbcca_correlations, filterbank, fit_trca, reference_bases,
                        trca_correlations, weights)
from data_io import cache_key, crop_cs, load_preprocessed, mat_path, segment_ns
from ocn import evaluate_ocn_fold, extract_candidates


def outer_assignments(config, cs_repeats, ns_groups):
    """CS 每频率重复编号决定折；NS 原始试次按模分折，保证每折 8 CS+8 NS。

    CS 按其重复编号成折：第 f 折取每个频率的第 f 次重复，故每折恰好 8 个 CS（8 个频率各一个）。
    NS 按原始试次组号对折数取模，同一原始试次的所有片段必然落入同一折。

    96 个 1s 片段来自 24 * 4，别让相邻片段分布用于训练和测试，解决方法就是让连续的四个片段分配带同一折。
    48 * 2 也直接这么处理即可，兼容的
    """
    fold_count = config["folds"]
    if cs_repeats != fold_count:
        raise ValueError("This protocol requires one CS repeat per class in each outer fold")
    return np.tile(np.arange(cs_repeats), 8), ns_groups % fold_count    # 每个人12次，分12折即可
    # [0,1,2,...,11, 0,1,2,...,11, ... ]  共 8 段，长度 96   每个折拿到 2 个原始试次 × 4 段 = 8 个 NS 片段
    # 对 NS3    每折 4 个试次 × 2 段 = 8 段


def feature_tensor(config, bank, model):
    """对 [子带, 样本, 通道, 时间] 输入返回 [样本, 类别, 子带] 的 eTRCA 特征。"""
    return np.stack([trca_correlations(config, bank[:, index], model, ensemble=True).T
                     for index in range(bank.shape[1])])


def train_fs_models(config, positive, negative):
    """每个频率训练一个 LDA；先验设为各 1/2，使用全部外层训练 NS 特征。"""
    models = []
    for target in range(positive.shape[1]):
        inputs = np.concatenate([positive[:, target], negative[:, target]])
        labels = np.r_[np.ones(len(positive), dtype=int), np.zeros(len(negative), dtype=int)]
        model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=config["lda_shrinkage"], priors=[0.5, 0.5])
        model.fit(inputs, labels)
        models.append(model)
    return models

def fs_predict(models, features, candidates):
    """按候选频率选择对应 LDA；LDA 接受为 CS 时输出频率，否则输出 NS=8。

    候选频率由 eTRCA 相关确定，接受与否由该候选对应的 LDA 判定。拒绝时统一输出 8"""
    predictions = np.full(len(features), 8, dtype=int)
    for index, target in enumerate(candidates):
        if models[target].predict(features[index, target][None])[0] == 1:
            predictions[index] = target
    return predictions

def ocn_candidates_cached(config, subject, state, signals):
    """缓存无需标签的网络统计量；候选参数变化或原文件变化都会更换缓存键。
    读取或生成 IFB-OCN 的候选特征缓存
    """
    import hashlib
    settings = {key: value for key, value in config.items() if key.startswith("ocn_")}
    settings["version"] = 1
    settings["cs_crop_delay"] = config["delay_seconds"]
    signature = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:12]
    source_key = cache_key(config, mat_path(config, subject, state), ["Fp1"])
    destination = Path(config["cache_dir"]) / f"{subject}_{state}_ocn_{source_key}_{signature}.npy"
    if destination.is_file():
        return np.load(destination, allow_pickle=False)
    print(f"  {subject} {state}: extracting recurrence-network candidates", flush=True)
    features = extract_candidates(config, signals)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.save(destination, features, allow_pickle=False)
    return features


# 主流程
def asynchronous_evaluation(config, subject):
    """完整运行某人的三组状态分类；任一失败会抛出错误。

    对三种 NS 分别构造 192 样本（96 CS + 96 NS）的九分类任务，用同一套 CS 侧计算（外层 eTRCA 模型与内部留一正类特征只算一次，三状态共享），
    执行 12 折分组验证。每折内完成：eTRCA 候选频率选择、逐候选 LDA 训练（FS 两变体）、IFB-OCN 的内层选带选参与 SVM 训练。
    """
    output = Path(config["output_dir"]) / "asynchronous"
    output.mkdir(parents=True, exist_ok=True)

    # 频率识别沿用九个后部电极，网络法（IFB-OCN）独立读取 Fp1，两个方法互不混用通道。
    cs = crop_cs(config, load_preprocessed(config, subject, "CS"), 1)
    class_count, repeats, channels, samples = cs.shape
    cs_bank = filterbank(config, cs)
    cs_flat = cs.reshape(-1, channels, samples)
    cs_labels = np.repeat(np.arange(class_count), repeats)
    reference = reference_bases(config, samples)
    cs_cca = np.stack([fbcca_correlations(config, cs_bank.reshape(4, -1, channels, samples)[:, index], reference).T
                       for index in range(class_count * repeats)])
    frontal_cs = crop_cs(config, load_preprocessed(config, subject, "CS", ["Fp1"]), 1).reshape(-1, samples)
    frontal_candidates = ocn_candidates_cached(config, subject, "CS", frontal_cs)
    rows, predictions, selection_log = [], [], []

    # 先为每个外层折构建 eTRCA 模型和内部留一正类特征，三种 NS 共享这些 CS 计算。
    prepared = []
    for fold in range(config["folds"]):
        keep = np.arange(repeats) != fold
        model = fit_trca(config, cs_bank[:, :, keep])
        positive = np.empty((repeats - 1, class_count, 4))
        for position, held_repeat in enumerate(np.flatnonzero(keep)):
            inner_keep = keep & (np.arange(repeats) != held_repeat)
            inner_model = fit_trca(config, cs_bank[:, :, inner_keep])
            for target in range(class_count):
                # 内层只提取该试次真实频率的特征；不能让该试次进入自己的模板。
                positive[position, target] = trca_correlations(
                    config, cs_bank[:, target, held_repeat], inner_model, ensemble=True
                )[:, target]
        prepared.append((model, positive))

    for state in ("NS1", "NS2", "NS3"):
        print(f"{subject}: asynchronous CS vs {state}", flush=True)
        ns, ns_groups = segment_ns(config, load_preprocessed(config, subject, state))
        ns_bank = filterbank(config, ns)
        ns_cca = np.stack([fbcca_correlations(config, ns_bank[:, index], reference).T for index in range(len(ns))])
        frontal_ns, _ = segment_ns(config, load_preprocessed(config, subject, state, ["Fp1"]))
        network_features = np.concatenate([frontal_candidates,
            ocn_candidates_cached(config, subject, state, frontal_ns[:, 0])])
        cs_folds, ns_folds = outer_assignments(config, repeats, ns_groups)
        folds = np.r_[cs_folds, ns_folds]
        labels = np.r_[cs_labels, np.full(len(ns), 8)]
        binary_labels = (labels != 8).astype(int)
        # CS 不同频率/重复是独立原始试次；NS 每个原始试次的多个片段共享组号。
        groups = np.r_[np.arange(len(cs_flat)), len(cs_flat) + ns_groups]
        algorithm_predictions = {name: np.full(len(labels), -1, dtype=int) for name in
                                 ("FS-eTRCA", "FS-eTRCA&FBCCA", "IFB-OCN-reconstructed")}
        # 逐折完成三种算法
        for fold, (model, positive) in enumerate(prepared):
            ns_train = ns_folds != fold
            test = folds == fold
            train = ~test
            if set(groups[train]) & set(groups[test]):
                raise AssertionError("Original trial leaked across the outer split")
            all_bank = np.concatenate([cs_bank.reshape(4, -1, channels, samples), ns_bank], axis=1)
            test_etr = feature_tensor(config, all_bank[:, test], model)
            negative = feature_tensor(config, ns_bank[:, ns_train], model)
            candidate = np.argmax(np.einsum("ncb,b->nc", test_etr ** config["fs_score_power"], weights(config)), axis=1)
            fs_models = train_fs_models(config, positive, negative)
            algorithm_predictions["FS-eTRCA"][test] = fs_predict(fs_models, test_etr, candidate)
            # CCA 是无训练参考信号法；按同一重复顺序拼接，形成每候选频率八维特征。
            cca_positive = cs_cca.reshape(class_count, repeats, class_count, 4)
            cca_positive = np.stack([cca_positive[target, np.arange(repeats) != fold, target]
                                     for target in range(class_count)], axis=1)
            positive_fused = np.concatenate([positive, cca_positive], axis=-1)
            negative_fused = np.concatenate([negative, ns_cca[ns_train]], axis=-1)
            fused_models = train_fs_models(config, positive_fused, negative_fused)
            test_fused = np.concatenate([test_etr, np.concatenate([cs_cca, ns_cca])[test]], axis=-1)
            algorithm_predictions["FS-eTRCA&FBCCA"][test] = fs_predict(fused_models, test_fused, candidate)
            predicted, choices = evaluate_ocn_fold(config, network_features, binary_labels, groups, train, test)
            algorithm_predictions["IFB-OCN-reconstructed"][test] = predicted
            selection_log.append(dict(subject=subject, state=state, fold=fold + 1, selected=choices))

        # 算指标 + 记预测
        for algorithm, predicted in algorithm_predictions.items():
            truth = binary_labels if algorithm.startswith("IFB") else labels
            if (predicted < 0).any():
                raise AssertionError("Some samples did not receive an outer-fold prediction")
            predicted_control = predicted == 1 if algorithm.startswith("IFB") else predicted != 8
            rows.append(dict(subject=subject, state=state, algorithm=algorithm,
                             accuracy=accuracy_score(truth, predicted),
                             balanced_accuracy=balanced_accuracy_score(truth, predicted),
                             cs_detection_rate=float(predicted_control[:len(cs_flat)].mean()),
                             ns_false_positive_rate=float(predicted_control[len(cs_flat):].mean()),
                             n_classes=2 if algorithm.startswith("IFB") else 9, n_samples=len(labels)))
            print(f"  {algorithm}: {rows[-1]['accuracy']:.2%}", flush=True)
            for index in range(len(labels)):
                predictions.append(dict(subject=subject, state=state, algorithm=algorithm,
                                        sample=index, fold=int(folds[index] + 1), original_group=int(groups[index]),
                                        true_class=int(truth[index]), predicted_class=int(predicted[index])))
    pd.DataFrame(rows).to_csv(output / f"{subject}_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(output / f"{subject}_predictions.csv", index=False)
    (output / f"{subject}_band_selection.json").write_text(json.dumps(selection_log, indent=2), encoding="utf-8")
    return rows
