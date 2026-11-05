"""异步状态分类（论文图 7）：不用同步提示，直接判断受试者此刻的状态。

两个任务
--------
FS 九分类：当前是 8 个闪烁频率中的哪一个，还是“空闲”？
    标签 0..7 = 8 个频率，标签 8 = NS（空闲）。
IFB-OCN 二分类：只判断“连续闪烁(CS)”还是“空闲(NS)”。
    标签 0 = NS，1 = CS。二分类和九分类的准确率不能直接排名。

样本怎么来
----------
CS：每个频率每次重复，取 0.14 s 视觉延迟之后的 1 秒 → 8×12 = 96 个样本。
NS：每个原始试次切成连续的 1 秒片段：4 s 的试次切 4 段（NS1/NS2），
    2 s 的试次切 2 段（NS3）→ 都是 96 个样本。
所以每个 NS 状态的任务都是 96 CS + 96 NS = 192 个样本。

为什么按“原始试次”分折
----------------------
NS 的相邻片段来自同一个 4 秒试次，如果按片段随机分折，训练集和测试集
会包含同一次试次的相邻片段，等于提前泄露答案。所以同一个原始试次的
所有片段必须落在同一折；CS 则按“第几次重复”分折。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from .algorithms import (band_weights, fbcca_correlations, filter_bank, fit_trca,
                         reference_bases, trca_correlations)
from .config import n_bands
from .data import cache_file, crop_cs, load_state, mat_path, segment_ns
from .ocn import evaluate_ocn_fold, extract_features

STATES = ("NS1", "NS2", "NS3")
NS_LABEL = 8
ALGORITHMS = ("FS-eTRCA", "FS-eTRCA&FBCCA", "IFB-OCN-reconstructed")


@dataclass
class CsData:
    """一名受试者的 CS 侧数据（三种 NS 状态共享，只算一次）。"""

    bank: np.ndarray            # [4 子带, 8 频率, 12 重复, 通道, 时间]
    cca: np.ndarray             # [96 样本, 8 候选频率, 4 子带]
    labels: np.ndarray          # [96]，每个样本的真实频率
    n_classes: int
    n_repeats: int


@dataclass
class OuterFold:
    """一个外层折的 CS 侧训练结果。"""

    model: tuple                # (templates, filters)，由 11 次重复训练得到
    positive: np.ndarray        # [11, 8, 4]：训练 CS 的 eTRCA 特征（内部留一）


@dataclass
class NsTask:
    """一种 NS 状态的测试数据。"""

    bank: np.ndarray            # [4, 96, 通道, 时间]
    cca: np.ndarray             # [96, 8, 4]
    ocn_features: np.ndarray    # [192, 5 频带, 8 参数, 2 统计量]
    raw_trial: np.ndarray       # [96]，每个 1 秒片段来自第几个原始试次


# ---------------------------------------------------------------------------
# CS 侧准备
# ---------------------------------------------------------------------------

def _load_cs(config: dict, subject: str) -> CsData:
    """读 CS、截 1 秒窗、做子带滤波，并算好每个样本的 FBCCA 相关。"""
    window = config["asynchronous"]["window_seconds"]
    cs = crop_cs(config, load_state(config, subject, "CS"), window)   # [8, 12, 通道, 时间]
    n_classes, n_repeats = cs.shape[0], cs.shape[1]

    bank = filter_bank(config, cs)
    bases = reference_bases(config, cs.shape[-1])
    # 把 [4, 8, 12, ...] 展平成 [4, 样本, ...]，便于逐个样本算 CCA。
    flat_bank = bank.reshape(n_bands(config), -1, *cs.shape[2:])
    cca = np.stack([fbcca_correlations(config, flat_bank[:, i], bases).T
                    for i in range(n_classes * n_repeats)])
    labels = np.repeat(np.arange(n_classes), n_repeats)
    return CsData(bank, cca, labels, n_classes, n_repeats)


def _train_outer_folds(config: dict, cs: CsData) -> list[OuterFold]:
    """为每个外层折训练 CS 模型，并生成“内部留一”的训练特征。

    内部留一：计算某个训练 CS 的特征时，先把这次试次从模板里剔除，
    否则“自己和自己比”，特征会虚高。这里只用外层训练数据，不碰测试数据。
    """
    folds = []
    for fold in range(config["asynchronous"]["folds"]):
        keep = np.arange(cs.n_repeats) != fold
        model = fit_trca(config, cs.bank[:, :, keep])

        positive = np.empty((keep.sum(), cs.n_classes, n_bands(config)))
        for position, held_repeat in enumerate(np.flatnonzero(keep)):
            inner_keep = keep & (np.arange(cs.n_repeats) != held_repeat)
            inner_model = fit_trca(config, cs.bank[:, :, inner_keep])
            for target in range(cs.n_classes):
                # 只取该试次真实频率对应的那一列相关；4 个子带作为特征。
                positive[position, target] = trca_correlations(
                    config, cs.bank[:, target, held_repeat], inner_model, ensemble=True
                )[:, target]
        folds.append(OuterFold(model, positive))
    return folds


# ---------------------------------------------------------------------------
# NS 侧准备
# ---------------------------------------------------------------------------

def _ocn_features(config: dict, subject: str, state: str, signals: np.ndarray) -> np.ndarray:
    """提取并缓存 IFB-OCN 的候选特征 [片段, 5 频带, 8 参数, 2 统计量]。

    这些特征只用信号本身，不需要标签，所以可以在交叉验证之前算好。
    真正需要标签的“选参数、选频带”发生在 evaluate_ocn_fold 内部。
    """
    frontal = [config["paradigm"]["frontal_channel"]]
    source = mat_path(config, subject, state)
    extra = {"ocn": config["ocn"],
             "window": config["asynchronous"]["window_seconds"],
             "delay": config["paradigm"]["delay_seconds"]}
    cached = cache_file(config, subject, state, source, frontal,
                        tag=f"ocn_{state}", extra=extra)
    if cached.is_file():
        return np.load(cached, allow_pickle=False)

    print(f"  {subject} {state}: 提取网络特征", flush=True)
    features = extract_features(config, signals)
    np.save(cached, features, allow_pickle=False)
    return features


def _build_task(config: dict, subject: str, state: str, cs: CsData) -> NsTask:
    """准备一种 NS 状态的 NS 样本、CCA 特征和网络特征。"""
    window = config["asynchronous"]["window_seconds"]
    samples = cs.bank.shape[-1]
    frontal = [config["paradigm"]["frontal_channel"]]

    ns, ns_groups = segment_ns(config, load_state(config, subject, state))   # [96, 通道, 时间]
    ns_bank = filter_bank(config, ns)
    bases = reference_bases(config, samples)
    ns_cca = np.stack([fbcca_correlations(config, ns_bank[:, i], bases).T
                       for i in range(len(ns))])

    # IFB-OCN 单独使用 Fp1 单通道；CS 和 NS 的特征拼在一起，共 192 个样本。
    cs_frontal = crop_cs(config, load_state(config, subject, "CS", frontal), window).reshape(-1, samples)
    ns_frontal = segment_ns(config, load_state(config, subject, state, frontal))[0][:, 0]
    ocn_features = np.concatenate([_ocn_features(config, subject, "CS", cs_frontal),
                                   _ocn_features(config, subject, state, ns_frontal)])
    return NsTask(ns_bank, ns_cca, ocn_features, ns_groups)


# ---------------------------------------------------------------------------
# 每个外层折里训练/预测三种算法
# ---------------------------------------------------------------------------

def _etrca_features(config: dict, bank: np.ndarray, model: tuple) -> np.ndarray:
    """[4, 样本, 通道, 时间] → [样本, 8 候选频率, 4 子带] 的 eTRCA 相关特征。"""
    return np.stack([trca_correlations(config, bank[:, i], model, ensemble=True).T
                     for i in range(bank.shape[1])])


def _training_cs_cca(cs: CsData, fold: int) -> np.ndarray:
    """外层训练 CS 的 CCA 特征 [11, 8 候选频率, 4 子带]。

    cs.cca 按 (真实频率, 重复) 排列。对候选频率 target，
    取“真实频率就是 target、且不是本折测试”的那些样本，只保留它自己那一列相关。
    """
    grid = cs.cca.reshape(cs.n_classes, cs.n_repeats, cs.n_classes, cs.cca.shape[-1])
    keep = np.arange(cs.n_repeats) != fold
    return np.stack([grid[target, keep, target] for target in range(cs.n_classes)], axis=1)


def _train_lda_models(config: dict, positive: np.ndarray, negative: np.ndarray) -> list:
    """每个候选频率训练一个二分类 LDA：正类=该频率的 CS，负类=NS。

    positive: [11, 8, 特征维]，negative: [88, 8, 特征维]。
    两类先验都设 0.5，避免 NS 样本多就把判决偏向 NS。
    """
    models = []
    for target in range(positive.shape[1]):
        features = np.concatenate([positive[:, target], negative[:, target]])
        labels = np.r_[np.ones(len(positive), dtype=int), np.zeros(len(negative), dtype=int)]
        model = LinearDiscriminantAnalysis(
            solver="lsqr", shrinkage=config["asynchronous"]["lda_shrinkage"], priors=[0.5, 0.5])
        model.fit(features, labels)
        models.append(model)
    return models


def _fs_predict(models: list, features: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """用 eTRCA 选出的候选频率，交给该频率的 LDA 判决。

    LDA 说是 CS → 输出该频率；LDA 说是 NS → 输出 8。
    所以这是九分类，而不是二分类。
    """
    predictions = np.full(len(features), NS_LABEL, dtype=int)
    for index, target in enumerate(candidates):
        if models[target].predict(features[index, target][None, :])[0] == 1:
            predictions[index] = target
    return predictions


def _evaluate_task(config: dict, subject: str, state: str, cs: CsData,
                   outer_folds: list[OuterFold], task: NsTask) -> tuple[list, list, list]:
    """对一种 NS 状态做 12 折交叉验证，返回指标行、预测行、选带日志。"""
    n_folds = config["asynchronous"]["folds"]
    n_cs = len(cs.labels)
    n_ns = len(task.raw_trial)

    # 每个样本属于第几折：CS 按重复编号，NS 按原始试次编号 mod 12。
    sample_folds = np.r_[np.tile(np.arange(cs.n_repeats), cs.n_classes),
                         task.raw_trial % n_folds]
    labels = np.r_[cs.labels, np.full(n_ns, NS_LABEL)]
    binary_labels = (labels != NS_LABEL).astype(int)
    # 同一个原始试次的片段共享组号，用于检查是否跨折。
    groups = np.r_[np.arange(n_cs), n_cs + task.raw_trial]

    # CS 的 [子带, 频率, 重复, 通道, 时间] 展平成 [子带, 样本, 通道, 时间]，
    # 才能和 NS 的 [子带, 样本, 通道, 时间] 拼在一起。
    cs_flat = cs.bank.reshape(n_bands(config), -1, *cs.bank.shape[3:])
    all_bank = np.concatenate([cs_flat, task.bank], axis=1)
    all_cca = np.concatenate([cs.cca, task.cca])

    predictions = {name: np.full(len(labels), -1, dtype=int) for name in ALGORITHMS}
    selection_log = []

    for fold, training in enumerate(outer_folds):
        is_test = sample_folds == fold
        ns_is_training = (task.raw_trial % n_folds) != fold
        if set(groups[~is_test]) & set(groups[is_test]):
            raise AssertionError("同一原始试次的片段跨了训练/测试折")

        # eTRCA 特征：测试样本 1 组，训练 NS 1 组。
        test_etrca = _etrca_features(config, all_bank[:, is_test], training.model)
        negative = _etrca_features(config, task.bank[:, ns_is_training], training.model)

        # 先用平方加权的 eTRCA 相关选出候选频率。
        weighted = test_etrca ** config["asynchronous"]["fs_score_power"]
        candidate = np.argmax(np.einsum("ncb,b->nc", weighted, band_weights(config)), axis=1)

        # (a) FS-eTRCA：4 维 eTRCA 相关特征 + 每个频率一个 LDA。
        models = _train_lda_models(config, training.positive, negative)
        predictions["FS-eTRCA"][is_test] = _fs_predict(models, test_etrca, candidate)

        # (b) FS-eTRCA&FBCCA：把 4 维 eTRCA 相关和 4 维 FBCCA 相关拼成 8 维。
        positive_fused = np.concatenate([training.positive, _training_cs_cca(cs, fold)], axis=-1)
        negative_fused = np.concatenate([negative, task.cca[ns_is_training]], axis=-1)
        models = _train_lda_models(config, positive_fused, negative_fused)
        test_fused = np.concatenate([test_etrca, all_cca[is_test]], axis=-1)
        predictions["FS-eTRCA&FBCCA"][is_test] = _fs_predict(models, test_fused, candidate)

        # (c) IFB-OCN：网络特征 + 训练内选参数/选频带 + 线性 SVM（二分类）。
        ocn_prediction, details = evaluate_ocn_fold(
            config, task.ocn_features, binary_labels, groups, ~is_test, is_test)
        predictions["IFB-OCN-reconstructed"][is_test] = ocn_prediction
        selection_log.append(dict(subject=subject, state=state, fold=fold + 1, selected=details))

    # 计算指标。九分类用 9 类标签；OCN 用二分类标签。
    metric_rows, prediction_rows = [], []
    for algorithm in ALGORITHMS:
        predicted = predictions[algorithm]
        if (predicted < 0).any():
            raise AssertionError("有样本没有拿到外层预测")

        is_cs = predicted == 1 if algorithm.startswith("IFB") else predicted != NS_LABEL
        truth = binary_labels if algorithm.startswith("IFB") else labels
        metric_rows.append(dict(
            subject=subject, state=state, algorithm=algorithm,
            accuracy=accuracy_score(truth, predicted),
            balanced_accuracy=balanced_accuracy_score(truth, predicted),
            cs_detection_rate=float(is_cs[:n_cs].mean()),
            ns_false_positive_rate=float(is_cs[n_cs:].mean()),
            n_classes=2 if algorithm.startswith("IFB") else 9,
            n_samples=len(labels)))
        print(f"  {algorithm}: {metric_rows[-1]['accuracy']:.2%}", flush=True)

        for index in range(len(labels)):
            prediction_rows.append(dict(
                subject=subject, state=state, algorithm=algorithm,
                sample=index, fold=int(sample_folds[index] + 1),
                original_group=int(groups[index]), true_class=int(truth[index]),
                predicted_class=int(predicted[index])))

    return metric_rows, prediction_rows, selection_log


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def run_asynchronous(config: dict, subject: str) -> pd.DataFrame:
    """对一个人跑完三种 NS 状态，写出指标、逐样本预测和选带日志。

    输出文件
    --------
    results/asynchronous/{subject}_metrics.csv
    results/asynchronous/{subject}_predictions.csv
    results/asynchronous/{subject}_band_selection.json
    """
    cs = _load_cs(config, subject)
    outer_folds = _train_outer_folds(config, cs)

    metric_rows, prediction_rows, selection_log = [], [], []
    for state in STATES:
        print(f"{subject}: 异步状态分类 CS vs {state}", flush=True)
        task = _build_task(config, subject, state, cs)
        metrics, predictions, selections = _evaluate_task(
            config, subject, state, cs, outer_folds, task)
        metric_rows += metrics
        prediction_rows += predictions
        selection_log += selections

    output = Path(config["paths"]["output"]) / "asynchronous"
    output.mkdir(parents=True, exist_ok=True)
    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame.to_csv(output / f"{subject}_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(output / f"{subject}_predictions.csv", index=False)
    (output / f"{subject}_band_selection.json").write_text(
        json.dumps(selection_log, indent=2), encoding="utf-8")
    return metrics_frame
