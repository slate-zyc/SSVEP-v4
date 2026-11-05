"""八分类频率识别（论文图 5、图 6）。

一句话：给一段 1~4 秒的脑电，判断受试者看的是 8 个闪烁频率中的哪一个。

每个人、每个窗长都要单独做一次 12 折交叉验证：

    第 f 折：把每个频率的“第 f 次重复”留作测试（共 8 个测试试次），
             其余 11 次重复用来训练 TRCA/eTRCA 的模板和空间滤波器。
    每个测试试次：FBCCA / TRCA / eTRCA 各打一次分，取分数最高的频率作为预测。

训练折和测试折严格分开，测试数据从不参与模板构建，所以准确率是“新数据”上的表现。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .algorithms import (decision_seconds, fbcca_scores, filter_bank, fit_trca,
                         itr, reference_bases, trca_scores)
from .config import n_frequencies
from .data import crop_cs, load_state

ALGORITHMS = ("FBCCA", "TRCA", "eTRCA")


def run_frequency(config: dict, subject: str, windows: list[float] | None = None) -> pd.DataFrame:
    """对一个人跑完所有窗长，写出逐试次预测和指标两张表。

    输出文件
    --------
    results/frequency/{subject}_predictions.csv
    results/frequency/{subject}_metrics.csv

    返回指标表（每个窗长 × 每个算法一行）。
    """
    windows = windows or config["frequency"]["windows_seconds"]
    n_classes = n_frequencies(config)
    n_folds = config["frequency"]["folds"]
    cs = load_state(config, subject, "CS")           # [8, 12, 通道, 22500]

    prediction_rows: list[dict] = []
    metric_rows: list[dict] = []

    for seconds in windows:
        print(f"{subject}: 频率识别，窗长 {seconds:g} s", flush=True)

        # 1. 截窗 + 4 子带滤波 → [4, 8, 12, 通道, 250×seconds]
        bank = filter_bank(config, crop_cs(config, cs, seconds))
        bases = reference_bases(config, bank.shape[-1])   # 8 个频率的参考基，本窗长内共用
        n_repeats = bank.shape[2]

        # 2. 12 折交叉验证：每折留出每个频率的同一次重复
        window_predictions = {name: [] for name in ALGORITHMS}
        for fold in range(n_folds):
            train_repeats = np.arange(n_repeats) != fold
            model = fit_trca(config, bank[:, :, train_repeats])   # 只喂训练试次

            for target in range(n_classes):
                test_trial = bank[:, target, fold]                # [4, 通道, 时间]
                scores = {
                    "FBCCA": fbcca_scores(config, test_trial, bases),
                    "TRCA": trca_scores(config, test_trial, model),
                    "eTRCA": trca_scores(config, test_trial, model, ensemble=True),
                }
                for name, values in scores.items():
                    predicted = int(np.argmax(values))
                    window_predictions[name].append(predicted == target)
                    prediction_rows.append(dict(
                        subject=subject, window_seconds=seconds, algorithm=name,
                        fold=fold + 1, true_class=target, predicted_class=predicted))

        # 3. 本窗长的准确率与 ITR（决策时间 = 窗长 + 提示 + 视觉延迟）
        duration = decision_seconds(config, seconds)
        for name in ALGORITHMS:
            correct = window_predictions[name]
            accuracy = float(np.mean(correct))
            metric_rows.append(dict(
                subject=subject, window_seconds=seconds, algorithm=name,
                accuracy=accuracy, itr_bits_min=itr(accuracy, n_classes, duration),
                n_trials=len(correct), decision_seconds=duration))
            print(f"  {name}: {accuracy:.2%}", flush=True)

    # 4. 保存
    output = Path(config["paths"]["output"]) / "frequency"
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(prediction_rows).to_csv(output / f"{subject}_predictions.csv", index=False)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output / f"{subject}_metrics.csv", index=False)
    return metrics
