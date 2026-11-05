"""按受试者执行八分类频率识别，保存逐试次预测和准确率/ITR。

每个窗长执行12折验证，各类别留下同一编号的试次作为测试，其余11次训练。
质量分析与跨人汇总分别由 quality_analysis.py、reporting.py 负责。
"""

from pathlib import Path

import numpy as np
import pandas as pd

from algorithms import fbcca_scores, filterbank, fit_trca, information_rate, reference_bases, trca_scores
from data_io import crop_cs, load_preprocessed      # crop_cs 按数据长度截窗


def frequency_evaluation(config, subject, windows):
    """每人每窗每算法得到 8×12=96 条独立的外层测试预测。

    对 windows 中的每个数据长度，先截窗并做四子带滤波，再执行 12 折验证：
    第 f 折把 8 个类别中第 f 次重复的试次全部留作测试，其余 11 次重复用于训练 TRCA/eTRCA 模板与空间滤波器。
    每折对 8 个测试试次分别用 FBCCA、TRCA、eTRCA 打分并取 argmax 作为预测类别。FBCCA 无需训练，
    但仍按同一折结构记录预测，以保持三种算法的记录格式与核验方式一致。
    """
    data = load_preprocessed(config, subject, "CS")
    rows, predictions = [], []
    output = Path(config["output_dir"]) / "frequency"
    output.mkdir(parents=True, exist_ok=True)
    for seconds in windows:
        print(f"{subject}: frequency recognition, {seconds:g}s", flush=True)
        bank = filterbank(config, crop_cs(config, data, seconds))
        class_count, trial_count = data.shape[:2]
        bases = reference_bases(config, bank.shape[-1])
        for fold in range(trial_count):
            # 每次模型只使用训练切片，测试试次从未参与模板和滤波器构建。
            keep = np.arange(trial_count) != fold
            model = fit_trca(config, bank[:, :, keep])
            for target in range(class_count):
                test = bank[:, target, fold]
                scores = {"FBCCA": fbcca_scores(config, test, bases),
                          "TRCA": trca_scores(config, test, model),
                          "eTRCA": trca_scores(config, test, model, ensemble=True)}
                for algorithm, values in scores.items():
                    predictions.append(dict(subject=subject, window_seconds=seconds, algorithm=algorithm,
                        fold=fold + 1, true_class=target, predicted_class=int(np.argmax(values))))
        for algorithm in ("FBCCA", "TRCA", "eTRCA"):
            selected = [row for row in predictions if row["window_seconds"] == seconds and row["algorithm"] == algorithm]
            accuracy = float(np.mean([row["true_class"] == row["predicted_class"] for row in selected]))
            # 标准对称信道 ITR 使用完整声明的决策时间，而非单独的观察窗时长。
            duration = seconds + config["cue_seconds"] + config["delay_seconds"]
            rows.append(dict(subject=subject, window_seconds=seconds, algorithm=algorithm, accuracy=accuracy,
                itr_bits_min=information_rate(accuracy, class_count, duration), n_trials=len(selected), decision_seconds=duration))
            print(f"  {algorithm}: {accuracy:.2%}", flush=True)
    pd.DataFrame(predictions).to_csv(output / f"{subject}_predictions.csv", index=False)
    pd.DataFrame(rows).to_csv(output / f"{subject}_metrics.csv", index=False)
    return rows
