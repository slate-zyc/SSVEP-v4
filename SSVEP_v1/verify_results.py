"""独立核验全体实验输出，不重新训练或修改任何原始数据。

检查受试者/算法/窗长覆盖、每折样本数、原始试次分组、逐样本真实标签，
并从预测重新计算准确率、平衡准确率和 ITR。任何不一致均抛出异常。
运行方式：在项目环境执行 python verify_results.py。
"""

from pathlib import Path
import ast
import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from algorithms import information_rate
from data_io import ROOT, read_config
from reporting import read_subject_tables, validate_predictions


def verify(config):
    """按配置的完整受试者集合逐条核验，并保存结构化验证结果。"""
    subjects = config["subjects"]
    output = Path(config["output_dir"])
    frequency = read_subject_tables(output, "frequency", "predictions", subjects)
    asynchronous = read_subject_tables(output, "asynchronous", "predictions", subjects)
    assert validate_predictions(frequency) == len(subjects) * len(config["windows"]) * 3 * 96
    assert validate_predictions(asynchronous, asynchronous=True) == len(subjects) * 3 * 3 * 192
    frequency_metrics = pd.read_csv(output / "frequency_metrics.csv")
    asynchronous_metrics = pd.read_csv(output / "asynchronous_metrics.csv")
    assert len(frequency_metrics) == len(subjects) * len(config["windows"]) * 3
    assert len(asynchronous_metrics) == len(subjects) * 3 * 3
    assert set(frequency.algorithm) == {"FBCCA", "TRCA", "eTRCA"}
    assert set(asynchronous.algorithm) == {"FS-eTRCA", "FS-eTRCA&FBCCA", "IFB-OCN-reconstructed"}
    assert set(frequency.window_seconds) == set(config["windows"])

    for row in frequency_metrics.itertuples():
        selected = frequency[(frequency.subject == row.subject) & (frequency.algorithm == row.algorithm)
                             & (frequency.window_seconds == row.window_seconds)]
        assert set(selected.fold) == set(range(1, 13))
        assert set(selected.true_class) == set(range(8))
        assert set(selected.predicted_class) <= set(range(8))
        assert (selected.groupby("fold").size() == 8).all()
        measured = accuracy_score(selected.true_class, selected.predicted_class)
        np.testing.assert_allclose(measured, row.accuracy)
        np.testing.assert_allclose(information_rate(measured, 8, row.decision_seconds), row.itr_bits_min)
    for row in asynchronous_metrics.itertuples():
        selected = asynchronous[(asynchronous.subject == row.subject) & (asynchronous.algorithm == row.algorithm)
                                & (asynchronous.state == row.state)].sort_values("sample")
        sample = selected["sample"].to_numpy()
        np.testing.assert_array_equal(sample, np.arange(192))
        is_cs = sample < 96
        expected = np.where(is_cs, sample // 12, 8)
        if row.algorithm.startswith("IFB"):
            expected = is_cs.astype(int)
            assert set(selected.predicted_class) <= {0, 1}
        else:
            assert set(selected.predicted_class) <= set(range(9))
        np.testing.assert_array_equal(expected, selected.true_class)
        segments_per_trial = 2 if row.state == "NS3" else 4
        expected_groups = np.where(is_cs, sample, 96 + (sample - 96) // segments_per_trial)
        expected_folds = np.where(is_cs, sample % 12, ((sample - 96) // segments_per_trial) % 12) + 1
        np.testing.assert_array_equal(expected_groups, selected.original_group)
        np.testing.assert_array_equal(expected_folds, selected.fold)
        np.testing.assert_allclose(accuracy_score(expected, selected.predicted_class), row.accuracy)
        np.testing.assert_allclose(balanced_accuracy_score(expected, selected.predicted_class), row.balanced_accuracy)
        predicted_control = selected.predicted_class == 1 if row.algorithm.startswith("IFB") else selected.predicted_class != 8
        np.testing.assert_allclose(predicted_control.to_numpy()[is_cs].mean(), row.cs_detection_rate)
        np.testing.assert_allclose(predicted_control.to_numpy()[~is_cs].mean(), row.ns_false_positive_rate)

    # 不执行源码，仅用 AST 检查文件开头的模块文档字符串要求。
    sources = list(ROOT.glob("*.py"))
    assert all(path.read_text(encoding="utf-8").startswith('"""') for path in sources)
    assert all(ast.get_docstring(ast.parse(path.read_text(encoding="utf-8"))) for path in sources)
    result = dict(subjects=len(subjects), frequency_predictions=len(frequency),
                  asynchronous_predictions=len(asynchronous), frequency_metrics=len(frequency_metrics),
                  asynchronous_metrics=len(asynchronous_metrics), documented_python_files=len(sources),
                  raw_trial_grouping="passed", independent_metric_recalculation="passed",
                  exact_paper_reproduction=False,
                  limitation="IFB-OCN reconstruction and explicitly documented protocol assumptions")
    (output / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    verify(read_config())
