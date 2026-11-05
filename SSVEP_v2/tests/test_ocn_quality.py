"""OCN 与信号质量模块的单元测试（对照已知答案的构造样例）。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ssvep.config import load_config
from ssvep.ocn import evaluate_ocn_fold, network_statistics
from ssvep.quality import snr_spectra


class NetworkTests(unittest.TestCase):
    """递归网络统计量。"""

    def test_triangle_and_path(self):
        """完整三角形：平均度 2、聚类系数 1；三节点链：平均度 4/3、聚类 0。"""
        triangle = np.ones((3, 3)) - np.eye(3)
        np.testing.assert_allclose(network_statistics(triangle), [2, 1])

        path = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        np.testing.assert_allclose(network_statistics(path), [4 / 3, 0])


class OcnSelectionTests(unittest.TestCase):
    """参数/频带选择只能看训练折。"""

    @classmethod
    def setUpClass(cls):
        cls.config = load_config()

    def test_selection_ignores_test_labels(self):
        """翻转外层测试标签后，选出的参数与外层预测必须完全不变。"""
        generator = np.random.default_rng(5)
        features = generator.normal(size=(48, 5, 2, 2))
        labels = np.tile([0, 1], 24)
        groups = np.arange(48)
        train = np.arange(48) >= 8
        test = ~train

        first_prediction, first_choice = evaluate_ocn_fold(
            self.config, features, labels, groups, train, test)

        changed = labels.copy()
        changed[test] = 1 - changed[test]      # 篡改测试标签
        second_prediction, second_choice = evaluate_ocn_fold(
            self.config, features, changed, groups, train, test)

        np.testing.assert_array_equal(first_prediction, second_prediction)
        self.assertEqual(first_choice, second_choice)


class SnrTests(unittest.TestCase):
    """SNR 公式的边界。"""

    def test_flat_spectrum_is_one(self):
        """平坦频谱的两种邻域比都应为 1（防止把系数 9 误写成 1.6）。"""
        frequencies = np.arange(0, 5.25, 0.25)
        conventional, paper = snr_spectra(np.ones((2, len(frequencies))), frequencies)
        np.testing.assert_allclose(conventional[:, 8], 1)
        np.testing.assert_allclose(paper[:, 8], 1)


if __name__ == "__main__":
    unittest.main()
