"""验证频率算法、递归网络统计量、NS 分组切窗和训练内参数选择。

测试使用已知结构的合成信号/网络，不依赖20GB真实数据；完整真实数据运行后
另外由 reporting.py 和 verify_results.py 核对预测数量及指标。
"""

import unittest

import numpy as np

from algorithms import fbcca_scores, filterbank, fit_trca, information_rate, orthogonal_basis, trca_scores
from data_io import read_config
from data_io import segment_ns
from asynchronous import fs_predict, outer_assignments
from ocn import evaluate_ocn_fold, network_statistics
from quality_analysis import snr_spectra


class AlgorithmTests(unittest.TestCase):
    """测试可预先确定正确答案的边界和构造样例。"""

    def test_itr_boundaries(self):
        """零准确率和机会水平不输出正 ITR，完全正确时达到理论上限。"""
        self.assertEqual(information_rate(1 / 8, 8, 2), 0)
        self.assertEqual(information_rate(0, 8, 2), 0)
        self.assertAlmostEqual(information_rate(1, 8, 2), 90)

    def test_rank_deficient_cca_input(self):
        """重复通道只能产生一个独立方向，SVD 不应将其误判为三维。"""
        times = np.arange(250) / 250
        wave = np.sin(2 * np.pi * 12 * times)
        basis = orthogonal_basis(np.stack([wave, wave, 2 * wave]))
        self.assertEqual(basis.shape[1], 1)

    def test_synthetic_frequency_and_trca(self):
        """三种算法都应识别具有已知频率和独立测试噪声的模拟信号。"""
        config = read_config()
        generator = np.random.default_rng(42)
        times = np.arange(500) / config["fs"]
        classes = []
        for frequency in config["frequencies"]:
            wave = np.sin(2 * np.pi * frequency * times) + 0.4 * np.sin(4 * np.pi * frequency * times)
            trials = np.stack([np.arange(1, 5)[:, None] * wave + generator.normal(0, 0.2, (4, 500))
                               for _ in range(4)])
            classes.append(trials)
        bank = filterbank(config, np.stack(classes))
        model = fit_trca(config, bank[:, :, :3])
        for target in range(8):
            trial = bank[:, target, 3]
            self.assertEqual(int(np.argmax(fbcca_scores(config, trial))), target)
            self.assertEqual(int(np.argmax(trca_scores(config, trial, model))), target)
            self.assertEqual(int(np.argmax(trca_scores(config, trial, model, ensemble=True))), target)

    def test_model_uses_training_trials_only(self):
        """改变外层留出试次不应改变用相同训练切片构建的 TRCA 模型。"""
        config = read_config()
        generator = np.random.default_rng(7)
        bank = generator.normal(size=(4, 8, 5, 3, 250))
        first = fit_trca(config, bank[:, :, :4].copy())
        bank[:, :, 4] = 1e6
        second = fit_trca(config, bank[:, :, :4].copy())
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])

    def test_network_triangle_and_path(self):
        """完整三角形的平均度=2且聚类=1，三节点路径的聚类=0。"""
        triangle = np.ones((3, 3)) - np.eye(3)
        np.testing.assert_allclose(network_statistics(triangle), [2, 1])
        path = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        np.testing.assert_allclose(network_statistics(path), [4 / 3, 0])

    def test_ns_segments_preserve_raw_trials(self):
        """切片顺序不能打乱时间/通道；24个NS1试次每折恰有8个片段。"""
        config = read_config()
        data = np.arange(24 * 2 * 1000).reshape(24, 2, 1000)
        segments, groups = segment_ns(config, data)
        self.assertEqual(segments.shape, (96, 2, 250))
        np.testing.assert_array_equal(segments[1, 1], data[0, 1, 250:500])
        _, folds = outer_assignments(config, 12, groups)
        np.testing.assert_array_equal(np.bincount(folds), np.full(12, 8))
        for fold in range(12):
            self.assertFalse(set(groups[folds == fold]) & set(groups[folds != fold]))

    def test_snr_flat_spectrum(self):
        """平坦频谱的两种邻域比均为1，防止把原文系数9误写为1.6。"""
        frequency = np.arange(0, 5.25, 0.25)
        conventional, paper = snr_spectra(np.ones((2, len(frequency))), frequency)
        np.testing.assert_allclose(conventional[:, 8], 1)
        np.testing.assert_allclose(paper[:, 8], 1)

    def test_ocn_selection_ignores_test_labels(self):
        """翻转外层测试标签，所选参数和外层预测必须完全相同。"""
        config = read_config()
        generator = np.random.default_rng(5)
        features = generator.normal(size=(48, 5, 2, 2))
        labels = np.tile([0, 1], 24)
        groups = np.arange(48)
        train = np.arange(48) >= 8
        first, choices = evaluate_ocn_fold(config, features, labels, groups, train, ~train)
        changed = labels.copy()
        changed[~train] = 1 - changed[~train]
        second, changed_choices = evaluate_ocn_fold(config, features, changed, groups, train, ~train)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(choices, changed_choices)


if __name__ == "__main__":
    unittest.main()
