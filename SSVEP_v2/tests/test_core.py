"""核心算法的单元测试：用“答案已知”的合成信号检查实现是否正确。

这些测试不需要 20GB 真实数据，几秒就能跑完：
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

# 让 `python -m unittest discover -s tests` 也能从项目根目录导入 ssvep 包。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ssvep.algorithms import (band_weights, fbcca_scores,
                              filter_bank, fit_trca, itr, orthonormal_basis,
                              trca_scores)
from ssvep.config import load_config
from ssvep.data import crop_cs, segment_ns


class AlgorithmTests(unittest.TestCase):
    """算法层的构造样例。"""

    @classmethod
    def setUpClass(cls):
        cls.config = load_config()

    def test_itr_boundaries(self):
        """低于机会水平的准确率不给正 ITR；全对时达到理论公式值。"""
        self.assertEqual(itr(1 / 8, 8, 2), 0)       # 机会水平
        self.assertEqual(itr(0, 8, 2), 0)           # 全错
        self.assertAlmostEqual(itr(1, 8, 2), 90)    # log2(8)*60/2 = 90 bit/min

    def test_rank_deficient_cca_input(self):
        """三路完全相同的信号只张成 1 维子空间，不能被当成 3 维。"""
        times = np.arange(250) / 250
        wave = np.sin(2 * np.pi * 12 * times)
        basis = orthonormal_basis(np.stack([wave, wave, 2 * wave]))
        self.assertEqual(basis.shape[1], 1)

    def test_filter_bank_shapes_and_weights(self):
        """子带滤波在轴 0 新增一维；权重随子带序号递减。"""
        signals = np.random.default_rng(0).normal(size=(8, 12, 9, 250))
        bank = filter_bank(self.config, signals)
        self.assertEqual(bank.shape, (4,) + signals.shape)

        weights = band_weights(self.config)
        self.assertAlmostEqual(float(weights[0]), 1.25)          # 1^-1.25 + 0.25
        self.assertTrue(np.all(np.diff(weights) < 0))            # 单调递减

    def test_synthetic_frequency_recognition(self):
        """合成 8 个已知频率的信号，三种算法都应识别正确。"""
        rng = np.random.default_rng(42)
        fs = self.config["eeg"]["fs"]
        times = np.arange(500) / fs

        classes = []
        for frequency in self.config["paradigm"]["frequencies"]:
            wave = (np.sin(2 * np.pi * frequency * times)
                    + 0.4 * np.sin(4 * np.pi * frequency * times))
            # [重复, 通道, 时间]：4 个通道是同一波形的不同混合，外加独立噪声。
            trials = np.stack([np.stack([(1 + channel) * wave
                                         + rng.normal(0, 0.2, 500) for channel in range(4)])
                               for _ in range(4)])
            classes.append(trials)

        bank = filter_bank(self.config, np.stack(classes))     # [4, 8, 4, 4, 500]
        model = fit_trca(self.config, bank[:, :, :3])           # 前 3 次重复训练
        bases = None                                            # 由函数内部按窗长生成

        for target in range(8):
            test_trial = bank[:, target, 3]
            self.assertEqual(int(np.argmax(fbcca_scores(self.config, test_trial, bases))), target)
            self.assertEqual(int(np.argmax(trca_scores(self.config, test_trial, model))), target)
            self.assertEqual(int(np.argmax(trca_scores(self.config, test_trial, model, ensemble=True))), target)

    def test_model_uses_training_trials_only(self):
        """改动测试试次不能改变用训练试次构建的模型（防止数据泄漏）。"""
        bank = np.random.default_rng(7).normal(size=(4, 8, 5, 3, 250))
        first = fit_trca(self.config, bank[:, :, :4].copy())

        bank[:, :, 4] = 1e6           # 把留出的第 5 次重复改成极端值
        second = fit_trca(self.config, bank[:, :, :4].copy())

        np.testing.assert_array_equal(first[0], second[0])    # 模板
        np.testing.assert_array_equal(first[1], second[1])    # 空间滤波器


class DataTests(unittest.TestCase):
    """数据切窗/切段的形状与分组。"""

    @classmethod
    def setUpClass(cls):
        cls.config = load_config()

    def test_ns_segments_preserve_trials(self):
        """切片顺序不能打乱；同一原始试次的片段要能被追溯。"""
        data = np.arange(24 * 2 * 1000, dtype=float).reshape(24, 2, 1000)
        segments, groups = segment_ns(self.config, data)

        self.assertEqual(segments.shape, (96, 2, 250))
        np.testing.assert_array_equal(segments[1, 1], data[0, 1, 250:500])
        self.assertEqual(groups[0], 0)
        self.assertEqual(groups[4], 1)

    def test_ns_fold_assignment_keeps_trials_together(self):
        """24 个 NS 试次按“试次编号 mod 12”分折，每折正好 8 段且不跨折。"""
        _, groups = segment_ns(self.config, np.zeros((24, 2, 1000)))
        folds = groups % 12

        self.assertEqual(len(folds), 96)
        np.testing.assert_array_equal(np.bincount(folds), np.full(12, 8))
        for fold in range(12):
            self.assertFalse(set(groups[folds == fold]) & set(groups[folds != fold]))

    def test_crop_cs_starts_after_visual_delay(self):
        """1 秒窗 = 视觉延迟 0.14 s 之后的 250 个点；越界要报错。"""
        cs = np.zeros((8, 12, 9, 1125))
        window = crop_cs(self.config, cs, 1)
        self.assertEqual(window.shape, (8, 12, 9, 250))
        with self.assertRaises(ValueError):
            crop_cs(self.config, cs, 5)


if __name__ == "__main__":
    unittest.main()
