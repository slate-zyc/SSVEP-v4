"""5 分钟看懂：一段脑电是怎么变成“你看到的是 12 Hz”的。

这个脚本只做一件事——用 S01 的 4 秒窗长，把 FBCCA 频率识别的完整链路跑一遍，
每一步都把数组形状和中间结果打印出来。读它不需要看其他模块。

运行：
    python quick_start.py                 # 默认 S01、4 秒
    python quick_start.py S02 2           # 换成 S02、2 秒

整个流程一共 5 步：
    1. 读数据          原始 MAT → [8 频率, 12 重复, 9 通道, 时间]
    2. 截窗            从 0.14 s 视觉延迟后取 4 秒
    3. 子带滤波        同一段信号变成 4 个频带的版本
    4. 与参考信号比相关 每个频带 × 每个候选频率得到 4×8 的相关系数
    5. 加权打分 + 取最大 得到预测频率，12 折交叉验证算准确率
"""

from __future__ import annotations

import sys

import numpy as np

from ssvep.algorithms import band_weights, fbcca_correlations, filter_bank, reference_bases
from ssvep.config import load_config
from ssvep.data import crop_cs, load_state


def main(subject: str = "S01", seconds: float = 4.0) -> None:
    config = load_config()
    fs = config["eeg"]["fs"]
    frequencies = config["paradigm"]["frequencies"]

    # ------------------------------------------------------------------
    # 第 1 步：读数据
    # 原始 MAT 是 [频率, 通道, 时间, 重复]，load_state 会选通道、预处理并转成
    # [频率, 重复, 通道, 时间]。预处理 = 去均值 → 降采样到 250 Hz → 50 Hz 陷波。
    # ------------------------------------------------------------------
    cs = load_state(config, subject, "CS")
    print(f"[1] {subject} 的 CS 数据形状：{cs.shape}")
    print("    含义：[8 个频率, 12 次重复, 9 个通道, 1125 个采样点]")
    print(f"    1125 点 / {fs} Hz = {1125 / fs:.1f} 秒（一次闪烁持续 4.5 秒）\n")

    # ------------------------------------------------------------------
    # 第 2 步：截窗
    # 刺激出现后视觉通路有约 0.14 秒延迟，所以从第 0.14 秒开始取。
    # ------------------------------------------------------------------
    window = crop_cs(config, cs, seconds)
    print(f"[2] 截取 {seconds:g} 秒后：{window.shape}")
    print(f"    起点 = 0.14 s × {fs} Hz = {round(0.14 * fs)} 个采样点\n")

    # ------------------------------------------------------------------
    # 第 3 步：子带滤波
    # 同一段信号过 4 个不同通带的滤波器，得到 4 个“版本”。
    # 低频版本保留最强的基频，高频版本更干净但信号弱，后面加权融合。
    # ------------------------------------------------------------------
    bank = filter_bank(config, window)
    print(f"[3] 子带滤波后：{bank.shape}")
    print("    含义：[4 个子带, 8 频率, 12 重复, 9 通道, 采样点]\n")

    # ------------------------------------------------------------------
    # 第 4 步：与 8 个频率的参考信号比相关
    # 参考信号是每个频率的正弦 + 余弦（含 3 次谐波），共 6 条。
    # CCA 找出“脑电子空间”和“参考子空间”之间最像的一对方向，
    # 它们的夹角余弦就是相关系数。
    # ------------------------------------------------------------------
    bases = reference_bases(config, bank.shape[-1])
    print(f"[4] 参考基：每个频率一个矩阵，形状 {bases[0].shape}（时间 × 6 个方向）")

    # 拿“真实频率 12 Hz（下标 4）、第 0 次重复”的试次演示一次。
    trial = bank[:, 4, 0]                       # [4 子带, 9 通道, 采样点]
    correlations = fbcca_correlations(config, trial, bases)   # [4 子带, 8 候选频率]
    print("    这个试次在 8 个候选频率上的相关系数：")
    for band_index, row in enumerate(correlations):
        print(f"      子带 {band_index + 1}: " +
              "  ".join(f"{frequency:>4g}Hz={value:.3f}"
                        for frequency, value in zip(frequencies, row)))
    print(f"    单看相关性，最像的是 {frequencies[int(np.argmax(correlations.mean(axis=0)))]} Hz\n")

    # ------------------------------------------------------------------
    # 第 5 步：加权打分并做 12 折交叉验证
    # 权重 w_m = m^(-1.25) + 0.25：低子带权重大，高子带只做补充。
    # 12 折：第 f 折拿每个频率的第 f 次重复当测试，其余 11 次当训练。
    # FBCCA 不需要训练（没有可学习的参数），所以每折只做打分。
    # ------------------------------------------------------------------
    weights = band_weights(config)
    print("[5] 子带权重：", np.round(weights, 3))

    correct = 0
    for fold in range(12):
        for target in range(8):
            test_trial = bank[:, target, fold]
            correlation = fbcca_correlations(config, test_trial, bases)
            score = weights @ correlation          # 4 个子带加权求和 → 8 个分数
            predicted = int(np.argmax(score))
            correct += predicted == target

    accuracy = correct / (12 * 8)
    print(f"\n    FBCCA 在 {subject}、{seconds:g} 秒窗长上的八分类准确率：{accuracy:.2%}")
    print("    （论文 4 秒窗长、24 人平均约 93.49%；单人数字会有波动）")
    print("\n想看“需要训练”的 TRCA/eTRCA 怎么实现，读 ssvep/algorithms.py 的 fit_trca；")
    print("想跑完整的 24 人实验，运行 run.py。")


if __name__ == "__main__":
    subject = sys.argv[1] if len(sys.argv) > 1 else "S01"
    seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0
    main(subject, seconds)
