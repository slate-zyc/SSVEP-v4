"""复现论文第 4.1 节：12 Hz 连续闪烁（CS）信号的质量分析（对应论文图 4）。

4.1 节只分析 12 Hz 这一类刺激，回答四个问题：

    (a) 24 人平均后，Oz 电极上的 12 Hz 响应波形长什么样？
    (b) 这段平均响应的幅度谱里，12、24、36、48 Hz 的峰有多高？
    (c) 这四个谐波在 63 个电极上怎么分布（头皮图）？
    (d) 按原文式 (3) 算出的 SNR 是多少，和原文数值差多少？

数据流（代码从上到下就是执行顺序）：

    MAT (8, 63, 22500, 12)
      → 取 12 Hz 块、12 次重复平均        (63, 22500)
      → 去均值 / 降采样 / 50 Hz 陷波       (63, 1125)
      → 24 人等权平均                      (63, 1125)
      → 取 0~4 秒做 FFT                    (63, 501)
      → 原文式 (3) 算 SNR                  (63, 501)
      → 四个谐波的幅度/SNR 表 + 对照表
      → 图 4（波形、幅度谱、头皮图、SNR）

运行：python reproduce_4_1.py
输出：results/section4_1/ 下 4 个 CSV 和 1 张 figure4.png
"""

from time import perf_counter

import matplotlib
matplotlib.use("Agg")               # 只把图存成文件，不弹窗口
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from config import (FFT_SAMPLES, FS, HARMONICS, OUTPUT_DIR, PAPER_AMPLITUDE,
                    PAPER_SNR, SUBJECTS)
from data import display_waveform, load_channel_names, load_twelve_hz_response


# ===========================================================================
# 1. 准备：输出目录、电极名、收集用的列表
# ===========================================================================
started = perf_counter()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 头皮图和全通道表需要电极名；Oz 用于图 (a)、(b)、(d) 的单通道曲线。
channel_names, oz_index = load_channel_names()
print(f"通道表：{len(channel_names)} 个电极，Oz 的下标是 {oz_index}")

responses = []   # 每个人的 12 Hz 预处理响应，[63, 1125]
waveforms = []   # 每个人 Oz 的 7-70 Hz 显示波形，[1125]


# ===========================================================================
# 2. 逐人读取并预处理
# ===========================================================================
# 每轮只载入一个人的 CS 文件（约 500 MB），处理完就丢掉原始大数组，
# 所以内存占用基本只有一个人的数据。
for subject in SUBJECTS:
    print(f"[{subject}] 读取并预处理 12 Hz 响应", flush=True)
    response = load_twelve_hz_response(subject)          # [63, 1125]
    responses.append(response)
    waveforms.append(display_waveform(response[oz_index]))


# ===========================================================================
# 3. 24 人等权平均
# ===========================================================================
# 每个人先对自己的 12 次重复平均过了，这里再对 24 人平均，每人权重 1/24。
grand_response = np.mean(responses, axis=0)      # [63, 1125]，用于频谱和 SNR
grand_waveform = np.mean(waveforms, axis=0)      # [1125]，只用于图 (a)
times = np.arange(grand_response.shape[-1]) / FS  # 0, 0.004, ..., 4.496 秒


# ===========================================================================
# 4. 单边幅度谱
# ===========================================================================
# 只取刺激开始后的 4 秒（1000 点），不补零、不加窗：
# 频率分辨率 = 250 / 1000 = 0.25 Hz，得到 0~125 Hz 共 501 个频点。
frequencies = np.fft.rfftfreq(FFT_SAMPLES, d=1 / FS)
amplitude = np.abs(np.fft.rfft(grand_response[:, :FFT_SAMPLES], axis=-1)) / FFT_SAMPLES
# 实数信号的正负频率对称，rfft 只保留了一半；除直流和 Nyquist 外都要乘 2，
# 乘 2 后谱峰高度才等于该频率成分的物理幅度（保留源数据单位）。
amplitude[:, 1:-1] *= 2     # [63, 501]


# ===========================================================================
# 5. 按原文式 (3) 计算 SNR
# ===========================================================================
# SNR(f) = (2/Δf + 1) × Amp(f) / Σ_{|x - f| ≤ 1 Hz} Amp(x)
# Δf = 0.25 Hz 时 2/Δf + 1 = 9，邻域正好 9 个频点（含中心），
# 所以原式等价于“中心频点幅度 ÷ 邻域 9 点平均幅度”，结果是幅度比而不是 dB。
delta_f = frequencies[1] - frequencies[0]     # 0.25 Hz
half_width = round(1 / delta_f)               # ±1 Hz 对应 4 个频率间隔

snr = np.full_like(amplitude, np.nan)         # 两端凑不满 9 个点，保持 NaN
for index in range(half_width, len(frequencies) - half_width):
    neighbor_sum = amplitude[:, index - half_width:index + half_width + 1].sum(axis=-1)
    snr[:, index] = (2 / delta_f + 1) * amplitude[:, index] / neighbor_sum

# 12/24/36/48 Hz 在 0.25 Hz 频率轴上的下标：48、96、144、192。
harmonic_indices = np.array(HARMONICS) * round(1 / delta_f)
oz_amplitude = amplitude[oz_index]            # [501] Oz 的幅度谱
oz_snr = snr[oz_index]                        # [501] Oz 的 SNR


# ===========================================================================
# 6. 与原文第 4.1 节的数值对照
# ===========================================================================
# 原文值只用来算差值，不参与任何计算或参数选择。
comparison = pd.DataFrame({
    "frequency_hz": HARMONICS,
    "amplitude_reproduced": oz_amplitude[harmonic_indices],
    "amplitude_paper": PAPER_AMPLITUDE,
    "snr_reproduced": oz_snr[harmonic_indices],
    "snr_paper": PAPER_SNR,
})
comparison["amplitude_difference"] = (comparison.amplitude_reproduced
                                      - comparison.amplitude_paper)
comparison["snr_difference"] = comparison.snr_reproduced - comparison.snr_paper


# ===========================================================================
# 7. 写 CSV（结果表）
# ===========================================================================
# 图 (a) 的时间轴和两种波形：原始平均响应 + 7-70 Hz 显示版本。
pd.DataFrame({
    "time_from_stimulus_onset_s": times,
    "oz_average": grand_response[oz_index],
    "oz_display_7_70_hz": grand_waveform,
}).to_csv(OUTPUT_DIR / "waveform_oz.csv", index=False)

# 图 (b) 和 (d) 共用的 501 个频点：幅度谱 + 式(3) SNR。
pd.DataFrame({
    "frequency_hz": frequencies,
    "amplitude": oz_amplitude,
    "snr_equation_3": oz_snr,
}).to_csv(OUTPUT_DIR / "spectrum_snr_oz.csv", index=False)

# 图 (c) 背后的数值：63 个电极 × 4 个谐波 = 252 行。
pd.DataFrame({
    "channel": np.repeat(channel_names, len(HARMONICS)),
    "frequency_hz": np.tile(HARMONICS, len(channel_names)),
    "amplitude": amplitude[:, harmonic_indices].ravel(),
    "snr_equation_3": snr[:, harmonic_indices].ravel(),
}).to_csv(OUTPUT_DIR / "harmonics_all_channels.csv", index=False)

comparison.to_csv(OUTPUT_DIR / "comparison_with_paper.csv", index=False)


# ===========================================================================
# 8. 头皮图的电极坐标
# ===========================================================================
# 原始数据只给了电极名，没有实测三维位置，所以只能用 MNE 的标准 10-20
# 模板坐标来画头皮图。这一点会在图的底部注明，避免被误当成实测位置。
montage = mne.channels.make_standard_montage("colin27_1020")
# 官方表里的大小写和模板不完全一致（如 Poz vs POz），统一转小写来匹配。
standard_names = {name.casefold(): name for name in montage.ch_names}
plot_names = [standard_names[name.casefold()] for name in channel_names]
info = mne.create_info(plot_names, sfreq=FS, ch_types="eeg")
info.set_montage(montage)


# ===========================================================================
# 9. 画图 4：四行依次是波形、幅度谱、头皮图、SNR
# ===========================================================================
plt.rcParams.update({"font.size": 10,
                     "axes.spines.top": False,
                     "axes.spines.right": False})
figure = plt.figure(figsize=(12, 13), layout="constrained")
grid = figure.add_gridspec(4, 4, height_ratios=[1.2, 1, 1.5, 1])
ax_wave = figure.add_subplot(grid[0, :])
ax_spec = figure.add_subplot(grid[1, :])
ax_maps = [figure.add_subplot(grid[2, column]) for column in range(4)]
ax_snr = figure.add_subplot(grid[3, :])
figure.suptitle("Section 4.1 | 12 Hz CS signal quality | 24 subjects, 12 trials each",
                fontsize=15)

# ---------- (a) Oz 平均波形 ----------
ax_wave.plot(times, grand_waveform, color="#2074a5", linewidth=0.8)
ax_wave.axvline(0, color="#d94b46", linestyle="--", label="Stimulus onset / offset")
ax_wave.axvline(4, color="#d94b46", linestyle="--")
ax_wave.axvline(0.14, color="#2c9d65", linestyle="--", label="140 ms reference")
ax_wave.axvline(4.14, color="#2c9d65", linestyle="--")
ax_wave.set(xlim=(-0.02, 4.5),
            xlabel="Time from stimulus onset (s)",
            ylabel="Amplitude (source units)",
            title="(a) Average Oz waveform, 7-70 Hz display filter")
ax_wave.legend(loc="lower right", fontsize=9)

# ---------- (b) 跨人平均响应的 Oz 幅度谱 ----------
ax_spec.plot(frequencies, oz_amplitude, color="#2074a5", linewidth=0.9)
ax_spec.set(xlim=(0, 70),
            xlabel="Frequency (Hz)",
            ylabel="Amplitude (source units)",
            title="(b) Spectrum of the grand-average Oz response, 0-4 s")

# ---------- (c) 四个谐波的头皮图 ----------
# 每张图有自己的色标范围，所以不要跨图直接比较颜色深浅。
for axis, frequency, index in zip(ax_maps, HARMONICS, harmonic_indices):
    plotted, _ = mne.viz.plot_topomap(amplitude[:, index], info,
                                      axes=axis, show=False, cmap="viridis")
    figure.colorbar(plotted, ax=axis, orientation="horizontal", shrink=0.9, pad=0.03)
    axis.set_title(f"{frequency} Hz")
ax_maps[0].text(0, 1.2, "(c) Harmonic scalp maps", transform=ax_maps[0].transAxes)

# ---------- (d) 与 (b) 同一段响应、同一条频率轴的 SNR ----------
ax_snr.plot(frequencies, oz_snr, color="#2074a5", linewidth=0.9)
ax_snr.set(xlim=(0, 70), ylim=(0, 9),
           xlabel="Frequency (Hz)",
           ylabel="SNR (amplitude ratio)",
           title="(d) SNR of the grand-average Oz response, paper equation (3)")
# 两条谱线都标出四个谐波的位置，方便对照谱峰。
for axis in [ax_spec, ax_snr]:
    for frequency in HARMONICS:
        axis.axvline(frequency, color="gray", linestyle=":", linewidth=0.7, alpha=0.5)

# 给总标题和底部说明留出空间（rect 依次是左、下、宽、高）。
figure.get_layout_engine().set(rect=(0, 0.035, 1, 0.925))
figure.text(0.5, 0.01,
            "Scalp maps: standard 10-20 coordinates; amplitudes retain the original data units.",
            ha="center", fontsize=9, color="#555555")
figure.savefig(OUTPUT_DIR / "figure4.png", dpi=200)
plt.close(figure)


# ===========================================================================
# 10. 在控制台打印对照表和耗时
# ===========================================================================
elapsed = perf_counter() - started
print()
print(comparison.to_string(index=False, float_format="%.4f"))
print(f"\n完成：{OUTPUT_DIR}\n用时：{elapsed:.1f} 秒")
