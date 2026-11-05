"""
重建信号质量图：Oz 电极 平均波形、幅度谱、SNR 和四个谐波的头皮分布。

先在每名受试者内部平均 12 Hz 的 12 个 CS 试次，再跨受试者等权平均。
头皮图使用标准 10–20 坐标；SNR同时输出论文的含中心点定义与排除中心点的邻域比。
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mne           # 画头皮地形图
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt    # 设计 Butterworth 滤波器；零相位滤波的 sos 版

from data_io import channel_table, quality_response     # 导入 评估信号质量 所需要的预处理函数

matplotlib.use("Agg")   # 关闭绘图弹窗

# 计算信噪比 SNR
def snr_spectra(amplitude, frequencies):
    """
    同时计算排除中心频点的邻域幅值比，以及论文的 (2/Δf+1) 幅值比。

    Δf=0.25时，±1Hz邻域含中心点共9个频点，系数为9，对应邻域均值。
    频谱分辨率 = 采样率 ÷ 采样点数 = 250 ÷ 1000 = 0.25 Hz

    论文中计算方式等价于：SNR_paper = Amp(f) / 邻域9个点的平均幅度  9 个点中包括 f 自己
    """
    conventional = np.full_like(amplitude, np.nan, dtype=float)     # 创建一个和模板形状完全相同、全部填充指定值的新数组
    paper = np.full_like(amplitude, np.nan, dtype=float)
    delta = frequencies[1] - frequencies[0]     # Δf=0.25
    for index, frequency in enumerate(frequencies):
        if frequency < 1 or frequency > frequencies[-1] - 1:    # 保证 ±1 Hz 邻域完整
            continue
        neighboring = np.abs(frequencies - frequency) <= 1 + 1e-9      # 布尔掩码，生成一个长度 501 的 True/False 数组
        excluding = neighboring.copy()
        excluding[index] = False    # 去掉中心频点
        # 计算排除中心频点的邻域幅值比
        conventional[..., index] = amplitude[..., index] / np.maximum(amplitude[..., excluding].mean(axis=-1), 1e-15)
        # 计算论文公式的邻域幅值比
        paper[..., index] = (2 / delta + 1) * amplitude[..., index] / np.maximum(
            amplitude[..., neighboring].sum(axis=-1), 1e-15)
    return conventional, paper


def standard_positions(config):
    """按官方通道索引排列标准坐标；不存在的电极会报错，不猜测或静默丢弃。"""
    mapping = channel_table(config)
    names = sorted(mapping, key=mapping.get)
    # standard_1020在MNE 1.13中改名为colin27_1020，采用新名称避免弃用警告。
    montage = mne.channels.make_standard_montage("colin27_1020")     # montage（蒙太奇）：MNE 术语，“一套电极名 + 坐标”。
    positions = {name.casefold(): position for name, position in montage.get_positions()["ch_pos"].items()}
    missing = set(names) - positions.keys()
    if missing:
        raise ValueError(f"Missing standard electrode coordinates: {missing}")
    info = mne.create_info(names, config["fs"], "eeg")
    info.set_montage(mne.channels.make_dig_montage(ch_pos={name: positions[name] for name in names}, coord_frame="head"))
    return names, info


def quality_evaluation(config, subjects):
    """逐人读取 63 通道平均响应，输出质量表、波形谱图和四谐波地形图。"""
    output = Path(config["output_dir"]) / "quality"
    output.mkdir(parents=True, exist_ok=True)   # 创建输出目录
    names, info = standard_positions(config)
    oz_index = names.index("oz")
    samples = round(config["fs"] * 4)   # 250 * 4 = 1000 采样点，不算休息的
    frequencies = np.fft.rfftfreq(samples, 1 / config["fs"])
    # FFT（快速傅里叶变换）预备：任何信号都能分解成许多频率正弦波的叠加；FFT 把“时间上的波形”变成“每个频率有多少幅度”，结果叫频谱。
    # np.fft.rfftfreq(n, d) 只生成频谱的横轴：n=1000 点、间隔 1/250 秒，返回 0 到 125 Hz 共 501 个频率值（间隔 0.25 Hz）。
    # rfft 的 r = real：实数信号的频谱正负频率对称，只需一半，501 = 1000/2 + 1。
    responses, harmonic_rows, spectrum_rows = [], [], []
    for subject in subjects:
        print(f"{subject}: 63-channel quality analysis", flush=True)
        response = quality_response(config, subject)    # 处理后的数据
        responses.append(response)
        amplitude = np.abs(np.fft.rfft(response[:, :samples], axis=-1)) * 2 / samples
        # 幅度谱计算，从内往外四步：
        # response[:, :samples]：取前 1000 点（0–4 s）；
        # np.fft.rfft(..., axis=-1)：沿时间轴 FFT → 复数数组（每个频点一个复数，编码幅度和相位），形状 (63, 501)；
        # np.abs(...)：复数的模 = 该频率成分的幅度；
        # * 2 / samples：单边谱归一化。FFT 原始输出的数值与点数有关，除以点数、乘 2（补偿被丢掉的负频率半边），
        # 归一化后的性质：幅度为 A 的纯正弦在频谱上显示为高度 A 的峰——数值才与物理幅度（μV）对应。
        conventional, paper = snr_spectra(amplitude, frequencies)       # 算该受试者两种 SNR，(63, 501)
        for harmonic in range(1, 5):    # 基频 + 第 2/3/4 次谐波 = 12、24、36、48 Hz
            index = int(np.argmin(abs(frequencies - 12 * harmonic)))
            # np.argmin(数组)：返回最小值的下标。abs(frequencies - 目标) 是“每个频点离目标多远”，argmin 找最近的频点。
            # 不直接找“频率恰好等于 12”的下标，是因为这种写法对任何目标频率都稳健，不依赖恰好整除的巧合。

            # 谐波 × 63 通道逐行记录：受试者、通道名、频率、幅度、两种 SNR。24 × 63 × 4 = 6048 行。
            for channel_index, name in enumerate(names):
                harmonic_rows.append(dict(subject=subject, channel=name, frequency_hz=frequencies[index],
                    amplitude_raw_units=amplitude[channel_index, index],
                    snr_neighbor_ratio=conventional[channel_index, index], snr_paper_formula=paper[channel_index, index]))
        # 该受试者 Oz 全部 501 个频点 → spectra_by_subject.csv（24 × 501 行）。
        for index, frequency in enumerate(frequencies):
            spectrum_rows.append(dict(subject=subject, frequency_hz=frequency,
                amplitude_raw_units=amplitude[oz_index, index],
                snr_neighbor_ratio=conventional[oz_index, index], snr_paper_formula=paper[oz_index, index]))
    grand = np.mean(responses, axis=0)  # 跨受试者等权平均
    sections = butter(4, [7, 70], btype="bandpass", fs=config["fs"], output="sos")  # 7–70 Hz 带通滤波
    waveform = sosfiltfilt(sections, grand[oz_index])
    spectrum = np.abs(np.fft.rfft(grand[:, :samples], axis=-1)) * 2 / samples   # 跨人平均后再算频谱
    times = np.arange(grand.shape[-1]) / config["fs"]   # 时间轴
    spectrum_table = pd.DataFrame(spectrum_rows)
    harmonic_table = pd.DataFrame(harmonic_rows)
    harmonic_table.to_csv(output / "harmonics_all_channels.csv", index=False)
    harmonic_table[harmonic_table.channel == "oz"].to_csv(output / "harmonics.csv", index=False)    # 只要oz的
    spectrum_table.to_csv(output / "spectra_by_subject.csv", index=False)
    pd.DataFrame(dict(time_seconds=times, grand_average=grand[oz_index], display_filtered=waveform)).to_csv(
        output / "waveform.csv", index=False)   # 时间、原始波形、显示滤波波形三列
    pd.DataFrame(grand.T, columns=names).to_csv(output / "grand_average_all_channels.csv", index=False)

    # SNR 先逐人计算再等权平均，与跨人平均波形的频谱不是同一统计量。
    figure, axes = plt.subplots(3, 1, figsize=(11, 10), layout="constrained")   # layout="constrained" 自动排版防重叠
    axes[0].plot(times, waveform, linewidth=0.8)
    axes[0].axvline(0, color="red", linestyle="--")
    axes[0].axvline(4, color="red", linestyle="--", label="Stimulus offset")
    axes[0].axvline(config["delay_seconds"], color="green", linestyle="--", label="Visual delay")
    axes[0].axvline(4 + config["delay_seconds"], color="green", linestyle="--")
    axes[0].set(xlabel="Time from stimulus onset (s)", ylabel="Amplitude (source units)",
                title=f"12 Hz target, Oz: {len(subjects)} subjects; 7-70 Hz display filter")
    axes[0].legend()
    axes[1].plot(frequencies, spectrum[oz_index])
    axes[1].set(xlim=(5, 60), xlabel="Frequency (Hz)", ylabel="Amplitude (source units)",
                title="Spectrum of grand-average 0-4 s response; 0.25 Hz resolution")
    average_snr = spectrum_table.groupby("frequency_hz").snr_paper_formula.mean()   # 先逐人 SNR、再跨人平均
    axes[2].plot(average_snr.index, average_snr.values)
    axes[2].set(xlim=(5, 60), xlabel="Frequency (Hz)", ylabel="SNR (amplitude ratio)",
                title="Mean per-subject SNR: paper equation (3), center bin included")
    # 谐波画参考线
    for axis in axes[1:]:
        for frequency in (12, 24, 36, 48):
            axis.axvline(frequency, color="gray", linestyle="--", alpha=0.4)
    figure.savefig(output / "figure4_1_waveform_spectrum.png", dpi=180)
    plt.close(figure)

    # 使用标准坐标而非实测坐标，图标题明确说明该限制。
    figure, axes = plt.subplots(1, 4, figsize=(13, 4), layout="constrained")
    for axis, frequency in zip(axes, (12, 24, 36, 48)):
        index = int(np.argmin(abs(frequencies - frequency)))
        plotted, _ = mne.viz.plot_topomap(spectrum[:, index], info, axes=axis, show=False, cmap="viridis")
        figure.colorbar(plotted, ax=axis, shrink=0.65)
        axis.set_title(f"{frequency} Hz")
    figure.suptitle("12 Hz target harmonics: standard 10-20 coordinates; amplitudes in source units")
    figure.savefig(output / "figure4_2_harmonic_topographies.png", dpi=180)
    plt.close(figure)
