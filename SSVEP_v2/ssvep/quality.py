"""信号质量图：Oz 平均波形、幅度谱、SNR 和四个谐波的头皮分布。

数据流：每名受试者把 12 Hz 刺激的 12 个 CS 试次先平均，再做预处理，
得到 63 通道 × 1125 点的响应。之后：

    1. 逐受试者做 FFT，得到幅度谱（0–125 Hz，分辨率 0.25 Hz）；
    2. 按两种 SNR 定义各算一遍，并挑出 12/24/36/48 Hz 四个谐波；
    3. 把受试者的响应等权平均，画 Oz 波形、平均谱和头皮地形图。

为什么先平均试次：预处理是线性运算，先平均再滤波和先滤波再平均在数学上等价，
但前者省内存也省时间。为什么用标准坐标：数据没有记录电极实测位置，
头皮图只能假设 10-20 标准坐标，图标题里必须写明这个限制。

IO 只出现在 run_quality 里：其余函数只做数组运算，方便单独测试。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from .config import sampling_rate
from .data import channel_table, load_quality_response

matplotlib.use("Agg")            # 只出图不弹窗

# 12 Hz 目标响应的基频与第 2/3/4 次谐波。
HARMONIC_FREQUENCIES = (12, 24, 36, 48)


# ---------------------------------------------------------------------------
# 1. 频谱与 SNR（纯数学）
# ---------------------------------------------------------------------------

def amplitude_spectrum(signals: np.ndarray, n_samples: int) -> np.ndarray:
    """单边幅度谱：|rfft| × 2 / 点数，输入 [..., 时间] → [..., 频点]。

    只取前 n_samples 个点（4 秒 = 1000 点），分辨率 = 采样率 / 点数 = 0.25 Hz。
    乘 2 是补偿 rfft 丢掉的负频率半边；除以点数让纯正弦的峰高等于它的物理幅度。
    """
    return np.abs(np.fft.rfft(signals[..., :n_samples], axis=-1)) * 2 / n_samples


def snr_spectra(amplitude: np.ndarray, frequencies: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray]:
    """输入幅度谱 [..., 频点]，返回两种 SNR（同形状，边缘频点为 NaN）。

    邻域 = 与中心频率相差不超过 1 Hz 的频点；分辨率 0.25 Hz 时共 9 个点。

        常规定义 conventional = A(f) / mean(邻域去掉中心)      —— 更严格，排除中心
        论文定义 paper        = (2/Δf+1) × A(f) / sum(邻域含中心)
        其中 2/Δf+1 = 2/0.25+1 = 9，也就是邻域点数，所以 paper 等于“中心幅度 / 邻域均值”。

    为什么边缘留 NaN：±1 Hz 邻域不完整时两种定义都会偏向边界，与原实现一致地跳过。
    """
    conventional = np.full_like(amplitude, np.nan, dtype=float)
    paper = np.full_like(amplitude, np.nan, dtype=float)
    delta = frequencies[1] - frequencies[0]

    for index, frequency in enumerate(frequencies):
        if frequency < 1 or frequency > frequencies[-1] - 1:
            continue                                   # 邻域不完整
        neighboring = np.abs(frequencies - frequency) <= 1 + 1e-9
        excluding = neighboring.copy()
        excluding[index] = False
        conventional[..., index] = amplitude[..., index] / np.maximum(
            amplitude[..., excluding].mean(axis=-1), 1e-15)
        paper[..., index] = (2 / delta + 1) * amplitude[..., index] / np.maximum(
            amplitude[..., neighboring].sum(axis=-1), 1e-15)
    return conventional, paper


# ---------------------------------------------------------------------------
# 2. 电极坐标
# ---------------------------------------------------------------------------

def standard_positions(config: dict) -> tuple[list[str], mne.Info]:
    """返回 (电极名列表, MNE Info)，电极顺序与官方通道表的列号一致。

    电极名来自 Channel Loc.xlsx（小写），坐标来自 MNE 的标准 10-20 蒙太奇。
    缺失任何电极都直接报错，不猜测、不静默丢弃。
    """
    mapping = channel_table(config)
    names = sorted(mapping, key=mapping.get)

    montage = mne.channels.make_standard_montage("colin27_1020")
    positions = {name.casefold(): position
                 for name, position in montage.get_positions()["ch_pos"].items()}
    missing = set(names) - positions.keys()
    if missing:
        raise ValueError(f"标准坐标里缺少这些电极: {missing}")

    info = mne.create_info(names, sampling_rate(config), "eeg")
    dig_montage = mne.channels.make_dig_montage(
        ch_pos={name: positions[name] for name in names}, coord_frame="head")
    info.set_montage(dig_montage)
    return names, info


# ---------------------------------------------------------------------------
# 3. 逐人整理成表格行（纯计算，不写文件）
# ---------------------------------------------------------------------------

def harmonic_rows(subject: str, names: list[str], frequencies: np.ndarray,
                  amplitude: np.ndarray, conventional: np.ndarray,
                  paper: np.ndarray) -> list[dict]:
    """一个人 × 4 个谐波 × 63 通道的幅度与两种 SNR，返回 252 行字典。

    harmonic_rows 是“写 CSV 前的数据行”；每行对应一个通道在一个谐波频点上的取值。
    """
    rows = []
    for harmonic in range(1, 5):
        index = int(np.argmin(np.abs(frequencies - 12 * harmonic)))
        for channel_index, name in enumerate(names):
            rows.append(dict(
                subject=subject, channel=name, frequency_hz=frequencies[index],
                amplitude_raw_units=amplitude[channel_index, index],
                snr_neighbor_ratio=conventional[channel_index, index],
                snr_paper_formula=paper[channel_index, index]))
    return rows


def spectrum_rows(subject: str, frequencies: np.ndarray, amplitude: np.ndarray,
                  conventional: np.ndarray, paper: np.ndarray,
                  oz_index: int) -> list[dict]:
    """一个人 Oz 电极全部 501 个频点的幅度与两种 SNR，返回 501 行字典。"""
    return [dict(subject=subject, frequency_hz=frequency,
                 amplitude_raw_units=amplitude[oz_index, index],
                 snr_neighbor_ratio=conventional[oz_index, index],
                 snr_paper_formula=paper[oz_index, index])
            for index, frequency in enumerate(frequencies)]


# ---------------------------------------------------------------------------
# 4. 画图（只创建 Figure，保存交给 run_quality）
# ---------------------------------------------------------------------------

def draw_waveform_spectrum(config: dict, times: np.ndarray, waveform: np.ndarray,
                           frequencies: np.ndarray, spectrum: np.ndarray,
                           oz_index: int, spectrum_table: pd.DataFrame,
                           n_subjects: int):
    """画三联图：Oz 平均波形、跨人平均谱、逐人 SNR 的跨人平均。

    竖线含义：红色虚线是刺激开始(0 s)和结束(4 s)，绿色虚线是视觉延迟(0.14 s)
    以及刺激结束后同样的延迟。
    """
    delay = config["paradigm"]["delay_seconds"]
    figure, axes = plt.subplots(3, 1, figsize=(11, 10), layout="constrained")

    axes[0].plot(times, waveform, linewidth=0.8)
    axes[0].axvline(0, color="red", linestyle="--")
    axes[0].axvline(4, color="red", linestyle="--", label="Stimulus offset")
    axes[0].axvline(delay, color="green", linestyle="--", label="Visual delay")
    axes[0].axvline(4 + delay, color="green", linestyle="--")
    axes[0].set(xlabel="Time from stimulus onset (s)",
                ylabel="Amplitude (source units)",
                title=f"12 Hz target, Oz: {n_subjects} subjects; 7-70 Hz display filter")
    axes[0].legend()

    axes[1].plot(frequencies, spectrum[oz_index])
    axes[1].set(xlim=(5, 60), xlabel="Frequency (Hz)",
                ylabel="Amplitude (source units)",
                title="Spectrum of grand-average 0-4 s response; 0.25 Hz resolution")

    # SNR 必须先逐人计算再跨人平均，不能拿平均波形去算，否则不是同一个统计量。
    average_snr = spectrum_table.groupby("frequency_hz").snr_paper_formula.mean()
    axes[2].plot(average_snr.index, average_snr.values)
    axes[2].set(xlim=(5, 60), xlabel="Frequency (Hz)", ylabel="SNR (amplitude ratio)",
                title="Mean per-subject SNR: paper equation (3), center bin included")

    for axis in axes[1:]:
        for frequency in HARMONIC_FREQUENCIES:
            axis.axvline(frequency, color="gray", linestyle="--", alpha=0.4)
    return figure


def draw_harmonic_topographies(spectrum: np.ndarray, frequencies: np.ndarray,
                               info: mne.Info):
    """画 12/24/36/48 Hz 四个谐波的 63 通道头皮分布，1×4 子图。

    数值是跨受试者平均后的幅度谱；坐标是标准 10-20，不是实测位置。
    """
    figure, axes = plt.subplots(1, 4, figsize=(13, 4), layout="constrained")
    for axis, frequency in zip(axes, HARMONIC_FREQUENCIES):
        index = int(np.argmin(np.abs(frequencies - frequency)))
        plotted, _ = mne.viz.plot_topomap(spectrum[:, index], info,
                                          axes=axis, show=False, cmap="viridis")
        figure.colorbar(plotted, ax=axis, shrink=0.65)
        axis.set_title(f"{frequency} Hz")
    figure.suptitle("12 Hz target harmonics: standard 10-20 coordinates; "
                    "amplitudes in source units")
    return figure


# ---------------------------------------------------------------------------
# 5. 顶层流程：唯一会读写文件的地方
# ---------------------------------------------------------------------------

def run_quality(config: dict, subjects: list[str]) -> None:
    """对给定受试者做信号质量分析，写出 CSV 和两张图到 paths.output/quality。

    输出文件（与原实现同名）：
        harmonics_all_channels.csv     受试者×谐波×63 通道的幅度与两种 SNR
        harmonics.csv                  同上，只保留 Oz
        spectra_by_subject.csv         每人 Oz 的 501 个频点
        waveform.csv                   Oz 跨人平均波形与 7–70 Hz 显示滤波
        grand_average_all_channels.csv 跨人平均的 63 通道响应
        figure4_1_waveform_spectrum.png
        figure4_2_harmonic_topographies.png
    """
    output = Path(config["paths"]["output"]) / "quality"
    output.mkdir(parents=True, exist_ok=True)

    names, info = standard_positions(config)
    oz_index = names.index("oz")
    fs = sampling_rate(config)
    samples = round(fs * 4)                       # 只用刺激后的 4 秒
    frequencies = np.fft.rfftfreq(samples, 1 / fs)

    responses, all_harmonic_rows, all_spectrum_rows = [], [], []
    for subject in subjects:
        print(f"{subject}: 63 通道信号质量分析", flush=True)
        response = load_quality_response(config, subject)
        responses.append(response)
        amplitude = amplitude_spectrum(response, samples)
        conventional, paper = snr_spectra(amplitude, frequencies)
        all_harmonic_rows += harmonic_rows(subject, names, frequencies,
                                           amplitude, conventional, paper)
        all_spectrum_rows += spectrum_rows(subject, frequencies, amplitude,
                                           conventional, paper, oz_index)

    # 跨受试者等权平均后再画波形和频谱。
    grand = np.mean(responses, axis=0)
    display_filter = butter(4, [7, 70], btype="bandpass", fs=fs, output="sos")
    waveform = sosfiltfilt(display_filter, grand[oz_index])
    spectrum = amplitude_spectrum(grand, samples)
    times = np.arange(grand.shape[-1]) / fs

    harmonic_table = pd.DataFrame(all_harmonic_rows)
    spectrum_table = pd.DataFrame(all_spectrum_rows)
    harmonic_table.to_csv(output / "harmonics_all_channels.csv", index=False)
    harmonic_table[harmonic_table.channel == "oz"].to_csv(output / "harmonics.csv", index=False)
    spectrum_table.to_csv(output / "spectra_by_subject.csv", index=False)
    pd.DataFrame(dict(time_seconds=times, grand_average=grand[oz_index],
                      display_filtered=waveform)).to_csv(output / "waveform.csv", index=False)
    pd.DataFrame(grand.T, columns=names).to_csv(
        output / "grand_average_all_channels.csv", index=False)

    figure = draw_waveform_spectrum(config, times, waveform, frequencies, spectrum,
                                    oz_index, spectrum_table, len(subjects))
    figure.savefig(output / "figure4_1_waveform_spectrum.png", dpi=180)
    plt.close(figure)

    figure = draw_harmonic_topographies(spectrum, frequencies, info)
    figure.savefig(output / "figure4_2_harmonic_topographies.png", dpi=180)
    plt.close(figure)
