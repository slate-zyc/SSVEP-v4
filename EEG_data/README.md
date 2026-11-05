# 共享原始 EEG 数据

本目录由 `SSVEP_v1`、`SSVEP_v2`、`SSVEP_v3`、`SSVEP_v4` 共用。

- 数据来源：[YSU asynchronous SSVEP-BCI，Figshare v3](https://doi.org/10.6084/m9.figshare.24906300.v3)。
- 对应论文：[Zhao et al., 2024](https://doi.org/10.1080/27706710.2024.2418650)。
- 文件布局：`S01/` 至 `S24/`，每人含 `sub_XX_data_CS.mat`、`sub_XX_data_NS1.mat`、`sub_XX_data_NS2.mat`、`sub_XX_data_NS3.mat`；另有 `Channel Loc.xlsx` 和 `Subjects' Info.xlsx`。

原始文件约 18.7 GiB，保留在本地，不上传至 Git。克隆项目后请从官方来源下载并解压到本目录；不要把数据分别复制到四个版本下。使用和引用要求以官方发布页为准。
