# SSVEP_v3：按论文小节组织的独立脚本

复现 [Zhao 等人 2024 年异步 SSVEP 数据集论文](https://doi.org/10.1080/27706710.2024.2418650)的第 4.1、4.2、4.3 节。每个脚本顺着数据处理流程展开，便于逐行阅读。

所有脚本直接读取大项目根目录下的 `../EEG_data/`，结果保存在本版本的 `results_full/`。路径根据脚本位置计算，因此可从任意工作目录启动。环境使用 [environment.yml](environment.yml)。

在本目录运行：

```powershell
conda activate SSVEP
python reproduce_4_1.py
python reproduce_4_2_fbcca.py
python reproduce_4_2_trca.py
python reproduce_4_2_etrca.py
python reproduce_4_3_fs.py
python reproduce_4_3_ifb_ocn.py
```

| 脚本 | 对应任务 |
| --- | --- |
| `reproduce_4_1.py` | 图 4：时域波形、幅度谱、谐波头皮图、SNR |
| `reproduce_4_2_fbcca.py` | 图 5/6：FBCCA 频率识别与 ITR |
| `reproduce_4_2_trca.py` | 图 5/6：TRCA 频率识别与 ITR |
| `reproduce_4_2_etrca.py` | 图 5/6：eTRCA 频率识别与 ITR |
| `reproduce_4_3_fs.py` | 图 7：FS-eTRCA 与融合 FBCCA 的九分类 |
| `reproduce_4_3_ifb_ocn.py` | 图 7：IFB-OCN 重建版 CS/NS 二分类 |

已有结果保持原样。独立复现中采用的协议与参数不代表原作者完整实现；尤其 IFB-OCN 应保留“重建版”限定。数据来源、参考文献和其他版本见 [项目说明](../README.md)，论文阅读资料见 [文献仓库](https://github.com/slate-zyc/relevant_work_for_thesis)。
