"""第 4.3 节：单文件运行 IFB-OCN 重建版，24 人分别完成 CS 与三种 NS 的二分类。

每行可执行代码均有中文注释；直接使用现有 SSVEP Conda 环境运行本文件。
从原始 MAT 重新计算，不导入项目内的其他脚本，不读取或写入中间缓存。
"""

# ==================== 来源、已知方法与本文件的补充选择 ====================
# 目标论文：https://doi.org/10.1080/27706710.2024.2418650，第 4.3 节明确取 Fp1、1 秒窗、12 折、最佳三个频带、线性 SVM。
# IFB-OCN 原文：https://doi.org/10.1016/j.jneumeth.2020.108776；可取得的正文未完整给出本数据集上的网络参数和优化配置。
# 网络定义参照同组 OCN 原文：https://doi.org/10.1155/2019/2167871；延迟状态向量为节点，欧氏距离阈值决定边，特征为平均度和平均局部聚类系数。
# 五个候选频带来自同组公开论文的 IFBOCN 部分：https://doi.org/10.1109/TNSRE.2023.3275547；delta、theta、alpha、beta、gamma 的边界分别见 BANDS。
# 逐片段标准化、四阶 Butterworth 滤波、八组网络候选参数、内部三折验证及 SVM 的 C=1 是本文件补充的实现选择，不是已核实的原作者参数。
# 因此本文件及输出明确标记为 IFB-OCN-reconstructed；它复现可得的方法结构，不宣称与原作者实现或目标论文数值完全一致。
# 训练内验证先为每个频带选网络参数，再选验证准确率最高的三个频带，拼成六维特征；外层测试标签不参与任何选择。
# CS 每种频率有 12 次重复，取刺激后 0.14-1.14 秒，共 96 段；NS1/NS2 的 24 个四秒记录各切四段，NS3 的 48 个两秒记录各切两段，也各为 96 段。
# 同一 NS 原始记录的全部相邻片段始终属于同一外层折和同一内层验证组，防止同一条记录同时进入训练和测试。
# NS 原始记录编号对 12 取模分折，这是目标论文未详细给出的分组实现选择；每折恰好测试八个 CS 和八个 NS。
# 标签固定为 1=CS、0=NS，统计二分类准确率；保留所有人及原始频率标签，包括 S11。

from itertools import product  # product 构造预先固定的延迟、嵌入维度、连接半径候选组合，只用于训练内部选择。
from pathlib import Path  # Path 用于定位原始数据和当前脚本所在的结果目录，不依赖启动脚本时的工作目录。
from time import perf_counter  # 单调计时器记录本次从读取数据到保存图表的总耗时，单位为秒。
import matplotlib  # 先导入绘图库本体，便于在 pyplot 创建画布前指定不弹窗的后端。
matplotlib.use("Agg")  # 使用 Agg 后端直接输出 PNG，完整计算期间不需要图形窗口交互。
import matplotlib.pyplot as plt  # pyplot 提供均值图、箱线图、图例及保存图片接口。
import numpy as np  # NumPy 处理脑电数组、网络邻接矩阵、训练测试索引和分类指标。
import pandas as pd  # pandas 读取通道名称表，整理逐试次预测、选带记录和跨人汇总并保存 CSV。
from scipy.io import loadmat  # loadmat 读取原始 MAT 中指定变量，不加载旧流程的任何缓存文件。
from scipy.signal import butter, filtfilt, iirnotch, resample_poly, sosfiltfilt  # 分别用于四阶带通、双向陷波、陷波器设计、抗混叠降采样及二阶节双向滤波。
from scipy.spatial.distance import cdist  # cdist 计算同一片段全部延迟状态向量之间的欧氏距离，决定网络节点之间是否连接。
from sklearn.model_selection import StratifiedGroupKFold  # 内层分层分组验证同时保留 CS/NS 比例，并保持同一 NS 原始记录的全部片段同组。
from sklearn.pipeline import make_pipeline  # 将特征缩放与 SVM 串联，使每次内部拟合只用其训练部分估计特征均值和标准差。
from sklearn.preprocessing import StandardScaler  # 对两维或六维网络特征做训练集标准化，测试特征使用训练部分学到的尺度。
from sklearn.svm import SVC  # SVC 的 kernel="linear" 对应线性核 SVM，用于二分类及训练内选带。
from threadpoolctl import threadpool_limits  # 控制小型网络矩阵乘法所用的 BLAS 线程数量，避免频繁开启线程拖慢计算。

DATA_DIR = Path(__file__).resolve().parent.parent / "EEG_data"  # 四个版本共用根目录数据；路径不依赖当前工作目录。
OUTPUT_DIR = Path(__file__).resolve().parent / "results_full" / "section4_3_ifb_ocn"  # 该脚本的最终预测、指标、选带日志和图表全部写到独立结果目录。
SUBJECTS = [f"S{number:02d}" for number in range(1, 25)]  # 按顺序处理 S01-S24，保留全部 24 名受试者。
STATES = ["NS1", "NS2", "NS3"]  # 三种 NS 分别与同一批 CS 构成独立的二分类任务，不把三种 NS 混成一个训练集。
FREQUENCIES = np.array([8, 9, 10.5, 11, 12, 13, 14, 15])  # 原始 CS 类别轴的八种刺激频率，保留 10.5 Hz 和原始类别顺序。
ORIGINAL_FS = 5000  # 官方 MAT 数据采样率为 5000 Hz，CS 有 22500 点，NS 记录有 20000 或 10000 点。
FS = 250  # 分析采样率为 250 Hz，每个一秒片段对应 250 个点；降采样先执行抗混叠 FIR 滤波。
FOLDS = 12  # 外层验证共 12 折，CS 的重复编号 0-11 直接决定对应外层测试折。
START = round(0.14 * FS)  # CS 截窗从第 35 个降采样点开始，精确对应刺激开始后的 0.14 秒。
BANDS = [(0.5, 3), (4, 7), (8, 13), (13, 30), (30, 60)]  # 五个候选个体化频带，单位 Hz，最后仅使用训练内选择的三个。
EMBEDDINGS = list(product([2, 4], [3, 5]))  # 四组 (延迟点数 tau,嵌入维度 m)，tau=2、4 点分别对应 8、16 毫秒。
RADII = [0.5, 1.0]  # 逐片段单位标准差信号空间中的两个欧氏连接半径，与原始数据的微伏幅值阈值不同。
PARAMETERS = [(delay, dimension, radius) for delay, dimension in EMBEDDINGS for radius in RADII]  # 把四组嵌入与两个半径排成八组候选，索引为嵌入编号*2+半径编号。
METHOD = "IFB-OCN-reconstructed"  # 所有输出都保留重建版标识，避免将补充参数选择误称为已核实的原作者实现。
PAPER_ACCURACY = np.array([70.51, 72.35, 69.34])  # 目标论文第 4.3 节给出的 CS-NS1、CS-NS2、CS-NS3 二分类平均准确率，单位百分数。

threadpool_limits(limits=1, user_api="blas")  # 网络只有约 240 个节点，限制为单线程 BLAS，减少大量小矩阵运算的线程调度开销。
started = perf_counter()  # 记录运行起点，最后输出完整计算耗时，不保存运行状态供下次接续。
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # 建立最终结果目录；重新运行从原始 MAT 开始并覆盖同名输出。
channel_table = pd.read_excel(DATA_DIR / "Channel Loc.xlsx")  # 从官方电极位置表读取 Fp1 对应的通道编号。
channel_lookup = dict(zip(channel_table["Channel Name"].str.casefold(), channel_table["Channel Index"] - 1))  # 将官方从 1 开始的编号转为 NumPy 零基编号，并统一电极名称大小写。
channel_index = channel_lookup["fp1"]  # 本方法只用目标论文指定的 Fp1，不使用后枕区的九个频率识别电极。
notch_b, notch_a = iirnotch(50, Q=30, fs=FS)  # 设计 50 Hz、Q=30 的陷波器，与已有 4.2 脚本保持相同公共预处理。
prediction_rows = []  # 保存 24 人*3 种 NS*192 段=13824 条外层测试预测，不保存训练集预测作为评价结果。
metric_rows = []  # 保存每人每种 NS 的一行指标，总计 24*3=72 行。
selection_rows = []  # 保存每人、每种 NS、每外层折所选三个频带及其参数，便于审查选带只使用训练数据。
confusion = np.zeros((3, 2, 2), dtype=int)  # 三种状态分别建立二分类混淆矩阵，行是真实类别，列是预测类别，0=NS、1=CS。

for subject in SUBJECTS:  # 每次处理一人，读取四个原始状态文件后计算全部三组二分类任务。
    print(f"[{subject}] 读取 Fp1，计算 IFB-OCN 重建版", flush=True)  # 显示当前受试者进度，flush 立即输出，便于观察耗时较长的特征提取。
    raw = loadmat(DATA_DIR / subject / f"sub_{subject[1:]}_data_CS.mat", variable_names=["data_CS"])["data_CS"]  # 读取该人的 (8,63,22500,12) CS 数据，不使用已有频率识别结果。
    cs = raw[:, channel_index, :, :].transpose(0, 2, 1).copy()  # 只选 Fp1 并转为 (8,12,22500)，使最后一轴统一为时间。
    del raw  # 释放完整 63 通道大数组的内存引用，保留当前单通道所需数据。
    cs -= cs.mean(axis=-1, keepdims=True)  # 各类各重复分别减去原始 4.5 秒记录的时间均值。
    cs = resample_poly(cs, up=1, down=ORIGINAL_FS // FS, axis=-1)  # 经过抗混叠多相 FIR 滤波，从 5000 Hz 降到 250 Hz。
    cs = filtfilt(notch_b, notch_a, cs, axis=-1)  # 对完整试次做双向 50 Hz 陷波，去除工频成分并避免引入相位偏移。
    cs = cs[..., START:START + FS].reshape(96, FS)  # 从每类每次重复提取 0.14-1.14 秒并展平，顺序为类别优先、重复次之。
    signals_by_state = [cs]  # 按 CS、NS1、NS2、NS3 保存当前人的一秒片段；这是本次计算的内存数据，不写中间文件。
    ns_groups_by_state = []  # 保存 NS 片段所属原始记录的零基编号，用于保持同一原始记录的全部片段同折。
    ns_segments_by_state = []  # 保存每个 NS 片段在原始记录内部的位置，供最终预测追溯。
    for state in STATES:  # 三种 NS 文件分别处理，各自得到 96 个一秒片段。
        variable = f"data_{state}"  # MAT 变量名与状态一致，例如 NS1 对应 data_NS1。
        raw = loadmat(DATA_DIR / subject / f"sub_{subject[1:]}_data_{state}.mat", variable_names=[variable])[variable]  # NS1/NS2 形状 (63,20000,24)，NS3 形状 (63,10000,48)。
        ns = raw[channel_index].T.copy()  # 提取 Fp1 并转为 (原始记录数,原始时间点数)，每行是完整的一条 NS 记录。
        del raw  # 释放未选取的 62 个通道，只保留本方法使用的单通道数据。
        ns -= ns.mean(axis=-1, keepdims=True)  # 对每条完整 NS 记录分别去时间均值，不跨原始记录计算均值。
        ns = resample_poly(ns, up=1, down=ORIGINAL_FS // FS, axis=-1)  # 降至 250 Hz，四秒记录变成 1000 点，两秒记录变成 500 点。
        ns = filtfilt(notch_b, notch_a, ns, axis=-1)  # 在各条完整原始记录内部做双向 50 Hz 陷波，不跨记录拼接滤波。
        segments_per_trial = ns.shape[-1] // FS  # NS1/NS2 每记录分为四段，NS3 分为两段；这里只处理固定官方数据的完整一秒窗口。
        ns_groups_by_state.append(np.repeat(np.arange(len(ns)), segments_per_trial))  # 原始记录号在它的所有片段中重复，例如 NS1 的 0 号记录对应四个 0。
        ns_segments_by_state.append(np.tile(np.arange(segments_per_trial), len(ns)))  # 对每条记录重复 0-3 或 0-1，标识片段的起始秒位置。
        signals_by_state.append(ns.reshape(96, FS))  # 直接按每条记录的时间顺序切成 96 段，不重叠、不丢弃最后一秒。
    signals = np.concatenate(signals_by_state, axis=0)  # 按四种状态合成 (384,250) 数组，每种状态连续占据 96 行。
    features = np.empty((384, len(BANDS), len(PARAMETERS), 2))  # 保存每个片段、每个频带、每组网络参数的平均度与聚类系数两维特征。
    for band_index, band in enumerate(BANDS):  # 五个候选频带全部提特征；只有后续训练内评估决定采用哪三个。
        sections = butter(4, band, btype="bandpass", fs=FS, output="sos")  # 固定采用四阶 Butterworth 带通，输出二阶节以稳定表示滤波器系数。
        filtered = sosfiltfilt(sections, signals, axis=-1)  # 对各个一秒片段分别做零相位带通，不把不同试次串接到同一滤波序列。
        for sample_index, signal in enumerate(filtered):  # 网络节点来自当前片段的延迟状态向量，不来自多个电极或其他样本。
            standardized = (signal - signal.mean()) / signal.std()  # 每个片段自身标准化为零均值、单位标准差，这一步不使用标签或其他片段的数据。
            for embedding_index, (delay, dimension) in enumerate(EMBEDDINGS):  # 同一嵌入下两个半径共用一次距离计算，避免重复构造相同的状态向量。
                node_count = FS - (dimension - 1) * delay  # 延迟向量必须完整落在一秒信号中，节点数例如 250-(5-1)*4=234。
                embedded = standardized[np.arange(node_count)[:, None] + np.arange(dimension)[None, :] * delay]  # 构造 (节点数,m) 状态矩阵，第 i 行为 x_i、x_(i+tau)、直到 x_(i+(m-1)*tau)。
                distances = cdist(embedded, embedded, metric="euclidean")  # 计算节点间对称欧氏距离矩阵，同一状态向量与自身的距离为零。
                for radius_index, radius in enumerate(RADII):  # 对预定的两个半径分别构图，半径和网络参数均由外层训练集内部验证选择。
                    adjacency = (distances <= radius).astype(float)  # 距离不大于半径时连边，得到对称二值邻接矩阵。
                    np.fill_diagonal(adjacency, 0)  # 按无自环网络的数学定义清除对角线，自身不能算作邻居。
                    degrees = adjacency.sum(axis=1)  # 每行边数即当前节点的度 k_i，网络平均度是它们的均值。
                    closed_paths = ((adjacency @ adjacency) * adjacency).sum(axis=1)  # A² 与 A 逐元素相乘后按行求和，得到每个节点参与的闭合三步路径数，即三角形数量的两倍。
                    clustering = np.divide(closed_paths, degrees * (degrees - 1), out=np.zeros_like(degrees), where=degrees > 1)  # 局部聚类系数为闭合路径数/[k_i*(k_i-1)]；度小于二时按网络定义取零，所有节点仍参与均值。
                    features[sample_index, band_index, embedding_index * len(RADII) + radius_index] = [degrees.mean(), clustering.mean()]  # 按固定八组参数顺序保存两维网络特征，不将任何标签加入特征向量。
        print(f"  网络特征 {band[0]:g}-{band[1]:g} Hz 已完成", flush=True)  # 每完成一个频带报告进度，避免构图阶段长期没有输出。

    for state_index, state in enumerate(STATES):  # 分别执行 CS-NS1、CS-NS2 和 CS-NS3 的外层验证，不共享不同 NS 的监督训练信息。
        inputs = np.concatenate([features[:96], features[(state_index + 1) * 96:(state_index + 2) * 96]])  # 当前二分类只取 CS 的 96 行和这一种 NS 的 96 行。
        labels = np.r_[np.ones(96, dtype=int), np.zeros(96, dtype=int)]  # 前 96 段为 CS=1，后 96 段为 NS=0，类别数量相等。
        original_trials = np.r_[np.arange(96), 96 + ns_groups_by_state[state_index]]  # CS 各段来自独立原始试次，NS 共享记录号；加 96 防止 CS 和 NS 组号意外相同。
        segment_indices = np.r_[np.zeros(96, dtype=int), ns_segments_by_state[state_index]]  # CS 只截一段，编号记为 0；NS 保留原始记录内的分段编号。
        assignments = np.r_[np.tile(np.arange(FOLDS), 8), ns_groups_by_state[state_index] % FOLDS]  # CS 按重复号分折，NS 按原始记录号对 12 取模分折，每折八个 CS 加八个 NS。
        predicted = np.empty(192, dtype=int)  # 用每个片段唯一一次外层测试预测填满，后面只根据这一数组统计指标。
        for fold in range(FOLDS):  # 共十二次外层训练测试，每个片段恰好被测试一次。
            train = np.flatnonzero(assignments != fold)  # 当前折 176 个训练片段的位置，测试片段完全不参与内部选带和选网络参数。
            test = np.flatnonzero(assignments == fold)  # 当前折 16 个测试片段的位置，同一 NS 原始记录的全部片段都在这一侧。
            splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=20261105)  # 内部三折只在 176 个外层训练片段中划分，固定随机种子保证重复运行相同。
            inner_splits = list(splitter.split(inputs[train], labels[train], original_trials[train]))  # 划分使用原始记录组号，确保内层验证也不把相邻 NS 片段拆到两侧。
            best_parameters = np.empty(len(BANDS), dtype=int)  # 每个候选频带分别保存内部验证最好的网络参数编号。
            band_accuracies = np.empty(len(BANDS))  # 保存五个频带各自最优参数的内部验证准确率，供后续排序。
            for band_index in range(len(BANDS)):  # 每个频带单独用两维网络特征比较八组参数。
                parameter_accuracies = []  # 保存当前频带八组候选在同一内部划分上的验证正确比例。
                for parameter_index in range(len(PARAMETERS)):  # 参数顺序预先固定，不根据测试结果扩展搜索范围。
                    candidate = inputs[train, band_index, parameter_index]  # 当前候选仅取外层训练片段，形状为 (176,2)。
                    correct = 0  # 累计三个内部验证折的正确数，每个外层训练片段作为内部验证样本一次。
                    for fit_indices, validation_indices in inner_splits:  # 下标是相对于 176 行外层训练数据的位置，不是全体 192 行的位置。
                        model = make_pipeline(StandardScaler(), SVC(kernel="linear", C=1.0))  # 特征缩放与线性 SVM 一起拟合，C 固定为 1，不用外层测试准确率调节。
                        model.fit(candidate[fit_indices], labels[train][fit_indices])  # 只用内部训练部分拟合缩放参数及分类超平面。
                        validation_prediction = model.predict(candidate[validation_indices])  # 将内部验证部分作为未知样本预测，未参与这一模型的拟合。
                        correct += np.count_nonzero(validation_prediction == labels[train][validation_indices])  # 累计内部验证正确数，最后除以 176 得到按片段等权的验证准确率。
                    parameter_accuracies.append(correct / len(train))  # 每个外层训练样本在三个验证折中恰好出现一次，分母固定为 176。
                best_parameters[band_index] = np.argmax(parameter_accuracies)  # 选内部验证准确率最大的候选，平分时取预定顺序中更早的一组。
                band_accuracies[band_index] = parameter_accuracies[best_parameters[band_index]]  # 保存当前频带最优网络参数所取得的内部验证准确率。
            selected = np.argsort(-band_accuracies, kind="stable")[:3]  # 按内部验证准确率由高到低选择三个频带，平分时保留预定频带顺序。
            combined = np.concatenate([inputs[:, band_index, best_parameters[band_index]] for band_index in selected], axis=1)  # 将三个已选频带的两维特征拼为 (192,6)，不平均掉不同频带的信息。
            model = make_pipeline(StandardScaler(), SVC(kernel="linear", C=1.0))  # 用相同固定线性 SVM 处理六维融合特征，缩放仍在训练管线内部完成。
            model.fit(combined[train], labels[train])  # 在全部 176 个外层训练片段上拟合最终模型，16 个测试片段始终留出。
            predicted[test] = model.predict(combined[test])  # 为当前折测试片段写入预测，十二折后每个位置恰好被填写一次。
            decision_values = model.decision_function(combined[test])  # 保存有符号 SVM 决策值，正值偏向 CS、负值偏向 NS，便于审查接受和拒绝结果。
            for rank, band_index in enumerate(selected):  # 记录三个已选频带的排名、参数及内部验证表现，每折三行。
                delay, dimension, radius = PARAMETERS[best_parameters[band_index]]  # 把参数编号还原为可读的延迟点数、嵌入维度和标准化空间半径。
                selection_rows.append({"subject": subject, "state": state, "fold": fold + 1, "rank": rank + 1, "low_hz": BANDS[band_index][0], "high_hz": BANDS[band_index][1], "delay_samples": delay, "dimension": dimension, "radius": radius, "inner_accuracy": band_accuracies[band_index]})  # 记录选带依据，inner_accuracy 仅来自外层训练集内部验证。
            for sample_index, decision in zip(test, decision_values):  # 按当前折测试位置保存预测及原始记录信息，便于独立重建分组和指标。
                prediction_rows.append({"subject": subject, "state": state, "method": METHOD, "fold": fold + 1, "sample": sample_index, "original_trial": original_trials[sample_index], "segment": segment_indices[sample_index], "true_class": labels[sample_index], "predicted_class": predicted[sample_index], "decision_value": decision})  # true_class/predicted_class 为二分类 0、1，不能当作九分类结果比较。
        accuracy = np.mean(predicted == labels)  # 二分类准确率基于当前人的全部 192 次外层测试结果，不使用内部验证准确率代替。
        cs_detection = np.mean(predicted[:96] == 1)  # 真正 CS 被识别为 CS 的比例，即控制状态检出率 TPR。
        ns_false_positive = np.mean(predicted[96:] == 1)  # 真正 NS 被误触发为 CS 的比例，即非控制状态假阳性率 FPR。
        np.add.at(confusion[state_index], (labels, predicted), 1)  # 将当前人的真实/预测类别配对加入当前 NS 任务的二分类混淆矩阵。
        metric_rows.append({"subject": subject, "state": state, "method": METHOD, "accuracy": accuracy, "cs_detection_rate": cs_detection, "ns_false_positive_rate": ns_false_positive, "n_classes": 2, "n_samples": len(labels)})  # 保存当前人当前 NS 任务的指标，不把二分类称作九分类。
        print(f"  CS-{state}: accuracy={accuracy:.2%}, TPR={cs_detection:.2%}, FPR={ns_false_positive:.2%}", flush=True)  # 显示本人的二分类结果和两类状态表现。

predictions = pd.DataFrame(prediction_rows)  # 整理全部 13824 条外层测试预测，每条记录可追溯到原始记录和片段。
metrics = pd.DataFrame(metric_rows)  # 整理 72 行每人每种 NS 的最终指标，后续跨人等权平均。
summary = metrics.groupby("state", as_index=False).agg(accuracy_mean=("accuracy", "mean"), accuracy_sd=("accuracy", "std"), cs_detection_mean=("cs_detection_rate", "mean"), ns_false_positive_mean=("ns_false_positive_rate", "mean"))  # 按三种 NS 分组，计算 24 人等权准确率、样本标准差及状态检出/误触发率。
summary["accuracy_sem"] = summary.accuracy_sd / np.sqrt(len(SUBJECTS))  # 标准误为跨人样本标准差除以 sqrt(24)，用于均值图的误差线。
comparison = summary[["state", "accuracy_mean"]].rename(columns={"accuracy_mean": "reproduced"})  # 原文只报告平均准确率，因此对照表采用同一指标，不猜测未给出的 TPR/FPR。
comparison["method"] = METHOD  # 对照表保留重建版标识，不能把补充参数配置误认为原作者完整配置。
comparison["reproduced"] *= 100  # 将计算得到的 0-1 准确率转成百分数，与原文数值的单位一致。
comparison["paper"] = PAPER_ACCURACY  # 三行顺序为 NS1、NS2、NS3，对应原文的 70.51%、72.35%、69.34%。
comparison["difference_percentage_points"] = comparison.reproduced - comparison.paper  # 有符号差值单位是百分点，直接反映与原文的差异。
confusion_table = pd.DataFrame({"state": np.repeat(STATES, 4), "true_class": np.tile(np.repeat([0, 1], 2), 3), "predicted_class": np.tile([0, 1], 6), "count": confusion.ravel()})  # 三个 2*2 混淆矩阵展平成十二行，行真实、列预测，类别 0=NS、1=CS。
predictions.to_csv(OUTPUT_DIR / "predictions.csv", index=False)  # 保存逐片段预测及其来源，重新运行会覆盖最终输出，下一次运行不读取本表。
metrics.to_csv(OUTPUT_DIR / "subject_metrics.csv", index=False)  # 保存 24 人各三种任务的 72 行指标，用于独立复算和个体分布图。
summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)  # 保存三种任务的跨人汇总，不保存任何特征缓存。
comparison.to_csv(OUTPUT_DIR / "comparison_with_paper.csv", index=False)  # 保存计算值、原文值及差值，明确区分实际计算与论文参考。
pd.DataFrame(selection_rows).to_csv(OUTPUT_DIR / "selected_bands.csv", index=False)  # 保存 24*3*12*3=2592 行训练内选带信息，记录最终模型使用的频带和网络参数。
confusion_table.to_csv(OUTPUT_DIR / "confusion_matrices.csv", index=False)  # 保存三个二分类任务的累计混淆计数，每种任务总计 24*192=4608。

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})  # 统一字体并隐藏顶部和右侧边框，使最终图表清楚可读。
figure, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")  # 一张图包含跨人均值对照和 24 人个体分布两个面板。
figure.suptitle("Section 4.3 | IFB-OCN reconstructed | Binary CS/NS", fontsize=14)  # 标题明确当前方法为重建版，任务是 CS/NS 二分类。
positions = np.arange(3)  # 三种 NS 任务在横轴上的位置固定为 0、1、2。
axes[0].bar(positions, summary.accuracy_mean * 100, yerr=summary.accuracy_sem * 100, capsize=4, color="#2074a5", label="Reproduced mean +/- SEM")  # 左图显示跨 24 人平均准确率和标准误，二者均转成百分数。
axes[0].scatter(positions, PAPER_ACCURACY, marker="D", facecolors="none", edgecolors="#d94b46", s=65, label="Paper mean")  # 空心菱形标出目标论文给出的参考平均准确率，不影响模型计算。
axes[0].set(xticks=positions, xticklabels=STATES, ylabel="Binary accuracy (%)", ylim=(0, 105), title="(a) CS versus each NS state")  # 纵轴明确为二分类准确率，不与九分类的类别均衡准确率混淆。
axes[0].legend(fontsize=9)  # 图例说明实测均值和论文参考点，误差线代表跨人 SEM。
axes[1].boxplot([metrics.loc[metrics.state == state, "accuracy"].to_numpy() * 100 for state in STATES], tick_labels=STATES, showmeans=True)  # 右图用全部 24 人的实际准确率画箱线图，另外标出均值。
axes[1].set(ylabel="Binary accuracy (%)", ylim=(0, 105), title="(b) Accuracy distribution: 24 subjects")  # 个体分布保留低表现者，纵轴为百分数。
figure.savefig(OUTPUT_DIR / "ifb_ocn_accuracy.png", dpi=200)  # 输出 2400*1000 像素 PNG，供直接查看与论文比较。
plt.close(figure)  # 图像写入磁盘后关闭画布，释放绘图资源。
print(comparison.to_string(index=False, float_format="%.4f"), flush=True)  # 打印三种任务的实际数值和原文差异，不把成功运行表述为数值一致。
print(f"完成：{OUTPUT_DIR}\n用时：{perf_counter() - started:.1f} 秒", flush=True)  # 输出结果目录及完整运行耗时，便于直接定位 CSV 和图表。
