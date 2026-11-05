"""
YSU 24 人 SSVEP 复现主入口，可直接在 PyCharm 中运行。

按顺序执行数据核验、信号质量分析、八分类频率识别、三组异步状态分类，
最终汇总表格和图。默认输出 results_full。
--resume 仅复用来源/参数/代码签名均匹配且成功完成的受试者阶段。
"""

import argparse                             # 解析命令行参数
from datetime import datetime, timezone     # 生成带时区的 UTC 时间戳
import hashlib
import importlib.metadata                   # 读取已安装包的版本号
import json
from pathlib import Path
import platform                             # 读 Python 版本

from threadpoolctl import threadpool_limits     # 管理线程

from asynchronous import asynchronous_evaluation
from data_io import ROOT, audit, mat_path, read_config
from evaluation import frequency_evaluation
from quality_analysis import quality_evaluation
from reporting import summarize_all


def experiment_signature(config, subject, windows, stage):
    """签名涵盖源码和各原始文件的元数据；不把不完整或旧参数输出当作已完成。"""
    contents = dict(config=config, subject=subject, windows=windows, stage=stage,
        code={path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in ROOT.glob("*.py")},
        sources={state: [mat_path(config, subject, state).stat().st_size,
                         mat_path(config, subject, state).stat().st_mtime_ns]
                 for state in ("CS", "NS1", "NS2", "NS3")})
    # 通道对应关系变更也会使旧的完成标记失效。
    contents["channel_table"] = hashlib.sha256((Path(config["data_dir"]) / "Channel Loc.xlsx").read_bytes()).hexdigest()
    return hashlib.sha256(json.dumps(contents, sort_keys=True).encode()).hexdigest()


def run_subject_stage(config, subject, windows, stage, resume):
    """运行单人单阶段；只有所有输出成功写完后才创建完成标记。"""
    folder = Path(config["output_dir"]) / stage
    folder.mkdir(parents=True, exist_ok=True)
    marker = folder / f"{subject}_complete.json"
    signature = experiment_signature(config, subject, windows, stage)
    required = [folder / f"{subject}_{suffix}.csv" for suffix in ("metrics", "predictions")]
    if stage == "asynchronous":
        required.append(folder / f"{subject}_band_selection.json")
    if resume and marker.is_file() and all(path.is_file() for path in required):
        if json.loads(marker.read_text(encoding="utf-8")).get("signature") == signature:
            print(f"Resume: {subject} {stage} already completed", flush=True)
            return
    if stage == "frequency":
        frequency_evaluation(config, subject, windows)
    else:
        asynchronous_evaluation(config, subject)
    marker.write_text(json.dumps(dict(signature=signature, completed_utc=datetime.now(timezone.utc).isoformat()), indent=2), encoding="utf-8")


def main():
    """解析命令、保存环境快照、逐人运行，并在最终汇总时验证预测数量。"""
    parser = argparse.ArgumentParser(description="YSU full offline SSVEP reproduction")
    parser.add_argument("--stage", choices=["audit", "quality", "frequency", "async", "all", "report"], default="all")
    parser.add_argument("--subjects", nargs="+", help="For example: S01 S02")
    parser.add_argument("--windows", nargs="+", type=float)
    parser.add_argument("--config")
    parser.add_argument("--resume", action="store_true", help="Reuse completed subject stages with matching signatures")
    arguments = parser.parse_args()
    config = read_config(arguments.config)
    subjects, windows = arguments.subjects or config["subjects"], arguments.windows or config["windows"]
    if len(set(subjects)) != len(subjects) or not subjects:
        parser.error("Subject identifiers must be unique")
    if any(seconds <= 0 or seconds > 4 for seconds in windows) or len(set(windows)) != len(windows):
        parser.error("Windows must be unique and in (0, 4]")
    if arguments.subjects or arguments.windows:
        label = "_".join(subjects) + "_" + "_".join(f"{seconds:g}s" for seconds in windows)
        config["output_dir"] = str(Path(config["output_dir"]) / "runs" / label)
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    inventory = audit(config, subjects)
    inventory.to_csv(output / "data_inventory.csv", index=False)
    print(f"Audited {len(inventory)} MAT files for {len(subjects)} subjects", flush=True)
    packages = ("numpy", "scipy", "pandas", "matplotlib", "scikit-learn", "openpyxl", "threadpoolctl", "mne")
    manifest = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), python=platform.python_version(),
        subjects=subjects, windows=windows, stage=arguments.stage, config=config,
        versions={name: importlib.metadata.version(name) for name in packages},
        code_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in ROOT.glob("*.py")})
    (output / f"manifest_{arguments.stage}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    # 同时保存环境中全部包版本，environment.yml 则用于安装顶层依赖。
    installed = sorted(f"{distribution.metadata['Name']}=={distribution.version}" for distribution in importlib.metadata.distributions())
    (output / "requirements-lock.txt").write_text("\n".join(installed) + "\n", encoding="utf-8")
    with threadpool_limits(limits=1):
        if arguments.stage in ("quality", "all"):
            quality_evaluation(config, subjects)
        if arguments.stage in ("frequency", "all"):
            for subject in subjects:
                run_subject_stage(config, subject, windows, "frequency", arguments.resume)
        if arguments.stage in ("async", "all"):
            for subject in subjects:
                run_subject_stage(config, subject, windows, "asynchronous", arguments.resume)
        if arguments.stage in ("frequency", "async", "all", "report"):
            summarize_all(config, subjects,
                          include_frequency=arguments.stage != "async",
                          include_async=arguments.stage != "frequency")
    print(f"Finished. Output: {output}", flush=True)


if __name__ == "__main__":
    main()
