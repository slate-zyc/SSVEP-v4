"""命令行入口：按阶段运行整个复现流程。

常用命令
--------
    python run.py                              # 全部阶段、全部 24 人（耗时长）
    python run.py --subjects S01               # 只跑一个人，快速体验
    python run.py --stage frequency            # 只跑频率识别
    python run.py --stage async --subjects S01
    python run.py --stage report               # 只重新汇总/画图
    python run.py --stage verify               # 独立核验已有结果

输出默认写在 results/ 下；指定受试者或窗长时，结果写到 results/runs/<标签>/，
避免“部分数据的结果”和“全量结果”混在一起。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

from threadpoolctl import threadpool_limits

from ssvep.asynchronous import run_asynchronous
from ssvep.config import ROOT, load_config
from ssvep.data import audit
from ssvep.frequency import run_frequency
from ssvep.quality import run_quality
from ssvep.report import summarize_all
from ssvep.verify import verify_all

STAGES = ("audit", "quality", "frequency", "async", "report", "verify", "all")


def write_manifest(config: dict, subjects: list[str], windows: list[float], stage: str) -> None:
    """保存本次运行的参数、软件版本和源码哈希，方便日后追溯“这份结果是谁算的”。"""
    output = Path(config["paths"]["output"])
    output.mkdir(parents=True, exist_ok=True)
    packages = ("numpy", "scipy", "pandas", "matplotlib", "scikit-learn", "openpyxl", "mne")
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "stage": stage,
        "subjects": subjects,
        "windows": windows,
        "config": config,
        "versions": {name: importlib.metadata.version(name) for name in packages},
        "code_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in sorted((ROOT / "ssvep").glob("*.py"))},
    }
    (output / f"manifest_{stage}.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def run_subject_stage(config: dict, subject: str, windows: list[float],
                      stage: str, skip_existing: bool) -> None:
    """跑一个人的一个阶段；--skip-existing 时若输出已存在就跳过。"""
    folder = Path(config["paths"]["output"]) / stage
    if skip_existing:
        required = [folder / f"{subject}_{suffix}.csv" for suffix in ("metrics", "predictions")]
        if stage == "asynchronous":
            required.append(folder / f"{subject}_band_selection.json")
        if all(path.is_file() for path in required):
            print(f"跳过 {subject} {stage}（结果已存在）", flush=True)
            return

    if stage == "frequency":
        run_frequency(config, subject, windows)
    else:
        run_asynchronous(config, subject)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SSVEP 异步数据集复现（易读版）")
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--subjects", nargs="+", help="例如 S01 S02；默认用 config.json 里的全部 24 人")
    parser.add_argument("--windows", nargs="+", type=float, help="频率识别的窗长（秒），默认 1 2 3 4")
    parser.add_argument("--config", help="改用别的配置文件")
    parser.add_argument("--output", help="覆盖输出目录")
    parser.add_argument("--skip-existing", action="store_true", help="跳过已有结果的受试者阶段")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    config = load_config(args.config)
    subjects = args.subjects or config["subjects"]
    windows = args.windows or config["frequency"]["windows_seconds"]

    if len(set(subjects)) != len(subjects):
        raise SystemExit("受试者编号不能重复")
    if any(not 0 < seconds <= 4 for seconds in windows) or len(set(windows)) != len(windows):
        raise SystemExit("窗长必须在 (0, 4] 秒内且不能重复")

    # 只跑部分受试者/窗长时，把结果放到单独的子目录，避免覆盖全量结果。
    if args.output:
        config["paths"]["output"] = args.output
    elif args.subjects or args.windows:
        label = "_".join(subjects) + "_" + "_".join(f"{seconds:g}s" for seconds in windows)
        config["paths"]["output"] = str(Path(config["paths"]["output"]) / "runs" / label)

    output = Path(config["paths"]["output"])
    output.mkdir(parents=True, exist_ok=True)
    print(f"输出目录：{output}", flush=True)

    # 数据维度检查总是执行：它是所有实验的第一道防线。
    inventory = audit(config, subjects)
    inventory.to_csv(output / "data_inventory.csv", index=False)
    print(f"已检查 {len(inventory)} 个 MAT 文件", flush=True)
    write_manifest(config, subjects, windows, args.stage)

    with threadpool_limits(limits=1):      # 避免 BLAS 线程过多反而拖慢
        if args.stage in ("quality", "all"):
            run_quality(config, subjects)
        if args.stage in ("frequency", "all"):
            for subject in subjects:
                run_subject_stage(config, subject, windows, "frequency", args.skip_existing)
        if args.stage in ("async", "all"):
            for subject in subjects:
                run_subject_stage(config, subject, windows, "asynchronous", args.skip_existing)
        if args.stage == "report":
            summarize_all(config, subjects)
        if args.stage == "verify":
            verify_all(config, subjects)
        if args.stage == "all":
            summarize_all(config, subjects)

    print(f"完成。输出：{output}", flush=True)


if __name__ == "__main__":
    main()
