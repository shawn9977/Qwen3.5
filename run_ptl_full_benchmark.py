#!/usr/bin/env python3
"""Run Qwen3.5 INT4/INT8 text+multimodal benchmarks and save logs.

This script reproduces the benchmark matrix from README section 7:
- INT4 + Text-only:   (1024, 64, warmup/iters=3/5), (1024, 512, 3/10)
- INT4 + Multimodal:  (prompt~1408, 64, 3/5),      (prompt~1408, 512, 3/10)
- INT8 + Text-only:   same as above
- INT8 + Multimodal:  same as above

It executes the existing benchmark scripts in this folder:
- benchmark_qwen3_5_openvino.py
- benchmark_qwen3_5_mm_realtext.py

And writes:
- raw logs per case
- summary CSV
- summary Markdown table
- summary JSON
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class CaseConfig:
    precision: str
    mode: str  # "text" or "multimodal"
    seq_or_prompt_tokens: int
    new_tokens: int
    warmup: int
    iters: int


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    parser = argparse.ArgumentParser(
        description="Run full PTL benchmark matrix for Qwen3.5 INT4/INT8 and save logs"
    )
    parser.add_argument(
        "--models-root",
        type=Path,
        default=project_root / "models",
        help="Directory containing Qwen3.5-35B-A3B-INT4 and Qwen3.5-35B-A3B-INT8",
    )
    parser.add_argument(
        "--qwen-dir",
        type=Path,
        default=script_dir,
        help="Directory containing benchmark_qwen3_5_openvino.py and benchmark_qwen3_5_mm_realtext.py",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=script_dir / "test_image.jpg",
        help="Image path used for multimodal benchmark",
    )
    parser.add_argument(
        "--python-bin",
        type=Path,
        default=Path(sys.executable),
        help="Python executable to run benchmark scripts",
    )
    parser.add_argument(
        "--device",
        default="GPU",
        help="OpenVINO device, e.g. GPU/CPU/NPU",
    )
    parser.add_argument(
        "--num-streams",
        default="1",
        help="OpenVINO NUM_STREAMS",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "benchmark_logs",
        help="Directory where logs and summary files are saved",
    )
    parser.add_argument(
        "--openvino-python-path",
        type=Path,
        default=script_dir / "openvino_35d22" / "bin" / "intel64" / "Release" / "python",
        help="Path appended to PYTHONPATH for custom OpenVINO Python package",
    )
    parser.add_argument(
        "--openvino-lib-path",
        type=Path,
        default=script_dir / "openvino_35d22" / "bin" / "intel64" / "Release",
        help="Path appended to LD_LIBRARY_PATH for custom OpenVINO runtime libs",
    )
    parser.add_argument(
        "--strict-device",
        action="store_true",
        help="Pass --strict-device to multimodal benchmark",
    )
    parser.add_argument(
        "--stop-on-fail",
        action="store_true",
        help="Stop immediately if any case fails",
    )
    return parser.parse_args()


def build_cases() -> list[CaseConfig]:
    cases: list[CaseConfig] = []
    for precision in ("INT4", "INT8"):
        cases.append(CaseConfig(precision, "text", 1024, 64, 3, 5))
        cases.append(CaseConfig(precision, "text", 1024, 512, 3, 10))
        # README reports prompt tokens around 1408 for input-text-tokens 1024.
        cases.append(CaseConfig(precision, "multimodal", 1408, 64, 3, 5))
        cases.append(CaseConfig(precision, "multimodal", 1408, 512, 3, 10))
    return cases


def make_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()

    if args.openvino_python_path.exists():
        old_py_path = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{args.openvino_python_path}:{old_py_path}" if old_py_path else str(args.openvino_python_path)
        )

    if args.openvino_lib_path.exists():
        old_ld_path = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = (
            f"{args.openvino_lib_path}:{old_ld_path}" if old_ld_path else str(args.openvino_lib_path)
        )

    return env


def extract_metric(pattern: str, text: str) -> Optional[float]:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    return float(match.group(1))


def extract_prompt_tokens(text: str) -> Optional[int]:
    match = re.search(r"Prompt\s+tokens:\s*(\d+)", text, flags=re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1))


def format_mode(mode: str) -> str:
    if mode == "text":
        return "Text-only (LLM)"
    return "Multimodal (image+text)"


def format_warmup_iters(warmup: int, iters: int) -> str:
    return f"{warmup}/{iters}"


def format_ttft(value: Optional[float]) -> str:
    return f"{value:.0f} ms" if value is not None else "N/A"


def format_tpot(value: Optional[float]) -> str:
    return f"{value:.1f} ms" if value is not None else "N/A"


def format_thr(value: Optional[float]) -> str:
    return f"{value:.2f} tok/s" if value is not None else "N/A"


def run_case(
    args: argparse.Namespace,
    env: dict[str, str],
    case: CaseConfig,
    case_index: int,
    total_cases: int,
    run_dir: Path,
) -> dict:
    model_dir = args.models_root / f"Qwen3.5-35B-A3B-{case.precision}"
    text_script = args.qwen_dir / "benchmark_qwen3_5_openvino.py"
    mm_script = args.qwen_dir / "benchmark_qwen3_5_mm_realtext.py"

    if not model_dir.exists():
        raise FileNotFoundError(f"Model dir not found: {model_dir}")
    if not text_script.exists():
        raise FileNotFoundError(f"Script not found: {text_script}")
    if not mm_script.exists():
        raise FileNotFoundError(f"Script not found: {mm_script}")

    case_name = (
        f"{case.precision}_{case.mode}_seq{case.seq_or_prompt_tokens}_new{case.new_tokens}_w{case.warmup}_i{case.iters}"
    )
    safe_case_name = case_name.replace("+", "plus")
    log_path = run_dir / f"{safe_case_name}.log"

    if case.mode == "text":
        cmd = [
            str(args.python_bin),
            str(text_script),
            "--model-xml",
            str(model_dir / "openvino_language_model.xml"),
            "--device",
            args.device,
            "--batch",
            "1",
            "--seq-len",
            "1024",
            "--decode-tokens",
            str(case.new_tokens),
            "--warmup",
            str(case.warmup),
            "--iters",
            str(case.iters),
            "--num-streams",
            str(args.num_streams),
        ]
    else:
        if not args.image.exists():
            raise FileNotFoundError(f"Image not found: {args.image}")
        cmd = [
            str(args.python_bin),
            str(mm_script),
            "--model-dir",
            str(model_dir),
            "--image",
            str(args.image),
            "--device",
            args.device,
            "--input-text-tokens",
            "1024",
            "--new-tokens",
            str(case.new_tokens),
            "--warmup",
            str(case.warmup),
            "--iters",
            str(case.iters),
            "--num-streams",
            str(args.num_streams),
        ]
        if args.strict_device:
            cmd.append("--strict-device")

    print(
        f"[{case_index}/{total_cases}] Running {case.precision} | {format_mode(case.mode)} | "
        f"tokens={case.seq_or_prompt_tokens} | new={case.new_tokens} | warmup/iters={case.warmup}/{case.iters}"
    )

    started = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(args.qwen_dir),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    elapsed_s = time.perf_counter() - started

    log_text = []
    log_text.append(f"# Case: {case_name}")
    log_text.append(f"# Command: {' '.join(cmd)}")
    log_text.append(f"# Exit code: {proc.returncode}")
    log_text.append(f"# Elapsed seconds: {elapsed_s:.3f}")
    log_text.append("# ---- output ----")
    log_text.append(proc.stdout)
    log_path.write_text("\n".join(log_text), encoding="utf-8")

    ttft = extract_metric(r"mean\s+TTFT:\s*([0-9]+(?:\.[0-9]+)?)\s*ms", proc.stdout)
    tpot = extract_metric(r"mean\s+TPOT:\s*([0-9]+(?:\.[0-9]+)?)\s*ms", proc.stdout)
    thr = extract_metric(r"Throughput:\s*([0-9]+(?:\.[0-9]+)?)\s*(?:tokens/s|tok/s)", proc.stdout)

    prompt_tokens_detected = extract_prompt_tokens(proc.stdout) if case.mode == "multimodal" else None

    seq_prompt = case.seq_or_prompt_tokens
    if case.mode == "multimodal" and prompt_tokens_detected is not None:
        seq_prompt = prompt_tokens_detected

    row = {
        "precision": case.precision,
        "mode": format_mode(case.mode),
        "seq_prompt_tokens": seq_prompt,
        "new_tokens": case.new_tokens,
        "warmup_iters": format_warmup_iters(case.warmup, case.iters),
        "ttft_ms": ttft,
        "tpot_ms": tpot,
        "throughput_tok_s": thr,
        "status": "OK" if proc.returncode == 0 else "FAIL",
        "exit_code": proc.returncode,
        "elapsed_s": elapsed_s,
        "log_file": log_path.name,
    }

    return row


def write_summary_csv(rows: list[dict], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "量化精度",
                "测试模式",
                "SeqLen / Prompt tokens",
                "New tokens",
                "Warmup/Iters",
                "TTFT (mean)",
                "TPOT (mean)",
                "Throughput",
                "Status",
                "ExitCode",
                "Elapsed(s)",
                "LogFile",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r["precision"],
                    r["mode"],
                    r["seq_prompt_tokens"],
                    r["new_tokens"],
                    r["warmup_iters"],
                    format_ttft(r["ttft_ms"]),
                    format_tpot(r["tpot_ms"]),
                    format_thr(r["throughput_tok_s"]),
                    r["status"],
                    r["exit_code"],
                    f"{r['elapsed_s']:.3f}",
                    r["log_file"],
                ]
            )


def write_summary_md(rows: list[dict], md_path: Path) -> None:
    md_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("| 量化精度 | 测试模式 | SeqLen / Prompt tokens | New tokens | Warmup/Iters | TTFT (mean) | TPOT (mean) | Throughput | Status |")
    lines.append("|---|---|---|---|---|---|---|---|---|")

    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(r["precision"]),
                    str(r["mode"]),
                    str(r["seq_prompt_tokens"]),
                    str(r["new_tokens"]),
                    str(r["warmup_iters"]),
                    format_ttft(r["ttft_ms"]),
                    format_tpot(r["tpot_ms"]),
                    format_thr(r["throughput_tok_s"]),
                    str(r["status"]),
                ]
            )
            + " |"
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_console_table(rows: list[dict]) -> None:
    headers = [
        "量化精度",
        "测试模式",
        "SeqLen/Prompt",
        "New",
        "Warmup/Iters",
        "TTFT",
        "TPOT",
        "Throughput",
        "Status",
    ]

    printable = []
    for r in rows:
        printable.append(
            [
                str(r["precision"]),
                str(r["mode"]),
                str(r["seq_prompt_tokens"]),
                str(r["new_tokens"]),
                str(r["warmup_iters"]),
                format_ttft(r["ttft_ms"]),
                format_tpot(r["tpot_ms"]),
                format_thr(r["throughput_tok_s"]),
                str(r["status"]),
            ]
        )

    widths = [len(h) for h in headers]
    for row in printable:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(values: list[str]) -> str:
        return " | ".join(values[i].ljust(widths[i]) for i in range(len(values)))

    sep = "-+-".join("-" * w for w in widths)

    print()
    print(fmt_row(headers))
    print(sep)
    for row in printable:
        print(fmt_row(row))
    print()


def main() -> None:
    args = parse_args()

    if not args.models_root.exists():
        raise FileNotFoundError(f"models-root does not exist: {args.models_root}")
    if not args.qwen_dir.exists():
        raise FileNotFoundError(f"qwen-dir does not exist: {args.qwen_dir}")
    if not args.python_bin.exists():
        raise FileNotFoundError(f"python-bin does not exist: {args.python_bin}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / f"ptl_full_benchmark_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(args)
    cases = build_cases()

    print("=== PTL Full Benchmark Runner ===")
    print(f"Models root:    {args.models_root}")
    print(f"Qwen dir:       {args.qwen_dir}")
    print(f"Image:          {args.image}")
    print(f"Python bin:     {args.python_bin}")
    print(f"Device:         {args.device}")
    print(f"NUM_STREAMS:    {args.num_streams}")
    print(f"Output run dir: {run_dir}")
    print(f"Cases:          {len(cases)}")
    print()

    rows: list[dict] = []
    failed = 0

    for idx, case in enumerate(cases, start=1):
        try:
            row = run_case(args, env, case, idx, len(cases), run_dir)
            rows.append(row)
            if row["status"] != "OK":
                failed += 1
                print(
                    f"  -> FAIL (exit={row['exit_code']}) | log={run_dir / row['log_file']}"
                )
                if args.stop_on_fail:
                    break
            else:
                print(
                    f"  -> OK | TTFT={format_ttft(row['ttft_ms'])}, TPOT={format_tpot(row['tpot_ms'])}, "
                    f"Throughput={format_thr(row['throughput_tok_s'])}"
                )
        except Exception as exc:
            failed += 1
            err_name = (
                f"{case.precision}_{case.mode}_seq{case.seq_or_prompt_tokens}_new{case.new_tokens}_"
                f"w{case.warmup}_i{case.iters}_exception.log"
            )
            err_log = run_dir / err_name
            err_log.write_text(f"Exception: {exc}\n", encoding="utf-8")

            rows.append(
                {
                    "precision": case.precision,
                    "mode": format_mode(case.mode),
                    "seq_prompt_tokens": case.seq_or_prompt_tokens,
                    "new_tokens": case.new_tokens,
                    "warmup_iters": format_warmup_iters(case.warmup, case.iters),
                    "ttft_ms": None,
                    "tpot_ms": None,
                    "throughput_tok_s": None,
                    "status": "FAIL",
                    "exit_code": -1,
                    "elapsed_s": 0.0,
                    "log_file": err_log.name,
                }
            )
            print(f"  -> FAIL (exception: {exc}) | log={err_log}")
            if args.stop_on_fail:
                break

    csv_path = run_dir / "summary.csv"
    md_path = run_dir / "summary.md"
    json_path = run_dir / "summary.json"

    write_summary_csv(rows, csv_path)
    write_summary_md(rows, md_path)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print_console_table(rows)

    print("=== Benchmark Completed ===")
    print(f"Total cases:   {len(rows)}")
    print(f"Failed cases:  {failed}")
    print(f"Summary CSV:   {csv_path}")
    print(f"Summary MD:    {md_path}")
    print(f"Summary JSON:  {json_path}")
    print(f"Raw logs dir:  {run_dir}")

    # Non-zero exit code if any case failed.
    if failed > 0:
        sys.exit(2)


if __name__ == "__main__":
    main()
