#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Collect per-case msprof op Task Duration for all isamin TC_PF cases."""

import argparse
import csv
import datetime
import json
import logging
import math
import os
import shutil
import statistics
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
EXPECTED_CASES = 200
WARMUP = 5
VALID_SAMPLES = 15
PERF_THRESHOLD = 0.8
LOGGER = logging.getLogger(__name__)


def load_cases():
    with (HERE / "isamin_test.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = [
            row
            for row in csv.DictReader(stream)
            if row["case_name"].startswith("TC_PF_")
        ]
    with (HERE / "gpu_baseline.csv").open(encoding="utf-8-sig", newline="") as stream:
        baseline = list(csv.DictReader(stream))
    if len(rows) != EXPECTED_CASES or len(baseline) != EXPECTED_CASES:
        raise RuntimeError("Expected 200 TC_PF cases and 200 GPU baseline rows")
    for case, reference in zip(rows, baseline):
        if int(case["n"]) != int(reference["n"]) or int(case["incx"]) != int(
            reference["incx"]
        ):
            raise RuntimeError("GPU baseline parameter mismatch: " + case["case_name"])
        if float(reference["gpu_ms"]) <= 0:
            raise RuntimeError("Invalid GPU baseline: " + case["case_name"])
    return rows, baseline


def resolve_executable(name):
    executable = shutil.which(name)
    if executable is None:
        raise RuntimeError(f"Executable not found: {name}")
    return Path(executable).resolve()


def write_environment(output, msprof_bin):
    commands = {
        "git_commit": ["git", "-C", str(REPO), "rev-parse", "HEAD"],
        "uname": ["uname", "-a"],
        "msprof_version": [str(msprof_bin), "-v"],
        "npu_smi": ["npu-smi", "info"],
    }
    lines = []
    for name, command in commands.items():
        try:
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=30,
                check=False,
            )
            lines.append(f"[{name}] rc={result.returncode}")
            lines.append(result.stdout.strip())
        except (OSError, subprocess.TimeoutExpired) as error:
            lines.append(f"[{name}] unavailable: {error}")
    lines.append(
        f"ASCEND_DEVICE_ID={os.environ.get('ASCEND_DEVICE_ID', '')} "
        f"BLAS_DEVICE_ID={os.environ.get('BLAS_DEVICE_ID', '')}"
    )
    (output / "environment.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    help_result = subprocess.run(
        [str(msprof_bin), "op", "--help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=30,
        check=False,
    )
    (output / "msprof_op_help.txt").write_text(help_result.stdout, encoding="utf-8")


def create_app(binary, output):
    app = output / "profile_app.sh"
    app.write_text(
        "#!/bin/bash\n"
        'exec "${ISAMIN_TEST_BINARY:-'
        + str(binary)
        + '}" --gtest_filter="$ISAMIN_GTEST_FILTER" --gtest_color=no\n',
        encoding="utf-8",
    )
    app.chmod(0o700)
    return app


def read_raw_samples(raw_dir):
    entries = []
    for path in raw_dir.rglob("OpBasicInfo_*.csv"):
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 1:
            raise RuntimeError(f"Expected one task per OpBasicInfo CSV: {path}")
        row = rows[0]
        if row["Op Name"] != "isamin_kernel":
            raise RuntimeError(f"Unexpected kernel {row['Op Name']}: {path}")
        duration = float(row["Task Duration(us)"])
        if not math.isfinite(duration) or duration <= 0:
            raise RuntimeError(f"Invalid Task Duration(us): {path}")
        launch_id = (
            int(path.parent.name) if path.parent.name.isdigit() else len(entries)
        )
        entries.append((launch_id, duration, path))
    entries.sort(key=lambda item: item[0])
    if len(entries) != VALID_SAMPLES:
        raise RuntimeError(f"Expected {VALID_SAMPLES} raw samples, got {len(entries)}")
    return entries


def build_command(msprof_bin, app, raw_dir):
    return [
        str(msprof_bin),
        "op",
        "--application=" + str(app),
        "--output=" + str(raw_dir),
        "--aic-metrics=BasicInfo",
        "--kernel-name=isamin_kernel",
        "--warm-up=" + str(WARMUP),
        "--launch-count=" + str(VALID_SAMPLES),
        "--replay-mode=kernel",
    ]


def run_case_command(args, command, case_name, case_dir):
    environment = dict(os.environ)
    environment["ISAMIN_PROFILE"] = "1"
    environment["ISAMIN_GTEST_FILTER"] = "Isamin/IsaminTest.CsvDriven/" + case_name
    environment["ISAMIN_TEST_BINARY"] = str(args.binary)
    for name in ("GTEST_FILTER", "GTEST_OUTPUT", "BLAS_TEST_CASES"):
        environment.pop(name, None)
    with (case_dir / "msprof.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
            timeout=args.case_timeout,
            check=False,
        )
    text = (case_dir / "msprof.log").read_text(encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(
            f"msprof failed for {case_name}; inspect {case_dir / 'msprof.log'}"
        )
    if "[  PASSED  ] 1 test." not in text:
        raise RuntimeError(f"GTest did not pass for {case_name}")
    if "[ERROR]" in text:
        raise RuntimeError(f"msprof reported ERROR for {case_name}")


def preserve_samples(case_name, entries, case_dir, output):
    preserved = case_dir / "OpBasicInfo"
    preserved.mkdir()
    samples = []
    for sample, (_, duration, path) in enumerate(entries):
        target = preserved / path.name
        shutil.copy2(path, target)
        samples.append(
            {
                "case_name": case_name,
                "sample": sample,
                "duration_us": duration,
                "raw_csv": str(target.relative_to(output)),
            }
        )
    return preserved, samples


def build_case_result(row, reference, values, case_dir, preserved, output):
    case_name = row["case_name"]
    mean_us = statistics.mean(values)
    gpu_us = float(reference["gpu_ms"]) * 1000.0
    limit_us = gpu_us / PERF_THRESHOLD
    ratio = gpu_us / mean_us
    return {
        "case_name": case_name,
        "n": int(row["n"]),
        "incx": int(row["incx"]),
        "baseline_id": reference["id"],
        "gpu_ms": float(reference["gpu_ms"]),
        "gpu_us": gpu_us,
        "limit_us": limit_us,
        "warmup": WARMUP,
        "samples": len(values),
        "mean_us": mean_us,
        "min_us": min(values),
        "max_us": max(values),
        "gpu_npu_ratio": ratio,
        "limit_utilization": mean_us / limit_us,
        "verdict": "PASS" if ratio >= PERF_THRESHOLD else "FAIL",
        "raw_dir": str(preserved.relative_to(output)),
        "log": str((case_dir / "msprof.log").relative_to(output)),
    }


def collect_case(args, msprof_bin, app, row, reference, index, output):
    case_name = row["case_name"]
    case_dir = output / "cases" / f"{index:04d}_{case_name}"
    case_dir.mkdir(parents=True)
    raw_dir = case_dir / "raw"
    command = build_command(msprof_bin, app, raw_dir)
    (case_dir / "command.json").write_text(
        json.dumps(command, indent=2) + "\n", encoding="utf-8"
    )
    run_case_command(args, command, case_name, case_dir)

    entries = read_raw_samples(raw_dir)
    preserved, samples = preserve_samples(case_name, entries, case_dir, output)
    shutil.rmtree(raw_dir)

    values = [sample["duration_us"] for sample in samples]
    result = build_case_result(row, reference, values, case_dir, preserved, output)
    return result, samples, command


def save_results(
    output, results, samples, commands, selected, start_index, selection_mode
):
    expected_samples = selected * VALID_SAMPLES
    if len(results) != selected or len(samples) != expected_samples:
        raise RuntimeError(
            f"Expected {selected} cases and {expected_samples} samples, got {len(results)} and {len(samples)}"
        )
    with (output / "performance_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    with (output / "performance_samples.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(samples[0]))
        writer.writeheader()
        writer.writerows(samples)
    (output / "commands.txt").write_text(
        "\n".join(" ".join(command) for command in commands) + "\n", encoding="utf-8"
    )

    passed = sum(result["verdict"] == "PASS" for result in results)
    failed = len(results) - passed
    summary = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "formal": selected == EXPECTED_CASES,
        "start_index": start_index,
        "selection_mode": selection_mode,
        "requested_cases": selected,
        "total_cases": len(results),
        "passed": passed,
        "failed": failed,
        "warmup": WARMUP,
        "valid_samples_per_case": VALID_SAMPLES,
        "total_samples": len(samples),
        "threshold": PERF_THRESHOLD,
        "metric": "msprof op BasicInfo Task Duration(us), arithmetic mean",
        "replay_mode": "kernel",
        "minimum_gpu_npu_ratio": min(result["gpu_npu_ratio"] for result in results),
        "maximum_limit_utilization": max(
            result["limit_utilization"] for result in results
        ),
    }
    (output / "performance_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary",
        type=Path,
        default=REPO / "build/test/isamin/isamin_test",
        help="isamin GTest binary",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New evidence directory"
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=EXPECTED_CASES,
        help="Smoke-test selection count",
    )
    parser.add_argument(
        "--start-index", type=int, default=0, help="Zero-based TC_PF selection start"
    )
    parser.add_argument(
        "--case-file", type=Path, help="Optional file containing exact case names"
    )
    parser.add_argument("--case-timeout", type=int, default=900)
    parser.add_argument("--msprof", type=Path, help="msprof executable")
    return parser.parse_args()


def select_cases(args, rows, baseline):
    row_index = {row["case_name"]: index for index, row in enumerate(rows)}
    if args.case_file:
        names = [
            line.strip()
            for line in args.case_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(names) != len(set(names)) or any(
            name not in row_index for name in names
        ):
            raise RuntimeError("Invalid or duplicate case in --case-file")
        selection = [
            (row_index[name], rows[row_index[name]], baseline[row_index[name]])
            for name in names
        ]
        return selection, min(item[0] for item in selection), "case_file"

    selected = args.max_cases
    selection = [
        (index, rows[index], baseline[index])
        for index in range(args.start_index, args.start_index + selected)
    ]
    return selection, args.start_index, "range"


def execute_cases(args, msprof_bin, app, selection):
    results, samples, commands = [], [], []
    for index, row, reference in selection:
        try:
            result, case_samples, command = collect_case(
                args, msprof_bin, app, row, reference, index, args.output
            )
            results.append(result)
            samples.extend(case_samples)
            commands.append(command)
            LOGGER.info(
                "[%s/%s] %s mean_us=%.3f %s",
                index + 1,
                EXPECTED_CASES,
                row["case_name"],
                result["mean_us"],
                result["verdict"],
            )
        except Exception as error:
            (args.output / "failure.txt").write_text(
                str(error) + "\n", encoding="utf-8"
            )
            raise
    return results, samples, commands


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    args.binary = args.binary.resolve()
    args.output = args.output.resolve()
    if not args.binary.is_file():
        raise RuntimeError("Test binary not found: " + str(args.binary))
    if args.output.exists():
        raise RuntimeError("Output directory already exists: " + str(args.output))
    if (
        args.start_index < 0
        or args.max_cases <= 0
        or args.start_index + args.max_cases > EXPECTED_CASES
    ):
        raise RuntimeError("Selection must stay inside the fixed 200-case TC_PF list")
    rows, baseline = load_cases()
    selection, start_index, selection_mode = select_cases(args, rows, baseline)
    selected = len(selection)
    msprof_bin = args.msprof.resolve() if args.msprof else resolve_executable("msprof")
    args.output.mkdir(parents=True)
    write_environment(args.output, msprof_bin)
    app = create_app(args.binary, args.output)

    results, samples, commands = execute_cases(args, msprof_bin, app, selection)

    summary = save_results(
        args.output, results, samples, commands, selected, start_index, selection_mode
    )
    LOGGER.info("%s", json.dumps(summary, indent=2))
    if summary["failed"] != 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
