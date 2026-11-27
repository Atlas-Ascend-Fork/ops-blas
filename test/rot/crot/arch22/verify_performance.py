#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Collect arch22 Crot task durations with msprof op; never builds the operator."""

import argparse
import csv
import logging
import math
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys


HERE = Path(__file__).resolve().parent
LOGGER = logging.getLogger(__name__)


def load_baselines():
    with (HERE / "crot_test.csv").open(encoding="utf-8", newline="") as source:
        cases = {
            row["case_name"]: row
            for row in csv.DictReader(source)
            if row["case_name"].startswith("TC_PF_")
        }
    with (HERE / "gpu_baseline.csv").open(encoding="utf-8", newline="") as source:
        baselines = list(csv.DictReader(source))
    names = [row["case_name"] for row in baselines]
    if len(names) != 200 or len(set(names)) != 200 or set(names) != set(cases):
        raise ValueError("Expected one distinct baseline for each of the 200 performance cases")
    for row in baselines:
        case = cases[row["case_name"]]
        if any(int(row[key]) != int(case[key]) for key in ("n", "incx", "incy")):
            raise ValueError(f"Baseline shape mismatch: {row['case_name']}")
        if not math.isfinite(float(row["gpu_ms"])) or float(row["gpu_ms"]) <= 0:
            raise ValueError(f"Invalid GPU duration: {row['case_name']}")
    return baselines


def read_profile_file(path):
    samples = []
    with path.open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            if "crot_kernel" not in row["Op Name"]:
                raise ValueError(f"Unexpected kernel in {path}: {row['Op Name']}")
            value = float(row["Task Duration(us)"])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid task duration in {path}: {value}")
            samples.append(value)
    return samples


def read_durations(directory):
    samples = []
    for path in sorted(directory.rglob("OpBasicInfo*.csv")):
        samples.extend(read_profile_file(path))
    # This harness launches exactly ten calls. Do not silently accept a partial
    # capture, aggregate row, duplicated report, or a previous run's samples.
    if len(samples) != 10:
        raise ValueError(f"Expected 10 task durations in {directory}, found {len(samples)}")
    return samples


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New directory for raw profiles and results")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--case", help="One exact TC_PF case name; default: all 200")
    parser.add_argument("--fail-fast", action="store_true", help="Stop at the first case over its performance limit")
    parser.add_argument("--timeout", type=int, default=600, help="Seconds per case")
    args = parser.parse_args()
    baselines = load_baselines()
    if args.case:
        baselines = [row for row in baselines if row["case_name"] == args.case]
        if not baselines:
            parser.error("Unknown performance case")
    return args, baselines


def profile_case(binary, output, baseline, env, timeout):
    name = baseline["case_name"]
    destination = output / name
    application = shlex.join([str(binary), f"--gtest_filter=CrotA2/CrotTest.CsvDriven/{name}"])
    command = [
        "msprof", "op", "--warm-up=5", "--launch-count=10", "--aic-metrics=BasicInfo",
        "--kernel-name=*crot_kernel*",
        f"--output={destination}", f"--application={application}",
    ]
    log_path = output / f"{name}.log"
    with log_path.open("w", encoding="utf-8") as log:
        log.write(shlex.join(command) + "\n")
        log.flush()
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                timeout=timeout, check=False)
    log_text = log_path.read_text(encoding="utf-8")
    marker = f"[PROFILE] case={name} successful_samples=10"
    if result.returncode != 0 or marker not in log_text:
        raise RuntimeError(f"{name}: profiling/application failed; inspect {name}.log")
    durations = read_durations(destination)
    average = statistics.mean(durations)
    gpu_us = float(baseline["gpu_ms"]) * 1000.0
    limit = gpu_us * 1.25
    return {
        "case_name": name, "n": baseline["n"], "incx": baseline["incx"], "incy": baseline["incy"],
        "samples": len(durations), "avg_us": average, "gpu_us": gpu_us, "limit_us": limit,
        "status": "PASS" if average <= limit else "FAIL",
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    args, baselines = parse_arguments()
    binary = args.binary.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, CROT_PROFILE="1", BLAS_DEVICE_ID=str(args.device))
    failures = 0
    with (output / "results.csv").open("w", encoding="utf-8", newline="") as report:
        writer = csv.DictWriter(report, fieldnames=[
            "case_name", "n", "incx", "incy", "samples", "avg_us", "gpu_us", "limit_us", "status"])
        writer.writeheader()
        for baseline in baselines:
            result = profile_case(binary, output, baseline, env, args.timeout)
            passed = result["status"] == "PASS"
            failures += not passed
            writer.writerow(result)
            report.flush()
            LOGGER.info("%s: %s avg_us=%.6f limit_us=%.6f",
                        result["case_name"], result["status"], result["avg_us"], result["limit_us"])
            if args.fail_fast and not passed:
                return 1
    return int(failures != 0)


if __name__ == "__main__":
    raise SystemExit(main())
