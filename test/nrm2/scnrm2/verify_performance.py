#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Verify scnrm2 strict event timings against the supplied GPU baseline."""
import argparse
import csv
import datetime
import logging
import math
import os
from pathlib import Path
import re
import subprocess
import sys

LOGGER = logging.getLogger(__name__)


class OutputFormatter(logging.Formatter):
    """Preserve captured subprocess output, including its trailing newline."""

    def format(self, record):
        return super().format(record) + getattr(record, "terminator", "\n")


def configure_logging():
    handler = logging.StreamHandler(sys.stdout)
    handler.terminator = ""
    handler.setFormatter(OutputFormatter("%(message)s"))
    LOGGER.handlers.clear()
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False

OP = "scnrm2"
FAMILY = "nrm2"
PERF_THRESHOLD = 0.4
STRICT_WARMUP = 20
STRICT_SAMPLES = 100
EXPECTED_CASES = 200
ARCH_BY_SOC = {"ascend950": "arch35", "ascend910b3": "arch22", "ascend910b4": "arch22"}
STRICT_RE = re.compile(
    r"\[STRICT_PERF\]\s+case=(\S+)\s+n=(\d+)\s+incx=(-?\d+)\s+"
    r"warmup=(\d+)\s+samples=(\d+)\s+average_us=([0-9.eE+-]+)"
)


def find_binary(repo):
    candidates = [
        repo / "build" / "test" / FAMILY / OP / f"{OP}_test",
        repo / "build" / "test" / OP / f"{OP}_test",
    ]
    return next((path for path in candidates if path.is_file() and os.access(path, os.X_OK)), None)


def parse_perf_params(repo, soc):
    arch = ARCH_BY_SOC.get(soc, "arch35")
    path = repo / "test" / FAMILY / OP / arch / f"{OP}_test.csv"
    with path.open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["case_name"].startswith("TC_PF_")]
    if len(rows) != EXPECTED_CASES:
        raise ValueError(f"expected {EXPECTED_CASES} TC_PF cases in {path}, got {len(rows)}")
    return rows


def load_baseline(path):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != EXPECTED_CASES:
        raise ValueError(f"expected {EXPECTED_CASES} baseline rows in {path}, got {len(rows)}")
    empty = [index for index, row in enumerate(rows) if not (row.get("gpu_ms") or "").strip()]
    if empty:
        raise ValueError(f"baseline {path} has empty gpu_ms at row indexes {empty[:5]}")
    invalid = []
    for index, row in enumerate(rows):
        try:
            gpu_us = float(row["gpu_ms"]) * 1000.0
        except ValueError:
            invalid.append(index)
            continue
        if not math.isfinite(gpu_us) or gpu_us <= 0:
            invalid.append(index)
    if invalid:
        raise ValueError(
            f"baseline {path} requires finite positive gpu_ms convertible to microseconds "
            f"at row indexes {invalid[:5]}"
        )
    return rows


def build(repo, soc):
    cmd = ["bash", "build.sh", f"--soc={soc}", f"--ops={OP}"]
    LOGGER.info("build: %s", " ".join(cmd))
    return subprocess.run(cmd, cwd=repo, timeout=600).returncode == 0


def run_gtest(binary, device, timeout, strict):
    env = os.environ.copy()
    if device is not None:
        env["ASCEND_DEVICE_ID"] = str(device)
    if strict:
        env["SCNRM2_STRICT_PERF"] = "1"
    cmd = [str(binary), "--gtest_filter=*TC_PF*", "--gtest_color=no", "--gtest_brief=1"]
    LOGGER.info("run: %s", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd, env=env, timeout=timeout, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        LOGGER.info(output, extra={"terminator": ""})
        LOGGER.error("ERROR: timed out after %ss", timeout)
        return 1, {}
    LOGGER.info(result.stdout, extra={"terminator": ""})
    timings = {}
    duplicate = []
    for line in result.stdout.splitlines():
        match = STRICT_RE.search(line)
        if not match:
            continue
        name = match.group(1)
        if name in timings:
            duplicate.append(name)
        timings[name] = {
            "n": int(match.group(2)),
            "incx": int(match.group(3)),
            "warmup": int(match.group(4)),
            "samples": int(match.group(5)),
            "average_us": float(match.group(6)),
        }
    if duplicate:
        LOGGER.error("ERROR: duplicate STRICT_PERF results: %s", sorted(set(duplicate))[:5])
        # 重复的 STRICT_PERF 记录意味着计时门禁结果不可信，必须判失败，
        # 而不是仅打印错误后仍按底层进程退出码放行（issue #534）
        return result.returncode or 1, timings
    return result.returncode, timings


def parse_args():
    parser = argparse.ArgumentParser(description="scnrm2 strict performance verifier")
    parser.add_argument("--repo", required=True, type=Path, help="ops-blas repository")
    parser.add_argument("--soc", required=True, help="for example ascend950")
    parser.add_argument("--baseline", required=True, type=Path, help="authoritative gpu_baseline.csv")
    parser.add_argument("--output", type=Path, help="comparison CSV (default: test directory)")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--device", type=int)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument(
        "--strict", action="store_true",
        default=os.environ.get("SCNRM2_STRICT_PERF") == "1",
        help="set SCNRM2_STRICT_PERF=1 and verify warmup/sample/event-us fields",
    )
    return parser, parser.parse_args()


def validate_paths(parser, args):

    repo = args.repo.resolve()
    baseline_path = args.baseline.resolve()
    if not repo.is_dir():
        parser.error(f"repository does not exist: {repo}")
    if not baseline_path.is_file():
        parser.error(f"baseline does not exist: {baseline_path}")
    return repo, baseline_path


def collect_rows(perf_cases, baseline, timings, strict):
    rows = []
    for index, (case, base) in enumerate(zip(perf_cases, baseline), 1001):
        name = case["case_name"]
        timing = timings.get(name)
        expected = f"TC_PF_{index}"
        if name != expected:
            failure = f"expected case order {expected}"
            timing = None
        elif timing is None:
            failure = "missing STRICT_PERF result"
        else:
            failure = check_timing(case, timing, base, strict)
        rows.append(make_row(case, timing, base, failure))
    return rows


def check_timing(case, timing, base, strict):
    failures = []
    if timing["n"] != int(case["n"]) or timing["incx"] != int(case["incx"]):
        failures.append("n/incx mismatch")
    if strict and timing["warmup"] != STRICT_WARMUP:
        failures.append(f"warmup={timing['warmup']}")
    if strict and timing["samples"] != STRICT_SAMPLES:
        failures.append(f"samples={timing['samples']}")
    if timing["average_us"] <= 0:
        failures.append("non-positive average_us")
    ratio = float(base["gpu_ms"]) * 1000.0 / timing["average_us"] if timing["average_us"] > 0 else 0.0
    if ratio < PERF_THRESHOLD:
        failures.append(f"ratio<{PERF_THRESHOLD}")
    return ";".join(failures)


def make_row(case, timing, base, failure):
    name = case["case_name"]
    if timing is None:
        return {"case_name": name, "n": case.get("n", ""), "incx": case.get("incx", ""),
                "warmup": "", "samples": "", "gpu_baseline_us": "", "npu_avg_us": "",
                "ratio": "", "verdict": "FAIL", "failure": failure}
    gpu_us = float(base["gpu_ms"]) * 1000.0
    average = timing["average_us"]
    ratio = gpu_us / average if average > 0 else 0.0
    return {"case_name": name, "n": timing["n"], "incx": timing["incx"],
            "warmup": timing["warmup"], "samples": timing["samples"],
            "gpu_baseline_us": f"{gpu_us:.6f}", "npu_avg_us": f"{average:.6f}",
            "ratio": f"{ratio:.6f}", "verdict": "FAIL" if failure else "PASS", "failure": failure}


def write_rows(rows, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def report(rows, output, returncode, unexpected):
    passed = sum(row["verdict"] == "PASS" for row in rows)
    ratios = [float(row["ratio"]) for row in rows if row["verdict"] == "PASS"]
    min_ratio = min(ratios) if ratios else 0.0
    min_case = next((row["case_name"] for row in rows
                     if row["verdict"] == "PASS" and float(row["ratio"]) == min_ratio), "N/A")
    LOGGER.info("output: %s", output)
    LOGGER.info("total=%s PASS=%s FAIL=%s", len(rows), passed, len(rows) - passed)
    LOGGER.info("min_ratio=%.6f min_case=%s threshold=%s", min_ratio, min_case, PERF_THRESHOLD)
    if returncode != 0:
        LOGGER.error("ERROR: gtest exited with %s", returncode)
    success = returncode == 0 and len(rows) == EXPECTED_CASES and passed == EXPECTED_CASES and not unexpected
    LOGGER.info("result: %s", "PASS" if success else "FAIL")
    return 0 if success else 1


def main():
    configure_logging()
    parser, args = parse_args()
    repo, baseline_path = validate_paths(parser, args)

    if not args.skip_build and not build(repo, args.soc):
        LOGGER.error("ERROR: build failed")
        return 1
    LOGGER.info("skip-build: true" if args.skip_build else "build: PASS")

    binary = find_binary(repo)
    if binary is None:
        LOGGER.error("ERROR: scnrm2_test binary not found")
        return 1
    LOGGER.info("binary: %s", binary)

    try:
        perf_cases = parse_perf_params(repo, args.soc)
        baseline = load_baseline(baseline_path)
    except (OSError, ValueError) as exc:
        LOGGER.error("ERROR: %s", exc)
        return 1

    returncode, timings = run_gtest(binary, args.device, args.timeout, args.strict)
    rows = collect_rows(perf_cases, baseline, timings, args.strict)

    unexpected = sorted(set(timings) - {case["case_name"] for case in perf_cases})
    if unexpected:
        LOGGER.error("ERROR: unexpected STRICT_PERF cases: %s", unexpected[:5])

    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    output = args.output or (Path(__file__).resolve().parent / f"results_npu_{OP}_{timestamp}.csv")
    write_rows(rows, output)
    return report(rows, output, returncode, unexpected)


if __name__ == "__main__":
    sys.exit(main())
