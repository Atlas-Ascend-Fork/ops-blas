# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Compare strict NPU event timings with the supplied GPU baseline."""
import csv
import logging
import math
import re
import sys
from pathlib import Path

PATTERN = re.compile(
    r"\[STRICT_PERF\] case=(\S+) n=(\d+) "
    r"incx=(-?\d+).*?samples=(\d+) average_us=([0-9.eE+-]+)"
)
PASS_RATIO_THRESHOLD = 0.4


def parse_timings(log_path):
    timed = {}
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = PATTERN.search(line)
        if not match:
            continue
        case_name = match.group(1)
        if case_name in timed:
            raise ValueError(f"Duplicate timing record for {case_name}")
        npu_us = float(match.group(5))
        if not math.isfinite(npu_us) or npu_us <= 0.0:
            raise ValueError(f"Invalid average_us for {case_name}: {npu_us}")
        timed[case_name] = (
            int(match.group(2)),
            int(match.group(3)),
            int(match.group(4)),
            npu_us,
        )
    return timed


def compare_timings(log_path, baseline_path, out_path):
    timed = parse_timings(log_path)
    rows = []
    with baseline_path.open(encoding="utf-8", newline="") as src:
        for index, baseline in enumerate(csv.DictReader(src), 1001):
            name = f"TC_PF_{index}"
            if name not in timed:
                raise ValueError(f"Missing timing record for {name}")
            n, incx, samples, npu = timed[name]
            gpu_ms = float(baseline["gpu_ms"])
            if not math.isfinite(gpu_ms) or gpu_ms <= 0.0:
                raise ValueError(f"Invalid gpu_ms for {name}: {gpu_ms}")
            gpu_us = gpu_ms * 1000.0
            ratio = gpu_us / npu
            rows.append(
                dict(
                    case_name=name,
                    n=n,
                    incx=incx,
                    gpu_baseline_us=f"{gpu_us:.6f}",
                    npu_avg_us=f"{npu:.6f}",
                    samples=samples,
                    ratio=f"{ratio:.6f}",
                    verdict="PASS" if ratio >= PASS_RATIO_THRESHOLD else "FAIL",
                )
            )
    if not rows:
        raise ValueError("Baseline CSV contains no rows")
    with out_path.open("w", encoding="utf-8", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if len(sys.argv) != 4:
        logging.error("Usage: strict_compare.py <log> <baseline.csv> <out.csv>")
        return 1
    log_path, baseline_path, out_path = map(Path, sys.argv[1:])
    try:
        rows = compare_timings(log_path, baseline_path, out_path)
    except (OSError, ValueError, KeyError) as exc:
        logging.error("Comparison failed: %s", exc)
        return 1
    passed = sum(row["verdict"] == "PASS" for row in rows)
    failed = len(rows) - passed
    logging.info("TOTAL=%d PASS=%d FAIL=%d", len(rows), passed, failed)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
