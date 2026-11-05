#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""ssbmv arch22 自测门禁：用例数断言 + 精度全量执行。

1) ssbmv_test.csv 必须恰好 1200 条：1000 条精度 + 200 条 TC_PF 性能用例；
2) 跑 -TC_PF 的精度套件，要求 0 失败；
3) 本次运行的用例数必须等于 gtest 列出的总数减去 200 条性能用例（手写 TEST_F 计入总数）。

用法：
    python3 test/sbmv/ssbmv/arch22/run_self_test.py
    python3 .../run_self_test.py --binary build/test/sbmv/ssbmv/ssbmv_test --device 1
"""

import argparse
import csv
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
CSV_PATH = HERE / "ssbmv_test.csv"
DEFAULT_BINARY = REPO / "build" / "test" / "sbmv" / "ssbmv" / "ssbmv_test"
EXPECTED_TOTAL = 1200
EXPECTED_ACCURACY = 1000
EXPECTED_PERF = 200

LOGGER = logging.getLogger("ssbmv-self-test")


def csv_case_counts(path):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    total = len(rows)
    perf = sum(1 for row in rows if row["case_name"].startswith("TC_PF_"))
    return total, total - perf, perf


def listed_test_count(binary):
    result = subprocess.run(
        [str(binary), "--gtest_list_tests"], capture_output=True, text=True, check=True
    )
    # Suite lines are "Suite." and case lines are indented; only case lines start with two spaces.
    return sum(1 for line in result.stdout.splitlines() if line.startswith("  ") and line.strip())


def run_accuracy(binary, device):
    env = os.environ.copy()
    if device is not None:
        env["ASCEND_DEVICE_ID"] = str(device)
    result = subprocess.run(
        [str(binary), "--gtest_filter=-*TC_PF*"],
        capture_output=True,
        text=True,
        env=env,
        timeout=1800,
    )
    output = result.stdout + result.stderr
    passed = 0
    failed = 0
    match = re.search(r"\[\s+PASSED\s+\]\s+(\d+)\s+tests?\.", output)
    if match:
        passed = int(match.group(1))
    match = re.search(r"\[\s+FAILED\s+\]\s+(\d+)\s+tests?, listed below", output)
    if match:
        failed = int(match.group(1))
    return result.returncode, passed, failed, output


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    parser = argparse.ArgumentParser(description="ssbmv arch22 self-test gate")
    parser.add_argument("--binary", default=str(DEFAULT_BINARY))
    parser.add_argument("--device", type=int, default=None)
    args = parser.parse_args()

    binary = Path(args.binary)
    if not binary.is_file():
        LOGGER.error("[self-test] binary not found: %s", binary)
        LOGGER.error(
            "build it first: cmake -S . -B build -DBUILD_TEST=ON -DTEST_NAMES=ssbmv && cmake --build build -j32"
        )
        return 1

    total, accuracy, perf = csv_case_counts(CSV_PATH)
    LOGGER.info("[self-test] csv=%s", CSV_PATH)
    LOGGER.info("[self-test] cases: total=%d accuracy=%d perf(TC_PF)=%d", total, accuracy, perf)
    ok = True
    if (total, accuracy, perf) != (EXPECTED_TOTAL, EXPECTED_ACCURACY, EXPECTED_PERF):
        LOGGER.error("[self-test] FAIL: expected %d/%d/%d", EXPECTED_TOTAL, EXPECTED_ACCURACY, EXPECTED_PERF)
        ok = False

    listed = listed_test_count(binary)
    expected_run = listed - perf
    LOGGER.info("[self-test] gtest lists %d tests; accuracy run is expected to execute %d", listed, expected_run)

    returncode, passed, failed, output = run_accuracy(binary, args.device)
    LOGGER.info("[self-test] accuracy suite: passed=%d failed=%d rc=%d", passed, failed, returncode)
    if returncode != 0 or failed != 0:
        LOGGER.error("[self-test] FAIL: accuracy suite reported failures")
        for line in output.splitlines():
            if line.startswith("[  FAILED  ]"):
                LOGGER.error("  %s", line)
        ok = False
    if passed != expected_run:
        LOGGER.error(
            "[self-test] FAIL: executed %d tests, expected %d (csv count and gtest registration disagree)",
            passed,
            expected_run,
        )
        ok = False

    if ok:
        LOGGER.info("[self-test] PASS")
        return 0
    LOGGER.error("[self-test] FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
