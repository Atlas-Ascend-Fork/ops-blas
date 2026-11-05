#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""aclblasCher 用例数与结果自检：确认 CSV 全部用例被执行且无静默丢行。

用法：
  python3 run_self_test.py --binary ./build/test/syr/cher/cher_test
退出码：0 全绿且条数一致；1 有用例失败/条数不符；2 环境或二进制缺失。
"""
import argparse
import csv
import json
import logging
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXPECTED_CASES = 1200
EXPECTED_PERF = 200

logging.basicConfig(level=logging.INFO, format="%(message)s")
LOG = logging.getLogger("cher_self_test")


def audit_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    names = [row.get("case_name", "") for row in rows]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("CSV has empty or duplicate case_name")
    perf = sum(1 for name in names if name.startswith("TC_PF_"))
    if len(rows) != EXPECTED_CASES or perf != EXPECTED_PERF:
        raise ValueError(
            f"CSV count mismatch: total={len(rows)} expected={EXPECTED_CASES}, "
            f"TC_PF_={perf} expected={EXPECTED_PERF}"
        )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default=os.path.join(HERE, "../../../build/test/syr/cher/cher_test"))
    ap.add_argument("--csv", default=os.path.join(HERE, "arch35/cher_test.csv"))
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument("--output", default=HERE, help="Directory for self_test_summary.json")
    args = ap.parse_args()

    binary = os.path.abspath(args.binary)
    if not os.path.isfile(binary):
        LOG.error("SKIP: cher_test binary not found: %s", binary)
        return 2

    try:
        rows = audit_csv(os.path.abspath(args.csv))
    except ValueError as err:
        LOG.error("%s", err)
        return 1
    LOG.info("CSV cases: %s (acc=%s pf=%s)", len(rows), len(rows) - EXPECTED_PERF, EXPECTED_PERF)

    proc = subprocess.run(
        [binary], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=args.timeout, check=False
    )
    text = proc.stdout or ""
    m = re.search(r"\[==========\] (\d+) tests from \d+ test suites? ran", text)
    ran = int(m.group(1)) if m else -1
    failed = len(re.findall(r"^\[  FAILED  \]", text, re.M))
    summary = {
        "csv_cases": len(rows),
        "expected_cases": EXPECTED_CASES,
        "expected_perf": EXPECTED_PERF,
        "tests_ran": ran,
        "failed_cases": failed,
        "exit_code": proc.returncode,
        "cmd": binary,
    }
    out_dir = os.path.abspath(args.output)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "self_test_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    LOG.info("%s", json.dumps(summary, indent=2))

    # CSV-driven suite plus fixed contract tests (CsvCaseCount / NullHandle / ...).
    ok = ran >= len(rows) and failed == 0 and proc.returncode == 0
    LOG.info("RESULT: %s", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
