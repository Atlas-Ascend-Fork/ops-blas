#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""aclblasCher 内存安全门禁：mssanitizer memcheck 跑契约/边界用例。

用法：
  python3 run_memcheck.py --binary ./build/test/syr/cher/cher_test
退出码：0 无错误；1 检出 WARNING/ERROR；3 未安装 mssanitizer（SKIP）。
"""
import argparse
import glob
import json
import logging
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(level=logging.INFO, format="%(message)s")
LOG = logging.getLogger("cher_memcheck")


def find_tool():
    for name in ("mssanitizer", "ms_sanitizer"):
        path = shutil.which(name)
        if path:
            return path
    hits = glob.glob("/usr/local/Ascend/**/bin/mssanitizer", recursive=True)
    return hits[0] if hits else None


def audit_memcheck(text, returncode):
    """Host-side gate used by test_validation.py (BL-34 real import)."""
    warnings = len(re.findall(r"WARNING: MemorySanitizer|ERROR: MemorySanitizer|LeakSanitizer", text or ""))
    starts = (text or "").count("Start memcheck")
    finished = len(re.findall(r"No error detected", text or ""))
    ok = warnings == 0 and returncode == 0 and starts > 0
    return {
        "warnings": warnings,
        "starts": starts,
        "finished_clean": finished,
        "exit_code": returncode,
        "ok": ok,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default=os.path.join(HERE, "../../../build/test/syr/cher/cher_test"))
    # Contract + edge CSV families; keep filter narrow so the gate finishes on 950.
    ap.add_argument(
        "--filter",
        default=(
            "*NullHandle*:*HostZero*:*DeviceZero*:*DeviceAlpha*:*NonzeroRejects*:"
            "*InvalidStructure*:*CsvCaseCount*:*TC_ED*:*TC_FL*"
        ),
    )
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--output", default=HERE)
    args = ap.parse_args()

    tool = find_tool()
    binary = os.path.abspath(args.binary)
    if tool is None:
        LOG.error("SKIP: mssanitizer not found; run this gate where the CANN sanitizer is installed.")
        return 3
    if not os.path.isfile(binary):
        LOG.error("SKIP: binary not found: %s", binary)
        return 3

    cmd = [tool, "--tool=memcheck", "--leak-check=yes", binary, "--gtest_filter=" + args.filter]
    LOG.info("RUN: %s", " ".join(cmd))
    proc = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=args.timeout, check=False
    )
    summary = audit_memcheck(proc.stdout or "", proc.returncode)
    summary["tool"] = tool
    out_dir = os.path.abspath(args.output)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "memcheck_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    LOG.info("%s", json.dumps(summary, indent=2))
    if summary["starts"] == 0:
        LOG.error(
            "FAIL: no 'Start memcheck' records; gtest filter matched 0 tests or mssanitizer did not run"
        )
    LOG.info("RESULT: %s", "PASS" if summary["ok"] else "FAIL")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
