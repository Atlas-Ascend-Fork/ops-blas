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
"""ssbmv arch22 的 mssanitizer memcheck 门禁。

用 mssanitizer 的 memcheck 工具跑一组代表用例（手写就地/workspace/大步长/宽带 k>=n/
beta==0 strided-y 用例，加上 CSV 里窄带、相位副本、宽带列形式、strided 各一条），
要求每个 kernel 都报 "No error detected"，
且 gtest 无失败。--filter 可以指定任意 gtest filter（如 '*TC_L0_*'）。

每个 ':' 分隔的 pattern 各起一个独立进程：mssanitizer 下「用户注入 64B workspace 后换回流」
（InPlaceWithoutWorkspaceReportsAllocFailed）会让同进程后续 launch 报 107017，这是工具与
用例的交互问题、与被检算子无关，分进程跑既能绕开，也让每条 pattern 的结果互不影响。

用法：
    python3 test/sbmv/ssbmv/arch22/run_memcheck.py
    python3 .../run_memcheck.py --filter '*TC_EX_0184*' --device 1
"""

import argparse
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
DEFAULT_BINARY = REPO / "build" / "test" / "sbmv" / "ssbmv" / "ssbmv_test"
DEFAULT_FILTER = (
    "*NullHandle*:*ZeroOrderIsNoOp:*AlphaZeroInPlaceIsExact:*InPlace*:*LargeStrides*:"
    "*RejectsStrideSpan*:*BetaZeroIgnoresStridedY*:*WideBandMatchesReference*:"
    "*TC_L0_001:*TC_L0_002:*TC_SQ_017:*TC_BW_081:*TC_INC_099:*TC_EX_0184"
)

LOGGER = logging.getLogger("ssbmv-memcheck")


def find_sanitizer():
    candidate = os.environ.get("MSSANITIZER") or shutil.which("mssanitizer")
    if not candidate or not os.path.isfile(candidate):
        return None
    return candidate


def count_tests(output, regex):
    match = re.search(regex, output)
    return int(match.group(1)) if match else 0


def run_pattern(sanitizer, binary, pattern, env):
    summary = {"pattern": pattern, "rc": -1, "kernels": 0, "clean": 0, "errors": 0, "passed": 0, "failed": 0,
               "output": "", "ok": False}
    with tempfile.TemporaryDirectory(prefix="ssbmv_memcheck_") as workdir:
        log_path = Path(workdir) / "mssanitizer.log"
        command = [
            sanitizer,
            "--tool=memcheck",
            "--log-file=" + str(log_path),
            "--",
            str(binary),
            "--gtest_filter=" + pattern,
        ]
        result = subprocess.run(command, cwd=workdir, capture_output=True, text=True, env=env, timeout=3600)
        output = result.stdout + result.stderr
        # With --log-file the per-kernel verdicts are written to the log, not to stdout.
        log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
        verdicts = output + log_text

        summary.update({
            "rc": result.returncode,
            "kernels": verdicts.count("Sanitizer finished on kernel"),
            "clean": verdicts.count("No error detected"),
            "errors": len(re.findall(r"\[mssanitizer\].*ERROR", verdicts)),
            "passed": count_tests(output, r"\[\s+PASSED\s+\]\s+(\d+)\s+tests?\."),
            "failed": count_tests(output, r"\[\s+FAILED\s+\]\s+(\d+)\s+tests?, listed below"),
            "output": output,
        })
    # A validation-only case (e.g. the stride-span rejection) passes gtest without launching a
    # kernel; require at least one passing test per pattern, and kernels > 0 over the whole run.
    # Every checked kernel must also carry the "No error detected" verdict.
    summary["ok"] = (
        summary["rc"] == 0 and summary["errors"] == 0 and summary["failed"] == 0 and summary["passed"] > 0 and
        summary["clean"] == summary["kernels"]
    )
    return summary


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    parser = argparse.ArgumentParser(description="ssbmv arch22 mssanitizer memcheck gate")
    parser.add_argument("--binary", default=str(DEFAULT_BINARY))
    parser.add_argument("--filter", default=DEFAULT_FILTER)
    parser.add_argument("--device", type=int, default=None)
    args = parser.parse_args()

    binary = Path(args.binary).resolve()
    if not binary.is_file():
        LOGGER.error("[memcheck] binary not found: %s", binary)
        return 1
    sanitizer = find_sanitizer()
    if sanitizer is None:
        LOGGER.error("[memcheck] mssanitizer not found; source the CANN set_env.sh or set MSSANITIZER")
        return 1

    env = os.environ.copy()
    if args.device is not None:
        env["ASCEND_DEVICE_ID"] = str(args.device)

    patterns = [pattern for pattern in args.filter.split(":") if pattern]
    results = []
    for pattern in patterns:
        summary = run_pattern(sanitizer, binary, pattern, env)
        results.append(summary)
        status = "ok" if summary["ok"] else "FAIL"
        LOGGER.info(
            "[memcheck] pattern=%s kernels=%d clean=%d gtest_passed=%d gtest_failed=%d sanitizer_errors=%d rc=%d %s",
            pattern, summary["kernels"], summary["clean"], summary["passed"], summary["failed"], summary["errors"],
            summary["rc"], status,
        )
        if not summary["ok"]:
            LOGGER.error("%s", summary["output"].rstrip())

    kernels = sum(item["kernels"] for item in results)
    clean = sum(item["clean"] for item in results)
    errors = sum(item["errors"] for item in results)
    failed = sum(item["failed"] for item in results)
    rc_bad = [item["pattern"] for item in results if item["rc"] != 0]
    no_kernel = [item["pattern"] for item in results if item["kernels"] == 0]
    ok = (not rc_bad) and errors == 0 and failed == 0 and kernels > 0 and clean == kernels
    if no_kernel:
        LOGGER.info("[memcheck] note: no kernel launched by %s (validation-only cases)", no_kernel)
    LOGGER.info(
        "[memcheck] totals: patterns=%d kernels=%d clean=%d gtest_failed=%d sanitizer_errors=%d nonzero_rc=%d",
        len(results), kernels, clean, failed, errors, len(rc_bad),
    )
    if ok:
        LOGGER.info("[memcheck] PASS")
        return 0
    LOGGER.error("[memcheck] FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
