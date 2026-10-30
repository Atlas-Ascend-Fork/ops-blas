#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Run contracts, boundaries, TN96 and alias regressions under real memcheck."""
import argparse
from pathlib import Path
import re
import subprocess
from test_gate import (ARCH, audit_csvs, check_jsonl, check_mapping, check_xml, environment,
                       identity, list_tests, write_summary)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New evidence directory")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--sanitizer", default="mssanitizer")
    parser.add_argument("--timeout", type=int, default=7200)
    return parser.parse_args()


def audit_memcheck(text, run_text):
    starts, clean = text.count("Start memcheck"), text.count("No error detected")
    clean_line = re.compile(
        r".*(?:No error detected|(?:ERROR|WARNING)\s*[:=]\s*0|\b0\s+(?:errors|warnings))\s*$", re.I)
    suspect = []
    for line in text.splitlines():
        if clean_line.fullmatch(line):
            continue
        if re.search(r"\b(error|warning|failed)\b|out.of.bounds|race condition", line, re.I):
            suspect.append(line)
    if re.search(r"stop(?:ping)? DBI|failed to create|instrumentation.*fail|CPU_MODEL_PASS", run_text, re.I):
        raise RuntimeError("Missing instrumentation or CPU mock detected")
    if not starts or starts != clean or suspect:
        raise RuntimeError(f"Incomplete/failed memcheck: starts={starts}, clean={clean}, suspect={suspect[:10]}")
    return {"start_checks": starts, "clean_checks": clean, "instrumentation_verified": True}


def main():
    args = parse_args()
    binary, output = args.binary.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    summary = {"status": "FAIL", "instrumentation_verified": False}
    try:
        summary.update(identity(binary), csv=audit_csvs())
        env = environment(ARCH / "sgemm_grouped_batched_test_full.csv", args.device, output / "cases.jsonl")
        check_mapping(list_tests(binary, env), ARCH / "sgemm_grouped_batched_test_full.csv")
        pattern = "*TC_ED_*:*TC_L0_*:Review/*:*Small8*"
        expected = list_tests(binary, env, pattern)
        if not any("TC_ED_" in name for name in expected) or not any("TC_REVIEW_" in name for name in expected):
            raise RuntimeError("Contract/boundary or TN96 discovery missing")
        xml, memlog = output / "result.xml", output / "memcheck.log"
        command = [args.sanitizer, "--tool=memcheck", f"--log-file={memlog}", "--", str(binary),
                   f"--gtest_filter={pattern}", f"--gtest_output=xml:{xml}"]
        summary["command"] = command
        with (output / "run.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=args.timeout, check=False)
        summary["returncode"] = result.returncode
        if result.returncode:
            raise RuntimeError(f"Memcheck process failed: {result.returncode}")
        summary.update(check_xml(xml, expected))
        summary["jsonl_cases"] = check_jsonl(output / "cases.jsonl", expected)
        text = memlog.read_text(encoding="utf-8", errors="replace")
        # A zero exit without positive DBI records is never accepted.
        run_text = (output / "run.log").read_text(encoding="utf-8", errors="replace")
        summary.update(audit_memcheck(text, run_text))
        summary.update(status="PASS", instrumentation_verified=True)
    except Exception as error:
        summary["error"] = repr(error)
    finally:
        write_summary(output, "memcheck_summary.json", summary)
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
