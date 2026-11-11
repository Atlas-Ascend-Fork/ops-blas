#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Run CSYMV boundary and contract cases under mssanitizer memcheck."""
import argparse
import json
import logging
import os
from pathlib import Path
import re

from run_self_test import audit_csvs, check_mapping, check_xml, environment, list_tests, run_suite


def audit_memcheck(text):
    starts = re.findall(r"Start memcheck sanitizer on kernel\s+([^\s.]+)", text)
    clean = re.findall(r"Sanitizer finished on kernel\s+([^\s.]+)\. No error detected\.", text)
    pending = {}
    for match in re.finditer(r"Start memcheck sanitizer on kernel\s+([^\s.]+)|"
                             r"Sanitizer finished on kernel\s+([^\s.]+)\. No error detected\.", text):
        start, end = match.groups()
        if start:
            pending[start] = pending.get(start, 0) + 1
        elif not pending.get(end, 0):
            raise ValueError("Memcheck completion without a matching start")
        else:
            pending[end] -= 1
    suspect = []
    for line in text.splitlines():
        if "No error detected" in line:
            continue
        if re.search(r"\b(?:ERROR|WARNING)\b|out.of.bounds|stop(?:ping)? DBI|"
                     r"failed to create|instrumentation.*fail|CPU_MODEL_PASS", line, re.I):
            suspect.append(line)
    if not starts or any(pending.values()) or suspect:
        raise ValueError(f"Incomplete/failed instrumentation: starts={len(starts)}, "
                         f"clean={len(clean)}, diagnostics={suspect[:10]}")
    if not any("csymv" in kernel.lower() for kernel in starts):
        raise ValueError("No instrumented CSYMV kernel")
    return {"start_checks": len(starts), "clean_checks": len(clean)}


def run_group(args, name, pattern, env, output):
    binary = args.binary.resolve()
    expected = list_tests(binary, pattern, env)
    xml, detail = output / f"{name}.xml", output / f"{name}_memcheck.log"
    command = [args.sanitizer, "--tool=memcheck", f"--log-file={detail}", "--", str(binary),
               f"--gtest_filter={pattern}", f"--gtest_output=xml:{xml}"]
    record = {"group": name, "status": "FAIL", "command": command, "selected_tests": expected}
    try:
        xml.unlink(missing_ok=True)
        detail.unlink(missing_ok=True)
        record["returncode"] = run_suite(command, output, name, env, args.timeout)
        record.update(check_xml(xml, expected))
        if record.get("returncode") != 0 or not record.get("passed", False):
            raise ValueError("GTest failure or incomplete execution")
        # Prefer the dedicated log; console output may duplicate the same records.
        text = detail.read_text(encoding="utf-8", errors="replace")
        record.update(audit_memcheck(text))
        console = (output / f"{name}.log").read_text(encoding="utf-8", errors="replace")
        if re.search(r"stop(?:ping)? DBI|failed to create|instrumentation.*fail|CPU_MODEL_PASS", console, re.I):
            raise ValueError("Instrumentation failed; see console log")
        record["status"] = "PASS"
    except Exception as error:
        record["error"] = str(error)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", type=int, default=int(os.environ.get("BLAS_DEVICE_ID", "0")))
    parser.add_argument("--sanitizer", default="mssanitizer")
    parser.add_argument("--timeout", type=int, default=14400)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = {"status": "FAIL", "groups": []}
    try:
        summary.update(audit_csvs())
        env = environment(args.device)
        check_mapping(list_tests(args.binary.resolve(), "*", env), performance=True)
        routes = list_tests(args.binary.resolve(), "Csymv/CsymvMemoryCheckTest.*", env)
        if {name.rsplit("/", 1)[-1] for name in routes} != {f"N{n}" for n in (16, 32, 64, 697, 2048, 4096, 8192)}:
            raise ValueError("Missing route boundary tests")
        groups = [("contracts", "*TC_ED_*:*TC_INC_*:*TC_EX_0203"),
                  ("routes", "Csymv/CsymvMemoryCheckTest.*"),
                  ("workspace", "CsymvTest.SmallWorkspaceFallsBack")]
        for name, pattern in groups:
            summary["groups"].append(run_group(args, name, pattern, env, output))
        if all(record["status"] == "PASS" for record in summary["groups"]):
            summary["status"] = "PASS"
    except Exception as error:
        summary["error"] = str(error)
    (output / "memcheck_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    logging.info("%s", json.dumps(summary))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
