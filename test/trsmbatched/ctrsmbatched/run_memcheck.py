#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Run arch22 boundary cases with an instrumented library and CANN mssanitizer."""

import json
import logging
from pathlib import Path
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET

import run_self_test as self_test


def sanitizer_result(report, application_log):
    kernel_log = report.read_text() if report.exists() else ""
    diagnostics = re.compile(r"^\s*\[mssanitizer\]\s*(?:warning|warn|error|fatal)\b.*$", re.I | re.M)
    messages = diagnostics.findall(kernel_log)
    messages += [line for line in diagnostics.findall(application_log.read_text()) if line not in set(messages)]
    errors = [line for line in messages if re.search(r"\]\s*(?:error|fatal)\b", line, re.I)]
    warnings = [line for line in messages if re.search(r"\]\s*warn(?:ing)?\b", line, re.I)]
    starts = re.findall(r'Start memcheck sanitizer on kernel "([^\n]+)"', kernel_log)
    finishes = re.findall(r'Sanitizer finished on kernel "([^\n]+)"', kernel_log)
    return dict(errors=len(errors), warnings=len(warnings), diagnostics=messages, scanned_kernels=len(starts),
                kernel_names=sorted(set(starts)), completed_kernels=len(finishes),
                scan_complete=bool(starts) and starts == finishes)


def instrumented_library(args, env, output):
    log = output / "linked-library.log"
    if self_test.run_command(["ldd", str(args.binary.resolve())], env, log, args.timeout, output) != 0:
        raise ValueError("ldd failed; see linked-library.log")
    match = re.search(r"^\s*libops_blas\.so\s+=>\s+(.+?)\s+\(", log.read_text(), re.M)
    if match is None:
        raise ValueError("test binary must dynamically load libops_blas.so")
    library = Path(match.group(1)).resolve()
    data = library.read_bytes()
    if b".SanitizerFileMapping" not in data or b"__sanitizer_report_" not in data:
        raise ValueError(
            "loaded libops_blas.so has no CANN sanitizer instrumentation; "
            "rebuild with --cce-enable-sanitizer"
        )
    return str(library)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = self_test.argument_parser(__doc__).parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, mode=0o750)
    summary = dict(status="FAIL", scope="selected" if args.cases else "boundary")
    try:
        names = self_test.read_csv_cases(args.csv)
        env = self_test.device_environment(args.device)
        summary["instrumented_library"] = instrumented_library(args, env, output)
        registered = self_test.registered_tests(args, env, output, names)
        cases = args.cases or [name for name in names if name.startswith(("TC_ED_", "TC_L0_")) or name == "TC_AB_075"]
        selected = self_test.select_tests(registered, cases)
        tool = shutil.which("mssanitizer")
        if tool is None:
            raise FileNotFoundError("mssanitizer is missing from PATH; load the matching CANN environment")
        xml = output / "gtest.xml"
        env.update(GTEST_FILTER=":".join(selected), GTEST_OUTPUT="xml:" + str(xml))
        report, log = output / "memcheck.log", output / "run.log"
        command = [tool, "--tool=memcheck", "--log-file=" + str(report), str(args.binary.resolve())]
        summary.update(tool=tool, command=command, selected_cases=len(selected), registered_cases=len(registered))
        summary["exit_code"] = self_test.run_command(command, env, log, args.timeout, output)
        summary.update(self_test.read_test_result(xml, selected))
        summary.update(sanitizer_result(report, log))
        passed = (summary["exit_code"] == 0 and summary["passed"] and summary["scan_complete"]
                  and summary["errors"] == 0 and summary["warnings"] == 0)
        summary["status"] = "PASS" if passed else "FAIL"
    except (OSError, ValueError, ET.ParseError, subprocess.TimeoutExpired) as error:
        summary["error"] = str(error)
    (output / "memcheck_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    logging.info("%s", json.dumps(summary, indent=2))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
