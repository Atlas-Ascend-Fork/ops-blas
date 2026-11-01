#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Run instrumented Cher2 memory checks in separate processes for each ACL fixture."""

import argparse
from collections import Counter
import json
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

from verify_accuracy import read_cases


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    default_binary = Path(__file__).resolve().parents[3] / "build-sanitize/test/syr2/cher2/cher2_test"
    parser.add_argument("--binary", type=Path, default=default_binary)
    parser.add_argument("--tool", default="mssanitizer")
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def test_groups():
    negative = [row["case_name"] for row in read_cases() if row["expect_result"] != "ACLBLAS_STATUS_SUCCESS"]
    if len(negative) != 23:
        raise RuntimeError("Cher2 CSV must contain exactly 23 negative cases")
    csv_names = negative + ["TC_PF_1001", "TC_PF_1002", "TC_PF_1004"]
    selected = ["Cher2Test.NullHandle"] + ["Cher2/Cher2Test.CsvDriven/" + name for name in csv_names]
    return [
        ("csv", selected, 27, {"cher2_aiv_kernel"}),
        ("regression", ["Cher2RegressionTest.*"], 4, {"cher2_kernel", "cher2_aiv_kernel"}),
    ]


def check_xml(xml, expected):
    if not xml.exists():
        return False
    try:
        tests = ET.parse(xml).getroot().findall(".//testcase")
    except ET.ParseError:
        return False
    return len(tests) == expected and all(
        test.attrib.get("result") == "completed"
        and test.find("failure") is None
        and test.find("skipped") is None
        for test in tests
    )


def check_group(name, expected, required, returncode, text, xml):
    passed = re.findall(r"\[\s*PASSED\s*\]\s*(\d+) tests?\.", text)
    diagnostics = [
        line for line in text.splitlines()
        if re.search(r"\b(?:WARNING|ERROR)\b", line) and "No error detected." not in line
    ]
    # Full demangling adds quotes and a C++ parameter list on current CANN.
    kernel = r'"?(\w+)(?:\([^"\n]*\))?"?'
    starts = Counter(re.findall(r"Start memcheck sanitizer on kernel " + kernel, text))
    finished = Counter(re.findall(r"Sanitizer finished on kernel " + kernel + r"\. No error detected\.", text))
    kernels_clean = required.issubset(starts) and starts == finished
    tests_clean = bool(passed) and all(int(value) == expected for value in passed) and check_xml(xml, expected)
    clean = returncode == 0 and tests_clean and kernels_clean and not diagnostics
    return dict(
        group=name, passed=clean, expected_tests=expected, reported_tests=passed,
        returncode=returncode, checked_kernels=dict(finished), diagnostics=diagnostics,
    )


def run_group(args, name, selected, expected, required, env):
    log = args.output / (name + ".log")
    detail = args.output / (name + "_detail.log")
    xml = args.output / (name + ".xml")
    detail.unlink(missing_ok=True)
    xml.unlink(missing_ok=True)
    command = [
        args.tool, "--tool=memcheck", "--leak-check=yes", "--log-file=" + str(detail),
        str(args.binary.resolve()),
    ]
    # This tool version parses application flags as its own options. GTest
    # environment variables preserve both the filter and XML output contract.
    child_env = dict(env, GTEST_FILTER=":".join(selected), GTEST_OUTPUT="xml:" + str(xml))
    with log.open("w", encoding="utf-8") as output:
        result = subprocess.run(command, env=child_env, stdout=output, stderr=subprocess.STDOUT, timeout=3600)
    text = log.read_text(encoding="utf-8", errors="replace")
    if detail.exists():
        text += "\n" + detail.read_text(encoding="utf-8", errors="replace")
    return check_group(name, expected, required, result.returncode, text, xml)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    tool = shutil.which(args.tool)
    if tool is None:
        raise RuntimeError("mssanitizer executable not found: " + args.tool)
    args.tool = str(Path(tool).resolve())
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    args.output.chmod(0o750)
    # Each process contains only one ACL fixture, avoiding tool state reuse
    # across the framework's aclFinalize/aclInit lifecycle.
    env = os.environ.copy()
    for key in ("CHER2_PROFILE", "CHER2_PROFILE_RANGE", "CHER2_PROFILE_SAMPLES", "BLAS_TEST_CASES", "GTEST_FILTER"):
        env.pop(key, None)
    checks = [run_group(args, *group, env) for group in test_groups()]
    clean = all(check["passed"] for check in checks)
    summary = dict(passed=clean, expected_tests=sum(check["expected_tests"] for check in checks), groups=checks)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    logging.info("%s", json.dumps(summary, ensure_ascii=False))
    if not clean:
        sys.exit(1)


if __name__ == "__main__":
    main()
