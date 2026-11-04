#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Run instrumented CgemmBatched memory checks in separate processes for each ACL fixture."""

import argparse
import csv
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

from run_self_test import cases, check_xml as verify_xml


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    default_binary = (
        Path(__file__).resolve().parents[3]
        / "build-sanitize/test/gemm_batched/cgemm_batched/cgemm_batched_test"
    )
    parser.add_argument("--binary", type=Path, default=default_binary)
    parser.add_argument("--tool", default="mssanitizer")
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def test_groups():
    cases(True)  # Validate both complete CSVs before selecting representatives.
    arch = Path(__file__).resolve().parent / "arch22"
    with (arch / "cgemm_batched_test.csv").open(newline="") as source:
        negative = [
            r["case_name"]
            for r in csv.DictReader(source)
            if r["expect_result"] != "ACLBLAS_STATUS_SUCCESS"
        ]
    selected = ["CgemmBatchedTest.KZeroIgnoresNonFiniteAlpha",
                "CgemmBatchedTest.LargeAlphaZeroBetaCancellation",
                "CgemmBatchedTest.OffsetPointersAndStreamOrdering"]
    selected += ["CgemmBatched/CgemmBatchedTest.CsvDriven/" + n
                 for n in negative + ["TC_L0_002", "TC_L0_003"]]
    selected += ["CgemmBatchedRegression/CgemmBatchedTest.CsvDriven/" + n
                 for n in ["TC_RG_055", "TC_RG_056", "TC_RG_057"]]
    return [("contract_and_paths", selected, len(selected), set())]


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
        line
        for line in text.splitlines()
        if re.search(r"\b(?:WARNING|ERROR)\b", line, re.IGNORECASE) and "No error detected." not in line
    ]
    # Full demangling adds quotes and a C++ parameter list on current CANN.
    kernel = r'(.+?)'
    starts = Counter(re.findall(r"Start memcheck sanitizer on kernel " + kernel + r"$", text, re.MULTILINE))
    finished = Counter(re.findall(r"Sanitizer finished on kernel " + kernel + r"\. No error detected\.", text))
    kernels_clean = bool(starts) and required.issubset(starts) and starts == finished
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
    record = check_group(name, expected, required, result.returncode, text, xml)
    try:
        verify_xml(xml, selected)
    except Exception as error:
        record.update(passed=False, error=str(error))
    return record


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    tool = shutil.which(args.tool)
    if tool is None:
        raise RuntimeError("mssanitizer executable not found: " + args.tool)
    args.tool = str(Path(tool).resolve())
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    args.output.chmod(0o750)
    # Each process contains only one ACL fixture, avoiding tool state reuse
    # across the framework's aclFinalize/aclInit lifecycle.
    env = os.environ.copy()
    for key in ("CHER2_PROFILE", "CHER2_PROFILE_RANGE", "CHER2_PROFILE_SAMPLES", "BLAS_TEST_CASES", "GTEST_FILTER"):
        env.pop(key, None)
    env["CGEMM_BATCHED_FULL_CSV"] = "1"
    for key in tuple(env):
        if key.startswith("GTEST_"):
            env.pop(key)
    checks = [run_group(args, *group, env) for group in test_groups()]
    clean = all(check["passed"] for check in checks)
    summary = dict(passed=clean, expected_tests=sum(check["expected_tests"] for check in checks), groups=checks)
    (args.output / "memcheck_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    logging.info("%s", json.dumps(summary, ensure_ascii=False))
    if not clean:
        sys.exit(1)


if __name__ == "__main__":
    main()
