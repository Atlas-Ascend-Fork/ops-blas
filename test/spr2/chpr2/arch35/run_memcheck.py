#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check real Chpr2 kernel launches and API failures with mssanitizer."""
import argparse
from collections import Counter
import csv
import json
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET


SUCCESS_CASES = {
    "upper_unit": ("TC_L0_001", "ACLBLAS_UPPER", 1, 1),
    "lower_unit": ("TC_L0_003", "ACLBLAS_LOWER", 1, 1),
    "upper_negative": ("TC_INC_088", "ACLBLAS_UPPER", -1, -1),
    "lower_negative": ("TC_INC_124", "ACLBLAS_LOWER", -1, -1),
}
NEGATIVE_CASES = ("TC_ED_188", "TC_ED_189", "TC_ED_192", "TC_ED_193")


def check_cases(csv_path):
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    lookup = {row["case_name"]: row for row in rows}
    if len(lookup) != len(rows):
        raise ValueError("Duplicate case names in CSV")
    for name, uplo, incx, incy in SUCCESS_CASES.values():
        row = lookup[name]
        if row["uplo"] != uplo or int(row["incx"]) != incx or int(row["incy"]) != incy:
            raise ValueError("Invalid successful kernel case: " + name)
        if int(row["n"]) <= 0 or row["expect_result"] != "ACLBLAS_STATUS_SUCCESS":
            raise ValueError("Invalid successful kernel case: " + name)
        if any(row[key] == "NULLPTR" for key in ("x_fill", "y_fill", "ap_fill")):
            raise ValueError("Invalid successful kernel case: " + name)
        if row["alpha_real"] == "null":
            raise ValueError("Invalid successful kernel case: " + name)
        if float(row["alpha_real"]) == 0 and float(row["alpha_imag"]) == 0:
            raise ValueError("Invalid successful kernel case: " + name)
    for name in NEGATIVE_CASES:
        if lookup[name]["expect_result"] == "ACLBLAS_STATUS_SUCCESS":
            raise ValueError("Expected a negative case: " + name)


def run_group(args, group, selected, require_kernel):
    output = args.output
    xml, log, detail = (output / (group + suffix) for suffix in (".xml", ".log", "_detail.log"))
    env = os.environ.copy()
    for key in ("CHPR2_PROFILE", "CHPR2_PROFILE_CALLS", "CHPR2_RESULTS", "BLAS_TEST_CASES", "GTEST_FILTER",
                "GTEST_OUTPUT"):
        env.pop(key, None)
    env.update(CHPR2_CSV=str(args.csv), BLAS_DEVICE_ID=str(args.device),
               GTEST_FILTER=":".join(selected), GTEST_OUTPUT="xml:" + str(xml))
    command = [args.tool, "--tool=memcheck", "--leak-check=yes", "--log-file=" + str(detail), str(args.binary)]
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                timeout=args.timeout, check=False)
    text = log.read_text(encoding="utf-8", errors="replace")
    if detail.exists():
        text += "\n" + detail.read_text(encoding="utf-8", errors="replace")
    pattern = r'"?([^"\n]+?)(?:\([^"\n]*\))?"?'
    starts = Counter(re.findall(r"Start memcheck sanitizer on kernel " + pattern + r"\s*$", text, re.MULTILINE))
    finished = Counter(re.findall(r"Sanitizer finished on kernel " + pattern + r"\. No error detected\.", text))
    diagnostics = []
    for line in text.splitlines():
        if re.search(r"\b(?:ERROR|WARNING)\b", line) and "No error detected." not in line:
            diagnostics.append(line)
    tests = ET.parse(xml).getroot().findall(".//testcase") if xml.is_file() else []
    actual = [test.get("name", "").rsplit("/", 1)[-1] for test in tests]
    expected = [name.rsplit("/", 1)[-1].rsplit(".", 1)[-1] for name in selected]
    xml_ok = (Counter(actual) == Counter(expected) and all(test.get("result") == "completed"
              and test.find("failure") is None and test.find("skipped") is None for test in tests))
    chpr2_starts = sum(count for name, count in starts.items() if "chpr2_kernel" in name)
    kernel_ok = starts == finished and (chpr2_starts == 1 if require_kernel else chpr2_starts == 0)
    passed = result.returncode == 0 and xml_ok and kernel_ok and not diagnostics
    return dict(group=group, passed=passed, returncode=result.returncode, expected_tests=len(selected),
                kernel_starts=dict(starts), kernel_finished=dict(finished), diagnostics=diagnostics)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--csv", type=Path, default=Path(__file__).with_name("chpr2_test.csv"))
    parser.add_argument("--tool", default="mssanitizer")
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()
    args.binary, args.csv, args.output = args.binary.resolve(), args.csv.resolve(), args.output.resolve()
    tool = shutil.which(args.tool)
    if not tool:
        raise RuntimeError("mssanitizer not found: " + args.tool)
    args.tool = str(Path(tool).resolve())
    check_cases(args.csv)
    args.output.mkdir(mode=0o750, parents=True, exist_ok=False)
    groups = [(group, ["*/" + case[0]], True) for group, case in SUCCESS_CASES.items()]
    groups.append(("negative", ["*/" + name for name in NEGATIVE_CASES] +
                   ["Chpr2Validation.NullHandlePrecedesAll"], False))
    checks = [run_group(args, *group) for group in groups]
    summary = dict(passed=all(check["passed"] for check in checks), groups=checks)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    logging.info("%s", json.dumps(summary, ensure_ascii=False))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
