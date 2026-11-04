#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Run CSV completeness and execution gates; never accept skipped cases."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

ARCH = Path(__file__).resolve().parent / "arch22"
SPECIAL = [
    "CgemmBatchedCsvTest.CsvCaseCount",
    "CgemmBatchedTest.NullHandle",
    "CgemmBatchedTest.KZeroIgnoresNonFiniteAlpha",
    "CgemmBatchedTest.FusedMultiplyAddReference",
    "CgemmBatchedTest.OffsetPointersAndStreamOrdering",
    "CgemmBatchedTest.ConcurrentStreamsIndependentWorkspaces",
    "CgemmBatchedTest.LargeAlphaZeroBetaCancellation",
    "CgemmBatchedTest.PackedAlphaIntermediateOverflow",
    "CgemmBatchedTest.LargeLeadingDimension",
]


def read_cases(filename, count):
    with (ARCH / filename).open(newline="") as source:
        data = list(csv.DictReader(source))
    if len(data) != count:
        raise ValueError(f"{filename}: expected {count}, got {len(data)}")
    if count != 1200:
        return data
    for prefix, expected in (("TC_EX_", 755), ("TC_PF_", 200)):
        if sum(r["case_name"].startswith(prefix) for r in data) != expected:
            raise ValueError(f"Invalid category count: {prefix}")
    return data


def cases(full):
    rows = []
    names = set()
    for filename, count, suite in (
        ("cgemm_batched_test.csv", 1200, "CgemmBatched"),
        ("cgemm_batched_regression.csv", 669, "CgemmBatchedRegression"),
    ):
        data = read_cases(filename, count)
        for row in data:
            name = row["case_name"]
            if not name or name in names:
                raise ValueError(f"Empty or duplicate case: {name}")
            names.add(name)
            if full or not name.startswith(("TC_EX_", "TC_PF_")):
                rows.append(f"{suite}/CgemmBatchedTest.CsvDriven/{name}")
    return rows + SPECIAL


def check_xml(path, expected):
    tests = ET.parse(path).getroot().findall(".//testcase")
    actual = [t.attrib["classname"] + "." + t.attrib["name"] for t in tests]
    if len(actual) != len(expected) or set(actual) != set(expected):
        raise ValueError("Executed tests do not match expected names/count")
    if any(t.find("failure") is not None or t.find("skipped") is not None
           or t.attrib.get("result") != "completed" for t in tests):
        raise ValueError("Failed, skipped or incomplete test")
    return len(tests)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    summary = {"passed": False, "mode": "smoke" if args.smoke else "full"}
    try:
        expected = cases(not args.smoke)
        summary["expected_tests"] = len(expected)
        env = os.environ.copy()
        for key in tuple(env):
            if key.startswith("GTEST_") or key.startswith("CGEMM_BATCHED_"):
                env.pop(key)
        env.update(BLAS_DEVICE_ID=str(args.device), CGEMM_BATCHED_FULL_CSV="0" if args.smoke else "1")
        xml = output / "results.xml"
        pattern = "*CsvDriven/*:" + ":".join(SPECIAL)
        with (output / "run.log").open("w") as log:
            result = subprocess.run([str(args.binary.resolve()), "--gtest_filter=" + pattern,
                "--gtest_output=xml:" + str(xml)], env=env, stdout=log, stderr=subprocess.STDOUT,
                timeout=14400, check=False)
        summary["returncode"] = result.returncode
        summary["executed_tests"] = check_xml(xml, expected)
        summary["passed"] = result.returncode == 0
    except Exception as error:
        summary["error"] = str(error)
    (output / "self_test_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
