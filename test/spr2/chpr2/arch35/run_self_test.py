#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Run Chpr2 accuracy cases and verify the complete recorded case set."""
import argparse
import csv
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--csv", type=Path, default=Path(__file__).with_name("chpr2_test.csv"))
    parser.add_argument("--output", required=True, type=Path, help="new output directory")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    binary, csv_path, output = args.binary.resolve(), args.csv.resolve(), args.output.resolve()
    cases = read_rows(csv_path)
    names = [row["case_name"] for row in cases]
    if len(names) != 1200 or len(set(names)) != 1200 or not all(names):
        raise ValueError("Unexpected CSV count, categories, empty name, or duplicate name")
    if (sum(name.startswith("TC_ED_") for name in names) != 28
            or sum(name.startswith("TC_PF_") for name in names) != 200):
        raise ValueError("Unexpected CSV count, categories, empty name, or duplicate name")
    expected = {name for name in names if not name.startswith("TC_PF_")}
    expected.update(f"TC_EXTRA_{i}" for i in range(32))
    output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    for key in ("BLAS_TEST_CASES", "GTEST_FILTER", "GTEST_OUTPUT", "CHPR2_PROFILE", "CHPR2_PROFILE_CALLS"):
        env.pop(key, None)
    env.update(CHPR2_CSV=str(csv_path), CHPR2_RESULTS=str(output / "accuracy.csv"), BLAS_DEVICE_ID=str(args.device))
    xml = output / "gtest.xml"
    command = [str(binary), "--gtest_filter=-*TC_PF*", "--gtest_color=no", "--gtest_output=xml:" + str(xml)]
    with (output / "accuracy.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    records = read_rows(output / "accuracy.csv") if (output / "accuracy.csv").exists() else []
    actual = [row["case_name"] for row in records]
    xml_ok = False
    if xml.is_file():
        root = ET.parse(xml).getroot()
        tests = root.findall(".//testcase")
        xml_ok = (len(tests) == len(expected) + 2 and int(root.get("failures", "0")) == 0
                  and int(root.get("errors", "0")) == 0
                  and all(test.find("failure") is None and test.find("skipped") is None for test in tests))
    passed = (result.returncode == 0 and xml_ok and len(actual) == len(expected)
              and len(set(actual)) == len(actual) and set(actual) == expected
              and all(row.get("passed") == "PASS" for row in records))
    summary = dict(passed=passed, expected_cases=len(expected), recorded_cases=len(actual),
                   missing=sorted(expected - set(actual)), unexpected=sorted(set(actual) - expected),
                   failed=[row["case_name"] for row in records if row.get("passed") != "PASS"],
                   gtest_xml_passed=xml_ok, returncode=result.returncode)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    logging.info("%s", json.dumps(summary, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
