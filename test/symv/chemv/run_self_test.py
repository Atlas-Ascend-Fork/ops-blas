# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Execute every Chemv CSV case and collect both component accuracy records."""
import argparse
from collections import Counter
import json
import logging
import math
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

from chemv_test_common import (
    HERE, case_test_name, executable, needs_computation, read_cases, run_logged,
    save_summary, supplemental_names, test_environment, validate_xml)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=HERE.parents[2] / "build/test/symv/chemv/chemv_test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    return parser.parse_args()


def precision_records(path, rows):
    records = []
    pattern = re.compile(r"^\[(TC_\w+)_(real|imag)\] (PASSED|FAILED) (.*)$")
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.fullmatch(line)
        if match:
            name, component, status, fields = match.groups()
            metrics = dict(field.split("=", 1) for field in fields.split() if "=" in field)
            records.append(dict(case=name, component=component, status=status, metrics=metrics))
    expected_names = [row["case_name"] for row in rows if needs_computation(row)]
    expected_names += ["TC_RS_" + str(index) for index in range(128)]
    expected = Counter((name, component) for name in expected_names for component in ("real", "imag"))
    if Counter((item["case"], item["component"]) for item in records) != expected:
        raise ValueError("Missing, duplicate or unexpected component precision records")
    for item in records:
        validate_precision(item)
    return records


def validate_precision(item):
    metrics = item["metrics"]
    values = {key: float(metrics[key]) for key in ("MERE", "MARE", "max_abs_error", "matched_ratio")}
    if not all(math.isfinite(value) and value >= 0 for value in values.values()):
        raise ValueError("Invalid numeric precision record: " + item["case"])
    threshold = float(metrics["mere_threshold"])
    multiplier = float(metrics["mare_multiplier"])
    if not math.isfinite(threshold) or not math.isfinite(multiplier) or min(threshold, multiplier) <= 0:
        raise ValueError("Invalid precision threshold: " + item["case"])
    numeric_pass = all((values["matched_ratio"] >= 0.99, values["max_abs_error"] <= 0.01,
                        values["MERE"] < threshold, values["MARE"] < threshold * multiplier,
                        int(metrics["near_zero_cap_violations"]) == 0,
                        int(metrics["nonfinite"]) == int(metrics["nonfinite_matched"])))
    if (not numeric_pass or item["status"] != "PASSED" or
            any(metrics[key] != "PASS" for key in ("mixed", "supplementary"))):
        raise ValueError("Failed precision record: " + item["case"])


def execute(args, summary):
    rows = read_cases()
    expected = [case_test_name(row["case_name"]) for row in rows] + supplemental_names()
    xml = args.output / "accuracy.xml"
    xml.unlink(missing_ok=True)
    command = [executable(args.binary), "--gtest_filter=Chemv/ChemvArch35Test.*:ChemvArch35Test.*",
               "--gtest_repeat=1", "--gtest_output=xml:" + str(xml)]
    summary.update(command=command, csv_cases=len(rows), accuracy_cases=1000, performance_cases=200)
    code = run_logged(command, args.output / "accuracy.log", test_environment(args.device))
    summary["returncode"] = code
    if code != 0:
        raise RuntimeError("Chemv self-test executable failed")
    summary["executed_tests"] = validate_xml(xml, expected)
    records = precision_records(args.output / "accuracy.log", rows)
    (args.output / "precision.json").write_text(json.dumps(records, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    summary.update(executed_csv_cases=summary["executed_tests"] - len(supplemental_names()),
                   component_records=len(records),
                   precision_maxima={key: max(float(item["metrics"][key]) for item in records)
                                     for key in ("MERE", "MARE", "max_abs_error")}, passed=True)


def main():
    args = parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    summary = dict(passed=False, device=args.device)
    save_summary(args.output, summary)
    try:
        execute(args, summary)
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.SubprocessError, ET.ParseError) as error:
        summary["error"] = str(error)
        logging.error("Chemv self-test failed: %s", error)
    save_summary(args.output, summary)
    logging.info("Chemv self-test summary: %s", args.output / "summary.json")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
