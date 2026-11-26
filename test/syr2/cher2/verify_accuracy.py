#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Run the complete CSV-driven GTest suite and reject missing/skipped cases."""

import argparse
import csv
import json
import logging
import sys
import os
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent


def read_cases(cases_csv=HERE / "arch35/cher2_test.csv"):
    with cases_csv.open(encoding="utf-8-sig", newline="") as source:
        raw = list(csv.DictReader(source))
    if len(raw) != 1200:
        raise RuntimeError(f"Cher2 CSV must contain exactly 1200 cases, got {len(raw)}")
    if any(None in row or any(value is None for value in row.values()) for row in raw):
        raise RuntimeError("Cher2 CSV contains a row with missing or extra fields")
    names = [row.get("case_name", "") for row in raw]
    if any(not name for name in names) or len(set(names)) != 1200:
        raise RuntimeError("Cher2 CSV contains an empty or duplicate case name")
    performance = sum(name.startswith("TC_PF_") for name in names)
    if performance != 200 or len(raw) - performance != 1000:
        raise RuntimeError("Cher2 CSV must contain 1000 accuracy and 200 performance cases")
    return raw


def expected_cases(cases_csv):
    raw = read_cases(cases_csv)
    expected = {row["case_name"] for row in raw}
    expected.update(row["case_name"] + "_NORMAL" for row in raw if not row["case_name"].startswith("TC_PF_"))
    expected.update("TC_DIST_" + str(i) for i in range(32))
    expected.add("NullHandle")
    if len(expected) != 2233:
        raise RuntimeError("Cher2 requires exactly 2233 distinct acceptance test names")
    return expected


def make_result_row(name, params, metrics):
    row = dict(params.get(name, {"case": name, "expected": "9"}))
    row["result"] = "PASS"
    for part in ("real", "imag"):
        for key, value in metrics.get(name, {}).get(part, {}).items():
            if key not in ("case", "part"):
                row[part + "_" + key] = float(value)
    row["category"] = "performance_correctness" if name.startswith("TC_PF_") else "accuracy"
    return row


def read_records(output, expected):
    log = (output / "accuracy.log").read_text(encoding="utf-8")
    params, metrics = {}, {}
    for line in log.splitlines():
        if line.startswith("[CASE] "):
            fields = dict(re.findall(r"(\w+)=([^\s]+)", line))
            params[fields["case"]] = fields
        elif line.startswith("[ACCURACY] "):
            fields = dict(re.findall(r"(\w+)=([^\s]+)", line))
            metrics.setdefault(fields["case"], {})[fields["part"]] = fields
    if set(params) != expected - {"NullHandle"}:
        raise RuntimeError("Missing case parameter records")
    return params, metrics


def validate_accuracy_xml(xml, expected):
    if not xml.is_file():
        raise RuntimeError("accuracy.xml is missing")
    try:
        root = ET.parse(xml).getroot()
        if any(int(root.attrib.get(key, 0)) != 0 for key in ("failures", "errors", "disabled", "skipped")):
            raise RuntimeError("GTest XML reports failures, errors, disabled or skipped tests")
    except ET.ParseError as exc:
        raise RuntimeError(f"Failed to parse accuracy.xml: {exc}") from exc
    except ValueError as exc:
        raise RuntimeError(f"Invalid GTest XML root counts: {exc}") from exc
    tests = root.findall(".//testcase")
    names = [t.attrib.get("name", "").split("/")[-1] for t in tests]
    if set(names) != expected or len(names) != len(expected):
        raise RuntimeError(
            f"Case set mismatch: expected {len(expected)}, got {len(names)}; "
            f"missing {sorted(expected - set(names))[:8]}"
        )
    if any(
        t.attrib.get("status") != "run"
        or t.attrib.get("result") != "completed"
        or t.find("failure") is not None
        or t.find("error") is not None
        or t.find("skipped") is not None
        for t in tests
    ):
        raise RuntimeError("GTest contains failure, error, skip, or incomplete case")
    return names


def build_environment():
    env = dict(os.environ)
    for key in [key for key in env if key.startswith("GTEST_")]:
        env.pop(key, None)
    for key in ("CHER2_PROFILE", "CHER2_PROFILE_RANGE", "CHER2_PROFILE_SAMPLES", "BLAS_TEST_CASES"):
        env.pop(key, None)
    return env


def summarize(output, cases_csv=HERE / "arch35/cher2_test.csv"):
    expected = expected_cases(cases_csv)
    names = validate_accuracy_xml(output / "accuracy.xml", expected)
    params, metrics = read_records(output, expected)
    rows = []
    for name in names:
        rows.append(make_result_row(name, params, metrics))
    # Non-empty successful updates must produce both independent component records.
    for row in rows:
        if row["case"] == "NullHandle":
            continue
        nonempty = (
            int(row["expected"]) == 0
            and int(row["n"]) > 0
            and (float(row["alpha_real"]) != 0 or float(row["alpha_imag"]) != 0)
        )
        if nonempty and set(metrics.get(row["case"], {})) != {"real", "imag"}:
            raise RuntimeError("Missing real/imaginary statistics: " + row["case"])
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (output / "accuracy_results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "total": len(rows),
        "passed": len(rows),
        "original_accuracy": 1000,
        "performance_correctness": 200,
        "normal_variants": 1000,
        "random_alpha": 32,
        "null_handle": 1,
        "golden": "Netlib CBLAS 3.10.0 cblas_cher2",
        "component_records": sum(len(v) for v in metrics.values()),
    }
    (output / "accuracy_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    logging.info("%s", json.dumps(summary))


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", type=Path, default=Path("build/test/syr2/cher2/cher2_test"))
    p.add_argument("--output", type=Path, default=Path("cher2_accuracy"))
    p.add_argument("--timeout", type=int, default=3600)
    p.add_argument("--summarize-only", action="store_true")
    a = p.parse_args()
    if a.timeout <= 0:
        p.error("--timeout must be a positive integer")
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=True)
    expected_cases(HERE / "arch35/cher2_test.csv")
    if not a.summarize_only:
        xml = a.output / "accuracy.xml"
        xml.unlink(missing_ok=True)
        env = build_environment()
        with (a.output / "accuracy.log").open("w", encoding="utf-8") as f:
            subprocess.run(
                [str(a.binary.resolve()), "--gtest_filter=-*Cher2RegressionTest*", "--gtest_output=xml:" + str(xml)],
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=a.timeout,
            )
    summarize(a.output)


if __name__ == "__main__":
    main()
