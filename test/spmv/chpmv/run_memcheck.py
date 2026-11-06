#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""用 mssanitizer 分进程检查 Chpmv 契约、真实内核路径和内存泄漏。"""

import argparse
from collections import Counter
import json
import logging
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

from run_self_test import clean_environment, executed_records, inventory, read_cases, run_logged


def no_launch(row):
    if row["expect_result"] != "ACLBLAS_STATUS_SUCCESS" or int(row["n"]) == 0:
        return True
    scalars = [row[key] for key in ("alpha_real", "alpha_imag", "beta_real", "beta_imag")]
    return "null" not in scalars and list(map(float, scalars)) == [0.0, 0.0, 1.0, 0.0]


def test_groups(rows):
    edge = [row for row in rows if row["case_name"].startswith("TC_ED_")]
    contract = [row["case_name"] for row in edge if no_launch(row)]
    launch = [row["case_name"] for row in edge if not no_launch(row)]
    simd_unit = []
    for uplo in ("ACLBLAS_UPPER", "ACLBLAS_LOWER"):
        eligible = []
        for row in rows:
            if row["uplo"] != uplo or int(row["n"]) < 512:
                continue
            if not (row["incx"] == row["incy"] == "1") or no_launch(row):
                continue
            if float(row["alpha_real"]) ** 2 + float(row["alpha_imag"]) ** 2 > 0:
                eligible.append(row)
        simd_unit.append(min(eligible, key=lambda row: (int(row["n"]), row["case_name"]))["case_name"])
    return [
        ("contract", ["ValidationOrder", "InvalidUploBeforeOtherArguments"] + contract, 0),
        ("edge_launch", launch, len(launch)),
        ("unit", ["TC_L0_001", "TC_L0_003"], 2),
        ("negative_simt", ["SUP_DIAG_0_33", "SUP_DIAG_1_33"], 2),
        ("simt_511", ["SUP_DIST_0_0_19", "SUP_DIST_0_1_19"], 2),
        ("simd_unit", simd_unit, 2),
        ("simd_512", ["SUP_DIST_0_0_20", "SUP_DIST_0_1_20"], 2),
        ("simd_tail_513", ["SUP_DIST_0_0_12", "SUP_DIST_0_1_12"], 2),
        ("simd_reuse_1025", ["SUP_DIST_0_0_13", "SUP_DIST_0_1_13"], 2),
    ]


def selected_names(registered, suffixes):
    selected = []
    for suffix in suffixes:
        matches = []
        for name in registered:
            if name.rsplit("/", 1)[-1] == suffix or name.rsplit(".", 1)[-1] == suffix:
                matches.append(name)
        if len(matches) != 1:
            raise ValueError("Missing or ambiguous memcheck case: " + suffix)
        selected.append(matches[0])
    if len(selected) != len(set(selected)):
        raise ValueError("Repeated case in a memcheck group")
    return selected


def check_sanitizer(text, expected_launches):
    started = Counter(re.findall(r"Start memcheck sanitizer on kernel (.+)", text))
    finished = Counter(re.findall(r"Sanitizer finished on kernel (.+)\. No error detected\.", text))
    diagnostics = [line for line in text.splitlines()
                   if re.search(r"\b(?:WARNING|ERROR|FATAL)\b|See all detected errors above|exited abnormally", line)]
    kernels = [name.strip('"') for name in started]
    matching = all(name == "chpmv_kernel" or name.startswith("chpmv_kernel(")
                   or name.startswith("_Z12chpmv_kernel") for name in kernels)
    clean = not diagnostics and matching and started == finished and sum(started.values()) == expected_launches
    return {"clean": clean, "kernels_started": dict(started), "kernels_finished": dict(finished),
            "expected_launches": expected_launches, "diagnostics": diagnostics}


def run_group(args, output, selected, launches):
    output.mkdir(mode=0o750)
    xml, detail = output / "results.xml", output / "sanitizer.log"
    command = [args.tool, "--tool=memcheck", "--leak-check=yes", "--log-level=warn",
               "--log-file=" + str(detail), "--", str(args.binary),
               "--gtest_filter=" + ":".join(selected), "--gtest_output=xml:" + str(xml)]
    result = {"name": output.name, "status": "FAIL", "expected_cases": selected, "command": command}
    try:
        result["exit_code"] = run_logged(command, output / "stdout.log", clean_environment(), args.timeout)
        text = (output / "stdout.log").read_text(encoding="utf-8", errors="replace")
        if detail.exists():
            text += "\n" + detail.read_text(encoding="utf-8", errors="replace")
        result.update(check_sanitizer(text, launches))
        records = executed_records(xml, selected)
        result["executed_cases"] = [record["name"] for record in records]
        if result.get("exit_code") == 0 and result.get("clean", False):
            result["status"] = "PASS"
    except (OSError, ValueError, KeyError, ET.ParseError, subprocess.TimeoutExpired) as error:
        result["error"] = str(error)
    return result


def verify_groups(args, output, summary):
    args.binary = args.binary.resolve(strict=True)
    if args.timeout <= 0:
        raise ValueError("Timeout must be positive")
    registered = inventory(args.binary, output, clean_environment(), args.timeout)
    groups = test_groups(read_cases())
    for name, suffixes, launches in groups:
        selected = selected_names(registered, suffixes)
        summary["groups"].append(run_group(args, output / name, selected, launches))
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if not all(group["status"] == "PASS" for group in summary["groups"]):
        raise ValueError("One or more memcheck groups failed")
    summary["status"] = "PASS"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--tool", default="mssanitizer")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=900, help="Timeout in seconds for each process")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False, mode=0o750)
    summary = {"status": "FAIL", "groups": [], "binary": str(args.binary.resolve()), "tool": args.tool}
    try:
        verify_groups(args, output, summary)
    except (OSError, ValueError, KeyError, ET.ParseError, subprocess.TimeoutExpired) as error:
        summary["error"] = str(error)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    logging.info("%s", json.dumps(summary))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    raise SystemExit(main())
