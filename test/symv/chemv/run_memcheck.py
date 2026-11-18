# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Run Chemv contracts and kernel-launching cases with mssanitizer memcheck."""
import argparse
from collections import Counter
import logging
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

from chemv_test_common import (
    HERE, case_test_name, executable, needs_computation, read_cases, run_logged,
    save_summary, supplemental_names, test_environment, validate_xml)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=HERE.parents[2] / "build-memcheck/test/symv/chemv/chemv_test")
    parser.add_argument("--tool", default="mssanitizer")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    return parser.parse_args()


def scalar(row, name, real, imag):
    return float(row[name + "_real"]) == real and float(row[name + "_imag"]) == imag


def reg_fast(row):
    return scalar(row, "alpha", 1, 0) and scalar(row, "beta", 0, 0) and row["incx"] == row["incy"] == "1"


def reg_shape(row, uplo, minimum, maximum):
    return row["uplo"] == uplo and reg_fast(row) and minimum <= int(row["n"]) < maximum


def reg_tail(row, uplo):
    return reg_shape(row, uplo, 65, 256) and int(row["n"]) % 64 != 0


def positive_stride_padding(row):
    increments = (int(row["incx"]), int(row["incy"]))
    return (min(increments) > 0 and max(increments) > 1 and
            int(row["lda"]) > int(row["n"]) and not scalar(row, "alpha", 0, 0))


def ex_tail(row, uplo):
    return (row["case_name"].startswith("TC_EX_") and row["uplo"] == uplo and
            int(row["n"]) > 64 and int(row["n"]) % 64 != 0)


def select_success_cases(rows):
    live = [row for row in rows if needs_computation(row)]
    criteria = {
        "upper_reg_tail": lambda row: reg_tail(row, "ACLBLAS_UPPER"),
        "lower_reg_tail": lambda row: reg_tail(row, "ACLBLAS_LOWER"),
        "upper_reg_middle": lambda row: reg_shape(row, "ACLBLAS_UPPER", 897, 1793),
        "lower_reg_middle": lambda row: reg_shape(row, "ACLBLAS_LOWER", 897, 1793),
        "upper_reg_large": lambda row: reg_shape(row, "ACLBLAS_UPPER", 2048, 2147483648),
        "lower_reg_large": lambda row: reg_shape(row, "ACLBLAS_LOWER", 2048, 2147483648),
        "positive_stride_padding": positive_stride_padding,
        "reg_padding": lambda row: reg_fast(row) and int(row["lda"]) > int(row["n"]),
        "negative_stride": lambda row: int(row["incx"]) < 0 and int(row["incy"]) < 0,
        "alpha_zero_beta_zero": lambda row: scalar(row, "alpha", 0, 0) and scalar(row, "beta", 0, 0),
        "alpha_zero_beta_other": lambda row: scalar(row, "alpha", 0, 0) and not scalar(row, "beta", 0, 0),
        "beta_one_compute": lambda row: scalar(row, "beta", 1, 0) and not scalar(row, "alpha", 0, 0),
        "upper_ex_tail": lambda row: ex_tail(row, "ACLBLAS_UPPER"),
        "lower_ex_tail": lambda row: ex_tail(row, "ACLBLAS_LOWER"),
    }
    selected = {}
    for label, predicate in criteria.items():
        matches = [row for row in live if predicate(row)]
        if not matches:
            raise ValueError("No kernel-launching case for memcheck coverage: " + label)
        selected[label] = matches[0]
    return selected


def test_groups(rows):
    contracts = [case_test_name(row["case_name"]) for row in rows if not needs_computation(row)]
    groups = [dict(name="contracts", tests=contracts + supplemental_names()[:1], require_kernel=False)]
    parameter_names = (
        "uplo", "n", "lda", "incx", "incy", "alpha_real", "alpha_imag", "beta_real", "beta_imag")
    for label, row in select_success_cases(rows).items():
        parameters = {}
        for key in parameter_names:
            parameters[key] = row[key]
        groups.append(dict(name=label, tests=[case_test_name(row["case_name"])], require_kernel=True,
                           parameters=parameters))
    return groups


def sanitizer_records(stdout, detail):
    text = stdout + "\n" + detail
    started = Counter(re.findall(r"Start memcheck sanitizer on kernel (.+)", text))
    finished = Counter(re.findall(r"Sanitizer finished on kernel (.+?)\. No error detected\.", text))
    diagnostics = []
    detail_lines = set(detail.splitlines())
    for line in text.splitlines():
        tool_line = line in detail_lines or "sanitizer" in line.lower()
        severity_line = line.rstrip()
        if severity_line.endswith(". No error detected."):
            severity_line = severity_line[:-len(" No error detected.")]
        if tool_line and re.search(r"\b(?:WARN|WARNING|ERROR|FATAL)\b", severity_line, re.IGNORECASE):
            diagnostics.append(line)
        if re.search(r"\b(?:leaks?|leaked bytes)\s*[:=]\s*[1-9][0-9]*", line, re.IGNORECASE):
            diagnostics.append(line)
    return dict(started_kernels=dict(started), finished_kernels=dict(finished),
                checked_kernels=sum(started.values()), diagnostics=diagnostics,
                balanced=started == finished)


def run_group(args, group, env):
    name = group["name"]
    xml = args.output / (name + ".xml")
    detail = args.output / (name + "_detail.log")
    log = args.output / (name + ".log")
    xml.unlink(missing_ok=True)
    detail.unlink(missing_ok=True)
    command = [args.tool, "--tool=memcheck", "--leak-check=yes",
               "--log-file=" + str(detail), args.binary]
    group_env = dict(env, GTEST_FILTER=":".join(group["tests"]), GTEST_REPEAT="1",
                     GTEST_OUTPUT="xml:" + str(xml))
    result = dict(group, passed=False, command=command, gtest_filter=group_env["GTEST_FILTER"])
    result["returncode"] = run_logged(command, log, group_env)
    output = log.read_text(encoding="utf-8", errors="replace")
    detail_text = detail.read_text(encoding="utf-8", errors="replace") if detail.exists() else ""
    result.update(sanitizer_records(output, detail_text))
    if result["returncode"] == 0:
        result["executed_tests"] = validate_xml(xml, group["tests"])
        chemv_kernels = any("chemv_kernel" in kernel for kernel in result["started_kernels"])
        result["passed"] = result["balanced"] and not result["diagnostics"] and (
            not group["require_kernel"] or (result["checked_kernels"] > 0 and chemv_kernels))
    return result


def execute(args, summary):
    groups = test_groups(read_cases())
    args.binary = executable(args.binary)
    args.tool = executable(args.tool)
    summary.update(binary=args.binary, sanitizer=args.tool, expected_groups=len(groups), groups=[])
    env = test_environment(args.device)
    for group in groups:
        result = run_group(args, group, env)
        summary["groups"].append(result)
        save_summary(args.output, summary)
        logging.info("Chemv memcheck %s: %s", group["name"], "PASS" if result["passed"] else "FAIL")
    summary.update(checked_kernels=sum(group["checked_kernels"] for group in summary["groups"]),
                   passed=all(group["passed"] for group in summary["groups"]))


def main():
    args = parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    args.output.chmod(0o750)
    summary = dict(passed=False, device=args.device)
    save_summary(args.output, summary)
    try:
        execute(args, summary)
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.SubprocessError, ET.ParseError) as error:
        summary["error"] = str(error)
        logging.error("Chemv memcheck failed: %s", error)
    save_summary(args.output, summary)
    logging.info("Chemv memcheck summary: %s", args.output / "summary.json")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
