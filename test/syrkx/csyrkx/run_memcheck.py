#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Run instrumented boundary, dispatch and nonfinite regressions; fail closed."""
import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET

from run_self_test import HERE, PREFIX, check_xml, clean_env, discover, load_cases

SPECIAL = [
    "NullHandle", "ValidationBeforeNoOp", "AddressableTensorSpansBeforeLaunch",
    "BetaZeroIgnoresPoisonedC", "ZeroKAndAlphaDoNotReadUnusedInputs", "GaussianComplexComponents",
    "PaddedLongKTailMemcheck", "FiniteComplexCancellationDespiteRealPartialOverflow", "MixedOverflowChunkRepairLarge",
    "MixedOverflowChunkRepairSmall",
    "NormalTransposeExtremeScalingOrder", "NonfiniteComplexTermsN1", "NonfiniteComplexTermsN33",
    "NonfiniteComplexTermsN65", "NonfiniteComplexTermsN257",
    "AlphaBeforeBOverflowDespiteFiniteFinalScale", "MixedOriginalCRiskSmallBetaOne",
    "MixedOriginalCRiskLargeComplexBeta",
    "InputRiskThresholdK1", "InputRiskThresholdK9", "WideRiskSlotsAdaptive",
]
KERNELS = ("csyrkx_direct_kernel", "csyrkx_ordered_kernel", "csyrkx_split_kernel",
           "gemm_cgemm_products_kernel", "csyrkx_epilogue_kernel")


def select_tests(names):
    # L0/ED include invalid/no-op inputs; the other cases exercise direct,
    # ordered, Cube one-part and two-part paths. The supplemental tests cover
    # beta/stride, poison/NaN/Inf, magnitude boundaries and workspace reuse.
    # PaddedLongKTailMemcheck reproduces the long-K partial-block write case.
    # MixedOverflow, nonfinite and original-C risk regressions use separate processes.
    # Cross-shape tests remain in run_self_test.py and can be run under racecheck.
    selected = [PREFIX + n for n in names if n.startswith(("TC_L0_", "TC_ED_"))]
    representatives = ("TC_SQ_020", "TC_SQ_029", "TC_SQ_036", "TC_FL_319", "TC_FL_320",
                       "TC_FL_322", "TC_FL_323")
    if not set(representatives).issubset(names) or len(selected) != 43:
        raise ValueError("required L0/ED or dispatch cases missing")
    selected += [PREFIX + n for n in representatives]
    selected += ["CsyrkxIndependentTest." + n for n in SPECIAL]
    return selected


def verify_sanitizer(text, required=(), allow_no_kernel=False):
    verify_console(text)
    start = Counter(re.findall(r"Start memcheck sanitizer on kernel\s+(.+?)\s*$", text, re.MULTILINE))
    done = Counter(re.findall(r"Sanitizer finished on kernel\s+(.+?)\.\s+No error detected\.", text))
    if start != done or (not start and not allow_no_kernel):
        raise ValueError("missing or unmatched sanitizer kernel start/clean completion")
    if any(not any(k in name for name in start) for k in required):
        raise ValueError("required kernel evidence missing: " + ", ".join(required))
    return dict(done)


def verify_console(text):
    # This exact warning only says that stack traces will lack source lines.
    # All other sanitizer warnings and errors remain failures.
    no_debug = re.compile(
        r"\[mssanitizer\] WARN: Kernel [A-Za-z_][A-Za-z0-9_]* missed debug_line information\. "
        r"Please recompile kernel with `-g' or `-gline-tables-only' option to enable callstack display\."
    )
    clean = re.compile(r"(?:\[mssanitizer\] )?Sanitizer finished on kernel .+\. No error detected\.")
    diagnostics = re.compile(
        r"\b(?:ERROR|WARNING|FATAL|WARN)\b|detected memory leaks|\bLeakSanitizer\b|"
        r"\b(?:direct|indirect) leak of\b|\b[1-9]\d* byte(?:\(s\)|s)? leaked\b", re.IGNORECASE
    )
    for line in text.splitlines():
        if no_debug.fullmatch(line.strip()) or clean.fullmatch(line.strip()):
            continue
        if diagnostics.search(line):
            raise ValueError("mssanitizer reported a diagnostic: " + line[:300])


def csv_requirements(row):
    n, k = int(row["n"]), int(row["k"])
    if row["expect_result"] != "ACLBLAS_STATUS_SUCCESS" or n == 0:
        return (), "invalid input or zero-size no-op"
    alpha_zero = float(row["alpha_real"]) == 0 and float(row["alpha_imag"]) == 0
    beta_one = float(row["beta_real"]) == 1 and float(row["beta_imag"]) == 0
    scale_only = k == 0 or alpha_zero
    small_direct = n <= 8 and k <= 32
    if scale_only and beta_one:
        return (), "scale-only beta-one no-op"
    if scale_only or small_direct:
        return ("csyrkx_direct_kernel",), "direct or scale-only"
    if n <= 64 and k <= 64:
        return ("csyrkx_ordered_kernel",), "ordered"
    mode = "Cube; two partitions" if k > 256 else "Cube; one partition"
    return KERNELS[2:], mode


def special_requirements(short):
    no_launch = {"NullHandle", "ValidationBeforeNoOp", "AddressableTensorSpansBeforeLaunch"}
    direct = {"BetaZeroIgnoresPoisonedC", "ZeroKAndAlphaDoNotReadUnusedInputs",
              "GaussianComplexComponents", "NonfiniteComplexTermsN1"}
    cube = {"PaddedLongKTailMemcheck", "MixedOverflowChunkRepairLarge", "MixedOverflowChunkRepairSmall",
            "MixedOriginalCRiskSmallBetaOne", "MixedOriginalCRiskLargeComplexBeta",
            "InputRiskThresholdK1", "InputRiskThresholdK9", "WideRiskSlotsAdaptive",
            "NonfiniteComplexTermsN65", "NonfiniteComplexTermsN257"}
    if short in no_launch:
        return (), "invalid input or no-op contract"
    if short in direct:
        return ("csyrkx_direct_kernel",), "direct path and input contracts"
    if short in cube:
        return KERNELS[2:], "Cube and risk regression"
    return ("csyrkx_ordered_kernel",), "ordered risk regression"


def requirements(selected, csv_path):
    with csv_path.open(newline="", encoding="utf-8-sig") as stream:
        rows = {row["case_name"]: row for row in csv.DictReader(stream)}
    result = {}
    for test in selected:
        if test.startswith(PREFIX):
            result[test] = csv_requirements(rows[test[len(PREFIX):]])
        else:
            result[test] = special_requirements(test.rsplit(".", 1)[1])
    return result


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--csv", type=Path, default=HERE / "arch35/csyrkx_test.csv")
    parser.add_argument("--tool", default="mssanitizer")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=14400)
    return parser.parse_args()


def execute_case(tool, binary, env, timeout, directory, record):
    name, required = record["name"], record["expected_kernels"]
    xml, detail, console = directory / "result.xml", directory / "sanitizer.log", directory / "run.log"
    child = dict(env, CSYRKX_MODE="accuracy", GTEST_FILTER=name, GTEST_OUTPUT="xml:" + str(xml))
    command = [tool, "--tool=memcheck", "--leak-check=yes", "--log-file=" + str(detail), str(binary)]
    with console.open("w", encoding="utf-8") as log:
        result = subprocess.run(command, env=child, stdout=log, stderr=subprocess.STDOUT,
                                timeout=timeout, check=False)
    record["returncode"] = result.returncode
    if result.returncode:
        raise ValueError(f"mssanitizer exited {result.returncode}")
    check_xml(xml, [name])
    text = detail.read_text(encoding="utf-8", errors="replace") if detail.is_file() else ""
    record["checked_kernels"] = verify_sanitizer(text, required, allow_no_kernel=not required)
    if not required and record["checked_kernels"]:
        raise ValueError("invalid/no-op case unexpectedly launched a kernel")
    out_text = console.read_text(encoding="utf-8", errors="replace")
    verify_console(out_text)
    record["passed"] = True


def run_case(tool, binary, env, timeout, directory, record):
    try:
        execute_case(tool, binary, env, timeout, directory, record)
    except (OSError, ValueError, ET.ParseError, subprocess.TimeoutExpired) as error:
        record["error"] = str(error)


def run_checks(args, out, summary):
    tool = shutil.which(args.tool)
    if not tool:
        raise ValueError("mssanitizer executable not found")
    binary, csv_path = args.binary.resolve(strict=True), args.csv.resolve(strict=True)
    names = load_cases(csv_path)
    selected = select_tests(names)
    env = clean_env(csv_path, args.device)
    found = discover(binary, env, out / "discovery.log")
    if {PREFIX + n for n in names} != {n for n in found if n.startswith(PREFIX)}:
        raise ValueError("CSV discovery mismatch")
    if not set(selected).issubset(found):
        raise ValueError("required memcheck GTests not discovered")
    plans = requirements(selected, csv_path)
    summary["cases"] = []
    seen = set()
    for index, name in enumerate(selected, 1):
        required, reason = plans[name]
        directory = out / f"{index:03d}_{name.rsplit('/', 1)[-1]}"
        directory.mkdir(mode=0o700)
        if os.name == "posix":
            directory.chmod(0o700)
        record = {"name": name, "expected_kernels": list(required), "coverage": reason, "passed": False}
        # Keep the failing case in the summary even when execution raises.
        summary["cases"].append(record)
        run_case(tool, binary, env, args.timeout, directory, record)
        if record["passed"]:
            checked = record.get("checked_kernels")
            if not isinstance(checked, dict):
                raise ValueError("missing or invalid sanitizer kernel evidence")
            seen.update(k for k in KERNELS if any(k in actual for actual in checked))
        (out / "memcheck_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        if not record["passed"]:
            raise ValueError(f"memcheck case failed: {name}: {record.get('error', 'missing diagnostic')}")
    if seen != set(KERNELS):
        raise ValueError("missing global dispatch evidence: " + ", ".join(sorted(set(KERNELS) - seen)))
    summary["executed_tests"] = len(selected)
    summary["covered_kernels"] = sorted(seen)
    summary["passed"] = True


def main():
    args = parse_arguments()
    out = args.output.resolve()
    out.mkdir(mode=0o700, parents=True, exist_ok=False)
    if os.name == "posix":
        out.chmod(0o700)
    summary = {"passed": False, "tool": "memcheck", "leak_check": True}
    try:
        run_checks(args, out, summary)
    except (OSError, ValueError, KeyError, ET.ParseError, subprocess.TimeoutExpired) as error:
        summary["error"] = str(error)
    (out / "memcheck_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
