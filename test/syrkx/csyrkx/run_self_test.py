#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Validate complete CSV discovery and run accuracy and optional performance execution.

The 200 PF cases are performance inputs. Their execution in performance mode is
not a golden accuracy check or a performance acceptance measurement. Use
--include-pf-golden to additionally run them in accuracy mode and report any
failures honestly; the 1000 non-PF cases are always checked against golden.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
PREFIX = "Csyrkx/CsyrkxIndependentTest.CsvDriven/"
INDEPENDENT_TEST_NAMES = (
    "NullHandle", "ValidationBeforeNoOp", "AddressableTensorSpansBeforeLaunch", "BetaZeroIgnoresPoisonedC",
    "ZeroKAndAlphaDoNotReadUnusedInputs", "GaussianComplexComponents", "LongKChunkTails",
    "FiniteComplexCancellationDespiteRealPartialOverflow", "MixedOverflowChunkRepair",
    "NormalTransposeExtremeScalingOrder", "NonfiniteComplexTermsDespiteFiniteRealPlanes",
    "AlphaBeforeBOverflowDespiteFiniteFinalScale", "MixedOriginalCRiskAndSingleBetaUpdate",
    "InputRiskThresholdBoundary", "PaddedLongKTailMemcheck", "MixedOverflowChunkRepairLarge",
    "MixedOverflowChunkRepairSmall", "NonfiniteComplexTermsN1", "NonfiniteComplexTermsN33",
    "NonfiniteComplexTermsN65", "NonfiniteComplexTermsN257", "MixedOriginalCRiskSmallBetaOne",
    "MixedOriginalCRiskLargeComplexBeta", "InputRiskThresholdK1", "InputRiskThresholdK9", "WideRiskSlotsAdaptive")
ACCEPTANCE_TEST_NAMES = (
    "TrimmedGoldenMatchesDenseNetlibBitwise", "MaxErrorPointDefinesHardLimit", "NonFiniteAndTolerance")
SUPPLEMENTAL = {"CsyrkxIndependentTest." + n for n in INDEPENDENT_TEST_NAMES}
SUPPLEMENTAL.update("CsyrkxAcceptanceRules." + n for n in ACCEPTANCE_TEST_NAMES)


def load_cases(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    names = [row["case_name"] for row in rows]
    error = "CSV must contain 1200 unique named cases, including 200 PF cases"
    if len(names) != 1200 or len(set(names)) != 1200:
        raise ValueError(error)
    if any(not valid_case_name(n) for n in names):
        raise ValueError(error)
    if sum(n.startswith("TC_PF_") for n in names) != 200:
        raise ValueError(error)
    return names


def valid_case_name(name):
    return isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9_]+", name)


def clean_env(csv_path, device):
    env = {}
    for key, value in os.environ.items():
        if key.startswith(("GTEST_", "CSYRKX_")) or key == "BLAS_TEST_CASES":
            continue
        env[key] = value
    env.update(CSYRKX_TEST_CSV=str(csv_path), BLAS_DEVICE_ID=str(device))
    return env


def discover(binary, env, output):
    result = subprocess.run([str(binary), "--gtest_list_tests"], env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180, check=False)
    output.write_text(result.stdout, encoding="utf-8")
    if result.returncode:
        raise ValueError("GTest discovery failed")
    suite = None
    found = []
    for line in result.stdout.splitlines():
        if line and not line[0].isspace() and line.endswith("."):
            suite = line.strip()
        elif line.startswith("  ") and suite:
            name = line.strip().split("  #", 1)[0]
            if name:
                found.append(suite + name)
    if len(found) != len(set(found)):
        raise ValueError("duplicate GTest names in discovery")
    return found


def incomplete_case(case):
    if case.get("status") != "run" or case.get("result") != "completed":
        return True
    return any(case.find(tag) is not None for tag in ("failure", "error", "skipped"))


def check_xml(path, expected):
    root = ET.parse(path).getroot()
    cases = root.findall(".//testcase")
    names = [c.get("classname", "") + "." + c.get("name", "") for c in cases]
    if len(names) != len(expected) or len(names) != len(set(names)) or set(names) != set(expected):
        raise ValueError("executed test names/count differ from requested set")
    if any(incomplete_case(c) for c in cases):
        raise ValueError("failed, skipped or incomplete test")
    if any(int(root.get(key, "0")) for key in ("failures", "errors", "disabled", "skipped")):
        raise ValueError("XML reports failures, errors, disabled or skipped tests")
    return len(cases)


def run_group(binary, out, name, tests, env, mode):
    xml = out / (name + ".xml")
    child = dict(env, CSYRKX_MODE=mode)
    with (out / (name + ".log")).open("w", encoding="utf-8") as log:
        result = subprocess.run([str(binary), "--gtest_filter=" + ":".join(tests),
                                 "--gtest_output=xml:" + str(xml)], env=child, stdout=log,
                                stderr=subprocess.STDOUT, timeout=14400, check=False)
    count = check_xml(xml, tests)
    if result.returncode:
        raise ValueError(f"{name} exited {result.returncode}")
    if mode == "performance":
        text = (out / (name + ".log")).read_text(encoding="utf-8", errors="replace")
        begin = re.findall(r"^CSYRKX_PERF_BEGIN,([^,\r\n]+),", text, re.MULTILINE)
        end = re.findall(r"^CSYRKX_PERF_END,([^,\r\n]+),successful_calls=56$", text, re.MULTILINE)
        wanted = [x[len(PREFIX):] for x in tests]
        if begin != wanted or end != wanted:
            raise ValueError("incomplete performance execution markers")
    return {"tests": count, "returncode": result.returncode, "mode": mode}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--csv", type=Path, default=HERE / "arch35/csyrkx_test.csv")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--include-pf-golden", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    summary = {"passed": False, "performance_is_accuracy": False, "performance_is_acceptance": False}
    try:
        binary, csv_path = args.binary.resolve(strict=True), args.csv.resolve(strict=True)
        names = load_cases(csv_path)
        env = clean_env(csv_path, args.device)
        found = discover(binary, env, out / "discovery.log")
        csv_tests = {PREFIX + n for n in names}
        actual_csv = {n for n in found if n.startswith(PREFIX)}
        if csv_tests != actual_csv:
            missing = len(csv_tests - actual_csv)
            extra_count = len(actual_csv - csv_tests)
            raise ValueError(f"CSV discovery mismatch: missing={missing}, extra={extra_count}")
        extra = [n for n in found if n.startswith(("CsyrkxIndependentTest.", "CsyrkxAcceptanceRules."))]
        if set(extra) != SUPPLEMENTAL:
            raise ValueError("supplemental GTest discovery mismatch")
        accuracy = [PREFIX + n for n in names if not n.startswith("TC_PF_")] + extra
        performance = [PREFIX + n for n in names if n.startswith("TC_PF_")]
        summary["discovered_csv"] = len(actual_csv)
        summary["accuracy"] = run_group(binary, out, "accuracy", accuracy, env, "accuracy")
        summary["performance_execution"] = run_group(binary, out, "performance", performance, env, "performance")
        if args.include_pf_golden:
            summary["pf_golden_accuracy"] = run_group(binary, out, "pf_golden", performance, env, "accuracy")
        summary["passed"] = True
    except (OSError, ValueError, KeyError, ET.ParseError, subprocess.TimeoutExpired) as error:
        summary["error"] = str(error)
    (out / "self_test_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
