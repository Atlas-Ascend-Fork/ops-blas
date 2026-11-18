#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Run aclblasHgemmBatched memory-safety contracts under mssanitizer memcheck.

The selected queue stresses the paths that most need memory-safety evidence:
all negative/boundary contracts (every non-SUCCESS row, i.e. the TC_ED queue) plus
the TC_L0 tiny baseline, the FLOAT16 direct copy-out (fixpipe F322F16 from the
FP32 L0C accumulator, no FP32 staging buffer) across NN/TN/NT/TT, the batchCount
boundaries (b=0 no-op, b=1 and b=1024), special-value fills (extreme/Inf/NaN in A
and C), and the beta==0 zero-write branches where C is filled with NaN
(TC_ED_144/168/169) so a stray read of C would surface as a diagnostic.

Each selected case runs in its own mssanitizer process: memcheck instruments a
single kernel launch, and reusing one process (hence one handle/workspace) across
several proofile-heavy launches makes the tool's GM shadow state carry over
between launches, producing cross-launch diagnostics that do not reproduce when
the same launch is checked in isolation. Per-case isolation is therefore the
correct methodology and provides per-case evidence. Every case must keep the full
mssanitizer log, GTest XML and the paired "Start memcheck" / "No error detected"
records; a missing instrumentation record, a diagnostic or a non-zero exit fails
the gate.
"""
import argparse
from collections import Counter
import hashlib
import json
import logging
from pathlib import Path
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

from run_self_test import SUITE, execution_env, read_cases, test_completed_cleanly

DIRECTORY = Path(__file__).resolve().parent

# Representative positive shapes: the FLOAT16 direct copy-out fast path
# (alpha == 1, beta == 0) plus one beta!=0 (temporary-output) path, spanning all
# NN/TN/NT/TT combinations.  batchCount boundaries cover the real lower bound
# (b=1 normal compute and b=0 no-op) and the upper bound (b=1024 batch-parallel).
REPRESENTATIVE_SHAPES = [
    "TC_MN_043",  # NN, 256x256x256, b=4, alpha=1 beta=0 (fixpipe direct out)
    "TC_MN_059",  # TN, 127x129x65, b=2, alpha=1 beta=0
    "TC_MN_060",  # NT, 256x1024x512, b=2, alpha=1 beta=0
    "TC_EX_0828", # TT, 128x31, b=1, beta=1 (temporary-output path)
    "TC_EX_0173", # CN, 512x511x512, b=1 (batchCount lower bound, normal compute)
    "TC_ED_132",  # NN, 8x8x8, b=0 (batchCount zero: legal no-op, no array access)
    "TC_CV_121",  # 32x32x32, b=1024 (batchCount upper bound, batch-parallel path)
]
# Special-value fills straight into the copy-out/epilogue: extreme magnitudes and
# Inf/NaN in A, B and C, so a fixpipe/epilogue cast or aliasing defect surfaces.
SPECIAL_VALUE_CASES = [
    "TC_FL_100",  # A = RANDOM_EXTREME
    "TC_FL_105",  # C = VALUE_NORM_INF
    "TC_FL_106",  # A = VALUE_NORM_NAN
]
# beta==0 zero-write branches where C is NaN-filled: a stray read of C would show
# up as a diagnostic instead of being silently multiplied by zero.
ZERO_WRITE_CASES = [
    "TC_ED_144",  # beta=0, C=NaN -> must be zero-filled without reading C
    "TC_ED_168",  # k=0, beta=0, C=NaN -> early-exit zero fill
    "TC_ED_169",  # alpha=0, beta=0, C=NaN -> early-exit zero fill
]


def expects_kernel(row):
    """True when a row must launch at least one compute/epilogue kernel.

    Invalid arguments and legal no-ops (m==0, n==0, batchCount==0) return before
    any kernel launch; every other SUCCESS row must produce instrumentation
    evidence, otherwise a silently-uninstrumented kernel could pass as "clean".
    """
    if row["expect_result"] != "ACLBLAS_STATUS_SUCCESS":
        return False
    m, n, batch = int(row["m"]), int(row["n"]), int(row["batchCount"])
    return m > 0 and n > 0 and batch > 0


def select_cases(rows):
    names = {row["case_name"]: row for row in rows}
    selected = [row["case_name"] for row in rows if row["expect_result"] != "ACLBLAS_STATUS_SUCCESS"]
    for name in REPRESENTATIVE_SHAPES + SPECIAL_VALUE_CASES + ZERO_WRITE_CASES:
        if name not in names:
            raise ValueError(f"Representative case '{name}' missing from CSV")
        if names[name]["expect_result"] != "ACLBLAS_STATUS_SUCCESS":
            raise ValueError(f"Representative case '{name}' is not a success row")
        selected.append(name)
    # Preserve CSV order, drop duplicates.
    seen = set()
    ordered = []
    for name in selected:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def sanitizer_result(text):
    starts = Counter(re.findall(r"Start memcheck sanitizer on kernel\s+(\S+)", text))
    finished = Counter(re.findall(r"Sanitizer finished on kernel\s+(\S+)\. No error detected\.", text))
    diagnostics = []
    for line in text.splitlines():
        if not re.search(r"\b(?:WARNING|ERROR|FATAL)\b", line, re.IGNORECASE):
            continue
        if "No error detected." in line:
            continue
        diagnostics.append(line)
    # Negative/boundary cases (e.g. invalid arguments, batchCount==0 no-ops)
    # return before launching any kernel, so zero instrumentation records is
    # valid for them.  A case passes when every started kernel finished clean and
    # no diagnostic was reported.
    has_evidence = starts == finished and not diagnostics
    return {
        "passed": has_evidence,
        "kernel_starts": sum(starts.values()),
        "clean_kernel_finishes": sum(finished.values()),
        "diagnostics": diagnostics,
    }


def check_xml(xml, case):
    root = ET.parse(xml).getroot()
    tests = root.findall(".//testcase")
    actual = [test.get("classname", "") + "." + test.get("name", "") for test in tests]
    expected = f"{SUITE}/{SUITE}Test.CsvDriven/{case}"
    if actual != [expected]:
        raise ValueError(f"Executed tests do not match the selected case: {actual}")
    for test in tests:
        if not test_completed_cleanly(test):
            raise ValueError(f"Failed, skipped or incomplete test: {test.get('name')}")
    return len(tests)


def run_case(args, case, env, require_kernel):
    stage = args.output / case
    stage.mkdir(mode=0o750)
    xml = stage / "gtest.xml"
    detail = stage / "sanitizer_detail.log"
    log = stage / "process.log"
    command = [
        args.tool,
        "--tool=memcheck",
        "--leak-check=yes",
        "--log-file=" + str(detail),
        "--",
        str(args.binary.resolve()),
        "--gtest_filter=" + f"*CsvDriven/{case}",
        "--gtest_output=xml:" + str(xml),
    ]
    record = {"case": case, "require_kernel": require_kernel, "command": command, "passed": False}
    (stage / "command.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    with log.open("w", encoding="utf-8") as output:
        result = subprocess.run(command, env=dict(env), stdout=output, stderr=subprocess.STDOUT, timeout=args.timeout)
    record["returncode"] = result.returncode
    text = log.read_text(encoding="utf-8", errors="replace")
    if detail.is_file():
        text += "\n" + detail.read_text(encoding="utf-8", errors="replace")
    (stage / "combined.log").write_text(text, encoding="utf-8")
    record["sanitizer"] = sanitizer_result(text)
    try:
        check_xml(xml, case)
        if result.returncode != 0:
            raise ValueError("Test process returned nonzero")
        if not record["sanitizer"]["passed"]:
            raise ValueError("Sanitizer reported diagnostics or incomplete kernel instrumentation")
        if require_kernel and record["sanitizer"]["kernel_starts"] == 0:
            raise ValueError("No kernel instrumentation evidence for a compute case")
        record["passed"] = True
    except (ValueError, OSError, ET.ParseError) as error:
        record["first_failure"] = str(error)
    (stage / "result.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    default = DIRECTORY.parents[2] / "build-sanitize/test/gemm/hgemmBatched/hgemmBatched_test"
    parser.add_argument("--binary", type=Path, default=default)
    parser.add_argument("--tool", default="mssanitizer")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--dry-run", action="store_true", help="Print the selected queue; no NPU call")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    args = parse_args()
    csv_path, rows = read_cases()
    selected = select_cases(rows)
    row_by_name = {row["case_name"]: row for row in rows}
    if args.dry_run:
        logging.info("%s", json.dumps({"status": "planned", "npu_calls": 0, "selected_cases": selected,
                                       "expected_tests": len(selected)}, indent=2))
        return 0
    if args.timeout <= 0:
        raise ValueError("timeout must be positive")
    tool = shutil.which(args.tool)
    if tool is None:
        raise RuntimeError("mssanitizer executable not found: " + args.tool)
    args.tool = str(Path(tool).resolve())
    if not args.binary.is_file():
        raise FileNotFoundError(args.binary)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    args.output.chmod(0o750)
    env = execution_env()
    env["BLAS_DEVICE_ID"] = str(args.device)
    results = []
    for case in selected:
        result = run_case(args, case, env, expects_kernel(row_by_name[case]))
        results.append(result)
        if not result["passed"]:
            break
    passed = len(results) == len(selected) and all(result["passed"] for result in results)
    summary = {
        "passed": passed,
        "expected_cases": len(selected),
        "clean_cases": sum(1 for result in results if result["passed"]),
        "csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "results": results,
        "untested": selected[len(results):],
    }
    (args.output / "memcheck_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    logging.info("%s", json.dumps(summary, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
