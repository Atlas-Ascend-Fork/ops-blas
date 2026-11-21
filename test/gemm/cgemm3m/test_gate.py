#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""CSV and execution completeness checks for the Cgemm3m review gates."""

import argparse
import csv
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import xml.etree.ElementTree as ET

CSV_PATH = Path(__file__).resolve().parent / "arch35/cgemm3m_test.csv"
CSV_PREFIX = "Cgemm3m/Cgemm3mTest.CsvDriven/"
SUPPLEMENTARY = {
    "Cgemm3mCsvIntegrity.CsvCaseCount",
    "Cgemm3mVerifier.MatchedRatioAndMaxError",
    "Cgemm3mVerifier.MagnitudeScaledTolerance",
    "Cgemm3mVerifier.NonfiniteRules",
    "Cgemm3mTest.NullHandle",
    "Cgemm3mTest.EmptyOutputPreservesSentinel",
    "Cgemm3mTest.ZeroProductAllowsNullInputs",
    "Cgemm3mTest.SegmentCancellation",
}


def audit_csv(path=CSV_PATH):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "case_name" not in reader.fieldnames:
            raise RuntimeError("Missing CSV case_name header")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise RuntimeError("Malformed CSV row")
    names = [row["case_name"] for row in rows]
    if any(not name or name != name.strip() for name in names) or len(set(names)) != len(names):
        raise RuntimeError("Empty or duplicate CSV case names")
    performance = sum(name.startswith("TC_PF_") for name in names)
    if len(names) != 1200 or performance != 200:
        raise RuntimeError(f"CSV count mismatch: total={len(names)}, performance={performance}")
    return names, {"total": len(names), "accuracy": len(names) - performance,
                   "performance": performance, "sha256": file_sha(path)}


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def environment(device, reference):
    if device < 0:
        raise ValueError("Device must be nonnegative")
    env = {key: value for key, value in os.environ.items() if not key.startswith("GTEST_")}
    for key in ("BLAS_TEST_CASES", "CGEMM3M_PROFILE", "CGEMM3M_WARMUP", "CGEMM3M_SAMPLES"):
        env.pop(key, None)
    libraries = [reference / name for name in ("libcblas.so", "libblas.so")]
    hashes = {str(path): file_sha(path) for path in libraries}
    env["LD_PRELOAD"] = " ".join(str(path) for path in libraries)
    env["BLAS_DEVICE_ID"] = str(device)
    return env, hashes


def run_logged(command, env, log, timeout):
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired as error:
                raise RuntimeError(f"Timed out; process group {process.pid} termination is pending") from error
            raise


def discover(binary, env, output, timeout):
    log = output / "discovery.log"
    code = run_logged([str(binary), "--gtest_list_tests", "--gtest_filter=*"], env, log, timeout)
    if code:
        raise RuntimeError(f"GTest discovery failed: exit={code}")
    suite, names = "", []
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        value = line.split("#", 1)[0].strip()
        if not value:
            continue
        if not line[0].isspace() and value.endswith("."):
            suite = value
        elif line[0].isspace() and suite:
            names.append(suite + value)
    if not names or len(names) != len(set(names)):
        raise RuntimeError("Empty or duplicate GTest discovery")
    return names


def check_mapping(names, cases):
    expected = {CSV_PREFIX + name for name in cases} | SUPPLEMENTARY
    if len(names) != len(expected) or set(names) != expected:
        raise RuntimeError(f"GTest/CSV mismatch: missing={sorted(expected - set(names))[:5]}, "
                           f"extra={sorted(set(names) - expected)[:5]}")
    return {"csv_cases": len(cases), "supplementary": len(SUPPLEMENTARY), "total": len(expected)}


def check_xml(path, expected):
    root = ET.parse(path).getroot()
    nodes = list(root.iter("testcase"))
    actual = [node.get("classname", "") + "." + node.get("name", "") for node in nodes]
    if not expected or len(actual) != len(set(actual)) or set(actual) != set(expected):
        raise RuntimeError("Missing, extra or duplicate executed tests")
    for node in nodes:
        if node.get("status") != "run" or node.get("result") != "completed":
            raise RuntimeError(f"Incomplete or skipped test: {node.get('name')}")
        if any(node.find(tag) is not None for tag in ("failure", "error", "skipped")):
            raise RuntimeError(f"Failed test: {node.get('name')}")
    for suite in root.iter():
        if suite.tag in ("testsuites", "testsuite"):
            if any(int(suite.get(key, "0")) for key in ("failures", "errors", "disabled", "skipped")):
                raise RuntimeError("Nonzero GTest failure/error/skip summary")
    return {"executed": len(actual), "failed": 0, "skipped": 0}


def audit_memcheck(text, run_text):
    starts = text.count("Start memcheck")
    clean = text.count("No error detected")
    if not starts or starts != clean:
        raise RuntimeError(f"Incomplete memcheck: starts={starts}, clean={clean}")
    for line in text.splitlines():
        residual = re.sub(r"No error detected\.?|\b(?:ERROR|WARNING)\s*[:=]\s*0\b|\b0 (?:errors|warnings)\b",
                          "", line, flags=re.I)
        if re.search(r"\b(error|warning|failed)\b|out.of.bounds|race condition", residual, re.I):
            raise RuntimeError(f"Memcheck diagnostic: {line[:300]}")
    if re.search(r"stop(?:ping)? DBI|failed to create|instrumentation.*fail|CPU_MODEL_PASS|"
                 r"records undetected|detection result maybe unreliable", run_text, re.I):
        raise RuntimeError("Instrumentation failed or CPU mock output detected")
    return {"start_checks": starts, "clean_checks": clean}


def parse_args(description, memcheck=False):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path, help="Prepared Netlib 3.10.0 build/lib directory")
    parser.add_argument("--output", required=True, type=Path, help="New evidence directory; never overwritten")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=400 if memcheck else 14400, help="Seconds per process")
    if memcheck:
        parser.add_argument("--sanitizer", default="mssanitizer")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    return args


def run_group(args, env, output, name, expected, memcheck=False, sanitizer_options=()):
    xml = output / f"{name}.xml"
    log = output / f"{name}.log"
    if not expected:
        raise RuntimeError(f"Empty test group: {name}")
    command = [str(args.binary.resolve()), "--gtest_filter=" + ":".join(sorted(expected)),
               f"--gtest_output=xml:{xml}", "--gtest_repeat=1", "--gtest_color=no"]
    memlog = output / f"{name}_memcheck.log"
    if memcheck:
        command = [args.sanitizer, "--tool=memcheck", f"--log-file={memlog}",
                   *sanitizer_options, "--", *command]
    record = {"command": command, "expected": len(expected)}
    code = run_logged(command, env, log, args.timeout)
    record["returncode"] = code
    if code:
        raise RuntimeError(f"{name} failed: exit={code}; see {log}")
    record.update(check_xml(xml, expected))
    if memcheck:
        record.update(audit_memcheck(memlog.read_text(encoding="utf-8", errors="replace"),
                                    log.read_text(encoding="utf-8", errors="replace")))
    return record


def run_isolated_group(args, env, output, group, selected, records):
    record = {"expected": len(selected), "executed": 0, "start_checks": 0,
              "clean_checks": 0, "runs": []}
    records[group] = record
    for index, case in enumerate(selected):
        result = run_group(args, env, output, f"{group}_{index:03d}", [case], memcheck=True)
        record["runs"].append(result)
        for key in ("executed", "start_checks", "clean_checks"):
            record[key] += result[key]
    if record["executed"] != len(selected):
        raise RuntimeError(f"Incomplete memcheck group: {group}")


def run_memcheck_groups(args, env, output, names, groups, isolated_groups, records):
    for group, patterns in groups:
        selected = [name for name in names if any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)]
        if not selected:
            raise RuntimeError(f"Empty test group: {group}")
        if group in isolated_groups:
            run_isolated_group(args, env, output, group, selected, records)
        else:
            records[group] = run_group(args, env, output, group, selected, memcheck=True)


def execute(args, groups=None, isolated_groups=()):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    summary = {"status": "FAIL", "device": args.device, "groups": {}}
    if groups is not None:
        summary["instrumentation_verified"] = False
    name = "memcheck_summary.json" if groups is not None else "self_test_summary.json"
    try:
        binary = args.binary.resolve()
        summary.update(binary=str(binary), binary_sha256=file_sha(binary))
        cases, summary["csv"] = audit_csv()
        env, summary["reference_libraries_sha256"] = environment(args.device, args.reference.resolve())
        names = discover(binary, env, output, args.timeout)
        summary["discovery"] = check_mapping(names, cases)
        if groups is None:
            summary["groups"]["full"] = run_group(args, env, output, "full", names)
        else:
            run_memcheck_groups(args, env, output, names, groups, isolated_groups, summary["groups"])
        summary["status"] = "PASS"
        if groups is not None:
            summary["instrumentation_verified"] = True
    except (OSError, RuntimeError, ValueError, ET.ParseError, subprocess.SubprocessError) as error:
        summary["error"] = str(error)
    finally:
        (output / name).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if summary["status"] == "PASS" else 1
