# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Evidence and completeness checks shared by the SGEMM test gates."""
import csv
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

ARCH = Path(__file__).resolve().parent / "arch22"
COUNTS = {
    "sgemm_grouped_batched_test.csv": (50, 0),
    "sgemm_grouped_batched_test_full.csv": (1000, 200),
    "supplemental.csv": (306, 0),
    "review_regression.csv": (3, 0),
}


def read_cases(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    names = [row["case_name"] for row in rows]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise RuntimeError(f"Empty or duplicate case IDs: {path}")
    return rows


def audit_csvs():
    evidence = {}
    for name, (accuracy, performance) in COUNTS.items():
        path = ARCH / name
        rows = read_cases(path)
        pf = sum(row["case_name"].startswith("TC_PF_") for row in rows)
        if len(rows) != accuracy + performance or pf != performance:
            raise RuntimeError(f"CSV count mismatch: {name}: total={len(rows)}, PF={pf}")
        evidence[name] = {"accuracy": len(rows) - pf, "performance": pf,
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    if read_cases(ARCH / "sgemm_grouped_batched_test.csv") != read_cases(
            ARCH / "sgemm_grouped_batched_test_full.csv")[:50]:
        raise RuntimeError("Default smoke CSV is not the original first 50 cases")
    return evidence


def environment(csv_path, device, jsonl):
    env = os.environ.copy()
    for key in ("GB_INCLUDE_PERF_ACCURACY", "GTEST_FILTER", "GTEST_TOTAL_SHARDS", "GTEST_SHARD_INDEX"):
        env.pop(key, None)
    env.update(GB_CASES_CSV=str(csv_path), GB_DEVICE=str(device), GB_JSONL=str(jsonl))
    return env


def list_tests(binary, env, pattern="*"):
    result = subprocess.run([str(binary), "--gtest_list_tests", f"--gtest_filter={pattern}"],
                            env=env, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=True)
    suite, names = "", []
    for line in result.stdout.splitlines():
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


def check_mapping(names, csv_path):
    expected = {row["case_name"] for row in read_cases(csv_path)}
    mapped = [name.rsplit("/", 1)[-1] for name in names if name.startswith("TaskCSV/")]
    if len(mapped) != len(expected) or set(mapped) != expected:
        raise RuntimeError(f"GTest/CSV mapping mismatch: {csv_path}")
    review = {row["case_name"] for row in read_cases(ARCH / "review_regression.csv")}
    found = {name.rsplit("/", 1)[-1] for name in names if name.startswith("Review/")}
    if found != review:
        raise RuntimeError("Mandatory TN96 golden tests missing")


def check_xml(path, expected):
    tests = list(ET.parse(path).getroot().iter("testcase"))
    actual = [node.get("classname", "") + "." + node.get("name", "") for node in tests]
    failures = []
    for name, node in zip(actual, tests):
        has_issue = any(node.find(tag) is not None for tag in ("failure", "error", "skipped"))
        if has_issue or node.get("status") == "notrun" or node.get("result") in ("skipped", "suppressed"):
            failures.append(name)
    if len(actual) != len(expected) or set(actual) != set(expected) or failures:
        raise RuntimeError(f"GTest execution incomplete/failed: expected={len(expected)}, "
                           f"actual={len(actual)}, failures/skips={failures}")
    return {"discovered": len(expected), "executed": len(actual), "failed_or_skipped": 0}


def check_jsonl(path, expected):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    wanted = Counter(name.rsplit("/", 1)[-1] for name in expected
                     if ".CSVAccuracy/" in name or ".CSVKernelTiming/" in name)
    actual = Counter(row["case_name"] for row in rows)
    if wanted != actual or any(row["status"] not in ("PASS", "MEASURED_NOT_ACCEPTED") for row in rows):
        raise RuntimeError("JSONL missing, duplicated, skipped, CPU-mocked or failed cases")
    return len(rows)


def write_summary(directory, name, data):
    (directory / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def identity(binary):
    return {"binary": str(binary), "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            "ld_library_path": os.environ.get("LD_LIBRARY_PATH", ""), "cpu_mock_accepted": False}
