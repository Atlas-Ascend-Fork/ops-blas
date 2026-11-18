# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Shared CSV, XML and process checks for Chemv test gates."""
import csv
from collections import Counter
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
CSV_COLUMNS = (
    "case_name", "description", "uplo", "n", "alpha_real", "alpha_imag", "a_fill", "lda", "x_fill", "incx",
    "beta_real", "beta_imag", "y_fill", "incy", "expect_result", "mere_threshold", "mare_multiplier", "random_seed")
BASELINE_COLUMNS = (
    "id", "uplo", "n", "alpha_real", "alpha_imag", "beta_real", "beta_imag", "incx", "incy", "lda", "gpu_ms")
SUPPLEMENTAL = ("NullHandlePrecedesDimensions", "RandomScalarDistributions")


def read_table(path, columns):
    with Path(path).open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source, strict=True)
        if reader.fieldnames != list(columns):
            raise ValueError("Unexpected CSV header: " + str(path))
        try:
            rows = list(reader)
        except csv.Error as error:
            raise ValueError("Malformed CSV: " + str(path)) from error
    if any(set(row) != set(columns) or any(value is None or not value.strip() for value in row.values())
           for row in rows):
        raise ValueError("Malformed or empty CSV fields: " + str(path))
    return rows


def performance_key(row):
    scalars = tuple(float(row[name]) for name in ("alpha_real", "alpha_imag", "beta_real", "beta_imag"))
    if not all(math.isfinite(value) for value in scalars):
        raise ValueError("Nonfinite performance scalar")
    return (row["uplo"].replace("ACLBLAS_", "", 1), int(row["n"]), scalars,
            (int(row["incx"]), int(row["incy"]), int(row["lda"])))


def read_cases(csv_path=HERE / "arch35/chemv_test.csv", baseline_path=HERE / "arch35/gpu_baseline.csv"):
    rows = read_table(csv_path, CSV_COLUMNS)
    names = [row["case_name"] for row in rows]
    if any(re.fullmatch(r"[A-Za-z0-9_]+", name) is None for name in names) or len(set(names)) != len(names):
        raise ValueError("Empty, invalid or duplicate case_name")
    perf = [row for row in rows if row["case_name"].startswith("TC_PF_")]
    if (len(rows), len(rows) - len(perf), len(perf)) != (1200, 1000, 200):
        raise ValueError("Chemv CSV requires 1200 cases: 1000 accuracy + 200 TC_PF_")
    baseline = read_table(baseline_path, BASELINE_COLUMNS)
    if len(baseline) != 200 or len({row["id"] for row in baseline}) != 200:
        raise ValueError("GPU baseline requires 200 uniquely named rows")
    if any(not math.isfinite(float(row["gpu_ms"])) or float(row["gpu_ms"]) <= 0 for row in baseline):
        raise ValueError("GPU baseline duration must be finite and positive")
    if Counter(map(performance_key, perf)) != Counter(map(performance_key, baseline)):
        raise ValueError("Performance CSV and GPU baseline inputs do not match")
    return rows


def needs_computation(row):
    if row["expect_result"] != "ACLBLAS_STATUS_SUCCESS" or int(row["n"]) == 0:
        return False
    alpha_zero = float(row["alpha_real"]) == 0 and float(row["alpha_imag"]) == 0
    beta_one = float(row["beta_real"]) == 1 and float(row["beta_imag"]) == 0
    return not (alpha_zero and beta_one)


def case_test_name(name):
    return "Chemv/ChemvArch35Test.CsvDriven/" + name


def supplemental_names():
    return ["ChemvArch35Test." + name for name in SUPPLEMENTAL]


def validate_xml(path, expected):
    root = ET.parse(path).getroot()
    if int(root.get("tests", -1)) != len(expected):
        raise ValueError("GTest XML count differs from selected cases")
    if any(int(root.get(key, -1)) != 0 for key in ("failures", "disabled", "errors")):
        raise ValueError("GTest XML reports failures, disabled tests or errors")
    seen = []
    for item in root.iter("testcase"):
        if item.get("status") != "run" or item.get("result") != "completed" or len(item):
            raise ValueError("A selected test failed, skipped or did not complete")
        seen.append(item.get("classname", "") + "." + item.get("name", ""))
    if Counter(seen) != Counter(expected):
        raise ValueError("GTest executed names differ from selected cases")
    return len(seen)


def test_environment(device):
    env = os.environ.copy()
    for key in tuple(env):
        if key.startswith("GTEST_") or key in ("CHEMV_PROFILE", "CHEMV_DUMP_DIR"):
            env.pop(key)
    env["BLAS_DEVICE_ID"] = str(device)
    env["ASCEND_DEVICE_ID"] = str(device)
    return env


def executable(path):
    resolved = shutil.which(str(path))
    if resolved is None:
        raise FileNotFoundError("Executable not found: " + str(path))
    return str(Path(resolved).resolve(strict=True))


def run_logged(command, log, env):
    with log.open("w", encoding="utf-8") as output:
        return subprocess.run(command, env=env, stdout=output, stderr=subprocess.STDOUT,
                              check=False, timeout=3600).returncode


def save_summary(output, summary):
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
