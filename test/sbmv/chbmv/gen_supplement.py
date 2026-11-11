#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Generate separate Chbmv coverage; never rewrite the task CSV or GPU baseline.

Only the paired random-coverage stratum has a 50/50 uniform/Gaussian split.
The task's original CSV and separate edge/special-value strata remain intact.
This tool checks data identities and CSV contracts, not NPU correctness.
"""

import argparse
import csv
import hashlib
import io
import json
import logging
import sys
from pathlib import Path
import random
import struct


ROOT = Path(__file__).resolve().parent
MAIN = ROOT / "arch22" / "chbmv_test.csv"
BASELINE = ROOT / "arch22" / "gpu_baseline.csv"
OUTPUT = ROOT / "arch22" / "chbmv_supplement_test.csv"
SOURCE_SHA = {
    "chbmv_test.csv": "190f8cd07fa4691051280ae421ecb8c721e63dd4abecc519e0beaf35613b658f",
    "gpu_baseline.csv": "dd2cb87a50589bde139bd42014e911eb4c8c05f31aac0fbdfb0633359bad4369",
}
UNIFORM = "RANDOM_NORM_5_5"
GAUSSIAN = "GAUSS_NORM_5_2"
SUCCESS = "ACLBLAS_STATUS_SUCCESS"
INVALID = "ACLBLAS_STATUS_INVALID_VALUE"

# (n, declared k, lda padding). k is never clamped for physical storage:
# UPPER diagonal stays at band row k even where declared k > n - 1.
SHAPES = [
    (1, 0, 0), (2, 1, 0), (3, 7, 1), (7, 11, 4),
    (8, 2, 0), (15, 14, 1), (16, 3, 4), (17, 31, 8),
    (31, 1, 16), (32, 7, 0), (33, 32, 1), (63, 8, 4),
    (64, 63, 8), (65, 16, 0), (127, 31, 4), (128, 127, 1),
    (129, 7, 16), (255, 64, 0), (256, 255, 8), (257, 17, 4),
    (511, 63, 1), (512, 127, 16), (1024, 255, 0), (2048, 64, 4),
]
STRIDES = [(1, 1), (2, -3), (-3, 2), (-1, -2), (3, 3), (-2, -1)]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_originals():
    for path in (MAIN, BASELINE):
        if sha256(path) != SOURCE_SHA[path.name]:
            raise ValueError(f"Original task artifact changed: {path}")
    with MAIN.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        original = list(reader)
    if len(original) != 1200 or sum(r["case_name"].startswith("TC_PF_") for r in original) != 200:
        raise ValueError("Original task CSV must retain 1000 accuracy and 200 PF rows")
    return fields, original


def f32_text(value):
    value = struct.unpack("<f", struct.pack("<f", value))[0]
    return format(value, ".9g")


def scalars(seed, gaussian):
    rng = random.Random(seed)
    values = []
    for _ in range(4):
        if gaussian:
            mu, sigma = rng.uniform(-5.0, 5.0), rng.uniform(0.1, 2.0)
            value = min(5.0, max(-5.0, rng.gauss(mu, sigma)))
        else:
            value = rng.uniform(-5.0, 5.0)
        values.append(f32_text(value))
    return values


def add_row(rows, name, description, **changes):
    row = {
        "case_name": name, "description": description, "uplo": "ACLBLAS_UPPER",
        "n": 17, "k": 3, "alpha_real": "1", "alpha_imag": "0.5",
        "a_fill": UNIFORM, "lda": 4, "x_fill": UNIFORM, "incx": 1,
        "beta_real": "-0.25", "beta_imag": "0.75", "y_fill": UNIFORM, "incy": 1,
        "expect_result": SUCCESS, "mere_threshold": "0.00012207031",
        "mare_multiplier": "10.0", "random_seed": 2026111100 + len(rows),
    }
    row.update(changes)
    rows.append({key: str(value) for key, value in row.items()})


def add_distribution_rows(rows):
    pair = 0
    for shape_index, (n, k, pad) in enumerate(SHAPES):
        ix, iy = STRIDES[shape_index % len(STRIDES)]
        for side in ("UPPER", "LOWER"):
            pair += 1
            for tag, fill in (("U", UNIFORM), ("G", GAUSSIAN)):
                ar, ai, br, bi = scalars(20261111000 + pair * 2 + (tag == "G"), tag == "G")
                add_row(rows, f"TC_SU_DIST_{pair:03d}_{tag}",
                    f"paired_{side}_n{n}_declaredk{k}_pad{pad}_inc{ix}_{iy}_{tag}",
                    uplo="ACLBLAS_" + side, n=n, k=k, lda=k + 1 + pad, incx=ix, incy=iy,
                    a_fill=fill, x_fill=fill, y_fill=fill,
                    alpha_real=ar, alpha_imag=ai, beta_real=br, beta_imag=bi)



def add_special_rows(rows):
    for side in ("UPPER", "LOWER"):
        for label, changes in [
            ("beta0_nan_old_y", {"n": 17, "k": 0, "lda": 1, "y_fill": "VALUE_NORM_NAN",
                                 "beta_real": "0", "beta_imag": "0"}),
            ("beta0_inf_old_y", {"n": 33, "k": 4, "lda": 7, "incy": -3,
                                 "y_fill": "VALUE_NORM_INF", "beta_real": "0", "beta_imag": "0"}),
            ("nan_x_diagonal", {"n": 9, "k": 0, "lda": 3, "incx": -2,
                                "x_fill": "VALUE_NORM_NAN", "beta_real": "0", "beta_imag": "0"}),
            ("inf_x_band", {"n": 17, "k": 1, "lda": 4, "x_fill": "VALUE_NORM_INF",
                            "beta_real": "0", "beta_imag": "0"}),
            ("inf_a_band", {"n": 8, "k": 2, "lda": 5, "a_fill": "VALUE_NORM_INF",
                            "beta_real": "0", "beta_imag": "0"}),
            ("nan_a_band", {"n": 8, "k": 2, "lda": 5, "a_fill": "VALUE_NORM_NAN",
                            "beta_real": "0", "beta_imag": "0"}),
            ("alpha0_scale_finite_y", {"n": 7, "k": 6, "lda": 9, "incx": -1, "incy": -2,
                                       "alpha_real": "0", "alpha_imag": "0"}),
            ("alpha0_beta0_nan_old_y", {"n": 19, "k": 3, "lda": 6, "y_fill": "VALUE_NORM_NAN",
                                       "alpha_real": "0", "alpha_imag": "0",
                                       "beta_real": "0", "beta_imag": "0"}),
            ("beta1_nan_old_y", {"y_fill": "VALUE_NORM_NAN", "beta_real": "1", "beta_imag": "0"}),
        ]:
            add_row(rows, f"TC_SU_SPECIAL_{side}_{label}", label, uplo="ACLBLAS_" + side, **changes)



def add_edge_rows(rows):
    edges = [
        ("n0_invalid_uplo", {"n": 0, "uplo": "999"}, SUCCESS),
        ("n0_negative_k", {"n": 0, "k": -1}, SUCCESS),
        ("n0_invalid_lda", {"n": 0, "lda": 0}, SUCCESS),
        ("n0_zero_incx", {"n": 0, "incx": 0}, SUCCESS),
        ("n0_intmin_incy", {"n": 0, "incy": -2147483648}, SUCCESS),
        ("n0_all_null", {"n": 0, "alpha_real": "null", "beta_real": "null",
                          "a_fill": "NULLPTR", "x_fill": "NULLPTR", "y_fill": "NULLPTR"}, SUCCESS),
        ("n0_all_invalid", {"n": 0, "uplo": "999", "k": -1, "lda": 0, "incx": 0,
                             "incy": -2147483648, "alpha_real": "null", "beta_real": "null",
                             "a_fill": "NULLPTR", "x_fill": "NULLPTR", "y_fill": "NULLPTR"}, SUCCESS),
        ("n0_intmax_k", {"n": 0, "k": 2147483647, "lda": 0}, SUCCESS),
        ("negative_n_precedence", {"n": -1, "uplo": "999", "k": -1, "lda": 0,
                                   "incx": 0, "incy": 0}, INVALID),
        ("n_intmin", {"n": -2147483648}, INVALID),
        ("k_intmax_lda_guard", {"n": 1, "k": 2147483647, "lda": 2147483647}, INVALID),
        ("incx_intmin_n1", {"n": 1, "incx": -2147483648}, INVALID),
        ("incy_intmin_n1", {"n": 1, "incy": -2147483648}, INVALID),
        ("incx_span_int32_plus1", {"n": 3, "incx": 1073741824}, INVALID),
        ("incy_negative_span_int32_plus1", {"n": 3, "incy": -1073741824}, INVALID),
        ("beta0_null_y_is_invalid", {"beta_real": "0", "beta_imag": "0", "y_fill": "NULLPTR"}, INVALID),
        ("alpha0_null_a_is_invalid", {"alpha_real": "0", "alpha_imag": "0", "a_fill": "NULLPTR"}, INVALID),
    ]
    for label, changes, expected in edges:
        add_row(rows, "TC_SU_EDGE_" + label, label, expect_result=expected, **changes)


def build_rows():
    rows = []
    add_distribution_rows(rows)
    add_special_rows(rows)
    add_edge_rows(rows)
    return rows


def validate_positive_dimensions(row):
    n, k, lda, ix, iy = (int(row[key]) for key in ("n", "k", "lda", "incx", "incy"))
    if row["expect_result"] != SUCCESS or n <= 0:
        return
    dimensions_valid = (k >= 0, lda >= k + 1, ix != 0, iy != 0)
    if not all(dimensions_valid):
        raise ValueError("Positive success case has invalid dimensions/strides")
    for inc in (ix, iy):
        if abs(inc) == 2147483648 or (n - 1) * abs(inc) > 2147483647:
            raise ValueError("Positive success case has overflow stride")


def collect_random_pairs(rows, fields):
    pairs = {}
    for row in rows:
        if set(row) != set(fields):
            raise ValueError("Supplement field schema differs from original task CSV")
        if row["case_name"].startswith("TC_SU_DIST_"):
            pair, tag = row["case_name"].rsplit("_", 1)
            pairs.setdefault(pair, {})[tag] = row
        validate_positive_dimensions(row)
        if row["x_fill"] in ("VALUE_NORM_INF", "VALUE_NORM_NAN"):
            if float(row["alpha_real"]) == float(row["alpha_imag"]) == 0.0:
                raise ValueError("Task excludes alpha=0 with Inf/NaN x")
    return pairs


def accumulate_pair_budget(totals, tag, row):
    budget = totals.get(tag)
    if budget is None:
        raise ValueError("Random pair tag must be U or G")
    fill = UNIFORM if tag == "U" else GAUSSIAN
    if any(row[key] != fill for key in ("a_fill", "x_fill", "y_fill")):
        raise ValueError("Random tensor distributions are not consistently paired")
    n = int(row["n"])
    increments = {"a": int(row["lda"]) * n,
                  "x": (n - 1) * abs(int(row["incx"])) + 1,
                  "y": (n - 1) * abs(int(row["incy"])) + 1,
                  "alpha_components": 2, "beta_components": 2}
    for key, value in increments.items():
        budget[key] = budget.get(key, 0) + value


def paired_budgets(pairs):
    totals = {"U": {"a": 0, "x": 0, "y": 0, "alpha_components": 0, "beta_components": 0},
              "G": {"a": 0, "x": 0, "y": 0, "alpha_components": 0, "beta_components": 0}}
    for pair in pairs.values():
        if set(pair) != {"U", "G"}:
            raise ValueError("Every random shape requires one uniform and one Gaussian row")
        uniform, gaussian = pair.get("U"), pair.get("G")
        for key in ("uplo", "n", "k", "lda", "incx", "incy"):
            if uniform.get(key) != gaussian.get(key):
                raise ValueError("Pair changes physical shape")
        for tag, row in pair.items():
            accumulate_pair_budget(totals, tag, row)
    if totals.get("U") != totals.get("G"):
        raise ValueError("Uniform/Gaussian physical and scalar sample budgets differ")
    return totals


def validate(rows, fields, original):
    names = [row["case_name"] for row in rows]
    if len(names) != len(set(names)) or set(names).intersection(row["case_name"] for row in original):
        raise ValueError("Supplement case names must be unique and disjoint from original names")
    pairs = collect_random_pairs(rows, fields)
    totals = paired_budgets(pairs)
    return {
        "supplement_rows": len(rows), "random_pairs": len(pairs), "random_rows": 2 * len(pairs),
        "special_rows": sum(name.startswith("TC_SU_SPECIAL_") for name in names),
        "edge_rows": sum(name.startswith("TC_SU_EDGE_") for name in names),
        "paired_physical_complex_counts_and_scalar_components": totals,
        "original_task_rows": len(original), "original_sha256": SOURCE_SHA,
        "qualification": "STATIC_CSV_CHECK_ONLY_NOT_NPU_PASS",
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify existing CSV without writing")
    args = parser.parse_args()
    fields, original = check_originals()
    rows = build_rows()
    summary = validate(rows, fields, original)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    encoded = buffer.getvalue().encode("utf-8")
    if args.check:
        if OUTPUT.read_bytes() != encoded:
            raise ValueError("Existing supplement CSV does not match deterministic generator")
    else:
        OUTPUT.write_bytes(encoded)
    check_originals()
    summary["supplement_sha256"] = hashlib.sha256(encoded).hexdigest()
    logging.info("%s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
