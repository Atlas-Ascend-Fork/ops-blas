#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Pair original random tensor draws with equal-budget Gaussian draws; no NPU execution."""
import argparse
import csv
import hashlib
import io
import json
import logging
import sys
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
CASES = ROOT / "arch22"
UNIFORM, GAUSSIAN = "RANDOM_NORM_5_5", "GAUSS_NORM_5_2"
SUCCESS = "ACLBLAS_STATUS_SUCCESS"
SEED_OFFSET = 1000000000
SOURCE_SHA = {
    "chbmv_test.csv": "190f8cd07fa4691051280ae421ecb8c721e63dd4abecc519e0beaf35613b658f",
    "chbmv_supplement_test.csv": "69ac4d00bc155224313a4a2672d929c13b354169ec3ac5125603c70a8fd1c347",
    "gpu_baseline.csv": "dd2cb87a50589bde139bd42014e911eb4c8c05f31aac0fbdfb0633359bad4369",
}
ORIGIN = {
    "official_task_zip_sha256": "152a3b97380f893a925ae638bf5b4fedad1afd9ec7fce477c80261f2ab98abf2",
    "official_gen_csv_py_sha256": "69e262093a8729894b46bd19ea572b02624628dd3dd0571a40147a4fd1519388",
    "official_random_methods": ["randrange"],
    "official_uniform_or_gaussian_scalar_calls": 0,
    "official_scalar_origin": "All original alpha/beta values are fixed SCALARS/literals/null; never random.uniform",
    "official_scalar_functions": {
        "TC_AB": "Gen.l2_scalar / fixed SCALARS", "TC_EX": "Gen.ex_extend / fixed SCALARS",
        "TC_PF": "Gen.perf / fixed alpha=(1,0), beta=(0,0)",
        "others": "Gen.l0_basic/l1_size/l1b_band/l4_ld_inc/l5_fill/l5b_cover/l6_edge / fixed literals",
    },
    "supplement_random_scalar_origin": "Only TC_SU_DIST_*_U/G, from gen_supplement.scalars; 48 pairs",
    "mirror_scalar_rule": "Preserve every fixed scalar exactly. Selected source rows contain no random scalar draw.",
    "scalar_gaussian_draws_added_by_this_mirror": 0,
}
TENSORS = {"A": ("a_fill", 4000), "x": ("x_fill", 1), "y": ("y_fill", 2)}
SCALARS = ("alpha_real", "alpha_imag", "beta_real", "beta_imag")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def required(mapping, key):
    value = mapping.get(key)
    if value is None:
        raise ValueError("Missing required field: " + key)
    return value


def originals():
    for name, expected in SOURCE_SHA.items():
        if digest(CASES / name) != expected:
            raise ValueError("Original artifact changed: " + name)
    result = {}
    fields = None
    for name in ("chbmv_test.csv", "chbmv_supplement_test.csv"):
        with (CASES / name).open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if fields is not None and fields != reader.fieldnames:
                raise ValueError("Source CSV schemas differ")
            fields = reader.fieldnames
            result[name] = list(reader)
    if len(result.get("chbmv_test.csv", [])) != 1200:
        raise ValueError("Original registry must remain 1200 + 131")
    if len(result.get("chbmv_supplement_test.csv", [])) != 131:
        raise ValueError("Original registry must remain 1200 + 131")
    return fields, result


def numeric(row):
    # Positive alpha=0,beta=1 compares numerically but need not launch a kernel.
    return int(row["n"]) > 0 and row["expect_result"] == SUCCESS


def scalar_origin(row):
    if re.fullmatch(r"TC_SU_DIST_[0-9]+_[UG]", row["case_name"]):
        return "uniform" if row["case_name"].endswith("_U") else "gaussian"
    return "fixed_nonrandom"


def budgets(row, parameter):
    n, k, lda = int(row["n"]), int(row["k"]), int(row["lda"])
    alpha_nonzero = float(row["alpha_real"]) != 0 or float(row["alpha_imag"]) != 0
    beta_nonzero = float(row["beta_real"]) != 0 or float(row["beta_imag"]) != 0
    if parameter == "A":
        effective = min(k, n - 1)
        off = n * effective - effective * (effective + 1) // 2
        return {"generated_real": lda * n, "generated_imag": lda * n,
                "physical_complex_span": lda * n,
                "mathematically_referenced_real": n + off if alpha_nonzero else 0,
                "mathematically_referenced_imag": off if alpha_nonzero else 0}
    inc = abs(int(row["incx" if parameter == "x" else "incy"]))
    used = alpha_nonzero if parameter == "x" else beta_nonzero
    # Stride gaps are fixed sentinels, not random draws.
    return {"generated_real": n, "generated_imag": n,
            "physical_complex_span": (n - 1) * inc + 1,
            "mathematically_referenced_real": n if used else 0,
            "mathematically_referenced_imag": n if used else 0}


def tensor_kind(token):
    return {UNIFORM: "uniform", GAUSSIAN: "gaussian"}.get(token, "fixed_or_special")


def empty_tensor_totals():
    metrics = ("generated_real", "generated_imag", "mathematically_referenced_real",
               "mathematically_referenced_imag")
    total = {}
    for parameter in TENSORS:
        kinds = {}
        for kind in ("uniform", "gaussian", "fixed_or_special"):
            kinds[kind] = dict.fromkeys(metrics, 0)
        total[parameter] = kinds
    return total


def global_accounting(rows):
    total = empty_tensor_totals()
    scalar = {}
    for parameter in ("alpha", "beta"):
        scalar[parameter] = dict.fromkeys(("uniform", "gaussian", "fixed_nonrandom"), 0)
    for row in rows:
        if not numeric(row):
            continue
        for parameter, (field, _) in TENSORS.items():
            budget = budgets(row, parameter)
            kind = tensor_kind(row[field])
            target = required(required(total, parameter), kind)
            for metric in target:
                target[metric] = required(target, metric) + required(budget, metric)
        for parameter in scalar:
            target = required(scalar, parameter)
            origin = scalar_origin(row)
            target[origin] = required(target, origin) + 2
    return {"tensor_components": total, "scalar_components": scalar,
            "tensor_uniform_gaussian_equal_by_parameter": {
                name: required(kinds, "uniform") == required(kinds, "gaussian") for name, kinds in total.items()},
            "random_scalar_uniform_gaussian_equal_by_parameter": {
                name: required(kinds, "uniform") == required(kinds, "gaussian") for name, kinds in scalar.items()},
            "fixed_special_and_no_numeric_rows_are_separate_strata": True,
            "all_scalar_values_including_fixed_claimed_50_50": False,
            "unit": "unique generated input components per numeric comparison, not repeated PF launches",
            "matrix_diagonal_imaginary_and_padding_draws": (
                "generated counts include them; mathematical counts exclude them")}


def skip_reason(filename, row):
    if not numeric(row):
        return "no_Golden_Compare_branch_n0_or_expected_error"
    if filename == "chbmv_supplement_test.csv" and row["case_name"].startswith("TC_SU_DIST_"):
        return "existing_48_U_G_pairs_retained_without_extra_mirroring"
    if not any(row[field] == UNIFORM for field, _ in TENSORS.values()):
        return "no_uniform_tensor_draw"
    return None


def select_sources(source):
    selected, skipped = [], []
    for filename, rows in source.items():
        for row in rows:
            reason = skip_reason(filename, row)
            if reason:
                skipped.append({"source_csv": filename, "source_case": row["case_name"], "reason": reason})
            else:
                selected.append((filename, row))
    return selected, skipped


def transform_tensors(original, row, source_seed, seed):
    transformed = {}
    for parameter, (field, offset) in TENSORS.items():
        if original[field] == UNIFORM:
            row[field] = GAUSSIAN
        transformed[parameter] = {
            "from": original[field], "to": row[field], "budget_each_partner": budgets(original, parameter),
            "source_real_seed": source_seed + offset, "source_imag_seed": source_seed + offset + 1000,
            "mirror_real_seed": seed + offset, "mirror_imag_seed": seed + offset + 1000}
    return transformed


def validate_fixed_fields(fields, original, row):
    if "TC_PF_" in row["case_name"]:
        raise ValueError("Mirror name must not match the formal PF filter")
    for field in SCALARS:
        if row[field] != original[field]:
            raise ValueError("Fixed scalar unexpectedly changed")
    changed_fields = {"case_name", "description", "random_seed", "a_fill", "x_fill", "y_fill"}
    for field in fields:
        if field not in changed_fields and row[field] != original[field]:
            raise ValueError("Geometry/fixed parameters changed: " + field)


def fixed_scalar_parameters(original, row):
    parameters = {}
    for name in ("alpha", "beta"):
        parameters[name] = {
            "source_values": [original[name + "_real"], original[name + "_imag"]],
            "mirror_values": [row[name + "_real"], row[name + "_imag"]],
            "origin": "fixed_nonrandom", "new_random_components": 0}
    return parameters


def make_mirror(filename, original, fields):
    row = dict(original)
    row["case_name"] = "TC_GS_FROM_" + original["case_name"][3:]
    row["description"] = "gaussian_mirror_of_" + original["case_name"]
    source_seed = int(original["random_seed"])
    seed = source_seed + SEED_OFFSET
    if seed + 5000 >= 2 ** 32:
        raise ValueError("Seed exceeds the actual uint32 generator domain")
    row["random_seed"] = str(seed)
    transformed = transform_tensors(original, row, source_seed, seed)
    validate_fixed_fields(fields, original, row)
    scalar_values = [float(original[field]) for field in SCALARS]
    mapping = {"source_csv": filename, "source_case": original["case_name"],
               "mirror_case": row["case_name"], "source_seed": source_seed, "mirror_seed": seed,
               "numeric_compare_but_may_quick_return": scalar_values == [0, 0, 1, 0],
               "tensor_parameters": transformed, "scalar_parameters": fixed_scalar_parameters(original, row),
               "source_parameters": original, "mirror_parameters": row}
    return row, mapping


def flattened_sources(source):
    rows = []
    for group in source.values():
        rows.extend(group)
    return rows


def validate_combined(all_source, mirrors):
    combined = all_source + mirrors
    names = [row["case_name"] for row in combined]
    seeds = [int(row["random_seed"]) for row in combined]
    if len(set(names)) != len(names) or len(set(seeds)) != len(seeds):
        raise ValueError("Case names/seeds collide")
    after = global_accounting(combined)
    if not all(after["tensor_uniform_gaussian_equal_by_parameter"].values()):
        raise ValueError("Aggregate random tensor component budgets are not balanced")
    if not all(after["random_scalar_uniform_gaussian_equal_by_parameter"].values()):
        raise ValueError("Existing random scalar budgets are not balanced")
    return after


def selected_source_counts(source, selected):
    counts = dict.fromkeys(source, 0)
    for filename, _ in selected:
        counts[filename] = counts.get(filename, 0) + 1
    return counts


def build_registry(source, selected, mirrors, after):
    all_source = flattened_sources(source)
    combined = all_source + mirrors
    directed = re.findall(r"TEST_F\(ChbmvTest,\s*(\w+)\)",
                          (CASES / "chbmv_test.cpp").read_text(encoding="utf-8"))
    if len(directed) != len(set(directed)):
        raise ValueError("Duplicated directed UT names")
    return {"status": "GENERATED_COVERAGE_NOT_NPU_EXECUTED", "source_sha256": SOURCE_SHA,
            "original_main_rows": len(source.get("chbmv_test.csv", [])),
            "original_supplement_rows": len(source.get("chbmv_supplement_test.csv", [])),
            "gaussian_mirror_rows": len(mirrors), "csv_total": len(combined),
            "numeric_compare_csv_rows": sum(numeric(row) for row in combined),
            "original_pf_rows": sum(row["case_name"].startswith("TC_PF_") for row in all_source),
            "gaussian_mirrors_from_original_pf_are_accuracy_only": sum(
                row["case_name"].startswith("TC_GS_FROM_PF_") for row in mirrors),
            "directed_ut_names": ["ChbmvTest." + name for name in directed],
            "directed_ut_count": len(directed), "all_registered_gtest_count": len(combined) + len(directed),
            "mirror_source_counts": selected_source_counts(source, selected),
            "optional_csv_omitted_registry": {
                "gaussian_absent_main_and_supplement_present": len(all_source) + len(directed)},
            "scalar_origin_facts": ORIGIN, "before": global_accounting(all_source), "after": after}


def encode_products(fields, mirrors, registry, mapping, skipped):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(mirrors)
    csv_bytes = buffer.getvalue().encode("utf-8")
    registry["gaussian_csv_sha256"] = hashlib.sha256(csv_bytes).hexdigest()
    details = {"source_sha256": SOURCE_SHA, "scalar_origin_facts": ORIGIN,
               "seed_rule": "mirror_seed=source_seed+1000000000; tensor real/imag use independent seed offsets",
               "all_fixed_parameters_preserved": True, "case_mappings": mapping, "skipped_sources": skipped}
    return {
        "chbmv_gaussian_mirror_test.csv": csv_bytes,
        "chbmv_gaussian_mirror_mapping.json": json.dumps(details, ensure_ascii=False, indent=2).encode("utf-8") + b"\n",
        "chbmv_case_registry.json": json.dumps(registry, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"}


def generate():
    fields, source = originals()
    selected, skipped = select_sources(source)
    mirrors, mapping = [], []
    for filename, original in selected:
        row, entry = make_mirror(filename, original, fields)
        mirrors.append(row)
        mapping.append(entry)
    after = validate_combined(flattened_sources(source), mirrors)
    registry = build_registry(source, selected, mirrors, after)
    return encode_products(fields, mirrors, registry, mapping, skipped), registry


def report_summary(registry):
    fields = ("status", "mirror_source_counts", "gaussian_mirror_rows", "csv_total", "numeric_compare_csv_rows",
              "original_pf_rows", "directed_ut_count", "all_registered_gtest_count", "gaussian_csv_sha256")
    summary = {}
    for field in fields:
        summary[field] = required(registry, field)
    logging.info("%s", json.dumps(summary, indent=2))
    after = required(registry, "after")
    logging.info("%s", json.dumps({
        "after_random_tensor_50_50": required(after, "tensor_uniform_gaussian_equal_by_parameter"),
        "after_random_scalar_50_50": required(after, "random_scalar_uniform_gaussian_equal_by_parameter"),
        "all_fixed_scalar_values_claimed_50_50": False,
        "new_gaussian_scalar_draws": required(ORIGIN, "scalar_gaussian_draws_added_by_this_mirror")}))


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify deterministic existing outputs; write nothing")
    parser.add_argument("--mapping-report", type=Path,
                        help="Optionally write/check the full per-case mapping at this path")
    args = parser.parse_args()
    products, registry = generate()
    for name, payload in products.items():
        if name == "chbmv_gaussian_mirror_mapping.json":
            if args.mapping_report is None:
                continue
            path = args.mapping_report
        else:
            path = CASES / name
        if args.check:
            if not path.exists() or path.read_bytes() != payload:
                raise ValueError("Deterministic output differs: " + name)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
    originals()
    report_summary(registry)


if __name__ == "__main__":
    main()
