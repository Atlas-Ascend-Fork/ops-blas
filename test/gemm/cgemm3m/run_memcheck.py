#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Run uninstrumented 950 quick diagnostics; results do not certify full memory safety."""

import json
import subprocess
import xml.etree.ElementTree as ET

from test_gate import (CSV_PREFIX, audit_csv, check_mapping, discover, environment, file_sha,
                       parse_args, run_group)

SMOKE = ("TC_SQ_039", "TC_PF_1001", "TC_PF_1200")
CRITICAL = ("TC_SQ_045", "TC_PF_1052")
KERNELS_PF1052 = (
    "cgemm3m_pack_kernel",
    "cgemm3m_permute_kernel",
    "cgemm3m_prefix_recompute_kernel",
)


def validate_build(binary):
    cache = next((root / "CMakeCache.txt" for root in binary.parents
                  if (root / "CMakeCache.txt").is_file()), None)
    if cache is None:
        raise RuntimeError("Missing CMakeCache.txt for sanitizer binary")
    lines = cache.read_text(encoding="utf-8", errors="replace").splitlines()
    build_type = next((line for line in lines if line.startswith("CMAKE_BUILD_TYPE:")), "")
    base = next((line for line in lines if line.startswith("CMAKE_ASC_FLAGS:")), "")
    diagnostic = next((line for line in lines if line.startswith("CMAKE_ASC_FLAGS_RELWITHDEBINFO:")), "")
    flags = base + " " + diagnostic
    if not build_type.endswith("=RelWithDebInfo") or "-O2" not in flags or "-g" not in flags:
        raise RuntimeError("Ascend 950 quick memcheck requires an O2/-g ASC diagnostic build")
    if "--cce-enable-sanitizer" in flags or "--sanitizer" in flags or "-O0" in flags:
        raise RuntimeError("Ascend 950 does not support sanitizer or O0 compilation flags")
    return {"cmake_cache": str(cache), "cmake_cache_sha256": file_sha(cache), "asc_flags": flags,
            "mode": "Ascend 950 quick check without sanitizer compilation flags",
            "compiler_instrumented": False, "memory_safety_certified": False,
            "limit": "This mode does not inspect all instructions"}


def record_group(groups, name, operation):
    groups[name] = {"status": "RUNNING"}
    try:
        result = operation()
    except (OSError, RuntimeError, ValueError, ET.ParseError, subprocess.SubprocessError) as error:
        groups[name] = {"status": "FAIL", "error": str(error)}
        raise
    result["status"] = "PASS_UNINSTRUMENTED"
    groups[name] = result
    return result


def planned_groups():
    names = ["l0", "ed", *(case.lower() for case in SMOKE), "sq045_path"]
    names.extend("pf1052_" + kernel.removeprefix("cgemm3m_") for kernel in KERNELS_PF1052)
    names.append("pf1052_pack_tail")
    for case in CRITICAL:
        names.append(case.lower() + "_select_all")
    return {name: {"status": "NOT_RUN"} for name in names}


def run_case(args, env, output, names, case, stem, options, summary):
    name = CSV_PREFIX + case
    if names.count(name) != 1:
        raise RuntimeError(f"Missing or duplicate GTest case: {case}")
    result = record_group(summary["groups"], stem, lambda: run_group(
        args, env, output, stem, [name], memcheck=True, sanitizer_options=options))
    result["case"] = case
    return result


def run_critical(args, env, output, names, summary):
    cache = ("--cache-size=128",)
    summary["groups"]["sq045_path"] = run_case(
        args, env, output, names, "TC_SQ_045", "sq045_path",
        (*cache, "--kernel-name=cgemm3m_path_kernel"), summary)
    for kernel in KERNELS_PF1052:
        stem = "pf1052_" + kernel.removeprefix("cgemm3m_")
        summary["groups"][stem] = run_case(
            args, env, output, names, "TC_PF_1052", stem,
            (*cache, "--kernel-name=" + kernel), summary)
    summary["groups"]["pf1052_pack_tail"] = run_case(
        args, env, output, names, "TC_PF_1052", "pf1052_pack_tail",
        ("--cache-size=1024", "--kernel-name=cgemm3m_pack_kernel", "--block-id=24"), summary)
    for case in CRITICAL:
        # Omitting --block-id checks every block. Keep total tool cache below its 24 GiB limit.
        stem = case.lower() + "_select_all"
        run_case(args, env, output, names, case, stem,
                 (*cache, "--kernel-name=cgemm3m_select_kernel"), summary)


def main():
    args = parse_args(__doc__, memcheck=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    summary = {"status": "FAIL", "compiler_instrumented": False, "memory_safety_certified": False,
               "device": args.device, "groups": planned_groups(),
               "tool_limitation": "Not established; incomplete records alone do not prove a tool defect",
               "scope": {"l0_ed": "all cases", "sq_pf": "representative cases",
                         "smoke_cases": SMOKE, "critical_cases": CRITICAL,
                         "select_blocks": "all blocks (tool default; --block-id omitted)"}}
    try:
        binary = args.binary.resolve()
        summary.update(binary=str(binary), binary_sha256=file_sha(binary),
                       memcheck_build=validate_build(binary))
        cases, summary["csv"] = audit_csv()
        env, summary["reference_libraries_sha256"] = environment(args.device, args.reference.resolve())
        names = discover(binary, env, output, args.timeout)
        summary["discovery"] = check_mapping(names, cases)
        for category, count in (("L0", 18), ("ED", 32)):
            selected = [name for name in names if name.startswith(CSV_PREFIX + f"TC_{category}_")]
            if len(selected) != count:
                raise RuntimeError(f"{category} contract count mismatch: {len(selected)} != {count}")
            record_group(summary["groups"], category.lower(), lambda: run_group(
                args, env, output, category.lower(), selected, memcheck=True))
        for case in SMOKE:
            summary["groups"][case.lower()] = run_case(
                args, env, output, names, case, case.lower(), ("--cache-size=128",), summary)
        run_critical(args, env, output, names, summary)
        summary["status"] = "PASS_UNINSTRUMENTED"
    except (OSError, RuntimeError, ValueError, ET.ParseError, subprocess.SubprocessError) as error:
        summary["error"] = str(error)
    finally:
        (output / "memcheck_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if summary["status"] == "PASS_UNINSTRUMENTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
