#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Run HgemmStridedBatched contracts and device paths under mssanitizer memcheck."""

from collections import Counter
from itertools import product
from pathlib import Path
import argparse
import csv
import hashlib
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys

from run_self_test import parse_cases, REPO, TEST_ROOT

SUCCESS = "ACLBLAS_STATUS_SUCCESS"
TRANSPOSES = ("ACLBLAS_OP_N", "ACLBLAS_OP_T", "ACLBLAS_OP_C")
TAIL_SIZES = (65, 127, 129, 255)
GROUPED_SIZES = (8, 16, 32, 64)
# More than eight matrices per core on the task's 28-Cube-core 950PR:
# exercises the eight-slot C ring, delayed Fixpipe, and partial four-batch groups.
GROUPED_BATCHES = 257
LAYOUT_PADDING = 3
BATCH_GAP = 17
SAMPLE_BATCHES = 3
SAMPLE_K = 33
SAMPLE_SEED = 834000
WORKSPACE_ALIGNMENT = 32
WORKSPACE_LD_ALIGNMENT = 8
FLOAT_BYTES = 4
STRIP_COLUMNS = 7
WORKSPACE_COLUMNS = 9  # Two strips (7+2); retain the cBaseBytes tail without many redundant launches.
WORKSPACE_GROUP_BATCHES = 2
TIMEOUT_SECONDS = 3600
KERNELS = {
    "direct": ("hgemm_batch_direct_pipeline",),
    "workspace": ("hgemm_batch_temporary_pipeline", "hgemm_batch_combine"),
    "grouped": ("hgemm_batch_grouped_pipeline",),
    "scale": ("hgemm_batch_combine",),
    "contracts": (),
}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def sample_case(name, shape, transposes, compact=False):
    m, n, k = shape
    ta, tb = transposes
    padding, gap = (0, 0) if compact else (LAYOUT_PADDING, BATCH_GAP)
    lda = max(1, m if ta == TRANSPOSES[0] else k) + padding
    ldb = max(1, k if tb == TRANSPOSES[0] else n) + padding
    ldc = m + padding
    return dict(
        case_name=name, description=name, transA=ta, transB=tb, m=m, n=n, k=k,
        alpha=1, beta=0, lda=lda, ldb=ldb, ldc=ldc,
        strideA=lda * (k if ta == TRANSPOSES[0] else m) + gap,
        strideB=ldb * (n if tb == TRANSPOSES[0] else k) + gap,
        strideC=ldc * n + gap, batchCount=SAMPLE_BATCHES,
        a_fill="UNIFORM", b_fill="UNIFORM", c_fill="UNIFORM",
        expect_result=SUCCESS, random_seed=SAMPLE_SEED,
    )


def workspace_cases(pair, index):
    m, n = TAIL_SIZES[0], WORKSPACE_COLUMNS
    column_bytes = ((m + WORKSPACE_LD_ALIGNMENT - 1) // WORKSPACE_LD_ALIGNMENT
                    * WORKSPACE_LD_ALIGNMENT * FLOAT_BYTES)
    rows = []
    for mode, columns in (("strip", STRIP_COLUMNS), ("group", n * WORKSPACE_GROUP_BATCHES)):
        row = sample_case(f"TC_MC_workspace_{mode}_{index}", (m, n, SAMPLE_K), pair)
        row.update(alpha=0.5, beta=-1, c_offset=1, workspace_offset=1,
                   workspace_bytes=column_bytes * columns + WORKSPACE_ALIGNMENT - 1)
        rows.append(row)
    return rows


def device_cases():
    groups = {name: [] for name in KERNELS if name != "contracts"}
    for index, pair in enumerate(product(TRANSPOSES, repeat=2)):
        for size in TAIL_SIZES:
            groups["direct"].append(sample_case(f"TC_MC_direct_{size}_{index}", (size, size, SAMPLE_K), pair))
        groups["workspace"].extend(workspace_cases(pair, index))
        for size in GROUPED_SIZES:
            row = sample_case(f"TC_MC_grouped_{size}_{index}", (size, size, size), pair, compact=True)
            row["batchCount"] = GROUPED_BATCHES
            groups["grouped"].append(row)
        for mode, k, alpha in (("k_zero", 0, 1), ("alpha_zero", SAMPLE_K, 0)):
            row = sample_case(f"TC_MC_scale_{mode}_{index}", (*TAIL_SIZES[:2], k), pair)
            row.update(alpha=alpha, beta=-1, c_offset=1)
            groups["scale"].append(row)
    return groups


def read_contracts(path):
    contracts = []
    with path.open(encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        for row in reader:
            if row["expect_result"] != SUCCESS or any(int(row[key]) == 0 for key in ("m", "n", "batchCount")):
                contracts.append(row)
    return contracts, fields


def load_plan():
    contracts, fields, sources = [], [], {}
    for name in ("hgemmStridedBatched_test.csv", "hgemmStridedBatched_supplement.csv"):
        path = TEST_ROOT / "arch35" / name
        sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows, columns = read_contracts(path)
        fields = list(dict.fromkeys(fields + columns))
        contracts.extend(rows)
    groups = dict(contracts=contracts, **device_cases())
    names = [row["case_name"] for rows in groups.values() for row in rows]
    if len(names) != len(set(names)) or any(not rows for rows in groups.values()):
        raise ValueError("Empty group or duplicate case name")
    return dict(groups=groups, fields=fields, source_csv_sha256=sources)


def sanitizer_result(text, required, minimum):
    # Full demangling includes quotes and spaces in C++ parameter types.
    start_names = re.findall(r'Start memcheck sanitizer on kernel\s+("[^"\r\n]+"|[^\r\n]+)', text)
    finish_names = re.findall(
        r'Sanitizer finished on kernel\s+("[^"\r\n]+"|[^\r\n]+?)\. No error detected\.', text)
    starts = Counter(name.strip('"') for name in start_names)
    finishes = Counter(name.strip('"') for name in finish_names)
    diagnostics = sanitizer_diagnostics(text)
    counts = {kernel: sum(count for name, count in starts.items() if name.startswith(kernel)) for kernel in required}
    hgemm_count = sum(count for name, count in starts.items() if name.startswith("hgemm_"))
    covered = all(count >= minimum for count in counts.values()) if required else hgemm_count == 0
    return dict(
        passed=(bool(text.strip()) or not required) and starts == finishes and not diagnostics and covered,
        kernel_starts=dict(starts), clean_kernel_finishes=dict(finishes),
        hgemm_kernel_starts=hgemm_count, required_kernel_counts=counts,
        minimum_per_required_kernel=minimum, diagnostics=diagnostics,
    )


def sanitizer_diagnostics(text):
    pattern = r"\b(?:warning|warn|error|fatal)\b|detected memory leaks|[1-9]\d* byte\(s\) leaked"
    return [line for line in text.splitlines()
            if re.search(pattern, line.replace("No error detected.", ""), re.I)]


def wait_process(process, timeout):
    # Kill the whole sanitizer/application process group on timeout, not just the wrapper.
    try:
        return dict(return_code=process.wait(timeout=timeout), timed_out=False)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        return dict(return_code=124, timed_out=True)


def execute(command, env, out, timeout):
    with (out / "process.log").open("w") as log:
        try:
            with subprocess.Popen(command, cwd=out, env=env, stdout=log, stderr=subprocess.STDOUT,
                                  start_new_session=True) as process:
                return wait_process(process, timeout)
        except OSError as exc:
            return dict(return_code=127, timed_out=False, launch_error=str(exc))


def execution_env(args, out):
    env = os.environ.copy()
    for key in ("HGEMM_PROFILE_REPEATS", "HGEMM_DUMP_DIR", "HGEMM_INPUT_DIR", "BLAS_TEST_CASES"):
        env.pop(key, None)
    for key in list(env):
        if key.startswith("GTEST_"):
            env.pop(key)
    env.update(HGEMM_CSV=str(out / "cases.csv"), HGEMM_RESULTS=str(out / "batches.jsonl"),
               HGEMM_DEVICE_MEMORY=args.device_memory, BLAS_DEVICE_ID=str(args.device))
    return env


def run_group(args, name, rows, fields):
    out = args.output / name
    out.mkdir(mode=0o750)
    with (out / "cases.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    command = [args.tool, "--tool=memcheck", "--leak-check=yes",
               "--log-file=" + str(out / "sanitizer.log"),
               "--", str(args.binary), "--gtest_filter=*/*/*", "--gtest_output=xml:" + str(out / "gtest.xml")]
    write_json(out / "command.json", dict(command=command, cases_csv_sha256=hashlib.sha256(
        (out / "cases.csv").read_bytes()).hexdigest()))
    record = dict(group=name, selected=[row["case_name"] for row in rows],
                  **execute(command, execution_env(args, out), out, args.timeout))
    return summarize_group(out, record)


def summarize_group(out, record):
    try:
        observed, failed, skipped, xml_error = parse_cases(out, None)
    except (KeyError, ValueError) as exc:
        observed, failed, skipped, xml_error = [], [], [], str(exc)
    expected = set(record["selected"])
    record.update(failed=failed, skipped=skipped, xml_error=xml_error,
                  missing=sorted(expected - set(observed)), unexpected=sorted(set(observed) - expected),
                  duplicate_count=len(observed) - len(set(observed)))
    detail = out / "sanitizer.log"
    record["report_exists"] = detail.is_file()
    # Count only the dedicated report: stdout can contain another copy of the same records.
    text = detail.read_text(errors="replace") if detail.is_file() else ""
    required = KERNELS[record["group"]]
    record["sanitizer"] = sanitizer_result(text, required, len(record["selected"]) if required else 0)
    # Host negative tests deliberately log OP_LOGE. Only sanitizer-tagged console
    # diagnostics supplement the dedicated report; do not count console kernel copies.
    console = (out / "process.log").read_text(errors="replace")
    tagged = "\n".join(line for line in console.splitlines() if re.search(r"\[mssanitizer\]|^=+", line, re.I))
    record["console_diagnostics"] = sanitizer_diagnostics(tagged)
    failures = ("failed", "skipped", "xml_error", "missing", "unexpected", "duplicate_count", "console_diagnostics")
    record["passed"] = (record["return_code"] == 0 and not record["timed_out"] and record["report_exists"]
                        and not any(record[key] for key in failures)
                        and record["sanitizer"]["passed"])
    write_json(out / "summary.json", record)
    return record


def run_plan(args, plan, summary):
    args.tool = shutil.which(args.tool)
    if args.tool is None or not args.binary.is_file():
        summary.update(status="unavailable", error="Missing mssanitizer or sanitizer-built test binary")
        return
    summary.update(status="running", tool=args.tool,
                   binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest())
    for name, rows in plan["groups"].items():
        result = run_group(args, name, rows, plan["fields"])
        summary["groups"].append(result)
        summary["untested"].remove(name)
        write_json(args.output / "summary.json", summary)
        logging.info("%s: %s", name, "PASS" if result["passed"] else "FAIL")
        if not result["passed"]:
            break
    summary["passed"] = not summary["untested"] and all(row["passed"] for row in summary["groups"])
    summary["status"] = "passed" if summary["passed"] else "failed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    binary = REPO / "build-memcheck/test/gemm/hgemmStridedBatched/hgemmStridedBatched_test"
    parser.add_argument("--binary", type=Path, default=binary)
    parser.add_argument("--tool", default="mssanitizer")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--device-memory", choices=["high-bandwidth", "normal"], default="high-bandwidth")
    parser.add_argument("--timeout", type=int, default=TIMEOUT_SECONDS, help="seconds per group")
    parser.add_argument("--dry-run", action="store_true", help="write plan only; never reports memcheck passed")
    args = parser.parse_args()
    if args.timeout <= 0 or args.device < 0:
        parser.error("timeout must be positive and device nonnegative")
    args.output, args.binary = args.output.resolve(), args.binary.resolve()
    args.output.mkdir(mode=0o750, parents=True, exist_ok=False)
    plan = load_plan()
    write_json(args.output / "plan.json", plan)
    summary = dict(passed=False, status="planned", binary=str(args.binary), device=args.device,
                   device_memory=args.device_memory, groups=[], untested=list(plan["groups"]))
    if not args.dry_run:
        run_plan(args, plan, summary)
    write_json(args.output / "summary.json", summary)
    logging.info("%s", json.dumps(dict(status=summary["status"], passed=summary["passed"], output=str(args.output))))
    return 0 if args.dry_run or summary["passed"] else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    sys.exit(main())
