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

"""Build and run CSV GTest; reject missing, skipped, duplicated or failed cases."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import csv
import fnmatch
import json
import logging
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

from reference.build_reference import build_reference
from reference.build_reference import tool_path

TEST_ROOT = Path(__file__).resolve().parent
REPO = TEST_ROOT.parents[2]


def build(repo):
    ref = repo / "build/reference"
    build_reference(ref)
    cann = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/cann-9.1.0"))
    env = os.environ.copy()
    env.update(EAGER_LIBRARY_PATH=str(cann / "lib64"), LINUX_INCLUDE_PATH=str(cann / "x86_64-linux/include"))
    subprocess.run(
        [
            tool_path("cmake"),
            "-S",
            str(repo),
            "-B",
            str(repo / "build"),
            "-DSOC_VERSION=ascend950",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DASCEND_CANN_PACKAGE_PATH={cann}",
            f'-DCMAKE_PREFIX_PATH={cann / "lib64/cmake"}',
            "-DBUILD_TEST=ON",
            "-DTEST_NAMES=hgemmStridedBatched",
            f"-DREFBLAS_INCLUDE_DIR={ref}",
            f'-DREFBLAS_LIB={ref / "libnetlib_sgemm.so"}',
            f'-DREFLAPACK_LIB={ref / "libnetlib_sgemm.so"}',
        ],
        env=env,
        check=True,
    )
    subprocess.run(
        [tool_path("cmake"), "--build", str(repo / "build"), "--target", "hgemmStridedBatched_test", "-j8"],
        env=env,
        check=True,
    )


def merge_shards(args, out):
    merge_error = None
    if args.workers <= 1:
        return None
    root = ET.Element("testsuites")
    try:
        for index in range(args.workers):
            for suite in ET.parse(out / f"gtest-{index}.xml").getroot():
                root.append(suite)
        ET.ElementTree(root).write(out / "gtest.xml", encoding="utf-8", xml_declaration=True)
    except (OSError, ET.ParseError) as exc:
        merge_error = str(exc)
    with (out / "accuracy.log").open("w") as log, (out / "batches.jsonl").open("w") as batches:
        for index in range(args.workers):
            log.write(f"Actual GTest shard {index}/{args.workers}\n")
            log.write((out / f"accuracy-{index}.log").read_text())
            data = out / f"batches-{index}.jsonl"
            if data.exists():
                batches.write(data.read_text())
    return merge_error


def parse_cases(out, merge_error):
    observed, failed, skipped = [], [], []
    parse_error = merge_error
    try:
        for case in ET.parse(out / "gtest.xml").iter("testcase"):
            name = case.attrib["name"].split("/")[-1]
            observed.append(name)
            if case.find("failure") is not None or case.find("error") is not None:
                failed.append(name)
            if case.find("skipped") is not None or case.attrib.get("result") == "suppressed":
                skipped.append(name)
    except (OSError, ET.ParseError) as exc:
        parse_error = str(exc)
    return observed, failed, skipped, parse_error


def run_tests(args, csv_path, out, selected, rows, binary, repo):
    env = os.environ.copy()
    env.update(
        HGEMM_CSV=str(csv_path),
        HGEMM_RESULTS=str(out / "batches.jsonl"),
        BLAS_DEVICE_ID=str(args.device),
        HGEMM_DEVICE_MEMORY=args.device_memory,
    )
    env.pop("HGEMM_PROFILE_REPEATS", None)
    env.pop("BLAS_TEST_CASES", None)
    env.pop("GTEST_SHARD_INDEX", None)
    env.pop("GTEST_TOTAL_SHARDS", None)
    env.pop("HGEMM_DUMP_DIR", None)
    env.pop("HGEMM_INPUT_DIR", None)
    if args.input_attachments:
        env["HGEMM_INPUT_DIR"] = str(args.input_attachments.resolve())
    if args.dump:
        env["HGEMM_DUMP_DIR"] = str(out / "attachments")

    def run_shard(index):
        shard_env = env.copy()
        xml = out / ("gtest.xml" if args.workers == 1 else f"gtest-{index}.xml")
        log_path = out / ("accuracy.log" if args.workers == 1 else f"accuracy-{index}.log")
        if args.workers > 1:
            shard_env.update(
                GTEST_TOTAL_SHARDS=str(args.workers),
                GTEST_SHARD_INDEX=str(index),
                HGEMM_RESULTS=str(out / f"batches-{index}.jsonl"),
            )
        # Python shell globs include [] sets, which GTest does not understand.
        # Translate the selected CSV inventory to exact case-name patterns.
        test_filter = "*/*/*" if len(selected) == len(rows) else ":".join("*/*/" + r["case_name"] for r in selected)
        cmd = [str(binary), f"--gtest_filter={test_filter}", f"--gtest_output=xml:{xml}"]
        with log_path.open("w") as log:
            try:
                r = subprocess.run(
                    cmd, cwd=repo, env=shard_env, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout
                )
                return dict(index=index, command=cmd, return_code=r.returncode, timed_out=False)
            except subprocess.TimeoutExpired:
                return dict(index=index, command=cmd, return_code=124, timed_out=True)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        shards = list(executor.map(run_shard, range(args.workers)))
    return shards


def write_validation_summary(args, code, timed_out, expected, observed, failed, skipped, parse_error, shards, out):
    summary = dict(
        device_memory=args.device_memory,
        return_code=code,
        timed_out=timed_out,
        requested=len(expected),
        executed=len(observed),
        failed=failed,
        skipped=skipped,
        missing=sorted(expected - set(observed)),
        unexpected=sorted(set(observed) - expected),
        duplicate_count=len(observed) - len(set(observed)),
        xml_error=parse_error,
        shards=shards,
    )
    summary["passed"] = (
        code == 0
        and not timed_out
        and not failed
        and not skipped
        and not summary["missing"]
        and not summary["unexpected"]
        and not summary["duplicate_count"]
        and parse_error is None
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (out / "accuracy.log").open("a") as log:
        log.write(
            "\nDriver aggregate summary from actual XML and process exits:\n" + json.dumps(summary, indent=2) + "\n"
        )
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, default=REPO)
    ap.add_argument("--csv", type=Path, default=TEST_ROOT / "arch35/hgemmStridedBatched_test.csv")
    ap.add_argument("--filter", default="*", help="shell-style case_name glob; e.g. TC_L0_*")
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--device-memory", choices=["high-bandwidth", "normal"], default="high-bandwidth")
    ap.add_argument("--timeout", type=int, default=43200)
    ap.add_argument("--dump", action="store_true", help="save this run actual FP16 input/output buffers")
    ap.add_argument("--workers", type=int, default=1, help="independent GTest shards for correctness only")
    ap.add_argument("--input-attachments", type=Path, help="replay the actual A/B/C_input binary attachments")
    ap.add_argument("--output", type=Path, required=True, help="new output directory")
    args = ap.parse_args()
    if not 1 <= args.workers <= 16:
        raise ValueError("--workers must be between 1 and 16")
    repo, csv_path, out = args.repo.resolve(), args.csv.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8-sig")))
    selected = [r for r in rows if fnmatch.fnmatchcase(r["case_name"], args.filter)]
    expected = {r["case_name"] for r in selected}
    if not expected or len(expected) != len(selected):
        raise ValueError("Empty or duplicate requested case set")
    (out / "cases.json").write_text(json.dumps(selected, indent=2) + "\n")
    if not args.skip_build:
        build(repo)
    binary = repo / "build/test/gemm/hgemmStridedBatched/hgemmStridedBatched_test"
    if not binary.is_file():
        raise ValueError(f"Missing actual test binary: {binary}")
    shards = run_tests(args, csv_path, out, selected, rows, binary, repo)

    code = next((s["return_code"] for s in shards if s["return_code"]), 0)
    timed_out = any(s["timed_out"] for s in shards)
    merge_error = merge_shards(args, out)

    observed, failed, skipped, parse_error = parse_cases(out, merge_error)

    summary = write_validation_summary(
        args, code, timed_out, expected, observed, failed, skipped, parse_error, shards, out
    )

    logging.info(json.dumps(summary, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    sys.exit(main())
