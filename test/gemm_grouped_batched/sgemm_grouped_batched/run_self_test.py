#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Run full accuracy and supplemental gates; performance timing is opt-in."""
import argparse
from pathlib import Path
import subprocess
from test_gate import (ARCH, audit_csvs, check_jsonl, check_mapping, check_xml, environment,
                       identity, list_tests, write_summary)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New evidence directory (never overwritten)")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", help="Explicit short run instead of full accuracy")
    parser.add_argument("--include-performance", action="store_true", help="Also execute all 200 timing cases")
    parser.add_argument("--timeout", type=int, default=14400, help="Seconds per suite")
    return parser.parse_args()


def main():
    args = parse_args()
    binary, output = args.binary.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    summary = {"status": "FAIL", "suites": [], "performance_acceptance": False}
    try:
        csv_evidence = audit_csvs()
        summary.update(identity(binary), csv=csv_evidence)
        # Audit discovery for every committed CSV even in explicit smoke mode.
        for name in csv_evidence:
            if name == "review_regression.csv":
                continue
            env = environment(ARCH / name, args.device, output / "discovery.jsonl")
            check_mapping(list_tests(binary, env), ARCH / name)
        suites = ["sgemm_grouped_batched_test.csv"] if args.smoke else [
            "sgemm_grouped_batched_test_full.csv", "supplemental.csv"]
        for index, name in enumerate(suites):
            stem = f"suite_{index}"
            env = environment(ARCH / name, args.device, output / (stem + ".jsonl"))
            pattern = "*" if args.include_performance else "*-*CSVKernelTiming*"
            expected = list_tests(binary, env, pattern)
            xml = output / (stem + ".xml")
            command = [str(binary), f"--gtest_filter={pattern}", f"--gtest_output=xml:{xml}"]
            with (output / (stem + ".log")).open("w", encoding="utf-8") as log:
                result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                        timeout=args.timeout, check=False)
            record = {"csv": name, "command": command, "returncode": result.returncode}
            summary["suites"].append(record)
            if result.returncode:
                raise RuntimeError(f"Suite failed: {name}, exit={result.returncode}")
            record.update(check_xml(xml, expected))
            record["jsonl_cases"] = check_jsonl(output / (stem + ".jsonl"), expected)
        summary["status"] = "PASS"
    except Exception as error:
        summary["error"] = repr(error)
    finally:
        write_summary(output, "self_test_summary.json", summary)
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
