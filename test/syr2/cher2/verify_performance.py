#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Collect actual msprof op task durations for all 200 published PF cases."""

import argparse
import csv
import json
import logging
import shutil
import sys
import math
import os
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent


def prepare_app(binary, output, fix):
    if not fix:
        return str(binary)
    # msopprof 26.1 in a container can convert public device 0 to host physical
    # device 7 before passing it to a driver that expects container-local IDs.
    # Override only the profiler's device lookup with the public AscendCL API.
    # No operator implementation, launch, clock, or duration is modified.
    source = output / "profiler_device.cpp"
    source.write_text(
        """#include <cstdint>
extern "C" int aclrtGetDevice(int32_t*);
class DeviceContext { public: static int32_t GetRunningDeviceId(); };
int32_t DeviceContext::GetRunningDeviceId() {
    int32_t device = 0;
    return aclrtGetDevice(&device) == 0 ? device : -1;
}
""",
        encoding="utf-8",
    )
    library = output / "profiler_device.so"
    compiler = shutil.which("g++")
    if compiler is None:
        raise RuntimeError("g++ not found")
    subprocess.run([str(Path(compiler).resolve()), "-shared", "-fPIC", str(source), "-o", str(library)], check=True)
    app = output / "profile_app.sh"
    app.write_text(
        "#!/bin/bash\nexport LD_PRELOAD="
        + shlex.quote(str(library))
        + ":${LD_PRELOAD}\nexec "
        + shlex.quote(str(binary))
        + ' "$@"\n',
        encoding="utf-8",
    )
    app.chmod(0o700)
    return str(app)


def kernel(row):
    return "cher2_aiv_kernel" if 256 <= int(row["n"]) <= 4096 else "cher2_kernel"


def read_durations(folder):
    result = {}
    for f in folder.rglob("OpBasicInfo_*.csv"):
        entries = list(csv.DictReader(f.open(encoding="utf-8-sig")))
        if len(entries) != 1:
            raise RuntimeError("Expected one raw task per CSV: " + str(f))
        entry = entries[0]
        name = entry["Op Name"]
        if name not in ("cher2_kernel", "cher2_aiv_kernel"):
            raise RuntimeError("Unexpected kernel: " + name)
        duration = float(entry["Task Duration(us)"])
        if not math.isfinite(duration) or duration <= 0:
            raise RuntimeError("Invalid task duration")
        result.setdefault(name, []).append((int(f.parent.name), duration, str(f.relative_to(folder))))
    for name, values in result.items():
        values.sort()
        if [v[0] for v in values] != list(range(len(values))):
            raise RuntimeError("Missing/duplicate raw launch IDs: " + name)
    return result


def load_cases():
    rows = [
        r
        for r in csv.DictReader((HERE / "arch35/cher2_test.csv").open(encoding="utf-8-sig"))
        if r["case_name"].startswith("TC_PF_")
    ]
    baseline = list(csv.DictReader((HERE / "gpu_baseline.csv").open(encoding="utf-8-sig")))
    if len(rows) != 200 or len(baseline) != 200:
        raise RuntimeError("Expected all 200 published PF cases and GPU baselines")
    for row, ref in zip(rows, baseline):
        for key in ("uplo", "n", "alpha_real", "alpha_imag", "incx", "incy", "lda"):
            if key == "uplo":
                valid = row[key].removeprefix("ACLBLAS_") == ref[key]
            else:
                valid = float(row[key]) == float(ref[key])
            if not valid:
                raise RuntimeError("GPU parameter mismatch: " + row["case_name"] + ":" + key)
    return rows, baseline


def profile_batch(a, app, batch, offset, env):
    prefix = a.output / ("batch_" + str(offset // a.batch_size))
    prefix.mkdir()
    xml = prefix / "gtest.xml"
    filt = ":".join("*" + r["case_name"] for r in batch)
    # msopprof tokenizes --application on whitespace without shell unquoting.
    if any(c.isspace() for c in app + str(xml)):
        raise RuntimeError("msprof op requires an application/output path without whitespace")
    application = " ".join([app, "--gtest_filter=" + filt, "--gtest_output=xml:" + str(xml)])
    cmd = [
        "msprof",
        "op",
        "--application=" + application,
        "--output=" + str(prefix / "raw"),
        "--aic-metrics=BasicInfo",
        "--warm-up=5",
        "--launch-count=" + str(len(batch) * 60),
        "--mstx=on",
        "--replay-mode=range",
    ]
    (prefix / "command.json").write_text(json.dumps(cmd, indent=2) + "\n", encoding="utf-8")
    with (prefix / "msprof.log").open("w", encoding="utf-8") as f:
        subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=a.timeout, check=True)
    text = (prefix / "msprof.log").read_text(encoding="utf-8")
    if "[ERROR]" in text:
        raise RuntimeError("msprof reported an error; inspect " + str(prefix / "msprof.log"))
    tests = ET.parse(xml).getroot().findall(".//testcase")
    if len(tests) != len(batch) or any(t.find("failure") is not None or t.find("skipped") is not None for t in tests):
        raise RuntimeError("Missing or failed PF GTest")
    raw = read_durations(prefix / "raw")
    return prefix, text, raw


def collect_batch_samples(batch, offset, baseline, prefix, text, raw):
    all_samples, results = [], []
    used = {name: 0 for name in raw}
    for index, row in enumerate(batch, offset):
        name = row["case_name"]
        calls = re.findall(
            r"\[PERF_CALL\] case=" + re.escape(name) + r" sample=(\d+) phase=(?:warmup|sample) status=0", text
        )
        if list(map(int, calls)) != list(range(65)):
            raise RuntimeError("Missing successful API calls: " + name)
        k = kernel(row)
        begin = used.get(k, 0)
        end = begin + 60
        samples = raw.get(k, [])[begin:end]
        if len(samples) != 60:
            raise RuntimeError("Missing task samples: " + name)
        used[k] = end
        values = [s[1] for s in samples]
        mean = statistics.mean(values)
        gpu = float(baseline[index]["gpu_ms"]) * 1000
        result = dict(
            row,
            baseline_id=baseline[index]["id"],
            kernel=k,
            warmup=5,
            samples=60,
            npu_mean_us=mean,
            npu_min_us=min(values),
            npu_max_us=max(values),
            gpu_us=gpu,
            ratio=gpu / mean,
            passed=gpu / mean >= 0.4,
        )
        results.append(result)
        for sample, (_, value, path) in enumerate(samples):
            all_samples.append(
                {"case": name, "sample": sample, "duration_us": value, "raw_csv": str(prefix.name + "/raw/" + path)}
            )
    if any(used.get(k, 0) != len(v) for k, v in raw.items()):
        raise RuntimeError("Unassigned raw samples")
    return results, all_samples


def save_results(a, results, all_samples):
    for name, data in [("performance_results.csv", results), ("performance_samples.csv", all_samples)]:
        with (a.output / name).open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    summary = {
        "total": 200,
        "passed": sum(r["passed"] for r in results),
        "samples": 12000,
        "warmup": 5,
        "valid_samples_per_case": 60,
        "threshold": 0.4,
        "minimum_ratio": min(r["ratio"] for r in results),
        "metric": "msprof op OpBasicInfo Task Duration(us), arithmetic mean",
        "container_device_fix": a.container_device_fix,
    }
    (a.output / "performance_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    logging.info("%s", json.dumps(summary))
    return summary["passed"] == 200


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", type=Path, default=Path("build/test/syr2/cher2/cher2_test"))
    p.add_argument("--output", type=Path, default=Path("cher2_performance"))
    p.add_argument("--container-device-fix", action="store_true")
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--batch-size", type=int, default=16, choices=range(1, 17))
    a = p.parse_args()
    a.binary, a.output = a.binary.resolve(), a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    rows, baseline = load_cases()
    app = prepare_app(a.binary, a.output, a.container_device_fix)
    env = dict(os.environ, CHER2_PROFILE="1", CHER2_PROFILE_RANGE="1")
    env.pop("BLAS_TEST_CASES", None)
    all_samples, results = [], []
    for offset in range(0, len(rows), a.batch_size):
        end = offset + a.batch_size
        batch = rows[offset:end]
        prefix, text, raw = profile_batch(a, app, batch, offset, env)
        batch_results, batch_samples = collect_batch_samples(batch, offset, baseline, prefix, text, raw)
        results.extend(batch_results)
        all_samples.extend(batch_samples)
        logging.info("Completed %d/200 cases", len(results))
    if not save_results(a, results, all_samples):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
