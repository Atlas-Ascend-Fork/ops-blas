#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Strictly validate all Cher2 msprof op batches and their raw Task Duration samples."""
import argparse
import json
import logging
import math
from pathlib import Path
import re
import statistics
import sys
from summarize_task_profile import KEYS, key, load_references, read_csv, require, required, write_results


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--cases', default=Path(__file__).parent / 'arch35/cher2_test.csv', type=Path)
    parser.add_argument('--baseline', default=Path(__file__).parent / 'gpu_baseline.csv', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    return parser.parse_args()


def validate_batch_log(batch, root, count):
    names = required(batch, 'cases')
    require(required(batch, 'returncode') == 0 and 0 < len(names) * count <= 1000,
            'Failed collection or invalid batch size')
    log = (root / required(batch, 'log')).read_text()
    require(not re.search(r'\[\s*(?:ERROR|FAILED)\s*\]', log), 'Collection/test error')
    require(f'[  PASSED  ] {len(names)} tests.' in log and 'Profiling data parse finished.' in log,
            'Incomplete GTest run or profiling parse')
    samples = re.findall(r'\[ CHER2 PERF \] (\S+) warmup=(\d+) samples=(\d+)', log)
    require([name for name, _, _ in samples] == names, 'Batch case order mismatch')
    require(all(warmup == '0' and int(size) == count for _, warmup, size in samples), 'Unexpected sampling count')
    require(log.count('Warm Up enabled. times:5') == len(names) * count, 'Missing profiler warmup records')
    return names


def read_batch_durations(batch, root, expected):
    profiles = list((root / required(batch, 'profile')).glob('OPPROF_*'))
    require(len(profiles) == 1, 'Expected exactly one op profile directory per batch')
    files = sorted(profiles[0].glob('cher2_kernel/*/OpBasicInfo*.csv'), key=lambda path: int(path.parent.name))
    require([int(path.parent.name) for path in files] == list(range(expected)), 'Missing/duplicate op samples')
    durations = []
    for path in files:
        rows = read_csv(path)
        require(len(rows) == 1 and required(rows[0], 'Op Name') == 'cher2_kernel', 'Unexpected op sample rows')
        value = float(required(rows[0], 'Task Duration(us)'))
        require(math.isfinite(value) and value > 0, 'Invalid op task duration')
        durations.append(value)
    return durations


def summarize_batch(names, durations, count, references, raw):
    case_map, baseline = references
    results = []
    for index, name in enumerate(names):
        require(name in case_map and name not in raw, 'Duplicate/unknown case')
        values = durations[index * count:(index + 1) * count]
        case = required(case_map, name)
        mean, mean10 = statistics.mean(values), statistics.mean(values[:10])
        gpu = required(baseline, key(case))
        results.append(dict(case_name=name, **{field: required(case, field) for field in KEYS}, samples=count,
                            npu_us=mean, npu_first10_us=mean10, gpu_us=gpu,
                            ratio=gpu / mean, ratio_first10=gpu / mean10,
                            verdict='PASS' if min(gpu / mean, gpu / mean10) >= 0.4 else 'FAIL', source='msprof_op'))
        raw[name] = values
    return results


def summarize_manifest(manifest_path, references):
    manifest = json.loads(manifest_path.read_text())
    count = required(manifest, 'samples')
    require(count in (10, 51) and required(manifest, 'warmup') == 5, 'Unexpected manifest sampling count')
    results, raw = [], {}
    for batch in required(manifest, 'batches'):
        names = validate_batch_log(batch, manifest_path.parent, count)
        durations = read_batch_durations(batch, manifest_path.parent, len(names) * count)
        results.extend(summarize_batch(names, durations, count, references, raw))
    case_map, _ = references
    require(set(raw) == set(case_map), 'Incomplete performance suite')
    return results, raw


def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s', stream=sys.stdout)
    args = parse_arguments()
    references = load_references(args.cases, args.baseline)
    results, raw = summarize_manifest(args.manifest, references)
    write_results(args.output, results, raw)
    failures = [row for row in results if required(row, 'verdict') == 'FAIL']
    logging.info('msprof op: %d/200 PASS. Min ratio=%.6f',
                 200 - len(failures), min(required(row, 'ratio') for row in results))
    for row in failures:
        logging.error('%s', row)
    return bool(failures)


if __name__ == '__main__':
    raise SystemExit(main())
