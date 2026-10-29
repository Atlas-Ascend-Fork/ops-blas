#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Validate ordinary msprof task samples. This is NOT an msprof op acceptance report."""
import argparse
from collections import Counter
import csv
import json
import logging
import math
from pathlib import Path
import re
import statistics
import sys

KEYS = ('uplo', 'n', 'alpha_real', 'alpha_imag', 'incx', 'incy', 'lda')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def required(mapping, field):
    value = mapping.get(field)
    require(value is not None, f'Missing required field: {field}')
    return value


def key(row):
    return (required(row, 'uplo').removeprefix('ACLBLAS_'),) + tuple(float(required(row, k)) for k in KEYS[1:])


def read_csv(path):
    with open(path, newline='') as stream:
        return list(csv.DictReader(stream))


def load_references(cases_path, baseline_path):
    cases = [row for row in read_csv(cases_path) if required(row, 'case_name').startswith('TC_PF_')]
    case_map = {required(row, 'case_name'): row for row in cases}
    require(len(case_map) == len(cases) == 200, 'Expected 200 unique performance cases')
    baseline_rows = read_csv(baseline_path)
    require(len(baseline_rows) == 200, 'Missing GPU baseline rows')
    require(Counter(key(row) for row in cases) == Counter(key(row) for row in baseline_rows),
            'Baseline parameter mismatch')
    baseline = {}
    for row in baseline_rows:
        value = float(required(row, 'gpu_ms')) * 1000
        require(math.isfinite(value) and value > 0, 'Invalid GPU baseline duration')
        shape = key(row)
        # The supplied baseline repeats five shapes with identical times.
        require(shape not in baseline or baseline.get(shape) == value, 'Conflicting duplicate baseline')
        baseline[shape] = value
    return case_map, baseline


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', required=True, type=Path)
    parser.add_argument('--log', required=True, type=Path)
    parser.add_argument('--cases', default=Path(__file__).parent / 'arch35/cher2_test.csv', type=Path)
    parser.add_argument('--baseline', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    return parser.parse_args()


def validate_task_log(log_path, case_map):
    log = log_path.read_text()
    require(not re.search(r'\[\s*(?:FAILED|ERROR)\s*\]', log), 'Failed run/collection')
    require('[  PASSED  ] 200 tests.' in log, 'Incomplete GTest run')
    samples = re.findall(r'\[ CHER2 PERF \] (\S+) warmup=(\d+) samples=(\d+)', log)
    require(len(samples) == 200 and {name for name, _, _ in samples} == set(case_map), 'Case mismatch')
    require(all(warmup == '5' and count == '51' for _, warmup, count in samples), 'Unexpected sampling count')
    return samples


def read_task_durations(profile):
    files = list(profile.glob('PROF_*/mindstudio_profiler_output/op_summary_*.csv'))
    require(len(files) == 1, 'Use a fresh output directory containing exactly one profile')
    rows = [row for row in read_csv(files[0]) if required(row, 'Op Name') == 'cher2_kernel']
    rows.sort(key=lambda row: float(required(row, 'Task Start Time(us)')))
    require(len(rows) == 200 * 56, 'Missing/duplicate kernel records')
    starts = [float(required(row, 'Task Start Time(us)')) for row in rows]
    require(len(set(starts)) == len(starts), 'Duplicate start times')
    durations = [float(required(row, 'Task Duration(us)')) for row in rows]
    require(all(math.isfinite(value) and value > 0 for value in durations), 'Invalid task duration')
    return durations


def summarize_tasks(samples, durations, case_map, baseline):
    result, raw = [], {}
    for index, (name, _, _) in enumerate(samples):
        case = required(case_map, name)
        values = durations[index * 56 + 5:(index + 1) * 56]
        gpu = required(baseline, key(case))
        mean10, mean51 = statistics.mean(values[:10]), statistics.mean(values)
        passed = min(gpu / mean10, gpu / mean51) >= 0.4
        result.append(dict(case_name=name, **{field: required(case, field) for field in KEYS},
                           npu_mean10_us=mean10, npu_mean51_us=mean51,
                           gpu_us=gpu, ratio10=gpu / mean10, ratio51=gpu / mean51,
                           verdict='PASS' if passed else 'FAIL', source='msprof_task_not_op'))
        raw[name] = dict(warmup_us=durations[index * 56:index * 56 + 5], effective_us=values)
    return result, raw


def write_results(output, results, raw):
    require(bool(results), 'No performance results')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    output.with_suffix('.samples.json').write_text(json.dumps(raw, indent=2) + '\n')


def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s', stream=sys.stdout)
    args = parse_arguments()
    case_map, baseline = load_references(args.cases, args.baseline)
    samples = validate_task_log(args.log, case_map)
    durations = read_task_durations(args.profile)
    results, raw = summarize_tasks(samples, durations, case_map, baseline)
    write_results(args.output, results, raw)
    failures = [row for row in results if required(row, 'verdict') != 'PASS']
    logging.info('Ordinary msprof task timing: %d/200 pass (both first 10 and all 51).', 200 - len(failures))
    logging.info('Min ratio: first10=%.6f, all51=%.6f',
                 min(required(row, 'ratio10') for row in results), min(required(row, 'ratio51') for row in results))
    for row in failures:
        logging.error('%s', row)
    return bool(failures)


if __name__ == '__main__':
    raise SystemExit(main())
