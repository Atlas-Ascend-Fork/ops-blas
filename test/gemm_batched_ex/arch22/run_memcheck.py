#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Run GemmBatchedEx contracts and representative paths under mssanitizer memcheck."""
import argparse
from collections import Counter
import hashlib
import json
import logging
from pathlib import Path
import re
import sys
from typing import NamedTuple
import xml.etree.ElementTree as ET

from run_self_test import DIRECTORY, SUITES, execute, execution_env, read_cases, validate_precision_log, validate_xml


class CoverageKey(NamedTuple):
    route: str
    trans_a: str
    trans_b: str
    device_scalars: bool
    nonzero_beta: bool
    padded_ldc: bool
    large_batch: bool


def is_large_short_k_mix(row, m, n, k):
    if not 32 < k <= 128:
        return False
    if min(m, n) < 128 or m * n < 32768:
        return False
    if float(row['alpha']) != 1 or float(row['beta']) != 0 or 'device_scalars' in row['description']:
        return False
    return not (k <= 64 and n <= 512 and int(row['ldc']) % 8 != 0)


def coverage_key(row, m, n, k, batch_count) -> CoverageKey:
    if k >= 2048:
        route = 'split'
    elif k >= 512:
        route = 'policy'
    elif max(m, n, k) <= 32:
        route = 'tiny'
    else:
        route = 'vector_or_cube'
    return CoverageKey(
        route=route,
        trans_a=row['transA'],
        trans_b=row['transB'],
        device_scalars='device_scalars' in row['description'],
        nonzero_beta=float(row['beta']) != 0,
        padded_ldc=int(row['ldc']) != m,
        large_batch=batch_count > 4,
    )


def representative_rows(suite, rows):
    if suite == 'special':
        return rows
    selected = {}
    for row in rows:
        if row['expect_result'] != 'ACLBLAS_STATUS_SUCCESS':
            selected[row['case_name']] = row
    representatives = {}
    for row in rows:
        if row['expect_result'] != 'ACLBLAS_STATUS_SUCCESS':
            continue
        m, n, k = (int(row[key]) for key in ('m', 'n', 'k'))
        batch_count = int(row['batchCount'])
        if m <= 0 or n <= 0 or batch_count <= 0:
            continue
        # Exercise every large short-K geometry affected by the two MIX pipelines.
        if is_large_short_k_mix(row, m, n, k):
            selected[row['case_name']] = row
        key = coverage_key(row, m, n, k, batch_count)
        # Keep coverage buckets while bounding instrumentation work per representative.
        work = m * n * max(k, 1) * batch_count
        previous = representatives.get(key)
        if previous is None or work < previous[0]:
            representatives[key] = (work, row)
    for _, row in representatives.values():
        selected[row['case_name']] = row
    return list(selected.values())


def sanitizer_result(text, tool='memcheck'):
    start_pattern = r'Start ' + re.escape(tool) + r' sanitizer on kernel[ \t]+([^\r\n]+)'
    finish_pattern = r'Sanitizer finished on kernel[ \t]+([^\r\n]+?)\. No error detected\.'
    starts = [name.strip().strip('"') for name in re.findall(start_pattern, text)]
    finished = [name.strip().strip('"') for name in re.findall(finish_pattern, text)]
    diagnostics = []
    for line in text.splitlines():
        if re.search(r'\b(?:WARNING|ERROR|FATAL)\b', line, flags=re.IGNORECASE) and 'No error detected.' not in line:
            diagnostics.append(line)
    gemm_starts = [name for name in starts if 'gemm_batched_ex' in name or 'GemmBatchedEx' in name]
    passed = bool(gemm_starts) and Counter(starts) == Counter(finished) and not diagnostics
    failed_pattern = r'Sanitizer finished on kernel[ \t]+([^\r\n]+?)\. See all detected errors above\.'
    failed = [name.strip().strip('"') for name in re.findall(failed_pattern, text)]
    unfinished = Counter(starts) - Counter(finished) - Counter(failed)
    return dict(passed=passed, kernel_starts=len(starts), clean_kernel_finishes=len(finished),
                failed_kernel_finishes=len(failed), unfinished_kernels=list(unfinished.elements()),
                all_kernel_finishes_match=Counter(starts) == Counter(finished) + Counter(failed),
                gemm_kernel_starts=len(gemm_starts), kernel_counts=dict(Counter(starts)), diagnostics=diagnostics)


def ownership_only(result):
    """Keep GM ownership warnings visible; they require a separate clean racecheck."""
    return (result['gemm_kernel_starts'] > 0 and result['all_kernel_finishes_match']
            and bool(result['diagnostics'])
            and all(re.fullmatch(r'====== WARNING: out of bounds of size \d+', line)
                    for line in result['diagnostics']))


def expects_kernel(row):
    if row['expect_result'] != 'ACLBLAS_STATUS_SUCCESS' or any(int(row[key]) <= 0 for key in ('m', 'n', 'batchCount')):
        return False
    # Host alpha/beta can produce C := C, so the API returns before launching.
    host_scalars = 'device_scalars' not in row['description']
    scale_only = int(row['k']) == 0 or (host_scalars and float(row['alpha']) == 0.0)
    return not (scale_only and host_scalars and float(row['beta']) == 1.0)


def run_tool(args, tool, stage, csv_path, rows, selected):
    stage.mkdir(mode=0o750)
    xml = stage / 'gtest.xml'
    detail = stage / 'sanitizer.log'
    command = [args.tool, '--tool=' + tool, '--log-file=' + str(detail),
               '--cache-size=' + str(args.cache_size)]
    if tool == 'memcheck':
        command.append('--leak-check=yes')
    command += ['--', str(args.binary.resolve()), '--gtest_filter=' + selected,
                '--gtest_output=xml:' + str(xml)]
    record = dict(command=command, passed=False)
    (stage / 'command.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    record['returncode'] = execute(command, execution_env(csv_path, stage / 'evidence'),
                                   stage / 'process.log', args.timeout)
    text = (stage / 'process.log').read_text(encoding='utf-8', errors='replace')
    if detail.is_file():
        text += '\n' + detail.read_text(encoding='utf-8', errors='replace')
    record['sanitizer'] = sanitizer_result(text, tool)
    try:
        validate_xml(xml, rows)
        validate_precision_log(text, rows)
        if record['returncode'] != 0:
            raise ValueError('Instrumented test process returned nonzero')
        record['tests_passed'] = True
        launches_expected = any(expects_kernel(row) for row in rows)
        sanitizer = record['sanitizer']
        no_launches = (sanitizer['kernel_starts'] == 0 and sanitizer['all_kernel_finishes_match']
                       and not sanitizer['diagnostics'])
        record['passed'] = sanitizer['passed'] if launches_expected else no_launches
    except (ValueError, OSError, ET.ParseError) as error:
        record['tests_passed'] = False
        record['first_failure'] = str(error)
    (stage / 'result.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    return record


def run_batch(args, stage, csv_path, rows):
    stage.mkdir(mode=0o750)
    selected = ':'.join(['GemmBatchedExContract.*'] + ['*CsvDriven/' + row['case_name'] for row in rows])
    memory = run_tool(args, 'memcheck', stage / 'memcheck', csv_path, rows, selected)
    record = dict(selected_cases=[row['case_name'] for row in rows], memcheck=memory, passed=memory['passed'])
    if not memory['passed'] and memory['tests_passed'] and ownership_only(memory['sanitizer']):
        races = run_tool(args, 'racecheck', stage / 'racecheck', csv_path, rows, selected)
        same_kernels = memory['sanitizer']['kernel_counts'] == races['sanitizer']['kernel_counts']
        # Official mssanitizer guide 6.1.3.2: ownership warnings require race analysis.
        # Never waive illegal accesses, register warnings, incomplete kernels or races.
        record.update(racecheck=races, ownership_warnings_reviewed=not memory['passed'],
                      passed=races['passed'] and same_kernels)
    (stage / 'result.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    return record


def run_group(args, suite):
    csv_path, all_rows = read_cases(suite)
    rows = representative_rows(suite, all_rows)
    stage = args.output / suite
    stage.mkdir(mode=0o750)
    results = []
    for first in range(0, len(rows), args.batch_size):
        batch = rows[first:first + args.batch_size]
        result = run_batch(args, stage / f'batch_{first // args.batch_size:03d}', csv_path, batch)
        results.append(result)
        logging.info('%s batch %d: %s', suite, len(results), 'PASS' if result['passed'] else 'FAIL')
        if not result['passed']:
            break
    record = dict(suite=suite, selected_cases=[row['case_name'] for row in rows], batches=results,
                  csv_sha256=hashlib.sha256(csv_path.read_bytes()).hexdigest(),
                  passed=sum(len(result['selected_cases']) for result in results) == len(rows)
                  and all(result['passed'] for result in results))
    (stage / 'result.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    return record


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    default = DIRECTORY.parents[2] / 'build-sanitize/test/gemm_batched_ex/gemm_batched_ex_test'
    parser.add_argument('--binary', type=Path, default=default)
    parser.add_argument('--tool', default='mssanitizer')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--suite', choices=SUITES, action='append')
    parser.add_argument('--timeout', type=int, default=7200)
    parser.add_argument('--batch-size', type=int, default=8, help='CSV cases per fresh sanitizer process')
    parser.add_argument('--cache-size', type=int, default=100, help='Sanitizer record buffer MB per block')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s', stream=sys.stdout)
    args = parse_args()
    suites = args.suite or list(SUITES)
    invalid_run_size = args.timeout <= 0 or args.batch_size <= 0
    valid_cache_size = 1 <= args.cache_size <= 8192
    if len(set(suites)) != len(suites) or invalid_run_size or not valid_cache_size:
        raise ValueError('Duplicate suite or invalid timeout, batch size or cache size')
    plan = [dict(suite=suite, cases=len(representative_rows(suite, read_cases(suite)[1]))) for suite in suites]
    if args.dry_run:
        logging.info('%s', json.dumps(dict(status='planned', npu_calls=0, groups=plan), indent=2))
        return 0
    if not args.binary.is_file():
        raise FileNotFoundError(args.binary)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    args.output.chmod(0o750)
    results = []
    for suite in suites:
        result = run_group(args, suite)
        results.append(result)
        if not result['passed']:
            break
    passed = len(results) == len(suites) and all(result['passed'] for result in results)
    summary = dict(passed=passed, groups=results, untested=suites[len(results):], planned=plan)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    logging.info('%s', json.dumps(summary, indent=2))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
