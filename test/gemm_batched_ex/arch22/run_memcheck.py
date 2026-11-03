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
import xml.etree.ElementTree as ET

from run_self_test import DIRECTORY, SUITES, execute, execution_env, read_cases, validate_precision_log, validate_xml


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
        if m <= 0 or n <= 0 or int(row['batchCount']) <= 0:
            continue
        route = ('split' if k >= 2048 else 'policy' if k >= 512 else
                 'tiny' if max(m, n, k) <= 32 else 'vector_or_cube')
        key = (route, row['transA'], row['transB'], 'device_scalars' in row['description'],
               float(row['beta']) != 0, int(row['ldc']) != m, int(row['batchCount']) > 4)
        representatives.setdefault(key, row)
    for row in representatives.values():
        selected[row['case_name']] = row
    return list(selected.values())


def sanitizer_result(text):
    starts = re.findall(r'Start memcheck sanitizer on kernel\s+(\S+)', text)
    finished = re.findall(r'Sanitizer finished on kernel\s+(\S+)\. No error detected\.', text)
    diagnostics = []
    for line in text.splitlines():
        if re.search(r'\b(?:WARNING|ERROR|FATAL)\b', line) and 'No error detected.' not in line:
            diagnostics.append(line)
    gemm_starts = [name for name in starts if name.startswith('gemm_batched_ex')]
    passed = bool(gemm_starts) and Counter(starts) == Counter(finished) and not diagnostics
    return dict(passed=passed, kernel_starts=len(starts), clean_kernel_finishes=len(finished),
                gemm_kernel_starts=len(gemm_starts), diagnostics=diagnostics)


def run_group(args, suite):
    csv_path, all_rows = read_cases(suite)
    rows = representative_rows(suite, all_rows)
    stage = args.output / suite
    stage.mkdir(mode=0o750)
    xml = stage / 'gtest.xml'
    detail = stage / 'sanitizer.log'
    selected = ':'.join(['GemmBatchedExContract.*'] + ['*CsvDriven/' + row['case_name'] for row in rows])
    command = [args.tool, '--tool=memcheck', '--leak-check=yes', '--log-file=' + str(detail), '--',
               str(args.binary.resolve()), '--gtest_filter=' + selected, '--gtest_output=xml:' + str(xml)]
    record = dict(suite=suite, selected_cases=[row['case_name'] for row in rows], command=command,
                  csv_sha256=hashlib.sha256(csv_path.read_bytes()).hexdigest(), passed=False)
    (stage / 'command.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    record['returncode'] = execute(command, execution_env(csv_path, stage / 'evidence'),
                                   stage / 'process.log', args.timeout)
    text = (stage / 'process.log').read_text(encoding='utf-8', errors='replace')
    if detail.is_file():
        text += '\n' + detail.read_text(encoding='utf-8', errors='replace')
    record['sanitizer'] = sanitizer_result(text)
    try:
        validate_xml(xml, rows)
        validate_precision_log(text, rows)
        if record['returncode'] != 0 or not record['sanitizer']['passed']:
            raise ValueError('Sanitizer failed or lacks complete real kernel instrumentation evidence')
        record['passed'] = True
    except (ValueError, OSError, ET.ParseError) as error:
        record['first_failure'] = str(error)
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
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s', stream=sys.stdout)
    args = parse_args()
    suites = args.suite or list(SUITES)
    if len(set(suites)) != len(suites) or args.timeout <= 0:
        raise ValueError('Duplicate suite or invalid timeout')
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
