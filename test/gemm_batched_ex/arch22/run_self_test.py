#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Run GemmBatchedEx CSV accuracy suites after the required performance gate."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

DIRECTORY = Path(__file__).resolve().parent
SUITES = {
    'main': ('gemm_batched_ex_test.csv', 2393),
    'fixed': ('fixed_scalar144.csv', 144),
    'random': ('random_scalar72.csv', 72),
    'special': ('special_zero432.csv', 432),
}
CONTRACT_TESTS = {'CsvCaseCount', 'SpecialValueComparison'}


def read_cases(suite):
    filename, expected = SUITES[suite]
    path = DIRECTORY / filename
    with path.open(newline='', encoding='utf-8') as source:
        rows = list(csv.DictReader(source))
    names = [row.get('case_name', '') for row in rows]
    if len(rows) != expected or len(set(names)) != expected:
        raise ValueError(f'{filename}: expected {expected} unique cases')
    if not all(re.fullmatch(r'[A-Za-z0-9_]+', name) for name in names):
        raise ValueError(f'{filename}: empty or unsafe case name')
    return path, rows


def validate_xml(path, rows):
    root = ET.parse(path).getroot()
    expected = Counter(row['case_name'] for row in rows)
    seen = Counter()
    contracts = Counter()
    for test in root.iter('testcase'):
        if test.get('status') != 'run' or test.get('result') != 'completed':
            raise ValueError('A GTest case was skipped, disabled or incomplete')
        name = test.get('name', '')
        if name.startswith('CsvDriven/'):
            seen[name.removeprefix('CsvDriven/')] += 1
        elif name in CONTRACT_TESTS:
            contracts[name] += 1
        else:
            raise ValueError(f'Unexpected GTest case: {name}')
    if seen != expected or contracts != Counter({name: 1 for name in CONTRACT_TESTS}):
        raise ValueError('CSV/GTest coverage mismatch (missing, duplicate or extra case)')
    if int(root.get('tests', '-1')) != len(rows) + len(CONTRACT_TESTS):
        raise ValueError('GTest total count differs from CSV plus contract tests')
    if any(int(root.get(key, '-1')) != 0 for key in ('failures', 'errors', 'disabled')):
        raise ValueError('GTest reported failures, errors or disabled tests')
    if any(test.find('failure') is not None for test in root.iter('testcase')):
        raise ValueError('GTest failure element present')


def validate_precision_log(text, rows):
    if 'GEMM_CENSUS_STOP_RUNTIME_OR_GUARD' in text:
        raise ValueError('Runtime, input or guard failure')
    census = Counter(re.findall(r'^GEMM_PRECISION_CENSUS (\S+) fresh batch=(\d+) passed=1\b', text, re.M))
    expected = Counter()
    for row in rows:
        positive = row['expect_result'] == 'ACLBLAS_STATUS_SUCCESS'
        active = int(row['m']) > 0 and int(row['n']) > 0
        if positive and active:
            expected.update((row['case_name'], str(batch)) for batch in range(int(row['batchCount'])))
    if census != expected:
        raise ValueError('Fresh-output numerical census is missing, repeated or failed')


def execution_env(csv_path, evidence):
    env = os.environ.copy()
    for key in ('GTEST_FILTER', 'GEMM_REFERENCE_CACHE', 'GEMM_OUTPUT_BLOB_POOL', 'GEMM_EXPECTED_CASES'):
        env.pop(key, None)
    env.update(GEMM_PHASE='precision', GEMM_TEST_CSV=str(csv_path.resolve()),
               GEMM_EVIDENCE_DIR=str(evidence.resolve()))
    return env


def execute(command, env, log, timeout):
    with log.open('x', encoding='utf-8') as output:
        try:
            completed = subprocess.run(command, env=env, stdout=output,
                                       stderr=subprocess.STDOUT, timeout=timeout, check=False)
            return completed.returncode
        except subprocess.TimeoutExpired:
            output.write('\nRUNNER_TIMEOUT\n')
            return 124
        except OSError as error:
            output.write(f'\nRUNNER_LAUNCH_ERROR: {error}\n')
            return 127


def run_suite(binary, output, suite, timeout):
    csv_path, rows = read_cases(suite)
    stage = output / suite
    stage.mkdir()
    xml = stage / 'gtest.xml'
    command = [str(binary.resolve()), '--gtest_filter=GemmBatchedExContract.*:*CsvDriven/*',
               '--gtest_break_on_failure=0', '--gtest_output=xml:' + str(xml.resolve())]
    record = dict(suite=suite, cases=len(rows), csv_sha256=hashlib.sha256(csv_path.read_bytes()).hexdigest(),
                  command=command, passed=False)
    (stage / 'command.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    record['returncode'] = execute(command, execution_env(csv_path, stage / 'evidence'),
                                   stage / 'process.log', timeout)
    text = (stage / 'process.log').read_text(encoding='utf-8', errors='replace')
    record['runtime_stop'] = (record['returncode'] not in (0, 1) or not xml.is_file()
                              or 'GEMM_CENSUS_STOP_RUNTIME_OR_GUARD' in text)
    try:
        validate_xml(xml, rows)
        validate_precision_log(text, rows)
        if record['returncode'] != 0:
            raise ValueError('Test process returned nonzero')
        record['passed'] = True
    except (ValueError, OSError, ET.ParseError) as error:
        record['first_failure'] = str(error)
    (stage / 'result.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    return record


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    default = DIRECTORY.parents[2] / 'build/test/gemm_batched_ex/gemm_batched_ex_test'
    parser.add_argument('--binary', type=Path, default=default)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--suite', choices=SUITES, action='append')
    parser.add_argument('--timeout', type=int, default=43200)
    parser.add_argument('--dry-run', action='store_true', help='Validate CSVs and print the plan; no NPU call')
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s', stream=sys.stdout)
    args = parse_args()
    suites = args.suite or list(SUITES)
    if len(set(suites)) != len(suites) or args.timeout <= 0:
        raise ValueError('Duplicate suite or invalid timeout')
    plan = [dict(suite=suite, csv=str(read_cases(suite)[0]), cases=len(read_cases(suite)[1])) for suite in suites]
    if args.dry_run:
        logging.info('%s', json.dumps(dict(status='planned', npu_calls=0, suites=plan), indent=2))
        return 0
    if not args.binary.is_file():
        raise FileNotFoundError(args.binary)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for suite in suites:
        result = run_suite(args.binary, args.output, suite, args.timeout)
        results.append(result)
        if result['runtime_stop']:
            break
    passed = len(results) == len(suites) and all(result['passed'] for result in results)
    summary = dict(passed=passed, expected_suites=suites, results=results,
                   untested=suites[len(results):], expected_csv_cases=sum(item['cases'] for item in plan))
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    logging.info('%s', json.dumps(summary, indent=2))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
