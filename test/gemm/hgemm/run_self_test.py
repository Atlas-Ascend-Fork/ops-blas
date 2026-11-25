#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Run the complete Hgemm accuracy suite and reject missing/duplicate CSV cases."""
import argparse
from collections import Counter
import csv
import json
import logging
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

ACCURACY_CASES = 1000
PERF_CASES = 200
SUCCESS_CASES = 980
NEGATIVE_CASES = 20


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    default_binary = Path(__file__).resolve().parents[3] / 'build/test/gemm/hgemm/hgemm_test'
    parser.add_argument('--binary', type=Path, default=default_binary)
    parser.add_argument('--output', required=True, type=Path)
    return parser.parse_args()


def read_cases():
    with (Path(__file__).parent / 'arch35/hgemm_test.csv').open() as source:
        rows = [row for row in csv.DictReader(source)]
    cases = [row.get('case_name') for row in rows]
    if not all(cases):
        raise ValueError('CSV contains a case with no name')
    if len(cases) != len(set(cases)):
        raise ValueError('Duplicate CSV case names detected')
    perf = [name for name in cases if name.startswith('TC_PF_')]
    if len(perf) != PERF_CASES:
        raise ValueError('Expected %d TC_PF cases, found %d' % (PERF_CASES, len(perf)))
    accuracy = [name for name in cases if not name.startswith('TC_PF_')]
    if len(accuracy) != ACCURACY_CASES:
        raise ValueError('Expected %d accuracy cases, found %d' % (ACCURACY_CASES, len(accuracy)))
    return cases


def validate_xml(xml, cases):
    root = ET.parse(xml).getroot()
    # Accuracy run excludes TC_PF: every accuracy CSV case maps to one
    # CsvDriven test plus the two dedicated TEST_F fixtures
    # (CsvCompositionAssert / ZeroingOrderedAfterInFlightGemm).
    expected_total = ACCURACY_CASES + 2
    if int(root.attrib.get('tests', -1)) != expected_total:
        raise ValueError('Unexpected test count: expected %d, got %s'
                         % (expected_total, root.attrib.get('tests')))
    if any(int(root.attrib.get(key, -1)) != 0 for key in ('failures', 'disabled', 'errors')):
        raise ValueError('Accuracy XML reports failures, disabled tests, or errors')
    seen = []
    for case in root.iter('testcase'):
        seen.append(case.attrib.get('name', '').split('/')[-1])
        if case.attrib.get('status') != 'run' or case.attrib.get('result') != 'completed':
            raise ValueError('An accuracy test did not complete')
    expected = set(cases) - {name for name in cases if name.startswith('TC_PF_')}
    expected |= {'CsvCompositionAssert', 'ZeroingOrderedAfterInFlightGemm'}
    if Counter(name for name in seen if name in expected) != Counter(sorted(expected)):
        raise ValueError('CSV coverage mismatch')


def accuracy_summary(log, returncode, csv_cases):
    text = log.read_text()
    summary = dict(tests=ACCURACY_CASES + 2, csv_cases=len(csv_cases), accuracy_cases=ACCURACY_CASES,
                  perf_cases=PERF_CASES, success_cases=SUCCESS_CASES, negative_cases=NEGATIVE_CASES,
                  failures=0, returncode=returncode)
    ratios = [float(value) for value in re.findall(r'\bmatchedRatio=([\d.eE+-]+)', text)]
    if not ratios:
        raise ValueError('No accuracy measurements for matchedRatio')
    summary['min_matched_ratio'] = min(ratios)
    max_abs = [float(value) for value in re.findall(r'\bmaxAbsErr=([\d.eE+-]+)', text)]
    if max_abs:
        summary['max_abs_error'] = max(max_abs)
    return summary


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    xml = args.output.resolve() / 'accuracy.xml'
    log = args.output / 'accuracy.log'
    cases = read_cases()
    with log.open('w') as output:
        result = subprocess.run([str(args.binary.resolve()), '--gtest_filter=-*TC_PF*',
                                 '--gtest_output=xml:' + str(xml)],
                                stdout=output, stderr=subprocess.STDOUT, timeout=3600)
    if result.returncode != 0:
        raise RuntimeError(f'Tests failed: see {log}')
    validate_xml(xml, cases)
    summary = accuracy_summary(log, result.returncode, cases)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    logging.info('%s', json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    main()
