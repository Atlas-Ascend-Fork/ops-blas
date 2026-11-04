#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Run the complete Csrot accuracy suite and reject missing/duplicate CSV cases."""
import argparse
from collections import Counter
import csv
import json
import logging
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    default_binary = Path(__file__).resolve().parents[3] / 'build/test/rot/csrot/csrot_test'
    parser.add_argument('--binary', type=Path, default=default_binary)
    parser.add_argument('--output', required=True, type=Path)
    return parser.parse_args()


def read_cases():
    with (Path(__file__).parent / 'arch35/csrot_test.csv').open() as source:
        cases = [row.get('case_name') for row in csv.DictReader(source)]
    if not all(cases):
        raise ValueError('CSV contains a case with no name')
    if len(cases) != len(set(cases)):
        raise ValueError('Duplicate CSV case names detected')
    return cases


def validate_xml(xml, cases):
    root = ET.parse(xml).getroot()
    # 全量运行：每条 CSV 用例一个 CsvDriven 测试 + 4 条专项 TEST_F（NulLHandle /
    # NulLHandleTakesPrecedenceOverQuickReturn / QuickReturnSkipsDataPointerValidation /
    # CsvCompositionAssert）。总数随 CSV 用例数变化，故不硬编码。
    expected_total = len(cases) + 4
    if int(root.attrib.get('tests', -1)) != expected_total:
        raise ValueError('Unexpected test count: expected %d' % expected_total)
    if any(int(root.attrib.get(key, -1)) != 0 for key in ('failures', 'disabled', 'errors')):
        raise ValueError('Accuracy XML reports failures, disabled tests, or errors')
    seen = []
    for case in root.iter('testcase'):
        seen.append(case.attrib.get('name', '').split('/')[-1])
        if case.attrib.get('status') != 'run' or case.attrib.get('result') != 'completed':
            raise ValueError('An accuracy test did not complete')
    expected = set(cases)
    if Counter(name for name in seen if name in expected) != Counter(cases):
        raise ValueError('CSV coverage mismatch')


def accuracy_summary(log, returncode, csv_cases):
    text = log.read_text()
    summary = dict(tests=len(csv_cases) + 4, csv_cases=len(csv_cases), failures=0, returncode=returncode)
    for metric in ('MERE', 'MARE'):
        values = [float(value) for value in re.findall(r'\b' + metric + r'=([\d.eE+-]+)', text)]
        if not values:
            raise ValueError('No accuracy measurements for ' + metric)
        summary[metric.lower()] = max(values)
    return summary


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    xml = args.output.resolve() / 'accuracy.xml'
    log = args.output / 'accuracy.log'
    cases = read_cases()
    with log.open('w') as output:
        result = subprocess.run([str(args.binary.resolve()), '--gtest_output=xml:' + str(xml)],
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
