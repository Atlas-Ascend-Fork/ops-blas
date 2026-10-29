#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Run instrumented memory checks in separate processes for each ACL test suite."""
import argparse
import csv
import json
import logging
import os
from pathlib import Path
import re
import subprocess


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    default_binary = Path(__file__).resolve().parents[3] / 'build-sanitize/test/syr2/cher2/cher2_test'
    parser.add_argument('--binary', type=Path, default=default_binary)
    parser.add_argument('--tool', default='mssanitizer')
    parser.add_argument('--output', required=True, type=Path)
    return parser.parse_args()


def test_groups():
    with (Path(__file__).parent / 'arch35/cher2_test.csv').open() as source:
        negative = []
        for row in csv.DictReader(source):
            if row.get('expect_result') == 'ACLBLAS_STATUS_SUCCESS':
                continue
            name = row.get('case_name')
            if not name:
                raise ValueError('Negative CSV case has no name')
            negative.append(name)
    return [
        ('contracts', ['Cher2Test.NullHandle', 'Cher2Test.NormalAndSpecialScalars',
                       'Cher2Test.MixedSpecialTilePaths']),
        ('csv', ['*TC_PF_1001', '*TC_PF_1002', '*TC_PF_1004'] + ['*' + name for name in negative]),
    ]


def check_group(name, selected, returncode, text):
    passed = re.findall(r'\[\s*PASSED\s*\]\s*(\d+) tests?\.', text)
    diagnostics = []
    for line in text.splitlines():
        if re.search(r'\b(?:WARNING|ERROR)\b', line) and 'No error detected.' not in line:
            diagnostics.append(line)
    starts = text.count('Start memcheck sanitizer on kernel cher2_kernel')
    finished = text.count('Sanitizer finished on kernel cher2_kernel. No error detected.')
    clean = (returncode == 0 and passed and all(int(value) == len(selected) for value in passed)
             and starts > 0 and starts == finished and not diagnostics)
    return dict(group=name, passed=bool(clean), expected_tests=len(selected), reported_tests=passed,
                returncode=returncode, checked_kernels=finished, diagnostics=diagnostics)


def run_group(args, name, selected, env):
    env['GTEST_FILTER'] = ':'.join(selected)
    log = args.output / (name + '.log')
    detail = args.output.resolve() / (name + '_detail.log')
    command = [args.tool, '--tool=memcheck', '--leak-check=yes', '--log-file=' + str(detail),
               str(args.binary.resolve())]
    with log.open('w') as output:
        result = subprocess.run(command, env=env, stdout=output, stderr=subprocess.STDOUT, timeout=3600)
    text = log.read_text() + ('\n' + detail.read_text() if detail.exists() else '')
    return check_group(name, selected, result.returncode, text)


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.output.chmod(0o750)
    # The framework resets ACL between suites. An isolated process prevents the
    # tool from reusing an instrumented function handle after that reset.
    env = os.environ.copy()
    env.pop('CHER2_PROFILE', None)
    checks = [run_group(args, name, selected, env) for name, selected in test_groups()]
    clean = all(check.get('passed', False) for check in checks)
    summary = dict(passed=clean, expected_tests=sum(check.get('expected_tests', 0) for check in checks),
                   checked_kernels=sum(check.get('checked_kernels', 0) for check in checks), groups=checks)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    logging.info('%s', json.dumps(summary, ensure_ascii=False))
    return not clean


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    raise SystemExit(main())
