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
from collections import Counter
from pathlib import Path
import re
import subprocess


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    default_binary = Path(__file__).resolve().parents[3] / 'build-sanitize/test/rot/csrot/csrot_test'
    parser.add_argument('--binary', type=Path, default=default_binary)
    parser.add_argument('--tool', default='mssanitizer')
    parser.add_argument('--output', required=True, type=Path)
    return parser.parse_args()


def test_groups():
    # 成功路径分组（reviewer 要求：必须让 kernel 真正被插桩，checked_kernels>0 为硬约束）。
    #   csrot 分发：incx==1 && incy==1 → csrot_aiv_kernel（membase 连续路径）；
    #   否则 → csrot_simt_kernel（stride 路径）。required 强制两 kernel 都启动且干净结束。
    #   - TC_PF_1001 / TC_PF_1200：TC_PF 首/尾（incx=incy=1 → aiv）。
    #   - TC_L0_001：incx=incy=1 小尺寸（aiv）。
    #   - TC_L0_002：incx=incy=2 小尺寸（simt）。
    #   - TC_L0_003：incx=incy=-1 小尺寸（simt，负步长）。
    #
    # contracts / csv 两组的用例均在 host/wrapper 预校验阶段返回（空句柄、n<=0 快速返回、
    # 空 c/s、步长溢出 INVALID_VALUE），不 launch kernel；其 required 为空，仅做 fail-on-diagnostic。
    with (Path(__file__).parent / 'arch35/csrot_test.csv').open() as source:
        negative = []
        for row in csv.DictReader(source):
            if row.get('expect_result') == 'ACLBLAS_STATUS_SUCCESS':
                continue
            name = row.get('case_name')
            if not name:
                raise ValueError('Negative CSV case has no name')
            negative.append(name)
    kernel_names = {'csrot_aiv_kernel', 'csrot_simt_kernel'}
    return [
        ('contracts', ['CsrotTest.NullHandle', 'CsrotTest.NullHandleTakesPrecedenceOverQuickReturn',
                       'CsrotTest.QuickReturnSkipsDataPointerValidation', 'CsrotTest.CsvCompositionAssert'],
         set()),
        ('csv', ['*' + name for name in negative], set()),
        ('smoke', ['*TC_PF_1001', '*TC_PF_1200', '*TC_L0_001', '*TC_L0_002', '*TC_L0_003'], kernel_names),
    ]


def check_group(name, selected, returncode, text, required):
    passed = re.findall(r'\[\s*PASSED\s*\]\s*(\d+) tests?\.', text)
    diagnostics = []
    for line in text.splitlines():
        if re.search(r'\b(?:WARNING|ERROR)\b', line) and 'No error detected.' not in line:
            diagnostics.append(line)
    started = Counter(re.findall(r'Start memcheck sanitizer on kernel (\S+)', text))
    finished = Counter(re.findall(r'Sanitizer finished on kernel (\S+)\. No error detected\.', text))
    kernels_ok = required.issubset(set(started)) and started == finished
    clean = (returncode == 0 and passed and all(int(value) == len(selected) for value in passed)
             and kernels_ok and not diagnostics)
    return dict(group=name, passed=bool(clean), expected_tests=len(selected), reported_tests=passed,
                returncode=returncode, checked_kernels=sum(started.values()),
                required_kernels=sorted(required), started_kernels=dict(started), diagnostics=diagnostics)


def run_group(args, name, selected, env, required):
    env['GTEST_FILTER'] = ':'.join(selected)
    log = args.output / (name + '.log')
    detail = args.output.resolve() / (name + '_detail.log')
    command = [args.tool, '--tool=memcheck', '--leak-check=yes', '--log-file=' + str(detail),
               str(args.binary.resolve())]
    with log.open('w') as output:
        result = subprocess.run(command, env=env, stdout=output, stderr=subprocess.STDOUT, timeout=3600)
    text = log.read_text() + ('\n' + detail.read_text() if detail.exists() else '')
    return check_group(name, selected, result.returncode, text, required)


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.output.chmod(0o750)
    # The framework resets ACL between suites. An isolated process prevents the
    # tool from reusing an instrumented function handle after that reset.
    env = os.environ.copy()
    env.pop('CSROT_PROFILE', None)
    checks = [run_group(args, name, selected, env, required) for name, selected, required in test_groups()]
    clean = all(check.get('passed', False) for check in checks)
    summary = dict(passed=clean, expected_tests=sum(check.get('expected_tests', 0) for check in checks),
                   checked_kernels=sum(check.get('checked_kernels', 0) for check in checks), groups=checks)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    logging.info('%s', json.dumps(summary, ensure_ascii=False))
    return not clean


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    raise SystemExit(main())
