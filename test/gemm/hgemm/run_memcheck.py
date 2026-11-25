#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Run instrumented memory checks in separate processes for each Hgemm test case.

Every smoke case runs in its own mssanitizer process: the library-level
workspace (shared through the aclblas handle across launches inside one
process) is re-created per case, which keeps the sanitizer's heap-block
records scoped to a single launch and avoids cross-case reuse noise.

Every kernel type of this operator must actually be launched and
instrumented across the smoke group: the cube matmul, the AIV epilogue
over the FP32 workspace, and the AIV beta scaling used by the k=0 and
alpha=0 degenerate paths.
"""
import argparse
import csv
import json
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

REQUIRED_KERNELS = {'hgemm_kernel_fp16', 'hgemm_alpha_beta_kernel', 'hgemm_scale_kernel_fp16'}

HERE = Path(__file__).parent
DEFAULT_BINARY = HERE.parents[2] / 'build' / 'test' / 'gemm' / 'hgemm' / 'hgemm_test'
LOGGER = logging.getLogger('hgemm-memcheck')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, default=DEFAULT_BINARY)
    parser.add_argument('--tool', default='mssanitizer')
    parser.add_argument('--output', required=True, type=Path)
    return parser.parse_args()


def find_sanitizer(tool):
    candidate = os.environ.get('MSSANITIZER') or shutil.which(tool)
    if not candidate or not Path(candidate).is_file():
        return None
    return candidate


def is_smoke_case(name, m):
    # Success-path smoke coverage spans the nine transA/transB combinations,
    # LD padding rows, the alpha/beta scalar branches, the degenerate
    # alpha=0, beta=0 and k=0 paths, the K-tail sizes and the alpha/beta
    # FP32 workspace round-trip samples.
    k_tails = ('63', '64', '65', '127', '511', '512', '513', '1023', '1024', '1025')
    if name.startswith(('TC_L0_', 'TC_LD_', 'TC_AB_', 'TC_ED_')):
        return True
    if name.startswith('TC_MN_') and m in k_tails:
        return True
    return name.startswith('TC_EX_') and int(name.split('_')[2]) % 100 == 0


def select_cases():
    # contracts: dedicated fixtures - the CSV composition guard and the
    # stream-ordering zeroing regression (launches the cube kernel plus the
    # async memset path on the handle stream).
    # csv: all negative cases (host-side validation, no kernel launch).
    with (HERE / 'arch35/hgemm_test.csv').open() as source:
        negative = []
        smoke = []
        for row in csv.DictReader(source):
            name = row.get('case_name')
            if not name:
                raise ValueError('CSV case has no name')
            if row.get('expect_result') != 'ACLBLAS_STATUS_SUCCESS':
                negative.append(name)
            elif is_smoke_case(name, row.get('m')):
                smoke.append(name)
    return negative, smoke


def run_pattern(sanitizer, binary, pattern, expected_pass, env):
    with tempfile.TemporaryDirectory(prefix='hgemm_memcheck_') as workdir:
        log_path = Path(workdir) / 'mssanitizer.log'
        command = [sanitizer, '--tool=memcheck', '--leak-check=yes',
                   '--log-file=' + str(log_path), '--', str(binary), '--gtest_filter=' + pattern]
        result = subprocess.run(command, cwd=workdir, capture_output=True, text=True, env=env, timeout=3600)
        output = result.stdout + result.stderr
        log_text = log_path.read_text(encoding='utf-8', errors='replace') if log_path.is_file() else ''
        verdicts = output + log_text
    passed = re.findall(r'\[\s+PASSED\s*\]\s*(\d+)\s+tests?\.', verdicts)
    failed = re.findall(r'\[\s+FAILED\s*\]\s*(\d+)\s+tests?,', verdicts)
    bad_word = re.compile(r'\b(?:WARNING|ERROR)\b')
    diagnostics = []
    for line in verdicts.splitlines():
        if bad_word.search(line) and 'No error detected.' not in line:
            diagnostics.append(line)
    started = set(re.findall(r'Start memcheck sanitizer on kernel (\S+)', verdicts))
    finished_ok = re.findall(r'Sanitizer finished on kernel \S+\. No error detected\.', verdicts)
    ok = (result.returncode == 0 and passed and int(passed[0]) == expected_pass
          and not failed and not diagnostics and len(finished_ok) == len(started))
    return dict(passed=bool(ok), returncode=result.returncode, reported_tests=passed,
                kernels=sorted(started), diagnostics=diagnostics[:8])


def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s', stream=sys.stdout)
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    binary = args.binary.resolve()
    if not binary.is_file():
        LOGGER.error('[memcheck] binary not found: %s', binary)
        return 1
    sanitizer = find_sanitizer(args.tool)
    if sanitizer is None:
        LOGGER.error('[memcheck] mssanitizer not found; source the CANN set_env.sh or set MSSANITIZER')
        return 1

    negative, smoke = select_cases()
    env = os.environ.copy()
    groups = []

    # contracts: the two dedicated fixtures share one process; the zeroing
    # regression launches the cube kernel, exercising a fresh workspace.
    pattern = '*HgemmTest.CsvCompositionAssert:*HgemmTest.ZeroingOrderedAfterInFlightGemm'
    result = run_pattern(sanitizer, binary, pattern, 2, env)
    groups.append(dict(group='contracts', expected_tests=2, **result))
    LOGGER.info('[memcheck] group=contracts kernels=%s %s',
                ','.join(result['kernels']) or '-', 'ok' if result['passed'] else 'FAIL')

    # csv: validation-only negative cases, one process, no kernel launches.
    pattern = ':'.join('*' + name for name in negative)
    result = run_pattern(sanitizer, binary, pattern, len(negative), env)
    groups.append(dict(group='csv', expected_tests=len(negative), **result))
    LOGGER.info('[memcheck] group=csv expected=%d %s', len(negative), 'ok' if result['passed'] else 'FAIL')

    # smoke: one process per case - the aclblas handle workspace is
    # process-scoped, so each launch is checked against freshly recorded
    # heap blocks (no cross-case reuse of sanitizer block records).
    smoke_results = []
    started_kernels = set()
    for case in smoke:
        result = run_pattern(sanitizer, binary, '*' + case, 1, env)
        result['case'] = case
        smoke_results.append(result)
        started_kernels.update(result['kernels'])
        LOGGER.info('[memcheck] case=%s kernels=%s %s',
                    case, ','.join(result['kernels']) or '-', 'ok' if result['passed'] else 'FAIL')
    smoke_ok = all(item['passed'] for item in smoke_results)
    kernels_covered = REQUIRED_KERNELS.issubset(started_kernels)
    groups.append(dict(group='smoke', expected_tests=len(smoke), passed=bool(smoke_ok and kernels_covered),
                       checked_kernels=sorted(started_kernels), required_kernels=sorted(REQUIRED_KERNELS),
                       failed_cases=[item['case'] for item in smoke_results if not item['passed']]))

    clean = all(group.get('passed', False) for group in groups)
    summary = dict(passed=clean, groups=groups)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    LOGGER.info('[memcheck] totals: smoke_cases=%d kernels=%d required=%s %s',
                len(smoke), len(started_kernels), sorted(REQUIRED_KERNELS), 'PASS' if clean else 'FAIL')
    return 0 if clean else 1


if __name__ == '__main__':
    sys.exit(main())
