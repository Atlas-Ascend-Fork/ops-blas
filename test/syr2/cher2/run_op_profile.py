#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Collect Cher2 msprof op samples in batches below the profiler's export limit."""
import argparse
import csv
import json
import logging
import os
from pathlib import Path
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tool', required=True, help='Path to the msopprof binary (the msprof op backend)')
    default_binary = Path(__file__).resolve().parents[3] / 'build/test/syr2/cher2/cher2_test'
    parser.add_argument('--binary', type=Path, default=default_binary)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--samples', type=int, default=10, choices=(10, 51))
    return parser.parse_args()


def read_cases():
    with (Path(__file__).parent / 'arch35/cher2_test.csv').open() as source:
        cases = []
        for row in csv.DictReader(source):
            name = row.get('case_name', '')
            if name.startswith('TC_PF_'):
                cases.append(name)
    if len(cases) != 200 or len(set(cases)) != 200:
        raise ValueError('Expected 200 unique performance CSV cases')
    return cases


def collect_batch(args, names, index, env):
    batch_name = f'batch_{index:02d}'
    profile = args.output.resolve() / batch_name
    log = args.output.resolve() / (batch_name + '.log')
    command = [args.tool, '--aic-metrics=BasicInfo', '--kernel-name=cher2_kernel',
               '--launch-count=' + str(len(names) * args.samples), '--warm-up=5',
               '--output=' + str(profile), str(args.binary.resolve()),
               '--gtest_filter=' + ':'.join('*' + name for name in names)]
    logging.info('%s: %s cases, %s samples', batch_name, len(names), len(names) * args.samples)
    with log.open('w') as output:
        result = subprocess.run(command, env=env, stdout=output, stderr=subprocess.STDOUT, timeout=3600)
    return dict(profile=batch_name, log=batch_name + '.log', cases=names,
                command=command, returncode=result.returncode)


def main():
    args = parse_args()
    if args.output.exists():
        raise ValueError('Use a new output directory')
    args.output.mkdir(parents=True)
    cases = read_cases()
    version = subprocess.run([args.tool, '--version'], check=True, text=True, capture_output=True).stdout
    env = os.environ.copy()
    env['CHER2_PROFILE'] = 'op'
    env['CHER2_PROFILE_SAMPLES'] = str(args.samples)
    batch_size = 1000 // args.samples
    batches = []
    manifest = dict(tool=args.tool, version=version, samples=args.samples, warmup=5, batches=batches)
    for offset in range(0, len(cases), batch_size):
        batch = collect_batch(args, cases[offset:offset + batch_size], offset // batch_size, env)
        batches.append(batch)
        (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        if batch.get('returncode') != 0:
            raise RuntimeError(f'Collection failed: {args.output / batch.get("log", "")}')
    summary = Path(__file__).with_name('summarize_op_profile.py')
    subprocess.run([str(Path(sys.executable).resolve()), str(summary),
                    '--manifest', str(args.output / 'manifest.json'),
                    '--output', str(args.output / 'performance.csv')], check=True)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    main()
