#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -euo pipefail
if [ "$#" -lt 2 ]; then
    echo "Usage: bash $0 /absolute/path/cher2k_test /absolute/report/directory [full|guards|perf|memcheck]" >&2
    exit 2
fi
test_binary=$(realpath "$1")
report_dir=$(realpath -m "$2")
mode=${3:-full}
mkdir -p "$report_dir"
unset CHER2K_SKIP_GOLDEN CHER2K_PERF_ITERATIONS CHER2K_PERF_WARMUP
export CHER2K_FULL_REGRESSION=1
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-8}
cd "$(dirname "$test_binary")"
case "$mode" in
    full)
        "$test_binary" --gtest_output="xml:$report_dir/full.xml" 2>&1 | tee "$report_dir/full.log"
        ;;
    guards)
        "$test_binary" --gtest_filter='*ScalarTransitions*' \
            --gtest_output="xml:$report_dir/guards.xml" 2>&1 | tee "$report_dir/guards.log"
        ;;
    perf)
        export CHER2K_PERF_WARMUP=10 CHER2K_PERF_ITERATIONS=60
        "$test_binary" --gtest_filter='*TC_PF_*' \
            --gtest_output="xml:$report_dir/perf.xml" 2>&1 | tee "$report_dir/perf.log"
        ;;
    memcheck)
        # Build this binary and its kernels with --cce-enable-sanitizer first.
        command -v mssanitizer >/dev/null
        # 旧产物必须先清除：新一轮未落报告时，残留的旧 XML/日志
        # 会通过下方 -s 与 -nt 新鲜度校验，掩盖本次运行失败（#1133）
        rm -f "$report_dir/memcheck.xml" "$report_dir/memcheck-report.log"
        scan_marker=$(mktemp "$report_dir/memcheck-start.XXXXXX")
        export GTEST_FILTER='*MemorySanitizerPaths'
        export GTEST_OUTPUT="xml:$report_dir/memcheck.xml"
        mssanitizer --tool=memcheck --log-file="$report_dir/memcheck-report.log" \
            "$test_binary" 2>&1 | tee "$report_dir/memcheck.log"
        test -s "$report_dir/memcheck.xml"
        test -s "$report_dir/memcheck-report.log"
        test "$report_dir/memcheck.xml" -nt "$scan_marker"
        grep -q 'No error detected.' "$report_dir/memcheck-report.log"
        findings=$(grep -Ei 'ERROR:|WARNING:' "$report_dir/memcheck-report.log" || true)
        if [ -n "$findings" ]; then
            printf '%s\n' "$findings" >&2
            echo "Device sanitizer findings require review." >&2
            exit 1
        fi
        ;;
    *) echo "Unknown validation mode: $mode" >&2; exit 2 ;;
esac
