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
    echo "Usage: bash $0 /absolute/path/blas_all_test /absolute/report/directory [full|memcheck]" >&2
    echo "  full     : run the whole Cgemm arch35 CSV suite (golden comparison)" >&2
    echo "  memcheck : run the Cgemm suite under mssanitizer --tool=memcheck;" >&2
    echo "             the binary must come from a --cce-enable-sanitizer build" >&2
    exit 2
fi
test_binary=$(realpath "$1")
report_dir=$(realpath -m "$2")
mode=${3:-full}
mkdir -p "$report_dir"
command -v mssanitizer >/dev/null || true
cd "$(dirname "$test_binary")"
# Cgemm arch35 用例（含 317 条 m>128、最大 m=2048）落在 CgemmTest 套件内
GTEST_FILTER='*Cgemm*CsvDriven*'
case "$mode" in
    full)
        "$test_binary" --gtest_filter="$GTEST_FILTER" \
            --gtest_output="xml:$report_dir/full.xml" 2>&1 | tee "$report_dir/full.log"
        ;;
    memcheck)
        # Build this binary and its kernels with --cce-enable-sanitizer first
        # (see VALIDATION.md). Golden comparison cannot catch issue #458: the
        # pre-fix overflow landed silently in adjacent queue buffers. Only a
        # device memcheck over the m>128 cases proves the UB is gone.
        command -v mssanitizer >/dev/null
        scan_marker=$(mktemp "$report_dir/memcheck-start.XXXXXX")
        export GTEST_FILTER="$GTEST_FILTER"
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
