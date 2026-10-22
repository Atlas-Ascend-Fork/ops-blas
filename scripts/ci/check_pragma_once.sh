#!/bin/bash
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
# CI lint 脚本：检查 PR 中新增/修改的 .h/.hpp 文件是否使用 #pragma once
#
# 用法（与 run_example.sh 一致，接收 PR 文件列表）：
#   bash scripts/ci/check_pragma_once.sh <pr_filelist.txt>
#
# pr_filelist.txt 格式：每行一个文件路径（相对仓库根目录），例如：
#   blas/scal/sscal/arch35/sscal_tiling_data.h
#   test/frame/blas_test.h
#   include/cann_ops_blas.h
#
# 仅检查 .h/.hpp 文件，其他文件类型自动跳过。
#
# 退出码：
#   0 — 所有头文件通过检查
#   1 — 存在使用 #ifndef include guard 或缺少 guard 的头文件

set -euo pipefail

ROOT_PATH=$(cd "$(dirname "$0")"/../.. && pwd)
cd "${ROOT_PATH}"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <pr_filelist.txt>"
    exit 1
fi

if [ "$1" = "--files" ]; then
    shift
    pr_filelist=$(mktemp)
    trap 'rm -f "$pr_filelist"' EXIT
    printf '%s\n' "$@" > "$pr_filelist"
else
    pr_filelist=$1
fi

if [ ! -f "$pr_filelist" ]; then
    echo "Error: File $pr_filelist not found"
    exit 1
fi

failed=0
checked=0

while IFS= read -r filepath || [ -n "$filepath" ]; do
    # 跳过空行
    [ -z "$filepath" ] && continue

    # 仅检查 .h / .hpp 文件
    case "$filepath" in
        *.h|*.hpp) ;;
        *) continue ;;
    esac

    # 跳过第三方/工具目录
    case "$filepath" in
        .opencode/*|asc-devkit/*|cann-recipes-infer/*|tilelang-ascend/*|msprof_trsm*|dev-doc/*|build/*|out/*)
            continue
            ;;
    esac

    # 文件不存在（可能是被删除的文件），跳过
    [ -f "$filepath" ] || continue

    checked=$((checked + 1))

    # 检查是否使用 #pragma once
    if grep -q '^#pragma once' "$filepath"; then
        continue
    fi

    # 使用 #ifndef include guard
    if grep -q '^#ifndef' "$filepath"; then
        echo "ERROR: $filepath uses #ifndef include guard. Please replace with #pragma once."
        failed=1
        continue
    fi

    # 完全没有 guard
    echo "ERROR: $filepath is missing #pragma once. Please add it after the copyright header."
    failed=1
done < "$pr_filelist"

echo "[LINT] Checked $checked header file(s) from PR."

if [ $failed -ne 0 ]; then
    echo ""
    echo "Include guard check failed. All .h/.hpp files must use '#pragma once'."
    echo "See Issue #162 for details."
    exit 1
fi

echo "[LINT] Include guard check passed."
exit 0
