#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""冒烟模式用例分级抽样（issue #457 全量耗时优化·方案2）。

背景：全量回归 6548 用例约 30 分钟；公共目录变更（如新增算子改
include/cann_ops_blas.h）必然触发 run_all，30 分钟硬超时。本脚本在
BLAS_TEST_MODE=smoke 时把每个算子的 CSV 用例做确定性抽样，
生成 BLAS_TEST_CASES 条目供 test_main.cpp 构造 gtest_filter。

抽样规则（通用，不依赖具体算子/前缀约定）：
  1. CSV 首列 case_name，按前缀分组（case_name 去掉末尾 _NNN 序号）；
  2. 负例（expect_result 列非 SUCCESS 值）全量保留；
  3. 组内按 case_name 排序后等距取样：小组（<=12）全保，
     大组保 max(4, size/8) 上限 24，边界样本（首/尾）必留；
  4. 找不到 CSV 或解析失败的算子跳过（回退全量，build.sh 不为其生成条目）。

用法：
    python3 smoke_sample.py <soc_arch> <op1> [op2 ...]
输出（stdout）：
    "op:case1,case2;op2:...;" 格式（BLAS_TEST_CASES 协议）。

CSV 与 C++ 读取器一致：接受 UTF-8 BOM，跳过空行和 # 注释，去除表头两端空白。
Host 回归：python3 -m unittest discover -s scripts/ci/tests -p test_csv_input_formats.py
"""

import csv
import glob
import os
import sys

# 组内保留上限与下限
SMALL_GROUP_KEEP = 12
LARGE_GROUP_KEEP = 24
LARGE_GROUP_STEP_DIVISOR = 8

SUCCESS_MARKERS = ("SUCCESS",)


def is_negative(row):
    expect = (row.get("expect_result") or row.get("ExpectResult") or "").upper()
    if not expect:
        return False
    return not any(marker in expect for marker in SUCCESS_MARKERS)


def group_prefix(case_name):
    # TC_EX_001 -> TC_EX；Ssymm_Basic_12 -> Ssymm_Basic
    idx = case_name.rfind("_")
    return case_name[:idx] if idx > 0 else case_name


def sample_group(names):
    """组内排序后等距取样，首尾必留，输出保持排序序。"""
    size = len(names)
    if size <= SMALL_GROUP_KEEP:
        return list(names)
    keep = max(4, min(LARGE_GROUP_KEEP, size // LARGE_GROUP_STEP_DIVISOR))
    step = max(1, size // keep)
    picked = set()
    for i in range(0, size, step):
        picked.add(names[i])
    picked.add(names[-1])  # 尾部边界（通常是大规模/极端参数）
    return sorted(picked)


def find_csv(op, soc_arch):
    # 优先当前 SOC 的 arch 子目录 CSV；退化到任意 arch/根 CSV
    patterns = [
        os.path.join("test", "**", soc_arch, f"{op}_test.csv"),
        os.path.join("test", "**", f"{op}_test.csv"),
    ]
    for pat in patterns:
        hits = sorted(glob.glob(pat, recursive=True))
        # 家族目录下同名 CSV 可能多个 arch，优先精确 arch 已在上面
        if hits:
            return hits[0]
    return None


def sample_op(op, soc_arch):
    csv_path = find_csv(op, soc_arch)
    if not csv_path:
        return None
    try:
        with open(csv_path, encoding="utf-8-sig", errors="ignore") as f:
            lines = (line for line in f if line.strip() and not line.lstrip().startswith("#"))
            reader = csv.DictReader(lines)
            if reader.fieldnames:
                reader.fieldnames = [name.strip() for name in reader.fieldnames]
            rows = list(reader)
    except OSError:
        return None
    if not rows:
        return None

    positive = {}
    negatives = []
    for row in rows:
        name = (row.get("case_name") or "").strip()
        if not name:
            continue
        if is_negative(row):
            negatives.append(name)
        else:
            positive.setdefault(group_prefix(name), []).append(name)

    picked = set(negatives)
    for prefix in sorted(positive):
        picked.update(sample_group(sorted(positive[prefix])))

    if not picked:
        return None
    return f"{op}:{','.join(sorted(picked))};"


def main():
    if len(sys.argv) < 3:
        print("usage: smoke_sample.py <soc_arch> <op> [op...]", file=sys.stderr)
        return 2
    soc_arch = sys.argv[1]
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    out = []
    for op in sys.argv[2:]:
        entry = sample_op(op, soc_arch)
        if entry:
            out.append(entry)
    sys.stdout.write("".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
