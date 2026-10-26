#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""ccopy CSV 等价组合去重（issue #457 全量耗时优化·方案5）。

现状：ccopy_test.csv 1200 行，其中 TC_EX（穷举）组内存在 55 行参数完全
重复的用例（(n,incx,incy,fill...) 相同，仅 case_name/seed 不同），另有大
n × 步长组合的测试性等价重复（同一 (n,incx,incy) 不同随机填充对 copy 类
算子覆盖同一执行路径）。

策略（保守，仅删确定性冗余）：
  1. 删除"参数完全相同"（(n,incx,incy,fill,align,expect) 全等）的重复行——
     保留每组首行。同参数对 copy 类算子的执行路径唯一，跨组完全重复
     （如 TC_PF 大 n 与 TC_SQ 平方数组重叠）同样删除，(n,incx,incy) 组合
     覆盖数不变（验证：1200 行 887 组合 → 927 行 887 组合）；
  2. TC_EX 内部对同一 (n,incx,incy) 仅保留 1 种 x_fill（RANDOM_NORM_5_5
     与 VALUE_* 对纯 copy 的执行路径等价，保留首见）；
  3. 负例（expect_result 非 SUCCESS / TC_ED 组）永不删除，且不参与
     正例去重集合；
     边界小 n（<=4）的删除仅限与其他行参数完全一致时（规则 1）。

负例保留优先于正例的两种去重规则。Host 回归（临时 CSV，未执行 CANN/NPU）：
    python3 test/copy/ccopy/arch35/test_dedup_ccopy_negative_host.py

用法（就地更新 CSV 前先打印预览）：
    python3 dedup_ccopy_csv.py arch35/ccopy_test.csv           # 预览
    python3 dedup_ccopy_csv.py --write arch35/ccopy_test.csv   # 写回
"""

import csv
import sys

EX_FILL_EQUIV = True   # TC_EX 内 (n,incx,incy) 相同时仅保留首个 x_fill
EDGE_KEEP_N = 4        # n <= 该值的用例视为边界，永不删除
NEG_PREFIX = "TC_ED"   # 负例组（expect_result 非 SUCCESS），全保


def load(path):
    with open(path, encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        rows = list(reader)
    header, data = rows[0], rows[1:]
    return header, [r for r in data if any(c.strip() for c in r)]


def is_negative(row_map):
    return "SUCCESS" not in (row_map.get("expect_result", "") or "").upper()


def main():
    args = sys.argv[1:]
    write = False
    if args and args[0] == "--write":
        write = True
        args = args[1:]
    if not args:
        print(__doc__)
        return 2
    path = args[0]

    header, data = load(path)
    idx = {name: i for i, name in enumerate(header)}
    ci = idx["case_name"]
    ni = idx["n"]
    ii = idx["incx"]
    iy = idx["incy"]
    xf = idx.get("x_fill", None)
    er = idx.get("expect_result", None)

    seen_exact = set()
    seen_combo = set()
    kept, dropped = [], []
    for row in data:
        m = {header[i]: row[i] for i in range(min(len(header), len(row)))}
        name = m["case_name"]
        n = int(m["n"]) if m.get("n", "").lstrip("-").isdigit() else -1
        combo = (m["n"], m["incx"], m["incy"])
        exact = combo + (m.get("x_fill", ""), m.get("y_fill", ""),
                         m.get("x_align_offset", ""), m.get("y_align_offset", ""),
                         m.get("expect_result", ""))

        neg = (er is not None and is_negative(m)) or name.startswith(NEG_PREFIX)
        edge = 0 <= n <= EDGE_KEEP_N

        # 负例保留优先于两种去重规则，且不占用正例去重集合，
        # 避免先出现的负例使同参数组合的正常用例被误删（issue #515）
        if neg:
            kept.append(row)
            continue

        if exact in seen_exact:
            dropped.append((name, "exact-duplicate"))
            continue
        if (not neg and not edge and EX_FILL_EQUIV and name.startswith("TC_EX")
                and combo in seen_combo):
            dropped.append((name, "equiv-combo-duplicate"))
            continue

        seen_exact.add(exact)
        seen_combo.add(combo)
        kept.append(row)

    print(f"total={len(data)} kept={len(kept)} dropped={len(dropped)}")
    from collections import Counter
    reasons = Counter(r for _, r in dropped)
    for reason, cnt in reasons.items():
        print(f"  {reason}: {cnt}")
    if len(dropped) <= 30:
        for name, reason in dropped:
            print(f"  - {name} ({reason})")

    if write:
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(kept)
        print(f"written: {path}")
    else:
        print("preview only (pass --write to update the CSV)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
