#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Generate cher2 test CSV: A = alpha*x*y^H + conjg(alpha)*y*x^H + A
（complex64，A 为 n×n Hermitian 矩阵，仅引用/更新 uplo 三角，对角虚部更新后置 0；alpha 为复数）

列格式约定（仓内无复数 cher2 CSV 工程，按同族 test/syr2/ 的 ReadMap 列名扩展）：
  基础结构对齐 test/syr2/ssyr2/ssyr2_param.h（uplo/n/alpha/x/incx/y/incy/a/lda），
  复数标量按 test/chemm/chemm_param.h 惯例拆为 alpha_real/alpha_imag，
  张量输入统一用 <参数>_fill 列（NULLPTR 表示空指针），标量空指针写字面量 null。
  case_name,description,uplo,n,alpha_real,alpha_imag,x_fill,incx,y_fill,incy,a_fill,lda,
  expect_result,mere_threshold,mare_multiplier,random_seed

任务书事实（1028_work/01_task_docs/1022_Atlas950PR/aclblasCher2/aclblasCher2_task_doc.md）：
  §2.3 签名: (handle, uplo, n, alpha, x, incx, y, incy, A, lda)，alpha 为 const aclblasComplex*
  §2.4 uplo 非法 → INVALID_ENUM；n<0 / incx=0 / incy=0 / lda<max(1,n) / alpha 指针空 → INVALID_VALUE
  §2.5 校验顺序: 先完成结构参数校验（n≥0、uplo、incx、incy、lda、alpha 指针），
       校验通过后 n=0 或 alpha=(0,0) 为合法 no-op（不访问 x/y/A）；
       n>0 且 alpha≠(0,0) 时 x/y/A 不得为空
  §3.3 典型性能 case: (512,UPPER) (1024,LOWER) (2048,UPPER) (4096,LOWER)，alpha=(1,0)，incx=incy=1，lda=n

用法（条数可灵活扩展）：
  python3 gen_csv.py                                       # 默认 1000 精度 + 200 性能
  python3 gen_csv.py --accuracy 1500 --perf 300
  python3 gen_csv.py --seed 20261028 --out /tmp/a.csv --baseline-out /tmp/b.csv

输入分布说明：当前 ops-blas 测试工程仅支持均匀分布（RANDOM_NORM_5_5 = [-5,5] 均匀）。
生态精度标准要求的 50% 正态分布依赖测试工程扩展（另行任务），扩展完成后通过 --dist mixed 切换。
"""

import argparse
import csv
import logging
import os
import random
import sys
from dataclasses import dataclass
from typing import Optional

# ---- 尺寸池 ----
SIZES = [1, 2, 3, 5, 7, 10, 12, 15, 16, 24, 32, 48, 64, 65, 96, 128, 200, 256, 400, 512, 800, 1024, 2048]
MED = [10, 12, 15, 24, 48, 96, 200, 400]
UPLO = ["ACLBLAS_UPPER", "ACLBLAS_LOWER"]
INCS = [1, 2, 3, -1, -2, -3]
# alpha 为复数 (real, imag)；含 (0,0) quick return、纯实/纯虚/一般复数/大值/共轭对
ALPHAS = [(1, 0), (0, 0), (0.5, 0.5), (0, 1), (-1, 0), (1, 1), (1000, 1000), (0.5, -0.5)]
FILL_RAND = "RANDOM_NORM_5_5"  # [-5,5] 均匀分布（生态标准值域；正态分布待测试工程扩展）
FILLS = [
    ("RANDOM_NORM_5_5", "rand"),
    ("VALUE_NORM_0", "zero"),
    ("RANDOM_ALTER", "alter"),
    ("RANDOM_EXTREME", "extreme"),
    ("VALUE_NORM_INF", "inf"),
    ("VALUE_NORM_NAN", "nan"),
]
MERE = "0.00012207031"  # 2^-13，与 ops-blas 仓测试工程惯例一致
MARE = "10.0"
MAXN = 4096  # 性能用例方阵上限：complex64 方阵 host 预算 512MB（n*n*8B ≤ 512MB）
INT_MIN = -2147483648

SUCCESS = "ACLBLAS_STATUS_SUCCESS"
INV = "ACLBLAS_STATUS_INVALID_VALUE"
ENUM = "ACLBLAS_STATUS_INVALID_ENUM"

# 任务书 §3.3 典型 case（逐参数精确一致，必须保持同步）
DOC_CASES = [("ACLBLAS_UPPER", 512), ("ACLBLAS_LOWER", 1024), ("ACLBLAS_UPPER", 2048), ("ACLBLAS_LOWER", 4096)]


@dataclass
class CaseOptions:
    incx: int = 1
    incy: int = 1
    exp: str = SUCCESS
    lda: Optional[int] = None
    af: str = FILL_RAND
    xf: str = FILL_RAND
    yf: str = FILL_RAND
    mere: str = MERE
    mare: str = MARE


class Gen:
    def __init__(self, seed):
        self.rows = []
        self.idx = 0
        self.rng = random.Random(seed)

    def add(self, nm, ds, uplo, n, ar=1, ai=0, **kwargs):
        options = CaseOptions(**kwargs)
        self.idx += 1
        self.rows.append(
            {
                "case_name": nm,
                "description": ds,
                "uplo": uplo,
                "n": n,
                "alpha_real": ar,
                "alpha_imag": ai,
                "x_fill": options.xf,
                "incx": options.incx,
                "y_fill": options.yf,
                "incy": options.incy,
                "a_fill": options.af,
                "lda": (max(1, n) if options.lda is None else options.lda),
                "expect_result": options.exp,
                "mere_threshold": options.mere,
                "mare_multiplier": options.mare,
                "random_seed": 20260000 + self.idx,
            }
        )

    # ---- 精度用例各类别 ----
    def l0_basic(self):
        for u in UPLO:
            for sz in [4, 8]:
                self.add(self._nm("TC_L0"), f"basic_{u.split('_')[-1]}_n{sz}", u, sz)

    def l1_size(self):
        for u in UPLO:
            for sz in SIZES:
                self.add(self._nm("TC_SQ"), f"sq_n{sz}_{u.split('_')[-1]}", u, sz)

    def l2_scalar(self):
        for ar, ai in ALPHAS:
            for u in UPLO:
                self.add(self._nm("TC_AB"), f"alpha_{ar}_{ai}_{u.split('_')[-1]}", u, 32, ar, ai)

    def l4_ld_inc(self):
        # 前导维 padding：lda = n+1 / n+8 / n+16
        for u in UPLO:
            for sz in [16, 32, 64]:
                for pad in [1, 8, 16]:
                    self.add(self._nm("TC_LD"), f"ld_{sz}_pad{pad}_{u.split('_')[-1]}", u, sz, lda=sz + pad)
        # 步长全组合（含负步长），UPPER 与 LOWER 各一组
        for u in UPLO:
            for ix in INCS:
                for iy in INCS:
                    self.add(self._nm("TC_INC"), f"inc_{ix}_{iy}_{u.split('_')[-1]}", u, 64, incx=ix, incy=iy)

    def l5_fill(self):
        for f, l in FILLS:
            self.add(self._nm("TC_FL"), f"fill_a_{l}", "ACLBLAS_UPPER", 32, af=f)
            self.add(self._nm("TC_FL"), f"fill_x_{l}", "ACLBLAS_LOWER", 32, xf=f)
            self.add(self._nm("TC_FL"), f"fill_y_{l}", "ACLBLAS_LOWER", 32, yf=f)

    def l5b_cover(self):
        for sz in MED:
            for u in UPLO:
                self.add(self._nm("TC_CV"), f"cv_{sz}_{u.split('_')[-1]}", u, sz)

    def l6_edge(self):
        upper, lower = UPLO
        # --- no-op（§2.5：结构参数先校验，通过后 n=0 或 alpha=(0,0) 才算 no-op） ---
        self.add(self._nm("TC_ED"), "n0_upper", upper, 0)
        self.add(self._nm("TC_ED"), "n0_lower", lower, 0)
        self.add(self._nm("TC_ED"), "n0_null_xyz", upper, 0, af="NULLPTR", xf="NULLPTR", yf="NULLPTR")
        self.add(self._nm("TC_ED"), "n0_invalid_uplo", "999", 0, exp=ENUM)
        self.add(self._nm("TC_ED"), "n0_incx0", upper, 0, incx=0, exp=INV)
        self.add(self._nm("TC_ED"), "n0_incy0", lower, 0, incy=0, exp=INV)
        self.add(self._nm("TC_ED"), "n0_null_alpha", upper, 0, ar="null", exp=INV)
        self.add(self._nm("TC_ED"), "n0_lda_zero", upper, 0, lda=0, exp=INV)
        self.add(self._nm("TC_ED"), "n1_upper", upper, 1)
        self.add(self._nm("TC_ED"), "n1_lower", lower, 1, ar=-1, ai=1)
        # alpha=(0,0)：合法 no-op，不访问 x/y/A（允许为空）
        self.add(self._nm("TC_ED"), "alpha0_null_xyz", upper, 32, 0, 0, af="NULLPTR", xf="NULLPTR", yf="NULLPTR")
        self.add(self._nm("TC_ED"), "alpha0_lower_noop", lower, 32, 0, 0)
        # alpha=(0,0) 但结构参数非法：仍返回结构错误码
        self.add(self._nm("TC_ED"), "alpha0_invalid_lda", upper, 32, 0, 0, lda=8, exp=INV)
        self.add(self._nm("TC_ED"), "alpha0_incx0", lower, 32, 0, 0, incx=0, exp=INV)
        # --- 负向：n<0 ---
        self.add(self._nm("TC_ED"), "neg_n", upper, -1, exp=INV)
        self.add(self._nm("TC_ED"), "neg_n_invalid_uplo", "999", -1, lda=0, incx=0, exp=INV)
        # --- 负向：空指针（标量 "null" / 张量 NULLPTR） ---
        self.add(self._nm("TC_ED"), "null_alpha", upper, 16, ar="null", exp=INV)
        self.add(self._nm("TC_ED"), "null_x", upper, 16, xf="NULLPTR", exp=INV)
        self.add(self._nm("TC_ED"), "null_y", lower, 16, yf="NULLPTR", exp=INV)
        self.add(self._nm("TC_ED"), "null_A", upper, 16, af="NULLPTR", exp=INV)
        # --- 负向：非法枚举（999）；uplo 校验早于 alpha 指针校验 ---
        self.add(self._nm("TC_ED"), "invalid_uplo", "999", 16, exp=ENUM)
        self.add(self._nm("TC_ED"), "invalid_uplo_null_alpha", "999", 16, ar="null", exp=ENUM)
        # --- 负向：非法前导维 ---
        self.add(self._nm("TC_ED"), "invalid_lda", upper, 16, lda=8, exp=INV)
        self.add(self._nm("TC_ED"), "lda_zero", upper, 16, lda=0, exp=INV)
        self.add(self._nm("TC_ED"), "lda_negative", lower, 16, lda=-1, exp=INV)
        self.add(self._nm("TC_ED"), "invalid_lda_and_incy0", upper, 16, lda=8, incy=0, exp=INV)
        # --- 负向：零步长与 INT_MIN 步长（|INT_MIN| 溢出长度估算，按仓内惯例判非法） ---
        self.add(self._nm("TC_ED"), "incx0", upper, 16, incx=0, exp=INV)
        self.add(self._nm("TC_ED"), "incy0", lower, 16, incy=0, exp=INV)
        self.add(self._nm("TC_ED"), "incx_intmin", upper, 16, incx=INT_MIN, exp=INV)
        self.add(self._nm("TC_ED"), "incy_intmin", lower, 16, incy=INT_MIN, exp=INV)
        # --- 特殊标量 ---
        self.add(self._nm("TC_ED"), "large_alpha", upper, 32, 1000, 1000)
        self.add(self._nm("TC_ED"), "pure_imag_alpha", lower, 32, 0, -2)

    def ex_extend(self, target):
        """TC_EX 扩展精度用例：确定性采样 尺寸×uplo×alpha×步长×前导维，补足到 target 条"""
        pool = [
            1,
            2,
            3,
            4,
            5,
            7,
            8,
            9,
            15,
            16,
            17,
            31,
            32,
            33,
            63,
            64,
            65,
            127,
            128,
            129,
            255,
            256,
            257,
            511,
            512,
            513,
            1000,
            1024,
            1025,
            2000,
            2048,
        ]
        i = 0
        while len([r for r in self.rows if not r["case_name"].startswith("TC_PF")]) < target:
            n = pool[self.rng.randrange(len(pool))]
            u = UPLO[i % len(UPLO)]
            ar, ai = ALPHAS[i % len(ALPHAS)]
            ix = INCS[self.rng.randrange(len(INCS))] if i % 3 == 0 else 1
            iy = INCS[self.rng.randrange(len(INCS))] if i % 3 == 1 else 1
            lda = n + 8 if i % 5 == 4 else None
            self.add(self._nm("TC_EX"), f"ex_{n}_{u.split('_')[-1]}", u, n, ar, ai, incx=ix, incy=iy, lda=lda)
            i += 1

    # ---- 性能用例 ----
    def perf(self, target):
        # 任务书 §3.3 的 4 条典型 case（逐参数精确一致：alpha=(1,0)、incx=incy=1、lda=n）
        for u, n in DOC_CASES:
            self.add(self._nm("TC_PF"), f"pf_doc_n{n}_{u.split('_')[-1]}", u, n, 1, 0)
        # 小尺寸延迟敏感区
        for sz in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
            self.add(self._nm("TC_PF"), f"pf_small_{sz}", "ACLBLAS_UPPER", sz, 1, 0)
        # 方阵规模对数扫描（512MB 预算内：n*n*8B ≤ 512MB）
        n_sq = min(60, target // 3)
        for i in range(n_sq):
            sz = int(round(1000 * (MAXN / 1000) ** (i / max(n_sq - 1, 1))))
            if sz * sz * 8 > 512 * 1024**2:
                continue
            u = UPLO[i % 2]
            self.add(self._nm("TC_PF"), f"pf_sq_{sz}_{u.split('_')[-1]}", u, sz, 1, 0)
        # 补足到 target：预算内随机尺寸（纯连续访存：incx=incy=1，lda 紧凑）
        while len([r for r in self.rows if r["case_name"].startswith("TC_PF")]) < target:
            n = self.rng.randrange(16, MAXN + 1)
            if n * n * 8 > 512 * 1024**2:
                continue
            u = UPLO[self.rng.randrange(2)]
            self.add(self._nm("TC_PF"), f"pf_mix_{n}_{u.split('_')[-1]}", u, n, 1, 0)

    def _nm(self, cat):
        return f"{cat}_{self.idx + 1:03d}" if cat != "TC_EX" else f"{cat}_{self.idx + 1:04d}"


FIELDS = [
    "case_name",
    "description",
    "uplo",
    "n",
    "alpha_real",
    "alpha_imag",
    "x_fill",
    "incx",
    "y_fill",
    "incy",
    "a_fill",
    "lda",
    "expect_result",
    "mere_threshold",
    "mare_multiplier",
    "random_seed",
]

# gpu_baseline 判定维度：任务书 §3.3 表格列（n/uplo/alpha/incx/incy/lda）
BASELINE_COLS = ["uplo", "n", "alpha_real", "alpha_imag", "incx", "incy", "lda"]


def validate_performance_rows(pf_rows):
    # 性能用例纯连续校验：非单位步长/非紧凑前导维会污染性能与内存基线
    for r in pf_rows:
        if not (r["incx"] == 1 and r["incy"] == 1):
            raise RuntimeError(f"性能用例含非单位步长: {r['case_name']}")
        if not (r["lda"] == max(1, r["n"])):
            raise RuntimeError(f"性能用例含非紧凑前导维: {r['case_name']}")
        if not ((r["alpha_real"], r["alpha_imag"]) == (1, 0)):
            raise RuntimeError(f"性能用例 alpha 异常: {r['case_name']}")


def write_csv_files(out, bl_out, rows, pf_rows):
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # gpu_baseline.csv：覆盖全部 PF 用例，gpu_ms 待基线测试后回填；前 4 条为任务书典型 case
    with open(bl_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id"] + BASELINE_COLS + ["gpu_ms"])
        for i, r in enumerate(pf_rows):
            prefix = "cher2-base" if i < len(DOC_CASES) else "cher2-perf"
            w.writerow(
                [f"{prefix}-{i:03d}"]
                + [r[c].replace("ACLBLAS_", "") if c == "uplo" else r[c] for c in BASELINE_COLS]
                + [""]
            )


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    ap = argparse.ArgumentParser(description="cher2 用例生成（条数可扩展）")
    ap.add_argument("--accuracy", type=int, default=1000, help="精度用例条数（默认1000）")
    ap.add_argument("--perf", type=int, default=200, help="性能/内存用例条数（默认200）")
    ap.add_argument("--seed", type=int, default=20261028, help="随机种子（固定可复现）")
    ap.add_argument(
        "--dist",
        choices=["uniform", "mixed"],
        default="uniform",
        help="输入分布；mixed（均匀50%%+正态50%%）依赖测试工程扩展，暂以均匀生成",
    )
    ap.add_argument("--out", default=None)
    ap.add_argument("--baseline-out", default=None)
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    out = args.out or os.path.join(here, "cher2_test.csv")
    bl_out = args.baseline_out or os.path.join(here, "gpu_baseline.csv")
    if args.dist == "mixed":
        logging.info("[提示] 正态分布依赖 ops-blas 测试工程扩展（另行任务），本次以均匀分布生成")

    g = Gen(args.seed)
    g.l0_basic()
    g.l1_size()
    g.l2_scalar()
    g.l4_ld_inc()
    g.l5_fill()
    g.l5b_cover()
    g.l6_edge()
    if args.accuracy > len(g.rows):
        g.ex_extend(args.accuracy)
    g.perf(args.perf)

    acc_rows = [r for r in g.rows if not r["case_name"].startswith("TC_PF")]
    pf_rows = [r for r in g.rows if r["case_name"].startswith("TC_PF")]
    validate_performance_rows(pf_rows)
    write_csv_files(out, bl_out, g.rows, pf_rows)

    from collections import Counter

    cats = Counter(r["case_name"].split("_")[1] for r in acc_rows)
    logging.info("%s: 共 %d 条（精度 %d + 性能/内存 %d）", out, len(g.rows), len(acc_rows), len(pf_rows))
    logging.info("精度用例分类: %s", dict(sorted(cats.items())))
    logging.info("%s: %d 条基线占位（gpu_ms 待回填）", bl_out, len(pf_rows))


if __name__ == "__main__":
    main()
