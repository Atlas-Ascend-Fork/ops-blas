#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
CtrsmBatched 精度验证脚本。
对比 NPU 输出与 golden 参考结果，判定精度是否达标。

使用方式:
    python verify_result.py <m> <n> <batch>

精度标准:
    - 相对容差 rtol = 1.22e-3 (2^-13 * 10)
    - 绝对容差 atol = 1e-3
    - 允许不超过 0.01% 的元素超标 (error_ratio <= 1e-4)
    - 本样例的有限值精度校验不接受 NaN/Inf，包括 golden 和复数的任一分量
"""

import logging
import sys

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

RTOL = 2**-13 * 10
ATOL = 1e-3
ERROR_RATIO_TOL = 1e-4


def verify_result(m, n, batch):
    all_pass = True
    max_mere = 0.0
    total_elements = 0
    total_fail_elements = 0

    for b in range(batch):
        golden_path = f"./test/data/output/golden_{b}.bin"
        output_path = f"./test/data/output/output_b_{b}.bin"

        golden = np.fromfile(golden_path, dtype=np.float32).view(np.complex64).reshape(m, n)
        output = np.fromfile(output_path, dtype=np.float32).view(np.complex64).reshape(m, n)

        ref_abs = np.maximum(np.abs(golden.real), np.abs(golden.imag))
        ref_abs = np.maximum(ref_abs, ATOL)

        mere_re = np.abs(output.real - golden.real) / ref_abs
        mere_im = np.abs(output.imag - golden.imag) / ref_abs
        mere_map = np.maximum(mere_re, mere_im)
        nonfinite_mask = ~(np.isfinite(golden) & np.isfinite(output))
        mere_map = np.where(nonfinite_mask, np.inf, mere_map)

        batch_mere = mere_map.max()
        if batch_mere > max_mere:
            max_mere = batch_mere

        fail_count = int(np.sum(mere_map > RTOL))
        total_elements += m * n
        total_fail_elements += fail_count

        if fail_count > 0:
            fail_ratio = fail_count / (m * n)
            if np.any(nonfinite_mask) or fail_ratio > ERROR_RATIO_TOL:
                all_pass = False
                worst_idx = np.unravel_index(np.argmax(mere_map), (m, n))
                logging.error(
                    f"  Batch {b}: MERE={batch_mere:.6e}, fail_elements={fail_count}/{m * n} "
                    f"({fail_ratio:.2e}) **FAIL**"
                )
                logging.error(
                    f"    Worst at [{worst_idx[0]},{worst_idx[1]}]: "
                    f"expected={golden[worst_idx]:.6f}, actual={output[worst_idx]:.6f}"
                )

    overall_ratio = total_fail_elements / total_elements if total_elements > 0 else 0.0
    return all_pass, max_mere, overall_ratio, total_fail_elements, total_elements


if __name__ == "__main__":
    try:
        m = int(sys.argv[1]) if len(sys.argv) > 1 else 16
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 16
        batch = int(sys.argv[3]) if len(sys.argv) > 3 else 1

        passed, max_mere, overall_ratio, fail_elems, total_elems = verify_result(m, n, batch)
        logging.info(f"Max MERE: {max_mere:.6e} (rtol={RTOL:.2e}, atol={ATOL:.1e})")
        logging.info(
            f"Fail elements: {fail_elems}/{total_elems} (ratio={overall_ratio:.2e}, tol={ERROR_RATIO_TOL:.1e})"
        )

        if passed:
            logging.info("[Success] Case accuracy verification passed.")
            sys.exit(0)
        else:
            logging.error("[Failed] Case accuracy verification failed.")
            sys.exit(1)
    except Exception as e:
        logging.error(f"[Error] {e}")
        sys.exit(1)
