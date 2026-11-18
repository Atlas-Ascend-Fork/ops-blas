#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import logging
import sys

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)


def parse_dimensions():
    m = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    batch_count = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    if m <= 0 or n <= 0 or batch_count <= 0:
        raise ValueError("m, n and batch must be positive for accuracy verification")

    return m, n, batch_count


def main(m, n, batch_count):

    # FLOAT32 community standard
    threshold = 2**-13            # MERE threshold
    mare_threshold = 10 * threshold
    small_value_thr = 2**-14      # below this -> small value domain (relative error unstable)
    small_abs_atol = 1e-4         # absolute tolerance for small-value subset
    all_pass = True

    for batch in range(batch_count):
        output = np.fromfile(f"output/output_b_{batch}.bin", dtype=np.float32).reshape(m, n)
        golden = np.fromfile(f"output/golden_b_{batch}.bin", dtype=np.float32).reshape(m, n)

        diff = np.abs(output - golden)
        abs_golden = np.abs(golden)

        normal_mask = abs_golden >= small_value_thr
        small_mask = ~normal_mask

        # Normal value domain: MERE / MARE on relative error
        if np.any(normal_mask):
            rel_err = diff[normal_mask] / (abs_golden[normal_mask] + 1e-7)
            mere = float(np.mean(rel_err))
            mare = float(np.max(rel_err))
        else:
            mere = 0.0
            mare = 0.0
        # Combined per-element tolerance (numpy.allclose style) guards near-zero
        # golden values where relative error is numerically unstable.
        elem_pass = diff <= (small_abs_atol + mare_threshold * abs_golden)
        normal_pass = (mere < threshold) and bool(np.all(elem_pass))

        # Small value domain: absolute error check
        if np.any(small_mask):
            small_err_count = int(np.sum(diff[small_mask] > small_abs_atol))
        else:
            small_err_count = 0
        small_pass = (small_err_count == 0)

        passed = normal_pass and small_pass
        status = "PASSED" if passed else "FAILED"
        log = logging.info if passed else logging.error
        log(f"  Batch {batch}: {status} | MERE={mere:.2e} (thr={threshold:.2e}) | "
            f"MARE={mare:.2e} (thr={mare_threshold:.2e}) | small_err={small_err_count}")
        if not passed:
            all_pass = False

    if all_pass:
        logging.info(f"All {batch_count} batches PASSED")
    else:
        logging.error(f"Some batches FAILED")
        sys.exit(1)


if __name__ == "__main__":
    try:
        dimensions = parse_dimensions()
    except ValueError as exc:
        logging.error("%s", exc)
        sys.exit(1)
    main(*dimensions)
