#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""CPU-only Netlib column slice; input/output mappings are inherited file descriptors."""

import ctypes as C
import mmap
import os
import sys


CBLAS_COL_MAJOR = 102
CBLAS_NO_TRANS = 111
CBLAS_INTEGER_PREFIX_ARGS = 6
FLOAT_BYTES = C.sizeof(C.c_float)


def load_sgemm(library):
    dll = C.CDLL(library)
    if dll.hgemm_reference_default_environment() != 0:
        raise ValueError("Cannot set IEEE default reference environment")
    gemm = dll.cblas_sgemm
    gemm.argtypes = [C.c_int] * CBLAS_INTEGER_PREFIX_ARGS + [
        C.c_float,
        C.c_void_p,
        C.c_int,
        C.c_void_p,
        C.c_int,
        C.c_float,
        C.c_void_p,
        C.c_int,
    ]
    gemm.restype = None
    return gemm


def main():
    library, fda, fdb, fdc, ta, tb, m, n, k, la, lb, alpha, beta, first, last = sys.argv[1:]
    ta, tb, m, n, k, la, lb, first, last = map(int, [ta, tb, m, n, k, la, lb, first, last])
    if not 0 <= first < last <= n:
        raise ValueError("Invalid reference column slice")
    mappings = [mmap.mmap(fd, os.fstat(fd).st_size, access=mmap.ACCESS_WRITE) for fd in map(int, [fda, fdb, fdc])]
    buffers = [(C.c_float * (len(v) // FLOAT_BYTES)).from_buffer(v) for v in mappings]
    a, b, out = [C.addressof(v) for v in buffers]
    b += (first * lb if tb == CBLAS_NO_TRANS else first) * FLOAT_BYTES
    out += first * m * FLOAT_BYTES
    gemm = load_sgemm(library)
    gemm(CBLAS_COL_MAJOR, ta, tb, m, last - first, k, float(alpha), a, la, b, lb, float(beta), out, m)
    # Mappings stay alive through the C call; the parent checks process status
    # before reading any result. There is no NPU runtime in this process.


if __name__ == "__main__":
    main()
