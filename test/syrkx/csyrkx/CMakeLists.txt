# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
# Independent acceptance tests. The parent test/CMakeLists.txt automatically
# discovers test/syrkx/csyrkx when TEST_NAMES contains csyrkx.
# Pin these cache entries to Netlib BLAS 3.10.0, NOT OpenBLAS:
#   -DREFBLAS_INCLUDE_DIR=/path/to/netlib-build/include
#   -DREFBLAS_LIB=/path/to/netlib/lib/libcblas.so
# libcblas.so must resolve its corresponding Netlib libblas.so at runtime.
ops_blas_add_gtest_tests(${OPS_BLAS} csyrkx_test)
# Classification/NaN handling and the golden must not inherit fast-math.
target_compile_options(csyrkx_test PRIVATE -fno-fast-math -ffp-contract=off)
message(STATUS "csyrkx golden requires Netlib 3.10.0: REFBLAS_LIB=${REFBLAS_LIB}; REFBLAS_INCLUDE_DIR=${REFBLAS_INCLUDE_DIR}")
