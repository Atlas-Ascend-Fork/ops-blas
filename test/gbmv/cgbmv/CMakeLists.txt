# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

if(SOC_VERSION STREQUAL "ascend950")
    ops_blas_add_gtest_tests(${OPS_BLAS} cgbmv_test cgbmv_benchmark)
    # ops_blas_add_gtest_tests 不注入优化选项，这两个 target 在 CMAKE_BUILD_TYPE=Release
    # 下仍拿到空的 CXX_FLAGS，CPU 侧造数与 golden 比对会退化为 -O0。
    target_compile_options(cgbmv_test PRIVATE -O2)
    target_compile_options(cgbmv_benchmark PRIVATE -O2)
else()
    ops_blas_add_tests(${OPS_BLAS})
endif()
