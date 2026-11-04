# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

ops_blas_add_gtest_tests(${OPS_BLAS} gemm_batched_ex_test)

# Reuse parseAclDataType / physRows / physCols from gemm_ex
target_include_directories(gemm_batched_ex_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../gemm_ex)

# The arch22 test TU calls host_utils.h inline helpers (GetAicCoreCount /
# GetAivCoreCount) whose OP_LOGE paths reference dlog symbols; unified_dlog is
# linked PRIVATE in the main library only, so the test target must link it
# explicitly (same as cher2k_test).
if(TARGET gemm_batched_ex_test)
    target_link_libraries(gemm_batched_ex_test PRIVATE unified_dlog)
endif()
