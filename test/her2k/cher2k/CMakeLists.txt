# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# GTest CSV-driven precision ST target (always built when the operator is enabled).
# arch22/cher2k_test.cpp = CSV precision + scalar-regression guards
# arch35/cher2k_test.cpp = CSV precision + TC_PF performance branch + §6.6 host unit tests
if("arch22" IN_LIST SOC_ARCH_DIRS OR "arch35" IN_LIST SOC_ARCH_DIRS)
    ops_blas_add_gtest_tests(${OPS_BLAS} cher2k_test)
endif()

# The arch35 wrapper (cher2k_npu_wrapper.h) and perf branch (cher2k_perf.h) error
# paths add OP_LOGE per code-quality.md §6.3, so the test target itself references
# dlog symbols; unified_dlog is linked PRIVATE in the main library only, so it must
# be linked here as well. The arch22 wrapper has no OP_LOGE and needs no extra link.
if("arch35" IN_LIST SOC_ARCH_DIRS)
    target_link_libraries(cher2k_test PRIVATE unified_dlog)
endif()
