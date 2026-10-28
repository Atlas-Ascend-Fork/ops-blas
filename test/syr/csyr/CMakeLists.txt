# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# GTest CSV-driven ST target: arch22/csyr_test.cpp and arch35/csyr_test.cpp are
# merged into the single csyr_test target by SOC selection (SOC_ARCH_DIRS).
# csyr_ut is the arch22 host-side unit-test target and only exists where the
# arch22 sources are built.
if("arch22" IN_LIST SOC_ARCH_DIRS OR "arch35" IN_LIST SOC_ARCH_DIRS)
    ops_blas_add_gtest_tests(${OPS_BLAS} csyr_test)
    if("arch22" IN_LIST SOC_ARCH_DIRS)
        ops_blas_add_gtest_tests(${OPS_BLAS} csyr_ut)
    endif()
endif()
