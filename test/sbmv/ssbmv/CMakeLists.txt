# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# Both the arch35 and the arch22 CSV suites are GTest driven (custom main, CSV case source), so the
# GTest registration is used for every SOC that ships a test source for this family.
ops_blas_add_gtest_tests(${OPS_BLAS} ssbmv_test)

# The performance gate ships with the arch22 test source only: its baseline is the A2/A3
# gpu_baseline.csv from the task package. Other SOCs build the accuracy suite alone.
if("arch22" IN_LIST SOC_ARCH_DIRS AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/arch22/ssbmv_perf.cpp)
    ops_blas_add_gtest_tests(${OPS_BLAS} ssbmv_perf)
endif()
