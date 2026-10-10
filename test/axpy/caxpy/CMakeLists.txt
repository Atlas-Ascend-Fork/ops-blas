# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# arch35 uses CSV-driven GTest cases and needs GTest linkage plus test/frame/test_main.cpp.
# arch22 keeps its legacy standalone test entry point and does not use GTest.
# Select the matching registration helper to preserve both test frameworks.
if(SOC_VERSION STREQUAL "ascend950")
    ops_blas_add_gtest_tests(${OPS_BLAS} caxpy_test)
else()
    ops_blas_add_tests(${OPS_BLAS})
endif()
