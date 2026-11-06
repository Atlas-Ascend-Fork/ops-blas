# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

if("arch22" IN_LIST SOC_ARCH_DIRS)
    foreach(_ex_interface gemm_ex sgemm_ex cgemm_ex)
        if(_ex_interface IN_LIST TEST_NAMES)
            ops_blas_add_gtest_tests(${OPS_BLAS} ${_ex_interface}_test)
        endif()
    endforeach()
else()
    ops_blas_add_gtest_tests(${OPS_BLAS} gemm_ex_test)
endif()