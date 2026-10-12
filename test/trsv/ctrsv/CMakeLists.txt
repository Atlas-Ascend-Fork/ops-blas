# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

ops_blas_add_gtest_tests(${OPS_BLAS} ctrsv_test)

find_library(CTRSV_NETLIB_LIBRARY NAMES blas)
if(NOT CTRSV_NETLIB_LIBRARY)
    message(FATAL_ERROR "Ctrsv correctness requires Netlib BLAS; set CTRSV_NETLIB_LIBRARY")
endif()
target_compile_definitions(ctrsv_test PRIVATE CTRSV_NETLIB_LIBRARY="${CTRSV_NETLIB_LIBRARY}")
target_link_libraries(ctrsv_test PRIVATE ${CMAKE_DL_LIBS})
