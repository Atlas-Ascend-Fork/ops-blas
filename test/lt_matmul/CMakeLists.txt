# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

set(LT_MATMUL_NETLIB_ROOT "$ENV{LT_MATMUL_NETLIB_ROOT}" CACHE PATH
    "Optional preinstalled task Netlib 3.10.0 reference prefix")
set(_lt_task_reference_default OFF)
if(LT_MATMUL_NETLIB_ROOT)
    set(_lt_task_reference_default ON)
endif()
option(LT_MATMUL_USE_TASK_NETLIB "Use the preinstalled task Netlib 3.10.0 reference" ${_lt_task_reference_default})

# Standard builds use the same preinstalled CBLAS/LAPACK dependencies as other
# operators. Task validation can select its exact reference without downloading
# packages, installing tools, or changing references for other test suites.
ops_blas_add_gtest_tests(${OPS_BLASLT} blasLtMatmul_test)
target_include_directories(blasLtMatmul_test PRIVATE ${CMAKE_SOURCE_DIR}/blasLt/internal/include)
target_compile_definitions(blasLtMatmul_test PRIVATE OPS_BLASLT_ARCH22=1)
target_compile_options(blasLtMatmul_test PRIVATE -ffp-contract=off)
target_link_libraries(blasLtMatmul_test PRIVATE ${CMAKE_DL_LIBS})

if(LT_MATMUL_USE_TASK_NETLIB)
    if(NOT EXISTS "${LT_MATMUL_NETLIB_ROOT}/.netlib_version")
        message(FATAL_ERROR "Set LT_MATMUL_NETLIB_ROOT to a preinstalled Netlib 3.10.0 reference, or disable LT_MATMUL_USE_TASK_NETLIB for repository CBLAS testing.")
    endif()
    file(READ "${LT_MATMUL_NETLIB_ROOT}/.netlib_version" _netlib_version)
    string(STRIP "${_netlib_version}" _netlib_version)
    if(NOT _netlib_version STREQUAL "3.10.0")
        message(FATAL_ERROR "Unexpected task Netlib version: ${_netlib_version}")
    endif()
    set(LT_MATMUL_NETLIB_LIBRARY "${LT_MATMUL_NETLIB_ROOT}/lib/liblt_netlib_sgemm.so")
    if(NOT EXISTS "${LT_MATMUL_NETLIB_LIBRARY}" OR NOT EXISTS "${LT_MATMUL_NETLIB_ROOT}/include/cblas.h")
        message(FATAL_ERROR "Incomplete preinstalled Netlib prefix: ${LT_MATMUL_NETLIB_ROOT}")
    endif()
    file(SHA256 "${LT_MATMUL_NETLIB_LIBRARY}" _netlib_sha256)
    target_include_directories(blasLtMatmul_test BEFORE PRIVATE "${LT_MATMUL_NETLIB_ROOT}/include")
    target_compile_definitions(blasLtMatmul_test PRIVATE
        LT_MATMUL_NETLIB_LIBRARY="${LT_MATMUL_NETLIB_LIBRARY}"
        LT_MATMUL_NETLIB_SHA256="${_netlib_sha256}")
    message(STATUS "LtMatmul task Netlib 3.10.0: ${LT_MATMUL_NETLIB_LIBRARY} (sha256=${_netlib_sha256})")
else()
    message(STATUS "LtMatmul reference: repository CBLAS (${REFBLAS_LIB})")
endif()
