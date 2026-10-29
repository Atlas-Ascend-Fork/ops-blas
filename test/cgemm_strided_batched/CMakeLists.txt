# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

ops_blas_add_gtest_tests(${OPS_BLAS} cgemm_strided_batched_test)
target_compile_options(cgemm_strided_batched_test PRIVATE -O2)
find_library(CGEMM_SB_NETLIB_LIBRARY NAMES blas libblas.so.3
    PATHS /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/blas /usr/lib64/blas NO_DEFAULT_PATH)
if(NOT CGEMM_SB_NETLIB_LIBRARY)
    message(FATAL_ERROR "Set CGEMM_SB_NETLIB_LIBRARY to the actual Netlib BLAS shared library.")
endif()
get_filename_component(CGEMM_SB_NETLIB_LIBRARY "${CGEMM_SB_NETLIB_LIBRARY}" REALPATH)
file(SHA256 "${CGEMM_SB_NETLIB_LIBRARY}" CGEMM_SB_NETLIB_SHA256)
target_compile_definitions(cgemm_strided_batched_test PRIVATE
    _GNU_SOURCE CGEMM_SB_NETLIB_LIBRARY="${CGEMM_SB_NETLIB_LIBRARY}"
    CGEMM_SB_NETLIB_SHA256="${CGEMM_SB_NETLIB_SHA256}")
get_target_property(_cgemm_test_libraries cgemm_strided_batched_test LINK_LIBRARIES)
list(REMOVE_ITEM _cgemm_test_libraries ${REFBLAS_LIB} ${REFLAPACK_LIB})
list(APPEND _cgemm_test_libraries ${CMAKE_DL_LIBS})
set_property(TARGET cgemm_strided_batched_test PROPERTY LINK_LIBRARIES "${_cgemm_test_libraries}")
message(STATUS "CGEMM Netlib reference: ${CGEMM_SB_NETLIB_LIBRARY}; SHA256=${CGEMM_SB_NETLIB_SHA256}")
