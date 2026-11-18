# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

if("arch35" IN_LIST SOC_ARCH_DIRS)
    if(NOT CHEMV_NETLIB_CONFIG OR NOT EXISTS "${CHEMV_NETLIB_CONFIG}")
        if("chemv" IN_LIST TEST_NAMES OR CHEMV_NETLIB_CONFIG)
            message(FATAL_ERROR "CHEMV requires a pinned Netlib 3.10.0 reference. Run test/symv/chemv/prepare_netlib.py and set CHEMV_NETLIB_CONFIG to netlib-reference.cmake.")
        endif()
        message(WARNING "Skipping Chemv tests: CHEMV_NETLIB_CONFIG is not configured.")
        return()
    endif()
    include("${CHEMV_NETLIB_CONFIG}")
    if(NOT CHEMV_REFERENCE_VERSION STREQUAL "3.10.0" OR
       NOT CHEMV_REFERENCE_SOURCE_SHA256 STREQUAL "328c1bea493a32cac5257d84157dc686cc3ab0b004e2bea22044e0a59f6f8a19")
        message(FATAL_ERROR "Wrong CHEMV golden source/version")
    endif()
    list(LENGTH REFBLAS_LIB _chemv_library_count)
    if(NOT _chemv_library_count EQUAL 2)
        message(FATAL_ERROR "CHEMV reference config must contain exactly CBLAS and BLAS libraries")
    endif()
    list(GET REFBLAS_LIB 0 _chemv_cblas_library)
    list(GET REFBLAS_LIB 1 _chemv_blas_library)
    if(NOT EXISTS "${_chemv_cblas_library}" OR NOT EXISTS "${_chemv_blas_library}")
        message(FATAL_ERROR "CHEMV pinned reference libraries do not exist")
    endif()
    get_filename_component(_chemv_cblas_library "${_chemv_cblas_library}" REALPATH)
    get_filename_component(_chemv_blas_library "${_chemv_blas_library}" REALPATH)
    # Escape paths as C string literals for direct GTest/combined-runner binding checks.
    string(REPLACE "\\" "\\\\" _chemv_cblas_literal "${_chemv_cblas_library}")
    string(REPLACE "\"" "\\\"" _chemv_cblas_literal "${_chemv_cblas_literal}")
    string(REPLACE "\\" "\\\\" _chemv_blas_literal "${_chemv_blas_library}")
    string(REPLACE "\"" "\\\"" _chemv_blas_literal "${_chemv_blas_literal}")
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/chemv_reference_config.h"
        "#pragma once\n#define CHEMV_EXPECTED_CBLAS_LIBRARY \"${_chemv_cblas_literal}\"\n#define CHEMV_EXPECTED_BLAS_LIBRARY \"${_chemv_blas_literal}\"\n")
    ops_blas_add_gtest_tests(${OPS_BLAS} chemv_test)
    target_include_directories(chemv_test PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
    target_link_libraries(chemv_test PRIVATE ${CMAKE_DL_LIBS})
    option(CHEMV_ENABLE_MEMCHECK "Build Chemv kernel with mssanitizer instrumentation" OFF)
    if(CHEMV_ENABLE_MEMCHECK)
        set_property(SOURCE "${PROJECT_SOURCE_DIR}/blas/symv/arch35/chemv_RegOrdered.cpp"
            DIRECTORY "${PROJECT_SOURCE_DIR}" APPEND PROPERTY COMPILE_OPTIONS "-g;--cce-enable-sanitizer")
    endif()
    find_package(Python3 3.8 REQUIRED COMPONENTS Interpreter)
    set(CHEMV_TEST_DEVICE 0 CACHE STRING "Logical device ID for Chemv test gates")
    foreach(_chemv_gate self_test memcheck)
        add_custom_target(chemv_${_chemv_gate}
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/run_${_chemv_gate}.py"
                --binary "$<TARGET_FILE:chemv_test>"
                --device "${CHEMV_TEST_DEVICE}"
                --output "${CMAKE_CURRENT_BINARY_DIR}/${_chemv_gate}_results"
            DEPENDS chemv_test
            VERBATIM
            USES_TERMINAL
        )
    endforeach()
    # The parent test directory creates blas_all_test after returning from here.
    # Keep its BLAS search/link inputs pinned too, without replacing global caches.
    set(REFBLAS_INCLUDE_DIR "${REFBLAS_INCLUDE_DIR}" PARENT_SCOPE)
    set(REFBLAS_LIB "${REFBLAS_LIB}" PARENT_SCOPE)
elseif("arch22" IN_LIST SOC_ARCH_DIRS)
    ops_blas_add_gtest_tests(${OPS_BLAS} chemv_test)
else()
    ops_blas_add_tests(${OPS_BLAS})
endif()
