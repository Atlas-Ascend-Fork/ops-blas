# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

if("arch35" IN_LIST SOC_ARCH_DIRS OR "arch22" IN_LIST SOC_ARCH_DIRS)
    ops_blas_add_gtest_tests(${OPS_BLAS} csymv_test)
else()
    ops_blas_add_tests(${OPS_BLAS})
endif()

if("arch22" IN_LIST SOC_ARCH_DIRS)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_custom_target(csymv_self_test
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/run_self_test.py"
            --binary "$<TARGET_FILE:csymv_test>"
            --output "${CMAKE_CURRENT_BINARY_DIR}/self_test"
        DEPENDS csymv_test
        USES_TERMINAL VERBATIM)
    add_custom_target(csymv_memcheck
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/run_memcheck.py"
            --binary "$<TARGET_FILE:csymv_test>"
            --output "${CMAKE_CURRENT_BINARY_DIR}/memcheck"
        DEPENDS csymv_test
        USES_TERMINAL VERBATIM)
    add_library(CANN::mspti SHARED IMPORTED GLOBAL)
    set_target_properties(CANN::mspti PROPERTIES
        IMPORTED_LOCATION "${ASCEND_CANN_PACKAGE_PATH}/lib64/libmspti.so"
        INTERFACE_INCLUDE_DIRECTORIES "${ASCEND_CANN_PACKAGE_PATH}/include/mspti")
    target_link_libraries(csymv_test PRIVATE CANN::mspti)
    cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}/test"
        CALL target_link_libraries blas_all_test PRIVATE CANN::mspti)
    set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/arch22/csymv_test.cpp"
        DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_SOURCE_DIR}/test"
        PROPERTIES COMPILE_OPTIONS "-O3;-ffp-contract=off")
endif()
