# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

if("arch35" IN_LIST SOC_ARCH_DIRS)
  ops_blas_add_gtest_tests(${OPS_BLAS} csyr2_golden_test)
endif()
ops_blas_add_gtest_tests(${OPS_BLAS} csyr2_test)

if(TARGET csyr2_test AND "arch22" IN_LIST SOC_ARCH_DIRS)
  target_include_directories(csyr2_test PRIVATE
    ${CMAKE_SOURCE_DIR}/blas/syr2/arch22
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/include
    $ENV{INC_INCLUDE_PATH}
    $ENV{ASCEND_OPP_PATH}/include
    $ENV{ASCEND_OPP_PATH}/built-in/op_proto/inc
  )
endif()
