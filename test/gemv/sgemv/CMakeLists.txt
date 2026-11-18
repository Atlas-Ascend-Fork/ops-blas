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
    ops_blas_add_gtest_tests(${OPS_BLAS} sgemv_test)
elseif("arch35" IN_LIST SOC_ARCH_DIRS)
    ops_blas_add_gtest_tests(${OPS_BLAS} sgemv_test)
else()
    message(FATAL_ERROR "sgemv_test requires arch22 or arch35")
endif()

# The shared CSV loader names a dtype not provided by every CANN 9.1 SDK.
# This FP32-only test must not advertise support for that missing dtype.
if(TARGET sgemv_test)
    include(CheckCXXSourceCompiles)
    set(CMAKE_REQUIRED_INCLUDES
        "${ASCEND_CANN_PACKAGE_PATH}/include;${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include")
    check_cxx_source_compiles(
        "#include <acl/acl.h>\nint main() { return static_cast<int>(ACL_HIFLOAT4); }"
        SGEMMV_HAVE_ACL_HIFLOAT4)
    if(NOT SGEMMV_HAVE_ACL_HIFLOAT4)
        target_compile_definitions(sgemv_test PRIVATE ACL_HIFLOAT4=ACL_DT_UNDEFINED)
    endif()
endif()
