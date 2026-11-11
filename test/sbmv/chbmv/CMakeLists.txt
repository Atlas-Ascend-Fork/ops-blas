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
    ops_blas_add_gtest_tests(${OPS_BLAS} chbmv_test)
    target_compile_options(chbmv_test PRIVATE -O2)
    # The shared CSV loader names a dtype not provided by every CANN 9.1 SDK.
    # This COMPLEX64-only test must not advertise support for that missing dtype.
    include(CheckCXXSourceCompiles)
    set(CMAKE_REQUIRED_INCLUDES
        "${ASCEND_CANN_PACKAGE_PATH}/include;${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include")
    check_cxx_source_compiles(
        "#include <acl/acl.h>\nint main() { return static_cast<int>(ACL_HIFLOAT4); }"
        CHBMV_HAVE_ACL_HIFLOAT4)
    if(NOT CHBMV_HAVE_ACL_HIFLOAT4)
        target_compile_definitions(chbmv_test PRIVATE ACL_HIFLOAT4=ACL_DT_UNDEFINED)
    endif()
elseif("arch35" IN_LIST SOC_ARCH_DIRS)
    ops_blas_add_gtest_tests(${OPS_BLAS} chbmv_test)
else()
    ops_blas_add_tests(${OPS_BLAS})
endif()
