# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

if("arch22" IN_LIST SOC_ARCH_DIRS OR "arch35" IN_LIST SOC_ARCH_DIRS)
    ops_blas_add_gtest_tests(${OPS_BLAS} chemv_test)
    include(CheckCXXSourceCompiles)
    set(_chemv_saved_includes "${CMAKE_REQUIRED_INCLUDES}")
    set(CMAKE_REQUIRED_INCLUDES "${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include")
    check_cxx_source_compiles(
        "#include \"acl/acl.h\"\nint main() { return static_cast<int>(ACL_HIFLOAT4); }"
        CHEMV_HAS_ACL_HIFLOAT4)
    set(CMAKE_REQUIRED_INCLUDES "${_chemv_saved_includes}")
    if(NOT CHEMV_HAS_ACL_HIFLOAT4 AND TARGET chemv_test)
        target_compile_definitions(chemv_test PRIVATE ACL_HIFLOAT4=42)
    endif()
else()
    ops_blas_add_tests(${OPS_BLAS})
endif()
