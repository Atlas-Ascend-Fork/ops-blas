# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

if("arch35" IN_LIST SOC_ARCH_DIRS)
    ops_blas_add_gtest_tests(${OPS_BLAS} cher2_test)
    target_sources(cher2_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/arch35/cher2_regression.cpp)
    target_link_libraries(cher2_test PRIVATE dl)
endif()
