# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

ops_blas_add_gtest_tests(${OPS_BLAS} stbsv_test)

# Frozen validation runtimes must override build-tree libraries. DT_RUNPATH,
# unlike DT_RPATH, respects the explicitly selected LD_LIBRARY_PATH.
if(TARGET stbsv_test AND UNIX)
    target_link_options(stbsv_test PRIVATE "-Wl,--enable-new-dtags")
    target_link_libraries(stbsv_test PRIVATE ${CMAKE_DL_LIBS})
endif()
