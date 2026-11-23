# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# TEMPLATE: 新结构算子仅在 ascend950（arch35）注册 GTest，其他 SOC 走旧测试入口。
# 将 {{op}} 替换为算子名（如 sger）。不区分 SOC 时可简化为：ops_blas_add_gtest_tests(${OPS_BLAS})
if(SOC_VERSION STREQUAL "ascend950")
    ops_blas_add_gtest_tests(${OPS_BLAS} {{op}}_test)
else()
    ops_blas_add_tests(${OPS_BLAS})
endif()
