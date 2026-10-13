/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once
#include <vector>
#include "acl/acl.h"
#include "cann_ops_blas.h"

inline aclblasStatus_t aclblasCrotg_npu(aclblasHandle_t handle, aclblasComplex* a,
                                        aclblasComplex* b, float* c, aclblasComplex* s)
{
    void *d_a = nullptr;
    void *d_b = nullptr;
    void *d_c = nullptr;
    void *d_s = nullptr;
    
    // 分配设备内存
    aclrtMalloc(&d_a, sizeof(aclblasComplex), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&d_b, sizeof(aclblasComplex), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&d_c, sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&d_s, sizeof(aclblasComplex), ACL_MEM_MALLOC_HUGE_FIRST);

    // 拷贝输入数据 Host -> Device
    aclrtMemcpy(d_a, sizeof(aclblasComplex), a, sizeof(aclblasComplex), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(d_b, sizeof(aclblasComplex), b, sizeof(aclblasComplex), ACL_MEMCPY_HOST_TO_DEVICE);

    // 调用设备端算子
    aclblasStatus_t ret = aclblasCrotg(handle, (aclblasComplex*)d_a, (aclblasComplex*)d_b,
                                       (float*)d_c, (aclblasComplex*)d_s);

    // 拷贝结果 Device -> Host (b 为只读输入，不需要拷贝回来)
    aclrtMemcpy(a, sizeof(aclblasComplex), d_a, sizeof(aclblasComplex), ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(c, sizeof(float), d_c, sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(s, sizeof(aclblasComplex), d_s, sizeof(aclblasComplex), ACL_MEMCPY_DEVICE_TO_HOST);

    // 释放设备内存
    aclrtFree(d_a);
    aclrtFree(d_b);
    aclrtFree(d_c);
    aclrtFree(d_s);
    return ret;
}
