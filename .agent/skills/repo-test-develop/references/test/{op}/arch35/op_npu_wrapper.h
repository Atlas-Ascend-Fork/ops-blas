/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// TEMPLATE: NPU wrapper（芯片相关 ACL 操作）。文件落地为 test/<family>/{{op}}/arch35/{{op}}_npu_wrapper.h
// 封装全部 ACL 操作：malloc -> H2D -> kernel -> sync -> D2H -> free。测试侧只传 host std::vector。
// 强制约束（代码检视必查 HIGH）：
//   - 每个 aclrtMalloc / aclrtMemcpy(H2D) / 算子调用 / aclrtSynchronizeDevice / aclrtMemcpy(D2H) 必须校验返回值
//   - 任一 ACL 失败后，必须先 free 已分配的 device 内存再返回错误码（防止泄漏）
//   - wrapper 内部不得调用业务算子日志接口（如 OP_LOGE），只返回结构化错误码
//   - 入参 nullptr 表示该 buffer 不参与，跳过对应 device 内存操作

#ifndef{{OP } } _NPU_H
#define{{OP } } _NPU_H

#include <cstdint>
#include "acl/acl.h"
#include "cann_ops_blas.h"

inline aclblasStatus_t aclblas
{
    {
        Op
    }
}
_npu(aclblasHandle_t handle /* TEMPLATE: + API 参数：维度 + const 指针 + 非常量指针 */)
{
    // 1. 快速路径：handle == nullptr 或规模 <= 0 -> 直接透传（由算子内部处理）
    if (handle == nullptr /* || n <= 0 */) {
        return aclblas{{Op}}(handle /* , 参数 */);
    }

    // 2. 计算 host 端需要搬运的字节数（考虑 stride / lda / 多维）
    const size_t xBytes = 0; // TEMPLATE
    const size_t yBytes = 0; // TEMPLATE

    // 3. 分配 device 内存 + H2D（每个 malloc / H2D 必须校验返回值）
    void* dX = nullptr;
    void* dY = nullptr;
    aclError aclRet;

    if (x != nullptr) {
        aclRet = aclrtMalloc(&dX, xBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (aclRet != ACL_SUCCESS)
            return ACLBLAS_STATUS_ALLOC_FAILED;
        aclRet = aclrtMemcpy(dX, xBytes, x, xBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (aclRet != ACL_SUCCESS) {
            aclrtFree(dX);
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }

    if (y != nullptr) {
        aclRet = aclrtMalloc(&dY, yBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (aclRet != ACL_SUCCESS) {
            if (dX)
                aclrtFree(dX);
            return ACLBLAS_STATUS_ALLOC_FAILED;
        }
        aclRet = aclrtMemcpy(dY, yBytes, y, yBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (aclRet != ACL_SUCCESS) {
            if (dX)
                aclrtFree(dX);
            aclrtFree(dY);
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }

    // 4. 调用算子（必须校验返回状态）
    aclblasStatus_t ret = aclblas{{Op}}(handle /* , 参数转换为 device 指针 */);
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        if (dX)
            aclrtFree(dX);
        if (dY)
            aclrtFree(dY);
        return ret;
    }

    // 5. 同步设备（必须校验返回值；异步算子改用 aclrtSynchronizeStream(h->stream)）
    aclRet = aclrtSynchronizeDevice();
    if (aclRet != ACL_SUCCESS) {
        if (dX)
            aclrtFree(dX);
        if (dY)
            aclrtFree(dY);
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    // 6. D2H（必须校验返回值）+ 释放
    if (y != nullptr && dY != nullptr) {
        aclRet = aclrtMemcpy(y, yBytes, dY, yBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        if (aclRet != ACL_SUCCESS) {
            if (dX)
                aclrtFree(dX);
            aclrtFree(dY);
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }

    if (dX)
        aclrtFree(dX);
    if (dY)
        aclrtFree(dY);
    return ret;
}

#endif // {{OP}}_NPU_H
