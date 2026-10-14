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

#include <cstdlib>
#include <gtest/gtest.h>
#include "acl/acl.h"
#include "cann_ops_blas.h"

#ifndef TEST_DEVICE_ID
#define TEST_DEVICE_ID 0
#endif

namespace blas_test_detail {

// 运行时设备号：优先读环境变量 BLAS_DEVICE_ID（多 device 分片并行用，
// build.sh 将算子列表按设备均分并逐片注入），未设置时回退编译期 TEST_DEVICE_ID。
inline int RuntimeDeviceId()
{
    const char* env = std::getenv("BLAS_DEVICE_ID");
    if (env != nullptr && *env != '\0') {
        char* end = nullptr;
        long value = std::strtol(env, &end, 10);
        if (end != env && *end == '\0' && value >= 0) {
            return static_cast<int>(value);
        }
    }
    return TEST_DEVICE_ID;
}

struct AclGuard {
    aclblasHandle_t handle = nullptr;
    aclrtStream stream = nullptr;
    bool cleaned = false;
    // suite 引用计数：SetUpTestSuite 递增、TearDownTestSuite 递减，
    // 仅当计数归零才销毁 handle/stream 并 aclFinalize。
    // 单算子二进制（1 个 suite）行为与原先一致；合并 Runner（多 suite 同进程，
    // issue #457 全量耗时优化）中前后 suite 复用同一份初始化，80 次 aclInit 降为 1 次。
    int refs = 0;

    ~AclGuard()
    {
        // Cleanup is primarily done in TearDownTestSuite to avoid heap corruption
        // during global destruction. This destructor is a safety fallback.
        if (cleaned) {
            return;
        }
        try {
            if (handle != nullptr) {
                aclblasDestroy(handle);
                handle = nullptr;
            }
            if (stream != nullptr) {
                aclrtDestroyStream(stream);
                stream = nullptr;
            }
            aclrtResetDevice(RuntimeDeviceId());
            aclFinalize();
        } catch (...) {
        }
    }
};

inline AclGuard& globalAcl()
{
    static AclGuard guard;
    return guard;
}

} // namespace blas_test_detail

template <typename ParamType>
class BlasTest : public ::testing::TestWithParam<ParamType> {
protected:
    static void SetUpTestSuite()
    {
        auto& guard = blas_test_detail::globalAcl();
        if (guard.handle == nullptr) {
            aclError initRet = aclInit(nullptr);
            ASSERT_TRUE(initRet == ACL_SUCCESS || initRet == ACL_ERROR_REPEAT_INITIALIZE)
                << "aclInit failed with error: " << initRet;
            ASSERT_EQ(aclrtSetDevice(blas_test_detail::RuntimeDeviceId()), ACL_SUCCESS);
            ASSERT_EQ(aclblasCreate(&guard.handle), ACLBLAS_STATUS_SUCCESS);
            ASSERT_EQ(aclrtCreateStream(&guard.stream), ACL_SUCCESS);
            ASSERT_EQ(aclblasSetStream(guard.handle, guard.stream), ACLBLAS_STATUS_SUCCESS);
            guard.cleaned = false;
        }
        guard.refs++;
        handle_ = guard.handle;
        stream_ = guard.stream;
    }

    static void TearDownTestSuite()
    {
        // Explicitly cleanup here instead of relying on global destructor
        // This avoids heap corruption during global destruction phase
        auto& guard = blas_test_detail::globalAcl();
        if (guard.cleaned)
            return;

        if (guard.refs > 0) {
            guard.refs--;
        }
        // 仍有 suite 在使用（合并 Runner 场景）时不销毁，下一个 suite 直接复用
        if (guard.refs > 0 || guard.handle == nullptr) {
            handle_ = guard.handle;
            stream_ = guard.stream;
            return;
        }

        aclblasDestroy(guard.handle);
        guard.handle = nullptr;
        aclrtSynchronizeStream(guard.stream);
        aclrtDestroyStream(guard.stream);
        guard.stream = nullptr;
        aclrtResetDevice(blas_test_detail::RuntimeDeviceId());
        aclFinalize();
        guard.cleaned = true;
        handle_ = nullptr;
        stream_ = nullptr;
    }

    static aclblasHandle_t handle_;
    static aclrtStream stream_;
};

template <typename ParamType>
aclblasHandle_t BlasTest<ParamType>::handle_ = nullptr;

template <typename ParamType>
aclrtStream BlasTest<ParamType>::stream_ = nullptr;
