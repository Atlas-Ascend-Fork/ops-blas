/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclblaslt_handle_impl.h
 * \brief Internal handle struct, library version / hardware constants, and device-capability
 *        queries. Not installed.
 */

#pragma once

#include "cann_ops_blasLt.h"

#include <acl/acl.h>
#include <cstddef>
#include <cstdint>

constexpr int ACLBLASLT_VERSION_MAJOR = 1;
constexpr int ACLBLASLT_VERSION_MINOR = 0;
constexpr int ACLBLASLT_VERSION_PATCH = 0;

constexpr uint32_t ACLBLASLT_HANDLE_MAGIC = 0xACBA1234;

// Ascend on-chip / peak defaults, used by the algo heuristic.
constexpr size_t L1_SIZE = 512 * 1024;
constexpr size_t L0_SIZE = 256;
constexpr uint32_t DEFAULT_AI_CORES = 8;
constexpr double DEFAULT_PEAK_TFLOPS = 140.0;
constexpr double DEFAULT_PEAK_GBPS = 900.0;

// issue #822: 移除仅被 create/destroy 触碰的遗留成员 internalWorkspace/workspaceSize/
// mutex/npuArch/maxSharedMemory/algoCache/algoCacheMaxSize/lruList 及 AlgoKey/
// AlgoKeyHasher/CacheEntry 类型——全仓无读写点，算法实际走每次调用的启发式排序，
// 不存在句柄级缓存；实际 workspace 由调用方按 heuristic 结果分配传入。
struct _aclblaslt_handle {
    uint32_t magic = ACLBLASLT_HANDLE_MAGIC;
    bool initialized = false;
    // version info
    int versionMajor = ACLBLASLT_VERSION_MAJOR;
    int versionMinor = ACLBLASLT_VERSION_MINOR;
    // AscendCL runtime
    aclrtContext context = nullptr;
    aclrtStream defaultStream = nullptr;
    int32_t deviceId = 0;
};

// Query the number of cube cores for a device, returning a safe fallback when the runtime query fails.
uint32_t QueryCubeCoreNum(int32_t deviceId);
