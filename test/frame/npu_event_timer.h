#pragma once
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <vector>

#include "acl/acl.h"
#include "acl/acl_rt.h"

namespace blas_test {
inline bool MeasureRequested() { return std::getenv("BLAS_MEASURE") != nullptr; }
inline int MeasureNpuEvent(aclrtStream stream, int warmup, int measurements,
    const std::function<bool()>& resetA, const std::function<bool()>& launch, std::vector<double>& outUs)
{
    if (stream == nullptr || warmup <= 0 || measurements <= 0) return 1;
    aclrtEvent evStart = nullptr, evEnd = nullptr;
    if (aclrtCreateEvent(&evStart) != ACL_SUCCESS || aclrtCreateEvent(&evEnd) != ACL_SUCCESS) return 2;
    const auto fail = [&](int code) { aclrtDestroyEvent(evStart); aclrtDestroyEvent(evEnd); return code; };
    for (int i = 0; i < warmup; ++i)
        if (!resetA() || !launch() || aclrtSynchronizeStream(stream) != ACL_SUCCESS) return fail(3);
    outUs.clear(); outUs.reserve(static_cast<size_t>(measurements));
    for (int i = 0; i < measurements; ++i) {
        if (!resetA()) return fail(4);
        if (aclrtRecordEvent(evStart, stream) != ACL_SUCCESS || !launch() ||
            aclrtRecordEvent(evEnd, stream) != ACL_SUCCESS || aclrtSynchronizeStream(stream) != ACL_SUCCESS)
            return fail(5);
        float ms = 0.0F;
        if (aclrtEventElapsedTime(&ms, evStart, evEnd) != ACL_SUCCESS) return fail(6);
        outUs.push_back(static_cast<double>(ms) * 1000.0);
    }
    return fail(0);
}
inline double MedianUs(std::vector<double> values)
{ if (values.empty()) return 0.0; std::sort(values.begin(), values.end()); return values[values.size() / 2]; }
inline double MeanUs(const std::vector<double>& values)
{ if (values.empty()) return 0.0; double sum = 0.0; for (double value : values) sum += value; return sum / static_cast<double>(values.size()); }
} // namespace blas_test
