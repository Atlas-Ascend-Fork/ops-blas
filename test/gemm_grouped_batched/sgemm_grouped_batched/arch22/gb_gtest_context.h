/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once
#include "gb_test_support.h"

namespace gb_test {
inline std::string SiblingCSV(const char* sourceFile, const char* name)
{
    std::string path(sourceFile);
    const auto slash = path.find_last_of("/\\");
    if (slash == std::string::npos)
        throw std::runtime_error("cannot locate the committed SGEMM task CSV");
    path.resize(slash + 1);
    return path + name;
}
inline std::string TaskCasesCSV(const char* sourceFile)
{
    if (const char* path = std::getenv("GB_CASES_CSV")) {
        if (!*path)
            throw std::runtime_error("GB_CASES_CSV is empty");
        return path;
    }
    return SiblingCSV(sourceFile, "sgemm_grouped_batched_test.csv");
}

// One lazy runtime owner for accuracy, performance and async GTest suites.
// The inline function's static is shared across translation units; listing
// cases does not call it and therefore does not initialize the NPU runtime.
inline Context& GTestContext()
{
#ifdef TEST_DEVICE_ID
    constexpr int defaultDevice = TEST_DEVICE_ID;
#else
    constexpr int defaultDevice = 0;
#endif
    static Context context(std::getenv("GB_DEVICE") ? Int(std::getenv("GB_DEVICE")) : defaultDevice);
    return context;
}
} // namespace gb_test
