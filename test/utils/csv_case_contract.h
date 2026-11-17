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

#include <cstddef>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

// Validate the unfiltered input first. Share names across companion CSV files.
template <typename Param>
void VerifyCsvComposition(
    const std::vector<Param>& cases, size_t expectedTotal, size_t expectedPerformance, const std::string& source,
    std::set<std::string>& names)
{
    if (cases.empty() || cases.size() != expectedTotal) {
        throw std::runtime_error(
            source + ": expected " + std::to_string(expectedTotal) + " CSV cases, got " + std::to_string(cases.size()));
    }
    size_t performance = 0;
    for (const auto& param : cases) {
        const auto& name = param.caseName;
        if (name.empty() || name.find_first_not_of(" \t\r\n") == std::string::npos || !names.insert(name).second) {
            throw std::runtime_error(source + ": empty or duplicate case_name: " + name);
        }
        performance += name.rfind("TC_PF_", 0U) == 0U;
    }
    if (performance != expectedPerformance) {
        throw std::runtime_error(
            source + ": expected " + std::to_string(expectedPerformance) + " performance cases, got " +
            std::to_string(performance));
    }
}

template <typename Param>
void VerifyCsvComposition(
    const std::vector<Param>& cases, size_t expectedTotal, size_t expectedPerformance, const std::string& source)
{
    std::set<std::string> names;
    VerifyCsvComposition(cases, expectedTotal, expectedPerformance, source, names);
}
