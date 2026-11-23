/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <unistd.h>
#include <gtest/gtest.h>

// 用例级增量过滤（issue #457）：
// CI PreSmoke 增量模式下，run_example.sh 依据 PR 变更的 *_test.csv 生成
// BLAS_TEST_CASES 环境变量，格式为 "<op1>:<case1>,<case2>;<op2>:ALL;<op 全量标记>ALL;"。
// 本二进制按自身可执行名匹配算子名，命中则构造 gtest_filter 只运行对应用例；
// 未设置 / 未命中 / ALL 时保持全量行为。
//
// 合并测试 Runner（blas_all_test，issue #457 全量耗时优化）：
//   - 可执行名为 blas_all_test（无 _test 后缀）时进入合并模式，此时
//     BLAS_TEST_CASES 的每个条目按 "op -> INSTANTIATE 前缀" 映射为对应 suite：
//     op:ALL → "<Prefix>/*"；op:case1,case2 → "<Prefix>/*case1*:<Prefix>/*case2*"。
//     op -> prefix 映射来自可执行同目录的 all_runner_suites.map（cmake configure
//     期生成，格式 "op:Prefix" 每行一条）。
//   - 条目值为裸 ALL（无 op 前缀）时运行全部 suite。
static std::string SelfExecutableName()
{
    std::string self = "/proc/self/exe";
    char buf[4096] = {0};
    ssize_t n = readlink(self.c_str(), buf, sizeof(buf) - 1);
    std::string exeName = "unknown";
    if (n > 0) {
        std::string path(buf, static_cast<size_t>(n));
        size_t pos = path.find_last_of('/');
        exeName = (pos == std::string::npos) ? path : path.substr(pos + 1);
    }
    return exeName;
}

static std::string ExecutableDir()
{
    std::string self = "/proc/self/exe";
    char buf[4096] = {0};
    ssize_t n = readlink(self.c_str(), buf, sizeof(buf) - 1);
    if (n <= 0) {
        return ".";
    }
    std::string path(buf, static_cast<size_t>(n));
    size_t pos = path.find_last_of('/');
    return (pos == std::string::npos) ? std::string(".") : path.substr(0, pos);
}

// 读取 all_runner_suites.map：op -> INSTANTIATE 前缀
static std::unordered_map<std::string, std::string> LoadSuiteMap()
{
    std::unordered_map<std::string, std::string> suiteMap;
    std::ifstream ifs(ExecutableDir() + "/all_runner_suites.map");
    std::string line;
    while (std::getline(ifs, line)) {
        size_t colon = line.find(':');
        if (colon == std::string::npos || colon == 0 || colon + 1 >= line.size()) {
            continue;
        }
        suiteMap[line.substr(0, colon)] = line.substr(colon + 1);
    }
    return suiteMap;
}

struct CaseEntry {
    std::string op;
    std::vector<std::string> cases; // 空 = ALL
};

static std::vector<CaseEntry> ParseCaseEntries(const std::string& raw)
{
    std::vector<CaseEntry> entries;
    std::string entry(raw);
    size_t start = 0;
    while (start <= entry.size()) {
        size_t semi = entry.find(';', start);
        std::string item = entry.substr(start, (semi == std::string::npos) ? std::string::npos : semi - start);
        if (!item.empty()) {
            size_t colon = item.find(':');
            CaseEntry e;
            if (colon == std::string::npos) {
                // 裸 ALL（全量标记）
                e.op = item;
                e.cases.push_back("ALL");
            } else {
                e.op = item.substr(0, colon);
                std::string cases = item.substr(colon + 1);
                size_t cs = 0;
                while (cs <= cases.size()) {
                    size_t comma = cases.find(',', cs);
                    std::string c = cases.substr(cs, (comma == std::string::npos) ? std::string::npos : comma - cs);
                    if (!c.empty()) {
                        e.cases.push_back(c);
                    }
                    if (comma == std::string::npos) {
                        break;
                    }
                    cs = comma + 1;
                }
            }
            if (!e.cases.empty()) {
                entries.push_back(e);
            }
        }
        if (semi == std::string::npos) {
            break;
        }
        start = semi + 1;
    }
    return entries;
}

static std::string BuildCaseFilter()
{
    const char* raw = std::getenv("BLAS_TEST_CASES");
    if (raw == nullptr || std::strlen(raw) == 0) {
        return "";
    }

    std::string exeName = SelfExecutableName();
    const std::string suffix = "_test";
    bool isAllRunner = (exeName == "blas_all_test");
    if (!isAllRunner) {
        if (exeName.size() > suffix.size() &&
            exeName.compare(exeName.size() - suffix.size(), suffix.size(), suffix) == 0) {
            exeName.resize(exeName.size() - suffix.size());
        }
    }

    std::string filter;
    auto append = [&filter](const std::string& pattern) {
        if (!filter.empty()) {
            filter += ":";
        }
        filter += pattern;
    };

    if (isAllRunner) {
        // 合并模式：op -> suite 前缀 → gtest pattern
        auto suiteMap = LoadSuiteMap();
        for (const auto& e : ParseCaseEntries(raw)) {
            if (e.cases.size() == 1 && e.cases[0] == "ALL") {
                // op:ALL 或裸 ALL：无 op 名则全部 suite，否则该 op 对应 suite
                if (e.op == "ALL") {
                    return ""; // 全量
                }
                auto it = suiteMap.find(e.op);
                if (it != suiteMap.end()) {
                    append(it->second + "/*");
                }
                continue;
            }
            auto it = suiteMap.find(e.op);
            if (it == suiteMap.end()) {
                continue;
            }
            for (const auto& c : e.cases) {
                if (c == "ALL") {
                    append(it->second + "/*");
                } else {
                    // 参数化全名：<Prefix>/<instance>.<TestName>，instance 即 case_name
                    append(it->second + "/*" + c + "*");
                }
            }
        }
        return filter;
    }

    // 单算子模式：遇到裸 ALL 或本算子的 ALL 时全量运行，并合并同算子的多个条目
    for (const auto& e : ParseCaseEntries(raw)) {
        if (e.op == "ALL" && e.cases.size() == 1 && e.cases[0] == "ALL") {
            return "";
        }
        if (e.op != exeName) {
            continue;
        }
        for (const auto& c : e.cases) {
            if (c == "ALL") {
                return "";
            }
            append("*" + c + "*");
        }
    }
    return filter;
}

int main(int argc, char* argv[])
{
    ::testing::InitGoogleTest(&argc, argv);

    std::string filter = BuildCaseFilter();
    if (!filter.empty()) {
        std::string existing = ::testing::GTEST_FLAG(filter);
        if (existing.empty() || existing == "*") {
            ::testing::GTEST_FLAG(filter) = filter;
        } else {
            // 调用方已显式传 --gtest_filter 时不覆盖
            ::testing::GTEST_FLAG(filter) = existing;
        }
    }
    return RUN_ALL_TESTS();
}
