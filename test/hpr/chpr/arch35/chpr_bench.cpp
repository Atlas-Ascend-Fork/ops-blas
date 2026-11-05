/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <climits>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include "fill.h"
#include "chpr_npu_wrapper.h"

namespace {
using namespace chpr35;
struct BenchCase {
    std::string id;
    aclblasFillMode_t uplo;
    int n;
    float alpha;
    int incx;
    BlasFillMode xFill;
    BlasFillMode apFill;
    uint32_t seed;
};
std::vector<std::string> Split(const std::string& line)
{
    std::vector<std::string> out;
    std::string field;
    bool quote = false;
    for (char ch : line) {
        if (ch == '"') {
            quote = !quote;
        } else if (ch == ',' && !quote) {
            out.push_back(field);
            field.clear();
        } else if (ch != '\r') {
            field += ch;
        }
    }
    if (quote) {
        throw std::runtime_error("Unterminated CSV quote");
    }
    out.push_back(field);
    return out;
}
std::vector<BenchCase> Load(const std::string& path)
{
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Cannot open CSV: " + path);
    }
    std::string line;
    std::getline(file, line);
    const auto keys = Split(line);
    std::vector<BenchCase> cases;
    std::set<std::string> ids;
    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }
        const auto values = Split(line);
        if (values.size() != keys.size()) {
            throw std::runtime_error("Malformed CSV row");
        }
        std::map<std::string, std::string> row;
        for (size_t i = 0; i < keys.size(); ++i) {
            row[keys[i]] = values[i];
        }
        const auto& id = row.at("case_name");
        if (!ids.insert(id).second) {
            throw std::runtime_error("Duplicate case: " + id);
        }
        if (id.rfind("TC_PF_", 0) != 0) {
            continue;
        }
        if (row.at("expect_result") != "ACLBLAS_STATUS_SUCCESS") {
            throw std::runtime_error("Performance case must expect SUCCESS: " + id);
        }
        const auto& uplo = row.at("uplo");
        if (uplo != "ACLBLAS_UPPER" && uplo != "ACLBLAS_LOWER") {
            throw std::runtime_error("Invalid uplo");
        }
        BenchCase c{
            id,
            uplo == "ACLBLAS_UPPER" ? ACLBLAS_UPPER : ACLBLAS_LOWER,
            std::stoi(row.at("n")),
            std::stof(row.at("alpha")),
            std::stoi(row.at("incx")),
            BlasFillMode(row.at("x_fill")),
            BlasFillMode(row.at("ap_fill")),
            static_cast<uint32_t>(std::stoul(row.at("random_seed")))};
        if (c.n <= 0 || c.incx == 0 || c.incx == INT_MIN || c.alpha == 0 || c.xFill.method == BlasFillMode::M_NULLPTR ||
            c.apFill.method == BlasFillMode::M_NULLPTR) {
            throw std::runtime_error("Invalid performance case: " + id);
        }
        cases.push_back(c);
    }
    if (ids.size() != 1200 || cases.size() != 200) {
        throw std::runtime_error("Expected 1200 rows and 200 PF cases");
    }
    return cases;
}
std::vector<aclblasComplex> MakeData(size_t count, const BlasFillMode& fill, uint32_t seed)
{
    const auto real = makeBlasArray(static_cast<int64_t>(count), fill, seed);
    const auto imag = makeBlasArray(static_cast<int64_t>(count), fill, seed + 1000U);
    std::vector<aclblasComplex> out(count);
    for (size_t i = 0; i < count; ++i) {
        out[i] = {real[i], imag[i]};
    }
    return out;
}
class Runtime {
public:
    aclblasHandle_t handle = nullptr;
    aclrtStream stream = nullptr;
    int device = 0;
    bool initialized = false, selected = false;
    void Init()
    {
        const char* env = std::getenv("BLAS_DEVICE_ID");
        if (env != nullptr) {
            device = std::stoi(env);
        }
        CheckAcl(aclInit(nullptr));
        initialized = true;
        CheckAcl(aclrtSetDevice(device));
        selected = true;
        if (aclblasCreate(&handle) != ACLBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("Create handle failed");
        }
        CheckAcl(aclrtCreateStream(&stream));
        if (aclblasSetStream(handle, stream) != ACLBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("Set stream failed");
        }
    }
    ~Runtime()
    {
        if (stream != nullptr) {
            aclrtSynchronizeStream(stream);
        }
        if (handle != nullptr) {
            aclblasDestroy(handle);
        }
        if (stream != nullptr) {
            aclrtDestroyStream(stream);
        }
        if (selected) {
            aclrtResetDevice(device);
        }
        if (initialized) {
            aclFinalize();
        }
    }
};
void Run(Runtime& runtime, const BenchCase& c, int warmup, int iterations, size_t ordinal)
{
    auto x = MakeData(VectorSize(c.n, c.incx), c.xFill, c.seed);
    auto ap = MakeData(PackedSize(c.n), c.apFill, c.seed + 1);
    for (int j = 0; j < c.n; ++j) {
        ap[DiagonalIndex(j, c.n, c.uplo)].imag = 0;
    }
    DeviceBuffer dx(x), dap(ap);
    auto launch = [&]() {
        const auto status = aclblasChpr(runtime.handle, c.uplo, c.n, &c.alpha, dx.get(), c.incx, dap.get());
        if (status != ACLBLAS_STATUS_SUCCESS) {
            throw std::runtime_error(c.id + " API status=" + std::to_string(status));
        }
    };
    for (int i = 0; i < warmup; ++i) {
        launch();
    }
    CheckAcl(aclrtSynchronizeStream(runtime.stream));
    for (int i = 0; i < iterations; ++i) {
        launch();
    }
    CheckAcl(aclrtSynchronizeStream(runtime.stream));
    std::cout << "CHPR_BENCH {\"case\":\"" << c.id << "\",\"n\":" << c.n << ",\"incx\":" << c.incx
              << ",\"warmup\":" << warmup << ",\"iterations\":" << iterations << ",\"ordinal\":" << ordinal
              << ",\"launch_begin\":" << ordinal * (warmup + iterations) << ",\"launch_count\":" << warmup + iterations
              << ",\"status\":\"SUCCESS\"}" << std::endl;
}
struct BenchOptions {
    std::string csv = std::string(__FILE__).substr(0, std::string(__FILE__).find_last_of("/\\")) + "/chpr_test.csv";
    std::set<std::string> requested;
    bool all = false;
    int warmup = 5;
    int iterations = 51;
};

void AddRequestedCase(std::set<std::string>& requested, const std::string& id)
{
    if (!requested.insert(id).second) {
        throw std::runtime_error("Duplicate requested case: " + id);
    }
}

void ReadCaseList(std::set<std::string>& requested, const std::string& path)
{
    std::ifstream list(path);
    if (!list) {
        throw std::runtime_error("Cannot open case list: " + path);
    }
    std::string id;
    while (std::getline(list, id)) {
        if (!id.empty() && id.back() == '\r') {
            id.pop_back();
        }
        if (!id.empty()) {
            AddRequestedCase(requested, id);
        }
    }
}

void ApplyOption(BenchOptions& options, const std::string& key, const std::string& value)
{
    if (key == "--csv") {
        options.csv = value;
    } else if (key == "--case") {
        AddRequestedCase(options.requested, value);
    } else if (key == "--case-list") {
        ReadCaseList(options.requested, value);
    } else if (key == "--warmup") {
        options.warmup = std::stoi(value);
    } else if (key == "--iterations") {
        options.iterations = std::stoi(value);
    } else {
        throw std::runtime_error("Unknown argument: " + key);
    }
}

BenchOptions ParseOptions(int argc, char** argv)
{
    BenchOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--all") {
            options.all = true;
            continue;
        }
        if (i + 1 >= argc) {
            throw std::runtime_error("Missing value: " + key);
        }
        const std::string value = argv[++i];
        ApplyOption(options, key, value);
    }
    if (options.warmup < 5 || options.iterations < 51 || options.all == !options.requested.empty()) {
        throw std::runtime_error("Require --all or --case/--case-list; warmup>=5, iterations>=51");
    }
    return options;
}

void RunSelected(Runtime& runtime, const std::vector<BenchCase>& cases, const BenchOptions& options)
{
    size_t done = 0;
    for (const auto& c : cases) {
        if (options.all || options.requested.count(c.id) != 0) {
            Run(runtime, c, options.warmup, options.iterations, done);
            ++done;
        }
    }
    if (done != (options.all ? 200U : options.requested.size())) {
        throw std::runtime_error("Case selection incomplete");
    }
}
} // namespace

int main(int argc, char** argv)
{
    try {
        const auto options = ParseOptions(argc, argv);
        const auto cases = Load(options.csv);
        Runtime runtime;
        runtime.Init();
        RunSelected(runtime, cases, options);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "CHPR_BENCH_ERROR " << e.what() << std::endl;
        return 1;
    }
}
