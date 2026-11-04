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
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include <cblas.h>
#include <unistd.h>

#include "blas_test.h"
#include "csv_loader.h"
#include "fill.h"
#include "gemm_batched_ex_param.h"
#include "host_utils.h"

namespace {
void AclCheck(aclError error)
{
    if (error != ACL_SUCCESS)
        throw std::runtime_error("ACL error " + std::to_string(error));
}

void Require(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

struct Case : GemmBatchedExParam {
    explicit Case(const csv_map& row) : GemmBatchedExParam(row)
    {
        alpha = std::strtof(ReadMap(row, "alpha", "1").c_str(), nullptr);
        beta = std::strtof(ReadMap(row, "beta", "0").c_str(), nullptr);
        // The shared parser has no ACL_DOUBLE name; retain the actual enum in
        // this negative test instead of silently converting it to FP16.
        if (ReadMap(row, "Atype") == "ACL_DOUBLE")
            Atype = ACL_DOUBLE;
        if (ReadMap(row, "Btype") == "ACL_DOUBLE")
            Btype = ACL_DOUBLE;
        if (ReadMap(row, "Ctype") == "ACL_DOUBLE")
            Ctype = ACL_DOUBLE;
    }
};

class DeviceBuffer {
public:
    static constexpr size_t GUARD = 64;
    DeviceBuffer(const void* source, size_t size, size_t offset = 0) : bytes_(size)
    {
        AclCheck(aclrtMalloc(&base_, bytes_ + 2 * GUARD + offset, ACL_MEM_MALLOC_HUGE_FIRST));
        data_ = static_cast<uint8_t*>(base_) + GUARD + offset;
        std::array<uint8_t, GUARD> guard;
        guard.fill(0xa5);
        AclCheck(aclrtMemcpy(data_ - GUARD, GUARD, guard.data(), GUARD, ACL_MEMCPY_HOST_TO_DEVICE));
        AclCheck(aclrtMemcpy(data_ + bytes_, GUARD, guard.data(), GUARD, ACL_MEMCPY_HOST_TO_DEVICE));
        if (source && bytes_)
            AclCheck(aclrtMemcpy(data_, bytes_, source, bytes_, ACL_MEMCPY_HOST_TO_DEVICE));
    }
    ~DeviceBuffer()
    {
        if (base_)
            aclrtFree(base_);
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    void* data() const { return data_; }
    size_t size() const { return bytes_; }
    void read(void* destination) const
    {
        if (bytes_)
            AclCheck(aclrtMemcpy(destination, bytes_, data_, bytes_, ACL_MEMCPY_DEVICE_TO_HOST));
    }
    void checkGuards() const
    {
        std::array<uint8_t, GUARD> before{};
        std::array<uint8_t, GUARD> after{};
        AclCheck(aclrtMemcpy(before.data(), GUARD, data_ - GUARD, GUARD, ACL_MEMCPY_DEVICE_TO_HOST));
        AclCheck(aclrtMemcpy(after.data(), GUARD, data_ + bytes_, GUARD, ACL_MEMCPY_DEVICE_TO_HOST));
        Require(
            std::all_of(before.begin(), before.end(), [](uint8_t x) { return x == 0xa5; }) &&
                std::all_of(after.begin(), after.end(), [](uint8_t x) { return x == 0xa5; }),
            "device buffer guard changed");
    }

private:
    void* base_ = nullptr;
    uint8_t* data_ = nullptr;
    size_t bytes_ = 0;
};

using Matrices = std::vector<std::vector<float>>;
std::vector<float> MakeMatrix(int rows, int columns, int leading, const BlasFillMode& fill, uint32_t seed)
{
    if (fill.method != BlasFillMode::M_GAUSS)
        return makeBlasMatrix(rows, columns, leading, fill, seed);
    if (rows <= 0 || columns <= 0)
        return {};
    std::vector<float> matrix(static_cast<size_t>(leading) * columns, kBlasSentinel);
    std::mt19937 random(seed);
    std::uniform_real_distribution<float> means(-5.0f, 5.0f);
    std::uniform_real_distribution<float> deviations(0.1f, 2.0f);
    const float mean = means(random);
    const float deviation = deviations(random);
    std::normal_distribution<float> distribution(mean, deviation);
    for (int col = 0; col < columns; ++col)
        for (int row = 0; row < rows; ++row)
            matrix[static_cast<size_t>(col) * leading + row] = distribution(random);
    return matrix;
}

struct Data {
    Matrices a;
    Matrices b;
    Matrices initialC;
    std::vector<std::unique_ptr<DeviceBuffer>> da;
    std::vector<std::unique_ptr<DeviceBuffer>> db;
    std::vector<std::unique_ptr<DeviceBuffer>> dc;
    std::unique_ptr<DeviceBuffer> ap;
    std::unique_ptr<DeviceBuffer> bp;
    std::unique_ptr<DeviceBuffer> cp;

    explicit Data(const Case& p)
    {
        std::vector<void*> pa;
        std::vector<void*> pb;
        std::vector<void*> pc;
        for (int batch = 0; batch < p.batchCount; ++batch) {
            const uint32_t seed = p.randomSeed + static_cast<uint32_t>(batch) * 3U;
            a.push_back(MakeMatrix(physRows(p.m, p.k, p.transA), physCols(p.m, p.k, p.transA), p.lda, p.aFill, seed));
            b.push_back(
                MakeMatrix(physRows(p.k, p.n, p.transB), physCols(p.k, p.n, p.transB), p.ldb, p.bFill, seed + 1));
            initialC.push_back(MakeMatrix(p.m, p.n, p.ldc, p.cFill, seed + 2));
            const size_t offset = p.description.find("offset4") == std::string::npos ? 0 : 4;
            da.emplace_back(new DeviceBuffer(a.back().data(), a.back().size() * sizeof(float), offset));
            db.emplace_back(new DeviceBuffer(b.back().data(), b.back().size() * sizeof(float), offset));
            dc.emplace_back(new DeviceBuffer(initialC.back().data(), initialC.back().size() * sizeof(float), offset));
            const bool noAb = p.description.find("noab") != std::string::npos;
            pa.push_back(noAb ? nullptr : da.back()->data());
            pb.push_back(noAb ? nullptr : db.back()->data());
            pc.push_back(dc.back()->data());
        }
        ap.reset(new DeviceBuffer(pa.data(), pa.size() * sizeof(void*)));
        bp.reset(new DeviceBuffer(pb.data(), pb.size() * sizeof(void*)));
        cp.reset(new DeviceBuffer(pc.data(), pc.size() * sizeof(void*)));
    }

    Matrices readC() const
    {
        Matrices output = initialC;
        for (size_t b = 0; b < dc.size(); ++b)
            dc[b]->read(output[b].data());
        return output;
    }

    void checkGuardsAndPadding(const Case& p, const Matrices& output) const
    {
        ap->checkGuards();
        bp->checkGuards();
        cp->checkGuards();
        for (size_t batch = 0; batch < dc.size(); ++batch) {
            da[batch]->checkGuards();
            db[batch]->checkGuards();
            dc[batch]->checkGuards();
            for (int col = 0; col < p.n; ++col) {
                const auto offset = static_cast<size_t>(col) * p.ldc + p.m;
                Require(
                    std::memcmp(
                        output[batch].data() + offset, initialC[batch].data() + offset,
                        static_cast<size_t>(p.ldc - p.m) * sizeof(float)) == 0,
                    "C padding changed");
            }
        }
    }

    void checkInputs() const
    {
        for (size_t batch = 0; batch < da.size(); ++batch) {
            auto read = a[batch];
            da[batch]->read(read.data());
            Require(std::memcmp(read.data(), a[batch].data(), read.size() * sizeof(float)) == 0, "A changed");
            read = b[batch];
            db[batch]->read(read.data());
            Require(std::memcmp(read.data(), b[batch].data(), read.size() * sizeof(float)) == 0, "B changed");
        }
    }
};

void WriteMatrices(const std::filesystem::path& path, const Matrices& matrices)
{
    Require(!std::filesystem::exists(path), "refuse to overwrite " + path.string());
    // Serialization only:the caller has already executed the new kernel and
    // read back actual C. Never supplies data to the numerical reference.
    const std::string name = path.filename().string();
    const bool timedOutput = name.size() > 11 && name.rfind("output_", 0) == 0 &&
                             name.substr(name.size() - 4) == ".bin" &&
                             name.find_first_not_of("0123456789", 7) == name.size() - 4;
    const char* pool = std::getenv("GEMM_OUTPUT_BLOB_POOL");
    if (pool && (name == "fresh_output.bin" || timedOutput)) {
        const auto existing = std::filesystem::path(pool) / path.parent_path().filename() / "fresh_output.bin";
        uint64_t bytes = 0;
        for (const auto& matrix : matrices)
            bytes += matrix.size() * sizeof(float);
        if (std::filesystem::is_regular_file(existing) && std::filesystem::file_size(existing) == bytes) {
            std::ifstream prior(existing, std::ios::binary);
            std::array<char, 65536> chunk{};
            bool equal = prior.good();
            for (const auto& matrix : matrices) {
                const char* actual = reinterpret_cast<const char*>(matrix.data());
                size_t remaining = matrix.size() * sizeof(float);
                while (equal && remaining) {
                    const size_t take = std::min(remaining, chunk.size());
                    prior.read(chunk.data(), take);
                    equal = prior.good() && std::memcmp(actual, chunk.data(), take) == 0;
                    remaining -= take;
                    actual += take;
                }
                if (!equal)
                    break;
            }
            if (equal && prior.peek() == std::ifstream::traits_type::eof()) {
                std::filesystem::create_symlink(std::filesystem::canonical(existing), path);
                return;
            }
        }
    }
    std::ofstream stream(path, std::ios::binary);
    for (const auto& matrix : matrices)
        stream.write(reinterpret_cast<const char*>(matrix.data()), matrix.size() * sizeof(float));
    Require(stream.good(), "snapshot write failed " + path.string());
}

Matrices ReadMatrices(const std::filesystem::path& path, const Matrices& shape)
{
    Matrices matrices = shape;
    std::ifstream stream(path, std::ios::binary);
    for (auto& matrix : matrices)
        stream.read(reinterpret_cast<char*>(matrix.data()), matrix.size() * sizeof(float));
    Require(
        stream.good() && stream.peek() == std::ifstream::traits_type::eof(), "snapshot size mismatch " + path.string());
    return matrices;
}

bool SameMatrices(const Matrices& a, const Matrices& b)
{
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].size() != b[i].size() || std::memcmp(a[i].data(), b[i].data(), a[i].size() * sizeof(float)))
            return false;
    }
    return true;
}

void ScaleGolden(const Case& p, std::vector<float>& matrix)
{
    for (int col = 0; col < p.n; ++col)
        for (int row = 0; row < p.m; ++row) {
            auto& value = matrix[static_cast<size_t>(col) * p.ldc + row];
            if (p.beta == 0.0f)
                value = 0.0f;
            else if (p.beta != 1.0f)
                value = p.beta * value;
        }
}

Matrices Golden(const Case& p, const Data& data)
{
    Matrices golden = data.initialC;
    for (int batch = 0; batch < p.batchCount; ++batch) {
        // Netlib SGEMM quick returns are explicit, including no read of C for beta=0.
        if (p.k == 0 || p.alpha == 0.0f) {
            ScaleGolden(p, golden[batch]);
        } else {
            cblas_sgemm(
                CblasColMajor, p.transA == ACLBLAS_OP_N ? CblasNoTrans : CblasTrans,
                p.transB == ACLBLAS_OP_N ? CblasNoTrans : CblasTrans, p.m, p.n, p.k, p.alpha, data.a[batch].data(),
                p.lda, data.b[batch].data(), p.ldb, p.beta, golden[batch].data(), p.ldc);
        }
    }
    return golden;
}

bool censusNumericalFailure = false;
bool censusTestCompleted = false;

struct PrecisionCensus {
    uint64_t finite = 0;
    uint64_t matched = 0;
    uint64_t special = 0;
    uint64_t specialMismatch = 0;
    uint64_t exactMismatch = 0;
    uint64_t hardFailures = 0;
    double maximum = 0;
};

void ReportPrecision(const Case& p, const std::string& label, int batch, bool exact, const PrecisionCensus& census)
{
    const double ratio = census.finite ? static_cast<double>(census.matched) / census.finite : 1.0;
    const bool passed =
        ratio >= 0.99 && census.hardFailures == 0 && census.specialMismatch == 0 && census.exactMismatch == 0;
    std::cout << "GEMM_PRECISION " << p.caseName << " " << label << " batch=" << batch << " finite=" << census.finite
              << " special=" << census.special << " matched_ratio=" << ratio << " max_abs_error=" << census.maximum
              << " exact=" << exact << std::endl;
    std::cout << "GEMM_PRECISION_CENSUS " << p.caseName << " " << label << " batch=" << batch << " passed=" << passed
              << " hard_failures=" << census.hardFailures << " exact_mismatches=" << census.exactMismatch
              << " special_mismatches=" << census.specialMismatch << std::endl;
    if (!passed) {
        censusNumericalFailure = true;
        ADD_FAILURE() << label << " numerical census failure batch=" << batch;
    }
}

bool EquivalentSpecial(float actual, float golden)
{
    return (std::isnan(actual) && std::isnan(golden)) ||
           (std::isinf(actual) && std::isinf(golden) && std::signbit(actual) == std::signbit(golden));
}

void Verify(const Case& p, const Matrices& actual, const Matrices& golden, const std::string& label)
{
    const bool exact = p.alpha == 0.0f || p.k == 0;
    for (int batch = 0; batch < p.batchCount; ++batch) {
        PrecisionCensus census;
        for (int col = 0; col < p.n; ++col)
            for (int row = 0; row < p.m; ++row) {
                const size_t i = static_cast<size_t>(col) * p.ldc + row;
                const float a = actual[batch][i];
                const float g = golden[batch][i];
                if (exact) {
                    census.exactMismatch += std::memcmp(&a, &g, sizeof(float)) != 0;
                    continue;
                }
                if (!std::isfinite(a) || !std::isfinite(g)) {
                    ++census.special;
                    census.specialMismatch += !EquivalentSpecial(a, g);
                    continue;
                }
                ++census.finite;
                const double error = std::fabs(static_cast<double>(a) - g);
                census.maximum = std::max(census.maximum, error);
                census.matched += error <= 0x1p-13 + 0x1p-13 * std::fabs(static_cast<double>(g));
                const float magnitude = std::fabs(g);
                const float adjacent =
                    std::nextafter(magnitude, magnitude == std::numeric_limits<float>::max() ? 0.0f : INFINITY);
                const double ulp = std::fabs(static_cast<double>(adjacent) - magnitude);
                census.hardFailures += error > std::max(1e-2, 32.0 * ulp);
            }
        ReportPrecision(p, label, batch, exact, census);
    }
}

class GemmBatchedExArch22Test : public BlasTest<Case> {
protected:
    void TearDown() override
    {
        if (HasFailure()) {
            if (censusNumericalFailure && censusTestCompleted) {
                std::cout << "GEMM_COLLECTED_NUMERICAL_FAILURE" << std::endl;
            } else {
                std::cout << "GEMM_CENSUS_STOP_RUNTIME_OR_GUARD" << std::endl;
                std::cerr.flush();
                std::_Exit(2);
            }
        }
    }
    static std::unique_ptr<DeviceBuffer> workspace;
    static void SetUpTestSuite()
    {
        BlasTest<Case>::SetUpTestSuite();
        std::cout << "GEMM_PLATFORM " << aclrtGetSocName() << " AIC=" << GetAicCoreCount()
                  << " AIV=" << GetAivCoreCount() << " PID=" << getpid() << std::endl;
        ASSERT_GT(GetAicCoreCount(), 0U);
        ASSERT_GT(GetAivCoreCount(), 0U);
        workspace.reset(new DeviceBuffer(nullptr, 32U * 1024U * 1024U));
        ASSERT_EQ(aclblasSetWorkspace(handle_, workspace->data(), workspace->size()), ACLBLAS_STATUS_SUCCESS);
    }
    static void TearDownTestSuite()
    {
        // All work is complete. Free borrowed user memory while ACL and its
        // device context are still alive; handle destruction does not free it.
        if (stream_)
            AclCheck(aclrtSynchronizeStream(stream_));
        workspace.reset();
        BlasTest<Case>::TearDownTestSuite();
    }
};
std::unique_ptr<DeviceBuffer> GemmBatchedExArch22Test::workspace;

const std::vector<Case>& LoadCases()
{
    const char* selected = std::getenv("GEMM_TEST_CSV");
    static const auto cases = GetCasesFromCsv<Case>(selected ? selected : ReplaceFileExtension2Csv(__FILE__));
    return cases;
}

size_t ExpectedCaseCount()
{
    const char* selected = std::getenv("GEMM_TEST_CSV");
    const auto name = selected ? std::filesystem::path(selected).filename().string() : "gemm_batched_ex_test.csv";
    if (name == "gemm_batched_ex_test.csv")
        return 2393;
    if (name == "fixed_scalar144.csv")
        return 144;
    if (name == "random_scalar72.csv")
        return 72;
    if (name == "special_zero432.csv")
        return 432;
    const char* count = std::getenv("GEMM_EXPECTED_CASES");
    Require(count != nullptr, "custom CSV requires GEMM_EXPECTED_CASES");
    size_t used = 0;
    const size_t expected = std::stoull(count, &used);
    Require(used == std::strlen(count) && expected > 0, "invalid GEMM_EXPECTED_CASES");
    return expected;
}

TEST(GemmBatchedExContract, CsvCaseCount)
{
    const auto& cases = LoadCases();
    ASSERT_EQ(cases.size(), ExpectedCaseCount());
    std::set<std::string> names;
    for (const auto& p : cases) {
        ASSERT_FALSE(p.caseName.empty());
        ASSERT_TRUE(names.insert(p.caseName).second) << "duplicate CSV case " << p.caseName;
    }
}

TEST(GemmBatchedExContract, SpecialValueComparison)
{
    EXPECT_TRUE(EquivalentSpecial(NAN, NAN));
    EXPECT_TRUE(EquivalentSpecial(INFINITY, INFINITY));
    EXPECT_TRUE(EquivalentSpecial(-INFINITY, -INFINITY));
    EXPECT_FALSE(EquivalentSpecial(NAN, INFINITY));
    EXPECT_FALSE(EquivalentSpecial(INFINITY, NAN));
    EXPECT_FALSE(EquivalentSpecial(INFINITY, -INFINITY));
    EXPECT_FALSE(EquivalentSpecial(NAN, 0.0f));
}

INSTANTIATE_TEST_SUITE_P(
    GemmBatchedEx, GemmBatchedExArch22Test, ::testing::ValuesIn(LoadCases()), PrintCaseInfoString<Case>);

std::filesystem::path EvidenceRoot(const Case& p)
{
    const char* directory = std::getenv("GEMM_EVIDENCE_DIR");
    if (directory == nullptr) {
        // Standard CI/smoke runs provide no evidence directory: capture no
        // evidence, but keep every numerical and guard verification active.
        return {};
    }
    Require(
        !p.caseName.empty() &&
            p.caseName.find_first_not_of("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_") ==
                std::string::npos,
        "unsafe case_name for evidence path");
    const auto root = std::filesystem::path(directory) / p.caseName;
    std::filesystem::create_directories(root);
    Require(std::filesystem::space(root).available > 8ULL * 1024 * 1024 * 1024, "census disk reserve below8GiB");
    return root;
}

void FreezeMatrix(const std::filesystem::path& path, const Matrices& matrix)
{
    if (!std::filesystem::exists(path))
        WriteMatrices(path, matrix);
    else
        Require(SameMatrices(ReadMatrices(path, matrix), matrix), "frozen input differs");
}

void FreezeInputs(const std::filesystem::path& root, const Data& data)
{
    if (root.empty())
        return;
    FreezeMatrix(root / "input_a.bin", data.a);
    FreezeMatrix(root / "input_b.bin", data.b);
    FreezeMatrix(root / "initial_c.bin", data.initialC);
}

template <typename Invoke>
void CheckQuickReturn(const Case& p, const Invoke& invoke, aclrtStream stream_, DeviceBuffer* workspace)
{
    float value = 23.0f;
    void* dummy = &value;
    const auto status = invoke(
        p.aarrayNull ? nullptr : &dummy, p.barrayNull ? nullptr : &dummy, p.carrayNull ? nullptr : &dummy,
        p.alphaNull ? nullptr : &p.alpha, p.betaNull ? nullptr : &p.beta);
    ASSERT_EQ(status, p.expectResult);
    AclCheck(aclrtSynchronizeStream(stream_));
    ASSERT_EQ(value, 23.0f);
    workspace->checkGuards();
}

struct CaseRun {
    const Case& p;
    const std::filesystem::path& root;
    Data& data;
    aclrtStream stream_;
    DeviceBuffer* workspace;
    DeviceBuffer& alphaDevice;
    DeviceBuffer& betaDevice;
    const void* alpha;
    const void* beta;
};

template <typename Invoke>
Matrices RunFresh(const CaseRun& x, const Invoke& invoke)
{
    const auto status = invoke(x.data.ap->data(), x.data.bp->data(), x.data.cp->data(), x.alpha, x.beta);
    Require(status == ACLBLAS_STATUS_SUCCESS, "operator status " + std::to_string(status));
    AclCheck(aclrtSynchronizeStream(x.stream_));
    auto output = x.data.readC();
    if (!x.root.empty())
        WriteMatrices(x.root / "fresh_output.bin", output);
    x.data.checkGuardsAndPadding(x.p, output);
    x.workspace->checkGuards();
    x.alphaDevice.checkGuards();
    x.betaDevice.checkGuards();
    return output;
}

template <typename Invoke>
void RunSample(
    const CaseRun& x, const Invoke& invoke, int i, aclrtEvent begin, aclrtEvent end, Matrices& first,
    std::ofstream& samples)
{
    std::cout << "GEMM_ATTEMPT " << x.p.caseName << " sample=" << i << std::endl;
    AclCheck(aclrtRecordEvent(begin, x.stream_));
    const auto start = std::chrono::steady_clock::now();
    const auto status = invoke(x.data.ap->data(), x.data.bp->data(), x.data.cp->data(), x.alpha, x.beta);
    Require(status == ACLBLAS_STATUS_SUCCESS, "performance launch failed");
    AclCheck(aclrtRecordEvent(end, x.stream_));
    AclCheck(aclrtSynchronizeStream(x.stream_));
    const auto stop = std::chrono::steady_clock::now();
    float milliseconds = 0;
    AclCheck(aclrtEventElapsedTime(&milliseconds, begin, end));
    auto output = x.data.readC();
    std::string file = "output_0.bin";
    if (!x.root.empty() && (i == 0 || !SameMatrices(first, output))) {
        file = "output_" + std::to_string(i) + ".bin";
        WriteMatrices(x.root / file, output);
        if (i == 0)
            first = output;
    }
    x.data.checkGuardsAndPadding(x.p, output);
    x.workspace->checkGuards();
    samples << i << ','
            << (i < 20 ? "natural_first20" :
                i < 25 ? "warmup5" :
                         "steady60")
            << ',' << std::setprecision(12) << milliseconds * 1000.0 << ','
            << std::chrono::duration<double, std::micro>(stop - start).count() << ',' << file << '\n';
    samples.flush();
}

template <typename Invoke>
void RunPerformance(const CaseRun& x, const Invoke& invoke)
{
    Require(x.p.beta == 0.0f, "performance case must not accumulate across samples");
    std::ofstream samples;
    if (!x.root.empty()) {
        Require(!std::filesystem::exists(x.root / "samples.csv"), "refuse to repeat an existing performance case");
        samples.open(x.root / "samples.csv");
        samples << "index,phase,event_us,api_sync_us,output_file\n";
    }
    Matrices first;
    aclrtEvent begin = nullptr;
    aclrtEvent end = nullptr;
    AclCheck(aclrtCreateEvent(&begin));
    AclCheck(aclrtCreateEvent(&end));
    for (int i = 0; i < 85; ++i) {
        RunSample(x, invoke, i, begin, end, first, samples);
    }
    AclCheck(aclrtDestroyEvent(begin));
    AclCheck(aclrtDestroyEvent(end));
    x.data.checkInputs();
    std::cout << "GEMM_PERFORMANCE_PENDING_PRECISION " << x.p.caseName << " samples=85" << std::endl;
}

void RecordPolicy(const CaseRun& x)
{
    if (x.root.empty())
        return;
    if (x.p.alpha == 1.0f && x.p.beta == 0.0f && x.p.k >= 512 && static_cast<uint64_t>(x.p.m) * x.p.n > 4096U / x.p.k) {
        Require(reinterpret_cast<uint64_t>(x.workspace->data()) % 64 == 0, "policy diagnostic workspace alignment");
        std::vector<uint32_t> flags(static_cast<size_t>(x.p.batchCount) * 16U);
        AclCheck(aclrtMemcpy(
            flags.data(), flags.size() * sizeof(uint32_t), x.workspace->data(), flags.size() * sizeof(uint32_t),
            ACL_MEMCPY_DEVICE_TO_HOST));
        const auto path = x.root / "policy_flags.csv";
        Require(!std::filesystem::exists(path), "refuse to overwrite policy diagnostic");
        std::ofstream policy(path);
        policy << "batch,reference_requested\n";
        for (int batch = 0; batch < x.p.batchCount; ++batch) {
            Require(flags[batch * 16U] <= 1U, "invalid policy flag");
            policy << batch << ',' << flags[batch * 16U] << '\n';
            std::cout << "GEMM_REFERENCE_POLICY " << x.p.caseName << " batch=" << batch
                      << " reference=" << flags[batch * 16U] << std::endl;
        }
        policy.flush();
        Require(policy.good(), "policy diagnostic write failed");
    }
}

Matrices LoadGolden(const CaseRun& x)
{
    Matrices golden;
    const char* referenceCache = std::getenv("GEMM_REFERENCE_CACHE");
    const auto cached = referenceCache ? std::filesystem::path(referenceCache) / x.p.caseName : std::filesystem::path();
    if (referenceCache && std::filesystem::exists(cached / "golden.bin")) {
        Require(
            SameMatrices(ReadMatrices(cached / "input_a.bin", x.data.a), x.data.a) &&
                SameMatrices(ReadMatrices(cached / "input_b.bin", x.data.b), x.data.b) &&
                SameMatrices(ReadMatrices(cached / "initial_c.bin", x.data.initialC), x.data.initialC),
            "cached reference input identity mismatch");
        golden = ReadMatrices(cached / "golden.bin", x.data.initialC);
        std::cout << "GEMM_INDEPENDENT_REFERENCE_REUSED " << cached.string() << std::endl;
    } else {
        golden = Golden(x.p, x.data);
    }
    if (!x.root.empty()) {
        const auto goldenPath = x.root / "golden.bin";
        if (std::filesystem::exists(goldenPath)) {
            Require(
                SameMatrices(ReadMatrices(goldenPath, x.data.initialC), golden), "immutable Golden identity differs");
        } else {
            WriteMatrices(goldenPath, golden);
        }
    }
    return golden;
}

void VerifySavedOutputs(const CaseRun& x, const Matrices& golden)
{
    if (x.root.empty())
        return;
    const auto samplesPath = x.root / "samples.csv";
    if (std::filesystem::exists(samplesPath)) {
        std::ifstream samples(samplesPath);
        std::string line;
        std::getline(samples, line);
        std::vector<std::string> checked;
        size_t count = 0;
        while (std::getline(samples, line)) {
            const auto columns = CsvSplitLine(line);
            Require(columns.size() == 5, "malformed saved sample");
            const auto& file = columns[4];
            if (std::find(checked.begin(), checked.end(), file) == checked.end()) {
                Verify(x.p, ReadMatrices(x.root / file, x.data.initialC), golden, file);
                checked.push_back(file);
            }
            ++count;
        }
        Require(count == 85, "saved output coverage is incomplete");
        std::cout << (censusNumericalFailure ? "GEMM_TIMED_OUTPUTS_CHECKED_WITH_FAILURE " :
                                               "GEMM_TIMED_OUTPUTS_VERIFIED ")
                  << x.p.caseName << " samples=" << count << std::endl;
    }
}

template <typename Invoke>
void RunValidCase(
    const Case& p, const std::filesystem::path& root, bool performance, const Invoke& invoke, aclrtStream stream,
    DeviceBuffer* workspace)
{
    Data data(p);
    FreezeInputs(root, data);
    const bool deviceScalar = p.description.find("device_scalars") != std::string::npos;
    DeviceBuffer alphaDevice(&p.alpha, sizeof(float));
    DeviceBuffer betaDevice(&p.beta, sizeof(float));
    const void* alpha = deviceScalar ? alphaDevice.data() : &p.alpha;
    const void* beta = deviceScalar ? betaDevice.data() : &p.beta;
    CaseRun x{p, root, data, stream, workspace, alphaDevice, betaDevice, alpha, beta};
    if (performance) {
        RunPerformance(x, invoke);
        return;
    }
    const auto output = RunFresh(x, invoke);
    data.checkInputs();
    RecordPolicy(x);
    const auto golden = LoadGolden(x);
    Verify(p, output, golden, "fresh");
    VerifySavedOutputs(x, golden);
    censusTestCompleted = true;
}

TEST_P(GemmBatchedExArch22Test, CsvDriven)
{
    const auto& p = GetParam();
    censusNumericalFailure = false;
    censusTestCompleted = false;
    const char* phaseEnv = std::getenv("GEMM_PHASE");
    const std::string phase = phaseEnv ? phaseEnv : "precision";
    const bool performance = phase == "performance";
    const auto root = EvidenceRoot(p);
    const auto testHandle = p.description.find("handle_null") == std::string::npos ? handle_ : nullptr;
    auto invoke = [&](const void* ap, const void* bp, void* cp, const void* alpha, const void* beta) {
        return aclblasGemmBatchedEx(
            testHandle, p.transA, p.transB, p.m, p.n, p.k, alpha, static_cast<const void* const*>(ap), p.Atype, p.lda,
            static_cast<const void* const*>(bp), p.Btype, p.ldb, beta, static_cast<void* const*>(cp), p.Ctype, p.ldc,
            p.batchCount, p.computeType, p.algo);
    };
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS || p.m == 0 || p.n == 0 || p.batchCount == 0) {
        ASSERT_NO_FATAL_FAILURE(CheckQuickReturn(p, invoke, stream_, workspace.get()));
        return;
    }
    RunValidCase(p, root, performance, invoke, stream_, workspace.get());
}

} // namespace
