/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file trsm_batched_test.cpp
 * \brief Direct-invoke test driver for StrsmBatched: reads input batches from files, runs the
 *        operator and writes the solution back for the Python accuracy check.
 *        Usage: trsm_batched_test m n batchCount side uplo transa diag [alpha]
 *        (side/uplo/transa/diag are 0/1 integers, matching the gen_data.py encoding.)
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <algorithm>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include "acl/acl.h"
#include "../../aclblas_minimal.h"

#define CHECK_ACL(expr)                                                                  \
    do {                                                                                 \
        aclError _ret = (expr);                                                          \
        if (_ret != ACL_SUCCESS) {                                                       \
            printf("[ERROR] acl call failed (%d) at %s:%d\n", _ret, __FILE__, __LINE__); \
            return _ret;                                                                 \
        }                                                                                \
    } while (0)

#define CHECK_RET(cond, msg)               \
    do {                                   \
        if (!(cond)) {                     \
            printf("[ERROR] %s\n", (msg)); \
            return -1;                     \
        }                                  \
    } while (0)

// RunOneXxxCase 返回 bool，CHECK_ACL 中的 return 的是 aclError，非 0 会被转换为 true（成功），
// 因此这里使用返回 false 的版本，失败时保持“用例失败”的语义。
#define CHECK_ACL_BOOL(expr)                                                             \
    do {                                                                                 \
        aclError _ret = (expr);                                                          \
        if (_ret != ACL_SUCCESS) {                                                       \
            printf("[ERROR] acl call failed (%d) at %s:%d\n", _ret, __FILE__, __LINE__); \
            return false;                                                                \
        }                                                                                \
    } while (0)

// 为每个 batch 分配 host/device 内存、从 .bin 读取 A/B 并拷贝到 device
static int LoadInputs(
    int32_t batchCount, size_t sizeA, size_t sizeB, std::vector<uint8_t*>& hostA, std::vector<uint8_t*>& hostB,
    std::vector<float*>& devA, std::vector<float*>& devB)
{
    for (int32_t i = 0; i < batchCount; i++) {
        CHECK_ACL(aclrtMallocHost((void**)&hostA[i], sizeA));
        CHECK_ACL(aclrtMalloc((void**)&devA[i], sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
        std::string fnameA = "input/input_a_" + std::to_string(i) + ".bin";
        ReadFile(fnameA, sizeA, hostA[i], sizeA);
        CHECK_ACL(aclrtMemcpy(devA[i], sizeA, hostA[i], sizeA, ACL_MEMCPY_HOST_TO_DEVICE));

        CHECK_ACL(aclrtMallocHost((void**)&hostB[i], sizeB));
        CHECK_ACL(aclrtMalloc((void**)&devB[i], sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
        std::string fnameB = "input/input_b_" + std::to_string(i) + ".bin";
        ReadFile(fnameB, sizeB, hostB[i], sizeB);
        CHECK_ACL(aclrtMemcpy(devB[i], sizeB, hostB[i], sizeB, ACL_MEMCPY_HOST_TO_DEVICE));
    }
    return 0;
}

// 将 device 结果回写 .bin，并释放所有 host/device 内存（未分配的指针保持 nullptr，跳过）
static int StoreOutputsAndFree(
    int32_t batchCount, size_t sizeB, std::vector<uint8_t*>& hostA, std::vector<uint8_t*>& hostB,
    std::vector<float*>& devA, std::vector<float*>& devB)
{
    for (int32_t i = 0; i < batchCount; i++) {
        CHECK_ACL(aclrtMemcpy(hostB[i], sizeB, devB[i], sizeB, ACL_MEMCPY_DEVICE_TO_HOST));
        std::string fname = "output/output_b_" + std::to_string(i) + ".bin";
        WriteFile(fname, hostB[i], sizeB);
    }
    FreeBatchPtrs(devA, hostA);
    FreeBatchPtrs(devB, hostB);
    return 0;
}

// 解析命令行参数并映射为算子枚举与尺寸
struct TrsmArgs {
    int32_t m, n, batchCount, sideI;
    aclblasSideMode_t side;
    aclblasFillMode_t uplo;
    aclblasOperation_t transa;
    aclblasDiagType_t diag;
    float alpha;
    int32_t lda, ldb;
    size_t sizeA, sizeB;
};

static TrsmArgs ParseTrsmArgs(int32_t argc, char* argv[])
{
    TrsmArgs a;
    a.m = (argc > 1) ? atoi(argv[1]) : 16;
    a.n = (argc > 2) ? atoi(argv[2]) : 16;
    a.batchCount = (argc > 3) ? atoi(argv[3]) : 1;
    a.sideI = (argc > 4) ? atoi(argv[4]) : 0;
    int32_t uploI = (argc > 5) ? atoi(argv[5]) : 0;
    int32_t transI = (argc > 6) ? atoi(argv[6]) : 0;
    int32_t diagI = (argc > 7) ? atoi(argv[7]) : 0;
    a.alpha = (argc > 8) ? (float)atof(argv[8]) : 1.0f;
    a.side = (a.sideI == 0) ? ACLBLAS_SIDE_LEFT : ACLBLAS_SIDE_RIGHT;
    a.uplo = (uploI == 0) ? ACLBLAS_UPPER : ACLBLAS_LOWER;
    a.transa = (transI == 0) ? ACLBLAS_OP_N : ACLBLAS_OP_T;
    a.diag = (diagI == 0) ? ACLBLAS_NON_UNIT : ACLBLAS_UNIT;
    a.lda = (a.sideI == 0) ? a.m : a.n;
    a.ldb = a.n;
    a.sizeA = (size_t)((a.sideI == 0) ? a.m : a.n) * a.lda * sizeof(float);
    // B 分配长度取 max(真实足迹 (n-1)*ldb+m, 旧式 m*ldb)：覆盖 host alpha=0 的整段
    // 清零范围与 kernel 的读写范围（m<n 默认步长时旧式 m*ldb 小于足迹，会越界写）
    a.sizeB = std::max((size_t)(a.n - 1) * a.ldb + a.m, (size_t)a.m * a.ldb) * sizeof(float);
    return a;
}

// 配置文件中的一个用例
struct CfgCase {
    int32_t m, n, batch, side, uplo, transa, diag;
    float alpha;
    std::string tag;
    int32_t ldb = 0; // 可选列（tag 之后）：>0 时按该步长调用；0 表示沿用默认 ldb=n
    int32_t expect = 0; // 可选列（tag 之后）：1 表示期望被入参校验拒绝（ACLBLAS_STATUS_INVALID_VALUE）
};

// 运行单个配置用例：从 <tag>/input_*.bin 加载 -> 调用算子 -> 成功则写 <tag>/output_b_*.bin
// issue #479: 失败路径统一释放本用例已分配的设备/主机内存（未分配的指针保持 nullptr，跳过）
static void FreeTrsmCaseBuffers(
    std::vector<uint8_t*>& hostA, std::vector<float*>& devA, std::vector<uint8_t*>& hostB, std::vector<float*>& devB)
{
    FreeBatchPtrs(devA, hostA);
    FreeBatchPtrs(devB, hostB);
}

// 普通用例（期望成功且 alpha!=0）的输入加载：分配 host/device 内存、读取 .bin 并拷贝到 device
// 失败时不做内部释放，由调用方统一走 FreeTrsmCaseBuffers（未分配的指针保持 nullptr，跳过）
static bool LoadTrsmCaseBuffers(
    const CfgCase& c, size_t sizeA, size_t sizeB, std::vector<uint8_t*>& hostA, std::vector<uint8_t*>& hostB,
    std::vector<float*>& devA, std::vector<float*>& devB)
{
    for (int32_t i = 0; i < c.batch; i++) {
        // issue #479: 分配失败时不能继续对空指针拷贝，需报错并返回失败
        if (aclrtMallocHost((void**)&hostA[i], sizeA) != ACL_SUCCESS ||
            aclrtMalloc((void**)&devA[i], sizeA, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
            aclrtMallocHost((void**)&hostB[i], sizeB) != ACL_SUCCESS ||
            aclrtMalloc((void**)&devB[i], sizeB, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
            printf("[ERROR] aclrtMalloc/aclrtMallocHost failed at %s:%d\n", __FILE__, __LINE__);
            return false;
        }
        std::string fnA = c.tag + "/input_a_" + std::to_string(i) + ".bin";
        ReadFile(fnA, sizeA, hostA[i], sizeA);
        if (aclrtMemcpy(devA[i], sizeA, hostA[i], sizeA, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
            printf("[ERROR] aclrtMemcpy failed at %s:%d\n", __FILE__, __LINE__);
            return false;
        }
        std::string fnB = c.tag + "/input_b_" + std::to_string(i) + ".bin";
        ReadFile(fnB, sizeB, hostB[i], sizeB);
        if (aclrtMemcpy(devB[i], sizeB, hostB[i], sizeB, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
            printf("[ERROR] aclrtMemcpy failed at %s:%d\n", __FILE__, __LINE__);
            return false;
        }
    }
    return true;
}

// alpha=0 用例仅整段清零 B：只分配 B 的 host/device 缓冲（无需输入数据文件）
static bool AllocTrsmZeroCaseBuffers(
    const CfgCase& c, size_t sizeB, std::vector<uint8_t*>& hostB, std::vector<float*>& devB)
{
    for (int32_t i = 0; i < c.batch; i++) {
        if (aclrtMallocHost((void**)&hostB[i], sizeB) != ACL_SUCCESS ||
            aclrtMalloc((void**)&devB[i], sizeB, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
            printf("[ERROR] aclrtMalloc/aclrtMallocHost failed at %s:%d\n", __FILE__, __LINE__);
            return false;
        }
    }
    return true;
}

// alpha=0 语义（X := 0）：断言整段 B 足迹（含最后一列）被清零
static bool VerifyTrsmZeroed(size_t sizeB, const std::vector<uint8_t*>& hostB)
{
    const size_t count = sizeB / sizeof(float);
    for (size_t i = 0; i < hostB.size(); i++) {
        const float* vals = reinterpret_cast<const float*>(hostB[i]);
        for (size_t k = 0; k < count; k++) {
            if (vals[k] != 0.0f) {
                printf("[ERROR] batch %zu: B[%zu]=%g not zeroed\n", i, k, vals[k]);
                return false;
            }
        }
    }
    return true;
}

static bool RunOneTrsmCase(_aclblas_handle& handle, const CfgCase& c)
{
    aclblasSideMode_t side = (c.side == 0) ? ACLBLAS_SIDE_LEFT : ACLBLAS_SIDE_RIGHT;
    aclblasFillMode_t uplo = (c.uplo == 0) ? ACLBLAS_UPPER : ACLBLAS_LOWER;
    aclblasOperation_t transa = (c.transa == 0) ? ACLBLAS_OP_N : ACLBLAS_OP_T;
    aclblasDiagType_t diag = (c.diag == 0) ? ACLBLAS_NON_UNIT : ACLBLAS_UNIT;
    int32_t lda = (c.side == 0) ? c.m : c.n;
    // 可选 ldb 列：>0 时使用配置步长（覆盖紧致 ldb=m 等场景）；0 时沿用原默认 ldb=n
    int32_t ldb = (c.ldb > 0) ? c.ldb : c.n;
    size_t sizeA = (size_t)((c.side == 0) ? c.m : c.n) * lda * sizeof(float);
    // 自定义 ldb 时按 B 的真实足迹分配（列主序 m×n、步长 ldb，末元素下标 (n-1)*ldb + m-1）
    // 两种分支统一取 max(足迹, m*ldb)：覆盖 host alpha=0 的整段清零范围与 kernel 读写范围
    size_t sizeB = std::max((size_t)(c.n - 1) * ldb + c.m, (size_t)c.m * ldb) * sizeof(float);

    std::vector<uint8_t*> hostA(c.batch), hostB(c.batch);
    std::vector<float*> devA(c.batch), devB(c.batch);
    // 校验拒绝用例（expect=1）不触碰设备内存；alpha=0 用例仅整段清零 B，均无需输入数据文件
    bool loaded = true;
    if (c.expect == 0 && c.alpha == 0.0f) {
        loaded = AllocTrsmZeroCaseBuffers(c, sizeB, hostB, devB);
    } else if (c.expect == 0) {
        loaded = LoadTrsmCaseBuffers(c, sizeA, sizeB, hostA, hostB, devA, devB);
    }
    if (!loaded) {
        FreeTrsmCaseBuffers(hostA, devA, hostB, devB);
        return false;
    }
    std::vector<const float*> aArray(devA.begin(), devA.end());
    std::vector<float*> bArray(devB.begin(), devB.end());
    aclblasStatus_t status = aclblasStrsmBatchedEx(
        (aclblasHandle_t)&handle, side, uplo, transa, diag, c.m, c.n, &c.alpha, aArray.data(), lda, bArray.data(), ldb,
        c.batch);
    // expect=1 的用例期望被入参校验拒绝；其余用例要求算子返回 SUCCESS
    bool ok = (c.expect == 1) ? (status == ACLBLAS_STATUS_INVALID_VALUE) : (status == ACLBLAS_STATUS_SUCCESS);
    if (ok && status == ACLBLAS_STATUS_SUCCESS) {
        for (int32_t i = 0; i < c.batch; i++) {
            CHECK_ACL_BOOL(aclrtMemcpy(hostB[i], sizeB, devB[i], sizeB, ACL_MEMCPY_DEVICE_TO_HOST));
            std::string fn = c.tag + "/output_b_" + std::to_string(i) + ".bin";
            WriteFile(fn, hostB[i], sizeB);
        }
        if (c.alpha == 0.0f) {
            ok = VerifyTrsmZeroed(sizeB, hostB);
        }
    }
    FreeTrsmCaseBuffers(hostA, devA, hostB, devB);
    return ok;
}

// 配置文件批量模式：一次进程内顺序运行全部用例（设备/流仅初始化一次）
static int RunConfigMode(const char* cfgPath)
{
    std::ifstream cfg(cfgPath);
    if (!cfg.is_open()) {
        printf("[ERROR] cannot open config %s\n", cfgPath);
        return 1;
    }
    std::vector<CfgCase> cases;
    std::string line;
    while (std::getline(cfg, line)) {
        if (line.empty() || line[0] == '#')
            continue;
        CfgCase c;
        std::istringstream ss(line);
        ss >> c.m >> c.n >> c.batch >> c.side >> c.uplo >> c.transa >> c.diag >> c.alpha >> c.tag;
        if (!ss.fail()) {
            // 可选列 ldb/expect 语义见 aclblas_minimal.h 的 ParseCfgOptionalColumns
            ParseCfgOptionalColumns(ss, c.ldb, c.expect);
            cases.push_back(c);
        }
    }
    int32_t deviceId = 0;
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));
    _aclblas_handle handle;
    handle.stream = stream;
    int total = 0, ok = 0, fail = 0;
    for (size_t i = 0; i < cases.size(); i++) {
        total++;
        if (RunOneTrsmCase(handle, cases[i])) {
            ok++;
            printf("  [%zu/%zu] %s OK\n", i + 1, cases.size(), cases[i].tag.c_str());
        } else {
            fail++;
            printf("  [%zu/%zu] %s KERNEL_FAIL\n", i + 1, cases.size(), cases[i].tag.c_str());
        }
    }
    printf("Kernel done: %d/%d passed\n", ok, total);
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
    return fail > 0 ? 1 : 0;
}

int32_t main(int32_t argc, char* argv[])
{
    // 配置文件批量模式：argv[1] 为非数字（文件路径）时，逐行运行全部用例（共享 handle）
    if (argc == 2 && argv[1][0] != '\0' && !std::isdigit((unsigned char)argv[1][0])) {
        return RunConfigMode(argv[1]);
    }
    TrsmArgs a = ParseTrsmArgs(argc, argv);

    int32_t deviceId = 0;
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    _aclblas_handle handle;
    handle.stream = stream;

    std::vector<uint8_t*> hostA(a.batchCount), hostB(a.batchCount);
    std::vector<float*> devA(a.batchCount), devB(a.batchCount);
    // alpha=0 仅整段清零 B：跳过输入文件加载（A 不参与，B 由算子清零），避免
    // m<n 时足迹大于 gen_data 的 m*n 文件长度导致 ReadFile 失败
    if (a.alpha == 0.0f) {
        for (int32_t i = 0; i < a.batchCount; i++) {
            if (aclrtMallocHost((void**)&hostB[i], a.sizeB) != ACL_SUCCESS ||
                aclrtMalloc((void**)&devB[i], a.sizeB, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
                printf("[ERROR] aclrtMalloc/aclrtMallocHost failed at %s:%d\n", __FILE__, __LINE__);
                return -1;
            }
        }
    } else {
        int loadRet = LoadInputs(a.batchCount, a.sizeA, a.sizeB, hostA, hostB, devA, devB);
        if (loadRet != 0)
            return loadRet;
    }

    std::vector<const float*> aArray(devA.begin(), devA.end());
    std::vector<float*> bArray(devB.begin(), devB.end());

    aclblasStatus_t status = aclblasStrsmBatchedEx(
        (aclblasHandle_t)&handle, a.side, a.uplo, a.transa, a.diag, a.m, a.n, &a.alpha, aArray.data(), a.lda,
        bArray.data(), a.ldb, a.batchCount);
    CHECK_RET(status == ACLBLAS_STATUS_SUCCESS, "aclblasStrsmBatchedEx failed");

    int storeRet = StoreOutputsAndFree(a.batchCount, a.sizeB, hostA, hostB, devA, devB);
    if (storeRet != 0)
        return storeRet;

    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
    return 0;
}
