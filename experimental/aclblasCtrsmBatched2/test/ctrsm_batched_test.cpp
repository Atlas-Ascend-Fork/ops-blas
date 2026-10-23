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
 * \file ctrsm_batched_test.cpp
 * \brief CtrsmBatched operator test
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <complex>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include "acl/acl.h"
#include "../../aclblas_minimal.h"

#define CHECK_ACL(expr) do { aclError _r = (expr); if (_r != ACL_SUCCESS) { \
    std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << _r << std::endl; return _r; } } while(0)

// RunOneCtrsmCase 返回 bool，CHECK_ACL 中的 return 的是 aclError，非 0 会被转换为 true（成功），
// 因此这里使用返回 false 的版本，失败时保持“用例失败”的语义。
#define CHECK_ACL_BOOL(expr) do { aclError _r = (expr); if (_r != ACL_SUCCESS) { \
    std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << _r << std::endl; return false; } } while(0)

// 配置文件模式：逐行解析用例，批量执行并统计通过数（供 msprof/回归共享 handle）
struct CfgCase { int m, n, batch, side, uplo, transa, diag; float ar, ai; std::string tag; };

// 解析配置文件为用例列表（每行: m n batch side uplo transa diag ar ai tag，# 开头为注释）
static void ParseConfigCases(std::ifstream& cfg, std::vector<CfgCase>& cases)
{
    std::string line;
    while (std::getline(cfg, line)) {
        if (line.empty() || line[0] == '#') continue;
        CfgCase c; std::istringstream ss(line);
        ss >> c.m >> c.n >> c.batch >> c.side >> c.uplo >> c.transa >> c.diag >> c.ar >> c.ai >> c.tag;
        if (!ss.fail()) cases.push_back(c);
    }
}

// 运行单个配置用例：分配/加载输入 -> 调用算子 -> 成功则回写输出，返回是否成功
// issue #479: 失败路径统一释放本用例已分配的设备/主机内存（未分配的指针保持 nullptr，跳过）
static void FreeCtrsmCaseBuffers(std::vector<std::complex<float>*>& hA, std::vector<void*>& dA,
                                 std::vector<std::complex<float>*>& hB, std::vector<void*>& dB)
{
    for (size_t b = 0; b < hA.size(); b++) {
        if (dA[b] != nullptr) { aclrtFree(dA[b]); }
        if (hA[b] != nullptr) { aclrtFreeHost(hA[b]); }
        if (dB[b] != nullptr) { aclrtFree(dB[b]); }
        if (hB[b] != nullptr) { aclrtFreeHost(hB[b]); }
    }
}

static bool RunOneCtrsmCase(_aclblas_handle& handle, const CfgCase& c)
{
    aclblasSideMode_t sm = c.side==0 ? ACLBLAS_SIDE_LEFT : ACLBLAS_SIDE_RIGHT;
    aclblasFillMode_t um = c.uplo==0 ? ACLBLAS_UPPER : ACLBLAS_LOWER;
    aclblasOperation_t tm = c.transa==0 ? ACLBLAS_OP_N : c.transa==1 ? ACLBLAS_OP_T : ACLBLAS_OP_C;
    aclblasDiagType_t dm = c.diag==0 ? ACLBLAS_NON_UNIT : ACLBLAS_UNIT;
    std::complex<float> alpha(c.ar, c.ai);
    int kDim = c.side==0 ? c.m : c.n;
    int lda=kDim, ldb=c.n;
    size_t sA=(size_t)kDim*lda*2*sizeof(float), sB=(size_t)c.m*ldb*2*sizeof(float);
    std::vector<std::complex<float>*> hA(c.batch), hB(c.batch);
    std::vector<void*> dA(c.batch), dB(c.batch);
    for (int b=0;b<c.batch;b++) {
        // issue #479: 分配失败时不能继续对空指针拷贝，需报错并释放已分配内存
        if (aclrtMallocHost((void**)&hA[b],sA) != ACL_SUCCESS || aclrtMalloc(&dA[b],sA,ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
            aclrtMallocHost((void**)&hB[b],sB) != ACL_SUCCESS || aclrtMalloc(&dB[b],sB,ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
            printf("[ERROR] aclrtMalloc/aclrtMallocHost failed at %s:%d\n", __FILE__, __LINE__);
            FreeCtrsmCaseBuffers(hA,dA,hB,dB);
            return false;
        }
        std::string fnA = c.tag + "/input_a_" + std::to_string(b) + ".bin"; ReadFile(fnA,sA,hA[b],sA);
        if (aclrtMemcpy(dA[b],sA,hA[b],sA,ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
            printf("[ERROR] aclrtMemcpy failed at %s:%d\n", __FILE__, __LINE__);
            FreeCtrsmCaseBuffers(hA,dA,hB,dB);
            return false;
        }
        std::string fnB = c.tag + "/input_b_" + std::to_string(b) + ".bin"; ReadFile(fnB,sB,hB[b],sB);
        if (aclrtMemcpy(dB[b],sB,hB[b],sB,ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
            printf("[ERROR] aclrtMemcpy failed at %s:%d\n", __FILE__, __LINE__);
            FreeCtrsmCaseBuffers(hA,dA,hB,dB);
            return false;
        }
    }
    auto ret = aclblasCtrsmBatched((aclblasHandle_t)&handle,sm,um,tm,dm,c.m,c.n,&alpha,
        (const std::complex<float>*const*)dA.data(),lda,(std::complex<float>*const*)dB.data(),ldb,c.batch);
    bool okCase = (ret==ACLBLAS_STATUS_SUCCESS);
    if (okCase) {
        for (int b=0;b<c.batch;b++) {
            CHECK_ACL_BOOL(aclrtMemcpy(hB[b],sB,dB[b],sB,ACL_MEMCPY_DEVICE_TO_HOST));
            std::string fn = c.tag + "/output_b_" + std::to_string(b) + ".bin"; WriteFile(fn,hB[b],sB);
        }
    }
    for (int b=0;b<c.batch;b++) { aclrtFree(dA[b]);aclrtFreeHost(hA[b]);aclrtFree(dB[b]);aclrtFreeHost(hB[b]); }
    return okCase;
}

static int RunConfigMode(const char* cfgPath)
{
    std::ifstream cfg(cfgPath);
    if (!cfg.is_open()) { std::cerr << "Cannot open " << cfgPath << std::endl; return 1; }
    std::vector<CfgCase> cases;
    ParseConfigCases(cfg, cases);

    int deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));
    _aclblas_handle handle; handle.stream = stream;
    int total=0, ok=0, fail=0;
    for (size_t i = 0; i < cases.size(); i++) {
        total++;
        if (RunOneCtrsmCase(handle, cases[i])) {
            ok++; printf("  [%zu/%zu] %s OK\n",i+1,cases.size(),cases[i].tag.c_str());
        } else {
            fail++; printf("  [%zu/%zu] %s KERNEL_FAIL\n",i+1,cases.size(),cases[i].tag.c_str());
        }
    }
    printf("Kernel done: %d/%d passed\n",ok,total);
    CHECK_ACL(aclrtDestroyStream(stream)); CHECK_ACL(aclrtResetDevice(deviceId)); CHECK_ACL(aclFinalize());
    return fail>0 ? 1 : 0;
}

// 单用例参数：由命令行解析而来
struct SingleArgs {
    int deviceId, m, n, batch, side, uplo, transa, diag;
    float alphaRe, alphaIm;
};

static SingleArgs ParseSingleArgs(int argc, char** argv)
{
    SingleArgs a;
    a.deviceId = std::atoi(argv[1]);
    a.m = (argc > 2) ? std::atoi(argv[2]) : 64;
    a.n = (argc > 3) ? std::atoi(argv[3]) : 64;
    a.batch = (argc > 4) ? std::atoi(argv[4]) : 1;
    a.side = (argc > 5) ? std::atoi(argv[5]) : 0;
    a.uplo = (argc > 6) ? std::atoi(argv[6]) : 0;
    a.transa = (argc > 7) ? std::atoi(argv[7]) : 0;
    a.diag = (argc > 8) ? std::atoi(argv[8]) : 0;
    a.alphaRe = (argc > 9) ? std::atof(argv[9]) : 1.0f;
    a.alphaIm = (argc > 10) ? std::atof(argv[10]) : 0.0f;
    return a;
}

// 加载 batch 组输入 A/B 到 host 与 device
static void LoadSingleInputs(int batch, size_t sizeA, size_t sizeB,
    std::vector<std::complex<float>*>& hostA, std::vector<std::complex<float>*>& hostB,
    std::vector<void*>& devA, std::vector<void*>& devB)
{
    for (int i = 0; i < batch; i++) {
        aclrtMallocHost((void**)&hostA[i], sizeA);
        aclrtMalloc(&devA[i], sizeA, ACL_MEM_MALLOC_HUGE_FIRST);
        std::string fnameA = "./test/data/input/input_a_" + std::to_string(i) + ".bin";
        ReadFile(fnameA, sizeA, hostA[i], sizeA);
        aclrtMemcpy(devA[i], sizeA, hostA[i], sizeA, ACL_MEMCPY_HOST_TO_DEVICE);

        aclrtMallocHost((void**)&hostB[i], sizeB);
        aclrtMalloc(&devB[i], sizeB, ACL_MEM_MALLOC_HUGE_FIRST);
        std::string fnameB = "./test/data/input/input_b_" + std::to_string(i) + ".bin";
        ReadFile(fnameB, sizeB, hostB[i], sizeB);
        aclrtMemcpy(devB[i], sizeB, hostB[i], sizeB, ACL_MEMCPY_HOST_TO_DEVICE);
    }
}

// 回写 device 输出到 .bin，并释放所有 host/device 内存
static void StoreSingleOutputsAndFree(int batch, size_t sizeB,
    std::vector<std::complex<float>*>& hostA, std::vector<std::complex<float>*>& hostB,
    std::vector<void*>& devA, std::vector<void*>& devB)
{
    for (int i = 0; i < batch; i++) {
        aclrtMemcpy(hostB[i], sizeB, devB[i], sizeB, ACL_MEMCPY_DEVICE_TO_HOST);
        std::string fname = "./test/data/output/output_b_" + std::to_string(i) + ".bin";
        WriteFile(fname, hostB[i], sizeB);
    }
    for (int i = 0; i < batch; i++) {
        aclrtFree(devA[i]); aclrtFreeHost(hostA[i]);
        aclrtFree(devB[i]); aclrtFreeHost(hostB[i]);
    }
}

// 单用例 CLI 模式：从命令行参数运行一次算子并回写输出
static int RunSingleCase(int argc, char** argv)
{
    SingleArgs a = ParseSingleArgs(argc, argv);

    CHECK_ACL(aclrtSetDevice(a.deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));
    _aclblas_handle handle;
    handle.stream = stream;

    aclblasSideMode_t sideMode = (a.side == 0) ? ACLBLAS_SIDE_LEFT : ACLBLAS_SIDE_RIGHT;
    aclblasFillMode_t uploMode = (a.uplo == 0) ? ACLBLAS_UPPER : ACLBLAS_LOWER;
    aclblasOperation_t transMode = (a.transa == 0) ? ACLBLAS_OP_N
                                 : (a.transa == 1) ? ACLBLAS_OP_T : ACLBLAS_OP_C;
    aclblasDiagType_t diagMode = (a.diag == 0) ? ACLBLAS_NON_UNIT : ACLBLAS_UNIT;
    std::complex<float> alpha(a.alphaRe, a.alphaIm);

    int kDim = (a.side == 0) ? a.m : a.n;
    int lda = kDim, ldb = a.n;
    size_t sizeA = (size_t)kDim * lda * 2 * sizeof(float);
    size_t sizeB = (size_t)a.m * ldb * 2 * sizeof(float);

    std::vector<std::complex<float>*> hostA(a.batch), hostB(a.batch);
    std::vector<void*> devA(a.batch), devB(a.batch);
    LoadSingleInputs(a.batch, sizeA, sizeB, hostA, hostB, devA, devB);

    std::cout << "[Info] Running CtrsmBatched: m=" << a.m << " n=" << a.n << " batch=" << a.batch
              << " side=" << a.side << " uplo=" << a.uplo << " transa=" << a.transa
              << " diag=" << a.diag << " alpha=(" << a.alphaRe << "," << a.alphaIm << ")" << std::endl;

    auto ret = aclblasCtrsmBatched(
        (aclblasHandle_t)&handle, sideMode, uploMode, transMode, diagMode,
        a.m, a.n, &alpha,
        (const std::complex<float>* const*)devA.data(), lda,
        (std::complex<float>* const*)devB.data(), ldb,
        a.batch);

    if (ret != ACLBLAS_STATUS_SUCCESS) {
        std::cerr << "[Error] aclblasCtrsmBatched failed. ERROR: " << ret << std::endl;
        return ret;
    }

    StoreSingleOutputsAndFree(a.batch, sizeB, hostA, hostB, devA, devB);
    std::cout << "[Info] Kernel execution succeeded. Output written to ./test/data/output/" << std::endl;

    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(a.deviceId));
    CHECK_ACL(aclFinalize());
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <deviceId> <m> <n> <batch> <side> <uplo> <transa> <diag> [alpha_re] [alpha_im]" << std::endl;
        std::cerr << "       " << argv[0] << " <config_file>" << std::endl;
        return 1;
    }

    CHECK_ACL(aclInit(nullptr));

    bool configMode = (argc == 2 || (argc >= 2 && argv[1][0] != '0' && !std::isdigit(argv[1][0])));
    if (argc == 2) configMode = true;

    if (configMode) {
        return RunConfigMode(argv[1]);
    }
    return RunSingleCase(argc, argv);
}
