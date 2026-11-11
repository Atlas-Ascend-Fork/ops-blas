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

#include <acl/acl.h>
#include <cstddef>
#include <cstdint>
#include <complex>
#include <cstdio>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include "../include/cann_ops_blas_common.h"

// GM_ADDR type for kernel function declarations
#define GM_ADDR uint8_t*

// Handle type
typedef void* aclblasHandle_t;

// Internal handle structure
struct _aclblas_handle {
    aclrtStream stream = nullptr;
    void* default_workspace = nullptr;
    size_t default_workspace_size = 0;
    void* user_workspace = nullptr;
    size_t user_workspace_size = 0;
    bool use_user_workspace = false;
};

// API declarations for TrsmBatched
#ifdef __cplusplus
extern "C" {
#endif

aclblasStatus_t aclblasStrsmBatchedEx(
    aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo, aclblasOperation_t trans,
    aclblasDiagType_t diag, int64_t m, int64_t n, const float* alpha, const float* const A[], int64_t lda,
    float* const B[], int64_t ldb, int64_t batchCount);

#ifdef __cplusplus
}
#endif

// C++ only API (uses std::complex)
#ifdef __cplusplus
aclblasStatus_t aclblasCtrsmBatched(
    aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo, aclblasOperation_t trans,
    aclblasDiagType_t diag, int64_t m, int64_t n, const std::complex<float>* alpha,
    const std::complex<float>* const A[], int64_t lda, std::complex<float>* const B[], int64_t ldb, int64_t batchCount);
#endif

// ---------------------------------------------------------------------------
// Test file I/O helpers (previously in each op's test/data_utils.h)
// ---------------------------------------------------------------------------

#ifndef ERROR_LOG
#define ERROR_LOG(fmt, args...) fprintf(stdout, "[ERROR]  " fmt "\n", ##args)
#endif

// 配置用例行可选尾列解析（trsm/ctrsm 批量测试 harness 共用）：
// ldb（0=默认 n）、expect（1=期望入参校验拒绝），缺失或非法时保持默认值
template <typename T>
inline void ParseCfgOptionalColumns(std::istringstream& ss, T& ldb, T& expect)
{
    if (!(ss >> ldb) || ldb < 0) {
        ldb = 0;
    }
    ss.clear();
    if (!(ss >> expect) || expect < 0) {
        expect = 0;
    }
}

// 批量释放 device/host 指针数组（未分配的指针保持 nullptr，跳过）：device 侧
// aclrtFree、host 侧 aclrtFreeHost（trsm/ctrsm 测试 harness 共用，两数组按 batch 等长）
template <typename TD, typename TH>
inline void FreeBatchPtrs(std::vector<TD*>& dev, std::vector<TH*>& host)
{
    for (size_t i = 0; i < host.size(); i++) {
        if (dev[i] != nullptr) {
            aclrtFree(dev[i]);
        }
        if (host[i] != nullptr) {
            aclrtFreeHost(host[i]);
        }
    }
}

/**
 * @brief Read data from file
 * @param [in] filePath: file path
 * @param [in] bufferSize: expected file size
 * @param [out] buffer: buffer to store data
 * @param [in] bufferLen: buffer length
 * @return read result
 */
inline bool ReadFile(const std::string& filePath, size_t bufferSize, void* buffer, size_t bufferLen)
{
    if (buffer == nullptr) {
        ERROR_LOG("buffer is nullptr");
        return false;
    }
    if (bufferSize > bufferLen) {
        ERROR_LOG("buffer size is larger than buffer length");
        return false;
    }

    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        ERROR_LOG("Open file failed. path = %s", filePath.c_str());
        return false;
    }

    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    if (fileSize != bufferSize) {
        ERROR_LOG("file size %zu != expected size %zu", fileSize, bufferSize);
        file.close();
        return false;
    }

    file.read(static_cast<char*>(buffer), bufferSize);
    if (!file) {
        ERROR_LOG("Read file failed");
        file.close();
        return false;
    }

    file.close();
    return true;
}

/**
 * @brief Write data to file
 * @param [in] filePath: file path
 * @param [in] buffer: data to write to file
 * @param [in] size: size to write
 * @return write result
 */
inline bool WriteFile(const std::string& filePath, const void* buffer, size_t size)
{
    if (buffer == nullptr) {
        ERROR_LOG("Write file failed. buffer is nullptr");
        return false;
    }

    int fd = open(filePath.c_str(), O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
    if (fd < 0) {
        ERROR_LOG("Open file failed. path = %s", filePath.c_str());
        return false;
    }

    ssize_t writeSize = write(fd, buffer, size);
    close(fd);
    if (static_cast<size_t>(writeSize) != size) {
        ERROR_LOG("Write file Failed.");
        return false;
    }

    return true;
}
