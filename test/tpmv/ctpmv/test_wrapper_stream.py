# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CPU-only regression for the host-buffer adapter; ACL calls are deferred stubs."""

import logging
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest


ACL_HEADER = r"""
#pragma once
#include <cstddef>
using aclError = int;
using aclrtStream = void*;
constexpr aclError ACL_SUCCESS = 0;
enum aclrtMemMallocPolicy { ACL_MEM_MALLOC_HUGE_FIRST };
enum aclrtMemcpyKind { ACL_MEMCPY_HOST_TO_DEVICE, ACL_MEMCPY_DEVICE_TO_HOST };
aclError aclrtMalloc(void**, size_t, aclrtMemMallocPolicy);
aclError aclrtFree(void*);
aclError aclrtMemcpy(void*, size_t, const void*, size_t, aclrtMemcpyKind);
aclError aclrtSynchronizeStream(aclrtStream);
aclError aclrtSynchronizeDevice();
"""

BLAS_HEADER = r"""
#pragma once
#include "acl/acl.h"
struct _aclblas_handle { aclrtStream stream; };
using aclblasHandle_t = _aclblas_handle*;
struct aclblasComplex { float real; float imag; };
enum aclblasStatus_t {
    ACLBLAS_STATUS_SUCCESS = 0, ACLBLAS_STATUS_ALLOC_FAILED = 2, ACLBLAS_STATUS_INVALID_VALUE = 3,
    ACLBLAS_STATUS_INTERNAL_ERROR = 6, ACLBLAS_STATUS_HANDLE_IS_NULLPTR = 9
};
enum aclblasFillMode_t { ACLBLAS_UPPER = 121 };
enum aclblasOperation_t { ACLBLAS_OP_N = 111 };
enum aclblasDiagType_t { ACLBLAS_NON_UNIT = 131 };
aclblasStatus_t aclblasGetStream(aclblasHandle_t, aclrtStream*);
aclblasStatus_t aclblasCtpmv(aclblasHandle_t, aclblasFillMode_t, aclblasOperation_t, aclblasDiagType_t,
                           int, const aclblasComplex*, aclblasComplex*, int);
"""

HARNESS = r"""
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#include "ctpmv_npu_wrapper.h"

namespace {
std::map<void*, size_t> buffers;
std::map<void*, int> pendingUses;
std::map<aclrtStream, std::vector<std::function<void()>>> queues;
aclblasStatus_t getStreamStatus = ACLBLAS_STATUS_SUCCESS;
aclblasStatus_t operationStatus = ACLBLAS_STATUS_SUCCESS;
aclError syncStatus = ACL_SUCCESS;
aclError readbackStatus = ACL_SUCCESS;
bool enqueueOnFailure = false;
bool unrelatedRan = false;
int failUpload = 0;
int getStreamCalls = 0;
int operationCalls = 0;
int streamSyncs = 0;
int deviceSyncs = 0;
int allocations = 0;
int frees = 0;
int uploads = 0;
int readbacks = 0;
aclrtStream syncedStream = reinterpret_cast<void*>(999);

void Check(bool condition, const char* message)
{
    if (!condition) { throw std::runtime_error(message); }
}

void Drain(aclrtStream stream)
{
    auto jobs = std::move(queues[stream]);
    queues[stream].clear();
    for (auto& job : jobs) { job(); }
}

void Cleanup()
{
    for (auto& entry : queues) { Drain(entry.first); }
    while (!buffers.empty()) { aclrtFree(buffers.begin()->first); }
}

aclblasStatus_t Run(aclblasHandle_t handle, aclblasComplex* hostX, int n = 2, int incx = 1)
{
    aclblasComplex ap[3] = {{1, 0}, {0, 0}, {1, 0}};
    return aclblasCtpmv_npu(handle, ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, n, ap, hostX, incx);
}

void Test(const std::string& name)
{
    _aclblas_handle handle{reinterpret_cast<void*>(1)};
    aclblasComplex hostX[3] = {{2, 1}, {3, -1}, {4, 2}};
    if (name == "explicit_stream" || name == "default_stream") {
        if (name == "default_stream") { handle.stream = nullptr; }
        queues[reinterpret_cast<void*>(2)].push_back([] { unrelatedRan = true; });
        Check(Run(&handle, hostX) == ACLBLAS_STATUS_SUCCESS, "normal call failed");
        Check(deviceSyncs == 0, "adapter synchronized the entire device");
        Check(streamSyncs == 1 && syncedStream == handle.stream, "wrong handle stream synchronized");
        Check(!unrelatedRan, "unrelated stream was drained");
        Check(hostX[0].real == 4 && hostX[1].real == 6, "return before copyback completion");
        Check(frees == 2 && buffers.empty(), "normal device buffers leaked");
        queues[reinterpret_cast<void*>(2)].clear();
    } else if (name == "get_stream_failure") {
        getStreamStatus = ACLBLAS_STATUS_INVALID_VALUE;
        Check(Run(&handle, hostX) == ACLBLAS_STATUS_INVALID_VALUE, "stream lookup error lost");
        Check(allocations == 0 && operationCalls == 0 && streamSyncs == 0, "work started after stream lookup failure");
    } else if (name == "submission_failure" || name == "partial_submission_failure") {
        operationStatus = ACLBLAS_STATUS_INVALID_VALUE;
        enqueueOnFailure = name == "partial_submission_failure";
        readbackStatus = 73;
        Check(Run(&handle, hostX) == ACLBLAS_STATUS_INVALID_VALUE, "operation error was overwritten");
        Check(streamSyncs == 1 && deviceSyncs == 0, "failed submission did not drain its own stream");
        Check(readbacks == 0 && hostX[0].real == 2, "failed operation copied back invalid result");
        Check(frees == 2 && queues[handle.stream].empty(), "pending work not completed before cleanup");
    } else if (name == "sync_failure" || name == "submission_and_sync_failure") {
        if (name == "submission_and_sync_failure") {
            operationStatus = ACLBLAS_STATUS_INVALID_VALUE;
            enqueueOnFailure = true;
        }
        syncStatus = 74;
        Check(Run(&handle, hostX) == ACLBLAS_STATUS_INTERNAL_ERROR, "sync failure hidden as expected validation result");
        Check(readbacks == 0 && frees == 0 && buffers.size() == 2, "uncompleted buffers read or freed");
        Check(!queues[handle.stream].empty(), "failed sync unexpectedly completed queue");
    } else if (name == "readback_failure") {
        readbackStatus = 75;
        Check(Run(&handle, hostX) == ACLBLAS_STATUS_INTERNAL_ERROR, "readback error lost");
        Check(streamSyncs == 1 && readbacks == 1 && frees == 2, "readback error cleanup incorrect");
    } else if (name == "first_upload_failure" || name == "second_upload_failure") {
        failUpload = name == "first_upload_failure" ? 1 : 2;
        Check(Run(&handle, hostX) == ACLBLAS_STATUS_INTERNAL_ERROR, "upload error lost");
        Check(operationCalls == 0 && streamSyncs == 0 && deviceSyncs == 0, "work enqueued after failed upload");
        Check(allocations == frees && buffers.empty(), "failed upload leaked buffers");
    } else if (name == "quick_return") {
        Check(Run(nullptr, hostX) == ACLBLAS_STATUS_HANDLE_IS_NULLPTR, "null handle not forwarded");
        Check(Run(&handle, hostX, 0) == ACLBLAS_STATUS_SUCCESS, "zero n not forwarded");
        Check(Run(&handle, hostX, -1) == ACLBLAS_STATUS_INVALID_VALUE, "negative n not forwarded");
        Check(getStreamCalls == 0 && allocations == 0 && streamSyncs == 0, "quick return introduced work");
    } else if (name == "negative_stride") {
        hostX[1] = {77, -77};
        Check(Run(&handle, hostX, 2, -2) == ACLBLAS_STATUS_SUCCESS, "negative stride call failed");
        Check(hostX[0].real == 4 && hostX[2].real == 8 && hostX[1].real == 77, "strided data or padding changed");
    } else {
        throw std::runtime_error("unknown test");
    }
}
}

aclError aclrtMalloc(void** ptr, size_t size, aclrtMemMallocPolicy)
{
    *ptr = std::malloc(size);
    if (*ptr == nullptr) { return 1; }
    buffers[*ptr] = size;
    allocations++;
    return ACL_SUCCESS;
}
aclError aclrtFree(void* ptr)
{
    Check(pendingUses[ptr] == 0, "buffer freed while queued work still references it");
    Check(buffers.erase(ptr) == 1, "invalid free");
    pendingUses.erase(ptr);
    frees++;
    std::free(ptr);
    return ACL_SUCCESS;
}
aclError aclrtMemcpy(void* dst, size_t capacity, const void* src, size_t bytes, aclrtMemcpyKind kind)
{
    Check(bytes <= capacity, "copy exceeds advertised capacity");
    if (kind == ACL_MEMCPY_HOST_TO_DEVICE) {
        uploads++;
        if (uploads == failUpload) { return 71; }
    } else {
        readbacks++;
        if (readbackStatus != ACL_SUCCESS) { return readbackStatus; }
    }
    std::memcpy(dst, src, bytes);
    return ACL_SUCCESS;
}
aclError aclrtSynchronizeStream(aclrtStream stream)
{
    streamSyncs++;
    syncedStream = stream;
    if (syncStatus != ACL_SUCCESS) { return syncStatus; }
    Drain(stream);
    return ACL_SUCCESS;
}
aclError aclrtSynchronizeDevice()
{
    deviceSyncs++;
    if (syncStatus != ACL_SUCCESS) { return syncStatus; }
    for (auto& entry : queues) { Drain(entry.first); }
    return ACL_SUCCESS;
}
aclblasStatus_t aclblasGetStream(aclblasHandle_t handle, aclrtStream* stream)
{
    getStreamCalls++;
    if (getStreamStatus != ACLBLAS_STATUS_SUCCESS) { return getStreamStatus; }
    *stream = handle->stream;
    return ACLBLAS_STATUS_SUCCESS;
}
aclblasStatus_t aclblasCtpmv(aclblasHandle_t handle, aclblasFillMode_t, aclblasOperation_t, aclblasDiagType_t,
                           int n, const aclblasComplex* ap, aclblasComplex* x, int incx)
{
    operationCalls++;
    if (handle == nullptr) { return ACLBLAS_STATUS_HANDLE_IS_NULLPTR; }
    if (n < 0) { return ACLBLAS_STATUS_INVALID_VALUE; }
    if (n == 0) { return ACLBLAS_STATUS_SUCCESS; }
    if (operationStatus == ACLBLAS_STATUS_SUCCESS || enqueueOnFailure) {
        auto* mutableAP = const_cast<aclblasComplex*>(ap);
        pendingUses[mutableAP]++;
        pendingUses[x]++;
        queues[handle->stream].push_back([=] {
            Check(buffers.count(mutableAP) == 1 && buffers.count(x) == 1, "queued buffers disappeared");
            for (int i = 0; i < n; ++i) {
                const size_t offset = static_cast<size_t>(i) * std::abs(incx);
                x[offset].real *= 2;
                x[offset].imag *= 2;
            }
            pendingUses[mutableAP]--;
            pendingUses[x]--;
        });
    }
    return operationStatus;
}

int main(int argc, char** argv)
{
    try {
        Check(argc == 2, "one test name required");
        Test(argv[1]);
        Cleanup();
        std::cout << "PASS " << argv[1] << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        Cleanup();
        return 1;
    }
}
"""


class WrapperStreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = os.environ.get("CXX", "c++")
        if shutil.which(compiler) is None:
            raise RuntimeError(f"C++ compiler unavailable: {compiler}")
        cls.temp = tempfile.TemporaryDirectory(prefix="ctpmv-wrapper-test-")
        cls.addClassCleanup(cls.temp.cleanup)
        root = Path(cls.temp.name)
        (root / "acl").mkdir()
        (root / "acl" / "acl.h").write_text(ACL_HEADER, encoding="utf-8")
        (root / "cann_ops_blas.h").write_text(BLAS_HEADER, encoding="utf-8")
        (root / "harness.cpp").write_text(HARNESS, encoding="utf-8")
        wrapper = Path(os.environ.get("CTPMV_WRAPPER_HEADER", Path(__file__).parent / "arch35" / "ctpmv_npu_wrapper.h"))
        shutil.copyfile(wrapper, root / "ctpmv_npu_wrapper.h")
        cls.executable = root / ("wrapper-test.exe" if os.name == "nt" else "wrapper-test")
        command = [compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-pedantic",
                   *shlex.split(os.environ.get("CXXFLAGS", "")), "-I", str(root),
                   str(root / "harness.cpp"), "-o", str(cls.executable)]
        result = subprocess.run(command, text=True, capture_output=True, timeout=60)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)

    def test_wrapper_stream_and_failure_boundaries(self):
        names = ["explicit_stream", "default_stream", "get_stream_failure", "submission_failure",
                 "partial_submission_failure", "sync_failure", "submission_and_sync_failure",
                 "readback_failure", "first_upload_failure", "second_upload_failure", "quick_return",
                 "negative_stride"]
        for name in names:
            with self.subTest(case=name):
                result = subprocess.run([str(self.executable), name], text=True, capture_output=True, timeout=15)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn(f"PASS {name}", result.stdout)
                logging.info("%s", result.stdout.strip())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    unittest.main(verbosity=2)
