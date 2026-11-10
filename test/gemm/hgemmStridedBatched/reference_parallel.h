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
#include <algorithm>
#include <cerrno>
#include <cfenv>
#include <cstdlib>
#include <dlfcn.h>
#include <fcntl.h>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <spawn.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>
#include <cblas.h>

namespace hgemm_reference {
constexpr uint64_t PARALLEL_MIN_VOLUME = 8'000'000'000ULL;
constexpr int MAX_REFERENCE_WORKERS = 32;
constexpr int SMALL_REFERENCE_WORKERS = 4;
constexpr int FIRST_PRIVATE_FD = 64;
constexpr int WORKER_A_FD = 3;
constexpr int WORKER_B_FD = 4;
constexpr int WORKER_C_FD = 5;
struct Environment {
    fenv_t saved;
    Environment()
    {
        if (std::fegetenv(&saved) != 0)
            throw std::runtime_error("Cannot read CPU floating environment");
        if (std::fesetenv(FE_DFL_ENV) != 0) {
            std::fesetenv(&saved);
            throw std::runtime_error("Cannot set CPU reference environment");
        }
    }
    ~Environment() { std::fesetenv(&saved); }
};
struct SharedFloats {
    int fd = -1;
    void* data = MAP_FAILED;
    size_t bytes;
    explicit SharedFloats(const std::vector<float>& input) : bytes(input.size() * sizeof(float))
    {
        int original = memfd_create("hgemm-netlib", MFD_CLOEXEC);
        if (original < 0)
            throw std::runtime_error("memfd_create failed");
        fd = fcntl(original, F_DUPFD_CLOEXEC, FIRST_PRIVATE_FD);
        close(original);
        if (fd < 0 || ftruncate(fd, bytes) != 0) {
            if (fd >= 0)
                close(fd);
            throw std::runtime_error("reference shared memory allocation failed");
        }
        data = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (data == MAP_FAILED) {
            close(fd);
            throw std::runtime_error("reference mmap failed");
        }
        std::copy_n(reinterpret_cast<const unsigned char*>(input.data()), bytes, static_cast<unsigned char*>(data));
    }
    SharedFloats(const SharedFloats&) = delete;
    SharedFloats& operator=(const SharedFloats&) = delete;
    ~SharedFloats()
    {
        if (data != MAP_FAILED)
            munmap(data, bytes);
        if (fd >= 0)
            close(fd);
    }
};

inline std::string ExactFloat(float value)
{
    std::ostringstream s;
    s << std::setprecision(std::numeric_limits<float>::max_digits10) << value;
    return s.str();
}

inline pid_t SpawnWorker(
    std::vector<std::string>& text, const SharedFloats& sa, const SharedFloats& sb, const SharedFloats& sc)
{
    std::vector<char*> argv;
    for (auto& s : text)
        argv.push_back(s.data());
    argv.push_back(nullptr);
    posix_spawn_file_actions_t actions;
    if (posix_spawn_file_actions_init(&actions) != 0) {
        return -1;
    }
    bool valid = posix_spawn_file_actions_adddup2(&actions, sa.fd, WORKER_A_FD) == 0 &&
                 posix_spawn_file_actions_adddup2(&actions, sb.fd, WORKER_B_FD) == 0 &&
                 posix_spawn_file_actions_adddup2(&actions, sc.fd, WORKER_C_FD) == 0;
    pid_t pid = 0;
    const int code = valid ? posix_spawnp(&pid, "python3", &actions, nullptr, argv.data(), ::environ) : EINVAL;
    posix_spawn_file_actions_destroy(&actions);
    return code == 0 ? pid : -1;
}

inline bool WaitForWorkers(const std::vector<pid_t>& children)
{
    bool failed = false;
    for (pid_t pid : children) {
        int status = 0;
        pid_t result;
        do {
            result = waitpid(pid, &status, 0);
        } while (result < 0 && errno == EINTR);
        if (result != pid || !WIFEXITED(status) || WEXITSTATUS(status) != 0)
            failed = true;
    }
    return failed;
}

inline std::vector<std::string> WorkerArguments(
    const std::filesystem::path& worker, const std::string& library, CBLAS_TRANSPOSE ta, CBLAS_TRANSPOSE tb, int m,
    int n, int k, int lda, int ldb, float alpha, float beta, int first, int last)
{
    return {
        "python3",
        worker.string(),
        library,
        std::to_string(WORKER_A_FD),
        std::to_string(WORKER_B_FD),
        std::to_string(WORKER_C_FD),
        std::to_string(ta),
        std::to_string(tb),
        std::to_string(m),
        std::to_string(n),
        std::to_string(k),
        std::to_string(lda),
        std::to_string(ldb),
        ExactFloat(alpha),
        ExactFloat(beta),
        std::to_string(first),
        std::to_string(last)};
}

inline void Gemm(
    CBLAS_TRANSPOSE ta, CBLAS_TRANSPOSE tb, int m, int n, int k, float alpha, const std::vector<float>& a, int lda,
    const std::vector<float>& b, int ldb, float beta, std::vector<float>& c, const std::filesystem::path& worker)
{
    Environment environment;
    const uint64_t volume = static_cast<uint64_t>(m) * n * k;
    uint64_t minimum = PARALLEL_MIN_VOLUME;
    if (const char* value = std::getenv("HGEMM_REFERENCE_MIN_VOLUME"))
        minimum = std::stoull(value);
    int workers = volume >= PARALLEL_MIN_VOLUME ? MAX_REFERENCE_WORKERS : SMALL_REFERENCE_WORKERS;
    if (const char* value = std::getenv("HGEMM_REFERENCE_WORKERS"))
        workers = std::stoi(value);
    workers = std::min(n, std::max(1, std::min(MAX_REFERENCE_WORKERS, workers)));
    if (volume < minimum || workers == 1) {
        cblas_sgemm(CblasColMajor, ta, tb, m, n, k, alpha, a.data(), lda, b.data(), ldb, beta, c.data(), m);
        return;
    }
    Dl_info info{};
    // Taking &cblas_sgemm may identify the executable's PLT stub. Resolve the
    // dynamic symbol itself so the child loads the exact linked shared object.
    void* symbol = dlsym(RTLD_DEFAULT, "cblas_sgemm");
    if (!symbol || !dladdr(symbol, &info) || !info.dli_fname)
        throw std::runtime_error("Cannot locate actual linked Netlib library");
    const auto library = std::filesystem::absolute(info.dli_fname).string();
    SharedFloats sa(a), sb(b), sc(c);
    std::vector<pid_t> children;
    bool failed = false;
    for (int i = 0; i < workers; ++i) {
        const int first = static_cast<int64_t>(n) * i / workers, last = static_cast<int64_t>(n) * (i + 1) / workers;
        auto text = WorkerArguments(worker, library, ta, tb, m, n, k, lda, ldb, alpha, beta, first, last);
        const pid_t pid = SpawnWorker(text, sa, sb, sc);
        if (pid < 0) {
            failed = true;
            break;
        }
        children.push_back(pid);
    }
    // Always join every launched child before destroying the shared mappings.
    const bool childFailed = WaitForWorkers(children);
    failed = failed || childFailed;
    if (failed || children.size() != static_cast<size_t>(workers))
        throw std::runtime_error("Netlib reference worker failed");
    std::copy_n(static_cast<const unsigned char*>(sc.data), sc.bytes, reinterpret_cast<unsigned char*>(c.data()));
}
} // namespace hgemm_reference
