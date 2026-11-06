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
 * \file aclblas_logger.cpp
 * \brief ops-blas log配置管理模块
 */

#include "aclblas_logger_manager.h"

namespace {
static AclBlas::_aclblas_logger_configure configure;
} // namespace

namespace AclBlas {

aclblasStatus_t aclblasLoggerConfigure(int logIsOn, int logToStdOut, int logToStdErr, const char* logFile)
{
    // issue #817: 配置仅做存储（预留接口），当前日志输出由外部框架 log/log.h 的
    // OP_LOGx 完成，不读取该配置对象；对外口径见 include/cann_ops_blas.h 的接口注释。
    // issue #601: logFile 以值拷贝保存（C ABI 边界不长期持有调用方裸指针），
    // 调用方在配置返回后释放/复用原缓冲不影响后续读取。
    configure.logIsOn = logIsOn;
    configure.logToStdOut = logToStdOut;
    configure.logToStdErr = logToStdErr;
    configure.logFile = (logFile != nullptr) ? std::string(logFile) : std::string();
    return aclblasStatus_t::ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasSetLoggerCallback(aclblasLogCallback userCallback)
{
    configure.userCallback = userCallback;
    return aclblasStatus_t::ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasGetLoggerCallback(aclblasLogCallback* userCallback)
{
    if (userCallback == nullptr) {
        return aclblasStatus_t::ACLBLAS_STATUS_INVALID_VALUE;
    }
    *userCallback = configure.userCallback;
    return aclblasStatus_t::ACLBLAS_STATUS_SUCCESS;
}

} // namespace AclBlas

extern "C" {

aclblasStatus_t aclblasLoggerConfigure(int logIsOn, int logToStdOut, int logToStdErr, const char* logFile)
{
    return AclBlas::aclblasLoggerConfigure(logIsOn, logToStdOut, logToStdErr, logFile);
}

aclblasStatus_t aclblasSetLoggerCallback(aclblasLogCallback userCallback)
{
    return AclBlas::aclblasSetLoggerCallback(userCallback);
}

aclblasStatus_t aclblasGetLoggerCallback(aclblasLogCallback* userCallback)
{
    return AclBlas::aclblasGetLoggerCallback(userCallback);
}

} // extern "C"
