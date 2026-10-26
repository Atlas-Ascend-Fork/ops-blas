# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

set(MAKESELF_NAME "makeself")
set(MAKESELF_PATH ${CANN_3RD_LIB_PATH}/makeself)

if (POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

# 默认配置的makeself还是不存在则下载
if (NOT EXISTS "${MAKESELF_PATH}/makeself-header.sh" OR NOT EXISTS "${MAKESELF_PATH}/makeself.sh")
    set(MAKESELF_URL "https://gitcode.com/cann-src-third-party/makeself/releases/download/release-2.5.0-patch1.0/makeself-release-2.5.0-patch1.tar.gz")
    message(STATUS "Downloading ${MAKESELF_NAME} from ${MAKESELF_URL}")

    include(FetchContent)
    FetchContent_Declare(
        ${MAKESELF_NAME}
        URL ${MAKESELF_URL}
        URL_HASH SHA256=bfa730a5763cdb267904a130e02b2e48e464986909c0733ff1c96495f620369a
        SOURCE_DIR "${MAKESELF_PATH}"  # 直接解压到此目录
    )
    FetchContent_MakeAvailable(${MAKESELF_NAME})
    # issue #608: chmod 目标为 FetchContent 实际解压目录 MAKESELF_PATH（原路径少一级
    # 且混入 -E env/CMAKE_TLS_VERIFY 残留参数导致命令必然失败）；结果必须检查，
    # exec 位加固失败提前告警而非后移到打包阶段才暴露
    execute_process(
        COMMAND chmod 700 "${MAKESELF_PATH}/makeself.sh"
        COMMAND chmod 700 "${MAKESELF_PATH}/makeself-header.sh"
        RESULT_VARIABLE CHMOD_RESULT
        ERROR_VARIABLE CHMOD_ERROR
    )
    if(NOT CHMOD_RESULT EQUAL 0)
        message(WARNING "makeself chmod failed: ${CHMOD_ERROR}")
    endif()
endif()