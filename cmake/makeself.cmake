# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
# makeself.cmake - CPack External 打包脚本，使用 makeself 生成 .run 自解压包
# 由 CPack 在 staging 完成后调用，变量由 CPack 传入

set(RUN_FILE "${CPACK_PACKAGE_DIRECTORY}/${CPACK_PACKAGE_FILE_NAME}.run")
set(MAKESELF_SCRIPT "${CPACK_MAKESELF_PATH}/makeself.sh")

if(NOT EXISTS "${MAKESELF_SCRIPT}")
  message(FATAL_ERROR "makeself.sh not found: ${MAKESELF_SCRIPT}")
endif()

if(NOT EXISTS "${CPACK_TEMPORARY_DIRECTORY}")
  message(FATAL_ERROR "Staging directory not found: ${CPACK_TEMPORARY_DIRECTORY}")
endif()

# 确保输出目录存在
file(MAKE_DIRECTORY "${CPACK_PACKAGE_DIRECTORY}")

# 使用 makeself 将 staging 目录打包为 .run 自解压包
# startup_script 为 install.sh，支持 --install、--uninstall、--install-path 等参数
# WORKING_DIRECTORY 确保 --help-header 能正确找到 help.info（路径相对于 staging 目录）
execute_process(
  COMMAND "${MAKESELF_SCRIPT}" --nocomp
          --help-header share/info/ops_blas/help.info
          "${CPACK_TEMPORARY_DIRECTORY}" "${RUN_FILE}"
          "${CPACK_PACKAGE_NAME} ${CPACK_PACKAGE_VERSION}"
          share/info/ops_blas/install.sh
  WORKING_DIRECTORY "${CPACK_TEMPORARY_DIRECTORY}"
  RESULT_VARIABLE MAKESELF_RESULT
  ERROR_VARIABLE MAKESELF_ERROR
)

if(MAKESELF_RESULT)
  message(FATAL_ERROR "makeself failed: ${MAKESELF_ERROR}")
endif()

# 告知 CPack 生成的包文件路径，CPack 会将其复制到构建目录
set(CPACK_EXTERNAL_BUILT_PACKAGES "${RUN_FILE}")