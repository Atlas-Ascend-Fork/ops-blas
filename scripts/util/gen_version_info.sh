#!/bin/bash
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
# 生成 version.info 文件，供 run 包使用

set -euo pipefail

# issue #728: 三个位置参数缺一不可，无参/漏参调用时空变量会拼出 /version.info
# 尝试写文件系统根目录；此处显式校验并给出用法后退出
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <ascend_install_dir> <gen_file_dir> <ops_blas_version>" >&2
    exit 1
fi

ascend_install_dir=$1
gen_file_dir=$2
ops_blas_version=$3

# ascend_install_dir 存在性软校验：不存在时跳过 compiler 版本提取并告警，
# 不阻断 version.info 生成
if [ ! -d "${ascend_install_dir}" ]; then
    echo "Warning: ascend_install_dir '${ascend_install_dir}' not found, skip compiler version extraction." >&2
fi

mkdir -p "${gen_file_dir}"

# create version.info
# Version: ops_blas 包版本
echo "Version=${ops_blas_version}" > "${gen_file_dir}/version.info"

# custom_opp_compiler_version: 从 Ascend compiler 读取
if [ -f "${ascend_install_dir}/compiler/version.info" ]; then
  # grep 无匹配时经 pipefail 会使脚本退出，|| true 保持"提取不到则留空"的现状语义
  compiler_version=$(grep "Version" -w "${ascend_install_dir}/compiler/version.info" | awk -F = '{print $2}' || true)
  echo "custom_opp_compiler_version=${compiler_version}" >> "${gen_file_dir}/version.info"
fi
