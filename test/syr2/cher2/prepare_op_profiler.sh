#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
set -euo pipefail

# Build the upstream injection component in an isolated copy of the installed
# profiler. This fixes logical device selection on the validation container.
: "${ASCEND_HOME_PATH:?Source CANN set_env.sh first}"
profiling_root="${1:?Usage: prepare_op_profiler.sh NEW_OUTPUT_DIRECTORY}"
revision=d1564487d9c0b27033f6bb51b19b21bec7be6aa5
if [[ -e "${profiling_root}" ]]; then
    echo "Use a new output directory: ${profiling_root}" >&2
    exit 1
fi
if [[ ! -f /usr/include/nlohmann/json.hpp ]]; then
    echo "Install nlohmann-json3-dev, CMake and a C++ compiler first." >&2
    exit 1
fi
mkdir -p "${profiling_root}"
profiling_root="$(realpath "${profiling_root}")"
git clone --no-checkout https://gitcode.com/Ascend/msopcom.git "${profiling_root}/source"
git -C "${profiling_root}/source" checkout "${revision}"
mkdir -p "${profiling_root}/source/thirdparty/json"
ln -s /usr/include "${profiling_root}/source/thirdparty/json/single_include"
cmake -S "${profiling_root}/source" -B "${profiling_root}/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "${profiling_root}/build" --target msopprof_injection -j8
cp -a "${ASCEND_HOME_PATH}/tools/msopprof" "${profiling_root}/msopprof"
cp "${profiling_root}/build/csrc/libmsopprof_injection.so" "${profiling_root}/msopprof/lib64/libmsopprof_injection.so"
"${profiling_root}/msopprof/bin/msopprof" --version | tee "${profiling_root}/version.txt"
