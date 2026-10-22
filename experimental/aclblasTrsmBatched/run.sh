#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

OP_NAME="test/trsm_batched_test"

SKIP_BUILD=0
POSITIONAL_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --skip-build) SKIP_BUILD=1 ;;
        *) POSITIONAL_ARGS+=("$arg") ;;
    esac
done

set -- ${POSITIONAL_ARGS[@]+"${POSITIONAL_ARGS[@]}"}

# Default params
M=${1:-16}; N=${2:-16}; BATCH=${3:-1}
SIDE=${4:-0}; UPLO=${5:-0}; TRANSA=${6:-0}; DIAG=${7:-0}; ALPHA=${8:-1.0}

die() { echo "ERROR: $*" >&2; exit 1; }

echo "=== [1/4] Setting CANN env ==="
[ -n "${ASCEND_HOME_PATH:-}" ] || die "ASCEND_HOME_PATH not set"
source "${ASCEND_HOME_PATH}/set_env.sh" || die "set_env.sh failed"

if [ "${SKIP_BUILD}" -eq 1 ]; then
    [ -f "build/${OP_NAME}" ] || die "--skip-build but build/${OP_NAME} not found"
    echo "=== [2/4] Skipping build ==="
else
    echo "=== [2/4] Building ==="
    mkdir -p build; cd build
    cmake .. || die "cmake failed"
    make -j4 || die "make failed"
    cd ..
fi

echo "=== [3/4] Generating test data ==="
cd build
python3 ../test/data/gen_data.py $M $N $BATCH $SIDE $UPLO $TRANSA $DIAG $ALPHA \
    || die "gen_data.py failed"

echo "=== [4/4] Running kernel ==="
rm -f output/output_b_*.bin
"./${OP_NAME}" $M $N $BATCH $SIDE $UPLO $TRANSA $DIAG $ALPHA \
    || die "Kernel run failed"

echo "=== Verifying ==="
python3 ../test/data/verify_result.py $M $N $BATCH $SIDE $UPLO $TRANSA $DIAG \
    || die "Verification failed"

echo "=== Done ==="
