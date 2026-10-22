# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

SKIP_BUILD=0
POSITIONAL_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --skip-build) SKIP_BUILD=1 ;;
        *) POSITIONAL_ARGS+=("$arg") ;;
    esac
done

set -- ${POSITIONAL_ARGS[@]+"${POSITIONAL_ARGS[@]}"}

M=${1:-16}; N=${2:-16}; BATCH=${3:-1}
SIDE=${4:-0}; UPLO=${5:-0}; TRANSA=${6:-0}; DIAG=${7:-0}
ALPHA_RE=${8:-1.0}; ALPHA_IM=${9:-0.0}

die() { echo "[Error] $*" >&2; exit 1; }

echo "=== [1/4] Setting CANN env ==="
[ -n "${ASCEND_HOME_PATH:-}" ] || die "ASCEND_HOME_PATH not set"
source "${ASCEND_HOME_PATH}/set_env.sh" || die "set_env.sh failed"

if [ "${SKIP_BUILD}" -eq 1 ]; then
    [ -f "build/test/ctrsm_batched_test" ] || die "--skip-build but build/test/ctrsm_batched_test not found"
    echo "=== [2/4] Skipping build ==="
else
    echo "=== [2/4] Building ==="
    mkdir -p build && cd build
    cmake .. || die "cmake failed"
    make -j8 || die "make failed"
    cd ..
fi

echo "=== [3/4] Generating test data ==="
python3 test/data/gen_data.py $M $N $BATCH $SIDE $UPLO $TRANSA $DIAG $ALPHA_RE $ALPHA_IM \
    || die "gen_data.py failed"

echo "=== [4/4] Running kernel ==="
./build/test/ctrsm_batched_test 0 $M $N $BATCH $SIDE $UPLO $TRANSA $DIAG $ALPHA_RE $ALPHA_IM \
    || die "Kernel run failed"

echo "=== Verifying ==="
python3 test/data/verify_result.py $M $N $BATCH \
    || die "Verification failed"

echo "=== Done ==="
