#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

set -e

BUILD_OPS=""
RUN_TEST=OFF
ENABLE_PACKAGE=FALSE
TEST_DEVICE_ID=0
TEST_DEVICE_COUNT=1
RUN_BLAS_ALL=OFF
BLAS_SMOKE=OFF

export BASE_PATH=$(
  cd "$(dirname $0)"
  pwd
)
BUILD_DIR="${BASE_PATH}/build"
export BUILD_PATH="${BASE_PATH}/build"
export BUILD_OUT_PATH="${BASE_PATH}/build_out"
ARCH_INFO=$(uname -m)

# issue #609: ASCEND_HOME_PATH 回退解析（支持仅设 ASCEND_TOOLKIT_HOME 的环境）
# 必须先于 include/lib 路径导出，否则导出 /include、/lib64 等非法路径
ASCEND_HOME="${ASCEND_HOME_PATH:-${ASCEND_TOOLKIT_HOME}}"
export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-${ASCEND_HOME}}"

export INCLUDE_PATH="${ASCEND_HOME_PATH}/include"
export ACLNN_INCLUDE_PATH="${INCLUDE_PATH}/aclnn"
export COMPILER_INCLUDE_PATH="${ASCEND_HOME_PATH}/include"
export GRAPH_INCLUDE_PATH="${COMPILER_INCLUDE_PATH}/graph"
export EXTERNAL_INCLUDE_PATH="${COMPILER_INCLUDE_PATH}/external"
export GE_INCLUDE_PATH="${COMPILER_INCLUDE_PATH}/ge"
# issue #658: ASCEND_OPP_PATH fallback (derive from ASCEND_HOME, align with #609)
ASCEND_OPP_PATH=${ASCEND_OPP_PATH:-${ASCEND_HOME}/opp}
export ASCEND_OPP_PATH
export INC_INCLUDE_PATH="${ASCEND_OPP_PATH}/built-in/op_proto/inc"
export LINUX_INCLUDE_PATH="${ASCEND_HOME_PATH}/${ARCH_INFO}-linux/include"
export EAGER_LIBRARY_OPP_PATH="${ASCEND_OPP_PATH}/lib64"
export EAGER_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64"
export GRAPH_LIBRARY_STUB_PATH="${ASCEND_HOME_PATH}/lib64/stub"
export GRAPH_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64"
CANN_3RD_LIB_PATH="${BUILD_PATH}/third_party"

# 分隔线和错误打印函数（参考ops-nn/build.sh格式）
dotted_line="----------------------------------------------------------------"
print_error() {
  echo
  echo $dotted_line
  local msg="$1"
  echo -e "\033[31m[ERROR] ${msg}\033[0m"
  echo $dotted_line
  echo
}

print_skip() {
  echo -e "\033[33m[SKIP] ${1}\033[0m"
}

# 支持的 SOC 版本
# 按字符串长度从长到短排序，避免前缀匹配时出错
SUPPORT_COMPUTE_UNIT_SHORT=("ascend910_93" "ascend910b" "ascend950" "ascend310p")
SUPPORT_COMPUTE_UNIT_SHORT=($(printf '%s\n' "${SUPPORT_COMPUTE_UNIT_SHORT[@]}" | awk '{print length($0) " " $0}' | sort -rn | cut -d ' ' -f2-))

# ==========================
# 解析参数
# ==========================
for arg in "$@"; do
    case $arg in
        --ops=*)
            BUILD_OPS="${arg#*=}"
            ;;
        --run)
            RUN_TEST=ON
            ;;
        --soc=*)
            SOC_VERSION="${arg#*=}"
            ;;
        --pkg)
            ENABLE_PACKAGE=TRUE
            ;;
        --device=*)
            TEST_DEVICE_ID="${arg#*=}"
            ;;
        --devices=*)
            # 多 device 并行分片（issue #457）：如 --devices=2 表示用设备 0..N-1
            # 按算子均分并行执行；与 --device 互斥，--devices 优先
            TEST_DEVICE_COUNT="${arg#*=}"
            ;;
        --blas-all)
            # 合并测试 Runner（issue #457）：运行单一 blas_all_test 可执行
            # （全部算子测试源聚合编译，suite 初始化仅 1 次），替代逐算子循环
            RUN_BLAS_ALL=ON
            ;;
        --smoke)
            # 冒烟抽样（issue #457）：按 CSV 分级抽样执行，负例与边界全保
            BLAS_SMOKE=ON
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Usage:"
            echo "  bash build.sh                                 # 只编译库"
            echo "  bash build.sh --ops=scopy,blasLtMatmul        # 编译指定算子(多算子，逗号分隔)"
            echo "  bash build.sh --run                           # 编译并运行所有算子测试"
            echo "  bash build.sh --ops=scopy,blasLtMatmul --run  # 编译并运行多个算子"
            echo "  bash build.sh --pkg                           # 编译并打包run包"
            echo "  bash build.sh --pkg --soc=ascend950           # 打包指定SOC的run包"
            echo "  bash build.sh --pkg --soc=ascend950 --ops=scopy --run  # 编译指定算子打包并运行测试"
            echo "  bash build.sh --ops=scopy --run --device=1     # 指定偏好测试设备(默认0，被占用时自动换空闲卡)"
            echo "  bash build.sh --ops=scopy,cherk --run --devices=2   # 按 2 个设备分片并行"
            echo "  bash build.sh --run --blas-all                 # 用合并 Runner 运行全部(单进程)"
            echo "  bash build.sh --run --smoke                    # 冒烟抽样模式运行全部"
            exit 1
            ;;
    esac
done

# 默认 SOC_VERSION
if [ -z "${SOC_VERSION}" ]; then
    SOC_VERSION="ascend910b3"
fi

# SOC_VERSION -> SOC_ARCH_DIRS 映射（与 CMakeLists.txt 保持一致）
soc_lower=$(echo "${SOC_VERSION}" | tr '[:upper:]' '[:lower:]')
if [[ "${soc_lower}" == ascend910b* ]] || [[ "${soc_lower}" == ascend910_93* ]]; then
    SOC_ARCH_DIRS=("arch22")
elif [[ "${soc_lower}" == ascend950* ]]; then
    SOC_ARCH_DIRS=("arch35")
elif [[ "${soc_lower}" == ascend310p* ]]; then
    SOC_ARCH_DIRS=("arch20")
else
    SOC_ARCH_DIRS=()
fi

# 展开家族名为具体算子列表
# 例如: --ops=gbmv → --ops=sgbmv (如果 test/gbmv/sgbmv/CMakeLists.txt 存在)
# 搜索逻辑: 在家族目录下遍历各算子子目录,检查是否有目标 archXX 目录
expand_family_ops() {
  local ops_str="$1"
  local expanded=""
  IFS=',' read -ra OPS_ARRAY <<< "${ops_str}"
  for op in "${OPS_ARRAY[@]}"; do
    if [ -f "${BASE_PATH}/test/${op}/CMakeLists.txt" ]; then
      # 直接算子名 (如 scopy, sgbmv)
      if [ -z "${expanded}" ]; then
        expanded="${op}"
      else
        expanded="${expanded},${op}"
      fi
    elif [ -d "${BASE_PATH}/test/${op}" ]; then
      # 可能是家族名,遍历子目录查找具体算子
      local found_any=FALSE
      for sub_dir in "${BASE_PATH}/test/${op}"/*/; do
        [ -d "${sub_dir}" ] || continue
        [ -f "${sub_dir}/CMakeLists.txt" ] || continue
        local sub_name=$(basename "${sub_dir}")
        # 检查该算子是否有当前 SOC 对应的 archXX 目录
        local has_arch=FALSE
        for arch_dir in "${SOC_ARCH_DIRS[@]}"; do
          if [ -d "${sub_dir}/${arch_dir}" ]; then
            has_arch=TRUE
            break
          fi
        done
        if [ "${has_arch}" = "TRUE" ]; then
          if [ -z "${expanded}" ]; then
            expanded="${sub_name}"
          else
            expanded="${expanded},${sub_name}"
          fi
          found_any=TRUE
        else
          echo "[INFO] ${sub_name}: skipped (no ${SOC_ARCH_DIRS[*]} implementation)" >&2
        fi
      done
      if [ "${found_any}" = "FALSE" ]; then
        echo "[WARN] Family '${op}' has no operators with ${SOC_ARCH_DIRS[*]} implementation" >&2
      fi
    else
      # 未知算子名,保留原样让 CMake 报错
      if [ -z "${expanded}" ]; then
        expanded="${op}"
      else
        expanded="${expanded},${op}"
      fi
    fi
  done
  echo "${expanded}"
}

# 展开家族名
if [ -n "${BUILD_OPS}" ]; then
  BUILD_OPS=$(expand_family_ops "${BUILD_OPS}")
fi

# 如果 --run 单独使用（没有指定算子），自动发现所有测试目录
if [ "${RUN_TEST}" == "ON" ] && [ -z "${BUILD_OPS}" ]; then
  # 从 test 目录下查找所有包含 CMakeLists.txt 的子目录（支持家族嵌套）
  AUTO_DISCOVER_OPS=""
  for dir in "${BASE_PATH}"/test/*/; do
    [ -d "${dir}" ] || continue
    dir_name=$(basename "${dir}")
    # 排除非算子目录
    if [ "${dir_name}" = "frame" ] || [ "${dir_name}" = "utils" ] || [ "${dir_name}" = "common" ]; then
      continue
    fi
    if [ -f "${dir}/CMakeLists.txt" ]; then
      # 直接算子目录 (如 test/scopy/)
      if [ -z "${AUTO_DISCOVER_OPS}" ]; then
        AUTO_DISCOVER_OPS="${dir_name}"
      else
        AUTO_DISCOVER_OPS="${AUTO_DISCOVER_OPS},${dir_name}"
      fi
    else
      # 可能是家族目录 (如 test/gbmv/sgbmv/)
      for sub_dir in "${dir}"/*/; do
        [ -d "${sub_dir}" ] || continue
        [ -f "${sub_dir}/CMakeLists.txt" ] || continue
        sub_name=$(basename "${sub_dir}")
        if [ -z "${AUTO_DISCOVER_OPS}" ]; then
          AUTO_DISCOVER_OPS="${sub_name}"
        else
          AUTO_DISCOVER_OPS="${AUTO_DISCOVER_OPS},${sub_name}"
        fi
      done
    fi
  done
  BUILD_OPS="${AUTO_DISCOVER_OPS}"
  echo "Auto-discovered tests: ${BUILD_OPS}"
fi

echo "BUILD_OPS=${BUILD_OPS}, RUN_TEST=${RUN_TEST}, ENABLE_PACKAGE=${ENABLE_PACKAGE}"

# ==========================
# 环境检查（Ascend/CANN）
# issue #609: ASCEND_HOME_PATH 回退解析已前移至脚本头部导出之前，此处仅保留缺省报错
# ==========================
if [ -z "${ASCEND_HOME_PATH}" ]; then
    echo "Error: Ascend/CANN environment is not configured."
    echo "Please source the Ascend environment script first, e.g.:"
    echo "  source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh"
    echo "  or source \${HOME}/Ascend/ascend-toolkit/latest/set_env.sh"
    exit 1
fi

# ==========================
# 运行测试前智能选择空闲设备
# TEST_DEVICE_ID 是编译期宏（-DTEST_DEVICE_ID），必须在 cmake 配置之前完成选卡。
# --devices=N 多卡分片模式跳过此逻辑：各分片进程通过 BLAS_DEVICE_ID=0..N-1
# 自行绑定设备（blas_test.h 运行时读取），无单卡偏好可言。
# ==========================
if [ "${RUN_TEST}" == "ON" ] && [ -n "${BUILD_OPS}" ] && [ "${TEST_DEVICE_COUNT}" -le 1 ]; then
    SELECT_DEVICE_SCRIPT="${BASE_PATH}/scripts/select_device.py"
    if [ ! -f "${SELECT_DEVICE_SCRIPT}" ]; then
        print_error "设备选择脚本不存在: ${SELECT_DEVICE_SCRIPT}"
        exit 1
    fi
    set +e
    SELECTED_DEVICE=$(python3 "${SELECT_DEVICE_SCRIPT}" --prefer "${TEST_DEVICE_ID}")
    select_ret=$?
    set -e
    case ${select_ret} in
        0)
            if [[ "${SELECTED_DEVICE}" =~ ^[0-9]+$ ]]; then
                if [ "${SELECTED_DEVICE}" != "${TEST_DEVICE_ID}" ]; then
                    echo "[INFO] 使用设备 ${SELECTED_DEVICE}（原偏好设备 ${TEST_DEVICE_ID} 被占用，已自动切换）"
                else
                    echo "[INFO] 使用设备 ${SELECTED_DEVICE}"
                fi
                TEST_DEVICE_ID="${SELECTED_DEVICE}"
            else
                print_error "设备选择结果异常: ${SELECTED_DEVICE}"
                exit 1
            fi
            ;;
        10)
            print_error "没有可用设备"
            exit 1
            ;;
        20)
            print_error "所有设备均被占用"
            exit 1
            ;;
        *)
            print_error "设备选择失败，select_device.py 退出码 ${select_ret}，详见上方日志"
            exit 1
            ;;
    esac
fi

# ==========================
# 构建
# ==========================
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"

# 默认 SOC_VERSION，ASCEND_CANN_PACKAGE_PATH 使用环境变量
if [ -z "${SOC_VERSION}" ]; then
    SOC_VERSION="ascend910b3"
fi

# 校验 SOC 是否在支持列表中（前缀匹配）
soc_lower=$(echo "${SOC_VERSION}" | tr '[:upper:]' '[:lower:]')
matched=""
for support_unit in "${SUPPORT_COMPUTE_UNIT_SHORT[@]}"; do
    if [[ "${soc_lower}" == "${support_unit}"* ]]; then
        matched="${support_unit}"
        break
    fi
done
if [ -z "${matched}" ]; then
    echo "Error: The soc [${SOC_VERSION}] is not supported."
    echo "Supported SOC: ${SUPPORT_COMPUTE_UNIT_SHORT[*]}"
    exit 1
fi

CMAKE_OPTIONS="-DSOC_VERSION=${SOC_VERSION} -DASCEND_CANN_PACKAGE_PATH=${ASCEND_HOME}"

# 帮助 find_package(ASC) 定位 ASCConfig.cmake（CMake 查找 ASC/ASCConfig.cmake 或 ASC/asc-config.cmake）
if [ -z "${ASC_DIR}" ]; then
    for p in "${ASCEND_HOME}/lib64/cmake" "${ASCEND_HOME}/compiler/latest/lib64/cmake" \
             "${ASCEND_HOME}/ascendc/lib64/cmake"; do
        if [ -f "${p}/ASC/ASCConfig.cmake" ] || [ -f "${p}/ASC/asc-config.cmake" ]; then
            CMAKE_OPTIONS="${CMAKE_OPTIONS} -DCMAKE_PREFIX_PATH=${p}"
            break
        fi
    done
fi
[ -n "${ASC_DIR}" ] && CMAKE_OPTIONS="${CMAKE_OPTIONS} -DASC_DIR=${ASC_DIR}"

if [ -n "${BUILD_OPS}" ]; then
    # 将逗号分隔转换为 CMake 列表格式（分号分隔）
    TEST_NAMES_CMAKE="${BUILD_OPS//,/;}"
    CMAKE_OPTIONS="${CMAKE_OPTIONS} -DBUILD_TEST=ON -DTEST_NAMES=${TEST_NAMES_CMAKE}"
fi

CMAKE_OPTIONS="${CMAKE_OPTIONS} -DTEST_DEVICE_ID=${TEST_DEVICE_ID}"

if [ "${ENABLE_PACKAGE}" == "TRUE" ]; then
    CMAKE_OPTIONS="${CMAKE_OPTIONS} -DENABLE_PACKAGE=ON"
fi

# issue #610: 路径展开加引号（rm -rf/mkdir/cmake），路径含空格时不再被拆分；
# CMAKE_OPTIONS 保持分词展开（多选项语义），如未来需支持带空格选项应改为 bash 数组
cmake -S "${BASE_PATH}" -B "${BUILD_DIR}" ${CMAKE_OPTIONS}
cmake --build "${BUILD_DIR}" -j
if [ "${ENABLE_PACKAGE}" == "TRUE" ]; then
    cmake --build "${BUILD_DIR}" --target package
else
    cmake --install "${BUILD_DIR}"
fi

# ==========================
# 运行算子测试
# ==========================
if [ "${RUN_TEST}" == "ON" ]; then
    if [ -z "${BUILD_OPS}" ]; then
        echo "Error: No tests found. Please check test directories or specify --ops=<算子名1,算子名2,...>"
        exit 1
    fi

    # 将逗号分隔的算子名转换为数组
    IFS=',' read -ra OP_ARRAY <<< "${BUILD_OPS}"

    # 读取 CMake 配置阶段生成的 skip 清单
    declare -A SKIP_REASON_MAP=()
    SKIPPED_FILE="${BUILD_DIR}/test/skipped_tests.list"
    if [ -f "${SKIPPED_FILE}" ]; then
        while IFS='|' read -r skip_op skip_reason || [ -n "${skip_op}" ]; do
            [ -z "${skip_op}" ] && continue
            SKIP_REASON_MAP["${skip_op}"]="${skip_reason}"
        done < "${SKIPPED_FILE}"
    fi

    # 可运行算子 = 请求算子去掉 skip 名单
    RUNNABLE_OPS=()
    SKIPPED_OPS=()
    for op in "${OP_ARRAY[@]}"; do
        if [ -n "${SKIP_REASON_MAP[${op}]+x}" ]; then
            echo ""
            echo "========== Skipping ${op}_test =========="
            print_skip "${op}_test: not supported on SOC '${SOC_VERSION}' (${SKIP_REASON_MAP[${op}]})"
            SKIPPED_OPS+=("${op}")
        else
            RUNNABLE_OPS+=("${op}")
        fi
    done

    # ------------------------------------------------------------
    # 冒烟抽样（issue #457·方案2）：BLAS_SMOKE=ON 时按 CSV 分级抽样生成
    # BLAS_TEST_CASES，负例与边界全保，groups 内等距取样。未生成条目的算子
    # （无 CSV/解析失败）保持该算子全量。
    # ------------------------------------------------------------
    if [ "${BLAS_SMOKE}" == "ON" ] && [ "${#RUNNABLE_OPS[@]}" -gt 0 ]; then
        sampled=$(python3 "${BASE_PATH}/scripts/ci/smoke_sample.py" "${SOC_ARCH_DIRS[0]}" "${RUNNABLE_OPS[@]}" 2>/dev/null || true)
        if [ -n "${sampled}" ]; then
            export BLAS_TEST_CASES="${sampled}"
            echo "Smoke sampling enabled: BLAS_TEST_CASES set (see scripts/ci/smoke_sample.py)."
        else
            echo "Smoke sampling unavailable (no CSV found), falling back to full cases."
        fi
    fi

    # ------------------------------------------------------------
    # 多 device 并行分片（issue #457·方案4）：TEST_DEVICE_COUNT>1 时把可运行
    # 算子均分为 N 片，第 i 片进程设 BLAS_DEVICE_ID=i（blas_test.h 运行时读取，
    # 优先于编译期 TEST_DEVICE_ID）后台并行执行，完成后汇总各片 PASS/FAIL。
    # ------------------------------------------------------------
    shard_run_one() {
        # $1=op $2=device_id
        local op="$1" dev="$2"
        local TEST_BIN="${BUILD_DIR}/test/${op}/${op}_test"
        if [ ! -f "${TEST_BIN}" ]; then
            for family_dir in "${BUILD_DIR}/test"/*/; do
                if [ -f "${family_dir}${op}/${op}_test" ]; then
                    TEST_BIN="${family_dir}${op}/${op}_test"
                    break
                fi
            done
        fi
        if [ ! -f "${TEST_BIN}" ]; then
            local family="${op:1}"
            [ -f "${BUILD_DIR}/test/${family}/${op}_test" ] && TEST_BIN="${BUILD_DIR}/test/${family}/${op}_test"
        fi
        if [ ! -f "${TEST_BIN}" ]; then
            echo "[FAIL] ${op}_test: binary not found (build may have failed)"
            return 1
        fi
        BLAS_DEVICE_ID="${dev}" "${TEST_BIN}" "$(dirname "${TEST_BIN}")"
    }

    run_sharded() {
        local n_shards="$1"
        shift
        local -a shard_ops=("$@")
        local -a pids=()
        local -a shard_ops_list=()
        local base=${#shard_ops[@]}
        local per=$(( (base + n_shards - 1) / n_shards ))
        rm -rf "${BUILD_DIR}/shard_status" && mkdir -p "${BUILD_DIR}/shard_status"

        for ((i=0; i<n_shards; i++)); do
            local slice=("${shard_ops[@]:i*per:per}")
            [ ${#slice[@]} -eq 0 ] && continue
            (
                for op in "${slice[@]}"; do
                    echo ""
                    echo "========== [device ${i}] Running ${op}_test =========="
                    set +e
                    shard_run_one "${op}" "${i}"
                    local rc=$?
                    set -e
                    echo "${op} ${rc}" >> "${BUILD_DIR}/shard_status/device_${i}.txt"
                done
            ) > "${BUILD_DIR}/shard_log_${i}.txt" 2>&1 &
            pids+=($!)
            shard_ops_list+=("$(IFS=,; echo "${slice[*]}")")
        done

        echo "Sharded run: ${n_shards} devices, ops per device: ${shard_ops_list[*]}"
        for pid in "${pids[@]}"; do
            wait "${pid}"
        done

        # 汇总分片状态
        for f in "${BUILD_DIR}"/shard_status/*.txt; do
            [ -f "${f}" ] || continue
            while read -r op rc; do
                if [ "${rc}" -eq 0 ]; then
                    echo "[PASS] ${op}_test (shard $(basename "${f}" .txt))"
                    PASSED_OPS+=("${op}")
                else
                    echo "[FAIL] ${op}_test (shard $(basename "${f}" .txt), exit code: ${rc})"
                    FAILED_OPS+=("${op}")
                fi
            done < "${f}"
        done
    }

    # 注意：SKIPPED_OPS 在上方 skip 判定循环里已记录跳过算子，
    # 此处只初始化 PASSED/FAILED，不能重置 SKIPPED_OPS（否则跳过信息丢失）
    FAILED_OPS=()
    PASSED_OPS=()

    # ------------------------------------------------------------
    # 合并测试 Runner（issue #457·方案1）：单一 blas_all_test 进程跑全部算子，
    # suite 级 aclInit 复用（80 次 → 1 次）。构建/执行失败时回退 per-op 循环。
    # ------------------------------------------------------------
    ALL_RUNNER_BIN="${BUILD_DIR}/test/blas_all_test"
    if [ "${RUN_BLAS_ALL}" == "ON" ] && [ -f "${ALL_RUNNER_BIN}" ]; then
        echo "========== Running blas_all_test (amalgamated runner) =========="
        if [ "${TEST_DEVICE_COUNT}" -gt 1 ]; then
            # 合并 Runner 单进程内多 suite 共享设备，分片仍按 suite（gtest_filter）
            # 由 test_main.cpp 处理，这里退化为单 device 执行
            echo "Note: --blas-all runs in one process; --devices is ignored."
        fi
        set +e
        "${ALL_RUNNER_BIN}" "$(dirname "${ALL_RUNNER_BIN}")"
        all_rc=$?
        set -e
        if [ ${all_rc} -eq 0 ]; then
            echo "[PASS] blas_all_test"
            echo ""
            echo "========================================"
            echo "Test Summary:"
            echo "  Amalgamated runner PASSED (all gtest suites)."
            echo "========================================"
            exit 0
        else
            echo "[WARN] blas_all_test failed (exit code: ${all_rc}), falling back to per-op run for precise attribution."
        fi
    elif [ "${RUN_BLAS_ALL}" == "ON" ]; then
        echo "[WARN] blas_all_test binary not found, falling back to per-op run."
    fi

    if [ "${TEST_DEVICE_COUNT}" -gt 1 ] && [ "${#RUNNABLE_OPS[@]}" -gt 0 ]; then
        run_sharded "${TEST_DEVICE_COUNT}" "${RUNNABLE_OPS[@]}"
    else
        for op in "${RUNNABLE_OPS[@]}"; do
            echo ""
            echo "========== Running ${op}_test =========="
            set +e
            shard_run_one "${op}" "${TEST_DEVICE_ID}"
            rc=$?
            set -e
            if [ ${rc} -eq 0 ]; then
                echo "[PASS] ${op}_test"
                PASSED_OPS+=("${op}")
            else
                echo "[FAIL] ${op}_test (exit code: ${rc})"
                FAILED_OPS+=("${op}")
            fi
        done
    fi

    # 汇总测试结果
    echo ""
    echo "========================================"
    echo "Test Summary:"
    echo "  Passed:  ${#PASSED_OPS[@]} - ${PASSED_OPS[*]}"
    echo "  Skipped: ${#SKIPPED_OPS[@]} - ${SKIPPED_OPS[*]} (not supported on ${SOC_VERSION})"
    echo "  Failed:  ${#FAILED_OPS[@]} - ${FAILED_OPS[*]}"
    echo "========================================"

    if [ ${#FAILED_OPS[@]} -gt 0 ]; then
        exit 1
    fi
fi
