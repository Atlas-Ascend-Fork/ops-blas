# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)

# 收集当前目录下的基础源文件
file(GLOB BASE_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# 自动收集所有算子目录的源文件（支持子目录中的架构特定目录）
set(OP_SRC_FILES "")
file(GLOB children RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
foreach(child ${children})
    if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${child})
        
        # 如果算子未被排除，则收集其源文件
        if(NOT is_excluded)
            # 递归收集非架构特定源码，避免与 archXX 目录重复收集
            file(GLOB_RECURSE dir_srcs ${CMAKE_CURRENT_SOURCE_DIR}/${child}/*.cpp)
            foreach(src_file ${dir_srcs})
                set(is_arch_specific FALSE)
                foreach(arch_dir ${ARCH_SPECIFIC_DIRS})
                    if(src_file MATCHES "/${arch_dir}/")
                        set(is_arch_specific TRUE)
                        break()
                    endif()
                endforeach()
                if(NOT is_arch_specific)
                    # 如果目标 SOC 的 arch 目录中存在同名文件，排除根目录版本避免重复符号
                    get_filename_component(_src_fname ${src_file} NAME)
                    get_filename_component(_src_dir ${src_file} DIRECTORY)
                    set(_has_arch_counterpart FALSE)
                    foreach(_arch_dir ${SOC_ARCH_DIRS})
                        if(EXISTS ${_src_dir}/${_arch_dir}/${_src_fname})
                            set(_has_arch_counterpart TRUE)
                            break()
                        endif()
                    endforeach()
                    if(NOT _has_arch_counterpart)
                        list(APPEND OP_SRC_FILES ${src_file})
                    endif()
                endif()
            endforeach()
        endif()
    endif()
endforeach()

# 收集子目录中的架构特定源文件
set(ARCH_SRC_FILES "")
foreach(arch_dir ${SOC_ARCH_DIRS})
    file(GLOB_RECURSE arch_dir_srcs
        ${CMAKE_CURRENT_SOURCE_DIR}/*/${arch_dir}/*.cpp
    )
    list(APPEND ARCH_SRC_FILES ${arch_dir_srcs})
endforeach()

# sgbmv_kernel_vec/l1.cpp 经 sgbmv_kernel.cpp 以单 TU 方式编译（arch22 设备代码不支持跨 TU 链接），
# 不作为独立编译单元收集。
list(FILTER ARCH_SRC_FILES EXCLUDE REGEX "sgbmv_kernel_(l1|vec)\\.cpp$")

if(DEFINED TEST_NAMES AND NOT "${TEST_NAMES}" STREQUAL "")
    set(_FILTERED_OP_SRC_FILES "")
    set(_FILTERED_ARCH_SRC_FILES "")
    # host 代码直接链接 gemm_kernel_do（定义于 blas/gemm/arch35/gemm_kernel.cpp）的算子：
    # 单算子构建（--ops）时保留该源文件，保证测试可执行文件链接通过。
    # 新增算子复用 gemm_kernel_do 时须同步登记到本表。
    set(_GEMM_KERNEL_DEP_OPS "cherk;cher2k;cherkx")
    set(_KEEP_GEMM_KERNEL FALSE)
    foreach(_t ${TEST_NAMES})
        if(_t IN_LIST _GEMM_KERNEL_DEP_OPS)
            set(_KEEP_GEMM_KERNEL TRUE)
        endif()
    endforeach()
    foreach(_src_file ${OP_SRC_FILES} ${ARCH_SRC_FILES})
        set(_keep_src FALSE)
        if(_src_file MATCHES "/common/")
            set(_keep_src TRUE)
        endif()
        if(_KEEP_GEMM_KERNEL AND _src_file MATCHES "/gemm/arch35/gemm_kernel[.]cpp$")
            set(_keep_src TRUE)
        endif()
        get_filename_component(_src_name "${_src_file}" NAME)
        foreach(_test_name ${TEST_NAMES})
            if(_src_file MATCHES "/${_test_name}/")
                set(_keep_src TRUE)
            endif()
            if(_src_name MATCHES "^${_test_name}_")
                set(_keep_src TRUE)
            endif()
            if(_test_name MATCHES "^[a-zA-Z]")
                string(SUBSTRING "${_test_name}" 1 -1 _stripped_name)
                # fallback: TEST_NAMES like ssyr must also keep family sources under syr/ (syr_host.cpp / syr_kernel.cpp)
                if(_src_file MATCHES "/${_stripped_name}/" AND (_src_name MATCHES "^${_test_name}_" OR _src_name MATCHES "^${_stripped_name}_"))
                    set(_keep_src TRUE)
                endif()
            endif()
        endforeach()
        if(_keep_src)
            if(_src_file IN_LIST OP_SRC_FILES)
                list(APPEND _FILTERED_OP_SRC_FILES ${_src_file})
            else()
                list(APPEND _FILTERED_ARCH_SRC_FILES ${_src_file})
            endif()
        endif()
    endforeach()
    set(OP_SRC_FILES ${_FILTERED_OP_SRC_FILES})
    set(ARCH_SRC_FILES ${_FILTERED_ARCH_SRC_FILES})
endif()

# 列表驱动：遍历 TENSOR_API_OPS，低版本时过滤对应算子源文件
# TRSM_BLOCKED 为特例（仅按文件名过滤 ARCH_SRC_FILES），保持单独处理
foreach(entry ${TENSOR_API_OPS})
    string(REPLACE "|" ";" _fields "${entry}")
    list(GET _fields 0 _var_suffix)
    list(GET _fields 1 _regex)
    if(NOT ENABLE_BLAS_${_var_suffix})
        list(FILTER OP_SRC_FILES EXCLUDE REGEX "${_regex}")
        list(FILTER ARCH_SRC_FILES EXCLUDE REGEX "${_regex}")
        message(STATUS "Skipping blas ${_regex} (requires asc-devkit >= 9.1, current ${ASC_DEVKIT_MAJOR}.${ASC_DEVKIT_MINOR})")
    endif()
endforeach()
if(NOT ENABLE_BLAS_TRSM_BLOCKED)
    list(FILTER ARCH_SRC_FILES EXCLUDE REGEX "strsm_blocked_kernel")
    message(STATUS "Skipping blas trsm blocked kernel (requires asc-devkit >= 9.1, current ${ASC_DEVKIT_MAJOR}.${ASC_DEVKIT_MINOR})")
endif()
set(ALL_BLAS_SRC_FILES
    ${BASE_SRC_FILES}
    ${OP_SRC_FILES}
    ${ARCH_SRC_FILES}
    PARENT_SCOPE
)
