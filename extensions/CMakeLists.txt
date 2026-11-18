# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use the License for the License.
# THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.20.0)

# Collect base source files (non-arch-specific)
file(GLOB BASE_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# Auto-collect operator source files from subdirectories
set(OP_SRC_FILES "")
file(GLOB children RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
foreach(child ${children})
    if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${child})
        # Recursively collect non-arch-specific source files
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
endforeach()

# Collect arch-specific source files for the target SOC
set(ARCH_SRC_FILES "")
foreach(arch_dir ${SOC_ARCH_DIRS})
    file(GLOB_RECURSE arch_dir_srcs
        ${CMAKE_CURRENT_SOURCE_DIR}/*/${arch_dir}/*.cpp
    )
    list(APPEND ARCH_SRC_FILES ${arch_dir_srcs})
endforeach()

set(ALL_BLAS_SRC_FILES
    ${ALL_BLAS_SRC_FILES}
    ${BASE_SRC_FILES}
    ${OP_SRC_FILES}
    ${ARCH_SRC_FILES}
    PARENT_SCOPE
)
