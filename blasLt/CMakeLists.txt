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

# Collect every blasLt source, then classify by role so the parent scope can assign compile
# languages without maintaining a per-file whitelist:
#   *_kernel.cpp            -> ASC (device kernels)
#   api / internal / engine / *_host.cpp -> CXX (pure host code)
# Architecture-specific operator code lives under <op>/<variant>/<archXX>/; only the arch dirs
# matching the current SOC (SOC_ARCH_DIRS) are compiled.
file(GLOB_RECURSE BLASLT_ALL_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

if("arch22" IN_LIST SOC_ARCH_DIRS)
    file(GLOB_RECURSE BLASLT_ARCH22_TASK_SRCS CONFIGURE_DEPENDS
        ${CMAKE_SOURCE_DIR}/blas/lt_matmul/arch22/*.cpp)
    list(APPEND BLASLT_ALL_SRCS ${BLASLT_ARCH22_TASK_SRCS})
endif()

set(BLASLT_HOST_SRCS "")
set(BLASLT_KERNEL_SRCS "")

foreach(src_file ${BLASLT_ALL_SRCS})
    # Skip sources under an architecture directory that does not match the current SOC.
    set(is_arch_specific FALSE)
    foreach(arch_dir ${ARCH_SPECIFIC_DIRS})
        if(src_file MATCHES "/${arch_dir}/")
            set(is_arch_specific TRUE)
        endif()
    endforeach()
    set(arch_selected FALSE)
    foreach(arch_dir ${SOC_ARCH_DIRS})
        if(src_file MATCHES "/${arch_dir}/")
            set(arch_selected TRUE)
        endif()
    endforeach()

    if(is_arch_specific AND NOT arch_selected)
        # architecture directory for another SOC: do not compile it here
    elseif(src_file MATCHES "_kernel\\.cpp$" OR
           src_file MATCHES "/lt_matmul/arch22/lt_matmul_(vector|ordered)\\.cpp$")
        list(APPEND BLASLT_KERNEL_SRCS ${src_file})
    else()
        list(APPEND BLASLT_HOST_SRCS ${src_file})
    endif()
endforeach()

# MXFP8/MXFP4 require asc-devkit >= 9.1 (ENABLE_BLASLT_MXFP8). Exclude the whole precision subtrees
# rather than filtering individual files.
if(NOT ENABLE_BLASLT_MXFP8)
    list(FILTER BLASLT_HOST_SRCS EXCLUDE REGEX "/mxfp8/")
    list(FILTER BLASLT_HOST_SRCS EXCLUDE REGEX "/mxfp4/")
    list(FILTER BLASLT_KERNEL_SRCS EXCLUDE REGEX "/mxfp8/")
    list(FILTER BLASLT_KERNEL_SRCS EXCLUDE REGEX "/mxfp4/")
    message(
        STATUS
        "Skipping blasLt MXFP8/MXFP4 (requires asc-devkit >= 9.1, current ${ASC_DEVKIT_MAJOR}.${ASC_DEVKIT_MINOR})")
endif()

# Exported to the parent scope; the top-level CMakeLists assigns LANGUAGE per list.
set(ALL_BLASLT_SRC_FILES ${BLASLT_HOST_SRCS} ${BLASLT_KERNEL_SRCS} PARENT_SCOPE)
set(ALL_BLASLT_HOST_SRC_FILES ${BLASLT_HOST_SRCS} PARENT_SCOPE)
set(ALL_BLASLT_KERNEL_SRC_FILES ${BLASLT_KERNEL_SRCS} PARENT_SCOPE)
