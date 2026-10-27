# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16.0)

project(ops_blas VERSION 1.0.0)

# ========= 基本配置 =========
if(NOT DEFINED SOC_VERSION OR SOC_VERSION STREQUAL "")
  set(SOC_VERSION ascend910b3)
endif()
string(TOLOWER "${SOC_VERSION}" SOC_VERSION_LOWER)
# SOC_VERSION -> NPU_ARCH 映射（ascend910b->dav-2201, ascend950->dav-3101）
if(SOC_VERSION_LOWER MATCHES "^ascend910b")
  set(NPU_ARCH "dav-2201")
elseif(SOC_VERSION_LOWER MATCHES "^ascend910_93")
  set(NPU_ARCH "dav-2201")
elseif(SOC_VERSION_LOWER MATCHES "^ascend950")
  set(NPU_ARCH "dav-3510")
elseif(SOC_VERSION_LOWER MATCHES "^ascend310p")
  set(NPU_ARCH "dav-1101")
else()
  message(FATAL_ERROR "Unsupported SOC_VERSION: ${SOC_VERSION}. Supported: ascend910b*, ascend910_93*, ascend950*, ascend310p*")
endif()
message(STATUS "SOC_VERSION=${SOC_VERSION}, NPU_ARCH=${NPU_ARCH}")

# ========= SOC 架构目录配置 =========
# 根据 SOC 返回需要编译的架构目录列表
function(get_soc_arch_dirs soc_version arch_dirs)
  string(TOLOWER "${soc_version}" soc_lower)
  set(dirs "")

  if(soc_lower MATCHES "^ascend910b")
    list(APPEND dirs "arch22")
  elseif(soc_lower MATCHES "^ascend910_93")
    list(APPEND dirs "arch22")
  elseif(soc_lower MATCHES "^ascend950")
    list(APPEND dirs "arch35")
  elseif(soc_lower MATCHES "^ascend310p")
    list(APPEND dirs "arch20")
  endif()

  set(${arch_dirs} ${dirs} PARENT_SCOPE)
endfunction()

# 获取当前 SOC 对应的架构目录
get_soc_arch_dirs(${SOC_VERSION} SOC_ARCH_DIRS)
message(STATUS "SOC_ARCH_DIRS=${SOC_ARCH_DIRS}")

# 定义所有支持的架构特定目录（全局唯一定义）
set(ARCH_SPECIFIC_DIRS arch35 arch22 arch20 CACHE INTERNAL "Architecture specific directories")

set(ASCEND_CANN_PACKAGE_PATH $ENV{ASCEND_HOME_PATH})

include(cmake/asc_devkit_version.cmake)
ops_blas_detect_asc_devkit_version()

set(RUN_MODE "npu" CACHE STRING "run mode: npu")
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type Release/Debug" FORCE)
endif()
set(CMAKE_ASC_FLAGS_RELEASE "-O3" CACHE STRING "AscendC compile flags (Release)" FORCE)
set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/out" CACHE STRING "install path" FORCE)

# 外部传参
option(ENABLE_PACKAGE "Enable build package" OFF)

# 引入 ops-tensor
include(cmake/third_party/ops-tensor.cmake)

# ========= 构建 blasLt 动态库 =========
set(ALL_BLASLT_SRC_FILES "")
set(OPS_BLASLT ops_blasLt)
find_package(ASC REQUIRED)
add_subdirectory(blasLt)

project(${OPS_BLASLT} LANGUAGES ASC CXX)
add_library(${OPS_BLASLT} SHARED ${ALL_BLASLT_SRC_FILES})

# issue #615: 删除对未初始化 ALL_BLAS_SRC_FILES 的空操作 set_source_files_properties
# （该变量在 blas 子目录加入后才填充，真正生效的调用位于 add_library(${OPS_BLAS}) 之前）

# cgerc has task-level microsecond latency targets. Keep its arch35 device
# implementation optimized even when the repository's test build uses Debug.
if("arch35" IN_LIST SOC_ARCH_DIRS)
    set_source_files_properties(
        ${CMAKE_CURRENT_SOURCE_DIR}/blas/gerc/arch35/cgerc_kernel.cpp
        PROPERTIES COMPILE_OPTIONS "-O3;-DNDEBUG;-funroll-loops;-ffp-contract=off"
    )
endif()

# Device kernels compile with the ASC frontend; all host code (api / internal / engine / *_host.cpp)
# compiles as plain C++. The classification is produced by blasLt/CMakeLists.txt, so no per-file
# language whitelist is maintained here.
set_source_files_properties(
    ${ALL_BLASLT_KERNEL_SRC_FILES}
    PROPERTIES LANGUAGE ASC
)
set_source_files_properties(
    ${ALL_BLASLT_HOST_SRC_FILES}
    PROPERTIES LANGUAGE CXX
)

set(_OPS_BLAS_ASC_ROOT "${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/asc")

# Compatibility headers from asc-devkit!5818, generated in the build tree.
set(_OPS_BLAS_TENSOR_COMPAT_ROOT "")
if("arch35" IN_LIST SOC_ARCH_DIRS AND EXISTS "${_OPS_BLAS_ASC_ROOT}/include/utils/base/sys_macros.h")
  set(_OPS_BLAS_TENSOR_COMPAT_ROOT "${CMAKE_CURRENT_BINARY_DIR}/generated/tensor_api_sdk_compat")
  foreach(_compat_name utils_impl.h utils_type.h)
    set(_compat_relative "impl/c_api/instr_impl/npu_arch_2201/utils_impl/${_compat_name}")
    set(_compat_source "${OPTENSOR_INCLUDE_DIR}/tensor_api/${_compat_relative}")
    set(_compat_output "${_OPS_BLAS_TENSOR_COMPAT_ROOT}/${_compat_relative}")
    if(NOT EXISTS "${_compat_source}")
      message(FATAL_ERROR "Expected pinned tensor_api header is missing: ${_compat_source}")
    endif()
    file(READ "${_compat_source}" _compat_original)
    string(REPLACE "#include \"impl/utils/sys_macros.h\""
                   "#include \"utils/base/sys_macros.h\""
                   _compat_patched "${_compat_original}")
    if(_compat_patched STREQUAL _compat_original)
      string(FIND "${_compat_original}" "#include \"utils/base/sys_macros.h\"" _compat_public)
      if(_compat_public EQUAL -1)
        message(FATAL_ERROR "Unexpected tensor_api include layout: ${_compat_source}")
      endif()
    endif()
    get_filename_component(_compat_output_dir "${_compat_output}" DIRECTORY)
    file(MAKE_DIRECTORY "${_compat_output_dir}")
    set(_compat_existing "")
    if(EXISTS "${_compat_output}")
      file(READ "${_compat_output}" _compat_existing)
    endif()
    if(NOT _compat_existing STREQUAL _compat_patched)
      file(WRITE "${_compat_output}" "${_compat_patched}")
    endif()
  endforeach()
  message(STATUS "Using build-local tensor_api SDK include compatibility headers (asc-devkit!5818)")
endif()
target_include_directories(${OPS_BLASLT} PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/op_common/
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/base/
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/
    ./include
    ${OPTENSOR_INCLUDE_DIR}
)
# Use -iquote for third_party tensor_api paths so impl headers (e.g.
# transform_impl.h) are resolved from third_party, not from CANN paths
# injected by --asc-aicore-lang (CANN's copy_impl.h lacks transform_impl.h).
# -iquote directories are searched before -I for quoted includes (#include "...").
# SHELL: prefix prevents CMake from stripping the second -iquote (CMake's
# Makefile generator treats -iquote as a single-arg flag and drops duplicates).
target_compile_options(${OPS_BLASLT} PRIVATE
    "$<$<AND:$<COMPILE_LANGUAGE:ASC>,$<BOOL:${_OPS_BLAS_TENSOR_COMPAT_ROOT}>>:SHELL:-iquote \"${_OPS_BLAS_TENSOR_COMPAT_ROOT}\">"
    $<$<COMPILE_LANGUAGE:ASC>:SHELL:-iquote ${OPTENSOR_INCLUDE_DIR}/tensor_api -iquote ${OPTENSOR_INCLUDE_DIR}/tensor_api/include -DIMPL_TENSOR_API_UTILS_INT_IMPL_H>
)

if(EXISTS "${_OPS_BLAS_ASC_ROOT}/include")
  target_include_directories(${OPS_BLASLT} PRIVATE
    "${_OPS_BLAS_ASC_ROOT}"
    "${_OPS_BLAS_ASC_ROOT}/include"
    "${_OPS_BLAS_ASC_ROOT}/include/utils"
    "${_OPS_BLAS_ASC_ROOT}/impl/basic_api"
    "${_OPS_BLAS_ASC_ROOT}/impl/c_api"
    "${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include"
  )
endif()

# Private include roots for the refactored blasLt layout. internal/ headers are never installed;
# each operator's arch dir exposes its co-located host/kernel/tiling headers.
target_include_directories(${OPS_BLASLT} PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/blasLt/internal/include
    ${CMAKE_CURRENT_LIST_DIR}/blasLt/common/helper
    ${CMAKE_CURRENT_LIST_DIR}/blasLt/matmul/engine
    ${CMAKE_CURRENT_LIST_DIR}/blasLt/matmul/common
    ${CMAKE_CURRENT_LIST_DIR}/blasLt/matmul/fp32/arch35
    ${CMAKE_CURRENT_LIST_DIR}/blasLt/matmul/mxfp8/arch35
    ${CMAKE_CURRENT_LIST_DIR}/blasLt/matmul/mxfp4/arch35
    ${CMAKE_CURRENT_LIST_DIR}/blasLt/matmul/epilogue/arch35
    ${CMAKE_CURRENT_LIST_DIR}/blasLt/matrix_transform/engine
    ${CMAKE_CURRENT_LIST_DIR}/blasLt/matrix_transform/arch35
)

target_compile_definitions(${OPS_BLASLT} PRIVATE
  ASC_DEVKIT_MAJOR=${ASC_DEVKIT_MAJOR}
  ASC_DEVKIT_MINOR=${ASC_DEVKIT_MINOR})
target_compile_features(${OPS_BLASLT} PRIVATE cxx_std_17)

set(ASC_WARN_SUPPRESS "-Wno-#pragma-messages" "-Wno-deprecated-declarations" "-Wno-c++20-extensions")

# Older CANN toolchains (e.g. 9.1.0) are missing two things that ops-tensor's
# tensor_api headers rely on: the arch-gated bf16/fp8/fp4 vector builtins seen by
# the ASC front-end's arch-less pass, and the __asc_aicore debug-bus helpers.
# Both probes are resolved by ops_blas_detect_asc_devkit_version(); when either
# fires, force-include the compat header so every ASC translation unit sees the
# shims before tensor_api/tensor.h.
set(_ASC_TENSOR_API_COMPAT_DEFS "")
if(ASC_TENSOR_API_DEBUG_BUS_SHIM)
  list(APPEND _ASC_TENSOR_API_COMPAT_DEFS ASC_TENSOR_API_DEBUG_BUS_SHIM=1)
endif()
if(ASC_TENSOR_API_ARCH_TYPES_SHIM)
  list(APPEND _ASC_TENSOR_API_COMPAT_DEFS ASC_TENSOR_API_ARCH_TYPES_SHIM=1)
endif()
if(_ASC_TENSOR_API_COMPAT_DEFS)
  set(_ASC_TENSOR_API_COMPAT_OPTION
      $<$<COMPILE_LANGUAGE:ASC>:SHELL:-include ${CMAKE_CURRENT_LIST_DIR}/blas/common/helper/tensor_api_asc_compat.h>)
else()
  set(_ASC_TENSOR_API_COMPAT_OPTION "")
endif()

target_compile_options(${OPS_BLASLT} PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${NPU_ARCH} ${ASC_WARN_SUPPRESS} -DASC_DEVKIT_MAJOR=${ASC_DEVKIT_MAJOR} -DASC_DEVKIT_MINOR=${ASC_DEVKIT_MINOR}>
    $<$<COMPILE_LANGUAGE:CXX>:-DASC_DEVKIT_MAJOR=${ASC_DEVKIT_MAJOR} -DASC_DEVKIT_MINOR=${ASC_DEVKIT_MINOR}>
    ${_ASC_TENSOR_API_COMPAT_OPTION}
)
target_compile_definitions(${OPS_BLASLT} PRIVATE ${_ASC_TENSOR_API_COMPAT_DEFS})

target_link_directories(${OPS_BLASLT} PRIVATE "${ASCEND_CANN_PACKAGE_PATH}/lib64")
target_link_libraries(
  ${OPS_BLASLT}
  PRIVATE ascendc_runtime ascendcl runtime register error_manager profapi ge_common_base unified_dlog mmpa dl
          ascend_dump c_sec)

# ========= 构建 blas 动态库 =========
set(ALL_BLAS_SRC_FILES "")
set(OPS_BLAS ops_blas)

add_subdirectory(blas)
add_subdirectory(extensions)

project(${OPS_BLAS} LANGUAGES ASC CXX)

add_library(${OPS_BLAS} SHARED ${ALL_BLAS_SRC_FILES})

set_source_files_properties(
    ${ALL_BLAS_SRC_FILES}
    PROPERTIES LANGUAGE ASC
)

if(SOC_VERSION_LOWER MATCHES "^ascend950")
    # CSYMV's register-tiled hot loops need optimization even though this project
    # intentionally defaults to a Debug build for the remaining operators.
    set_source_files_properties(
        "${CMAKE_CURRENT_LIST_DIR}/blas/symv/arch35/csymv_kernel.cpp"
        PROPERTIES COMPILE_OPTIONS "-O3;-DNDEBUG"
    )
endif()

target_include_directories(${OPS_BLAS} PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/op_common/
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/base/
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/
    ${ASCEND_CANN_PACKAGE_PATH}/include/op_common/
    ./include
    ${CMAKE_CURRENT_LIST_DIR}/blas
    ${CMAKE_CURRENT_LIST_DIR}/extensions
    ${CMAKE_CURRENT_LIST_DIR}/extensions/complexmatdot/arch22
    "${_OPS_BLAS_ASC_ROOT}"
    "${_OPS_BLAS_ASC_ROOT}/include"
    "${_OPS_BLAS_ASC_ROOT}/include/utils"
    "${_OPS_BLAS_ASC_ROOT}/impl/basic_api"
    "${_OPS_BLAS_ASC_ROOT}/impl/c_api"
    "${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include"
    ${OPTENSOR_INCLUDE_DIR}
)

target_link_directories(${OPS_BLAS} PRIVATE "${ASCEND_CANN_PACKAGE_PATH}/lib64")
target_link_libraries(
  ${OPS_BLAS}
  PRIVATE tiling_api platform c_sec ascendc_runtime ascendcl runtime register)

if("${CMAKE_CURRENT_SOURCE_DIR}/blas/nrm2/arch35/scnrm2_host.cpp" IN_LIST ALL_BLAS_SRC_FILES)
  target_link_libraries(${OPS_BLAS} PRIVATE opapi nnopbase)
endif()

target_compile_definitions(${OPS_BLAS} PRIVATE
  ASC_DEVKIT_MAJOR=${ASC_DEVKIT_MAJOR}
  ASC_DEVKIT_MINOR=${ASC_DEVKIT_MINOR})

target_compile_options(${OPS_BLAS} PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${NPU_ARCH} ${ASC_WARN_SUPPRESS} -DASC_DEVKIT_MAJOR=${ASC_DEVKIT_MAJOR} -DASC_DEVKIT_MINOR=${ASC_DEVKIT_MINOR}>
    "$<$<AND:$<COMPILE_LANGUAGE:ASC>,$<BOOL:${_OPS_BLAS_TENSOR_COMPAT_ROOT}>>:SHELL:-iquote \"${_OPS_BLAS_TENSOR_COMPAT_ROOT}\">"
    $<$<COMPILE_LANGUAGE:ASC>:SHELL:-iquote ${OPTENSOR_INCLUDE_DIR}/tensor_api -iquote ${OPTENSOR_INCLUDE_DIR}/tensor_api/include -DIMPL_TENSOR_API_UTILS_INT_IMPL_H>
    $<$<COMPILE_LANGUAGE:CXX>:-DASC_DEVKIT_MAJOR=${ASC_DEVKIT_MAJOR} -DASC_DEVKIT_MINOR=${ASC_DEVKIT_MINOR}>
    ${_ASC_TENSOR_API_COMPAT_OPTION}
)
target_compile_definitions(${OPS_BLAS} PRIVATE ${_ASC_TENSOR_API_COMPAT_DEFS})

# ========= 安装规则（CPack 打包时放入 lib64 和 include） =========
install(TARGETS ${OPS_BLASLT} ${OPS_BLAS}
    LIBRARY DESTINATION lib64
)
install(FILES
    include/cann_ops_blas.h
    include/cann_ops_blasLt.h
    include/cann_ops_blas_common.h
    DESTINATION include
)

# ========= 构建测试程序 =========
option(BUILD_TEST "Build test programs" OFF)
if(BUILD_TEST)
    add_subdirectory(test)
endif()

if(ENABLE_PACKAGE)
    include(cmake/package.cmake)
    pack(${SOC_VERSION})
endif()

# Keep the validated Chemm SIMT/Cube kernels optimized even in Debug builds.
if("arch35" IN_LIST SOC_ARCH_DIRS)
    set_source_files_properties(
        "${CMAKE_CURRENT_LIST_DIR}/blas/hemm/arch35/chemm_kernel.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/blas/hemm/arch35/chemm_cube_kernel.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/blas/hemm/arch35/chemm_combine_norisk_kernel.cpp"
        PROPERTIES COMPILE_OPTIONS "-O2"
    )
endif()
