# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

include_guard(GLOBAL)

function(ops_blas_enable_gemmex_catlass target_name)
    if(NOT TARGET "${target_name}")
        message(FATAL_ERROR "GemmEx CATLASS requires an existing ops-blas target")
    endif()
    # A3 also uses arch22. Other architectures and builds without the native
    # GemmEx source must neither fetch CATLASS nor inherit its architecture macro.
    if(NOT "arch22" IN_LIST SOC_ARCH_DIRS)
        return()
    endif()
    get_target_property(_target_sources "${target_name}" SOURCES)
    set(_has_native FALSE)
    foreach(_source IN LISTS _target_sources)
        if(_source MATCHES "(^|/)gemm_ex/arch22/gemm_ex_matmul_kernel[.]cpp$")
            set(_has_native TRUE)
        endif()
    endforeach()
    if(NOT _has_native)
        return()
    endif()

    set(_revision "14d5536bb59cb5822ea3f3e2090b0917d7a4e7b4")
    set(_expected_tree "84526ab444dfd492f0de76dfc31f78defe9c042fe48f3f1d7def4233400d0db7")
    set(OPS_BLAS_CATLASS_SOURCE_DIR "${OPS_BLAS_CATLASS_SOURCE_DIR}" CACHE PATH
        "Existing CATLASS checkout/export at the required pinned revision")
    option(OPS_BLAS_CATLASS_OFFLINE "Require an existing verified CATLASS source without fetching" OFF)
    set(OPS_BLAS_CATLASS_GIT_REPOSITORY "https://gitcode.com/cann/catlass.git" CACHE STRING
        "CATLASS upstream or mirror; required revision and header content remain fixed")
    set(_source_dir "${OPS_BLAS_CATLASS_SOURCE_DIR}")
    # Retain the development option only as an explicit verified source input.
    # It no longer turns on unpinned global includes/architecture definitions.
    if(DEFINED GEMMEX_CATLASS_INCLUDE AND NOT "${GEMMEX_CATLASS_INCLUDE}" STREQUAL "")
        get_filename_component(_legacy_source "${GEMMEX_CATLASS_INCLUDE}/.." REALPATH)
        if(NOT "${_source_dir}" STREQUAL "")
            get_filename_component(_new_source "${_source_dir}" REALPATH)
            if(NOT _new_source STREQUAL _legacy_source)
                message(FATAL_ERROR "Conflicting CATLASS source and legacy include paths")
            endif()
        endif()
        set(_source_dir "${_legacy_source}")
    endif()
    if("${_source_dir}" STREQUAL "")
        if(DEFINED CANN_3RD_LIB_PATH AND NOT "${CANN_3RD_LIB_PATH}" STREQUAL "")
            set(_source_dir "${CANN_3RD_LIB_PATH}/catlass")
        else()
            set(_source_dir "${CMAKE_BINARY_DIR}/third_party/catlass")
        endif()
        if(NOT EXISTS "${_source_dir}/include/catlass/catlass.hpp")
            if(OPS_BLAS_CATLASS_OFFLINE OR FETCHCONTENT_FULLY_DISCONNECTED)
                message(FATAL_ERROR
                    "Pinned CATLASS cache is missing in offline mode. "
                    "Set OPS_BLAS_CATLASS_SOURCE_DIR to revision ${_revision}.")
            endif()
            include(FetchContent)
            # Older FetchContent loses an empty GIT_SUBMODULES argument while
            # forwarding it to its subbuild. Scope Git's submodule pathspec to
            # the pinned include/ tree instead; it contains no submodules.
            # This avoids unrelated googletest/AscendNPU-IR downloads on 3.16.
            if(POLICY CMP0097)
                cmake_policy(SET CMP0097 NEW)
            endif()
            set(_header_only_subdir "")
            if(NOT CMAKE_VERSION VERSION_LESS 3.18)
                # The pinned include/ tree has no CMakeLists.txt, so modern
                # MakeAvailable will populate headers without adding examples.
                set(_header_only_subdir SOURCE_SUBDIR include)
            endif()
            FetchContent_Declare(ops_blas_gemmex_catlass
                GIT_REPOSITORY "${OPS_BLAS_CATLASS_GIT_REPOSITORY}"
                GIT_TAG "${_revision}"
                GIT_SHALLOW FALSE
                GIT_SUBMODULES include
                UPDATE_DISCONNECTED TRUE
                SOURCE_DIR "${_source_dir}"
                BINARY_DIR "${CMAKE_BINARY_DIR}/_deps/gemmex-catlass-unused"
                ${_header_only_subdir})
            if(CMAKE_VERSION VERSION_LESS 3.18)
                FetchContent_GetProperties(ops_blas_gemmex_catlass)
                if(NOT ops_blas_gemmex_catlass_POPULATED)
                    FetchContent_Populate(ops_blas_gemmex_catlass)
                endif()
            else()
                FetchContent_MakeAvailable(ops_blas_gemmex_catlass)
            endif()
        endif()
    endif()
    if(NOT IS_DIRECTORY "${_source_dir}" OR
       NOT EXISTS "${_source_dir}/include/catlass/catlass.hpp" OR
       NOT EXISTS "${_source_dir}/include/tla/tensor.hpp")
        message(FATAL_ERROR "CATLASS source is incomplete: ${_source_dir}")
    endif()
    get_filename_component(_source_dir "${_source_dir}" REALPATH)
    if(EXISTS "${_source_dir}/.git")
        find_package(Git REQUIRED)
        execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${_source_dir}" rev-parse HEAD
            RESULT_VARIABLE _git_status OUTPUT_VARIABLE _head OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET)
        if(NOT _git_status EQUAL 0 OR NOT _head STREQUAL _revision)
            message(FATAL_ERROR "CATLASS checkout must be ${_revision}; found ${_head}. Local checkout was not changed.")
        endif()
    endif()
    # A commit name alone does not reject dirty headers. Exports without .git
    # are also supported by checking all public Catlass/TLA headers and licenses.
    file(GLOB_RECURSE _headers LIST_DIRECTORIES FALSE RELATIVE "${_source_dir}"
        "${_source_dir}/include/*")
    set(_files LICENSE LICENSE-APACHE ${_headers})
    list(SORT _files)
    set(_hash_records "")
    foreach(_file IN LISTS _files)
        if(NOT EXISTS "${_source_dir}/${_file}")
            message(FATAL_ERROR "Required CATLASS license/header is missing: ${_file}")
        endif()
        file(SHA256 "${_source_dir}/${_file}" _sha)
        string(APPEND _hash_records "${_file} ${_sha}\n")
    endforeach()
    string(SHA256 _actual_tree "${_hash_records}")
    if(NOT _actual_tree STREQUAL _expected_tree)
        message(FATAL_ERROR
            "CATLASS headers/licenses differ from pinned revision ${_revision}. "
            "Expected ${_expected_tree}, found ${_actual_tree}. Source was not reset or overwritten.")
    endif()
    # Keep the same header precedence as the validated development -I option.
    # Source properties keep arch22's private dependency out of ordinary Sgemm
    # and every other operator in the same shared-library target.
    foreach(_source IN LISTS _target_sources)
        if(_source MATCHES "(^|/)gemm_ex/arch22/.*[.]cpp$")
            set_property(SOURCE "${_source}" APPEND PROPERTY
                INCLUDE_DIRECTORIES "${_source_dir}/include")
            set_property(SOURCE "${_source}" APPEND PROPERTY
                COMPILE_DEFINITIONS GEMMEX_CATLASS_ENABLED CATLASS_ARCH=2201)
        endif()
    endforeach()
    # These isolated short-K kernels retain all explicit producer/consumer
    # waits. Each compute/output pair shares one translation unit and retains
    # its validated sync option; kernel launches remain separate.
    set_property(SOURCE
        "${PROJECT_SOURCE_DIR}/blas/gemm_ex/arch22/gemmex_short_nn_kernel.cpp"
        "${PROJECT_SOURCE_DIR}/blas/gemm_ex/arch22/sgemm_parity_nn_kernel.cpp"
        APPEND PROPERTY COMPILE_OPTIONS "--cce-auto-sync=off")
    set_property(TARGET "${target_name}" PROPERTY OPS_BLAS_CATLASS_REVISION "${_revision}")
    set_property(TARGET "${target_name}" PROPERTY OPS_BLAS_CATLASS_HEADER_SHA256 "${_actual_tree}")
    message(STATUS "GemmEx CATLASS ${_revision}: verified headers/licenses at ${_source_dir}")
endfunction()
