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

function(_ops_blas_write_preflight_report directory result output)
    file(READ "${OPS_BLAS_REFERENCE_ROOT}/reference.json" _manifest)
    set(_report "mode=source\nprofile=${OPS_BLAS_REFERENCE_ID}\narchitecture=${OPS_BLAS_REFERENCE_ARCH}\n")
    foreach(_field source_url source_sha256 cc fc flags)
        string(JSON _value GET "${_manifest}" "${_field}")
        string(APPEND _report "${_field}=${_value}\n")
    endforeach()
    string(APPEND _report "headers=${REFBLAS_INCLUDE_DIR}\nlibraries=${REFBLAS_LIB}\n")
    string(APPEND _report "preflight_exit=${result}\n${output}")
    file(WRITE "${directory}/report.txt" "${_report}")
endfunction()

# A GLOBAL property lasts one configure only. Never reuse an earlier configure's result.
function(_ops_blas_preflight_reference)
    set(_signature "${OPS_BLAS_REFERENCE_ID}|${REFBLAS_INCLUDE_DIR}|${REFBLAS_LIB}")
    get_property(_checked GLOBAL PROPERTY OPS_BLAS_PREFLIGHT_REFERENCE)
    if("${_checked}" STREQUAL "${_signature}")
        return()
    endif()
    if(CMAKE_CROSSCOMPILING)
        message(FATAL_ERROR "Reference BLAS preflight requires a native host build")
    endif()
    set(_directory "${CMAKE_BINARY_DIR}/reference-blas-preflight")
    set(_executable "${_directory}/reference_blas_probe")
    unset(_compiled CACHE)
    try_compile(_compiled "${_directory}/build"
        "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../scripts/reference_blas/probe.cpp"
        CMAKE_FLAGS "-DINCLUDE_DIRECTORIES:STRING=${REFBLAS_INCLUDE_DIR}"
        COMPILE_DEFINITIONS "-DOPS_BLAS_REFERENCE_PROFILE=\"${OPS_BLAS_REFERENCE_ID}\""
            -fno-fast-math -ffp-contract=off
        LINK_LIBRARIES ${REFBLAS_LIB} ${CMAKE_DL_LIBS}
        CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON
        COPY_FILE "${_executable}"
        OUTPUT_VARIABLE _build_output)
    file(WRITE "${_directory}/build.log" "${_build_output}")
    if(NOT _compiled)
        _ops_blas_write_preflight_report("${_directory}" "build-failed" "${_build_output}")
        message(FATAL_ERROR "Reference BLAS CPU preflight failed to build:\n${_build_output}")
    endif()
    execute_process(COMMAND "${_executable}" TIMEOUT 15
        RESULT_VARIABLE _result OUTPUT_VARIABLE _output ERROR_VARIABLE _error)
    file(WRITE "${_directory}/run.log" "${_output}${_error}")
    _ops_blas_write_preflight_report("${_directory}" "${_result}" "${_output}${_error}")
    if(NOT "${_result}" STREQUAL "0")
        message(FATAL_ERROR "Reference BLAS CPU preflight failed (${_result}):\n${_output}${_error}")
    endif()
    message(STATUS "Reference BLAS CPU preflight passed:\n${_output}")
    message(STATUS "Reference environment report: ${_directory}/report.txt")
    set_property(GLOBAL PROPERTY OPS_BLAS_PREFLIGHT_REFERENCE "${_signature}")
endfunction()

function(_ops_blas_load_source_reference)
    set(_reference_root "$ENV{OPS_BLAS_REFERENCE_ROOT}")
    if(NOT _reference_root)
        set(_reference_root "${CMAKE_SOURCE_DIR}/third_party/reference-blas")
    endif()
    set(OPS_BLAS_REFERENCE_ROOT "${_reference_root}" CACHE PATH "Reference built by scripts/build_reference_blas.py")
    if(NOT EXISTS "${OPS_BLAS_REFERENCE_ROOT}/reference.cmake")
        message(FATAL_ERROR "The fixed source-built BLAS reference is missing. Run:\n"
            "  python3 scripts/build_reference_blas.py --archive <lapack-v3.10.0.tar.gz>\n"
            "Or set -DOPS_BLAS_REFERENCE_ROOT=<existing reference bundle>.\n"
            "Source mode does not fall back to system libraries.")
    endif()
    include("${OPS_BLAS_REFERENCE_ROOT}/reference.cmake")
    if(NOT OPS_BLAS_REFERENCE_ID STREQUAL "netlib-3.10.0-flang18.1.8-gcc9.4.0-strict-v1"
       OR NOT OPS_BLAS_REFERENCE_ARCH STREQUAL CMAKE_HOST_SYSTEM_PROCESSOR)
        message(FATAL_ERROR "Reference bundle profile or host architecture does not match")
    endif()
    foreach(_reference_library IN LISTS REFBLAS_LIB)
        if(IS_ABSOLUTE "${_reference_library}" AND NOT EXISTS "${_reference_library}")
            message(FATAL_ERROR "Reference bundle is incomplete: ${_reference_library}")
        endif()
    endforeach()
    _ops_blas_preflight_reference()
    set(REFBLAS_INCLUDE_DIR "${REFBLAS_INCLUDE_DIR}" PARENT_SCOPE)
    set(REFBLAS_LIB "${REFBLAS_LIB}" PARENT_SCOPE)
    set(OPS_BLAS_REFERENCE_ID "${OPS_BLAS_REFERENCE_ID}" PARENT_SCOPE)
endfunction()
