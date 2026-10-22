# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

ops_blas_add_gtest_tests(${OPS_BLAS} strmm_test)

if("arch22" IN_LIST SOC_ARCH_DIRS)
    # The A2/A3 task requires reference Netlib STRMM. A library named libblas can
    # instead be an OpenBLAS alternatives symlink. Check the actual implementation,
    # then bind the adapter to that file rather than the process-global strmm_ symbol.
    set(STRMM_NETLIB_LIB "$ENV{STRMM_NETLIB_LIB}" CACHE FILEPATH
        "Netlib reference BLAS shared library for the arch22 STRMM golden")
    if(CMAKE_CROSSCOMPILING)
        message(FATAL_ERROR "STRMM golden validation must run on the test host; configure natively with Netlib BLAS.")
    endif()

    if(STRMM_NETLIB_LIB)
        set(_strmm_candidates "${STRMM_NETLIB_LIB}")
    else()
        set(_strmm_candidates "${REFBLAS_LIB}")
        foreach(_dir IN LISTS CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES)
            list(APPEND _strmm_candidates "${_dir}/blas/libblas.so" "${_dir}/libblas.so")
        endforeach()
    endif()
    list(REMOVE_DUPLICATES _strmm_candidates)

    set(_strmm_selected "")
    set(_strmm_errors "")
    foreach(_candidate IN LISTS _strmm_candidates)
        if(NOT EXISTS "${_candidate}")
            continue()
        endif()
        get_filename_component(_candidate "${_candidate}" REALPATH)
        unset(_strmm_probe_run CACHE)
        unset(_strmm_probe_build CACHE)
        try_run(_strmm_probe_run _strmm_probe_build
            "${CMAKE_CURRENT_BINARY_DIR}/netlib_probe"
            "${CMAKE_CURRENT_LIST_DIR}/arch22/strmm_test_netlib.cpp"
            CMAKE_FLAGS "-DCMAKE_CXX_STANDARD=17" "-DINCLUDE_DIRECTORIES:STRING=${REFBLAS_INCLUDE_DIR}"
            COMPILE_DEFINITIONS -DSTRMM_NETLIB_PROBE "-DSTRMM_NETLIB_LIBRARY=\"${_candidate}\""
            LINK_LIBRARIES ${CMAKE_DL_LIBS}
            COMPILE_OUTPUT_VARIABLE _strmm_probe_compile_output
            RUN_OUTPUT_VARIABLE _strmm_probe_output)
        # A Fortran XERBLA/STOP may exit with status zero on an ABI mismatch. Require
        # the success marker emitted only after both calls returned and were checked.
        if(_strmm_probe_build AND "${_strmm_probe_run}" STREQUAL "0" AND
           _strmm_probe_output MATCHES "STRMM golden.*Netlib strmm_:")
            set(_strmm_selected "${_candidate}")
            break()
        endif()
        string(APPEND _strmm_errors "\n${_candidate}: ${_strmm_probe_output}")
        if(NOT _strmm_probe_build)
            string(APPEND _strmm_errors "${_strmm_probe_compile_output}")
        endif()
    endforeach()

    if(NOT _strmm_selected)
        message(FATAL_ERROR
            "No usable Netlib reference STRMM library. OpenBLAS is not the task's golden.\n"
            "Install reference BLAS and set -DSTRMM_NETLIB_LIB=/absolute/path/to/libblas.so,\n"
            "or export STRMM_NETLIB_LIB before running build.sh.\n"
            "Checked: ${_strmm_candidates}\n${_strmm_errors}")
    endif()
    set(STRMM_NETLIB_LIB "${_strmm_selected}" CACHE FILEPATH
        "Netlib reference BLAS shared library for the arch22 STRMM golden" FORCE)
    file(SHA256 "${STRMM_NETLIB_LIB}" _strmm_netlib_sha256)
    message(STATUS "STRMM Netlib golden: ${STRMM_NETLIB_LIB} (sha256=${_strmm_netlib_sha256})")
    target_compile_definitions(strmm_test PRIVATE "STRMM_NETLIB_LIBRARY=\"${STRMM_NETLIB_LIB}\"")
    target_link_libraries(strmm_test PRIVATE ${CMAKE_DL_LIBS})
endif()
