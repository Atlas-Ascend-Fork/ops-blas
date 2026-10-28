# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

if(NOT DEFINED CANN_3RD_LIB_PATH)
  set(CANN_3RD_LIB_PATH "${CMAKE_BINARY_DIR}/third_party")
endif()

set(OPTENSOR_TAG_ID c1326e7a7fb30536dc3517ac40e06935aba5e88d)

if(EXISTS "${CANN_3RD_LIB_PATH}/ops-tensor")
  get_filename_component(OPTENSOR_SOURCE_PATH ${CANN_3RD_LIB_PATH}/ops-tensor REALPATH)
  message(STATUS "Find ops-tensor source dir: ${OPTENSOR_SOURCE_PATH}")
  # Idempotent handling of an existing clone (issue #648): skip when HEAD already
  # matches the pinned SHA; otherwise only touch a clean worktree and give clear
  # guidance for a dirty one instead of force-checking-out user state.
  execute_process(
    COMMAND git rev-parse HEAD
    WORKING_DIRECTORY ${OPTENSOR_SOURCE_PATH}
    OUTPUT_VARIABLE OPTENSOR_HEAD_SHA
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE OPTENSOR_HEAD_RESULT
  )
  if(NOT OPTENSOR_HEAD_RESULT EQUAL 0)
    message(FATAL_ERROR "ops-tensor: cannot read HEAD of the existing clone at ${OPTENSOR_SOURCE_PATH}")
  endif()
  if("${OPTENSOR_HEAD_SHA}" STREQUAL "${OPTENSOR_TAG_ID}")
    message(STATUS "ops-tensor already at ${OPTENSOR_TAG_ID}")
  else()
    execute_process(
      COMMAND git status --porcelain --untracked-files=no
      WORKING_DIRECTORY ${OPTENSOR_SOURCE_PATH}
      OUTPUT_VARIABLE OPTENSOR_DIRTY
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
      RESULT_VARIABLE OPTENSOR_STATUS_RESULT
    )
    if(NOT OPTENSOR_STATUS_RESULT EQUAL 0)
      message(FATAL_ERROR "ops-tensor: git status failed on ${OPTENSOR_SOURCE_PATH}")
    endif()
    if(NOT "${OPTENSOR_DIRTY}" STREQUAL "")
      message(WARNING
        "ops-tensor: ${OPTENSOR_SOURCE_PATH} is at ${OPTENSOR_HEAD_SHA} (expected ${OPTENSOR_TAG_ID}) "
        "and has uncommitted changes; leaving it untouched. Commit or stash the changes, "
        "or check out ${OPTENSOR_TAG_ID} manually, then re-run cmake.")
    else()
      execute_process(
        COMMAND git checkout ${OPTENSOR_TAG_ID}
        WORKING_DIRECTORY ${OPTENSOR_SOURCE_PATH}
        RESULT_VARIABLE EXEC_RESULT
        OUTPUT_VARIABLE EXEC_INFO
        ERROR_VARIABLE EXEC_ERROR
      )
      if(NOT "${EXEC_RESULT}" STREQUAL "0")
        message(FATAL_ERROR "Git checkout failed! result: ${EXEC_RESULT}, error: ${EXEC_ERROR}")
      endif()
    endif()
  endif()

  # Submodule: try local cache first (offline-friendly), fall back to network
  execute_process(
    COMMAND git submodule update --init --recursive --no-fetch
    WORKING_DIRECTORY ${OPTENSOR_SOURCE_PATH}
    RESULT_VARIABLE SUBMODULE_RESULT
    OUTPUT_QUIET ERROR_QUIET
  )
  if(NOT SUBMODULE_RESULT EQUAL 0)
    message(STATUS "ops-tensor submodule: local cache insufficient, fetching from remote")
    execute_process(
      COMMAND git submodule update --init --recursive
      WORKING_DIRECTORY ${OPTENSOR_SOURCE_PATH}
      TIMEOUT 300
      RESULT_VARIABLE SUBMODULE_RESULT
      ERROR_VARIABLE SUBMODULE_ERROR
      OUTPUT_QUIET
    )
    if(NOT SUBMODULE_RESULT EQUAL 0)
      message(WARNING "ops-tensor submodule update failed: ${SUBMODULE_ERROR}")
    else()
      message(STATUS "ops-tensor submodule update")
    endif()
  endif()
else()
  include(FetchContent)

  FetchContent_Declare(
    ops-tensor
    GIT_REPOSITORY https://gitcode.com/cann/ops-tensor.git
    GIT_TAG ${OPTENSOR_TAG_ID}
    GIT_PROGRESS TRUE
    SOURCE_DIR ${CANN_3RD_LIB_PATH}/ops-tensor)

  FetchContent_Populate(ops-tensor)

  set(OPTENSOR_SOURCE_PATH ${CANN_3RD_LIB_PATH}/ops-tensor)

  execute_process(
    COMMAND git submodule update --init --recursive
    WORKING_DIRECTORY ${OPTENSOR_SOURCE_PATH}
    TIMEOUT 300
    RESULT_VARIABLE SUBMODULE_RESULT
    ERROR_VARIABLE SUBMODULE_ERROR
    OUTPUT_QUIET
  )
  if(NOT SUBMODULE_RESULT EQUAL 0)
    message(WARNING "ops-tensor submodule update failed: ${SUBMODULE_ERROR}")
  else()
    message(STATUS "ops-tensor submodule update")
  endif()
endif()

# Fail fast if the tensor_api submodule is not at the commit recorded by the pinned ops-tensor tag:
# a partial/failed submodule update (the WARNING paths above) would otherwise let CI silently compile
# against stale or missing tensor_api headers.
execute_process(
  COMMAND git ls-tree HEAD include/tensor_api
  WORKING_DIRECTORY ${OPTENSOR_SOURCE_PATH}
  OUTPUT_VARIABLE _OPTENSOR_SUBMODULE_LS
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET)
string(REPLACE "\t" " " _OPTENSOR_SUBMODULE_LS "${_OPTENSOR_SUBMODULE_LS}")
separate_arguments(_OPTENSOR_SUBMODULE_PARTS UNIX_COMMAND "${_OPTENSOR_SUBMODULE_LS}")
set(_OPTENSOR_SUBMODULE_MODE "")
set(_OPTENSOR_SUBMODULE_EXPECTED "")
list(LENGTH _OPTENSOR_SUBMODULE_PARTS _OPTENSOR_SUBMODULE_N)
if(_OPTENSOR_SUBMODULE_N GREATER_EQUAL 3)
  list(GET _OPTENSOR_SUBMODULE_PARTS 0 _OPTENSOR_SUBMODULE_MODE)
  list(GET _OPTENSOR_SUBMODULE_PARTS 2 _OPTENSOR_SUBMODULE_EXPECTED)
endif()
if(_OPTENSOR_SUBMODULE_MODE STREQUAL "160000")
  execute_process(
    COMMAND git rev-parse HEAD
    WORKING_DIRECTORY ${OPTENSOR_SOURCE_PATH}/include/tensor_api
    OUTPUT_VARIABLE _OPTENSOR_SUBMODULE_ACTUAL
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
  if(NOT _OPTENSOR_SUBMODULE_ACTUAL STREQUAL _OPTENSOR_SUBMODULE_EXPECTED)
    message(
      FATAL_ERROR
        "ops-tensor submodule 'include/tensor_api' is not at the commit required by tag ${OPTENSOR_TAG_ID}.\n"
        "  expected: ${_OPTENSOR_SUBMODULE_EXPECTED}\n"
        "  actual:   ${_OPTENSOR_SUBMODULE_ACTUAL}\n"
        "Fix: git -C ${OPTENSOR_SOURCE_PATH} submodule update --init --recursive")
  endif()
endif()

set(OPTENSOR_INCLUDE_DIR "${OPTENSOR_SOURCE_PATH}/include")

if(NOT EXISTS "${OPTENSOR_INCLUDE_DIR}/tensor_api" AND NOT EXISTS "${OPTENSOR_INCLUDE_DIR}/blaze")
  message(
    FATAL_ERROR
      "ops-tensor headers not found: expected tensor_api/ or blaze/ under ${OPTENSOR_INCLUDE_DIR}. "
      "Set sibling clone at ../ops-tensor or ensure FetchContent succeeded.")
endif()
