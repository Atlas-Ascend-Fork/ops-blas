# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

# 读取 asc_devkit_version.h，判断是否满足所需版本：ASC_DEVKIT_MAJOR >= 9 && ASC_DEVKIT_MINOR > 0
# - MXFP8/MXFP4：blasLt 矩阵乘法，需 asc-devkit >= 9.1
# - TENSOR_API_OPS 列表中的算子：仅 arch35(ascend950) 使用 tensor_api，需 asc-devkit >= 9.1；
#   其他架构不受此版本限制。新增算子只需在下方列表中加一行。
# - TRSM_BLOCKED：仅 arch35(ascend950) 的 strsm_blocked_kernel 使用 tensor_api，需 asc-devkit >= 9.1；
#   strsm_kernel(SIMT path) 不依赖 tensor_api，低版本下 fallback 到 SIMT（特例，不纳入列表驱动）
function(ops_blas_detect_asc_devkit_version)
  set(_header
      "${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/version/asc_devkit_version.h")
  set(ASC_DEVKIT_MAJOR 0)
  set(ASC_DEVKIT_MINOR 0)
  set(ENABLE_BLASLT_MXFP8 FALSE)

  # 列表驱动的 tensor_api 算子注册。
  # 格式: VAR_SUFFIX|blas_filter_regex|test_name1,test_name2
  # 使用 '|' 作为字段分隔符（CMake 列表以 ';' 分隔，字段内不能用 ';'）。
  # 新增算子只需在此列表中添加一行，cmake/asc_devkit_version.cmake、blas/CMakeLists.txt、
  # cmake/test.cmake、test/CMakeLists.txt 四个文件中的 foreach 会自动处理。
  set(TENSOR_API_OPS
      "STRMM|/trmm/arch35/strmm|strmm"
      "SDGMM|/dgmm/arch35/sdgmm|sdgmm"
      "GEMM_BATCHED|/gemm_batched/arch35/gemm_batched_|sgemm_batched,cgemm_batched"
      "SGEMM3M|/gemm3m/arch35/sgemm3m|sgemm3m"
      "SSYRK|/syrk/arch35/ssyrk|ssyrk"
      "SSYR2K|/syr2k/arch35/ssyr2k|ssyr2k"
      "CSYR2K|/syr2k/arch35/csyr2k|csyr2k"
      "SSYRKX|/syrkx/arch35/ssyrkx|ssyrkx"
      "CHERK|/herk/arch35/cherk|cherk"
      "CHER2K|/her2k/arch35/cher2k|cher2k"
      "STRSMBATCHED|/trsmbatched/arch35/strsmbatched|strsmbatched"
      "GEMM_STRIDED_BATCHED|/gemm_strided_batched/arch35/gemm_strided_batched_|gemm_strided_batched"
      "CGEMM_STRIDED_BATCHED|/gemm_strided_batched/arch35/cgemm_strided_batched_|cgemm_strided_batched"
      "GEMM|/gemm/arch35/gemm_|gemm"
  )

  # 初始化所有 tensor_api 算子为 TRUE
  foreach(entry ${TENSOR_API_OPS})
    string(REPLACE "|" ";" _fields "${entry}")
    list(GET _fields 0 _suffix)
    set(ENABLE_BLAS_${_suffix} TRUE)
  endforeach()
  set(ENABLE_BLAS_TRSM_BLOCKED TRUE)

  if(EXISTS "${_header}")
    file(READ "${_header}" _version_content)
    if(_version_content MATCHES "#define ASC_DEVKIT_MAJOR ([0-9]+)")
      set(ASC_DEVKIT_MAJOR "${CMAKE_MATCH_1}")
    endif()
    if(_version_content MATCHES "#define ASC_DEVKIT_MINOR ([0-9]+)")
      set(ASC_DEVKIT_MINOR "${CMAKE_MATCH_1}")
    endif()
    # MXFP8/MXFP4 need the full AscendC::Reg SIMD API (CastTrait/Cast/DataCopy/...),
    # which the blaze fmm epilogue pulls in through kernel_basic_intf.h when
    # ASC_DEVKIT_MAJOR >= 9. Some 9.1 devkits report version 9.1 but do NOT wire
    # AscendC::Reg into kernel_basic_intf.h, so a pure version check is insufficient
    # and the build fails with "no member 'CastTrait' in namespace AscendC::Reg".
    # Probe the actual capability instead of trusting the version number.
    set(_reg_api_wired FALSE)
    set(_kb_intf
        "${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/asc/include/basic_api/kernel_basic_intf.h")
    if(EXISTS "${_kb_intf}")
      file(READ "${_kb_intf}" _kb_content)
      if(_kb_content MATCHES "reg_compute" OR _kb_content MATCHES "namespace Reg")
        set(_reg_api_wired TRUE)
      endif()
    endif()
    if(ASC_DEVKIT_MAJOR GREATER_EQUAL 9 AND ASC_DEVKIT_MINOR GREATER 0 AND _reg_api_wired)
      set(ENABLE_BLASLT_MXFP8 TRUE)
    else()
      set(ENABLE_BLASLT_MXFP8 FALSE)
      if(ASC_DEVKIT_MAJOR GREATER_EQUAL 9 AND ASC_DEVKIT_MINOR GREATER 0 AND NOT _reg_api_wired)
        message(STATUS
                "AscendC::Reg SIMD API not wired in devkit ${ASC_DEVKIT_MAJOR}.${ASC_DEVKIT_MINOR}; "
                "disabling blasLt MXFP8/MXFP4 (requires AscendC::Reg via kernel_basic_intf.h)")
      endif()
    endif()
    # ops-tensor (>= 781745c) 的 print_tensor_debug_bus_impl.h 依赖工具链提供的
    # __asc_aicore::get_debug_bus_local_addr_shift / get_debug_bus_loop_steplen。
    # 新版本 CANN 有这两个 helper，但 CANN 9.1.0 的
    # asc/impl/utils/debug/npu_arch_3510/asc_aicore_dump_utils.h 只有寄存器级原语，
    # 于是每个 include tensor_api/tensor.h 的 cube 算子都会编译失败
    # （no template named 'get_debug_bus_local_addr_shift' in namespace '__asc_aicore'）。
    # 探测工具链头文件：缺失时置位 ASC_TENSOR_API_DEBUG_BUS_SHIM，由 build 强制
    # include 兼容头补上这两个声明。
    #
    # 两个 shim 都只对 arch35（A5 / DAV_3510）有意义，因此只在目标 SOC 属于 arch35 时
    # 才探测并启用；否则 arch22 等目标也会被强制 include 兼容头（cann-9.2.0 上曾因此
    # 在 arch22 全量报重定义）。
    set(ASC_TENSOR_API_DEBUG_BUS_SHIM FALSE)
    set(ASC_TENSOR_API_ARCH_TYPES_SHIM FALSE)
    if("arch35" IN_LIST SOC_ARCH_DIRS)
      # 定位 ASC 编译器并准备探测用 include 目录（与 kernel 编译保持一致）。
      set(_asc_probe_compiler "")
      foreach(_candidate
              "${ASCEND_CANN_PACKAGE_PATH}/bin/bisheng"
              "${ASCEND_CANN_PACKAGE_PATH}/tools/bisheng_compiler/bin/bisheng")
        if(EXISTS "${_candidate}")
          set(_asc_probe_compiler "${_candidate}")
          break()
        endif()
      endforeach()
      set(_asc_asc_root "${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/asc")
      set(_asc_probe_includes
          -I"${_asc_asc_root}"
          -I"${_asc_asc_root}/include"
          -I"${_asc_asc_root}/include/utils"
          -I"${_asc_asc_root}/impl/basic_api"
          -I"${_asc_asc_root}/impl/c_api"
          -I"${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include")

      if(_asc_probe_compiler)
        # debug-bus：9.1.0 的工具链缺这两个 helper，9.2.0 已有。用编译探测（引用即报错）
        # 判断，比 grep 头文件更可靠。
        set(_debug_bus_probe_src "${CMAKE_BINARY_DIR}/asc_debug_bus_probe.cpp")
        file(WRITE "${_debug_bus_probe_src}"
"#include \"kernel_operator.h\"\n"
"#include \"impl/utils/debug/npu_arch_3510/asc_aicore_dump_utils.h\"\n"
"using AscDebugBusProbeShift = decltype(__asc_aicore::get_debug_bus_local_addr_shift<AscendC::Hardware::L1>());\n"
"using AscDebugBusProbeStep = decltype(__asc_aicore::get_debug_bus_loop_steplen<AscendC::Hardware::L1>());\n"
"int asc_debug_bus_probe() { return 0; }\n")
        execute_process(
          COMMAND "${_asc_probe_compiler}" --npu-arch=${NPU_ARCH} -std=c++17
                  --asc-aicore-lang -fsyntax-only ${_asc_probe_includes}
                  "${_debug_bus_probe_src}"
          RESULT_VARIABLE _debug_bus_probe_rc
          OUTPUT_VARIABLE _debug_bus_probe_out
          ERROR_VARIABLE _debug_bus_probe_out)
        if(NOT _debug_bus_probe_rc EQUAL 0 AND
           (_debug_bus_probe_out MATCHES "get_debug_bus_local_addr_shift" OR
            _debug_bus_probe_out MATCHES "get_debug_bus_loop_steplen"))
          set(ASC_TENSOR_API_DEBUG_BUS_SHIM TRUE)
          message(STATUS
                  "CANN debug-bus API lacks get_debug_bus_local_addr_shift / "
                  "get_debug_bus_loop_steplen; enabling the tensor_api debug-bus compat shim")
        endif()

        # ops-tensor 的 c_api 头（asc_simd.h -> reg_compute、constant_impl.h）无条件使用
        # bf16/fp8/fp4 向量内建类型和 int4x2_t。ASC 前端会把同一 TU 编译两遍（一遍带
        # __NPU_ARCH__、一遍不带），不同 CANN 表现不同：
        #   * CANN 9.1.0：声明被 (__NPU_ARCH__ == 3510) 门控，"无 arch"那一遍看不到它们，
        #     报 unknown type name 'vector_bf16' / 'int4x2_t'，需要补别名；
        #   * CANN 9.2.0：无条件声明，无需处理（强行补别名会 typedef redefinition）。
        # 头文件文本结构不足以区分，同样用编译探测，并且只在真的报出这两个类型名时才启用。
        set(_asc_arch_types_probe_src "${CMAKE_BINARY_DIR}/asc_arch_types_probe.cpp")
        file(WRITE "${_asc_arch_types_probe_src}"
"#include \"kernel_operator.h\"\n"
"using AscArchTypesProbeBf16 = vector_bf16;\n"
"using AscArchTypesProbeInt4 = int4x2_t;\n"
"int asc_arch_types_probe() { return 0; }\n")
        execute_process(
          COMMAND "${_asc_probe_compiler}" --npu-arch=${NPU_ARCH} -std=c++17
                  --asc-aicore-lang -fsyntax-only ${_asc_probe_includes}
                  "${_asc_arch_types_probe_src}"
          RESULT_VARIABLE _asc_arch_types_probe_rc
          OUTPUT_VARIABLE _asc_arch_types_probe_out
          ERROR_VARIABLE _asc_arch_types_probe_out)
        if(NOT _asc_arch_types_probe_rc EQUAL 0 AND
           (_asc_arch_types_probe_out MATCHES "unknown type name 'vector_bf16'" OR
            _asc_arch_types_probe_out MATCHES "unknown type name 'int4x2_t'"))
          set(ASC_TENSOR_API_ARCH_TYPES_SHIM TRUE)
          message(STATUS
                  "ASC front-end arch-less pass cannot see the bf16/fp8/fp4 vector builtins; "
                  "enabling the tensor_api arch-less-pass type shim")
        endif()
      else()
        message(WARNING
                "bisheng not found under ${ASCEND_CANN_PACKAGE_PATH}; cannot probe the toolchain "
                "capabilities, leaving the tensor_api compat shims disabled")
      endif()
    endif()

    # arch35 的 tensor_api 算子需 devkit >= 9.1；其他架构不受限
    if("arch35" IN_LIST SOC_ARCH_DIRS AND NOT (ASC_DEVKIT_MAJOR GREATER 9 OR (ASC_DEVKIT_MAJOR EQUAL 9 AND ASC_DEVKIT_MINOR GREATER 0)))
      foreach(entry ${TENSOR_API_OPS})
        string(REPLACE "|" ";" _fields "${entry}")
        list(GET _fields 0 _suffix)
        set(ENABLE_BLAS_${_suffix} FALSE)
      endforeach()
      set(ENABLE_BLAS_TRSM_BLOCKED FALSE)
    endif()
  else()
    foreach(entry ${TENSOR_API_OPS})
      string(REPLACE "|" ";" _fields "${entry}")
      list(GET _fields 0 _suffix)
      set(ENABLE_BLAS_${_suffix} FALSE)
    endforeach()
    set(ENABLE_BLAS_TRSM_BLOCKED FALSE)
    message(WARNING "asc_devkit_version.h not found: ${_header}, tensor_api ops and TRSM_BLOCKED will be skipped")
  endif()

  set(ASC_DEVKIT_MAJOR ${ASC_DEVKIT_MAJOR} PARENT_SCOPE)
  set(ASC_DEVKIT_MINOR ${ASC_DEVKIT_MINOR} PARENT_SCOPE)
  set(ASC_TENSOR_API_DEBUG_BUS_SHIM ${ASC_TENSOR_API_DEBUG_BUS_SHIM} PARENT_SCOPE)
  set(ASC_TENSOR_API_ARCH_TYPES_SHIM ${ASC_TENSOR_API_ARCH_TYPES_SHIM} PARENT_SCOPE)
  set(ENABLE_BLASLT_MXFP8 ${ENABLE_BLASLT_MXFP8} PARENT_SCOPE)
  foreach(entry ${TENSOR_API_OPS})
    string(REPLACE "|" ";" _fields "${entry}")
    list(GET _fields 0 _suffix)
    set(ENABLE_BLAS_${_suffix} ${ENABLE_BLAS_${_suffix}} PARENT_SCOPE)
  endforeach()
  set(ENABLE_BLAS_TRSM_BLOCKED ${ENABLE_BLAS_TRSM_BLOCKED} PARENT_SCOPE)
  set(TENSOR_API_OPS ${TENSOR_API_OPS} PARENT_SCOPE)

  set(_enable_status "")
  foreach(entry ${TENSOR_API_OPS})
    string(REPLACE "|" ";" _fields "${entry}")
    list(GET _fields 0 _suffix)
    string(APPEND _enable_status ", ENABLE_BLAS_${_suffix}=${ENABLE_BLAS_${_suffix}}")
  endforeach()
  string(APPEND _enable_status ", ENABLE_BLAS_TRSM_BLOCKED=${ENABLE_BLAS_TRSM_BLOCKED}")
  message(STATUS "ASC_DEVKIT_MAJOR=${ASC_DEVKIT_MAJOR}, ASC_DEVKIT_MINOR=${ASC_DEVKIT_MINOR}, ENABLE_BLASLT_MXFP8=${ENABLE_BLASLT_MXFP8}${_enable_status}")
endfunction()
