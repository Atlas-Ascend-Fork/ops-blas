# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# issue #661: 测试目标不得依赖 build.sh 专有的环境变量（LINUX_INCLUDE_PATH/EAGER_LIBRARY_PATH），
# 否则绕过 build.sh 直连 cmake 时链接输入退化为 "/libascendcl.so"。libascendcl 优先从
# ASCEND_CANN_PACKAGE_PATH 缓存变量解析（build.sh 场景两者指向同一目录，行为不变），环境变量作回退。
set(BLAS_ASCENDCL_LIB "${ASCEND_CANN_PACKAGE_PATH}/lib64/libascendcl.so")
if(NOT EXISTS "${BLAS_ASCENDCL_LIB}" AND EXISTS "$ENV{EAGER_LIBRARY_PATH}/libascendcl.so")
    set(BLAS_ASCENDCL_LIB "$ENV{EAGER_LIBRARY_PATH}/libascendcl.so")
endif()

# 检查 blas/<op_name>/ 下是否有当前 SOC 可编译的算子实现
function(_ops_blas_has_blas_op_sources op_name out_var)
    set(has_sources FALSE)

    # 列表驱动：遍历 TENSOR_API_OPS，低版本时跳过对应算子的实现与测试
    foreach(entry ${TENSOR_API_OPS})
        string(REPLACE "|" ";" _fields "${entry}")
        list(GET _fields 0 _var_suffix)
        list(GET _fields 2 _test_names)
        if(NOT ENABLE_BLAS_${_var_suffix})
            string(REPLACE "," ";" _test_name_list "${_test_names}")
            foreach(_tn ${_test_name_list})
                if(op_name STREQUAL "${_tn}")
                    set(${out_var} FALSE PARENT_SCOPE)
                    return()
                endif()
            endforeach()
        endif()
    endforeach()

    foreach(arch_dir ${SOC_ARCH_DIRS})
        file(GLOB arch_dir_srcs ${CMAKE_SOURCE_DIR}/blas/${op_name}/${arch_dir}/*.cpp
                                  ${CMAKE_SOURCE_DIR}/blas/*/${op_name}/${arch_dir}/*.cpp
                                  ${CMAKE_SOURCE_DIR}/blas/*/${arch_dir}/${op_name}_*.cpp
                                  ${CMAKE_SOURCE_DIR}/extensions/*/${op_name}/${arch_dir}/*.cpp
                                  ${CMAKE_SOURCE_DIR}/extensions/${op_name}/${arch_dir}/*.cpp)
        if(arch_dir_srcs)
            set(has_sources TRUE)
            break()
        endif()
    endforeach()
    # Flat directory: blas/{dir}/archXX/ without type subdirectory
    if(NOT has_sources AND op_name MATCHES "^[a-zA-Z]")
        string(SUBSTRING "${op_name}" 1 -1 _stripped)
        foreach(arch_dir ${SOC_ARCH_DIRS})
            file(GLOB arch_dir_srcs ${CMAKE_SOURCE_DIR}/blas/${_stripped}/${arch_dir}/*.cpp
                                     ${CMAKE_SOURCE_DIR}/extensions/${_stripped}/${arch_dir}/*.cpp)
            if(arch_dir_srcs)
                set(has_sources TRUE)
                break()
            endif()
        endforeach()
    endif()
    if(NOT has_sources)
        file(GLOB base_srcs ${CMAKE_SOURCE_DIR}/blas/${op_name}/*.cpp
                              ${CMAKE_SOURCE_DIR}/blas/*/${op_name}/*.cpp
                              ${CMAKE_SOURCE_DIR}/extensions/${op_name}/*.cpp
                              ${CMAKE_SOURCE_DIR}/extensions/*/${op_name}/*.cpp)
        if(base_srcs)
            set(has_sources TRUE)
        endif()
    endif()
    set(${out_var} ${has_sources} PARENT_SCOPE)
endfunction()

# 检查 blasLt/ 下是否有当前 SOC 可编译的算子实现。
# 测试名如 blasLtMatmul 会匹配 blasLt 下名称包含 matmul 的子目录（如 matmul_fp32、matmul_mxfp8）。
function(_ops_blas_has_blaslt_op_sources test_name out_var)
    set(has_sources FALSE)
    if(NOT test_name MATCHES "^blasLt")
        set(${out_var} FALSE PARENT_SCOPE)
        return()
    endif()

    string(REGEX REPLACE "^blasLt" "" _op_suffix ${test_name})
    string(TOLOWER "${_op_suffix}" _op_suffix_lower)

    file(GLOB children RELATIVE ${CMAKE_SOURCE_DIR}/blasLt ${CMAKE_SOURCE_DIR}/blasLt/*)
    foreach(child ${children})
        if(NOT IS_DIRECTORY ${CMAKE_SOURCE_DIR}/blasLt/${child})
            continue()
        endif()
        if(child STREQUAL "include" OR child STREQUAL "utils" OR child STREQUAL "internal" OR
           child STREQUAL "api" OR child STREQUAL "common")
            continue()
        endif()
        # Match the operator family ignoring underscores so directory names like "matrix_transform"
        # still match the test suffix "matrixtransform".
        string(TOLOWER "${child}" _child_lower)
        string(REPLACE "_" "" _child_norm "${_child_lower}")
        string(REPLACE "_" "" _op_suffix_norm "${_op_suffix_lower}")
        if(NOT _child_norm MATCHES "${_op_suffix_norm}")
            continue()
        endif()

        file(GLOB_RECURSE dir_srcs ${CMAKE_SOURCE_DIR}/blasLt/${child}/*.cpp)
        foreach(src_file ${dir_srcs})
            set(is_arch_specific FALSE)
            foreach(arch_dir ${ARCH_SPECIFIC_DIRS})
                if(src_file MATCHES "/${arch_dir}/")
                    set(is_arch_specific TRUE)
                    break()
                endif()
            endforeach()
            if(NOT is_arch_specific)
                set(has_sources TRUE)
                break()
            endif()
        endforeach()
        if(has_sources)
            break()
        endif()

        foreach(arch_dir ${SOC_ARCH_DIRS})
            file(GLOB arch_dir_srcs ${CMAKE_SOURCE_DIR}/blasLt/${child}/${arch_dir}/*.cpp)
            if(arch_dir_srcs)
                if((child STREQUAL "matmul_mxfp8" OR child STREQUAL "matmul_mxfp4") AND NOT ENABLE_BLASLT_MXFP8)
                    continue()
                endif()
                set(has_sources TRUE)
                break()
            endif()
        endforeach()
        if(has_sources)
            break()
        endif()
    endforeach()

    set(${out_var} ${has_sources} PARENT_SCOPE)
endfunction()

# 检查算子在当前 SOC 下是否有可编译实现（同时覆盖 blas/ 与 blasLt/ 两种目录布局）
function(ops_blas_has_op_sources_for_soc test_name out_var)
    _ops_blas_has_blas_op_sources(${test_name} _blas_has)
    if(_blas_has)
        set(${out_var} TRUE PARENT_SCOPE)
        return()
    endif()
    _ops_blas_has_blaslt_op_sources(${test_name} _blaslt_has)
    set(${out_var} ${_blaslt_has} PARENT_SCOPE)
endfunction()

# 为指定测试目标收集源文件：根目录 ${target}.cpp + 当前 SOC 架构目录下的同名/同前缀补充源
function(ops_blas_get_test_target_sources target out_var)
    set(sources "")
    set(root_src ${CMAKE_CURRENT_SOURCE_DIR}/${target}.cpp)

    # Check if any arch directory has a replacement for the root test
    set(_has_arch_test FALSE)
    foreach(arch_dir ${SOC_ARCH_DIRS})
        set(arch_src ${CMAKE_CURRENT_SOURCE_DIR}/${arch_dir}/${target}.cpp)
        if(EXISTS ${arch_src})
            list(APPEND sources ${arch_src})
            set(_has_arch_test TRUE)
        endif()
        file(GLOB arch_supp_srcs ${CMAKE_CURRENT_SOURCE_DIR}/${arch_dir}/${target}_*.cpp)
        list(APPEND sources ${arch_supp_srcs})
    endforeach()

    # Only include root test if no arch-specific replacement exists.
    # Root tests may use old API conventions (e.g. host pointers) that are
    # incompatible with arch-specific host implementations (e.g. device pointers).
    if(NOT _has_arch_test AND EXISTS ${root_src})
        list(APPEND sources ${root_src})
    endif()

    if(NOT sources)
        message(FATAL_ERROR "No test sources found for target '${target}' in ${CMAKE_CURRENT_SOURCE_DIR}")
    endif()

    set(${out_var} ${sources} PARENT_SCOPE)
endfunction()

function(_ops_blas_register_test_target target link_lib)
    _ops_blas_ensure_refblas_found()
    ops_blas_get_test_target_sources(${target} _test_srcs)
    add_executable(${target} ${_test_srcs})

    if(DEFINED TEST_DEVICE_ID)
        target_compile_definitions(${target} PRIVATE TEST_DEVICE_ID=${TEST_DEVICE_ID})
    endif()

    target_include_directories(${target} PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/test/frame
        ${CMAKE_SOURCE_DIR}/test/utils
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/blas/common/helper
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/op_common/
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/base/
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/
        $ENV{LINUX_INCLUDE_PATH}
        ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include
        ${REFBLAS_INCLUDE_DIR}
    )
    target_compile_features(${target} PRIVATE cxx_std_17)
    target_link_libraries(${target} PRIVATE
        ${link_lib}
        ${BLAS_ASCENDCL_LIB}
        ${REFBLAS_LIB}
        ${REFLAPACK_LIB}
    )

    _ops_blas_copy_test_config_files(${target})
endfunction()

# Locate GTest once per configure (CSV-driven ST uses custom main, not gtest_main).
function(_ops_blas_ensure_gtest_found)
    if(NOT GTEST_LIB OR NOT GTEST_INCLUDE_DIR)
        find_path(GTEST_INCLUDE_DIR gtest/gtest.h PATHS /usr/local/include /usr/include)
        find_library(GTEST_LIB gtest PATHS /usr/local/lib /usr/lib)
    endif()
endfunction()

# Locate reference BLAS (CBLAS) and LAPACK once per configure.
function(_ops_blas_ensure_refblas_found)
    find_path(REFBLAS_INCLUDE_DIR cblas.h
        PATHS /usr/local/include /usr/include /usr/include/x86_64-linux-gnu
              ${HOMEBREW_PREFIX}/include
        PATH_SUFFIXES openblas)
    find_library(REFBLAS_LIB NAMES openblas blas
        PATHS /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu
              ${HOMEBREW_PREFIX}/lib
        PATH_SUFFIXES blas)
    find_library(REFLAPACK_LIB NAMES lapack
        PATHS /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu
              ${HOMEBREW_PREFIX}/lib
        PATH_SUFFIXES lapack)
    if(NOT REFBLAS_INCLUDE_DIR OR NOT REFBLAS_LIB)
        message(FATAL_ERROR "Reference BLAS (cblas.h / libblas) not found. Install via:\n"
            "  Debian/Ubuntu:         apt-get install libblas-dev\n"
            "  RHEL/CentOS/openEuler: dnf install openblas-devel (or yum install openblas-devel)\n"
            "  macOS:                 brew install openblas\n"
            "Or run ./install_deps.sh to install dependencies automatically.")
    endif()
    if(NOT REFLAPACK_LIB)
        message(WARNING "Reference LAPACK (liblapack) not found. "
            "LAPACK-based golden tests will fail to link.")
    endif()
endfunction()

# Copy CSV/JSON from test root and from current-SOC arch subdirs (e.g. arch35/).
function(_ops_blas_copy_test_config_files target)
    set(_cfg_files "")
    file(GLOB _root_json "${CMAKE_CURRENT_SOURCE_DIR}/*.json")
    file(GLOB _root_csv "${CMAKE_CURRENT_SOURCE_DIR}/*.csv")
    list(APPEND _cfg_files ${_root_json} ${_root_csv})

    foreach(arch_dir ${SOC_ARCH_DIRS})
        file(GLOB _arch_json "${CMAKE_CURRENT_SOURCE_DIR}/${arch_dir}/*.json")
        file(GLOB _arch_csv "${CMAKE_CURRENT_SOURCE_DIR}/${arch_dir}/*.csv")
        list(APPEND _cfg_files ${_arch_json} ${_arch_csv})
    endforeach()

    # Legacy single-file JSON under build/json_configs/
    set(_json_src "${CMAKE_BINARY_DIR}/json_configs/${target}_testcases.json")
    set(_json_src_alt "${CMAKE_SOURCE_DIR}/build/test/json_configs/${target}_testcases.json")
    if(EXISTS ${_json_src})
        list(APPEND _cfg_files ${_json_src})
    elseif(EXISTS ${_json_src_alt})
        list(APPEND _cfg_files ${_json_src_alt})
    endif()

    foreach(_cfg_file ${_cfg_files})
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy ${_cfg_file}
                    $<TARGET_FILE_DIR:${target}>)
    endforeach()
endfunction()

# Register GTest-based CSV/JSON ST target (custom main; links gtest only, not gtest_main).
function(_ops_blas_register_gtest_target target link_lib)
    _ops_blas_ensure_gtest_found()
    _ops_blas_ensure_refblas_found()
    ops_blas_get_test_target_sources(${target} _test_srcs)
    list(APPEND _test_srcs ${CMAKE_SOURCE_DIR}/test/frame/test_main.cpp)
    add_executable(${target} ${_test_srcs})
    # 记录 gtest 目标清单：合并测试 Runner（blas_all_test，issue #457）据此聚合全部源
    set_property(GLOBAL APPEND PROPERTY OPS_BLAS_GTEST_TARGETS ${target})

    if(DEFINED TEST_DEVICE_ID)
        target_compile_definitions(${target} PRIVATE TEST_DEVICE_ID=${TEST_DEVICE_ID})
    endif()

    set(_extra_includes "")
    foreach(arch_dir ${SOC_ARCH_DIRS})
        if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${arch_dir}")
            list(APPEND _extra_includes "${CMAKE_CURRENT_SOURCE_DIR}/${arch_dir}")
        endif()
    endforeach()

    target_include_directories(${target} PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/test/frame
        ${CMAKE_SOURCE_DIR}/test/utils
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/blas/common/helper
        ${_extra_includes}
        $ENV{LINUX_INCLUDE_PATH}
        ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include
        ${GTEST_INCLUDE_DIR}
        ${REFBLAS_INCLUDE_DIR}
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/op_common/
        ${ASCEND_CANN_PACKAGE_PATH}/include/op_common/
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/base/
    )
    target_compile_features(${target} PRIVATE cxx_std_17)
    target_link_directories(${target} PRIVATE "${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/lib64")
    target_link_libraries(${target} PRIVATE
        ${link_lib}
        ${BLAS_ASCENDCL_LIB}
        ${GTEST_LIB}
        ${REFBLAS_LIB}
        ${REFLAPACK_LIB}
        pthread
        tiling_api
        platform
        register
        c_sec
    )

    _ops_blas_copy_test_config_files(${target})
endfunction()

function(_ops_blas_discover_test_targets out_var)
    set(targets "")

    file(GLOB root_srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
    foreach(src ${root_srcs})
        get_filename_component(target ${src} NAME_WE)
        list(APPEND targets ${target})
    endforeach()

    foreach(arch_dir ${SOC_ARCH_DIRS})
        file(GLOB arch_srcs ${CMAKE_CURRENT_SOURCE_DIR}/${arch_dir}/*.cpp)
        foreach(src ${arch_srcs})
            get_filename_component(target ${src} NAME_WE)
            if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${target}.cpp)
                continue()
            endif()
            list(APPEND targets ${target})
        endforeach()
    endforeach()

    if(targets)
        list(REMOVE_DUPLICATES targets)
    endif()
    set(${out_var} ${targets} PARENT_SCOPE)
endfunction()

# 注册测试可执行文件。ARGN 为空时自动发现本目录 target；否则仅注册指定 target 列表。
function(ops_blas_add_tests link_lib)
    if(ARGN)
        set(targets ${ARGN})
    else()
        _ops_blas_discover_test_targets(targets)
    endif()

    if(NOT targets)
        message(FATAL_ERROR "No test targets to register in ${CMAKE_CURRENT_SOURCE_DIR}")
    endif()

    foreach(target ${targets})
        if(TARGET ${target})
            continue()
        endif()
        _ops_blas_register_test_target(${target} ${link_lib})
    endforeach()
endfunction()

# Register GTest CSV/JSON ST targets. ARGN empty -> auto-discover *_test.cpp in root/arch dirs.
function(ops_blas_add_gtest_tests link_lib)
    if(ARGN)
        set(targets ${ARGN})
    else()
        _ops_blas_discover_test_targets(targets)
    endif()

    if(NOT targets)
        message(FATAL_ERROR "No GTest test targets to register in ${CMAKE_CURRENT_SOURCE_DIR}")
    endif()

    foreach(target ${targets})
        if(TARGET ${target})
            continue()
        endif()
        _ops_blas_register_gtest_target(${target} ${link_lib})
    endforeach()
endfunction()

# ----------------------------------------------------------------------------------------------------------
# 合并测试 Runner（blas_all_test，issue #457 全量耗时优化）。
# 把全部 *_test 目标的测试源聚合编译进单一可执行文件：
#   - aclInit/SetDevice/aclblasCreate 等 suite 级初始化由 blas_test.h 的引用计数守卫
#     在 suite 间复用，80 次 aclInit 降为 1 次（每 suite 约 3s，共省约 4 分钟）；
#   - 跨文件重名全局符号（如 16 个 *_test.cpp 里的 VerifyResult）由
#     amalgamate_conflicts.py 生成清单、localize_launcher.sh 对 .o 做 objcopy
#     --localize-symbols 降级为局部符号，规避 multiple definition；
#   - BLAS_TEST_CASES 里 op:ALL / op:case1,case2 的算子选择由 test_main.cpp
#     按可执行名（blas_all_test 无 _test 后缀 → 走 ALL 匹配分支）与 suite 前缀
#     映射文件（all_runner_suites.map）共同支持。
# 独立目标仍按原样构建：blas_all_test 仅在 --run-blas-all 时由 build.sh 使用，
# 以及 per-op 目标构建/链接失败时的回退路径。
# ----------------------------------------------------------------------------------------------------------
function(_ops_blas_register_all_runner link_lib)
    get_property(_gtest_targets GLOBAL PROPERTY OPS_BLAS_GTEST_TARGETS)
    if(NOT _gtest_targets)
        return()
    endif()

    _ops_blas_ensure_gtest_found()
    _ops_blas_ensure_refblas_found()

    # 1) 聚合源：每个目标的测试源（不含各自的 test_main.cpp，只保留一份 main）
    set(_all_srcs "")
    set(_all_dirs "")
    foreach(_tgt ${_gtest_targets})
        get_target_property(_srcs ${_tgt} SOURCES)
        foreach(_src ${_srcs})
            if(_src MATCHES "test_main\\.cpp$")
                continue()
            endif()
            list(APPEND _all_srcs ${_src})
            get_filename_component(_src_dir "${_src}" DIRECTORY)
            if(_src_dir)
                list(APPEND _all_dirs "${_src_dir}")
            endif()
        endforeach()
    endforeach()
    list(REMOVE_DUPLICATES _all_dirs)
    list(APPEND _all_srcs ${CMAKE_SOURCE_DIR}/test/frame/test_main.cpp)

    # 2) 冲突符号清单（configure 期生成，编译 launcher 消费）。
    # 输出文件由脚本 -o 参数显式写入（VERBATIM 下 shell 重定向会被转义失效）
    set(_conflict_list "${CMAKE_CURRENT_BINARY_DIR}/all_runner_conflict_symbols.txt")
    add_custom_command(
        OUTPUT "${_conflict_list}"
        COMMAND python3 ${CMAKE_SOURCE_DIR}/test/frame/amalgamate_conflicts.py
                -o "${_conflict_list}" ${_all_srcs}
        DEPENDS ${_all_srcs} ${CMAKE_SOURCE_DIR}/test/frame/amalgamate_conflicts.py
        COMMENT "amalgamate: detecting cross-file global symbol conflicts"
        VERBATIM)
    add_custom_target(blas_all_conflict_list DEPENDS "${_conflict_list}")

    # 3) suite 前缀映射：op -> INSTANTIATE_TEST_SUITE_P 前缀（test_main.cpp ALL 匹配用）
    set(_suite_map "${CMAKE_CURRENT_BINARY_DIR}/all_runner_suites.map")
    set(_map_content "")
    foreach(_tgt ${_gtest_targets})
        set(_prefix "")
        # 从目标源文件提取 INSTANTIATE 前缀（每个 gtest 目标恰好一个）。
        # 书写形态含单行与多行两种：INSTANTIATE_TEST_SUITE_P(\n    Prefix, Class,
        # 用 python 统一处理（CMake REGEX 对多行匹配支持有限）
        get_target_property(_srcs ${_tgt} SOURCES)
        foreach(_src ${_srcs})
            if(NOT _src MATCHES "\\.cpp$")
                continue()
            endif()
            execute_process(
                COMMAND python3 -c
                        "import re,sys;src=open(sys.argv[1],encoding='utf-8',errors='ignore').read();m=re.search(r'INSTANTIATE_TEST_SUITE_P\\(\\s*([A-Za-z0-9_]+)',src);print(m.group(1) if m else '')"
                        "${_src}"
                OUTPUT_VARIABLE _prefix
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
                RESULT_VARIABLE _prefix_rc)
            if(_prefix_rc EQUAL 0 AND _prefix)
                break()
            endif()
        endforeach()
            if(_prefix)
                # 键写两份：完整 target 名（ccopy_test）与去掉 _test 后缀的算子名
                # （ccopy，BLAS_TEST_CASES 里 run_example.sh 使用的是算子名）
                string(APPEND _map_content "${_tgt}:${_prefix}\n")
                string(REGEX REPLACE "_test$" "" _op_key "${_tgt}")
                if(NOT _op_key STREQUAL "${_tgt}")
                    string(APPEND _map_content "${_op_key}:${_prefix}\n")
                endif()
            else()
                message(WARNING "amalgamate: no INSTANTIATE_TEST_SUITE_P prefix found for '${_tgt}', "
                    "BLAS_TEST_CASES op-level selection will not cover it in blas_all_test")
            endif()
        endforeach()
        file(WRITE "${_suite_map}" "${_map_content}")

    # 4) 编译 launcher：对 .o 做冲突符号本地化（仅本目标）。
    # 以 /bin/bash 前缀执行脚本，不依赖 configure_file 产物的执行权限位。
    set(_launcher "${CMAKE_CURRENT_BINARY_DIR}/localize_launcher_used.sh")
    configure_file(${CMAKE_SOURCE_DIR}/test/frame/localize_launcher.sh
                   "${_launcher}" COPYONLY)

    add_executable(blas_all_test ${_all_srcs})
    add_dependencies(blas_all_test blas_all_conflict_list)
    set_target_properties(blas_all_test PROPERTIES
        CXX_COMPILER_LAUNCHER "/bin/bash;${_launcher};${_conflict_list}")

    if(DEFINED TEST_DEVICE_ID)
        target_compile_definitions(blas_all_test PRIVATE TEST_DEVICE_ID=${TEST_DEVICE_ID})
    endif()

    # 5) include 目录并集：每个源所在目录 + 各目标已注册的 include 路径
    set(_all_includes "")
    foreach(_tgt ${_gtest_targets})
        get_target_property(_inc ${_tgt} INCLUDE_DIRECTORIES)
        if(_inc)
            list(APPEND _all_includes ${_inc})
        endif()
    endforeach()
    foreach(_dir ${_all_dirs})
        list(APPEND _all_includes ${_dir})
        foreach(arch_dir ${SOC_ARCH_DIRS})
            if(IS_DIRECTORY "${_dir}/${arch_dir}")
                list(APPEND _all_includes "${_dir}/${arch_dir}")
            endif()
        endforeach()
    endforeach()
    list(REMOVE_DUPLICATES _all_includes)
    target_include_directories(blas_all_test PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/test/frame
        ${CMAKE_SOURCE_DIR}/test/utils
        ${CMAKE_SOURCE_DIR}/blas/common/helper
        ${_all_includes}
        $ENV{LINUX_INCLUDE_PATH}
        ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include
        ${GTEST_INCLUDE_DIR}
        ${REFBLAS_INCLUDE_DIR}
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/op_common/
        ${ASCEND_CANN_PACKAGE_PATH}/include/op_common/
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/base/
    )
    # 6) 聚合 per-target 编译定义（如 ssymm 的 SSYMM_ARCH35=1）。
    # 宏名按算子前缀命名（SSYMM_/CCOPY_ 等），跨算子无同名冲突；含生成器
    # 表达式（$<...>）或与 TEST_DEVICE_ID 相关的条目跳过。
    set(_all_defs "")
    foreach(_tgt ${_gtest_targets})
        get_target_property(_defs ${_tgt} COMPILE_DEFINITIONS)
        if(_defs)
            foreach(_def ${_defs})
                if(_def MATCHES "^\\$<" OR _def MATCHES "^TEST_DEVICE_ID" OR _def STREQUAL "")
                    continue()
                endif()
                list(APPEND _all_defs "${_def}")
            endforeach()
        endif()
    endforeach()
    list(REMOVE_DUPLICATES _all_defs)
    if(_all_defs)
        target_compile_definitions(blas_all_test PRIVATE ${_all_defs})
    endif()

    target_compile_features(blas_all_test PRIVATE cxx_std_17)
    target_link_directories(blas_all_test PRIVATE "${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/lib64")
    target_link_libraries(blas_all_test PRIVATE
        ${link_lib}
        ${BLAS_ASCENDCL_LIB}
        ${GTEST_LIB}
        ${REFBLAS_LIB}
        ${REFLAPACK_LIB}
        pthread
        tiling_api
        platform
        register
        c_sec
        unified_dlog
    )

    # 6) suite 映射与 CSV/JSON 一起拷到可执行目录，test_main.cpp 运行时读取
    add_custom_command(TARGET blas_all_test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${_suite_map}" $<TARGET_FILE_DIR:blas_all_test>/all_runner_suites.map
        COMMENT "amalgamate: installing all_runner_suites.map")
    _ops_blas_copy_test_config_files(blas_all_test)
endfunction()
