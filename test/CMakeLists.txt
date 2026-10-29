if(BUILD_TEST)
    include(${CMAKE_SOURCE_DIR}/cmake/test.cmake)

    set(_OPS_BLAS_BUILT_TESTS "")
    set(_OPS_BLAS_ADDED_DIRS "")

    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/skipped_tests.list "")

    # 支持多算子列表（TEST_NAMES 是分号分隔的 CMake 列表）
    foreach(TEST_NAME ${TEST_NAMES})
        # 解析测试目录路径：支持直接目录 (test/scopy/) 和家族嵌套 (test/gbmv/sgbmv/)
        set(TEST_DIR "")
        if(TEST_NAME STREQUAL "chemm" AND "arch35" IN_LIST SOC_ARCH_DIRS)
            set(TEST_DIR "symm/chemm")
        elseif(TEST_NAME STREQUAL "cgemm_strided_batched" AND "arch35" IN_LIST SOC_ARCH_DIRS)
            set(TEST_DIR "gemm_strided_batched/cgemm_strided_batched")
        elseif(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${TEST_NAME})
            set(TEST_DIR "${TEST_NAME}")
        else()
            # 搜索家族子目录
            file(GLOB family_dirs RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
            foreach(family_dir ${family_dirs})
                if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${family_dir}
                   AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${family_dir}/${TEST_NAME})
                    set(TEST_DIR "${family_dir}/${TEST_NAME}")
                    break()
                endif()
            endforeach()
        endif()

        # Flat directory without type prefix
        if("${TEST_DIR}" STREQUAL "" AND TEST_NAME MATCHES "^[a-zA-Z]")
            string(SUBSTRING "${TEST_NAME}" 1 -1 _stripped)
            if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${_stripped})
                set(TEST_DIR "${_stripped}")
            endif()
        endif()

        if("${TEST_DIR}" STREQUAL "")
            message(FATAL_ERROR
                "Test directory 'test/${TEST_NAME}' does not exist (checked flat and family layouts). "
                "Please check the operator name specified in --ops or TEST_NAMES.")
        endif()

        # 检查当前 SOC 架构下是否有算子实现源码（blas/ 或 blasLt/ 目录布局）
        ops_blas_has_op_sources_for_soc(${TEST_NAME} TEST_HAS_OP_SOURCES)

        # 检查当前 SOC 架构下是否有可编译的测试源码
        set(TEST_HAS_TEST_SOURCES FALSE)
        set(TEST_CPP "${TEST_NAME}_test.cpp")
        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${TEST_DIR}/${TEST_CPP})
            set(TEST_HAS_TEST_SOURCES TRUE)
        endif()
        foreach(arch_dir ${SOC_ARCH_DIRS})
            if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${TEST_DIR}/${arch_dir}/${TEST_CPP})
                set(TEST_HAS_TEST_SOURCES TRUE)
                break()
            endif()
            file(GLOB arch_test_supp_srcs
                ${CMAKE_CURRENT_SOURCE_DIR}/${TEST_DIR}/${arch_dir}/${TEST_NAME}_test_*.cpp)
            if(arch_test_supp_srcs)
                set(TEST_HAS_TEST_SOURCES TRUE)
                break()
            endif()
        endforeach()

        if(NOT TEST_HAS_OP_SOURCES OR NOT TEST_HAS_TEST_SOURCES)
            set(SKIP_REASONS "")
            if(NOT TEST_HAS_OP_SOURCES)
                # 列表驱动：遍历 TENSOR_API_OPS，低版本时给出对应的 skip reason
                set(_matched_tensor_op FALSE)
                foreach(entry ${TENSOR_API_OPS})
                    string(REPLACE "|" ";" _fields "${entry}")
                    list(GET _fields 0 _var_suffix)
                    list(GET _fields 2 _test_names)
                    if(NOT ENABLE_BLAS_${_var_suffix})
                        string(REPLACE "," ";" _test_name_list "${_test_names}")
                        foreach(_tn ${_test_name_list})
                            if(TEST_NAME STREQUAL "${_tn}")
                                list(APPEND SKIP_REASONS
                                    "requires asc-devkit >= 9.1, current ${ASC_DEVKIT_MAJOR}.${ASC_DEVKIT_MINOR}")
                                set(_matched_tensor_op TRUE)
                                break()
                            endif()
                        endforeach()
                    endif()
                    if(_matched_tensor_op)
                        break()
                    endif()
                endforeach()
                if(NOT _matched_tensor_op)
                    list(APPEND SKIP_REASONS
                        "operator implementation not found for '${SOC_VERSION}' "
                        "(checked blas/${TEST_NAME}/ and blasLt/*/)")
                endif()
            endif()
            if(NOT TEST_HAS_TEST_SOURCES)
                list(APPEND SKIP_REASONS
                    "test directory 'test/${TEST_DIR}/' contains no test sources for '${SOC_VERSION}'")
            endif()
            string(JOIN "; " SKIP_REASONS_STR ${SKIP_REASONS})
            if(TEST_NAME MATCHES "^blasLt")
                set(ACLBLAS_OP_NAME "acl${TEST_NAME}")
            else()
                string(SUBSTRING "${TEST_NAME}" 0 1 _api_first)
                string(TOUPPER "${_api_first}" _api_first)
                string(SUBSTRING "${TEST_NAME}" 1 -1 _api_rest)
                set(ACLBLAS_OP_NAME "aclblas${_api_first}${_api_rest}")
            endif()
            message(STATUS
                "Skipping test '${TEST_NAME}': ${SKIP_REASONS_STR}. "
                "Please verify that '${ACLBLAS_OP_NAME}' supports '${SOC_VERSION}'.")
            string(REPLACE ";" "," SKIP_REASONS_SAFE "${SKIP_REASONS_STR}")
            string(REPLACE "\n" " " SKIP_REASONS_SAFE "${SKIP_REASONS_SAFE}")
            file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/skipped_tests.list
                "${TEST_NAME}|${SKIP_REASONS_SAFE}\n")
            continue()
        endif()

        list(FIND _OPS_BLAS_ADDED_DIRS "${TEST_DIR}" _added_idx)
        if(_added_idx EQUAL -1)
            list(APPEND _OPS_BLAS_ADDED_DIRS "${TEST_DIR}")
            list(APPEND _OPS_BLAS_BUILT_TESTS "${TEST_NAME}")
            add_subdirectory(${TEST_DIR})
        endif()
    endforeach()

    # 生成测试清单，供 build.sh --run 区分 skip / fail
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/built_tests.list "")
    foreach(test_name ${_OPS_BLAS_BUILT_TESTS})
        file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/built_tests.list "${test_name}\n")
    endforeach()

    # 合并测试 Runner（issue #457 全量耗时优化）：聚合全部 gtest 目标的测试源
    # 为单一 blas_all_test 可执行，suite 初始化（aclInit 等）在 suite 间复用，
    # 80 次 aclInit 降为 1 次。独立 per-op 目标仍按原样构建（回退与单算子场景用）。
    _ops_blas_register_all_runner(${OPS_BLAS})
    if("arch22" IN_LIST SOC_ARCH_DIRS AND TARGET cgemm_strided_batched_test AND TARGET blas_all_test)
        # The CGEMM Netlib reference loader is also compiled into the combined runner.
        target_link_libraries(blas_all_test PRIVATE ${CMAKE_DL_LIBS})
    endif()
endif()
