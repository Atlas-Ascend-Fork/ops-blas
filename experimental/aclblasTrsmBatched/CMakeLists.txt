cmake_minimum_required(VERSION 3.20.0)
find_package(ASC REQUIRED)

project(trsm_batched LANGUAGES ASC CXX)
set(CMAKE_CXX_STANDARD 17)

set(ACL_INCLUDE_DIR "$ENV{ASCEND_HOME_PATH}/aarch64-linux/include")
set(ACL_LIB_DIR "$ENV{ASCEND_HOME_PATH}/lib64")

add_subdirectory(test)
