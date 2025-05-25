cmake_minimum_required(VERSION 3.20.0)

project(mlir-tutorial LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(BUILD_DEPS ON)

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

include(AddLLVM)
include(TableGen)



# Point to your submodule build
set(CMAKE_PREFIX_PATH "${PROJECT_SOURCE_DIR}/externals/llvm-project/build/lib/cmake/mlir;${PROJECT_SOURCE_DIR}/externals/llvm-project/build/lib/cmake/llvm")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(AddMLIR)
# Only add the build include directory
include_directories(BEFORE ${CMAKE_SOURCE_DIR}/externals/llvm-project/build/include)
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/externals/llvm-project)
include_directories(${PROJECT_BINARY_DIR})

# message(STATUS "Fetching or-tools...")
# include(FetchContent)
# FetchContent_Declare(
#   or-tools
#   GIT_REPOSITORY https://github.com/google/or-tools.git
#   GIT_TAG        v9.11
# )
# FetchContent_MakeAvailable(or-tools)
# message(STATUS "Done fetching or-tools")

add_subdirectory(tests)
add_subdirectory(tools)
add_subdirectory(lib)
