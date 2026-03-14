include(${CMAKE_CURRENT_LIST_DIR}/toolchain-gcc-linux-base.cmake)

set(CMAKE_SYSTEM_PROCESSOR arm64)

set(triple aarch64-linux-gnu)
set(CMAKE_C_COMPILER_TARGET ${triple})
set(CMAKE_CXX_COMPILER_TARGET ${triple})