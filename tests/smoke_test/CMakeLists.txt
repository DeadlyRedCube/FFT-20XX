cmake_minimum_required (VERSION 4.1.0)

project("smoke_test")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_EXTENSIONS OFF)

add_executable(smoke_test)

target_sources(
  smoke_test

  PRIVATE
    "smoke_test.cpp"
    "../../src/reference/real.cpp"
)

target_link_libraries(smoke_test fft-20xx)
