cmake_minimum_required (VERSION 4.1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_EXTENSIONS OFF)

project ("FFT-20XX")

add_library(fft-20xx)
add_executable(fft-20xx-test)

target_sources(
  fft-20xx
    
  PUBLIC FILE_SET HEADERS FILES
    "include/fft_20xx.h"

  PRIVATE
    "fft_20xx_bit_reverse_copy.h"
    "fft_20xx_common.h"
    "fft_20xx_complex.h"
    "fft_20xx_math.h"
    "fft_20xx_platform.h"
    "fft_20xx_run.h"
    "fft_20xx_table.h"

    "fft_20xx_scalar.cpp" 
    "fft_20xx_table.cpp"
    "fft_20xx_select.cpp"
)

target_sources(
  fft-20xx-test
    
  PRIVATE
    "main.cpp" 
)

target_link_libraries(fft-20xx-test fft-20xx)