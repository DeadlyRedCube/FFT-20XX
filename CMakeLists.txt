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
    "fft_20xx_avx_common.h"
    "fft_20xx_bit_reverse_copy.h"
    "fft_20xx_common.h"
    "fft_20xx_complex.h"
    "fft_20xx_math.h"
    "fft_20xx_platform.h"
    "fft_20xx_run.h"
    "fft_20xx_table.h"
    "fft_20xx_run.h"

    "fft_20xx_scalar.cpp"
    "fft_20xx_table.cpp"
    "fft_20xx_select.cpp"
)

target_sources(
  fft-20xx-test

  PRIVATE
    "main.cpp"
)

if ((CMAKE_SYSTEM_PROCESSOR MATCHES "x86")
    OR (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    OR (CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64")
    OR (CMAKE_SYSTEM_PROCESSOR MATCHES "x64")
  )
  target_sources(
      fft-20xx

      PRIVATE
        "fft_20xx_sse2.cpp"
        "fft_20xx_avx.cpp"
        "fft_20xx_fma.cpp"
  )

  if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    set_source_files_properties(fft_20xx_sse2.cpp PROPERTIES COMPILE_OPTIONS "/arch:SSE2")
    set_source_files_properties(fft_20xx_avx.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX")
    set_source_files_properties(fft_20xx_fma.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
  else()
    set_source_files_properties(fft_20xx_avx.cpp PROPERTIES COMPILE_OPTIONS -march=core2)
    set_source_files_properties(fft_20xx_avx.cpp PROPERTIES COMPILE_OPTIONS -march=sandybridge)
    set_source_files_properties(fft_20xx_fma.cpp PROPERTIES COMPILE_OPTIONS -mfma)
  endif()
elseif((CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    OR (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    OR (CMAKE_SYSTEM_PROCESSOR MATCHES "armv7a")
  )

  target_sources(
      fft-20xx

      PRIVATE
        "fft_20xx_neon.cpp"
  )
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm")
  # nothing to do here
else()
  if (CMAKE_SYSTEM_PROCESSOR STREQUAL "")
    message(FATAL_ERROR "CMAKE_SYSTEM_PROCESSOR must be set.")
  else()
    message(FATAL_ERROR "Unrecognized CMAKE_SYSTEM_PROCESSOR value '${CMAKE_SYSTEM_PROCESSOR}'")
  endif()
endif()

target_link_libraries(fft-20xx-test fft-20xx)