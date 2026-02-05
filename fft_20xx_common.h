#pragma once

#include <algorithm>
#include <cinttypes>
#include <cmath>

#include "include/fft_20xx.h"

#include "fft_20xx_complex.h"
#include "fft_20xx_table.h"

#ifdef __clang__
  #define COMPILER_CLANG 1
#else
  #define COMPILER_CLANG 0
#endif

#if defined(_MSC_VER) && !defined(__clang__)
  #define COMPILER_MSVC 1
#else
  #define COMPILER_MSVC 0
#endif

#if defined (__GNU__) && !defined(__clang__) && !defined(__llvm__) && !defined(__INTEL_COMPILER)
  #define COMPILER_GCC 1
#else
  #define COMPILER_GCC 0
#endif

#if defined(_M_ARM64) || defined(__aarch64__)
  #define CPU_ARM_ANY 1
  #define CPU_ARM64 1
  #define CPU_ARM32 0
#elif defined(_M_ARM) || defined (__arm__)
  #define CPU_ARM_ANY 1
  #define CPU_ARM64 0
  #define CPU_ARM32 1
#else
  #define CPU_ARM 0
  #define CPU_ARM64 0
  #define CPU_ARM32 0
#endif

#if defined(_M_X64) || (__x86_64__)
  #define CPU_X86_ANY 1
  #define CPU_X64 1
  #define CPU_X86 0
#elif defined(_M_IX86) || (__i386__)
  #define CPU_X86_ANY 1
  #define CPU_X64 0
  #define CPU_X86 1
#else
  #define CPU_X86_ANY 0
  #define CPU_X64 0
  #define CPU_X86 0
#endif  

#if COMPILER_MSVC
  #define FORCE_INLINE __forceinline
  #pragma warning (disable: 5045) // Compiler will insert Spectre mitigation for memory load if /Qspectre switch specified
#else
  #define FORCE_INLINE inline
#endif

#include "fft_20xx_complex.h"

namespace FFT20XX
{
  enum class Dir
  {
    Forward,
    Reverse,
  };

  template <typename F>
  static constexpr auto InvSqrt2 = F(0.7071067811865476);

  template <Dir D, typename F>
  auto Rot(Complex<F> v, Complex<F> w) -> Complex<F>
  {
    if constexpr (D == Dir::Forward)
      { return v * w; }
    else
      {  } // !!! multiply with conjugate (as a single operation), I think
  }

  template <Dir D, typename F>
  auto Rot45(Complex<F> v) -> Complex<F>
  {
    if constexpr (D == Dir::Forward)
      { return { .r = InvSqrt2<F> * (v.r + v.i), .i = InvSqrt2<F> * (v.i - v.r) }; }
    else
      { } // !!!
  }

  template <Dir D, typename F>
  auto Rot90(Complex<F> v) -> Complex<F>
  {
    if constexpr (D == Dir::Forward)
      { return v * -II; }
    else
      { return v * II; }
  }

  template <Dir D, typename F>
  auto Rot135(Complex<F> v) -> Complex<F>
  {
    if constexpr (D == Dir::Forward)
      { return { .r = InvSqrt2<F> * (v.i - v.r), .i = -InvSqrt2<F> * (v.r + v.i) }; }
    else
      { } // !!!
  }
}

