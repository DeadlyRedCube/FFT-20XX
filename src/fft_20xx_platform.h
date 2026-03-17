#pragma once

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

#if defined (_MSC_VER) && defined(__clang__)
  #define COMPILER_CLANG_PRETENDING_TO_BE_MSVC 1
#else
  #define COMPILER_CLANG_PRETENDING_TO_BE_MSVC 0
#endif

#if defined(_WIN32)
  #define OS_WINDOWS 1
#else
  #define OS_WINDOWS 0
#endif

#if defined (__GNUC__) && !defined(__clang__) && !defined(__llvm__) && !defined(__INTEL_COMPILER)
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
  #define CPU_ARM_ANY 0
  #define CPU_ARM64 0
  #define CPU_ARM32 0
#endif

#if defined(_M_X64) || defined(__x86_64__)
  #define CPU_X86_ANY 1
  #define CPU_X64 1
  #define CPU_X86 0
#elif defined(_M_IX86) || defined(__i386__)
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
//#elif COMPILER_CLANG
//  #define FORCE_INLINE __attribute__((always_inline)) inline
#else
  #define FORCE_INLINE inline
#endif

#if defined(__SSE2__) || CPU_X64 || COMPILER_MSVC // All x64 processors have SSE and SSE2, MSVC always sets this.
  #define SIMD_SSE2 1
#else
  #define SIMD_SSE2 0
#endif

#if defined(__SSE3__) || defined(__AVX__) // AVX implies all the other lower SSEs (but MSVC won't set them)
  #define SIMD_SSE3 1
#else
  #define SIMD_SSE3 0
#endif

#if defined(__SSE4_1__) || defined(__AVX__)
  #define SIMD_SSE4_1 1
#else
  #define SIMD_SSE4_1 0
#endif

#if defined(__SSE4_2__) || defined(__AVX__)
  #define SIMD_SSE4_2 1
#else
  #define SIMD_SSE4_2 0
#endif

#if defined(__AVX__)
  #define SIMD_AVX 1
#else
  #define SIMD_AVX 0
#endif

#if defined(__FMA__) || (COMPILER_MSVC && defined(__AVX2__))
  #define SIMD_FMA 1
#else
  #define SIMD_FMA 0
#endif

#if defined(__AVX2__)
  #define SIMD_AVX2 1
#else
  #define SIMD_AVX2 0
#endif

#if defined(__ARM_NEON)
  #define SIMD_NEON 1
#else
  #define SIMD_NEON 0
#endif

#if defined(__ARM_NEON) && CPU_ARM64
  #define SIMD_NEON64 1
#else
  #define SIMD_NEON64 0
#endif


#if COMPILER_MSVC
  #pragma warning (disable: 4514) // unreferenced inline function has been removed
  #pragma warning (disable: 4710) // function not inlined
  #pragma warning (disable: 4711) // function selected for automatic inline expansion
  #pragma warning (disable: 5045) // Compiler will insert Spectre mitigation for memory load if /Qspectre switch specified
#endif

#include <algorithm>
#include <bit>
#include <cinttypes>
#include <cmath>

#if SIMD_NEON || SIMD_NEON64
  #include <arm_neon.h>
#endif

#if SIMD_SSE2
  #include <immintrin.h>
#endif

