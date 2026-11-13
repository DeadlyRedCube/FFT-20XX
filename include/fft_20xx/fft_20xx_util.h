#ifndef FFT_20XX_UTIL_H_INCLUDE_GUARD
  #define FFT_20XX_UTIL_H_INCLUDE_GUARD

  #include "fft_20xx/fft_20xx.h"

  #if defined(_M_X64) || defined(__x86_64__) || defined(_M_IX86) || defined(__i386__)
    #define FFT_20XX_CPU_X86_ANY 1
  #else
    #define FFT_20XX_CPU_X86_ANY 0
  #endif

  #if defined(_M_ARM64) || defined(__aarch64__) || defined(_M_ARM) || defined (__arm__)
    #define FFT_20XX_CPU_ARM_ANY 1
  #else
    #define FFT_20XX_CPU_ARM_ANY 0
  #endif

  #ifdef __cplusplus
    #ifndef FFT_20XX_INCLUDING_FROM_CPP
      #error This header is intended for C only. Please include "fft_20xx_util.hpp" instead.
    #endif

    extern "C"
    {
      namespace fft_20xx::c
      {
  #endif
        // Allocate the given number of bytes, aligned to given power of two alignment.
        void *fft_20xx_allocate_bytes_aligned(size_t byte_count, size_t pow_2_alignment);

        // Allocate the given number of bytes, aligned to the system's max SIMD size.
        void *fft_20xx_allocate_bytes_simd_aligned(size_t byte_count);

        // Free the given block of aligned memory.
        void fft_20xx_free_aligned(const void *memory);

        #if FFT_20XX_CPU_X86_ANY
          typedef enum fft_20xx_simd_support_flags_t
          {
            FFT_20XX_SIMD_SUPPORT_FLAG_NONE    = 0x00,

            FFT_20XX_SIMD_SUPPORT_FLAG_SSE2    = 0x01,
            FFT_20XX_SIMD_SUPPORT_FLAG_SSE3    = 0x02,
            FFT_20XX_SIMD_SUPPORT_FLAG_SSSE3   = 0x04,
            FFT_20XX_SIMD_SUPPORT_FLAG_SSE4_1  = 0x08,
            FFT_20XX_SIMD_SUPPORT_FLAG_SSE4_2  = 0x10,
            FFT_20XX_SIMD_SUPPORT_FLAG_AVX     = 0x20,
            FFT_20XX_SIMD_SUPPORT_FLAG_FMA     = 0x40,
            FFT_20XX_SIMD_SUPPORT_FLAG_AVX2    = 0x80,
          } fft_20xx_simd_support_flags_t;

          fft_20xx_simd_support_flags_t fft_20xx_get_simd_support();
        #endif

        #if FFT_20XX_CPU_X86_ANY || FFT_20XX_CPU_ARM_ANY
          void fft_20xx_set_flush_subnormals_to_zero(bool flush_to_zero);
        #endif

  #ifdef __cplusplus
      } // namespace
    } // extern "C"
  #endif
#endif