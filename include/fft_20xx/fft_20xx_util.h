#ifndef FFT_20XX_UTIL_INCLUDE_GUARD
  #define FFT_20XX_UTIL_INCLUDE_GUARD

  #include "fft_20xx/fft_20xx.h"

  #if defined(_MSC_VER)
    #pragma warning (push)
    #pragma warning (disable: 4514) // unreferenced inline function has been removed
  #endif

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
    extern "C"
    {
      namespace fft_20xx::c
      {
  #endif
        #if FFT_20XX_CPU_X86_ANY
          typedef enum simd_support_flags_t
          {
            SIMD_SUPPORT_FLAG_NONE = 0x00,

            SIMD_SUPPORT_FLAG_SSE2    = 0x01,
            SIMD_SUPPORT_FLAG_SSE3    = 0x02,
            SIMD_SUPPORT_FLAG_SSSE3   = 0x04,
            SIMD_SUPPORT_FLAG_SSE4_1  = 0x08,
            SIMD_SUPPORT_FLAG_SSE4_2  = 0x10,
            SIMD_SUPPORT_FLAG_AVX     = 0x20,
            SIMD_SUPPORT_FLAG_FMA     = 0x40,
            SIMD_SUPPORT_FLAG_AVX2    = 0x80,
          } simd_support_flags_t;

          simd_support_flags_t fft_20xx_get_simd_support();
        #endif

        #if FFT_20XX_CPU_X86_ANY || FFT_20XX_CPU_ARM_ANY
          void fft_20xx_set_flush_subnormals_to_zero(bool flush_to_zero);
        #endif

        void *fft_20xx_allocate_simd_aligned(size_t byte_count);
        void fft_20xx_free_simd_aligned(void *memory);
  #ifdef __cplusplus
      } // namespace
    } // extern "C"


    namespace fft_20xx
    {
      inline auto allocate_simd_aligned(size_t byte_count) -> void *
        { return c::fft_20xx_allocate_simd_aligned(byte_count); }

      template <typename T>
      auto allocate_simd_aligned(size_t element_count) -> T *
        { return static_cast<T *>(allocate_simd_aligned(element_count * sizeof(T))); }

      inline void free_simd_aligned(void *memory)
        { c::fft_20xx_free_simd_aligned(memory); }

      #if FFT_20XX_CPU_X86_ANY
        enum class simd_support_flags
        {
          none = c::SIMD_SUPPORT_FLAG_NONE,
          sse2 = c::SIMD_SUPPORT_FLAG_SSE2,
          sse3 = c::SIMD_SUPPORT_FLAG_SSE3,
          ssse3 = c::SIMD_SUPPORT_FLAG_SSSE3,
          sse4_1 = c::SIMD_SUPPORT_FLAG_SSE4_1,
          sse4_2 = c::SIMD_SUPPORT_FLAG_SSE4_2,
          avx = c::SIMD_SUPPORT_FLAG_AVX,
          fma = c::SIMD_SUPPORT_FLAG_FMA,
          avx2 = c::SIMD_SUPPORT_FLAG_AVX2,
        };

        inline auto operator&(simd_support_flags a, simd_support_flags b) -> simd_support_flags
          { return simd_support_flags(int(a) & int(b)); }
        inline auto operator&=(simd_support_flags &a, simd_support_flags b) -> simd_support_flags &
          { a = a & b; return a; }
        inline auto operator|(simd_support_flags a, simd_support_flags b) -> simd_support_flags
          { return simd_support_flags(int(a) | int(b)); }
        inline auto operator|=(simd_support_flags &a, simd_support_flags b) -> simd_support_flags &
          { a = a | b; return a; }
        inline auto operator^(simd_support_flags a, simd_support_flags b) -> simd_support_flags
          { return simd_support_flags(int(a) ^ int(b)); }
        inline auto operator^=(simd_support_flags &a, simd_support_flags b) -> simd_support_flags &
          { a = a ^ b; return a; }
        inline auto operator~(simd_support_flags a) -> simd_support_flags
          { return simd_support_flags(~int(a)); }

        inline auto get_simd_support() -> simd_support_flags
          { return simd_support_flags(c::fft_20xx_get_simd_support()); }
      #endif

      #if FFT_20XX_CPU_X86_ANY || FFT_20XX_CPU_ARM_ANY
        inline void set_flush_subnormals_to_zero(bool flush_to_zero)
          { c::fft_20xx_set_flush_subnormals_to_zero(flush_to_zero); }
      #endif
    }
  #endif

  #if defined(_MSC_VER)
    #pragma warning (pop)
  #endif
#endif