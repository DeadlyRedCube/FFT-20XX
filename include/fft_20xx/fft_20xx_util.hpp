#ifndef FFT_20XX_UTIL_HPP_INCLUDE_GUARD
  #define FFT_20XX_UTIL_HPP_INCLUDE_GUARD

  #include "fft_20xx/fft_20xx.hpp"

  #define FFT_20XX_INCLUDING_FROM_CPP
  #include "fft_20xx/fft_20xx_util.h"
  #undef FFT_20XX_INCLUDING_FROM_CPP

  #if defined(_MSC_VER)
    #pragma warning (push)
    #pragma warning (disable: 4514) // unreferenced inline function has been removed
  #endif

  namespace fft_20xx
  {
    // Allocate the given number of bytes, aligned to given power of two alignment.
    void *allocate_bytes_aligned(size_t byte_count, size_t pow_2_alignment);

    // Allocate the given number of bytes, aligned to the system's max SIMD size.
    inline auto allocate_bytes_simd_aligned(size_t byte_count) -> void *;

    // Allocate the given number of typed elements, aligned to the system's max SIMD size.
    template <typename T> auto allocate_trivial_type_simd_aligned(size_t element_count) -> T *;

    // Free the given block of aligned memory.
    inline void free_aligned(const void *memory);

    #if FFT_20XX_CPU_X86_ANY
      enum class simd_support_flags
      {
        none = c::FFT_20XX_SIMD_SUPPORT_FLAG_NONE,
        sse2 = c::FFT_20XX_SIMD_SUPPORT_FLAG_SSE2,
        sse3 = c::FFT_20XX_SIMD_SUPPORT_FLAG_SSE3,
        ssse3 = c::FFT_20XX_SIMD_SUPPORT_FLAG_SSSE3,
        sse4_1 = c::FFT_20XX_SIMD_SUPPORT_FLAG_SSE4_1,
        sse4_2 = c::FFT_20XX_SIMD_SUPPORT_FLAG_SSE4_2,
        avx = c::FFT_20XX_SIMD_SUPPORT_FLAG_AVX,
        fma = c::FFT_20XX_SIMD_SUPPORT_FLAG_FMA,
        avx2 = c::FFT_20XX_SIMD_SUPPORT_FLAG_AVX2,
      };

      // Allow simd_support_flags to act as flags
      inline auto operator&(simd_support_flags a, simd_support_flags b) -> simd_support_flags;
      inline auto operator&=(simd_support_flags &a, simd_support_flags b) -> simd_support_flags &;
      inline auto operator|(simd_support_flags a, simd_support_flags b) -> simd_support_flags;
      inline auto operator|=(simd_support_flags &a, simd_support_flags b) -> simd_support_flags &;
      inline auto operator^(simd_support_flags a, simd_support_flags b) -> simd_support_flags;
      inline auto operator^=(simd_support_flags &a, simd_support_flags b) -> simd_support_flags &;
      inline auto operator~(simd_support_flags a) -> simd_support_flags;

      // Get the SIMD support reported by the current CPU.
      inline auto get_simd_support() -> simd_support_flags;
    #endif

    #if FFT_20XX_CPU_X86_ANY || FFT_20XX_CPU_ARM_ANY
      // Set the CPU flags in the current thread to flush subnormal (extremely tiny) float values to 0, for performance
      //  when the parameter is true, or disable the subnormal flushing when false.
      inline void set_flush_subnormals_to_zero(bool flush_to_zero);
    #endif



    // Inline function definitions

    inline auto allocate_bytes_aligned(size_t byte_count, size_t pow_2_alignment) -> void *
      { return c::fft_20xx_allocate_bytes_aligned(byte_count, pow_2_alignment); }

    inline auto allocate_bytes_simd_aligned(size_t byte_count) -> void *
      { return c::fft_20xx_allocate_bytes_simd_aligned(byte_count); }

    template <typename T>
    auto allocate_trivial_type_simd_aligned(size_t element_count) -> T *
    {
      static_assert(std::is_trivially_constructible_v<T> && std::is_trivially_destructible_v<T>);

      return static_cast<T *>(allocate_bytes_aligned(
        element_count * sizeof(T),
        std::max(alignof(T), c::get_simd_alignment())));
    }

    inline void free_aligned(const void *memory)
      { c::fft_20xx_free_aligned(memory); }

    #if FFT_20XX_CPU_X86_ANY
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

  #if defined(_MSC_VER)
    #pragma warning (pop)
  #endif
#endif