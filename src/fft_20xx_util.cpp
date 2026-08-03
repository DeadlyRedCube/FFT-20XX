#include "fft_20xx/fft_20xx.hpp"
#include "fft_20xx/fft_20xx_util.hpp"

#include "fft_20xx_common.h"
#include "fft_20xx_platform.h"

#include <cstdlib>
#include <optional>

#if COMPILER_MSVC
  #include <malloc.h>
#endif

#if (COMPILER_GCC || COMPILER_CLANG) && !OS_WINDOWS && CPU_X86_ANY
  #include <cpuid.h>
#endif

namespace fft_20xx
{
  #if CPU_X86_ANY
    static const auto s_simdSupport =
      []
      {
        auto s = simd_support_flags::none;

        #if COMPILER_MSVC || COMPILER_CLANG_PRETENDING_TO_BE_MSVC
          int data[4] {}; // corresponds to EAX, EBX, ECX, and EDX outputs
          __cpuidex(data, 0, 0);
          auto maxParam = std::uint32_t(data[0]);

          std::ranges::fill(data, 0);
          __cpuidex(data, 1, 0);
          auto ecx = std::uint32_t(data[2]);
          auto edx = std::uint32_t(data[3]);
        #else
          std::uint32_t eax {};
          std::uint32_t ebx {};
          std::uint32_t ecx {};
          std::uint32_t edx {};
          __get_cpuid(0, &eax, &ebx, &ecx, &edx);
          auto maxParam = eax;
          eax = ebx = ecx = edx = 0;
          __get_cpuid(1, &eax, &ebx, &ecx, &edx);
        #endif

        if (edx & (1u << 28))
          { s |= simd_support_flags::sse2; }
        if (ecx & (1u << 0))
          { s |= simd_support_flags::sse3; }
        if (ecx & (1u << 9))
          { s |= simd_support_flags::ssse3; }
        if (ecx & (1u << 19))
          { s |= simd_support_flags::sse4_1; }
        if (ecx & (1u << 20))
          { s |= simd_support_flags::sse4_2; }

        if (maxParam >= 7 && (ecx & (1u << 28)))
        {
          // Need to check the xgetbv instruction to see if the OS itself supports the YMM registerss.
          //  There's likely an additional bit to check here if we want to validate AVX512 support (ZMM registers)
          #if COMPILER_MSVC || COMPILER_CLANG_PRETENDING_TO_BE_MSVC
            auto xcr0 = _xgetbv(0);
          #else
            std::uint32_t xcr0;
            __asm__("xgetbv" : "=a"(xcr0) : "c"(0) : "%edx" );
          #endif
          if ((xcr0 & 0x06) == 0x06) // this indicates that XMM and YMM registers will save
          {
            s |= simd_support_flags::avx;
            if (ecx & (1u << 12))
              { s |= simd_support_flags::fma; }

            #if COMPILER_MSVC || COMPILER_CLANG_PRETENDING_TO_BE_MSVC
              std::ranges::fill(data, 0);
              __cpuidex(data, 7, 0);
              auto ebx = std::uint32_t(data[1]);
              ecx = std::uint32_t(data[2]);
            #else
              eax = 7;
              ebx = ecx = edx = 0;

              // For some reason, __get_cpuid doesn't give the correct results for level 7. Why? WHY?
              __asm__("cpuid;" : "+b"(ebx), "+a"(eax), "+c"(ecx), "=d"(edx));
            #endif

            if (ebx & (1u << 5))
              { s |= simd_support_flags::avx2; }

            // Could check for various AVX512 things here if support is added (currently my CPU doesn't even support it,
            //  so I can't test it)
          }
        }

        return s;
      }();
  #endif
}

extern "C"
{
  size_t get_simd_alignment()
    { return FFT20XX::SIMDAlignment; }

  #if CPU_X86_ANY
    fft_20xx::c::simd_support_flags_t fft_20xx_get_simd_support()
      { return fft_20xx::c::simd_support_flags_t(fft_20xx::s_simdSupport); }
  #endif


  void *fft_20xx_allocate_bytes_aligned(std::size_t byte_count, std::size_t pow_2_alignment)
  {
    assert(FFT20XX::IsPowerOfTwo(pow_2_alignment));

    #if defined (_MSC_VER)
      return _aligned_malloc(byte_count,  pow_2_alignment);
    #else
      return std::aligned_alloc(pow_2_alignment, byte_count);
    #endif
  }

  void *fft_20xx_allocate_bytes_simd_aligned(std::size_t byte_count)
    { return fft_20xx_allocate_bytes_aligned(byte_count, FFT20XX::SIMDAlignment); }

  void fft_20xx_free_simd_aligned(const void *mem)
  {
    #if defined (_MSC_VER)
      _aligned_free(const_cast<void *>(mem));
    #else
      std::free(const_cast<void *>(mem));
    #endif
  }

  #if FFT_20XX_CPU_X86_ANY
    void fft_20xx_set_flush_subnormals_to_zero(bool flush_to_zero)
    {
      auto prevCSR = _mm_getcsr();


      _mm_setcsr(
        (prevCSR & ~(_MM_FLUSH_ZERO_MASK | _MM_DENORMALS_ZERO_MASK))
          | (flush_to_zero ? (_MM_FLUSH_ZERO_ON | _MM_DENORMALS_ZERO_ON) : 0));
    }
  #elif FFT_20XX_CPU_ARM_ANY
    void fft_20xx_set_flush_subnormals_to_zero(bool flush_to_zero)
    {
      #if defined(_MSC_VER)
        std::uint32_t current;
        _controlfp_s(&current, _DN_FLUSH, _MCW_DN);
      #elif defined(_M_ARM64) || defined(__aarch64__)
        // From:
        //  https://developer.arm.com/documentation/ddi0595/2021-06/AArch64-Registers/FPCR--Floating-point-Control-Register
        std::uint64_t current;
        __asm__ ("mrs %0, fpcr" : "=r"(current) : ); // Read the current control values
        if (flush_to_zero)
          { current |= (1u << 24); }// Add the flush-to-zero flag
        else
          { current &= ~(1u << 24); } // Clear the flush-to-zero flag
        __asm__ ("msr fpcr, %0" : : "r"(current)); // Write the updated value
      #else
        // From:
        //  https://developer.arm.com/documentation/ddi0595/2021-06/AArch32-Registers/FPSCR--Floating-Point-Status-and-Control-Register
        std::uint32_t current;
        __asm__ ("vmrs %0, fpscr" : "=r"(current) : ); // Read the current control values
        if (flush_to_zero)
          { current |= (1ull << 24); } // Add the flush-to-zero flag
        else
          { current &= ~(1ull << 24); } // Clear the flush-to-zero flag
        __asm__ ("vmsr fpscr, %0" : : "r"(current)); // Write the updated value
      #endif
    }
  #else
    #error Unknown CPU type.
  #endif
}
