#include "fft_20xx/fft_20xx.h"

#include "fft_20xx_common.h"

using namespace FFT20XX;

#if (COMPILER_GCC || COMPILER_CLANG) && !OS_WINDOWS && CPU_X86_ANY
  #include <cpuid.h>
#endif

#if 1
extern "C"
{
  using namespace fft_20xx::c;

#define MAKE_C_OVERLOADS(suffix) \
  void fft_cplx_interleaved_f32##suffix(const fft_table_f32 *table, const float *input, float *output, size_t fftLength); \
  void fft_cplx_interleaved_f64##suffix(const fft_table_f64 *table, const double *input, double *output, size_t fftLength); \
  \
  void fft_cplx_f32##suffix(const fft_table_f32 *table, const std::complex<float> *input, std::complex<float> *output, size_t fftLength) \
    { fft_cplx_interleaved_f32##suffix(table, reinterpret_cast<const float *>(input), reinterpret_cast<float *>(output), fftLength); } \
  \
  void fft_cplx_f64##suffix(const fft_table_f64 *table, const std::complex<double> *input, std::complex<double> *output, size_t fftLength) \
    { fft_cplx_interleaved_f64##suffix(table, reinterpret_cast<const double *>(input), reinterpret_cast<double *>(output), fftLength); } \
  \
  void ifft_cplx_interleaved_f32##suffix(const fft_table_f32 *table, const float *input, float *output, size_t fftLength); \
  void ifft_cplx_interleaved_f64##suffix(const fft_table_f64 *table, const double *input, double *output, size_t fftLength); \
  \
  void ifft_cplx_f32##suffix(const fft_table_f32 *table, const std::complex<float> *input, std::complex<float> *output, size_t fftLength) \
    { ifft_cplx_interleaved_f32##suffix(table, reinterpret_cast<const float *>(input), reinterpret_cast<float *>(output), fftLength); } \
  \
  void ifft_cplx_f64##suffix(const fft_table_f64 *table, const std::complex<double> *input, std::complex<double> *output, size_t fftLength) \
    { ifft_cplx_interleaved_f64##suffix(table, reinterpret_cast<const double *>(input), reinterpret_cast<double *>(output), fftLength); } \
  \
  void fft_real_f32##suffix(const fft_table_f32 *table, const float *input, float *output, size_t fftLength); \
  void fft_real_f64##suffix(const fft_table_f64 *table, const double *input, double *output, size_t fftLength); \
  \
  void ifft_real_f32##suffix(const fft_table_f32 *table, const float *input, float *output, size_t fftLength); \
  void ifft_real_f64##suffix(const fft_table_f64 *table, const double *input, double *output, size_t fftLength); \
  \

  MAKE_C_OVERLOADS()
  MAKE_C_OVERLOADS(_scalar)

  #if CPU_X86 || CPU_X64
    MAKE_C_OVERLOADS(_sse2);
    MAKE_C_OVERLOADS(_avx);
    MAKE_C_OVERLOADS(_fma);
  #elif SIMD_NEON
    MAKE_C_OVERLOADS(_neon);
  #endif
}


#if CPU_X86 || CPU_X64
  enum class SIMDSupport
  {
    None    = 0x00,
    SSE2    = 0x01,
    SSE3    = 0x02,
    SSSE3   = 0x04,
    SSE4_1  = 0x08,
    SSE4_2  = 0x10,
    AVX     = 0x20,
    FMA     = 0x40,
    AVX2    = 0x80,
  };

  inline SIMDSupport operator&(SIMDSupport a, SIMDSupport b) { return SIMDSupport(int(a) & int(b)); }
  inline SIMDSupport &operator&=(SIMDSupport &a, SIMDSupport b) { a = a & b; return a; }
  inline SIMDSupport operator|(SIMDSupport a, SIMDSupport b) { return SIMDSupport(int(a) | int(b)); }
  inline SIMDSupport &operator|=(SIMDSupport &a, SIMDSupport b) { a = a | b; return a; }
  inline SIMDSupport operator^(SIMDSupport a, SIMDSupport b) { return SIMDSupport(int(a) ^ int(b)); }
  inline SIMDSupport &operator^=(SIMDSupport &a, SIMDSupport b) { a = a ^ b; return a; }
  inline SIMDSupport operator~(SIMDSupport a) { return SIMDSupport(~int(a)); }


  [[maybe_unused]] static const SIMDSupport s_simdSupport =
    []
    {
      SIMDSupport s = SIMDSupport::None;

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
        { s |= SIMDSupport::SSE2; }
      if (ecx & (1u << 0))
        { s |= SIMDSupport::SSE3; }
      if (ecx & (1u << 9))
        { s |= SIMDSupport::SSSE3; }
      if (ecx & (1u << 19))
        { s |= SIMDSupport::SSE4_1; }
      if (ecx & (1u << 20))
        { s |= SIMDSupport::SSE4_2; }

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
          s |= SIMDSupport::AVX;
          if (ecx & (1u << 12))
            { s |= SIMDSupport::FMA; }

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
            { s |= SIMDSupport::AVX2; }

          // Could check for various AVX512 things here if support is added
        }
      }

      return s;
    }();
#endif

extern "C"
{
  #if SIMD_AVX
    #if SIMD_FMA
      #define SELECT(name, ...) name##_fma(__VA_ARGS__)
    #else
      do \
      { \
        if ((s_simdSupport & (SIMDSupport::FMA | SIMDSupport::AVX)) == (SIMDSupport::FMA | SIMDSupport::AVX)) \
          { name##_fma(__VA_ARGS__); } \
        else \
          { name##_avx(__VA_ARGS__); } \
      } while (false)
    #endif
  #elif SIMD_SSE2
    #define SELECT(name, ...) \
      do \
      { \
        if ((s_simdSupport & (SIMDSupport::FMA | SIMDSupport::AVX)) == (SIMDSupport::FMA | SIMDSupport::AVX)) \
          { name##_fma(__VA_ARGS__); } \
        else if ((s_simdSupport & SIMDSupport::AVX) == SIMDSupport::AVX) \
          { name##_avx(__VA_ARGS__); } \
        else \
          { name##_sse2(__VA_ARGS__); } \
      } while (false)
  #elif CPU_X86 || CPU_X64
    #define SELECT(name, ...) \
      do \
      { \
        if ((s_simdSupport & (SIMDSupport::FMA | SIMDSupport::AVX)) == (SIMDSupport::FMA | SIMDSupport::AVX)) \
          { name##_fma(__VA_ARGS__); } \
        else if ((s_simdSupport & SIMDSupport::AVX) == SIMDSupport::AVX) \
          { name##_avx(__VA_ARGS__); } \
        else if ((s_simdSupport & SIMDSupport::SSE2) == SIMDSupport::SSE2) \
          { name##_sse2(__VA_ARGS__); } \
        else \
          { name##_scalar(__VA_ARGS__); } \
      } while (false)
  #elif SIMD_NEON
    #define SELECT(name, ...) name##_neon(__VA_ARGS__)
  #else
    #define SELECT(name, ...) name##_scalar(__VA_ARGS__)
  #endif

  void fft_cplx_interleaved_f32(const fft_table_f32 *table, const float *input, float *output, size_t length)
    { SELECT(fft_cplx_interleaved_f32, table, input, output, length); }

  void fft_cplx_interleaved_f64(const fft_table_f64 *table, const double *input, double *output, size_t length)
    { SELECT(fft_cplx_interleaved_f64, table, input, output, length); }

  void ifft_cplx_interleaved_f32(const fft_table_f32 *table, const float *input, float *output, size_t length)
    { SELECT(ifft_cplx_interleaved_f32, table, input, output, length); }

  void ifft_cplx_interleaved_f64(const fft_table_f64 *table, const double *input, double *output, size_t length)
    { SELECT(ifft_cplx_interleaved_f64, table, input, output, length); }

  void fft_real_f32(const fft_table_f32 *table, const float *input, float *output, size_t length)
    { SELECT(fft_real_f32, table, input, output, length); }

  void ifft_real_f32(const fft_table_f32 *table, const float *input, float *output, size_t length)
    { SELECT(ifft_real_f32, table, input, output, length); }

  void fft_real_f64(const fft_table_f64 *table, const double *input, double *output, size_t length)
    { SELECT(fft_real_f64, table, input, output, length); }

  void ifft_real_f64(const fft_table_f64 *table, const double *input, double *output, size_t length)
    { SELECT(ifft_real_f64, table, input, output, length); }
}
#endif