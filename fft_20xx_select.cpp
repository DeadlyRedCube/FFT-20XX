#include "include/fft_20xx.h"

#include "fft_20xx_common.h"

using namespace FFT20XX;

#if (COMPILER_GCC || COMPILER_CLANG) && !OS_WINDOWS
  #include <cpuid.h>
#endif


extern "C"
{
  void FFTComplex32(const FFTTable32 *table, const std::complex<float> *input, std::complex<float> *output, size_t fftLength)
  {
    FFTComplex32Interleaved(table, reinterpret_cast<const float *>(input), reinterpret_cast<float *>(output), fftLength);
  }

  void FFTComplex64(const FFTTable64 *table, const std::complex<double> *input, std::complex<double> *output, size_t fftLength)
  {
    FFTComplex64Interleaved(table, reinterpret_cast<const double *>(input), reinterpret_cast<double *>(output), fftLength);
  }

  void IFFTComplex32(const FFTTable32 *table, const std::complex<float> *input, std::complex<float> *output, size_t fftLength)
  {
    IFFTComplex32Interleaved(table, reinterpret_cast<const float *>(input), reinterpret_cast<float *>(output), fftLength);
  }

  void IFFTComplex64(const FFTTable64 *table, const std::complex<double> *input, std::complex<double> *output, size_t fftLength)
  {
    IFFTComplex64Interleaved(table, reinterpret_cast<const double *>(input), reinterpret_cast<double *>(output), fftLength);
  }



  void FFTComplex32_Scalar(const FFTTable32 *table, const std::complex<float> *input, std::complex<float> *output, size_t fftLength)
  {
    FFTComplex32Interleaved_Scalar(table, reinterpret_cast<const float *>(input), reinterpret_cast<float *>(output), fftLength);
  }

  void FFTComplex64_Scalar(const FFTTable64 *table, const std::complex<double> *input, std::complex<double> *output, size_t fftLength)
  {
    FFTComplex64Interleaved_Scalar(table, reinterpret_cast<const double *>(input), reinterpret_cast<double *>(output), fftLength);
  }

  void IFFTComplex32_Scalar(const FFTTable32 *table, const std::complex<float> *input, std::complex<float> *output, size_t fftLength)
  {
    IFFTComplex32Interleaved_Scalar(table, reinterpret_cast<const float *>(input), reinterpret_cast<float *>(output), fftLength);
  }

  void IFFTComplex64_Scalar(const FFTTable64 *table, const std::complex<double> *input, std::complex<double> *output, size_t fftLength)
  {
    IFFTComplex64Interleaved_Scalar(table, reinterpret_cast<const double *>(input), reinterpret_cast<double *>(output), fftLength);
  }


  #if CPU_X86 || CPU_X64
    void FFTComplex32_FMA(const FFTTable32 *table, const std::complex<float> *input, std::complex<float> *output, size_t fftLength)
    {
      FFTComplex32Interleaved_FMA(table, reinterpret_cast<const float *>(input), reinterpret_cast<float *>(output), fftLength);
    }

    void FFTComplex64_FMA(const FFTTable64 *table, const std::complex<double> *input, std::complex<double> *output, size_t fftLength)
    {
      FFTComplex64Interleaved_FMA(table, reinterpret_cast<const double *>(input), reinterpret_cast<double *>(output), fftLength);
    }

    void IFFTComplex32_FMA(const FFTTable32 *table, const std::complex<float> *input, std::complex<float> *output, size_t fftLength)
    {
      IFFTComplex32Interleaved_FMA(table, reinterpret_cast<const float *>(input), reinterpret_cast<float *>(output), fftLength);
    }

    void IFFTComplex64_FMA(const FFTTable64 *table, const std::complex<double> *input, std::complex<double> *output, size_t fftLength)
    {
      IFFTComplex64Interleaved_FMA(table, reinterpret_cast<const double *>(input), reinterpret_cast<double *>(output), fftLength);
    }
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

SIMDSupport operator&(SIMDSupport a, SIMDSupport b) { return SIMDSupport(int(a) & int(b)); }
SIMDSupport &operator&=(SIMDSupport &a, SIMDSupport b) { a = a & b; return a; }
SIMDSupport operator|(SIMDSupport a, SIMDSupport b) { return SIMDSupport(int(a) | int(b)); }
SIMDSupport &operator|=(SIMDSupport &a, SIMDSupport b) { a = a | b; return a; }
SIMDSupport operator^(SIMDSupport a, SIMDSupport b) { return SIMDSupport(int(a) ^ int(b)); }
SIMDSupport &operator^=(SIMDSupport &a, SIMDSupport b) { a = a ^ b; return a; }
SIMDSupport operator~(SIMDSupport a) { return SIMDSupport(~int(a)); }

  
[[maybe_unused]] static const SIMDSupport s_simdSupport = 
  []
  {
    SIMDSupport s = SIMDSupport::None;

    #if COMPILER_MSVC
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
      #if COMPILER_MSVC
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
          
        #if COMPILER_MSVC
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
  void FFTComplex32Interleaved(const FFTTable32 *table, const float *input, float *output, size_t fftLength)
  {
    #if SIMD_AVX && SIMD_FMA
      FFTComplex32Interleaved_FMA(table, input, output, fftLength);
    #elif CPU_X86 || CPU_X64
      if ((s_simdSupport & (SIMDSupport::FMA | SIMDSupport::AVX)) == (SIMDSupport::FMA | SIMDSupport::AVX))
        { FFTComplex32Interleaved_FMA(table, input, output, fftLength); }
      else
        { FFTComplex32Interleaved_Scalar(table, input, output, fftLength); }
    #endif
  }

  void IFFTComplex32Interleaved(const FFTTable32 *table, const float *input, float *output, size_t fftLength)
  {
    #if SIMD_AVX && SIMD_FMA
      IFFTComplex32Interleaved_FMA(table, input, output, fftLength);
    #elif CPU_X86 || CPU_X64
      if ((s_simdSupport & (SIMDSupport::FMA | SIMDSupport::AVX)) == (SIMDSupport::FMA | SIMDSupport::AVX))
        { IFFTComplex32Interleaved_FMA(table, input, output, fftLength); }
      else
        { IFFTComplex32Interleaved_Scalar(table, input, output, fftLength); }
    #endif
  }

  void FFTComplex64Interleaved(const FFTTable64 *table, const double *input, double *output, size_t fftLength)
  {
    #if SIMD_AVX && SIMD_FMA
      FFTComplex64Interleaved_FMA(table, input, output, fftLength);
    #elif CPU_X86 || CPU_X64
      if ((s_simdSupport & (SIMDSupport::FMA | SIMDSupport::AVX)) == (SIMDSupport::FMA | SIMDSupport::AVX))
        { FFTComplex64Interleaved_FMA(table, input, output, fftLength); }
      else
        { FFTComplex64Interleaved_Scalar(table, input, output, fftLength); }
    #endif
  }

  void IFFTComplex64Interleaved(const FFTTable64 *table, const double *input, double *output, size_t fftLength)
  {
    #if SIMD_AVX && SIMD_FMA
      IFFTComplex64Interleaved_FMA(table, input, output, fftLength);
    #elif CPU_X86 || CPU_X64
      if ((s_simdSupport & (SIMDSupport::FMA | SIMDSupport::AVX)) == (SIMDSupport::FMA | SIMDSupport::AVX))
        { IFFTComplex64Interleaved_FMA(table, input, output, fftLength); }
      else
        { IFFTComplex64Interleaved_Scalar(table, input, output, fftLength); }
    #endif
  }
}
