#include "fft_20xx_common.h"


static_assert(CPU_X86 || CPU_X64);

static_assert(SIMD_AVX);
static_assert(SIMD_FMA);

#include <concepts>
#include <cinttypes>
#include <immintrin.h>

#if COMPILER_GCC
  #pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

namespace FFT20XX::FMA
{
  #include "fft_20xx_avx_common.h"
}


extern "C"
{
  using namespace FFT20XX;

  #define COMMON_ASSERTS() \
    assert(table != nullptr); \
    assert(input != nullptr); \
    assert(output != nullptr); \
    assert(IsPowerOfTwo(fftLength)); \
    assert(t->MaxFFTLength() >= fftLength)

  void FFTComplex32Interleaved_FMA(const FFTTable32 *table, const float *input, float *output, size_t fftLength)
  {
    auto t = reinterpret_cast<const FFTTableImpl<float> *>(table);
    COMMON_ASSERTS();
    FMA::RunComplexFFT<Dir::Forward>(t->Data(), input, output, fftLength);
  }

  void FFTComplex64Interleaved_FMA(const FFTTable64 *table, const double *input, double *output, size_t fftLength)
  {
    auto t = reinterpret_cast<const FFTTableImpl<double> *>(table);
    COMMON_ASSERTS();
    FMA::RunComplexFFT<Dir::Forward>(t->Data(), input, output, fftLength);
  }


  void IFFTComplex32Interleaved_FMA(const FFTTable32 *table, const float *input, float *output, size_t fftLength)
  {
    auto t = reinterpret_cast<const FFTTableImpl<float> *>(table);
    COMMON_ASSERTS();
    FMA::RunComplexFFT<Dir::Inverse>(t->Data(), input, output, fftLength);
  }

  void IFFTComplex64Interleaved_FMA(const FFTTable64 *table, const double *input, double *output, size_t fftLength)
  {
    auto t = reinterpret_cast<const FFTTableImpl<double> *>(table);
    COMMON_ASSERTS();
    FMA::RunComplexFFT<Dir::Inverse>(t->Data(), input, output, fftLength);
  }

  void FFTReal32_FMA(const FFTTable32 *table, const float *input, float *output, size_t fftLength)
  {
    auto t = reinterpret_cast<const FFTTableImpl<float> *>(table);
    COMMON_ASSERTS();
    FMA::RunRealFwdFFT(t->Data(), input, output, fftLength);
  }

  void FFTReal64_FMA(const FFTTable64 *table, const double *input, double *output, size_t fftLength)
  {
    auto t = reinterpret_cast<const FFTTableImpl<double> *>(table);
    COMMON_ASSERTS();
    FMA::RunRealFwdFFT(t->Data(), input, output, fftLength);
  }

  void IFFTReal32_FMA(const FFTTable32 *table, const float *input, float *output, size_t fftLength)
  {
    auto t = reinterpret_cast<const FFTTableImpl<float> *>(table);
    COMMON_ASSERTS();
    FMA::RunRealInvFFT(t->Data(), input, output, fftLength);
  }

  void IFFTReal64_FMA(const FFTTable64 *table, const double *input, double *output, size_t fftLength)
  {
    auto t = reinterpret_cast<const FFTTableImpl<double> *>(table);
    COMMON_ASSERTS();
    FMA::RunRealInvFFT(t->Data(), input, output, fftLength);
  }
}