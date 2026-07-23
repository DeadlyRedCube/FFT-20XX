#include "fft_20xx_common.h"
#include "fft_20xx_table.h"

static_assert(CPU_X86 || CPU_X64);

static_assert(SIMD_AVX);

#if COMPILER_GCC
  #pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

// Ensure that this file only compiles with AVX support even if FMA is supported by the global settings
#if SIMD_FMA
  #undef SIMD_FMA
  #define SIMD_FMA 0
#endif

namespace FFT20XX::AVX
{
  #include "fft_20xx_avx_common.h"
}

using namespace FFT20XX;
using namespace fft_20xx::c;
extern "C"
{
  void fft_cplx_interleaved_f32_avx(const fft_table_f32 *table, const float *input, float *output, size_t fftLength)
  {
    [[maybe_unused]] auto t = reinterpret_cast<const FFTTableImpl<float> *>(table);
    COMMON_ASSERTS();
    AVX::RunComplexFFT<Dir::Forward>(t->Data(), input, output, fftLength);
  }


  void fft_cplx_interleaved_f64_avx(const fft_table_f64 *table, const double *input, double *output, size_t fftLength)
  {
    [[maybe_unused]] auto t = reinterpret_cast<const FFTTableImpl<double> *>(table);
    COMMON_ASSERTS();
    AVX::RunComplexFFT<Dir::Forward>(t->Data(), input, output, fftLength);
  }


  void ifft_cplx_interleaved_f32_avx(const fft_table_f32 *table, const float *input, float *output, size_t fftLength)
  {
    [[maybe_unused]] auto t = reinterpret_cast<const FFTTableImpl<float> *>(table);
    COMMON_ASSERTS();
    AVX::RunComplexFFT<Dir::Inverse>(t->Data(), input, output, fftLength);
  }


  void ifft_cplx_interleaved_f64_avx(const fft_table_f64 *table, const double *input, double *output, size_t fftLength)
  {
    [[maybe_unused]] auto t = reinterpret_cast<const FFTTableImpl<double> *>(table);
    COMMON_ASSERTS();
    AVX::RunComplexFFT<Dir::Inverse>(t->Data(), input, output, fftLength);
  }
}