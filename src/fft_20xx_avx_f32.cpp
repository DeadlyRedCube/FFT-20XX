#ifdef _MSC_VER
  #pragma warning (disable: 4711) // function <x> selected for automatic inline expansion
#endif

#include "fft_20xx_common.h"
#include "fft_20xx_table.h"
#include "fft_20xx_tiny_cplx.h"
#include "fft_20xx_tiny_real.h"

static_assert(CPU_X86 || CPU_X64);

static_assert(SIMD_AVX);
static_assert(!SIMD_FMA);

#if COMPILER_GCC
  #pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

using namespace FFT20XX;

namespace FFT20XX::AVX32
{
  #include "fft_20xx_avx_f32_common.h"
}


using namespace FFT20XX;
using namespace fft_20xx::c;
extern "C"
{
  void fft_cplx_f32_sse2(const fft_table_cplx_f32 *table, const float *input, float *output, size_t fftLength);
  void ifft_cplx_f32_sse2(const fft_table_cplx_f32 *table, const float *input, float *output, size_t fftLength);
  void fft_real_f32_sse2(const fft_table_real_f32 *table, const float *input, float *output, size_t fftLength);
  void ifft_real_f32_sse2(const fft_table_real_f32 *table, const float *input, float *output, size_t fftLength);

  void fft_cplx_f32_avx(const fft_table_cplx_f32 *table, const float *input, float *output, size_t fftLength)
  {
    if (fftLength < 64)
      { return fft_cplx_f32_sse2(table, input, output, fftLength); }
    COMMON_ASSERTS();
    AVX32::RunComplex<Dir::Forward>(GetTableData(table), input, output, fftLength);
  }


  void ifft_cplx_f32_avx(const fft_table_cplx_f32 *table, const float *input, float *output, size_t fftLength)
  {
    if (fftLength < 64)
      { return ifft_cplx_f32_sse2(table, input, output, fftLength); }
    COMMON_ASSERTS();
    AVX32::RunComplex<Dir::Inverse, Alignment::Aligned>(GetTableData(table), input, output, fftLength);
  }


  void fft_real_f32_avx(const fft_table_real_f32 *table, const float *input, float *output, size_t fftLength)
  {
    // Do this test before COMMON_ASSERTS since the called function will do those
    if (fftLength < AVX32::MinimumRealFFTLength)
    {
      // AVX's 8-long vectors are too large to make these nice so fall back on SSE2.
      return fft_real_f32_sse2(table, input, output, fftLength);
    }

    COMMON_ASSERTS();
    AVX32::RealForwardFFT(GetTableData(table), input, output, fftLength);
  }


  void ifft_real_f32_avx(const fft_table_real_f32 *table, const float *input, float *output, size_t fftLength)
  {
    if (fftLength < AVX32::MinimumRealFFTLength)
    {
      // AVX's 8-long vectors are too large to make these nice so fall back on SSE2.
      return ifft_real_f32_sse2(table, input, output, fftLength);
    }

    COMMON_ASSERTS();
    AVX32::RealInverseFFT(GetTableData(table), input, output, fftLength);
  }
}
