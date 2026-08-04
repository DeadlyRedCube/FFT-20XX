#ifdef _MSC_VER
  #pragma warning (disable: 4711) // function <x> selected for automatic inline expansion
#endif

#include "fft_20xx_common.h"
#include "fft_20xx_table.h"
#include "fft_20xx_tiny_cplx.h"
#include "fft_20xx_tiny_real.h"

static_assert(CPU_X86 || CPU_X64);

static_assert(SIMD_AVX && !SIMD_FMA);

using namespace FFT20XX;

namespace FFT20XX::AVX64
{
  #include "fft_20xx_avx_f64_common.h"
}


using namespace FFT20XX;
using namespace fft_20xx::c;
extern "C"
{
  void fft_cplx_f64_avx(const fft_table_cplx_f64 *table, const double *input, double *output, size_t fftLength)
  {
    COMMON_ASSERTS();
    AVX64::RunComplex<Dir::Forward>(GetTableData(table), input, output, fftLength);
  }


  void ifft_cplx_f64_avx(const fft_table_cplx_f64 *table, const double *input, double *output, size_t fftLength)
  {
    COMMON_ASSERTS();
    AVX64::RunComplex<Dir::Inverse>(GetTableData(table), input, output, fftLength);
  }

  void fft_real_f64_avx(const fft_table_real_f64 *table, const double *input, double *output, size_t fftLength)
  {
    COMMON_ASSERTS();
    AVX64::RealForwardFFT(GetTableData(table), input, output, fftLength);
  }


  void ifft_real_f64_avx(const fft_table_real_f64 *table, const double *input, double *output, size_t fftLength)
  {
    COMMON_ASSERTS();
    AVX64::RealInverseFFT(GetTableData(table), input, output, fftLength);
  }
}
