#ifdef _MSC_VER
  #pragma warning (disable: 4711) // function <x> selected for automatic inline expansion
#endif

#include "fft_20xx_common.h"
#include "fft_20xx_table.h"
#include "fft_20xx_tiny_cplx.h"
#include "fft_20xx_tiny_real.h"

static_assert(CPU_X86 || CPU_X64);

static_assert(SIMD_AVX && SIMD_FMA);

#if COMPILER_GCC
  #pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

using namespace FFT20XX;

namespace FFT20XX::FMA64
{
  #include "fft_20xx_avx_f64_common.h"
}


using namespace FFT20XX;
using namespace fft_20xx::c;
extern "C"
{
  void fft_cplx_f64_fma(const fft_table_cplx_f64 *table, const double *input, double *output, size_t fftLength)
  {
    COMMON_ASSERTS();
    if (FMA64::IsSIMDAligned(input) && FMA64::IsSIMDAligned(output))
      { FMA64::RunComplex<Dir::Forward, Alignment::Aligned>(GetTableData(table), input, output, fftLength); }
    else
      { FMA64::RunComplex<Dir::Forward, Alignment::Unaligned>(GetTableData(table), input, output, fftLength); }
  }


  void ifft_cplx_f64_fma(const fft_table_cplx_f64 *table, const double *input, double *output, size_t fftLength)
  {
    COMMON_ASSERTS();
    if (FMA64::IsSIMDAligned(input) && FMA64::IsSIMDAligned(output))
      { FMA64::RunComplex<Dir::Inverse, Alignment::Aligned>(GetTableData(table), input, output, fftLength); }
    else
      { FMA64::RunComplex<Dir::Inverse, Alignment::Unaligned>(GetTableData(table), input, output, fftLength); }
  }

  void fft_real_f64_fma(const fft_table_real_f64 *table, const double *input, double *output, size_t fftLength)
  {
    COMMON_ASSERTS();
    FMA64::RealForwardFFT<Alignment::Unaligned>(GetTableData(table), input, output, fftLength);
  }


  void ifft_real_f64_fma(const fft_table_real_f64 *table, const double *input, double *output, size_t fftLength)
  {
    COMMON_ASSERTS();
    FMA64::RealInverseFFT<Alignment::Unaligned>(GetTableData(table), input, output, fftLength);
  }
}
