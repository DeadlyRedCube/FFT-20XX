#include "fft_20xx_common.h"
#include "fft_20xx_table.h"


extern "C"
{
  namespace fft_20xx::c
  {
    const fft_table_f32 *allocate_fft_table_f32(size_t maxFFTLength)
      { return reinterpret_cast<fft_table_f32 *>(new FFT20XX::FFTTableImpl<float>(maxFFTLength)); }

    void free_fft_table_f32(const fft_table_f32 *table)
      { delete reinterpret_cast<const FFT20XX::FFTTableImpl<float> *>(table); }

    const fft_table_f64 *allocate_fft_table_f64(size_t maxFFTLength)
      { return reinterpret_cast<fft_table_f64 *>(new FFT20XX::FFTTableImpl<double>(maxFFTLength)); }

    void free_fft_table_f64(const fft_table_f64 *table)
      { delete reinterpret_cast<const FFT20XX::FFTTableImpl<double> *>(table); }
  }
}
