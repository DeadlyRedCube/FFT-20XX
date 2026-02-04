#include "fft_20xx_table.h"
#include "include/fft_20xx.h"


extern "C"
{
  const FFTTable32 *AllocateFFTTable32(size_t maxFFTLength)
    { return reinterpret_cast<FFTTable32 *>(new FFT20XX::FFTTableImpl<float>(maxFFTLength)); }

  void FreeFFTTable32(const FFTTable32 *table)
    { delete reinterpret_cast<const FFT20XX::FFTTableImpl<float> *>(table); }

  const FFTTable64 *AllocateFFTTable64(size_t maxFFTLength)
    { return reinterpret_cast<FFTTable64 *>(new FFT20XX::FFTTableImpl<double>(maxFFTLength)); }

  void FreeFFTTable64(const FFTTable64 *table)
    { delete reinterpret_cast<const FFT20XX::FFTTableImpl<double> *>(table); }
}