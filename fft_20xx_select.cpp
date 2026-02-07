#include "include/fft_20xx.h"

#include "fft_20xx_common.h"

using namespace FFT20XX;

extern "C"
{


  void FFTComplex32Interleaved(const FFTTable32 *table, const float *input, float *output, size_t fftLength)
  {
    // !!! choose based on CPUID
    FFTComplex32Interleaved_FMA(table, input, output, fftLength);
  }

  void IFFTComplex32Interleaved(const FFTTable32 *table, const float *input, float *output, size_t fftLength)
  {
    // !!! choose based on CPUID
    IFFTComplex32Interleaved_FMA(table, input, output, fftLength);
  }

  void FFTComplex64Interleaved(const FFTTable64 *table, const double *input, double *output, size_t fftLength)
  {
    // !!! choose based on CPUID
    FFTComplex64Interleaved_FMA(table, input, output, fftLength);
  }

  void IFFTComplex64Interleaved(const FFTTable64 *table, const double *input, double *output, size_t fftLength)
  {
    // !!! choose based on CPUID
    IFFTComplex64Interleaved_FMA(table, input, output, fftLength);
  }



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
}
