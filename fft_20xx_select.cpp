#include "include/fft_20xx.h"

#include "fft_20xx_common.h"

namespace FFT20XX::Scalar
{
  template <std::floating_point F>
  void ForwardFFT(const Complex<F> *table, const Complex<F> *input, Complex<F> *output, std::size_t fftLength);

  template <std::floating_point F>
  void InverseFFT(const Complex<F> *table, const Complex<F> *input, Complex<F> *output, std::size_t fftLength);
}


extern "C"
{
  void FFTComplex32Interleaved(const FFTTable32 *table, const float *input, float *output, size_t fftLength)
  {
    // !!! Choose based on CPUID
    using namespace FFT20XX;

    assert(table != nullptr);
    assert(input != nullptr);
    assert(output != nullptr);
    assert(IsPowerOfTwo(fftLength));
    auto t = reinterpret_cast<const FFTTableImpl<float> *>(table);

    assert(t->MaxFFTLength() >= fftLength);

    const auto *inCplx = reinterpret_cast<const Complex<float> *>(input);
    auto *outCplx = reinterpret_cast<Complex<float> *>(output);
    Scalar::ForwardFFT<float>(t->Data(), inCplx, outCplx, fftLength);
  }

  void FFTComplex64Interleaved(const FFTTable64 *table, const double *input, double *output, size_t fftLength)
  {
    // !!! Choose based on CPUID
    using namespace FFT20XX;

    assert(table != nullptr);
    assert(input != nullptr);
    assert(output != nullptr);
    assert(IsPowerOfTwo(fftLength));
    auto t = reinterpret_cast<const FFTTableImpl<double> *>(table);

    assert(t->MaxFFTLength() >= fftLength);

    const auto *inCplx = reinterpret_cast<const Complex<double> *>(input);
    auto *outCplx = reinterpret_cast<Complex<double> *>(output);
    Scalar::ForwardFFT<double>(t->Data(), inCplx, outCplx, fftLength);
  }


  void IFFTComplex32Interleaved(const FFTTable32 *table, const float *input, float *output, size_t fftLength)
  {
    // !!! Choose based on CPUID
    using namespace FFT20XX;

    assert(table != nullptr);
    assert(input != nullptr);
    assert(output != nullptr);
    assert(IsPowerOfTwo(fftLength));
    auto t = reinterpret_cast<const FFTTableImpl<float> *>(table);

    assert(t->MaxFFTLength() >= fftLength);

    const auto *inCplx = reinterpret_cast<const Complex<float> *>(input);
    auto *outCplx = reinterpret_cast<Complex<float> *>(output);
    Scalar::InverseFFT<float>(t->Data(), inCplx, outCplx, fftLength);
  }

  void IFFTComplex64Interleaved(const FFTTable64 *table, const double *input, double *output, size_t fftLength)
  {
    // !!! Choose based on CPUID
    using namespace FFT20XX;

    assert(table != nullptr);
    assert(input != nullptr);
    assert(output != nullptr);
    assert(IsPowerOfTwo(fftLength));
    auto t = reinterpret_cast<const FFTTableImpl<double> *>(table);

    assert(t->MaxFFTLength() >= fftLength);

    const auto *inCplx = reinterpret_cast<const Complex<double> *>(input);
    auto *outCplx = reinterpret_cast<Complex<double> *>(output);
    Scalar::InverseFFT<double>(t->Data(), inCplx, outCplx, fftLength);
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
}
