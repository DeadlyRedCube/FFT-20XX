#include "fft_20xx_common.h"

#include <concepts>
#include <cinttypes>



#include <vector>

namespace FFT20XX::Scalar
{
  template <std::floating_point F>
  constexpr auto SIMDScalarCount = 2u;
  
  template <std::floating_point F>
  constexpr auto SIMDComplexCount = 1u;

  template <std::floating_point F>
  using SIMDComplex = Complex<F>;

    template <std::floating_point F>
  FORCE_INLINE auto LoadComplex(const Complex<F> *in) -> SIMDComplex<F>
    { return *in; }

  template <std::floating_point F>
  FORCE_INLINE void StoreComplex(Complex<F> *out, SIMDComplex<F> v)
    { *out = v; }



  #include "fft_20xx_run.h"


  template <std::floating_point F>
  void ForwardFFT(const Complex<F> *table, const Complex<F> *input, Complex<F> *output, std::size_t fftLength)
    { RunFFT<Dir::Forward, F, 1>(table, input, output, fftLength); }

  template void ForwardFFT(const Complex<float> *table, const Complex<float> *input, Complex<float> *output, std::size_t fftLength);
  template void ForwardFFT(const Complex<double> *table, const Complex<double> *input, Complex<double> *output, std::size_t fftLength);

  template <std::floating_point F>
  void InverseFFT(const Complex<F> *table, const Complex<F> *input, Complex<F> *output, std::size_t fftLength)
    { RunFFT<Dir::Inverse, F, 1>(table, input, output, fftLength); }

  template void InverseFFT(const Complex<float> *table, const Complex<float> *input, Complex<float> *output, std::size_t fftLength);
  template void InverseFFT(const Complex<double> *table, const Complex<double> *input, Complex<double> *output, std::size_t fftLength);
}

using namespace FFT20XX;

extern "C"
{
  #define COMMON_ASSERTS() \
    assert(table != nullptr); \
    assert(input != nullptr); \
    assert(output != nullptr); \
    assert(IsPowerOfTwo(fftLength)); \
    assert(t->MaxFFTLength() >= fftLength)

  void FFTComplex32Interleaved_Scalar(const FFTTable32 *table, const float *input, float *output, size_t fftLength)
  {
    [[maybe_unused]] auto t = reinterpret_cast<const FFTTableImpl<float> *>(table);
    COMMON_ASSERTS();

    Scalar::ForwardFFT<float>(
      t->Data(),
      reinterpret_cast<const Complex<float> *>(input), 
      reinterpret_cast<Complex<float> *>(output), 
      fftLength);
  }

  void FFTComplex64Interleaved_Scalar(const FFTTable64 *table, const double *input, double *output, size_t fftLength)
  {
    [[maybe_unused]] auto t = reinterpret_cast<const FFTTableImpl<double> *>(table);
    COMMON_ASSERTS();
    Scalar::ForwardFFT<double>(
      t->Data(),
      reinterpret_cast<const Complex<double> *>(input), 
      reinterpret_cast<Complex<double> *>(output), 
      fftLength);
  }


  void IFFTComplex32Interleaved_Scalar(const FFTTable32 *table, const float *input, float *output, size_t fftLength)
  {
    [[maybe_unused]] auto t = reinterpret_cast<const FFTTableImpl<float> *>(table);
    COMMON_ASSERTS();

    Scalar::InverseFFT<float>(
      t->Data(),
      reinterpret_cast<const Complex<float> *>(input), 
      reinterpret_cast<Complex<float> *>(output), 
      fftLength);
  }

  void IFFTComplex64Interleaved_Scalar(const FFTTable64 *table, const double *input, double *output, size_t fftLength)
  {
    [[maybe_unused]] auto t = reinterpret_cast<const FFTTableImpl<double> *>(table);
    COMMON_ASSERTS();
    Scalar::InverseFFT<double>(
      t->Data(),
      reinterpret_cast<const Complex<double> *>(input), 
      reinterpret_cast<Complex<double> *>(output), 
      fftLength);
  }
}