#include "fft_20xx_common.h"

#include <concepts>
#include <cinttypes>



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



  static_assert(!SIMD_AVX);



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
