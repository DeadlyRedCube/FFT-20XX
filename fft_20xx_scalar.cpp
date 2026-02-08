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


  template <Dir D, std::floating_point F>
  FORCE_INLINE void SmallFFT(const Complex<F> *, const Complex<F> *input, Complex<F> *output, std::size_t fftLength)
  {
    switch (fftLength)
    {
    case 1:
      output[0] = input[0];
      break;

    case 2:
      {
        auto x0 = input[0];
        auto x1 = input[1];
        output[0] = x0 + x1;
        output[1] = x0 - x1;
      }
      break;

    case 4:
      {
        auto x0 = input[0];
        auto x2 = input[1]; // As usual, intentional reverse of x1/x2
        auto x1 = input[2];
        auto x3 = input[3];
        
        auto sum02 = x0 + x2;
        auto diff02 = x0 - x2;
        auto sum13 = x1 + x3;
        auto iDiff13 = Rot90<D>(x1 - x3);
        output[0] = sum02 + sum13;
        output[1] = diff02 + iDiff13;
        output[2] = sum02 - sum13;
        output[3] = diff02 - iDiff13;
      }
      break;
    }
  }

  #include "fft_20xx_run.h"
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

    Scalar::RunFFT<Dir::Forward, float>(
      t->Data(),
      reinterpret_cast<const Complex<float> *>(input), 
      reinterpret_cast<Complex<float> *>(output), 
      fftLength);
  }

  void FFTComplex64Interleaved_Scalar(const FFTTable64 *table, const double *input, double *output, size_t fftLength)
  {
    [[maybe_unused]] auto t = reinterpret_cast<const FFTTableImpl<double> *>(table);
    COMMON_ASSERTS();
    Scalar::RunFFT<Dir::Forward, double>(
      t->Data(),
      reinterpret_cast<const Complex<double> *>(input), 
      reinterpret_cast<Complex<double> *>(output), 
      fftLength);
  }


  void IFFTComplex32Interleaved_Scalar(const FFTTable32 *table, const float *input, float *output, size_t fftLength)
  {
    [[maybe_unused]] auto t = reinterpret_cast<const FFTTableImpl<float> *>(table);
    COMMON_ASSERTS();

    Scalar::RunFFT<Dir::Inverse, float>(
      t->Data(),
      reinterpret_cast<const Complex<float> *>(input), 
      reinterpret_cast<Complex<float> *>(output), 
      fftLength);
  }

  void IFFTComplex64Interleaved_Scalar(const FFTTable64 *table, const double *input, double *output, size_t fftLength)
  {
    [[maybe_unused]] auto t = reinterpret_cast<const FFTTableImpl<double> *>(table);
    COMMON_ASSERTS();
    Scalar::RunFFT<Dir::Inverse, double>(
      t->Data(),
      reinterpret_cast<const Complex<double> *>(input), 
      reinterpret_cast<Complex<double> *>(output), 
      fftLength);
  }
}