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

  template <typename F>
  void RunRealFwdFFT(const FFTTableImpl<F> *table, const F *input, F *output, size_t fftLength)
  {
    if (fftLength <= 2)
    {
      // For tiny real FFTs, all the outputs are real-valued.
      if (fftLength == 1)
      {
        output[0] = input[0];
        return;
      }

      auto a = input[0];
      auto b = input[0];
      output[0] = a + b;
      output[1] = a - b;
      return;
    }

    auto tableData = reinterpret_cast<const FFTTableImpl<F> *>(table)->Data();
    const auto complexFFTLength = fftLength / 2;

    auto cplxIn = reinterpret_cast<const Complex<F> *>(input);
    auto cplxOut = reinterpret_cast<Complex<F> *>(output);

    // A real-valued-input FFT can be calculated as an 1/2 length FFT of complex values. This takes advantage of FFTs
    //  being linear (i.e. FFT(sequenceA) + FFT(sequenceB) == FFT(elementwiseAddition(sequenceA, sequenceB))).
    // Thus, if we treat the real numbers as if every pair of them were a complex number (real number sequence
    //  [a b c d ...] pretending to be [(a + b*i), (c + d*i) ...]), the FFT of that is the same as:
    //    FFT(Even-Indexed Real Values) + FFT(i*(Odd-Indexed Real Values))
    //  (note that the odd values are multiplied by "i" to make the complex number sum).
    RunFFT<Dir::Forward>(tableData, cplxIn, cplxOut, complexFFTLength);

    // To get the true result out, then, we treat the even/odd values as if we were doing a decimation-in-time FFT.
    //  We only have half as many (complex) values as the actual length of the FFT, but we only need half (plus one) -
    //  for real-valued FFTs, the second half of the values are a conjugate mirror of the first:
    //     RealFFT(k) == Conjugate(RealFFT(N - k))
    // Two notes about this:
    //  - the indexing is modulo, which matters where k == 0 (RealFFT(N) becomes RealFFT(0))
    //  - The values at k == 0 and k == N/2 have to match their conjugates, which means they *must* be real values
    //
    // We need half + 1 output values (which is more space than in the output array), but since FFT(0) and FFT(N/2) are
    //  real numbers, we can pack them together as a special case in output[0] and output[1] respectively (they're not
    //  truly a complex pair like the rest of the output). This is pretty standard.

    // Do that now: pack the DC and Nyquist terms (0 and N/2) into the first "complex" output. This also allows us to
    //  skip k == 0 in the loop and avoid the modulo indexing issue.
    {
      auto x0r = output[0];
      auto x0i = output[1];

      output[0] = x0r + x0i;
      output[1] = x0r - x0i;
    }


    // For the rest, use the conjugate property to calculate the real values effectively from the outside in.


    auto kRevUnshifted = BitReverseIncrement(0u);
    auto shift = BitReverseShiftFromBitCount<uint32_t>(Log2Floor(complexFFTLength));

    // The middle complex output ends up just needing to be its own conjugate
    cplxOut[complexFFTLength / 2].i = - cplxOut[complexFFTLength / 2].i;

    for (auto k = 1u; k < complexFFTLength / 2; k++)
    {
      auto x0 = cplxOut[k];
      auto x1Conjugate = Conjugate(cplxOut[complexFFTLength - k]);
      auto even = (x0 + x1Conjugate) * F(0.5);
      auto odd = Rot90<Dir::Forward>((x0 - x1Conjugate) * F(0.5));

      auto kRev = kRevUnshifted >> shift;
      auto oddRotated = Rot<Dir::Forward>(odd, tableData[kRev]);
      cplxOut[k] = even + oddRotated;
      cplxOut[complexFFTLength - k] = Conjugate(even - oddRotated);

      kRevUnshifted = BitReverseIncrement(kRevUnshifted);
    }
  }

  template void RunRealFwdFFT(const FFTTableImpl<float> *, const float *, float *, size_t);
  template void RunRealFwdFFT(const FFTTableImpl<double> *, const double *, double *, size_t);


  template <typename F>
  void RunRealInvFFT(const FFTTableImpl<F> *table, const F *input, F *output, size_t fftLength)
  {
    if (fftLength <= 2)
    {
      // For tiny real FFTs, all the outputs are real-valued.
      if (fftLength == 1)
      {
        output[0] = input[0];
        return;
      }

      auto a = input[0];
      auto b = input[0];
      output[0] = a + b;
      output[1] = a - b;
      return;
    }

    auto tableData = reinterpret_cast<const FFTTableImpl<F> *>(table)->Data();
    const auto complexFFTLength = fftLength / 2;

    auto cplxIn = reinterpret_cast<const Complex<F> *>(input);
    auto cplxOut = reinterpret_cast<Complex<F> *>(output);

    auto kRevUnshifted = BitReverseIncrement(0u);
    auto shift = BitReverseShiftFromBitCount<uint32_t>(Log2Floor(complexFFTLength));

    // Undo what the real forward FFT does before we run the complex IFFT
    for (auto k = 1u; k < complexFFTLength / 2; k++)
    {
      auto x0 = cplxIn[k];
      auto x1Conjugate = Conjugate(cplxIn[complexFFTLength - k]);
      auto even = (x0 + x1Conjugate) * F(0.5);

      auto kRev = kRevUnshifted >> shift;
      auto oddUnRotated = Rot<Dir::Inverse>((x0 - x1Conjugate) * 0.5f, tableData[kRev]);
      auto odd = Rot90<Dir::Inverse>(oddUnRotated);

      cplxOut[k] = even + odd;
      cplxOut[complexFFTLength - k] = Conjugate(even - odd);

      kRevUnshifted = BitReverseIncrement(kRevUnshifted);
    }

    cplxOut[complexFFTLength / 2] = Conjugate(cplxIn[complexFFTLength / 2]);

    {
      auto x0r = input[0];
      auto x0i = input[1];


      output[0] = (x0r + x0i) * 0.5f;
      output[1] = (x0r - x0i) * 0.5f;
    }

    RunFFT<Dir::Inverse>(tableData, cplxOut, cplxOut, complexFFTLength);
  }
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

  void FFTReal32_Scalar(const FFTTable32 *table, const float *input, float *output, size_t fftLength)
    { Scalar::RunRealFwdFFT(reinterpret_cast<const FFTTableImpl<float> *>(table), input, output, fftLength); }

  void FFTReal64_Scalar(const FFTTable64 *table, const double *input, double *output, size_t fftLength)
    { Scalar::RunRealFwdFFT(reinterpret_cast<const FFTTableImpl<double> *>(table), input, output, fftLength); }

  void FFTReal32(const FFTTable32 *table, const float *input, float *output, size_t fftLength) // !!!
    { FFTReal32_Scalar(table, input, output, fftLength); }

  void FFTReal64(const FFTTable64 *table, const double *input, double *output, size_t fftLength) // !!!
    { FFTReal64_Scalar(table, input, output, fftLength); }

  void IFFTReal32_Scalar(const FFTTable32 *table, const float *input, float *output, size_t fftLength)
    { Scalar::RunRealInvFFT(reinterpret_cast<const FFTTableImpl<float> *>(table), input, output, fftLength); }

  void IFFTReal64_Scalar(const FFTTable64 *table, const double *input, double *output, size_t fftLength)
    { Scalar::RunRealInvFFT(reinterpret_cast<const FFTTableImpl<double> *>(table), input, output, fftLength); }

  void IFFTReal32(const FFTTable32 *table, const float *input, float *output, size_t fftLength) // !!!
    { IFFTReal32_Scalar(table, input, output, fftLength); }

  void IFFTReal64(const FFTTable64 *table, const double *input, double *output, size_t fftLength) // !!!
    { IFFTReal64_Scalar(table, input, output, fftLength); }
}