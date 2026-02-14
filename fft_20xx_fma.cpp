#include "fft_20xx_common.h"


static_assert(CPU_X86 || CPU_X64);

static_assert(SIMD_AVX);
static_assert(SIMD_FMA);

#include <concepts>
#include <cinttypes>
#include <immintrin.h>


namespace FFT20XX::Scalar
{
  template <Dir D, std::floating_point F>
  void RunFFT(const Complex<F> *table, const Complex<F> *input, Complex<F> *output, std::size_t fftLength);

  template <std::floating_point F>
  void RunRealFwdFFT(const FFTTableImpl<F> *, const F *, F *, std::size_t);
}

namespace FFT20XX::FMA
{
  #if COMPILER_GCC
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wignored-attributes"
  #endif

  template <std::floating_point F>
  using SIMDNativeType = std::conditional_t<std::same_as<std::remove_cvref_t<F>, float>, __m256, __m256d>;

  template <std::floating_point F>
  class SIMDComplex
  {
  public:
    using N = SIMDNativeType<F>;

    SIMDComplex() = default;
    SIMDComplex(N n) :nat(n) {}

    operator N() const { return nat; }

    explicit FORCE_INLINE SIMDComplex(Complex<F> c)
    {
      if constexpr (std::same_as<F, float>)
        { nat = _mm256_castpd_ps(_mm256_set1_pd(std::bit_cast<double>(c))); }
      else
        { nat = _mm256_set_pd(c.i, c.r, c.i, c.r); }
    }

    friend FORCE_INLINE auto operator+(SIMDComplex a, SIMDComplex b) -> SIMDComplex
    {
      if constexpr (std::same_as<F, float>)
        { return _mm256_add_ps(a, b); }
      else
        { return _mm256_add_pd(a, b); }
    }

    friend FORCE_INLINE auto operator-(SIMDComplex a, SIMDComplex b) -> SIMDComplex
    {
      if constexpr (std::same_as<F, float>)
        { return _mm256_sub_ps(a, b); }
      else
        { return _mm256_sub_pd(a, b); }
    }

    friend FORCE_INLINE auto operator*(SIMDComplex a, F b) -> SIMDComplex
    {
      if constexpr (std::same_as<F, float>)
        { return _mm256_mul_ps(a, _mm256_set1_ps(b)); }
      else
        { return _mm256_mul_pd(a, _mm256_set1_pd(b)); }
    }

    friend FORCE_INLINE auto operator*(SIMDComplex a, SIMDComplex b) -> SIMDComplex
    {
      if constexpr (std::same_as<F, float>)
        { return _mm256_fmaddsub_ps(b, DuplicateEven(a), _mm256_mul_ps(SwapEvenOdd(b), DuplicateOdd(a))); }
      else
        { return _mm256_fmaddsub_pd(b, DuplicateEven(a), _mm256_mul_pd(SwapEvenOdd(b), DuplicateOdd(a))); }
    }

    friend FORCE_INLINE auto MultiplyWithConjugate(SIMDComplex a, SIMDComplex b) -> SIMDComplex
    {
      if constexpr (std::same_as<F, float>)
        { return _mm256_fmsubadd_ps(SwapEvenOdd(b), DuplicateOdd(a), _mm256_mul_ps(b, DuplicateEven(a))); }
      else
        { return _mm256_fmsubadd_pd(SwapEvenOdd(b), DuplicateOdd(a), _mm256_mul_pd(b, DuplicateEven(a))); }
    }

    template <Dir D>
    friend FORCE_INLINE auto Rot45(SIMDComplex a) -> SIMDComplex
    {
      if constexpr (std::same_as<F, float>)
      {
        if constexpr (D == Dir::Forward)
          { return _mm256_mul_ps(_mm256_set1_ps(InvSqrt2<float>), _mm256_fmsubadd_ps(_mm256_set1_ps(1.0f), a, SwapEvenOdd(a)));  }
        else
          { return _mm256_mul_ps(_mm256_set1_ps(InvSqrt2<float>), _mm256_addsub_ps(a, SwapEvenOdd(a)));  }
      }
      else
      {
        if constexpr (D == Dir::Forward)
          { return _mm256_mul_pd(_mm256_set1_pd(InvSqrt2<double>), _mm256_fmsubadd_pd(_mm256_set1_pd(1.0), a, SwapEvenOdd(a))); }
        else
          { return _mm256_mul_pd(_mm256_set1_pd(InvSqrt2<double>), _mm256_addsub_pd(a, SwapEvenOdd(a))); }
      }
    }

    template <Dir D>
    friend FORCE_INLINE auto Rot90(SIMDComplex a) -> SIMDComplex
    {
      if constexpr (std::same_as<F, float>)
      {
        if constexpr (D == Dir::Forward)
          { return SwapEvenOdd(_mm256_addsub_ps(_mm256_setzero_ps(), a)); }
        else
          { return _mm256_addsub_ps(_mm256_setzero_ps(), SwapEvenOdd(a)); }
      }
      else
      {
        if constexpr (D == Dir::Forward)
          { return SwapEvenOdd(_mm256_addsub_pd(_mm256_setzero_pd(), a)); }
        else
          { return _mm256_addsub_pd(_mm256_setzero_pd(), SwapEvenOdd(a)); }
      }
    }


    template <Dir D>
    friend FORCE_INLINE auto Rot135(SIMDComplex a) -> SIMDComplex
    {
      if constexpr (std::same_as<F, float>)
      {
        if constexpr (D == Dir::Forward)
          { return _mm256_mul_ps(_mm256_set1_ps(-InvSqrt2<float>), _mm256_addsub_ps(a, SwapEvenOdd(a)));  }
        else
          { return _mm256_mul_ps(_mm256_set1_ps(-InvSqrt2<float>), _mm256_fmsubadd_ps(_mm256_set1_ps(1.0f), a, SwapEvenOdd(a)));  }
      }
      else
      {
        if constexpr (D == Dir::Forward)
          { return _mm256_mul_pd(_mm256_set1_pd(-InvSqrt2<double>), _mm256_addsub_pd(a, SwapEvenOdd(a)));  }
        else
          { return _mm256_mul_pd(_mm256_set1_pd(-InvSqrt2<double>), _mm256_fmsubadd_pd(_mm256_set1_pd(1.0f), a, SwapEvenOdd(a)));  }
      }
    }


    template <Dir D>
    friend FORCE_INLINE auto Rot(SIMDComplex a, SIMDComplex w) -> SIMDComplex
    {
      if constexpr (std::same_as<F, float>)
      {
        if (D == Dir::Forward)
          { return a * w; }
        else
          { return MultiplyWithConjugate(a, w); }
      }
      else
      {
        if (D == Dir::Forward)
          { return a * w; }
        else
          { return MultiplyWithConjugate(a, w); }
      }
    }

  private:
    N nat;

    static FORCE_INLINE auto SwapEvenOdd(N v)
    {
      if constexpr (std::same_as<F, float>)
        { return _mm256_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1)); }
      else
        { return _mm256_shuffle_pd(v, v, 0b0101); }
    }

    static FORCE_INLINE auto DuplicateEven(N v)
    {
      if constexpr (std::same_as<F, float>)
        { return _mm256_shuffle_ps(v, v, _MM_SHUFFLE(2, 2, 0, 0)); }
      else
        { return _mm256_shuffle_pd(v, v, 0); }
    }

    static FORCE_INLINE auto DuplicateOdd(N v)
    {
      if constexpr (std::same_as<F, float>)
        { return _mm256_shuffle_ps(v, v, _MM_SHUFFLE(3, 3, 1, 1)); }
      else
        { return _mm256_shuffle_pd(v, v, 0b1111); }
    }
  };


  #if COMPILER_GCC
    #pragma GCC diagnostic pop
  #endif

  template <std::floating_point F>
  constexpr auto SIMDScalarCount = std::same_as<std::remove_cvref_t<F>, float> ? 8u : 4u;

  template <std::floating_point F>
  constexpr auto SIMDComplexCount = SIMDScalarCount<F> / 2;


  template <std::floating_point F>
  FORCE_INLINE auto LoadComplex(const Complex<F> *in) -> SIMDComplex<F>
  {
    if constexpr (std::same_as<F, float>)
      { return _mm256_loadu_ps(reinterpret_cast<const float *>(in)); }
    else
      { return _mm256_loadu_pd(reinterpret_cast<const double *>(in)); }
  }

  template <std::floating_point F>
  FORCE_INLINE auto StoreComplex(Complex<F> *out, SIMDComplex<F> v)
  {
    if constexpr (std::same_as<F, float>)
      { _mm256_storeu_ps(reinterpret_cast<float *>(out), v); }
    else
      { _mm256_storeu_pd(reinterpret_cast<double *>(out), v); }
  }

  template <std::floating_point F>
  FORCE_INLINE void LoadContiguousRadix4Set(
    const Complex<F> *input,
    SIMDComplex<F> &x0,
    SIMDComplex<F> &x1,
    SIMDComplex<F> &x2,
    SIMDComplex<F> &x3)
  {
    if constexpr (std::same_as<F, float>)
    {

      // This is effectively a 4x4 complex matrix transpose from memory - memory layout is
      //    x00 x10 x20 x30 | x01 x11 x21 x31 | x02 x12 x22 x32 | x03 x13 x23 x33
      // Start by doing a series of 128-bit loads to effectively swap the upper right and lower-left quadrants
      //  of the matrix.
      // Note that I tested doing 256-bit loads and doing the swaps using _mm256_permute2f128_ps and that was slightly
      //  slower.
      auto x00x10x02x12 = _mm256_insertf128_ps(
        _mm256_castps128_ps256(_mm_loadu_ps(reinterpret_cast<const float *>(&input[0]))),
        _mm_loadu_ps(reinterpret_cast<const float *>(&input[8])),
        1);
      auto x20x30x22x32 = _mm256_insertf128_ps(
        _mm256_castps128_ps256(_mm_loadu_ps(reinterpret_cast<const float *>(&input[2]))),
        _mm_loadu_ps(reinterpret_cast<const float *>(&input[10])),
        1);
      auto x01x11x03x13 = _mm256_insertf128_ps(
        _mm256_castps128_ps256(_mm_loadu_ps(reinterpret_cast<const float *>(&input[4]))),
        _mm_loadu_ps(reinterpret_cast<const float *>(&input[12])),
        1);
      auto x21x31x23x33 = _mm256_insertf128_ps(
        _mm256_castps128_ps256(_mm_loadu_ps(reinterpret_cast<const float *>(&input[6]))),
        _mm_loadu_ps(reinterpret_cast<const float *>(&input[14])),
        1);

      // Now, shuffle around the remaining elements, intentionally swapping x1 and x2.
      x0 = _mm256_shuffle_ps(x00x10x02x12, x01x11x03x13, _MM_SHUFFLE(1, 0, 1, 0));
      x2 = _mm256_shuffle_ps(x00x10x02x12, x01x11x03x13, _MM_SHUFFLE(3, 2, 3, 2));
      x1 = _mm256_shuffle_ps(x20x30x22x32, x21x31x23x33, _MM_SHUFFLE(1, 0, 1, 0));
      x3 = _mm256_shuffle_ps(x20x30x22x32, x21x31x23x33, _MM_SHUFFLE(3, 2, 3, 2));
    }
    else
    {
      // Like with the f32 case, load the 128-bit halves independently. However, since each half is just a single
      //  complex number (2 doubles), there's no additional shuffling to be done.
      x0 = _mm256_insertf128_pd(
        _mm256_castpd128_pd256(_mm_loadu_pd(reinterpret_cast<const double *>(&input[0]))),
        _mm_loadu_pd(reinterpret_cast<const double *>(&input[4])),
        1);
      x2 = _mm256_insertf128_pd(
        _mm256_castpd128_pd256(_mm_loadu_pd(reinterpret_cast<const double *>(&input[1]))),
        _mm_loadu_pd(reinterpret_cast<const double *>(&input[5])),
        1);
      x1 = _mm256_insertf128_pd(
        _mm256_castpd128_pd256(_mm_loadu_pd(reinterpret_cast<const double *>(&input[2]))),
        _mm_loadu_pd(reinterpret_cast<const double *>(&input[6])),
        1);
      x3 = _mm256_insertf128_pd(
        _mm256_castpd128_pd256(_mm_loadu_pd(reinterpret_cast<const double *>(&input[3]))),
        _mm_loadu_pd(reinterpret_cast<const double *>(&input[7])),
        1);
    }
  }


  template <std::floating_point F>
  FORCE_INLINE void StoreContiguousRadix4Set(
    Complex<F> *output,
    SIMDComplex<F> x0,
    SIMDComplex<F> x1,
    SIMDComplex<F> x2,
    SIMDComplex<F> x3)
  {
    if constexpr (std::same_as<F, float>)
    {
      // Do the opposite of the LoadAndTranspose function
      auto x00x10x02x12 = _mm256_shuffle_ps(x0, x1, _MM_SHUFFLE(1, 0, 1, 0));
      auto x01x11x03x13 = _mm256_shuffle_ps(x0, x1, _MM_SHUFFLE(3, 2, 3, 2));
      auto x20x30x22x32 = _mm256_shuffle_ps(x2, x3, _MM_SHUFFLE(1, 0, 1, 0));
      auto x21x31x23x33 = _mm256_shuffle_ps(x2, x3, _MM_SHUFFLE(3, 2, 3, 2));

      _mm_storeu_ps(reinterpret_cast<float *>(&output[0]), _mm256_castps256_ps128(x00x10x02x12));
      _mm_storeu_ps(reinterpret_cast<float *>(&output[2]), _mm256_castps256_ps128(x20x30x22x32));
      _mm_storeu_ps(reinterpret_cast<float *>(&output[4]), _mm256_castps256_ps128(x01x11x03x13));
      _mm_storeu_ps(reinterpret_cast<float *>(&output[6]), _mm256_castps256_ps128(x21x31x23x33));
      _mm_storeu_ps(reinterpret_cast<float *>(&output[8]), _mm256_extractf128_ps(x00x10x02x12, 1));
      _mm_storeu_ps(reinterpret_cast<float *>(&output[10]), _mm256_extractf128_ps(x20x30x22x32, 1));
      _mm_storeu_ps(reinterpret_cast<float *>(&output[12]), _mm256_extractf128_ps(x01x11x03x13, 1));
      _mm_storeu_ps(reinterpret_cast<float *>(&output[14]), _mm256_extractf128_ps(x21x31x23x33, 1));
    }
    else
    {
      _mm_storeu_pd(reinterpret_cast<double *>(&output[0]), _mm256_castpd256_pd128(x0));
      _mm_storeu_pd(reinterpret_cast<double *>(&output[1]), _mm256_castpd256_pd128(x1));
      _mm_storeu_pd(reinterpret_cast<double *>(&output[2]), _mm256_castpd256_pd128(x2));
      _mm_storeu_pd(reinterpret_cast<double *>(&output[3]), _mm256_castpd256_pd128(x3));
      _mm_storeu_pd(reinterpret_cast<double *>(&output[4]), _mm256_extractf128_pd(x0, 1));
      _mm_storeu_pd(reinterpret_cast<double *>(&output[5]), _mm256_extractf128_pd(x1, 1));
      _mm_storeu_pd(reinterpret_cast<double *>(&output[6]), _mm256_extractf128_pd(x2, 1));
      _mm_storeu_pd(reinterpret_cast<double *>(&output[7]), _mm256_extractf128_pd(x3, 1));
    }
  }


  template <std::floating_point F> requires (SIMDComplexCount<F> == 2 || SIMDComplexCount<F> == 4)
  FORCE_INLINE void LoadFirstContiguousAngles(const Complex<F> *table, SIMDComplex<F> &w1, SIMDComplex<F> &w2, SIMDComplex<F> &w3)
  {
      alignas(SIMDNativeType<F>) Complex<F> w1Array[SIMDComplexCount<F>];
      alignas(SIMDNativeType<F>) Complex<F> w2Array[SIMDComplexCount<F>];
      alignas(SIMDNativeType<F>) Complex<F> w3Array[SIMDComplexCount<F>];
      w1Array[0] = { 1.0f, 0.0f };
      w2Array[0] = { 1.0f, 0.0f };
      w3Array[0] = { 1.0f, 0.0f };

      w1Array[1] = { InvSqrt2<F>, -InvSqrt2<F> };
      w2Array[1] = { 0.0f, -1.0f };
      w3Array[1] = { -InvSqrt2<F>, -InvSqrt2<F> };

    if constexpr (SIMDComplexCount<F> > 2)
    {
      // The first 2 contiguous butterflies use these angles from the table
      w1Array[2] = table[4];
      w2Array[2] = table[2];
      w3Array[2] = table[5];
      w1Array[3] = table[6];
      w2Array[3] = w2Array[2] * -II;
      w3Array[3] = table[7];
    }

    w1 = LoadComplex(w1Array);
    w2 = LoadComplex(w2Array);
    w3 = LoadComplex(w3Array);
  }


  template <std::floating_point F> requires (SIMDComplexCount<F> == 2 || SIMDComplexCount<F> == 4)
  FORCE_INLINE void LoadContiguousAngles(const Complex<F> *table, std::size_t groupIndex, SIMDComplex<F> &w1, SIMDComplex<F> &w2, SIMDComplex<F> &w3)
  {
    alignas(SIMDNativeType<F>) Complex<F> w1Array[SIMDComplexCount<F>];
    alignas(SIMDNativeType<F>) Complex<F> w2Array[SIMDComplexCount<F>];
    alignas(SIMDNativeType<F>) Complex<F> w3Array[SIMDComplexCount<F>];
    w1Array[0] = table[groupIndex << 1];
    w2Array[0] = table[groupIndex];
    w3Array[0] = table[(groupIndex << 1) + 1];

    w1Array[1] = table[(groupIndex  << 1) + 2];
    w2Array[1] = w2Array[0] * -II;
    w3Array[1] = table[(groupIndex << 1) + 3];

    if constexpr (SIMDComplexCount<F> > 2)
    {
      w1Array[2] = table[(groupIndex << 1) + 4];
      w2Array[2] = table[groupIndex + 2];
      w3Array[2] = table[(groupIndex << 1) + 5];
      w1Array[3] = table[(groupIndex  << 1) + 6];
      w2Array[3] = w2Array[2] * -II;
      w3Array[3] = table[(groupIndex << 1) + 7];
    }

    w1 = LoadComplex<F>(w1Array);
    w2 = LoadComplex<F>(w2Array);
    w3 = LoadComplex<F>(w3Array);
  }

  template <Dir D, std::floating_point F>
  FORCE_INLINE void SmallFFT(const Complex<F> *table, const Complex<F> *input, Complex<F> *output, std::size_t fftLength)
    { Scalar::RunFFT<D>(table, input, output, fftLength); }

  #include "fft_20xx_run.h"

  template <typename F>
  void RunRealFwdFFT(const FFTTableImpl<F> *table, const F *input, F *output, size_t fftLength)
  {
    //if (fftLength <= SIMDScalarCount<F>)
//      { return Scalar::RunRealFwdFFT(table, input, output, fftLength); }

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
    auto x0r = output[0];
    auto x0i = output[1];

    // Calculate the dc/nyquist/middle values up front - they're going to effectively get garbage written to them
    //  during the processing, but we can just set them to these values when we're done.
    auto dc = x0r + x0i;
    auto nyquist = x0r - x0i;
    auto middle = Conjugate(cplxOut[complexFFTLength / 2]);

    auto kRevUnshifted = 0u;
    auto shift = BitReverseShiftFromBitCount<uint32_t>(Log2Floor(complexFFTLength));

    if constexpr (std::same_as<F, float>)
    {
      auto half = _mm256_set1_ps(0.5f);

      // We're going to use these to do some permutation/shuffling to avoid unaligned loads/stores
      // TODO: could potentially switch this behavior on whether or not cplxOut is aligned or not - if it's not,
      //  we could just do this as a single loadu and be done with it.
      auto prevX0 = _mm256_loadu_ps(reinterpret_cast<float *>(&cplxOut[0]));

      // We don't actually care what the previous value of this is
      auto prevLo = _mm256_undefined_ps();

      for (auto k = 0u; k < complexFFTLength / 2; k += SIMDComplexCount<F>)
      {
        // Load the next AVX set, which includes the last complex value we want to process for this iteration, then
        //  use a permute and a shuffle to go from:
        //   (prev) A B C D | E F G H    (cur) I J K L | M N O P
        //  to:
        //   C D E F | G H I J
        auto x0Unshifted = _mm256_loadu_ps(reinterpret_cast<float *>(&cplxOut[k + SIMDComplexCount<F>]));
        auto mid = _mm256_permute2f128_ps(prevX0, x0Unshifted, 0b0010'0001);
        auto x0 = _mm256_shuffle_ps(prevX0, mid, _MM_SHUFFLE(1, 0, 3, 2));

        // Update the prev value with what we just loaded - it will contain the first 3 complex values we process next time
        //  (assuming there *is* a next time)
        prevX0 = x0Unshifted;

        // Now load from the end and reverse it
        auto x1Unflipped = _mm256_loadu_ps(reinterpret_cast<float *>(&cplxOut[complexFFTLength - k - SIMDComplexCount<F>]));

        // This could be a single permute4x64 if this were AVX2
        auto x1FlipHalves = _mm256_permute2f128_ps(x1Unflipped, x1Unflipped, 0b00'01);

        // Halving x1 here allows us to use fmsubadd and fmaddsub to do the add/subtract of x1's conjugate and it saves
        //  an extra instruction (otherwise we'd need two muls somewhere)
        auto halfX1 = _mm256_mul_ps(half, _mm256_permute_ps(x1FlipHalves, _MM_SHUFFLE(1, 0, 3, 2)));

        auto even = _mm256_fmsubadd_ps(half, x0, halfX1);

        // Load our angles
        alignas(SIMDComplex<F>) Complex<F> angles[4];
        kRevUnshifted = BitReverseIncrement(kRevUnshifted);
        angles[0] = tableData[kRevUnshifted >> shift];
        kRevUnshifted = BitReverseIncrement(kRevUnshifted);
        angles[1] = tableData[kRevUnshifted >> shift];
        kRevUnshifted = BitReverseIncrement(kRevUnshifted);
        angles[2] = tableData[kRevUnshifted >> shift];
        kRevUnshifted = BitReverseIncrement(kRevUnshifted);
        angles[3] = tableData[kRevUnshifted >> shift];
        auto a = _mm256_loadu_ps(reinterpret_cast<float *>(angles));

        // Now calculate the rotated odd value, which has a couple bits that are "backwards" from the canonical version
        //  to eliminate a couple instructions
        auto s = _mm256_fmaddsub_ps(half, x0, halfX1);
        auto oddRotated =  _mm256_fmsubadd_ps(
          a,
          _mm256_permute_ps(s, _MM_SHUFFLE(3, 3, 1, 1)),
          _mm256_mul_ps(_mm256_permute_ps(a, _MM_SHUFFLE(2, 3, 0, 1)), _mm256_permute_ps(s, _MM_SHUFFLE(2, 2, 0, 0))));

        // now we need the conjugate of oddRotated (reversed back into end-of-array ordering)
        auto conj = _mm256_addsub_ps(_mm256_setzero_ps(), _mm256_sub_ps(oddRotated, even));

        // Again, with AVX2 this would be a permute4x64
        auto conjFlipped = _mm256_permute2f128_ps(conj, conj, 0b00'01);
        auto conjReversed = _mm256_shuffle_ps(conjFlipped, conjFlipped, _MM_SHUFFLE(1, 0, 3, 2));

        //cplxOut[k] = even + oddRotated;
        auto loResult = _mm256_add_ps(even, oddRotated);

        // Similar to the load, we use a previous value (which contains the last complex calculated last time) to
        //  shuffle/permute things into place to get the correctly-aligned value to store.
        auto loMid = _mm256_permute2f128_ps(prevLo, loResult, 0b0010'0001);
        auto loFinal = _mm256_shuffle_ps(loMid, loResult, _MM_SHUFFLE(1, 0, 3, 2));
        prevLo = loResult;
        _mm256_storeu_ps(reinterpret_cast<float *>(&cplxOut[k]), loFinal);
        _mm256_storeu_ps(reinterpret_cast<float *>(&cplxOut[complexFFTLength - k - SIMDComplexCount<F>]), conjReversed);
      }

      // We corrupted these two values in the above loop, but that's okay! Set them to their correct values here.
      cplxOut[0] = { dc, nyquist };
      cplxOut[complexFFTLength / 2] = middle;
    }
    else
    {
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

      cplxOut[0] = { dc, nyquist };
      cplxOut[complexFFTLength / 2] = middle;
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

  void FFTComplex32Interleaved_FMA(const FFTTable32 *table, const float *input, float *output, size_t fftLength)
  {
    auto t = reinterpret_cast<const FFTTableImpl<float> *>(table);
    COMMON_ASSERTS();

    FMA::RunFFT<Dir::Forward, float>(
      t->Data(),
      reinterpret_cast<const Complex<float> *>(input),
      reinterpret_cast<Complex<float> *>(output),
      fftLength);
  }

  void FFTComplex64Interleaved_FMA(const FFTTable64 *table, const double *input, double *output, size_t fftLength)
  {
    auto t = reinterpret_cast<const FFTTableImpl<double> *>(table);
    COMMON_ASSERTS();
    FMA::RunFFT<Dir::Forward, double>(
      t->Data(),
      reinterpret_cast<const Complex<double> *>(input),
      reinterpret_cast<Complex<double> *>(output),
      fftLength);
  }


  void IFFTComplex32Interleaved_FMA(const FFTTable32 *table, const float *input, float *output, size_t fftLength)
  {
    auto t = reinterpret_cast<const FFTTableImpl<float> *>(table);
    COMMON_ASSERTS();

    FMA::RunFFT<Dir::Inverse, float>(
      t->Data(),
      reinterpret_cast<const Complex<float> *>(input),
      reinterpret_cast<Complex<float> *>(output),
      fftLength);
  }

  void IFFTComplex64Interleaved_FMA(const FFTTable64 *table, const double *input, double *output, size_t fftLength)
  {
    auto t = reinterpret_cast<const FFTTableImpl<double> *>(table);
    COMMON_ASSERTS();
    FMA::RunFFT<Dir::Inverse, double>(
      t->Data(),
      reinterpret_cast<const Complex<double> *>(input),
      reinterpret_cast<Complex<double> *>(output),
      fftLength);
  }


  void FFTReal32_FMA(const FFTTable32 *table, const float *input, float *output, size_t fftLength)
    { FMA::RunRealFwdFFT(reinterpret_cast<const FFTTableImpl<float> *>(table), input, output, fftLength); }

  void FFTReal64_FMA(const FFTTable64 *table, const double *input, double *output, size_t fftLength)
    { FMA::RunRealFwdFFT(reinterpret_cast<const FFTTableImpl<double> *>(table), input, output, fftLength); }

  void IFFTReal32_FMA(const FFTTable32 *table, const float *input, float *output, size_t fftLength)
    { FMA::RunRealInvFFT(reinterpret_cast<const FFTTableImpl<float> *>(table), input, output, fftLength); }

  void IFFTReal64_FMA(const FFTTable64 *table, const double *input, double *output, size_t fftLength)
    { FMA::RunRealInvFFT(reinterpret_cast<const FFTTableImpl<double> *>(table), input, output, fftLength); }
}