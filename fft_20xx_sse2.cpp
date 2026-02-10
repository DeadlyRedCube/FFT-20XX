#include "fft_20xx_common.h"


static_assert(CPU_X86 || CPU_X64);

static_assert(SIMD_SSE2);

#include <concepts>
#include <cinttypes>
#include <immintrin.h>

namespace FFT20XX::Scalar
{
  template <Dir D, std::floating_point F>
  void RunFFT(const Complex<F> *table, const Complex<F> *input, Complex<F> *output, std::size_t fftLength);

}

namespace FFT20XX::SSE2
{
  #if COMPILER_GCC
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wignored-attributes"
  #endif

  template <std::floating_point F>
  using SIMDNativeType = std::conditional_t<std::same_as<std::remove_cvref_t<F>, float>, __m128, __m128d>;

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
        { nat = _mm_castpd_ps(_mm_set1_pd(std::bit_cast<double>(c))); }
      else
        { nat = _mm_set_pd(c.i, c.r); }
    }

    friend FORCE_INLINE auto operator+(SIMDComplex a, SIMDComplex b) -> SIMDComplex
    { 
      if constexpr (std::same_as<F, float>)
        { return _mm_add_ps(a, b); }
      else
        { return _mm_add_pd(a, b); }
    }

    friend FORCE_INLINE auto operator-(SIMDComplex a, SIMDComplex b) -> SIMDComplex
    { 
      if constexpr (std::same_as<F, float>)
        { return _mm_sub_ps(a, b); }
      else
        { return _mm_sub_pd(a, b); }
    }

    friend FORCE_INLINE auto operator*(SIMDComplex a, F b) -> SIMDComplex
    {
      if constexpr (std::same_as<F, float>)
        { return _mm_mul_ps(a, _mm_set1_ps(b)); }
      else
        { return _mm_mul_pd(a, _mm_set1_pd(b)); }
    }

    friend FORCE_INLINE auto operator*(SIMDComplex a, SIMDComplex b) -> SIMDComplex
    {
      // ar * br - ai * bi, ar*bi + br*ai
      if constexpr (std::same_as<F, float>)
        { return _mm_sub_ps(_mm_mul_ps(b, DuplicateEven(a)), NegateOddElements(_mm_mul_ps(SwapEvenOdd(b), DuplicateOdd(a)))); }
      else
        { return _mm_sub_pd(_mm_mul_pd(b, DuplicateEven(a)), NegateOddElements(_mm_mul_pd(SwapEvenOdd(b), DuplicateOdd(a)))); }
    }

    friend FORCE_INLINE auto MultiplyWithConjugate(SIMDComplex a, SIMDComplex b) -> SIMDComplex
    {
      if constexpr (std::same_as<F, float>)
        { return _mm_add_ps(_mm_mul_ps(SwapEvenOdd(b), DuplicateOdd(a)), NegateOddElements(_mm_mul_ps(b, DuplicateEven(a)))); }
      else
        { return _mm_add_pd(_mm_mul_pd(SwapEvenOdd(b), DuplicateOdd(a)), NegateOddElements(_mm_mul_pd(b, DuplicateEven(a)))); }
    }

    template <Dir D>
    friend FORCE_INLINE auto Rot45(SIMDComplex a) -> SIMDComplex
    {  
      if constexpr (std::same_as<F, float>)
      {
        if constexpr (D == Dir::Forward)
          { return _mm_mul_ps(_mm_set1_ps(InvSqrt2<float>), _mm_add_ps(a, NegateOddElements(SwapEvenOdd(a))));  }
        else
          { return _mm_mul_ps(_mm_set1_ps(InvSqrt2<float>), _mm_sub_ps(a, NegateOddElements(SwapEvenOdd(a))));  }
      }
      else
      {
        if constexpr (D == Dir::Forward)
          { return _mm_mul_pd(_mm_set1_pd(InvSqrt2<double>), _mm_add_pd(a, NegateOddElements(SwapEvenOdd(a)))); }
        else
          { return _mm_mul_pd(_mm_set1_pd(InvSqrt2<double>), _mm_sub_pd(a, NegateOddElements(SwapEvenOdd(a)))); }
      }
    }

    template <Dir D>
    friend FORCE_INLINE auto Rot90(SIMDComplex a) -> SIMDComplex
    { 
      if constexpr (D == Dir::Forward)
        { return NegateOddElements(SwapEvenOdd(a)); }
      else
        { return SwapEvenOdd(NegateOddElements(a)); }
    }


    template <Dir D>
    friend FORCE_INLINE auto Rot135(SIMDComplex a) -> SIMDComplex
    { 
      if constexpr (std::same_as<F, float>)
      {
        if constexpr (D == Dir::Forward)
          { return _mm_mul_ps(_mm_set1_ps(-InvSqrt2<float>), _mm_sub_ps(a, NegateOddElements(SwapEvenOdd(a))));  }
        else
          { return _mm_mul_ps(_mm_set1_ps(-InvSqrt2<float>), _mm_add_ps(a, NegateOddElements(SwapEvenOdd(a))));  }
      }
      else
      {
        if constexpr (D == Dir::Forward)
          { return _mm_mul_pd(_mm_set1_pd(-InvSqrt2<double>), _mm_sub_pd(a, NegateOddElements(SwapEvenOdd(a))));  }
        else
          { return _mm_mul_pd(_mm_set1_pd(-InvSqrt2<double>), _mm_add_pd(a, NegateOddElements(SwapEvenOdd(a))));  }
      }
    }


    template <Dir D>
    friend FORCE_INLINE auto Rot(SIMDComplex a, SIMDComplex w) -> SIMDComplex
    { 
      if (D == Dir::Forward)
        { return a * w; }
      else
        { return MultiplyWithConjugate(a, w); }
    }

  private:
    N nat;
    static FORCE_INLINE auto NegateOddElements(N v) -> SIMDComplex
    {
      if constexpr (std::same_as<F, float>)
        { return _mm_xor_ps(v, _mm_castpd_ps(_mm_set1_pd(-0.0))); }
      else
        { return _mm_xor_pd(v, _mm_set_pd(-0.0, 0.0)); }
    }


    static FORCE_INLINE auto SwapEvenOdd(N v)
    { 
      if constexpr (std::same_as<F, float>)
        { return _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1)); }
      else
        { return _mm_shuffle_pd(v, v, 0b01); }
    }

    static FORCE_INLINE auto DuplicateEven(N v)
    { 
      if constexpr (std::same_as<F, float>)
        { return _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 2, 0, 0)); }
      else
        { return _mm_shuffle_pd(v, v, 0); }
    }

    static FORCE_INLINE auto DuplicateOdd(N v)
    {
      if constexpr (std::same_as<F, float>)
        { return _mm_shuffle_ps(v, v, _MM_SHUFFLE(3, 3, 1, 1)); }
      else
        { return _mm_shuffle_pd(v, v, 0b11); }
    }
  };


  #if COMPILER_GCC
    #pragma GCC diagnostic pop
  #endif

  template <std::floating_point F>
  constexpr auto SIMDScalarCount = std::same_as<std::remove_cvref_t<F>, float> ? 4u : 2u;

  template <std::floating_point F>
  constexpr auto SIMDComplexCount = SIMDScalarCount<F> / 2;


  template <std::floating_point F>
  FORCE_INLINE auto LoadComplex(const Complex<F> *in) -> SIMDComplex<F>
  { 
    if constexpr (std::same_as<F, float>)
      { return _mm_loadu_ps(reinterpret_cast<const float *>(in)); }
    else
      { return _mm_loadu_pd(reinterpret_cast<const double *>(in)); }
  }

  template <std::floating_point F>
  FORCE_INLINE auto StoreComplex(Complex<F> *out, SIMDComplex<F> v)
  { 
    if constexpr (std::same_as<F, float>)
      { _mm_storeu_ps(reinterpret_cast<float *>(out), v); }
    else
      { _mm_storeu_pd(reinterpret_cast<double *>(out), v); }
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
      auto x00x10 = _mm_load_ps(reinterpret_cast<const float *>(&input[0]));
      auto x20x30 = _mm_load_ps(reinterpret_cast<const float *>(&input[2]));
      auto x01x11 = _mm_load_ps(reinterpret_cast<const float *>(&input[4]));
      auto x21x31 = _mm_load_ps(reinterpret_cast<const float *>(&input[6]));

      x0 = _mm_shuffle_ps(x00x10, x01x11, _MM_SHUFFLE(1, 0, 1, 0));
      x2 = _mm_shuffle_ps(x00x10, x01x11, _MM_SHUFFLE(3, 2, 3, 2));
      x1 = _mm_shuffle_ps(x20x30, x21x31, _MM_SHUFFLE(1, 0, 1, 0));
      x3 = _mm_shuffle_ps(x20x30, x21x31, _MM_SHUFFLE(3, 2, 3, 2));
    }
    else
    {
      x0 = _mm_load_pd(reinterpret_cast<const double *>(&input[0]));
      x2 = _mm_load_pd(reinterpret_cast<const double *>(&input[1]));
      x1 = _mm_load_pd(reinterpret_cast<const double *>(&input[2]));
      x3 = _mm_load_pd(reinterpret_cast<const double *>(&input[3]));
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
      _mm_store_ps(reinterpret_cast<float *>(&output[0]), _mm_shuffle_ps(x0, x1, _MM_SHUFFLE(1, 0, 1, 0)));
      _mm_store_ps(reinterpret_cast<float *>(&output[2]), _mm_shuffle_ps(x2, x3, _MM_SHUFFLE(1, 0, 1, 0)));
      _mm_store_ps(reinterpret_cast<float *>(&output[4]), _mm_shuffle_ps(x0, x1, _MM_SHUFFLE(3, 2, 3, 2)));
      _mm_store_ps(reinterpret_cast<float *>(&output[6]), _mm_shuffle_ps(x2, x3, _MM_SHUFFLE(3, 2, 3, 2)));
    }
    else
    {
      _mm_store_pd(reinterpret_cast<double *>(&output[0]), x0);
      _mm_store_pd(reinterpret_cast<double *>(&output[1]), x1);
      _mm_store_pd(reinterpret_cast<double *>(&output[2]), x2);
      _mm_store_pd(reinterpret_cast<double *>(&output[3]), x3);
    }
  }


  template <std::floating_point F>
  FORCE_INLINE void LoadFirstContiguousAngles(const Complex<F> *, SIMDComplex<F> &w1, SIMDComplex<F> &w2, SIMDComplex<F> &w3)
  {
      alignas(SIMDNativeType<F>) Complex<F> w1Array[SIMDComplexCount<F>];
      alignas(SIMDNativeType<F>) Complex<F> w2Array[SIMDComplexCount<F>];
      alignas(SIMDNativeType<F>) Complex<F> w3Array[SIMDComplexCount<F>];
      w1Array[0] = { 1.0f, 0.0f };
      w2Array[0] = { 1.0f, 0.0f };
      w3Array[0] = { 1.0f, 0.0f };

    if constexpr (SIMDComplexCount<F> > 1)
    {
      w1Array[1] = { InvSqrt2<F>, -InvSqrt2<F> };
      w2Array[1] = { 0.0f, -1.0f };
      w3Array[1] = { -InvSqrt2<F>, -InvSqrt2<F> };
    }

    w1 = LoadComplex(w1Array);
    w2 = LoadComplex(w2Array);
    w3 = LoadComplex(w3Array);
  }


  template <std::floating_point F>
  FORCE_INLINE void LoadContiguousAngles(const Complex<F> *table, std::size_t groupIndex, SIMDComplex<F> &w1, SIMDComplex<F> &w2, SIMDComplex<F> &w3)
  {
    alignas(SIMDNativeType<F>) Complex<F> w1Array[SIMDComplexCount<F>];
    alignas(SIMDNativeType<F>) Complex<F> w2Array[SIMDComplexCount<F>];
    alignas(SIMDNativeType<F>) Complex<F> w3Array[SIMDComplexCount<F>];
    w1Array[0] = table[groupIndex << 1];
    w2Array[0] = table[groupIndex];
    w3Array[0] = table[(groupIndex << 1) + 1];

    if constexpr (SIMDComplexCount<F> > 1)
    {
      w1Array[1] = table[(groupIndex  << 1) + 2];
      w2Array[1] = w2Array[0] * -II;
      w3Array[1] = table[(groupIndex << 1) + 3];
    }

    w1 = LoadComplex<F>(w1Array);
    w2 = LoadComplex<F>(w2Array);
    w3 = LoadComplex<F>(w3Array);
  }

  template <Dir D, std::floating_point F>
  FORCE_INLINE void SmallFFT(const Complex<F> *table, const Complex<F> *input, Complex<F> *output, std::size_t fftLength)
    { Scalar::RunFFT<D, F>(table, input, output, fftLength); }

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

  void FFTComplex32Interleaved_SSE2(const FFTTable32 *table, const float *input, float *output, size_t fftLength)
  {
    auto t = reinterpret_cast<const FFTTableImpl<float> *>(table);
    COMMON_ASSERTS();

    SSE2::RunFFT<Dir::Forward, float>(
      t->Data(), 
      reinterpret_cast<const Complex<float> *>(input), 
      reinterpret_cast<Complex<float> *>(output), 
      fftLength);
  }

  void FFTComplex64Interleaved_SSE2(const FFTTable64 *table, const double *input, double *output, size_t fftLength)
  {
    auto t = reinterpret_cast<const FFTTableImpl<double> *>(table);
    COMMON_ASSERTS();
    SSE2::RunFFT<Dir::Forward, double>(
      t->Data(), 
      reinterpret_cast<const Complex<double> *>(input), 
      reinterpret_cast<Complex<double> *>(output), 
      fftLength);
  }


  void IFFTComplex32Interleaved_SSE2(const FFTTable32 *table, const float *input, float *output, size_t fftLength)
  {
    auto t = reinterpret_cast<const FFTTableImpl<float> *>(table);
    COMMON_ASSERTS();

    SSE2::RunFFT<Dir::Inverse, float>(
      t->Data(), 
      reinterpret_cast<const Complex<float> *>(input), 
      reinterpret_cast<Complex<float> *>(output), 
      fftLength);
  }

  void IFFTComplex64Interleaved_SSE2(const FFTTable64 *table, const double *input, double *output, size_t fftLength)
  {
    auto t = reinterpret_cast<const FFTTableImpl<double> *>(table);
    COMMON_ASSERTS();
    SSE2::RunFFT<Dir::Inverse, double>(
      t->Data(), 
      reinterpret_cast<const Complex<double> *>(input), 
      reinterpret_cast<Complex<double> *>(output), 
      fftLength);
  }
}