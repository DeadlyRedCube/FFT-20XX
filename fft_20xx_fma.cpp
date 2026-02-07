#include "fft_20xx_common.h"


static_assert(CPU_X86 || CPU_X64);

static_assert(SIMD_AVX);
static_assert(SIMD_FMA);

#include <concepts>
#include <cinttypes>
#include <immintrin.h>


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
      auto a = _mm_loadu_ps(reinterpret_cast<const float *>(&input[0]));
      auto b = _mm_loadu_ps(reinterpret_cast<const float *>(&input[2]));
      auto c = _mm_loadu_ps(reinterpret_cast<const float *>(&input[4]));
      auto d = _mm_loadu_ps(reinterpret_cast<const float *>(&input[6]));
      auto e = _mm_loadu_ps(reinterpret_cast<const float *>(&input[8]));
      auto f = _mm_loadu_ps(reinterpret_cast<const float *>(&input[10]));
      auto g = _mm_loadu_ps(reinterpret_cast<const float *>(&input[12]));
      auto h = _mm_loadu_ps(reinterpret_cast<const float *>(&input[14]));

      // Now what we have is effectively:
      //  a: (x0r0 x0i0 x1r0 x1i0), b: (x2r0 x2i0 x3r0 x3i0)
      //  c: (x0r1 x0i1 x1r1 x1i1), d: (x2r1 x2i1 x3r1 x3i1)
      //  e: (x0r2 x0i2 x1r2 x1i2), f: (x2r2 x2i2 x3r2 x3i2)
      //  g: (x0r3 x0i3 x1r3 x1i3), h: (x2r3 x2i3 x3r3 x3i3)

      // We want to do a 4x4 (complex) matrix transpose on this data.
      // First thing is to combine the 128 bits things
      auto t0 = _mm256_insertf128_ps(_mm256_castps128_ps256(a), c, 1);
      auto t1 = _mm256_insertf128_ps(_mm256_castps128_ps256(b), d, 1);
      auto t2 = _mm256_insertf128_ps(_mm256_castps128_ps256(e), g, 1);
      auto t3 = _mm256_insertf128_ps(_mm256_castps128_ps256(f), h, 1);

      // Now t0 is (x0r0 x0i0 x1r0 x1i0 x0r2 x0i2 x1r2 x1i2), t1-t3 are similar.
      // Do some shuffles into the final position
      x0 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0));
      x2 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2));
      x1 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0));
      x3 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2));
    }
    else
    {
      // Load the 128-bit halves separately (makes the interleaving easier)
      auto x0Lo = _mm_loadu_pd(reinterpret_cast<const double *>(&input[0]));
      auto x2Lo = _mm_loadu_pd(reinterpret_cast<const double *>(&input[1]));
      auto x1Lo = _mm_loadu_pd(reinterpret_cast<const double *>(&input[2]));
      auto x3Lo = _mm_loadu_pd(reinterpret_cast<const double *>(&input[3]));
      auto x0Hi = _mm_loadu_pd(reinterpret_cast<const double *>(&input[4]));
      auto x2Hi = _mm_loadu_pd(reinterpret_cast<const double *>(&input[5]));
      auto x1Hi = _mm_loadu_pd(reinterpret_cast<const double *>(&input[6]));
      auto x3Hi = _mm_loadu_pd(reinterpret_cast<const double *>(&input[7]));

      x0 = _mm256_insertf128_pd(_mm256_castpd128_pd256(x0Lo), x0Hi, 1);
      x1 = _mm256_insertf128_pd(_mm256_castpd128_pd256(x1Lo), x1Hi, 1);
      x2 = _mm256_insertf128_pd(_mm256_castpd128_pd256(x2Lo), x2Hi, 1);
      x3 = _mm256_insertf128_pd(_mm256_castpd128_pd256(x3Lo), x3Hi, 1);
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
      auto a = _mm256_shuffle_ps(x0, x1, _MM_SHUFFLE(1, 0, 1, 0));
      auto b = _mm256_shuffle_ps(x0, x1, _MM_SHUFFLE(3, 2, 3, 2));
      auto c = _mm256_shuffle_ps(x2, x3, _MM_SHUFFLE(1, 0, 1, 0));
      auto d = _mm256_shuffle_ps(x2, x3, _MM_SHUFFLE(3, 2, 3, 2));

      auto aHi = _mm256_extractf128_ps(a, 1);
      auto bHi = _mm256_extractf128_ps(b, 1);
      auto cHi = _mm256_extractf128_ps(c, 1);
      auto dHi = _mm256_extractf128_ps(d, 1);

      _mm_storeu_ps(reinterpret_cast<float *>(&output[0]), _mm256_castps256_ps128(a));
      _mm_storeu_ps(reinterpret_cast<float *>(&output[2]), _mm256_castps256_ps128(c));
      _mm_storeu_ps(reinterpret_cast<float *>(&output[4]), aHi);
      _mm_storeu_ps(reinterpret_cast<float *>(&output[6]), cHi);
      _mm_storeu_ps(reinterpret_cast<float *>(&output[8]), _mm256_castps256_ps128(b));
      _mm_storeu_ps(reinterpret_cast<float *>(&output[10]), _mm256_castps256_ps128(d));
      _mm_storeu_ps(reinterpret_cast<float *>(&output[12]), bHi);
      _mm_storeu_ps(reinterpret_cast<float *>(&output[14]), dHi);
    }
    else
    {
      auto x0Hi = _mm256_extractf128_pd(x0, 1);
      auto x1Hi = _mm256_extractf128_pd(x1, 1);
      auto x2Hi = _mm256_extractf128_pd(x2, 1);
      auto x3Hi = _mm256_extractf128_pd(x3, 1);
      _mm_storeu_pd(reinterpret_cast<double *>(&output[0]), _mm256_castpd256_pd128(x0));
      _mm_storeu_pd(reinterpret_cast<double *>(&output[1]), _mm256_castpd256_pd128(x1));
      _mm_storeu_pd(reinterpret_cast<double *>(&output[2]), _mm256_castpd256_pd128(x2));
      _mm_storeu_pd(reinterpret_cast<double *>(&output[3]), _mm256_castpd256_pd128(x3));
      _mm_storeu_pd(reinterpret_cast<double *>(&output[4]), x0Hi);
      _mm_storeu_pd(reinterpret_cast<double *>(&output[5]), x1Hi);
      _mm_storeu_pd(reinterpret_cast<double *>(&output[6]), x2Hi);
      _mm_storeu_pd(reinterpret_cast<double *>(&output[7]), x3Hi);
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

      constexpr auto HalfIndex = (SIMDComplexCount<F> == 2) ? 1u : 2u;
      w1Array[HalfIndex] = { InvSqrt2<float>, -InvSqrt2<float> };
      w2Array[HalfIndex] = { 0.0f, -1.0f };
      w3Array[HalfIndex] = { -InvSqrt2<float>, -InvSqrt2<float> };

    if constexpr (SIMDComplexCount<F> > 2)
    {
      w1Array[2] = { InvSqrt2<float>, -InvSqrt2<float> };
      w2Array[2] = { 0.0f, -1.0f };
      w3Array[2] = { -InvSqrt2<float>, -InvSqrt2<float> };

      // The first 2 contiguous butterflies use these angles from the table
      w1Array[1] = table[4];
      w2Array[1] = table[2];
      w3Array[1] = table[5];
      w1Array[3] = table[6];
      w2Array[3] = w2Array[1] * -II;
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

    constexpr auto HalfIndex = (SIMDComplexCount<F> == 2) ? 1u : 2u;
    w1Array[HalfIndex] = table[(groupIndex  << 1) + 2];
    w2Array[HalfIndex] = w2Array[0] * -II;
    w3Array[HalfIndex] = table[(groupIndex << 1) + 3];

    if constexpr (SIMDComplexCount<F> > 2)
    {
      w1Array[0] = table[groupIndex << 1];
      w2Array[0] = table[groupIndex];
      w3Array[0] = table[(groupIndex << 1) + 1];
      w1Array[2] = table[(groupIndex  << 1) + 2];
      w2Array[2] = w2Array[0] * -II;
      w3Array[2] = table[(groupIndex << 1) + 3];

      w1Array[1] = table[(groupIndex << 1) + 4];
      w2Array[1] = table[groupIndex + 2];
      w3Array[1] = table[(groupIndex << 1) + 5];
      w1Array[3] = table[(groupIndex  << 1) + 6];
      w2Array[3] = w2Array[1] * -II;
      w3Array[3] = table[(groupIndex << 1) + 7];
    }

    w1 = LoadComplex<F>(w1Array);
    w2 = LoadComplex<F>(w2Array);
    w3 = LoadComplex<F>(w3Array);
  }

  #include "fft_20xx_run.h"


  template <std::floating_point F>
  void ForwardFFT(const Complex<F> *table, const Complex<F> *input, Complex<F> *output, std::size_t fftLength)
    { RunFFT<Dir::Forward, F, SIMDComplexCount<F>>(table, input, output, fftLength); }

  template void ForwardFFT(const Complex<float> *table, const Complex<float> *input, Complex<float> *output, std::size_t fftLength);
  template void ForwardFFT(const Complex<double> *table, const Complex<double> *input, Complex<double> *output, std::size_t fftLength);

  template <std::floating_point F>
  void InverseFFT(const Complex<F> *table, const Complex<F> *input, Complex<F> *output, std::size_t fftLength)
    { RunFFT<Dir::Inverse, F, SIMDComplexCount<F>>(table, input, output, fftLength); }

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

  void FFTComplex32Interleaved_FMA(const FFTTable32 *table, const float *input, float *output, size_t fftLength)
  {
    auto t = reinterpret_cast<const FFTTableImpl<float> *>(table);
    COMMON_ASSERTS();

    FMA::ForwardFFT<float>(
      t->Data(), 
      reinterpret_cast<const Complex<float> *>(input), 
      reinterpret_cast<Complex<float> *>(output), 
      fftLength);
  }

  void FFTComplex64Interleaved_FMA(const FFTTable64 *table, const double *input, double *output, size_t fftLength)
  {
    auto t = reinterpret_cast<const FFTTableImpl<double> *>(table);
    COMMON_ASSERTS();
    FMA::ForwardFFT<double>(
      t->Data(), 
      reinterpret_cast<const Complex<double> *>(input), 
      reinterpret_cast<Complex<double> *>(output), 
      fftLength);
  }


  void IFFTComplex32Interleaved_FMA(const FFTTable32 *table, const float *input, float *output, size_t fftLength)
  {
    auto t = reinterpret_cast<const FFTTableImpl<float> *>(table);
    COMMON_ASSERTS();

    FMA::InverseFFT<float>(
      t->Data(), 
      reinterpret_cast<const Complex<float> *>(input), 
      reinterpret_cast<Complex<float> *>(output), 
      fftLength);
  }

  void IFFTComplex64Interleaved_FMA(const FFTTable64 *table, const double *input, double *output, size_t fftLength)
  {
    auto t = reinterpret_cast<const FFTTableImpl<double> *>(table);
    COMMON_ASSERTS();
    FMA::InverseFFT<double>(
      t->Data(), 
      reinterpret_cast<const Complex<double> *>(input), 
      reinterpret_cast<Complex<double> *>(output), 
      fftLength);
  }
}