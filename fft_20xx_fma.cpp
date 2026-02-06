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

  #if COMPILER_GCC
    #pragma GCC diagnostic pop
  #endif

  template <std::floating_point F>
  constexpr auto SIMDScalarCount = std::same_as<std::remove_cvref_t<F>, float> ? 8u : 4u;

  template <std::floating_point F>
  constexpr auto SIMDComplexCount = SIMDScalarCount<F> / 2;

  // !!! all fma and addsub/subadd instructions are FMA
  FORCE_INLINE auto SwapEvenOdd(__m256d v)
    { return _mm256_shuffle_pd(v, v, 0b0101); }

  FORCE_INLINE auto SwapEvenOdd(__m256  v)
    { return _mm256_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1)); }

  FORCE_INLINE auto DuplicateEven(__m256d v)
    { return _mm256_shuffle_pd(v, v, 0); }

  FORCE_INLINE auto DuplicateEven(__m256 v)
    { return _mm256_shuffle_ps(v, v, _MM_SHUFFLE(2, 2, 0, 0)); }

  FORCE_INLINE auto DuplicateOdd(__m256d v)
    { return _mm256_shuffle_pd(v, v, 0b1111); }

  FORCE_INLINE auto DuplicateOdd(__m256 v)
    { return _mm256_shuffle_ps(v, v, _MM_SHUFFLE(3, 3, 1, 1)); }



  FORCE_INLINE auto ComplexMultiply(__m256d a, __m256d b)
    { return _mm256_fmaddsub_pd(b, DuplicateEven(a), _mm256_mul_pd(SwapEvenOdd(b), DuplicateOdd(a))); }

  FORCE_INLINE auto ComplexMultiply(__m256 a, __m256 b)
    { return _mm256_fmaddsub_ps(b, DuplicateEven(a), _mm256_mul_ps(SwapEvenOdd(b), DuplicateOdd(a))); }

  FORCE_INLINE auto ComplexMultiplyConj(__m256d a, __m256d b)
    { return _mm256_fmsubadd_pd(SwapEvenOdd(b), DuplicateOdd(a), _mm256_mul_pd(b, DuplicateEven(a))); }

  FORCE_INLINE auto ComplexMultiplyConj(__m256 a, __m256 b)
    { return _mm256_fmsubadd_ps(SwapEvenOdd(b), DuplicateOdd(a), _mm256_mul_ps(b, DuplicateEven(a))); }

  template <Dir D>
  FORCE_INLINE auto Rot45(__m256d a)
  {  
    if constexpr (D == Dir::Forward)
      { return _mm256_mul_pd(_mm256_set1_pd(InvSqrt2<double>), _mm256_fmsubadd_pd(_mm256_set1_pd(1.0), a, SwapEvenOdd(a))); }
    else
      { return _mm256_mul_pd(_mm256_set1_pd(InvSqrt2<double>), _mm256_addsub_pd(a, SwapEvenOdd(a))); }
  }

  template <Dir D>
  FORCE_INLINE auto Rot45(__m256 a)
  { 
    if constexpr (D == Dir::Forward)
      { return _mm256_mul_ps(_mm256_set1_ps(InvSqrt2<float>), _mm256_fmsubadd_ps(_mm256_set1_ps(1.0f), a, SwapEvenOdd(a)));  }
    else
      { return _mm256_mul_ps(_mm256_set1_ps(InvSqrt2<float>), _mm256_addsub_ps(a, SwapEvenOdd(a)));  }
  }

  template <Dir D>
  FORCE_INLINE auto Rot90(__m256d a)
  { 
    if constexpr (D == Dir::Forward)
      { return SwapEvenOdd(_mm256_addsub_pd(_mm256_setzero_pd(), a)); }
    else
      { return _mm256_addsub_pd(_mm256_setzero_pd(), SwapEvenOdd(a)); }
  }

  template <Dir D>
  FORCE_INLINE auto Rot90(__m256 a)
  { 
    if constexpr (D == Dir::Forward)
      { return SwapEvenOdd(_mm256_addsub_ps(_mm256_setzero_ps(), a)); }
    else
      { return _mm256_addsub_ps(_mm256_setzero_ps(), SwapEvenOdd(a)); }
  }

  template <Dir D>
  FORCE_INLINE auto Rot135(__m256d a)
  { 
    if constexpr (D == Dir::Forward)
      { return _mm256_mul_pd(_mm256_set1_pd(-InvSqrt2<double>), _mm256_addsub_pd(a, SwapEvenOdd(a)));  }
    else
      { return _mm256_mul_pd(_mm256_set1_pd(-InvSqrt2<double>), _mm256_fmsubadd_pd(_mm256_set1_pd(1.0f), a, SwapEvenOdd(a)));  }
  }

  template <Dir D>
  FORCE_INLINE auto Rot135(__m256 a)
  { 
    if constexpr (D == Dir::Forward)
      { return _mm256_mul_ps(_mm256_set1_ps(-InvSqrt2<float>), _mm256_addsub_ps(a, SwapEvenOdd(a)));  }
    else
      { return _mm256_mul_ps(_mm256_set1_ps(-InvSqrt2<float>), _mm256_fmsubadd_ps(_mm256_set1_ps(1.0f), a, SwapEvenOdd(a)));  }
  }

  template <Dir D>
  FORCE_INLINE auto Rot(__m256 a, __m256 w)
  { 
    if (D == Dir::Forward)
      { return ComplexMultiply(a, w); }
    else
      { return ComplexMultiplyConj(a, w); }
  }

  template <Dir D>
  FORCE_INLINE auto Rot(__m256d a, __m256d w)
  { 
    if (D == Dir::Forward)
      { return ComplexMultiply(a, w); }
    else
      { return ComplexMultiplyConj(a, w); }
  }

  FORCE_INLINE auto Set1Complex(Complex<double> c)
    { return _mm256_set_pd(c.i, c.r, c.i, c.r); }

  FORCE_INLINE auto Set1Complex(Complex<float> c)
    { return _mm256_castpd_ps(_mm256_set1_pd(std::bit_cast<double>(c))); }

  FORCE_INLINE auto LoadComplex(const Complex<float> *in)
    { return _mm256_loadu_ps(reinterpret_cast<const float *>(in)); }

  FORCE_INLINE auto LoadComplex(const Complex<double> *in)
    { return _mm256_loadu_pd(reinterpret_cast<const double *>(in)); }

  FORCE_INLINE auto StoreComplex(Complex<float> *out, __m256 v)
    { _mm256_storeu_ps(reinterpret_cast<float *>(out), v); }

  FORCE_INLINE auto StoreComplex(Complex<double> *out, __m256d v)
    { _mm256_storeu_pd(reinterpret_cast<double *>(out), v); }

  FORCE_INLINE auto Add(__m256 a, __m256 b)
    { return _mm256_add_ps(a, b); }

  FORCE_INLINE auto Add(__m256d a, __m256d b)
    { return _mm256_add_pd(a, b); }

  FORCE_INLINE auto Sub(__m256 a, __m256 b)
    { return _mm256_sub_ps(a, b); }

  FORCE_INLINE auto Sub(__m256d a, __m256d b)
    { return _mm256_sub_pd(a, b); }

  FORCE_INLINE void LoadContiguousRadix4(
    const Complex<float> *input,
    __m256 &x0,
    __m256 &x1,
    __m256 &x2,
    __m256 &x3)
  {
    //
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

  FORCE_INLINE void LoadContiguousRadix4(
    const Complex<double> *input,
    __m256d &x0,
    __m256d &x1,
    __m256d &x2,
    __m256d &x3)
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


  FORCE_INLINE void TransposeAndSaveRadix4(
    Complex<float> *output,
    __m256 x0,
    __m256 x1,
    __m256 x2,
    __m256 x3)
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

  FORCE_INLINE void TransposeAndSaveRadix4(
    Complex<double> *output,
    __m256d x0,
    __m256d x1,
    __m256d x2,
    __m256d x3)
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


  template <Dir D>
  FORCE_INLINE void Radix4ButterflyNoRotation(auto &x0, auto &x1, auto &x2, auto &x3)
  {
    auto sum02 = Add(x0, x2);
    auto diff02 = Sub(x0, x2);
    auto sum13 = Add(x1, x3);
    auto iDiff13 = Rot90<D>(Sub(x1, x3));
    x0 = Add(sum02, sum13);
    x1 = Add(diff02, iDiff13);
    x2 = Sub(sum02, sum13);
    x3 = Sub(diff02, iDiff13);
  }

  template <Dir D>
  FORCE_INLINE void Radix4Butterfly45Rotation(auto &x0, auto &x1, auto &x2, auto &x3)
  {
    auto sum02 = Add(x0, x2);
    auto diff02 = Sub(x0, x2);
    auto sum13 = Add(x1, x3);
    auto iDiff13 = Rot90<D>(Sub(x1, x3));
    x0 = Add(sum02, sum13);
    x1 = Rot45<D>(Add(diff02, iDiff13));
    x2 = Rot90<D>(Sub(sum02, sum13));
    x3 = Rot135<D>(Sub(diff02, iDiff13));
  }

  template <Dir D>
  FORCE_INLINE void Radix4Butterfly(auto &x0, auto &x1, auto &x2, auto &x3, auto w1, auto w2, auto w3)
  {
    auto sum02 = Add(x0, x2);
    auto diff02 = Sub(x0, x2);
    auto sum13 = Add(x1, x3);
    auto iDiff13 = Rot90<D>(Sub(x1, x3));
    x0 = Add(sum02, sum13);
    x1 = Rot<D>(Add(diff02, iDiff13), w1);
    x2 = Rot<D>(Sub(sum02, sum13), w2);
    x3 = Rot<D>(Sub(diff02, iDiff13), w3);
  }

  template <std::floating_point F> requires (SIMDComplexCount<F> == 2 || SIMDComplexCount<F> == 4)
  FORCE_INLINE void LoadFirstContiguousAngles(const Complex<F> *table, SIMDNativeType<F> &w1, SIMDNativeType<F> &w2, SIMDNativeType<F> &w3)
  {
    if constexpr (SIMDComplexCount<F> == 2)
    {
      w1 = _mm256_set_pd(-InvSqrt2<double>, InvSqrt2<double>, 0.0, 1.0);
      w2 = _mm256_set_pd(-1.0, 0.0, 0.0, 1.0);
      w3 = _mm256_set_pd(-InvSqrt2<double>, -InvSqrt2<double>, 0.0, 1.0);
    }
    else
    {
      alignas(SIMDNativeType<F>) Complex<float> w1Array[SIMDComplexCount<F>];
      alignas(SIMDNativeType<F>) Complex<float> w2Array[SIMDComplexCount<F>];
      alignas(SIMDNativeType<F>) Complex<float> w3Array[SIMDComplexCount<F>];
      w1Array[0] = { 1.0f, 0.0f };
      w2Array[0] = { 1.0f, 0.0f };
      w3Array[0] = { 1.0f, 0.0f };
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
      w1 = LoadComplex(w1Array);
      w2 = LoadComplex(w2Array);
      w3 = LoadComplex(w3Array);
    }
  }


  template <std::floating_point F> requires (SIMDComplexCount<F> == 2 || SIMDComplexCount<F> == 4)
  FORCE_INLINE void LoadContiguousAngles(const Complex<F> *table, std::size_t groupIndex, SIMDNativeType<F> &w1, SIMDNativeType<F> &w2, SIMDNativeType<F> &w3)
  {
    alignas(SIMDNativeType<F>) Complex<F> w1Array[SIMDComplexCount<F>];
    alignas(SIMDNativeType<F>) Complex<F> w2Array[SIMDComplexCount<F>];
    alignas(SIMDNativeType<F>) Complex<F> w3Array[SIMDComplexCount<F>];
    if constexpr (SIMDComplexCount<F> == 2)
    {
        w1Array[0] = table[groupIndex << 1];
        w2Array[0] = table[groupIndex];
        w3Array[0] = table[(groupIndex << 1) + 1];
        w1Array[1] = table[(groupIndex  << 1) + 2];
        w2Array[1] = w2Array[0] * -II;
        w3Array[1] = table[(groupIndex << 1) + 3];
    }
    else
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

    w1 = LoadComplex(w1Array);
    w2 = LoadComplex(w2Array);
    w3 = LoadComplex(w3Array);
  }


  template <Dir D, std::floating_point F>
  FORCE_INLINE void DoNoRotationButterfly(Complex<F> *data, std::size_t baseIndex, std::size_t butterflyStep)
  {
    auto i0 = baseIndex;
    auto i1 = i0 + butterflyStep;
    auto i2 = i1 + butterflyStep;
    auto i3 = i2 + butterflyStep;

    auto x0 = LoadComplex(&data[i0]);
    auto x2 = LoadComplex(&data[i1]);
    auto x1 = LoadComplex(&data[i2]);
    auto x3 = LoadComplex(&data[i3]);

    Radix4ButterflyNoRotation<D>(x0, x1, x2, x3);

    StoreComplex(&data[i0], x0);
    StoreComplex(&data[i1], x1);
    StoreComplex(&data[i2], x2);
    StoreComplex(&data[i3], x3);
  }


  template <Dir D, std::floating_point F>
  FORCE_INLINE void DoFixed45Butterfly(Complex<F> *data, std::size_t baseIndex, std::size_t butterflyStep)
  {
    auto i0 = baseIndex;
    auto i1 = i0 + butterflyStep;
    auto i2 = i1 + butterflyStep;
    auto i3 = i2 + butterflyStep;

    auto x0 = LoadComplex(&data[i0]);
    auto x2 = LoadComplex(&data[i1]);
    auto x1 = LoadComplex(&data[i2]);
    auto x3 = LoadComplex(&data[i3]);

    Radix4Butterfly45Rotation<D>(x0, x1, x2, x3);

    StoreComplex(&data[i0], x0);
    StoreComplex(&data[i1], x1);
    StoreComplex(&data[i2], x2);
    StoreComplex(&data[i3], x3);
  }

  template <Dir D, std::floating_point F>
  FORCE_INLINE void DoMainButterfly(
    Complex<F> *data,
    std::size_t baseIndex,
    std::size_t butterflyStep,
    Complex<F> w1In,
    Complex<F> w2In,
    Complex<F> w3In)
  {
    auto w1 = Set1Complex(w1In);
    auto w2 = Set1Complex(w2In);
    auto w3 = Set1Complex(w3In);

    auto i0 = baseIndex;
    auto i1 = i0 + butterflyStep;
    auto i2 = i1 + butterflyStep;
    auto i3 = i2 + butterflyStep;
    auto x0 = LoadComplex(&data[i0]);
    auto x2 = LoadComplex(&data[i1]);
    auto x1 = LoadComplex(&data[i2]);
    auto x3 = LoadComplex(&data[i3]);

    Radix4Butterfly<D>(x0, x1, x2, x3, w1, w2, w3);

    StoreComplex(&data[i0], x0);
    StoreComplex(&data[i1], x1);
    StoreComplex(&data[i2], x2);
    StoreComplex(&data[i3], x3);
  }


  template <Dir D, std::floating_point F>
  FORCE_INLINE void RunFirstContiguousButterflySet(const Complex<F> *table, Complex<F> *data)
  {
    SIMDNativeType<F> x0, x1, x2, x3;
    SIMDNativeType<F> w1, w2, w3;
    LoadContiguousRadix4(data, x0, x1, x2, x3);
    LoadFirstContiguousAngles<F>(table, w1, w2, w3);
    Radix4Butterfly<D>(x0, x1, x2, x3, w1, w2, w3);
    TransposeAndSaveRadix4(data, x0, x1, x2, x3);
  }

  template <Dir D, std::floating_point F>
  FORCE_INLINE void RunContiguousButterflySet(const Complex<F> *table, Complex<F> *data, std::size_t groupIndex)
  {
    SIMDNativeType<F> x0, x1, x2, x3;
    SIMDNativeType<F> w1, w2, w3;
    LoadContiguousRadix4(data, x0, x1, x2, x3);
    LoadContiguousAngles<F>(table, groupIndex, w1, w2, w3);
    Radix4Butterfly<D>(x0, x1, x2, x3, w1, w2, w3);
    TransposeAndSaveRadix4(data, x0, x1, x2, x3);
  }

  template <std::floating_point F>
  FORCE_INLINE void DoNoRotationRadix2Butterfly(
    Complex<F> *data,
    std::size_t baseIndex,
    std::size_t butterflyStep)
  {
    auto i0 = baseIndex;
    auto i1 = i0 + butterflyStep;

    auto even = LoadComplex(&data[i0]);
    auto odd = LoadComplex(&data[i1]);

    StoreComplex(&data[i0], Add(even, odd));
    StoreComplex(&data[i1], Sub(even, odd));
  }


template <std::floating_point F>
FORCE_INLINE void RescaleForIFFT(Complex<F> *data, std::size_t fftLength, std::size_t dataOffset, std::size_t spanLength)
{
  F scale = F(1.0) / F(fftLength);
  for (auto i = dataOffset; i < dataOffset + spanLength; i++)
    { data[i] *= scale; }
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
