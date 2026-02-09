#include "fft_20xx_common.h"


static_assert(SIMD_NEON);

#include <concepts>
#include <cinttypes>

#include <arm_neon.h>

namespace FFT20XX::Scalar
{
  template <Dir D, std::floating_point F>
  void RunFFT(const Complex<F> *table, const Complex<F> *input, Complex<F> *output, std::size_t fftLength);

}

// !!! 32-bit float output is wrong :|
namespace FFT20XX::NEON
{
  #if COMPILER_GCC
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wignored-attributes"
  #endif

  template <std::floating_point F>
  using SIMDNativeType = std::conditional_t<std::same_as<std::remove_cvref_t<F>, float>, float32x4_t, float64x2_t>;

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
      { 
        nat = vreinterpretq_f32_f64(vdupq_n_f64(vreinterpret_f64_f32(float32x2_t{ c.r, c.i })[0]));
      }
      else
      { 
        nat = { c.r, c.i };
      }
    }

    friend FORCE_INLINE auto operator+(SIMDComplex a, SIMDComplex b) -> SIMDComplex
    { 
      if constexpr (std::same_as<F, float>)
        { return vaddq_f32(a, b); }
      else
        { return vaddq_f64(a, b); }
    }

    friend FORCE_INLINE auto operator-(SIMDComplex a, SIMDComplex b) -> SIMDComplex
    { 
      if constexpr (std::same_as<F, float>)
        { return vsubq_f32(a, b); }
      else
        { return vsubq_f64(a, b); }
    }

    friend FORCE_INLINE auto operator*(SIMDComplex a, F b) -> SIMDComplex
    {
      if constexpr (std::same_as<F, float>)
        { return vmulq_f32(a, vdupq_n_f32(b)); }
      else
        { return vmulq_f64(a, vdupq_n_f64(b)); }
    }

    friend FORCE_INLINE auto operator*(SIMDComplex a, SIMDComplex b) -> SIMDComplex
    {
      // ar * br - ai * bi, ar*bi + br*ai
      if constexpr (std::same_as<F, float>)
        { return vfmaq_f32(SwapEvenOdd(NegateOddElements(vmulq_f32(b, DuplicateOdd(a)))), DuplicateEven(a), b); }
      else
        {  return vfmaq_f64(SwapEvenOdd(NegateOddElements(vmulq_f64(b, DuplicateOdd(a)))), DuplicateEven(a), b); }
    }

    friend FORCE_INLINE auto MultiplyWithConjugate(SIMDComplex a, SIMDComplex b) -> SIMDComplex
    {
      // ar * br - ai * (-bi), ar*(-bi) + br*ai
      // bi*ai + br.ar, br*ai - bi*ar
      // (b.i, b.r) * a.i + NegateOdd(b * a.r);
      if constexpr (std::same_as<F, float>)
        { return vfmaq_f32(NegateOddElements(vmulq_f32(b, DuplicateEven(a))), DuplicateOdd(a), SwapEvenOdd(b)); }
      else
        { return vfmaq_f64(NegateOddElements(vmulq_f64(b, DuplicateEven(a))), DuplicateOdd(a), SwapEvenOdd(b)); }
    }

    template <Dir D>
    friend FORCE_INLINE auto Rot45(SIMDComplex a) -> SIMDComplex
    {  
      if constexpr (std::same_as<F, float>)
      {
        if constexpr (D == Dir::Forward)
          { return vmulq_f32(vdupq_n_f32(InvSqrt2<float>), vaddq_f32(a, NegateOddElements(SwapEvenOdd(a)))); }
        else
          { return vmulq_f32(vdupq_n_f32(InvSqrt2<float>), vaddq_f32(a, SwapEvenOdd(NegateOddElements(a)))); }
      }
      else
      {
        if constexpr (D == Dir::Forward)
          { return vmulq_f64(vdupq_n_f64(InvSqrt2<double>), vaddq_f64(a, NegateOddElements(SwapEvenOdd(a)))); }
        else
          { return vmulq_f64(vdupq_n_f64(InvSqrt2<double>), vaddq_f64(a, SwapEvenOdd(NegateOddElements(a)))); }
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
          { return vmulq_f32(vdupq_n_f32(-InvSqrt2<float>), vaddq_f32(a, SwapEvenOdd(NegateOddElements(a)))); }
        else
          { return vmulq_f32(vdupq_n_f32(-InvSqrt2<float>), vaddq_f32(a, NegateOddElements(SwapEvenOdd(a)))); }
      }
      else
      {
        if constexpr (D == Dir::Forward)
          { return vmulq_f64(vdupq_n_f64(-InvSqrt2<double>), vaddq_f64(a, SwapEvenOdd(NegateOddElements(a)))); }
        else
          { return vmulq_f64(vdupq_n_f64(-InvSqrt2<double>), vaddq_f64(a, NegateOddElements(SwapEvenOdd(a)))); }
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

    static FORCE_INLINE auto NegateOddElements(N v)
    {
      if constexpr (std::same_as<F, float>)
      {
        auto negateOddElements = vreinterpretq_u32_f64(vnegq_f64(vdupq_n_f64(0.0)));
        return vreinterpretq_f32_u32(veorq_u32(negateOddElements, v));
      }
      else
      {
        uint64x2_t negateOddElements { 0, 0x80000000'00000000 };
        return vreinterpretq_f64_u64(veorq_u64(negateOddElements, v));
      }
    }

    static FORCE_INLINE auto SwapEvenOdd(N v)
    { 
      if constexpr (std::same_as<F, float>)
        { return vrev64q_f32(v); }
      else
        { return vextq_f64(v, v, 1); }
    }

    static FORCE_INLINE auto DuplicateEven(N v)
    { 
      if constexpr (std::same_as<F, float>)
        { return vtrn1q_f32(v, v); }
      else
        { return vdupq_n_f64(v[0]); }
    }

    static FORCE_INLINE auto DuplicateOdd(N v)
    {
      if constexpr (std::same_as<F, float>)
        { return vtrn2q_f32(v, v); }
      else
        { return vdupq_n_f64(v[1]); }
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
      { return vld1q_f32(reinterpret_cast<const float *>(in)); }
    else
      { return vld1q_f64(reinterpret_cast<const double *>(in)); }
  }

  template <std::floating_point F>
  FORCE_INLINE auto StoreComplex(Complex<F> *out, SIMDComplex<F> v)
  { 
    if constexpr (std::same_as<F, float>)
      { vst1q_f32(reinterpret_cast<float *>(out), v); }
    else
      { vst1q_f64(reinterpret_cast<double *>(out), v); }
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
      auto x00x10 = vld1q_f32(reinterpret_cast<const float *>(&input[0]));
      auto x20x30 = vld1q_f32(reinterpret_cast<const float *>(&input[2]));
      auto x01x11 = vld1q_f32(reinterpret_cast<const float *>(&input[4]));
      auto x21x31 = vld1q_f32(reinterpret_cast<const float *>(&input[6]));

      x0 = vreinterpretq_f32_f64(vzip1q_f64(vreinterpretq_f64_f32(x00x10), vreinterpretq_f64_f32(x01x11)));
      x1 = vreinterpretq_f32_f64(vzip2q_f64(vreinterpretq_f64_f32(x00x10), vreinterpretq_f64_f32(x01x11)));
      x2 = vreinterpretq_f32_f64(vzip1q_f64(vreinterpretq_f64_f32(x20x30), vreinterpretq_f64_f32(x21x31)));
      x3 = vreinterpretq_f32_f64(vzip1q_f64(vreinterpretq_f64_f32(x21x31), vreinterpretq_f64_f32(x21x31)));
    }
    else
    {
      x0 = vld1q_f64(reinterpret_cast<const double *>(&input[0]));
      x1 = vld1q_f64(reinterpret_cast<const double *>(&input[1]));
      x2 = vld1q_f64(reinterpret_cast<const double *>(&input[2]));
      x3 = vld1q_f64(reinterpret_cast<const double *>(&input[3]));
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
      vst1q_f32(reinterpret_cast<float *>(&output[0]), vreinterpretq_f32_f64(vzip1q_f64(vreinterpretq_f64_f32(x0), vreinterpretq_f64_f32(x1))));
      vst1q_f32(reinterpret_cast<float *>(&output[2]), vreinterpretq_f32_f64(vzip2q_f64(vreinterpretq_f64_f32(x0), vreinterpretq_f64_f32(x1))));
      vst1q_f32(reinterpret_cast<float *>(&output[4]), vreinterpretq_f32_f64(vzip1q_f64(vreinterpretq_f64_f32(x2), vreinterpretq_f64_f32(x3))));
      vst1q_f32(reinterpret_cast<float *>(&output[6]), vreinterpretq_f32_f64(vzip2q_f64(vreinterpretq_f64_f32(x2), vreinterpretq_f64_f32(x3))));
    }
    else
    {
      vst1q_f64(reinterpret_cast<double *>(&output[0]), x0);
      vst1q_f64(reinterpret_cast<double *>(&output[1]), x1);
      vst1q_f64(reinterpret_cast<double *>(&output[2]), x2);
      vst1q_f64(reinterpret_cast<double *>(&output[3]), x3);
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
      w2Array[3] = w2Array[1] * -II;
      w3Array[3] = table[(groupIndex << 1) + 7];
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

  void FFTComplex32Interleaved_NEON(const FFTTable32 *table, const float *input, float *output, size_t fftLength)
  {
    auto t = reinterpret_cast<const FFTTableImpl<float> *>(table);
    COMMON_ASSERTS();

    NEON::RunFFT<Dir::Forward, float>(
      t->Data(), 
      reinterpret_cast<const Complex<float> *>(input), 
      reinterpret_cast<Complex<float> *>(output), 
      fftLength);
  }

  void FFTComplex64Interleaved_NEON(const FFTTable64 *table, const double *input, double *output, size_t fftLength)
  {
    auto t = reinterpret_cast<const FFTTableImpl<double> *>(table);
    COMMON_ASSERTS();
    NEON::RunFFT<Dir::Forward, double>(
      t->Data(), 
      reinterpret_cast<const Complex<double> *>(input), 
      reinterpret_cast<Complex<double> *>(output), 
      fftLength);
  }


  void IFFTComplex32Interleaved_NEON(const FFTTable32 *table, const float *input, float *output, size_t fftLength)
  {
    auto t = reinterpret_cast<const FFTTableImpl<float> *>(table);
    COMMON_ASSERTS();

    NEON::RunFFT<Dir::Inverse, float>(
      t->Data(), 
      reinterpret_cast<const Complex<float> *>(input), 
      reinterpret_cast<Complex<float> *>(output), 
      fftLength);
  }

  void IFFTComplex64Interleaved_NEON(const FFTTable64 *table, const double *input, double *output, size_t fftLength)
  {
    auto t = reinterpret_cast<const FFTTableImpl<double> *>(table);
    COMMON_ASSERTS();
    NEON::RunFFT<Dir::Inverse, double>(
      t->Data(), 
      reinterpret_cast<const Complex<double> *>(input), 
      reinterpret_cast<Complex<double> *>(output), 
      fftLength);
  }
}