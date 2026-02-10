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
    SIMDComplex(N rr, N ii) :r(rr), i(ii) {}


    explicit FORCE_INLINE SIMDComplex(Complex<F> c)
    {
      if constexpr (std::same_as<F, float>)
      {
        auto d = vld2q_dup_f32(reinterpret_cast<F *>(&c));
        r = d.val[0];
        i = d.val[1];
      }
      else
      {
        auto d = vld2q_dup_f64(reinterpret_cast<F *>(&c));
        r = d.val[0];
        i = d.val[1];
      }
    }

    friend FORCE_INLINE auto operator+(SIMDComplex a, SIMDComplex b) -> SIMDComplex
    { 
      if constexpr (std::same_as<F, float>)
        { return { vaddq_f32(a.r, b.r), vaddq_f32(a.i, b.i) }; }
      else
        { return { vaddq_f64(a.r, b.r), vaddq_f64(a.i, b.i) }; }
    }

    friend FORCE_INLINE auto operator-(SIMDComplex a, SIMDComplex b) -> SIMDComplex
    { 
      if constexpr (std::same_as<F, float>)
        { return { vsubq_f32(a.r, b.r), vsubq_f32(a.i, b.i) }; }
      else
        { return { vsubq_f64(a.r, b.r), vsubq_f64(a.i, b.i) }; }
    }

    friend FORCE_INLINE auto operator*(SIMDComplex a, F b) -> SIMDComplex
    {
      if constexpr (std::same_as<F, float>)
        {  return { vmulq_n_f32(a.r, b), vmulq_n_f32(a.i, b) };  }
      else
        {  return { vmulq_n_f64(a.r, b), vmulq_n_f64(a.i, b) };  }
    }

    friend FORCE_INLINE auto operator*(SIMDComplex a, SIMDComplex b) -> SIMDComplex
    {
      // ar * br - ai * bi, ar*bi + br*ai
      if constexpr (std::same_as<F, float>)
        { return { vfmsq_f32(vmulq_f32(a.r, b.r), a.i, b.i), vfmaq_f32(vmulq_f32(b.r, a.i), a.r, b.i) }; }
      else
        { return { vfmsq_f64(vmulq_f64(a.r, b.r), a.i, b.i), vfmaq_f64(vmulq_f64(b.r, a.i), a.r, b.i) }; }
    }

    friend FORCE_INLINE auto MultiplyWithConjugate(SIMDComplex a, SIMDComplex b) -> SIMDComplex
    {
      // ar * br - ai * (-bi), ar*(-bi) + br*ai
      // ar * br + ai * bi, br*ai - ar*bi
      if constexpr (std::same_as<F, float>)
        { return { vfmaq_f32(vmulq_f32(a.i, b.i), a.r, b.r), vfmsq_f32(vmulq_f32(b.r, a.i), a.r, b.i) }; }
      else
        { return { vfmaq_f64(vmulq_f64(a.i, b.i), a.r, b.r), vfmsq_f64(vmulq_f64(b.r, a.i), a.r, b.i) }; }
    }

    template <Dir D>
    friend FORCE_INLINE auto Rot45(SIMDComplex a) -> SIMDComplex
    {  
      if constexpr (D == Dir::Forward)
      {
        if constexpr (std::same_as<F, float>)
          { return { vmulq_n_f32(vaddq_f32(a.r, a.i), InvSqrt2<F>), vmulq_n_f32(vsubq_f32(a.i, a.r), InvSqrt2<F>) }; }
        else
          { return { vmulq_n_f64(vaddq_f64(a.r, a.i), InvSqrt2<F>), vmulq_n_f64(vsubq_f64(a.i, a.r), InvSqrt2<F>) }; }
      }
      else
      {
        if constexpr (std::same_as<F, float>)
          { return { vmulq_n_f32(vsubq_f32(a.r, a.i), InvSqrt2<F>), vmulq_n_f32(vaddq_f32(a.i, a.r), InvSqrt2<F>) }; }
        else
          { return { vmulq_n_f64(vsubq_f64(a.r, a.i), InvSqrt2<F>), vmulq_n_f64(vaddq_f64(a.i, a.r), InvSqrt2<F>) }; }
      }
    }

    template <Dir D>
    friend FORCE_INLINE auto Rot90(SIMDComplex a) -> SIMDComplex
    {
      if constexpr (D == Dir::Forward)
      {
        if constexpr (std::same_as<F, float>)
          { return { a.i, vnegq_f32(a.r) }; }
        else
          { return { a.i, vnegq_f64(a.r) }; }
      }
      else
      {
        if constexpr (std::same_as<F, float>)
          { return { vnegq_f32(a.i), a.r }; }
        else
          { return { vnegq_f64(a.i), a.r }; }
      }
    }


    template <Dir D>
    friend FORCE_INLINE auto Rot135(SIMDComplex a) -> SIMDComplex
    {
      const auto nisqrt = -InvSqrt2<F>;
      if constexpr (D == Dir::Forward)
      {
        if constexpr (std::same_as<F, float>)
          { return { vmulq_n_f32(vsubq_f32(a.r, a.i), nisqrt), vmulq_n_f32(vaddq_f32(a.i, a.r), nisqrt) }; }
        else
          { return { vmulq_n_f64(vsubq_f64(a.r, a.i), nisqrt), vmulq_n_f64(vaddq_f64(a.i, a.r), nisqrt) }; }
      }
      else
      {
        if constexpr (std::same_as<F, float>)
          { return { vmulq_n_f32(vaddq_f32(a.r, a.i), nisqrt), vmulq_n_f32(vsubq_f32(a.i, a.r), nisqrt) }; }
        else
          { return { vmulq_n_f64(vaddq_f64(a.r, a.i), nisqrt), vmulq_n_f64(vsubq_f64(a.i, a.r), nisqrt) }; }
      }
    }


    template <Dir D>
    friend FORCE_INLINE auto Rot(SIMDComplex a, SIMDComplex w) -> SIMDComplex
    {
      if constexpr (D == Dir::Forward)
        { return a * w; }
      else
        { return MultiplyWithConjugate(a, w); }
    }

    N r;
    N i;
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
    { 
      auto d = vld2q_f32(reinterpret_cast<const float *>(in)); 
      return { d.val[0], d.val[1] };
    }
    else
    { 
      auto d = vld2q_f64(reinterpret_cast<const double *>(in)); 
      return { d.val[0], d.val[1] };
    }
  }

  template <std::floating_point F>
  FORCE_INLINE auto StoreComplex(Complex<F> *out, SIMDComplex<F> v)
  { 
    if constexpr (std::same_as<F, float>)
    { 
      auto d = float32x4x2_t { v.r, v.i };
      vst2q_f32(reinterpret_cast<float *>(out), d); 
    }
    else
    { 
      auto d = float64x2x2_t { v.r, v.i };
      vst2q_f64(reinterpret_cast<double *>(out), d);
    }
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
      // r00 r20 r01 r21 | i00 i20 i02 i21 | r10 r30 r11 r31 | i10 i30 i11 i31
      auto a = vld4q_f32(reinterpret_cast<const F *>(input));

      // r02 r22 r03 r23 | i02 i22 i03 i23 | r12 r32 r13 r33 | i12 i32 i13 i33
      auto b = vld4q_f32(reinterpret_cast<const F *>(input) + 16);

      x0 = { vuzp1q_f32(a.val[0], b.val[0]), vuzp1q_f32(a.val[1], b.val[1]) };
      x2 = { vuzp1q_f32(a.val[2], b.val[2]), vuzp1q_f32(a.val[3], b.val[3]) };
      x1 = { vuzp2q_f32(a.val[0], b.val[0]), vuzp2q_f32(a.val[1], b.val[1]) };
      x3 = { vuzp2q_f32(a.val[2], b.val[2]), vuzp2q_f32(a.val[3], b.val[3]) };
    }
    else
    {
      // r00 r20 | i00 i20 | r10 r30 | i10 i30
      auto a = vld4q_f64(reinterpret_cast<const double *>(input));

      // r01 r21 | i01 i21 | r11 r31 | i11 i31
      auto b = vld4q_f64(reinterpret_cast<const double *>(input) + 8);
      x0 = { vuzp1q_f64(a.val[0], b.val[0]), vuzp1q_f64(a.val[1], b.val[1]) };
      x2 = { vuzp1q_f64(a.val[2], b.val[2]), vuzp1q_f64(a.val[3], b.val[3]) };
      x1 = { vuzp2q_f64(a.val[0], b.val[0]), vuzp2q_f64(a.val[1], b.val[1]) };
      x3 = { vuzp2q_f64(a.val[2], b.val[2]), vuzp2q_f64(a.val[3], b.val[3]) };
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
      // r00 r20 r01 r21 | i00 i20 i02 i21 | r10 r30 r11 r31 | i10 i30 i11 i31
      auto a = float32x4x4_t { vzip1q_f32(x0.r, x2.r), vzip1q_f32(x0.i, x2.i), vzip1q_f32(x1.r, x3.r), vzip1q_f32(x1.i, x3.i) };
        
      // r02 r22 r03 r23 | i02 i22 i03 i23 | r12 r32 r13 r33 | i12 i32 i13 i33
      auto b = float32x4x4_t { vzip2q_f32(x0.r, x2.r), vzip2q_f32(x0.i, x2.i), vzip2q_f32(x1.r, x3.r), vzip2q_f32(x1.i, x3.i) };

      vst4q_f32(reinterpret_cast<F *>(output), a);
      vst4q_f32(reinterpret_cast<F *>(output) + 16, b);
    }
    else
    {
      auto a = float64x2x4_t { vzip1q_f64(x0.r, x2.r), vzip1q_f64(x0.i, x2.i), vzip1q_f64(x1.r, x3.r), vzip1q_f64(x1.i, x3.i) };
      auto b = float64x2x4_t { vzip2q_f64(x0.r, x2.r), vzip2q_f64(x0.i, x2.i), vzip2q_f64(x1.r, x3.r), vzip2q_f64(x1.i, x3.i) };

      vst4q_f64(reinterpret_cast<F *>(output), a);
      vst4q_f64(reinterpret_cast<F *>(output) + 8, b);
    }
  }


  template <std::floating_point F>
  FORCE_INLINE void LoadFirstContiguousAngles(const Complex<F> *table, SIMDComplex<F> &w1, SIMDComplex<F> &w2, SIMDComplex<F> &w3)
  {
    if constexpr (std::same_as<F, float>)
    {
      auto w1w3 = vld4q_f32(reinterpret_cast<const F *>(&table[0 * 2]));
      auto w2r = vcombine_f32(vld1_f32(reinterpret_cast<const F *>(&table[0 + 0])), vld1_f32(reinterpret_cast<const F *>(&table[0 + 2])));
      auto w2i = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f64(vdupq_n_u64(0x80000000'00000000ull)), vreinterpretq_u32_f32(vrev64q_f32(w2r))));

      w1 = { w1w3.val[0], w1w3.val[1] };
      w2 = { w2r, w2i };
      w3 = { w1w3.val[2], w1w3.val[3] };
    }
    else
    {
      auto w1w3 = vld4q_f64(reinterpret_cast<const F *>(&table[0 * 2]));
      auto w2r = vld1q_f64(reinterpret_cast<const F *>(&table[0]));
      auto w2i = vextq_f64(w2r, vnegq_f64(w2r), 1);

      w1 = { w1w3.val[0], w1w3.val[1] };
      w2 = { w2r, w2i };
      w3 = { w1w3.val[2], w1w3.val[3] };
    }
  }


  template <std::floating_point F>
  FORCE_INLINE void LoadContiguousAngles(const Complex<F> *table, std::size_t groupIndex, SIMDComplex<F> &w1, SIMDComplex<F> &w2, SIMDComplex<F> &w3)
  {
    if constexpr (std::same_as<F, float>)
    {
      auto w1w3 = vld4q_f32(reinterpret_cast<const F *>(&table[groupIndex << 1]));
      auto w2r = vcombine_f32(vld1_f32(reinterpret_cast<const F *>(&table[groupIndex])), vld1_f32(reinterpret_cast<const F *>(&table[groupIndex + 2])));
      auto w2i = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f64(vdupq_n_u64(0x80000000'00000000ull)), vreinterpretq_u32_f32(vrev64q_f32(w2r))));

      w1 = { w1w3.val[0], w1w3.val[1] };
      w2 = { w2r, w2i };
      w3 = { w1w3.val[2], w1w3.val[3] };
    }
    else
    {
      auto w1w3 = vld4q_f64(reinterpret_cast<const F *>(&table[groupIndex << 1]));
      auto w2r = vld1q_f64(reinterpret_cast<const F *>(&table[groupIndex]));
      auto w2i = vextq_f64(w2r, vnegq_f64(w2r), 1);

      w1 = { w1w3.val[0], w1w3.val[1] };
      w2 = { w2r, w2i };
      w3 = { w1w3.val[2], w1w3.val[3] };
    }
  }

  #define CUSTOM_BUTTERFLY_LOAD 1

  template <std::floating_point F>
  FORCE_INLINE void LoadPrimaryMainButterflyAngles(const Complex<F> *table, std::size_t groupIndex, SIMDComplex<F> &w1, SIMDComplex<F> &w2, SIMDComplex<F> &w3)
  {
    if constexpr (std::same_as<F, float>)
    {
      auto w1w3s = vld4q_dup_f32(reinterpret_cast<const F *>(&table[groupIndex << 1]));
      auto w2s = vld2q_dup_f32(reinterpret_cast<const F *>(&table[groupIndex]));

      w1 = { w1w3s.val[0], w1w3s.val[1] };
      w2 = { w2s.val[0], w2s.val[1] };
      w3 = { w1w3s.val[2], w1w3s.val[3] };
    }
    else
    {
      auto w1w3s = vld4q_dup_f64(reinterpret_cast<const F *>(&table[groupIndex << 1]));
      auto w2s = vld2q_dup_f64(reinterpret_cast<const F *>(&table[groupIndex]));

      w1 = { w1w3s.val[0], w1w3s.val[1] };
      w2 = { w2s.val[0], w2s.val[1] };
      w3 = { w1w3s.val[2], w1w3s.val[3] };
    }
  }

  template <std::floating_point F>
  FORCE_INLINE void MakeSecondaryMainButterflyAngles(const Complex<F> *table, std::size_t groupIndex, SIMDComplex<F> &w1, SIMDComplex<F> &w2, SIMDComplex<F> &w3)
  {
    if constexpr (std::same_as<F, float>)
    {
      auto w1w3s = vld4q_dup_f32(reinterpret_cast<const F *>(&table[(groupIndex << 1) + 2]));
      auto t = w2.r;
      w2.r = w2.i;
      w2.i = vnegq_f32(t);

      w1 = { w1w3s.val[0], w1w3s.val[1] };
      w3 = { w1w3s.val[2], w1w3s.val[3] };
    }
    else
    {
      auto w1w3s = vld4q_dup_f64(reinterpret_cast<const F *>(&table[(groupIndex << 1) + 2]));
      auto t = w2.r;
      w2.r = w2.i;
      w2.i = vnegq_f64(t);

      w1 = { w1w3s.val[0], w1w3s.val[1] };
      w3 = { w1w3s.val[2], w1w3s.val[3] };
    }
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