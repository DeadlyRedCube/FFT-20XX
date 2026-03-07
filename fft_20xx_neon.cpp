#include "fft_20xx_common.h"


static_assert(SIMD_NEON);

#include <concepts>
#include <cinttypes>

#include <arm_neon.h>

namespace FFT20XX::NEON
{
  template <typename T> concept f32 = std::same_as<float, T> || std::same_as<float32x4_t, T>;
  template <typename T> concept simd_vec = std::same_as<std::remove_cvref_t<T>, float32x4_t> || std::same_as<std::remove_cvref_t<T>, float64x2_t>;
  template <typename T> using Vec = std::conditional_t<f32<T>, float32x4_t, float64x2_t>;

  template <typename> constexpr uint32_t VecSize = 0;

  template <> constexpr uint32_t VecSize<Vec<float>> = 4u;
  template <> constexpr uint32_t VecSize<Vec<double>> = 2u;
  
  constexpr bool UseFMAOptimization = false;

  template <std::floating_point F>
  static auto LoadA(const F *ptr) -> Vec<F>
  {
    if constexpr (f32<F>)
      { return vld1q_f32(ptr); }
    else
      { return vld1q_f64(ptr); }
  }
  template <std::floating_point F>
  static auto Load(const F *ptr) -> Vec<F>
    { return LoadA(ptr); }

  template <std::floating_point F>
  static auto Load4(const F *ptr) -> Vec<F>
    { return LoadA(ptr); }

  template <std::floating_point F>
  static auto LoadDeinterleaved(const F *ptr) -> std::pair<Vec<F>, Vec<F>>
  {
    if constexpr (f32<F>)
    {
      auto d = vld2q_f32(ptr);
      return { d.val[0], d.val[1] };
    }
    else
    {
      auto d = vld2q_f64(ptr);
      return { d.val[0], d.val[1] };
    }
  }

  template <std::floating_point F>
  static void Store2x2Transposed(F *ptrA, F *ptrB, Vec<F> a, Vec<F> b)
  {
    auto ta = vtrn1q_f64(a, b);
    auto tb = vtrn2q_f64(a, b);
    vst1q_f64(ptrA, ta);
    vst1q_f64(ptrB, tb);
  }

  template <std::floating_point F>
  static void Store4x4Transposed(F *ptrA, F *ptrB, F *ptrC, F *ptrD, Vec<F> a, Vec<F> b, Vec<F> c, Vec<F> d)
  {
    static_assert(f32<F>);
    auto ta = vtrn1q_f32(a, b);
    auto tb = vtrn2q_f32(a, b);
    auto tc = vtrn1q_f32(c, d);
    auto td = vtrn2q_f32(c, d);

    a = vreinterpretq_f32_f64(vtrn1q_f64(vreinterpretq_f64_f32(ta), vreinterpretq_f64_f32(tc)));
    b = vreinterpretq_f32_f64(vtrn1q_f64(vreinterpretq_f64_f32(tb), vreinterpretq_f64_f32(td)));
    c = vreinterpretq_f32_f64(vtrn2q_f64(vreinterpretq_f64_f32(ta), vreinterpretq_f64_f32(tc)));
    d = vreinterpretq_f32_f64(vtrn2q_f64(vreinterpretq_f64_f32(tb), vreinterpretq_f64_f32(td)));
    vst1q_f32(ptrA, a);
    vst1q_f32(ptrB, b);
    vst1q_f32(ptrC, c);
    vst1q_f32(ptrD, d);

  }

  template <std::floating_point F>
  static void Store(F *ptr, Vec<F> a)
  {
    if constexpr (f32<F>)
      { vst1q_f32(ptr, a); }
    else 
      { vst1q_f64(ptr, a); }
  }

  template <std::floating_point F>
  static void StoreInterleaved(F *ptr, Vec<F> a, Vec<F> b)
  {
    if constexpr (f32<F>)
      { vst2q_f32(ptr, (float32x4x2_t{a, b})); }
    else
      { vst2q_f64(ptr, (float64x2x2_t{a, b})); }
  }

  template <std::floating_point F, std::same_as<Vec<F>> ... Vs>
  static void Scale(F scale, Vs &...vs)
  {
    if constexpr (f32<F>)
      { ((vs = vmulq_n_f32(vs, scale)), ...); }
    else
      { ((vs = vmulq_n_f64(vs, scale)), ...); }
  }
  
  template <simd_vec V>
  static V Add(V a, V b)
  {
    if constexpr (f32<V>)
      { return vaddq_f32(a, b); }
    else
      { return vaddq_f64(a, b); }
  }

  template <simd_vec V>
  static V Sub(V a, V b)
  {
    if constexpr (f32<V>)
      { return vsubq_f32(a, b); }
    else
      { return vsubq_f64(a, b); }
  }

  template <simd_vec V>
  static V Mul(V a, V b)
  {
    if constexpr (f32<V>)
      { return vmulq_f32(a, b); }
    else
      { return vmulq_f64(a, b); }
  }

  template <simd_vec V>
  static auto Set1(std::conditional_t<f32<V>, float, double> f) -> V
  {
    if constexpr (f32<V>)
      { return vdupq_n_f32(f); }
    else
      { return vdupq_n_f64(f); }
  }
  
  template <simd_vec V>
  static V FMAdd(V a, V b, V c)
  {
    if constexpr (f32<V>)
      { return vmlaq_f32(c, b, a); }
    else
      { return vfmaq_f64(c, b, a); }
  }

  template <simd_vec V>
  static V FNMAdd(V a, V b, V c)
  {
    if constexpr (f32<V>)
      { return vmlsq_f32(c, b, a); }
    else
      { return vfmsq_f64(c, b, a); }
  }

  template <simd_vec V>
  static V FMSub(V a, V b, V c)
  {
    if constexpr (f32<V>)
      { return vnegq_f32(vmlsq_f32(c, b, a)); }
    else
      { return vnegq_f64(vfmsq_f64(c, b, a)); }
  }

  template <simd_vec V>
  static V MAdd(V a, V b, V c)
    { return FMAdd(a, b, c); }

  template <simd_vec V>
  static V NMAdd(V a, V b, V c)
    { return FNMAdd(a, b, c); }
  
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

    NEON::DITRadix4FFT<Dir::Forward>(
      t->Data(),
      reinterpret_cast<const Complex<float> *>(input),
      reinterpret_cast<Complex<float> *>(output),
      fftLength);
  }

  void FFTComplex64Interleaved_NEON(const FFTTable64 *table, const double *input, double *output, size_t fftLength)
  {
    #if SIMD_NEON64
      auto t = reinterpret_cast<const FFTTableImpl<double> *>(table);
      COMMON_ASSERTS();

      NEON::DITRadix4FFT<Dir::Forward>(
        t->Data(),
        reinterpret_cast<const Complex<double> *>(input),
        reinterpret_cast<Complex<double> *>(output),
        fftLength);
    #else
      FFTComplex64Interleaved_Scalar(table, input, output, fftLength);
    #endif
  }


  void IFFTComplex32Interleaved_NEON(const FFTTable32 *table, const float *input, float *output, size_t fftLength)
  {
    auto t = reinterpret_cast<const FFTTableImpl<float> *>(table);
    COMMON_ASSERTS();

    NEON::DITRadix4FFT<Dir::Inverse>(
      t->Data(),
      reinterpret_cast<const Complex<float> *>(input),
      reinterpret_cast<Complex<float> *>(output),
      fftLength);
  }

  void IFFTComplex64Interleaved_NEON(const FFTTable64 *table, const double *input, double *output, size_t fftLength)
  {
    #if SIMD_NEON64
      auto t = reinterpret_cast<const FFTTableImpl<double> *>(table);
      COMMON_ASSERTS();
      NEON::DITRadix4FFT<Dir::Inverse>(
        t->Data(),
        reinterpret_cast<const Complex<double> *>(input),
        reinterpret_cast<Complex<double> *>(output),
        fftLength);
    #else
      IFFTComplex64Interleaved_Scalar(table, input, output, fftLength);
    #endif
  }
}