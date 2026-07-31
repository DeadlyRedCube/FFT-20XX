#ifdef _MSC_VER
  #pragma warning (disable: 4711) // function <x> selected for automatic inline expansion
#endif

#include "fft_20xx_common.h"
#include "fft_20xx_table.h"
#include "fft_20xx_tiny_real.h"

static_assert(CPU_ARM64);
static_assert(SIMD_NEON);

#include <arm_neon.h>

namespace FFT20XX::NEON
{
  template <std::floating_point F>
  bool IsSIMDAligned(const F *ptr)
    { return (uintptr_t(ptr) & (sizeof(float32x4_t) - 1)) == 0; }

  constexpr auto MinimumRealFFTLength = 32;
  constexpr bool UseFMAOptimization = false;
  constexpr auto VecSize = 4u;
  using VecType = float32x4_t;
  using ElementType = float;

  template <Alignment A>
  [[nodiscard]] FORCE_INLINE auto Load(const float *p)
  {
    return vld1q_f32(p);
  }



  template <Alignment A>
  FORCE_INLINE void Store(float *p, float32x4_t v)
  {
    vst1q_f32(p, v);
  }



  [[nodiscard]] FORCE_INLINE auto Add(float32x4_t a, float32x4_t b)
  {
    return vaddq_f32(a, b);
  }

  [[nodiscard]] FORCE_INLINE auto Sub(float32x4_t a, float32x4_t b)
  {
    return vsubq_f32(a, b);
  }

  [[nodiscard]] FORCE_INLINE auto Mul(float32x4_t a, float32x4_t b)
  {
    return vmulq_f32(a, b);
  }

  [[nodiscard]] FORCE_INLINE auto Mul(float32x4_t a, float b)
  {
    return vmulq_n_f32(a, b);
  }

  [[nodiscard]] FORCE_INLINE auto MAdd(float32x4_t a, auto b, float32x4_t c) -> float32x4_t
  {
    if constexpr (std::floating_point<decltype(b)>)
      { return vmlaq_n_f32(c, a, b); }
    else
      { return vmlaq_f32(c, b, a); }
  }

  [[nodiscard]] FORCE_INLINE auto NMAdd(float32x4_t a, auto b, float32x4_t c) -> float32x4_t
  {
    if constexpr (std::floating_point<decltype(b)>)
      { return vmlsq_n_f32(c, a, b); }
    else
      { return vmlsq_f32(c, b, a); }
  }

  [[nodiscard]] FORCE_INLINE auto ReplaceFirstOf(float32x4_t value, auto newFirst) -> float32x4_t
  {
    if constexpr (std::floating_point<decltype(newFirst)>)
      { return vsetq_lane_f32(newFirst, value, 0); }
    else
      { return vsetq_lane_f32(vgetq_lane_f32(newFirst, 0), value, 0); }
  }

  [[nodiscard]] FORCE_INLINE auto InsertTopHalf(float32x4_t lo, float32x4_t hi) -> float32x4_t
  {
    return vcombine_f32(vget_low_f32(lo), vget_high_f32(hi));
  }

  FORCE_INLINE void Transpose4(float32x4_t &a, float32x4_t &b, float32x4_t &c, float32x4_t &d)
  {
    auto ta = vtrn1q_f32(a, b);
    auto tb = vtrn2q_f32(a, b);
    auto tc = vtrn1q_f32(c, d);
    auto td = vtrn2q_f32(c, d);

    a = vreinterpretq_f32_f64(vtrn1q_f64(vreinterpretq_f64_f32(ta), vreinterpretq_f64_f32(tc)));
    b = vreinterpretq_f32_f64(vtrn1q_f64(vreinterpretq_f64_f32(tb), vreinterpretq_f64_f32(td)));
    c = vreinterpretq_f32_f64(vtrn2q_f64(vreinterpretq_f64_f32(ta), vreinterpretq_f64_f32(tc)));
    d = vreinterpretq_f32_f64(vtrn2q_f64(vreinterpretq_f64_f32(tb), vreinterpretq_f64_f32(td)));
  }



  // Takes the first two elements of aceg0 and bdfh0 and puts them in t, then the second two elements from each into bdfh0.
  FORCE_INLINE void SwapRealComponentsForLastDoubleFlip(float32x4_t aceg0, float32x4_t &bdfh0, float32x4_t &t)
  {
    t =  vcombine_f32(vget_low_f32(aceg0), vget_low_f32(bdfh0));
    bdfh0 = vcombine_f32(vget_high_f32(aceg0), vget_high_f32(bdfh0));
  }



  FORCE_INLINE void Deinterleave4(auto a, auto &b, auto &aOut)
  {
    auto r = vuzpq_f32(a, b);
    aOut = r.val[0];
    b = r.val[1];
  }



  FORCE_INLINE void Interleave4(auto a, auto &b, auto &aOut)
  {
    auto r = vzipq_f32(a, b);
    aOut = r.val[0];
    b = r.val[1];
  }



  #include "fft_20xx_algo_real_fwd.h"
  #include "fft_20xx_algo_real_inv.h"
}


using namespace FFT20XX;
using namespace fft_20xx::c;
extern "C"
{
  void fft_real_f32_neon(const fft_table_f32 *table, const float *input, float *output, size_t fftLength)
  {
    auto t = reinterpret_cast<const FFTTableImpl<float> *>(table);
    COMMON_ASSERTS();
    NEON::RealForwardFFT<Alignment::Unaligned>(t->RealData(), input, output, fftLength);
  }



  void ifft_real_f32_neon(const fft_table_f32 *table, const float *input, float *output, size_t fftLength)
  {
    auto t = reinterpret_cast<const FFTTableImpl<float> *>(table);
    COMMON_ASSERTS();
    NEON::RealInverseFFT<Alignment::Unaligned>(t->RealData(), input, output, fftLength);
  }
}
