#ifdef _MSC_VER
  #pragma warning (disable: 4711) // function <x> selected for automatic inline expansion
#endif

#include "fft_20xx_common.h"
#include "fft_20xx_table.h"
#include "fft_20xx_tiny_cplx.h"
#include "fft_20xx_tiny_real.h"

static_assert(CPU_ARM64);
static_assert(SIMD_NEON);

#if COMPILER_GCC
  #pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

using namespace FFT20XX;

namespace FFT20XX::NEON32
{
  constexpr uint32_t VecSize = 4u;
  using VecType = float32x4_t;
  using ElementType = float;
  constexpr auto MinimumRealFFTLength = 32;
  constexpr bool UseFMAOptimization = false;

  template <typename T>
  struct Angles
  {
    T w0;
    T w1;
    T w2;
  };


  bool IsSIMDAligned(const float *ptr)
    { return (uintptr_t(ptr) & (sizeof(VecType) - 1)) == 0; }


  FORCE_INLINE auto Set1(float f) -> VecType
  {
    return vdupq_n_f32(f);
  }



  template <Alignment Align>
  FORCE_INLINE auto Load(const float *ptr)
  {
    return vld1q_f32(ptr);
  }

  template <Alignment Align>
  FORCE_INLINE void Store(float *ptr, VecType v)
  {
    vst1q_f32(ptr, v);
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

  FORCE_INLINE void Transpose4x4(float32x4_t &a, float32x4_t &b, float32x4_t &c, float32x4_t &d)
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

  // Functions specific to complex FFTs ///////////////////////////////////////////////////////////////////////////////

  template <Alignment Align>
  FORCE_INLINE auto LoadPackedComplex(const float *ptr) -> ComplexVec<VecType>
  {
    auto r = vld2q_f32(ptr);
    return { r.val[0], r.val[1] };
  }



  template <Alignment Align>
  FORCE_INLINE void StorePackedComplex(float *ptr, ComplexVec<VecType> v)
  {
    vst2q_f32(ptr, (float32x4x2_t{v.r, v.i}));
  }

  // Functions specific to real FFTs //////////////////////////////////////////////////////////////////////////////////

  // Takes the first two elements of aceg0 and bdfh0 and puts them in t, then the second two elements from each into bdfh0.
  FORCE_INLINE void SwapRealComponentsForLastDoubleFlip(float32x4_t aceg0, float32x4_t &bdfh0, float32x4_t &t)
  {
    t =  vcombine_f32(vget_low_f32(aceg0), vget_low_f32(bdfh0));
    bdfh0 = vcombine_f32(vget_high_f32(aceg0), vget_high_f32(bdfh0));
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

  #include "fft_20xx_algo_cplx.h"
  #include "fft_20xx_algo_real_common.h"
  #include "fft_20xx_algo_real_fwd.h"
  #include "fft_20xx_algo_real_inv.h"
}


using namespace FFT20XX;
using namespace fft_20xx::c;
extern "C"
{
  void fft_cplx_f32_neon(const fft_table_cplx_f32 *table, const float *input, float *output, size_t fftLength)
  {
    COMMON_ASSERTS();
    NEON32::RunComplex<Dir::Forward, Alignment::Unaligned>(GetTableData(table), input, output, fftLength);
  }

  void ifft_cplx_f32_neon(const fft_table_cplx_f32 *table, const float *input, float *output, size_t fftLength)
  {
    COMMON_ASSERTS();
    NEON32::RunComplex<Dir::Inverse, Alignment::Unaligned>(GetTableData(table), input, output, fftLength);
  }

  void fft_real_f32_neon(const fft_table_real_f32 *table, const float *input, float *output, size_t fftLength)
  {
    COMMON_ASSERTS();
    NEON32::RealForwardFFT<Alignment::Unaligned>(GetTableData(table), input, output, fftLength);
  }

  void ifft_real_f32_neon(const fft_table_real_f32 *table, const float *input, float *output, size_t fftLength)
  {
    COMMON_ASSERTS();
    NEON32::RealInverseFFT<Alignment::Unaligned>(GetTableData(table), input, output, fftLength);
  }
}
