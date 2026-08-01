#ifdef _MSC_VER
  #pragma warning (disable: 4711) // function <x> selected for automatic inline expansion
#endif

#include "fft_20xx_common.h"
#include "fft_20xx_table.h"
#include "fft_20xx_tiny_cplx.h"

static_assert(CPU_ARM64);
static_assert(SIMD_NEON);

#if COMPILER_GCC
  #pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

using namespace FFT20XX;

namespace FFT20XX::NEONCplx32
{
  constexpr uint32_t VecSize = 4u;
  using VecType = float32x4_t;
  using ElementType = float;

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



  FORCE_INLINE void Transpose4x4(
    ComplexVec<VecType> &a,
    ComplexVec<VecType> &b,
    ComplexVec<VecType> &c,
    ComplexVec<VecType> &d)
  {
    {
      auto ta = vtrn1q_f32(a.r, b.r);
      auto tb = vtrn2q_f32(a.r, b.r);
      auto tc = vtrn1q_f32(c.r, d.r);
      auto td = vtrn2q_f32(c.r, d.r);

      a.r = vreinterpretq_f32_f64(vtrn1q_f64(vreinterpretq_f64_f32(ta), vreinterpretq_f64_f32(tc)));
      b.r = vreinterpretq_f32_f64(vtrn1q_f64(vreinterpretq_f64_f32(tb), vreinterpretq_f64_f32(td)));
      c.r = vreinterpretq_f32_f64(vtrn2q_f64(vreinterpretq_f64_f32(ta), vreinterpretq_f64_f32(tc)));
      d.r = vreinterpretq_f32_f64(vtrn2q_f64(vreinterpretq_f64_f32(tb), vreinterpretq_f64_f32(td)));
    }

    {
      auto ta = vtrn1q_f32(a.i, b.i);
      auto tb = vtrn2q_f32(a.i, b.i);
      auto tc = vtrn1q_f32(c.i, d.i);
      auto td = vtrn2q_f32(c.i, d.i);

      a.i = vreinterpretq_f32_f64(vtrn1q_f64(vreinterpretq_f64_f32(ta), vreinterpretq_f64_f32(tc)));
      b.i = vreinterpretq_f32_f64(vtrn1q_f64(vreinterpretq_f64_f32(tb), vreinterpretq_f64_f32(td)));
      c.i = vreinterpretq_f32_f64(vtrn2q_f64(vreinterpretq_f64_f32(ta), vreinterpretq_f64_f32(tc)));
      d.i = vreinterpretq_f32_f64(vtrn2q_f64(vreinterpretq_f64_f32(tb), vreinterpretq_f64_f32(td)));
    }
  }

  #include "fft_20xx_algo_cplx.h"
}


using namespace FFT20XX;
using namespace fft_20xx::c;
extern "C"
{
  void fft_cplx_f32_neon(const fft_table_cplx_f32 *table, const float *input, float *output, size_t fftLength)
  {
    COMMON_ASSERTS();
    NEONCplx32::RunComplex<Dir::Forward, Alignment::Unaligned>(GetTableData(table), input, output, fftLength);
  }


  void ifft_cplx_f32_neon(const fft_table_cplx_f32 *table, const float *input, float *output, size_t fftLength)
  {
    COMMON_ASSERTS();
    NEONCplx32::RunComplex<Dir::Inverse, Alignment::Unaligned>(GetTableData(table), input, output, fftLength);
  }
}
