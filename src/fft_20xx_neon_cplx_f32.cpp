#ifdef _MSC_VER
  #pragma warning (disable: 4711) // function <x> selected for automatic inline expansion
#endif

#include "fft_20xx_common.h"
#include "fft_20xx_table.h"
#include "fft_20xx_tiny.h"

static_assert(CPU_ARM64);
static_assert(SIMD_NEON);

#include <arm_neon.h>


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
  FORCE_INLINE auto LoadUnpackedComplex(const float *ptr) -> ComplexVec<VecType>
    { return { Load<Align>(ptr), Load<Align>(ptr + VecSize) }; }


  template <Alignment Align>
  FORCE_INLINE void StoreUnpackedComplex(float *ptr, ComplexVec<VecType> v)
  {
    Store<Align>(ptr, v.r);
    Store<Align>(ptr + VecSize, v.i);
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





  FORCE_INLINE void AddSub(ComplexVec<VecType> &a, ComplexVec<VecType> &b)
  {
    auto t = vaddq_f32(a.r, b.r);
    b.r = vsubq_f32(a.r, b.r);
    a.r = t;
    t = vaddq_f32(a.i, b.i);
    b.i = vsubq_f32(a.i, b.i);
    a.i = t;
  }



  template <Dir D>
  FORCE_INLINE void AddSubRot45(ComplexVec<VecType> &a, ComplexVec<VecType> &b)
  {
    if constexpr (D == Dir::Forward)
    {
      auto t = vaddq_f32(b.r, b.i);
      b.i = vsubq_f32(b.i, b.r);
      b.r = vmulq_n_f32(t, InvSqrt2<float>);
      b.i = vmulq_n_f32(b.i, InvSqrt2<float>);
    }
    else
    {
      auto t = vsubq_f32(b.r, b.i);
      b.i = vaddq_f32(b.i, b.r);
      b.r = vmulq_n_f32(t, InvSqrt2<float>);
      b.i = vmulq_n_f32(b.i, InvSqrt2<float>);
    }

    auto t = vaddq_f32(a.r, b.r);
    b.r = vsubq_f32(a.r, b.r);
    a.r = t;
    t = vaddq_f32(a.i, b.i);
    b.i = vsubq_f32(a.i, b.i);
    a.i = t;
  }



  template <Dir D>
  FORCE_INLINE void AddSubRot90(ComplexVec<VecType> &a, ComplexVec<VecType> &b)
  {
    if constexpr (D == Dir::Forward)
    {
      auto t = vaddq_f32(a.r, b.i);
      b.i = vsubq_f32(a.r, b.i);
      a.r = t;
      t = vsubq_f32(a.i, b.r);
      b.r = vaddq_f32(a.i, b.r);
      a.i = t;
    }
    else
    {
      auto t = vsubq_f32(a.r, b.i);
      b.i = vaddq_f32(a.r, b.i);
      a.r = t;
      t = vaddq_f32(a.i, b.r);
      b.r = vsubq_f32(a.i, b.r);
      a.i = t;
    }
    std::swap(b.r, b.i);
  }



  template <Dir D>
  FORCE_INLINE void AddSubRot135(ComplexVec<VecType> &a, ComplexVec<VecType> &b)
  {
    if constexpr (D == Dir::Forward)
    {
      auto t = vsubq_f32(b.i, b.r);
      b.i = vaddq_f32(b.i, b.r); // This is actually -b.i
      b.r = vmulq_n_f32(t, InvSqrt2<float>);
      b.i = vmulq_n_f32(b.i, InvSqrt2<float>);

      t = vaddq_f32(a.r, b.r);
      b.r = vsubq_f32(a.r, b.r);
      a.r = t;
      t = vsubq_f32(a.i, b.i); // Because b.i is -b.i, addition is subtraction!
      b.i = vaddq_f32(a.i, b.i);
      a.i = t;
    }
    else
    {
      auto t = vaddq_f32(b.i, b.r); // This is actually -b.r
      b.i = vsubq_f32(b.r, b.i);
      b.r = vmulq_n_f32(t, InvSqrt2<float>);
      b.i = vmulq_n_f32(b.i, InvSqrt2<float>);

      t = vsubq_f32(a.r, b.r); // because b.r is actually -b.r, swap the addition/subtraction
      b.r = vaddq_f32(a.r, b.r);
      a.r = t;
      t = vaddq_f32(a.i, b.i);
      b.i = vsubq_f32(a.i, b.i);
      a.i = t;
    }
  }


  [[nodiscard]] FORCE_INLINE auto ComplexMultiply(ComplexVec<VecType> a, ComplexVec<VecType> b) -> ComplexVec<VecType>
  {
    auto t = vmlsq_f32(vmulq_f32(a.r, b.r), a.i, b.i);
    a.i = vmlaq_f32(vmulq_f32(a.r, b.i), a.i, b.r);
    a.r = t;
    return a;
  }

  [[nodiscard]] FORCE_INLINE auto ScalarMultiply(ComplexVec<VecType> a, VecType b) -> ComplexVec<VecType>
  {
    return { vmulq_f32(a.r, b), vmulq_f32(a.i, b) };
  }

  template <Dir D>
  FORCE_INLINE void Rotate(ComplexVec<VecType> &a, ComplexVec<VecType> w)
  {
    if constexpr (D == Dir::Forward)
    {
      auto t = vmlsq_f32(vmulq_f32(a.r, w.r), a.i, w.i);
      a.i = vmlaq_f32(vmulq_f32(a.r, w.i), a.i, w.r);
      a.r = t;
    }
    else
    {
      auto t = vmlaq_f32(vmulq_f32(a.r, w.r), a.i, w.i);
      a.i = vmlsq_f32(vmulq_f32(a.i, w.r), a.r, w.i);
      a.r = t;
    }
  }


  template <Dir D>
  FORCE_INLINE void Rotate(ComplexVec<VecType> &a, Complex<float> w)
  {
    if constexpr (D == Dir::Forward)
    {
      auto t = vmlsq_n_f32(vmulq_n_f32(a.r, w.r), a.i, w.i);
      a.i = vmlaq_n_f32(vmulq_n_f32(a.r, w.i), a.i, w.r);
      a.r = t;
    }
    else
    {
      auto t = vmlaq_n_f32(vmulq_n_f32(a.r, w.r), a.i, w.i);
      a.i = vmlsq_n_f32(vmulq_n_f32(a.i, w.r), a.r, w.i);
      a.r = t;
    }
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
  void fft_cplx_interleaved_f32_neon(const fft_table_f32 *table, const float *input, float *output, size_t fftLength)
  {
    [[maybe_unused]] auto t = reinterpret_cast<const FFTTableImpl<float> *>(table);
    COMMON_ASSERTS();
    NEONCplx32::RunComplex<Dir::Forward, Alignment::Unaligned>(t->Data(), input, output, fftLength);
  }


  void ifft_cplx_interleaved_f32_neon(const fft_table_f32 *table, const float *input, float *output, size_t fftLength)
  {
    [[maybe_unused]] auto t = reinterpret_cast<const FFTTableImpl<float> *>(table);
    COMMON_ASSERTS();
    NEONCplx32::RunComplex<Dir::Inverse, Alignment::Unaligned>(t->Data(), input, output, fftLength);
  }
}
