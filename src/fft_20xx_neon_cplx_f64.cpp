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

namespace FFT20XX::NEONCplx64
{
  constexpr uint32_t VecSize = 2u;
  using VecType = float64x2_t;
  using ElementType = double;

  template <typename T>
  struct Angles
  {
    T w0;
    T w1;
    T w2;
  };


  bool IsSIMDAligned(const double *ptr)
    { return (uintptr_t(ptr) & (sizeof(VecType) - 1)) == 0; }


  FORCE_INLINE auto Set1(double f) -> VecType
  {
    return vdupq_n_f64(f);
  }



  template <Alignment Align>
  FORCE_INLINE auto Load(const double *ptr) -> VecType
  {
    return vld1q_f64(ptr);
  }

  template <Alignment Align>
  FORCE_INLINE void Store(double *ptr, VecType v)
  {
    vst1q_f64(ptr, v);
  }



  template <Alignment Align>
  FORCE_INLINE auto LoadUnpackedComplex(const double *ptr) -> ComplexVec<VecType>
    { return { Load<Align>(ptr), Load<Align>(ptr + VecSize) }; }


  template <Alignment Align>
  FORCE_INLINE void StoreUnpackedComplex(double *ptr, ComplexVec<VecType> v)
  {
    Store<Align>(ptr, v.r);
    Store<Align>(ptr + VecSize, v.i);
  }



  template <Alignment Align>
  FORCE_INLINE auto LoadPackedComplex(const double *ptr) -> ComplexVec<VecType>
  {
    auto r = vld2q_f64(ptr);
    return { r.val[0], r.val[1] };
  }



  template <Alignment Align>
  FORCE_INLINE void StorePackedComplex(double *ptr, ComplexVec<VecType> v)
  {
    vst2q_f64(ptr, (float64x2x2_t{v.r, v.i}));
  }



  FORCE_INLINE void AddSub(ComplexVec<VecType> &a, ComplexVec<VecType> &b)
  {
    auto t = vaddq_f64(a.r, b.r);
    b.r = vsubq_f64(a.r, b.r);
    a.r = t;
    t = vaddq_f64(a.i, b.i);
    b.i = vsubq_f64(a.i, b.i);
    a.i = t;
  }



  template <Dir D>
  FORCE_INLINE void AddSubRot45(ComplexVec<VecType> &a, ComplexVec<VecType> &b)
  {
    if constexpr (D == Dir::Forward)
    {
      auto t = vaddq_f64(b.r, b.i);
      b.i = vsubq_f64(b.i, b.r);
      b.r = vmulq_n_f64(t, InvSqrt2<double>);
      b.i = vmulq_n_f64(b.i, InvSqrt2<double>);
    }
    else
    {
      auto t = vsubq_f64(b.r, b.i);
      b.i = vaddq_f64(b.i, b.r);
      b.r = vmulq_n_f64(t, InvSqrt2<double>);
      b.i = vmulq_n_f64(b.i, InvSqrt2<double>);
    }

    auto t = vaddq_f64(a.r, b.r);
    b.r = vsubq_f64(a.r, b.r);
    a.r = t;
    t = vaddq_f64(a.i, b.i);
    b.i = vsubq_f64(a.i, b.i);
    a.i = t;
  }



  template <Dir D>
  FORCE_INLINE void AddSubRot90(ComplexVec<VecType> &a, ComplexVec<VecType> &b)
  {
    if constexpr (D == Dir::Forward)
    {
      auto t = vaddq_f64(a.r, b.i);
      b.i = vsubq_f64(a.r, b.i);
      a.r = t;
      t = vsubq_f64(a.i, b.r);
      b.r = vaddq_f64(a.i, b.r);
      a.i = t;
    }
    else
    {
      auto t = vsubq_f64(a.r, b.i);
      b.i = vaddq_f64(a.r, b.i);
      a.r = t;
      t = vaddq_f64(a.i, b.r);
      b.r = vsubq_f64(a.i, b.r);
      a.i = t;
    }
    std::swap(b.r, b.i);
  }



  template <Dir D>
  FORCE_INLINE void AddSubRot135(ComplexVec<VecType> &a, ComplexVec<VecType> &b)
  {
    if constexpr (D == Dir::Forward)
    {
      auto t = vsubq_f64(b.i, b.r);
      b.i = vaddq_f64(b.i, b.r); // This is actually -b.i
      b.r = vmulq_n_f64(t, InvSqrt2<double>);
      b.i = vmulq_n_f64(b.i, InvSqrt2<double>);

      t = vaddq_f64(a.r, b.r);
      b.r = vsubq_f64(a.r, b.r);
      a.r = t;
      t = vsubq_f64(a.i, b.i); // Because b.i is -b.i, addition is subtraction!
      b.i = vaddq_f64(a.i, b.i);
      a.i = t;
    }
    else
    {
      auto t = vaddq_f64(b.i, b.r); // This is actually -b.r
      b.i = vsubq_f64(b.r, b.i);
      b.r = vmulq_n_f64(t, InvSqrt2<double>);
      b.i = vmulq_n_f64(b.i, InvSqrt2<double>);

      t = vsubq_f64(a.r, b.r); // because b.r is actually -b.r, swap the addition/subtraction
      b.r = vaddq_f64(a.r, b.r);
      a.r = t;
      t = vaddq_f64(a.i, b.i);
      b.i = vsubq_f64(a.i, b.i);
      a.i = t;
    }
  }


  [[nodiscard]] FORCE_INLINE auto ComplexMultiply(ComplexVec<VecType> a, ComplexVec<VecType> b) -> ComplexVec<VecType>
  {
    auto t = vfmsq_f64(vmulq_f64(a.r, b.r), a.i, b.i);
    a.i = vfmaq_f64(vmulq_f64(a.r, b.i), a.i, b.r);
    a.r = t;
    return a;
  }

  [[nodiscard]] FORCE_INLINE auto ScalarMultiply(ComplexVec<VecType> a, VecType b) -> ComplexVec<VecType>
  {
    return { vmulq_f64(a.r, b), vmulq_f64(a.i, b) };
  }

  template <Dir D>
  FORCE_INLINE void Rotate(ComplexVec<VecType> &a, ComplexVec<VecType> w)
  {
    if constexpr (D == Dir::Forward)
    {
      auto t = vfmsq_f64(vmulq_f64(a.r, w.r), a.i, w.i);
      a.i = vfmaq_f64(vmulq_f64(a.r, w.i), a.i, w.r);
      a.r = t;
    }
    else
    {
      auto t = vfmaq_f64(vmulq_f64(a.r, w.r), a.i, w.i);
      a.i = vfmsq_f64(vmulq_f64(a.i, w.r), a.r, w.i);
      a.r = t;
    }
  }


  template <Dir D, std::floating_point F>
  FORCE_INLINE void Rotate(ComplexVec<VecType> &a, Complex<F> w)
  {
    if constexpr (D == Dir::Forward)
    {
      auto t = vfmsq_n_f64(vmulq_n_f64(a.r, w.r), a.i, w.i);
      a.i = vfmaq_n_f64(vmulq_n_f64(a.r, w.i), a.i, w.r);
      a.r = t;
    }
    else
    {
      auto t = vfmaq_n_f64(vmulq_n_f64(a.r, w.r), a.i, w.i);
      a.i = vfmsq_n_f64(vmulq_n_f64(a.i, w.r), a.r, w.i);
      a.r = t;
    }
  }



  FORCE_INLINE void Transpose2x2(
    ComplexVec<VecType> &a,
    ComplexVec<VecType> &b)
  {
    {
      auto t = vtrn1q_f64(a.r, b.r);
      b.r = vtrn2q_f64(a.r, b.r);
      a.r = t;
    }

    {
      auto t = vtrn1q_f64(a.i, b.i);
      b.i = vtrn2q_f64(a.i, b.i);
      a.i = t;
    }
  }

  #include "fft_20xx_algo_cplx.h"
}


using namespace FFT20XX;
using namespace fft_20xx::c;
extern "C"
{
  void fft_cplx_f64_neon(const fft_table_cplx_f64 *table, const double *input, double *output, size_t fftLength)
  {
    COMMON_ASSERTS();
    NEONCplx64::RunComplex<Dir::Forward, Alignment::Unaligned>(GetTableData(table), input, output, fftLength);
  }


  void ifft_cplx_f64_neon(const fft_table_cplx_f64 *table, const double *input, double *output, size_t fftLength)
  {
    COMMON_ASSERTS();
    NEONCplx64::RunComplex<Dir::Inverse, Alignment::Unaligned>(GetTableData(table), input, output, fftLength);
  }
}
