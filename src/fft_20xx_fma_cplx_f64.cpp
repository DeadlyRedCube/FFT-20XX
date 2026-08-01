#ifdef _MSC_VER
  #pragma warning (disable: 4711) // function <x> selected for automatic inline expansion
#endif

#include "fft_20xx_common.h"
#include "fft_20xx_table.h"
#include "fft_20xx_tiny_cplx.h"

static_assert(CPU_X86 || CPU_X64);

static_assert(SIMD_SSE2);

#if COMPILER_GCC
  #pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

using namespace FFT20XX;

namespace FFT20XX::FMACplx64
{
  constexpr uint32_t VecSize = 4u;
  using VecType = __m256d;
  using ElementType = double;

  template <typename T>
  struct Angles
  {
    T w0;
    T w1;
  };


  bool IsSIMDAligned(const double *ptr)
    { return (uintptr_t(ptr) & (sizeof(VecType) - 1)) == 0; }


  FORCE_INLINE auto Set1(double f) -> VecType
  {
    return _mm256_set1_pd(f);
  }



  template <Alignment Align>
  FORCE_INLINE auto Load(const double *ptr)
  {
    if constexpr (Align == Alignment::Aligned)
      { return _mm256_load_pd(ptr); }
    else
      { return _mm256_loadu_pd(ptr); }
  }

  template <Alignment Align>
  FORCE_INLINE void Store(double *ptr, VecType v)
  {
    if constexpr (Align == Alignment::Aligned)
      { return _mm256_store_pd(ptr, v); }
    else
      { return _mm256_storeu_pd(ptr, v); }
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
    auto a = Load<Align>(ptr);
    auto b = Load<Align>(ptr + VecSize);

    // (A a | B b) (C c | D d) -> (A a | C c)
    auto firstHalves = _mm256_permute2f128_pd(a, b, 0x20);

    // (A a | B b) (C c | D d) -> (B b | D d)
    auto secondHalves = _mm256_permute2f128_pd(a, b, 0x31);

    return
      {
        // (A a | C c) (B b | D d) -> (A B | C D)
        _mm256_shuffle_pd(firstHalves, secondHalves, 0b0'0'0'0),

        // (A a | C c) (B b | D d) -> (a b | c d)
        _mm256_shuffle_pd(firstHalves, secondHalves, 0b1'1'1'1),
      };
  }



  template <Alignment Align>
  FORCE_INLINE void StorePackedComplex(double *ptr, ComplexVec<VecType> v)
  {
    // (A B C D) (a b c d) -> (A a C c)
    auto loHalves = _mm256_unpacklo_pd(v.r, v.i);

    // (A B C D) (a b c d) -> (B b D d)
    auto hiHalves = _mm256_unpackhi_pd(v.r, v.i);

    // (A a C c) (B b D d) -> (A a B b)
    auto a = _mm256_permute2f128_pd(loHalves, hiHalves, 0x20);

    // (A a C c) (B b D d) -> (C c D d)
    auto b = _mm256_permute2f128_pd(loHalves, hiHalves, 0x31);

    _mm256_storeu_pd(ptr, a);
    _mm256_storeu_pd(ptr + 4, b);
  }





  FORCE_INLINE void AddSub(ComplexVec<VecType> &a, ComplexVec<VecType> &b)
  {
    auto t = _mm256_add_pd(a.r, b.r);
    b.r = _mm256_sub_pd(a.r, b.r);
    a.r = t;
    t = _mm256_add_pd(a.i, b.i);
    b.i = _mm256_sub_pd(a.i, b.i);
    a.i = t;
  }



  template <Dir D>
  FORCE_INLINE void AddSubRot45(ComplexVec<VecType> &a, ComplexVec<VecType> &b)
  {
    const auto s = _mm256_set1_pd(InvSqrt2<double>);
    if constexpr (D == Dir::Forward)
    {
      auto t = _mm256_add_pd(b.r, b.i);
      b.i = _mm256_sub_pd(b.i, b.r);
      b.r = _mm256_mul_pd(s, t);
      b.i = _mm256_mul_pd(s, b.i);
    }
    else
    {
      auto t = _mm256_sub_pd(b.r, b.i);
      b.i = _mm256_add_pd(b.i, b.r);
      b.r = _mm256_mul_pd(s, t);
      b.i = _mm256_mul_pd(s, b.i);
    }

    auto t = _mm256_add_pd(a.r, b.r);
    b.r = _mm256_sub_pd(a.r, b.r);
    a.r = t;
    t = _mm256_add_pd(a.i, b.i);
    b.i = _mm256_sub_pd(a.i, b.i);
    a.i = t;
  }



  template <Dir D>
  FORCE_INLINE void AddSubRot90(ComplexVec<VecType> &a, ComplexVec<VecType> &b)
  {
    if constexpr (D == Dir::Forward)
    {
      auto t = _mm256_add_pd(a.r, b.i);
      b.i = _mm256_sub_pd(a.r, b.i);
      a.r = t;
      t = _mm256_sub_pd(a.i, b.r);
      b.r = _mm256_add_pd(a.i, b.r);
      a.i = t;
    }
    else
    {
      auto t = _mm256_sub_pd(a.r, b.i);
      b.i = _mm256_add_pd(a.r, b.i);
      a.r = t;
      t = _mm256_add_pd(a.i, b.r);
      b.r = _mm256_sub_pd(a.i, b.r);
      a.i = t;
    }
    std::swap(b.r, b.i);
  }



  template <Dir D>
  FORCE_INLINE void AddSubRot135(ComplexVec<VecType> &a, ComplexVec<VecType> &b)
  {
    const auto s = _mm256_set1_pd(InvSqrt2<double>);
    if constexpr (D == Dir::Forward)
    {
      auto t = _mm256_sub_pd(b.i, b.r);
      b.i = _mm256_add_pd(b.i, b.r); // This is actually -b.i
      b.r = _mm256_mul_pd(s, t);
      b.i = _mm256_mul_pd(s, b.i);

      t = _mm256_add_pd(a.r, b.r);
      b.r = _mm256_sub_pd(a.r, b.r);
      a.r = t;
      t = _mm256_sub_pd(a.i, b.i); // Because b.i is -b.i, addition is subtraction!
      b.i = _mm256_add_pd(a.i, b.i);
      a.i = t;
    }
    else
    {
      auto t = _mm256_add_pd(b.i, b.r); // This is actually -b.r
      b.i = _mm256_sub_pd(b.r, b.i);
      b.r = _mm256_mul_pd(s, t);
      b.i = _mm256_mul_pd(s, b.i);

      t = _mm256_sub_pd(a.r, b.r); // because b.r is actually -b.r, swap the addition/subtraction
      b.r = _mm256_add_pd(a.r, b.r);
      a.r = t;
      t = _mm256_add_pd(a.i, b.i);
      b.i = _mm256_sub_pd(a.i, b.i);
      a.i = t;
    }
  }


  [[nodiscard]] FORCE_INLINE auto ComplexMultiply(ComplexVec<VecType> a, ComplexVec<VecType> b) -> ComplexVec<VecType>
  {
    auto t = _mm256_fmsub_pd(a.r, b.r, _mm256_mul_pd(a.i, b.i));
    a.i = _mm256_fmadd_pd(a.r, b.i, _mm256_mul_pd(a.i, b.r));
    a.r = t;
    return a;
  }

  [[nodiscard]] FORCE_INLINE auto ScalarMultiply(ComplexVec<VecType> a, VecType b) -> ComplexVec<VecType>
  {
    return { _mm256_mul_pd(a.r, b), _mm256_mul_pd(a.i, b) };
  }

  template <Dir D>
  FORCE_INLINE void AddSubRot(ComplexVec<VecType> &a, ComplexVec<VecType> &b, auto angle)
  {
    ComplexVec<VecType> w;
    if constexpr (std::same_as<decltype(angle), Complex<ElementType>>)
    {
      w.r = _mm256_set1_pd(angle.r);
      w.i = _mm256_set1_pd(angle.i);
    }
    else
    {
      w = angle;
    }

    auto two = _mm256_set1_pd(2.0);
    if constexpr (D == Dir::Forward)
    {
      auto t1 = _mm256_fmadd_pd(b.r, w.r, _mm256_fnmadd_pd(b.i, w.i, a.r));
      auto t2 = _mm256_fmadd_pd(b.r, w.i, _mm256_fmadd_pd(b.i, w.r, a.i));
      b.r = _mm256_fmsub_pd(two, a.r, t1);
      a.r = t1;
      b.i = _mm256_fmsub_pd(two, a.i, t2);
      a.i = t2;
    }
    else
    {
      auto t1 = _mm256_fmadd_pd(b.r, w.r, _mm256_fmadd_pd(b.i, w.i, a.r));
      auto t2 = _mm256_fnmadd_pd(b.r, w.i, _mm256_fmadd_pd(b.i, w.r, a.i));
      b.r = _mm256_fmsub_pd(two, a.r, t1);
      a.r = t1;
      b.i = _mm256_fmsub_pd(two, a.i, t2);
      a.i = t2;
    }
  }



  template <Dir D>
  FORCE_INLINE void AddSubRotPlus90(ComplexVec<VecType> &a, ComplexVec<VecType> &b, auto angle)
  {
    ComplexVec<VecType> w;
    if constexpr (std::same_as<decltype(angle), Complex<ElementType>>)
    {
      w.r = _mm256_set1_pd(angle.r);
      w.i = _mm256_set1_pd(angle.i);
    }
    else
    {
      w = angle;
    }

    auto two = _mm256_set1_pd(2.0);
    if constexpr (D == Dir::Forward)
    {
      auto t1 = _mm256_fmadd_pd(b.r, w.i, _mm256_fmadd_pd(b.i, w.r, a.r));
      auto t2 = _mm256_fnmadd_pd(b.r, w.r, _mm256_fmadd_pd(b.i, w.i, a.i));
      b.r = _mm256_fmsub_pd(two, a.r, t1);
      a.r = t1;
      b.i = _mm256_fmsub_pd(two, a.i, t2);
      a.i = t2;
    }
    else
    {
      auto t1 = _mm256_fmadd_pd(b.r, w.i, _mm256_fnmadd_pd(b.i, w.r, a.r));
      auto t2 = _mm256_fmadd_pd(b.r, w.r, _mm256_fmadd_pd(b.i, w.i, a.i));
      b.r = _mm256_fmsub_pd(two, a.r, t1);
      a.r = t1;
      b.i = _mm256_fmsub_pd(two, a.i, t2);
      a.i = t2;
    }
  }

  FORCE_INLINE void Transpose4x4(
    ComplexVec<__m256d> &a,
    ComplexVec<__m256d> &b,
    ComplexVec<__m256d> &c,
    ComplexVec<__m256d> &d)
  {
    // For f64 we have to transpose cross-lane, start by swapping all the 2x2s
    auto t0 = _mm256_unpacklo_pd(a.r, b.r);
    auto t1 = _mm256_unpackhi_pd(a.r, b.r);
    auto t2 = _mm256_unpacklo_pd(c.r, d.r);
    auto t3 = _mm256_unpackhi_pd(c.r, d.r);

    // Now swap the upper-right and lower-left 2x2 chunks
    a.r = _mm256_permute2f128_pd(t0, t2, 0x20);
    b.r = _mm256_permute2f128_pd(t1, t3, 0x20);
    c.r = _mm256_permute2f128_pd(t0, t2, 0x31);
    d.r = _mm256_permute2f128_pd(t1, t3, 0x31);

    t0 = _mm256_unpacklo_pd(a.i, b.i);
    t1 = _mm256_unpackhi_pd(a.i, b.i);
    t2 = _mm256_unpacklo_pd(c.i, d.i);
    t3 = _mm256_unpackhi_pd(c.i, d.i);

    // Now swap the upper-right and lower-left 2x2 chunks
    a.i = _mm256_permute2f128_pd(t0, t2, 0x20);
    b.i = _mm256_permute2f128_pd(t1, t3, 0x20);
    c.i = _mm256_permute2f128_pd(t0, t2, 0x31);
    d.i = _mm256_permute2f128_pd(t1, t3, 0x31);
  }

  #include "fft_20xx_algo_cplx.h"
}


using namespace FFT20XX;
using namespace fft_20xx::c;
extern "C"
{
  void fft_cplx_f64_fma(const fft_table_cplx_f64 *table, const double *input, double *output, size_t fftLength)
  {
    COMMON_ASSERTS();
    if (FMACplx64::IsSIMDAligned(input) && FMACplx64::IsSIMDAligned(output))
      { FMACplx64::RunComplex<Dir::Forward, Alignment::Aligned>(GetTableData(table), input, output, fftLength); }
    else
      { FMACplx64::RunComplex<Dir::Forward, Alignment::Unaligned>(GetTableData(table), input, output, fftLength); }
  }


  void ifft_cplx_f64_fma(const fft_table_cplx_f64 *table, const double *input, double *output, size_t fftLength)
  {
    COMMON_ASSERTS();
    if (FMACplx64::IsSIMDAligned(input) && FMACplx64::IsSIMDAligned(output))
      { FMACplx64::RunComplex<Dir::Inverse, Alignment::Aligned>(GetTableData(table), input, output, fftLength); }
    else
      { FMACplx64::RunComplex<Dir::Inverse, Alignment::Unaligned>(GetTableData(table), input, output, fftLength); }
  }
}
