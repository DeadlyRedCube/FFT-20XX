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

namespace FFT20XX::FMA32Cplx
{
  constexpr uint32_t VecSize = 8u;
  using VecType = __m256;
  using ElementType = float;

  template <typename T>
  struct Angles
  {
    T w0;
    T w1;
  };


  bool IsSIMDAligned(const float *ptr)
    { return (uintptr_t(ptr) & (sizeof(VecType) - 1)) == 0; }


  FORCE_INLINE auto Set1(float f) -> VecType
  {
    return _mm256_set1_ps(f);
  }

  [[nodiscard]] FORCE_INLINE auto Set2(float lo, float hi) -> VecType
  {
    return _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_set1_ps(lo)), _mm_set1_ps(hi), 1);
  }


  template <Alignment Align>
  FORCE_INLINE auto Load(const float *ptr)
  {
    if constexpr (Align == Alignment::Aligned)
      { return _mm256_load_ps(ptr); }
    else
      { return _mm256_loadu_ps(ptr); }
  }

  template <Alignment Align>
  FORCE_INLINE void Store(float *ptr, VecType v)
  {
    if constexpr (Align == Alignment::Aligned)
      { return _mm256_store_ps(ptr, v); }
    else
      { return _mm256_storeu_ps(ptr, v); }
  }

  template <Alignment Align>
  FORCE_INLINE auto LoadHalf(const float *ptr)
  {
    if constexpr (Align == Alignment::Aligned)
      { return _mm_load_ps(ptr); }
    else
      { return _mm_loadu_ps(ptr); }
  }

  template <Alignment Align>
  FORCE_INLINE void Store(float *ptr, __m128 v)
  {
    if constexpr (Align == Alignment::Aligned)
      { return _mm_store_ps(ptr, v); }
    else
      { return _mm_storeu_ps(ptr, v); }
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
  FORCE_INLINE auto LoadUnpackedComplexHalves(const float *ptr0, const float *ptr1) -> ComplexVec<VecType>
  {
    auto r = _mm256_insertf128_ps(_mm256_castps128_ps256(LoadHalf<Align>(ptr0)), LoadHalf<Align>(ptr1), 1);
    auto i = _mm256_insertf128_ps(_mm256_castps128_ps256(LoadHalf<Align>(ptr0 + VecSize)), LoadHalf<Align>(ptr1 + VecSize), 1);
    return { r, i };
  }

  template <Alignment Align>
  FORCE_INLINE void StoreUnpackedComplexHalves(float *ptr0, float *ptr1, ComplexVec<VecType> v)
  {
    Store<Align>(ptr0, _mm256_castps256_ps128(v.r));
    Store<Align>(ptr1, _mm256_extractf128_ps(v.r, 1));
    Store<Align>(ptr0 + VecSize, _mm256_castps256_ps128(v.i));
    Store<Align>(ptr1 + VecSize, _mm256_extractf128_ps(v.i, 1));
  }


  template <Alignment Align>
  FORCE_INLINE auto LoadPackedComplex(const float *ptr) -> ComplexVec<VecType>
  {
    auto [r, i] = LoadUnpackedComplex<Align>(ptr);
    auto t = _mm256_permute2f128_ps(r, i, 0x20);
    i = _mm256_permute2f128_ps(r, i, 0x31);
    r = _mm256_shuffle_ps(t, i, _MM_SHUFFLE(2, 0, 2, 0));
    i = _mm256_shuffle_ps(t, i, _MM_SHUFFLE(3, 1, 3, 1));
    return {r, i};
  }



  template <Alignment Align>
  FORCE_INLINE void StorePackedComplex(float *ptr, ComplexVec<VecType> v)
  {
      // (A B C D E F G H) (a b c d e f g h) -> (A a B b E e F f)
      auto AaBbEeFf = _mm256_unpacklo_ps(v.r, v.i);

      // (A B C D E F G H) (a b c d e f g h) -> (C c D d G g H h)
      auto CcDdGgHh = _mm256_unpackhi_ps(v.r, v.i);

      auto EeFf = _mm256_extractf128_ps(AaBbEeFf, 1);
      auto GgHh = _mm256_extractf128_ps(CcDdGgHh, 1);
      auto AaBb = _mm256_castps256_ps128(AaBbEeFf);
      auto CcDd = _mm256_castps256_ps128(CcDdGgHh);

      if constexpr (Align == Alignment::Aligned)
      {
        _mm_store_ps(ptr + 0, AaBb);
        _mm_store_ps(ptr + 4, CcDd);
        _mm_store_ps(ptr + 8, EeFf);
        _mm_store_ps(ptr + 12, GgHh);
      }
      else
      {
        _mm_storeu_ps(ptr + 0, AaBb);
        _mm_storeu_ps(ptr + 4, CcDd);
        _mm_storeu_ps(ptr + 8, EeFf);
        _mm_storeu_ps(ptr + 12, GgHh);
      }
  }


    template <Alignment Align>
  FORCE_INLINE auto StoreUnpackedLowerHalves(float *ptr, ComplexVec<VecType> a, ComplexVec<VecType> b)
  {
    auto aR = _mm256_castps256_ps128(a.r);
    auto aI = _mm256_castps256_ps128(a.i);
    auto bR = _mm256_castps256_ps128(b.r);
    auto bI = _mm256_castps256_ps128(b.i);
    Store<Align>(ptr, aR);
    Store<Align>(ptr + 4, bR);
    Store<Align>(ptr + VecSize, aI);
    Store<Align>(ptr + 4 + VecSize, bI);
  }

  template <Alignment Align>
  FORCE_INLINE auto StoreUnpackedUpperHalves(float *ptr, ComplexVec<VecType> a, ComplexVec<VecType> b)
  {
    auto aR = _mm256_extractf128_ps(a.r, 1);
    auto aI = _mm256_extractf128_ps(a.i, 1);
    auto bR = _mm256_extractf128_ps(b.r, 1);
    auto bI = _mm256_extractf128_ps(b.i, 1);
    Store<Align>(ptr, aR);
    Store<Align>(ptr + 4, bR);
    Store<Align>(ptr + VecSize, aI);
    Store<Align>(ptr + 4 + VecSize, bI);
  }

  FORCE_INLINE void AddSub(ComplexVec<VecType> &a, ComplexVec<VecType> &b)
  {
    auto t = _mm256_add_ps(a.r, b.r);
    b.r = _mm256_sub_ps(a.r, b.r);
    a.r = t;
    t = _mm256_add_ps(a.i, b.i);
    b.i = _mm256_sub_ps(a.i, b.i);
    a.i = t;
  }



  template <Dir D>
  FORCE_INLINE void AddSubRot45(ComplexVec<VecType> &a, ComplexVec<VecType> &b)
  {
    const auto s = _mm256_set1_ps(InvSqrt2<float>);
    if constexpr (D == Dir::Forward)
    {
      auto t = _mm256_add_ps(b.r, b.i);
      b.i = _mm256_sub_ps(b.i, b.r);
      b.r = _mm256_mul_ps(s, t);
      b.i = _mm256_mul_ps(s, b.i);
    }
    else
    {
      auto t = _mm256_sub_ps(b.r, b.i);
      b.i = _mm256_add_ps(b.i, b.r);
      b.r = _mm256_mul_ps(s, t);
      b.i = _mm256_mul_ps(s, b.i);
    }

    auto t = _mm256_add_ps(a.r, b.r);
    b.r = _mm256_sub_ps(a.r, b.r);
    a.r = t;
    t = _mm256_add_ps(a.i, b.i);
    b.i = _mm256_sub_ps(a.i, b.i);
    a.i = t;
  }



  template <Dir D>
  FORCE_INLINE void AddSubRot90(ComplexVec<VecType> &a, ComplexVec<VecType> &b)
  {
    if constexpr (D == Dir::Forward)
    {
      auto t = _mm256_add_ps(a.r, b.i);
      b.i = _mm256_sub_ps(a.r, b.i);
      a.r = t;
      t = _mm256_sub_ps(a.i, b.r);
      b.r = _mm256_add_ps(a.i, b.r);
      a.i = t;
    }
    else
    {
      auto t = _mm256_sub_ps(a.r, b.i);
      b.i = _mm256_add_ps(a.r, b.i);
      a.r = t;
      t = _mm256_add_ps(a.i, b.r);
      b.r = _mm256_sub_ps(a.i, b.r);
      a.i = t;
    }
    std::swap(b.r, b.i);
  }



  template <Dir D>
  FORCE_INLINE void AddSubRot135(ComplexVec<VecType> &a, ComplexVec<VecType> &b)
  {
    const auto s = _mm256_set1_ps(InvSqrt2<float>);
    if constexpr (D == Dir::Forward)
    {
      auto t = _mm256_sub_ps(b.i, b.r);
      b.i = _mm256_add_ps(b.i, b.r); // This is actually -b.i
      b.r = _mm256_mul_ps(s, t);
      b.i = _mm256_mul_ps(s, b.i);

      t = _mm256_add_ps(a.r, b.r);
      b.r = _mm256_sub_ps(a.r, b.r);
      a.r = t;
      t = _mm256_sub_ps(a.i, b.i); // Because b.i is -b.i, addition is subtraction!
      b.i = _mm256_add_ps(a.i, b.i);
      a.i = t;
    }
    else
    {
      auto t = _mm256_add_ps(b.i, b.r); // This is actually -b.r
      b.i = _mm256_sub_ps(b.r, b.i);
      b.r = _mm256_mul_ps(s, t);
      b.i = _mm256_mul_ps(s, b.i);

      t = _mm256_sub_ps(a.r, b.r); // because b.r is actually -b.r, swap the addition/subtraction
      b.r = _mm256_add_ps(a.r, b.r);
      a.r = t;
      t = _mm256_add_ps(a.i, b.i);
      b.i = _mm256_sub_ps(a.i, b.i);
      a.i = t;
    }
  }


  template <Dir D>
  FORCE_INLINE void AddSubRot(ComplexVec<VecType> &a, ComplexVec<VecType> &b, auto angle)
  {
    ComplexVec<VecType> w;
    if constexpr (std::same_as<decltype(angle), Complex<ElementType>>)
    {
      w.r = _mm256_set1_ps(angle.r);
      w.i = _mm256_set1_ps(angle.i);
    }
    else if constexpr (std::same_as<decltype(angle), std::pair<Complex<ElementType>, Complex<ElementType>>>)
    {
      w.r = Set2(angle.first.r, angle.second.r);
      w.i = Set2(angle.first.i, angle.second.i);
    }
    else
    {
      w = angle;
    }

    auto two = _mm256_set1_ps(2.0f);
    if constexpr (D == Dir::Forward)
    {
      auto t1 = _mm256_fmadd_ps(b.r, w.r, _mm256_fnmadd_ps(b.i, w.i, a.r));
      auto t2 = _mm256_fmadd_ps(b.r, w.i, _mm256_fmadd_ps(b.i, w.r, a.i));
      b.r = _mm256_fmsub_ps(two, a.r, t1);
      a.r = t1;
      b.i = _mm256_fmsub_ps(two, a.i, t2);
      a.i = t2;
    }
    else
    {
      auto t1 = _mm256_fmadd_ps(b.r, w.r, _mm256_fmadd_ps(b.i, w.i, a.r));
      auto t2 = _mm256_fnmadd_ps(b.r, w.i, _mm256_fmadd_ps(b.i, w.r, a.i));
      b.r = _mm256_fmsub_ps(two, a.r, t1);
      a.r = t1;
      b.i = _mm256_fmsub_ps(two, a.i, t2);
      a.i = t2;
    }
  }



  template <Dir D>
  FORCE_INLINE void AddSubRotPlus90(ComplexVec<VecType> &a, ComplexVec<VecType> &b, auto angle)
  {
    ComplexVec<VecType> w;
    if constexpr (std::same_as<decltype(angle), Complex<ElementType>>)
    {
      w.r = _mm256_set1_ps(angle.r);
      w.i = _mm256_set1_ps(angle.i);
    }
    else if constexpr (std::same_as<decltype(angle), std::pair<Complex<ElementType>, Complex<ElementType>>>)
    {
      w.r = Set2(angle.first.r, angle.second.r);
      w.i = Set2(angle.first.i, angle.second.i);
    }
    else
    {
      w = angle;
    }

    auto two = _mm256_set1_ps(2.0f);
    if constexpr (D == Dir::Forward)
    {
      auto t1 = _mm256_fmadd_ps(b.r, w.i, _mm256_fmadd_ps(b.i, w.r, a.r));
      auto t2 = _mm256_fnmadd_ps(b.r, w.r, _mm256_fmadd_ps(b.i, w.i, a.i));
      b.r = _mm256_fmsub_ps(two, a.r, t1);
      a.r = t1;
      b.i = _mm256_fmsub_ps(two, a.i, t2);
      a.i = t2;
    }
    else
    {
      auto t1 = _mm256_fmadd_ps(b.r, w.i, _mm256_fnmadd_ps(b.i, w.r, a.r));
      auto t2 = _mm256_fmadd_ps(b.r, w.r, _mm256_fmadd_ps(b.i, w.i, a.i));
      b.r = _mm256_fmsub_ps(two, a.r, t1);
      a.r = t1;
      b.i = _mm256_fmsub_ps(two, a.i, t2);
      a.i = t2;
    }
  }


  [[nodiscard]] FORCE_INLINE auto ComplexMultiply(ComplexVec<VecType> a, ComplexVec<VecType> b) -> ComplexVec<VecType>
  {
    auto t = _mm256_fmsub_ps(a.r, b.r, _mm256_mul_ps(a.i, b.i));
    a.i = _mm256_fmadd_ps(a.r, b.i, _mm256_mul_ps(a.i, b.r));
    a.r = t;
    return a;
  }

  [[nodiscard]] FORCE_INLINE auto ScalarMultiply(ComplexVec<VecType> a, VecType b) -> ComplexVec<VecType>
  {
    return { _mm256_mul_ps(a.r, b), _mm256_mul_ps(a.i, b) };
  }

  FORCE_INLINE void Transpose4x4(
    ComplexVec<VecType> &a,
    ComplexVec<VecType> &b,
    ComplexVec<VecType> &c,
    ComplexVec<VecType> &d)
  {
    {
      auto t1 = _mm256_shuffle_ps(c.r, d.r, 0x44);
      d.r = _mm256_shuffle_ps(c.r, d.r, 0xEE);
      auto t2 = _mm256_shuffle_ps(a.r, b.r, 0xEE);
      b.r = _mm256_shuffle_ps(a.r, b.r, 0x44);

      a.r = _mm256_shuffle_ps(b.r, t1, 0x88);
      b.r = _mm256_shuffle_ps(b.r, t1, 0xDD);
      c.r = _mm256_shuffle_ps(t2, d.r, 0x88);
      d.r = _mm256_shuffle_ps(t2, d.r, 0xDD);
    }

    {
      auto t1 = _mm256_shuffle_ps(c.i, d.i, 0x44);
      d.i = _mm256_shuffle_ps(c.i, d.i, 0xEE);
      auto t2 = _mm256_shuffle_ps(a.i, b.i, 0xEE);
      b.i = _mm256_shuffle_ps(a.i, b.i, 0x44);

      a.i = _mm256_shuffle_ps(b.i, t1, 0x88);
      b.i = _mm256_shuffle_ps(b.i, t1, 0xDD);
      c.i = _mm256_shuffle_ps(t2, d.i, 0x88);
      d.i = _mm256_shuffle_ps(t2, d.i, 0xDD);
    }
  }

  #include "fft_20xx_algo_cplx.h"
}


using namespace FFT20XX;
using namespace fft_20xx::c;
extern "C"
{
  void fft_cplx_interleaved_f32_sse2(const fft_table_f32 *table, const float *input, float *output, size_t fftLength);

  void fft_cplx_interleaved_f32_fma(const fft_table_f32 *table, const float *input, float *output, size_t fftLength)
  {
    if (fftLength < 64)
      { return fft_cplx_interleaved_f32_sse2(table, input, output, fftLength); }
    [[maybe_unused]] auto t = reinterpret_cast<const FFTTableImpl<float> *>(table);
    COMMON_ASSERTS();
    if (FMA32Cplx::IsSIMDAligned(input) && FMA32Cplx::IsSIMDAligned(output))
      { FMA32Cplx::RunComplex<Dir::Forward, Alignment::Aligned>(t->Data(), input, output, fftLength); }
    else
      { FMA32Cplx::RunComplex<Dir::Forward, Alignment::Unaligned>(t->Data(), input, output, fftLength); }
  }

  void ifft_cplx_interleaved_f32_sse2(const fft_table_f32 *table, const float *input, float *output, size_t fftLength);

  void ifft_cplx_interleaved_f32_fma(const fft_table_f32 *table, const float *input, float *output, size_t fftLength)
  {
    if (fftLength < 64)
      { return ifft_cplx_interleaved_f32_sse2(table, input, output, fftLength); }
    [[maybe_unused]] auto t = reinterpret_cast<const FFTTableImpl<float> *>(table);
    COMMON_ASSERTS();
    if (FMA32Cplx::IsSIMDAligned(input) && FMA32Cplx::IsSIMDAligned(output))
      { FMA32Cplx::RunComplex<Dir::Inverse, Alignment::Aligned>(t->Data(), input, output, fftLength); }
    else
      { FMA32Cplx::RunComplex<Dir::Inverse, Alignment::Unaligned>(t->Data(), input, output, fftLength); }
  }
}
