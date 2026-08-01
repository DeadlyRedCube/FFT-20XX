#ifdef _MSC_VER
  #pragma warning (disable: 4711) // function <x> selected for automatic inline expansion
#endif

#include "fft_20xx_common.h"
#include "fft_20xx_table.h"
#include "fft_20xx_tiny_cplx.h"
#include "fft_20xx_tiny_real.h"

static_assert(CPU_X86 || CPU_X64);

static_assert(SIMD_FMA);

#if COMPILER_GCC
  #pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

using namespace FFT20XX;

namespace FFT20XX::FMA32
{
  constexpr uint32_t VecSize = 8u;
  using VecType = __m256;
  using HalfType = __m128;
  using ElementType = float;
  constexpr auto MinimumRealFFTLength = 128;
  constexpr bool UseFMAOptimization = true;

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
  FORCE_INLINE auto LoadHalf(const float *ptr)
  {
    if constexpr (Align == Alignment::Aligned)
      { return _mm_load_ps(ptr); }
    else
      { return _mm_loadu_ps(ptr); }
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
  FORCE_INLINE void Store(float *ptr, __m128 v)
  {
    if constexpr (Align == Alignment::Aligned)
      { return _mm_store_ps(ptr, v); }
    else
      { return _mm_storeu_ps(ptr, v); }
  }

  [[nodiscard]] FORCE_INLINE auto Add(__m256 a, __m256 b) -> __m256
  {
    return _mm256_add_ps(a, b);
  }

  [[nodiscard]] FORCE_INLINE auto Sub(__m256 a, __m256 b) -> __m256
  {
    return _mm256_sub_ps(a, b);
  }

  [[nodiscard]] FORCE_INLINE auto Mul(__m256 a, __m256 b) -> __m256
  {
    return _mm256_mul_ps(a, b);
  }

  [[nodiscard]] FORCE_INLINE auto Mul(__m256 a, float b) -> __m256
  {
    return _mm256_mul_ps(a, _mm256_set1_ps(b));
  }

  [[nodiscard]] FORCE_INLINE auto Mul(__m256 a, std::pair<float, float> b) -> __m256
  {
    return _mm256_mul_ps(a, Set2(b.first, b.second));
  }

  [[nodiscard]] FORCE_INLINE auto Add(__m128 a, __m128 b) -> __m128
  {
    return _mm_add_ps(a, b);
  }

  [[nodiscard]] FORCE_INLINE auto Sub(__m128 a, __m128 b) -> __m128
  {
    return _mm_sub_ps(a, b);
  }

  [[nodiscard]] FORCE_INLINE auto Mul(__m128 a, __m128 b) -> __m128
  {
    return _mm_mul_ps(a, b);
  }

  [[nodiscard]] FORCE_INLINE auto Mul(__m128 a, float b) -> __m128
  {
    return _mm_mul_ps(a, _mm_set1_ps(b));
  }

  [[nodiscard]] FORCE_INLINE auto MAdd(__m256 a, __m256 b, __m256 c) -> __m256
  {
    return _mm256_fmadd_ps(a, b, c);
  }

  [[nodiscard]] FORCE_INLINE auto NMAdd(__m256 a, __m256 b, __m256 c) -> __m256
  {
    return _mm256_fnmadd_ps(a, b, c);
  }

  [[nodiscard]] FORCE_INLINE auto MAdd(__m128 a, __m128 b, __m128 c) -> __m128
  {
    return _mm_fmadd_ps(a, b, c);
  }

  [[nodiscard]] FORCE_INLINE auto NMAdd(__m128 a, __m128 b, __m128 c) -> __m128
  {
    return _mm_fnmadd_ps(a, b, c);
  }

  [[nodiscard]] FORCE_INLINE auto MAdd(__m256 a, float b, __m256 c) -> __m256
  {
    return _mm256_fmadd_ps(a, _mm256_set1_ps(b), c);
  }

  [[nodiscard]] FORCE_INLINE auto NMAdd(__m256 a, float b, __m256 c) -> __m256
  {
    return _mm256_fnmadd_ps(a, _mm256_set1_ps(b), c);
  }

  [[nodiscard]] FORCE_INLINE auto MAdd(__m128 a, float b, __m128 c) -> __m128
  {
    return _mm_fmadd_ps(a, _mm_set1_ps(b), c);
  }

  [[nodiscard]] FORCE_INLINE auto NMAdd(__m128 a, float b, __m128 c) -> __m128
  {
    return _mm_fnmadd_ps(a, _mm_set1_ps(b), c);
  }

  [[nodiscard]] FORCE_INLINE auto MAdd(__m256 a, std::pair<float, float> b, __m256 c) -> __m256
  {
    return _mm256_fmadd_ps(a, Set2(b.first, b.second), c);
  }

  [[nodiscard]] FORCE_INLINE auto NMAdd(__m256 a, std::pair<float, float> b, __m256 c) -> __m256
  {
    return _mm256_fnmadd_ps(a, Set2(b.first, b.second), c);
  }

  FORCE_INLINE void Transpose4x4(__m256 &a, __m256 &b, __m256 &c, __m256 &d)
  {
    auto t0 = _mm256_shuffle_ps(a, b, 0xEE);
    a = _mm256_shuffle_ps(a, b, 0x44);
    auto t1 = _mm256_shuffle_ps(c, d, 0x44);
    d = _mm256_shuffle_ps(c, d, 0xEE);

    b = _mm256_shuffle_ps(a, t1, 0xDD);
    a = _mm256_shuffle_ps(a, t1, 0x88);
    c = _mm256_shuffle_ps(t0, d, 0x88);
    d = _mm256_shuffle_ps(t0, d, 0xDD);
  }

  // Functions needed specifically to support the FMA optimizations ///////////////////////////////////////////////////
  [[nodiscard]] FORCE_INLINE auto MSub(__m256 a, float b, __m256 c) -> __m256
  {
    return _mm256_fmsub_ps(a, _mm256_set1_ps(b), c);
  }

  [[nodiscard]] FORCE_INLINE auto MSub(__m128 a, float b, __m128 c) -> __m128
  {
    return _mm_fmsub_ps(a, _mm_set1_ps(b), c);
  }
  // Functions specific to complex FFTs ///////////////////////////////////////////////////////////////////////////////

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
    auto r = Load<Align>(ptr);
    auto i = Load<Align>(ptr + VecSize);
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

  // Functions specific to real FFTs //////////////////////////////////////////////////////////////////////////////////

  template <Alignment A>
  FORCE_INLINE void StoreLowerHalves(float *p, __m256 a, __m256 b)
  {
    Store<A>(p, _mm256_castps256_ps128(a));
    Store<A>(p + 4, _mm256_castps256_ps128(b));
  }

  template <Alignment A>
  FORCE_INLINE void StoreUpperHalves(float *p, __m256 a, __m256 b)
  {
    Store<A>(p, _mm256_extractf128_ps(a, 1));
    Store<A>(p + 4, _mm256_extractf128_ps(b, 1));
  }


  template <Alignment A>
  [[nodiscard]] FORCE_INLINE auto LoadHalves(const float *pA, const float *pB) -> VecType
  {
    auto t = _mm256_castps128_ps256(LoadHalf<A>(pB));
    return _mm256_insertf128_ps(_mm256_castps128_ps256(LoadHalf<A>(pA)), _mm256_castps256_ps128(t), 1);
  }

  template <Alignment A, std::floating_point F, std::same_as<__m256>...V>
  FORCE_INLINE void LoadStridedHalves(const F *baseA, const F *baseB, size_t stride, V &... vs)
  {
    size_t offset = 0;
    ((vs = LoadHalves<A>(baseA + offset, baseB + offset), offset += stride), ...);
  }

  template <Alignment A>
  FORCE_INLINE void StoreHalves(float *pA, float *pB, __m256 v)
  {
    auto t = _mm256_castps128_ps256(_mm256_extractf128_ps(v, 1));
    Store<A>(pA, _mm256_castps256_ps128(v));
    Store<A>(pB, _mm256_castps256_ps128(t));
  }

  template <Alignment A, std::floating_point F, std::same_as<__m256> ...V>
  FORCE_INLINE void StoreStridedHalves(F *baseA, F *baseB, size_t stride, V...v)
  {
    size_t offset = 0;
    ((StoreHalves<A>(baseA + offset, baseB + offset, v), offset += stride), ...);
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

  // Takes the first two elements of acegACEG0 and bdfh0 and puts them in t, then the second two elements from each into bdfh0.
  FORCE_INLINE void SwapRealComponentsForLastDoubleFlip(__m256 aceg0, __m256 &bdfh0, __m256 &t)
  {
    t = _mm256_shuffle_ps(aceg0, bdfh0, 0b01'00'01'00);
    auto t2 = _mm256_shuffle_ps(aceg0, bdfh0, 0b11'10'11'10);
    t = _mm256_blend_ps(aceg0, t, 0b00001111);
    bdfh0 = _mm256_blend_ps(bdfh0, t2, 0b00001111);
  }

  [[nodiscard]] FORCE_INLINE auto ReplaceFirstOf(__m256 value, auto newFirst) -> __m256
  {
    // For AVX the best version I've found is:
    //  (works great in all 3 tested compilers)
    if constexpr (std::floating_point<decltype(newFirst)>)
      { return _mm256_blend_ps(value, _mm256_castps128_ps256(_mm_set_ss(newFirst)), 0b00000001); }
    else
      { return _mm256_blend_ps(value, newFirst, 0b00000001); }
  }

  [[nodiscard]] FORCE_INLINE auto InsertTopHalf(__m256 lo, __m256 hi) -> __m256
  {
    return _mm256_blend_ps(lo, hi, 0b11110000);
  }

  [[nodiscard]] FORCE_INLINE auto InsertTopHalf(__m128 lo, __m128 hi) -> __m128
  {
    return _mm_shuffle_ps(lo, hi, 0b11'10'01'00);
  }

  FORCE_INLINE void Deinterleave4(__m256 a, __m256 &b, __m256 &aOut)
  {
    aOut = _mm256_shuffle_ps(a, b, 0b10'00'10'00);
    b = _mm256_shuffle_ps(a, b, 0b11'01'11'01);
  }


  FORCE_INLINE void Interleave4(__m256 a, __m256 &b, __m256 &aOut)
  {
    aOut = _mm256_unpacklo_ps(a, b);
    b = _mm256_unpackhi_ps(a, b);
  }

  #include "fft_20xx_algo_cplx.h"
  #include "fft_20xx_algo_real_common.h"
  #include "fft_20xx_algo_real_fwd.h"
  #include "fft_20xx_algo_real_inv.h"

  static_assert(TangleVecSize == 4);
  static_assert(std::same_as<TangleVecType, __m128>);
}


using namespace FFT20XX;
using namespace fft_20xx::c;
extern "C"
{
  void fft_cplx_f32_sse2(const fft_table_cplx_f32 *table, const float *input, float *output, size_t fftLength);

  void fft_cplx_f32_fma(const fft_table_cplx_f32 *table, const float *input, float *output, size_t fftLength)
  {
    if (fftLength < 64)
      { return fft_cplx_f32_sse2(table, input, output, fftLength); }
    COMMON_ASSERTS();
    if (FMA32::IsSIMDAligned(input) && FMA32::IsSIMDAligned(output))
      { FMA32::RunComplex<Dir::Forward, Alignment::Aligned>(GetTableData(table), input, output, fftLength); }
    else
      { FMA32::RunComplex<Dir::Forward, Alignment::Unaligned>(GetTableData(table), input, output, fftLength); }
  }

  void ifft_cplx_f32_sse2(const fft_table_cplx_f32 *table, const float *input, float *output, size_t fftLength);

  void ifft_cplx_f32_fma(const fft_table_cplx_f32 *table, const float *input, float *output, size_t fftLength)
  {
    if (fftLength < 64)
      { return ifft_cplx_f32_sse2(table, input, output, fftLength); }
    COMMON_ASSERTS();
    if (FMA32::IsSIMDAligned(input) && FMA32::IsSIMDAligned(output))
      { FMA32::RunComplex<Dir::Inverse, Alignment::Aligned>(GetTableData(table), input, output, fftLength); }
    else
      { FMA32::RunComplex<Dir::Inverse, Alignment::Unaligned>(GetTableData(table), input, output, fftLength); }
  }

  void fft_real_f32_sse2(const fft_table_real_f32 *table, const float *input, float *output, size_t fftLength);

  void fft_real_f32_fma(const fft_table_real_f32 *table, const float *input, float *output, size_t fftLength)
  {
    // Do this test before COMMON_ASSERTS since the called function will do those
    if (fftLength < FMA32::MinimumRealFFTLength)
    {
      // AVX's 8-long vectors are too large to make these nice so fall back on SSE2.
      return fft_real_f32_sse2(table, input, output, fftLength);
    }

    COMMON_ASSERTS();
    FMA32::RealForwardFFT<Alignment::Unaligned>(GetTableData(table), input, output, fftLength);
  }


  void ifft_real_f32_sse2(const fft_table_real_f32 *table, const float *input, float *output, size_t fftLength);

  void ifft_real_f32_fma(const fft_table_real_f32 *table, const float *input, float *output, size_t fftLength)
  {
    if (fftLength < FMA32::MinimumRealFFTLength)
    {
      // AVX's 8-long vectors are too large to make these nice so fall back on SSE2.
      return ifft_real_f32_sse2(table, input, output, fftLength);
    }

    COMMON_ASSERTS();
    FMA32::RealInverseFFT<Alignment::Unaligned>(GetTableData(table), input, output, fftLength);
  }
}
