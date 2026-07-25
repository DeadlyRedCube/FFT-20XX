#ifdef _MSC_VER
  #pragma warning (disable: 4711) // function <x> selected for automatic inline expansion
#endif

#include "fft_20xx_common.h"
#include "fft_20xx_table.h"
#include "fft_20xx_tiny_real.h"

static_assert(CPU_X86 || CPU_X64);

static_assert(SIMD_AVX);

#if COMPILER_GCC
  #pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

namespace FFT20XX::AVX
{
  template <std::floating_point F>
  bool IsSIMDAligned(const F *ptr)
    { return (uintptr_t(ptr) & (sizeof(__m256) - 1)) == 0; }

  constexpr auto MinimumRealFFTLength = 128;
  constexpr bool HasNativeFMA = false;

  constexpr auto VecSize = 8u;
  using VecType = __m256;
  using HalfType = __m128;
  using ElementType = float;

  template <Alignment A>
  [[nodiscard]] FORCE_INLINE auto Load(const float *p) -> VecType
  {
    if constexpr (A == Alignment::Aligned)
      { return _mm256_load_ps(p); }
    else
      { return _mm256_loadu_ps(p); }
  }

  template <Alignment A>
  [[nodiscard]] FORCE_INLINE auto LoadHalf(const float *p) -> HalfType
  {
    if constexpr (A == Alignment::Aligned)
      { return _mm_load_ps(p); }
    else
      { return _mm_loadu_ps(p); }
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
  FORCE_INLINE void Store(float *p, __m256 v)
  {
    if constexpr (A == Alignment::Aligned)
      { _mm256_store_ps(p, v); }
    else
      { _mm256_storeu_ps(p, v); }
  }

  template <Alignment A>
  FORCE_INLINE void Store(float *p, __m128 v)
  {
    if constexpr (A == Alignment::Aligned)
      { _mm_store_ps(p, v); }
    else
      { _mm_storeu_ps(p, v); }
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

  [[nodiscard]] FORCE_INLINE auto Set2(float lo, float hi) -> __m256
  {
    return _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_set1_ps(lo)), _mm_set1_ps(hi), 1);
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

  [[nodiscard]] FORCE_INLINE auto MAdd(__m256 a, auto b, __m256 c) -> __m256
  {
    return Add(Mul(a, b), c);
  }

  [[nodiscard]] FORCE_INLINE auto NMAdd(__m256 a, auto b, __m256 c) -> __m256
  {
    return Sub(c, Mul(a, b));
  }

  [[nodiscard]] FORCE_INLINE auto MAdd(__m128 a, auto b, __m128 c) -> __m128
  {
    return Add(Mul(a, b), c);
  }

  [[nodiscard]] FORCE_INLINE auto NMAdd(__m128 a, auto b, __m128 c) -> __m128
  {
    return Sub(c, Mul(a, b));
  }

  [[nodiscard]] FORCE_INLINE auto Mx2Sub(__m256 a, __m256 b) -> __m256
  {
    return _mm256_sub_ps(_mm256_add_ps(a, a), b);
  }

  [[nodiscard]] FORCE_INLINE auto NMx2Add(__m256 a, __m256 b) -> __m256
  {
    return _mm256_sub_ps(b, _mm256_add_ps(a, a));
  }


  [[nodiscard]] FORCE_INLINE auto Mx2Sub(__m128 a, __m128 b) -> __m128
  {
    return _mm_sub_ps(_mm_add_ps(a, a), b);
  }

  [[nodiscard]] FORCE_INLINE auto NMx2Add(__m128 a, __m128 b) -> __m128
  {
    return _mm_sub_ps(b, _mm_add_ps(a, a));
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


  FORCE_INLINE void Transpose4(__m256 &a, __m256 &b, __m256 &c, __m256 &d)
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



  // Takes the first two elements of acegACEG0 and bdfh0 and puts them in t, then the second two elements from each into bdfh0.
  FORCE_INLINE void SwapRealComponentsForLastDoubleFlip(__m256 aceg0, __m256 &bdfh0, __m256 &t, __m256 &t2)
  {
    t = _mm256_shuffle_ps(aceg0, bdfh0, 0b01'00'01'00);
    t2 = _mm256_shuffle_ps(aceg0, bdfh0, 0b11'10'11'10);
    t = _mm256_blend_ps(aceg0, t, 0b00001111);
    bdfh0 = _mm256_blend_ps(bdfh0, t2, 0b00001111);
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



  #include "fft_20xx_algo_real_fwd.h"
  #include "fft_20xx_algo_real_inv.h"

  static_assert(TangleVecSize == 4);
  static_assert(std::same_as<TangleVecType, __m128>);
}


using namespace FFT20XX;
using namespace fft_20xx::c;
extern "C"
{
  void fft_real_f32_sse2(const fft_table_f32 *table, const float *input, float *output, size_t fftLength);

  void fft_real_f32_avx(const fft_table_f32 *table, const float *input, float *output, size_t fftLength)
  {
    // Do this test before COMMON_ASSERTS since the called function will do those
    if (fftLength < AVX::MinimumRealFFTLength)
    {
      // AVX's 8-long vectors are too large to make these nice so fall back on SSE2.
      return fft_real_f32_sse2(table, input, output, fftLength);
    }

    auto t = reinterpret_cast<const FFTTableImpl<float> *>(table);
    COMMON_ASSERTS();
    AVX::RealForwardFFT<Alignment::Unaligned>(t->RealData(), input, output, fftLength);
  }


  void ifft_real_f32_sse2(const fft_table_f32 *table, const float *input, float *output, size_t fftLength);

  void ifft_real_f32_avx(const fft_table_f32 *table, const float *input, float *output, size_t fftLength)
  {
    if (fftLength < AVX::MinimumRealFFTLength)
    {
      // AVX's 8-long vectors are too large to make these nice so fall back on SSE2.
      return ifft_real_f32_sse2(table, input, output, fftLength);
    }

    auto t = reinterpret_cast<const FFTTableImpl<float> *>(table);
    COMMON_ASSERTS();
    AVX::RealInverseFFT<Alignment::Unaligned>(t->RealData(), input, output, fftLength);
  }
}
