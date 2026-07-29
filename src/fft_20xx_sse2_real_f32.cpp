#ifdef _MSC_VER
  #pragma warning (disable: 4711) // function <x> selected for automatic inline expansion
#endif

#include "fft_20xx_common.h"
#include "fft_20xx_table.h"
#include "fft_20xx_tiny_real.h"

static_assert(CPU_X86 || CPU_X64);

static_assert(SIMD_SSE2);

#if COMPILER_GCC
  #pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

namespace FFT20XX::SSE2
{
  template <std::floating_point F>
  bool IsSIMDAligned(const F *ptr)
    { return (uintptr_t(ptr) & (sizeof(__m128) - 1)) == 0; }

  constexpr auto MinimumRealFFTLength = 32;
  constexpr bool UseFMAOptimization = false;
  constexpr auto VecSize = 4u;
  using VecType = __m128;
  using ElementType = float;

  template <Alignment A>
  [[nodiscard]] FORCE_INLINE auto Load(const float *p)
  {
    if constexpr (A == Alignment::Aligned)
      { return _mm_load_ps(p); }
    else
      { return _mm_loadu_ps(p); }
  }



  template <Alignment A>
  FORCE_INLINE void Store(float *p, __m128 v)
  {
    if constexpr (A == Alignment::Aligned)
      { _mm_store_ps(p, v); }
    else
      { _mm_storeu_ps(p, v); }
  }



  [[nodiscard]] FORCE_INLINE auto Add(__m128 a, __m128 b)
  {
    return _mm_add_ps(a, b);
  }

  [[nodiscard]] FORCE_INLINE auto Sub(__m128 a, __m128 b)
  {
    return _mm_sub_ps(a, b);
  }

  [[nodiscard]] FORCE_INLINE auto Mul(__m128 a, __m128 b)
  {
    return _mm_mul_ps(a, b);
  }

  [[nodiscard]] FORCE_INLINE auto Mul(__m128 a, float b)
  {
    return _mm_mul_ps(a, _mm_set1_ps(b));
  }

  [[nodiscard]] FORCE_INLINE auto MAdd(__m128 a, auto b, __m128 c) -> __m128
  {
    return Add(Mul(a, b), c);
  }

  [[nodiscard]] FORCE_INLINE auto NMAdd(__m128 a, auto b, __m128 c) -> __m128
  {
    return Sub(c, Mul(a, b));
  }

  [[nodiscard]] FORCE_INLINE auto Mx2Sub(__m128 a, __m128 b) -> __m128
  {
    return _mm_sub_ps(_mm_add_ps(a, a), b);
  }

  [[nodiscard]] FORCE_INLINE auto NMx2Add(__m128 a, __m128 b) -> __m128
  {
    return _mm_sub_ps(b, _mm_add_ps(a, a));
  }

  [[nodiscard]] FORCE_INLINE auto ReplaceFirstOf(auto value, auto newFirst)
  {
    // For AVX the best version I've found is:
    // _mm256_blend_ps(a, _mm256_castps128_ps256(_mm_set_ss(d)), 0b00000001);
    //  (works great in all 3 tested compilers)
    if constexpr (std::floating_point<decltype(newFirst)>)
    {
      #if COMPILER_MSVC
        // MSVC doesn't optimize the intrinsics as well so do this directly
        value.m128_f32[0] = newFirst;
        return value;
      #else
        return _mm_move_ss(value, _mm_set_ss(newFirst)); // Will compile to a single instruction
      #endif
    }
    else
      { return _mm_move_ss(value, newFirst); }
  }

  [[nodiscard]] FORCE_INLINE auto InsertTopHalf(auto lo, auto hi)
  {
    return _mm_shuffle_ps(lo, hi, 0b11'10'01'00);
  }

  FORCE_INLINE void Transpose4(auto &a, auto &b, auto &c, auto &d)
  {
    auto t0 = _mm_shuffle_ps(a, b, 0xEE);
    a = _mm_shuffle_ps(a, b, 0x44);
    auto t1 = _mm_shuffle_ps(c, d, 0x44);
    d = _mm_shuffle_ps(c, d, 0xEE);

    b = _mm_shuffle_ps(a, t1, 0xDD);
    a = _mm_shuffle_ps(a, t1, 0x88);
    c = _mm_shuffle_ps(t0, d, 0x88);
    d = _mm_shuffle_ps(t0, d, 0xDD);
  }



  // Takes the first two elements of aceg0 and bdfh0 and puts them in t, then the second two elements from each into bdfh0.
  FORCE_INLINE void SwapRealComponentsForLastDoubleFlip(auto aceg0, auto &bdfh0, auto &t)
  {
    t = _mm_shuffle_ps(aceg0, bdfh0, 0b01'00'01'00);
    bdfh0 = _mm_shuffle_ps(aceg0, bdfh0, 0b11'10'11'10);
  }



  FORCE_INLINE void Deinterleave4(auto a, auto &b, auto &aOut)
  {
    aOut = _mm_shuffle_ps(a, b, 0b10'00'10'00);
    b = _mm_shuffle_ps(a, b, 0b11'01'11'01);
  }



  FORCE_INLINE void Interleave4(auto a, auto &b, auto &aOut)
  {
    aOut = _mm_unpacklo_ps(a, b);
    b = _mm_unpackhi_ps(a, b);
  }



  #include "fft_20xx_algo_real_fwd.h"
  #include "fft_20xx_algo_real_inv.h"
}


using namespace FFT20XX;
using namespace fft_20xx::c;
extern "C"
{
  void fft_real_f32_sse2(const fft_table_f32 *table, const float *input, float *output, size_t fftLength)
  {
    auto t = reinterpret_cast<const FFTTableImpl<float> *>(table);
    COMMON_ASSERTS();
    SSE2::RealForwardFFT<Alignment::Unaligned>(t->RealData(), input, output, fftLength);
  }



  void ifft_real_f32_sse2(const fft_table_f32 *table, const float *input, float *output, size_t fftLength)
  {
    auto t = reinterpret_cast<const FFTTableImpl<float> *>(table);
    COMMON_ASSERTS();
    SSE2::RealInverseFFT<Alignment::Unaligned>(t->RealData(), input, output, fftLength);
  }
}
