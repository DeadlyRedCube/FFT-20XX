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
    { return (uintptr_t(ptr) & (sizeof(__m128d) - 1)) == 0; }

  constexpr auto MinimumRealFFTLength = 32;

  constexpr auto VecSize = 2u;
  using VecType = __m128d;
  using ElementType = double;

  template <Alignment A>
  [[nodiscard]] FORCE_INLINE auto Load(const double *p)
  {
    if constexpr (A == Alignment::Aligned)
      { return _mm_load_pd(p); }
    else
      { return _mm_loadu_pd(p); }
  }



  template <Alignment A>
  FORCE_INLINE void Store(double *p, __m128d v)
  {
    if constexpr (A == Alignment::Aligned)
      { _mm_store_pd(p, v); }
    else
      { _mm_storeu_pd(p, v); }
  }



  [[nodiscard]] FORCE_INLINE auto Set1(double v)
  {
    return _mm_set1_pd(v);
  }


  [[nodiscard]] FORCE_INLINE auto Add(auto a, auto b)
  {
    return _mm_add_pd(a, b);
  }

  [[nodiscard]] FORCE_INLINE auto Sub(auto a, auto b)
  {
    return _mm_sub_pd(a, b);
  }

  [[nodiscard]] FORCE_INLINE auto Mul(auto a, auto b)
  {
    return _mm_mul_pd(a, b);
  }

  [[nodiscard]] FORCE_INLINE auto ReplaceFirstOf(auto value, auto newFirst)
  {
    // NOTE: For AVX the best version I've found is
    // _mm256_blend_pd(a, _mm256_castpd128_pd256(_mm_set_sd(d)), 0b0001);
    if constexpr (std::floating_point<decltype(newFirst)>)
    {
      #if COMPILER_MSVC
        // Like with 32-bit SSE, the intrinsics don't optimize well so do it directly
        value.m128d_f64[0] = newFirst;
        return value;
      #else
        return _mm_move_sd(value, _mm_set_sd(newFirst)); // Will compile to a single instruction
      #endif
    }
    else
      { return _mm_move_sd(value, newFirst); }
  }

  [[nodiscard]] FORCE_INLINE auto InsertTopHalf(auto lo, auto hi)
  {
    return _mm_shuffle_pd(lo, hi, 0b11'10'01'00);
  }

  // Takes the first two elements of aceg0 and bdfh0 and puts them in t, then the second two elements from each into bdfh0.
  FORCE_INLINE void SwapRealComponentsForLastDoubleFlip(auto aceg0, auto &bdfh0, auto &t)
  {
    t = _mm_shuffle_ps(aceg0, bdfh0, 0b01'00'01'00);
    bdfh0 = _mm_shuffle_ps(aceg0, bdfh0, 0b11'10'11'10);
  }


  FORCE_INLINE void Transpose2(auto a, auto &b, auto &aOut)
  {
    aOut = _mm_shuffle_pd(a, b, 0b00);
    b = _mm_shuffle_pd(a, b, 0b11);
  }

  #include "fft_20xx_algo_real_fwd.h"
  #include "fft_20xx_algo_real_inv.h"
}


using namespace FFT20XX;
using namespace fft_20xx::c;
extern "C"
{
  void fft_real_f64_sse2(const fft_table_f64 *table, const double *input, double *output, size_t fftLength)
  {
    auto t = reinterpret_cast<const FFTTableImpl<double> *>(table);
    COMMON_ASSERTS();
    SSE2::RealForwardFFT<Alignment::Unaligned>(t->RealData(), input, output, fftLength);
  }



  void ifft_real_f64_sse2(const fft_table_f64 *table, const double *input, double *output, size_t fftLength)
  {
    auto t = reinterpret_cast<const FFTTableImpl<double> *>(table);
    COMMON_ASSERTS();
    SSE2::RealInverseFFT<Alignment::Unaligned>(t->RealData(), input, output, fftLength);
  }
}
