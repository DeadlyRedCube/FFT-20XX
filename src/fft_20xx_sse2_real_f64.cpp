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
  constexpr bool UseFMAOptimization = false;

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



  [[nodiscard]] FORCE_INLINE auto Add(__m128d a, __m128d b)
  {
    return _mm_add_pd(a, b);
  }

  [[nodiscard]] FORCE_INLINE auto Sub(__m128d a, __m128d b)
  {
    return _mm_sub_pd(a, b);
  }

  [[nodiscard]] FORCE_INLINE auto Mul(__m128d a, __m128d b)
  {
    return _mm_mul_pd(a, b);
  }

  [[nodiscard]] FORCE_INLINE auto Mul(__m128d a, double b)
  {
    return _mm_mul_pd(a, _mm_set1_pd(b));
  }

  [[nodiscard]] FORCE_INLINE auto MAdd(__m128d a, auto b, __m128d c) -> __m128d
  {
    return Add(Mul(a, b), c);
  }

  [[nodiscard]] FORCE_INLINE auto NMAdd(__m128d a, auto b, __m128d c) -> __m128d
  {
    return Sub(c, Mul(a, b));
  }

  [[nodiscard]] FORCE_INLINE auto Mx2Sub(__m128d a, __m128d b) -> __m128d
  {
    return _mm_sub_pd(_mm_add_pd(a, a), b);
  }

  [[nodiscard]] FORCE_INLINE auto NMx2Add(__m128d a, __m128d b) -> __m128d
  {
    return _mm_sub_pd(b, _mm_add_pd(a, a));
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

  FORCE_INLINE void Transpose2(auto a, auto &b, auto &aOut)
  {
    aOut = _mm_shuffle_pd(a, b, 0b00);
    b = _mm_shuffle_pd(a, b, 0b11);
  }

  #include "fft_20xx_algo_real_common.h"
  #include "fft_20xx_algo_real_fwd.h"
  #include "fft_20xx_algo_real_inv.h"
}


using namespace FFT20XX;
using namespace fft_20xx::c;
extern "C"
{
  void fft_real_f64_sse2(const fft_table_real_f64 *table, const double *input, double *output, size_t fftLength)
  {
    COMMON_ASSERTS();
    SSE2::RealForwardFFT<Alignment::Unaligned>(GetTableData(table), input, output, fftLength);
  }



  void ifft_real_f64_sse2(const fft_table_real_f64 *table, const double *input, double *output, size_t fftLength)
  {
    COMMON_ASSERTS();
    SSE2::RealInverseFFT<Alignment::Unaligned>(GetTableData(table), input, output, fftLength);
  }
}
