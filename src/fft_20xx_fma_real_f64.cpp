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

namespace FFT20XX::FMA64
{
  template <std::floating_point F>
  bool IsSIMDAligned(const F *ptr)
    { return (uintptr_t(ptr) & (sizeof(__m256d) - 1)) == 0; }

  constexpr auto MinimumRealFFTLength = 32;
  constexpr bool UseFMAOptimization = true;

  constexpr auto VecSize = 4u;
  using VecType = __m256d;
  using ElementType = double;

  template <Alignment A>
  [[nodiscard]] FORCE_INLINE auto Load(const double *p)
  {
    if constexpr (A == Alignment::Aligned)
      { return _mm256_load_pd(p); }
    else
      { return _mm256_loadu_pd(p); }
  }



  template <Alignment A>
  FORCE_INLINE void Store(double *p, __m256d v)
  {
    if constexpr (A == Alignment::Aligned)
      { _mm256_store_pd(p, v); }
    else
      { _mm256_storeu_pd(p, v); }
  }



  [[nodiscard]] FORCE_INLINE auto Add(__m256d a, __m256d b)
  {
    return _mm256_add_pd(a, b);
  }

  [[nodiscard]] FORCE_INLINE auto Sub(__m256d a, __m256d b)
  {
    return _mm256_sub_pd(a, b);
  }

  [[nodiscard]] FORCE_INLINE auto Mul(__m256d a, __m256d b)
  {
    return _mm256_mul_pd(a, b);
  }

  [[nodiscard]] FORCE_INLINE auto Mul(__m256d a, double b)
  {
    return _mm256_mul_pd(a, _mm256_set1_pd(b));
  }

  [[nodiscard]] FORCE_INLINE auto MAdd(__m256d a, __m256d b, __m256d c) -> __m256d
  {
    return _mm256_fmadd_pd(a, b, c);
  }

  [[nodiscard]] FORCE_INLINE auto NMAdd(__m256d a, __m256d b, __m256d c) -> __m256d
  {
    return _mm256_fnmadd_pd(a, b, c);
  }

  [[nodiscard]] FORCE_INLINE auto MAdd(__m256d a, double b, __m256d c) -> __m256d
  {
    return _mm256_fmadd_pd(a, _mm256_set1_pd(b), c);
  }

  [[nodiscard]] FORCE_INLINE auto NMAdd(__m256d a, double b, __m256d c) -> __m256d
  {
    return _mm256_fnmadd_pd(a, _mm256_set1_pd(b), c);
  }

  [[nodiscard]] FORCE_INLINE auto Mx2Sub(__m256d a, __m256d b) -> __m256d
  {
    return _mm256_fmsub_pd(_mm256_set1_pd(2.0), a, b);
  }

  [[nodiscard]] FORCE_INLINE auto NMx2Add(__m256d a, __m256d b) -> __m256d
  {
    return _mm256_fnmadd_pd(_mm256_set1_pd(2.0), a, b);
  }


  [[nodiscard]] FORCE_INLINE auto ReplaceFirstOf(auto value, auto newFirst)
  {
    if constexpr (std::floating_point<decltype(newFirst)>)
    {
      return _mm256_blend_pd(value, _mm256_castpd128_pd256(_mm_set_sd(newFirst)), 0b0001);
    }
    else
      { return _mm256_blend_pd(value, newFirst, 0b0001); }
  }

  [[nodiscard]] FORCE_INLINE auto InsertTopHalf(auto lo, auto hi)
  {
    return _mm256_blend_pd(lo, hi, 0b1100);
  }

  FORCE_INLINE void Transpose4(auto &a, auto &b, auto &c, auto &d)
  {
    auto t0 = _mm256_unpackhi_pd(a, b);
    a = _mm256_unpacklo_pd(a, b);
    auto t1 = _mm256_unpacklo_pd(c, d);
    d = _mm256_unpackhi_pd(c, d);

    c = _mm256_permute2f128_pd(a, t1, 0x31);
    a = _mm256_permute2f128_pd(a, t1, 0x20);
    b = _mm256_permute2f128_pd(t0, d, 0x20);
    d = _mm256_permute2f128_pd(t0, d, 0x31);
  }



  // Takes the first two elements of aceg0 and bdfh0 and puts them in t, then the second two elements from each into bdfh0.
  FORCE_INLINE void SwapRealComponentsForLastDoubleFlip(auto aceg0, auto &bdfh0, auto &t)
  {
    t = _mm256_insertf128_pd(aceg0, _mm256_castpd256_pd128(bdfh0), 1);
    bdfh0 = _mm256_permute2f128_pd(aceg0, bdfh0, 0x31);
  }



  // This function deinterleave "a, b" to "aOut, b". It also modifies "a" but because the caller is discarding the
  //  result, it's fine.
  FORCE_INLINE void Deinterleave4(auto &a, auto &b, auto &aOut)
  {
    aOut = _mm256_permute2f128_pd(a, b, 0x20);  // First halves of each
    a = _mm256_permute2f128_pd(a, b, 0x31);     // Second halves of each

    b = _mm256_shuffle_pd(aOut, a, 0b1'1'1'1);
    aOut = _mm256_shuffle_pd(aOut, a, 0b0'0'0'0);
  }



  // This function interleaves "a, b" to "aOut, b". It also modifies "a", like the above function.
  FORCE_INLINE void Interleave4(auto &a, auto &b, auto &aOut)
  {
    aOut = _mm256_unpacklo_pd(a, b);
    a = _mm256_unpackhi_pd(a, b);

    b = _mm256_permute2f128_pd(aOut, a, 0x31);
    aOut = _mm256_permute2f128_pd(aOut, a, 0x20);
  }



  #include "fft_20xx_algo_real_common.h"
  #include "fft_20xx_algo_real_fwd.h"
  #include "fft_20xx_algo_real_inv.h"
}


using namespace FFT20XX;
using namespace fft_20xx::c;
extern "C"
{
  void fft_real_f64_fma(const fft_table_f64 *table, const double *input, double *output, size_t fftLength)
  {
    auto t = reinterpret_cast<const FFTTableImpl<double> *>(table);
    COMMON_ASSERTS();
    FMA64::RealForwardFFT<Alignment::Unaligned>(t->RealData(), input, output, fftLength);
  }


  void ifft_real_f64_fma(const fft_table_f64 *table, const double *input, double *output, size_t fftLength)
  {
    auto t = reinterpret_cast<const FFTTableImpl<double> *>(table);
    COMMON_ASSERTS();
    FMA64::RealInverseFFT<Alignment::Unaligned>(t->RealData(), input, output, fftLength);
  }
}
