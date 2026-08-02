#if COMPILER_GCC
  #pragma GCC diagnostic ignored "-Wignored-attributes"
#endif
constexpr uint32_t VecSize = 4u;
using VecType = __m256d;
using ElementType = double;
constexpr auto MinimumRealFFTLength = 32;
constexpr bool UseFMAOptimization = SIMD_FMA;

template <typename T>
struct Angles
{
  T w0;
  T w1;
  #if !SIMD_FMA
    T w2;
  #endif
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

[[nodiscard]] FORCE_INLINE auto Add(__m256d a, __m256d b) -> __m256d
{
  return _mm256_add_pd(a, b);
}

[[nodiscard]] FORCE_INLINE auto Sub(__m256d a, __m256d b) -> __m256d
{
  return _mm256_sub_pd(a, b);
}

[[nodiscard]] FORCE_INLINE auto Mul(__m256d a, __m256d b) -> __m256d
{
  return _mm256_mul_pd(a, b);
}

[[nodiscard]] FORCE_INLINE auto Mul(__m256d a, double b) -> __m256d
{
  return _mm256_mul_pd(a, _mm256_set1_pd(b));
}

#if SIMD_FMA
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
#else
  [[nodiscard]] FORCE_INLINE auto MAdd(__m256d a, auto b, __m256d c) -> __m256d
  {
    return Add(Mul(a, b), c);
  }

  [[nodiscard]] FORCE_INLINE auto NMAdd(__m256d a, auto b, __m256d c) -> __m256d
  {
    return Sub(c, Mul(a, b));
  }
#endif  

FORCE_INLINE void Transpose4x4(auto &a, auto &b, auto &c, auto &d)
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

// Functions needed specifically to support the FMA optimizations ///////////////////////////////////////////////////

#if SIMD_FMA
  [[nodiscard]] FORCE_INLINE auto MSub(__m256d a, __m256d b, __m256d c) -> __m256d
  {
    return _mm256_fmsub_pd(a, b, c);
  }

  [[nodiscard]] FORCE_INLINE auto MSub(__m256d a, double b, __m256d c) -> __m256d
  {
    return _mm256_fmsub_pd(a, _mm256_set1_pd(b), c);
  }
#endif

// Functions specific to complex FFTs ///////////////////////////////////////////////////////////////////////////////

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

// Functions specific to real FFTs //////////////////////////////////////////////////////////////////////////////////

// Takes the first two elements of aceg0 and bdfh0 and puts them in t, then the second two elements from each into bdfh0.
FORCE_INLINE void SwapRealComponentsForLastDoubleFlip(auto aceg0, auto &bdfh0, auto &t)
{
  t = _mm256_insertf128_pd(aceg0, _mm256_castpd256_pd128(bdfh0), 1);
  bdfh0 = _mm256_permute2f128_pd(aceg0, bdfh0, 0x31);
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

#include "fft_20xx_algo_cplx.h"
#include "fft_20xx_algo_real_common.h"
#include "fft_20xx_algo_real_fwd.h"
#include "fft_20xx_algo_real_inv.h"
