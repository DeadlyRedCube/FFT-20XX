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

namespace FFT20XX::AVXCplx64
{
  constexpr uint32_t VecSize = 4u;
  using VecType = __m256d;
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

  [[nodiscard]] FORCE_INLINE auto MAdd(__m256d a, auto b, __m256d c) -> __m256d
  {
    return Add(Mul(a, b), c);
  }

  [[nodiscard]] FORCE_INLINE auto NMAdd(__m256d a, auto b, __m256d c) -> __m256d
  {
    return Sub(c, Mul(a, b));
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
  void fft_cplx_f64_avx(const fft_table_cplx_f64 *table, const double *input, double *output, size_t fftLength)
  {
    COMMON_ASSERTS();
    if (AVXCplx64::IsSIMDAligned(input) && AVXCplx64::IsSIMDAligned(output))
      { AVXCplx64::RunComplex<Dir::Forward, Alignment::Aligned>(GetTableData(table), input, output, fftLength); }
    else
      { AVXCplx64::RunComplex<Dir::Forward, Alignment::Unaligned>(GetTableData(table), input, output, fftLength); }
  }


  void ifft_cplx_f64_avx(const fft_table_cplx_f64 *table, const double *input, double *output, size_t fftLength)
  {
    COMMON_ASSERTS();
    if (AVXCplx64::IsSIMDAligned(input) && AVXCplx64::IsSIMDAligned(output))
      { AVXCplx64::RunComplex<Dir::Inverse, Alignment::Aligned>(GetTableData(table), input, output, fftLength); }
    else
      { AVXCplx64::RunComplex<Dir::Inverse, Alignment::Unaligned>(GetTableData(table), input, output, fftLength); }
  }
}
