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

namespace FFT20XX::SSE2
{
  constexpr uint32_t VecSize = 4u;
  using VecType = __m128;
  using ElementType = float;

  template <typename T>
  struct Angles
  {
    T w0;
    T w1;
    T w2;
  };


  bool IsSIMDAligned(const float *ptr)
    { return (uintptr_t(ptr) & (sizeof(__m128) - 1)) == 0; }


  FORCE_INLINE auto Set1(float f) -> VecType
  {
    return _mm_set1_ps(f);
  }



  template <Alignment Align>
  FORCE_INLINE auto Load(const float *ptr)
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
      { return _mm_store_ps(ptr, v); }
    else
      { return _mm_storeu_ps(ptr, v); }
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

  template <Alignment Align>
  FORCE_INLINE auto LoadPackedComplex(const float *ptr) -> ComplexVec<VecType>
  {
    auto t0 = Load<Align>(ptr);
    auto t1 = Load<Align>(ptr + VecSize);
    return { _mm_shuffle_ps(t0, t1, _MM_SHUFFLE(2, 0, 2, 0)), _mm_shuffle_ps(t0, t1, _MM_SHUFFLE(3, 1, 3, 1)) };
  }



  template <Alignment Align>
  FORCE_INLINE void StorePackedComplex(float *ptr, ComplexVec<VecType> v)
  {
    auto aa = _mm_unpacklo_ps(v.r, v.i);
    auto bb = _mm_unpackhi_ps(v.r, v.i);
    Store<Align>(ptr, aa);
    Store<Align>(ptr + 4, bb);
  }



  FORCE_INLINE void Transpose4x4(
    ComplexVec<__m128> &a,
    ComplexVec<__m128> &b,
    ComplexVec<__m128> &c,
    ComplexVec<__m128> &d)
  {
    _MM_TRANSPOSE4_PS(a.r, b.r, c.r, d.r);
    _MM_TRANSPOSE4_PS(a.i, b.i, c.i, d.i);
  }

  #include "fft_20xx_algo_cplx.h"
}


using namespace FFT20XX;
using namespace fft_20xx::c;
extern "C"
{
  void fft_cplx_f32_sse2(const fft_table_cplx_f32 *table, const float *input, float *output, size_t fftLength)
  {
    COMMON_ASSERTS();
    if (SSE2::IsSIMDAligned(input) && SSE2::IsSIMDAligned(output))
      { SSE2::RunComplex<Dir::Forward, Alignment::Aligned>(GetTableData(table), input, output, fftLength); }
    else
      { SSE2::RunComplex<Dir::Forward, Alignment::Unaligned>(GetTableData(table), input, output, fftLength); }
  }


  void ifft_cplx_f32_sse2(const fft_table_cplx_f32 *table, const float *input, float *output, size_t fftLength)
  {
    COMMON_ASSERTS();
    if (SSE2::IsSIMDAligned(input) && SSE2::IsSIMDAligned(output))
      { SSE2::RunComplex<Dir::Inverse, Alignment::Aligned>(GetTableData(table), input, output, fftLength); }
    else
      { SSE2::RunComplex<Dir::Inverse, Alignment::Unaligned>(GetTableData(table), input, output, fftLength); }
  }
}
