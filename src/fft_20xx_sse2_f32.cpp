#ifdef _MSC_VER
  #pragma warning (disable: 4711) // function <x> selected for automatic inline expansion
#endif

#include "fft_20xx_common.h"
#include "fft_20xx_table.h"
#include "fft_20xx_tiny_cplx.h"
#include "fft_20xx_tiny_real.h"

static_assert(CPU_X86 || CPU_X64);

static_assert(SIMD_SSE2);

#if COMPILER_GCC
  #pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

using namespace FFT20XX;

namespace FFT20XX::SSE2F32
{
  constexpr uint32_t VecSize = 4u;
  using VecType = __m128;
  using ElementType = float;
  constexpr auto MinimumRealFFTLength = 32;
  constexpr bool UseFMAOptimization = false;

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

  FORCE_INLINE void Transpose4x4(auto &a, auto &b, auto &c, auto &d)
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

  // Functions specific to complex FFTs ///////////////////////////////////////////////////////////////////////////////

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

  // Functions specific to real FFTs //////////////////////////////////////////////////////////////////////////////////

  // Takes the first two elements of aceg0 and bdfh0 and puts them in t, then the second two elements from each into bdfh0.
  FORCE_INLINE void SwapRealComponentsForLastDoubleFlip(auto aceg0, auto &bdfh0, auto &t)
  {
    t = _mm_shuffle_ps(aceg0, bdfh0, 0b01'00'01'00);
    bdfh0 = _mm_shuffle_ps(aceg0, bdfh0, 0b11'10'11'10);
  }

  [[nodiscard]] FORCE_INLINE auto ReplaceFirstOf(auto value, auto newFirst)
  {
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

  #include "fft_20xx_algo_cplx.h"
  #include "fft_20xx_algo_real_common.h"
  #include "fft_20xx_algo_real_fwd.h"
  #include "fft_20xx_algo_real_inv.h"
}


using namespace FFT20XX;
using namespace fft_20xx::c;
extern "C"
{
  void fft_cplx_f32_sse2(const fft_table_cplx_f32 *table, const float *input, float *output, size_t fftLength)
  {
    COMMON_ASSERTS();
    if (SSE2F32::IsSIMDAligned(input) && SSE2F32::IsSIMDAligned(output))
      { SSE2F32::RunComplex<Dir::Forward, Alignment::Aligned>(GetTableData(table), input, output, fftLength); }
    else
      { SSE2F32::RunComplex<Dir::Forward, Alignment::Unaligned>(GetTableData(table), input, output, fftLength); }
  }


  void ifft_cplx_f32_sse2(const fft_table_cplx_f32 *table, const float *input, float *output, size_t fftLength)
  {
    COMMON_ASSERTS();
    if (SSE2F32::IsSIMDAligned(input) && SSE2F32::IsSIMDAligned(output))
      { SSE2F32::RunComplex<Dir::Inverse, Alignment::Aligned>(GetTableData(table), input, output, fftLength); }
    else
      { SSE2F32::RunComplex<Dir::Inverse, Alignment::Unaligned>(GetTableData(table), input, output, fftLength); }
  }

  void fft_real_f32_sse2(const fft_table_cplx_f32 *table, const float *input, float *output, size_t fftLength)
  {
    COMMON_ASSERTS();
    SSE2F32::RealForwardFFT<Alignment::Unaligned>(GetTableData(table), input, output, fftLength);
  }


  void ifft_real_f32_sse2(const fft_table_cplx_f32 *table, const float *input, float *output, size_t fftLength)
  {
    COMMON_ASSERTS();
    SSE2F32::RealInverseFFT<Alignment::Unaligned>(GetTableData(table), input, output, fftLength);
  }
}
