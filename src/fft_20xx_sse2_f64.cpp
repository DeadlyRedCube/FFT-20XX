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

namespace FFT20XX::SSE264
{
  constexpr uint32_t VecSize = 2u;
  using VecType = __m128d;
  using ElementType = double;
  constexpr auto MinimumRealFFTLength = 32;
  constexpr bool UseFMAOptimization = false;

  template <typename T>
  struct Angles
  {
    T w0;
    T w1;
    T w2;
  };


  bool IsSIMDAligned(const double *ptr)
    { return (uintptr_t(ptr) & (sizeof(__m128) - 1)) == 0; }


  FORCE_INLINE auto Set1(double f) -> VecType
  {
    return _mm_set1_pd(f);
  }



  template <Alignment Align>
  FORCE_INLINE auto Load(const double *ptr) -> VecType
  {
    if constexpr (Align == Alignment::Aligned)
      { return _mm_load_pd(ptr); }
    else
      { return _mm_loadu_pd(ptr); }
  }

  template <Alignment Align>
  FORCE_INLINE void Store(double *ptr, VecType v)
  {
    if constexpr (Align == Alignment::Aligned)
      { return _mm_store_pd(ptr, v); }
    else
      { return _mm_storeu_pd(ptr, v); }
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

  FORCE_INLINE void Transpose2x2(__m128d a, __m128d &b, __m128d &aOut)
  {
    aOut = _mm_shuffle_pd(a, b, 0b00);
    b = _mm_shuffle_pd(a, b, 0b11);
  }

  // Functions specific to complex FFTs ///////////////////////////////////////////////////////////////////////////////

  template <Alignment Align>
  FORCE_INLINE auto LoadPackedComplex(const double *ptr) -> ComplexVec<VecType>
  {
    auto t0 = Load<Align>(ptr);
    auto t1 = Load<Align>(ptr + VecSize);
    return { _mm_shuffle_pd(t0, t1, _MM_SHUFFLE2(0, 0)), _mm_shuffle_pd(t0, t1, _MM_SHUFFLE2(1, 1)) };
  }



  template <Alignment Align>
  FORCE_INLINE void StorePackedComplex(double *ptr, ComplexVec<VecType> v)
  {
    auto aa = _mm_shuffle_pd(v.r, v.i, _MM_SHUFFLE2(0, 0));
    auto bb = _mm_shuffle_pd(v.r, v.i, _MM_SHUFFLE2(1, 1));
    Store<Align>(ptr, aa);
    Store<Align>(ptr + 2, bb);
  }

  // Functions specific to real FFTs //////////////////////////////////////////////////////////////////////////////////

  [[nodiscard]] FORCE_INLINE auto ReplaceFirstOf(auto value, auto newFirst)
  {
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


  #include "fft_20xx_algo_cplx.h"
  #include "fft_20xx_algo_real_common.h"
  #include "fft_20xx_algo_real_fwd.h"
  #include "fft_20xx_algo_real_inv.h"
}


using namespace FFT20XX;
using namespace fft_20xx::c;
extern "C"
{
  void fft_cplx_f64_sse2(const fft_table_cplx_f64 *table, const double *input, double *output, size_t fftLength)
  {
    COMMON_ASSERTS();
    if (SSE264::IsSIMDAligned(input) && SSE264::IsSIMDAligned(output))
      { SSE264::RunComplex<Dir::Forward, Alignment::Aligned>(GetTableData(table), input, output, fftLength); }
    else
      { SSE264::RunComplex<Dir::Forward, Alignment::Unaligned>(GetTableData(table), input, output, fftLength); }
  }

  void ifft_cplx_f64_sse2(const fft_table_cplx_f64 *table, const double *input, double *output, size_t fftLength)
  {
    COMMON_ASSERTS();
    if (SSE264::IsSIMDAligned(input) && SSE264::IsSIMDAligned(output))
      { SSE264::RunComplex<Dir::Inverse, Alignment::Aligned>(GetTableData(table), input, output, fftLength); }
    else
      { SSE264::RunComplex<Dir::Inverse, Alignment::Unaligned>(GetTableData(table), input, output, fftLength); }
  }

  void fft_real_f64_sse2(const fft_table_real_f64 *table, const double *input, double *output, size_t fftLength)
  {
    COMMON_ASSERTS();
    SSE264::RealForwardFFT<Alignment::Unaligned>(GetTableData(table), input, output, fftLength);
  }

  void ifft_real_f64_sse2(const fft_table_real_f64 *table, const double *input, double *output, size_t fftLength)
  {
    COMMON_ASSERTS();
    SSE264::RealInverseFFT<Alignment::Unaligned>(GetTableData(table), input, output, fftLength);
  }
}
