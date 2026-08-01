#ifdef _MSC_VER
  #pragma warning (disable: 4711) // function <x> selected for automatic inline expansion
#endif

#include "fft_20xx_common.h"
#include "fft_20xx_table.h"
#include "fft_20xx_tiny_cplx.h"
#include "fft_20xx_tiny_real.h"

static_assert(CPU_ARM64);
static_assert(SIMD_NEON);

#if COMPILER_GCC
  #pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

using namespace FFT20XX;

namespace FFT20XX::NEON64
{
  constexpr uint32_t VecSize = 2u;
  using VecType = float64x2_t;
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
    { return (uintptr_t(ptr) & (sizeof(VecType) - 1)) == 0; }


  FORCE_INLINE auto Set1(double f) -> VecType
  {
    return vdupq_n_f64(f);
  }



  template <Alignment Align>
  FORCE_INLINE auto Load(const double *ptr) -> VecType
  {
    return vld1q_f64(ptr);
  }

  template <Alignment Align>
  FORCE_INLINE void Store(double *ptr, VecType v)
  {
    vst1q_f64(ptr, v);
  }

  [[nodiscard]] FORCE_INLINE auto Add(float64x2_t a, float64x2_t b)
  {
    return vaddq_f64(a, b);
  }

  [[nodiscard]] FORCE_INLINE auto Sub(float64x2_t a, float64x2_t b)
  {
    return vsubq_f64(a, b);
  }

  [[nodiscard]] FORCE_INLINE auto Mul(float64x2_t a, float64x2_t b)
  {
    return vmulq_f64(a, b);
  }

  [[nodiscard]] FORCE_INLINE auto Mul(float64x2_t a, double b)
  {
    return vmulq_n_f64(a, b);
  }

  [[nodiscard]] FORCE_INLINE auto MAdd(float64x2_t a, auto b, float64x2_t c) -> float64x2_t
  {
    if constexpr (std::floating_point<decltype(b)>)
      { return vfmaq_n_f64(c, a, b); }
    else
      { return vfmaq_f64(c, b, a); }
  }

  [[nodiscard]] FORCE_INLINE auto NMAdd(float64x2_t a, auto b, float64x2_t c) -> float64x2_t
  {
    if constexpr (std::floating_point<decltype(b)>)
      { return vfmsq_n_f64(c, a, b); }
    else
      { return vfmsq_f64(c, b, a); }
  }

  FORCE_INLINE void Transpose2x2(auto a, auto &b, auto &aOut)
  {
    aOut = vtrn1q_f64(a, b);
    b = vtrn2q_f64(a, b);
  }

  // Functions specific to complex FFTs ///////////////////////////////////////////////////////////////////////////////

  template <Alignment Align>
  FORCE_INLINE auto LoadPackedComplex(const double *ptr) -> ComplexVec<VecType>
  {
    auto r = vld2q_f64(ptr);
    return { r.val[0], r.val[1] };
  }



  template <Alignment Align>
  FORCE_INLINE void StorePackedComplex(double *ptr, ComplexVec<VecType> v)
  {
    vst2q_f64(ptr, (float64x2x2_t{v.r, v.i}));
  }

  // Functions specific to real FFTs //////////////////////////////////////////////////////////////////////////////////

  [[nodiscard]] FORCE_INLINE auto ReplaceFirstOf(auto value, auto newFirst)
  {
    if constexpr (std::floating_point<decltype(newFirst)>)
      { return vsetq_lane_f64(newFirst, value, 0); }
    else
      { return vsetq_lane_f64(vgetq_lane_f64(newFirst, 0), value, 0); }
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
  void fft_cplx_f64_neon(const fft_table_cplx_f64 *table, const double *input, double *output, size_t fftLength)
  {
    COMMON_ASSERTS();
    NEON64::RunComplex<Dir::Forward, Alignment::Unaligned>(GetTableData(table), input, output, fftLength);
  }


  void ifft_cplx_f64_neon(const fft_table_cplx_f64 *table, const double *input, double *output, size_t fftLength)
  {
    COMMON_ASSERTS();
    NEON64::RunComplex<Dir::Inverse, Alignment::Unaligned>(GetTableData(table), input, output, fftLength);
  }

  void fft_real_f64_neon(const fft_table_real_f64 *table, const double *input, double *output, size_t fftLength)
  {
    COMMON_ASSERTS();
    NEON64::RealForwardFFT<Alignment::Unaligned>(GetTableData(table), input, output, fftLength);
  }



  void ifft_real_f64_neon(const fft_table_real_f64 *table, const double *input, double *output, size_t fftLength)
  {
    COMMON_ASSERTS();
    NEON64::RealInverseFFT<Alignment::Unaligned>(GetTableData(table), input, output, fftLength);
  }
}
