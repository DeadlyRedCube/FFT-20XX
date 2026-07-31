#ifdef _MSC_VER
  #pragma warning (disable: 4711) // function <x> selected for automatic inline expansion
#endif

#include "fft_20xx_common.h"
#include "fft_20xx_table.h"
#include "fft_20xx_tiny_real.h"

static_assert(CPU_ARM64);
static_assert(SIMD_NEON);

#include <arm_neon.h>

#if COMPILER_GCC
  #pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

namespace FFT20XX::NEONReal64
{
  template <std::floating_point F>
  bool IsSIMDAligned(const F *ptr)
    { return (uintptr_t(ptr) & (sizeof(float64x2_t) - 1)) == 0; }

  constexpr auto MinimumRealFFTLength = 32;
  constexpr bool UseFMAOptimization = false;

  constexpr auto VecSize = 2u;
  using VecType = float64x2_t;
  using ElementType = double;

  template <Alignment A>
  [[nodiscard]] FORCE_INLINE auto Load(const double *p)
  {
    return vld1q_f64(p);
  }



  template <Alignment A>
  FORCE_INLINE void Store(double *p, float64x2_t v)
  {
    vst1q_f64(p, v);
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

  [[nodiscard]] FORCE_INLINE auto ReplaceFirstOf(auto value, auto newFirst)
  {
    if constexpr (std::floating_point<decltype(newFirst)>)
      { return vsetq_lane_f64(newFirst, value, 0); }
    else
      { return vsetq_lane_f64(vgetq_lane_f64(newFirst, 0), value, 0); }
  }

  FORCE_INLINE void Transpose2(auto a, auto &b, auto &aOut)
  {
    aOut = vtrn1q_f64(a, b);
    b = vtrn2q_f64(a, b);
  }

  #include "fft_20xx_algo_real_fwd.h"
  #include "fft_20xx_algo_real_inv.h"
}


using namespace FFT20XX;
using namespace fft_20xx::c;
extern "C"
{
  void fft_real_f64_neon(const fft_table_f64 *table, const double *input, double *output, size_t fftLength)
  {
    auto t = reinterpret_cast<const FFTTableImpl<double> *>(table);
    COMMON_ASSERTS();
    NEONReal64::RealForwardFFT<Alignment::Unaligned>(t->RealData(), input, output, fftLength);
  }



  void ifft_real_f64_neon(const fft_table_f64 *table, const double *input, double *output, size_t fftLength)
  {
    auto t = reinterpret_cast<const FFTTableImpl<double> *>(table);
    COMMON_ASSERTS();
    NEONReal64::RealInverseFFT<Alignment::Unaligned>(t->RealData(), input, output, fftLength);
  }
}
