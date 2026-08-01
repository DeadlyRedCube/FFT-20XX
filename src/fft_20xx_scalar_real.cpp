#ifdef _MSC_VER
  #pragma warning (disable: 4711) // function <x> selected for automatic inline expansion
#endif

#include "fft_20xx_common.h"
#include "fft_20xx_table.h"
#include "fft_20xx_tiny_real.h"

#if COMPILER_GCC
  #pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

namespace FFT20XX::ScalarReal
{
  constexpr auto MinimumRealFFTLength = 32;
  constexpr bool UseFMAOptimization = false;
  constexpr auto VecSize = 4u;

  template <std::floating_point F>
  struct Vec
  {
    F v[VecSize];
  };


  template <std::floating_point F>
  bool IsSIMDAligned(const F *ptr)
    { return (uintptr_t(ptr) & (sizeof(Vec<F>) - 1)) == 0; }


  template <Alignment A, std::floating_point F>
  [[nodiscard]] FORCE_INLINE auto Load(const F *ptr) -> Vec<F>
  {
    Vec<F> r;
    for (auto i = 0u; i < VecSize; i++)
      { r.v[i] = ptr[i]; }
    return r;
  }



  template <Alignment A, std::floating_point F>
  FORCE_INLINE void Store(F *ptr, Vec<F> v)
  {
    for (auto i = 0u; i < VecSize; i++)
      { ptr[i] = v.v[i]; }
  }



  template <std::floating_point F>
  FORCE_INLINE Vec<F> Add(Vec<F> a, Vec<F> b)
  {
    Vec<F> r;
    for (auto i = 0u; i < VecSize; i++)
      { r.v[i] = a.v[i] + b.v[i]; }
    return r;
  }

  template <std::floating_point F>
  FORCE_INLINE Vec<F> Sub(Vec<F> a, Vec<F> b)
  {
    Vec<F> r;
    for (auto i = 0u; i < VecSize; i++)
      { r.v[i] = a.v[i] - b.v[i]; }
    return r;
  }

  template <std::floating_point F>
  FORCE_INLINE Vec<F> Mul(Vec<F> a, Vec<F> b)
  {
    Vec<F> r;
    for (auto i = 0u; i < VecSize; i++)
      { r.v[i] = a.v[i] * b.v[i]; }
    return r;
  }

  template <std::floating_point F>
  FORCE_INLINE Vec<F> Mul(Vec<F> a, F b)
  {
    Vec<F> r;
    for (auto i = 0u; i < VecSize; i++)
      { r.v[i] = a.v[i] * b; }
    return r;
  }

  template <std::floating_point F>
  [[nodiscard]] FORCE_INLINE auto MAdd(Vec<F> a, auto b, Vec<F> c) -> Vec<F>
  {
    return Add(Mul(a, b), c);
  }

  template <std::floating_point F>
  [[nodiscard]] FORCE_INLINE auto NMAdd(Vec<F> a, auto b, Vec<F> c) -> Vec<F>
  {
    return Sub(c, Mul(a, b));
  }

  template <std::floating_point F>
  [[nodiscard]] FORCE_INLINE auto ReplaceFirstOf(Vec<F> value, auto newFirst) -> Vec<F>
  {
    // For AVX the best version I've found is:
    // _mm256_blend_ps(a, _mm256_castps128_ps256(_mm_set_ss(d)), 0b00000001);
    //  (works great in all 3 tested compilers)
    if constexpr (std::floating_point<decltype(newFirst)>)
      { value.v[0] = newFirst; }
    else
      { value.v[0] = newFirst.v[0]; }
    return value;
  }

  template <std::floating_point F>
  [[nodiscard]] FORCE_INLINE auto InsertTopHalf(Vec<F> lo, Vec<F> hi) -> Vec<F>
  {
    for (auto i = 0u; i < VecSize / 2; i++)
      { lo.v[i + VecSize / 2] = hi.v[i + VecSize / 2]; }
    return lo;
  }

  template <std::floating_point F>
  FORCE_INLINE void Transpose4(Vec<F> &a, Vec<F> &b, Vec<F> &c, Vec<F> &d)
  {
    for (auto x = 0u; x < VecSize; x += 4u)
    {
      std::swap(a.v[x + 1], b.v[x + 0]);
      std::swap(a.v[x + 2], c.v[x + 0]);
      std::swap(a.v[x + 3], d.v[x + 0]);
      std::swap(b.v[x + 2], c.v[x + 1]);
      std::swap(b.v[x + 3], d.v[x + 1]);
      std::swap(c.v[x + 3], d.v[x + 2]);
    }
  }



  // Takes the first two elements of aceg0 and bdfh0 and puts them in t, then the second two elements from each into bdfh0.
  template <std::floating_point F>
  FORCE_INLINE void SwapRealComponentsForLastDoubleFlip(Vec<F> aceg0, Vec<F> &bdfh0, Vec<F> &t)
  {
    constexpr auto H = VecSize / 2;
    for (auto i = 0u; i < H; i++)
    {
      t.v[i] = aceg0.v[i];
      t.v[i + H] = bdfh0.v[i];
      bdfh0.v[i] = aceg0.v[i + H];
    }
  }



  template <std::floating_point F>
  FORCE_INLINE void Deinterleave4(Vec<F> a, Vec<F> &b, Vec<F> &aOut)
  {
    constexpr auto H = VecSize / 2;
    Vec<F> t;
    for (auto i = 0u; i < H; i++)
    {
      aOut.v[i + 0] = a.v[2 * i];
      aOut.v[i + H] = b.v[2 * i];
      t.v[i + 0] = a.v[2 * i + 1];
      t.v[i + H] = b.v[2 * i + 1];
    }

    b = t;
  }



  template <std::floating_point F>
  FORCE_INLINE void Interleave4(Vec<F> a, Vec<F> &b, Vec<F> &aOut)
  {
    constexpr auto H = VecSize / 2;
    Vec<F> bOut;
    for (auto i = 0u; i < H; i++)
    {
      aOut.v[2 * i] = a.v[i + 0];
      aOut.v[2 * i + 1] = b.v[i + 0];
      bOut.v[2 * i] = a.v[i + H];
      bOut.v[2 * i + 1] = b.v[i + H];
    }

    b = bOut;
  }


  namespace F32
  {
    using ElementType = float;
    using VecType = Vec<ElementType>;
    #include "fft_20xx_algo_real_common.h"
    #include "fft_20xx_algo_real_fwd.h"
    #include "fft_20xx_algo_real_inv.h"
  }

  namespace F64
  {
    using ElementType = double;
    using VecType = Vec<ElementType>;
    #include "fft_20xx_algo_real_common.h"
    #include "fft_20xx_algo_real_fwd.h"
    #include "fft_20xx_algo_real_inv.h"
  }
}


using namespace FFT20XX;
using namespace fft_20xx::c;
extern "C"
{
  void fft_real_f32_scalar(const fft_table_f32 *table, const float *input, float *output, size_t fftLength)
  {
    auto t = reinterpret_cast<const FFTTableImpl<float> *>(table);
    COMMON_ASSERTS();
    ScalarReal::F32::RealForwardFFT<Alignment::Unaligned>(t->RealData(), input, output, fftLength);
  }



  void ifft_real_f32_scalar(const fft_table_f32 *table, const float *input, float *output, size_t fftLength)
  {
    auto t = reinterpret_cast<const FFTTableImpl<float> *>(table);
    COMMON_ASSERTS();
    ScalarReal::F32::RealInverseFFT<Alignment::Unaligned>(t->RealData(), input, output, fftLength);
  }


  void fft_real_f64_scalar(const fft_table_f64 *table, const double *input, double *output, size_t fftLength)
  {
    auto t = reinterpret_cast<const FFTTableImpl<double> *>(table);
    COMMON_ASSERTS();
    ScalarReal::F64::RealForwardFFT<Alignment::Unaligned>(t->RealData(), input, output, fftLength);
  }



  void ifft_real_f64_scalar(const fft_table_f64 *table, const double *input, double *output, size_t fftLength)
  {
    auto t = reinterpret_cast<const FFTTableImpl<double> *>(table);
    COMMON_ASSERTS();
    ScalarReal::F64::RealInverseFFT<Alignment::Unaligned>(t->RealData(), input, output, fftLength);
  }
}
