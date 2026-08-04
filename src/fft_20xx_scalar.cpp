#ifdef _MSC_VER
  #pragma warning (disable: 4711) // function <x> selected for automatic inline expansion
#endif

#include "fft_20xx_common.h"
#include "fft_20xx_table.h"
#include "fft_20xx_tiny_cplx.h"
#include "fft_20xx_tiny_real.h"

#if COMPILER_GCC
  #pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

using namespace FFT20XX;

namespace FFT20XX::Scalar
{
  constexpr uint32_t VecSize = 4u;
  constexpr auto MinimumRealFFTLength = 32;
  constexpr bool UseFMAOptimization = false;

  template <std::floating_point F>
  struct Vec
  {
    F v[VecSize];
  };

  template <typename T>
  struct Angles
  {
    T w0;
    T w1;
    T w2;
  };


  template <std::floating_point F>
  FORCE_INLINE auto Set1(F f) -> Vec<F>
  {
    Vec<F> r;
    for (auto &v : r.v)
      { v = f; }
    return r;
  }



  template <Alignment Align, std::floating_point F>
  FORCE_INLINE auto Load(const F *ptr)
  {
    Vec<F> r;
    for (auto i = 0u; i < VecSize; i++)
      { r.v[i] = ptr[i]; }
    return r;
  }

  template <Alignment Align, std::floating_point F>
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


  [[nodiscard]] FORCE_INLINE auto MAdd(auto a, auto b, auto c)
  {
    return Add(Mul(a, b), c);
  }

  [[nodiscard]] FORCE_INLINE auto NMAdd(auto a, auto b, auto c)
  {
    return Sub(c, Mul(a, b));
  }



  template <std::floating_point F>
  FORCE_INLINE void Transpose4x4(
    Vec<F> &a,
    Vec<F> &b,
    Vec<F> &c,
    Vec<F> &d)
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

  // Functions specific to complex FFTs ///////////////////////////////////////////////////////////////////////////////

  template <Alignment Align, std::floating_point F>
  FORCE_INLINE auto LoadPackedComplex(const F *ptr) -> ComplexVec<Vec<F>>
  {
    ComplexVec<Vec<F>> r;
    for (auto i = 0u; i < VecSize; i++)
    {
      r.r.v[i] = ptr[2 * i + 0];
      r.i.v[i] = ptr[2 * i + 1];
    }

    return r;
  }



  template <Alignment Align, std::floating_point F>
  FORCE_INLINE void StorePackedComplex(F *ptr, ComplexVec<Vec<F>> v)
  {
    for (auto i = 0u; i < VecSize; i++)
    {
      ptr[2 * i + 0] = v.r.v[i];
      ptr[2 * i + 1] = v.i.v[i];
    }
  }

  // Functions specific to real FFTs //////////////////////////////////////////////////////////////////////////////////

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
    #include "fft_20xx_algo_cplx.h"
    #include "fft_20xx_algo_real_common.h"
    #include "fft_20xx_algo_real_fwd.h"
    #include "fft_20xx_algo_real_inv.h"
  }

  namespace F64
  {
    using ElementType = double;
    using VecType = Vec<ElementType>;
    #include "fft_20xx_algo_cplx.h"
    #include "fft_20xx_algo_real_common.h"
    #include "fft_20xx_algo_real_fwd.h"
    #include "fft_20xx_algo_real_inv.h"
  }
}


using namespace FFT20XX;
using namespace fft_20xx::c;
extern "C"
{
  void fft_cplx_f32_scalar(const fft_table_cplx_f32 *table, const float *input, float *output, size_t fftLength)
  {
    COMMON_ASSERTS();
    Scalar::F32::RunComplex<Dir::Forward>(GetTableData(table), input, output, fftLength);
  }


  void ifft_cplx_f32_scalar(const fft_table_cplx_f32 *table, const float *input, float *output, size_t fftLength)
  {
    COMMON_ASSERTS();
    Scalar::F32::RunComplex<Dir::Inverse>(GetTableData(table), input, output, fftLength);
  }

  void fft_cplx_f64_scalar(const fft_table_cplx_f64 *table, const double *input, double *output, size_t fftLength)
  {
    COMMON_ASSERTS();
    Scalar::F64::RunComplex<Dir::Forward>(GetTableData(table), input, output, fftLength);
  }


  void ifft_cplx_f64_scalar(const fft_table_cplx_f64 *table, const double *input, double *output, size_t fftLength)
  {
    COMMON_ASSERTS();
    Scalar::F64::RunComplex<Dir::Inverse>(GetTableData(table), input, output, fftLength);
  }

  void fft_real_f32_scalar(const fft_table_real_f32 *table, const float *input, float *output, size_t fftLength)
  {
    COMMON_ASSERTS();
    Scalar::F32::RealForwardFFT(GetTableData(table), input, output, fftLength);
  }



  void ifft_real_f32_scalar(const fft_table_real_f32 *table, const float *input, float *output, size_t fftLength)
  {
    COMMON_ASSERTS();
    Scalar::F32::RealInverseFFT(GetTableData(table), input, output, fftLength);
  }


  void fft_real_f64_scalar(const fft_table_real_f64 *table, const double *input, double *output, size_t fftLength)
  {
    COMMON_ASSERTS();
    Scalar::F64::RealForwardFFT(GetTableData(table), input, output, fftLength);
  }



  void ifft_real_f64_scalar(const fft_table_real_f64 *table, const double *input, double *output, size_t fftLength)
  {
    COMMON_ASSERTS();
    Scalar::F64::RealInverseFFT(GetTableData(table), input, output, fftLength);
  }
}
