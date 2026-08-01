#ifdef _MSC_VER
  #pragma warning (disable: 4711) // function <x> selected for automatic inline expansion
#endif

#include "fft_20xx_common.h"
#include "fft_20xx_table.h"
#include "fft_20xx_tiny_cplx.h"

#if COMPILER_GCC
  #pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

using namespace FFT20XX;

namespace FFT20XX::ScalarCplx
{
  constexpr uint32_t VecSize = 4u;

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



  template <Alignment Align, std::floating_point F>
  FORCE_INLINE auto LoadUnpackedComplex(const F *ptr) -> ComplexVec<Vec<F>>
    { return { Load<Align>(ptr), Load<Align>(ptr + VecSize) }; }


  template <Alignment Align, std::floating_point F>
  FORCE_INLINE void StoreUnpackedComplex(F *ptr, ComplexVec<Vec<F>> v)
  {
    Store<Align>(ptr, v.r);
    Store<Align>(ptr + VecSize, v.i);
  }



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

  template <std::floating_point F>
  FORCE_INLINE void Transpose4x4(
    ComplexVec<Vec<F>> &a,
    ComplexVec<Vec<F>> &b,
    ComplexVec<Vec<F>> &c,
    ComplexVec<Vec<F>> &d)
  {
    Transpose4x4(a.r, b.r, c.r, d.r);
    Transpose4x4(a.i, b.i, c.i, d.i);
  }

  namespace F32
  {
    using ElementType = float;
    using VecType = Vec<ElementType>;
    #include "fft_20xx_algo_cplx.h"
  }

  namespace F64
  {
    using ElementType = double;
    using VecType = Vec<ElementType>;
    #include "fft_20xx_algo_cplx.h"
  }
}


using namespace FFT20XX;
using namespace fft_20xx::c;
extern "C"
{
  void fft_cplx_f32_scalar(const fft_table_cplx_f32 *table, const float *input, float *output, size_t fftLength)
  {
    COMMON_ASSERTS();
    ScalarCplx::F32::RunComplex<Dir::Forward, Alignment::Unaligned>(GetTableData(table), input, output, fftLength);
  }


  void ifft_cplx_f32_scalar(const fft_table_cplx_f32 *table, const float *input, float *output, size_t fftLength)
  {
    COMMON_ASSERTS();
    ScalarCplx::F32::RunComplex<Dir::Inverse, Alignment::Unaligned>(GetTableData(table), input, output, fftLength);
  }

  void fft_cplx_f64_scalar(const fft_table_cplx_f64 *table, const double *input, double *output, size_t fftLength)
  {
    COMMON_ASSERTS();
    ScalarCplx::F64::RunComplex<Dir::Forward, Alignment::Unaligned>(GetTableData(table), input, output, fftLength);
  }


  void ifft_cplx_f64_scalar(const fft_table_cplx_f64 *table, const double *input, double *output, size_t fftLength)
  {
    COMMON_ASSERTS();
    ScalarCplx::F64::RunComplex<Dir::Inverse, Alignment::Unaligned>(GetTableData(table), input, output, fftLength);
  }
}
