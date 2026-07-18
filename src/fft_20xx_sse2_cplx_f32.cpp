#ifdef _MSC_VER
  #pragma warning (disable: 4711) // function <x> selected for automatic inline expansion
#endif

#include "fft_20xx_common.h"
#include "fft_20xx_table.h"
#include "fft_20xx_tiny.h"

static_assert(CPU_X86 || CPU_X64);

static_assert(SIMD_SSE2);

#if COMPILER_GCC
  #pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

using namespace FFT20XX;

namespace FFT20XX::SSE2
{
  template <typename T> concept simd_vec = std::same_as<std::remove_cvref_t<T>, __m128> || std::same_as<std::remove_cvref_t<T>, __m128d>;
  template <typename T> using Vec = __m128;

  template <typename> constexpr uint32_t VecSize = 4u;

  template <typename T>
  struct Angles
  {
    T w0;
    T w1;
    T w2;
  };


  template <std::floating_point F>
  bool IsSIMDAligned(const F *ptr)
    { return (uintptr_t(ptr) & (sizeof(__m128) - 1)) == 0; }


  template <std::floating_point F>
  FORCE_INLINE auto Set1(F f) -> Vec<F>
  {
    return _mm_set1_ps(f);
  }



  template <Alignment Align, std::floating_point F>
  FORCE_INLINE auto Load(const F *ptr) -> Vec<F>
  {
    if constexpr (Align == Alignment::Aligned)
      { return _mm_load_ps(ptr); }
    else
      { return _mm_loadu_ps(ptr); }
  }

  template <Alignment Align, std::floating_point F>
  FORCE_INLINE void Store(F *ptr, Vec<F> v)
  {
    if constexpr (Align == Alignment::Aligned)
      { return _mm_store_ps(ptr, v); }
    else
      { return _mm_storeu_ps(ptr, v); }
  }



  template <Alignment Align, std::floating_point F>
  FORCE_INLINE auto LoadUnpackedComplex(const F *ptr) -> ComplexVec<Vec<F>>
    { return { Load<Align>(ptr), Load<Align>(ptr + VecSize<F>) }; }


  template <Alignment Align, std::floating_point F>
  FORCE_INLINE void StoreUnpackedComplex(F *ptr, ComplexVec<Vec<F>> v)
  {
    Store<Align>(ptr, v.r);
    Store<Align>(ptr + VecSize<F>, v.i);
  }



  template <Alignment Align, std::floating_point F>
  FORCE_INLINE auto LoadPackedComplex(const F *ptr) -> ComplexVec<Vec<F>>
  {
    auto t0 = Load<Align>(ptr);
    auto t1 = Load<Align>(ptr + VecSize<F>);
    return { _mm_shuffle_ps(t0, t1, _MM_SHUFFLE(2, 0, 2, 0)), _mm_shuffle_ps(t0, t1, _MM_SHUFFLE(3, 1, 3, 1)) };
  }



  template <Alignment Align, std::floating_point F>
  FORCE_INLINE void StorePackedComplex(F *ptr, ComplexVec<Vec<F>> v)
  {
    auto aa = _mm_unpacklo_ps(v.r, v.i);
    auto bb = _mm_unpackhi_ps(v.r, v.i);
    Store<Align>(ptr, aa);
    Store<Align>(ptr + 4, bb);
  }





  template <simd_vec V>
  FORCE_INLINE void AddSub(ComplexVec<V> &a, ComplexVec<V> &b)
  {
    auto t = _mm_add_ps(a.r, b.r);
    b.r = _mm_sub_ps(a.r, b.r);
    a.r = t;
    t = _mm_add_ps(a.i, b.i);
    b.i = _mm_sub_ps(a.i, b.i);
    a.i = t;
  }



  template <Dir D, simd_vec V>
  FORCE_INLINE void AddSubRot45(ComplexVec<V> &a, ComplexVec<V> &b)
  {
    const auto s = _mm_set1_ps(InvSqrt2<float>);
    if constexpr (D == Dir::Forward)
    {
      auto t = _mm_add_ps(b.r, b.i);
      b.i = _mm_sub_ps(b.i, b.r);
      b.r = _mm_mul_ps(s, t);
      b.i = _mm_mul_ps(s, b.i);
    }
    else
    {
      auto t = _mm_sub_ps(b.r, b.i);
      b.i = _mm_add_ps(b.i, b.r);
      b.r = _mm_mul_ps(s, t);
      b.i = _mm_mul_ps(s, b.i);
    }

    auto t = _mm_add_ps(a.r, b.r);
    b.r = _mm_sub_ps(a.r, b.r);
    a.r = t;
    t = _mm_add_ps(a.i, b.i);
    b.i = _mm_sub_ps(a.i, b.i);
    a.i = t;
  }



  template <Dir D, simd_vec V>
  FORCE_INLINE void AddSubRot90(ComplexVec<V> &a, ComplexVec<V> &b)
  {
    if constexpr (D == Dir::Forward)
    {
      auto t = _mm_add_ps(a.r, b.i);
      b.i = _mm_sub_ps(a.r, b.i);
      a.r = t;
      t = _mm_sub_ps(a.i, b.r);
      b.r = _mm_add_ps(a.i, b.r);
      a.i = t;
    }
    else
    {
      auto t = _mm_sub_ps(a.r, b.i);
      b.i = _mm_add_ps(a.r, b.i);
      a.r = t;
      t = _mm_add_ps(a.i, b.r);
      b.r = _mm_sub_ps(a.i, b.r);
      a.i = t;
    }
    std::swap(b.r, b.i);
  }



  template <Dir D, simd_vec V>
  FORCE_INLINE void AddSubRot135(ComplexVec<V> &a, ComplexVec<V> &b)
  {
    const auto s = _mm_set1_ps(InvSqrt2<float>);
    if constexpr (D == Dir::Forward)
    {
      auto t = _mm_sub_ps(b.i, b.r);
      b.i = _mm_add_ps(b.i, b.r); // This is actually -b.i
      b.r = _mm_mul_ps(s, t);
      b.i = _mm_mul_ps(s, b.i);

      t = _mm_add_ps(a.r, b.r);
      b.r = _mm_sub_ps(a.r, b.r);
      a.r = t;
      t = _mm_sub_ps(a.i, b.i); // Because b.i is -b.i, addition is subtraction!
      b.i = _mm_add_ps(a.i, b.i);
      a.i = t;
    }
    else
    {
      auto t = _mm_add_ps(b.i, b.r); // This is actually -b.r
      b.i = _mm_sub_ps(b.r, b.i);
      b.r = _mm_mul_ps(s, t);
      b.i = _mm_mul_ps(s, b.i);

      t = _mm_sub_ps(a.r, b.r); // because b.r is actually -b.r, swap the addition/subtraction
      b.r = _mm_add_ps(a.r, b.r);
      a.r = t;
      t = _mm_add_ps(a.i, b.i);
      b.i = _mm_sub_ps(a.i, b.i);
      a.i = t;
    }
  }


  template <simd_vec V>
  [[nodiscard]] FORCE_INLINE auto ComplexMultiply(ComplexVec<V> a, ComplexVec<V> b) -> ComplexVec<V>
  {
    auto t = _mm_sub_ps(_mm_mul_ps(a.r, b.r), _mm_mul_ps(a.i, b.i));
    a.i = _mm_add_ps(_mm_mul_ps(a.r, b.i), _mm_mul_ps(a.i, b.r));
    a.r = t;
    return a;
  }

  template <simd_vec V>
  [[nodiscard]] FORCE_INLINE auto ScalarMultiply(ComplexVec<V> a, V b) -> ComplexVec<V>
  {
    return { _mm_mul_ps(a.r, b), _mm_mul_ps(a.i, b) };
  }

  template <Dir D, simd_vec V>
  FORCE_INLINE void Rotate(ComplexVec<V> &a, ComplexVec<V> w)
  {
    if constexpr (D == Dir::Forward)
    {
      auto t = _mm_sub_ps(_mm_mul_ps(a.r, w.r), _mm_mul_ps(a.i, w.i));
      a.i = _mm_add_ps(_mm_mul_ps(a.r, w.i), _mm_mul_ps(a.i, w.r));
      a.r = t;
    }
    else
    {
      auto t = _mm_add_ps(_mm_mul_ps(a.r, w.r), _mm_mul_ps(a.i, w.i));
      a.i = _mm_sub_ps(_mm_mul_ps(a.i, w.r), _mm_mul_ps(a.r, w.i));
      a.r = t;
    }
  }


  template <Dir D, simd_vec V, std::floating_point F>
  FORCE_INLINE void Rotate(ComplexVec<V> &a, Complex<F> w)
  {
    auto wR = _mm_set1_ps(w.r);
    auto wI = _mm_set1_ps(w.i);
    if constexpr (D == Dir::Forward)
    {
      auto t = _mm_sub_ps(_mm_mul_ps(a.r, wR), _mm_mul_ps(a.i, wI));
      a.i = _mm_add_ps(_mm_mul_ps(a.r, wI), _mm_mul_ps(a.i, wR));
      a.r = t;
    }
    else
    {
      auto t = _mm_add_ps(_mm_mul_ps(a.r, wR), _mm_mul_ps(a.i, wI));
      a.i = _mm_sub_ps(_mm_mul_ps(a.i, wR), _mm_mul_ps(a.r, wI));
      a.r = t;
    }
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
  void fft_cplx_interleaved_f32_sse2(const fft_table_f32 *table, const float *input, float *output, size_t fftLength)
  {
    [[maybe_unused]] auto t = reinterpret_cast<const FFTTableImpl<float> *>(table);
    COMMON_ASSERTS();
    if (SSE2::IsSIMDAligned(input) && SSE2::IsSIMDAligned(output))
      { SSE2::RunComplex<Dir::Forward, Alignment::Aligned>(t->Data(), input, output, fftLength); }
    else
      { SSE2::RunComplex<Dir::Forward, Alignment::Unaligned>(t->Data(), input, output, fftLength); }
  }


  void ifft_cplx_interleaved_f32_sse2(const fft_table_f32 *table, const float *input, float *output, size_t fftLength)
  {
    [[maybe_unused]] auto t = reinterpret_cast<const FFTTableImpl<float> *>(table);
    COMMON_ASSERTS();
    if (SSE2::IsSIMDAligned(input) && SSE2::IsSIMDAligned(output))
      { SSE2::RunComplex<Dir::Inverse, Alignment::Aligned>(t->Data(), input, output, fftLength); }
    else
      { SSE2::RunComplex<Dir::Inverse, Alignment::Unaligned>(t->Data(), input, output, fftLength); }
  }
}
