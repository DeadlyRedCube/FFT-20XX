#include "fft_20xx_common.h"


static_assert(CPU_X86 || CPU_X64);

static_assert(SIMD_SSE2);

#include <concepts>
#include <cinttypes>
#include <immintrin.h>

namespace FFT20XX::SSE2
{
  template <typename T> concept f32 = std::same_as<float, T> || std::same_as<__m128, T>;
  template <typename T> concept simd_vec = std::same_as<std::remove_cvref_t<T>, __m128> || std::same_as<std::remove_cvref_t<T>, __m128d>;
  template <typename T> using Vec = std::conditional_t<f32<T>, __m128, __m128d>;

  template <typename> constexpr uint32_t VecSize = 0;

  template <> constexpr uint32_t VecSize<Vec<float>> = 4u;
  template <> constexpr uint32_t VecSize<Vec<double>> = 2u;

  constexpr bool UseFMAOptimization = false;

  template <std::floating_point F>
  static auto LoadA(const F *ptr) -> Vec<F>
  {
    if constexpr (f32<F>)
      { return _mm_load_ps(ptr); }
    else
      { return _mm_load_pd(ptr); }
  }
  template <std::floating_point F>
  static auto Load(const F *ptr) -> Vec<F>
  {
    if constexpr (f32<F>)
      { return _mm_loadu_ps(ptr); }
    else
      { return _mm_loadu_pd(ptr); }
  }

  template <std::floating_point F>
  static auto Load4(const F *ptr) -> Vec<F>
    { return Load(ptr); }

  template <std::floating_point F>
  static auto LoadDeinterleaved(const F *ptr) -> std::pair<Vec<F>, Vec<F>>
  {
    if constexpr (f32<F>)
    {
      auto t0 = _mm_loadu_ps(ptr);
      auto t1 = _mm_loadu_ps(ptr + 4u);
      return { _mm_shuffle_ps(t0, t1, _MM_SHUFFLE(2, 0, 2, 0)), _mm_shuffle_ps(t0, t1, _MM_SHUFFLE(3, 1, 3, 1)) };
    }
    else
    {
      auto t0 = _mm_loadu_pd(ptr);
      auto t1 = _mm_loadu_pd(ptr + 2u);
      return { _mm_shuffle_pd(t0, t1, _MM_SHUFFLE2(0, 0)), _mm_shuffle_pd(t0, t1, _MM_SHUFFLE2(1, 1)) };
    }
  }

  template <std::floating_point F>
  static void Store2x2Transposed(F *ptrA, F *ptrB, Vec<F> a, Vec<F> b)
  {
    auto aa = _mm_shuffle_pd(a, b, _MM_SHUFFLE2(0, 0));
    auto bb = _mm_shuffle_pd(a, b, _MM_SHUFFLE2(1, 1));
    _mm_storeu_pd(ptrA, aa);
    _mm_storeu_pd(ptrB, bb);
  }

  template <std::floating_point F>
  static void Store4x4Transposed(F *ptrA, F *ptrB, F *ptrC, F *ptrD, Vec<F> a, Vec<F> b, Vec<F> c, Vec<F> d)
  {
    static_assert(f32<F>);
    _MM_TRANSPOSE4_PS(a, b, c, d);
    _mm_storeu_ps(ptrA, a);
    _mm_storeu_ps(ptrB, b);
    _mm_storeu_ps(ptrC, c);
    _mm_storeu_ps(ptrD, d);
  }

  template <std::floating_point F>
  static void Store(F *ptr, Vec<F> a)
  {
    if constexpr (f32<F>)
      { _mm_storeu_ps(ptr, a); }
    else 
      { _mm_storeu_pd(ptr, a); }
  }

  template <std::floating_point F>
  static void StoreInterleaved(F *ptr, Vec<F> a, Vec<F> b)
  {
    if constexpr (f32<F>)
    {
      auto aa = _mm_unpacklo_ps(a, b);
      auto bb = _mm_unpackhi_ps(a, b);
      _mm_storeu_ps(ptr, aa);
      _mm_storeu_ps(ptr + 4, bb);
    }
    else
    {
      auto aa = _mm_shuffle_pd(a, b, _MM_SHUFFLE2(0, 0));
      auto bb = _mm_shuffle_pd(a, b, _MM_SHUFFLE2(1, 1));
      _mm_storeu_pd(ptr, aa);
      _mm_storeu_pd(ptr + 2, bb);
    }
  }

  template <std::floating_point F, std::same_as<Vec<F>> ... Vs>
  static void Scale(F scale, Vs &...vs)
  {
    if constexpr (f32<F>)
    {
      auto s = _mm_set1_ps(scale);
      ((vs = _mm_mul_ps(s, vs)), ...);
    }
    else
    {
      auto s = _mm_set1_pd(scale);
      ((vs = _mm_mul_pd(s, vs)), ...);
    }
  }
  
  template <simd_vec V>
  static V Add(V a, V b)
  {
    if constexpr (f32<V>)
      { return _mm_add_ps(a, b); }
    else
      { return _mm_add_pd(a, b); }
  }

  template <simd_vec V>
  static V Sub(V a, V b)
  {
    if constexpr (f32<V>)
      { return _mm_sub_ps(a, b); }
    else
      { return _mm_sub_pd(a, b); }
  }

  template <simd_vec V>
  static V Mul(V a, V b)
  {
    if constexpr (f32<V>)
      { return _mm_mul_ps(a, b); }
    else
      { return _mm_mul_pd(a, b); }
  }

  template <simd_vec V>
  static V MAdd(V a, V b, V c)
    { return Add(Mul(a, b), c); }

  template <simd_vec V>
  static V NMAdd(V a, V b, V c)
    { return Sub(c, Mul(a, b)); }

  #include "fft_20xx_run.h"
}


using namespace FFT20XX;

extern "C"
{
  #define COMMON_ASSERTS() \
    assert(table != nullptr); \
    assert(input != nullptr); \
    assert(output != nullptr); \
    assert(IsPowerOfTwo(fftLength)); \
    assert(t->MaxFFTLength() >= fftLength)

  void FFTComplex32Interleaved_SSE2(const FFTTable32 *table, const float *input, float *output, size_t fftLength)
  {
    auto t = reinterpret_cast<const FFTTableImpl<float> *>(table);
    COMMON_ASSERTS();
    SSE2::DITRadix4FFT<Dir::Forward>(
      t->Data(),
      reinterpret_cast<const Complex<float> *>(input),
      reinterpret_cast<Complex<float> *>(output),
      fftLength);
  }

  void FFTComplex64Interleaved_SSE2(const FFTTable64 *table, const double *input, double *output, size_t fftLength)
  {
    auto t = reinterpret_cast<const FFTTableImpl<double> *>(table);
    COMMON_ASSERTS();
    SSE2::DITRadix4FFT<Dir::Forward>(
      t->Data(),
      reinterpret_cast<const Complex<double> *>(input),
      reinterpret_cast<Complex<double> *>(output),
      fftLength);
  }


  void IFFTComplex32Interleaved_SSE2(const FFTTable32 *table, const float *input, float *output, size_t fftLength)
  {
    auto t = reinterpret_cast<const FFTTableImpl<float> *>(table);
    COMMON_ASSERTS();
    SSE2::DITRadix4FFT<Dir::Inverse>(
      t->Data(),
      reinterpret_cast<const Complex<float> *>(input),
      reinterpret_cast<Complex<float> *>(output),
      fftLength);
  }

  void IFFTComplex64Interleaved_SSE2(const FFTTable64 *table, const double *input, double *output, size_t fftLength)
  {
    auto t = reinterpret_cast<const FFTTableImpl<double> *>(table);
    COMMON_ASSERTS();
    SSE2::DITRadix4FFT<Dir::Inverse>(
      t->Data(),
      reinterpret_cast<const Complex<double> *>(input),
      reinterpret_cast<Complex<double> *>(output),
      fftLength);
  }
}