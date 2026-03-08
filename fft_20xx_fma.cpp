#include "fft_20xx_common.h"


static_assert(CPU_X86 || CPU_X64);

static_assert(SIMD_AVX);
static_assert(SIMD_FMA);

#include <concepts>
#include <cinttypes>
#include <immintrin.h>

#if COMPILER_GCC
  #pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

namespace FFT20XX::FMA
{
  template <typename T> concept f32 = std::same_as<float, T> || std::same_as<__m128, T> || std::same_as<__m256, T>;
  template <typename T> concept simd_vec = std::same_as<std::remove_cvref_t<T>, __m256> || std::same_as<std::remove_cvref_t<T>, __m256d>;
  template <typename T> concept simd_half_vec = std::same_as<std::remove_cvref_t<T>, __m128> || std::same_as<std::remove_cvref_t<T>, __m128d>;

  template <typename T> using Vec = std::conditional_t<f32<T>, __m256, __m256d>;

  template <typename> constexpr uint32_t VecSize = 0;

  template <> constexpr uint32_t VecSize<Vec<float>> = 8u;
  template <> constexpr uint32_t VecSize<Vec<double>> = 4u;
  template <> constexpr uint32_t VecSize<__m128> = 4u;

  constexpr bool UseFMAOptimization = true;

  template <std::floating_point F>
  static auto LoadA(const F *ptr) -> Vec<F>
  {
    if constexpr (f32<F>)
      { return _mm256_load_ps(ptr);  }
    else
      { return _mm256_load_pd(ptr);  }
  }

  template <std::floating_point F>
  static auto Load(const F *ptr) -> Vec<F>
  {
    if constexpr (f32<F>)
      { return _mm256_loadu_ps(ptr);  }
    else
      { return _mm256_loadu_pd(ptr);  }
  }

  static auto LoadHalf(const float *ptr) -> __m128
    { return _mm_loadu_ps(ptr); }

  static auto SplitHalves(__m256 v) -> std::pair<__m128, __m128>
    { return { _mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1) }; }

  static auto LoadHalves(const float *ptrA, size_t offset)
    { return _mm256_set_m128(_mm_loadu_ps(ptrA + offset), _mm_loadu_ps(ptrA)); }
  
  template <std::floating_point F>
  static auto Load4(const F *ptr) -> Vec<F>
  {
    if constexpr (f32<F>)
    {
      auto half = _mm_load_ps(ptr); // aligned load
      return _mm256_set_m128(half, half);
    }
    else
      { return _mm256_load_pd(ptr); }
  }

  static auto LoadAnglesRadix4FromCount4Half(const float *ptr, size_t angleIndex) -> std::tuple<__m128, __m128, __m128, __m128>
  { 
    return 
      { 
        LoadHalf(ptr + angleIndex),
        LoadHalf(ptr + 4 + angleIndex),
        LoadHalf(ptr + 8 + angleIndex),
        LoadHalf(ptr + 16 + angleIndex)
      };
  }

  template <std::floating_point F>
  static auto LoadDeinterleaved(const F *ptr) -> std::pair<Vec<F>, Vec<F>>
  {
    auto a = Load(ptr);
    auto b = Load(ptr + VecSize<Vec<F>>);

    if constexpr (f32<F>)
    {
       // (A a B b | C c D d) (E e F f | G g H h) -> (A a B b | E e F f)
      auto firstHalves = _mm256_permute2f128_ps(a, b, 0x20);

      // (A a B b | C c D d) (E e F f | G g H h) -> (C c D d | G g H h)
      auto secondHalves = _mm256_permute2f128_ps(a, b, 0x31);

      return 
        { 
          // (A a B b | E e F f) (C c D d | G g H h) -> (A B C D | E F G H)
          _mm256_shuffle_ps(firstHalves, secondHalves, _MM_SHUFFLE(2, 0, 2, 0)),

          // (A a B b | E e F f) (C c D d | G g H h) -> (a b c d | e f g h)
          _mm256_shuffle_ps(firstHalves, secondHalves, _MM_SHUFFLE(3, 1, 3, 1)),
        };
    }
    else
    {
      // (A a | B b) (C c | D d) -> (A a | C c)
      auto firstHalves = _mm256_permute2f128_pd(a, b, 0x20);

      // (A a | B b) (C c | D d) -> (B b | D d)
      auto secondHalves = _mm256_permute2f128_pd(a, b, 0x31);
      
      return
        {
          // (A a | C c) (B b | D d) -> (A B | C D)
          _mm256_shuffle_pd(firstHalves, secondHalves, 0b0'0'0'0),

          // (A a | C c) (B b | D d) -> (a b | c d)
          _mm256_shuffle_pd(firstHalves, secondHalves, 0b1'1'1'1),
        };
    }
  }

  static auto LoadDeinterleavedHalf(const float *ptr) -> std::pair<__m128, __m128>
  {
    auto t0 = _mm_loadu_ps(ptr);
    auto t1 = _mm_loadu_ps(ptr + 4u);
    return { _mm_shuffle_ps(t0, t1, _MM_SHUFFLE(2, 0, 2, 0)), _mm_shuffle_ps(t0, t1, _MM_SHUFFLE(3, 1, 3, 1)) };
  }


  static auto StoreHalves(float *ptr, size_t offset, __m256 v)
    { _mm256_storeu2_m128(ptr + offset, ptr, v); }
  
  template <std::floating_point F>
  static void Store4x4Transposed(F *ptrA, F *ptrB, F *ptrC, F *ptrD, Vec<F> a, Vec<F> b, Vec<F> c, Vec<F> d)
  {
    if constexpr (f32<F>)
    {
      // 4x4 transform on both halves of the AVX vector at once (this is the _MM_TRANSPOSE4 macro but with _mm256 instead)
      auto t0 = _mm256_shuffle_ps(a, b, 0x44);
      auto t2 = _mm256_shuffle_ps(a, b, 0xEE);
      auto t1 = _mm256_shuffle_ps(c, d, 0x44);
      auto t3 = _mm256_shuffle_ps(c, d, 0xEE);

      a = _mm256_shuffle_ps(t0, t1, 0x88);
      b = _mm256_shuffle_ps(t0, t1, 0xDD);
      c = _mm256_shuffle_ps(t2, t3, 0x88);
      d = _mm256_shuffle_ps(t2, t3, 0xDD);

      _mm256_storeu_ps(ptrA, a);
      _mm256_storeu_ps(ptrB, b);
      _mm256_storeu_ps(ptrC, c);
      _mm256_storeu_ps(ptrD, d);
    }
    else
    {
      // For f64 we have to transpose cross-lane, start by swapping all the 2x2s
      auto t0 = _mm256_unpacklo_pd(a, b);
      auto t1 = _mm256_unpackhi_pd(a, b);
      auto t2 = _mm256_unpacklo_pd(c, d);
      auto t3 = _mm256_unpackhi_pd(c, d);

      // Now swap the upper-right and lower-left 2x2 chunks
      a = _mm256_permute2f128_pd(t0, t2, 0x20);
      b = _mm256_permute2f128_pd(t1, t3, 0x20);
      c = _mm256_permute2f128_pd(t0, t2, 0x31);
      d = _mm256_permute2f128_pd(t1, t3, 0x31);

      _mm256_storeu_pd(ptrA, a);
      _mm256_storeu_pd(ptrB, b);
      _mm256_storeu_pd(ptrC, c);
      _mm256_storeu_pd(ptrD, d);
    }
  }

  static void Store4x4Transposed(float *ptrA, float *ptrB, float *ptrC, float *ptrD, __m128 a, __m128 b, __m128 c, __m128 d)
  {
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
      { _mm256_storeu_ps(ptr, a); }
    else
      { _mm256_storeu_pd(ptr, a); }
  }

  // Need a half-store for f32s
  static void Store(float *ptr, __m128 a)
  {
    _mm_storeu_ps(ptr, a);
  }

  template <std::floating_point F>
  static void StoreInterleaved(F *ptr, Vec<F> a, Vec<F> b)
  {
    if constexpr (f32<F>)
    {
      // (A B C D E F G H) (a b c d e f g h) -> (A a B b E e F f)
      auto loHalves = _mm256_unpacklo_ps(a, b);

      // (A B C D E F G H) (a b c d e f g h) -> (C c D d G g H h)
      auto hiHalves = _mm256_unpackhi_ps(a, b);

      // (A a B b E e F f) (C c D d G g H h) -> (A a B b C c D d)
      a = _mm256_permute2f128_ps(loHalves, hiHalves, 0x20);

      // (A a B b E e F f) (C c D d G g H h) -> (E e F f G g H h)
      b = _mm256_permute2f128_ps(loHalves, hiHalves, 0x31);

      _mm256_storeu_ps(ptr, a);
      _mm256_storeu_ps(ptr + 8, b);
    }
    else
    {
      // (A B C D) (a b c d) -> (A a C c)
      auto loHalves = _mm256_unpacklo_pd(a, b);

      // (A B C D) (a b c d) -> (B b D d)
      auto hiHalves = _mm256_unpackhi_pd(a, b);

      // (A a C c) (B b D d) -> (A a B b)
      a = _mm256_permute2f128_pd(loHalves, hiHalves, 0x20);

      // (A a C c) (B b D d) -> (C c D d)
      b = _mm256_permute2f128_pd(loHalves, hiHalves, 0x31);

      _mm256_storeu_pd(ptr, a);
      _mm256_storeu_pd(ptr + 4, b);
    }
  }

  static void StoreInterleaved(float *ptr, __m128 a, __m128 b)
  {
    auto aa = _mm_unpacklo_ps(a, b);
    auto bb = _mm_unpackhi_ps(a, b);
    _mm_storeu_ps(ptr, aa);
    _mm_storeu_ps(ptr + 4, bb);
  }

  template <typename V>
  static auto Set1(std::conditional_t<f32<V>, float, double> f) -> V
  {
    if constexpr (f32<V>)
    { 
      if constexpr (std::same_as<V, __m128>)
        { return _mm_set1_ps(f); }
      else
        { return _mm256_set1_ps(f); }
    }
    else
      { return _mm256_set1_pd(f); }
  }

  template <simd_vec V>
  static V Add(V a, V b)
  {
    if constexpr (f32<V>)
      { return _mm256_add_ps(a, b); }
    else
      { return _mm256_add_pd(a, b); }
  }

  static __m128 Add(__m128 a, __m128 b)
    { return _mm_add_ps(a, b); }

  template <simd_vec V>
  static V Sub(V a, V b)
  {
    if constexpr (f32<V>)
      { return _mm256_sub_ps(a, b); }
    else
      { return _mm256_sub_pd(a, b); }
  }

  static __m128 Sub(__m128 a, __m128 b)
    { return _mm_sub_ps(a, b); }

  template <simd_vec V>
  static V Mul(V a, V b)
  {
    if constexpr (f32<V>)
      { return _mm256_mul_ps(a, b); }
    else
      { return _mm256_mul_pd(a, b); }
  }

  
  template <simd_vec V>
  static V FMAdd(V a, V b, V c)
  {
    if constexpr (f32<V>)
      { return _mm256_fmadd_ps(a, b, c); }
    else
      { return _mm256_fmadd_pd(a, b, c); }
  }

  static __m128 FMAdd(__m128 a, __m128 b, __m128 c)
    { return _mm_fmadd_ps(a, b, c); }

  template <simd_vec V>
  static V FMSub(V a, V b, V c)
  {
    if constexpr (f32<V>)
      { return _mm256_fmsub_ps(a, b, c); }
    else
      { return _mm256_fmsub_pd(a, b, c); }
  }

  static __m128 FMSub(__m128 a, __m128 b, __m128 c)
    { return _mm_fmsub_ps(a, b, c); }

  template <simd_vec V>
  static V FNMAdd(V a, V b, V c) // -(a * b) + c
  {
    if constexpr (f32<V>)
      { return _mm256_fnmadd_ps(a, b, c); }
    else
      { return _mm256_fnmadd_pd(a, b, c); }
  }

  static __m128 FNMAdd(__m128 a, __m128 b, __m128 c)
    { return _mm_fnmadd_ps(a, b, c); }

  template <std::floating_point F, std::same_as<Vec<F>> ... Vs>
  static void Scale(F scale, Vs &...vs)
  {
    if constexpr (f32<F>)
    {
      auto s = _mm256_set1_ps(scale);
      ((vs = _mm256_mul_ps(s, vs)), ...);
    }
    else
    {
      auto s = _mm256_set1_pd(scale);
      ((vs = _mm256_mul_pd(s, vs)), ...);
    }
  }
  
  template <std::floating_point F, std::same_as<__m128> ... Vs>
  static void Scale(F scale, Vs &...vs)
  {
    auto s = _mm_set1_ps(scale);
    ((vs = _mm_mul_ps(s, vs)), ...);
  }
  
  template <typename V>
  static V MAdd(V a, V b, V c)
    { return FMAdd(a, b, c); }

  template <typename V>
  static V NMAdd(V a, V b, V c) // -(a * b) + c
    { return FNMAdd(a, b, c); }

  #include "fft_20xx_run.h"
}


extern "C"
{
  using namespace FFT20XX;

  #define COMMON_ASSERTS() \
    assert(table != nullptr); \
    assert(input != nullptr); \
    assert(output != nullptr); \
    assert(IsPowerOfTwo(fftLength)); \
    assert(t->MaxFFTLength() >= fftLength)

  void FFTComplex32Interleaved_FMA(const FFTTable32 *table, const float *input, float *output, size_t fftLength)
  {
    auto t = reinterpret_cast<const FFTTableImpl<float> *>(table);
    COMMON_ASSERTS();
    FMA::DITRadix4FFT<Dir::Forward>(
      t->Data(),
      reinterpret_cast<const Complex<float> *>(input),
      reinterpret_cast<Complex<float> *>(output),
      fftLength);
  }

  void FFTComplex64Interleaved_FMA(const FFTTable64 *table, const double *input, double *output, size_t fftLength)
  {
    auto t = reinterpret_cast<const FFTTableImpl<double> *>(table);
    COMMON_ASSERTS();
    FMA::DITRadix4FFT<Dir::Forward>(
      t->Data(),
      reinterpret_cast<const Complex<double> *>(input),
      reinterpret_cast<Complex<double> *>(output),
      fftLength);
  }


  void IFFTComplex32Interleaved_FMA(const FFTTable32 *table, const float *input, float *output, size_t fftLength)
  {
    auto t = reinterpret_cast<const FFTTableImpl<float> *>(table);
    COMMON_ASSERTS();
    FMA::DITRadix4FFT<Dir::Inverse>(
      t->Data(),
      reinterpret_cast<const Complex<float> *>(input),
      reinterpret_cast<Complex<float> *>(output),
      fftLength);
  }

  void IFFTComplex64Interleaved_FMA(const FFTTable64 *table, const double *input, double *output, size_t fftLength)
  {
    auto t = reinterpret_cast<const FFTTableImpl<double> *>(table);
    COMMON_ASSERTS();
    FMA::DITRadix4FFT<Dir::Inverse>(
      t->Data(),
      reinterpret_cast<const Complex<double> *>(input),
      reinterpret_cast<Complex<double> *>(output),
      fftLength);
  }
}