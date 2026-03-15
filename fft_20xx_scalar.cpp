#include "fft_20xx_common.h"
#include "fft_20xx_table.h"


namespace FFT20XX::Scalar
{
  template <typename T, uint32_t N>
  struct ScalarVec
  {
    using Type = T;
    static constexpr auto Length = N;

    T data[N];

    T &operator[](std::integral auto i)
      { return data[i]; }

    const T &operator[](std::integral auto i) const
      { return data[i]; }
  };

  constexpr uint32_t StandardSize = 8u;

  template <typename T> concept f32 = std::same_as<float, T>
    || std::same_as<ScalarVec<float, StandardSize>, T>
    || std::same_as<ScalarVec<float, StandardSize / 2>, T>;

  template <typename T> using Vec = ScalarVec<T, StandardSize>;

  template <typename> constexpr uint32_t VecSize = StandardSize;

  constexpr bool IsScalar = true;
  constexpr bool UseFMAOptimization = false;

  template <std::floating_point F>
  static auto LoadA(const F *ptr) -> Vec<F>
  {
    Vec<F> r;
    for (auto i = 0u; i < StandardSize; i++)
      { r[i] = ptr[i]; }
    return r;
  }
  template <std::floating_point F>
  static auto Load(const F *ptr) -> Vec<F>
    { return LoadA(ptr); }

  template <std::floating_point F>
  static auto LoadHalf(const F *ptr)
  {
    ScalarVec<F, StandardSize / 2> r;
    for (auto i = 0u; i < StandardSize / 2; i++)
      { r[i] = ptr[i]; }
    return r;
  }

  template <typename F>
  static auto SplitHalves(Vec<F> v)
  {
    Pair<ScalarVec<F, StandardSize / 2>, ScalarVec<F, StandardSize / 2>> r;
    for (auto i = 0u; i < StandardSize / 2; i++)
    {
      r.first[i] = v[i];
      r.second[i] = v[i + StandardSize / 2];
    }
    return r;
  }

  template <typename F>
  static auto LoadHalves(const F *ptr, size_t offset)
  {
    Vec<F> r;
    for (auto i = 0u; i < StandardSize / 2; i++)
    {
      r[i] = ptr[i];
      r[i + StandardSize / 2] = ptr[i + offset];
    }
    return r;
  }


  template <std::floating_point F>
  static auto Load4(const F *ptr) -> Vec<F>
  {
    Vec<F> r;
    for (auto i = 0u; i < StandardSize; i++)
      { r[i] = ptr[i & 3]; }
    return r;
  }

  template <std::floating_point F>
  static auto LoadDeinterleaved(const F *ptr) -> Pair<Vec<F>, Vec<F>>
  {
    Pair<Vec<F>, Vec<F>> r;
    for (auto i = 0u; i < StandardSize; i++)
    {
      r.first[i] = ptr[2 * i + 0];
      r.second[i] = ptr[2 * i + 1];
    }
    return r;
  }

  template <std::floating_point F>
  static auto LoadDeinterleavedHalf(const F *ptr)
  {
    Pair<ScalarVec<F, StandardSize / 2>, ScalarVec<F, StandardSize / 2>> r;
    for (auto i = 0u; i < StandardSize / 2; i++)
    {
      r.first[i] = ptr[2 * i + 0];
      r.second[i] = ptr[2 * i + 1];
    }
    return r;
  }


  template <std::floating_point F>
  static auto StoreHalves(F *ptr, size_t offset, Vec<F> v)
  {
    for (auto i = 0u; i < StandardSize / 2; i++)
    {
      ptr[i] = v[i];
      ptr[i + offset] = v[i + StandardSize / 2];
    }
  }

  template <std::floating_point F, typename V>
  static void Store(F *ptr, V a)
  {
    for (auto i = 0u; i < V::Length; i++)
      { ptr[i] = a[i]; }
  }

  template <std::floating_point F, typename V>
  static void Store4x4Transposed(F *ptrA, F *ptrB, F *ptrC, F *ptrD, V a, V b, V c, V d)
  {
    for (auto x = 0u; x < V::Length; x += 4u)
    {
      std::swap(a[x + 1], b[x + 0]);
      std::swap(a[x + 2], c[x + 0]);
      std::swap(a[x + 3], d[x + 0]);
      std::swap(b[x + 2], c[x + 1]);
      std::swap(b[x + 3], d[x + 1]);
      std::swap(c[x + 3], d[x + 2]);
    }

    Store(ptrA, a);
    Store(ptrB, b);
    Store(ptrC, c);
    Store(ptrD, d);
  }

  template <std::floating_point F, typename V>
  static void StoreInterleaved(F *ptr, V a, V b)
  {
    for (auto i = 0u; i < V::Length; i++)
    {
      ptr[2 * i + 0] = a[i];
      ptr[2 * i + 1] = b[i];
    }
  }

  template <typename V>
  static V Add(V a, V b)
  {
    V r;
    for (auto i = 0u; i < V::Length; i++)
      { r[i] = a[i] + b[i]; }
    return r;
  }

  template <typename V>
  static V Sub(V a, V b)
  {
    V r;
    for (auto i = 0u; i < V::Length; i++)
      { r[i] = a[i] - b[i]; }
    return r;
  }

  template <typename V>
  static V Mul(V a, V b)
  {
    V r;
    for (auto i = 0u; i < V::Length; i++)
      { r[i] = a[i] * b[i]; }
    return r;
  }

  template <typename V, std::floating_point F>
  static V Mul(V a, F b)
  {
    for (auto i = 0u; i < V::Length; i++)
      { a[i] *= b; };
    return a;
  }

  template <std::floating_point F, typename ... Vs>
  static void Scale(F scale, Vs &...vs)
    { ((vs = Mul(vs, scale)), ...); }

  template <typename V>
  static V MAdd(V a, V b, V c)
    { return Add(Mul(a, b), c); }

  template <typename V>
  static V NMAdd(V a, V b, V c)
    { return Sub(c, Mul(a, b)); }

  #include "fft_20xx_run.h"

}

using namespace FFT20XX;

using namespace fft_20xx::c;
extern "C"
{
  void fft_cplx_interleaved_f32_scalar(const fft_table_f32 *table, const float *input, float *output, size_t fftLength)
  {
    [[maybe_unused]] auto t = reinterpret_cast<const FFTTableImpl<float> *>(table);
    COMMON_ASSERTS();
    Scalar::RunComplexFFT<Dir::Forward>(t->Data(), input, output, fftLength);
  }


  void fft_cplx_interleaved_f64_scalar(const fft_table_f64 *table, const double *input, double *output, size_t fftLength)
  {
    [[maybe_unused]] auto t = reinterpret_cast<const FFTTableImpl<double> *>(table);
    COMMON_ASSERTS();
    Scalar::RunComplexFFT<Dir::Forward>(t->Data(), input, output, fftLength);
  }


  void ifft_cplx_interleaved_f32_scalar(const fft_table_f32 *table, const float *input, float *output, size_t fftLength)
  {
    [[maybe_unused]] auto t = reinterpret_cast<const FFTTableImpl<float> *>(table);
    COMMON_ASSERTS();
    Scalar::RunComplexFFT<Dir::Inverse>(t->Data(), input, output, fftLength);
  }


  void ifft_cplx_interleaved_f64_scalar(const fft_table_f64 *table, const double *input, double *output, size_t fftLength)
  {
    [[maybe_unused]] auto t = reinterpret_cast<const FFTTableImpl<double> *>(table);
    COMMON_ASSERTS();
    Scalar::RunComplexFFT<Dir::Inverse>(t->Data(), input, output, fftLength);
  }

  void fft_real_f32_scalar(const fft_table_f32 *table, const float *input, float *output, size_t fftLength)
  {
    auto t = reinterpret_cast<const FFTTableImpl<float> *>(table);
    COMMON_ASSERTS();
    Scalar::RunRealFwdFFT(t->Data(), input, output, fftLength);
  }

  void fft_real_f64_scalar(const fft_table_f64 *table, const double *input, double *output, size_t fftLength)
  {
    auto t = reinterpret_cast<const FFTTableImpl<double> *>(table);
    COMMON_ASSERTS();
    Scalar::RunRealFwdFFT(t->Data(), input, output, fftLength);
  }

  void ifft_real_f32_scalar(const fft_table_f32 *table, const float *input, float *output, size_t fftLength)
  {
    auto t = reinterpret_cast<const FFTTableImpl<float> *>(table);
    COMMON_ASSERTS();
    Scalar::RunRealInvFFT(t->Data(), input, output, fftLength);
  }

  void ifft_real_f64_scalar(const fft_table_f64 *table, const double *input, double *output, size_t fftLength)
  {
    auto t = reinterpret_cast<const FFTTableImpl<double> *>(table);
    COMMON_ASSERTS();
    Scalar::RunRealInvFFT(t->Data(), input, output, fftLength);
  }
}