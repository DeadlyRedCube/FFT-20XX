#ifdef _MSC_VER
  #pragma warning (disable: 4710) // function not inlined
  #pragma warning (disable: 4711) // function <x> selected for automatic inline expansion
  #pragma warning (disable: 5045) // Compiler will insert Spectre mitigation for memory load
#endif

#include "fft_20xx/fft_20xx.h"

#include <cassert>
#include <chrono>
#include <cmath>
#include <complex.h>
#include <complex>
#include <cstdlib>
#include <random>
#include <utility>
#include <vector>

#if defined(_M_X64) || defined(__x86_64__) || defined(_M_IX86) || defined(__i386__)
  #include <immintrin.h>
#endif

template <typename T>
T *AlignedAlloc(std::size_t count, std::size_t elementAlignment)
{
  #if defined (_MSC_VER)
    return static_cast<T *>(_aligned_malloc(count * sizeof(T),  elementAlignment * sizeof(T)));
  #else
    return static_cast<T *>(std::aligned_alloc(elementAlignment * sizeof(T), count * sizeof(T)));
  #endif
}



template <typename T>
void AlignedFree(T *mem)
{
  #if defined (_MSC_VER)
    _aligned_free(mem);
  #else
    std::free(mem);
  #endif
}



enum class BufferType
{
  InPlace,
  Separate,
};



#if defined(_M_X64) || defined(__x86_64__) || defined(_M_IX86) || defined(__i386__)
  void FlushSubnormals()
  {
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  }
#elif defined(_M_ARM64) || defined(__aarch64__) || defined(_M_ARM) || defined (__arm__)
  void FlushSubnormals()
  {
    #if defined(_MSC_VER)
      std::uint32_t current;
      _controlfp_s(&current, _DN_FLUSH, _MCW_DN);
    #elif defined(_M_ARM64) || defined(__aarch64__)
      // From:
      //  https://developer.arm.com/documentation/ddi0595/2021-06/AArch64-Registers/FPCR--Floating-point-Control-Register
      std::uint64_t current;
      __asm__ ("mrs %0, fpcr" : "=r"(current) : ); // Read the current control values
      current |= (1 << 24); // Add the flush-to-zero flag
      __asm__ ("msr fpcr, %0" : : "r"(current)); // Write the updated value
    #else
      // From:
      //  https://developer.arm.com/documentation/ddi0595/2021-06/AArch32-Registers/FPSCR--Floating-Point-Status-and-Control-Register
      std::uint32_t current;
      __asm__ ("vmrs %0, fpscr" : "=r"(current) : ); // Read the current control values
      current |= (1 << 24); // Add the flush-to-zero flag
      __asm__ ("vmsr fpscr, %0" : : "r"(current)); // Write the updated value
    #endif
  }
#else
  #error Unknown CPU type.
#endif

#if defined(_M_X64) || defined(__x86_64__) || defined(_M_IX86) || defined(__i386__)
  #define CPU_X86_ANY 1
#else
  #define CPU_X86_ANY 0
#endif

#if defined(_M_ARM64) || defined(__aarch64__) || defined(_M_ARM) || defined (__arm__)
  #define CPU_ARM_ANY 1
#else
  #define CPU_ARM_ANY 0
#endif



extern "C"
{
  using namespace fft_20xx::c;

  void fft_cplx_f32_scalar(const fft_table_f32 *table, const std::complex<float> *input, std::complex<float> *output, size_t length);
  void fft_cplx_f64_scalar(const fft_table_f64 *table, const std::complex<double> *input, std::complex<double> *output, size_t length);
  void ifft_cplx_f32_scalar(const fft_table_f32 *table, const std::complex<float> *input, std::complex<float> *output, size_t length);
  void ifft_cplx_f64_scalar(const fft_table_f64 *table, const std::complex<double> *input, std::complex<double> *output, size_t length);
  void fft_real_f32_scalar(const fft_table_f32 *table, const float *input, float *output, size_t length);
  void fft_real_f64_scalar(const fft_table_f64 *table, const double *input, double *output, size_t length);
  void ifft_real_f32_scalar(const fft_table_f32 *table, const float *input, float *output, size_t length);
  void ifft_real_f64_scalar(const fft_table_f64 *table, const double *input, double *output, size_t length);

  #if CPU_X86_ANY
    void fft_cplx_f32_sse2(const fft_table_f32 *table, const std::complex<float> *input, std::complex<float> *output, size_t length);
    void fft_cplx_f64_sse2(const fft_table_f64 *table, const std::complex<double> *input, std::complex<double> *output, size_t length);
    void ifft_cplx_f32_sse2(const fft_table_f32 *table, const std::complex<float> *input, std::complex<float> *output, size_t length);
    void ifft_cplx_f64_sse2(const fft_table_f64 *table, const std::complex<double> *input, std::complex<double> *output, size_t length);
    void fft_real_f32_sse2(const fft_table_f32 *table, const float *input, float *output, size_t length);
    void fft_real_f64_sse2(const fft_table_f64 *table, const double *input, double *output, size_t length);
    void ifft_real_f32_sse2(const fft_table_f32 *table, const float *input, float *output, size_t length);
    void ifft_real_f64_sse2(const fft_table_f64 *table, const double *input, double *output, size_t length);

    void fft_cplx_f32_avx(const fft_table_f32 *table, const std::complex<float> *input, std::complex<float> *output, size_t length);
    void fft_cplx_f64_avx(const fft_table_f64 *table, const std::complex<double> *input, std::complex<double> *output, size_t length);
    void ifft_cplx_f32_avx(const fft_table_f32 *table, const std::complex<float> *input, std::complex<float> *output, size_t length);
    void ifft_cplx_f64_avx(const fft_table_f64 *table, const std::complex<double> *input, std::complex<double> *output, size_t length);
    void fft_real_f32_avx(const fft_table_f32 *table, const float *input, float *output, size_t length);
    void fft_real_f64_avx(const fft_table_f64 *table, const double *input, double *output, size_t length);
    void ifft_real_f32_avx(const fft_table_f32 *table, const float *input, float *output, size_t length);
    void ifft_real_f64_avx(const fft_table_f64 *table, const double *input, double *output, size_t length);

    void fft_cplx_f32_fma(const fft_table_f32 *table, const std::complex<float> *input, std::complex<float> *output, size_t length);
    void fft_cplx_f64_fma(const fft_table_f64 *table, const std::complex<double> *input, std::complex<double> *output, size_t length);
    void ifft_cplx_f32_fma(const fft_table_f32 *table, const std::complex<float> *input, std::complex<float> *output, size_t length);
    void ifft_cplx_f64_fma(const fft_table_f64 *table, const std::complex<double> *input, std::complex<double> *output, size_t length);
    void fft_real_f32_fma(const fft_table_f32 *table, const float *input, float *output, size_t length);
    void fft_real_f64_fma(const fft_table_f64 *table, const double *input, double *output, size_t length);
    void ifft_real_f32_fma(const fft_table_f32 *table, const float *input, float *output, size_t length);
    void ifft_real_f64_fma(const fft_table_f64 *table, const double *input, double *output, size_t length);
  #elif CPU_ARM_ANY
    void fft_cplx_f32_neon(const fft_table_f32 *table, const std::complex<float> *input, std::complex<float> *output, size_t length);
    void fft_cplx_f64_neon(const fft_table_f64 *table, const std::complex<double> *input, std::complex<double> *output, size_t length);
    void ifft_cplx_f32_neon(const fft_table_f32 *table, const std::complex<float> *input, std::complex<float> *output, size_t length);
    void ifft_cplx_f64_neon(const fft_table_f64 *table, const std::complex<double> *input, std::complex<double> *output, size_t length);
    void fft_real_f32_neon(const fft_table_f32 *table, const float *input, float *output, size_t length);
    void fft_real_f64_neon(const fft_table_f64 *table, const double *input, double *output, size_t length);
    void ifft_real_f32_neon(const fft_table_f32 *table, const float *input, float *output, size_t length);
    void ifft_real_f64_neon(const fft_table_f64 *table, const double *input, double *output, size_t length);
  #endif
}



template <std::floating_point F>
constexpr auto Rotation(F angle) -> std::complex<F>
  { return { std::cos(angle), std::sin(angle) }; }



template <std::floating_point F>
constexpr auto Rotation(std::size_t n, std::size_t N) -> std::complex<F>
{
  constexpr auto NegTwoPi = F(-6.283185307179586);
  auto angle = NegTwoPi * F(n) / F(N);
  return Rotation(angle);
}



auto BitRev(std::integral auto v)
{
  if constexpr (sizeof(v) == sizeof(uint64_t))
  {
    v = ((v >>  1) & 0x55555555'55555555) | ((v & 0x55555555'55555555) << 1);
    v = ((v >>  2) & 0x33333333'33333333) | ((v & 0x33333333'33333333) << 2);
    v = ((v >>  4) & 0x0f0f0f0f'0f0f0f0f) | ((v & 0x0f0f0f0f'0f0f0f0f) << 4);
    v = ((v >>  8) & 0x00ff00ff'00ff00ff) | ((v & 0x00ff00ff'00ff00ff) << 8);
    v = ((v >> 16) & 0x0000ffff'0000ffff) | ((v & 0x0000ffff'0000ffff) << 16);
    v = (v >> 32) | (v << 32);
  }
  else
  {
    static_assert(sizeof(v) == sizeof(uint32_t));
    v = ((v >>  1) & 0x55555555) | ((v & 0x55555555) << 1);
    v = ((v >>  2) & 0x33333333) | ((v & 0x33333333) << 2);
    v = ((v >>  4) & 0x0f0f0f0f) | ((v & 0x0f0f0f0f) << 4);
    v = ((v >>  8) & 0x00ff00ff) | ((v & 0x00ff00ff) << 8);
    v = (v >> 16) | (v << 16);
  }

  return v;
}



template <std::floating_point F>
void BasicDITRadix2FFT(const std::complex<F> *input, std::complex<F> *output, std::size_t fftLength)
{
  assert(input != output);
  for (auto i = size_t(0); i < fftLength; i++)
  {
    const auto fftLengthBitCount = std::uint32_t(std::numeric_limits<decltype(fftLength)>::digits - 1 - std::countl_zero(fftLength));
    const auto shift = std::uint32_t(std::numeric_limits<size_t>::digits - fftLengthBitCount);

    output[i] = input[BitRev(i) >> shift];
  }

  // Start with a bunch of 2-element FFTs
  for (auto innerLen = 2u; innerLen <= fftLength; innerLen *= 2)
  {
    for (auto fftIndex = 0u; fftIndex < fftLength / innerLen; fftIndex++)
    {
      for (auto butterflyIndex = 0u; butterflyIndex < innerLen / 2; butterflyIndex++)
      {
        auto i0 = fftIndex * innerLen + butterflyIndex;
        auto i1 = i0 + innerLen / 2;

        auto even = output[i0];
        auto odd = output[i1] * Rotation<F>(butterflyIndex, innerLen);

        output[i0] = even + odd;
        output[i1] = even - odd;
      }
    }
  }
}



template <std::floating_point F>
void BasicDITRadix2IFFT(const std::complex<F> *input, std::complex<F> *output, std::size_t fftLength)
{
  assert(input != output);
  for (auto i = size_t(0); i < fftLength; i++)
  {
    const auto fftLengthBitCount = std::uint32_t(std::numeric_limits<decltype(fftLength)>::digits - 1 - std::countl_zero(fftLength));
    const auto shift = std::uint32_t(std::numeric_limits<size_t>::digits - fftLengthBitCount);

    output[i] = input[BitRev(i) >> shift];
  }

  F scale = F(1) / F(fftLength);

  for (auto i = 0u; i < fftLength; i++)
  {
    output[i] *= scale;
  }

  // Start with a bunch of 2-element FFTs
  for (auto innerLen = 2u; innerLen <= fftLength; innerLen *= 2)
  {
    for (auto fftIndex = 0u; fftIndex < fftLength / innerLen; fftIndex++)
    {
      for (auto butterflyIndex = 0u; butterflyIndex < innerLen / 2; butterflyIndex++)
      {
        auto i0 = fftIndex * innerLen + butterflyIndex;
        auto i1 = i0 + innerLen / 2;

        auto even = output[i0];
        auto odd = output[i1] * std::conj(Rotation<F>(butterflyIndex, innerLen));

        output[i0] = even + odd;
        output[i1] = even - odd;
      }
    }
  }
}



enum class TestType
{
  Complex,
  Real,
};

template <TestType T, BufferType B, typename F>
void SmokeTest(const char *name, auto testFwd, auto testInv)
{
  printf(
    "Testing %s, %s, %s, %s...",
    name,
    T == TestType::Complex ? "complex" : "real",
    B == BufferType::InPlace ? "In-Place" : "Separate",
    std::same_as<F, float> ? "f32" : "f64");

  constexpr auto MaxFFTLength = 32768u;
  bool hadError = false;

  for (auto size = 1u; size <= MaxFFTLength; size *= 2)
  {
    auto table = fft_20xx::fft_table<F>{size};

    using ElementType = std::conditional_t<T == TestType::Complex, std::complex<F>, F>;

    std::vector<std::complex<double>> inputD(size);
    std::vector<std::complex<double>> outputGood(size);
    std::vector<std::complex<double>> ioutputGood(size);

    std::vector<ElementType> input(size);
    std::vector<ElementType> inputIFFT(size);
    std::vector<ElementType> outputTest(size);
    std::vector<ElementType> ioutputTest(size);

    auto seq = std::seed_seq{1, 2, 3, 4, 5};
    std::mt19937_64 rnd{seq};
    auto gen = [&] { return std::uniform_real_distribution<F>{F(-0.5), F(0.5)}(rnd); };

    for (auto i = 0u; i < input.size(); i++)
    {
      if constexpr (T == TestType::Complex)
        { input[i] = { gen(), gen() }; }
      else
        { input[i] = gen(); }
      inputD[i] = input[i];
    }


    BasicDITRadix2FFT(inputD.data(), outputGood.data(), size);
    BasicDITRadix2IFFT(outputGood.data(), ioutputGood.data(), size);

    if constexpr (T == TestType::Complex)
    {
      for (auto i = 0u; i < size; i++)
        { inputIFFT[i] = { F(outputGood[i].real()), F(outputGood[i].imag()) }; }
    }
    else
    {
      inputIFFT[0] = F(outputGood[0].real());
      if (size > 1)
      {
        inputIFFT[1] = F(outputGood[size / 2].real());
        for (auto i = 2u; i < input.size(); i += 2)
        {
          inputIFFT[i] = F(outputGood[i / 2].real());
          inputIFFT[i + 1] = F(outputGood[i / 2].imag());
        }
      }
    }

    if constexpr (B == BufferType::InPlace)
    {
      std::ranges::copy(input, outputTest.begin());
      std::ranges::copy(inputIFFT, ioutputTest.begin());
      testFwd(table, outputTest.data(), outputTest.data(), size);
      testInv(table, ioutputTest.data(), ioutputTest.data(), size);
    }
    else
    {
      testFwd(table, input.data(), outputTest.data(), size);
      testInv(table, inputIFFT.data(), ioutputTest.data(), size);
    }

    constexpr auto Threshold = std::same_as<F, float> ? F(3e-5) : F(2e-13);

    #ifdef _MSC_VER
      #pragma warning (push)
      #pragma warning (disable : 4820) // padding added after data member
    #endif

    struct ErrorInfo
    {
      double amount;
      const char *direction;
      const char *component;
      size_t index;
      double good;
      double test;
    };

    #ifdef _MSC_VER
      #pragma warning (pop)
    #endif
    std::optional<ErrorInfo> errorInfo;

    auto Error =
      [](double a, double b) -> double
      {
        auto max = std::max(std::abs(a), std::abs(b));
        int maxExp;
        std::frexp(max, &maxExp);
        auto scale = std::ldexp(F(1), maxExp);

        auto err = std::abs(b - a);
        if (scale > 1.0)
          { err /= scale; }
        return err;
      };

    for (auto i = 0u; i < size; i++)
    {
      if constexpr (T == TestType::Complex)
      {
        if (auto err = Error(outputGood[i].real(), outputTest[i].real()); err > Threshold)
        {
          errorInfo = { err, "forward", " r", i, outputGood[i].real(), outputTest[i].real() };
          break;
        }
        if (auto err = Error(outputGood[i].imag(), outputTest[i].imag()); err > Threshold)
        {
          errorInfo = { err, "forward", " i", i, outputGood[i].imag(), outputTest[i].imag() };
          break;
        }

        if (auto err = Error(ioutputGood[i].real(), ioutputTest[i].real()); err > Threshold)
        {
          errorInfo = { err, "inverse", " r", i, ioutputGood[i].real(), ioutputTest[i].real() };
          break;
        }

        if (auto err = Error(ioutputGood[i].imag(), ioutputTest[i].imag()); err > Threshold)
        {
          errorInfo = { err, "inverse", " i", i, ioutputGood[i].imag(), ioutputTest[i].imag() };
          break;
        }
      }
      else
      {
        if ((i & 1) == 0)
        {
          if (auto err = Error(outputGood[i / 2].real(), outputTest[i]); err > Threshold)
          {
            errorInfo = { err, "forward", (i == 0) ? " dc" : " r", i, outputGood[i / 2].real(), outputTest[i] };
            break;
          }
        }
        else if (i == 1)
        {
          if (auto err = Error(outputGood[size / 2].real(), outputTest[i]); err > Threshold)
          {
            errorInfo = { err, "forward", " nyquist", i, outputGood[size / 2].real(), outputTest[i] };
            break;
          }
        }
        else
        {
          if (auto err = Error(outputGood[i / 2].imag(), outputTest[i]); err > Threshold)
          {
            errorInfo = { err, "forward", " i", i, outputGood[i / 2].imag(), outputTest[i] };
            break;
          }
        }

        if (auto err = Error(ioutputGood[i].real(), ioutputTest[i]); err > Threshold)
        {
          errorInfo = { err, "inverse", "", i, ioutputGood[i].real(), ioutputTest[i] };
          break;
        }
      }
    }

    if (errorInfo.has_value())
    {
      hadError = true;
      fprintf(
        stderr,
        "\n  Error: %s (length %u), at %u%s: %.20f != %.20f (err: %e)",
        errorInfo->direction,
        size,
        uint32_t(errorInfo->index),
        errorInfo->component,
        errorInfo->good,
        errorInfo->test,
        errorInfo->amount);
    }
  }

  if (hadError)
    { printf("\n\n"); }
  else
    { printf("Done\n"); }
}



int main()
{
  FlushSubnormals();

  using enum BufferType;
  using enum TestType;
  auto doTests =
    []<BufferType B>()
    {
      SmokeTest<Complex, B, float>("Auto", [&](auto &&...args) { fft_20xx::fft_cplx(args...); }, [&](auto &&...args) { fft_20xx::ifft_cplx(args...); });
      SmokeTest<Complex, B, double>("Auto", [&](auto &&...args) { fft_20xx::fft_cplx(args...); }, [&](auto &&...args) { fft_20xx::ifft_cplx(args...); });

      SmokeTest<Complex, B, float>("Scalar", [&](auto &&...args) { fft_cplx_f32_scalar(args...); }, [&](auto &&...args) { ifft_cplx_f32_scalar(args...); });
      SmokeTest<Complex, B, double>("Scalar", [&](auto &&...args) { fft_cplx_f64_scalar(args...); }, [&](auto &&...args) { ifft_cplx_f64_scalar(args...); });

      #if CPU_X86_ANY
        SmokeTest<Complex, B, float>("SSE2", [&](auto &&...args) { fft_cplx_f32_sse2(args...); }, [&](auto &&...args) { ifft_cplx_f32_sse2(args...); });
        SmokeTest<Complex, B, double>("SSE2", [&](auto &&...args) { fft_cplx_f64_sse2(args...); }, [&](auto &&...args) { ifft_cplx_f64_sse2(args...); });

        SmokeTest<Complex, B, float>("AVX", [&](auto &&...args) { fft_cplx_f32_avx(args...); }, [&](auto &&...args) { ifft_cplx_f32_avx(args...); });
        SmokeTest<Complex, B, double>("AVX", [&](auto &&...args) { fft_cplx_f64_avx(args...); }, [&](auto &&...args) { ifft_cplx_f64_avx(args...); });

        SmokeTest<Complex, B, float>("AVX w/FMA", [&](auto &&...args) { fft_cplx_f32_fma(args...); }, [&](auto &&...args) { ifft_cplx_f32_fma(args...); });
        SmokeTest<Complex, B, double>("AVX w/FMA", [&](auto &&...args) { fft_cplx_f64_fma(args...); }, [&](auto &&...args) { ifft_cplx_f64_fma(args...); });
      #elif CPU_ARM_ANY
        SmokeTest<Complex, B, float>("NEON", [&](auto &&...args) { fft_cplx_f32_neon(args...); }, [&](auto &&...args) { ifft_cplx_f32_neon(args...); });
        SmokeTest<Complex, B, double>("NEON", [&](auto &&...args) { fft_cplx_f64_neon(args...); }, [&](auto &&...args) { ifft_cplx_f64_neon(args...); });
      #endif

      SmokeTest<Real, B, float>("Auto", [&](auto &&...args) { fft_20xx::fft_real(args...); }, [&](auto &&...args) { fft_20xx::ifft_real(args...); });
      SmokeTest<Real, B, double>("Auto", [&](auto &&...args) { fft_20xx::fft_real(args...); }, [&](auto &&...args) { fft_20xx::ifft_real(args...); });

      SmokeTest<Real, B, float>("Scalar", [&](auto &&...args) { fft_real_f32_scalar(args...); }, [&](auto &&...args) { ifft_real_f32_scalar(args...); });
      SmokeTest<Real, B, double>("Scalar", [&](auto &&...args) { fft_real_f64_scalar(args...); }, [&](auto &&...args) { ifft_real_f64_scalar(args...); });
      #if CPU_X86_ANY
        SmokeTest<Real, B, float>("SSE2", [&](auto &&...args) { fft_real_f32_sse2(args...); }, [&](auto &&...args) { ifft_real_f32_sse2(args...); });
        SmokeTest<Real, B, double>("SSE2", [&](auto &&...args) { fft_real_f64_sse2(args...); }, [&](auto &&...args) { ifft_real_f64_sse2(args...); });

        SmokeTest<Real, B, float>("AVX", [&](auto &&...args) { fft_real_f32_avx(args...); }, [&](auto &&...args) { ifft_real_f32_avx(args...); });
        SmokeTest<Real, B, double>("AVX", [&](auto &&...args) { fft_real_f64_avx(args...); }, [&](auto &&...args) { ifft_real_f64_avx(args...); });

        SmokeTest<Real, B, float>("AVX w/FMA", [&](auto &&...args) { fft_real_f32_fma(args...); }, [&](auto &&...args) { ifft_real_f32_fma(args...); });
        SmokeTest<Real, B, double>("AVX w/FMA", [&](auto &&...args) { fft_real_f64_fma(args...); }, [&](auto &&...args) { ifft_real_f64_fma(args...); });
      #elif CPU_ARM_ANY
        SmokeTest<Real, B, float>("NEON", [&](auto &&...args) { fft_real_f32_neon(args...); }, [&](auto &&...args) { ifft_real_f32_neon(args...); });
        SmokeTest<Real, B, double>("NEON", [&](auto &&...args) { fft_real_f64_neon(args...); }, [&](auto &&...args) { ifft_real_f64_neon(args...); });
      #endif
    };

  doTests.template operator()<InPlace>();
  doTests.template operator()<Separate>();

  return 0;
}