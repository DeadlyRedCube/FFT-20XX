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

#define MULTIBENCH 0


#if defined(__clang__)
  __attribute__((always_inline)) inline void AvoidOptimizing(auto &&v)
  {
    asm volatile("" : "+r,m"(v) : : "memory");
  }
#elif defined (__GNUC__)
  __attribute__((always_inline)) inline void AvoidOptimizing(auto &&v)
  {
    asm volatile("" : "+m,r"(v) : : "memory");
  }
#elif defined(_MSC_VER)
  thread_local void const volatile * volatile avoid_opt_ptr = nullptr;
  __forceinline  void AvoidOptimizing(auto &&v)
  {
    avoid_opt_ptr = reinterpret_cast<void const volatile *>(&v);
    _ReadWriteBarrier();
  }
#endif

enum class BufferType
{
  InPlace,
  Separate,
};

template<BufferType B, std::floating_point F>
void Benchmark(const char *name, auto &&fftFunction, std::uint32_t maxFFTLength, auto &&initSize, auto &&deinitSize)
{
  printf("%s, %s, %s\n", name, (B == BufferType::InPlace ? "In-Place" : "Separate"), std::same_as<F, float> ? "f32" : "f64");
  (void)maxFFTLength;

  for (auto length = 64u; length <= maxFFTLength; length *= 2)
  //auto length = 256u;
  {
    initSize(length);
    #if MULTIBENCH
      printf("  Length %7u", length);
      fprintf(stderr, "length %u\n", length);
    #else
      printf("  Length %7u: ", length);
    #endif

    #if MULTIBENCH
      for (auto iterationCount = 0u; iterationCount < 128; iterationCount++)
    #endif
    {
      #if MULTIBENCH
        if (iterationCount % 5 == 0)
          { fprintf(stderr, "."); }
      #endif
      auto vals = AlignedAlloc<std::complex<F>>(length, 8);
      auto outs = AlignedAlloc<std::complex<F>>(length, 8);
      for (auto  i = 0u; i < length; i++)
        { vals[i] = { F(i), -F(2*i) }; }

      if constexpr (B == BufferType::InPlace)
        { std::ranges::copy(vals, vals + length, outs); }

      auto primeCount = 10u;
      auto count = 32u * 1024u * 1024u / length;
      #if !MULTIBENCH
        count = 32 * 1024 * 1024u / length;
      #endif

      for (auto i = 0u; i < primeCount; i++)
      {
        if constexpr(B == BufferType::InPlace)
          { fftFunction(outs, outs, length); }
        else
        {
          fftFunction(vals, outs, length);
          std::swap(vals, outs);
        }
      }

      auto start = std::chrono::steady_clock::now();

      for (auto i = 0u; i < count; i++)
      {
        if constexpr(B == BufferType::InPlace)
        {
          AvoidOptimizing(outs);
          fftFunction(outs, outs, length);
          AvoidOptimizing(outs);
        }
        else
        {
          AvoidOptimizing(vals);
          fftFunction(vals, outs, length);
          AvoidOptimizing(outs);
          std::swap(vals, outs);
        }
      }

      auto elapsed = std::chrono::steady_clock::now() - start;
      auto c = uint32_t(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count());

      #if MULTIBENCH
        printf(", %8u", c);
      #else
        printf("%8u\n", c);
      #endif

      AlignedFree(vals);
      AlignedFree(outs);
    }

    #if MULTIBENCH
      printf("\n");
      fprintf(stderr, "\n");
    #endif

    deinitSize();
  }
  printf("\n");
}


template<BufferType B, std::floating_point F>
void Benchmark(const char *name, auto &&fftFunction, std::uint32_t maxFFTLength)
  { Benchmark<B, F>(name, fftFunction, maxFFTLength, [](size_t){}, []{}); }

template<BufferType B, std::floating_point F>
void BenchmarkReal(const char *name, auto fftFunction, std::uint32_t maxFFTLength, auto &&initSize, auto &&deinitSize)
{
  printf("%s, %s, %s\n", name, (B == BufferType::InPlace ? "In-Place" : "Separate"), std::same_as<F, float> ? "f32" : "f64");
  for (auto length = 64u; length <= maxFFTLength; length *= 2)
  {
    initSize(length);

    #if MULTIBENCH
      printf("  Length %7u", length);
      fprintf(stderr, "length %u\n", length);
    #else
      printf("  Length %7u: ", length);
    #endif

    #if MULTIBENCH
      for (auto iterationCount = 0u; iterationCount < 128; iterationCount++)
    #endif
    {
      #if MULTIBENCH
        if (iterationCount % 5 == 0)
          { fprintf(stderr, "."); }
      #endif
      auto vals = AlignedAlloc<F>(length, 8);
      auto outs = AlignedAlloc<F>(length, 8);
      for (auto  i = 0u; i < length; i++)
        { vals[i] = F(i); }

      if constexpr (B == BufferType::InPlace)
        { std::ranges::copy(vals, vals + length, outs); }

      auto primeCount = 10u;
      auto count = 32u * 1024u * 1024u / length;

      for (auto i = 0u; i < primeCount; i++)
      {
        if constexpr(B == BufferType::InPlace)
          { fftFunction(outs, outs, length); }
        else
        {
          fftFunction(vals, outs, length);
          std::swap(vals, outs);
        }
      }

      auto start = std::chrono::steady_clock::now();

      for (auto i = 0u; i < count; i++)
      {
        if constexpr(B == BufferType::InPlace)
          { fftFunction(outs, outs, length); }
        else
        {
          fftFunction(vals, outs, length);
          std::swap(vals, outs);
        }
      }

      auto elapsed = std::chrono::steady_clock::now() - start;
      auto c = uint32_t(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count());

      #if MULTIBENCH
        printf(", %8u", c);
      #else
        printf("%8u\n", c);
      #endif

      AlignedFree(vals);
      AlignedFree(outs);

    }

    #if MULTIBENCH
      printf("\n");
      fprintf(stderr, "\n");
    #endif

    deinitSize();
  }
  printf("\n");
}


template<BufferType B, std::floating_point F>
void BenchmarkReal(const char *name, auto &&fftFunction, std::uint32_t maxFFTLength)
  { BenchmarkReal<B, F>(name, fftFunction, maxFFTLength, [](size_t){}, []{}); }


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

template <std::floating_point F>
  void BasicDITRadix2FFT(const std::complex<F> *input, std::complex<F> *output, std::size_t fftLength);

template <std::floating_point F>
  void BasicDITRadix2IFFT(const std::complex<F> *input, std::complex<F> *output, std::size_t fftLength);

template <BufferType B, typename F>
void SmokeTest(const char *name, auto testFwd, auto testInv)
{
  constexpr auto MaxFFTLength = 32768u;

  for (auto size = 1u; size <= MaxFFTLength; size *= 2)
  {
    auto table = fft_20xx::fft_table<F>{size};

    std::vector<std::complex<F>> input(size);
    std::vector<std::complex<double>> inputD(size);
    std::vector<std::complex<double>> outputGood(size);
    std::vector<std::complex<F>> outputTest(size);
    std::vector<std::complex<double>> ioutputGood(size);
    std::vector<std::complex<F>> ioutputTest(size);

    auto seq = std::seed_seq{1, 2, 3, 4, 5};
    std::mt19937_64 rnd{seq};
    auto gen = [&] { return std::uniform_real_distribution<F>{F(-0.5), F(0.5)}(rnd); };

    for (auto i = 0u; i < input.size(); i++)
    {
      input[i] = { gen(), gen() };
      inputD[i] = input[i];
    }


    BasicDITRadix2FFT(inputD.data(), outputGood.data(), size);
    BasicDITRadix2IFFT(inputD.data(), ioutputGood.data(), size);

    if constexpr (B == BufferType::InPlace)
    {
      std::ranges::copy(input, outputTest.begin());
      std::ranges::copy(input, ioutputTest.begin());
      testFwd(table, outputTest.data(), outputTest.data(), size);
      testInv(table, ioutputTest.data(), ioutputTest.data(), size);
    }
    else
    {
      testFwd(table, input.data(), outputTest.data(), size);
      testInv(table, input.data(), ioutputTest.data(), size);
    }

    constexpr auto Threshold = std::same_as<F, float> ? F(1e-4) : F(1e-12);

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
      if (Error(outputGood[i].real(), outputTest[i].real()) > Threshold)
      {
        fprintf(
          stderr,
          "Fwd FFT %s, %s, %s %u, at %u r: %.20f != %.20f (err: %e)\n",
          name,
          B == BufferType::InPlace ? "In-Place" : "Separate",
          std::same_as<F, float> ? "f32" : "f64",
          size,
          i,
          outputGood[i].real(),
          outputTest[i].real(),
          Error(outputGood[i].real(), outputTest[i].real()));
        break;
      }
      if (Error(outputGood[i].imag(), outputTest[i].imag()) > Threshold)
      {
        fprintf(
          stderr,
          "Fwd FFT %s, %s, %s %u, at %u i: %.20f != %.20f (err: %e)\n",
          name,
          B == BufferType::InPlace ? "In-Place" : "Separate",
          std::same_as<F, float> ? "f32" : "f64",
          size,
          i,
          outputGood[i].imag(),
          outputTest[i].imag(),
          Error(outputGood[i].imag(), outputTest[i].imag()));
        break;
      }
    }

    for (auto i = 0u; i < size; i++)
    {
      if (Error(ioutputGood[i].real(), ioutputTest[i].real()) > Threshold)
      {
        fprintf(
          stderr,
          "Inv FFT %s, %s, %s %u, at %u r: %.20f != %.20f\n",
          name,
          B == BufferType::InPlace ? "In-Place" : "Separate",
          std::same_as<F, float> ? "f32" : "f64",
          size,
          i,
          ioutputGood[i].real(),
          ioutputTest[i].real());
        break;
      }

      if (Error(ioutputGood[i].imag(), ioutputTest[i].imag()) > Threshold)
      {
        fprintf(
          stderr,
          "Inv FFT %s, %s, %s %u, at %u i: %.20f != %.20f\n",
          name,
          B == BufferType::InPlace ? "In-Place" : "Separate",
          std::same_as<F, float> ? "f32" : "f64",
          size,
          i,
          ioutputGood[i].imag(),
          ioutputTest[i].imag());
        break;
      }
    }
  }
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


int main()
{
  FlushSubnormals();

  constexpr auto MaxFFTLength = 512 * 1024u;

  using enum BufferType;
  SmokeTest<InPlace, float>("Scalar", [&](auto &&...args) { fft_20xx::fft_cplx(args...); }, [&](auto &&...args) { fft_20xx::ifft_cplx(args...); });
  SmokeTest<InPlace, double>("Scalar", [&](auto &&...args) { fft_20xx::fft_cplx(args...); }, [&](auto &&...args) { fft_20xx::ifft_cplx(args...); });

  fft_20xx::fft_table<float> table32{MaxFFTLength};
  fft_20xx::fft_table<double> table64{MaxFFTLength};

  Benchmark<InPlace, float>("Auto Fwd", [&](auto &&...args) { fft_20xx::fft_cplx(table32, args...); }, MaxFFTLength);
  Benchmark<InPlace, double>("Auto Fwd", [&](auto &&...args) { fft_20xx::fft_cplx(table64, args...); }, MaxFFTLength);
  Benchmark<Separate, float>("Auto Fwd", [&](auto &&...args) { fft_20xx::fft_cplx(table32, args...); }, MaxFFTLength);
  Benchmark<Separate, double>("Auto Fwd", [&](auto &&...args) { fft_20xx::fft_cplx(table64, args...); }, MaxFFTLength);

  Benchmark<InPlace, float>("Auto Inv", [&](auto &&...args) { fft_20xx::ifft_cplx(table32, args...); }, MaxFFTLength);
  Benchmark<InPlace, double>("Auto Inv", [&](auto &&...args) { fft_20xx::ifft_cplx(table64, args...); }, MaxFFTLength);
  Benchmark<Separate, float>("Auto Inv", [&](auto &&...args) { fft_20xx::ifft_cplx(table32, args...); }, MaxFFTLength);
  Benchmark<Separate, double>("Auto Inv", [&](auto &&...args) { fft_20xx::ifft_cplx(table64, args...); }, MaxFFTLength);

  return 0;
}