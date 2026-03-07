#ifdef _MSC_VER
  #pragma warning (disable: 4710) // function not inlined
  #pragma warning (disable: 4711) // function <x> selected for automatic inline expansion
  #pragma warning (disable: 5045) // Compiler will insert Spectre mitigation for memory load
#endif

#include "fft_20xx_platform.h"

#include "include/fft_20xx.h"

#include <chrono>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <random>
#include <utility>
#include <vector>

#include "fft_20xx_bit_reverse_copy.h"
#include "fft_20xx_math.h"
#include "fft_20xx_table.h"

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
  __attribute__((always_inline)) void AvoidOptimizing(auto &&v)
  {
    asm volatile("" : "+r,m"(v) : : "memory");
  }
#elif defined (__GNUC__)
  __attribute__((always_inline)) void AvoidOptimizing(auto &&v)
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
void BenchmarkReal(const char *name, auto fftFunction, std::uint32_t maxFFTLength)
{
  printf("%s\n", name);
  for (auto length = 64u; length <= maxFFTLength; length *= 2)
  {
    #if MULTIBENCH
      printf("  Length %7u", length);
      fprintf(stderr, "length %u\n", length);
    #else
      printf("  Length %7u: ", length);
    #endif

    #if MULTIBENCH
      for (auto iterationCount = 0u; iterationCount < 100; iterationCount++)
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
  }
  printf("\n");
}


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
  void BasicDITRadix2FFT(const std::complex<F> *input, std::complex<F> *output, std::uint32_t fftLength);

template <std::floating_point F>
  void BasicDITRadix2IFFT(const std::complex<F> *input, std::complex<F> *output, std::uint32_t fftLength);

template <BufferType B, typename F>
void SmokeTest(const char *name, auto testFwd, auto testInv)
{
  constexpr auto MaxFFTLength = 32768u;

  for (auto size = 1u; size <= MaxFFTLength; size *= 2)
  {
    [[maybe_unused]] auto table32 = AllocateFFTTable32(size);
    [[maybe_unused]] auto table64 = AllocateFFTTable64(size);

    auto *table =
      [&]
      {
        if constexpr (std::same_as<F, float>)
          { return table32; }
        else
          { return table64; }
      }();

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
        goto continueOuter; 
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
        goto continueOuter; 
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
        goto continueOuter; 
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
        goto continueOuter; 
      }
    }

  continueOuter:;
    FreeFFTTable32(table32);
    FreeFFTTable64(table64);
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

template <std::floating_point F>
void BasicDITRadix2FFT(const std::complex<F> *input, std::complex<F> *output, std::uint32_t fftLength)
{
  FFT20XX::BitReverseCopy(reinterpret_cast<const FFT20XX::Complex<F> *>(input), reinterpret_cast<FFT20XX::Complex<F> *>(output), fftLength);

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
void BasicDITRadix2IFFT(const std::complex<F> *input, std::complex<F> *output, std::uint32_t fftLength)
{
  FFT20XX::BitReverseCopy(reinterpret_cast<const FFT20XX::Complex<F> *>(input), reinterpret_cast<FFT20XX::Complex<F> *>(output), fftLength);

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


  template <typename F>
  using FFTTable = std::conditional_t<std::same_as<F, float>, FFTTable32, FFTTable64>;

int main()
{
  FlushSubnormals();

  constexpr auto MaxFFTLength = 512 * 1024u;
  [[maybe_unused]] auto table32 = AllocateFFTTable32(MaxFFTLength);
  [[maybe_unused]] auto table64 = AllocateFFTTable64(MaxFFTLength);

  #if 0
    auto size = 16u;
    using F = float;
    std::vector<std::complex<F>> input(size);
    std::vector<std::complex<F>> output(size);
    std::vector<std::complex<F>> output2(size);
    std::vector<F> inputReal(size);
    std::vector<F> outputReal(size);
    std::vector<F> outputReal2(size);


    for (auto i = 0u; i < input.size(); i++)
    {
      input[i] = {F(i), -F(2*i) };
      inputReal[i] = F(i);
    }

    auto table = 
      [&]<typename FF>
      {
        if constexpr(std::same_as<F, float>)
          { return table32; }
        else
          { return table64; }
      }.template operator()<F>();

    BasicDITRadix2FFT(input.data(), output.data(), size);
    BasicDITRadix2IFFT(output.data(), output2.data(), size);
    #if CPU_X86_ANY
      FFTComplex32_SSE2(table, input.data(), output2.data(), size);
    #else
      FFTComplex32_NEON(table, input.data(), output2.data(), size);
    #endif
    //FFTComplex64_FMA(table, input.data(), output2.data(), size);
    //IFFTComplex64_FMA(table, output2.data(), output.data(), size);

    /*for (auto &c : output)
    {
      c = { std::round(c.real()), std::round(c.imag()) };
    }

    for (auto i = 0u; i < size; i++)
    {
      //assert(output[i] == output2[i]);
    }*/

    /*printf("\n\n");
    for (auto v : output)
    {
      if (v.imag() < 0)
        { printf("%f - i*%f\n", v.real(), -v.imag()); }
      else
        { printf("%f + i*%f,\n", v.real(), v.imag()); }
    }*/

    //FFTReal32_FMA(table32, inputReal.data(), outputReal.data(), size);
    //IFFTReal32(table32, outputReal.data(), outputReal2.data(), size);

    /*printf("\n\n");
    printf("%f\n", outputReal[0]);
    for (auto i = 2u; i < outputReal.size(); i+=2)
    {
      if (outputReal[i + 1] < 0)
        { printf("%f - i*%f\n", outputReal[i + 0], -outputReal[i + 1]); }
      else
        { printf("%f + i*%f,\n", outputReal[i + 0], outputReal[i + 1]); }
    }
    printf("%f\n", outputReal[1]);*/

  #endif

  using enum BufferType;

  SmokeTest<InPlace, float>("Scalar", [&](auto &&...args) { FFTComplex32_Scalar(args...); }, [&](auto &&...args) { IFFTComplex32_Scalar(args...); });
  SmokeTest<InPlace, double>("Scalar", [&](auto &&...args) { FFTComplex64_Scalar(args...); }, [&](auto &&...args) { IFFTComplex64_Scalar(args...); });

  Benchmark<InPlace, float>("Scalar", [&](auto &&...args) { FFTComplex32_Scalar(table32, args...); }, MaxFFTLength);
  #if CPU_X86_ANY
    SmokeTest<InPlace, float>("SSE2", [&](auto &&...args) { FFTComplex32_SSE2(args...); }, [&](auto &&...args) { IFFTComplex32_SSE2(args...); });
    SmokeTest<InPlace, double>("SSE2", [&](auto &&...args) { FFTComplex64_SSE2(args...); }, [&](auto &&...args) { IFFTComplex64_SSE2(args...); });
    SmokeTest<InPlace, float>("AVX", [&](auto &&...args) { FFTComplex32_AVX(args...); }, [&](auto &&...args) { IFFTComplex32_AVX(args...); });
    SmokeTest<InPlace, double>("AVX", [&](auto &&...args) { FFTComplex64_AVX(args...); }, [&](auto &&...args) { IFFTComplex64_AVX(args...); });
    SmokeTest<InPlace, float>("FMA", [&](auto &&...args) { FFTComplex32_FMA(args...); }, [&](auto &&...args) { IFFTComplex32_FMA(args...); });
    SmokeTest<InPlace, double>("FMA", [&](auto &&...args) { FFTComplex64_FMA(args...); }, [&](auto &&...args) { IFFTComplex64_FMA(args...); });

    Benchmark<InPlace, float>("SSE2", [&](auto &&...args) { FFTComplex32_SSE2(table32, args...); }, MaxFFTLength);
    Benchmark<InPlace, float>("AVX", [&](auto &&...args) { FFTComplex32_AVX(table32, args...); }, MaxFFTLength);
    Benchmark<InPlace, float>("FMA", [&](auto &&...args) { FFTComplex32_FMA(table32, args...); }, MaxFFTLength);
  #else
    SmokeTest<InPlace, float>("NEON", [&](auto &&...args) { FFTComplex32_NEON(args...); }, [&](auto &&...args) { IFFTComplex32_NEON(args...); });
    SmokeTest<InPlace, double>("NEON", [&](auto &&...args) { FFTComplex64_NEON(args...); }, [&](auto &&...args) { IFFTComplex64_NEON(args...); });
    Benchmark<InPlace, float>("NEON", [&](auto &&...args) { FFTComplex32_NEON(table32, args...); }, MaxFFTLength);
  #endif
  return 0;
}