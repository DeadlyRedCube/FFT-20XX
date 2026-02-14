#ifdef _MSC_VER
  #pragma warning (disable: 4710) // function not inlined
  #pragma warning (disable: 4711) // function <x> selected for automatic inline expansion
  #pragma warning (disable: 5045) // Compiler will insert Spectre mitigation for memory load
#endif

#include "fft_20xx_platform.h"

#include "include/fft_20xx.h"

#include <chrono>
#include <complex>
#include <cstdlib>
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



template<std::floating_point F>
void Benchmark(const char *name, auto fftFunction, std::uint32_t maxFFTLength)
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
      auto vals = AlignedAlloc<std::complex<F>>(length, 8);
      auto outs = AlignedAlloc<std::complex<F>>(length, 8);
      for (auto  i = 0u; i < length; i++)
        { vals[i] = { F(i), -F(2*i) }; }

      auto primeCount = 10u;
      auto count = 32u * 1024u * 1024u / length;

      for (auto i = 0u; i < primeCount; i++)
      {
        fftFunction(vals, outs, length);
        std::swap(vals, outs);
      }

      auto start = std::chrono::steady_clock::now();

      for (auto i = 0u; i < count; i++)
      {
        fftFunction(vals, outs, length);
        std::swap(vals, outs);
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


template<std::floating_point F>
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

      auto primeCount = 10u;
      auto count = 32u * 1024u * 1024u / length;

      for (auto i = 0u; i < primeCount; i++)
      {
        fftFunction(vals, outs, length);
        std::swap(vals, outs);
      }

      auto start = std::chrono::steady_clock::now();

      for (auto i = 0u; i < count; i++)
      {
        fftFunction(vals, outs, length);
        std::swap(vals, outs);
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

template <typename F>
void SmokeTest(const char *name, auto testFwd, auto testInv)
{
  constexpr auto MaxFFTLength = 4096;
  [[maybe_unused]] auto table32 = AllocateFFTTable32(MaxFFTLength);
  [[maybe_unused]] auto table64 = AllocateFFTTable64(MaxFFTLength);

  auto *table =
    [&]
    {
      if constexpr (std::same_as<F, float>)
        { return table32; }
      else
        { return table64; }
    }();

  for (auto size = 1u; size <= MaxFFTLength; size *= 2)
  {
    std::vector<std::complex<F>> input(size);
    std::vector<std::complex<F>> outputGood(size);
    std::vector<std::complex<F>> outputTest(size);
    std::vector<std::complex<F>> ioutputGood(size);
    std::vector<std::complex<F>> ioutputTest(size);

    for (auto i = 0u; i < input.size(); i++)
      { input[i] = {F(i), -F(2 * i) }; }


    if constexpr (std::same_as<F, float>)
    {
      FFTComplex32_Scalar(table, input.data(), outputGood.data(), size);
      IFFTComplex32_Scalar(table, input.data(), ioutputGood.data(), size);
    }
    else
    {
      FFTComplex64_Scalar(table, input.data(), outputGood.data(), size);
      IFFTComplex64_Scalar(table, input.data(), ioutputGood.data(), size);
    }

    testFwd(table, input.data(), outputTest.data(), size);
    testInv(table, input.data(), ioutputTest.data(), size);

    // The float32 epsilon is fairly high because of differences in using fma vs not
    constexpr F Epsilon = F(std::same_as<F, float> ? 5e-3 : 1e-9);

    for (auto i = 0u; i < size; i++)
    {
      auto maxR = std::max(std::abs(outputTest[i].real()), std::abs(outputGood[i].real()));
      auto maxI = std::max(std::abs(outputTest[i].imag()), std::abs(outputGood[i].imag()));
      if (std::abs(outputTest[i].real() - outputGood[i].real()) / maxR > Epsilon)
        { printf("Fwd FFT %s %u, at %u r: %f != %f\n", name, size, i, outputGood[i].real(), outputTest[i].real()); abort(); }
      if (std::abs(outputTest[i].imag() - outputGood[i].imag()) / maxI > Epsilon)
        { printf("Fwd FFT %s %u, at %u i: %f != %f\n", name, size, i, outputGood[i].imag(), outputTest[i].imag()); abort(); }
    }

    for (auto i = 0u; i < size; i++)
    {
      auto maxR = std::max(std::abs(ioutputTest[i].real()), std::abs(ioutputGood[i].real()));
      auto maxI = std::max(std::abs(ioutputTest[i].imag()), std::abs(ioutputGood[i].imag()));
      if (std::abs(ioutputTest[i].real() - ioutputGood[i].real()) / maxR > Epsilon)
        { printf("Inv FFT %s %u, at %u r: %f != %f\n", name, size, i, ioutputGood[i].real(), ioutputTest[i].real()); abort(); }
      if (std::abs(ioutputTest[i].imag() - ioutputGood[i].imag()) / maxI > Epsilon)
        { printf("Inv FFT %s %u, at %u i: %f != %f\n", name, size, i, ioutputGood[i].imag(), ioutputTest[i].imag()); abort(); }
    }
  }

  FreeFFTTable32(table32);
  FreeFFTTable64(table64);
}

int main()
{
  FlushSubnormals();

  constexpr auto MaxFFTLength = 512 * 1024u;
  [[maybe_unused]] auto table32 = AllocateFFTTable32(MaxFFTLength);
  [[maybe_unused]] auto table64 = AllocateFFTTable64(MaxFFTLength);


  #if 1
    using F = float;
    constexpr auto size = 32u;
    std::vector<std::complex<F>> input(size);
    std::vector<std::complex<F>> output(size);
    std::vector<std::complex<F>> output2(size);
    std::vector<F> inputReal(size);
    std::vector<F> outputReal(size);
    std::vector<F> outputReal2(size);

    for (auto i = 0u; i < input.size(); i++)
    {
      input[i] = {F(i), 0 };
      inputReal[i] = F(i);
    }

    FFTComplex32_Scalar(table32, input.data(), output.data(), size);
    FFTComplex32_NEON(table32, input.data(), output2.data(), size);

    printf("\n\n");
    for (auto v : output)
    {
      if (v.imag() < 0)
        { printf("%f - i*%f\n", v.real(), -v.imag()); }
      else
        { printf("%f + i*%f,\n", v.real(), v.imag()); }
    }

    /*FFTReal32_FMA(table32, inputReal.data(), outputReal.data(), size);
    IFFTReal32(table32, outputReal.data(), outputReal2.data(), size);

    printf("\n\n");
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

  BenchmarkReal<float>("Scalar   (f32, Real)", [&](auto &&...args) { FFTReal32(table32, args...); }, MaxFFTLength);
  //BenchmarkReal<float>("AVX/FMA  (f32, Real)", [&](auto &&...args) { FFTReal32_FMA(table32, args...); }, MaxFFTLength);

  Benchmark<float>("Auto     (f32)", [&](auto &&...args) { FFTComplex32(table32, args...); }, MaxFFTLength);
  Benchmark<float>("Scalar   (f32)", [&](auto &&...args) { FFTComplex32_Scalar(table32, args...); }, MaxFFTLength);

  #if defined(_M_X64) || defined(__x86_64__) || defined(_M_IX86) || defined(__i386__)
    SmokeTest<float>("SSE2 f32", [&](auto &&...args) { FFTComplex32_SSE2(args...); }, [&](auto &&...args) { IFFTComplex32_SSE2(args...); });
    Benchmark<float>("SSE2     (f32)", [&](auto &&...args) { FFTComplex32_SSE2(table32, args...); }, MaxFFTLength);
    SmokeTest<float>("AVX f32", [&](auto &&...args) { FFTComplex32_AVX(args...); }, [&](auto &&...args) { IFFTComplex32_AVX(args...); });
    Benchmark<float>("AVX only (f32)", [&](auto &&...args) { FFTComplex32_AVX(table32, args...); }, MaxFFTLength);
    SmokeTest<float>("FMA f32", [&](auto &&...args) { FFTComplex32_FMA(args...); }, [&](auto &&...args) { IFFTComplex32_FMA(args...); });
    Benchmark<float>("AVX/FMA  (f32)", [&](auto &&...args) { FFTComplex32_FMA(table32, args...); }, MaxFFTLength);
  #elif defined(__ARM_NEON) && (defined(_M_ARM64) || defined(__aarch64__))
    SmokeTest<float>("NEON f32", [&](auto &&...args) { FFTComplex32_NEON(args...); }, [&](auto &&...args) { IFFTComplex32_NEON(args...); });
    Benchmark<float>("NEON     (f32)", [&](auto &&...args) { FFTComplex32_NEON(table32, args...); }, MaxFFTLength);
  #endif

  Benchmark<double>("Scalar   (f64)", [&](auto &&...args) { FFTComplex64_Scalar(table64, args...); }, MaxFFTLength);

  #if defined(_M_X64) || defined(__x86_64__) || defined(_M_IX86) || defined(__i386__)
    SmokeTest<double>("SSE2 f64", [&](auto &&...args) { FFTComplex64_SSE2(args...); }, [&](auto &&...args) { IFFTComplex64_SSE2(args...); });
    Benchmark<double>("SSE2     (f64)", [&](auto &&...args) { FFTComplex64_SSE2(table64, args...); }, MaxFFTLength);
    SmokeTest<double>("AVX f64", [&](auto &&...args) { FFTComplex64_AVX(args...); }, [&](auto &&...args) { IFFTComplex64_AVX(args...); });
    Benchmark<double>("AVX only (f64)", [&](auto &&...args) { FFTComplex64_AVX(table64, args...); }, MaxFFTLength);
    SmokeTest<double>("FMA f64", [&](auto &&...args) { FFTComplex64_FMA(args...); }, [&](auto &&...args) { IFFTComplex64_FMA(args...); });
    Benchmark<double>("AVX/FMA  (f64)", [&](auto &&...args) { FFTComplex64_FMA(table64, args...); }, MaxFFTLength);
  #elif defined(__ARM_NEON) && (defined(_M_ARM64) || defined(__aarch64__))
    SmokeTest<double>("NEON f64", [&](auto &&...args) { FFTComplex64_NEON(args...); }, [&](auto &&...args) { IFFTComplex64_NEON(args...); });
    Benchmark<double>("NEON     (f64)", [&](auto &&...args) { FFTComplex64_NEON(table64, args...); }, MaxFFTLength);
  #endif
  FreeFFTTable32(table32);
  FreeFFTTable64(table64);
  return 0;
}