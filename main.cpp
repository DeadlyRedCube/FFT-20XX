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
        { vals[i] = { F(i), 0 }; }

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

int main()
{
  FlushSubnormals();

  constexpr auto MaxFFTLength = 512 * 1024u;
  auto table32 = AllocateFFTTable32(MaxFFTLength);
  auto table64 = AllocateFFTTable64(MaxFFTLength);

  
  using F = float;
  constexpr auto size = 16u;
  std::vector<std::complex<F>> input(size);
  std::vector<std::complex<F>> output(size);
  std::vector<std::complex<F>> output2(size);

  for (auto i = 0u; i < input.size(); i++)
  {
    input[i] = {F(i), -F(2 * i) };
  }

  FFTComplex32_AVX(table32, input.data(), output.data(), size);

  IFFTComplex32_AVX(table32, output.data(), output2.data(), size);
   
  printf("\n\n");

  printf("\n\n");
  for (auto v : output2)
  {
    if (v.imag() < 0)
      { printf("%f - i*%f\n", v.real(), -v.imag()); }
    else
      { printf("%f + i*%f,\n", v.real(), v.imag()); }
  }

  // Flags are here: 
  //  https://en.wikipedia.org/wiki/CPUID
  //

  Benchmark<float>("Auto     (f32)", [&](auto &&...args) { FFTComplex32(table32, args...); }, MaxFFTLength);
  Benchmark<float>("Scalar   (f32)", [&](auto &&...args) { FFTComplex32_Scalar(table32, args...); }, MaxFFTLength);

  #if defined(_M_X64) || defined(__x86_64__) || defined(_M_IX86) || defined(__i386__)
    Benchmark<float>("SSE2     (f32)", [&](auto &&...args) { FFTComplex32_SSE2(table32, args...); }, MaxFFTLength);
    Benchmark<float>("AVX only (f32)", [&](auto &&...args) { FFTComplex32_AVX(table32, args...); }, MaxFFTLength);
    Benchmark<float>("AVX/FMA  (f32)", [&](auto &&...args) { FFTComplex32_FMA(table32, args...); }, MaxFFTLength);
  #elif defined(__ARM_NEON) && (defined(_M_ARM64) || defined(__aarch64__))
    Benchmark<float>("NEON     (f32)", [&](auto &&...args) { FFTComplex32_NEON(table32, args...); }, MaxFFTLength);
  #endif

  Benchmark<double>("Auto     (f64)", [&](auto &&...args) { FFTComplex64(table64, args...); }, MaxFFTLength);
  Benchmark<double>("Scalar   (f64)", [&](auto &&...args) { FFTComplex64_Scalar(table64, args...); }, MaxFFTLength);
  
  #if defined(_M_X64) || defined(__x86_64__) || defined(_M_IX86) || defined(__i386__)
    Benchmark<double>("SSE2     (f64)", [&](auto &&...args) { FFTComplex64_SSE2(table64, args...); }, MaxFFTLength);
    Benchmark<double>("AVX only (f64)", [&](auto &&...args) { FFTComplex64_AVX(table64, args...); }, MaxFFTLength);
    Benchmark<double>("AVX/FMA  (f64)", [&](auto &&...args) { FFTComplex64_FMA(table64, args...); }, MaxFFTLength);
  #elif defined(__ARM_NEON) && (defined(_M_ARM64) || defined(__aarch64__))
    Benchmark<double>("NEON     (f64)", [&](auto &&...args) { FFTComplex64_NEON(table64, args...); }, MaxFFTLength);
  #endif
  FreeFFTTable32(table32);
  FreeFFTTable64(table64);
  return 0;
}