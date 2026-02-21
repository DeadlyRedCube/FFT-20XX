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
#include <random>
#include <vector>

#include "fft_20xx_bit_reverse_copy.h"
#include "fft_20xx_math.h"

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

#define MULTIBENCH 1



enum class BufferType
{
  InPlace,
  Separate,
};

template<BufferType B, std::floating_point F>
void Benchmark(const char *name, auto fftFunction, std::uint32_t maxFFTLength)
{
  printf("%s, %s, %s\n", name, (B == BufferType::InPlace ? "In-Place" : "Separate"), std::same_as<F, float> ? "f32" : "f64");
  (void)maxFFTLength;

  for (auto length = 64u; length <= maxFFTLength; length *= 2)
  //auto length = 128u;
  {
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
      auto vals = AlignedAlloc<FFT20XX::Complex<F>>(length, 8);
      auto outs = AlignedAlloc<FFT20XX::Complex<F>>(length, 8);
      for (auto  i = 0u; i < length; i++)
        { vals[i] = { F(i), -F(2*i) }; }

      if constexpr (B == BufferType::InPlace)
        { std::ranges::copy(vals, vals + length, outs); }

      auto primeCount = 10u;
      auto count = 32u * 1024u * 1024u / length * 4;

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

template <BufferType B, typename F>
void SmokeTest(const char *name, auto testFwd, auto testInv, size_t maxFFTLength)
{
  [[maybe_unused]] auto table32 = AllocateFFTTable32(maxFFTLength);
  [[maybe_unused]] auto table64 = AllocateFFTTable64(maxFFTLength);

  auto *table =
    [&]
    {
      if constexpr (std::same_as<F, float>)
        { return table32; }
      else
        { return table64; }
    }();

  for (auto size = 1u; size <= maxFFTLength; size *= 2)
  {
    std::vector<std::complex<F>> input(size);
    std::vector<std::complex<double>> inputD(size);
    std::vector<std::complex<double>> outputGood(size);
    std::vector<std::complex<F>> outputTest(size);
    std::vector<std::complex<double>> ioutputGood(size);
    std::vector<std::complex<F>> ioutputTest(size);

    for (auto i = 0u; i < input.size(); i++)
    { 
      input[i] = {F(i), -F(2 * i) }; 
      inputD[i] = input[i];
    }


    FFTComplex64_Scalar(table64, inputD.data(), outputGood.data(), size);
    IFFTComplex64_Scalar(table64, inputD.data(), ioutputGood.data(), size);

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

    // The float32 epsilon is fairly high because of differences in using fma vs not
    constexpr F Epsilon = F(std::same_as<F, float> ? 2e-2 : 1e-7);

    for (auto i = 0u; i < size; i++)
    {
      auto maxR = std::max(std::abs(double(outputTest[i].real())), std::abs(outputGood[i].real()));
      auto maxI = std::max(std::abs(double(outputTest[i].imag())), std::abs(outputGood[i].imag()));
      if (std::abs(outputTest[i].real() - outputGood[i].real()) / maxR > Epsilon)
        { printf("Fwd FFT %s %u, at %u r: %f != %f\n", name, size, i, outputGood[i].real(), outputTest[i].real()); goto continueOuter; }
      if (std::abs(outputTest[i].imag() - outputGood[i].imag()) / maxI > Epsilon)
        { printf("Fwd FFT %s %u, at %u i: %f != %f\n", name, size, i, outputGood[i].imag(), outputTest[i].imag()); goto continueOuter; }
    }

    for (auto i = 0u; i < size; i++)
    {
      auto maxR = std::max(double(std::abs(ioutputTest[i].real())), std::abs(ioutputGood[i].real()));
      auto maxI = std::max(double(std::abs(ioutputTest[i].imag())), std::abs(ioutputGood[i].imag()));
      if (std::abs(ioutputTest[i].real() - ioutputGood[i].real()) / maxR > Epsilon)
        { printf("Inv FFT %s %u, at %u r: %f != %f\n", name, size, i, ioutputGood[i].real(), ioutputTest[i].real()); goto continueOuter; }
      if (std::abs(ioutputTest[i].imag() - ioutputGood[i].imag()) / maxI > Epsilon)
        { printf("Inv FFT %s %u, at %u i: %f != %f\n", name, size, i, ioutputGood[i].imag(), ioutputTest[i].imag()); goto continueOuter; }
    }

  continueOuter:;
  }

  FreeFFTTable32(table32);
  FreeFFTTable64(table64);
}


enum class Inclusivity
{
  Inclusive,
  ExclusiveEnd,
};


using enum Inclusivity;

template <Inclusivity E, size_t A, size_t B, typename Func> requires (A <= B)
FORCE_INLINE constexpr void Unroll(Func &&f)
{
  constexpr size_t IterCount = B - A + ((E == Inclusive) ? 1 : 0);
  return []<size_t... I>(Func &&f, std::index_sequence<I...>)
    {
      (f.template operator()<I + A>(), ...);
    }(std::forward<Func>(f), std::make_index_sequence<IterCount>());
}

#ifdef _MSC_VER
  #define LAMBDA_FORCE_INLINE [[msvc::forceinline]]
#else
  #define LAMBDA_FORCE_INLINE
#endif

template <size_t N, std::floating_point F>
void TransposeBlockInPlace(
  FFT20XX::Complex<F> *inOut, 
  uint32_t shift)
{
  Unroll<ExclusiveEnd, 0, N>(
    [&]<size_t i>() LAMBDA_FORCE_INLINE
    {
      constexpr auto iRevUnshifted = FFT20XX::BitReverse(i);
      auto iRev = iRevUnshifted >> shift;
      Unroll<ExclusiveEnd, 0, i>(
        [&]<size_t j>() LAMBDA_FORCE_INLINE
        {
          constexpr auto jRevUnshifted = FFT20XX::BitReverse(j);
          auto jRev = jRevUnshifted >> shift;
          std::swap(inOut[jRev + i], inOut[iRev + j]);
        });
    });
}

template <size_t N, std::floating_point F>
void TransposeBlockSeparate(
  const FFT20XX::Complex<F> *input, 
  FFT20XX::Complex<F> *output, 
  uint32_t shift)
{
  Unroll<ExclusiveEnd, 0, N>(
    [&]<size_t i>() LAMBDA_FORCE_INLINE
    {
      constexpr auto iRevUnshifted = FFT20XX::BitReverse(i);
      auto iRev = iRevUnshifted >> shift;
      Unroll<ExclusiveEnd, 0, N>(
        [&]<size_t j>() LAMBDA_FORCE_INLINE
        {
          constexpr auto jRevUnshifted = FFT20XX::BitReverse(j);
          auto jRev = jRevUnshifted >> shift;
          output[jRev + i] = input[iRev + j];
          output[iRev + j] = input[jRev + i];
        });
    });
}

template <size_t FFTLen, std::floating_point F>
void TransposeBlockSelect(
  const FFT20XX::Complex<F> *input, 
  FFT20XX::Complex<F> *output, 
  uint32_t shift)
{
  constexpr auto N = size_t(1) << (FFT20XX::Log2Floor(FFTLen) / 2);
  if constexpr (FFT20XX::IsPowerOfFourAssumingIsPowerOfTwo(FFTLen))
  {
    if (input == output)
      { TransposeBlockInPlace<N>(output, shift); }
    else
      { TransposeBlockSeparate<N>(input, output, shift); }
  }
  else
  {
    if (input == output)
    { 
      TransposeBlockInPlace<N>(output, shift); 
      TransposeBlockInPlace<N>(output + N, shift); 
    }
    else
    { 
      TransposeBlockSeparate<N>(input, output, shift); 
      TransposeBlockSeparate<N>(input + N, output + N, shift); 
    }
  }
}

void TransposeDiagonalBlockInPlace(
  FFT20XX::Complex<float> *inOut, 
  size_t innerDim, 
  uint32_t shift)
{
  for (size_t i = 0, iRevUnshifted = 0; i < innerDim; i++, iRevUnshifted = FFT20XX::BitReverseIncrement(iRevUnshifted))
  {
    auto iRev = iRevUnshifted >> shift;
    for (size_t j = 0, jRevUnshifted = 0; j < i; j++, jRevUnshifted = FFT20XX::BitReverseIncrement(jRevUnshifted))
    {
      auto jRev = jRevUnshifted >> shift;
      std::swap(inOut[jRev + i], inOut[iRev + j]);
    }
  }
}

void TransposeDiagonalBlockSeparate(
  const FFT20XX::Complex<float> *input, 
  FFT20XX::Complex<float> *output, 
  size_t innerDim, 
  uint32_t shift)
{
  for (size_t i = 0, iRevUnshifted = 0; i < innerDim; i++, iRevUnshifted = FFT20XX::BitReverseIncrement(iRevUnshifted))
  {
    auto iRev = iRevUnshifted >> shift;
    for (size_t j = 0, jRevUnshifted = 0; j < innerDim; j++, jRevUnshifted = FFT20XX::BitReverseIncrement(jRevUnshifted))
    {
      auto jRev = jRevUnshifted >> shift;
      output[jRev + i] = input[iRev + j];
      output[iRev + j] = input[jRev + i];
    }
  }
}

void TransposeRecursive(
  const FFT20XX::Complex<float> *input, 
  FFT20XX::Complex<float> *output, 
  size_t innerDim, 
  uint32_t shift)
{
  if (innerDim <= 32)
  {
    if (input != output)
      { return TransposeDiagonalBlockSeparate(input, output, innerDim, shift); }
    else
      { return TransposeDiagonalBlockInPlace(output, innerDim, shift); }
  }

  auto halfDim = innerDim / 2;
  auto midY = FFT20XX::BitReverse(halfDim) >> shift;
  TransposeRecursive(input, output, halfDim, shift);
  TransposeRecursive(input + halfDim, output + halfDim, halfDim, shift);
  TransposeRecursive(input + midY, output + midY, halfDim, shift);
  TransposeRecursive(input + midY + halfDim, output + midY + halfDim, halfDim, shift);

  // Swap corners
  for (size_t i = 0, iRevUnshifted = 0; i < halfDim; i++, iRevUnshifted = FFT20XX::BitReverseIncrement(iRevUnshifted))
  {
    auto iRev = iRevUnshifted >> shift;
    for (size_t j = 0; j < halfDim; j++)
      { std::swap(output[iRev + (j + halfDim)], output[(iRev + midY) + j]); }
  }
}

void SwapBlocks(
  const FFT20XX::Complex<float> *inBlock1, 
  const FFT20XX::Complex<float> *inBlock2, 
  FFT20XX::Complex<float> *outBlock1, 
  FFT20XX::Complex<float> *outBlock2, 
  size_t innerDim, 
  uint32_t shift)
{
  for (size_t i = 0, iRevUnshifted = 0; i < innerDim; i++, iRevUnshifted = FFT20XX::BitReverseIncrement(iRevUnshifted))
  {
    auto iRev = iRevUnshifted >> shift;
    for (size_t j = 0, jRevUnshifted = 0; j < innerDim; j++, jRevUnshifted = FFT20XX::BitReverseIncrement(jRevUnshifted))
    {
      auto jRev = jRevUnshifted >> shift;
      auto t = inBlock1[iRev + j];
      outBlock1[iRev + j] = inBlock2[jRev + i];
      outBlock2[jRev + i] = t;

      t = inBlock1[jRev + i];
      outBlock1[jRev + i] = inBlock2[iRev + j];
      outBlock2[iRev + j] = t;
    }
  }
}


void TransposeTest(const FFT20XX::Complex<float> *input, FFT20XX::Complex<float> *output, size_t length)
{
  #if 0
    constexpr auto MinDim = 32u;
    constexpr auto MinDimShift = FFT20XX::Log2Floor(MinDim);
    const auto bitCount = FFT20XX::Log2Floor(length);
    const auto dim = size_t(1) << (bitCount / 2);
    const auto shift = FFT20XX::BitReverseShiftFromBitCount<size_t>(bitCount);
    switch (length)
    {
    case 1: return TransposeBlockSelect<1>(input, output, shift);
    case 2: return TransposeBlockSelect<2>(input, output, shift);
    case 4: return TransposeBlockSelect<4>(input, output, shift);
    case 8: return TransposeBlockSelect<8>(input, output, shift);
    case 16: return TransposeBlockSelect<16>(input, output, shift);
    case 32: return TransposeBlockSelect<32>(input, output, shift);
    case 64: return TransposeBlockSelect<64>(input, output, shift);
    case 128: return TransposeBlockSelect<128>(input, output, shift);
    case 256: return TransposeBlockSelect<256>(input, output, shift);
    case 512: return TransposeBlockSelect<512>(input, output, shift);
    case 1024: return TransposeBlockSelect<1024>(input, output, shift);
    }

    // Diagonals first
    while (true)
    {
      if (input == output)
      {
        for (auto i = size_t(0); i < dim; i += MinDim)
          { TransposeDiagonalBlockInPlace(output + i + (FFT20XX::BitReverse(i) >> shift), MinDim, shift); }
      }
      else
      {
        for (auto i = size_t(0); i < dim; i += MinDim)
        { 
          auto offset = i + (FFT20XX::BitReverse(i) >> shift);
          TransposeDiagonalBlockSeparate(input + offset, output + offset, MinDim, shift);
        }
      }

      // Now all the block swaps
      const auto iterDim = dim / MinDim;
      for (size_t i = 0, iRevUnshifted = 0; i < iterDim; i++, iRevUnshifted = FFT20XX::BitReverseIncrement(iRevUnshifted))
      {
        auto iRev = iRevUnshifted >> (shift + MinDimShift);
        for (size_t j = 0, jRevUnshifted = 0; j < i; j++, jRevUnshifted = FFT20XX::BitReverseIncrement(jRevUnshifted))
        {
          auto jRev = jRevUnshifted >> (shift + MinDimShift);
          auto block1Off = (j << MinDimShift) + iRev;
          auto block2Off = jRev + (i << MinDimShift);
          SwapBlocks(input + block1Off, input + block2Off, output + block1Off, output + block2Off, MinDim, shift);
        }
      }

      if (FFT20XX::IsPowerOfFourAssumingIsPowerOfTwo(length))
        { return; }

      // This was not a power of four, so we have to do a second pass, as a non-power-of-two transpose ends up working
      //  like two transposes side-by-side.
      length >>= 1; // Update this just so that it'll exit on the next loop
      input += dim;
      output += dim;
    }

  #elif 1
    const auto bitCount = FFT20XX::Log2Floor(length);
    const auto dim = size_t(1) << (bitCount / 2);
  
    const auto shift = FFT20XX::BitReverseShiftFromBitCount<size_t>(bitCount);
    // std::ranges::copy(input, input + length, output);

    TransposeRecursive(input, output, dim, shift);
    if (!FFT20XX::IsPowerOfFourAssumingIsPowerOfTwo(length))
      { TransposeRecursive(input + dim, output + dim, dim, shift); }
  #else
    for (size_t i = 0, iRevUnshifted = 0; i < dim; i++, iRevUnshifted = FFT20XX::BitReverseIncrement(iRevUnshifted))
    {
      auto iRev = iRevUnshifted >> shift;
      for (size_t j = 0, jRevUnshifted = 0; j < i; j++, jRevUnshifted = FFT20XX::BitReverseIncrement(jRevUnshifted))
      {
        auto jRev = jRevUnshifted >> shift;
        std::swap(output[jRev + i], output[iRev + j]);
      }
    }
  #endif
}


int main()
{
  FlushSubnormals();

  constexpr auto MaxFFTLength = 512 * 1024u;
  [[maybe_unused]] auto table32 = AllocateFFTTable32(MaxFFTLength);
  [[maybe_unused]] auto table64 = AllocateFFTTable64(MaxFFTLength);

  #if 0
    constexpr auto size = 2048u;
    std::vector<FFT20XX::Complex<float>> input(size);
    std::vector<FFT20XX::Complex<float>> outputG(size);
    std::vector<FFT20XX::Complex<float>> outputT(size);

    for (auto i = 0u; i < input.size(); i++)
    {
      input[i] = {float(i), float(i) };
    }

    TransposeTest(input.data(), outputG.data(), size);
    FFT20XX::BitReverseCopy(input.data(), outputT.data(), size);

  #endif

  #if 0
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

    FFTComplex32_Scalar(table32, input.data(), output.data(), size);
    FFTComplex32_FMA(table32, input.data(), output2.data(), size);

    printf("\n\n");
    for (auto v : output)
    {
      if (v.imag() < 0)
        { printf("%f - i*%f\n", v.real(), -v.imag()); }
      else
        { printf("%f + i*%f,\n", v.real(), v.imag()); }
    }

    FFTReal32_FMA(table32, inputReal.data(), outputReal.data(), size);
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
    printf("%f\n", outputReal[1]);

  #endif

  using enum BufferType;

  //SmokeTest<InPlace, double>("Auto f64 In-Place", [&](auto &&...args) { FFTComplex64(args...); }, [&](auto &&...args) { IFFTComplex64(args...); }, MaxFFTLength);
  //SmokeTest<Separate, double>("Auto f64 Separate", [&](auto &&...args) { FFTComplex64(args...); }, [&](auto &&...args) { IFFTComplex64(args...); }, MaxFFTLength);
  Benchmark<Separate, float>("Outer T (No Copy)", [&](auto &&...args) { FFT20XX::BitReverseCopy(args...); }, MaxFFTLength);
  // Benchmark<InPlace, float>("Transpose", [&](auto &&...args) { TransposeTest(args...); }, MaxFFTLength);
  //Benchmark<Separate, float>("32: Separate Bit Reverse", [&](auto &&...args) { FFT20XX::BitReverseCopy(args...); }, MaxFFTLength);
  //Benchmark<InPlace, double>("64: In-Place Bit Reverse", [&](auto &&...args) { FFT20XX::BitReverseCopy(args...); }, MaxFFTLength);
  //Benchmark<Separate, double>("64: Separate Bit Reverse", [&](auto &&...args) { FFT20XX::BitReverseCopy(args...); }, MaxFFTLength);
  // SmokeTest<InPlace, float>("FMA f32 In Place", [&](auto &&...args) { FFTComplex32_FMA(args...); }, [&](auto &&...args) { IFFTComplex32_FMA(args...); }, MaxFFTLength);
  //SmokeTest<Separate, float>("Scalar f32 Separate", [&](auto &&...args) { FFTComplex32_Scalar(args...); }, [&](auto &&...args) { IFFTComplex32_Scalar(args...); }, MaxFFTLength);
  //SmokeTest<Separate, float>("AVX f32 Separate", [&](auto &&...args) { FFTComplex32_AVX(args...); }, [&](auto &&...args) { IFFTComplex32_AVX(args...); }, MaxFFTLength);
  //SmokeTest<Separate, float>("FMA f32 Separate", [&](auto &&...args) { FFTComplex32_FMA(args...); }, [&](auto &&...args) { IFFTComplex32_FMA(args...); }, MaxFFTLength);

  //Benchmark<InPlace, float>("NEON (f32) In-Place", [&](auto &&...args) { FFTComplex32_NEON(table32, args...); }, MaxFFTLength);
  //Benchmark<Separate, float>("NEON (f32) Separate", [&](auto &&...args) { FFTComplex32_NEON(table32, args...); }, MaxFFTLength);
  //Benchmark<InPlace, double>("NEON (f64) In-Place", [&](auto &&...args) { FFTComplex64_NEON(table64, args...); }, MaxFFTLength);
  //Benchmark<Separate, double>("NEON (f64) Separate", [&](auto &&...args) { FFTComplex64_NEON(table64, args...); }, MaxFFTLength);
  
  /*BenchmarkReal<float>("Scalar   (f32, Real)", [&](auto &&...args) { FFTReal32(table32, args...); }, MaxFFTLength);
  BenchmarkReal<float>("AVX/FMA  (f32, Real)", [&](auto &&...args) { FFTReal32_FMA(table32, args...); }, MaxFFTLength);

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
  FreeFFTTable64(table64);*/
  return 0;
}