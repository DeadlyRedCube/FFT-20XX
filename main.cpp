#include "fft_20xx_common.h"

#include "include/fft_20xx.h"


#include <complex>
#include <vector>

using namespace FFT20XX;



int main()
{
  using F = double;
  constexpr auto MaxFFTLength = 65536u;
  auto table32 = AllocateFFTTable32(MaxFFTLength);
  auto table64 = AllocateFFTTable64(MaxFFTLength);

  constexpr auto size = 1024u;
  std::vector<std::complex<F>> input(size);
  std::vector<std::complex<F>> output(size);
  std::vector<std::complex<F>> output2(size);

  for (auto i = 0u; i < input.size(); i++)
  {
    input[i] = {F(i), -F(2 * i) };
  }

  FFTComplex64(table64, input.data(), output.data(), size);

  IFFTComplex64(table64, output.data(), output2.data(), size);

  printf("\n\n");
  #if COMPILER_CLANG
    printf("CLANG BUILD\n");
  #elif COMPILER_GCC
    printf("GCC build\n");
  #elif COMPILER_MSVC
    printf("MSVC build\n");
  #else
    printf("Unknown compiler\n");
  #endif

  #if CPU_X64
    printf("  X64\n");
  #elif CPU_X86
    printf("  X86\n");
  #elif CPU_ARM32
    printf("  ARM (32-bit)\n");
  #elif CPU_ARM64
    printf("  ARM (64-bit)\n");
  #else
    printf("  Unknown CPU\n");
  #endif

  printf("Compiled with:\n");
  #if SIMD_SSE2
    printf("  - SSE2\n");
  #endif
  #if SIMD_SSE3
    printf("  - SSE3\n");
  #endif
  #if SIMD_SSE4_1
    printf("  - SSE4.1\n");
  #endif
  #if SIMD_SSE4_2
    printf("  - SSE4.2\n");
  #endif
  #if SIMD_AVX
    printf("  - AVX2\n");
  #endif
  #if SIMD_AVX2
    printf("  - AVX2\n");
  #endif
  #if SIMD_NEON
    printf("  - ARM NEON\n");
  #endif
  #if SIMD_NEON64
    printf("  - ARM64 NEON\n");
  #endif

  printf("\n\n");
  for (auto v : output2)
  {
    if (v.imag() < 0)
      { printf("%f - i*%f\n", v.real(), -v.imag()); }
    else
      { printf("%f + i*%f,\n", v.real(), v.imag()); }
  }

  FreeFFTTable32(table32);
  FreeFFTTable64(table64);
  return 0;
}