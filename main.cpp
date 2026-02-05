#include "include/fft_20xx.h"

#include "fft_20xx_common.h"
#include <complex>
#include <vector>

#if COMPILER_MSVC
  #pragma warning (disable : 5045)
#endif

int main()
{
  constexpr auto MaxFFTLength = 65536u;
  auto table32 = AllocateFFTTable32(MaxFFTLength);
  auto table64 = AllocateFFTTable64(MaxFFTLength);

  std::vector<std::complex<float>> input(64);
  std::vector<std::complex<float>> output(64);

  for (auto i = 0u; i < input.size(); i++)
  {
    input[i] = {float(i), -float(2 * i) };
  }

  FFTComplex32(table32, reinterpret_cast<float *>(input.data()), reinterpret_cast<float *>(output.data()), 64);

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
  printf("\n\n");
  for (auto v : output)
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