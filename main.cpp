#include "Include/fft_20xx.h"

#include <complex>
#include <vector>

#ifdef _MSC_VER
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

  FreeFFTTable32(table32);
  FreeFFTTable64(table64);
  return 0;
}