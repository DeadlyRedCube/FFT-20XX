#ifdef _MSC_VER
  #pragma warning (disable: 4710) // function not inlined
  #pragma warning (disable: 4711) // function <x> selected for automatic inline expansion
  #pragma warning (disable: 5045) // Compiler will insert Spectre mitigation for memory load
#endif

#include "include/fft_20xx.h"

#include <chrono>
#include <complex>
#include <vector>

template<std::floating_point F>
void Benchmark(const char *name, auto fftFunction, std::uint32_t maxFFTLength)
{
  printf("Algo: %s\n", name);
  for (auto length = 64u; length <= maxFFTLength; length *= 2)
  {
    printf("  Length %7u: ", length);

    std::vector<std::complex<F>> vals(length);
    std::vector<std::complex<F>> outs(length);
    for (auto  i = 0u; i < length; i++)
      { vals[i] = { F(i), 0 }; }

    auto primeCount = 10u;
    auto count = 32u * 1024u * 1024u / length;

    for (auto i = 0u; i < primeCount; i++)
    {
      fftFunction(vals.data(), outs.data(), length);
      std::swap(vals, outs);
    }

    auto start = std::chrono::steady_clock::now();

    for (auto i = 0u; i < count; i++)
    {
      fftFunction(vals.data(), outs.data(), length);
      std::swap(vals, outs);
    }

    auto elapsed = std::chrono::steady_clock::now() - start;
    auto c = uint32_t(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count());
    printf("%8u ms\n", c);
  }
  printf("\n");
}


int main()
{
  constexpr auto MaxFFTLength = 512 * 1024u;
  auto table32 = AllocateFFTTable32(MaxFFTLength);
  auto table64 = AllocateFFTTable64(MaxFFTLength);

  /*
  using F = float;
  constexpr auto size = 32u;
  std::vector<std::complex<F>> input(size);
  std::vector<std::complex<F>> output(size);
  std::vector<std::complex<F>> output2(size);

  for (auto i = 0u; i < input.size(); i++)
  {
    input[i] = {F(i), -F(2 * i) };
  }

  FFTComplex32(table32, input.data(), output.data(), size);

  IFFTComplex32(table32, output.data(), output2.data(), size);
   
  printf("\n\n");

  printf("\n\n");
  for (auto v : output)
  {
    if (v.imag() < 0)
      { printf("%f - i*%f\n", v.real(), -v.imag()); }
    else
      { printf("%f + i*%f,\n", v.real(), v.imag()); }
  }*/

  Benchmark<float>("Scalar (f32)", [&](auto &&...args) { FFTComplex32_Scalar(table32, args...); }, MaxFFTLength);
  Benchmark<float>("AVX/FMA (f32)", [&](auto &&...args) { FFTComplex32_FMA(table32, args...); }, MaxFFTLength);

  Benchmark<double>("Scalar (f64)", [&](auto &&...args) { FFTComplex64_Scalar(table64, args...); }, MaxFFTLength);
  Benchmark<double>("AVX/FMA (f64)", [&](auto &&...args) { FFTComplex64_FMA(table64, args...); }, MaxFFTLength);

  FreeFFTTable32(table32);
  FreeFFTTable64(table64);
  return 0;
}