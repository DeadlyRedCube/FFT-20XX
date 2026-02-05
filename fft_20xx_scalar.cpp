#include "fft_20xx_common.h"

#include <concepts>
#include <cinttypes>



namespace FFT20XX::Scalar
{

  FORCE_INLINE void Radix4ButterflyNoRotation(auto &x0, auto &x1, auto &x2, auto &x3)
  {
    auto sum02 = x0 + x2;
    auto diff02 = x0 - x2;
    auto sum13 = x1 + x3;
    auto iDiff13 = (x1 - x3) * -II;

    x0 = sum02 + sum13;
    x1 = diff02 + iDiff13;
    x2 = sum02 - sum13;
    x3 = diff02 - iDiff13;
  }

  FORCE_INLINE void Radix4Butterfly45Rotation(auto &x0, auto &x1, auto &x2, auto &x3)
  {
    auto sum02 = x0 + x2;
    auto diff02 = x0 - x2;
    auto sum13 = x1 + x3;
    auto iDiff13 = (x1 - x3) * -II;

    x0 = sum02 + sum13;
    x1 = Rot45<Dir::Forward>(diff02 + iDiff13);
    x2 = Rot90<Dir::Forward>(sum02 - sum13);
    x3 = Rot135<Dir::Forward>(diff02 - iDiff13);
  }

  FORCE_INLINE void Radix4Butterfly(auto &x0, auto &x1, auto &x2, auto &x3, auto w1, auto w2, auto w3)
  {
    auto sum02 = x0 + x2;
    auto diff02 = x0 - x2;
    auto sum13 = x1 + x3;
    auto iDiff13 = (x1 - x3) * -II;

    x0 = sum02 + sum13;
    x1 = Rot<Dir::Forward>(diff02 + iDiff13, w1);
    x2 = Rot<Dir::Forward>(sum02 - sum13, w2);
    x3 = Rot<Dir::Forward>(diff02 - iDiff13, w3);
  }

  template <std::floating_point F>
  FORCE_INLINE void DoFirstContiguousButterfly(const Complex<F> *, Complex<F> *data)
  {
    auto x0 = data[0];
    auto x2 = data[1]; // Intentionally reversed x1/x2
    auto x1 = data[2];
    auto x3 = data[3];

    Radix4ButterflyNoRotation(x0, x1, x2, x3);

    data[0] = x0;
    data[1] = x1;
    data[2] = x2;
    data[3] = x3;
  }

  template <std::floating_point F>
  FORCE_INLINE void DoContiguousButterfly(const Complex<F> *table, Complex<F> *data, std::size_t groupIndex)
  {
    auto x0 = data[0];
    auto x2 = data[1]; // Intentionally reversed x1/x2
    auto x1 = data[2];
    auto x3 = data[3];

    auto w1 = table[groupIndex << 1];
    auto w2 = table[groupIndex];
    auto w3 = table[(groupIndex << 1) + 1];
    Radix4Butterfly(x0, x1, x2, x3, w1, w2, w3);

    data[0] = x0;
    data[1] = x1;
    data[2] = x2;
    data[3] = x3;
  }

  template <std::floating_point F>
  FORCE_INLINE void DoNoRotationButterfly(Complex<F> *data, std::size_t baseIndex, std::size_t butterflyStep)
  {
    auto i0 = baseIndex;
    auto i1 = i0 + butterflyStep;
    auto i2 = i1 + butterflyStep;
    auto i3 = i2 + butterflyStep;

    auto x0 = data[i0];
    auto x2 = data[i1];
    auto x1 = data[i2];
    auto x3 = data[i3];

    Radix4ButterflyNoRotation(x0, x1, x2, x3);

    data[i0] = x0;
    data[i1] = x1;
    data[i2] = x2;
    data[i3] = x3;
  }


  template <std::floating_point F>
  FORCE_INLINE void DoFixed45Butterfly(Complex<F> *data, std::uint32_t baseIndex, std::uint32_t butterflyStep)
  {
    auto i0 = baseIndex;
    auto i1 = i0 + butterflyStep;
    auto i2 = i1 + butterflyStep;
    auto i3 = i2 + butterflyStep;

    auto x0 = data[i0];
    auto x2 = data[i1];
    auto x1 = data[i2];
    auto x3 = data[i3];

    Radix4Butterfly45Rotation(x0, x1, x2, x3);

    data[i0] = x0;
    data[i1] = x1;
    data[i2] = x2;
    data[i3] = x3;
  }

  template <std::floating_point F>
  FORCE_INLINE void DoMainButterfly(
    Complex<F> *data,
    std::size_t baseIndex,
    std::size_t butterflyStep,
    Complex<F> w1,
    Complex<F> w2,
    Complex<F> w3)
  {
    auto i0 = baseIndex;
    auto i1 = i0 + butterflyStep;
    auto i2 = i1 + butterflyStep;
    auto i3 = i2 + butterflyStep;

    auto x0 = data[i0];
    auto x2 = data[i1]; // Intentionally reversed x1/x2
    auto x1 = data[i2];
    auto x3 = data[i3];

    Radix4Butterfly(x0, x1, x2, x3, w1, w2, w3);

    data[i0] = x0;
    data[i1] = x1;
    data[i2] = x2;
    data[i3] = x3;
  }


  template <std::floating_point F>
  FORCE_INLINE void RunFirstContiguousButterflySet(const Complex<F> *, Complex<F> *data)
  {
    DoNoRotationButterfly(data, 0, 1);
    DoFixed45Butterfly(data, 4, 1);
  }

  template <std::floating_point F>
  FORCE_INLINE void RunContiguousButterflySet(const Complex<F> *table, Complex<F> *data, std::size_t groupIndex)
  {
    auto w1 = table[groupIndex << 1];
    auto w2 = table[groupIndex];
    auto w3 = table[(groupIndex << 1) + 1];
    DoMainButterfly(data, 0u, 1u, w1, w2, w3);

    w1 = table[(groupIndex << 1) + 2];
    w2 *= -II;
    w3 = table[(groupIndex << 1) + 3];
    DoMainButterfly(data, 4u, 1u, w1, w2, w3);
  }

  template <std::floating_point F>
  FORCE_INLINE void RunMultipleMainButterflies(const Complex<F> *table, Complex<F> *data, std::size_t groupIndex, std::size_t butterflyStep)
  {
    auto groupBase = groupIndex * butterflyStep * 4;
    auto w1 = table[groupIndex << 1];
    auto w2 = table[groupIndex];
    auto w3 = table[(groupIndex << 1) + 1];
    for (auto butterflyIndex = 0u; butterflyIndex < butterflyStep; butterflyIndex ++)
      { DoMainButterfly<F>(data, groupBase + butterflyIndex, butterflyStep, w1, w2, w3); }

    // Next group is a 45 degree rotation from the current one.
    w1 = table[(groupIndex  << 1) + 2];
    w2 *= -II; // This 90 degree rotate without using Rot is fine
    w3 = table[(groupIndex << 1) + 3];

    groupBase += butterflyStep * 4;
    for (auto butterflyIndex = 0u; butterflyIndex < butterflyStep; butterflyIndex ++)
      { DoMainButterfly<F>(data, groupBase + butterflyIndex, butterflyStep, w1, w2, w3); }
  }

  template <std::floating_point F>
  FORCE_INLINE void DoNoRotationRadix2Butterfly(
    Complex<F> *data,
    std::size_t baseIndex,
    std::size_t butterflyStep)
  {
    auto i0 = baseIndex;
    auto i1 = i0 + butterflyStep;

    auto even = data[i0];
    auto odd = data[i1];

    data[i0] = even + odd;
    data[i1] = even - odd;
  }


  #include "fft_20xx_run.h"


  template <std::floating_point F>
  void ForwardFFT(const Complex<F> *table, const Complex<F> *input, Complex<F> *output, std::size_t fftLength)
    { RunFFT<F, 1>(table, input, output, fftLength); }

  template void ForwardFFT(const Complex<float> *table, const Complex<float> *input, Complex<float> *output, std::size_t fftLength);
  template void ForwardFFT(const Complex<double> *table, const Complex<double> *input, Complex<double> *output, std::size_t fftLength);
}
