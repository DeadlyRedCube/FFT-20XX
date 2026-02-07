#pragma once 

template <Dir D>
FORCE_INLINE void Radix4ButterflyNoRotation(auto &x0, auto &x1, auto &x2, auto &x3)
{
  auto sum02 = x0 + x2;
  auto diff02 = x0 - x2;
  auto sum13 = x1 + x3;
  auto iDiff13 = Rot90<D>(x1 - x3);

  x0 = sum02 + sum13;
  x1 = diff02 + iDiff13;
  x2 = sum02 - sum13;
  x3 = diff02 - iDiff13;
}

template <Dir D>
FORCE_INLINE void Radix4Butterfly45Rotation(auto &x0, auto &x1, auto &x2, auto &x3)
{
  auto sum02 = x0 + x2;
  auto diff02 = x0 - x2;
  auto sum13 = x1 + x3;
  auto iDiff13 = Rot90<D>(x1 - x3);

  x0 = sum02 + sum13;
  x1 = Rot45<D>(diff02 + iDiff13);
  x2 = Rot90<D>(sum02 - sum13);
  x3 = Rot135<D>(diff02 - iDiff13);
}

template <Dir D>
FORCE_INLINE void Radix4Butterfly(auto &x0, auto &x1, auto &x2, auto &x3, auto w1, auto w2, auto w3)
{
  auto sum02 = x0 + x2;
  auto diff02 = x0 - x2;
  auto sum13 = x1 + x3;
  auto iDiff13 = Rot90<D>(x1 - x3);

  x0 = sum02 + sum13;
  x1 = Rot<D>(diff02 + iDiff13, w1);
  x2 = Rot<D>(sum02 - sum13, w2);
  x3 = Rot<D>(diff02 - iDiff13, w3);
}

template <Dir D, std::floating_point F>
FORCE_INLINE void DoNoRotationButterflySet(Complex<F> *data, std::size_t baseIndex, std::size_t butterflyStep)
{
  auto i0 = baseIndex;
  auto i1 = i0 + butterflyStep;
  auto i2 = i1 + butterflyStep;
  auto i3 = i2 + butterflyStep;

  auto x0 = LoadComplex<F>(&data[i0]);
  auto x2 = LoadComplex<F>(&data[i1]);
  auto x1 = LoadComplex<F>(&data[i2]);
  auto x3 = LoadComplex<F>(&data[i3]);

  Radix4ButterflyNoRotation<D>(x0, x1, x2, x3);

  StoreComplex(&data[i0], x0);
  StoreComplex(&data[i1], x1);
  StoreComplex(&data[i2], x2);
  StoreComplex(&data[i3], x3);
}


template <Dir D, std::floating_point F>
FORCE_INLINE void DoFixed45ButterflySet(Complex<F> *data, std::uint32_t baseIndex, std::uint32_t butterflyStep)
{
  auto i0 = baseIndex;
  auto i1 = i0 + butterflyStep;
  auto i2 = i1 + butterflyStep;
  auto i3 = i2 + butterflyStep;

  auto x0 = LoadComplex<F>(&data[i0]);
  auto x2 = LoadComplex<F>(&data[i1]);
  auto x1 = LoadComplex<F>(&data[i2]);
  auto x3 = LoadComplex<F>(&data[i3]);

  Radix4Butterfly45Rotation<D>(x0, x1, x2, x3);

  StoreComplex(&data[i0], x0);
  StoreComplex(&data[i1], x1);
  StoreComplex(&data[i2], x2);
  StoreComplex(&data[i3], x3);
}

template <Dir D, std::floating_point F>
FORCE_INLINE void DoMainButterfly(
  Complex<F> *data,
  std::size_t baseIndex,
  std::size_t butterflyStep,
  Complex<F> w1In,
  Complex<F> w2In,
  Complex<F> w3In)
{
  auto w1 = SIMDComplex{w1In};
  auto w2 = SIMDComplex{w2In};
  auto w3 = SIMDComplex{w3In};

  auto i0 = baseIndex;
  auto i1 = i0 + butterflyStep;
  auto i2 = i1 + butterflyStep;
  auto i3 = i2 + butterflyStep;
  auto x0 = LoadComplex<F>(&data[i0]);
  auto x2 = LoadComplex<F>(&data[i1]);
  auto x1 = LoadComplex<F>(&data[i2]);
  auto x3 = LoadComplex<F>(&data[i3]);

  Radix4Butterfly<D>(x0, x1, x2, x3, w1, w2, w3);

  StoreComplex(&data[i0], x0);
  StoreComplex(&data[i1], x1);
  StoreComplex(&data[i2], x2);
  StoreComplex(&data[i3], x3);
}


template <Dir D, std::floating_point F>
FORCE_INLINE void RunFirstContiguousButterflySetMin2(const Complex<F> *table, Complex<F> *data)
{
  if constexpr (SIMDComplexCount<F> == 1)
  {
    DoNoRotationButterflySet<D>(data, 0, 1);
    DoFixed45ButterflySet<D>(data, 4, 1);
    std::ignore = table;
  }
  else
  {
    SIMDComplex<F> x0, x1, x2, x3;
    SIMDComplex<F> w1, w2, w3;
    LoadContiguousRadix4Set(data, x0, x1, x2, x3);
    LoadFirstContiguousAngles<F>(table, w1, w2, w3);
    Radix4Butterfly<D>(x0, x1, x2, x3, w1, w2, w3);
    StoreContiguousRadix4Set(data, x0, x1, x2, x3);
  }
}

template <Dir D, std::floating_point F>
FORCE_INLINE void RunContiguousButterflySetMin2(const Complex<F> *table, Complex<F> *data, std::size_t groupIndex)
{
  if constexpr (SIMDComplexCount<F> == 1)
  {
    auto w1 = table[groupIndex << 1];
    auto w2 = table[groupIndex];
    auto w3 = table[(groupIndex << 1) + 1];
    DoMainButterfly<D>(data, 0u, 1u, w1, w2, w3);

    w1 = table[(groupIndex << 1) + 2];
    w2 *= -II;
    w3 = table[(groupIndex << 1) + 3];
    DoMainButterfly<D>(data, 4u, 1u, w1, w2, w3);
  }
  else
  {
    SIMDComplex<F> x0, x1, x2, x3;
    SIMDComplex<F> w1, w2, w3;
    LoadContiguousRadix4Set(data, x0, x1, x2, x3);
    LoadContiguousAngles<F>(table, groupIndex, w1, w2, w3);
    Radix4Butterfly<D>(x0, x1, x2, x3, w1, w2, w3);
    StoreContiguousRadix4Set(data, x0, x1, x2, x3);
  }
}

template <std::floating_point F>
FORCE_INLINE void DoNoRotationRadix2Butterfly(
  Complex<F> *data,
  std::size_t baseIndex,
  std::size_t butterflyStep)
{
  auto i0 = baseIndex;
  auto i1 = i0 + butterflyStep;

  auto even = LoadComplex<F>(&data[i0]);
  auto odd = LoadComplex<F>(&data[i1]);

  StoreComplex(&data[i0], even + odd);
  StoreComplex(&data[i1], even - odd);
}


template <std::floating_point F>
FORCE_INLINE void RescaleForIFFT(Complex<F> *data, std::size_t fftLength, std::size_t dataOffset, std::size_t spanLength)
{
F scale = F(1.0) / F(fftLength);
for (auto i = dataOffset; i < dataOffset + spanLength; i += SIMDComplexCount<F>)
{ 
  auto d = LoadComplex(data + i);
  StoreComplex(data + i, d * scale);
}
}
template <Dir D, std::floating_point F, std::uint32_t NativeComplexCount>
FORCE_INLINE void RunMultipleNoRotationButterflies(Complex<F> *data, std::uint32_t groupBase, std::uint32_t butterflyStep)
{
  for (auto butterflyIndex = 0u; butterflyIndex < butterflyStep; butterflyIndex += NativeComplexCount)
    { DoNoRotationButterflySet<D, F>(data, groupBase + butterflyIndex, butterflyStep); }
}

template <Dir D, std::floating_point F, std::uint32_t NativeComplexCount>
FORCE_INLINE void RunMultipleFixed45Butterflies(Complex<F> *data, std::uint32_t groupBase, std::uint32_t butterflyStep)
{
  for (auto butterflyIndex = 0u; butterflyIndex < butterflyStep; butterflyIndex += NativeComplexCount)
    { DoFixed45ButterflySet<D, F>(data, groupBase + butterflyIndex, butterflyStep); }
}


template <Dir D, std::floating_point F, std::uint32_t NativeComplexCount>
FORCE_INLINE void RunMultipleMainButterflies(const Complex<F> *table, Complex<F> *data, std::size_t groupIndex, std::size_t butterflyStep)
{
  auto groupBase = groupIndex * butterflyStep * 4;
  auto w1 = table[groupIndex << 1];
  auto w2 = table[groupIndex];
  auto w3 = table[(groupIndex << 1) + 1];
  for (auto butterflyIndex = 0u; butterflyIndex < butterflyStep; butterflyIndex += NativeComplexCount)
    { DoMainButterfly<D, F>(data, groupBase + butterflyIndex, butterflyStep, w1, w2, w3); }

  // Next group is a 45 degree rotation from the current one.
  w1 = table[(groupIndex  << 1) + 2];
  w2 *= -II; // This 90 degree rotate without using Rot is fine
  w3 = table[(groupIndex << 1) + 3];

  groupBase += butterflyStep * 4;
  for (auto butterflyIndex = 0u; butterflyIndex < butterflyStep; butterflyIndex += NativeComplexCount)
    { DoMainButterfly<D, F>(data, groupBase + butterflyIndex, butterflyStep, w1, w2, w3); }
}


template <Dir D, std::floating_point F, std::uint32_t NativeComplexCount>
FORCE_INLINE void RunFirstPartialFFT(const Complex<F> *table, const Complex<F> *input, Complex<F> *output, std::size_t fftLength, std::size_t spanLength)
{
  constexpr auto ComplexCountPerSet = std::max(2u, NativeComplexCount);

  PartialBitReverseCopy(input, output, fftLength, 0u, spanLength);

  if (D == Dir::Inverse)
    { RescaleForIFFT(output, fftLength, 0u, spanLength); }

  {
    RunFirstContiguousButterflySetMin2<D>(table, output);

    // Now iterate the rest of the groups two at a time
    for (auto groupIndex = ComplexCountPerSet; groupIndex < spanLength / 4; groupIndex += ComplexCountPerSet)
      { RunContiguousButterflySetMin2<D>(table, output + groupIndex * 4, groupIndex); }
  }

  for (auto butterflySpan = 16u; butterflySpan < spanLength; butterflySpan *= 4)
  {
    // First group has no rotation
    RunMultipleNoRotationButterflies<D, F, NativeComplexCount>(output, 0u, butterflySpan / 4);

    // Due to the bit-reversed ordering, the second group is a pure 45 degree rotation
    RunMultipleFixed45Butterflies<D, F, NativeComplexCount>(output, butterflySpan, butterflySpan / 4);

    // Now iterate the rest of the groups two at a time
    for (auto groupIndex = 2u; groupIndex < spanLength / butterflySpan; groupIndex += 2)
      { RunMultipleMainButterflies<D, F, NativeComplexCount>(table, output, groupIndex, butterflySpan / 4); }
  }
}

template <Dir D, std::floating_point F, std::uint32_t NativeComplexCount>
FORCE_INLINE void RunPartialFFT(const Complex<F> *table, const Complex<F> *input, Complex<F> *output, std::size_t fftLength, std::size_t dataOffset, std::size_t spanLength)
{
  constexpr auto ComplexCountPerSet = std::max(2u, NativeComplexCount);

  PartialBitReverseCopy(input, output, fftLength, dataOffset, dataOffset + spanLength);

  if (D == Dir::Inverse)
    { RescaleForIFFT(output, fftLength, dataOffset, spanLength); }

  for (auto groupIndex = dataOffset / 4u; groupIndex < (dataOffset + spanLength) / 4u; groupIndex += ComplexCountPerSet)
    { RunContiguousButterflySetMin2<D>(table, output + groupIndex * 4, groupIndex); }

  for (auto butterflySpan = 16u; butterflySpan < spanLength; butterflySpan *= 4)
  {
    for (auto groupIndex = dataOffset / butterflySpan; groupIndex < (dataOffset + spanLength) / butterflySpan; groupIndex += 2)
      { RunMultipleMainButterflies<D, F, NativeComplexCount>(table, output, groupIndex, butterflySpan / 4); }
  }
}


template <Dir D, std::floating_point F, std::uint32_t NativeComplexCount>
void RunFFT(const Complex<F> *table, const Complex<F> *input, Complex<F> *output, std::size_t fftLength)
{
  // Too-small FFTs should be done as scalar.
  if (fftLength <= 8)
  {
    // !!! Hard-coded small FFTs

  }

  constexpr auto MaxSpanLength = std::size_t(1024);
  if (fftLength > MaxSpanLength)
  {
    RunFirstPartialFFT<D, F, NativeComplexCount>(table, input, output, fftLength, MaxSpanLength);

    for (auto offset = MaxSpanLength; offset < fftLength; offset += MaxSpanLength)
      { RunPartialFFT<D, F, NativeComplexCount>(table, input, output, fftLength, offset, MaxSpanLength); }

    // Now the partials are done, the last steps span farther so just let them roll
    for (auto butterflySpan = MaxSpanLength; butterflySpan < fftLength; butterflySpan *= 4)
    {
      for (auto groupIndex = 0u; groupIndex < fftLength / butterflySpan; groupIndex += 2)
        { RunMultipleMainButterflies<D, F, NativeComplexCount>(table, output, groupIndex, butterflySpan / 4); }
    }
  }
  else
  {
    // We just need to run one that is the full size of the
    RunFirstPartialFFT<D, F, NativeComplexCount>(table, input, output, fftLength, fftLength);
  }


  if (const bool isPowerOfFour = ((fftLength & 0x55555555) != 0); isPowerOfFour)
  {
    // Last power of 4
    for (auto butterflyIndex = 0u; butterflyIndex < fftLength / 4; butterflyIndex += NativeComplexCount)
      { DoNoRotationButterflySet<D>(output, butterflyIndex, fftLength / 4); }
  }
  else
  {      // One last Radix 2 step
    auto butterflySpan = fftLength;
    for (auto butterflyIndex = 0u; butterflyIndex < butterflySpan / 2; butterflyIndex += NativeComplexCount)
      { DoNoRotationRadix2Butterfly(output, butterflyIndex, butterflySpan / 2); }
  }
}
