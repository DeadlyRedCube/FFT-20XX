#pragma once 

template <Dir D, std::floating_point F, std::uint32_t NativeComplexCount>
FORCE_INLINE void RunMultipleNoRotationButterflies(Complex<F> *data, std::uint32_t groupBase, std::uint32_t butterflyStep)
{
  for (auto butterflyIndex = 0u; butterflyIndex < butterflyStep; butterflyIndex += NativeComplexCount)
    { DoNoRotationButterfly<D, F>(data, groupBase + butterflyIndex, butterflyStep); }
}

template <Dir D, std::floating_point F, std::uint32_t NativeComplexCount>
FORCE_INLINE void RunMultipleFixed45Butterflies(Complex<F> *data, std::uint32_t groupBase, std::uint32_t butterflyStep)
{
  for (auto butterflyIndex = 0u; butterflyIndex < butterflyStep; butterflyIndex += NativeComplexCount)
    { DoFixed45Butterfly<D, F>(data, groupBase + butterflyIndex, butterflyStep); }
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
    RunFirstContiguousButterflySet<D>(table, output);

    // Now iterate the rest of the groups two at a time
    for (auto groupIndex = ComplexCountPerSet; groupIndex < spanLength / 4; groupIndex += ComplexCountPerSet)
      { RunContiguousButterflySet<D>(table, output + groupIndex * 4, groupIndex); }
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
    { RunContiguousButterflySet<D>(table, output + groupIndex * 4, groupIndex); }

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
      { DoNoRotationButterfly<D>(output, butterflyIndex, fftLength / 4); }
  }
  else
  {      // One last Radix 2 step
    auto butterflySpan = fftLength;
    for (auto butterflyIndex = 0u; butterflyIndex < butterflySpan / 2; butterflyIndex += NativeComplexCount)
      { DoNoRotationRadix2Butterfly(output, butterflyIndex, butterflySpan / 2); }
  }
}
