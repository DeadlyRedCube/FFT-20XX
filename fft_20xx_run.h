template <std::floating_point F, std::uint32_t ComplexCountPerStep>
FORCE_INLINE void RunFirstPartialFFT(const Complex<F> *input, Complex<F> *output, std::size_t fftLength, std::size_t spanLength)
{
  PartialBitReverseCopyAllowInPlaceSIMD(input, output, fftLength, 0u, spanLength);

  {
    DoFirstContiguousButterfly(output);

    // Now iterate the rest of the groups two at a time
    for (auto groupIndex = ComplexCountPerStep; groupIndex < spanLength / 4; groupIndex += ComplexCountPerStep)
      { DoContiguousButterfly(output + groupIndex * 4, groupIndex); }
  }

  // Now do all but the very last iteration
  for (auto butterflySpan = 16u; butterflySpan < spanLength; butterflySpan *= 4)
  {
    // First group has no rotation
    RunMultipleNoRotationButterflies<F>(output, 0u, butterflySpan / 4);

    // Due to the bit-reversed ordering, the second group is a pure 45 degree rotation
    RunMultipleFixed45Butterflies<F>(output, butterflySpan, butterflySpan / 4);

    // Now iterate the rest of the groups two at a time
    for (auto groupIndex = 2u; groupIndex < spanLength / butterflySpan; groupIndex += 2)
      { RunMultipleMainButterflies(output, groupIndex, butterflySpan / 4); }
  }
}

template <std::floating_point F, std::uint32_t NativeComplexCount>
FORCE_INLINE void RunMultipleNoRotationButterflies(Complex<F> *data, std::uint32_t groupBase, std::uint32_t butterflyStep)
{
  for (auto butterflyIndex = 0u; butterflyIndex < butterflyStep; butterflyIndex += NativeComplexCount)
    { DoNoRotationButterfly<F>(data, groupBase + butterflyIndex, butterflyStep); }
}

template <std::floating_point F, std::uint32_t NativeComplexCount>
FORCE_INLINE void RunMultipleFixed45Butterflies(Complex<F> *data, std::uint32_t groupBase, std::uint32_t butterflyStep)
{
  for (auto butterflyIndex = 0u; butterflyIndex < butterflyStep; butterflyIndex += NativeComplexCount)
    { DoFixed45Butterfly<F>(data, groupBase + butterflyIndex, butterflyStep); }
}


template <std::floating_point F, std::uint32_t NativeComplexCount>
FORCE_INLINE void RunFirstPartialFFT(const Complex<F> *table, const Complex<F> *input, Complex<F> *output, std::size_t fftLength, std::size_t spanLength)
{
  constexpr auto ComplexCountPerSet = std::max(2u, NativeComplexCount);

  PartialBitReverseCopy(input, output, fftLength, 0u, spanLength);

  {
    RunFirstContiguousButterflySet(table, output);

    // Now iterate the rest of the groups two at a time
    for (auto groupIndex = ComplexCountPerSet; groupIndex < spanLength / 4; groupIndex += ComplexCountPerSet)
      { RunContiguousButterflySet(table, output + groupIndex * 4, groupIndex); }
  }

  for (auto butterflySpan = 16u; butterflySpan < spanLength; butterflySpan *= 4)
  {
    // First group has no rotation
    RunMultipleNoRotationButterflies<F, NativeComplexCount>(output, 0u, butterflySpan / 4);

    // Due to the bit-reversed ordering, the second group is a pure 45 degree rotation
    RunMultipleFixed45Butterflies<F, NativeComplexCount>(output, butterflySpan, butterflySpan / 4);

    // Now iterate the rest of the groups two at a time
    for (auto groupIndex = 2u; groupIndex < spanLength / butterflySpan; groupIndex += 2)
      { RunMultipleMainButterflies(table, output, groupIndex, butterflySpan / 4); }
  }
}

template <std::floating_point F, std::uint32_t NativeComplexCount>
FORCE_INLINE void RunPartialFFT(const Complex<F> *table, const Complex<F> *input, Complex<F> *output, std::size_t fftLength, std::size_t dataOffset, std::size_t spanLength)
{
  constexpr auto ComplexCountPerSet = std::max(2u, NativeComplexCount);

  PartialBitReverseCopy(input, output, fftLength, dataOffset, dataOffset + spanLength);

  for (auto groupIndex = dataOffset / 4u; groupIndex < (dataOffset + spanLength) / 4u; groupIndex += ComplexCountPerSet)
    { RunContiguousButterflySet(table, output + groupIndex * 4, groupIndex); }

  for (auto butterflySpan = 16u; butterflySpan < spanLength; butterflySpan *= 4)
  {
    for (auto groupIndex = dataOffset / butterflySpan; groupIndex < (dataOffset + spanLength) / butterflySpan; groupIndex += 2)
      { RunMultipleMainButterflies(table, output, groupIndex, butterflySpan / 4); }
  }
}

template <std::floating_point F, std::uint32_t NativeComplexCount>
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
    RunFirstPartialFFT<F, NativeComplexCount>(table, input, output, fftLength, MaxSpanLength);

    for (auto offset = MaxSpanLength; offset < fftLength; offset += MaxSpanLength)
      { RunPartialFFT<F, NativeComplexCount>(table, input, output, fftLength, offset, MaxSpanLength); }

    // Now the partials are done, the last steps span farther so just let them roll
    for (auto butterflySpan = MaxSpanLength; butterflySpan < fftLength; butterflySpan *= 4)
    {
      for (auto groupIndex = 0u; groupIndex < fftLength / butterflySpan; groupIndex += 2)
        { RunMultipleMainButterflies(table, output, groupIndex, butterflySpan / 4); }
    }
  }
  else
  {
    // We just need to run one that is the full size of the
    RunFirstPartialFFT<F, NativeComplexCount>(table, input, output, fftLength, fftLength);
  }


  if (const bool isPowerOfFour = ((fftLength & 0x55555555) != 0); isPowerOfFour)
  {
    // Last power of 4
    for (auto butterflyIndex = 0u; butterflyIndex < fftLength / 4; butterflyIndex += NativeComplexCount)
      { DoNoRotationButterfly(output, butterflyIndex, fftLength / 4); }
  }
  else
  {      // One last Radix 2 step
    auto butterflySpan = fftLength;
    for (auto butterflyIndex = 0u; butterflyIndex < butterflySpan / 2; butterflyIndex += NativeComplexCount)
      { DoNoRotationRadix2Butterfly(output, butterflyIndex, butterflySpan / 2); }
  }
}
