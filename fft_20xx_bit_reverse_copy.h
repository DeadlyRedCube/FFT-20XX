#pragma once

#include <algorithm>
#include <cinttypes>
#include <ranges>
#include "fft_20xx_complex.h"
#include "fft_20xx_math.h"



namespace FFT20XX
{
  template <std::floating_point F>
  void PartialBitReverseCopy(
    const Complex<F> *input,
    Complex<F> *output,
    std::size_t fftLength,
    std::size_t startIndex,
    std::size_t endIndex)
  {
    const auto bitCount = Log2Floor(fftLength);

    if (input != output)
    {
      constexpr auto LowCutoffFFTLength = 8192 * sizeof(double) / sizeof(F);
      constexpr auto HighCutoffFFTLength = 512*1024 * sizeof(double) / sizeof(F);

      if (fftLength >= HighCutoffFFTLength)
      {
        // Above this threshold it is more efficient to blast the memory from input to output then do the in-place
        //  reordering rather than doing it straight from input...but only do the copy on the initial index.
        if (startIndex == 0)
          { std::ranges::copy(input, input + fftLength, output); }
        input = output;
      }
      else if (fftLength >= LowCutoffFFTLength)
      {
        // Otherwise, above this threshold it's faster to do an element-by-element copy from input to output.
        auto revUnshifted = BitReverse(startIndex);
        auto revShift = BitReverseShiftFromBitCount<std::size_t>(bitCount);
        for (auto i = startIndex; i < endIndex; i++)
        {
          output[i] = input[revUnshifted >> revShift];
          revUnshifted = BitReverseIncrement(revUnshifted);
        }
        return;
      }
    }

    const auto iterationCount = std::size_t(1u) << (bitCount / 2);

    // Already good!
    if (startIndex >= iterationCount)
      { return; }

    endIndex = std::min(endIndex, iterationCount);
    const auto bitReverseShift = BitReverseShiftFromBitCount<std::size_t>(bitCount);
    if ((bitCount & 1) == 0)
    {
      // This is a power-of-four FFT so there's no center bit.

      // Iterate through half the bits
      auto reversedBaseValueUnshifted = BitReverse(startIndex);
      for (auto baseValue = startIndex; baseValue < endIndex; baseValue++)
      {
        // Iterate through the bitValues until we reach ones where the reversed value is the same as the non-reversed
        //  (a binary palindrome), at which point we stop.

        auto iUnshifted = baseValue << bitReverseShift;
        auto ri = reversedBaseValueUnshifted >> bitReverseShift;
        auto target = ri | baseValue;
        while (ri < target)
        {
          auto i = iUnshifted >> bitReverseShift;
          auto t = input[i]; // Store input[i] as a temp in case this is an in-place swap
          output[i] = input[ri];
          output[ri] = t;

          ri++;
          iUnshifted = BitReverseIncrement(iUnshifted);
        }

        // This value doesn't move but copy it just in case this isn't an in-place
        output[ri] = input[ri];

        reversedBaseValueUnshifted = BitReverseIncrement(reversedBaseValueUnshifted);
      }
    }
    else
    {
      // This is a power-of-two FFT so there's a center bit.

      // Iterate through half the bits
      auto reversedBaseValueUnshifted = BitReverse(startIndex);
      for (auto baseValue = startIndex; baseValue < endIndex; baseValue++)
      {
        // Iterate through the bitValues until we reach ones where the reversed value is the same as the non-reversed
        //  (a binary palindrome), at which point we stop.

        auto iUnshifted = baseValue << bitReverseShift;
        auto ri = reversedBaseValueUnshifted >> bitReverseShift;
        auto target = ri | baseValue;
        do
        {
          auto i = iUnshifted >> bitReverseShift;
          auto t = input[i]; // Store input[i] as a temp in case this is an in-place swap
          output[i] = input[ri];
          output[ri] = t;

          t = input[i | iterationCount];
          output[i | iterationCount] = input[ri | iterationCount];
          output[ri | iterationCount] = t;

          ri++;
          iUnshifted = BitReverseIncrement(iUnshifted);
        } while (ri < target);

        // This value doesn't move but copy it just in case this isn't an in-place
        output[ri] = input[ri];
        output[ri | iterationCount] = input[ri | iterationCount];

        reversedBaseValueUnshifted = BitReverseIncrement(reversedBaseValueUnshifted);
      }
    }
  }


  template <std::floating_point F>
  void BitReverseCopy(const Complex<F> *input, Complex<F> *output, std::size_t fftLength)
    { PartialBitReverseCopy(input, output, fftLength, 0, fftLength); }
}
