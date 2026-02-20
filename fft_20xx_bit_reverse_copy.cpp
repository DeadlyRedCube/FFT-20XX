#include "fft_20xx_common.h"

#include "include/fft_20xx.h"



#include <algorithm>
#include <cinttypes>
#include <ranges>
#include "fft_20xx_complex.h"
#include "fft_20xx_math.h"


#define USE_ARM_RBIT CPU_ARM_ANY

#define NEW_COPY 1
namespace FFT20XX
{
  #if NEW_COPY
    template <size_t TopBit, size_t BottomBit, std::floating_point F> 
    FORCE_INLINE void BitReverseCopyTemplatedDisjoint(
      const Complex<F> *inputA, 
      const Complex<F> *inputB, 
      Complex<F> *outputA, 
      Complex<F> *outputB, 
      uint32_t upperShift)
    {
      auto i0 = size_t(0);
      auto i1 = (TopBit << upperShift);
      auto i2 = BottomBit;
      auto i3 = i1 | i2;

      if constexpr (TopBit / BottomBit > 2)
      {
        // recurse
        constexpr auto T2 = TopBit >> 1;
        constexpr auto B2 = BottomBit << 1;
        BitReverseCopyTemplatedDisjoint<T2, B2, F>(inputA + i0, inputB + i0, outputA + i0, outputB + i0, upperShift);
        BitReverseCopyTemplatedDisjoint<T2, B2, F>(inputA + i1, inputB + i2, outputA + i1, outputB + i2, upperShift);
        BitReverseCopyTemplatedDisjoint<T2, B2, F>(inputA + i2, inputB + i1, outputA + i2, outputB + i1, upperShift);
        BitReverseCopyTemplatedDisjoint<T2, B2, F>(inputA + i3, inputB + i3, outputA + i3, outputB + i3, upperShift);
      }
      else if constexpr(TopBit != BottomBit)
      {
        auto t = inputA[i0];
        outputA[i0] = inputB[i0];
        outputB[i0] = t;
        t = inputA[i3];
        outputA[i3] = inputB[i3];
        outputB[i3] = t;
        t = inputA[i1];
        outputA[i1] = inputB[i2];
        outputB[i2] = t;
        t = inputA[i2];
        outputA[i2] = inputB[i1];
        outputB[i1] = t;
      }
      else
      {
        auto t = inputA[i0];
        outputA[i0] = inputB[i0];
        outputB[i0] = t;
        t = inputA[i2];
        outputA[i2] = inputB[i2];
        outputB[i2] = t;
      }
    }

    template <size_t TopBit, size_t BottomBit, std::floating_point F> 
    FORCE_INLINE void BitReverseCopyTemplatedInPosition(const Complex<F> *input, Complex<F> *output, uint32_t upperShift)
    {
      const auto a0 = size_t(0);
      const auto a1 = TopBit << upperShift;
      const auto a2 = BottomBit;
      const auto a3 = a1 | a2;

      if constexpr (TopBit / BottomBit > 2)
      {
        // recurse
        constexpr auto T2 = TopBit >> 1;
        constexpr auto B2 = BottomBit << 1;
        // swap anything where neither top or bottom bit is set
        BitReverseCopyTemplatedInPosition<T2, B2, F>(input + a0, output + a0, upperShift);
        BitReverseCopyTemplatedDisjoint<T2, B2, F>(
          input + a1, 
          input + a2,
          output + a1,
          output + a2, 
          upperShift);
        BitReverseCopyTemplatedInPosition<T2, B2, F>(input + a3, output + a3, upperShift);
      }
      else if constexpr (TopBit > BottomBit)
      {
        output[a0] = input[a0];
        auto t = input[a1];
        output[a1] = input[a2];
        output[a2] = t;
        output[a3] = input[a3];
      }
      else
      {
        output[a0] = input[a0];
        output[a2] = input[a2];
      }
    }
  #endif


  template <std::floating_point F>
  void BitReverseCopy(const Complex<F> *input, Complex<F> *output, std::size_t fftLength)
  {
    #if NEW_COPY
      // Measured 16, 64, 256, and 1024 on my machine and 64 has the overall best performance (slightly slower than 1k at low sizes
      static constexpr auto SubFFTLength = size_t(64);
      static_assert(SubFFTLength >= 16);
      static_assert(IsPowerOfTwo(SubFFTLength));
      static_assert(IsPowerOfFourAssumingIsPowerOfTwo(SubFFTLength));

      switch (fftLength)
      {
      case 1: output[0] = input[0]; return;
      case 2: return BitReverseCopyTemplatedInPosition<1, 1>(input, output, 0);
      case 4: return BitReverseCopyTemplatedInPosition<2, 1>(input, output, 0);
      case 8: return BitReverseCopyTemplatedInPosition<4, 1>(input, output, 0);
      case 16: return BitReverseCopyTemplatedInPosition<8, 1>(input, output, 0);
      case 32:
        if constexpr (32 <= SubFFTLength)
          { return BitReverseCopyTemplatedInPosition<16, 1>(input, output, 0); }
      case 64:
        if constexpr (64 <= SubFFTLength)
          { return BitReverseCopyTemplatedInPosition<32, 1>(input, output, 0); }
        [[fallthrough]];
      case 128:
        if constexpr (128 <= SubFFTLength)
          { return BitReverseCopyTemplatedInPosition<64, 1>(input, output, 0); }
        [[fallthrough]];
      case 256:
        if constexpr (256 <= SubFFTLength)
          { return BitReverseCopyTemplatedInPosition<128, 1>(input, output, 0); }
        [[fallthrough]];
      case 512:
        if constexpr (512 <= SubFFTLength)
          { return BitReverseCopyTemplatedInPosition<256, 1>(input, output, 0); }
        [[fallthrough]];
      case 1024:
        if constexpr (1024 <= SubFFTLength)
          { return BitReverseCopyTemplatedInPosition<512, 1>(input, output, 0); }
        [[fallthrough]];
      default:
        {
          #if 1
          constexpr auto CopyThreshold = 2048u;
          if (input != output && fftLength > CopyThreshold)
          { 
            // It's faster to blast through a copy from input to output first, then do the reversing.
            std::ranges::copy(input, input + fftLength, output); 
            input = output;
          }
          #endif

          constexpr auto SubTopBit = SubFFTLength / 2;
          constexpr auto MaxSubFFTLengthBitCount = Log2Floor(SubFFTLength);
          constexpr auto MiddleShift = MaxSubFFTLengthBitCount / 2;
          const auto fftBitCount = Log2Floor(fftLength);
          const auto middleBitCount = fftBitCount - MaxSubFFTLengthBitCount;

          const auto middleCount = size_t(1) << middleBitCount;
          auto revUnshifted = size_t(0);
          const auto revShift = BitReverseShiftFromBitCount<size_t>(middleBitCount);

          for (auto i = size_t(0); i < middleCount; i++)
          {
            auto rev = revUnshifted >> revShift;

            if (i < rev)
            { 
              auto aOffset = i << MiddleShift;
              auto bOffset = rev << MiddleShift;
              BitReverseCopyTemplatedDisjoint<SubTopBit, 1>(
                input + aOffset,
                input + bOffset,
                output + aOffset,
                output + bOffset,
                middleBitCount);
            }
            else if (i == rev)
            { 
              auto offset = i<< MiddleShift;
              BitReverseCopyTemplatedInPosition<SubTopBit, 1>(
                input + offset, 
                output + offset, 
                middleBitCount); 
            }

            revUnshifted = BitReverseIncrement(revUnshifted);
          }
        }
        break;
      }
    #else
      const auto bitCount = Log2Floor(fftLength);

      if (input != output)
      {
        constexpr auto LowCutoffFFTLength = 8192 * sizeof(double) / sizeof(F);
        constexpr auto HighCutoffFFTLength = 512*1024 * sizeof(double) / sizeof(F);

        if (fftLength >= HighCutoffFFTLength)
        {
          // Above this threshold it is more efficient to blast the memory from input to output then do the in-place
          //  reordering rather than doing it straight from input...but only do the copy on the initial index.
          std::ranges::copy(input, input + fftLength, output);
          input = output;
        }
        else if (fftLength >= LowCutoffFFTLength)
        {
          // Otherwise, above this threshold it's faster to do an element-by-element copy from input to output.

          #if !USE_ARM_RBIT
            auto revUnshifted = size_t(0);
          #endif
          const auto revShift = BitReverseShiftFromBitCount<std::size_t>(bitCount);
          for (auto i = size_t(0); i < fftLength; i++)
          {
            #if USE_ARM_RBIT
              std::size_t rev;
              asm("rbit %0,%1" : "=r"(rev) : "r"(i));
              rev >>= revShift;
            #else
              auto rev = revUnshifted >> revShift;
              revUnshifted = BitReverseIncrement(revUnshifted);
            #endif

            output[i] = input[rev];
          }
          return;
        }
      }

      const auto iterationCount = std::size_t(1u) << (bitCount / 2);

      const auto bitReverseShift = BitReverseShiftFromBitCount<std::size_t>(bitCount);
      if ((bitCount & 1) == 0)
      {
        // This is a power-of-four FFT so there's no center bit.

        // Iterate through half the bits
        #if !USE_ARM_RBIT
          auto reversedBaseValueUnshifted = size_t(0);
        #endif
        for (auto baseValue = size_t(0); baseValue < iterationCount; baseValue++)
        {
          // Iterate through the bitValues until we reach ones where the reversed value is the same as the non-reversed
          //  (a binary palindrome), at which point we stop.

          #if USE_ARM_RBIT
            std:: size_t ri;
            asm("rbit %0,%1" : "=r"(ri) : "r"(baseValue));
            ri >>= bitReverseShift;
          #else
            auto iUnshifted = baseValue << bitReverseShift;
            auto ri = reversedBaseValueUnshifted >> bitReverseShift;
          #endif
          auto target = ri | baseValue;
          while (ri < target)
          {
            #if USE_ARM_RBIT
              std::size_t i;
              asm("rbit %0,%1" : "=r"(i) : "r"(ri));
              i >>= bitReverseShift;
            #else
              auto i = iUnshifted >> bitReverseShift;
              iUnshifted = BitReverseIncrement(iUnshifted);
            #endif

            auto t = input[i]; // Store input[i] as a temp in case this is an in-place swap
            output[i] = input[ri];
            output[ri] = t;

            ri++;
          }

          // This value doesn't move but copy it just in case this isn't an in-place
          output[ri] = input[ri];

          #if !USE_ARM_RBIT
            reversedBaseValueUnshifted = BitReverseIncrement(reversedBaseValueUnshifted);
          #endif
        }
      }
      else
      {
        // This is a power-of-two FFT so there's a center bit.

        // Iterate through half the bits
        auto reversedBaseValueUnshifted = size_t(0);
        for (auto baseValue = size_t(0); baseValue < iterationCount; baseValue++)
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
    #endif
  }


  template void BitReverseCopy(const Complex<float> *input, Complex<float> *output, std::size_t fftLength);
  template void BitReverseCopy(const Complex<double> *input, Complex<double> *output, std::size_t fftLength);
}
