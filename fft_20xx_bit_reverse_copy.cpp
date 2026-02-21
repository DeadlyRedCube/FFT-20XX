#include "fft_20xx_common.h"

#include "include/fft_20xx.h"



#include <algorithm>
#include <cinttypes>
#include <ranges>
#include "fft_20xx_complex.h"
#include "fft_20xx_math.h"


namespace FFT20XX
{
  static constexpr auto MaxSizeTBit = size_t(std::numeric_limits<std::make_signed_t<size_t>>::min());

  template <size_t FFTSize, size_t TopBit, size_t BottomBit, std::floating_point F> 
  FORCE_INLINE void BitReverseCopyDisjoint(
    const Complex<F> *inputA, 
    const Complex<F> *inputB, 
    Complex<F> *outputA, 
    Complex<F> *outputB, 
    uint32_t shift)
  {
    auto i0 = size_t(0);
    auto i1 = TopBit >> shift;
    auto i2 = BottomBit;
    auto i3 = i1 | i2;

    if constexpr (BottomBit < (size_t(1) << ((Log2Floor(FFTSize) - 1) / 2)))
    {
      // recurse
      constexpr auto T2 = TopBit >> 1;
      constexpr auto B2 = BottomBit << 1;
      BitReverseCopyDisjoint<FFTSize, T2, B2, F>(inputA + i0, inputB + i0, outputA + i0, outputB + i0, shift);
      BitReverseCopyDisjoint<FFTSize, T2, B2, F>(inputA + i1, inputB + i2, outputA + i1, outputB + i2, shift);
      BitReverseCopyDisjoint<FFTSize, T2, B2, F>(inputA + i2, inputB + i1, outputA + i2, outputB + i1, shift);
      BitReverseCopyDisjoint<FFTSize, T2, B2, F>(inputA + i3, inputB + i3, outputA + i3, outputB + i3, shift);
    }
    else if constexpr(IsPowerOfFourAssumingIsPowerOfTwo(FFTSize))
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

  template <size_t FFTSize, size_t TopBit, size_t BottomBit, std::floating_point F> 
  FORCE_INLINE void BitReverseCopyInPosition(const Complex<F> *input, Complex<F> *output, uint32_t shift)
  {
    const auto a0 = size_t(0);
    const auto a1 = TopBit >> shift;
    const auto a2 = BottomBit;
    const auto a3 = a1 | a2;

    if constexpr (BottomBit < (size_t(1) << ((Log2Floor(FFTSize) - 1) / 2)))
    {
      // recurse
      constexpr auto T2 = TopBit >> 1;
      constexpr auto B2 = BottomBit << 1;
      // swap anything where neither top or bottom bit is set
      BitReverseCopyInPosition<FFTSize, T2, B2, F>(input + a0, output + a0, shift);
      BitReverseCopyDisjoint<FFTSize, T2, B2, F>(
        input + a1, 
        input + a2,
        output + a1,
        output + a2, 
        shift);
      BitReverseCopyInPosition<FFTSize, T2, B2, F>(input + a3, output + a3, shift);
    }
    else if constexpr (IsPowerOfFourAssumingIsPowerOfTwo(FFTSize))
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


  template <size_t FFTSize, std::floating_point F> 
  void BitReverseCopyDisjoint(
      const Complex<F> *inputA, 
      const Complex<F> *inputB, 
      Complex<F> *outputA, 
      Complex<F> *outputB, 
      uint32_t shift)
    { BitReverseCopyDisjoint<FFTSize, MaxSizeTBit, 1>(inputA, inputB, outputA, outputB, shift); }

  template <size_t FFTSize, std::floating_point F> 
  void BitReverseCopyInPosition(const Complex<F> *input, Complex<F> *output, uint32_t shift)
    { BitReverseCopyInPosition<FFTSize, MaxSizeTBit, 1>(input, output, shift); }


  template <size_t FFTLength, std::floating_point F>
  void BitReverseCopyFixed(const Complex<F> *input, Complex<F> *output)
  {
    constexpr auto BitCount = Log2Floor(FFTLength);
    constexpr auto Shift = BitReverseShiftFromBitCount<size_t>(BitCount);
    return BitReverseCopyInPosition<FFTLength, MaxSizeTBit, 1>(input, output, Shift);
  }


  template <std::floating_point F>
  void BitReverseCopy(const Complex<F> *input, Complex<F> *output, std::size_t fftLength)
  {
    // Measured 16, 64, 256, and 1024 on my machine and 64 has the overall best performance (slightly slower than 1k at low sizes
    constexpr auto SubFFTLength = size_t(64);
    constexpr auto SubFFTBitCount = Log2Floor(SubFFTLength);
    constexpr auto SubFFTDim = size_t(1) << (SubFFTBitCount / 2);
    static_assert(SubFFTLength >= 16);
    static_assert(IsPowerOfTwo(SubFFTLength));
    static_assert(IsPowerOfFourAssumingIsPowerOfTwo(SubFFTLength));

    switch (fftLength)
    {
    case 1: output[0] = input[0]; return;
    case 2: return BitReverseCopyFixed<2>(input, output);
    case 4: return BitReverseCopyFixed<4>(input, output);
    case 8: return BitReverseCopyFixed<8>(input, output);
    case 16: return BitReverseCopyFixed<16>(input, output);
    case 32: return BitReverseCopyFixed<32>(input, output);
    case 64: return BitReverseCopyFixed<64>(input, output);
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

        const auto fftBitCount = Log2Floor(fftLength);
        const auto fullDim = size_t(1) << (fftBitCount / 2);
        const auto fullShift = BitReverseShiftFromBitCount<size_t>(fftBitCount);
        constexpr auto LowShift = SubFFTBitCount / 2;
        const auto highShift = fullShift + LowShift;

        while (true)
        {
          #if !CPU_ARM_ANY
            auto iRevUnshifted = size_t(0);
          #endif
          for (auto i = size_t(0); i < fullDim; i += SubFFTDim)
          {
            #if CPU_ARM_ANY
              auto iRevUnshifted = BitReverse(i);
            #endif
            
            auto iRev = iRevUnshifted >> highShift;
            {
              auto offset = i + iRev;
              BitReverseCopyInPosition<SubFFTLength>(
                input + offset, 
                output + offset, 
                fullShift); 
            }

            #if !CPU_ARM_ANY
              auto jRevUnshifted = size_t(0);
            #endif

            for (auto j = size_t(0); j < i; j += SubFFTDim)
            {
              #if CPU_ARM_ANY
                auto jRevUnshifted = BitReverse(j);
              #endif
              auto jRev = jRevUnshifted >> highShift;
              auto aOffset = i + jRev;
              auto bOffset = j + iRev;
              BitReverseCopyDisjoint<SubFFTLength>(
                input + aOffset,
                input + bOffset,
                output + aOffset,
                output + bOffset,
                fullShift);
              
              #if !CPU_ARM_ANY
                jRevUnshifted = BitReverseIncrement(jRevUnshifted);
              #endif
            }
            
            #if !CPU_ARM_ANY
              iRevUnshifted = BitReverseIncrement(iRevUnshifted);
            #endif
          }

          if (IsPowerOfFourAssumingIsPowerOfTwo(fftLength))
            { break; }

          fftLength >>= 1;
          input += fullDim;
          output += fullDim;
        }
      }
      break;
    }

  }


  template void BitReverseCopy(const Complex<float> *input, Complex<float> *output, std::size_t fftLength);
  template void BitReverseCopy(const Complex<double> *input, Complex<double> *output, std::size_t fftLength);
}
