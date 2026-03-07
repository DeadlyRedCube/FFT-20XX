#pragma once

#include "fft_20xx_complex.h"
#include "fft_20xx_math.h"


namespace FFT20XX
{
  static constexpr auto MaxSizeTBit = size_t(std::numeric_limits<std::make_signed_t<size_t>>::min());

  #if SIMD_AVX
    FORCE_INLINE void AVXTranspose4x4(
      const Complex<float> *r0In,
      const Complex<float> *r1In,
      const Complex<float> *r2In,
      const Complex<float> *r3In,
      __m256 &AEIM,
      __m256 &BFJN,
      __m256 &CGKO,
      __m256 &DHLP)
    {
      // Transpose
      //   A B C D
      //   E F G H
      //   I J K L
      //   M N O P
      //
      // to:
      //   A E I M
      //   B F J N
      //   C G K O
      //   D H L P
      auto AB = _mm_loadu_ps(reinterpret_cast<const float *>(r0In));
      auto CD = _mm_loadu_ps(reinterpret_cast<const float *>(r0In) + 4);
      auto EF = _mm_loadu_ps(reinterpret_cast<const float *>(r1In));
      auto GH = _mm_loadu_ps(reinterpret_cast<const float *>(r1In) + 4);
      auto IJ = _mm_loadu_ps(reinterpret_cast<const float *>(r2In));
      auto KL = _mm_loadu_ps(reinterpret_cast<const float *>(r2In) + 4);
      auto MN = _mm_loadu_ps(reinterpret_cast<const float *>(r3In));
      auto OP = _mm_loadu_ps(reinterpret_cast<const float *>(r3In) + 4);
      auto ABIJ = _mm256_insertf128_ps(_mm256_castps128_ps256(AB), IJ, 1);
      auto CDKL = _mm256_insertf128_ps(_mm256_castps128_ps256(CD), KL, 1);
      auto EFMN = _mm256_insertf128_ps(_mm256_castps128_ps256(EF), MN, 1);
      auto GHOP = _mm256_insertf128_ps(_mm256_castps128_ps256(GH), OP, 1);

      AEIM = _mm256_shuffle_ps(ABIJ, EFMN, _MM_SHUFFLE(1, 0, 1, 0));
      BFJN = _mm256_shuffle_ps(ABIJ, EFMN, _MM_SHUFFLE(3, 2, 3, 2));
      CGKO = _mm256_shuffle_ps(CDKL, GHOP, _MM_SHUFFLE(1, 0, 1, 0));
      DHLP = _mm256_shuffle_ps(CDKL, GHOP, _MM_SHUFFLE(3, 2, 3, 2));
    }

    FORCE_INLINE void AVXStoreRows(
      __m256 AEIM,
      __m256 BFJN,
      __m256 CGKO,
      __m256 DHLP,
      Complex<float> *r0Out,
      Complex<float> *r1Out,
      Complex<float> *r2Out,
      Complex<float> *r3Out)
    {
      _mm256_storeu_ps(reinterpret_cast<float *>(r0Out), AEIM);
      _mm256_storeu_ps(reinterpret_cast<float *>(r1Out), BFJN);
      _mm256_storeu_ps(reinterpret_cast<float *>(r2Out), CGKO);
      _mm256_storeu_ps(reinterpret_cast<float *>(r3Out), DHLP);
    }
  #endif

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
  { 
    #if SIMD_AVX
      if constexpr (FFTSize == 64 && std::same_as<F, float>)
      {
        constexpr auto RowOffsetUnshifted = BitReverse(size_t(2));
        constexpr auto XBlockOffset = size_t(4);
        constexpr auto YBlockOffsetUnshifted = BitReverse(size_t(4));

        auto rowOffset = RowOffsetUnshifted >> shift;
        auto yBlockOffset = YBlockOffsetUnshifted >> shift;

        auto b0 = size_t(0);
        auto b1 = XBlockOffset;
        auto b2 = yBlockOffset;
        auto b3 = b1 + b2;

        auto r0 = size_t(0);
        auto r1 = 2 * rowOffset;
        auto r2 = rowOffset;
        auto r3 = 3 * rowOffset;
        
        __m256 a, b, c, d;
        __m256 e, f, g, h;
        AVXTranspose4x4(inputA + b0 + r0, inputA + b0 + r1, inputA + b0 + r2, inputA + b0 + r3, a, b, c, d);
        AVXTranspose4x4(inputB + b0 + r0, inputB + b0 + r1, inputB + b0 + r2, inputB + b0 + r3, e, f, g, h);
        AVXStoreRows(e, f, g, h, outputA + b0 + r0, outputA + b0 + r1, outputA + b0 + r2, outputA + b0 + r3);
        AVXStoreRows(a, b, c, d, outputB + b0 + r0, outputB + b0 + r1, outputB + b0 + r2, outputB + b0 + r3);

        AVXTranspose4x4(inputA + b1 + r0, inputA + b1 + r1, inputA + b1 + r2, inputA + b1 + r3, a, b, c, d);
        AVXTranspose4x4(inputB + b2 + r0, inputB + b2 + r1, inputB + b2 + r2, inputB + b2 + r3, e, f, g, h);
        AVXStoreRows(e, f, g, h, outputA + b1 + r0, outputA + b1 + r1, outputA + b1 + r2, outputA + b1 + r3);
        AVXStoreRows(a, b, c, d, outputB + b2 + r0, outputB + b2 + r1, outputB + b2 + r2, outputB + b2 + r3);

        AVXTranspose4x4(inputA + b2 + r0, inputA + b2 + r1, inputA + b2 + r2, inputA + b2 + r3, a, b, c, d);
        AVXTranspose4x4(inputB + b1 + r0, inputB + b1 + r1, inputB + b1 + r2, inputB + b1 + r3, e, f, g, h);
        AVXStoreRows(e, f, g, h, outputA + b2 + r0, outputA + b2 + r1, outputA + b2 + r2, outputA + b2 + r3);
        AVXStoreRows(a, b, c, d, outputB + b1 + r0, outputB + b1 + r1, outputB + b1 + r2, outputB + b1 + r3);

        AVXTranspose4x4(inputA + b3 + r0, inputA + b3 + r1, inputA + b3 + r2, inputA + b3 + r3, a, b, c, d);
        AVXTranspose4x4(inputB + b3 + r0, inputB + b3 + r1, inputB + b3 + r2, inputB + b3 + r3, e, f, g, h);
        AVXStoreRows(e, f, g, h, outputA + b3 + r0, outputA + b3 + r1, outputA + b3 + r2, outputA + b3 + r3);
        AVXStoreRows(a, b, c, d, outputB + b3 + r0, outputB + b3 + r1, outputB + b3 + r2, outputB + b3 + r3);
      }
      else
    #endif
    {
      BitReverseCopyDisjoint<FFTSize, MaxSizeTBit, 1>(inputA, inputB, outputA, outputB, shift); 
    }
  }

  template <size_t FFTSize, std::floating_point F> 
  void BitReverseCopyInPosition(const Complex<F> *input, Complex<F> *output, uint32_t shift)
  { 
    #if SIMD_AVX
      if constexpr (FFTSize == 64 && std::same_as<F, float>)
      {
        constexpr auto RowOffsetUnshifted = BitReverse(size_t(2));
        constexpr auto XBlockOffset = size_t(4);
        constexpr auto YBlockOffsetUnshifted = BitReverse(size_t(4));

        auto rowOffset = RowOffsetUnshifted >> shift;
        auto yBlockOffset = YBlockOffsetUnshifted >> shift;

        auto b0 = size_t(0);
        auto b1 = XBlockOffset;
        auto b2 = yBlockOffset;
        auto b3 = b1 + b2;

        auto r0 = size_t(0);
        auto r1 = 2 * rowOffset;
        auto r2 = rowOffset;
        auto r3 = 3 * rowOffset;
        
        __m256 a, b, c, d;
        AVXTranspose4x4(input + b0 + r0, input + b0 + r1, input + b0 + r2, input + b0 + r3, a, b, c, d);
        AVXStoreRows(a, b, c, d, output + b0 + r0, output + b0 + r1, output + b0 + r2, output + b0 + r3);

        AVXTranspose4x4(input + b1 + r0, input + b1 + r1, input + b1 + r2, input + b1 + r3, a, b, c, d);

        __m256 e, f, g, h;
        AVXTranspose4x4(input + b2 + r0, input + b2 + r1, input + b2 + r2, input + b2 + r3, e, f, g, h);
        AVXStoreRows(e, f, g, h, output + b1 + r0, output + b1 + r1, output + b1 + r2, output + b1 + r3);
        AVXStoreRows(a, b, c, d, output + b2 + r0, output + b2 + r1, output + b2 + r2, output + b2 + r3);

        AVXTranspose4x4(input + b3 + r0, input + b3 + r1, input + b3 + r2, input + b3 + r3, a, b, c, d);
        AVXStoreRows(a, b, c, d, output + b3 + r0, output + b3 + r1, output + b3 + r2, output + b3 + r3);
      }
      else
    #endif
    {
      BitReverseCopyInPosition<FFTSize, MaxSizeTBit, 1>(input, output, shift); 
    }
  }


  template <size_t FFTLength, std::floating_point F>
  void BitReverseCopyFixed(const Complex<F> *input, Complex<F> *output)
  {
    #if SIMD_AVX
      if constexpr (FFTLength == 64 && std::same_as<F, float>)
      {
        // 
        __m256 ul0, ul1, ul2, ul3;
        AVXTranspose4x4(input, input + 32, input + 16, input + 48, ul0, ul1, ul2, ul3);
        AVXStoreRows(ul0, ul1, ul2, ul3, output, output + 32, output + 16, output + 48);

        __m256 ur0, ur1, ur2, ur3;
        AVXTranspose4x4(input + 4, input + 36, input + 20, input + 52, ur0, ur1, ur2, ur3);

        __m256 ll0, ll1, ll2, ll3;
        AVXTranspose4x4(input + 8, input + 40, input + 24, input + 56, ll0, ll1, ll2, ll3);
        AVXStoreRows(ur0, ur1, ur2, ur3, output + 8, output + 40, output + 24, output + 56);
        AVXStoreRows(ll0, ll1, ll2, ll3, output + 4, output + 36, output + 20, output + 52);

        __m256 lr0, lr1, lr2, lr3;
        AVXTranspose4x4(input + 12, input + 44, input + 28, input + 60, lr0, lr1, lr2, lr3);
        AVXStoreRows(lr0, lr1, lr2, lr3, output + 12, output + 44, output + 28, output + 60);
      }
      else
    #endif
    {
      constexpr auto BitCount = Log2Floor(FFTLength);
      constexpr auto Shift = BitReverseShiftFromBitCount<size_t>(BitCount);
      return BitReverseCopyInPosition<FFTLength, MaxSizeTBit, 1>(input, output, Shift);
    }
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
        #if CPU_ARM_ANY
          constexpr auto LowShift = 0u;
        #else
          constexpr auto LowShift = SubFFTBitCount / 2;
        #endif
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
}
