#ifdef _MSC_VER
  #pragma warning (disable : 4710) // Function not inlined
  #pragma warning (disable : 4711)
  #pragma warning (disable : 5045)
#endif

#include "fft_20xx/fft_20xx.h"
#include "fft_20xx/fft_20xx_util.h"

#include <bit>
#include <cmath>
#include <complex>


namespace FFT20XX
{
  namespace
  {
    // Some helper functions here just to make it easier



    constexpr size_t operator""_z(unsigned long long l)
      { return size_t(l); }

    template <std::unsigned_integral I>
    constexpr auto IsPowerOfFourAssumingIsPowerOfTwo(I v) -> bool
      { return (v & I(0x55555555'55555555)) != 0; }

    constexpr auto Rotation(double angle) -> std::complex<double>
      { return { std::cos(angle), std::sin(angle) }; }


    constexpr auto Rotation(std::size_t n, std::size_t N) -> std::complex<double>
    {
      constexpr auto NegTwoPi = -6.283185307179586;
      auto angle = NegTwoPi * double(n) / double(N);
      return Rotation(angle);
    }

    auto BitRev(std::integral auto v)
    {
      if constexpr (sizeof(v) == sizeof(uint64_t))
      {
        v = ((v >>  1) & 0x55555555'55555555) | ((v & 0x55555555'55555555) << 1);
        v = ((v >>  2) & 0x33333333'33333333) | ((v & 0x33333333'33333333) << 2);
        v = ((v >>  4) & 0x0f0f0f0f'0f0f0f0f) | ((v & 0x0f0f0f0f'0f0f0f0f) << 4);
        v = ((v >>  8) & 0x00ff00ff'00ff00ff) | ((v & 0x00ff00ff'00ff00ff) << 8);
        v = ((v >> 16) & 0x0000ffff'0000ffff) | ((v & 0x0000ffff'0000ffff) << 16);
        v = (v >> 32) | (v << 32);
      }
      else
      {
        static_assert(sizeof(v) == sizeof(uint32_t));
        v = ((v >>  1) & 0x55555555) | ((v & 0x55555555) << 1);
        v = ((v >>  2) & 0x33333333) | ((v & 0x33333333) << 2);
        v = ((v >>  4) & 0x0f0f0f0f) | ((v & 0x0f0f0f0f) << 4);
        v = ((v >>  8) & 0x00ff00ff) | ((v & 0x00ff00ff) << 8);
        v = (v >> 16) | (v << 16);
      }

      return v;
    }


    auto Load8(const double *data, size_t startIndex, size_t increment)
    {
      return std::tuple
      {
        data[startIndex + 0 * increment],
        data[startIndex + 1 * increment],
        data[startIndex + 2 * increment],
        data[startIndex + 3 * increment],
        data[startIndex + 4 * increment],
        data[startIndex + 5 * increment],
        data[startIndex + 6 * increment],
        data[startIndex + 7 * increment],
      };
    }


    auto Store8(
      double *data,
      size_t startIndex,
      size_t increment,
      double a,
      double b,
      double c,
      double d,
      double e,
      double f,
      double g,
      double h)
    {
      data[startIndex + 0 * increment] = a;
      data[startIndex + 1 * increment] = b;
      data[startIndex + 2 * increment] = c;
      data[startIndex + 3 * increment] = d;
      data[startIndex + 4 * increment] = e;
      data[startIndex + 5 * increment] = f;
      data[startIndex + 6 * increment] = g;
      data[startIndex + 7 * increment] = h;
    }




    size_t TangleIndex(size_t i)
    {
      auto mask = (i & (i - 1)) >> 1;

      // Turns out, this transformation is called "Gray code" (thanks to Matthijs De Smedt for figuring this out)
      // It's possible (via wikipedia) to calculate this with a single "carryless multiply" instruction on
      // Platforms where that's a thing - but I couldn't find a clean way to test that in C++.
      mask ^= mask >> 32;
      mask ^= mask >> 16;
      mask ^= mask >> 8;
      mask ^= mask >> 4;
      mask ^= mask >> 2;
      mask ^= mask >> 1;

      return (i - (mask & 1)) ^ mask;
    }

    void DoFullyRealDIFRadix4(double &a, double &b, double &c, double &d, double &e, double &f, double &g, double &h)
    {
      constexpr auto InvSqrt2 = 0.7071067811865475244;
      constexpr auto angle = std::complex{InvSqrt2, -InvSqrt2};
      std::tie(a, e) = std::tuple{a + e, a - e};
      std::tie(b, f) = std::tuple{b + f, b - f};
      std::tie(c, g) = std::tuple{c + g, g - c};
      std::tie(d, h) = std::tuple{d + h, h - d};

      // Second radix 2
      std::tie(a, c) = std::tuple{a + c, a - c};
      std::tie(b, d) = std::tuple{b + d, d - b};

      auto eg = std::complex{e, g};
      auto fh = std::complex{f, h};
      fh *= angle;
      std::tie(e, g) = std::tuple{eg.real() + fh.real(), eg.real() - fh.real()};
      std::tie(f, h) = std::tuple{eg.imag() + fh.imag(), fh.imag() - eg.imag()};
    }


    void DoFullyRealInverseDIFRadix4(double &a, double &b, double &c, double &d, double &e, double &f, double &g, double &h)
    {
      constexpr auto InvSqrt2 = 0.7071067811865475244;
      constexpr auto angle = std::complex{InvSqrt2, InvSqrt2}; // conjugate of the forward angle

      auto eg = std::complex{e + g, f - h};
      auto fh = std::complex{e - g, f + h};

      fh *= angle;

      e = eg.real();
      f = fh.real();
      g = eg.imag();
      h = fh.imag();

      std::tie(a, c) = std::tuple{a + c, a - c};
      std::tie(b, d) = std::tuple{b - d, b + d};

      std::tie(a, e) = std::tuple{a + e, a - e};
      std::tie(b, f) = std::tuple{b + f, b - f};
      std::tie(c, g) = std::tuple{c - g, c + g};
      std::tie(d, h) = std::tuple{d - h, d + h};
    }


    void DoComplexRealDITRadix4(
      double &a,
      double &b,
      double &c,
      double &d,
      double &e,
      double &f,
      double &g,
      double &h,
      std::complex<double> angle0,
      std::complex<double> angle1)
    {
      // On input, our four complex values are split with the reals in the first half and the imaginary components in
      //  the second half.
      auto A = std::complex{a, e};
      auto B = std::complex{b, f};
      auto C = std::complex{c, g};
      auto D = std::complex{d, h};

      C *= angle0;
      D *= angle0;

      std::tie(A, C) = std::tuple{A + C, A - C};
      std::tie(B, D) = std::tuple{B + D, B - D};

      // For this algorithm to work, C and D need to be the conjugates of themselves
      C = std::conj(C);
      D = std::conj(D);
      B *= angle1;
      D *= std::complex{-angle1.imag(), -angle1.real()};

      std::tie(A, B) = std::tuple{A + B, A - B};
      std::tie(C, D) = std::tuple{C + D, C - D};

      // Again, make these the conjugates
      B = std::conj(B);
      D = std::conj(D);

      // On output, the real/imaginary components are now interleaved
      a = A.real();
      b = A.imag();
      c = B.real();
      d = B.imag();
      e = C.real();
      f = C.imag();
      g = D.real();
      h = D.imag();
    }


    void DoComplexRealDIFInverseRadix4(
      double &a,
      double &b,
      double &c,
      double &d,
      double &e,
      double &f,
      double &g,
      double &h,
      std::complex<double> angle0,
      std::complex<double> angle1)
    {
      // This is effectively the opposite of the above Radix4.
      auto A = std::complex{a, b};
      auto B = std::complex{c, d};
      auto C = std::complex{e, f};
      auto D = std::complex{g, h};

      B = std::conj(B);
      D = std::conj(D);

      std::tie(A, B) = std::tuple{A + B, A - B};
      std::tie(C, D) = std::tuple{C + D, C - D};

      D *= std::complex{-angle1.imag(), angle1.real()};
      B *= std::conj(angle1);
      D = std::conj(D);
      C = std::conj(C);

      std::tie(A, C) = std::tuple{A + C, A - C};
      std::tie(B, D) = std::tuple{B + D, B - D};

      D *= std::conj(angle0);
      C *= std::conj(angle0);

      a = A.real();
      b = B.real();
      c = C.real();
      d = D.real();
      e = A.imag();
      f = B.imag();
      g = C.imag();
      h = D.imag();
    }


    // The very first pass does either an effective radix 4 or 8 pass, depending on whether the real FFT length is a
    //  power of 4 or not.
    void Radix4Or8InitialStepRealFwd(const double *input, double *output, size_t length)
    {
      // The first radix 4 pass has no bit reversals and is always fully-real input (by definition)
      for (auto i = 0u; i < length / 8; i++)
      {
        auto [a, b, c, d, e, f, g, h] = Load8(input, i, length / 8);
        DoFullyRealDIFRadix4(a, b, c, d, e, f, g, h);
        Store8(output, i, length / 8, a, b, c, d, e, f, g, h);
      }

      if (IsPowerOfFourAssumingIsPowerOfTwo(length))
      {
        // If the real count is a power of 4, then we have an odd radix2 pass.

        const auto stride = length / 16;

        // Do the reals (subset 0)
        {
          for (auto i = 0u; i < stride; i++)
          {
            auto a = input[i + 0 * stride];
            auto b = input[i + 1 * stride];
            auto c = input[i + 2 * stride];
            auto d = input[i + 3 * stride];

            output[i + 0 * stride] = a + c;
            output[i + 1 * stride] = b + d;
            output[i + 2 * stride] = a - c;
            output[i + 3 * stride] = d - b; // reversed!
          }
        }

        // Now do the complex values
        for (auto subsetIndex = 1u; subsetIndex < 4; subsetIndex++)
        {
          const auto angleIndex = TangleIndex(BitRev(subsetIndex) >> (3 + std::countl_zero(length)));
          const auto angle = Rotation(angleIndex, length);

          auto start = subsetIndex * stride * 4;
          for (auto i = 0u; i < stride; i++)
          {
            auto a = std::complex{input[start + i + 0 * stride], input[start + i + 2 * stride]};
            auto b = std::complex{input[start + i + 1 * stride], input[start + i + 3 * stride]};
            b *= angle;

            output[start + i + 0 * stride] = a.real() + b.real();
            output[start + i + 1 * stride] = a.imag() + b.imag();
            output[start + i + 2 * stride] = a.real() - b.real();
            output[start + i + 3 * stride] = b.imag() - a.imag(); // reversed as conjugate
          }
        }
      }
    }


    // The next passes don't have any bit flipping yet (those happen in the back half of the algorithm) so this is a
    //  straightforward loop over all of the butterflies doing standard radix 4 passes (where the first pass still has
    //  fully real inputs and is treated differently)
    void NoFlipRadix4PassRealFwd(double *data, size_t length, size_t subLength)
    {
      const auto angleCount = 2 * length / subLength;

      for (auto subsetStartIndex = 0_z; subsetStartIndex < length; subsetStartIndex += subLength)
      {
        const auto angleIndex = TangleIndex(BitRev(subsetStartIndex) >> (1 + std::countl_zero(length)));
        const auto angle0 = Rotation(angleIndex, angleCount * 2);
        const auto angle1 = Rotation(angleIndex, angleCount * 4);

        for (auto i = 0u; i < subLength / 8; i++)
        {
          auto [a, b, c, d, e, f, g, h] = Load8(data, subsetStartIndex + i, subLength / 8);
          if (subsetStartIndex == 0)
            { DoFullyRealDIFRadix4(a, b, c, d, e, f, g, h); }
          else
            { DoComplexRealDITRadix4(a, b, c, d, e, f, g, h, angle0, angle1); }
          Store8(data, subsetStartIndex + i, subLength / 8, a, b, c, d, e, f, g, h);
        }
      }
    }



    // For FFTs where an odd number of bits flip (i.e. there's a middle bit that doesn't flip), there needs to be a
    //  special pass in the middle that does a single bit flip instead of the usual 2-bit-flips-per-radix-4 approach.
    //  The behavior of this pass differs based on whether or not the length is a power of 4 or not (it changes the
    //  pattern of transposition in memory)
    void SingleFlipRadix4PassRealFwd(double *data, size_t length, size_t subLength)
    {
      if (IsPowerOfFourAssumingIsPowerOfTwo(length))
      {
        // The bits that flip for power of four lengths touch two separate radix4s - however, they are conveniently
        //  contiguous in memory.
        const auto angleCount = 2 * length / subLength;

        for (auto subsetStartIndex = 0_z; subsetStartIndex < length; subsetStartIndex += subLength * 2)
        {
          const auto angleIndex0PreTangle = BitRev(subsetStartIndex) >> (1 + std::countl_zero(length));
          const auto angleIndex1PreTangle = angleIndex0PreTangle + subLength / 8;

          const auto angleIndex0 = TangleIndex(angleIndex0PreTangle);
          const auto angleIndex1 = TangleIndex(angleIndex1PreTangle);
          const auto angle0_0 = Rotation(angleIndex0, angleCount * 2);
          const auto angle1_0 = Rotation(angleIndex0, angleCount * 4);
          const auto angle0_1 = Rotation(angleIndex1, angleCount * 2);
          const auto angle1_1 = Rotation(angleIndex1, angleCount * 4);

          for (auto i = 0u; i < subLength / 8; i++)
          {
            auto [a0, b0, c0, d0, e0, f0, g0, h0] = Load8(data, subsetStartIndex + i, subLength / 8);
            auto [a1, b1, c1, d1, e1, f1, g1, h1] = Load8(data, subsetStartIndex + i + subLength, subLength / 8);
            if (subsetStartIndex == 0)
              { DoFullyRealDIFRadix4(a0, b0, c0, d0, e0, f0, g0, h0); }
            else
              { DoComplexRealDITRadix4(a0, b0, c0, d0, e0, f0, g0, h0, angle0_0, angle1_0); }
            DoComplexRealDITRadix4(a1, b1, c1, d1, e1, f1, g1, h1, angle0_1, angle1_1);

            std::swap(c0, a1); std::swap(d0, b1);
            std::swap(g0, e1); std::swap(h0, f1);
            Store8(data, subsetStartIndex + i, subLength / 8, a0, b0, c0, d0, e0, f0, g0, h0);
            Store8(data, subsetStartIndex + i + subLength, subLength / 8, a1, b1, c1, d1, e1, f1, g1, h1);
          }
        }
      }
      else
      {
        // The bits that flip for non-power-of-four lengths are nice because they fit squarely within a single
        //  radix 4 butterfly (swapping the two inner elements), so this case is even simpler!
        const auto angleCount = 2 * length / subLength;

        for (auto subsetStartIndex = 0_z; subsetStartIndex < length; subsetStartIndex += subLength)
        {
          const auto angleIndex = TangleIndex(BitRev(subsetStartIndex) >> (1 + std::countl_zero(length)));
          const auto angle0 = Rotation(angleIndex, angleCount * 2);
          const auto angle1 = Rotation(angleIndex, angleCount * 4);

          for (auto i = 0u; i < subLength / 8; i++)
          {
            auto [a, b, c, d, e, f, g, h] = Load8(data, subsetStartIndex + i, subLength / 8);
            if (subsetStartIndex == 0)
              { DoFullyRealDIFRadix4(a, b, c, d, e, f, g, h); }
            else
              { DoComplexRealDITRadix4(a, b, c, d, e, f, g, h, angle0, angle1); }

            // Do the flip
            std::swap(c, e);
            std::swap(d, f);

            Store8(data, subsetStartIndex + i, subLength / 8, a, b, c, d, e, f, g, h);
          }
        }
      }
    }


    // Once we've started flipping bits, there will be a series of double-flips (where two pairs of bits swap at once)
    //  This is effectively a 4x4 matrix transpose, so we need to load 4 radix 4s, process them, then write them back
    //  out to transposed locations.
    void DoubleFlipRadix4PassRealFwd(double *data, size_t length, size_t subLength)
    {
      const auto flipStride = length / subLength * 2;
      const auto flipGroupLength = flipStride * 4;

      const auto shift = 1 + std::countl_zero(length);
      const auto reversedFlipStride = BitRev(flipStride) >> shift;

      for (auto flipGroupIndex = 0_z; flipGroupIndex < length; flipGroupIndex += flipGroupLength)
      {
        const auto reversedFlipGroupIndex = BitRev(flipGroupIndex) >> shift;
        for (auto subsetIndex = 0_z; subsetIndex < flipStride; subsetIndex += subLength)
        {
          const auto start0 = subsetIndex + flipGroupIndex;
          auto start1 = start0 +  1 * flipStride;
          auto start2 = start0 +  2 * flipStride;
          auto start3 = start0 +  3 * flipStride;
          const auto preTangle0 = (subsetIndex >> 1) | reversedFlipGroupIndex;
          const auto preTangle1 = preTangle0 | reversedFlipStride;
          const auto preTangle2 = preTangle0 | (reversedFlipStride >> 1);
          const auto preTangle3 = preTangle1 | preTangle2;
          const auto angleIndex0 = TangleIndex(preTangle0);
          const auto angleIndex1 = TangleIndex(preTangle1);
          const auto angleIndex2 = TangleIndex(preTangle2);
          const auto angleIndex3 = TangleIndex(preTangle3);

          const auto angle0_0 = Rotation(angleIndex0, flipStride * 2);
          const auto angle1_0 = Rotation(angleIndex0, flipStride * 4);
          const auto angle0_1 = Rotation(angleIndex1, flipStride * 2);
          const auto angle1_1 = Rotation(angleIndex1, flipStride * 4);
          const auto angle0_2 = Rotation(angleIndex2, flipStride * 2);
          const auto angle1_2 = Rotation(angleIndex2, flipStride * 4);
          const auto angle0_3 = Rotation(angleIndex3, flipStride * 2);
          const auto angle1_3 = Rotation(angleIndex3, flipStride * 4);

          for (auto i = 0u; i < subLength / 8; i++)
          {
            auto [a0, b0, c0, d0, e0, f0, g0, h0] = Load8(data, start0 + i, subLength / 8);
            auto [a1, b1, c1, d1, e1, f1, g1, h1] = Load8(data, start1 + i, subLength / 8);
            auto [a2, b2, c2, d2, e2, f2, g2, h2] = Load8(data, start2 + i, subLength / 8);
            auto [a3, b3, c3, d3, e3, f3, g3, h3] = Load8(data, start3 + i, subLength / 8);

            if (start0 == 0)
              { DoFullyRealDIFRadix4(a0, b0, c0, d0, e0, f0, g0, h0); }
            else
              { DoComplexRealDITRadix4(a0, b0, c0, d0, e0, f0, g0, h0, angle0_0, angle1_0); }

            DoComplexRealDITRadix4(a1, b1, c1, d1, e1, f1, g1, h1, angle0_1, angle1_1);
            DoComplexRealDITRadix4(a2, b2, c2, d2, e2, f2, g2, h2, angle0_2, angle1_2);
            DoComplexRealDITRadix4(a3, b3, c3, d3, e3, f3, g3, h3, angle0_3, angle1_3);

            // Transpose and store
            Store8(data, start0 + i, subLength / 8, a0, b0, a2, b2, a1, b1, a3, b3);
            Store8(data, start1 + i, subLength / 8, e0, f0, e2, f2, e1, f1, e3, f3);
            Store8(data, start2 + i, subLength / 8, c0, d0, c2, d2, c1, d1, c3, d3);
            Store8(data, start3 + i, subLength / 8, g0, h0, g2, h2, g1, h1, g3, h3);
          }
        }
      }
    }
  }



  // After the last radix 4 pass has run, we have one final fixup needed to get things into the correct output order,
  //  as the real-input FFT algorithm outputs in a strange order where the bottom half minus one of every power of 2
  //  size (8 pairs of values and up) are all in reversed order. The simplest version of this algorithm is to loop
  //  through these sizes reversing the ends of each from smallest size to the full FFT length.
  void UntangleRealFFT(double *data, size_t length)
  {
    for (auto stride = 16_z; stride <= length; stride *= 2)
    {
      for (auto offset = 0_z; offset < length; offset += stride)
      {
        auto d = data + offset + stride / 2;

        for (auto i = size_t(2); i < stride / 4; i += 2)
        {
          std::swap(d[(i + 0)], d[stride / 2 - (i + 0)]);
          std::swap(d[(i + 1)], d[stride / 2 - (i - 1)]);
        }
      }
    }
  }



  void RealForwardReference(const double *input, double *output, size_t length)
  {
    constexpr auto MinimumPiecewiseRealFFTLength = 32;
    if (length < MinimumPiecewiseRealFFTLength)
    {
      // 32 is the cutoff here, which would run an initial radix 4 then the final radix 4
      //  note that AVX will either need to fall back to SSE for 32 and 64 or special-case them (which might be good
      //  since they could fit completely within registers anyway)
      return;
    }

    const auto bitCount = std::numeric_limits<size_t>::digits - std::countl_zero(length) - 1;
    const auto bitReverseCount = (bitCount - 1) >> 1;

    // The first pass is a special case
    Radix4Or8InitialStepRealFwd(input, output, length);

    auto subLength = length >> (IsPowerOfFourAssumingIsPowerOfTwo(length) ? 3 : 2);
    for (const auto endSubLength = (4_z << bitReverseCount); subLength > endSubLength; subLength >>= 2)
      { NoFlipRadix4PassRealFwd(output, length, subLength); }

    if ((bitReverseCount & 1) != 0)
    {
      SingleFlipRadix4PassRealFwd(output, length, subLength);
      subLength >>= 2;
    }

    for (; subLength >= 4; subLength >>= 2)
      { DoubleFlipRadix4PassRealFwd(output, length, subLength); }

    // There is one final "pass" necessary - to compute the DC and nyquist terms (which are elements 0 and 1 in the
    //  array). Just one last radix 2 (since the real values are all effectively a radix 2 pass behind the complex
    //  values, and by the end there are just 2 real values left).
    auto a = output[0];
    auto b = output[1];
    output[0] = a + b;
    output[1] = a - b;

    UntangleRealFFT(output, length);
  }


  /// INVERSE /////////////////////////////////////////////////////////////////////////////////////////////////////////
  namespace
  {
    void TangleRealFFT(double *data, size_t length)
    {
      // The difference between this and Untangle is that this works the stride from largest to smallest instead of
      //  the other way around.
      for (auto stride = length; stride >= 16_z; stride /= 2)
      {
        for (auto offset = 0_z; offset < length; offset += stride)
        {
          auto d = data + offset + stride / 2;

          for (auto i = size_t(2); i < stride / 4; i += 2)
          {
            std::swap(d[(i + 0)], d[stride / 2 - (i + 0)]);
            std::swap(d[(i + 1)], d[stride / 2 - (i - 1)]);
          }
        }
      }
    }


    void DoubleFlipRadix4PassRealInv(double *data, size_t length, size_t subLength)
    {
      const auto flipStride = length / subLength * 2;
      const auto flipGroupLength = flipStride * 4;

      const auto shift = 1 + std::countl_zero(length);
      const auto reversedFlipStride = BitRev(flipStride) >> shift;

      for (auto flipGroupIndex = 0_z; flipGroupIndex < length; flipGroupIndex += flipGroupLength)
      {
        const auto reversedFlipGroupIndex = BitRev(flipGroupIndex) >> shift;
        for (auto subsetIndex = 0_z; subsetIndex < flipStride; subsetIndex += subLength)
        {
          const auto start0 = subsetIndex + flipGroupIndex;
          auto start1 = start0 +  1 * flipStride;
          auto start2 = start0 +  2 * flipStride;
          auto start3 = start0 +  3 * flipStride;
          const auto preTangle0 = (subsetIndex >> 1) | reversedFlipGroupIndex;
          const auto preTangle1 = preTangle0 | reversedFlipStride;
          const auto preTangle2 = preTangle0 | (reversedFlipStride >> 1);
          const auto preTangle3 = preTangle1 | preTangle2;
          const auto angleIndex0 = TangleIndex(preTangle0);
          const auto angleIndex1 = TangleIndex(preTangle1);
          const auto angleIndex2 = TangleIndex(preTangle2);
          const auto angleIndex3 = TangleIndex(preTangle3);

          const auto angle0_0 = Rotation(angleIndex0, flipStride * 2);
          const auto angle1_0 = Rotation(angleIndex0, flipStride * 4);
          const auto angle0_1 = Rotation(angleIndex1, flipStride * 2);
          const auto angle1_1 = Rotation(angleIndex1, flipStride * 4);
          const auto angle0_2 = Rotation(angleIndex2, flipStride * 2);
          const auto angle1_2 = Rotation(angleIndex2, flipStride * 4);
          const auto angle0_3 = Rotation(angleIndex3, flipStride * 2);
          const auto angle1_3 = Rotation(angleIndex3, flipStride * 4);

          for (auto i = 0u; i < subLength / 8; i++)
          {
            auto [a0, b0, a2, b2, a1, b1, a3, b3] = Load8(data, start0 + i, subLength / 8);
            auto [e0, f0, e2, f2, e1, f1, e3, f3] = Load8(data, start1 + i, subLength / 8);
            auto [c0, d0, c2, d2, c1, d1, c3, d3] = Load8(data, start2 + i, subLength / 8);
            auto [g0, h0, g2, h2, g1, h1, g3, h3] = Load8(data, start3 + i, subLength / 8);

            if (start0 == 0)
              { DoFullyRealInverseDIFRadix4(a0, b0, c0, d0, e0, f0, g0, h0); }
            else
              { DoComplexRealDIFInverseRadix4(a0, b0, c0, d0, e0, f0, g0, h0, angle0_0, angle1_0); }

            DoComplexRealDIFInverseRadix4(a1, b1, c1, d1, e1, f1, g1, h1, angle0_1, angle1_1);
            DoComplexRealDIFInverseRadix4(a2, b2, c2, d2, e2, f2, g2, h2, angle0_2, angle1_2);
            DoComplexRealDIFInverseRadix4(a3, b3, c3, d3, e3, f3, g3, h3, angle0_3, angle1_3);

            Store8(data, start0 + i, subLength / 8, a0, b0, c0, d0, e0, f0, g0, h0);
            Store8(data, start1 + i, subLength / 8, a1, b1, c1, d1, e1, f1, g1, h1);
            Store8(data, start2 + i, subLength / 8, a2, b2, c2, d2, e2, f2, g2, h2);
            Store8(data, start3 + i, subLength / 8, a3, b3, c3, d3, e3, f3, g3, h3);
          }
        }
      }
    }

    // For FFTs where an odd number of bits flip (i.e. there's a middle bit that doesn't flip), there needs to be a
    //  special pass in the middle that does a single bit flip instead of the usual 2-bit-flips-per-radix-4 approach.
    //  The behavior of this pass differs based on whether or not the length is a power of 4 or not (it changes the
    //  pattern of transposition in memory)
    void SingleFlipRadix4PassRealInv(double *data, size_t length, size_t subLength)
    {
      if (IsPowerOfFourAssumingIsPowerOfTwo(length))
      {
        // The bits that flip for power of four lengths touch two separate radix4s - however, they are conveniently
        //  contiguous in memory.
        const auto angleCount = 2 * length / subLength;

        for (auto subsetStartIndex = 0_z; subsetStartIndex < length; subsetStartIndex += subLength * 2)
        {
          const auto angleIndex0PreTangle = BitRev(subsetStartIndex) >> (1 + std::countl_zero(length));
          const auto angleIndex1PreTangle = angleIndex0PreTangle + subLength / 8;

          const auto angleIndex0 = TangleIndex(angleIndex0PreTangle);
          const auto angleIndex1 = TangleIndex(angleIndex1PreTangle);
          const auto angle0_0 = Rotation(angleIndex0, angleCount * 2);
          const auto angle1_0 = Rotation(angleIndex0, angleCount * 4);
          const auto angle0_1 = Rotation(angleIndex1, angleCount * 2);
          const auto angle1_1 = Rotation(angleIndex1, angleCount * 4);

          for (auto i = 0u; i < subLength / 8; i++)
          {
            auto [a0, b0, a1, b1, e0, f0, e1, f1] = Load8(data, subsetStartIndex + i, subLength / 8);
            auto [c0, d0, c1, d1, g0, h0, g1, h1] = Load8(data, subsetStartIndex + i + subLength, subLength / 8);

            if (subsetStartIndex == 0)
              { DoFullyRealInverseDIFRadix4(a0, b0, c0, d0, e0, f0, g0, h0); }
            else
              { DoComplexRealDIFInverseRadix4(a0, b0, c0, d0, e0, f0, g0, h0, angle0_0, angle1_0); }
            DoComplexRealDIFInverseRadix4(a1, b1, c1, d1, e1, f1, g1, h1, angle0_1, angle1_1);

            Store8(data, subsetStartIndex + i, subLength / 8, a0, b0, c0, d0, e0, f0, g0, h0);
            Store8(data, subsetStartIndex + i + subLength, subLength / 8, a1, b1, c1, d1, e1, f1, g1, h1);
          }
        }
      }
      else
      {
        // The bits that flip for non-power-of-four lengths are nice because they fit squarely within a single
        //  radix 4 butterfly (swapping the two inner elements), so this case is even simpler!
        const auto angleCount = 2 * length / subLength;

        for (auto subsetStartIndex = 0_z; subsetStartIndex < length; subsetStartIndex += subLength)
        {
          const auto angleIndex = TangleIndex(BitRev(subsetStartIndex) >> (1 + std::countl_zero(length)));
          const auto angle0 = Rotation(angleIndex, angleCount * 2);
          const auto angle1 = Rotation(angleIndex, angleCount * 4);

          for (auto i = 0u; i < subLength / 8; i++)
          {
            auto [a, b, e, f, c, d, g, h] = Load8(data, subsetStartIndex + i, subLength / 8);

            if (subsetStartIndex == 0)
              { DoFullyRealInverseDIFRadix4(a, b, c, d, e, f, g, h); }
            else
              { DoComplexRealDIFInverseRadix4(a, b, c, d, e, f, g, h, angle0, angle1); }

            Store8(data, subsetStartIndex + i, subLength / 8, a, b, c, d, e, f, g, h);
          }
        }
      }
    }

    void NoFlipRadix4PassRealInv(double *data, size_t length, size_t subLength)
    {
      const auto angleCount = 2 * length / subLength;

      for (auto subsetStartIndex = 0_z; subsetStartIndex < length; subsetStartIndex += subLength)
      {
        const auto angleIndex = TangleIndex(BitRev(subsetStartIndex) >> (1 + std::countl_zero(length)));
        const auto angle0 = Rotation(angleIndex, angleCount * 2);
        const auto angle1 = Rotation(angleIndex, angleCount * 4);

        for (auto i = 0u; i < subLength / 8; i++)
        {
          auto [a, b, c, d, e, f, g, h] = Load8(data, subsetStartIndex + i, subLength / 8);
          if (subsetStartIndex == 0)
            { DoFullyRealInverseDIFRadix4(a, b, c, d, e, f, g, h); }
          else
            { DoComplexRealDIFInverseRadix4(a, b, c, d, e, f, g, h, angle0, angle1); }
          Store8(data, subsetStartIndex + i, subLength / 8, a, b, c, d, e, f, g, h);
        }
      }
    }


    // The very first pass does either an effective radix 4 or 8 pass, depending on whether the real FFT length is a
    //  power of 4 or not.
    void Radix4Or8FinalStepRealInv(double *data, size_t length)
    {
      if (IsPowerOfFourAssumingIsPowerOfTwo(length))
      {
        // If the real count is a power of 4, then we have an odd radix2 pass.

        const auto stride = length / 16;

        // Do the reals (subset 0)
        {
          for (auto i = 0u; i < stride; i++)
          {
            auto a = data[i + 0 * stride];
            auto b = data[i + 1 * stride];
            auto c = data[i + 2 * stride];
            auto d = data[i + 3 * stride];

            data[i + 0 * stride] = a + c;
            data[i + 1 * stride] = b - d;
            data[i + 2 * stride] = a - c;
            data[i + 3 * stride] = b + d;
          }
        }

        // Now do the complex values
        for (auto subsetIndex = 1u; subsetIndex < 4; subsetIndex++)
        {
          const auto angleIndex = TangleIndex(BitRev(subsetIndex) >> (3 + std::countl_zero(length)));
          const auto angle = Rotation(angleIndex, length);

          auto start = subsetIndex * stride * 4;
          for (auto i = 0u; i < stride; i++)
          {
            auto ar = data[start + i + 0 * stride];
            auto ai = data[start + i + 1 * stride];
            auto br = data[start + i + 2 * stride];
            auto bi = data[start + i + 3 * stride];

            std::tie(ar, br) = std::tuple{ar + br, ar - br};
            std::tie(ai, bi) = std::tuple{ai - bi, ai + bi};

            auto b = std::complex{br, bi};
            b *= std::conj(angle);

            data[start + i + 0 * stride] = ar;
            data[start + i + 1 * stride] = b.real();
            data[start + i + 2 * stride] = ai;
            data[start + i + 3 * stride] = b.imag();
          }
        }
      }

      // Divide by length/2 to get into the correct space
      const auto scale = 2.0 / double(length);
      for (auto i = 0u; i < length / 8; i++)
      {
        auto [a, b, c, d, e, f, g, h] = Load8(data, i, length / 8);
        DoFullyRealInverseDIFRadix4(a, b, c, d, e, f, g, h);

        a *= scale;
        b *= scale;
        c *= scale;
        d *= scale;
        e *= scale;
        f *= scale;
        g *= scale;
        h *= scale;
        Store8(data, i, length / 8, a, b, c, d, e, f, g, h);
      }
    }
  }

  void RealInverseReference(const double *input, double *output, size_t length)
  {
    const auto bitCount = std::numeric_limits<size_t>::digits - std::countl_zero(length) - 1;
    const auto bitReverseCount = (bitCount - 1) >> 1;

    std::copy(input, input + length, output);
    TangleRealFFT(output, length);

    // This one step needs to scale down by half as it's the only set that does one extra "radix2" step vs. the rest
    //  of the elements
    auto a = output[0];
    auto b = output[1];
    output[0] = (a + b) / 2;
    output[1] = (a - b) / 2;

    const auto maxSubLength = length >> (IsPowerOfFourAssumingIsPowerOfTwo(length) ? 3 : 2);
    const auto maxDoubleFlipSubLength = 4_z << (bitReverseCount - ((bitReverseCount & 1) ? 2 : 0));
    auto subLength = 8_z;
    for (; subLength <= maxDoubleFlipSubLength; subLength <<= 2)
    {
      DoubleFlipRadix4PassRealInv(output, length, subLength);
    }

    if ((bitReverseCount & 1) != 0)
    {
      SingleFlipRadix4PassRealInv(output, length, subLength);
      subLength <<= 2;
    }

    while (subLength <= maxSubLength)
    {
      NoFlipRadix4PassRealInv(output, length, subLength);
      subLength <<= 2;
    }

    Radix4Or8FinalStepRealInv(output, length);
  }
}