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

template <Dir D, typename V>
FORCE_INLINE void DoRadix4NoRotation(V &a, V &b, V &c, V &d)
{
  AddSub(a, c);
  AddSub(b, d);
  AddSub(a, b);
  AddSubRot90<D>(c, d);
}

template <Dir D, typename V>
FORCE_INLINE void DoRadix4FirstRotation(V &a, V &b, V &c, V &d)
{
  AddSubRot90<D>(a, c);
  AddSubRot90<D>(b, d);
  AddSubRot45<D>(a, b);
  AddSubRot135<D>(c, d);
}

template <Dir D, Alignment Align, std::floating_point F>
void Radix4Or8InitialStep(const F *input, F *output, const size_t length)
{
  if (!IsPowerOfFourAssumingIsPowerOfTwo(length))
  {
    // Double the stride to skip over complex pairs.
    const auto doubleStride = length / 4;

    for (auto i = 0_z; i < doubleStride; i += 2 * VecSize<F>)
    {
      // load with deinterleave
      auto a = LoadPackedComplex<Align>(&input[doubleStride * 0 + i]);
      auto c = LoadPackedComplex<Align>(&input[doubleStride * 2 + i]);
      auto e = LoadPackedComplex<Align>(&input[doubleStride * 4 + i]);
      auto g = LoadPackedComplex<Align>(&input[doubleStride * 6 + i]);

      // First Radix2: A+-E, B+-F, C+-G, D+-H
      // Second Radix2: A' +- C', B' +- D', E' +- r90(G'), F' +- r90(H')
      // Third Radix 2: A'' +- B'', C'' +- r90(D''), E'' +- r45(F''), G'' +- r135(H'')
      // However they're going to be interleaved to minimize extra registers
      AddSub(a, e);
      AddSub(c, g);
      AddSub(a, c);
      AddSubRot90<D>(e, g);

      // Currently loaded: A'', C'', E'', G'' (8 vectors). Load b and f (12 vectors total)
      auto b = LoadPackedComplex<Align>(&input[doubleStride * 1 + i]);
      auto f = LoadPackedComplex<Align>(&input[doubleStride * 5 + i]);

      AddSub(b, f);

      // Okay now we finally need all 16 loaded so now we get tricky
      // This actually sucks because we need to deinterleave the vectors to load them which requires a single
      //  temporary register, which pushes us over into 17 :|
      auto d = LoadPackedComplex<Align>(&input[doubleStride * 3 + i]);
      auto h = LoadPackedComplex<Align>(&input[doubleStride * 7 + i]);

      AddSub(d, h);
      AddSub(b, d);
      AddSub(a, b);

      // Store A''', B''' - now we have 14 registers again:
      //  C'', D'', E'', F', G'', H'

      StoreUnpackedComplex<Align>(&output[doubleStride*0 + i], a);
      StoreUnpackedComplex<Align>(&output[doubleStride*1 + i], b);

      // C''', D''' = C'' +- r90(D'')
      AddSubRot90<D>(c, d);
      StoreUnpackedComplex<Align>(&output[doubleStride*2 + i], c);
      StoreUnpackedComplex<Align>(&output[doubleStride*3 + i], d);

      // Now we have E'', F', G'', H'
      AddSubRot90<D>(f, h);
      AddSubRot45<D>(e, f);
      StoreUnpackedComplex<Align>(&output[doubleStride*4 + i], e);
      StoreUnpackedComplex<Align>(&output[doubleStride*5 + i], f);

      AddSubRot135<D>(g, h);
      StoreUnpackedComplex<Align>(&output[doubleStride*6 + i], g);
      StoreUnpackedComplex<Align>(&output[doubleStride*7 + i], h);
    }
  }
  else
  {
    const auto doubleStride = length / 2;
    for (auto i = 0_z; i < doubleStride; i += 2 * VecSize<F>)
    {
      auto a = LoadPackedComplex<Align>(&input[doubleStride * 0 + i]);
      auto b = LoadPackedComplex<Align>(&input[doubleStride * 1 + i]);
      auto c = LoadPackedComplex<Align>(&input[doubleStride * 2 + i]);
      auto d = LoadPackedComplex<Align>(&input[doubleStride * 3 + i]);

      DoRadix4NoRotation<D>(a, b, c, d);

      StoreUnpackedComplex<Align>(&output[doubleStride*0 + i], a);
      StoreUnpackedComplex<Align>(&output[doubleStride*1 + i], b);
      StoreUnpackedComplex<Align>(&output[doubleStride*2 + i], c);
      StoreUnpackedComplex<Align>(&output[doubleStride*3 + i], d);
    }
  }
}


template <Dir D, typename V, typename A> requires requires(A a) { a.w0; a.w1; }
FORCE_INLINE void DoRadix4(V &a, V &b, V &c, V &d, A angles)
{
  if constexpr (requires (A aa) { aa.w2; })
  {
    Rotate<D>(b, angles.w0);
    Rotate<D>(c, angles.w1);
    Rotate<D>(d, angles.w2);
    AddSub(a, c);
    AddSub(b, d);
    AddSub(a, b);
    AddSubRot90<D>(c, d);
  }
  else
  {
    // Two-angle (FMA-optimized) version!
    AddSubRot<D>(a, c, angles.w0);
    AddSubRot<D>(b, d, angles.w0);
    AddSubRot<D>(a, b, angles.w1);
    AddSubRotPlus90<D>(c, d, angles.w1);
  }
}

template <std::floating_point F>
auto LoadSingleAngle(const F *tableData, size_t angleIndex, size_t angleCount) -> Angles<Complex<F>>
{
  static_assert(requires (Angles<F> a) { a.w0; a.w1; });
  auto i1 = angleCount * 4;
  auto r1 = (angleCount == 4) ? 8 : (angleCount * 3);

  // Special handling for 8-angle indexing, as they're arranged differently for alignment reasons.
  //  The default math would index the "angleCount == 8" angles from offset 12, but they actually start at 8.
  //  This masks off the bottom bits of 12 to turn it into an 8.
  auto r0 = (angleCount == 4) ? 0 : ((r1 >> 1) & ~7_z);
  auto i0 = (angleCount == 4) ? 4 : (i1 >> 1);

  auto w1 = Complex{ tableData[r0 + angleIndex], tableData[i0 + angleIndex] };
  auto w2 = Complex{ tableData[r1 + angleIndex], tableData[i1 + angleIndex] };

  if constexpr (requires(Angles<F> a) { a.w2; })
  {
    return { w2, w1, w1 * w2 };
  }
  else
  {
    return std::tuple{w1, w2};
  }
}

template <std::floating_point F>
auto LoadSingleAngleSet(const F *tableData, size_t angleIndex, size_t angleCount) -> Angles<ComplexVec<Vec<F>>>
{
  auto angles = LoadSingleAngle(tableData, angleIndex, angleCount);
  if constexpr (requires (decltype(angles) a) { a.w2; })
  {
    return
      {
        ComplexVec{ Set1(angles.w0.r), Set1(angles.w0.i) },
        ComplexVec{ Set1(angles.w1.r), Set1(angles.w1.i) },
        ComplexVec{ Set1(angles.w2.r), Set1(angles.w2.i) },
      };
   }
   else
   {
    return
      {
        ComplexVec{ Set1(angles.w0.r), Set1(angles.w0.i) },
        ComplexVec{ Set1(angles.w1.r), Set1(angles.w1.i) },
      };
   }
}

template <Dir D, Alignment Align, std::floating_point F>
void NoFlipRadix4Pass(const F *tableData, F *data, const size_t length, const size_t radix4Stride)
{
  const auto doubleStride = radix4Stride * 2;

  // The first set of stride works with no rotations (besides the 90)
  {
    for (auto i = 0_z; i < doubleStride; i += 2 * VecSize<F>)
    {
      auto a = LoadUnpackedComplex<Align>(&data[doubleStride*0 + i]);
      auto b = LoadUnpackedComplex<Align>(&data[doubleStride*1 + i]);
      auto c = LoadUnpackedComplex<Align>(&data[doubleStride*2 + i]);
      auto d = LoadUnpackedComplex<Align>(&data[doubleStride*3 + i]);

      DoRadix4NoRotation<D>(a, b, c, d);

      StoreUnpackedComplex<Align>(&data[doubleStride*0 + i], a);
      StoreUnpackedComplex<Align>(&data[doubleStride*1 + i], b);
      StoreUnpackedComplex<Align>(&data[doubleStride*2 + i], c);
      StoreUnpackedComplex<Align>(&data[doubleStride*3 + i], d);
    }
  }

  // The second set has 90 then 45 (so 45, 90, 135 as R4)
  {
    const auto sectionStart = data + (radix4Stride * 4) * 2;

    for (auto i = 0_z; i < doubleStride; i += 2 * VecSize<F>)
    {
      auto a = LoadUnpackedComplex<Align>(&sectionStart[doubleStride*0 + i]);
      auto b = LoadUnpackedComplex<Align>(&sectionStart[doubleStride*1 + i]);
      auto c = LoadUnpackedComplex<Align>(&sectionStart[doubleStride*2 + i]);
      auto d = LoadUnpackedComplex<Align>(&sectionStart[doubleStride*3 + i]);

      DoRadix4FirstRotation<D>(a, b, c, d);

      StoreUnpackedComplex<Align>(&sectionStart[doubleStride*0 + i], a);
      StoreUnpackedComplex<Align>(&sectionStart[doubleStride*1 + i], b);
      StoreUnpackedComplex<Align>(&sectionStart[doubleStride*2 + i], c);
      StoreUnpackedComplex<Align>(&sectionStart[doubleStride*3 + i], d);
    }
  }

  // The rest need to load angles
  const auto sectionCount = length / (radix4Stride * 4);
  for (auto sectionIndex = 2_z; sectionIndex < sectionCount; sectionIndex++)
  {
    // !!! on x64 do this as a bit reverse increment instead
    const auto angleIndex = BitRev(sectionIndex) >> (1 + std::countl_zero(sectionCount));

    const auto angles = LoadSingleAngleSet(tableData, angleIndex, sectionCount);

    const auto sectionStart = data + (sectionIndex * radix4Stride * 4) * 2;

    for (auto i = 0_z; i < doubleStride; i += 2 * VecSize<F>)
    {
      auto a = LoadUnpackedComplex<Align>(&sectionStart[doubleStride*0 + i]);
      auto b = LoadUnpackedComplex<Align>(&sectionStart[doubleStride*1 + i]);
      auto c = LoadUnpackedComplex<Align>(&sectionStart[doubleStride*2 + i]);
      auto d = LoadUnpackedComplex<Align>(&sectionStart[doubleStride*3 + i]);

      DoRadix4<D>(a, b, c, d, angles);

      StoreUnpackedComplex<Align>(&sectionStart[doubleStride*0 + i], a);
      StoreUnpackedComplex<Align>(&sectionStart[doubleStride*1 + i], b);
      StoreUnpackedComplex<Align>(&sectionStart[doubleStride*2 + i], c);
      StoreUnpackedComplex<Align>(&sectionStart[doubleStride*3 + i], d);
    }
  }
}


template <Dir D, Alignment Align, std::floating_point F>
void SingleFlipRadix4Pass(const F *tableData, F *data, const size_t length, const size_t radix4Stride)
{
    // There's a different flip pattern for power of 2 and power of 4 single flips
  const auto doubleStride = radix4Stride * 2;
  if (IsPowerOfFourAssumingIsPowerOfTwo(length))
  {
    // For power of four lengths, the flip happens completely within the butterfly (b and c flip). 1024 is a good
    //  test size for this case

    // The first set of stride works with no rotations (besides the 90)
    {
      for (auto i = 0_z; i < doubleStride; i += 2 * VecSize<F>)
      {
        auto a = LoadUnpackedComplex<Align>(&data[doubleStride*0 + i]);
        auto b = LoadUnpackedComplex<Align>(&data[doubleStride*1 + i]);
        auto c = LoadUnpackedComplex<Align>(&data[doubleStride*2 + i]);
        auto d = LoadUnpackedComplex<Align>(&data[doubleStride*3 + i]);

        DoRadix4NoRotation<D>(a, b, c, d);

        // Write the middle two (c & b) out of order (that's the bit flip). The other two loops do the same.
        StoreUnpackedComplex<Align>(&data[doubleStride*0 + i], a);
        StoreUnpackedComplex<Align>(&data[doubleStride*1 + i], c);
        StoreUnpackedComplex<Align>(&data[doubleStride*2 + i], b);
        StoreUnpackedComplex<Align>(&data[doubleStride*3 + i], d);
      }
    }

    // The second set has 90 then 45 (so 45, 90, 135 as R4)
    {
      const auto sectionStart = data + (radix4Stride * 4) * 2;

      for (auto i = 0_z; i < doubleStride; i += 2 * VecSize<F>)
      {
        auto a = LoadUnpackedComplex<Align>(&sectionStart[doubleStride*0 + i]);
        auto b = LoadUnpackedComplex<Align>(&sectionStart[doubleStride*1 + i]);
        auto c = LoadUnpackedComplex<Align>(&sectionStart[doubleStride*2 + i]);
        auto d = LoadUnpackedComplex<Align>(&sectionStart[doubleStride*3 + i]);

        DoRadix4FirstRotation<D>(a, b, c, d);

        StoreUnpackedComplex<Align>(&sectionStart[doubleStride*0 + i], a);
        StoreUnpackedComplex<Align>(&sectionStart[doubleStride*1 + i], c);
        StoreUnpackedComplex<Align>(&sectionStart[doubleStride*2 + i], b);
        StoreUnpackedComplex<Align>(&sectionStart[doubleStride*3 + i], d);
      }
    }

    const auto sectionCount = length / (radix4Stride * 4);
    for (auto sectionIndex = 2_z; sectionIndex < sectionCount; sectionIndex++)
    {
      const auto angleIndex = BitRev(sectionIndex) >> (1 + std::countl_zero(sectionCount));
      auto angles = LoadSingleAngleSet(tableData, angleIndex, sectionCount);

      const auto sectionStart = data + (sectionIndex * radix4Stride * 4) * 2;

      for (auto i = 0_z; i < doubleStride; i += 2 * VecSize<F>)
      {
        auto a = LoadUnpackedComplex<Align>(&sectionStart[doubleStride*0 + i]);
        auto b = LoadUnpackedComplex<Align>(&sectionStart[doubleStride*1 + i]);
        auto c = LoadUnpackedComplex<Align>(&sectionStart[doubleStride*2 + i]);
        auto d = LoadUnpackedComplex<Align>(&sectionStart[doubleStride*3 + i]);

        DoRadix4<D>(a, b, c, d, angles);

        StoreUnpackedComplex<Align>(&sectionStart[doubleStride*0 + i],  a);
        StoreUnpackedComplex<Align>(&sectionStart[doubleStride*1 + i],  c);
        StoreUnpackedComplex<Align>(&sectionStart[doubleStride*2 + i],  b);
        StoreUnpackedComplex<Align>(&sectionStart[doubleStride*3 + i],  d);
      }
    }
  }
  else
  {
    // For non-power-of-four lengths, the flip requires a second butterfly as well. The good news is, it's the *next
    //  sequential* butterfly, which means for the first two butterflies they're still both nice and special-case
    //  (first one is no-rotation, second one is InvSqrt2-only)
    // Then (using the first row as a1/b1/c1/d1 and the second as a2/b2/c2/d2) swap b2/a1 and d2/c1 (the rest stay)
    // 2048 is a good test size for this case.

    // The first two sections are the no-rotation and simple-rotation cases so special case them for efficiency.
    {
      const auto sectionStart0 = data;
      const auto sectionStart1 = sectionStart0 + (radix4Stride * 4) * 2;

      for (auto i = 0_z; i < doubleStride; i += 2 * VecSize<F>)
      {
        auto a = LoadUnpackedComplex<Align>(&sectionStart0[doubleStride*0 + i]);
        auto b = LoadUnpackedComplex<Align>(&sectionStart0[doubleStride*1 + i]);
        auto c = LoadUnpackedComplex<Align>(&sectionStart0[doubleStride*2 + i]);
        auto d = LoadUnpackedComplex<Align>(&sectionStart0[doubleStride*3 + i]);

        DoRadix4NoRotation<D>(a, b, c, d);

        // These can be stored back in-place
        StoreUnpackedComplex<Align>(&sectionStart0[doubleStride*0 + i], a);
        StoreUnpackedComplex<Align>(&sectionStart0[doubleStride*2 + i], c);

        // Now load the two values that get replaced with section 0 values and store over them
        a = LoadUnpackedComplex<Align>(&sectionStart1[doubleStride*0 + i]);

        StoreUnpackedComplex<Align>(&sectionStart1[doubleStride*0 + i], b);

        c = LoadUnpackedComplex<Align>(&sectionStart1[doubleStride*2 + i]);
        StoreUnpackedComplex<Align>(&sectionStart1[doubleStride*2 + i], d);

        // Load the remaining 2
        b = LoadUnpackedComplex<Align>(&sectionStart1[doubleStride*1 + i]);

        d = LoadUnpackedComplex<Align>(&sectionStart1[doubleStride*3 + i]);

        DoRadix4FirstRotation<D>(a, b, c, d);

        StoreUnpackedComplex<Align>(&sectionStart0[doubleStride*1 + i], a);
        StoreUnpackedComplex<Align>(&sectionStart1[doubleStride*1 + i], b);
        StoreUnpackedComplex<Align>(&sectionStart0[doubleStride*3 + i], c);
        StoreUnpackedComplex<Align>(&sectionStart1[doubleStride*3 + i], d);
      }
    }

    const auto sectionCount = length / (radix4Stride * 4);
    for (auto sectionIndex = 2_z; sectionIndex < sectionCount; sectionIndex += 2)
    {
      // !!! Without native bit reverse, do this as a bitrev increment instead
      const auto angleIndex0 = BitRev(sectionIndex) >> (1 + std::countl_zero(sectionCount));
      auto angles0 = LoadSingleAngle(tableData, angleIndex0, sectionCount);
      const auto angleIndex1 = angleIndex0 + (sectionCount >> 1);
      auto angles1 = LoadSingleAngle(tableData, angleIndex1, sectionCount);

      const auto sectionStart0 = data + (sectionIndex * radix4Stride * 4) * 2;
      const auto sectionStart1 = sectionStart0 + (radix4Stride * 4) * 2;

      for (auto i = 0_z; i < doubleStride; i += 2 * VecSize<F>)
      {
        auto a = LoadUnpackedComplex<Align>(&sectionStart0[doubleStride*0 + i]);
        auto b = LoadUnpackedComplex<Align>(&sectionStart0[doubleStride*1 + i]);
        auto c = LoadUnpackedComplex<Align>(&sectionStart0[doubleStride*2 + i]);
        auto d = LoadUnpackedComplex<Align>(&sectionStart0[doubleStride*3 + i]);

        DoRadix4<D>(a, b, c, d, angles0);

        // These can be stored back in-place
        StoreUnpackedComplex<Align>(&sectionStart0[doubleStride*0 + i], a);
        StoreUnpackedComplex<Align>(&sectionStart0[doubleStride*2 + i], c);

        // Now load the two values that get replaced with section 0 values and store over them
        a = LoadUnpackedComplex<Align>(&sectionStart1[doubleStride*0 + i]);
        StoreUnpackedComplex<Align>(&sectionStart1[doubleStride*0 + i], b);

        c = LoadUnpackedComplex<Align>(&sectionStart1[doubleStride*2 + i]);
        StoreUnpackedComplex<Align>(&sectionStart1[doubleStride*2 + i], d);

        // Load the remaining 2
        b = LoadUnpackedComplex<Align>(&sectionStart1[doubleStride*1 + i]);
        d = LoadUnpackedComplex<Align>(&sectionStart1[doubleStride*3 + i]);

        DoRadix4<D>(a, b, c, d, angles1);

        StoreUnpackedComplex<Align>(&sectionStart0[doubleStride*1 + i], a);
        StoreUnpackedComplex<Align>(&sectionStart1[doubleStride*1 + i], b);
        StoreUnpackedComplex<Align>(&sectionStart0[doubleStride*3 + i], c);
        StoreUnpackedComplex<Align>(&sectionStart1[doubleStride*3 + i], d);
      }
    }
  }
}

template <Dir D, Alignment Align, std::floating_point F>
void DoubleFlipRadix4Pass(
  const F *tableData,
  F *data,
  const size_t length,
  const size_t radix4Stride)
{
  const auto sectionCount = length / (radix4Stride * 4);
  const auto flipStride = sectionCount;

  const auto shift = std::countl_zero(length - 1);
  const auto angleIndexFlipInsert = BitRev(flipStride) >> shift;

  for (auto groupStartIndex = 0_z; groupStartIndex < length; groupStartIndex += flipStride * 4)
  {
    for (auto butterflySetStartIndex = 0_z; butterflySetStartIndex < flipStride; butterflySetStartIndex += radix4Stride * 4)
    {
      // we need 4 sets of 3 angles each
      const auto angleIndex0 = (BitRev(groupStartIndex) >> shift) + butterflySetStartIndex;
      const auto angleIndex2 = angleIndex0 | angleIndexFlipInsert;
      const auto angleIndex1 = angleIndex0 | (angleIndexFlipInsert >> 1);
      const auto angleIndex3 = angleIndex1 | angleIndex2;

      const auto angles0 = LoadSingleAngle(tableData, angleIndex0, sectionCount);
      const auto angles2 = LoadSingleAngle(tableData, angleIndex2, sectionCount);
      const auto angles1 = LoadSingleAngle(tableData, angleIndex1, sectionCount);
      const auto angles3 = LoadSingleAngle(tableData, angleIndex3, sectionCount);

      // Use the partially-bit-reversed "groupStartIndex + butterflySetStartIndex" to get the angle
      for (auto butterflyStartIndex = 0_z; butterflyStartIndex < radix4Stride; butterflyStartIndex += VecSize<F>)
      {
        auto i = groupStartIndex + butterflySetStartIndex + butterflyStartIndex;

        auto a0 = LoadUnpackedComplex<Align>(&data[2 * (i + flipStride * 0 + radix4Stride * 0)]);
        auto b0 = LoadUnpackedComplex<Align>(&data[2 * (i + flipStride * 0 + radix4Stride * 1)]);
        auto c0 = LoadUnpackedComplex<Align>(&data[2 * (i + flipStride * 0 + radix4Stride * 2)]);
        auto d0 = LoadUnpackedComplex<Align>(&data[2 * (i + flipStride * 0 + radix4Stride * 3)]);

        DoRadix4<D>(a0, b0, c0, d0, angles0);

        // Now store all of the '0's, swapping in 'a's as we go
        StoreUnpackedComplex<Align>(&data[2 * (i + flipStride * 0 + radix4Stride * 0)], a0);

        auto a1 = LoadUnpackedComplex<Align>(&data[2 * (i + flipStride * 2 + radix4Stride * 0)]);
        StoreUnpackedComplex<Align>(&data[2 * (i + flipStride * 2 + radix4Stride * 0)], b0);

        auto a2 = LoadUnpackedComplex<Align>(&data[2 * (i + flipStride * 1 + radix4Stride * 0)]);
        StoreUnpackedComplex<Align>(&data[2 * (i + flipStride * 1 + radix4Stride * 0)], c0);

        auto a3 = LoadUnpackedComplex<Align>(&data[2 * (i + flipStride * 3 + radix4Stride * 0)]);
        StoreUnpackedComplex<Align>(&data[2 * (i + flipStride * 3 + radix4Stride * 0)], d0);

        // At this point we have 6 loaded sets (a1, a2, a3)

        auto b1 = LoadUnpackedComplex<Align>(&data[2 * (i + flipStride * 2 + radix4Stride * 1)]);
        auto c1 = LoadUnpackedComplex<Align>(&data[2 * (i + flipStride * 2 + radix4Stride * 2)]);
        auto d1 = LoadUnpackedComplex<Align>(&data[2 * (i + flipStride * 2 + radix4Stride * 3)]);

        // Now there are 12 loaded (a1, b1, c1, d1, a2, a3)
        DoRadix4<D>(a1, b1, c1, d1, angles1);

        // Store everything (Swapping in bs as we go)
        StoreUnpackedComplex<Align>(&data[2 * (i + flipStride * 0 + radix4Stride * 1)], a1);
        StoreUnpackedComplex<Align>(&data[2 * (i + flipStride * 2 + radix4Stride * 1)], b1);

        auto b2 = LoadUnpackedComplex<Align>(&data[2 * (i + flipStride * 1 + radix4Stride * 1)]);
        StoreUnpackedComplex<Align>(&data[2 * (i + flipStride * 1 + radix4Stride * 1)], c1);

        auto b3 = LoadUnpackedComplex<Align>(&data[2 * (i + flipStride * 3 + radix4Stride * 1)]);
        StoreUnpackedComplex<Align>(&data[2 * (i + flipStride * 3 + radix4Stride * 1)], d1);

        // Now there are 8 loaded (a2, b2, a3, b3)

        auto c2 = LoadUnpackedComplex<Align>(&data[2 * (i + flipStride * 1 + radix4Stride * 2)]);
        auto d2 = LoadUnpackedComplex<Align>(&data[2 * (i + flipStride * 1 + radix4Stride * 3)]);

        // Back up to 12 (a2, b2, c2, d2, a3, b3)
        DoRadix4<D>(a2, b2, c2, d2, angles2);

        StoreUnpackedComplex<Align>(&data[2 * (i + flipStride * 0 + radix4Stride * 2) + 0], a2);
        StoreUnpackedComplex<Align>(&data[2 * (i + flipStride * 2 + radix4Stride * 2) + 0], b2);
        StoreUnpackedComplex<Align>(&data[2 * (i + flipStride * 1 + radix4Stride * 2) + 0], c2);

        auto c3 = LoadUnpackedComplex<Align>(&data[2 * (i + flipStride * 3 + radix4Stride * 2)]);
        StoreUnpackedComplex<Align>(&data[2 * (i + flipStride * 3 + radix4Stride * 2) + 0], d2);

        auto d3 = LoadUnpackedComplex<Align>(&data[2 * (i + flipStride * 3 + radix4Stride * 3)]);

        // Now just the '3's are left (8 sets)

        DoRadix4<D>(a3, b3, c3, d3, angles3);

        StoreUnpackedComplex<Align>(&data[2 * (i + flipStride * 0 + radix4Stride * 3)], a3);
        StoreUnpackedComplex<Align>(&data[2 * (i + flipStride * 2 + radix4Stride * 3)], b3);
        StoreUnpackedComplex<Align>(&data[2 * (i + flipStride * 1 + radix4Stride * 3)], c3);
        StoreUnpackedComplex<Align>(&data[2 * (i + flipStride * 3 + radix4Stride * 3)], d3);
      }
    }
  }
}

template <std::floating_point F>
auto LoadFinalAngles(const F *tableData, size_t angleIndex, size_t angleCount) -> Angles<ComplexVec<Vec<F>>>
{
  auto i1 = angleCount * 4;
  auto r1 = (angleCount == 4) ? 8 : (angleCount * 3);

  // Special handling for 8-angle indexing, as they're arranged differently for alignment reasons.
  //  The default math would index the "angleCount == 8" angles from offset 12, but they actually start at 8.
  //  This masks off the bottom bits of 12 to turn it into an 8.
  auto r0 = (angleCount == 4) ? 0 : ((r1 >> 1) & ~7_z);
  auto i0 = (angleCount == 4) ? 4 : (i1 >> 1);

  auto w1 = ComplexVec
    {
      Load<Alignment::Aligned>(&tableData[r0 + angleIndex]),
      Load<Alignment::Aligned>(&tableData[i0 + angleIndex]),
    };

  auto w2 = ComplexVec
    {
      Load<Alignment::Aligned>(&tableData[r1 + angleIndex]),
      Load<Alignment::Aligned>(&tableData[i1 + angleIndex]),
    };


  if constexpr (requires (Angles<ComplexVec<Vec<F>>> a) { a.w2; })
  {
    return {w2, w1, ComplexMultiply(w1, w2)};
  }
  else
  {
    return {w1, w2};
  }
}


template <Dir D, Alignment Align, std::floating_point F>
void LastDoubleFlipRadix4Pass(const F *tableData, F *data, const size_t length)
{
  const auto flipStride = length / 4;

  if constexpr (VecSize<F> >= 4u)
  {
    for (auto butterflySetStartIndex = 0_z; butterflySetStartIndex < flipStride; butterflySetStartIndex += VecSize<F>)
    {
      // Use the partially-bit-reversed "groupStartIndex + butterflySetStartIndex" to get the angle

      auto i = butterflySetStartIndex;

      auto a = LoadUnpackedComplex<Align>(&data[2 * (i + flipStride * 0)]);
      auto b = LoadUnpackedComplex<Align>(&data[2 * (i + flipStride * 2)]);
      auto c = LoadUnpackedComplex<Align>(&data[2 * (i + flipStride * 1)]);
      auto d = LoadUnpackedComplex<Align>(&data[2 * (i + flipStride * 3)]);

      Transpose4x4(a, b, c, d);

      auto angles = LoadFinalAngles(tableData, butterflySetStartIndex, flipStride);
      DoRadix4<D>(a, b, c, d, angles);

      if constexpr (D == Dir::Inverse)
      {
        const auto scale = Set1(F(1) / F(length));
        a = ScalarMultiply(a, scale);
        b = ScalarMultiply(b, scale);
        c = ScalarMultiply(c, scale);
        d = ScalarMultiply(d, scale);
      }

      StorePackedComplex<Align>(&data[2 * (i + flipStride * 0)], a);
      StorePackedComplex<Align>(&data[2 * (i + flipStride * 2)], b);
      StorePackedComplex<Align>(&data[2 * (i + flipStride * 1)], c);
      StorePackedComplex<Align>(&data[2 * (i + flipStride * 3)], d);
    }
  }
  else if constexpr (VecSize<F> == 2u)
  {
    for (auto butterflySetStartIndex = 0_z; butterflySetStartIndex < flipStride; butterflySetStartIndex += 2 * VecSize<F>)
    {
      // Use the partially-bit-reversed "groupStartIndex + butterflySetStartIndex" to get the angle

      auto i = butterflySetStartIndex;

      const auto ImaginaryOffset = 4u; // 4 because these are unpacked as (R R | I I)

      auto a01 = LoadUnpackedComplex<Align>(&data[2 * (i + flipStride * 0)]);      // Loads a0, b0
      auto b01 = LoadUnpackedComplex<Align>(&data[2 * (i + flipStride * 2)]);      // a1, b1
      auto c01 = LoadUnpackedComplex<Align>(&data[2 * (i + flipStride * 0)] + ImaginaryOffset);  // c0, d0
      auto d01 = LoadUnpackedComplex<Align>(&data[2 * (i + flipStride * 2)] + ImaginaryOffset);  // d1, d1

      Transpose2x2(a01, b01);
      Transpose2x2(c01, d01);

      auto angles = LoadFinalAngles(tableData, butterflySetStartIndex, flipStride);
      DoRadix4<D>(a01, b01, c01, d01, angles);
      if constexpr (D == Dir::Inverse)
      {
        const auto scale = Set1(F(1) / F(length));
        a01 = ScalarMultiply(a01, scale);
        b01 = ScalarMultiply(b01, scale);
        c01 = ScalarMultiply(c01, scale);
        d01 = ScalarMultiply(d01, scale);
      }

      StorePackedComplex<Align>(&data[2 * (i + flipStride * 0)], a01);
      StorePackedComplex<Align>(&data[2 * (i + flipStride * 2)], b01);
      auto a23 = LoadUnpackedComplex<Align>(&data[2 * (i + flipStride * 1)]);
      StorePackedComplex<Align>(&data[2 * (i + flipStride * 1)], c01);
      auto b23 = LoadUnpackedComplex<Align>(&data[2 * (i + flipStride * 3)]);
      StorePackedComplex<Align>(&data[2 * (i + flipStride * 3)], d01);

      auto c23 = LoadUnpackedComplex<Align>(&data[2 * (i + flipStride * 1)] + ImaginaryOffset);
      auto d23 = LoadUnpackedComplex<Align>(&data[2 * (i + flipStride * 3)] + ImaginaryOffset);
      Transpose2x2(a23, b23);
      Transpose2x2(c23, d23);

      angles = LoadFinalAngles(tableData, butterflySetStartIndex + 2, flipStride);
      DoRadix4<D>(a23, b23, c23, d23, angles);
      if constexpr (D == Dir::Inverse)
      {
        const auto scale = Set1(F(1) / F(length));
        a23 = ScalarMultiply(a23, scale);
        b23 = ScalarMultiply(b23, scale);
        c23 = ScalarMultiply(c23, scale);
        d23 = ScalarMultiply(d23, scale);
      }

      StorePackedComplex<Align>(&data[2 * (i + flipStride * 0) + ImaginaryOffset], a23);
      StorePackedComplex<Align>(&data[2 * (i + flipStride * 2) + ImaginaryOffset], b23);
      StorePackedComplex<Align>(&data[2 * (i + flipStride * 1) + ImaginaryOffset], c23);
      StorePackedComplex<Align>(&data[2 * (i + flipStride * 3) + ImaginaryOffset], d23);
    }
  }
  else
  {
    // !!! Scalar
    static_assert(VecSize<F> == 1u);
  }
}



constexpr auto SIMDSetSize = 4u;

template <Dir D, Alignment Align, std::floating_point F>
void RunComplex(
  const F *tableData,
  const F *input,
  F *output,
  size_t length)
{
  if constexpr (requires { { RunSmallComplex(tableData, input, output, length) } -> std::same_as<bool>; })
  {
    // There's a special implementation of small FFTs, so call it
    if (RunSmallComplex(tableData, input, output, length))
      { return; }
  }
  else
  {
    if (length < TinyFFTThreshold)
    {
      RunTinyFFT<D>(input, output, length);
      return;
    }
  }

  const auto bitCount = std::numeric_limits<size_t>::digits - std::countl_zero(length) - 1;
  const auto bitReverseCount = bitCount >> 1;

  // Step 0: Either a radix 4 or 8 pass (it's radix 4 for power-of-4 FFTs and radix 8 for just power-of-2 FFTs, where
  //  there would otherwise be a spare Radix 2 somewhere in there)
  Radix4Or8InitialStep<D, Align>(input, output, length);

  auto radix4Stride = length >> (IsPowerOfFourAssumingIsPowerOfTwo(length) ? 4 : 5);

  // The remainder of the first half of the steps (rounded down) have no bit flips to them, they're just regular boring
  //  radix 4 passes.
  for (const auto endStride = (1_z << bitReverseCount); radix4Stride >= endStride; radix4Stride >>= 2)
    { NoFlipRadix4Pass<D, Align>(tableData, output, length, radix4Stride); }

  if ((bitReverseCount & 1) != 0)
  {
    // If there is an odd number of bit reverses needed, then the first reverse is going to be a single bit flip.
    SingleFlipRadix4Pass<D, Align>(tableData, output, length, radix4Stride);
    radix4Stride >>= 2;
  }

  // Loop until the stride gets to be too small to properly handle as a radix 4 without swapping within a SIMD set.
  //  (more generally this should be "round up to nearest power of 4 but meh, this is good enough)
  constexpr auto StopStride = (SIMDSetSize == 8) ? 16u : SIMDSetSize;
  for (; radix4Stride >= StopStride; radix4Stride >>= 2)
    { DoubleFlipRadix4Pass<D, Align>(tableData, output, length, radix4Stride); }

  if constexpr (SIMDSetSize > 4u)
  {
    static_assert(SIMDSetSize <= 8u);
    // This needs a special-case because the second-to-last pass has a stride that is smaller than the SIMD set.
    SecondToLastDoubleFlipRadix4Pass<D, Align>(tableData, output, length);
  }

  // Step N - 1:
  //  The final flip (Which has to transpose within a SIMD set)
  LastDoubleFlipRadix4Pass<D, Align>(tableData, output, length);
}



// Real FFT ///////////////////////////////////////////////////////////////////////////////////////////////////////////

#if 0
template <Alignment Align, std::floating_point F>
void RunForwardRealPostprocess(F *data, size_t fftLength)
{
  // Real-input-valued FFTs ultimately get calculated as a half-length FFT of complex values. FFTs are linear (i.e.
  //  FFT(inputA) + FFT(inputB) == FFT(elementwiseAddition(inputA + inputB)), so treating the input (real-valued) data
  //  as if it were actually alternating real/imaginary components (that is, for input reals [a, b, c, d ...] we just
  //  pretend it's [(a + b*i), (c + d*i), ...] and run it as a complex FFT that's half the length), then what we end
  //  up with an FFT that is equivalent to FFT(data[0, 2, 4, 6, ...]) + FFT(i * data[1, 3, 5, 7, ...]).
  //
  // Once we've done that we have the FFTs of the evens and the odds summed together. If we pull them apart, we can
  //  then do one final radix 2 decimation-in-time pass to get the final result.
  // Every Real FFT's output has two real values and N - 2 complex values, where the latter half of the values are
  //  mirrored complex conjugates of the first half, like so (where A* is the conjugate of A):
  //   RealFFT(8Elements) = DC[real] A B C Nyquist[real] C* B* A*
  // We know that the result of each of the above even- and odd-indexed FFTs, since both had real-valued inputs, will
  //  be conjugates of each other, and we can use that information to pull the two FFTs apart. If you look at our
  //  data (which has already been run as a complex data before reaching this function), it basically has the following
  //   values:
  //  (DC0 + i*DC1) (A0 + i*A1) (B0 + i*B1) (C0 + i*C1) (Nyq0 + i*Nyq1) (C0* + i*C1*) (B0* + i*B1*) (A0* + i*A1*)
  //
  //  DC0 and DC1 are both real (by nature of real FFT results), so those components are already fine as-is. Ditto for
  //   Nyq0 and Nyq1. These represent the DC and nyquist values for the even and odd indices of the original real data.
  //
  //  To get the complex values, we can take advantage of the conjugate nature of the even/odd outputs and solve.
  //  Using A0/A1 as an example, pulling apart the following values into their imaginary components (Xr and Xi):
  //   (A0 + i*A1) == { A0r - A1i , A0i + A1r }
  //   (A0* + i*A1*) == { A0r + A1i, -A0i + A1r }
  // We can use those values to solve for A0r and A0i:
  //   A0r = [(A0r - A1i) + (A0r + A1i)] / 2
  //   A0i = [(A0i + A1r) - (-A0i + A1r)] / 2
  // (Solving for A1r and A1i are similar)
  //
  // That gives us the original even (DC0 A0 B0 C0 Nyq0) and odd (DC1 A1 B1 C1 Nyq1) values (conjugate values are
  //  implied). We can then run those values through a standard radix 2 DIT to get the final values.
  //
  // However, there's still the matter of storage! The above example did two interleaved 8-element FFTs using the even
  //  and odd inputs of a larger 16-element FFT. The output of that should *technically* have 16 elements. But, as
  //  stated, there's conjugate mirroring so we only need to actually output 9 values (DC, A, B, C, D, E, F, G, Nyq).
  //  That would require 18 float values (for 9 complex pairs), but because DC and Nyquist are both guaranteed to be
  //  real values, we'll pack them both into data[0] and data[1], so they together take up one "complex value", then
  //  the remaining 14 floats can store the 7 non-conjugate complex values.
  //
  // That was a lot of comment! But I wanted to be able to remember how this works later!

  const auto complexFFTLength = fftLength / 2;
  auto cplxData = reinterpret_cast<Complex<F> *>(data);

  constexpr auto ImaginaryComponentOffset = VecSize<F>;

  auto x0r = data[0];
  auto x0i = data[ImaginaryComponentOffset];

  // The radix2 butterfly dc and nyquist results use these first two components.
  auto dc = x0r + x0i;
  auto nyquist = x0r - x0i;

  // The very middle complex element is also real values, so a + Rot90(b) == a - b*i.
  auto middle = Complex { output[complexFFTLength], -output[complexFFTLength + ImaginaryComponentOffset] };

  for (auto k = 1u; k < complexFFTLength / 2; k++)
  {
    auto x0 = cplxData[k];
    auto x1Conjugate = Conjugate(cplxData[complexFFTLength - k]);
    auto even = (x0 + x1Conjugate) * F(0.5);
    auto odd = Rot90<Dir::Forward>((x0 - x1Conjugate) * F(0.5));

    auto angle =
      Complex<F>
      {
        tableData[angleCount * 3 / 2 + k],
        tableData[angleCount * 3 / 2 + k + angleCount / 2],
      };
    auto oddRotated = Rot<Dir::Forward>(odd, angle);
    cplxData[k] = even + oddRotated;
    cplxData[complexFFTLength - k] = Conjugate(even - oddRotated);
  }

  cplxOut[0] = { dc, nyquist };
  cplxOut[complexFFTLength / 2] = middle;
}
#endif