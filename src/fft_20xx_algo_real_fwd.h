template <Alignment Align, std::floating_point F>
void Radix4InitialStepRealFwd(const F *input, F *output, size_t length)
{
  const auto stride = length / 8;
  for (auto i = 0u; i < stride; i += VecSize)
  {
    // First radix 2
    VecType a, b, c, d, e, f, g, h;
    LoadStridedSets<Align>(input + i, stride, a, b, c, d, e, f, g, h);
    DITRealInputRadix4Pair(a, b, c, d, e, f, g, h);
    StoreStridedSets<Align>(output + i, stride, a, b, c, d, e, f, g, h);
  }
}



template <Alignment Align, std::floating_point F>
inline void Radix2FwdPass(F *data, size_t length)
{
  // Do the reals (subset 0)
  const auto stride = length / 16;
  {
    for (auto i = 0u; i < stride; i += VecSize)
    {
      auto a = Load<Align>(&data[i + 0 * stride]);
      auto b = Load<Align>(&data[i + 1 * stride]);
      auto c = Load<Align>(&data[i + 2 * stride]);
      auto d = Load<Align>(&data[i + 3 * stride]);

      VecType a2, b2;
      DITRealInputRadix2Pair(a, b, c, d, a2, b2);

      Store<Align>(&data[i + 0 * stride], a2);
      Store<Align>(&data[i + 1 * stride], b2);
      Store<Align>(&data[i + 2 * stride], c);
      Store<Align>(&data[i + 3 * stride], d);
    }
  }

  // Now do the complex values
  const F swRs[3] = { InvSqrt2<F>, CosPiOver8<F>, SinPiOver8<F> };
  const F swIs[3] = { -InvSqrt2<F>, -SinPiOver8<F>, -CosPiOver8<F> };

  for (auto complexSubset = 0u; complexSubset < 3; complexSubset++)
  {
    const auto wR = swRs[complexSubset];
    const auto wI = swIs[complexSubset];

    auto start = (1 + complexSubset) * stride * 4;
    for (auto i = 0u; i < stride; i += VecSize)
    {
      auto aR = Load<Align>(&data[start + i + 0 * stride]);
      auto bR = Load<Align>(&data[start + i + 1 * stride]);
      auto aI = Load<Align>(&data[start + i + 2 * stride]);
      auto bI = Load<Align>(&data[start + i + 3 * stride]);

      VecType aR2, aI2;
      DITComplexRadix2(aR, aI, bR, bI, wR, wI, aR2, aI2);

      Store<Align>(&data[start + i + 0 * stride], aR2);
      Store<Align>(&data[start + i + 1 * stride], aI2);
      Store<Align>(&data[start + i + 2 * stride], bR);
      Store<Align>(&data[start + i + 3 * stride], bI);
    }
  }
}



template <Alignment Align, std::floating_point F>
inline void NoFlipRadix4PassRealFwd(const F *tableData, F *data, size_t length, size_t subLength)
{
  const auto angleCount = 2 * length / subLength;

  const auto stride = subLength / 8;

  // Do the reals (subset 0)
  {
    for (auto i = 0u; i < stride; i += VecSize)
    {
      VecType a, b, c, d, e, f, g, h;
      LoadStridedSets<Align>(data + i, stride, a, b, c, d, e, f, g, h);
      DITRealInputRadix4Pair(a, b, c, d, e, f, g, h);
      StoreStridedSets<Align>(data + i, stride, a, b, c, d, e, f, g, h);
    }
  }

  // Now do the complex values
  for (auto subsetStartIndex = subLength; subsetStartIndex < length; subsetStartIndex += subLength)
  {
    const auto angleIndex = BitRev(subsetStartIndex) >> (1 + std::countl_zero(length));
    const auto angles = LoadScalarAngleGroup(tableData, angleIndex, angleCount);

    for (auto i = subsetStartIndex; i < subsetStartIndex + stride; i += VecSize)
    {
      VecType aR, bR, cR, dR, aI, bI, cI, dI;
      LoadStridedSets<Align>(data + i, stride, aR, bR, cR, dR, aI, bI, cI, dI);
      DITComplexRadix4(aR, aI, bR, bI, cR, cI, dR, dI, angles);
      StoreStridedSets<Align>(data + i, stride, aR, aI, bR, bI, cR, cI, dR, dI);
    }
  }
}



template <Alignment Align, std::floating_point F, uint32_t VS = VecSize>
inline void SingleFlipRadix4PassRealFwd(const F *tableData, F *data, size_t length, size_t subLength)
{
  if (IsPowerOfFourAssumingIsPowerOfTwo(length))
  {
    //This flip needs 2 radix 4s, but they are conveniently sequential
    const auto angleCount = 2 * length / subLength;
    const auto stride = subLength / 8;
    const auto subsetStep = subLength * 2;

    if constexpr(VS > 4)
    {
      static_assert(VS == 8);
      if (VS > stride)
      {
        assert(stride == 4 && subLength == 32 && length == 256);

        // Do the first real case as half vetors
        using HalfVec = decltype(LoadHalf<Align>(data));
        {
          const auto angleIndex1 = stride;
          const auto angles = LoadScalarAngleGroup(tableData, angleIndex1, angleCount);

          HalfVec r06, r07, r08, r09, r10, r11, r12, r13;
          LoadStridedSets<Align>(data, stride, r06, r07, r08, r09, r10, r11, r12, r13);

          DITRealInputRadix4Pair(r06, r07, r08, r09, r10, r11, r12, r13);

          Store<Align>(&data[0 * stride], r06);
          Store<Align>(&data[1 * stride], r07);
          Store<Align>(&data[4 * stride], r10);
          Store<Align>(&data[5 * stride], r11);

          r06 = LoadHalf<Align>(&data[subLength + 0 * stride]);
          r07 = LoadHalf<Align>(&data[subLength + 1 * stride]);
          Store<Align>(&data[subLength + 0 * stride], r08);
          Store<Align>(&data[subLength + 1 * stride], r09);

          r08 = LoadHalf<Align>(&data[subLength + 4 * stride]);
          r09 = LoadHalf<Align>(&data[subLength + 5 * stride]);
          Store<Align>(&data[subLength + 4 * stride], r12);
          Store<Align>(&data[subLength + 5 * stride], r13);

          r10 = LoadHalf<Align>(&data[subLength + 2 * stride]);
          r11 = LoadHalf<Align>(&data[subLength + 3 * stride]);
          r12 = LoadHalf<Align>(&data[subLength + 6 * stride]);
          r13 = LoadHalf<Align>(&data[subLength + 7 * stride]);

          // Back up to 15, but now we can jettison `z` to be 14, and now we have 2 temp registers, so 16 total
          DITComplexRadix4(r06, r08, r07, r09, r10, r12, r11, r13, angles);

          Store<Align>(&data[2 * stride], r06);
          Store<Align>(&data[3 * stride], r08);
          Store<Align>(&data[6 * stride], r10);
          Store<Align>(&data[7 * stride], r12);
          Store<Align>(&data[subLength + 2 * stride], r07);
          Store<Align>(&data[subLength + 3 * stride], r09);
          Store<Align>(&data[subLength + 6 * stride], r11);
          Store<Align>(&data[subLength + 7 * stride], r13);
        }

        // Now also do the first complex set as half vectors (to re-align so we can then do two subsets at once)
        {
          const auto angleIndex0 = BitRev(subsetStep) >> (1 + std::countl_zero(length));
          const auto angleIndex1 = angleIndex0 + stride;

          const auto sw0s = LoadScalarAngleGroup(tableData, angleIndex0, angleCount);
          const auto sw1s = LoadScalarAngleGroup(tableData, angleIndex1, angleCount);

          // There are 5 registers left over. I looked into potentially using them to pre-load some of the angles here, but
          //  it actually made it run slightly slower - I think with less register pressure the optimizer can reorder
          //  things more cleverly.
          HalfVec r00, r01, r02, r03, r04, r05, r06, r07;
          LoadStridedSets<Align>(data + subsetStep, stride, r00, r01, r02, r03, r04, r05, r06, r07);
          DITComplexRadix4(r00, r04, r01, r05, r02, r06, r03, r07, sw0s);

          Store<Align>(&data[subsetStep + 0 * stride], r00);
          Store<Align>(&data[subsetStep + 1 * stride], r04);
          Store<Align>(&data[subsetStep + 4 * stride], r02);
          Store<Align>(&data[subsetStep + 5 * stride], r06);

          r00 = LoadHalf<Align>(&data[subsetStep + subLength + 0 * stride]);
          r02 = LoadHalf<Align>(&data[subsetStep + subLength + 1 * stride]);
          Store<Align>(&data[subsetStep + subLength + 0 * stride], r01);
          Store<Align>(&data[subsetStep + subLength + 1 * stride], r05);
          r01 = LoadHalf<Align>(&data[subsetStep + subLength + 4 * stride]);
          r04 = LoadHalf<Align>(&data[subsetStep + subLength + 5 * stride]);
          Store<Align>(&data[subsetStep + subLength + 4 * stride], r03);
          Store<Align>(&data[subsetStep + subLength + 5 * stride], r07);

          r03 = LoadHalf<Align>(&data[subsetStep + subLength + 2 * stride]);
          r05 = LoadHalf<Align>(&data[subsetStep + subLength + 3 * stride]);
          r06 = LoadHalf<Align>(&data[subsetStep + subLength + 6 * stride]);
          r07 = LoadHalf<Align>(&data[subsetStep + subLength + 7 * stride]);

          DITComplexRadix4(r00, r01, r02, r04, r03, r06, r05, r07, sw1s);

          Store<Align>(&data[subsetStep + 2 * stride], r00);
          Store<Align>(&data[subsetStep + 3 * stride], r01);
          Store<Align>(&data[subsetStep + 6 * stride], r03);
          Store<Align>(&data[subsetStep + 7 * stride], r06);
          Store<Align>(&data[subsetStep + subLength + 2 * stride], r02);
          Store<Align>(&data[subsetStep + subLength + 3 * stride], r04);
          Store<Align>(&data[subsetStep + subLength + 6 * stride], r05);
          Store<Align>(&data[subsetStep + subLength + 7 * stride], r07);
        }

        // now do the last two as a single block
        const auto subsetStartIndex = subsetStep * 2;
        {
          const auto angleIndex0A = BitRev(subsetStartIndex) >> (1 + std::countl_zero(length));
          const auto angleIndex1A = angleIndex0A + stride;
          const auto angleIndex0B = BitRev(subsetStartIndex + subsetStep) >> (1 + std::countl_zero(length));
          const auto angleIndex1B = angleIndex0B + stride;

          const auto angles0 = LoadScalarAngleGroupX2(tableData, angleIndex0A, angleIndex0B, angleCount);
          const auto angles1 = LoadScalarAngleGroupX2(tableData, angleIndex1A, angleIndex1B, angleCount);

          VecType r00, r01, r02, r03, r04, r05, r06, r07;
          const auto baseA = data + subsetStartIndex;
          const auto baseB = baseA + subsetStep;
          LoadStridedHalves<Align>(baseA, baseB, stride, r00, r01, r02, r03, r04, r05, r06, r07);
          DITComplexRadix4(r00, r04, r01, r05, r02, r06, r03, r07, angles0);

          StoreHalves<Align>(&baseA[0 * stride], &baseB[0 * stride], r00);
          StoreHalves<Align>(&baseA[1 * stride], &baseB[1 * stride], r04);
          StoreHalves<Align>(&baseA[4 * stride], &baseB[4 * stride], r02);
          StoreHalves<Align>(&baseA[5 * stride], &baseB[5 * stride], r06);

          r00 = LoadHalves<Align>(&baseA[subLength + 0 * stride], &baseB[subLength + 0 * stride]);
          r02 = LoadHalves<Align>(&baseA[subLength + 1 * stride], &baseB[subLength + 1 * stride]);
          StoreHalves<Align>(&baseA[subLength + 0 * stride], &baseB[subLength + 0 * stride], r01);
          StoreHalves<Align>(&baseA[subLength + 1 * stride], &baseB[subLength + 1 * stride], r05);
          r01 = LoadHalves<Align>(&baseA[subLength + 4 * stride], &baseB[subLength + 4 * stride]);
          r04 = LoadHalves<Align>(&baseA[subLength + 5 * stride], &baseB[subLength + 5 * stride]);
          StoreHalves<Align>(&baseA[subLength + 4 * stride], &baseB[subLength + 4 * stride], r03);
          StoreHalves<Align>(&baseA[subLength + 5 * stride], &baseB[subLength + 5 * stride], r07);

          r03 = LoadHalves<Align>(&baseA[subLength + 2 * stride], &baseB[subLength + 2 * stride]);
          r05 = LoadHalves<Align>(&baseA[subLength + 3 * stride], &baseB[subLength + 3 * stride]);
          r06 = LoadHalves<Align>(&baseA[subLength + 6 * stride], &baseB[subLength + 6 * stride]);
          r07 = LoadHalves<Align>(&baseA[subLength + 7 * stride], &baseB[subLength + 7 * stride]);

          DITComplexRadix4(r00, r01, r02, r04, r03, r06, r05, r07, angles1);

          StoreHalves<Align>(&baseA[2 * stride], &baseB[2 * stride], r00);
          StoreHalves<Align>(&baseA[3 * stride], &baseB[3 * stride], r01);
          StoreHalves<Align>(&baseA[6 * stride], &baseB[6 * stride], r03);
          StoreHalves<Align>(&baseA[7 * stride], &baseB[7 * stride], r06);
          StoreHalves<Align>(&baseA[subLength + 2 * stride], &baseB[subLength + 2 * stride], r02);
          StoreHalves<Align>(&baseA[subLength + 3 * stride], &baseB[subLength + 3 * stride], r04);
          StoreHalves<Align>(&baseA[subLength + 6 * stride], &baseB[subLength + 6 * stride], r05);
          StoreHalves<Align>(&baseA[subLength + 7 * stride], &baseB[subLength + 7 * stride], r07);
        }

        return;
      }
    }

    {
      const auto angleIndex1 = stride;
      const auto angles = LoadScalarAngleGroup(tableData, angleIndex1, angleCount);

      for (auto i = 0u; i < subLength / 8; i += VecSize)
      {
        VecType r06, r07, r08, r09, r10, r11, r12, r13;
        LoadStridedSets<Align>(data + i, stride, r06, r07, r08, r09, r10, r11, r12, r13);

        DITRealInputRadix4Pair(r06, r07, r08, r09, r10, r11, r12, r13);

        Store<Align>(&data[i + 0 * stride], r06);
        Store<Align>(&data[i + 1 * stride], r07);
        Store<Align>(&data[i + 4 * stride], r10);
        Store<Align>(&data[i + 5 * stride], r11);

        r06 = Load<Align>(&data[i + subLength + 0 * stride]);
        r07 = Load<Align>(&data[i + subLength + 1 * stride]);
        Store<Align>(&data[i + subLength + 0 * stride], r08);
        Store<Align>(&data[i + subLength + 1 * stride], r09);

        r08 = Load<Align>(&data[i + subLength + 4 * stride]);
        r09 = Load<Align>(&data[i + subLength + 5 * stride]);
        Store<Align>(&data[i + subLength + 4 * stride], r12);
        Store<Align>(&data[i + subLength + 5 * stride], r13);

        r10 = Load<Align>(&data[i + subLength + 2 * stride]);
        r11 = Load<Align>(&data[i + subLength + 3 * stride]);
        r12 = Load<Align>(&data[i + subLength + 6 * stride]);
        r13 = Load<Align>(&data[i + subLength + 7 * stride]);

        DITComplexRadix4(r06, r08, r07, r09, r10, r12, r11, r13, angles);

        Store<Align>(&data[i + 2 * stride], r06);
        Store<Align>(&data[i + 3 * stride], r08);
        Store<Align>(&data[i + 6 * stride], r10);
        Store<Align>(&data[i + 7 * stride], r12);
        Store<Align>(&data[i + subLength + 2 * stride], r07);
        Store<Align>(&data[i + subLength + 3 * stride], r09);
        Store<Align>(&data[i + subLength + 6 * stride], r11);
        Store<Align>(&data[i + subLength + 7 * stride], r13);
      }
    }

    for (auto subsetStartIndex = subsetStep; subsetStartIndex < length; subsetStartIndex += subsetStep)
    {
      const auto angleIndex0 = BitRev(subsetStartIndex) >> (1 + std::countl_zero(length));
      const auto angleIndex1 = angleIndex0 + stride;

      const auto sw0s
        = LoadScalarAngleGroup(tableData, angleIndex0, angleCount);

      const auto sw1s
        = LoadScalarAngleGroup(tableData, angleIndex1, angleCount);

      // There are 5 registers left over. I looked into potentially using them to pre-load some of the angles here, but
      //  it actually made it run slightly slower - I think with less register pressure the optimizer can reorder
      //  things more cleverly.
      for (auto i = subsetStartIndex; i < subsetStartIndex + stride; i += VecSize)
      {
        VecType r00, r01, r02, r03, r04, r05, r06, r07;
        LoadStridedSets<Align>(data + i, stride, r00, r01, r02, r03, r04, r05, r06, r07);
        DITComplexRadix4(r00, r04, r01, r05, r02, r06, r03, r07, sw0s);

        Store<Align>(&data[i + 0 * stride], r00);
        Store<Align>(&data[i + 1 * stride], r04);
        Store<Align>(&data[i + 4 * stride], r02);
        Store<Align>(&data[i + 5 * stride], r06);

        r00 = Load<Align>(&data[i + subLength + 0 * stride]);
        r02 = Load<Align>(&data[i + subLength + 1 * stride]);
        Store<Align>(&data[i + subLength + 0 * stride], r01);
        Store<Align>(&data[i + subLength + 1 * stride], r05);
        r01 = Load<Align>(&data[i + subLength + 4 * stride]);
        r04 = Load<Align>(&data[i + subLength + 5 * stride]);
        Store<Align>(&data[i + subLength + 4 * stride], r03);
        Store<Align>(&data[i + subLength + 5 * stride], r07);

        r03 = Load<Align>(&data[i + subLength + 2 * stride]);
        r05 = Load<Align>(&data[i + subLength + 3 * stride]);
        r06 = Load<Align>(&data[i + subLength + 6 * stride]);
        r07 = Load<Align>(&data[i + subLength + 7 * stride]);

        DITComplexRadix4(r00, r01, r02, r04, r03, r06, r05, r07, sw1s);

        Store<Align>(&data[i + 2 * stride], r00);
        Store<Align>(&data[i + 3 * stride], r01);
        Store<Align>(&data[i + 6 * stride], r03);
        Store<Align>(&data[i + 7 * stride], r06);
        Store<Align>(&data[i + subLength + 2 * stride], r02);
        Store<Align>(&data[i + subLength + 3 * stride], r04);
        Store<Align>(&data[i + subLength + 6 * stride], r05);
        Store<Align>(&data[i + subLength + 7 * stride], r07);
      }
    }
  }
  else
  {
    // This one is great because it's a swap within a single radix 4!
    const auto angleCount = 2 * length / subLength;

    // Do the reals (subset 0)
    const auto stride = subLength / 8;

    if constexpr(VS > 4)
    {
      static_assert(VS == 8);
      if (VS > stride)
      {
        assert(stride == 4 && subLength == 32 && length == 128);

        // Do the first real set and the first complex set as half-vectors
        using HalfVec = decltype(LoadHalf<Align>(data));
        {
          HalfVec a, b, c, d, e, f, g, h;
          LoadStridedSets<Align>(data, stride, a, b, c, d, e, f, g, h);
          DITRealInputRadix4Pair(a, b, c, d, e, f, g, h);
          StoreStridedSets<Align>(data, stride, a, b, e, f, c, d, g, h);
        }

        {
          const auto angleIndex = BitRev(subLength) >> (1 + std::countl_zero(length));
          const auto angles = LoadScalarAngleGroup(tableData, angleIndex, angleCount);

          HalfVec aR, bR, cR, dR, aI, bI, cI, dI;
          LoadStridedSets<Align>(data + subLength, stride, aR, bR, cR, dR, aI, bI, cI, dI);
          DITComplexRadix4(aR, aI, bR, bI, cR, cI, dR, dI, angles);
          StoreStridedSets<Align>(data + subLength, stride, aR, aI, cR, cI, bR, bI, dR, dI);
        }

        // Now do two subsets of complex values at a time
        for (auto subsetStartIndex = subLength * 2; subsetStartIndex < length; subsetStartIndex += subLength * 2)
        {
          const auto angleIndexA = BitRev(subsetStartIndex) >> (1 + std::countl_zero(length));
          const auto angleIndexB = BitRev(subsetStartIndex + subLength) >> (1 + std::countl_zero(length));
          auto angles = LoadScalarAngleGroupX2(tableData, angleIndexA, angleIndexB, angleCount);

          VecType aR, bR, cR, dR, aI, bI, cI, dI;
          auto *baseA = data + subsetStartIndex;
          auto *baseB = baseA + subLength;
          LoadStridedHalves<Align>(baseA, baseB, stride, aR, bR, cR, dR, aI, bI, cI, dI);
          DITComplexRadix4(aR, aI, bR, bI, cR, cI, dR, dI, angles);
          StoreStridedHalves<Align>(baseA, baseB, stride, aR, aI, cR, cI, bR, bI, dR, dI);
        }
        return;
      }
    }

    {
      for (auto i = 0u; i < stride; i += VecSize)
      {
        VecType a, b, c, d, e, f, g, h;
        LoadStridedSets<Align>(data + i, stride, a, b, c, d, e, f, g, h);
        DITRealInputRadix4Pair(a, b, c, d, e, f, g, h);
        StoreStridedSets<Align>(data + i, stride, a, b, e, f, c, d, g, h);
      }

      // Now do the complex values
      for (auto subsetStartIndex = subLength; subsetStartIndex < length; subsetStartIndex += subLength)
      {
        const auto angleIndex = BitRev(subsetStartIndex) >> (1 + std::countl_zero(length));
        const auto angles = LoadScalarAngleGroup(tableData, angleIndex, angleCount);
        for (auto i = subsetStartIndex; i < subsetStartIndex + stride; i += VecSize)
        {
          VecType aR, bR, cR, dR, aI, bI, cI, dI;
          LoadStridedSets<Align>(data + i, stride, aR, bR, cR, dR, aI, bI, cI, dI);
          DITComplexRadix4(aR, aI, bR, bI, cR, cI, dR, dI, angles);
          StoreStridedSets<Align>(data + i, stride, aR, aI, cR, cI, bR, bI, dR, dI);
        }
      }
    }
  }
}



template <Alignment Align, std::floating_point F>
inline void DoubleFlipRadix4PassRealFwd(const F *tableData, F *data, size_t length, size_t subLength)
{
  const auto flipStride = length / subLength * 2;
  const auto flipGroupLength = flipStride * 4;

  const auto stride = subLength / 8;

  const auto shift = 1 + std::countl_zero(length);
  const auto reversedFlipStride = BitRev(flipStride) >> shift;

  // First flip group's first *subgroup* has the real-input radix4s
  {
    constexpr auto start0 = 0_z;
    auto start1 = start0 +  1 * flipStride;
    auto start2 = start0 +  2 * flipStride;
    auto start3 = start0 +  3 * flipStride;
    const auto angleIndex1 = reversedFlipStride;
    const auto angleIndex2 = (reversedFlipStride >> 1);
    const auto angleIndex3 = angleIndex1 | angleIndex2;

    const auto sw1s
      = LoadScalarAngleGroup(tableData, angleIndex1, flipStride);
    const auto sw2s
      = LoadScalarAngleGroup(tableData, angleIndex2, flipStride);
    const auto sw3s
      = LoadScalarAngleGroup(tableData, angleIndex3, flipStride);
    for (auto i = 0u;;)
    {
      VecType r00, r01, r02, r03, r04, r05, r06, r07, r08, r09, r10, r11;

      r00 = Load<Align>(&data[start0 + i + 0 * stride]);
      r01 = Load<Align>(&data[start0 + i + 1 * stride]);
      r02 = Load<Align>(&data[start0 + i + 2 * stride]);
      r03 = Load<Align>(&data[start0 + i + 3 * stride]);
      r04 = Load<Align>(&data[start0 + i + 4 * stride]);
      r05 = Load<Align>(&data[start0 + i + 5 * stride]);
      r06 = Load<Align>(&data[start0 + i + 6 * stride]);
      r07 = Load<Align>(&data[start0 + i + 7 * stride]);

      DITRealInputRadix4Pair(r00, r01, r02, r03, r04, r05, r06, r07);

      Store<Align>(&data[start0 + i + 0 * stride], r00);
      Store<Align>(&data[start0 + i + 1 * stride], r01);

      r00 = Load<Align>(&data[start2 + i + 0 * stride]);
      r01 = Load<Align>(&data[start2 + i + 1 * stride]);
      Store<Align>(&data[start2 + i + 0 * stride], r02);
      Store<Align>(&data[start2 + i + 1 * stride], r03);

      r02 = Load<Align>(&data[start3 + i + 0 * stride]);
      r03 = Load<Align>(&data[start3 + i + 1 * stride]);
      Store<Align>(&data[start3 + i + 0 * stride], r06);
      Store<Align>(&data[start3 + i + 1 * stride], r07);

      r06 = Load<Align>(&data[start1 + i + 0 * stride]);
      r07 = Load<Align>(&data[start1 + i + 1 * stride]);
      Store<Align>(&data[start1 + i + 0 * stride], r04);
      Store<Align>(&data[start1 + i + 1 * stride], r05);

      r04 = Load<Align>(&data[start1 + i + 2 * stride]);
      r05 = Load<Align>(&data[start1 + i + 3 * stride]);
      r08 = Load<Align>(&data[start1 + i + 4 * stride]);
      r09 = Load<Align>(&data[start1 + i + 5 * stride]);
      r10 = Load<Align>(&data[start1 + i + 6 * stride]);
      r11 = Load<Align>(&data[start1 + i + 7 * stride]);


      DITComplexRadix4(r06, r08, r07, r09, r04, r10, r05, r11, sw1s);

      Store<Align>(&data[start0 + i + 4 * stride], r06);
      Store<Align>(&data[start0 + i + 5 * stride], r08);
      Store<Align>(&data[start1 + i + 4 * stride], r04);
      Store<Align>(&data[start1 + i + 5 * stride], r10);

      r04 = Load<Align>(&data[start3 + i + 4 * stride]);
      r06 = Load<Align>(&data[start3 + i + 5 * stride]);
      Store<Align>(&data[start3 + i + 4 * stride], r05);
      Store<Align>(&data[start3 + i + 5 * stride], r11);

      r05 = Load<Align>(&data[start2 + i + 4 * stride]);
      r08 = Load<Align>(&data[start2 + i + 5 * stride]);
      Store<Align>(&data[start2 + i + 4 * stride], r07);
      Store<Align>(&data[start2 + i + 5 * stride], r09);

      r07 = Load<Align>(&data[start2 + i + 2 * stride]);
      r09 = Load<Align>(&data[start2 + i + 3 * stride]);
      r10 = Load<Align>(&data[start2 + i + 6 * stride]);
      r11 = Load<Align>(&data[start2 + i + 7 * stride]);

      DITComplexRadix4(r00, r05, r01, r08, r07, r10, r09, r11, sw2s);

      Store<Align>(&data[start0 + i + 2 * stride], r00);
      Store<Align>(&data[start0 + i + 3 * stride], r05);
      Store<Align>(&data[start1 + i + 2 * stride], r07);
      Store<Align>(&data[start1 + i + 3 * stride], r10);
      Store<Align>(&data[start2 + i + 2 * stride], r01);
      Store<Align>(&data[start2 + i + 3 * stride], r08);

      r00 = Load<Align>(&data[start3 + i + 2 * stride]);
      r01 = Load<Align>(&data[start3 + i + 3 * stride]);
      Store<Align>(&data[start3 + i + 2 * stride], r09);
      Store<Align>(&data[start3 + i + 3 * stride], r11);

      r05 = Load<Align>(&data[start3 + i + 6 * stride]);
      r07 = Load<Align>(&data[start3 + i + 7 * stride]);

      DITComplexRadix4(r02, r04, r03, r06, r00, r05, r01, r07, sw3s);

      // Transpose and store
      Store<Align>(&data[start0 + i + 6 * stride], r02);
      Store<Align>(&data[start0 + i + 7 * stride], r04);
      Store<Align>(&data[start1 + i + 6 * stride], r00);
      Store<Align>(&data[start1 + i + 7 * stride], r05);
      Store<Align>(&data[start2 + i + 6 * stride], r03);
      Store<Align>(&data[start2 + i + 7 * stride], r06);
      Store<Align>(&data[start3 + i + 6 * stride], r01);
      Store<Align>(&data[start3 + i + 7 * stride], r07);

      i += VecSize;
      if (i >= stride)
        { break; }
    }
  }

  // effectively start one subset in on the first grouping
  auto flipGroupIndex = 0_z;
  auto reversedFlipGroupIndex = 0_z;
  auto subsetIndex = subLength;
  while (true)
  {
    while (subsetIndex < flipStride)
    {
      const auto start0 = subsetIndex + flipGroupIndex;
      auto start1 = start0 +  1 * flipStride;
      auto start2 = start0 +  2 * flipStride;
      auto start3 = start0 +  3 * flipStride;
      const auto angleIndex0 = (subsetIndex >> 1) | reversedFlipGroupIndex;
      const auto angleIndex1 = angleIndex0 | reversedFlipStride;
      const auto angleIndex2 = angleIndex0 | (reversedFlipStride >> 1);
      const auto angleIndex3 = angleIndex1 | angleIndex2;

      const auto sw0s = LoadScalarAngleGroup(tableData, angleIndex0, flipStride);
      const auto sw1s = LoadScalarAngleGroup(tableData, angleIndex1, flipStride);
      const auto sw2s = LoadScalarAngleGroup(tableData, angleIndex2, flipStride);
      const auto sw3s = LoadScalarAngleGroup(tableData, angleIndex3, flipStride);
      for (auto i = 0u;;)
      {
        VecType r00, r01, r02, r03, r04, r05, r06, r07, r08, r09, r10, r11;

        r00 = Load<Align>(&data[start0 + i + 0 * stride]);
        r01 = Load<Align>(&data[start0 + i + 1 * stride]);
        r02 = Load<Align>(&data[start0 + i + 2 * stride]);
        r03 = Load<Align>(&data[start0 + i + 3 * stride]);
        r04 = Load<Align>(&data[start0 + i + 4 * stride]);
        r05 = Load<Align>(&data[start0 + i + 5 * stride]);
        r06 = Load<Align>(&data[start0 + i + 6 * stride]);
        r07 = Load<Align>(&data[start0 + i + 7 * stride]);

        DITComplexRadix4(r00, r04, r01, r05, r02, r06, r03, r07, sw0s);

        Store<Align>(&data[start0 + i + 0 * stride], r00);
        Store<Align>(&data[start0 + i + 1 * stride], r04);

        r00 = Load<Align>(&data[start2 + i + 0 * stride]);
        r04 = Load<Align>(&data[start2 + i + 1 * stride]);
        Store<Align>(&data[start2 + i + 0 * stride], r01);
        Store<Align>(&data[start2 + i + 1 * stride], r05);

        r01 = Load<Align>(&data[start1 + i + 0 * stride]);
        r05 = Load<Align>(&data[start1 + i + 1 * stride]);
        Store<Align>(&data[start1 + i + 0 * stride], r02);
        Store<Align>(&data[start1 + i + 1 * stride], r06);

        r02 = Load<Align>(&data[start3 + i + 0 * stride]);
        r06 = Load<Align>(&data[start3 + i + 1 * stride]);
        Store<Align>(&data[start3 + i + 0 * stride], r03);
        Store<Align>(&data[start3 + i + 1 * stride], r07);


        r03 = Load<Align>(&data[start1 + i + 2 * stride]);
        r07 = Load<Align>(&data[start1 + i + 3 * stride]);
        r08 = Load<Align>(&data[start1 + i + 4 * stride]);
        r09 = Load<Align>(&data[start1 + i + 5 * stride]);
        r10 = Load<Align>(&data[start1 + i + 6 * stride]);
        r11 = Load<Align>(&data[start1 + i + 7 * stride]);

        DITComplexRadix4(r01, r08, r05, r09, r03, r10, r07, r11, sw1s);

        Store<Align>(&data[start0 + i + 4 * stride], r01);
        Store<Align>(&data[start0 + i + 5 * stride], r08);
        Store<Align>(&data[start1 + i + 4 * stride], r03);
        Store<Align>(&data[start1 + i + 5 * stride], r10);

        r01 = Load<Align>(&data[start2 + i + 4 * stride]);
        r03 = Load<Align>(&data[start2 + i + 5 * stride]);
        Store<Align>(&data[start2 + i + 4 * stride], r05);
        Store<Align>(&data[start2 + i + 5 * stride], r09);

        r05 = Load<Align>(&data[start3 + i + 4 * stride]);
        r08 = Load<Align>(&data[start3 + i + 5 * stride]);
        Store<Align>(&data[start3 + i + 4 * stride], r07);
        Store<Align>(&data[start3 + i + 5 * stride], r11);

        r07 = Load<Align>(&data[start2 + i + 2 * stride]);
        r09 = Load<Align>(&data[start2 + i + 3 * stride]);
        r10 = Load<Align>(&data[start2 + i + 6 * stride]);
        r11 = Load<Align>(&data[start2 + i + 7 * stride]);

        DITComplexRadix4(r00, r01, r04, r03, r07, r10, r09, r11, sw2s);

        Store<Align>(&data[start0 + i + 2 * stride], r00);
        Store<Align>(&data[start0 + i + 3 * stride], r01);
        Store<Align>(&data[start1 + i + 2 * stride], r07);
        Store<Align>(&data[start1 + i + 3 * stride], r10);
        Store<Align>(&data[start2 + i + 2 * stride], r04);
        Store<Align>(&data[start2 + i + 3 * stride], r03);

        r00 = Load<Align>(&data[start3 + i + 2 * stride]);
        r01 = Load<Align>(&data[start3 + i + 3 * stride]);
        Store<Align>(&data[start3 + i + 2 * stride], r09);
        Store<Align>(&data[start3 + i + 3 * stride], r11);

        r03 = Load<Align>(&data[start3 + i + 6 * stride]);
        r04 = Load<Align>(&data[start3 + i + 7 * stride]);

        DITComplexRadix4(r02, r05, r06, r08, r00, r03, r01, r04, sw3s);

        // Transpose and store
        Store<Align>(&data[start0 + i + 6 * stride], r02);
        Store<Align>(&data[start0 + i + 7 * stride], r05);
        Store<Align>(&data[start1 + i + 6 * stride], r00);
        Store<Align>(&data[start1 + i + 7 * stride], r03);
        Store<Align>(&data[start2 + i + 6 * stride], r06);
        Store<Align>(&data[start2 + i + 7 * stride], r08);
        Store<Align>(&data[start3 + i + 6 * stride], r01);
        Store<Align>(&data[start3 + i + 7 * stride], r04);

        i += VecSize;
        if (i >= stride)
          { break; }
      }

      subsetIndex += subLength;
    }

    flipGroupIndex += flipGroupLength;
    if (flipGroupIndex >= length)
      { break; }
    subsetIndex = 0_z;
    reversedFlipGroupIndex = BitRev(flipGroupIndex) >> shift;
  }
}



template <Alignment Align, std::floating_point F>
inline void SecondToLastDoubleFlipRadix4PassRealFwd(const F *tableData, F *data, size_t length)
{
  constexpr auto subLength = 32_z;
  const auto flipStride = length / subLength * 2;
  const auto flipGroupLength = flipStride * 4;

  const auto stride = subLength / 8;

  const auto shift = 1 + std::countl_zero(length);
  const auto reversedFlipStride = BitRev(flipStride) >> shift;

  using HalfVec = decltype(LoadHalf<Align>(data));

  // First flip group's first *subgroup* has the real-input radix4s. Do this as half-vectors
  {
    constexpr auto start0 = 0_z;
    auto start1 = start0 +  1 * flipStride;
    auto start2 = start0 +  2 * flipStride;
    auto start3 = start0 +  3 * flipStride;
    const auto angleIndex1 = reversedFlipStride;
    const auto angleIndex2 = (reversedFlipStride >> 1);
    const auto angleIndex3 = angleIndex1 | angleIndex2;

    const auto sw0s = LoadScalarAngleGroup(tableData, angleIndex1, flipStride);
    const auto angles23 = LoadScalarAngleGroupX2(tableData, angleIndex2, angleIndex3, flipStride);

    HalfVec h00, h01, h02, h03, h04, h05, h06, h07;
    VecType r00, r01, r02, r03, r04, r05, r06, r07;

    h00 = LoadHalf<Align>(&data[start0 + 0 * stride]);
    h01 = LoadHalf<Align>(&data[start0 + 1 * stride]);
    h02 = LoadHalf<Align>(&data[start0 + 2 * stride]);
    h03 = LoadHalf<Align>(&data[start0 + 3 * stride]);
    h04 = LoadHalf<Align>(&data[start0 + 4 * stride]);
    h05 = LoadHalf<Align>(&data[start0 + 5 * stride]);
    h06 = LoadHalf<Align>(&data[start0 + 6 * stride]);
    h07 = LoadHalf<Align>(&data[start0 + 7 * stride]);

    DITRealInputRadix4Pair(h00, h01, h02, h03, h04, h05, h06, h07);
    Store<Align>(&data[start0 + 0 * stride], h00);
    Store<Align>(&data[start0 + 1 * stride], h01);
    h00 = LoadHalf<Align>(&data[start1 + 0 * stride]);
    h01 = LoadHalf<Align>(&data[start1 + 1 * stride]);
    Store<Align>(&data[start1 + 0 * stride], h04);
    Store<Align>(&data[start1 + 1 * stride], h05);
    r00 = LoadHalves<Align>(&data[start2 + 0 * stride], &data[start3 + 0 * stride]);
    r01 = LoadHalves<Align>(&data[start2 + 1 * stride], &data[start3 + 1 * stride]);
    Store<Align>(&data[start2 + 0 * stride], h02);
    Store<Align>(&data[start2 + 1 * stride], h03);
    Store<Align>(&data[start3 + 0 * stride], h06);
    Store<Align>(&data[start3 + 1 * stride], h07);

    h02 = LoadHalf<Align>(&data[start1 + 2 * stride]);
    h03 = LoadHalf<Align>(&data[start1 + 3 * stride]);
    h04 = LoadHalf<Align>(&data[start1 + 4 * stride]);
    h05 = LoadHalf<Align>(&data[start1 + 5 * stride]);
    h06 = LoadHalf<Align>(&data[start1 + 6 * stride]);
    h07 = LoadHalf<Align>(&data[start1 + 7 * stride]);

    DITComplexRadix4(h00, h04, h01, h05, h02, h06, h03, h07, sw0s);
    Store<Align>(&data[start0 + 4 * stride], h00);
    Store<Align>(&data[start0 + 5 * stride], h04);
    Store<Align>(&data[start1 + 4 * stride], h02);
    Store<Align>(&data[start1 + 5 * stride], h06);
    r04 = LoadHalves<Align>(&data[start2 + 4 * stride], &data[start3 + 4 * stride]);
    r05 = LoadHalves<Align>(&data[start2 + 5 * stride], &data[start3 + 5 * stride]);
    Store<Align>(&data[start2 + 4 * stride], h01);
    Store<Align>(&data[start2 + 5 * stride], h05);
    Store<Align>(&data[start3 + 4 * stride], h03);
    Store<Align>(&data[start3 + 5 * stride], h07);

    r02 = LoadHalves<Align>(&data[start2 + 2 * stride], &data[start3 + 2 * stride]);
    r03 = LoadHalves<Align>(&data[start2 + 3 * stride], &data[start3 + 3 * stride]);
    r06 = LoadHalves<Align>(&data[start2 + 6 * stride], &data[start3 + 6 * stride]);
    r07 = LoadHalves<Align>(&data[start2 + 7 * stride], &data[start3 + 7 * stride]);

    DITComplexRadix4(r00, r04, r01, r05, r02, r06, r03, r07, angles23);

    StoreLowerHalves<Align>(&data[start0 + 2 * stride], r00, r04);
    StoreLowerHalves<Align>(&data[start2 + 2 * stride], r01, r05);
    StoreLowerHalves<Align>(&data[start1 + 2 * stride], r02, r06);
    StoreLowerHalves<Align>(&data[start3 + 2 * stride], r03, r07);
    StoreUpperHalves<Align>(&data[start0 + 6 * stride], r00, r04);
    StoreUpperHalves<Align>(&data[start2 + 6 * stride], r01, r05);
    StoreUpperHalves<Align>(&data[start1 + 6 * stride], r02, r06);
    StoreUpperHalves<Align>(&data[start3 + 6 * stride], r03, r07);
  }

  // effectively start one subset in on the first grouping
  auto flipGroupIndex = 0_z;
  auto reversedFlipGroupIndex = 0_z;
  auto subsetIndex = subLength;
  while (true)
  {
    while (subsetIndex < flipStride)
    {
      const auto start0 = subsetIndex + flipGroupIndex;
      auto start1 = start0 +  1 * flipStride;
      auto start2 = start0 +  2 * flipStride;
      auto start3 = start0 +  3 * flipStride;
      const auto angleIndex0 = (subsetIndex >> 1) | reversedFlipGroupIndex;
      const auto angleIndex1 = angleIndex0 | reversedFlipStride;
      const auto angleIndex2 = angleIndex0 | (reversedFlipStride >> 1);
      const auto angleIndex3 = angleIndex1 | angleIndex2;

      const auto angles01 = LoadScalarAngleGroupX2(tableData, angleIndex0, angleIndex1, flipStride);
      const auto angles23 = LoadScalarAngleGroupX2(tableData, angleIndex2, angleIndex3, flipStride);

      VecType r00, r01, r02, r03, r04, r05, r06, r07;

      r00 = LoadHalves<Align>(&data[start0 + 0 * stride], &data[start1 + 0 * stride]);
      r01 = LoadHalves<Align>(&data[start0 + 1 * stride], &data[start1 + 1 * stride]);
      r02 = LoadHalves<Align>(&data[start0 + 2 * stride], &data[start1 + 2 * stride]);
      r03 = LoadHalves<Align>(&data[start0 + 3 * stride], &data[start1 + 3 * stride]);
      r04 = LoadHalves<Align>(&data[start0 + 4 * stride], &data[start1 + 4 * stride]);
      r05 = LoadHalves<Align>(&data[start0 + 5 * stride], &data[start1 + 5 * stride]);
      r06 = LoadHalves<Align>(&data[start0 + 6 * stride], &data[start1 + 6 * stride]);
      r07 = LoadHalves<Align>(&data[start0 + 7 * stride], &data[start1 + 7 * stride]);

      DITComplexRadix4(r00, r04, r01, r05, r02, r06, r03, r07, angles01);

      StoreLowerHalves<Align>(&data[start0 + 0 * stride], r00, r04);
      StoreLowerHalves<Align>(&data[start1 + 0 * stride], r02, r06);
      StoreUpperHalves<Align>(&data[start0 + 4 * stride], r00, r04);
      StoreUpperHalves<Align>(&data[start1 + 4 * stride], r02, r06);

      r00 = LoadHalves<Align>(&data[start2 + 0 * stride], &data[start3 + 0 * stride]);
      r02 = LoadHalves<Align>(&data[start2 + 1 * stride], &data[start3 + 1 * stride]);
      r04 = LoadHalves<Align>(&data[start2 + 4 * stride], &data[start3 + 4 * stride]);
      r06 = LoadHalves<Align>(&data[start2 + 5 * stride], &data[start3 + 5 * stride]);
      StoreLowerHalves<Align>(&data[start2 + 0 * stride], r01, r05);
      StoreLowerHalves<Align>(&data[start3 + 0 * stride], r03, r07);
      StoreUpperHalves<Align>(&data[start2 + 4 * stride], r01, r05);
      StoreUpperHalves<Align>(&data[start3 + 4 * stride], r03, r07);

      r01 = LoadHalves<Align>(&data[start2 + 2 * stride], &data[start3 + 2 * stride]);
      r03 = LoadHalves<Align>(&data[start2 + 3 * stride], &data[start3 + 3 * stride]);
      r05 = LoadHalves<Align>(&data[start2 + 6 * stride], &data[start3 + 6 * stride]);
      r07 = LoadHalves<Align>(&data[start2 + 7 * stride], &data[start3 + 7 * stride]);

      DITComplexRadix4(r00, r04, r02, r06, r01, r05, r03, r07, angles23);

      StoreLowerHalves<Align>(&data[start0 + 2 * stride], r00, r04);
      StoreLowerHalves<Align>(&data[start1 + 2 * stride], r01, r05);
      StoreUpperHalves<Align>(&data[start0 + 6 * stride], r00, r04);
      StoreUpperHalves<Align>(&data[start1 + 6 * stride], r01, r05);

      StoreLowerHalves<Align>(&data[start2 + 2 * stride], r02, r06);
      StoreLowerHalves<Align>(&data[start3 + 2 * stride], r03, r07);

      StoreUpperHalves<Align>(&data[start2 + 6 * stride], r02, r06);
      StoreUpperHalves<Align>(&data[start3 + 6 * stride], r03, r07);

      subsetIndex += subLength;
    }

    flipGroupIndex += flipGroupLength;
    if (flipGroupIndex >= length)
      { break; }
    subsetIndex = 0_z;
    reversedFlipGroupIndex = BitRev(flipGroupIndex) >> shift;
  }
}



template <Alignment Align, std::floating_point F, uint32_t VV = VecSize, typename VT = VecType>
inline void LastDoubleFlipRadix4PassRealFwd(const F *tableData, F *data, size_t length)
{
  if constexpr (VV == 2)
  {
    const auto flipStride = length / 4;
    {
      auto start1 = 1 * flipStride;
      auto start2 = 2 * flipStride;
      auto start3 = 3 * flipStride;

      VT r00, r01, r02, r03, r04, r05, r06, r07, r08, r09, r10, r11;

      LoadStridedSets<Align>(data, 2, r00, r01, r02, r03);
      LoadStridedSets<Align>(data + start2, 2, r04, r05, r06, r07);

      Transpose2x2(r00, r04, r08);
      Transpose2x2(r01, r06, r00);
      Transpose2x2(r02, r05, r01);
      Transpose2x2(r03, r07, r02);

      auto angles0 = LoadSIMDAngleDouble(tableData, 0, flipStride);
      FinalRealDITRealComplexRadix4(r08, r00, r04, r06, r01, r02, r05, r07, angles0.w1r, angles0.w1i, angles0.w0r, angles0.w0i);

      Transpose2x2(r08, r00, r03);
      Transpose2x2(r04, r06, r08);
      Transpose2x2(r01, r02, r04);
      Transpose2x2(r05, r07, r01);

      StoreStridedSets<Align>(data, 2, r03, r00);
      StoreStridedSets<Align>(data + start2, 2, r08, r06);

      LoadStridedSets<Align>(data + start1, 2, r00, r03, r05, r06);
      LoadStridedSets<Align>(data + start3, 2, r08, r09, r10, r11);
      StoreStridedSets<Align>(data + start1, 2, r04, r02);
      StoreStridedSets<Align>(data + start3, 2, r01, r07);

      Transpose2x2(r00, r08, r01);
      Transpose2x2(r05, r10, r00);
      Transpose2x2(r03, r09, r02);
      Transpose2x2(r06, r11, r04);

      auto angles1 = LoadSIMDAngleGroup(tableData, 2, flipStride);
      DITComplexRadix4(r01, r00, r08, r10, r02, r04, r09, r11, angles1);

      Transpose2x2(r01, r00, r03);
      Transpose2x2(r08, r10, r01);
      Transpose2x2(r02, r04, r06);
      Transpose2x2(r09, r11, r02);

      StoreStridedSets<Align>(data + 4, 2, r03, r00);
      StoreStridedSets<Align>(data + start2 + 4, 2, r01, r10);
      StoreStridedSets<Align>(data + start1 + 4, 2, r06, r04);
      StoreStridedSets<Align>(data + start3 + 4, 2, r02, r11);
    }

    // Do the scalar step
    {
      auto a = data[0];
      auto b = data[1];
      data[0] = a + b;
      data[1] = a - b;
    }


    for (auto subsetIndex = 8_z; subsetIndex < flipStride; subsetIndex += 8)
    {
      const auto start0 = subsetIndex;
      auto start1 = start0 +  1 * flipStride;
      auto start2 = start0 +  2 * flipStride;
      auto start3 = start0 +  3 * flipStride;
      const auto angleIndex0 = (subsetIndex >> 1);

      VT r00, r01, r02, r03, r04, r05, r06, r07, r08, r09, r10, r11, r12;

      LoadStridedSets<Align>(data + start0, 2, r00, r01, r02, r03);
      LoadStridedSets<Align>(data + start2, 2, r04, r05, r06, r07);

      Transpose2x2(r00, r04, r08);
      Transpose2x2(r02, r06, r00);
      Transpose2x2(r01, r05, r02);
      Transpose2x2(r03, r07, r01);

      auto angles0 = LoadSIMDAngleGroup(tableData, angleIndex0, flipStride);
      DITComplexRadix4(r08, r00, r04, r06, r02, r01, r05, r07, angles0);

      Transpose2x2(r08, r00, r03);
      Transpose2x2(r04, r06, r08);
      Transpose2x2(r02, r01, r04);
      Transpose2x2(r05, r07, r02);

      StoreStridedSets<Align>(data + start0, 2, r03, r00);
      StoreStridedSets<Align>(data + start2, 2, r08, r06);
      LoadStridedSets<Align>(data + start1, 2, r00, r03, r05, r08);
      LoadStridedSets<Align>(data + start3, 2, r09, r10, r11, r12);
      StoreStridedSets<Align>(data + start1, 2, r04, r01);
      StoreStridedSets<Align>(data + start3, 2, r02, r07);

      Transpose2x2(r00, r09, r01);
      Transpose2x2(r05, r11, r00);
      Transpose2x2(r03, r10, r02);
      Transpose2x2(r08, r12, r03);

      auto angles1 = LoadSIMDAngleGroup(tableData, angleIndex0 + 2, flipStride);
      DITComplexRadix4(r01, r00, r09, r11, r02, r03, r10, r12, angles1);

      Transpose2x2(r01, r00, r04);
      Transpose2x2(r09, r11, r01);
      Transpose2x2(r02, r03, r05);
      Transpose2x2(r10, r12, r02);

      StoreStridedSets<Align>(data + start0 + 4, 2, r04, r00);
      StoreStridedSets<Align>(data + start1 + 4, 2, r05, r03);
      StoreStridedSets<Align>(data + start2 + 4, 2, r01, r11);
      StoreStridedSets<Align>(data + start3 + 4, 2, r02, r12);
    }
  }
  else if constexpr (VV == 4)
  {
    const auto flipStride = length / 4;

    // The first has one real and 3 complex radix 4s
    {
      const auto start1 = 1_z * flipStride;
      const auto start2 = 2_z * flipStride;
      const auto start3 = 3_z * flipStride;

      VT r00, r01, r02, r03, r04, r05, r06, r07, r08;
      r00 = Load<Align>(&data[0]);
      r01 = Load<Align>(&data[4]);
      r02 = Load<Align>(&data[start1 + 0]);
      r03 = Load<Align>(&data[start1 + 4]);
      r04 = Load<Align>(&data[start2 + 0]);
      r05 = Load<Align>(&data[start2 + 4]);
      r06 = Load<Align>(&data[start3 + 0]);
      r07 = Load<Align>(&data[start3 + 4]);

      // swap {e0, g0} and {b0, d0} because the real butterfly needs them in this order instead.
      SwapRealComponentsForLastDoubleFlip(r00, r01, r08);

      // Transpose these - note that 1 and 2 are out of order - this makes the deinterleave at the end simpler.
      Transpose4x4(r08, r04, r02, r06);
      Transpose4x4(r01, r05, r03, r07);

      // [0] angles are special-cased here because they're from a "previous" radix2 step (which hasn't been angle-
      //  reversed yet, effectively, so it's still in bit reverse order, hence the 1/sqrt(2) terms)
      auto angles = LoadSIMDAngleDouble(tableData, 0, flipStride);
      FinalRealDITRealComplexRadix4(r08, r01, r04, r05, r02, r03, r06, r07, angles.w1r, angles.w1i, angles.w0r, angles.w0i);

      // Re-interleave the pairs (a/b, c/d, e/f, g/h)
      Interleave4(r08, r01, r00);
      Interleave4(r04, r05, r08);
      Interleave4(r02, r03, r04);
      Interleave4(r06, r07, r02);

      Store<Align>(&data[0], r00);
      Store<Align>(&data[4], r01);
      Store<Align>(&data[start1 + 0], r04);
      Store<Align>(&data[start1 + 4], r03);
      Store<Align>(&data[start2 + 0], r08);
      Store<Align>(&data[start2 + 4], r05);
      Store<Align>(&data[start3 + 0], r02);
      Store<Align>(&data[start3 + 4], r07);
    }

    // Do this final scalar step
    {
      auto a = data[0];
      auto b = data[1];
      data[0] = a + b;
      data[1] = a - b;
    }

    // Keep this as a standard `for` loop because the condition may actually be false at the start for small FFTs.
    for (auto subsetIndex = 8_z; subsetIndex < flipStride; subsetIndex += 8)
    {
      const auto start0 = subsetIndex;
      auto start1 = start0 + 1 * flipStride;
      auto start2 = start0 + 2 * flipStride;
      auto start3 = start0 + 3 * flipStride;
      const auto angleIndex0 = subsetIndex >> 1;

      VT r00, r01, r02, r03, r04, r05, r06, r07, r08;

      r00 = Load<Align>(&data[start0 + 0]);
      r01 = Load<Align>(&data[start0 + 4]);
      r02 = Load<Align>(&data[start1 + 0]);
      r03 = Load<Align>(&data[start1 + 4]);
      r04 = Load<Align>(&data[start2 + 0]);
      r05 = Load<Align>(&data[start2 + 4]);
      r06 = Load<Align>(&data[start3 + 0]);
      r07 = Load<Align>(&data[start3 + 4]);

      // Transpose these - note that 1 and 2 are out of order - this makes the deinterleave at the end simpler (and
      //  also lines the values up with the loaded angles)

      Transpose4x4(r00, r04, r02, r06);
      Transpose4x4(r01, r05, r03, r07);

      auto angles = LoadSIMDAngleGroup(tableData, angleIndex0, flipStride);
      DITComplexRadix4(r00, r01, r04, r05, r02, r03, r06, r07, angles);

      // Reinterleave
      Interleave4(r00, r01, r08);
      Interleave4(r04, r05, r00);
      Interleave4(r02, r03, r04);
      Interleave4(r06, r07, r02);

      Store<Align>(&data[start0 + 0], r08);
      Store<Align>(&data[start0 + 4], r01);
      Store<Align>(&data[start1 + 0], r04);
      Store<Align>(&data[start1 + 4], r03);
      Store<Align>(&data[start2 + 0], r00);
      Store<Align>(&data[start2 + 4], r05);
      Store<Align>(&data[start3 + 0], r02);
      Store<Align>(&data[start3 + 4], r07);
    }
  }
  else
  {
    static_assert(VV == 8);

    const auto flipStride = length / 4;

    // The first has one real and 7 complex radix 4s
    {
      const auto start1 = 1_z * flipStride;
      const auto start2 = 2_z * flipStride;
      const auto start3 = 3_z * flipStride;

      VT r00, r01, r02, r03, r04, r05, r06, r07, r08;
      r00 = LoadHalves<Align>(&data[0], &data[0 + 8]);
      r01 = LoadHalves<Align>(&data[4], &data[4 + 8]);
      r02 = LoadHalves<Align>(&data[start1 + 0], &data[start1 + 0 + 8]);
      r03 = LoadHalves<Align>(&data[start1 + 4], &data[start1 + 4 + 8]);
      r04 = LoadHalves<Align>(&data[start2 + 0], &data[start2 + 0 + 8]);
      r05 = LoadHalves<Align>(&data[start2 + 4], &data[start2 + 4 + 8]);
      r06 = LoadHalves<Align>(&data[start3 + 0], &data[start3 + 0 + 8]);
      r07 = LoadHalves<Align>(&data[start3 + 4], &data[start3 + 4 + 8]);

      // swap {e0, g0} and {b0, d0} because the real butterfly needs them in this order instead.
      SwapRealComponentsForLastDoubleFlip(r00, r01, r08);

      // Transpose these - note that 1 and 2 are out of order - this makes the deinterleave at the end simpler.
      Transpose4x4(r08, r04, r02, r06);
      Transpose4x4(r01, r05, r03, r07);

      auto angles = LoadSIMDAngleDouble(tableData, 0, flipStride);
      FinalRealDITRealComplexRadix4(r08, r01, r04, r05, r02, r03, r06, r07, angles.w1r, angles.w1i, angles.w0r, angles.w0i);

      // Re-interleave the pairs (a/b, c/d, e/f, g/h)
      Interleave4(r08, r01, r00);
      Interleave4(r04, r05, r08);
      Interleave4(r02, r03, r04);
      Interleave4(r06, r07, r02);

      StoreHalves<Align>(&data[0], &data[0 + 8], r00);
      StoreHalves<Align>(&data[4], &data[4 + 8], r01);
      StoreHalves<Align>(&data[start1 + 0], &data[start1 + 0 + 8], r04);
      StoreHalves<Align>(&data[start1 + 4], &data[start1 + 4 + 8], r03);
      StoreHalves<Align>(&data[start2 + 0], &data[start2 + 0 + 8], r08);
      StoreHalves<Align>(&data[start2 + 4], &data[start2 + 4 + 8], r05);
      StoreHalves<Align>(&data[start3 + 0], &data[start3 + 0 + 8], r02);
      StoreHalves<Align>(&data[start3 + 4], &data[start3 + 4 + 8], r07);
    }

    // Do this final scalar step
    {
      auto a = data[0];
      auto b = data[1];
      data[0] = a + b;
      data[1] = a - b;
    }

    // Keep this as a standard `for` loop because the condition may actually be false at the start for small FFTs.
    //  This is basically the same routine as the VecSize == 4 case except we're doing two subsets at a time.
    for (auto subsetIndex = 16_z; subsetIndex < flipStride; subsetIndex += 16)
    {
      const auto start0 = subsetIndex;
      auto start1 = start0 + 1 * flipStride;
      auto start2 = start0 + 2 * flipStride;
      auto start3 = start0 + 3 * flipStride;
      const auto angleIndex0 = subsetIndex >> 1;

      VT r00, r01, r02, r03, r04, r05, r06, r07, r08;

      r00 = LoadHalves<Align>(&data[start0 + 0], &data[start0 + 0 + 8]);
      r01 = LoadHalves<Align>(&data[start0 + 4], &data[start0 + 4 + 8]);
      r02 = LoadHalves<Align>(&data[start1 + 0], &data[start1 + 0 + 8]);
      r03 = LoadHalves<Align>(&data[start1 + 4], &data[start1 + 4 + 8]);
      r04 = LoadHalves<Align>(&data[start2 + 0], &data[start2 + 0 + 8]);
      r05 = LoadHalves<Align>(&data[start2 + 4], &data[start2 + 4 + 8]);
      r06 = LoadHalves<Align>(&data[start3 + 0], &data[start3 + 0 + 8]);
      r07 = LoadHalves<Align>(&data[start3 + 4], &data[start3 + 4 + 8]);

      // Transpose these - note that 1 and 2 are out of order - this makes the deinterleave at the end simpler (and
      //  also lines the values up with the loaded angles)

      Transpose4x4(r00, r04, r02, r06);
      Transpose4x4(r01, r05, r03, r07);

      auto angles = LoadSIMDAngleGroup(tableData, angleIndex0, flipStride);
      DITComplexRadix4(r00, r01, r04, r05, r02, r03, r06, r07, angles);

      // Reinterleave
      Interleave4(r00, r01, r08);
      Interleave4(r04, r05, r00);
      Interleave4(r02, r03, r04);
      Interleave4(r06, r07, r02);

      StoreHalves<Align>(&data[start0 + 0], &data[start0 + 0 + 8], r08);
      StoreHalves<Align>(&data[start0 + 4], &data[start0 + 4 + 8], r01);
      StoreHalves<Align>(&data[start1 + 0], &data[start1 + 0 + 8], r04);
      StoreHalves<Align>(&data[start1 + 4], &data[start1 + 4 + 8], r03);
      StoreHalves<Align>(&data[start2 + 0], &data[start2 + 0 + 8], r00);
      StoreHalves<Align>(&data[start2 + 4], &data[start2 + 4 + 8], r05);
      StoreHalves<Align>(&data[start3 + 0], &data[start3 + 0 + 8], r02);
      StoreHalves<Align>(&data[start3 + 4], &data[start3 + 4 + 8], r07);
    }
  }
}


template <Alignment Align, std::floating_point F, uint32_t VS = VecSize>
void UntangleRealFFT(F *data, size_t length)
{
  if (length < 16)
    { return; }    // Nothing needs to happen for extremely short FFTs

  // This whole function relies on the convenient fact that the routine for handling the first
  //  16 or 32 or 64 etc elements of one of these reversals is the same for *all sizes*, so
  //  the first 128 elements of a 128-length FFT untangle in the exact same way that the first
  //  128 elements of a 512k FFT. That remains true at every power of 2 step up.
  //
  // For any FFTs larger than 128 we'll loop through doing longer untangles, but stop before
  //  the final 128 floats, which are special cased at the end of the function.
  //
  // Note that we loop from bottom up here, because it means we do a smaller sub-untangle
  //  within a larger one *before* the larger one happens, so that by the time we get to
  //  the larger one, the elements within are in a consistent order.
  // Example: In a 512-length FFT, we'll do the following:
  //  - untangle the last 32 pairs (64 floats) from indices 448..511
  //  - untangle the last 64 pairs (indices 384..511), doing the 8- and 16-pair flips in the
  //    first 32 of this 64-pair set, then doing the whole 64 that includes the previous 32
  //  - untangle the 32 pairs before that (indices 320..383)
  //  - untangle the 128 pairs including everything we've touched thus far (indices 256..511)
  //  - untangle the 32 before *that* (indices 192..255)
  //  - untangle the 64 before that (128..255)
  //  - At that point, we'll break out of the loop and go to the special case handling of the
  //    remaining elements (which will do 64..127, 32..64, 16..31, then finally 8..15.
  //    0..7 always already in the correct order and need no flippy flopping)

  switch (length)
  {
  default: //Any size > 128 has to do the long untangles
    for (size_t offset = length;;)
    {
      // An inner loop to handle the difference between the OuterOnly untangles (which leave
      //  any reversed triples within a set of 4 (i.e. A D C B) reversed) vs the final untangles
      //  (which would take A D C B and write it as A B C D, putting everything into the final
      //  order).
      while (true)
      {
        // We have a bespoke function to untangle 32 pairs at a time, so we're theoretically
        //  stepping by 32 pairs (64 floats).
        // *However* counting from the end to the start the length pattern we'll see looks like:
        // 32 64 32 128 32 64 32 256 32 ...
        // That is, we'll do a full 32, then the next length is 64 (it reverses the area that
        //  includes the 32 we just handles). The next one will be a solo 32 again, then the
        //  next is a 128 (flipping all of the values we've seen).
        // Hopefully this makes sense? To think about it another way, imagine you wanted to know the
        //  largest power of 2 that a given number is. 1 is 2^0, 2 is 2^1, 3 is a multiple of 1,
        //   4 is 2^2, 5 is a multiple of 1 again, 6 is a multiple of 2, etc.
        //  That pattern is 1 2 1 4 1 2 1 8 1 2 1 4 1 2 1 16 ...
        //  The above pattern is the same, just multiplied by 32 (as it's our step size).
        //
        // Anyway, because 32 pairs (64 floats, again) is a bespoke case and it is going to always
        //  happen every other time, we'll actually step by 64 pairs (128 floats) and handle it,
        //  then handle the larger case (which includes the 16- and 8-pair cases within the top
        //  32 of it that happens before doing the large flip.
        offset -= 128;

        Tangle32Pairs<Align, TangleType::ForwardInner>(data + offset + 64, data + offset + 64);

        // PERF: Hypothetically these could be folded into the long untangle function as the startup
        //  code, but for now just handle these like this.
        Tangle16Pairs<Align, TangleType::ForwardInner>(data + offset + 32, data + offset + 32);
        Tangle8Pairs<Align, TangleType::ForwardInner>(data + offset + 16, data + offset + 16);

        // Get the highest power of 2 this offset is a multiple of
        //  (offset & (offset - 1)) removes the lowest set bit from `offset` and then xoring it with
        //  `offset` again clears all the bits that didn't change and leave the one bit that did,
        //  which is said lowest set bit (which is the largest power of 2 it's a multiple of).
        const auto subLength = (offset & (offset - 1)) ^ offset;

        // If the offset *is* that power of 2, then this is the largest flip for this area of the
        //  FFT data, and we want to do it as a finalize step, so break out of the inner loop.
        if (subLength == offset)
          { break; }

        LargeTanglePairs<Align, TangleType::ForwardInner>(data + offset, data + offset, subLength);
      }

      LargeTanglePairs<Align, TangleType::ForwardLargest>(data + offset, data + offset, offset);

      if (offset == 128)
        { break; }
    }

    // Now do the bespoke flips, falling through from largest to smallest
    [[fallthrough]];

  case 128:
    Tangle32Pairs<Align, TangleType::ForwardLargest>(data + 64, data + 64);
    [[fallthrough]];

  case 64:
    Tangle16Pairs<Align, TangleType::ForwardLargest>(data + 32, data + 32);
    [[fallthrough]];

  case 32:
    Tangle8Pairs<Align, TangleType::ForwardLargest>(data + 16, data + 16);
    [[fallthrough]];

  case 16:
    TangleFinal4Pairs<Align>(data + 8, data + 8);
    break;
  }
}

template <Alignment Align, std::floating_point F, uint32_t VS = VecSize>
void RealForwardFFT(const F *tableData, const F *input, F *output, size_t fftLength)
{
  auto length = fftLength;
  if (length < MinimumRealFFTLength)
  {
    // 32 is the cutoff here, which would run an initial radix 4 then the final radix 4
    //  note that AVX will either need to fall back to SSE for 32 and 64 or special-case them (which might be good
    //  since they could fit completely within registers anyway)
    TinyRealFwd(input, output, fftLength);
    return;
  }

  const auto bitCount = std::numeric_limits<size_t>::digits - std::countl_zero(length) - 1;
  const auto bitReverseCount = (bitCount - 1) >> 1;

  Radix4InitialStepRealFwd<Align>(input, output, length);

  auto subLength = length >> 2;

  if (IsPowerOfFourAssumingIsPowerOfTwo(length))
  {
    Radix2FwdPass<Align>(output, length);
    subLength >>= 1;
  }

  for (const auto endSubLength = (4_z << bitReverseCount); subLength > endSubLength; subLength >>= 2)
    { NoFlipRadix4PassRealFwd<Align>(tableData, output, length, subLength); }

  if ((bitReverseCount & 1) != 0)
  {
    SingleFlipRadix4PassRealFwd<Align>(tableData, output, length, subLength);
    subLength >>= 2;
  }

  constexpr auto DoubleFlipLimit = (VS > 4) ? 32 : 8;
  for (; subLength > DoubleFlipLimit; subLength >>= 2)
    { DoubleFlipRadix4PassRealFwd<Align>(tableData, output, length, subLength); }

  if constexpr(VS > 4)
  {
    static_assert(VS == 8);

    if (subLength > 8)
    {
      SecondToLastDoubleFlipRadix4PassRealFwd<Align>(tableData, output, length);
    }
  }

  LastDoubleFlipRadix4PassRealFwd<Align>(tableData, output, length);

  // Do the final pass to get from the "tangled" order that the real FFT algorithm outputs in to
  //  be in-order.
  UntangleRealFFT<Align>(output, length);
}

template <std::floating_point F>
void RealForwardFFT(const F *tableData, const F *input, F *output, size_t fftLength)
{
  // Only bother with aligned vs. unaligned switch on non-ARM CPUs where the vector alignment is larger than float
  //  alignment (ARM has the same instruction for aligned vs. not so the switch doesn't matter there)
  if constexpr (!CPU_ARM_ANY && alignof(VecType) > alignof(F))
  {
    if (IsAlignedPtr(input, alignof(VecType)) && IsAlignedPtr(output, alignof(VecType)))
    {
      RealForwardFFT<Alignment::Aligned>(tableData, input, output, fftLength);
      return;
    }
  }

  RealForwardFFT<Alignment::Unaligned>(tableData, input, output, fftLength);
}
