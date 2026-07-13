template <Alignment Align, std::floating_point F>
void Radix4InitialStepRealFwd(const F *input, F *output, size_t length)
{
  const auto z = Set1(InvSqrt2<F>);
  const auto stride = length / 8;
  for (auto i = 0u; i < stride; i += VecSize)
  {
    // First radix 2
    auto [a, b, c, d, e, f, g, h] = LoadStridedSets<Align, 8>(input + i, stride);
    DITRealInputRadix4Pair(a, b, c, d, e, f, g, h, z);
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

      DITRealInputRadix2Pair(a, b, c, d);

      Store<Align>(&data[i + 0 * stride], a);
      Store<Align>(&data[i + 1 * stride], b);
      Store<Align>(&data[i + 2 * stride], c);
      Store<Align>(&data[i + 3 * stride], d);
    }
  }

  // Now do the complex values
  const F swRs[3] = { InvSqrt2<F>, CosPiOver8<F>, SinPiOver8<F> };
  const F swIs[3] = { -InvSqrt2<F>, -SinPiOver8<F>, -CosPiOver8<F> };

  for (auto complexSubset = 0u; complexSubset < 3; complexSubset++)
  {
    const auto wR = Set1(swRs[complexSubset]);
    const auto wI = Set1(swIs[complexSubset]);

    auto start = (1 + complexSubset) * stride * 4;
    for (auto i = 0u; i < stride; i += VecSize)
    {
      auto aR = Load<Align>(&data[start + i + 0 * stride]);
      auto bR = Load<Align>(&data[start + i + 1 * stride]);
      auto aI = Load<Align>(&data[start + i + 2 * stride]);
      auto bI = Load<Align>(&data[start + i + 3 * stride]);

      DITComplexRadix2(aR, aI, bR, bI, wR, wI);

      Store<Align>(&data[start + i + 0 * stride], aR);
      Store<Align>(&data[start + i + 1 * stride], aI);
      Store<Align>(&data[start + i + 2 * stride], bR);
      Store<Align>(&data[start + i + 3 * stride], bI);
    }
  }
}



FORCE_INLINE auto GetScalarAngleIndices(size_t angleIndex, size_t angleCount)
{
  auto r1 = angleCount * 2;
  auto r0 = angleCount;
  auto i1 = r1 + angleCount;
  auto i0 = r0 + angleCount / 2;

  return std::tuple{ r0 + angleIndex, i0 + angleIndex, r1 + angleIndex, i1 + angleIndex };
}



template <std::floating_point F>
FORCE_INLINE auto LoadScalarAngleTriple(const F *tableData, size_t angleIndex, size_t angleCount)
{
  auto [r0, i0, r1, i1] = GetScalarAngleIndices(angleIndex, angleCount);
  auto wr1 = tableData[r0];
  auto wi1 = tableData[i0];
  auto wr2 = tableData[r1];
  auto wi2 = tableData[i1];

  // Third angle calculated based on the first two.
  return std::tuple{wr2, wi2, wr1, wi1, wr1 * wr2 - wi1 * wi2, wr1 * wi2 + wi1 * wr2};
}



template <std::floating_point F>
FORCE_INLINE auto LoadSIMDAngleDouble(const F *tableData, size_t angleIndex, size_t angleCount)
{
  auto [r0, i0, r1, i1] = GetScalarAngleIndices(angleIndex, angleCount);
  auto wr1 = Load<Alignment::Aligned>(&tableData[r0]);
  auto wi1 = Load<Alignment::Aligned>(&tableData[i0]);
  auto wr2 = Load<Alignment::Aligned>(&tableData[r1]);
  auto wi2 = Load<Alignment::Aligned>(&tableData[i1]);

  return std::tuple{wr2, wi2, wr1, wi1};
}



template <Alignment Align, std::floating_point F>
inline void NoFlipRadix4PassRealFwd(const F *tableData, F *data, size_t length, size_t subLength)
{
  const auto angleCount = 2 * length / subLength;

  const auto stride = subLength / 8;

  // Do the reals (subset 0)
  {
    const auto z = Set1(InvSqrt2<F>);

    for (auto i = 0u; i < stride; i += VecSize)
    {
      auto [a, b, c, d, e, f, g, h] = LoadStridedSets<Align, 8>(data + i, stride);
      DITRealInputRadix4Pair(a, b, c, d, e, f, g, h, z);
      StoreStridedSets<Align>(data + i, stride, a, b, c, d, e, f, g, h);
    }
  }

  // Now do the complex values
  for (auto subsetStartIndex = subLength; subsetStartIndex < length; subsetStartIndex += subLength)
  {
    const auto angleIndex = BitRev(subsetStartIndex) >> (1 + std::countl_zero(length));
    const auto [sw0r, sw0i, sw1r, sw1i, sw2r, sw2i] = LoadScalarAngleTriple(tableData, angleIndex, angleCount);

    const auto w0r = Set1(sw0r);
    const auto w0i = Set1(sw0i);
    const auto w1r = Set1(sw1r);
    const auto w1i = Set1(sw1i);
    const auto w2r = Set1(sw2r);
    const auto w2i = Set1(sw2i);

    for (auto i = subsetStartIndex; i < subsetStartIndex + stride; i += VecSize)
    {
      auto [aR, bR, cR, dR, aI, bI, cI, dI] = LoadStridedSets<Align, 8>(data + i, stride);
      DITComplexRadix4(aR, aI, bR, bI, cR, cI, dR, dI, w0r, w0i, w1r, w1i, w2r, w2i);
      StoreStridedSets<Align>(data + i, stride, aR, aI, bR, bI, cR, cI, dR, dI);
    }
  }
}



template <Alignment Align, std::floating_point F>
inline void SingleFlipRadix4PassRealFwd(const F *tableData, F *data, size_t length, size_t subLength)
{
  if (IsPowerOfFourAssumingIsPowerOfTwo(length))
  {
    //This flip needs 2 radix 4s, but they are conveniently sequential
    const auto angleCount = 2 * length / subLength;
    const auto stride = subLength / 8;

    {
      const auto angleIndex1 = stride;
      const auto [sw0r_1, sw0i_1, sw1r_1, sw1i_1, sw2r_1, sw2i_1]
        = LoadScalarAngleTriple(tableData, angleIndex1, angleCount);

      // If we load these once, they can stay resident (we'll need to re-set 1/sqrt2 every loop)
      const auto w0r_1 = Set1(sw0r_1);
      const auto w0i_1 = Set1(sw0i_1);
      const auto w1r_1 = Set1(sw1r_1);
      const auto w1i_1 = Set1(sw1i_1);
      const auto w2r_1 = Set1(sw2r_1);
      const auto w2i_1 = Set1(sw2i_1);

      for (auto i = 0u; i < subLength / 8; i += VecSize)
      {
        // 8+6 == 14 loaded ( + 1 temp)
        auto [a0, b0, c0, d0, e0, f0, g0, h0] = LoadStridedSets<Align, 8>(data + i, stride);

        DITRealInputRadix4Pair(a0, b0, c0, d0, e0, f0, g0, h0, InvSqrt2<F>);

        // Now down to 11
        Store<Align>(&data[i + 0 * stride], a0);
        Store<Align>(&data[i + 1 * stride], b0);
        Store<Align>(&data[i + 4 * stride], e0);
        Store<Align>(&data[i + 5 * stride], f0);

        auto aR1 = Load<Align>(&data[i + subLength + 0 * stride]);
        auto bR1 = Load<Align>(&data[i + subLength + 1 * stride]);
        Store<Align>(&data[i + subLength + 0 * stride], c0);
        Store<Align>(&data[i + subLength + 1 * stride], d0);

        auto aI1 = Load<Align>(&data[i + subLength + 4 * stride]);
        auto bI1 = Load<Align>(&data[i + subLength + 5 * stride]);
        Store<Align>(&data[i + subLength + 4 * stride], g0);
        Store<Align>(&data[i + subLength + 5 * stride], h0);

        // Still 11

        auto cR1 = Load<Align>(&data[i + subLength + 2 * stride]);
        auto dR1 = Load<Align>(&data[i + subLength + 3 * stride]);
        auto cI1 = Load<Align>(&data[i + subLength + 6 * stride]);
        auto dI1 = Load<Align>(&data[i + subLength + 7 * stride]);

        // Back up to 15, but now we can jettison `z` to be 14, and now we have 2 temp registers, so 16 total
        DITComplexRadix4(aR1, aI1, bR1, bI1, cR1, cI1, dR1, dI1, w0r_1, w0i_1, w1r_1, w1i_1, w2r_1, w2i_1);

        Store<Align>(&data[i + 2 * stride], aR1);
        Store<Align>(&data[i + 3 * stride], aI1);
        Store<Align>(&data[i + 6 * stride], cR1);
        Store<Align>(&data[i + 7 * stride], cI1);
        Store<Align>(&data[i + subLength + 2 * stride], bR1);
        Store<Align>(&data[i + subLength + 3 * stride], bI1);
        Store<Align>(&data[i + subLength + 6 * stride], dR1);
        Store<Align>(&data[i + subLength + 7 * stride], dI1);
      }
    }

    const auto subsetStep = subLength * 2;
    for (auto subsetStartIndex = subsetStep; subsetStartIndex < length; subsetStartIndex += subsetStep)
    {
      const auto angleIndex0 = BitRev(subsetStartIndex) >> (1 + std::countl_zero(length));
      const auto angleIndex1 = angleIndex0 + stride;

      const auto [sw0r_0, sw0i_0, sw1r_0, sw1i_0, sw2r_0, sw2i_0]
        = LoadScalarAngleTriple(tableData, angleIndex0, angleCount);

      const auto [sw0r_1, sw0i_1, sw1r_1, sw1i_1, sw2r_1, sw2i_1]
        = LoadScalarAngleTriple(tableData, angleIndex1, angleCount);

      for (auto i = subsetStartIndex; i < subsetStartIndex + stride; i += VecSize)
      {
        auto [aR0, bR0, cR0, dR0, aI0, bI0, cI0, dI0] = LoadStridedSets<Align, 8>(data + i, stride);

        DITComplexRadix4(aR0, aI0, bR0, bI0, cR0, cI0, dR0, dI0, sw0r_0, sw0i_0, sw1r_0, sw1i_0, sw2r_0, sw2i_0);

        Store<Align>(&data[i + 0 * stride], aR0);
        Store<Align>(&data[i + 1 * stride], aI0);
        Store<Align>(&data[i + 4 * stride], cR0);
        Store<Align>(&data[i + 5 * stride], cI0);

        auto aR1 = Load<Align>(&data[i + subLength + 0 * stride]);
        auto bR1 = Load<Align>(&data[i + subLength + 1 * stride]);
        Store<Align>(&data[i + subLength + 0 * stride], bR0);
        Store<Align>(&data[i + subLength + 1 * stride], bI0);
        auto aI1 = Load<Align>(&data[i + subLength + 4 * stride]);
        auto bI1 = Load<Align>(&data[i + subLength + 5 * stride]);
        Store<Align>(&data[i + subLength + 4 * stride], dR0);
        Store<Align>(&data[i + subLength + 5 * stride], dI0);

        auto cR1 = Load<Align>(&data[i + subLength + 2 * stride]);
        auto dR1 = Load<Align>(&data[i + subLength + 3 * stride]);
        auto cI1 = Load<Align>(&data[i + subLength + 6 * stride]);
        auto dI1 = Load<Align>(&data[i + subLength + 7 * stride]);

        DITComplexRadix4(aR1, aI1, bR1, bI1, cR1, cI1, dR1, dI1, sw0r_1, sw0i_1, sw1r_1, sw1i_1, sw2r_1, sw2i_1);

        Store<Align>(&data[i + 2 * stride], aR1);
        Store<Align>(&data[i + 3 * stride], aI1);
        Store<Align>(&data[i + 6 * stride], cR1);
        Store<Align>(&data[i + 7 * stride], cI1);
        Store<Align>(&data[i + subLength + 2 * stride], bR1);
        Store<Align>(&data[i + subLength + 3 * stride], bI1);
        Store<Align>(&data[i + subLength + 6 * stride], dR1);
        Store<Align>(&data[i + subLength + 7 * stride], dI1);
      }
    }
  }
  else
  {
    // This one is great because it's a swap within a single radix 4!
    const auto angleCount = 2 * length / subLength;

    // Do the reals (subset 0)
    const auto stride = subLength / 8;
    {
      const auto z = Set1(InvSqrt2<F>);

      for (auto i = 0u; i < subLength / 8; i += VecSize)
      {
        auto [a, b, c, d, e, f, g, h] = LoadStridedSets<Align, 8>(data + i, stride);
        DITRealInputRadix4Pair(a, b, c, d, e, f, g, h, z);
        StoreStridedSets<Align>(data + i, stride, a, b, e, f, c, d, g, h);
      }
    }

    // Now do the complex values
    for (auto subsetStartIndex = subLength; subsetStartIndex < length; subsetStartIndex += subLength)
    {
      const auto angleIndex = BitRev(subsetStartIndex) >> (1 + std::countl_zero(length));
      const auto [sw0r, sw0i, sw1r, sw1i, sw2r, sw2i] = LoadScalarAngleTriple(tableData, angleIndex, angleCount);

      const auto w0r = Set1(sw0r);
      const auto w0i = Set1(sw0i);
      const auto w1r = Set1(sw1r);
      const auto w1i = Set1(sw1i);
      const auto w2r = Set1(sw2r);
      const auto w2i = Set1(sw2i);

      for (auto i = subsetStartIndex; i < subsetStartIndex + stride; i += VecSize)
      {
        auto [aR, bR, cR, dR, aI, bI, cI, dI] = LoadStridedSets<Align, 8>(data + i, stride);
        DITComplexRadix4(aR, aI, bR, bI, cR, cI, dR, dI, w0r, w0i, w1r, w1i, w2r, w2i);
        StoreStridedSets<Align>(data + i, stride, aR, aI, cR, cI, bR, bI, dR, dI);
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

    const auto [sw0r_1, sw0i_1, sw1r_1, sw1i_1, sw2r_1, sw2i_1]
      = LoadScalarAngleTriple(tableData, angleIndex1, flipStride);
    const auto [sw0r_2, sw0i_2, sw1r_2, sw1i_2, sw2r_2, sw2i_2]
      = LoadScalarAngleTriple(tableData, angleIndex2, flipStride);
    const auto [sw0r_3, sw0i_3, sw1r_3, sw1i_3, sw2r_3, sw2i_3]
      = LoadScalarAngleTriple(tableData, angleIndex3, flipStride);
    for (auto i = 0u;;)
    {
      // 8 sets loaded
      auto a0 = Load<Align>(&data[start0 + i + 0 * stride]);
      auto b0 = Load<Align>(&data[start0 + i + 1 * stride]);
      auto c0 = Load<Align>(&data[start0 + i + 2 * stride]);
      auto d0 = Load<Align>(&data[start0 + i + 3 * stride]);
      auto e0 = Load<Align>(&data[start0 + i + 4 * stride]);
      auto f0 = Load<Align>(&data[start0 + i + 5 * stride]);
      auto g0 = Load<Align>(&data[start0 + i + 6 * stride]);
      auto h0 = Load<Align>(&data[start0 + i + 7 * stride]);

      DITRealInputRadix4Pair(a0, b0, c0, d0, e0, f0, g0, h0, InvSqrt2<F>);

      Store<Align>(&data[start0 + i + 0 * stride], a0);
      Store<Align>(&data[start0 + i + 1 * stride], b0);

      // 6 sets loaded
      auto a2 = Load<Align>(&data[start2 + i + 0 * stride]);
      auto c2 = Load<Align>(&data[start2 + i + 1 * stride]);
      Store<Align>(&data[start2 + i + 0 * stride], c0);
      Store<Align>(&data[start2 + i + 1 * stride], d0);

      auto a3 = Load<Align>(&data[start3 + i + 0 * stride]);
      auto c3 = Load<Align>(&data[start3 + i + 1 * stride]);
      Store<Align>(&data[start3 + i + 0 * stride], g0);
      Store<Align>(&data[start3 + i + 1 * stride], h0);

      auto a1 = Load<Align>(&data[start1 + i + 0 * stride]);
      auto c1 = Load<Align>(&data[start1 + i + 1 * stride]);
      Store<Align>(&data[start1 + i + 0 * stride], e0);
      Store<Align>(&data[start1 + i + 1 * stride], f0);

      // still 6 sets loaded

      auto e1 = Load<Align>(&data[start1 + i + 2 * stride]);
      auto g1 = Load<Align>(&data[start1 + i + 3 * stride]);
      auto b1 = Load<Align>(&data[start1 + i + 4 * stride]);
      auto d1 = Load<Align>(&data[start1 + i + 5 * stride]);
      auto f1 = Load<Align>(&data[start1 + i + 6 * stride]);
      auto h1 = Load<Align>(&data[start1 + i + 7 * stride]);

      // now 12 sets - R4 needs 4 additional registers (2 to hold current angle, 2 temp) so this is 16 total
      DITComplexRadix4(a1, b1, c1, d1, e1, f1, g1, h1, sw0r_1, sw0i_1, sw1r_1, sw1i_1, sw2r_1, sw2i_1);

      Store<Align>(&data[start0 + i + 4 * stride], a1);
      Store<Align>(&data[start0 + i + 5 * stride], b1);
      Store<Align>(&data[start1 + i + 4 * stride], e1);
      Store<Align>(&data[start1 + i + 5 * stride], f1);

      // now 8 sets loaded

      auto b3 = Load<Align>(&data[start3 + i + 4 * stride]);
      auto d3 = Load<Align>(&data[start3 + i + 5 * stride]);
      Store<Align>(&data[start3 + i + 4 * stride], g1);
      Store<Align>(&data[start3 + i + 5 * stride], h1);

      auto b2 = Load<Align>(&data[start2 + i + 4 * stride]);
      auto d2 = Load<Align>(&data[start2 + i + 5 * stride]);
      Store<Align>(&data[start2 + i + 4 * stride], c1);
      Store<Align>(&data[start2 + i + 5 * stride], d1);

      // Still 8 loaded

      auto e2 = Load<Align>(&data[start2 + i + 2 * stride]);
      auto g2 = Load<Align>(&data[start2 + i + 3 * stride]);
      auto f2 = Load<Align>(&data[start2 + i + 6 * stride]);
      auto h2 = Load<Align>(&data[start2 + i + 7 * stride]);

      // Back up to 12 - still good on register space
      DITComplexRadix4(a2, b2, c2, d2, e2, f2, g2, h2, sw0r_2, sw0i_2, sw1r_2, sw1i_2, sw2r_2, sw2i_2);

      Store<Align>(&data[start0 + i + 2 * stride], a2);
      Store<Align>(&data[start0 + i + 3 * stride], b2);
      Store<Align>(&data[start1 + i + 2 * stride], e2);
      Store<Align>(&data[start1 + i + 3 * stride], f2);
      Store<Align>(&data[start2 + i + 2 * stride], c2);
      Store<Align>(&data[start2 + i + 3 * stride], d2);

      // Down to 6 registers now

      auto e3 = Load<Align>(&data[start3 + i + 2 * stride]);
      auto g3 = Load<Align>(&data[start3 + i + 3 * stride]);
      Store<Align>(&data[start3 + i + 2 * stride], g2);
      Store<Align>(&data[start3 + i + 3 * stride], h2);

      auto f3 = Load<Align>(&data[start3 + i + 6 * stride]);
      auto h3 = Load<Align>(&data[start3 + i + 7 * stride]);

      // At 8

      DITComplexRadix4(a3, b3, c3, d3, e3, f3, g3, h3, sw0r_3, sw0i_3, sw1r_3, sw1i_3, sw2r_3, sw2i_3);

      // Transpose and store
      Store<Align>(&data[start0 + i + 6 * stride], a3);
      Store<Align>(&data[start0 + i + 7 * stride], b3);
      Store<Align>(&data[start1 + i + 6 * stride], e3);
      Store<Align>(&data[start1 + i + 7 * stride], f3);
      Store<Align>(&data[start2 + i + 6 * stride], c3);
      Store<Align>(&data[start2 + i + 7 * stride], d3);
      Store<Align>(&data[start3 + i + 6 * stride], g3);
      Store<Align>(&data[start3 + i + 7 * stride], h3);

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

      const auto [sw0r_0, sw0i_0, sw1r_0, sw1i_0, sw2r_0, sw2i_0]
        = LoadScalarAngleTriple(tableData, angleIndex0, flipStride);
      const auto [sw0r_1, sw0i_1, sw1r_1, sw1i_1, sw2r_1, sw2i_1]
        = LoadScalarAngleTriple(tableData, angleIndex1, flipStride);
      const auto [sw0r_2, sw0i_2, sw1r_2, sw1i_2, sw2r_2, sw2i_2]
        = LoadScalarAngleTriple(tableData, angleIndex2, flipStride);
      const auto [sw0r_3, sw0i_3, sw1r_3, sw1i_3, sw2r_3, sw2i_3]
        = LoadScalarAngleTriple(tableData, angleIndex3, flipStride);
      for (auto i = 0u;;)
      {
        auto a0 = Load<Align>(&data[start0 + i + 0 * stride]);
        auto c0 = Load<Align>(&data[start0 + i + 1 * stride]);
        auto e0 = Load<Align>(&data[start0 + i + 2 * stride]);
        auto g0 = Load<Align>(&data[start0 + i + 3 * stride]);
        auto b0 = Load<Align>(&data[start0 + i + 4 * stride]);
        auto d0 = Load<Align>(&data[start0 + i + 5 * stride]);
        auto f0 = Load<Align>(&data[start0 + i + 6 * stride]);
        auto h0 = Load<Align>(&data[start0 + i + 7 * stride]);

        DITComplexRadix4(a0, b0, c0, d0, e0, f0, g0, h0, sw0r_0, sw0i_0, sw1r_0, sw1i_0, sw2r_0, sw2i_0);

        Store<Align>(&data[start0 + i + 0 * stride], a0);
        Store<Align>(&data[start0 + i + 1 * stride], b0);

        auto a2 = Load<Align>(&data[start2 + i + 0 * stride]);
        auto c2 = Load<Align>(&data[start2 + i + 1 * stride]);
        Store<Align>(&data[start2 + i + 0 * stride], c0);
        Store<Align>(&data[start2 + i + 1 * stride], d0);

        auto a1 = Load<Align>(&data[start1 + i + 0 * stride]);
        auto c1 = Load<Align>(&data[start1 + i + 1 * stride]);
        Store<Align>(&data[start1 + i + 0 * stride], e0);
        Store<Align>(&data[start1 + i + 1 * stride], f0);

        auto a3 = Load<Align>(&data[start3 + i + 0 * stride]);
        auto c3 = Load<Align>(&data[start3 + i + 1 * stride]);
        Store<Align>(&data[start3 + i + 0 * stride], g0);
        Store<Align>(&data[start3 + i + 1 * stride], h0);


        auto e1 = Load<Align>(&data[start1 + i + 2 * stride]);
        auto g1 = Load<Align>(&data[start1 + i + 3 * stride]);
        auto b1 = Load<Align>(&data[start1 + i + 4 * stride]);
        auto d1 = Load<Align>(&data[start1 + i + 5 * stride]);
        auto f1 = Load<Align>(&data[start1 + i + 6 * stride]);
        auto h1 = Load<Align>(&data[start1 + i + 7 * stride]);

        DITComplexRadix4(a1, b1, c1, d1, e1, f1, g1, h1, sw0r_1, sw0i_1, sw1r_1, sw1i_1, sw2r_1, sw2i_1);

        Store<Align>(&data[start0 + i + 4 * stride], a1);
        Store<Align>(&data[start0 + i + 5 * stride], b1);
        Store<Align>(&data[start1 + i + 4 * stride], e1);
        Store<Align>(&data[start1 + i + 5 * stride], f1);

        auto b2 = Load<Align>(&data[start2 + i + 4 * stride]);
        auto d2 = Load<Align>(&data[start2 + i + 5 * stride]);
        Store<Align>(&data[start2 + i + 4 * stride], c1);
        Store<Align>(&data[start2 + i + 5 * stride], d1);

        auto b3 = Load<Align>(&data[start3 + i + 4 * stride]);
        auto d3 = Load<Align>(&data[start3 + i + 5 * stride]);
        Store<Align>(&data[start3 + i + 4 * stride], g1);
        Store<Align>(&data[start3 + i + 5 * stride], h1);

        auto e2 = Load<Align>(&data[start2 + i + 2 * stride]);
        auto g2 = Load<Align>(&data[start2 + i + 3 * stride]);
        auto f2 = Load<Align>(&data[start2 + i + 6 * stride]);
        auto h2 = Load<Align>(&data[start2 + i + 7 * stride]);

        DITComplexRadix4(a2, b2, c2, d2, e2, f2, g2, h2, sw0r_2, sw0i_2, sw1r_2, sw1i_2, sw2r_2, sw2i_2);

        Store<Align>(&data[start0 + i + 2 * stride], a2);
        Store<Align>(&data[start0 + i + 3 * stride], b2);
        Store<Align>(&data[start1 + i + 2 * stride], e2);
        Store<Align>(&data[start1 + i + 3 * stride], f2);
        Store<Align>(&data[start2 + i + 2 * stride], c2);
        Store<Align>(&data[start2 + i + 3 * stride], d2);

        auto e3 = Load<Align>(&data[start3 + i + 2 * stride]);
        auto g3 = Load<Align>(&data[start3 + i + 3 * stride]);
        Store<Align>(&data[start3 + i + 2 * stride], g2);
        Store<Align>(&data[start3 + i + 3 * stride], h2);

        auto f3 = Load<Align>(&data[start3 + i + 6 * stride]);
        auto h3 = Load<Align>(&data[start3 + i + 7 * stride]);

        DITComplexRadix4(a3, b3, c3, d3, e3, f3, g3, h3, sw0r_3, sw0i_3, sw1r_3, sw1i_3, sw2r_3, sw2i_3);

        // Transpose and store
        Store<Align>(&data[start0 + i + 6 * stride], a3);
        Store<Align>(&data[start0 + i + 7 * stride], b3);
        Store<Align>(&data[start1 + i + 6 * stride], e3);
        Store<Align>(&data[start1 + i + 7 * stride], f3);
        Store<Align>(&data[start2 + i + 6 * stride], c3);
        Store<Align>(&data[start2 + i + 7 * stride], d3);
        Store<Align>(&data[start3 + i + 6 * stride], g3);
        Store<Align>(&data[start3 + i + 7 * stride], h3);

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
inline void LastDoubleFlipRadix4PassRealFwd(const F *tableData, F *data, size_t length)
{
  const auto flipStride = length / 4;

  // The first has one real and 3 complex radix 4s
  {
    const auto start1 = 1_z * flipStride;
    const auto start2 = 2_z * flipStride;
    const auto start3 = 3_z * flipStride;

    auto aceg0 = Load<Align>(&data[0]);
    auto bdfh0 = Load<Align>(&data[4]);
    auto aceg1 = Load<Align>(&data[start1 + 0]);
    auto bdfh1 = Load<Align>(&data[start1 + 4]);
    auto aceg2 = Load<Align>(&data[start2 + 0]);
    auto bdfh2 = Load<Align>(&data[start2 + 4]);
    auto aceg3 = Load<Align>(&data[start3 + 0]);
    auto bdfh3 = Load<Align>(&data[start3 + 4]);

    // swap {e0, g0} and {b0, d0} because the real butterfly needs them in this order instead.
    SwapRealComponentsForLastDoubleFlip(aceg0, bdfh0);

    // Transpose these - note that 1 and 2 are out of order - this makes the deinterleave at the end simpler.
    auto [a, c, e, g] = Transpose4(aceg0, aceg2, aceg1, aceg3);
    auto [b, d, f, h] = Transpose4(bdfh0, bdfh2, bdfh1, bdfh3);

    // [0] angles are special-cased here because they're from a "previous" radix2 step (which hasn't been angle-
    //  reversed yet, effectively, so it's still in bit reverse order, hence the 1/sqrt(2) terms)
    const auto [w1R, w1I, w0R, w0I] = LoadSIMDAngleDouble(tableData, 0, flipStride);
    FinalRealDITRealComplexRadix4(a, b, c, d, e, f, g, h, w0R, w0I, w1R, w1I);

    // Re-interleave the pairs (a/b, c/d, e/f, g/h)
    auto [ab01, ab23] = Interleave4(a, b);
    auto [cd01, cd23] = Interleave4(c, d);
    auto [ef01, ef23] = Interleave4(e, f);
    auto [gh01, gh23] = Interleave4(g, h);

    Store<Align>(&data[0], ab01);
    Store<Align>(&data[4], ab23);
    Store<Align>(&data[start1 + 0], ef01);
    Store<Align>(&data[start1 + 4], ef23);
    Store<Align>(&data[start2 + 0], cd01);
    Store<Align>(&data[start2 + 4], cd23);
    Store<Align>(&data[start3 + 0], gh01);
    Store<Align>(&data[start3 + 4], gh23);
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

    auto aceg0 = Load<Align>(&data[start0 + 0]);
    auto bdfh0 = Load<Align>(&data[start0 + 4]);
    auto aceg1 = Load<Align>(&data[start1 + 0]);
    auto bdfh1 = Load<Align>(&data[start1 + 4]);
    auto aceg2 = Load<Align>(&data[start2 + 0]);
    auto bdfh2 = Load<Align>(&data[start2 + 4]);
    auto aceg3 = Load<Align>(&data[start3 + 0]);
    auto bdfh3 = Load<Align>(&data[start3 + 4]);

    // Transpose these - note that 1 and 2 are out of order - this makes the deinterleave at the end simpler (and
    //  also lines the values up with the loaded angles)
    auto [a, c, e, g] = Transpose4(aceg0, aceg2, aceg1, aceg3);
    auto [b, d, f, h] = Transpose4(bdfh0, bdfh2, bdfh1, bdfh3);

    const auto [w0R, w0I, w1R, w1I] = LoadSIMDAngleDouble(tableData, angleIndex0, flipStride);
    const auto [w2R, w2I] = ComplexMultiply(w0R, w0I, w1R, w1I);

    DITComplexRadix4(a, b, c, d, e, f, g, h, w0R, w0I, w1R, w1I, w2R, w2I);

    // Reinterleave
    auto [ab01, ab23] = Interleave4(a, b);
    auto [cd01, cd23] = Interleave4(c, d);
    auto [ef01, ef23] = Interleave4(e, f);
    auto [gh01, gh23] = Interleave4(g, h);

    Store<Align>(&data[start0 + 0], ab01);
    Store<Align>(&data[start0 + 4], ab23);
    Store<Align>(&data[start1 + 0], ef01);
    Store<Align>(&data[start1 + 4], ef23);
    Store<Align>(&data[start2 + 0], cd01);
    Store<Align>(&data[start2 + 4], cd23);
    Store<Align>(&data[start3 + 0], gh01);
    Store<Align>(&data[start3 + 4], gh23);
  }
}


template <Alignment Align>
void UntangleRealFFT(float *data, size_t length)
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
    TangleFinal4Pairs<Align, TangleType::InverseLargest>(data + 8, data + 8);
    break;
  }
}

template <Alignment Align, std::floating_point F>
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

  for (; subLength > 8; subLength >>= 2)
    { DoubleFlipRadix4PassRealFwd<Align>(tableData, output, length, subLength); }

  LastDoubleFlipRadix4PassRealFwd<Align>(tableData, output, length);

  // Do the final pass to get from the "tangled" order that the real FFT algorithm outputs in to
  //  be in-order.
  UntangleRealFFT<Align>(output, length);
}
