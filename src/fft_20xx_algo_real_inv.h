  template <Alignment Align, std::floating_point F>
  void TangleRealFFT(const F *input, F *output, size_t length)
  {
    if (length < 16)
      { return; }

    switch (length)
    {
    default:
      for (size_t outer = length / 2;;)
      {
        LargeTanglePairs<Align, TangleType::InverseLargest>(input + outer, output + outer, outer);
        Tangle8Pairs<Align, TangleType::InverseInner>(output + outer + 16, output + outer + 16);
        Tangle16Pairs<Align, TangleType::InverseInner>(output + outer + 32, output + outer + 32);
        Tangle32Pairs<Align, TangleType::InverseInner>(output + outer + 64, output + outer + 64);

        for (auto offset = outer + 128; offset < 2 * outer; offset += 128)
        {
          const auto subLength = (offset & (offset - 1)) ^ offset;
          LargeTanglePairs<Align, TangleType::InverseInner>(output + offset, output + offset, subLength);
          Tangle8Pairs<Align, TangleType::InverseInner>(output + offset + 16, output + offset + 16);
          Tangle16Pairs<Align, TangleType::InverseInner>(output + offset + 32, output + offset + 32);
          Tangle32Pairs<Align, TangleType::InverseInner>(output + offset + 64, output + offset + 64);
        }

        outer /= 2;
        if (outer <= 64)
          { break; }
      }

      [[fallthrough]];

    case 128:
      Tangle32Pairs<Align, TangleType::InverseLargest>(input + 64, output + 64);
      [[fallthrough]];

    case 64:
      Tangle16Pairs<Align, TangleType::InverseLargest>(input + 32, output + 32);
      [[fallthrough]];

    case 32:
      Tangle8Pairs<Align, TangleType::InverseLargest>(input + 16, output + 16);
      [[fallthrough]];

    case 16:
      Copy4Pairs<Align>(input, output);
      TangleFinal4Pairs<Align, TangleType::InverseLargest>(input + 8, output + 8);
      return;
    }
  }


  template <Alignment Align, std::floating_point F>
  void FirstDoubleFlipRadix4PassRealInv(const F *tableData, F *data, size_t length)
  {
    const auto flipStride = length / 4;

    // Start with the scalar step
    {
      auto a = data[0];
      auto b = data[1];
      data[0] = (a + b) / 2;
      data[1] = (a - b) / 2;
    }

    {
      const auto start1 = 1_z * flipStride;
      const auto start2 = 2_z * flipStride;
      const auto start3 = 3_z * flipStride;

      auto ab01 = Load<Align>(&data[0]);
      auto ab23 = Load<Align>(&data[4]);
      auto ef01 = Load<Align>(&data[start1 + 0]);
      auto ef23 = Load<Align>(&data[start1 + 4]);
      auto cd01 = Load<Align>(&data[start2 + 0]);
      auto cd23 = Load<Align>(&data[start2 + 4]);
      auto gh01 = Load<Align>(&data[start3 + 0]);
      auto gh23 = Load<Align>(&data[start3 + 4]);

      // Re-interleave the pairs (a/b, c/d, e/f, g/h)
      auto [a, b] = Deinterleave4(ab01, ab23);
      auto [c, d] = Deinterleave4(cd01, cd23);
      auto [e, f] = Deinterleave4(ef01, ef23);
      auto [g, h] = Deinterleave4(gh01, gh23);

      // [0] angles are special-cased here because they're from a "previous" radix2 step (which hasn't been angle-
      //  reversed yet, effectively, so it's still in bit reverse order, hence the 1/sqrt(2) terms)
      const auto [w1R, w1I, w0R, w0I] = LoadSIMDAngleDouble(tableData, 0, flipStride);
      FirstRealDITRealComplexRadix4Inv(a, b, c, d, e, f, g, h, w0R, w0I, w1R, w1I);

      // Transpose these - note that 1 and 2 are out of order - this makes the deinterleave at the end simpler.
      auto [aceg0, aceg2, aceg1, aceg3] = Transpose4(a, c, e, g);
      auto [bdfh0, bdfh2, bdfh1, bdfh3] = Transpose4(b, d, f, h);

      // swap {e0, g0} and {b0, d0} because the real butterfly needs them in this order instead.
      SwapRealComponentsForLastDoubleFlip(aceg0, bdfh0);

      Store<Align>(&data[0], aceg0);
      Store<Align>(&data[4], bdfh0);
      Store<Align>(&data[start1 + 0], aceg1);
      Store<Align>(&data[start1 + 4], bdfh1);
      Store<Align>(&data[start2 + 0], aceg2);
      Store<Align>(&data[start2 + 4], bdfh2);
      Store<Align>(&data[start3 + 0], aceg3);
      Store<Align>(&data[start3 + 4], bdfh3);
    }

    for (auto subsetIndex = 8_z; subsetIndex < flipStride; subsetIndex += 8)
    {
      const auto start0 = subsetIndex;
      auto start1 = start0 + 1 * flipStride;
      auto start2 = start0 + 2 * flipStride;
      auto start3 = start0 + 3 * flipStride;
      const auto angleIndex0 = subsetIndex >> 1;

      auto ab01 = Load<Align>(&data[start0 + 0]);
      auto ab23 = Load<Align>(&data[start0 + 4]);
      auto ef01 = Load<Align>(&data[start1 + 0]);
      auto ef23 = Load<Align>(&data[start1 + 4]);
      auto cd01 = Load<Align>(&data[start2 + 0]);
      auto cd23 = Load<Align>(&data[start2 + 4]);
      auto gh01 = Load<Align>(&data[start3 + 0]);
      auto gh23 = Load<Align>(&data[start3 + 4]);

      auto [a, b] = Deinterleave4(ab01, ab23);
      auto [c, d] = Deinterleave4(cd01, cd23);
      auto [e, f] = Deinterleave4(ef01, ef23);
      auto [g, h] = Deinterleave4(gh01, gh23);

      const auto [w0R, w0I, w1R, w1I] = LoadSIMDAngleDouble(tableData, angleIndex0, flipStride);
      const auto [w2R, w2I] = ComplexMultiply(w0R, w0I, w1R, w1I);
      DIFComplexRadix4Inv(a, b, c, d, e, f, g, h, w0R, w0I, w1R, w1I, w2R, w2I);

      auto [aceg0, aceg2, aceg1, aceg3] = Transpose4(a, c, e, g);
      auto [bdfh0, bdfh2, bdfh1, bdfh3] = Transpose4(b, d, f, h);

      Store<Align>(&data[start0 + 0], aceg0);
      Store<Align>(&data[start0 + 4], bdfh0);
      Store<Align>(&data[start1 + 0], aceg1);
      Store<Align>(&data[start1 + 4], bdfh1);
      Store<Align>(&data[start2 + 0], aceg2);
      Store<Align>(&data[start2 + 4], bdfh2);
      Store<Align>(&data[start3 + 0], aceg3);
      Store<Align>(&data[start3 + 4], bdfh3);
    }
  }


  template <Alignment Align, std::floating_point F>
  void DoubleFlipRadix4PassRealInv(const F *tableData, F *data, size_t length, size_t subLength)
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
        auto c0 = Load<Align>(&data[start2 + i + 0 * stride]);
        auto d0 = Load<Align>(&data[start2 + i + 1 * stride]);
        auto e0 = Load<Align>(&data[start1 + i + 0 * stride]);
        auto f0 = Load<Align>(&data[start1 + i + 1 * stride]);
        auto g0 = Load<Align>(&data[start3 + i + 0 * stride]);
        auto h0 = Load<Align>(&data[start3 + i + 1 * stride]);

        DIFRealOutputRadix4PairInv(a0, b0, c0, d0, e0, f0, g0, h0, InvSqrt2<F>);

        Store<Align>(&data[start0 + i + 0 * stride], a0);
        Store<Align>(&data[start0 + i + 1 * stride], b0);
        auto a2 = Load<Align>(&data[start0 + i + 2 * stride]);
        auto b2 = Load<Align>(&data[start0 + i + 3 * stride]);
        Store<Align>(&data[start0 + i + 2 * stride], c0);
        Store<Align>(&data[start0 + i + 3 * stride], d0);
        auto a1 = Load<Align>(&data[start0 + i + 4 * stride]);
        auto b1 = Load<Align>(&data[start0 + i + 5 * stride]);
        Store<Align>(&data[start0 + i + 4 * stride], e0);
        Store<Align>(&data[start0 + i + 5 * stride], f0);
        auto a3 = Load<Align>(&data[start0 + i + 6 * stride]);
        auto b3 = Load<Align>(&data[start0 + i + 7 * stride]);
        Store<Align>(&data[start0 + i + 6 * stride], g0);
        Store<Align>(&data[start0 + i + 7 * stride], h0);


        auto c1 = Load<Align>(&data[start2 + i + 4 * stride]);
        auto d1 = Load<Align>(&data[start2 + i + 5 * stride]);
        auto e1 = Load<Align>(&data[start1 + i + 4 * stride]);
        auto f1 = Load<Align>(&data[start1 + i + 5 * stride]);
        auto g1 = Load<Align>(&data[start3 + i + 4 * stride]);
        auto h1 = Load<Align>(&data[start3 + i + 5 * stride]);

        DIFComplexRadix4Inv(a1, b1, c1, d1, e1, f1, g1, h1, sw0r_1, sw0i_1, sw1r_1, sw1i_1, sw2r_1, sw2i_1);

        Store<Align>(&data[start1 + i + 0 * stride], a1);
        Store<Align>(&data[start1 + i + 1 * stride], c1);
        Store<Align>(&data[start1 + i + 4 * stride], b1);
        Store<Align>(&data[start1 + i + 5 * stride], d1);

        auto e2 = Load<Align>(&data[start1 + i + 2 * stride]);
        auto f2 = Load<Align>(&data[start1 + i + 3 * stride]);
        Store<Align>(&data[start1 + i + 2 * stride], e1);
        Store<Align>(&data[start1 + i + 3 * stride], g1);
        auto e3 = Load<Align>(&data[start1 + i + 6 * stride]);
        auto f3 = Load<Align>(&data[start1 + i + 7 * stride]);
        Store<Align>(&data[start1 + i + 6 * stride], f1);
        Store<Align>(&data[start1 + i + 7 * stride], h1);


        auto c2 = Load<Align>(&data[start2 + i + 2 * stride]);
        auto d2 = Load<Align>(&data[start2 + i + 3 * stride]);
        auto g2 = Load<Align>(&data[start3 + i + 2 * stride]);
        auto h2 = Load<Align>(&data[start3 + i + 3 * stride]);

        DIFComplexRadix4Inv(a2, b2, c2, d2, e2, f2, g2, h2, sw0r_2, sw0i_2, sw1r_2, sw1i_2, sw2r_2, sw2i_2);

        Store<Align>(&data[start2 + i + 0 * stride], a2);
        Store<Align>(&data[start2 + i + 1 * stride], c2);
        Store<Align>(&data[start2 + i + 2 * stride], e2);
        Store<Align>(&data[start2 + i + 3 * stride], g2);
        Store<Align>(&data[start2 + i + 4 * stride], b2);
        Store<Align>(&data[start2 + i + 5 * stride], d2);
        auto c3 = Load<Align>(&data[start2 + i + 6 * stride]);
        auto d3 = Load<Align>(&data[start2 + i + 7 * stride]);
        Store<Align>(&data[start2 + i + 6 * stride], f2);
        Store<Align>(&data[start2 + i + 7 * stride], h2);

        auto g3 = Load<Align>(&data[start3 + i + 6 * stride]);
        auto h3 = Load<Align>(&data[start3 + i + 7 * stride]);

        DIFComplexRadix4Inv(a3, b3, c3, d3, e3, f3, g3, h3, sw0r_3, sw0i_3, sw1r_3, sw1i_3, sw2r_3, sw2i_3);

        // Transpose and store
        Store<Align>(&data[start3 + i + 0 * stride], a3);
        Store<Align>(&data[start3 + i + 1 * stride], c3);
        Store<Align>(&data[start3 + i + 2 * stride], e3);
        Store<Align>(&data[start3 + i + 3 * stride], g3);
        Store<Align>(&data[start3 + i + 4 * stride], b3);
        Store<Align>(&data[start3 + i + 5 * stride], d3);
        Store<Align>(&data[start3 + i + 6 * stride], f3);
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
          auto b0 = Load<Align>(&data[start0 + i + 1 * stride]);
          auto c0 = Load<Align>(&data[start2 + i + 0 * stride]);
          auto d0 = Load<Align>(&data[start2 + i + 1 * stride]);
          auto e0 = Load<Align>(&data[start1 + i + 0 * stride]);
          auto f0 = Load<Align>(&data[start1 + i + 1 * stride]);
          auto g0 = Load<Align>(&data[start3 + i + 0 * stride]);
          auto h0 = Load<Align>(&data[start3 + i + 1 * stride]);

          DIFComplexRadix4Inv(a0, b0, c0, d0, e0, f0, g0, h0, sw0r_0, sw0i_0, sw1r_0, sw1i_0, sw2r_0, sw2i_0);

          Store<Align>(&data[start0 + i + 0 * stride], a0);
          Store<Align>(&data[start0 + i + 1 * stride], c0);
          auto a2 = Load<Align>(&data[start0 + i + 2 * stride]);
          auto b2 = Load<Align>(&data[start0 + i + 3 * stride]);
          Store<Align>(&data[start0 + i + 2 * stride], e0);
          Store<Align>(&data[start0 + i + 3 * stride], g0);
          auto a1 = Load<Align>(&data[start0 + i + 4 * stride]);
          auto b1 = Load<Align>(&data[start0 + i + 5 * stride]);
          Store<Align>(&data[start0 + i + 4 * stride], b0);
          Store<Align>(&data[start0 + i + 5 * stride], d0);
          auto a3 = Load<Align>(&data[start0 + i + 6 * stride]);
          auto b3 = Load<Align>(&data[start0 + i + 7 * stride]);
          Store<Align>(&data[start0 + i + 6 * stride], f0);
          Store<Align>(&data[start0 + i + 7 * stride], h0);

          auto c1 = Load<Align>(&data[start2 + i + 4 * stride]);
          auto d1 = Load<Align>(&data[start2 + i + 5 * stride]);
          auto g1 = Load<Align>(&data[start3 + i + 4 * stride]);
          auto e1 = Load<Align>(&data[start1 + i + 4 * stride]);
          auto f1 = Load<Align>(&data[start1 + i + 5 * stride]);
          auto h1 = Load<Align>(&data[start3 + i + 5 * stride]);
          DIFComplexRadix4Inv(a1, b1, c1, d1, e1, f1, g1, h1, sw0r_1, sw0i_1, sw1r_1, sw1i_1, sw2r_1, sw2i_1);

          Store<Align>(&data[start1 + i + 0 * stride], a1);
          Store<Align>(&data[start1 + i + 1 * stride], c1);
          Store<Align>(&data[start1 + i + 4 * stride], b1);
          Store<Align>(&data[start1 + i + 5 * stride], d1);

          auto e2 = Load<Align>(&data[start1 + i + 2 * stride]);
          auto f2 = Load<Align>(&data[start1 + i + 3 * stride]);
          Store<Align>(&data[start1 + i + 2 * stride], e1);
          Store<Align>(&data[start1 + i + 3 * stride], g1);
          auto e3 = Load<Align>(&data[start1 + i + 6 * stride]);
          auto f3 = Load<Align>(&data[start1 + i + 7 * stride]);
          Store<Align>(&data[start1 + i + 6 * stride], f1);
          Store<Align>(&data[start1 + i + 7 * stride], h1);

          auto c2 = Load<Align>(&data[start2 + i + 2 * stride]);
          auto d2 = Load<Align>(&data[start2 + i + 3 * stride]);
          auto g2 = Load<Align>(&data[start3 + i + 2 * stride]);
          auto h2 = Load<Align>(&data[start3 + i + 3 * stride]);

          DIFComplexRadix4Inv(a2, b2, c2, d2, e2, f2, g2, h2, sw0r_2, sw0i_2, sw1r_2, sw1i_2, sw2r_2, sw2i_2);

          Store<Align>(&data[start2 + i + 0 * stride], a2);
          Store<Align>(&data[start2 + i + 1 * stride], c2);
          Store<Align>(&data[start2 + i + 2 * stride], e2);
          Store<Align>(&data[start2 + i + 3 * stride], g2);
          Store<Align>(&data[start2 + i + 4 * stride], b2);
          Store<Align>(&data[start2 + i + 5 * stride], d2);
          auto c3 = Load<Align>(&data[start2 + i + 6 * stride]);
          auto d3 = Load<Align>(&data[start2 + i + 7 * stride]);
          Store<Align>(&data[start2 + i + 6 * stride], f2);
          Store<Align>(&data[start2 + i + 7 * stride], h2);

          auto g3 = Load<Align>(&data[start3 + i + 6 * stride]);
          auto h3 = Load<Align>(&data[start3 + i + 7 * stride]);

          DIFComplexRadix4Inv(a3, b3, c3, d3, e3, f3, g3, h3, sw0r_3, sw0i_3, sw1r_3, sw1i_3, sw2r_3, sw2i_3);

          Store<Align>(&data[start3 + i + 0 * stride], a3);
          Store<Align>(&data[start3 + i + 1 * stride], c3);
          Store<Align>(&data[start3 + i + 2 * stride], e3);
          Store<Align>(&data[start3 + i + 3 * stride], g3);
          Store<Align>(&data[start3 + i + 4 * stride], b3);
          Store<Align>(&data[start3 + i + 5 * stride], d3);
          Store<Align>(&data[start3 + i + 6 * stride], f3);
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

  // For FFTs where an odd number of bits flip (i.e. there's a middle bit that doesn't flip), there needs to be a
  //  special pass in the middle that does a single bit flip instead of the usual 2-bit-flips-per-radix-4 approach.
  //  The behavior of this pass differs based on whether or not the length is a power of 4 or not (it changes the
  //  pattern of transposition in memory)
  template <Alignment Align, std::floating_point F>
  void SingleFlipRadix4PassRealInv(const F *tableData, F *data, size_t length, size_t subLength)
  {
    if (IsPowerOfFourAssumingIsPowerOfTwo(length))
    {
      // The bits that flip for power of four lengths touch two separate radix4s - however, they are conveniently
      //  contiguous in memory.
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
          auto a0 = Load<Align>(&data[i + 0 * stride]);
          auto b0 = Load<Align>(&data[i + 1 * stride]);
          auto e0 = Load<Align>(&data[i + 4 * stride]);
          auto f0 = Load<Align>(&data[i + 5 * stride]);
          auto c0 = Load<Align>(&data[i + subLength + 0 * stride]);
          auto d0 = Load<Align>(&data[i + subLength + 1 * stride]);
          auto g0 = Load<Align>(&data[i + subLength + 4 * stride]);
          auto h0 = Load<Align>(&data[i + subLength + 5 * stride]);

          DIFRealOutputRadix4PairInv(a0, b0, c0, d0, e0, f0, g0, h0, InvSqrt2<F>);

          Store<Align>(data + i + 0 * stride, a0);
          Store<Align>(data + i + 1 * stride, b0);
          Store<Align>(data + i + 4 * stride, e0);
          Store<Align>(data + i + 5 * stride, f0);

          auto aR1 = Load<Align>(&data[i + 2 * stride]);
          auto aI1 = Load<Align>(&data[i + 3 * stride]);
          auto cR1 = Load<Align>(&data[i + 6 * stride]);
          auto cI1 = Load<Align>(&data[i + 7 * stride]);
          Store<Align>(data + i + 2 * stride, c0);
          Store<Align>(data + i + 3 * stride, d0);
          Store<Align>(data + i + 6 * stride, g0);
          Store<Align>(data + i + 7 * stride, h0);

          auto bR1 = Load<Align>(&data[i + subLength + 2 * stride]);
          auto bI1 = Load<Align>(&data[i + subLength + 3 * stride]);
          auto dR1 = Load<Align>(&data[i + subLength + 6 * stride]);
          auto dI1 = Load<Align>(&data[i + subLength + 7 * stride]);

          DIFComplexRadix4Inv(aR1, aI1, bR1, bI1, cR1, cI1, dR1, dI1, w0r_1, w0i_1, w1r_1, w1i_1, w2r_1, w2i_1);

          Store<Align>(&data[i + subLength + 0 * stride], aR1);
          Store<Align>(&data[i + subLength + 1 * stride], bR1);
          Store<Align>(&data[i + subLength + 4 * stride], aI1);
          Store<Align>(&data[i + subLength + 5 * stride], bI1);
          Store<Align>(&data[i + subLength + 2 * stride], cR1);
          Store<Align>(&data[i + subLength + 3 * stride], dR1);
          Store<Align>(&data[i + subLength + 6 * stride], cI1);
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
          auto aR0 = Load<Align>(&data[i + 0 * stride]);
          auto aI0 = Load<Align>(&data[i + 1 * stride]);
          auto bR0 = Load<Align>(&data[i + subLength + 0 * stride]);
          auto bI0 = Load<Align>(&data[i + subLength + 1 * stride]);
          auto cR0 = Load<Align>(&data[i + 4 * stride]);
          auto cI0 = Load<Align>(&data[i + 5 * stride]);
          auto dR0 = Load<Align>(&data[i + subLength + 4 * stride]);
          auto dI0 = Load<Align>(&data[i + subLength + 5 * stride]);
          DIFComplexRadix4Inv(aR0, aI0, bR0, bI0, cR0, cI0, dR0, dI0, sw0r_0, sw0i_0, sw1r_0, sw1i_0, sw2r_0, sw2i_0);

          Store<Align>(data + i + 0 * stride, aR0);
          Store<Align>(data + i + 1 * stride, bR0);
          Store<Align>(data + i + 4 * stride, aI0);
          Store<Align>(data + i + 5 * stride, bI0);

          auto aR1 = Load<Align>(&data[i + 2 * stride]);
          auto aI1 = Load<Align>(&data[i + 3 * stride]);
          auto cR1 = Load<Align>(&data[i + 6 * stride]);
          auto cI1 = Load<Align>(&data[i + 7 * stride]);
          Store<Align>(data + i + 2 * stride, cR0);
          Store<Align>(data + i + 3 * stride, dR0);
          Store<Align>(data + i + 6 * stride, cI0);
          Store<Align>(data + i + 7 * stride, dI0);

          auto bR1 = Load<Align>(&data[i + subLength + 2 * stride]);
          auto bI1 = Load<Align>(&data[i + subLength + 3 * stride]);
          auto dR1 = Load<Align>(&data[i + subLength + 6 * stride]);
          auto dI1 = Load<Align>(&data[i + subLength + 7 * stride]);

          DIFComplexRadix4Inv(aR1, aI1, bR1, bI1, cR1, cI1, dR1, dI1, sw0r_1, sw0i_1, sw1r_1, sw1i_1, sw2r_1, sw2i_1);

          Store<Align>(&data[i + subLength + 0 * stride], aR1);
          Store<Align>(&data[i + subLength + 1 * stride], bR1);
          Store<Align>(&data[i + subLength + 4 * stride], aI1);
          Store<Align>(&data[i + subLength + 5 * stride], bI1);
          Store<Align>(&data[i + subLength + 2 * stride], cR1);
          Store<Align>(&data[i + subLength + 3 * stride], dR1);
          Store<Align>(&data[i + subLength + 6 * stride], cI1);
          Store<Align>(&data[i + subLength + 7 * stride], dI1);
        }
      }
    }
    else
    {
      // The bits that flip for non-power-of-four lengths are nice because they fit squarely within a single
      //  radix 4 butterfly (swapping the two inner elements), so this case is even simpler!
      const auto angleCount = 2 * length / subLength;

      // Do the reals (subset 0)
      const auto stride = subLength / 8;
      {
        const auto z = Set1(InvSqrt2<F>);

        for (auto i = 0u; i < subLength / 8; i += VecSize)
        {
          auto [a, b, e, f, c, d, g, h] = LoadStridedSets<Align, 8>(data + i, stride);
          DIFRealOutputRadix4PairInv(a, b, c, d, e, f, g, h, z);
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
          auto [aR, aI, cR, cI, bR, bI, dR, dI] = LoadStridedSets<Align, 8>(data + i, stride);
          DIFComplexRadix4Inv(aR, aI, bR, bI, cR, cI, dR, dI, w0r, w0i, w1r, w1i, w2r, w2i);
          StoreStridedSets<Align>(data + i, stride, aR, bR, cR, dR, aI, bI, cI, dI);
        }
      }
    }
  }


  template <Alignment Align, std::floating_point F>
  void NoFlipRadix4PassRealInv(const F *tableData, F *data, size_t length, size_t subLength)
  {
    const auto angleCount = 2 * length / subLength;

    const auto stride = subLength / 8;

    // Do the reals (subset 0)
    {
      const auto z = Set1(InvSqrt2<F>);

      for (auto i = 0u; i < stride; i += VecSize)
      {
        auto [a, b, c, d, e, f, g, h] = LoadStridedSets<Align, 8>(data + i, stride);
        DIFRealOutputRadix4PairInv(a, b, c, d, e, f, g, h, z);
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
        auto [aR, aI, bR, bI, cR, cI, dR, dI] = LoadStridedSets<Align, 8>(data + i, stride);
        DIFComplexRadix4Inv(aR, aI, bR, bI, cR, cI, dR, dI, w0r, w0i, w1r, w1i, w2r, w2i);
        StoreStridedSets<Align>(data + i, stride, aR, bR, cR, dR, aI, bI, cI, dI);
      }
    }
  }


  template <Alignment Align, std::floating_point F>
  void Radix2InvPass(F *data, size_t length)
  {
    // If the real count is a power of 4, then we have an odd radix2 pass.

    const auto stride = length / 16;

    // Do the reals (subset 0)
    {
      for (auto i = 0u; i < stride; i += VecSize)
      {
        auto a = Load<Align>(&data[i + 0 * stride]);
        auto b = Load<Align>(&data[i + 1 * stride]);
        auto c = Load<Align>(&data[i + 2 * stride]);
        auto d = Load<Align>(&data[i + 3 * stride]);

        DITRealOutputRadix2PairInv(a, b, c, d);

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
        auto aI = Load<Align>(&data[start + i + 1 * stride]);
        auto bR = Load<Align>(&data[start + i + 2 * stride]);
        auto bI = Load<Align>(&data[start + i + 3 * stride]);

        DIFComplexRadix2Inv(aR, aI, bR, bI, wR, wI);

        Store<Align>(&data[start + i + 0 * stride], aR);
        Store<Align>(&data[start + i + 1 * stride], bR);
        Store<Align>(&data[start + i + 2 * stride], aI);
        Store<Align>(&data[start + i + 3 * stride], bI);
      }
    }
  }

  // The very first pass does either an effective radix 4 or 8 pass, depending on whether the real FFT length is a
  //  power of 4 or not.
  template <Alignment Align, std::floating_point F>
  void Radix4FinalStepRealInv(F *data, size_t length)
  {
    // Divide by length/2 to get into the correct space
    const auto scale = Set1(F(2.0) / F(length));
    const auto z = Set1(InvSqrt2<F>);
    const auto stride = length / 8;
    for (auto i = 0u; i < stride; i += VecSize)
    {
      auto [a, b, c, d, e, f, g, h] = LoadStridedSets<Align, 8>(data + i, stride);
      DIFRealOutputRadix4PairInv(a, b, c, d, e, f, g, h, z);

      // !!! Maybe this can be baked into some other step for efficiency?
      a = Mul(a, scale);
      b = Mul(b, scale);
      c = Mul(c, scale);
      d = Mul(d, scale);
      e = Mul(e, scale);
      f = Mul(f, scale);
      g = Mul(g, scale);
      h = Mul(h, scale);
      StoreStridedSets<Align>(data + i, stride, a, b, c, d, e, f, g, h);
    }
  }


  template <Alignment Align, std::floating_point F>
  void RealInverseFFT(const F *tableData, const F *input, F *output, size_t length)
  {
    if (length < MinimumRealFFTLength)
    {
      TinyRealInv(input, output, length);
      return;
    }

    const auto bitCount = std::numeric_limits<size_t>::digits - std::countl_zero(length) - 1;
    const auto bitReverseCount = (bitCount - 1) >> 1;

    TangleRealFFT<Align>(input, output, length);

    const auto maxSubLength = length >> (IsPowerOfFourAssumingIsPowerOfTwo(length) ? 3 : 2);
    const auto maxDoubleFlipSubLength = 4_z << (bitReverseCount - ((bitReverseCount & 1) ? 2 : 0));

    FirstDoubleFlipRadix4PassRealInv<Align>(tableData, output, length);

    auto subLength = 32_z;
    for (; subLength <= maxDoubleFlipSubLength; subLength <<= 2)
    {
      DoubleFlipRadix4PassRealInv<Align>(tableData, output, length, subLength);
    }

    if ((bitReverseCount & 1) != 0)
    {
      SingleFlipRadix4PassRealInv<Align>(tableData, output, length, subLength);
      subLength <<= 2;
    }

    while (subLength <= maxSubLength)
    {
      NoFlipRadix4PassRealInv<Align>(tableData, output, length, subLength);
      subLength <<= 2;
    }

    if (IsPowerOfFourAssumingIsPowerOfTwo(length))
    {
      Radix2InvPass<Align>(output, length);
    }

    Radix4FinalStepRealInv<Align>(output, length);
  }