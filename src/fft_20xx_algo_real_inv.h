#include "fft_20xx_algo_real_common.h"

  template <Alignment Align, uint32_t VV = VecSize, typename VT = VecType>
  void FirstDoubleFlipRadix4PassRealInv(const ElementType *tableData, ElementType *data, size_t length)
  {
    // Start with the scalar step
    {
      auto a = data[0];
      auto b = data[1];
      data[0] = (a + b) / 2;
      data[1] = (a - b) / 2;
    }

    if constexpr (VV == 2)
    {
      const auto flipStride = length / 4;

      {
        auto start1 = 1 * flipStride;
        auto start2 = 2 * flipStride;
        auto start3 = 3 * flipStride;

        VT r00, r01, r02, r03, r04, r05, r06, r07, r08, r09, r10, r11, r12, r13, r14, r15;
        LoadStridedSets<Align>(data, 2, r00, r01);
        LoadStridedSets<Align>(data + start1, 2, r02, r03);
        LoadStridedSets<Align>(data + start2, 2, r04, r05);
        LoadStridedSets<Align>(data + start3, 2, r06, r07);

        // Deinterleave the real/imaginary components
        Transpose2(r00, r01, r08);
        Transpose2(r04, r05, r00);
        Transpose2(r02, r03, r04);
        Transpose2(r06, r07, r02);

        LoadSIMDAngleDouble(tableData, 0, flipStride, r06, r09, r10, r11);
        FirstRealDITRealComplexRadix4Inv(r08, r01, r00, r05, r04, r03, r02, r07, r10, r11, r06, r09, r12, r13);

        // Now do 2 quadrants of a 4x4 matrix transpose
        Transpose2(r08, r00, r06);
        Transpose2(r04, r02, r08);
        Transpose2(r01, r05, r04);
        Transpose2(r03, r07, r01);

        // Store the two quadrants that are in spots we already loaded from
        //  (note that there's a bit of extra shuffle here for the real values vs. the fully-complex subsets below)
        StoreStridedSets<Align>(data, 2, r06, r04);
        StoreStridedSets<Align>(data + start2, 2, r00, r02);

        // Now swap the two other quadrants for values in the new memory area
        LoadStridedSets<Align>(data + 4, 2, r00, r02);
        LoadStridedSets<Align>(data + start2 + 4, 2, r04, r06);
        StoreStridedSets<Align>(data + 4, 2, r08, r01);
        StoreStridedSets<Align>(data + start2 + 4, 2, r05, r07);

        // Load the last 2 new quadrants
        LoadStridedSets<Align>(data + start1 + 4, 2, r01, r03);
        LoadStridedSets<Align>(data + start3 + 4, 2, r05, r07);

        // Once again deinterleave the real/imaginary components
        Transpose2(r00, r02, r08);
        Transpose2(r04, r06, r00);
        Transpose2(r01, r03, r04);
        Transpose2(r05, r07, r01);

        LoadSIMDAngleDouble(tableData, 2, flipStride, r05, r09, r10, r11);
        ComplexMultiply(r05, r09, r10, r11, r12, r13, r14, r15);
        DIFComplexRadix4Inv(r08, r02, r00, r06, r04, r03, r01, r07, r05, r09, r10, r11, r12, r13, r14, r15);

        // The remainder of our 4x4 transpose
        Transpose2(r08, r00, r05);
        Transpose2(r04, r01, r08);
        Transpose2(r02, r06, r04);
        Transpose2(r03, r07, r02);
        StoreStridedSets<Align>(data + start1, 2, r05, r08);
        StoreStridedSets<Align>(data + start3, 2, r00, r01);
        StoreStridedSets<Align>(data + start1 + 4, 2, r04, r02);
        StoreStridedSets<Align>(data + start3 + 4, 2, r06, r07);
      }

      for (auto subsetIndex = 8_z; subsetIndex < flipStride; subsetIndex += 8)
      {
        const auto start0 = subsetIndex;
        auto start1 = start0 +  1 * flipStride;
        auto start2 = start0 +  2 * flipStride;
        auto start3 = start0 +  3 * flipStride;
        const auto angleIndex0 = (subsetIndex >> 1);

        VT r00, r01, r02, r03, r04, r05, r06, r07, r08, r09, r10, r11, r12, r13, r14, r15;
        LoadStridedSets<Align>(data + start0, 2, r00, r01);
        LoadStridedSets<Align>(data + start1, 2, r02, r03);
        LoadStridedSets<Align>(data + start2, 2, r04, r05);
        LoadStridedSets<Align>(data + start3, 2, r06, r07);

        Transpose2(r00, r01, r08);
        Transpose2(r04, r05, r00);
        Transpose2(r02, r03, r04);
        Transpose2(r06, r07, r02);

        LoadSIMDAngleDouble(tableData, angleIndex0, flipStride, r09, r10, r11, r12);
        ComplexMultiply(r09, r10, r11, r12, r13, r14, r15, r06);
        DIFComplexRadix4Inv(r08, r01, r00, r05, r04, r03, r02, r07, r09, r10, r11, r12, r13, r14, r15, r06);

        Transpose2(r08, r00, r06);
        Transpose2(r04, r02, r08);
        Transpose2(r01, r05, r04);
        Transpose2(r03, r07, r01);

        StoreStridedSets<Align>(data + start0, 2, r06, r08);
        StoreStridedSets<Align>(data + start2, 2, r00, r02);

        LoadStridedSets<Align>(data + start0 + 4, 2, r00, r02);
        LoadStridedSets<Align>(data + start2 + 4, 2, r06, r08);
        StoreStridedSets<Align>(data + start0 + 4, 2, r04, r01);
        StoreStridedSets<Align>(data + start2 + 4, 2, r05, r07);
        LoadStridedSets<Align>(data + start1 + 4, 2, r01, r03);
        LoadStridedSets<Align>(data + start3 + 4, 2, r04, r05);

        Transpose2(r00, r02, r07);
        Transpose2(r06, r08, r00);
        Transpose2(r01, r03, r06);
        Transpose2(r04, r05, r01);

        LoadSIMDAngleDouble(tableData, angleIndex0 + 2, flipStride, r04, r09, r10, r11);
        ComplexMultiply(r04, r09, r10, r11, r12, r13, r14, r15);
        DIFComplexRadix4Inv(r07, r02, r00, r08, r06, r03, r01, r05, r04, r09, r10, r11, r12, r13, r14, r15);

        Transpose2(r07, r00, r04);
        Transpose2(r06, r01, r07);
        Transpose2(r02, r08, r06);
        Transpose2(r03, r05, r02);

        StoreStridedSets<Align>(data + start1, 2, r04, r07);
        StoreStridedSets<Align>(data + start3, 2, r00, r01);
        StoreStridedSets<Align>(data + start1 + 4, 2, r06, r02);
        StoreStridedSets<Align>(data + start3 + 4, 2, r08, r05);
      }
    }
    else
    {
      static_assert(VV == 4);
      const auto flipStride = length / 4;

      {
        const auto start1 = 1_z * flipStride;
        const auto start2 = 2_z * flipStride;
        const auto start3 = 3_z * flipStride;

        VT r00, r01, r02, r03, r04, r05, r06, r07, r08, r09, r10, r11, r12, r13;
        r00 = Load<Align>(&data[0]);
        r01 = Load<Align>(&data[4]);
        r02 = Load<Align>(&data[start1 + 0]);
        r03 = Load<Align>(&data[start1 + 4]);
        r04 = Load<Align>(&data[start2 + 0]);
        r05 = Load<Align>(&data[start2 + 4]);
        r06 = Load<Align>(&data[start3 + 0]);
        r07 = Load<Align>(&data[start3 + 4]);

        // Re-interleave the pairs (a/b, c/d, e/f, g/h)
        Deinterleave4(r00, r01, r08);
        Deinterleave4(r04, r05, r00);
        Deinterleave4(r02, r03, r04);
        Deinterleave4(r06, r07, r02);

        // reuse r06
        LoadSIMDAngleDouble(tableData, 0, flipStride, r06, r09, r10, r11);

        FirstRealDITRealComplexRadix4Inv(r08, r01, r00, r05, r04, r03, r02, r07, r10, r11, r06, r09, r12, r13);

        // Transpose these - note that 1 and 2 are out of order - this makes the deinterleave at the end simpler.
        Transpose4(r08, r00, r04, r02, r06, r12);
        Transpose4(r01, r05, r03, r07, r06, r12);

        // swap {e0, g0} and {b0, d0} because the real butterfly needs them in this order instead.
        SwapRealComponentsForLastDoubleFlip(r08, r01, r06);

        Store<Align>(&data[0], r06);
        Store<Align>(&data[4], r01);
        Store<Align>(&data[start1 + 0], r04);
        Store<Align>(&data[start1 + 4], r03);
        Store<Align>(&data[start2 + 0], r00);
        Store<Align>(&data[start2 + 4], r05);
        Store<Align>(&data[start3 + 0], r02);
        Store<Align>(&data[start3 + 4], r07);
      }

      for (auto subsetIndex = 8_z; subsetIndex < flipStride; subsetIndex += 8)
      {
        const auto start0 = subsetIndex;
        auto start1 = start0 + 1 * flipStride;
        auto start2 = start0 + 2 * flipStride;
        auto start3 = start0 + 3 * flipStride;
        const auto angleIndex0 = subsetIndex >> 1;

        // "Give me ALL the registers that you have"
        VT r00, r01, r02, r03, r04, r05, r06, r07, r08, r09, r10, r11, r12, r13, r14, r15;
        r00 = Load<Align>(&data[start0 + 0]);
        r01 = Load<Align>(&data[start0 + 4]);
        r02 = Load<Align>(&data[start1 + 0]);
        r03 = Load<Align>(&data[start1 + 4]);
        r04 = Load<Align>(&data[start2 + 0]);
        r05 = Load<Align>(&data[start2 + 4]);
        r06 = Load<Align>(&data[start3 + 0]);
        r07 = Load<Align>(&data[start3 + 4]);


        Deinterleave4(r00, r01, r08);
        Deinterleave4(r04, r05, r00);
        Deinterleave4(r02, r03, r04);
        Deinterleave4(r06, r07, r02);

        LoadSIMDAngleDouble(tableData, angleIndex0, flipStride, r09, r10, r11, r12);
        ComplexMultiply(r09, r10, r11, r12, r06, r13, r14, r15);
        DIFComplexRadix4Inv(r08, r01, r00, r05, r04, r03, r02, r07, r09, r10, r11, r12, r06, r13, r14, r15);

        Transpose4(r08, r00, r04, r02, r14, r15);
        Transpose4(r01, r05, r03, r07, r14, r15);

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
  }


  template <Alignment Align>
  void DoubleFlipRadix4PassRealInv(const ElementType *tableData, ElementType *data, size_t length, size_t subLength)
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
        VecType r00, r01, r02, r03, r04, r05, r06, r07, r08, r09, r10, r11, r12, r13, r14;
        r00 = Load<Align>(&data[start0 + i + 0 * stride]);
        r01 = Load<Align>(&data[start0 + i + 1 * stride]);
        r02 = Load<Align>(&data[start2 + i + 0 * stride]);
        r03 = Load<Align>(&data[start2 + i + 1 * stride]);
        r04 = Load<Align>(&data[start1 + i + 0 * stride]);
        r05 = Load<Align>(&data[start1 + i + 1 * stride]);
        r06 = Load<Align>(&data[start3 + i + 0 * stride]);
        r07 = Load<Align>(&data[start3 + i + 1 * stride]);

        r12 = Set1(InvSqrt2<ElementType>);
        DIFRealOutputRadix4PairInv(r00, r01, r02, r03, r04, r05, r06, r07, r12, r08);

        Store<Align>(&data[start0 + i + 0 * stride], r00);
        Store<Align>(&data[start0 + i + 1 * stride], r01);
        r00 = Load<Align>(&data[start0 + i + 2 * stride]);
        r01 = Load<Align>(&data[start0 + i + 3 * stride]);
        Store<Align>(&data[start0 + i + 2 * stride], r02);
        Store<Align>(&data[start0 + i + 3 * stride], r03);
        r02 = Load<Align>(&data[start0 + i + 4 * stride]);
        r03 = Load<Align>(&data[start0 + i + 5 * stride]);
        Store<Align>(&data[start0 + i + 4 * stride], r04);
        Store<Align>(&data[start0 + i + 5 * stride], r05);
        r04 = Load<Align>(&data[start0 + i + 6 * stride]);
        r05 = Load<Align>(&data[start0 + i + 7 * stride]);
        Store<Align>(&data[start0 + i + 6 * stride], r06);
        Store<Align>(&data[start0 + i + 7 * stride], r07);


        r06 = Load<Align>(&data[start2 + i + 4 * stride]);
        r07 = Load<Align>(&data[start2 + i + 5 * stride]);
        r08 = Load<Align>(&data[start1 + i + 4 * stride]);
        r09 = Load<Align>(&data[start1 + i + 5 * stride]);
        r10 = Load<Align>(&data[start3 + i + 4 * stride]);
        r11 = Load<Align>(&data[start3 + i + 5 * stride]);

        DIFComplexRadix4Inv(r02, r03, r06, r07, r08, r09, r10, r11, sw0r_1, sw0i_1, sw1r_1, sw1i_1, sw2r_1, sw2i_1, r12, r13, r14);

        Store<Align>(&data[start1 + i + 0 * stride], r02);
        Store<Align>(&data[start1 + i + 1 * stride], r06);
        Store<Align>(&data[start1 + i + 4 * stride], r03);
        Store<Align>(&data[start1 + i + 5 * stride], r07);

        r02 = Load<Align>(&data[start1 + i + 2 * stride]);
        r03 = Load<Align>(&data[start1 + i + 3 * stride]);
        Store<Align>(&data[start1 + i + 2 * stride], r08);
        Store<Align>(&data[start1 + i + 3 * stride], r10);
        r06 = Load<Align>(&data[start1 + i + 6 * stride]);
        r07 = Load<Align>(&data[start1 + i + 7 * stride]);
        Store<Align>(&data[start1 + i + 6 * stride], r09);
        Store<Align>(&data[start1 + i + 7 * stride], r11);


        r08 = Load<Align>(&data[start2 + i + 2 * stride]);
        r09 = Load<Align>(&data[start2 + i + 3 * stride]);
        r10 = Load<Align>(&data[start3 + i + 2 * stride]);
        r11 = Load<Align>(&data[start3 + i + 3 * stride]);

        DIFComplexRadix4Inv(r00, r01, r08, r09, r02, r03, r10, r11, sw0r_2, sw0i_2, sw1r_2, sw1i_2, sw2r_2, sw2i_2, r12, r13, r14);

        Store<Align>(&data[start2 + i + 0 * stride], r00);
        Store<Align>(&data[start2 + i + 1 * stride], r08);
        Store<Align>(&data[start2 + i + 2 * stride], r02);
        Store<Align>(&data[start2 + i + 3 * stride], r10);
        Store<Align>(&data[start2 + i + 4 * stride], r01);
        Store<Align>(&data[start2 + i + 5 * stride], r09);
        r00 = Load<Align>(&data[start2 + i + 6 * stride]);
        r01 = Load<Align>(&data[start2 + i + 7 * stride]);
        Store<Align>(&data[start2 + i + 6 * stride], r03);
        Store<Align>(&data[start2 + i + 7 * stride], r11);

        r02 = Load<Align>(&data[start3 + i + 6 * stride]);
        r03 = Load<Align>(&data[start3 + i + 7 * stride]);

        DIFComplexRadix4Inv(r04, r05, r00, r01, r06, r07, r02, r03, sw0r_3, sw0i_3, sw1r_3, sw1i_3, sw2r_3, sw2i_3, r12, r13, r14);

        // Transpose and store
        Store<Align>(&data[start3 + i + 0 * stride], r04);
        Store<Align>(&data[start3 + i + 1 * stride], r00);
        Store<Align>(&data[start3 + i + 2 * stride], r06);
        Store<Align>(&data[start3 + i + 3 * stride], r02);
        Store<Align>(&data[start3 + i + 4 * stride], r05);
        Store<Align>(&data[start3 + i + 5 * stride], r01);
        Store<Align>(&data[start3 + i + 6 * stride], r07);
        Store<Align>(&data[start3 + i + 7 * stride], r03);

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
          VecType r00, r01, r02, r03, r04, r05, r06, r07, r08, r09, r10, r11, r12, r13, r14;
          r00 = Load<Align>(&data[start0 + i + 0 * stride]);
          r01 = Load<Align>(&data[start0 + i + 1 * stride]);
          r02 = Load<Align>(&data[start2 + i + 0 * stride]);
          r03 = Load<Align>(&data[start2 + i + 1 * stride]);
          r04 = Load<Align>(&data[start1 + i + 0 * stride]);
          r05 = Load<Align>(&data[start1 + i + 1 * stride]);
          r06 = Load<Align>(&data[start3 + i + 0 * stride]);
          r07 = Load<Align>(&data[start3 + i + 1 * stride]);

          DIFComplexRadix4Inv(r00, r01, r02, r03, r04, r05, r06, r07, sw0r_0, sw0i_0, sw1r_0, sw1i_0, sw2r_0, sw2i_0, r12, r13, r14);

          Store<Align>(&data[start0 + i + 0 * stride], r00);
          Store<Align>(&data[start0 + i + 1 * stride], r02);
          r00 = Load<Align>(&data[start0 + i + 2 * stride]);
          r02 = Load<Align>(&data[start0 + i + 3 * stride]);
          Store<Align>(&data[start0 + i + 2 * stride], r04);
          Store<Align>(&data[start0 + i + 3 * stride], r06);
          r04 = Load<Align>(&data[start0 + i + 4 * stride]);
          r06 = Load<Align>(&data[start0 + i + 5 * stride]);
          Store<Align>(&data[start0 + i + 4 * stride], r01);
          Store<Align>(&data[start0 + i + 5 * stride], r03);
          r01 = Load<Align>(&data[start0 + i + 6 * stride]);
          r03 = Load<Align>(&data[start0 + i + 7 * stride]);
          Store<Align>(&data[start0 + i + 6 * stride], r05);
          Store<Align>(&data[start0 + i + 7 * stride], r07);

          r05 = Load<Align>(&data[start2 + i + 4 * stride]);
          r07 = Load<Align>(&data[start2 + i + 5 * stride]);
          r08 = Load<Align>(&data[start3 + i + 4 * stride]);
          r09 = Load<Align>(&data[start1 + i + 4 * stride]);
          r10 = Load<Align>(&data[start1 + i + 5 * stride]);
          r11 = Load<Align>(&data[start3 + i + 5 * stride]);
          DIFComplexRadix4Inv(r04, r06, r05, r07, r09, r10, r08, r11, sw0r_1, sw0i_1, sw1r_1, sw1i_1, sw2r_1, sw2i_1, r12, r13, r14);

          Store<Align>(&data[start1 + i + 0 * stride], r04);
          Store<Align>(&data[start1 + i + 1 * stride], r05);
          Store<Align>(&data[start1 + i + 4 * stride], r06);
          Store<Align>(&data[start1 + i + 5 * stride], r07);

          r04 = Load<Align>(&data[start1 + i + 2 * stride]);
          r05 = Load<Align>(&data[start1 + i + 3 * stride]);
          Store<Align>(&data[start1 + i + 2 * stride], r09);
          Store<Align>(&data[start1 + i + 3 * stride], r08);
          r06 = Load<Align>(&data[start1 + i + 6 * stride]);
          r07 = Load<Align>(&data[start1 + i + 7 * stride]);
          Store<Align>(&data[start1 + i + 6 * stride], r10);
          Store<Align>(&data[start1 + i + 7 * stride], r11);

          r08 = Load<Align>(&data[start2 + i + 2 * stride]);
          r09 = Load<Align>(&data[start2 + i + 3 * stride]);
          r10 = Load<Align>(&data[start3 + i + 2 * stride]);
          r11 = Load<Align>(&data[start3 + i + 3 * stride]);

          DIFComplexRadix4Inv(r00, r02, r08, r09, r04, r05, r10, r11, sw0r_2, sw0i_2, sw1r_2, sw1i_2, sw2r_2, sw2i_2, r12, r13, r14);

          Store<Align>(&data[start2 + i + 0 * stride], r00);
          Store<Align>(&data[start2 + i + 1 * stride], r08);
          Store<Align>(&data[start2 + i + 2 * stride], r04);
          Store<Align>(&data[start2 + i + 3 * stride], r10);
          Store<Align>(&data[start2 + i + 4 * stride], r02);
          Store<Align>(&data[start2 + i + 5 * stride], r09);
          r00 = Load<Align>(&data[start2 + i + 6 * stride]);
          r02 = Load<Align>(&data[start2 + i + 7 * stride]);
          Store<Align>(&data[start2 + i + 6 * stride], r05);
          Store<Align>(&data[start2 + i + 7 * stride], r11);

          r04 = Load<Align>(&data[start3 + i + 6 * stride]);
          r05 = Load<Align>(&data[start3 + i + 7 * stride]);

          DIFComplexRadix4Inv(r01, r03, r00, r02, r06, r07, r04, r05, sw0r_3, sw0i_3, sw1r_3, sw1i_3, sw2r_3, sw2i_3, r12, r13, r14);

          Store<Align>(&data[start3 + i + 0 * stride], r01);
          Store<Align>(&data[start3 + i + 1 * stride], r00);
          Store<Align>(&data[start3 + i + 2 * stride], r06);
          Store<Align>(&data[start3 + i + 3 * stride], r04);
          Store<Align>(&data[start3 + i + 4 * stride], r03);
          Store<Align>(&data[start3 + i + 5 * stride], r02);
          Store<Align>(&data[start3 + i + 6 * stride], r07);
          Store<Align>(&data[start3 + i + 7 * stride], r05);

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
  template <Alignment Align>
  void SingleFlipRadix4PassRealInv(const ElementType *tableData, ElementType *data, size_t length, size_t subLength)
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
        const auto r00 = Set1(sw0r_1);
        const auto r01 = Set1(sw0i_1);
        const auto r02 = Set1(sw1r_1);
        const auto r03 = Set1(sw1i_1);
        const auto r04 = Set1(sw2r_1);
        const auto r05 = Set1(sw2i_1);

        for (auto i = 0u; i < subLength / 8; i += VecSize)
        {
          VecType r06, r07, r08, r09, r10, r11, r12, r13, r14, r15;
          r06 = Load<Align>(&data[i + 0 * stride]);
          r07 = Load<Align>(&data[i + 1 * stride]);
          r08 = Load<Align>(&data[i + 4 * stride]);
          r09 = Load<Align>(&data[i + 5 * stride]);
          r10 = Load<Align>(&data[i + subLength + 0 * stride]);
          r11 = Load<Align>(&data[i + subLength + 1 * stride]);
          r12 = Load<Align>(&data[i + subLength + 4 * stride]);
          r13 = Load<Align>(&data[i + subLength + 5 * stride]);

          // Unfortunately we're one register shy on x64 of being able to keep this resident like the other angles.
          r14 = Set1(InvSqrt2<ElementType>);
          DIFRealOutputRadix4PairInv(r06, r07, r10, r11, r08, r09, r12, r13, r14, r15);

          Store<Align>(data + i + 0 * stride, r06);
          Store<Align>(data + i + 1 * stride, r07);
          Store<Align>(data + i + 4 * stride, r08);
          Store<Align>(data + i + 5 * stride, r09);

          r06 = Load<Align>(&data[i + 2 * stride]);
          r07 = Load<Align>(&data[i + 3 * stride]);
          r08 = Load<Align>(&data[i + 6 * stride]);
          r09 = Load<Align>(&data[i + 7 * stride]);
          Store<Align>(data + i + 2 * stride, r10);
          Store<Align>(data + i + 3 * stride, r11);
          Store<Align>(data + i + 6 * stride, r12);
          Store<Align>(data + i + 7 * stride, r13);

          r10 = Load<Align>(&data[i + subLength + 2 * stride]);
          r11 = Load<Align>(&data[i + subLength + 3 * stride]);
          r12 = Load<Align>(&data[i + subLength + 6 * stride]);
          r13 = Load<Align>(&data[i + subLength + 7 * stride]);

          DIFComplexRadix4Inv(r06, r07, r10, r11, r08, r09, r12, r13, r00, r01, r02, r03, r04, r05, r14, r15);

          Store<Align>(&data[i + subLength + 0 * stride], r06);
          Store<Align>(&data[i + subLength + 1 * stride], r10);
          Store<Align>(&data[i + subLength + 4 * stride], r07);
          Store<Align>(&data[i + subLength + 5 * stride], r11);
          Store<Align>(&data[i + subLength + 2 * stride], r08);
          Store<Align>(&data[i + subLength + 3 * stride], r12);
          Store<Align>(&data[i + subLength + 6 * stride], r09);
          Store<Align>(&data[i + subLength + 7 * stride], r13);
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

        // We have enough registers to pull exactly one value out of the loop
        const VecType r08 = Set1(sw0r_0);
        for (auto i = subsetStartIndex; i < subsetStartIndex + stride; i += VecSize)
        {
          VecType r00, r01, r02, r03, r04, r05, r06, r07, r09, r10, r11, r12, r13, r14, r15;
          r00 = Load<Align>(&data[i + 0 * stride]);
          r01 = Load<Align>(&data[i + 1 * stride]);
          r02 = Load<Align>(&data[i + subLength + 0 * stride]);
          r03 = Load<Align>(&data[i + subLength + 1 * stride]);
          r04 = Load<Align>(&data[i + 4 * stride]);
          r05 = Load<Align>(&data[i + 5 * stride]);
          r06 = Load<Align>(&data[i + subLength + 4 * stride]);
          r07 = Load<Align>(&data[i + subLength + 5 * stride]);
          r09 = Set1(sw0i_0);
          r10 = Set1(sw1r_0);
          r11 = Set1(sw1i_0);
          r12 = Set1(sw2r_0);
          r13 = Set1(sw2i_0);
          DIFComplexRadix4Inv(r00, r01, r02, r03, r04, r05, r06, r07, r08, r09, r10, r11, r12, r13, r14, r15);

          Store<Align>(data + i + 0 * stride, r00);
          Store<Align>(data + i + 1 * stride, r02);
          Store<Align>(data + i + 4 * stride, r01);
          Store<Align>(data + i + 5 * stride, r03);

          r00 = Load<Align>(&data[i + 2 * stride]);
          r01 = Load<Align>(&data[i + 3 * stride]);
          r02 = Load<Align>(&data[i + 6 * stride]);
          r03 = Load<Align>(&data[i + 7 * stride]);
          Store<Align>(data + i + 2 * stride, r04);
          Store<Align>(data + i + 3 * stride, r06);
          Store<Align>(data + i + 6 * stride, r05);
          Store<Align>(data + i + 7 * stride, r07);

          r04 = Load<Align>(&data[i + subLength + 2 * stride]);
          r05 = Load<Align>(&data[i + subLength + 3 * stride]);
          r06 = Load<Align>(&data[i + subLength + 6 * stride]);
          r07 = Load<Align>(&data[i + subLength + 7 * stride]);

          DIFComplexRadix4Inv(r00, r01, r04, r05, r02, r03, r06, r07, sw0r_1, sw0i_1, sw1r_1, sw1i_1, sw2r_1, sw2i_1, r09, r10, r11);

          Store<Align>(&data[i + subLength + 0 * stride], r00);
          Store<Align>(&data[i + subLength + 1 * stride], r04);
          Store<Align>(&data[i + subLength + 4 * stride], r01);
          Store<Align>(&data[i + subLength + 5 * stride], r05);
          Store<Align>(&data[i + subLength + 2 * stride], r02);
          Store<Align>(&data[i + subLength + 3 * stride], r06);
          Store<Align>(&data[i + subLength + 6 * stride], r03);
          Store<Align>(&data[i + subLength + 7 * stride], r07);
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
        const auto z = Set1(InvSqrt2<ElementType>);

        for (auto i = 0u; i < subLength / 8; i += VecSize)
        {
          VecType a, b, e, f, c, d, g, h, t1;
          LoadStridedSets<Align>(data + i, stride, a, b, e, f, c, d, g, h);
          DIFRealOutputRadix4PairInv(a, b, c, d, e, f, g, h, z, t1);
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
          VecType aR, aI, cR, cI, bR, bI, dR, dI, t1, t2;
          LoadStridedSets<Align>(data + i, stride, aR, aI, cR, cI, bR, bI, dR, dI);
          DIFComplexRadix4Inv(aR, aI, bR, bI, cR, cI, dR, dI, w0r, w0i, w1r, w1i, w2r, w2i, t1, t2);
          StoreStridedSets<Align>(data + i, stride, aR, bR, cR, dR, aI, bI, cI, dI);
        }
      }
    }
  }


  template <Alignment Align>
  void NoFlipRadix4PassRealInv(const ElementType *tableData, ElementType *data, size_t length, size_t subLength)
  {
    const auto angleCount = 2 * length / subLength;

    const auto stride = subLength / 8;

    // Do the reals (subset 0)
    {
      const auto z = Set1(InvSqrt2<ElementType>);

      for (auto i = 0u; i < stride; i += VecSize)
      {
        VecType a, b, c, d, e, f, g, h, t1;
        LoadStridedSets<Align>(data + i, stride, a, b, c, d, e, f, g, h);
        DIFRealOutputRadix4PairInv(a, b, c, d, e, f, g, h, z, t1);
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
        VecType aR, aI, bR, bI, cR, cI, dR, dI, t1, t2;
        LoadStridedSets<Align>(data + i, stride, aR, aI, bR, bI, cR, cI, dR, dI);
        DIFComplexRadix4Inv(aR, aI, bR, bI, cR, cI, dR, dI, w0r, w0i, w1r, w1i, w2r, w2i, t1, t2);
        StoreStridedSets<Align>(data + i, stride, aR, bR, cR, dR, aI, bI, cI, dI);
      }
    }
  }


  template <Alignment Align>
  void Radix2InvPass(ElementType *data, size_t length)
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

        VecType a2, b2;
        DITRealOutputRadix2PairInv(a, b, c, d, a2, b2);

        Store<Align>(&data[i + 0 * stride], a2);
        Store<Align>(&data[i + 1 * stride], b2);
        Store<Align>(&data[i + 2 * stride], c);
        Store<Align>(&data[i + 3 * stride], d);
      }
    }

    // Now do the complex values
    const ElementType swRs[3] = { InvSqrt2<ElementType>, CosPiOver8<ElementType>, SinPiOver8<ElementType> };
    const ElementType swIs[3] = { -InvSqrt2<ElementType>, -SinPiOver8<ElementType>, -CosPiOver8<ElementType> };

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

        VecType aR2, aI2;
        DIFComplexRadix2Inv(aR, aI, bR, bI, wR, wI, aR2, aI2);

        Store<Align>(&data[start + i + 0 * stride], aR2);
        Store<Align>(&data[start + i + 1 * stride], bR);
        Store<Align>(&data[start + i + 2 * stride], aI2);
        Store<Align>(&data[start + i + 3 * stride], bI);
      }
    }
  }

  // The very first pass does either an effective radix 4 or 8 pass, depending on whether the real FFT length is a
  //  power of 4 or not.
  template <Alignment Align>
  void Radix4FinalStepRealInv(ElementType *data, size_t length)
  {
    // Divide by length/2 to get into the correct space
    const auto scale = Set1(ElementType(2.0) / ElementType(length));
    const auto z = Set1(InvSqrt2<ElementType>);
    const auto stride = length / 8;
    for (auto i = 0u; i < stride; i += VecSize)
    {
      VecType a, b, c, d, e, f, g, h, t1;
      LoadStridedSets<Align>(data + i, stride, a, b, c, d, e, f, g, h);
      DIFRealOutputRadix4PairInv(a, b, c, d, e, f, g, h, z, t1);

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


  template <Alignment Align>
  void RealInverseFFT(const ElementType *tableData, const ElementType *input, ElementType *output, size_t length)
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