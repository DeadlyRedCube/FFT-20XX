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

      VT r00, r01, r02, r03, r04, r05, r06, r07, r08;
      LoadStridedSets<Align>(data, 2, r00, r01);
      LoadStridedSets<Align>(data + start1, 2, r02, r03);
      LoadStridedSets<Align>(data + start2, 2, r04, r05);
      LoadStridedSets<Align>(data + start3, 2, r06, r07);

      // Deinterleave the real/imaginary components
      Transpose2(r00, r01, r08);
      Transpose2(r04, r05, r00);
      Transpose2(r02, r03, r04);
      Transpose2(r06, r07, r02);

      auto angles0 = LoadSIMDAngleDouble(tableData, 0, flipStride);
      FirstRealDIFRealComplexRadix4Inv(r08, r01, r00, r05, r04, r03, r02, r07, angles0.w1r, angles0.w1i, angles0.w0r, angles0.w0i);

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

      auto angles1 = LoadSIMDAngleGroup(tableData, 2, flipStride);
      DIFComplexRadix4Inv(r08, r02, r00, r06, r04, r03, r01, r07, angles1);

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

      VT r00, r01, r02, r03, r04, r05, r06, r07, r08;

      LoadStridedSets<Align>(data + start0, 2, r00, r01);
      LoadStridedSets<Align>(data + start1, 2, r02, r03);
      LoadStridedSets<Align>(data + start2, 2, r04, r05);
      LoadStridedSets<Align>(data + start3, 2, r06, r07);

      Transpose2(r00, r01, r08);
      Transpose2(r04, r05, r00);
      Transpose2(r02, r03, r04);
      Transpose2(r06, r07, r02);

      auto angles0 = LoadSIMDAngleGroup(tableData, angleIndex0, flipStride);
      DIFComplexRadix4Inv(r08, r01, r00, r05, r04, r03, r02, r07, angles0);

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

      auto angles1 = LoadSIMDAngleGroup(tableData, angleIndex0 + 2, flipStride);
      DIFComplexRadix4Inv(r07, r02, r00, r08, r06, r03, r01, r05, angles1);

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
  else if constexpr (VV == 4)
  {
    const auto flipStride = length / 4;

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

      // Re-interleave the pairs (a/b, c/d, e/f, g/h)
      Deinterleave4(r00, r01, r08);
      Deinterleave4(r04, r05, r00);
      Deinterleave4(r02, r03, r04);
      Deinterleave4(r06, r07, r02);

      // reuse r06
      auto angles = LoadSIMDAngleDouble(tableData, 0, flipStride);

      FirstRealDIFRealComplexRadix4Inv(r08, r01, r00, r05, r04, r03, r02, r07, angles.w1r, angles.w1i, angles.w0r, angles.w0i);

      // Transpose these - note that 1 and 2 are out of order - this makes the deinterleave at the end simpler.
      Transpose4(r08, r00, r04, r02);
      Transpose4(r01, r05, r03, r07);

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
      VT r00, r01, r02, r03, r04, r05, r06, r07, r08;
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

      auto angles = LoadSIMDAngleGroup(tableData, angleIndex0, flipStride);
      DIFComplexRadix4Inv(r08, r01, r00, r05, r04, r03, r02, r07, angles);

      Transpose4(r08, r00, r04, r02);
      Transpose4(r01, r05, r03, r07);

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
    static_assert (VV == 8);

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

      Deinterleave4(r00, r01, r08);
      Deinterleave4(r04, r05, r00);
      Deinterleave4(r02, r03, r04);
      Deinterleave4(r06, r07, r02);

      auto angles = LoadSIMDAngleDouble(tableData, 0, flipStride);
      FirstRealDIFRealComplexRadix4Inv(r08, r01, r00, r05, r04, r03, r02, r07, angles.w1r, angles.w1i, angles.w0r, angles.w0i);

      Transpose4(r08, r00, r04, r02);
      Transpose4(r01, r05, r03, r07);

      // swap {e0, g0} and {b0, d0} because the real butterfly needs them in this order instead.
      SwapRealComponentsForLastDoubleFlip(r08, r01, r06);
      StoreHalves<Align>(&data[0], &data[0 + 8], r06);
      StoreHalves<Align>(&data[4], &data[4 + 8], r01);
      StoreHalves<Align>(&data[start1 + 0], &data[start1 + 0 + 8], r04);
      StoreHalves<Align>(&data[start1 + 4], &data[start1 + 4 + 8], r03);
      StoreHalves<Align>(&data[start2 + 0], &data[start2 + 0 + 8], r00);
      StoreHalves<Align>(&data[start2 + 4], &data[start2 + 4 + 8], r05);
      StoreHalves<Align>(&data[start3 + 0], &data[start3 + 0 + 8], r02);
      StoreHalves<Align>(&data[start3 + 4], &data[start3 + 4 + 8], r07);
    }

    // Keep this as a standard `for` loop because the condition may actually be false at the start for small FFTs.
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

      Deinterleave4(r00, r01, r08);
      Deinterleave4(r04, r05, r00);
      Deinterleave4(r02, r03, r04);
      Deinterleave4(r06, r07, r02);

      auto angles = LoadSIMDAngleGroup(tableData, angleIndex0, flipStride);
      DIFComplexRadix4Inv(r08, r01, r00, r05, r04, r03, r02, r07, angles);

      Transpose4(r08, r00, r04, r02);
      Transpose4(r01, r05, r03, r07);

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
void SecondDoubleFlipRadix4PassRealInv(const F *tableData, F *data, size_t length)
{
  static_assert(VS == 8);

  constexpr auto subLength = 32_z;
  const auto flipStride = length / subLength * 2;
  const auto flipGroupLength = flipStride * 4;

  const auto stride = subLength / 8;

  const auto shift = 1 + std::countl_zero(length);
  const auto reversedFlipStride = BitRev(flipStride) >> shift;

  // First flip group's first *subgroup* has the real-input radix4s. Do this as half-vectors
  using HalfVec = decltype(LoadHalf<Align>(data));

  {
    constexpr auto start0 = 0_z;
    auto start1 = start0 +  1 * flipStride;
    auto start2 = start0 +  2 * flipStride;
    auto start3 = start0 +  3 * flipStride;
    const auto angleIndex1 = reversedFlipStride;
    const auto angleIndex2 = (reversedFlipStride >> 1);
    const auto angleIndex3 = angleIndex1 | angleIndex2;

    const auto angles1 = LoadScalarAngleGroup(tableData, angleIndex1, flipStride);
    const auto angles2 = LoadScalarAngleGroup(tableData, angleIndex2, flipStride);
    const auto angles3 = LoadScalarAngleGroup(tableData, angleIndex3, flipStride);

    for (auto i = 0u;;)
    {
      HalfVec r00, r01, r02, r03, r04, r05, r06, r07, r08, r09, r10, r11;
      r00 = LoadHalf<Align>(&data[start0 + i + 0 * stride]);
      r01 = LoadHalf<Align>(&data[start0 + i + 1 * stride]);
      r02 = LoadHalf<Align>(&data[start2 + i + 0 * stride]);
      r03 = LoadHalf<Align>(&data[start2 + i + 1 * stride]);
      r04 = LoadHalf<Align>(&data[start1 + i + 0 * stride]);
      r05 = LoadHalf<Align>(&data[start1 + i + 1 * stride]);
      r06 = LoadHalf<Align>(&data[start3 + i + 0 * stride]);
      r07 = LoadHalf<Align>(&data[start3 + i + 1 * stride]);

      DIFRealOutputRadix4PairInv(r00, r01, r02, r03, r04, r05, r06, r07);

      Store<Align>(&data[start0 + i + 0 * stride], r00);
      Store<Align>(&data[start0 + i + 1 * stride], r01);
      r00 = LoadHalf<Align>(&data[start0 + i + 2 * stride]);
      r01 = LoadHalf<Align>(&data[start0 + i + 3 * stride]);
      Store<Align>(&data[start0 + i + 2 * stride], r02);
      Store<Align>(&data[start0 + i + 3 * stride], r03);
      r02 = LoadHalf<Align>(&data[start0 + i + 4 * stride]);
      r03 = LoadHalf<Align>(&data[start0 + i + 5 * stride]);
      Store<Align>(&data[start0 + i + 4 * stride], r04);
      Store<Align>(&data[start0 + i + 5 * stride], r05);
      r04 = LoadHalf<Align>(&data[start0 + i + 6 * stride]);
      r05 = LoadHalf<Align>(&data[start0 + i + 7 * stride]);
      Store<Align>(&data[start0 + i + 6 * stride], r06);
      Store<Align>(&data[start0 + i + 7 * stride], r07);


      r06 = LoadHalf<Align>(&data[start2 + i + 4 * stride]);
      r07 = LoadHalf<Align>(&data[start2 + i + 5 * stride]);
      r08 = LoadHalf<Align>(&data[start1 + i + 4 * stride]);
      r09 = LoadHalf<Align>(&data[start1 + i + 5 * stride]);
      r10 = LoadHalf<Align>(&data[start3 + i + 4 * stride]);
      r11 = LoadHalf<Align>(&data[start3 + i + 5 * stride]);

      DIFComplexRadix4Inv(r02, r03, r06, r07, r08, r09, r10, r11, angles1);

      Store<Align>(&data[start1 + i + 0 * stride], r02);
      Store<Align>(&data[start1 + i + 1 * stride], r06);
      Store<Align>(&data[start1 + i + 4 * stride], r03);
      Store<Align>(&data[start1 + i + 5 * stride], r07);

      r02 = LoadHalf<Align>(&data[start1 + i + 2 * stride]);
      r03 = LoadHalf<Align>(&data[start1 + i + 3 * stride]);
      Store<Align>(&data[start1 + i + 2 * stride], r08);
      Store<Align>(&data[start1 + i + 3 * stride], r10);
      r06 = LoadHalf<Align>(&data[start1 + i + 6 * stride]);
      r07 = LoadHalf<Align>(&data[start1 + i + 7 * stride]);
      Store<Align>(&data[start1 + i + 6 * stride], r09);
      Store<Align>(&data[start1 + i + 7 * stride], r11);


      r08 = LoadHalf<Align>(&data[start2 + i + 2 * stride]);
      r09 = LoadHalf<Align>(&data[start2 + i + 3 * stride]);
      r10 = LoadHalf<Align>(&data[start3 + i + 2 * stride]);
      r11 = LoadHalf<Align>(&data[start3 + i + 3 * stride]);

      DIFComplexRadix4Inv(r00, r01, r08, r09, r02, r03, r10, r11, angles2);

      Store<Align>(&data[start2 + i + 0 * stride], r00);
      Store<Align>(&data[start2 + i + 1 * stride], r08);
      Store<Align>(&data[start2 + i + 2 * stride], r02);
      Store<Align>(&data[start2 + i + 3 * stride], r10);
      Store<Align>(&data[start2 + i + 4 * stride], r01);
      Store<Align>(&data[start2 + i + 5 * stride], r09);
      r00 = LoadHalf<Align>(&data[start2 + i + 6 * stride]);
      r01 = LoadHalf<Align>(&data[start2 + i + 7 * stride]);
      Store<Align>(&data[start2 + i + 6 * stride], r03);
      Store<Align>(&data[start2 + i + 7 * stride], r11);

      r02 = LoadHalf<Align>(&data[start3 + i + 6 * stride]);
      r03 = LoadHalf<Align>(&data[start3 + i + 7 * stride]);

      DIFComplexRadix4Inv(r04, r05, r00, r01, r06, r07, r02, r03, angles3);

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

      const auto angles01 = LoadScalarAngleGroupX2(tableData, angleIndex0, angleIndex1, flipStride);
      const auto angles23 = LoadScalarAngleGroupX2(tableData, angleIndex2, angleIndex3, flipStride);

      for (auto i = 0u;;)
      {
        // We can do this as two effective radix4s instead of 4
        VecType r00, r01, r02, r03, r04, r05, r06, r07;
        r00 = LoadHalves<Align>(&data[start0 + i + 0 * stride], &data[start0 + i + 4 * stride]);
        r01 = LoadHalves<Align>(&data[start0 + i + 1 * stride], &data[start0 + i + 5 * stride]);
        r02 = LoadHalves<Align>(&data[start2 + i + 0 * stride], &data[start2 + i + 4 * stride]);
        r03 = LoadHalves<Align>(&data[start2 + i + 1 * stride], &data[start2 + i + 5 * stride]);
        r04 = LoadHalves<Align>(&data[start1 + i + 0 * stride], &data[start1 + i + 4 * stride]);
        r05 = LoadHalves<Align>(&data[start1 + i + 1 * stride], &data[start1 + i + 5 * stride]);
        r06 = LoadHalves<Align>(&data[start3 + i + 0 * stride], &data[start3 + i + 4 * stride]);
        r07 = LoadHalves<Align>(&data[start3 + i + 1 * stride], &data[start3 + i + 5 * stride]);

        DIFComplexRadix4Inv(r00, r01, r02, r03, r04, r05, r06, r07, angles01);

        StoreLowerHalves<Align>(&data[start0 + i + 0 * stride], r00, r02);
        StoreUpperHalves<Align>(&data[start1 + i + 0 * stride], r00, r02);
        StoreLowerHalves<Align>(&data[start0 + i + 4 * stride], r01, r03);
        StoreUpperHalves<Align>(&data[start1 + i + 4 * stride], r01, r03);

        r00 = LoadHalves<Align>(&data[start0 + i + 2 * stride], &data[start0 + i + 6 * stride]);
        r01 = LoadHalves<Align>(&data[start0 + i + 3 * stride], &data[start0 + i + 7 * stride]);
        r02 = LoadHalves<Align>(&data[start1 + i + 2 * stride], &data[start1 + i + 6 * stride]);
        r03 = LoadHalves<Align>(&data[start1 + i + 3 * stride], &data[start1 + i + 7 * stride]);

        StoreLowerHalves<Align>(&data[start0 + i + 2 * stride], r04, r06);
        StoreLowerHalves<Align>(&data[start0 + i + 6 * stride], r05, r07);
        StoreUpperHalves<Align>(&data[start1 + i + 2 * stride], r04, r06);
        StoreUpperHalves<Align>(&data[start1 + i + 6 * stride], r05, r07);

        r04 = LoadHalves<Align>(&data[start2 + i + 2 * stride], &data[start2 + i + 6 * stride]);
        r05 = LoadHalves<Align>(&data[start2 + i + 3 * stride], &data[start2 + i + 7 * stride]);
        r06 = LoadHalves<Align>(&data[start3 + i + 2 * stride], &data[start3 + i + 6 * stride]);
        r07 = LoadHalves<Align>(&data[start3 + i + 3 * stride], &data[start3 + i + 7 * stride]);

        DIFComplexRadix4Inv(r00, r01, r04, r05, r02, r03, r06, r07, angles23);

        StoreLowerHalves<Align>(&data[start2 + i + 0 * stride], r00, r04);
        StoreLowerHalves<Align>(&data[start2 + i + 2 * stride], r02, r06);
        StoreLowerHalves<Align>(&data[start2 + i + 4 * stride], r01, r05);
        StoreLowerHalves<Align>(&data[start2 + i + 6 * stride], r03, r07);

        StoreUpperHalves<Align>(&data[start3 + i + 0 * stride], r00, r04);
        StoreUpperHalves<Align>(&data[start3 + i + 2 * stride], r02, r06);
        StoreUpperHalves<Align>(&data[start3 + i + 4 * stride], r01, r05);
        StoreUpperHalves<Align>(&data[start3 + i + 6 * stride], r03, r07);

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

    const auto angles1 = LoadScalarAngleGroup(tableData, angleIndex1, flipStride);
    const auto angles2 = LoadScalarAngleGroup(tableData, angleIndex2, flipStride);
    const auto angles3 = LoadScalarAngleGroup(tableData, angleIndex3, flipStride);

    for (auto i = 0u;;)
    {
      VecType r00, r01, r02, r03, r04, r05, r06, r07, r08, r09, r10, r11;
      r00 = Load<Align>(&data[start0 + i + 0 * stride]);
      r01 = Load<Align>(&data[start0 + i + 1 * stride]);
      r02 = Load<Align>(&data[start2 + i + 0 * stride]);
      r03 = Load<Align>(&data[start2 + i + 1 * stride]);
      r04 = Load<Align>(&data[start1 + i + 0 * stride]);
      r05 = Load<Align>(&data[start1 + i + 1 * stride]);
      r06 = Load<Align>(&data[start3 + i + 0 * stride]);
      r07 = Load<Align>(&data[start3 + i + 1 * stride]);

      DIFRealOutputRadix4PairInv(r00, r01, r02, r03, r04, r05, r06, r07);

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

      DIFComplexRadix4Inv(r02, r03, r06, r07, r08, r09, r10, r11, angles1);

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

      DIFComplexRadix4Inv(r00, r01, r08, r09, r02, r03, r10, r11, angles2);

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

      DIFComplexRadix4Inv(r04, r05, r00, r01, r06, r07, r02, r03, angles3);

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

      const auto angles0 = LoadScalarAngleGroup(tableData, angleIndex0, flipStride);
      const auto angles1 = LoadScalarAngleGroup(tableData, angleIndex1, flipStride);
      const auto angles2 = LoadScalarAngleGroup(tableData, angleIndex2, flipStride);
      const auto angles3 = LoadScalarAngleGroup(tableData, angleIndex3, flipStride);
      for (auto i = 0u;;)
      {
        VecType r00, r01, r02, r03, r04, r05, r06, r07, r08, r09, r10, r11;
        r00 = Load<Align>(&data[start0 + i + 0 * stride]);
        r01 = Load<Align>(&data[start0 + i + 1 * stride]);
        r02 = Load<Align>(&data[start2 + i + 0 * stride]);
        r03 = Load<Align>(&data[start2 + i + 1 * stride]);
        r04 = Load<Align>(&data[start1 + i + 0 * stride]);
        r05 = Load<Align>(&data[start1 + i + 1 * stride]);
        r06 = Load<Align>(&data[start3 + i + 0 * stride]);
        r07 = Load<Align>(&data[start3 + i + 1 * stride]);

        DIFComplexRadix4Inv(r00, r01, r02, r03, r04, r05, r06, r07, angles0);

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
        DIFComplexRadix4Inv(r04, r06, r05, r07, r09, r10, r08, r11, angles1);

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

        DIFComplexRadix4Inv(r00, r02, r08, r09, r04, r05, r10, r11, angles2);

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

        DIFComplexRadix4Inv(r01, r03, r00, r02, r06, r07, r04, r05, angles3);

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
template <Alignment Align, std::floating_point F,  uint32_t VS = VecSize>
void SingleFlipRadix4PassRealInv(const F *tableData, F *data, size_t length, size_t subLength)
{
  if (IsPowerOfFourAssumingIsPowerOfTwo(length))
  {
    // The bits that flip for power of four lengths touch two separate radix4s - however, they are conveniently
    //  contiguous in memory.
    const auto angleCount = 2 * length / subLength;
    const auto stride = subLength / 8;

    if constexpr (VS > 4)
    {
      static_assert(VS == 8);
      if (VS > stride)
      {
        const auto subsetStep = subLength * 2;

        assert(stride == 4 && subLength == 32 && length == 256);

        // Do the first real case as half vetors
        using HalfVec = decltype(LoadHalf<Align>(data));
        {
          const auto angleIndex1 = stride;
          const auto angles1 = LoadScalarAngleGroup(tableData, angleIndex1, angleCount);

          for (auto i = 0u; i < subLength / 8; i += VecSize)
          {
            HalfVec r06, r07, r08, r09, r10, r11, r12, r13;
            r06 = LoadHalf<Align>(&data[i + 0 * stride]);
            r07 = LoadHalf<Align>(&data[i + 1 * stride]);
            r08 = LoadHalf<Align>(&data[i + 4 * stride]);
            r09 = LoadHalf<Align>(&data[i + 5 * stride]);
            r10 = LoadHalf<Align>(&data[i + subLength + 0 * stride]);
            r11 = LoadHalf<Align>(&data[i + subLength + 1 * stride]);
            r12 = LoadHalf<Align>(&data[i + subLength + 4 * stride]);
            r13 = LoadHalf<Align>(&data[i + subLength + 5 * stride]);

            // Unfortunately we're one register shy on x64 of being able to keep this resident like the other angles.
            DIFRealOutputRadix4PairInv(r06, r07, r10, r11, r08, r09, r12, r13);

            Store<Align>(data + i + 0 * stride, r06);
            Store<Align>(data + i + 1 * stride, r07);
            Store<Align>(data + i + 4 * stride, r08);
            Store<Align>(data + i + 5 * stride, r09);

            r06 = LoadHalf<Align>(&data[i + 2 * stride]);
            r07 = LoadHalf<Align>(&data[i + 3 * stride]);
            r08 = LoadHalf<Align>(&data[i + 6 * stride]);
            r09 = LoadHalf<Align>(&data[i + 7 * stride]);
            Store<Align>(data + i + 2 * stride, r10);
            Store<Align>(data + i + 3 * stride, r11);
            Store<Align>(data + i + 6 * stride, r12);
            Store<Align>(data + i + 7 * stride, r13);

            r10 = LoadHalf<Align>(&data[i + subLength + 2 * stride]);
            r11 = LoadHalf<Align>(&data[i + subLength + 3 * stride]);
            r12 = LoadHalf<Align>(&data[i + subLength + 6 * stride]);
            r13 = LoadHalf<Align>(&data[i + subLength + 7 * stride]);

            DIFComplexRadix4Inv(r06, r07, r10, r11, r08, r09, r12, r13, angles1);

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

        // Now also do the first complex set as half vectors (to re-align so we can then do two subsets at once)
        {
          const auto angleIndex0 = BitRev(subsetStep) >> (1 + std::countl_zero(length));
          const auto angleIndex1 = angleIndex0 + stride;

          const auto angles0 = LoadScalarAngleGroup(tableData, angleIndex0, angleCount);
          const auto angles1 = LoadScalarAngleGroup(tableData, angleIndex1, angleCount);

          HalfVec r00, r01, r02, r03, r04, r05, r06, r07;
          r00 = LoadHalf<Align>(&data[subsetStep + 0 * stride]);
          r01 = LoadHalf<Align>(&data[subsetStep + 1 * stride]);
          r02 = LoadHalf<Align>(&data[subsetStep + subLength + 0 * stride]);
          r03 = LoadHalf<Align>(&data[subsetStep + subLength + 1 * stride]);
          r04 = LoadHalf<Align>(&data[subsetStep + 4 * stride]);
          r05 = LoadHalf<Align>(&data[subsetStep + 5 * stride]);
          r06 = LoadHalf<Align>(&data[subsetStep + subLength + 4 * stride]);
          r07 = LoadHalf<Align>(&data[subsetStep + subLength + 5 * stride]);
          DIFComplexRadix4Inv(r00, r01, r02, r03, r04, r05, r06, r07, angles0);

          Store<Align>(data + subsetStep + 0 * stride, r00);
          Store<Align>(data + subsetStep + 1 * stride, r02);
          Store<Align>(data + subsetStep + 4 * stride, r01);
          Store<Align>(data + subsetStep + 5 * stride, r03);

          r00 = LoadHalf<Align>(&data[subsetStep + 2 * stride]);
          r01 = LoadHalf<Align>(&data[subsetStep + 3 * stride]);
          r02 = LoadHalf<Align>(&data[subsetStep + 6 * stride]);
          r03 = LoadHalf<Align>(&data[subsetStep + 7 * stride]);
          Store<Align>(data + subsetStep + 2 * stride, r04);
          Store<Align>(data + subsetStep + 3 * stride, r06);
          Store<Align>(data + subsetStep + 6 * stride, r05);
          Store<Align>(data + subsetStep + 7 * stride, r07);

          r04 = LoadHalf<Align>(&data[subsetStep + subLength + 2 * stride]);
          r05 = LoadHalf<Align>(&data[subsetStep + subLength + 3 * stride]);
          r06 = LoadHalf<Align>(&data[subsetStep + subLength + 6 * stride]);
          r07 = LoadHalf<Align>(&data[subsetStep + subLength + 7 * stride]);

          DIFComplexRadix4Inv(r00, r01, r04, r05, r02, r03, r06, r07, angles1);

          Store<Align>(&data[subsetStep + subLength + 0 * stride], r00);
          Store<Align>(&data[subsetStep + subLength + 1 * stride], r04);
          Store<Align>(&data[subsetStep + subLength + 4 * stride], r01);
          Store<Align>(&data[subsetStep + subLength + 5 * stride], r05);
          Store<Align>(&data[subsetStep + subLength + 2 * stride], r02);
          Store<Align>(&data[subsetStep + subLength + 3 * stride], r06);
          Store<Align>(&data[subsetStep + subLength + 6 * stride], r03);
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

          const auto baseA = data + subsetStartIndex;
          const auto baseB = baseA + subsetStep;
          VecType r00, r01, r02, r03, r04, r05, r06, r07;

          r00 = LoadHalves<Align>(&baseA[0 * stride], &baseB[0 * stride]);
          r01 = LoadHalves<Align>(&baseA[1 * stride], &baseB[1 * stride]);
          r02 = LoadHalves<Align>(&baseA[4 * stride], &baseB[4 * stride]);
          r03 = LoadHalves<Align>(&baseA[5 * stride], &baseB[5 * stride]);
          r04 = LoadHalves<Align>(&baseA[subLength + 0 * stride], &baseB[subLength + 0 * stride]);
          r05 = LoadHalves<Align>(&baseA[subLength + 1 * stride], &baseB[subLength + 1 * stride]);
          r06 = LoadHalves<Align>(&baseA[subLength + 4 * stride], &baseB[subLength + 4 * stride]);
          r07 = LoadHalves<Align>(&baseA[subLength + 5 * stride], &baseB[subLength + 5 * stride]);

          DIFComplexRadix4Inv(r00, r01, r04, r05, r02, r03, r06, r07, angles0);

          StoreHalves<Align>(&baseA[0 * stride], &baseB[0 * stride], r00);
          StoreHalves<Align>(&baseA[1 * stride], &baseB[1 * stride], r04);
          StoreHalves<Align>(&baseA[4 * stride], &baseB[4 * stride], r01);
          StoreHalves<Align>(&baseA[5 * stride], &baseB[5 * stride], r05);

          r00 = LoadHalves<Align>(&baseA[2 * stride], &baseB[2 * stride]);
          r01 = LoadHalves<Align>(&baseA[3 * stride], &baseB[3 * stride]);
          r04 = LoadHalves<Align>(&baseA[6 * stride], &baseB[6 * stride]);
          r05 = LoadHalves<Align>(&baseA[7 * stride], &baseB[7 * stride]);

          StoreHalves<Align>(&baseA[2 * stride], &baseB[2 * stride], r02);
          StoreHalves<Align>(&baseA[3 * stride], &baseB[3 * stride], r06);
          StoreHalves<Align>(&baseA[6 * stride], &baseB[6 * stride], r03);
          StoreHalves<Align>(&baseA[7 * stride], &baseB[7 * stride], r07);

          r02 = LoadHalves<Align>(&baseA[subLength + 2 * stride], &baseB[subLength + 2 * stride]);
          r03 = LoadHalves<Align>(&baseA[subLength + 3 * stride], &baseB[subLength + 3 * stride]);
          r06 = LoadHalves<Align>(&baseA[subLength + 6 * stride], &baseB[subLength + 6 * stride]);
          r07 = LoadHalves<Align>(&baseA[subLength + 7 * stride], &baseB[subLength + 7 * stride]);

          DIFComplexRadix4Inv(r00, r01, r02, r03, r04, r05, r06, r07, angles1);

          StoreHalves<Align>(&baseA[subLength + 0 * stride], &baseB[subLength + 0 * stride], r00);
          StoreHalves<Align>(&baseA[subLength + 1 * stride], &baseB[subLength + 1 * stride], r02);
          StoreHalves<Align>(&baseA[subLength + 2 * stride], &baseB[subLength + 2 * stride], r04);
          StoreHalves<Align>(&baseA[subLength + 3 * stride], &baseB[subLength + 3 * stride], r06);
          StoreHalves<Align>(&baseA[subLength + 4 * stride], &baseB[subLength + 4 * stride], r01);
          StoreHalves<Align>(&baseA[subLength + 5 * stride], &baseB[subLength + 5 * stride], r03);
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
        r06 = Load<Align>(&data[i + 0 * stride]);
        r07 = Load<Align>(&data[i + 1 * stride]);
        r08 = Load<Align>(&data[i + 4 * stride]);
        r09 = Load<Align>(&data[i + 5 * stride]);
        r10 = Load<Align>(&data[i + subLength + 0 * stride]);
        r11 = Load<Align>(&data[i + subLength + 1 * stride]);
        r12 = Load<Align>(&data[i + subLength + 4 * stride]);
        r13 = Load<Align>(&data[i + subLength + 5 * stride]);

        // Unfortunately we're one register shy on x64 of being able to keep this resident like the other angles.
        DIFRealOutputRadix4PairInv(r06, r07, r10, r11, r08, r09, r12, r13);

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

        DIFComplexRadix4Inv(r06, r07, r10, r11, r08, r09, r12, r13, angles);

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

      const auto angles0 = LoadScalarAngleGroup(tableData, angleIndex0, angleCount);

      const auto angles1 = LoadScalarAngleGroup(tableData, angleIndex1, angleCount);

      for (auto i = subsetStartIndex; i < subsetStartIndex + stride; i += VecSize)
      {
        VecType r00, r01, r02, r03, r04, r05, r06, r07;
        r00 = Load<Align>(&data[i + 0 * stride]);
        r01 = Load<Align>(&data[i + 1 * stride]);
        r02 = Load<Align>(&data[i + subLength + 0 * stride]);
        r03 = Load<Align>(&data[i + subLength + 1 * stride]);
        r04 = Load<Align>(&data[i + 4 * stride]);
        r05 = Load<Align>(&data[i + 5 * stride]);
        r06 = Load<Align>(&data[i + subLength + 4 * stride]);
        r07 = Load<Align>(&data[i + subLength + 5 * stride]);
        DIFComplexRadix4Inv(r00, r01, r02, r03, r04, r05, r06, r07, angles0);

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

        DIFComplexRadix4Inv(r00, r01, r04, r05, r02, r03, r06, r07, angles1);

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

    if constexpr (VS > 4)
    {
      static_assert(VS == 8);
      if (VS > stride)
      {
        // Need a special case for 128-length FFT
        assert(stride == 4 && subLength == 32 && length == 128);

        // Do the first real set and the first complex set as half-vectors
        using HalfVec = decltype(LoadHalf<Align>(data));
        {
          HalfVec a, b, c, d, e, f, g, h;
          LoadStridedSets<Align>(data, stride, a, b, e, f, c, d, g, h);
          DIFRealOutputRadix4PairInv(a, b, c, d, e, f, g, h);
          StoreStridedSets<Align>(data, stride, a, b, c, d, e, f, g, h);
        }

        {
          const auto angleIndex = BitRev(subLength) >> (1 + std::countl_zero(length));
          const auto angles = LoadScalarAngleGroup(tableData, angleIndex, angleCount);

          HalfVec aR, bR, cR, dR, aI, bI, cI, dI;
          LoadStridedSets<Align>(data + subLength, stride, aR, aI, cR, cI, bR, bI, dR, dI);
          DIFComplexRadix4Inv(aR, aI, bR, bI, cR, cI, dR, dI, angles);
          StoreStridedSets<Align>(data + subLength, stride, aR, bR, cR, dR, aI, bI, cI, dI);
        }

        // Now do two subsets of complex values at a time
        for (auto subsetStartIndex = subLength * 2; subsetStartIndex < length; subsetStartIndex += subLength * 2)
        {
          const auto angleIndexA = BitRev(subsetStartIndex) >> (1 + std::countl_zero(length));
          const auto angleIndexB = BitRev(subsetStartIndex + subLength) >> (1 + std::countl_zero(length));
          const auto angles = LoadScalarAngleGroupX2(tableData, angleIndexA, angleIndexB, angleCount);

          VecType aR, bR, cR, dR, aI, bI, cI, dI;
          auto *baseA = data + subsetStartIndex;
          auto *baseB = baseA + subLength;
          LoadStridedHalves<Align>(baseA, baseB, stride, aR, aI, cR, cI, bR, bI, dR, dI);
          DIFComplexRadix4Inv(aR, aI, bR, bI, cR, cI, dR, dI, angles);
          StoreStridedHalves<Align>(baseA, baseB, stride, aR, bR, cR, dR, aI, bI, cI, dI);
        }
        return;
      }
    }

    for (auto i = 0u; i < subLength / 8; i += VecSize)
    {
      VecType a, b, e, f, c, d, g, h;
      LoadStridedSets<Align>(data + i, stride, a, b, e, f, c, d, g, h);
      DIFRealOutputRadix4PairInv(a, b, c, d, e, f, g, h);
      StoreStridedSets<Align>(data + i, stride, a, b, c, d, e, f, g, h);
    }

    // Now do the complex values
    for (auto subsetStartIndex = subLength; subsetStartIndex < length; subsetStartIndex += subLength)
    {
      const auto angleIndex = BitRev(subsetStartIndex) >> (1 + std::countl_zero(length));
      const auto angles = LoadScalarAngleGroup(tableData, angleIndex, angleCount);

      for (auto i = subsetStartIndex; i < subsetStartIndex + stride; i += VecSize)
      {
        VecType aR, aI, cR, cI, bR, bI, dR, dI;
        LoadStridedSets<Align>(data + i, stride, aR, aI, cR, cI, bR, bI, dR, dI);
        DIFComplexRadix4Inv(aR, aI, bR, bI, cR, cI, dR, dI, angles);
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
  for (auto i = 0u; i < stride; i += VecSize)
  {
    VecType a, b, c, d, e, f, g, h;
    LoadStridedSets<Align>(data + i, stride, a, b, c, d, e, f, g, h);
    DIFRealOutputRadix4PairInv(a, b, c, d, e, f, g, h);
    StoreStridedSets<Align>(data + i, stride, a, b, c, d, e, f, g, h);
  }

  // Now do the complex values
  for (auto subsetStartIndex = subLength; subsetStartIndex < length; subsetStartIndex += subLength)
  {
    const auto angleIndex = BitRev(subsetStartIndex) >> (1 + std::countl_zero(length));
    const auto angles = LoadScalarAngleGroup(tableData, angleIndex, angleCount);

    for (auto i = subsetStartIndex; i < subsetStartIndex + stride; i += VecSize)
    {
      VecType aR, aI, bR, bI, cR, cI, dR, dI;
      LoadStridedSets<Align>(data + i, stride, aR, aI, bR, bI, cR, cI, dR, dI);
      DIFComplexRadix4Inv(aR, aI, bR, bI, cR, cI, dR, dI, angles);
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
    const auto wR = swRs[complexSubset];
    const auto wI = swIs[complexSubset];

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
  const auto scale = ElementType(2.0) / ElementType(length);
  const auto stride = length / 8;
  for (auto i = 0u; i < stride; i += VecSize)
  {
    VecType a, b, c, d, e, f, g, h;
    LoadStridedSets<Align>(data + i, stride, a, b, c, d, e, f, g, h);
    DIFRealOutputRadix4PairInv(a, b, c, d, e, f, g, h);

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


template <Alignment Align, uint32_t VS = VecSize>
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
  if constexpr (VS > 4)
  {
    if (maxDoubleFlipSubLength >= subLength)
    {
      SecondDoubleFlipRadix4PassRealInv<Align>(tableData, output, length);
      subLength <<= 2;
    }
  }

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