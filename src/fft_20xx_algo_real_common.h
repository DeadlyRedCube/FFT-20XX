#pragma once

FORCE_INLINE void ComplexMultiply(auto aR, auto aI, auto bR, auto bI, auto &cR, auto &cI, auto &tR, auto &tI)
{
  tR = Mul(aI, bR);
  tI = Mul(aI, bI);
  cR = Mul(aR, bR);
  cI = Mul(aR, bI);
  cR = Sub(cR, tI);
  cI = Add(cI, tR);
}


template <Alignment A, std::floating_point F, typename...V>
FORCE_INLINE void LoadStridedSets(const F *base, size_t stride, V &... vs)
{
  size_t offset = 0;
  ((vs = Load<A>(base + offset), offset += stride), ...);
}



template <Alignment A, std::floating_point F, typename...V>
FORCE_INLINE void StoreStridedSets(F *base, size_t stride, V...v)
{
  size_t offset = 0;
  ((Store<A>(base + offset, v), offset += stride), ...);
}



FORCE_INLINE void DITRealInputRadix2Pair(auto &a, auto &b, auto &c, auto &d, auto &aOut, auto &bOut)
{
  aOut = Add(a, c);
  bOut = Add(b, d);
  c = Sub(a, c);
  d = Sub(d, b);
}



FORCE_INLINE void DITRealOutputRadix2PairInv(auto &a, auto &b, auto &c, auto &d, auto &aOut, auto &bOut)
{
  aOut = Add(a, c);
  bOut = Sub(b, d);
  c = Sub(a, c);
  d = Add(d, b);
}



FORCE_INLINE void DITComplexRadix2(auto &aR, auto &aI, auto &bR, auto &bI, auto wR, auto wI, auto &aROut, auto &aIOut)
{
  // Rotate b * w
  aROut = Mul(bR, wR);
  aIOut = Mul(bI, wI);
  bR = Mul(bR, wI);
  bI = Mul(bI, wR);
  aROut = Sub(aROut, aIOut);
  bI = Add(bR, bI);
  // bR = t1;

  bR = Sub(aR, aROut);
  aROut = Add(aR, aROut);
  aIOut = Add(aI, bI);
  bI = Sub(bI, aI);
}



FORCE_INLINE void DIFComplexRadix2Inv(auto &aR, auto &aI, auto &bR, auto &bI, auto wR, auto wI, auto &aROut, auto &aIOut)
{
  aROut = Add(aR, bR);
  aIOut = Sub(aI, bI);
  bR = Sub(aR, bR);
  bI = Add(aI, bI);

  // Rotate b * conj(w), using aR/aI as temp storage since they're no longer needed
  aR = Mul(bR, wI);
  aI = Mul(bI, wI);
  bR = Mul(bR, wR);
  bI = Mul(bI, wR);
  bR = Add(bR, aI);
  bI = Sub(bI, aR);
}



FORCE_INLINE void DITRealInputRadix4Pair(auto &a, auto &b, auto &c, auto &d, auto &e, auto &f, auto &g, auto &h, auto s, auto &t)
{
  // Do the radix4 pair, shuffling data around through other named registers as temp space to get things into the correct
  //  location by the end.
  t = Sub(b, f);
  b = Add(b, f);
  f = Add(h, d);
  h = Sub(h, d);

  d = Sub(f, b);
  b = Add(f, b);

  f = Add(a, e);
  e = Sub(a, e);

  a = Mul(s, Add(t, h));
  t = Mul(s, Sub(h, t));

  h = Sub(g, c);
  c = Add(g, c);
  g = Sub(e, a);
  e = Add(e, a);
  a = Add(f, c);
  c = Sub(f, c);

  f = Add(t, h);
  h = Sub(t, h);
}



FORCE_INLINE void DIFRealOutputRadix4PairInv(
  auto &a,
  auto &b,
  auto &c,
  auto &d,
  auto &e,
  auto &f,
  auto &g,
  auto &h,
  auto s,
  auto &t)
{
  // Like the others, this uses a single temp register and uses other temporarily-unused values as additional storage.
  t = Sub(e, g);
  e = Add(e, g);
  g = Sub(f, h);
  h = Add(f, h);

  f = Mul(s, Sub(t, h));
  t = Mul(s, Add(t, h));

  h = Add(a, c);
  a = Sub(a, c);

  c = Sub(b, d);
  d = Add(b, d);

  b = Add(c, f);
  f = Sub(c, f);

  c = Sub(a, g);
  g = Add(a, g);

  a = Add(h, e);
  e = Sub(h, e);

  h = Add(d, t);
  d = Sub(d, t);
}



FORCE_INLINE void DITComplexRadix4(
  auto &aR,
  auto &aI,
  auto &bR,
  auto &bI,
  auto &cR,
  auto &cI,
  auto &dR,
  auto &dI,
  std::floating_point auto w0r,
  std::floating_point auto w0i,
  std::floating_point auto w1r,
  std::floating_point auto w1i,
  std::floating_point auto w2r,
  std::floating_point auto w2i,
  auto &t0,
  auto &t1,
  auto &t2)
{
  t1 = Set1(w1i);
  t0 = Set1(w1r);

  // Start by rotating c by w1
  t2 = Mul(cR, t1);
  t1 = Mul(cI, t1);
  cR = Mul(cR, t0);
  t0 = Mul(cI, t0);
  cR = Sub(cR, t1);
  cI = Add(t2, t0);

  t0 = Set1(w0r);
  t1 = Set1(w0i);

  // Start the a/c butterfly while that loads
  t2 = Add(aR, cR);
  cR = Sub(aR, cR);

  // Rotate b by w0
  aR = Mul(bR, t1);
  t1 = Mul(bI, t1);
  bR = Mul(bR, t0);
  t0 = Mul(bI, t0);

  bR = Sub(bR, t1);
  t1 = Add(aI, cI);
  cI = Sub(aI, cI);
  aI = Set1(w2i);
  bI = Add(aR, t0);
  t0 = Set1(w2r);

  // aR is unused here, angles are loaded as t0,aI

  // now rotate d by w0
  aR = Mul(dR, aI);
  aI = Mul(dI, aI);
  dI = Mul(dI, t0);
  t0 = Mul(dR, t0);
  t0 = Sub(t0, aI);
  dI = Add(aR, dI);

  // Do the last butterflies and get everything back into their proper registers
  aR = Add(bR, t0);
  aI = Add(bI, dI);
  dI = Sub(bI, dI);
  t0 = Sub(t0, bR); // actually -dR

  bI = Sub(aI, t1);
  aI = Add(t1, aI);
  bR = Sub(t2, aR);
  aR = Add(t2, aR);

  dR = Add(cR, dI);
  cR = Sub(cR, dI);
  dI = Add(t0, cI);
  cI = Sub(t0, cI);
};

// This function uses all 16 registers!
FORCE_INLINE void DITComplexRadix4(
  auto &aR,
  auto &aI,
  auto &bR,
  auto &bI,
  auto &cR,
  auto &cI,
  auto &dR,
  auto &dI,
  auto w0r,
  auto w0i,
  auto w1r,
  auto w1i,
  auto w2r,
  auto w2i,
  auto &t1,
  auto &t2)
{
  // Rotate c by w1
  t1 = Mul(cR, w1r);
  t2 = Mul(cI, w1i);
  cR = Mul(cR, w1i);
  cI = Mul(cI, w1r);
  t1 = Sub(t1, t2);
  cI = Add(cR, cI);

  // Do the a/c butterfly while that loads
  cR = Sub(aR, t1);
  t1 = Add(aR, t1);
  aR = Add(aI, cI);
  cI = Sub(aI, cI);

  // use aI and t2 as temp registers
  // Rotate b by w0
  aI = Mul(bR, w0r);
  t2 = Mul(bI, w0i);
  bR = Mul(bR, w0i);
  bI = Mul(bI, w0r);

    // Load w2 in the middle of this
  bI = Add(bR, bI);
  bR = Sub(aI, t2);

  // now rotate d by w2
  aI = Mul(dR, w2r);
  t2 = Mul(dI, w2i);
  dR = Mul(dR, w2i);
  dI = Mul(dI, w2r);
  dI = Add(dR, dI);
  aI = Sub(aI, t2);

  // Do the last butterflies
  dR = Sub(aI, bR); // actually -dR
  bR = Add(bR, aI);
  t2 = Sub(bI, dI);
  bI = Add(bI, dI);

  aI = Add(aR, bI);
  bI = Sub(bI, aR);
  aR = Add(t1, bR);
  bR = Sub(t1, bR);

  dI = Add(dR, cI);
  cI = Sub(dR, cI);
  dR = Add(cR, t2);
  cR = Sub(cR, t2);
};



FORCE_INLINE void DIFComplexRadix4Inv(
  VecType &aR,
  VecType &aI,
  VecType &bR,
  VecType &bI,
  VecType &cR,
  VecType &cI,
  VecType &dR,
  VecType &dI,
  std::floating_point auto w0r,
  std::floating_point auto w0i,
  std::floating_point auto w1r,
  std::floating_point auto w1i,
  std::floating_point auto w2r,
  std::floating_point auto w2i,
  VecType &t0,
  VecType &t1,
  VecType &t2)
{
  // This is effectively the opposite of the above Radix4.
  t0 = Add(dI, cI);
  t2 = Sub(dR, cR);
  cR = Add(dR, cR);
  cI = Sub(dI, cI);

  // This frees up dR and dI

  dR = Add(aR, bR);
  bR = Sub(aR, bR);
  dI = Sub(aI, bI);
  bI = Add(aI, bI);

  aR = Add(dR, cR);
  cR = Sub(dR, cR);
  aI = Add(dI, cI);
  cI = Sub(dI, cI);

  // aR and aI are *finalized* at this point - do not use!

  dR = Add(bR, t0);
  bR = Sub(bR, t0);
  t0 = Set1(w1r);      // Load the angles in the middle of this to get them ready
  t1 = Set1(w1i);
  dI = Sub(bI, t2);
  bI = Add(bI, t2);

  // Rotate c by w1
  t2 = Mul(cR, t0);
  cR = Mul(cR, t1);
  t1 = Mul(cI, t1);
  cI = Mul(cI, t0);
  t0 = Set1(w0r);  // Preload this angle for the b rotation
  cI = Sub(cI, cR);
  cR = Add(t2, t1);

  // Rotate b by w0
  t1 = Set1(w0i);
  t2 = Mul(bR, t0);
  bR = Mul(bR, t1);
  t1 = Mul(bI, t1);
  bI = Mul(bI, t0);
  t0 = Set1(w2r); // Preload this angle for the d rotation
  bI = Sub(bI, bR);
  bR = Add(t2, t1);

  // now rotate d by w2
  t1  = Set1(w2i);
  t2 = Mul(dR, t0);
  dR = Mul(dR, t1);
  t1 = Mul(dI, t1);
  dI = Mul(dI, t0);
  dI = Sub(dI, dR);
  dR = Add(t2, t1);
};



FORCE_INLINE void DIFComplexRadix4Inv(
  VecType &aR,
  VecType &aI,
  VecType &bR,
  VecType &bI,
  VecType &cR,
  VecType &cI,
  VecType &dR,
  VecType &dI,
  const VecType w0r,
  const VecType w0i,
  const VecType w1r,
  const VecType w1i,
  const VecType w2r,
  const VecType w2i,
  VecType &t1,
  VecType &t2)
{
  // This is effectively the opposite of the above Radix4.
  t1 = Add(dR, cR);
  dR = Sub(dR, cR);
  t2 = Sub(dI, cI);
  dI = Add(dI, cI);

  cR = Sub(aR, bR);
  aR = Add(aR, bR);
  cI = Add(aI, bI);
  aI = Sub(aI, bI);

  bR = Sub(cR, dI);
  dI = Add(cR, dI);
  bI = Add(cI, dR);
  dR = Sub(cI, dR);

  cR = Sub(aR, t1);
  aR = Add(aR, t1);
  cI = Sub(aI, t2);
  aI = Add(aI, t2);

  // aR/aI are finalized here do not use them as temp!

  t1 = Mul(bR, w0r);
  t2 = Mul(bI, w0i);
  bR = Mul(bR, w0i);
  bI = Mul(bI, w0r);

  bI = Sub(bI, bR);
  bR = Add(t1, t2);

  t1 = Mul(dI, w2r);
  t2 = Mul(dR, w2i);
  dI = Mul(dI, w2i);
  dR = Mul(dR, w2r);
  dI = Sub(dR, dI);
  dR = Add(t1, t2);

  t1 = Mul(cR, w1r);
  t2 = Mul(cI, w1i);
  cR = Mul(cR, w1i);
  cI = Mul(cI, w1r);
  cI = Sub(cI, cR);
  cR = Add(t1, t2);
};


// !!! register optimization
FORCE_INLINE void FinalRealDITRealComplexRadix4(
  auto &a,
  auto &b,
  auto &c,
  auto &d,
  auto &e,
  auto &f,
  auto &g,
  auto &h,
  auto &w0R, // Note that the angles here are mutable (and used as scratch) - they don't have to persist across loops
  auto &w0I, //  in the caller
  auto &w1R,
  auto &w1I,
  auto &t0,
  auto &t1)
{
  // For the final real DIT we need a second w1 angle that is just 0.707, -0.707 (but the complex w1 values remain
  //  the same)

  // ef *= angle0;
  t0 = Mul(e, w0R);
  t1 = Mul(f, w0I);
  e = Mul(e, w0I);
  f = Mul(f, w0R);
  f = Add(e, f);
  e = Sub(t0, t1);

  // gh *= angle0;
  t0 = Mul(g, w0R);
  t1 = Mul(h, w0I);
  g = Mul(g, w0I);
  h = Mul(h, w0R);
  w0R = Sub(t0, t1);
  w0I = Add(g, h);

  // g, h are free here
  t0 = Add(a, e);
  e = Sub(a, e);
  t1 = Add(f, b);
  f = Sub(f, b);

  g = Sub(w0R, c); // Do -g here so that we can leverage it for the 90 degree rotation later
  c = Add(w0R, c);
  h = Sub(w0I, d);
  d = Add(w0I, d);

  // Need to do another swap (this time of b0 and c0, again for the "real" radix's different ordering)
  b = ReplaceFirstOf(t1, c);
  c = ReplaceFirstOf(c, t1);

  // t0 is in use, but t1, w0R, w0I are not.

  // cd *= angle1;
  w0R = Mul(c, w1R);
  w0I = Mul(d, w1I);
  c = Mul(c, w1I);
  d = Mul(d, w1R);
  w1R = ReplaceFirstOf(w1R, InvSqrt2<ElementType>);
  w1I = ReplaceFirstOf(w1I, -InvSqrt2<ElementType>);

  t1 = Add(c, d);
  c = Sub(w0R, w0I);

  // gh *= {-angle1.imag(), -angle1.real()};
  //  note that this takes advantage of g being stored as -g above
  w0R = Mul(g, w1R);
  w0I = Mul(h, w1I);
  g = Mul(g, w1I);
  h = Mul(h, w1R);
  w0R = Sub(w0R, w0I);
  h = Add(g, h);

  a = Add(t0, c);
  c = Sub(t0, c);
  d = Sub(t1, b);
  b = Add(t1, b);

  g = Sub(e, h);
  e = Add(e, h);
  h = Sub(w0R, f);
  f = Add(w0R, f);
}



FORCE_INLINE void FirstRealDITRealComplexRadix4Inv(
  auto &a,
  auto &b,
  auto &c,
  auto &d,
  auto &e,
  auto &f,
  auto &g,
  auto &h,
  auto &w0R, // Like above, these are mutable and used as scratch because the caller does not need them to persist
  auto &w0I,
  auto &w1R,
  auto &w1I,
  auto &t0,
  auto &t1)
{
  t0 = Sub(e, g);
  e = Add(e, g);
  g = Add(f, h);
  f = Sub(f, h);

  h = Sub(a, c);
  a = Add(a, c);
  t1 = Add(b, d);
  b = Sub(b, d);

  // c, d are free

  d = Mul(h, w1R);
  h = Mul(h, w1I);
  c = Mul(t1, w1I);
  t1 = Mul(t1, w1R);
  c = Add(d, c);
  d = Sub(t1, h);

  // For the inverse of the final real DIT we need a second w1 angle that is just 0.707, -0.707 (but the complex w1
  //  values remain the same)
  w1R = ReplaceFirstOf(w1R, InvSqrt2<ElementType>);
  w1I = ReplaceFirstOf(w1I, -InvSqrt2<ElementType>);

  t1 = Mul(g, w1R);
  w1R = Mul(t0, w1R);
  t0 = Mul(t0, w1I);
  w1I = Mul(g, w1I);
  g = Add(t1, t0);
  h = Sub(w1R, w1I);

  t0 = ReplaceFirstOf(c, b);
  b = ReplaceFirstOf(b, c);

  c = Sub(t0, g);
  g = Add(t0, g);
  t1 = Add(d, h);
  d = Sub(d, h);

  w1R = Sub(a, e);
  a = Add(a, e);
  w1I = Add(b, f);
  b = Sub(b, f);

  f = Mul(g, w0R);
  e = Mul(t1, w0I);
  h = Mul(t1, w0R);
  g = Mul(g, w0I);
  h = Sub(h, g);
  g = Add(f, e);

  t0 = Mul(w1R, w0R);
  t1 = Mul(w1I, w0I);
  w0I = Mul(w1R, w0I);
  w0R= Mul(w1I, w0R);
  e = Add(t0, t1);
  f = Sub(w0R, w0I);
}


template <Alignment A, TangleType Type, uint32_t VV = VecSize>
void LargeTanglePairs(const ElementType *input, ElementType *output, size_t subLength)
{
  // For example in this routine, going to use a 32-pair (64-element) example of ABCDEFGHIJKLMNOPQponmlkjihgfedcb as
  //  the inputs, and reverse them in blocks of 4, with ForwardLargest and InverseLargest flipping the last 3 elements
  //  in every other quad of pairs (i.e. ABCD -> ADCB). InverseLargest will reverse the first, third, fifth, etc. quads,
  //  and ForwardLargest will reverse the second, fourth, sixth, etc. quads.
  // There are 3 possible results:
  //  - For ForwardLargest we result in: Adcb|efgh|ilkj|mnop|QNOP|MLKJ|IFGH|EDCB
  //  - For InverseLargest:              abcd|ehgf|ijkl|mpon|QPON|MJKL|IHGF|EBCD
  //  - Otherwise (no 3-reverses):        Adcb|ehgf|ilkj|mpon|QNOP|MJKL|IFGH|EBCD

  if constexpr (VV == 2)
  {
    // The 2-element case is simpler than the others
    {
      // Start by loading the first 4 elements and last 3 elements
      VecType t0, t1, t2, t3, l1, l2, l3;
      LoadStridedSets<A>(input, 2, t0, t1, t2, t3);
      LoadStridedSets<A>(input + subLength - 6, 2, l1, l2, l3);

      // The first element doesn't move (but input might not be the same pointer as output)
      Store<A>(output, t0);

      // The elements loaded from the end write out right after (reversed for InverseLargest)
      if constexpr (Type == TangleType::InverseLargest)
        { StoreStridedSets<A>(output + 2, 2, l3, l2, l1); }
      else
        { StoreStridedSets<A>(output + 2, 2, l1, l2, l3); }

      // Elements 1-3 write out at the end
      if constexpr (Type == TangleType::ForwardLargest)
        { StoreStridedSets<A>(output + subLength - 6, 2, t3, t2, t1); }
      else
        { StoreStridedSets<A>(output + subLength - 6, 2, t1, t2, t3); }
    }

    for (auto i = 8u;;)
    {
      const auto bottom = subLength - i - 14;

      // We have 16 registers so load all of the next 8 pairs from the top and from the bottom.
      VecType t0, t1, t2, t3, t4, t5, t6, t7, l0, l1, l2, l3, l4, l5, l6, l7;
      LoadStridedSets<A>(input + i, 2, t0, t1, t2, t3, t4, t5, t6, t7);
      LoadStridedSets<A>(input + bottom, 2, l0, l1, l2, l3, l4, l5, l6, l7);

      // These store in different orders depending on the type
      if constexpr (Type == TangleType::ForwardLargest)
      {
        StoreStridedSets<A>(output + i, 2, l7, l6, l5, l4, l3, l0, l1, l2);
        StoreStridedSets<A>(output + bottom, 2, t7, t6, t5, t4, t1, t2, t3, t0);
      }
      else if constexpr(Type == TangleType::InverseLargest)
      {
        StoreStridedSets<A>(output + i, 2, l7, l4, l5, l6, l3, l2, l1, l0);
        StoreStridedSets<A>(output + bottom, 2, t5, t6, t7, t4, t3, t2, t1, t0);
      }
      else
      {
        StoreStridedSets<A>(output + i, 2, l7, l4, l5, l6, l3, l0, l1, l2);
        StoreStridedSets<A>(output + bottom, 2, t5, t6, t7, t4, t1, t2, t3, t0);
      }

      i += 16;
      if (i >= subLength / 2 - 8)
        { break; }
    }

    // The last middle section just needs to deal with 9 values
    {
      const auto offset = subLength / 2 - 8;
      VecType t0, t1, t2, t3, m, l0, l1, l2, l3;
      LoadStridedSets<A>(input + offset, 2, t0, t1, t2, t3, m, l0, l1, l2, l3);
      if constexpr (Type == TangleType::ForwardLargest)
        { StoreStridedSets<A>(output + offset, 2, l3, l2, l1, l0, m, t1, t2, t3, t0); }
      else if constexpr (Type == TangleType::InverseLargest)
        { StoreStridedSets<A>(output + offset, 2, l3, l0, l1, l2, m, t3, t2, t1, t0); }
      else
        { StoreStridedSets<A>(output + offset, 2, l3, l0, l1, l2, m, t1, t2, t3, t0); }
    }
  }
  else
  {
    static_assert(VV == 4);

    // Load AB CD (as prevTopA and B) and second-to-last vector `ed` as prevTopB
    VecType prevTopA, prevTopB;
    LoadStridedSets<A>(input, 4, prevTopA, prevTopB);
    auto prevLowA = Load<A>(input + subLength - 8);

    // This first section gets us into the condition where the loop expects
    {
      VecType r03, r04, r05, r06, r07;
      // Load the last value in the array as r03 (it's `cb`)
      //  that write to the beginning of the output.
      r03 = Load<A>(input + subLength - 4);
      if constexpr (Type == TangleType::InverseLargest)
      {
        // In this case, we do an extra flip so we get the first elements (r04|r03) as Ab|cd
        r04 = InsertTopHalf(prevTopA, r03);
        r03 = InsertTopHalf(r03, prevLowA);
      }
      else
      {
        // No extra rotation so keep `dcb` in order (leaving r03 unchanged) to make the first elements Ad|cb.
        r04 = InsertTopHalf(prevTopA, prevLowA);
      }

      // Write these values out
      StoreStridedSets<A>(output, 4, r04, r03);

      // Load the next value in from the left (EF), and proceed to calculate the values that go at the very end of
      //  our array.
      r05 = Load<A>(input + 8);

      if constexpr (Type == TangleType::ForwardLargest)
      {
        // Do an extra swap of the three values here to make r05|prevTopB ED|CB
        r04 = InsertTopHalf(r05, prevTopB);
        prevTopB = InsertTopHalf(prevTopB, prevTopA);
      }
      else
      {
        // Leave the bottom 3 values in order and make EB|CD
        r04 = InsertTopHalf(r05, prevTopA);
      }

      // Store the very end values of our array.
      assert((output + subLength - 8) != (input + 12));
      StoreStridedSets<A>(output + subLength - 8, 4, r04, prevTopB);

      // We're going to do one more set! Load the next one in from the left (GH) and the next two in from the right
      //  (`ih` and `gf`) and write the next values at the start
      prevTopB = Load<A>(input + 12);
      LoadStridedSets<A>(input + subLength - 16, 4, r06, r07);
      if constexpr (Type == TangleType::ForwardLargest)
      {
        // Prep `ef|gh` to write out
        r04 = InsertTopHalf(prevLowA, r07);
        r07 = InsertTopHalf(r07, r06);
      }
      else
      {
        // We'll write `eh|gf`
        r04 = InsertTopHalf(prevLowA, r06);
      }
      StoreStridedSets<A>(output + 8, 4, r04, r07);
      prevTopA = r05; // This is now EF, to start the loop
      // prevTopB here is now GH
      prevLowA = r06; // This is now `ih` to start the loop
    }

    // Step by 16s through (doing 2 sets of 4 pairs at a time, half of which flip in
    //  Shuffle mode and half of which don't)
    for (auto i = size_t(16);;)
    {
      // Load the next starting values in from the left (IJ|KL on the first loop)
      VecType r03, r04, r05, r06, r07, r08;
      LoadStridedSets<A>(input + i, 4, r03, r04);

      if constexpr (Type == TangleType::InverseLargest)
      {
        // Prep IH|GF to write out on first loop
        r05 = InsertTopHalf(r03, prevTopB);
        prevTopB = InsertTopHalf(prevTopB, prevTopA);
      }
      else
      {
        // Prep IF|GH to write out on first loop
        r05 = InsertTopHalf(r03, prevTopA);
      }

      StoreStridedSets<A>(output + subLength - i, 4, r05, prevTopB);

      // Load the next values in from the bottom (`ml|kj` on first loop)
      LoadStridedSets<A>(input + subLength - i - 8, 4, r06, r07);

      if constexpr (Type == TangleType::InverseLargest)
      {
        // prep `ij|kl`
        r08 = InsertTopHalf(prevLowA, r07);
        r07 = InsertTopHalf(r07, r06);
      }
      else
      {
        // prep `il|kj`
        r08 = InsertTopHalf(prevLowA, r06);
      }
      StoreStridedSets<A>(output + i, 4, r08, r07);

      // Next ones in from the start again (MN|OP)
      LoadStridedSets<A>(input + i + 8, 4, prevTopA, prevTopB);

      if constexpr (Type == TangleType::ForwardLargest)
      {
        // Prep ML|KJ to write out
        r05 = InsertTopHalf(prevTopA, r04);
        r04 = InsertTopHalf(r04, r03);
      }
      else
      {
        // Prep MJ|KL
        r05 = InsertTopHalf(prevTopA, r03);
      }

      StoreStridedSets<A>(output + subLength - i - 8, 4, r05, r04);

      // One last section on this loop, load the next values in from the bottom (`Qp|on`)
      prevLowA = Load<A>(input + subLength - i - 16);
      r07 = Load<A>(input + subLength - i - 12);

      if constexpr (Type == TangleType::ForwardLargest)
      {
        // Prep `mn|op`
        r08 = InsertTopHalf(r06, r07);
        r07 = InsertTopHalf(r07, prevLowA);
      }
      else
      {
        // Prep `mp|on`
        r08 = InsertTopHalf(r06, prevLowA);
      }

      StoreStridedSets<A>(output + i + 8, 4, r08, r07);

      // At the end of the first (and only) loop in our example data, we have:
      //  prevTopA == MN (and `M` has already been used)
      //  prevTopB == OP
      //  prevLowA == Qp (and `Q` has already been used)
      i += 16;
      if (i >= subLength / 2)
        { break; }
    }

    // The very middle value needs to be written now.
    {
      VecType finalA, finalB;
      if constexpr (Type == TangleType::InverseLargest)
      {
        // In our example data, this makes QP|ON
        finalA = InsertTopHalf(prevLowA, prevTopB);
        finalB = InsertTopHalf(prevTopB, prevTopA);
      }
      else
      {
        // In our example data, this makes QN|OP
        finalA = InsertTopHalf(prevLowA, prevTopA);
        finalB = prevTopB;
      }

      StoreStridedSets<A>(output + subLength / 2, 4, finalA, finalB);
    }
  }
}



template <Alignment Align, uint32_t VV = VecSize>
void Copy4Pairs(const ElementType *input, ElementType *output)
{
  if constexpr (VV == 2)
  {
    VecType r0, r1, r2, r3;
    LoadStridedSets<Align>(input, 2, r0, r1, r2, r3);
    StoreStridedSets<Align>(output, 2, r0, r1, r2, r3);
  }
  else
  {
    static_assert(VV == 4);
    VecType Aa, Bb;
    LoadStridedSets<Align>(input, 4, Aa, Bb);
    StoreStridedSets<Align>(output, 4, Aa, Bb);
  }
}



template <Alignment Align, TangleType Type, uint32_t VV = VecSize>
void TangleFinal4Pairs(const ElementType *input, ElementType *output)
{
  if constexpr (Type == TangleType::ForwardLargest)
  {
    //  AbBa -> AaBb
    if constexpr (VV == 2)
    {
      VecType A, b, B, a;
      LoadStridedSets<Align>(input, 2, A, b, B, a);
      StoreStridedSets<Align>(output, 2, A, a, B, b);
    }
    else
    {
      static_assert(VV == 4);
      VecType Ab, Ba;
      LoadStridedSets<Align>(input, 4, Ab, Ba);
      auto Aa = InsertTopHalf(Ab, Ba);
      auto Bb = InsertTopHalf(Ba, Ab);
      StoreStridedSets<Align>(output, 4, Aa, Bb);
    }
  }
  else
  {
    // AaBb -> AbBa
    static_assert(Type == TangleType::InverseLargest);

    if constexpr (VV == 2)
    {
      VecType A, a, B, b;
      LoadStridedSets<Align>(input, 2, A, a, B, b);
      StoreStridedSets<Align>(output, 2, A, b, B, a);
    }
    else
    {
      static_assert(VV == 4);
      VecType Aa, Bb;
      LoadStridedSets<Align>(input, 4, Aa, Bb);
      auto Ab = InsertTopHalf(Aa, Bb);
      auto Ba = InsertTopHalf(Bb, Aa);
      StoreStridedSets<Align>(output, 4, Ab, Ba);
    }
  }
}



template <Alignment Align, TangleType Type, uint32_t VV = VecSize>
void Tangle8Pairs(const ElementType *input, ElementType *output)
{
  if constexpr (Type == TangleType::ForwardInner)
  {
    // AcDdCaBb -> AaBbCcDd
    if constexpr (VV == 2)
    {
      VecType A, a, B, b, C, c, D, d;
      LoadStridedSets<Align>(input, 2, A, c, D, d, C, a, B, b);
      StoreStridedSets<Align>(output, 2, A, a, B, b, C, c, D, d);
    }
    else
    {
      static_assert(VV == 4);
      VecType Ac, Dd, Ca, Bb;
      LoadStridedSets<Align>(input, 8, Ac, Ca);
      auto Aa = InsertTopHalf(Ac, Ca);
      auto Cc = InsertTopHalf(Ca, Ac);
      LoadStridedSets<Align>(input + 4, 8, Dd, Bb);
      StoreStridedSets<Align>(output, 4, Aa, Bb, Cc, Dd);
    }
  }
  else if constexpr (Type == TangleType::ForwardLargest)
  {
    // AdDcCaBb -> AaBbCcDd
    if constexpr (VV == 2)
    {
      VecType A, a, B, b, C, c, D, d;
      LoadStridedSets<Align>(input, 2, A, d, D, c, C, a, B, b);
      StoreStridedSets<Align>(output, 2, A, a, B, b, C, c, D, d);
    }
    else
    {
      static_assert(VV == 4);
      VecType Ad, Dc, Ca, Bb;
      LoadStridedSets<Align>(input, 4, Ad, Dc, Ca, Bb);
      auto Aa = InsertTopHalf(Ad, Ca);
      auto Dd = InsertTopHalf(Dc, Ad);
      auto Cc = InsertTopHalf(Ca, Dc);
      StoreStridedSets<Align>(output, 4, Aa, Bb, Cc, Dd);
    }
  }
  else if constexpr (Type == TangleType::InverseInner)
  {
    // AaBbCcDd -> AcDdCaBb
    if constexpr (VV == 2)
    {
      VecType A, a, B, b, C, c, D, d;
      LoadStridedSets<Align>(input, 2, A, a, B, b, C, c, D, d);
      StoreStridedSets<Align>(output, 2, A, c, D, d, C, a, B, b);
    }
    else
    {
      static_assert(VV == 4);
      VecType Aa, Bb, Cc, Dd;
      LoadStridedSets<Align>(input, 4, Aa, Bb, Cc, Dd);
      auto Ac = InsertTopHalf(Aa, Cc);
      auto Ca = InsertTopHalf(Cc, Aa);
      StoreStridedSets<Align>(output, 4, Ac, Dd, Ca, Bb);
    }
  }
  else
  {
    // AaBbCcDd -> AdDcCaBb
    static_assert(Type == TangleType::InverseLargest);

    if constexpr (VV == 2)
    {
      VecType A, a, B, b, C, c, D, d;
      LoadStridedSets<Align>(input, 2, A, a, B, b, C, c, D, d);
      StoreStridedSets<Align>(output, 2, A, d, D, c, C, a, B, b);
    }
    else
    {
      static_assert(VV == 4);
      VecType Aa, Bb, Cc, Dd;
      LoadStridedSets<Align>(input, 4, Aa, Bb, Cc, Dd);
      auto Ad = InsertTopHalf(Aa, Dd);
      auto Dc = InsertTopHalf(Dd, Cc);
      auto Ca = InsertTopHalf(Cc, Aa);
      StoreStridedSets<Align>(output, 4, Ad, Dc, Ca, Bb);
    }
  }
}



template <Alignment Align, TangleType Type, uint32_t VV = VecSize>
void Tangle16Pairs(const ElementType *input, ElementType *output)
{
  if constexpr (Type == TangleType::ForwardInner)
  {
    // AgHhGeFfEaBbCcDd -> AaBbCcDdEeFfGgHh
    if constexpr (VV == 2)
    {
      // With this one, every pair is a whole vector so this is easy to just move things around.
      VecType r00, r01, r02, r03, r04, r05, r06, r07;
      r00 = Load<Align>(input + 0);
      Store<Align>(output + 0, r00);

      r00 = Load<Align>(input + 2);
      r01 = Load<Align>(input + 18);
      Store<Align>(output + 2, r01);

      r01 = Load<Align>(input + 4);
      r02 = Load<Align>(input + 20);
      Store<Align>(output + 4, r02);

      r02 = Load<Align>(input + 6);
      r03 = Load<Align>(input + 22);
      Store<Align>(output + 6, r03);

      r03 = Load<Align>(input + 8);
      r04 = Load<Align>(input + 24);
      Store<Align>(output + 8, r04);

      r04 = Load<Align>(input + 10);
      r05 = Load<Align>(input + 26);
      Store<Align>(output + 10, r05);

      r05 = Load<Align>(input + 12);
      r06 = Load<Align>(input + 28);
      Store<Align>(output + 12, r06);

      r06 = Load<Align>(input + 14);
      r07 = Load<Align>(input + 30);
      Store<Align>(output + 14, r07);

      r07 = Load<Align>(input + 16);
      Store<Align>(output + 16, r07);

      Store<Align>(output + 18, r04);
      Store<Align>(output + 20, r05);
      Store<Align>(output + 22, r06);
      Store<Align>(output + 24, r03);
      Store<Align>(output + 26, r00);
      Store<Align>(output + 28, r01);
      Store<Align>(output + 30, r02);
    }
    else
    {
      static_assert(VV == 4);
      VecType r00, r01, r02, r03, r04, r05, r06, r07;
      LoadStridedSets<Align>(input, 4, r00, r01, r02);
      r03 = InsertTopHalf(r02, r00);
      LoadStridedSets<Align>(input + 12, 4, r04, r05);
      r00 = InsertTopHalf(r00, r05);
      r02 = InsertTopHalf(r05, r02);
      LoadStridedSets<Align>(input + 20, 4, r05, r06, r07);
      StoreStridedSets<Align>(output, 4, r00, r05, r06, r07, r02, r04, r03, r01);
    }
  }
  else if constexpr (Type == TangleType::ForwardLargest)
  {
    // AhHgGeFfEaBbCdDc -> AaBbCcDdEeFfGgHh
    if constexpr (VV == 2)
    {
      VecType r00, r01, r02, r03, r04, r05, r06, r07;
      r00 = Load<Align>(input + 0);
      Store<Align>(output + 0, r00);

      r00 = Load<Align>(input + 2);
      r01 = Load<Align>(input + 18);
      Store<Align>(output + 2, r01);

      r01 = Load<Align>(input + 4);
      r02 = Load<Align>(input + 20);
      Store<Align>(output + 4, r02);

      r02 = Load<Align>(input + 6);
      r03 = Load<Align>(input + 22);
      Store<Align>(output + 6, r03);

      r03 = Load<Align>(input + 8);
      r04 = Load<Align>(input + 24);
      Store<Align>(output + 8, r04);

      r04 = Load<Align>(input + 10);
      r05 = Load<Align>(input + 30);
      Store<Align>(output + 10, r05);

      r05 = Load<Align>(input + 12);
      r06 = Load<Align>(input + 28);
      Store<Align>(output + 12, r06);

      r06 = Load<Align>(input + 14);
      r07 = Load<Align>(input + 26);
      Store<Align>(output + 14, r07);

      r07 = Load<Align>(input + 16);
      Store<Align>(output + 16, r07);
      Store<Align>(output + 18, r04);
      Store<Align>(output + 20, r05);
      Store<Align>(output + 22, r06);
      Store<Align>(output + 24, r03);
      Store<Align>(output + 26, r02);
      Store<Align>(output + 28, r01);
      Store<Align>(output + 30, r00);
    }
    else
    {
      static_assert(VV == 4);
      VecType r00, r01, r02, r03, r04, r05, r06, r07, r08;
      LoadStridedSets<Align>(input, 4, r00, r01, r02);
      r03 = InsertTopHalf(r01, r00);
      r01 = InsertTopHalf(r02, r01);
      LoadStridedSets<Align>(input + 12, 4, r04, r05);
      r02 = InsertTopHalf(r05, r02);
      r00 = InsertTopHalf(r00, r05);
      LoadStridedSets<Align>(input + 20, 4, r05, r06, r07);
      r08 = InsertTopHalf(r06, r07);
      r06 = InsertTopHalf(r07, r06);
      StoreStridedSets<Align>(output, 4, r00, r05, r08, r06, r02, r04, r01, r03);
    }
  }
  else if constexpr (Type == TangleType::InverseInner)
  {
    // AaBbCcDdEeFfGgHh -> AgHhGeFfEaBbCcDd
    if constexpr (VV == 2)
    {
      auto A = Load<Align>(input + 0);
      auto a = Load<Align>(input + 2);
      auto B = Load<Align>(input + 4);
      auto b = Load<Align>(input + 6);
      auto C = Load<Align>(input + 8);
      auto c = Load<Align>(input + 10);
      auto D = Load<Align>(input + 12);
      auto d = Load<Align>(input + 14);
      auto E = Load<Align>(input + 16);
      auto e = Load<Align>(input + 18);
      auto F = Load<Align>(input + 20);
      auto f = Load<Align>(input + 22);
      auto G = Load<Align>(input + 24);
      auto g = Load<Align>(input + 26);
      auto H = Load<Align>(input + 28);
      auto h = Load<Align>(input + 30);

      Store<Align>(output + 0, A);
      Store<Align>(output + 2, g);
      Store<Align>(output + 4, H);
      Store<Align>(output + 6, h);
      Store<Align>(output + 8, G);
      Store<Align>(output + 10, e);
      Store<Align>(output + 12, F);
      Store<Align>(output + 14, f);
      Store<Align>(output + 16, E);
      Store<Align>(output + 18, a);
      Store<Align>(output + 20, B);
      Store<Align>(output + 22, b);
      Store<Align>(output + 24, C);
      Store<Align>(output + 26, c);
      Store<Align>(output + 28, D);
      Store<Align>(output + 30, d);
    }
    else
    {
      VecType r00, r01, r02, r03, r04, r05, r06, r07, r08;
      LoadStridedSets<Align>(input, 4, r00, r01, r02, r03, r04, r05, r06, r07);
      r08 = InsertTopHalf(r00, r06);
      r06 = InsertTopHalf(r06, r04);
      r00 = InsertTopHalf(r04, r00);
      StoreStridedSets<Align>(output, 4, r08, r07, r06, r05, r00, r01, r02, r03);
    }
  }
  else
  {
    // AaBbCcDdEeFfGgHh -> AhHgGeFfEaBbCdDc
    if constexpr (VV == 2)
    {
      auto A = Load<Align>(input + 0);
      auto a = Load<Align>(input + 2);
      auto B = Load<Align>(input + 4);
      auto b = Load<Align>(input + 6);
      auto C = Load<Align>(input + 8);
      auto c = Load<Align>(input + 10);
      auto D = Load<Align>(input + 12);
      auto d = Load<Align>(input + 14);
      auto E = Load<Align>(input + 16);
      auto e = Load<Align>(input + 18);
      auto F = Load<Align>(input + 20);
      auto f = Load<Align>(input + 22);
      auto G = Load<Align>(input + 24);
      auto g = Load<Align>(input + 26);
      auto H = Load<Align>(input + 28);
      auto h = Load<Align>(input + 30);

      Store<Align>(output + 0, A);
      Store<Align>(output + 2, h);
      Store<Align>(output + 4, H);
      Store<Align>(output + 6, g);
      Store<Align>(output + 8, G);
      Store<Align>(output + 10, e);
      Store<Align>(output + 12, F);
      Store<Align>(output + 14, f);
      Store<Align>(output + 16, E);
      Store<Align>(output + 18, a);
      Store<Align>(output + 20, B);
      Store<Align>(output + 22, b);
      Store<Align>(output + 24, C);
      Store<Align>(output + 26, d);
      Store<Align>(output + 28, D);
      Store<Align>(output + 30, c);
    }
    else
    {
      VecType r00, r01, r02, r03, r04, r05, r06, r07, r08;
      LoadStridedSets<Align>(input, 4, r00, r01, r02, r03, r04, r05, r06, r07);
      r08 = InsertTopHalf(r00, r07);
      r07 = InsertTopHalf(r07, r06);
      r06 = InsertTopHalf(r06, r04);
      r00 = InsertTopHalf(r04, r00);
      auto r09 = InsertTopHalf(r02, r03);
      r02 = InsertTopHalf(r03, r02);
      StoreStridedSets<Align>(output, 4, r08, r07, r06, r05, r00, r01, r09, r02);
    }
  }
}



// Swaps 64 elements (32 pairs)
template <Alignment Align, TangleType Type, uint32_t VV = VecSize>
void Tangle32Pairs(const ElementType *input, ElementType *output)
{
  if constexpr (Type == TangleType::ForwardInner)
  {
    // AoPpOmNnMiJjKkLlIaBbCcDdEgHhGeFf -> AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPp
    if constexpr (VV == 2)
    {
      auto A = Load<Align>(input + 0);
      auto o = Load<Align>(input + 2);
      auto P = Load<Align>(input + 4);
      auto p = Load<Align>(input + 6);
      auto O = Load<Align>(input + 8);
      auto m = Load<Align>(input + 10);
      auto N = Load<Align>(input + 12);
      auto n = Load<Align>(input + 14);
      auto M = Load<Align>(input + 16);
      auto i = Load<Align>(input + 18);
      auto J = Load<Align>(input + 20);
      auto j = Load<Align>(input + 22);
      auto K = Load<Align>(input + 24);
      auto k = Load<Align>(input + 26);
      auto L = Load<Align>(input + 28);
      auto l = Load<Align>(input + 30);
      auto I = Load<Align>(input + 32);
      auto a = Load<Align>(input + 34);
      auto B = Load<Align>(input + 36);
      auto b = Load<Align>(input + 38);
      auto C = Load<Align>(input + 40);
      auto c = Load<Align>(input + 42);
      auto D = Load<Align>(input + 44);
      auto d = Load<Align>(input + 46);
      auto E = Load<Align>(input + 48);
      auto g = Load<Align>(input + 50);
      auto H = Load<Align>(input + 52);
      auto h = Load<Align>(input + 54);
      auto G = Load<Align>(input + 56);
      auto e = Load<Align>(input + 58);
      auto F = Load<Align>(input + 60);
      auto f = Load<Align>(input + 62);

      Store<Align>(output + 0, A);
      Store<Align>(output + 2, a);
      Store<Align>(output + 4, B);
      Store<Align>(output + 6, b);
      Store<Align>(output + 8, C);
      Store<Align>(output + 10, c);
      Store<Align>(output + 12, D);
      Store<Align>(output + 14, d);
      Store<Align>(output + 16, E);
      Store<Align>(output + 18, e);
      Store<Align>(output + 20, F);
      Store<Align>(output + 22, f);
      Store<Align>(output + 24, G);
      Store<Align>(output + 26, g);
      Store<Align>(output + 28, H);
      Store<Align>(output + 30, h);
      Store<Align>(output + 32, I);
      Store<Align>(output + 34, i);
      Store<Align>(output + 36, J);
      Store<Align>(output + 38, j);
      Store<Align>(output + 40, K);
      Store<Align>(output + 42, k);
      Store<Align>(output + 44, L);
      Store<Align>(output + 46, l);
      Store<Align>(output + 48, M);
      Store<Align>(output + 50, m);
      Store<Align>(output + 52, N);
      Store<Align>(output + 54, n);
      Store<Align>(output + 56, O);
      Store<Align>(output + 58, o);
      Store<Align>(output + 60, P);
      Store<Align>(output + 62, p);
    }
    else
    {
      static_assert(VV == 4);

      // Conveniently with the loads and shuffles ordered as such, we fit within the x64 register space.
      VecType r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
      LoadStridedSets<Align>(input, 4, r0, r1, r2, r3, r4);
      r5 = InsertTopHalf(r2, r0);
      r2 = InsertTopHalf(r4, r2);
      LoadStridedSets<Align>(input + 20, 4, r6, r7, r8, r9);
      r0 = InsertTopHalf(r0, r9);
      r4 = InsertTopHalf(r9, r4);
      r9 = Load<Align>(input + 36);
      LoadStridedSets<Align>(input + 40, 4, r10, r11, r12, r13, r14);
      r15 = InsertTopHalf(r14, r12);
      r12 = InsertTopHalf(r12, r14);
      r14 = Load<Align>(input + 60);
      StoreStridedSets<Align>(output, 4, r0, r9, r10, r11, r12, r14, r15, r13, r4, r6, r7, r8, r2, r3, r5, r1);
    }
  }
  else if constexpr (Type == TangleType::ForwardLargest)
  {
    // ApPoOmNnMiJjKlLkIaBbCdDcEhHgGeFf -> AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPp
    //  (note that pPo, lLk, dDc, and hHg are reversed from the above)
    if constexpr (VV == 2)
    {
      auto A = Load<Align>(input + 0);
      auto p = Load<Align>(input + 2);
      auto P = Load<Align>(input + 4);
      auto o = Load<Align>(input + 6);
      auto O = Load<Align>(input + 8);
      auto m = Load<Align>(input + 10);
      auto N = Load<Align>(input + 12);
      auto n = Load<Align>(input + 14);
      auto M = Load<Align>(input + 16);
      auto i = Load<Align>(input + 18);
      auto J = Load<Align>(input + 20);
      auto j = Load<Align>(input + 22);
      auto K = Load<Align>(input + 24);
      auto l = Load<Align>(input + 26);
      auto L = Load<Align>(input + 28);
      auto k = Load<Align>(input + 30);
      auto I = Load<Align>(input + 32);
      auto a = Load<Align>(input + 34);
      auto B = Load<Align>(input + 36);
      auto b = Load<Align>(input + 38);
      auto C = Load<Align>(input + 40);
      auto d = Load<Align>(input + 42);
      auto D = Load<Align>(input + 44);
      auto c = Load<Align>(input + 46);
      auto E = Load<Align>(input + 48);
      auto h = Load<Align>(input + 50);
      auto H = Load<Align>(input + 52);
      auto g = Load<Align>(input + 54);
      auto G = Load<Align>(input + 56);
      auto e = Load<Align>(input + 58);
      auto F = Load<Align>(input + 60);
      auto f = Load<Align>(input + 62);

      Store<Align>(output + 0, A);
      Store<Align>(output + 2, a);
      Store<Align>(output + 4, B);
      Store<Align>(output + 6, b);
      Store<Align>(output + 8, C);
      Store<Align>(output + 10, c);
      Store<Align>(output + 12, D);
      Store<Align>(output + 14, d);
      Store<Align>(output + 16, E);
      Store<Align>(output + 18, e);
      Store<Align>(output + 20, F);
      Store<Align>(output + 22, f);
      Store<Align>(output + 24, G);
      Store<Align>(output + 26, g);
      Store<Align>(output + 28, H);
      Store<Align>(output + 30, h);
      Store<Align>(output + 32, I);
      Store<Align>(output + 34, i);
      Store<Align>(output + 36, J);
      Store<Align>(output + 38, j);
      Store<Align>(output + 40, K);
      Store<Align>(output + 42, k);
      Store<Align>(output + 44, L);
      Store<Align>(output + 46, l);
      Store<Align>(output + 48, M);
      Store<Align>(output + 50, m);
      Store<Align>(output + 52, N);
      Store<Align>(output + 54, n);
      Store<Align>(output + 56, O);
      Store<Align>(output + 58, o);
      Store<Align>(output + 60, P);
      Store<Align>(output + 62, p);
    }
    else
    {
      static_assert(VV == 4);
      VecType r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
      LoadStridedSets<Align>(input, 4, r0, r1, r2, r3, r4);
      r5 = InsertTopHalf(r2, r1);
      r1 = InsertTopHalf(r1, r0);
      r2 = InsertTopHalf(r4, r2);
      LoadStridedSets<Align>(input + 20, 4, r6, r7, r8);
      r9 = InsertTopHalf(r7, r8);
      r7 = InsertTopHalf(r8, r7);
      r8 = Load<Align>(input + 32);
      r0 = InsertTopHalf(r0, r8);
      r4 = InsertTopHalf(r8, r4);
      LoadStridedSets<Align>(input + 36, 4, r10, r11, r12);
      r8 = InsertTopHalf(r11, r12);
      r11 = InsertTopHalf(r12, r11);
      LoadStridedSets<Align>(input + 48, 4, r13, r14, r15);
      r12 = InsertTopHalf(r14, r13);
      r14 = InsertTopHalf(r15, r14);
      r13 = InsertTopHalf(r13, r15);
      r15 = Load<Align>(input + 60);
      StoreStridedSets<Align>(output, 4, r0, r10, r8, r11, r13, r15, r14, r12, r4, r6, r9, r7, r2, r3, r5, r1);
    }
  }
  else if constexpr (Type == TangleType::InverseInner)
  {
    // AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPp -> AoPpOmNnMiJjKkLlIaBbCcDdEgHhGeFf
    if constexpr (VV == 2)
    {
      auto A = Load<Align>(input + 0);
      auto a = Load<Align>(input + 2);
      auto B = Load<Align>(input + 4);
      auto b = Load<Align>(input + 6);
      auto C = Load<Align>(input + 8);
      auto c = Load<Align>(input + 10);
      auto D = Load<Align>(input + 12);
      auto d = Load<Align>(input + 14);
      auto E = Load<Align>(input + 16);
      auto e = Load<Align>(input + 18);
      auto F = Load<Align>(input + 20);
      auto f = Load<Align>(input + 22);
      auto G = Load<Align>(input + 24);
      auto g = Load<Align>(input + 26);
      auto H = Load<Align>(input + 28);
      auto h = Load<Align>(input + 30);
      auto I = Load<Align>(input + 32);
      auto i = Load<Align>(input + 34);
      auto J = Load<Align>(input + 36);
      auto j = Load<Align>(input + 38);
      auto K = Load<Align>(input + 40);
      auto k = Load<Align>(input + 42);
      auto L = Load<Align>(input + 44);
      auto l = Load<Align>(input + 46);
      auto M = Load<Align>(input + 48);
      auto m = Load<Align>(input + 50);
      auto N = Load<Align>(input + 52);
      auto n = Load<Align>(input + 54);
      auto O = Load<Align>(input + 56);
      auto o = Load<Align>(input + 58);
      auto P = Load<Align>(input + 60);
      auto p = Load<Align>(input + 62);

      Store<Align>(output + 0, A);
      Store<Align>(output + 2, o);
      Store<Align>(output + 4, P);
      Store<Align>(output + 6, p);
      Store<Align>(output + 8, O);
      Store<Align>(output + 10, m);
      Store<Align>(output + 12, N);
      Store<Align>(output + 14, n);
      Store<Align>(output + 16, M);
      Store<Align>(output + 18, i);
      Store<Align>(output + 20, J);
      Store<Align>(output + 22, j);
      Store<Align>(output + 24, K);
      Store<Align>(output + 26, k);
      Store<Align>(output + 28, L);
      Store<Align>(output + 30, l);
      Store<Align>(output + 32, I);
      Store<Align>(output + 34, a);
      Store<Align>(output + 36, B);
      Store<Align>(output + 38, b);
      Store<Align>(output + 40, C);
      Store<Align>(output + 42, c);
      Store<Align>(output + 44, D);
      Store<Align>(output + 46, d);
      Store<Align>(output + 48, E);
      Store<Align>(output + 50, g);
      Store<Align>(output + 52, H);
      Store<Align>(output + 54, h);
      Store<Align>(output + 56, G);
      Store<Align>(output + 58, e);
      Store<Align>(output + 60, F);
      Store<Align>(output + 62, f);
    }
    else
    {
      static_assert(VV == 4);
      VecType r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
      LoadStridedSets<Align>(input, 4, r0, r1, r2, r3, r4, r5, r6);
      r7 = InsertTopHalf(r6, r4);
      r4 = InsertTopHalf(r4, r6);
      LoadStridedSets<Align>(input + 28, 4, r8, r9);
      r6 = InsertTopHalf(r9, r0);
      LoadStridedSets<Align>(input + 36, 4, r10, r11, r12, r13);
      r9 = InsertTopHalf(r13, r9);
      LoadStridedSets<Align>(input + 52, 4, r14, r15);
      r0 = InsertTopHalf(r0, r15);
      r13 = InsertTopHalf(r15, r13);
      r15 = Load<Align>(input + 60);
      StoreStridedSets<Align>(output, 4, r0, r15, r13, r14, r9, r10, r11, r12, r6, r1, r2, r3, r4, r8, r7, r5);
    }
  }
  else if constexpr (Type == TangleType::InverseLargest)
  {
    // AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPp -> ApPoOmNnMiJjKlLkIaBbCdDcEhHgGeFf
    if constexpr (VV == 2)
    {
      auto A = Load<Align>(input + 0);
      auto a = Load<Align>(input + 2);
      auto B = Load<Align>(input + 4);
      auto b = Load<Align>(input + 6);
      auto C = Load<Align>(input + 8);
      auto c = Load<Align>(input + 10);
      auto D = Load<Align>(input + 12);
      auto d = Load<Align>(input + 14);
      auto E = Load<Align>(input + 16);
      auto e = Load<Align>(input + 18);
      auto F = Load<Align>(input + 20);
      auto f = Load<Align>(input + 22);
      auto G = Load<Align>(input + 24);
      auto g = Load<Align>(input + 26);
      auto H = Load<Align>(input + 28);
      auto h = Load<Align>(input + 30);
      auto I = Load<Align>(input + 32);
      auto i = Load<Align>(input + 34);
      auto J = Load<Align>(input + 36);
      auto j = Load<Align>(input + 38);
      auto K = Load<Align>(input + 40);
      auto k = Load<Align>(input + 42);
      auto L = Load<Align>(input + 44);
      auto l = Load<Align>(input + 46);
      auto M = Load<Align>(input + 48);
      auto m = Load<Align>(input + 50);
      auto N = Load<Align>(input + 52);
      auto n = Load<Align>(input + 54);
      auto O = Load<Align>(input + 56);
      auto o = Load<Align>(input + 58);
      auto P = Load<Align>(input + 60);
      auto p = Load<Align>(input + 62);

      Store<Align>(output + 0, A);
      Store<Align>(output + 2, p);
      Store<Align>(output + 4, P);
      Store<Align>(output + 6, o);
      Store<Align>(output + 8, O);
      Store<Align>(output + 10, m);
      Store<Align>(output + 12, N);
      Store<Align>(output + 14, n);
      Store<Align>(output + 16, M);
      Store<Align>(output + 18, i);
      Store<Align>(output + 20, J);
      Store<Align>(output + 22, j);
      Store<Align>(output + 24, K);
      Store<Align>(output + 26, l);
      Store<Align>(output + 28, L);
      Store<Align>(output + 30, k);
      Store<Align>(output + 32, I);
      Store<Align>(output + 34, a);
      Store<Align>(output + 36, B);
      Store<Align>(output + 38, b);
      Store<Align>(output + 40, C);
      Store<Align>(output + 42, d);
      Store<Align>(output + 44, D);
      Store<Align>(output + 46, c);
      Store<Align>(output + 48, E);
      Store<Align>(output + 50, h);
      Store<Align>(output + 52, H);
      Store<Align>(output + 54, g);
      Store<Align>(output + 56, G);
      Store<Align>(output + 58, e);
      Store<Align>(output + 60, F);
      Store<Align>(output + 62, f);
    }
    else
    {
      static_assert(VV == 4);
      VecType r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
      LoadStridedSets<Align>(input, 4, r0, r1, r2, r3);
      r4 = InsertTopHalf(r2, r3);
      r2 = InsertTopHalf(r3, r2);
      LoadStridedSets<Align>(input + 16, 4, r5, r6, r7);
      r3 = InsertTopHalf(r7, r5);
      r8 = Load<Align>(input + 28);
      r5 = InsertTopHalf(r5, r8);
      r8 = InsertTopHalf(r8, r7);
      LoadStridedSets<Align>(input + 32, 4, r9, r10, r11, r12);
      r7 = InsertTopHalf(r11, r12);
      r11 = InsertTopHalf(r12, r11);
      r13 = Load<Align>(input + 48);
      r12 = InsertTopHalf(r9, r0);
      r9 = InsertTopHalf(r13, r9);
      LoadStridedSets<Align>(input + 56, 4, r14, r15);
      r0 = InsertTopHalf(r0, r15);
      r15 = InsertTopHalf(r15, r14);
      r13 = InsertTopHalf(r14, r13);
      r14 = Load<Align>(input + 52);
      StoreStridedSets<Align>(output, 4, r0, r15, r13, r14, r9, r10, r7, r11, r12, r1, r4, r2, r5, r8, r3, r6);
    }
  }
}


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