FORCE_INLINE void ComplexMultiply(auto aR, auto aI, auto bR, auto bI, auto &cR, auto &cI)
{
  auto tR = Mul(aR, bR);
  auto tI = Mul(aI, bR);
  cR = NMAdd(aI, bI, tR);
  cI = MAdd(aR, bI, tI);
}


template <Alignment A, std::floating_point F, typename...V>
FORCE_INLINE void LoadStridedSets(const F *base, size_t stride, V &... vs)
{
  size_t offset = 0;
  if constexpr ((std::same_as<std::remove_reference_t<V>, VecType> && ...))
  {
    ((vs = Load<A>(base + offset), offset += stride), ...);
  }
  else
  {
    ((vs = LoadHalf<A>(base + offset), offset += stride), ...);
  }
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
  if constexpr (UseFMAOptimization)
  {
    aROut = NMAdd(bI, wI, aR);
    aROut = MAdd(bR, wR, aROut);
    aIOut = MAdd(bR, wI, aI);
    aIOut = MAdd(bI, wR, aIOut);
    bR = Mx2Sub(aR, aROut);
    bI = NMx2Add(aI, aIOut);
  }
  else
  {
    aROut = Mul(bI, wI);
    aIOut = Mul(bR, wI);
    bR = NMAdd(bR, wR, aROut);
    bI = MAdd(bI, wR, aIOut);
    aROut = Sub(aR, bR);
    aIOut = Add(aI, bI);
    bR = Add(aR, bR);
    bI = Sub(bI, aI);
  }
}



FORCE_INLINE void DIFComplexRadix2Inv(auto &aR, auto &aI, auto &bR, auto &bI, auto wR, auto wI, auto &aROut, auto &aIOut)
{
  aROut = Add(aR, bR);
  aIOut = Sub(aI, bI);
  bR = Sub(bR, aR); // actually -bR
  bI = Add(aI, bI);

  // Rotate b * conj(w), using aR/aI as temp storage since they're no longer needed
  aR = Mul(bR, wI);
  aI = Mul(bI, wI);
  bR = NMAdd(bR, wR, aI);
  bI = MAdd(bI, wR, aR);
}



FORCE_INLINE void DITRealInputRadix4Pair(auto &a, auto &b, auto &c, auto &d, auto &e, auto &f, auto &g, auto &h)
{
  // Do the radix4 pair, shuffling data around through other named registers as temp space to get things into the correct
  //  location by the end.
  auto t = Sub(b, f);
  b = Add(b, f);
  f = Add(h, d);
  h = Sub(h, d);

  d = Sub(f, b);
  b = Add(f, b);

  f = Add(a, e);
  e = Sub(a, e);

  a = Mul(Add(t, h), InvSqrt2<ElementType>);
  t = Mul(Sub(h, t), InvSqrt2<ElementType>);

  h = Sub(g, c);
  c = Add(g, c);
  g = Sub(e, a);
  e = Add(e, a);
  a = Add(f, c);
  c = Sub(f, c);

  f = Add(t, h);
  h = Sub(t, h);
}



FORCE_INLINE void DIFRealOutputRadix4PairInv(auto &a, auto &b, auto &c, auto &d, auto &e, auto &f, auto &g, auto &h)
{
  // Like the others, this uses a single temp register and uses other temporarily-unused values as additional storage.
  auto t = Sub(e, g);
  e = Add(e, g);
  g = Sub(f, h);
  h = Add(f, h);

  f = Mul(Sub(t, h), InvSqrt2<ElementType>);
  t = Mul(Add(t, h), InvSqrt2<ElementType>);

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


template <typename F>
struct TwoAngles
{
  F w0r;
  F w0i;
  F w1r;
  F w1i;
};

template <typename F>
struct ThreeAngles
{
  F w0r;
  F w0i;
  F w1r;
  F w1i;
  F w2r;
  F w2i;
};

template <std::floating_point F>
struct TwoScalarAnglesX2
{
  TwoScalarAnglesX2(TwoAngles<F> a, TwoAngles<F> b)
    : w0r{a.w0r, b.w0r}
    , w0i{a.w0i, b.w0i}
    , w1r{a.w1r, b.w1r}
    , w1i{a.w1i, b.w1i}
    { }
  std::pair<F, F> w0r;
  std::pair<F, F> w0i;
  std::pair<F, F> w1r;
  std::pair<F, F> w1i;
};



template <std::floating_point F>
struct ThreeScalarAnglesX2
{
  ThreeScalarAnglesX2(ThreeAngles<F> a, ThreeAngles<F> b)
    : w0r{a.w0r, b.w0r}
    , w0i{a.w0i, b.w0i}
    , w1r{a.w1r, b.w1r}
    , w1i{a.w1i, b.w1i}
    , w2r{a.w2r, b.w2r}
    , w2i{a.w2i, b.w2i}
    { }
  std::pair<F, F> w0r;
  std::pair<F, F> w0i;
  std::pair<F, F> w1r;
  std::pair<F, F> w1i;
  std::pair<F, F> w2r;
  std::pair<F, F> w2i;
};


FORCE_INLINE auto GetScalarAngleIndices(size_t angleIndex, size_t angleCount)
{
  auto r1 = angleCount * 2;
  auto r0 = angleCount;
  auto i1 = r1 + angleCount;
  auto i0 = r0 + angleCount / 2;

  return std::tuple{ r0 + angleIndex, i0 + angleIndex, r1 + angleIndex, i1 + angleIndex };
}



template <std::floating_point F>
FORCE_INLINE auto LoadScalarAngleGroup(const F *tableData, size_t angleIndex, size_t angleCount)
{
  auto [r0, i0, r1, i1] = GetScalarAngleIndices(angleIndex, angleCount);
  auto wr1 = tableData[r0];
  auto wi1 = tableData[i0];
  auto wr2 = tableData[r1];
  auto wi2 = tableData[i1];

  if constexpr (UseFMAOptimization)
    { return TwoAngles<F>{wr2, wi2, wr1, wi1}; }
  else
  {
    // Third angle calculated based on the first two.
    return ThreeAngles<F>{wr2, wi2, wr1, wi1, wr1 * wr2 - wi1 * wi2, wr1 * wi2 + wi1 * wr2};
  }
}


template <std::floating_point F>
FORCE_INLINE auto LoadScalarAngleGroupX2(const F *tableData, size_t angleIndexA, size_t angleIndexB, size_t angleCount)
{
  using T = std::conditional_t<UseFMAOptimization, TwoScalarAnglesX2<F>, ThreeScalarAnglesX2<F>>;
  return T{LoadScalarAngleGroup(tableData, angleIndexA, angleCount), LoadScalarAngleGroup(tableData, angleIndexB, angleCount)};
}

template <std::floating_point F>
FORCE_INLINE auto LoadSIMDAngleDouble(const F *tableData, size_t angleIndex, size_t angleCount)
{
  auto [r0, i0, r1, i1] = GetScalarAngleIndices(angleIndex, angleCount);
  auto wr1 = Load<Alignment::Aligned>(&tableData[r0]);
  auto wi1 = Load<Alignment::Aligned>(&tableData[i0]);
  auto wr2 = Load<Alignment::Aligned>(&tableData[r1]);
  auto wi2 = Load<Alignment::Aligned>(&tableData[i1]);
  return TwoAngles{ wr2, wi2, wr1, wi1 };
}


template <std::floating_point F>
FORCE_INLINE auto LoadSIMDAngleGroup(const F *tableData, size_t angleIndex, size_t angleCount)
{
  auto two = LoadSIMDAngleDouble(tableData, angleIndex, angleCount);
  if constexpr (UseFMAOptimization)
    { return two; }
  else
  {
    VecType w2r, w2i;
    ComplexMultiply(two.w0r, two.w0i, two.w1r, two.w1i, w2r, w2i);
    return ThreeAngles{two.w0r, two.w0i, two.w1r, two.w1i, w2r, w2i};
  }
}



// This function uses all 16 registers!
template <typename V, typename Angles>
FORCE_INLINE void DITComplexRadix4(V &aR, V &aI, V &bR, V &bI, V &cR, V &cI, V &dR, V &dI, const Angles &angles)
{
  // Rotate c by w1
  if constexpr (requires (Angles a) { a.w2r; })
  {
    auto t2 = Mul(cR, angles.w1r);
    cR = Mul(cR, angles.w1i);
    auto t1 = NMAdd(cI, angles.w1i, t2);
    cI = MAdd(cI, angles.w1r, cR);

    cR = Sub(aR, t1);
    t1 = Add(aR, t1);
    aR = Add(aI, cI);
    cI = Sub(aI, cI);

    // use aI and t2 as temp registers
    // Rotate b by w0
    aI = Mul(bR, angles.w0i);
    t2 = Mul(bI, angles.w0i);
    bI = MAdd(bI, angles.w0r, aI);
    bR = NMAdd(bR, angles.w0r, t2); // actually -bR

    // now rotate d by w2

    t2 = Mul(dR, angles.w2r);
    dR = Mul(dR, angles.w2i);
    aI = NMAdd(dI, angles.w2i, t2); // this is rotated dR
    dI = MAdd(dI, angles.w2r, dR);

    // Do the last butterflies
    dR = Add(aI, bR); // actually -dR
    bR = Sub(aI, bR);
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
  }
  else
  {
    auto t1 = NMAdd(dI, angles.w1i, MAdd(dR, angles.w1r, bR));
    auto t2 = MAdd(dI, angles.w1r, MAdd(dR, angles.w1i, bI));
    dR = Mx2Sub(bR, t1);
    dI = NMx2Add(bI, t2);

    bR = NMAdd(cI, angles.w1i, MAdd(cR, angles.w1r, aR));
    bI = MAdd(cI, angles.w1r, MAdd(cR, angles.w1i, aI));
    aR = Mx2Sub(aR, bR);
    aI = NMx2Add(aI, bI);

    cR = NMAdd(dR, angles.w0i, MAdd(dI, angles.w0r, aR));
    cI = NMAdd(dR, angles.w0r, NMAdd(dI, angles.w0i, aI));
    dR = Mx2Sub(aR, cR);
    dI = NMx2Add(aI, cI);

    aR = NMAdd(t2, angles.w0i, MAdd(t1, angles.w0r, bR));
    aI = MAdd(t2, angles.w0r, MAdd(t1, angles.w0i, bI));
    bR = Mx2Sub(bR, aR);
    bI = NMx2Add(bI, aI);
  }
};



template <typename VT, typename Angles>
FORCE_INLINE void DIFComplexRadix4Inv(
  VT &aR,
  VT &aI,
  VT &bR,
  VT &bI,
  VT &cR,
  VT &cI,
  VT &dR,
  VT &dI,
  const Angles &angles)
{
  if constexpr (requires (Angles a) { a.w2r; })
  {
    // This is effectively the opposite of the above Radix4.
    auto t1 = Add(dR, cR);
    dR = Sub(dR, cR);
    auto t2 = Sub(dI, cI);
    dI = Add(dI, cI);

    cR = Sub(aR, bR);
    aR = Add(aR, bR);
    cI = Add(aI, bI);
    aI = Sub(aI, bI);

    bR = Sub(dI, cR); // actually -bR
    dI = Add(cR, dI);
    bI = Add(cI, dR);
    dR = Sub(cI, dR);

    cR = Sub(t1, aR); // actually -cR
    aR = Add(aR, t1);
    cI = Sub(aI, t2);
    aI = Add(aI, t2);

    // aR/aI are finalized here do not use them as temp!

    t1 = Mul(bR, angles.w0i);
    t2 = Mul(bI, angles.w0i);
    bR = NMAdd(bR, angles.w0r, t2);
    bI = MAdd(bI, angles.w0r, t1);

    t1 = Mul(dI, angles.w2r);
    t2 = Mul(dR, angles.w2r);
    dR = MAdd(dR, angles.w2i, t1);
    dI = NMAdd(dI, angles.w2i, t2);

    t1 = Mul(cI, angles.w1i);
    t2 = Mul(cR, angles.w1i);
    cR = NMAdd(cR, angles.w1r, t1);
    cI = MAdd(cI, angles.w1r, t2);
  }
  else
  {
    auto t1 = Sub(aR, bR);
    auto t2 = Sub(aI, bI);
    aI = Add(aI, bI);
    aR = Add(aR, bR);

    bI = Add(cI, dI);
    bR = Add(cR, dR);
    cR = Sub(cR, dR);
    cI = Sub(cI, dI);

    dR = MAdd(aI, angles.w0i, Mul(t1, angles.w0r));
    dI = NMAdd(t1, angles.w0i, Mul(aI, angles.w0r));

    aI = Sub(t2, cI);
    cI = Add(t2, cI);
    t2 = Sub(aR, bR);
    aR = Add(aR, bR);

    bR = NMAdd(bI, angles.w0r, NMAdd(cR, angles.w0i, dR));
    bI = MAdd(bI, angles.w0i, NMAdd(cR, angles.w0r, dI));
    cR = Mx2Sub(dR, bR);
    dI = Mx2Sub(dI, bI);

    dR = MAdd(dI, angles.w1i, Mul(cR, angles.w1r));
    dI = NMAdd(cR, angles.w1i, Mul(dI, angles.w1r));

    cR = MAdd(cI, angles.w1i, Mul(t2, angles.w1r));
    cI = NMAdd(t2, angles.w1i, Mul(cI, angles.w1r));
  }
};


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
  auto &w1I)
{
  // For the final real DIT we need a second w1 angle that is just 0.707, -0.707 (but the complex w1 values remain
  //  the same)

  if constexpr (UseFMAOptimization)
  {
    auto t1 = MAdd(e, w0R, NMAdd(f, w0I, a));
    auto t2 = MAdd(h, w0R, MAdd(g, w0I, d));

    e = MAdd(e, w0I, MAdd(f, w0R, b));
    w0I = NMAdd(h, w0I, MAdd(g, w0R, c));

    f = NMx2Add(c, w0I); // Do -g here so that we can leverage it for the 90 degree rotation later

    g = Mx2Sub(a, t1);
    h = NMx2Add(b, e);

    w0R = NMx2Add(d, t2);

    d = ReplaceFirstOf(e, w0I);
    b = ReplaceFirstOf(w0I, e);

    a = NMAdd(t2, w1I, MAdd(b, w1R, t1));
    c = Mx2Sub(t1, a);

    b = MAdd(b, w1I, MAdd(t2, w1R, d));
    d = NMx2Add(d, b);

    w1I = ReplaceFirstOf(w1I, -InvSqrt2<ElementType>);
    w1R = ReplaceFirstOf(w1R, InvSqrt2<ElementType>);

    e = MAdd(w0R, w1R, MAdd(f, w1I, g));
    f = NMAdd(w0R, w1I, MAdd(f, w1R, h));
    g = Mx2Sub(g, e);
    h = NMx2Add(h, f);
  }
  else
  {
    auto t0 = Mul(e, w0I);
    auto t1 = Mul(f, w0I);
    f = MAdd(f, w0R, t0);
    e = NMAdd(e, w0R, t1); // -e

    // gh *= angle0;
    t0 = Mul(g, w0R);
    g = Mul(g, w0I);
    w0I = NMAdd(h, w0I, t0);
    t1 = MAdd(h, w0R, g);

    // g, h are free here
    t0 = Sub(a, e);
    e = Add(a, e);
    w0R = Add(f, b);
    f = Sub(f, b);

    g = Sub(w0I, c); // Do -g here so that we can leverage it for the 90 degree rotation later
    c = Add(w0I, c);
    h = Sub(t1, d);
    d = Add(t1, d);

    // Need to do another swap (this time of b0 and c0, again for the "real" radix's different ordering)
    b = ReplaceFirstOf(w0R, c);
    c = ReplaceFirstOf(c, w0R);

    // t0 is in use, but t1, w0R, w0I are not.

    // cd *= angle1;
    w0I = Mul(d, w1I);
    d = Mul(d, w1R);
    t1 = MAdd(c, w1I, d);
    c = NMAdd(c, w1R, w0I);

    w1R = ReplaceFirstOf(w1R, InvSqrt2<ElementType>);
    w1I = ReplaceFirstOf(w1I, -InvSqrt2<ElementType>);

    a = Sub(t0, c);
    c = Add(t0, c);

    // gh *= {-angle1.imag(), -angle1.real()};
    //  note that this takes advantage of g being stored as -g above
    w0R = Mul(g, w1R);
    g = Mul(g, w1I);
    w0R = NMAdd(h, w1I, w0R);
    h = MAdd(h, w1R, g);

    d = Sub(t1, b);
    b = Add(t1, b);

    g = Sub(e, h);
    e = Add(e, h);
    h = Sub(w0R, f);
    f = Add(w0R, f);
  }
}



FORCE_INLINE void FirstRealDIFRealComplexRadix4Inv(
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
  auto &w1I)
{
  // This one is the same regardless of whether or not we do the FMA optimization

  auto t0 = Sub(e, g);
  e = Add(e, g);
  g = Add(f, h);
  f = Sub(f, h);

  h = Sub(a, c);
  a = Add(a, c);
  c = Add(b, d);
  b = Sub(b, d);

  // t1, d are free

  d = Mul(h, w1R);
  auto t1 = Mul(c, w1R);
  c = MAdd(c, w1I, d);
  d = NMAdd(h, w1I, t1);

  // For the inverse of the final real DIT we need a second w1 angle that is just 0.707, -0.707 (but the complex w1
  //  values remain the same)
  w1R = ReplaceFirstOf(w1R, InvSqrt2<ElementType>);
  w1I = ReplaceFirstOf(w1I, -InvSqrt2<ElementType>);

  t1 = Mul(t0, w1R);
  t0 = Mul(t0, w1I);
  h = NMAdd(g, w1I, t1);
  g = MAdd(g, w1R, t0);

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

  e = Mul(t1, w0I);
  t1 = Mul(t1, w0R);
  h = NMAdd(g, w0I, t1);
  g = MAdd(g, w0R, e);

  t1 = Mul(w1I, w0I);
  t0 = Mul(w1I, w0R);
  e = MAdd(w1R, w0R, t1);
  f = NMAdd(w0I, w1R, t0);
}


template <Alignment A, uint32_t VS = VecSize>
auto LoadForTangle(const ElementType *input)
{
  if constexpr (VS == 8)
    { return LoadHalf<A>(input); }
  else
    { return Load<A>(input); }
}

using TangleVecType = decltype(LoadForTangle<Alignment::Aligned>(std::declval<const ElementType *>()));
constexpr auto TangleVecSize = std::same_as<TangleVecType, VecType> ? VecSize : (VecSize / 2);

template <Alignment A, TangleType Type, uint32_t VV = TangleVecSize, typename VT = TangleVecType>
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
      VT t0, t1, t2, t3, l1, l2, l3;
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
      VT t0, t1, t2, t3, t4, t5, t6, t7, l0, l1, l2, l3, l4, l5, l6, l7;
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
    VT prevTopA, prevTopB;
    LoadStridedSets<A>(input, 4, prevTopA, prevTopB);
    auto prevLowA = LoadForTangle<A>(input + subLength - 8);

    // This first section gets us into the condition where the loop expects
    {
      VT r03, r04, r05, r06, r07;
      // Load the last value in the array as r03 (it's `cb`)
      //  that write to the beginning of the output.
      r03 = LoadForTangle<A>(input + subLength - 4);
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
      r05 = LoadForTangle<A>(input + 8);

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
      prevTopB = LoadForTangle<A>(input + 12);
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
      VT r03, r04, r05, r06, r07, r08;
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
      prevLowA = LoadForTangle<A>(input + subLength - i - 16);
      r07 = LoadForTangle<A>(input + subLength - i - 12);

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
      VT finalA, finalB;
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



template <Alignment Align, uint32_t VV = TangleVecSize, typename VT = TangleVecType>
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
    VT Aa, Bb;
    LoadStridedSets<Align>(input, 4, Aa, Bb);
    StoreStridedSets<Align>(output, 4, Aa, Bb);
  }
}



template <Alignment Align, TangleType Type, uint32_t VV = TangleVecSize, typename VT = TangleVecType>
void TangleFinal4Pairs(const ElementType *input, ElementType *output)
{
  if constexpr (Type == TangleType::ForwardLargest)
  {
    //  AbBa -> AaBb
    if constexpr (VV == 2)
    {
      VT A, b, B, a;
      LoadStridedSets<Align>(input, 2, A, b, B, a);
      StoreStridedSets<Align>(output, 2, A, a, B, b);
    }
    else
    {
      static_assert(VV == 4);
      VT Ab, Ba;
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
      VT A, a, B, b;
      LoadStridedSets<Align>(input, 2, A, a, B, b);
      StoreStridedSets<Align>(output, 2, A, b, B, a);
    }
    else
    {
      static_assert(VV == 4);
      VT Aa, Bb;
      LoadStridedSets<Align>(input, 4, Aa, Bb);
      auto Ab = InsertTopHalf(Aa, Bb);
      auto Ba = InsertTopHalf(Bb, Aa);
      StoreStridedSets<Align>(output, 4, Ab, Ba);
    }
  }
}



template <Alignment Align, TangleType Type, uint32_t VV = TangleVecSize, typename VT = TangleVecType>
void Tangle8Pairs(const ElementType *input, ElementType *output)
{
  if constexpr (Type == TangleType::ForwardInner)
  {
    // AcDdCaBb -> AaBbCcDd
    if constexpr (VV == 2)
    {
      VT A, a, B, b, C, c, D, d;
      LoadStridedSets<Align>(input, 2, A, c, D, d, C, a, B, b);
      StoreStridedSets<Align>(output, 2, A, a, B, b, C, c, D, d);
    }
    else
    {
      static_assert(VV == 4);
      VT Ac, Dd, Ca, Bb;
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
      VT A, a, B, b, C, c, D, d;
      LoadStridedSets<Align>(input, 2, A, d, D, c, C, a, B, b);
      StoreStridedSets<Align>(output, 2, A, a, B, b, C, c, D, d);
    }
    else
    {
      static_assert(VV == 4);
      VT Ad, Dc, Ca, Bb;
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
      VT A, a, B, b, C, c, D, d;
      LoadStridedSets<Align>(input, 2, A, a, B, b, C, c, D, d);
      StoreStridedSets<Align>(output, 2, A, c, D, d, C, a, B, b);
    }
    else
    {
      static_assert(VV == 4);
      VT Aa, Bb, Cc, Dd;
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
      VT A, a, B, b, C, c, D, d;
      LoadStridedSets<Align>(input, 2, A, a, B, b, C, c, D, d);
      StoreStridedSets<Align>(output, 2, A, d, D, c, C, a, B, b);
    }
    else
    {
      static_assert(VV == 4);
      VT Aa, Bb, Cc, Dd;
      LoadStridedSets<Align>(input, 4, Aa, Bb, Cc, Dd);
      auto Ad = InsertTopHalf(Aa, Dd);
      auto Dc = InsertTopHalf(Dd, Cc);
      auto Ca = InsertTopHalf(Cc, Aa);
      StoreStridedSets<Align>(output, 4, Ad, Dc, Ca, Bb);
    }
  }
}



template <Alignment Align, TangleType Type, uint32_t VV = TangleVecSize, typename VT = TangleVecType>
void Tangle16Pairs(const ElementType *input, ElementType *output)
{
  if constexpr (Type == TangleType::ForwardInner)
  {
    // AgHhGeFfEaBbCcDd -> AaBbCcDdEeFfGgHh
    if constexpr (VV == 2)
    {
      VT A, a, B, b, C, c, D, d, E, e, F, f, G, g, H, h;
      LoadStridedSets<Align>(input, 2, A, g, H, h, G, e, F, f, E, a, B, b, C, c, D, d);
      StoreStridedSets<Align>(output, 2, A, a, B, b, C, c, D, d, E, e, F, f, G, g, H, h);
    }
    else
    {
      static_assert(VV == 4);
      VT r00, r01, r02, r03, r04, r05, r06, r07;
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
      VT A, a, B, b, C, c, D, d, E, e, F, f, G, g, H, h;
      LoadStridedSets<Align>(input, 2, A, h, H, g, G, e, F, f, E, a, B, b, C, d, D, c);
      StoreStridedSets<Align>(output, 2, A, a, B, b, C, c, D, d, E, e, F, f, G, g, H, h);
    }
    else
    {
      static_assert(VV == 4);
      VT r00, r01, r02, r03, r04, r05, r06, r07, r08;
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
      VT A, a, B, b, C, c, D, d, E, e, F, f, G, g, H, h;
      LoadStridedSets<Align>(input, 2, A, a, B, b, C, c, D, d, E, e, F, f, G, g, H, h);
      StoreStridedSets<Align>(output, 2, A, g, H, h, G, e, F, f, E, a, B, b, C, c, D, d);
    }
    else
    {
      VT r00, r01, r02, r03, r04, r05, r06, r07, r08;
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
      VT A, a, B, b, C, c, D, d, E, e, F, f, G, g, H, h;
      LoadStridedSets<Align>(input, 2, A, a, B, b, C, c, D, d, E, e, F, f, G, g, H, h);
      StoreStridedSets<Align>(output, 2, A, h, H, g, G, e, F, f, E, a, B, b, C, d, D, c);
    }
    else
    {
      VT r00, r01, r02, r03, r04, r05, r06, r07, r08;
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
template <Alignment Align, TangleType Type, uint32_t VV = TangleVecSize, typename VT = TangleVecType>
void Tangle32Pairs(const ElementType *input, ElementType *output)
{
  if constexpr (Type == TangleType::ForwardInner)
  {
    // AoPpOmNnMiJjKkLlIaBbCcDdEgHhGeFf -> AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPp
    if constexpr (VV == 2)
    {
      VT r00, r01, r02, r03, r04, r05, r06, r07, r08, r09, r10, r11, r12, r13, r14, r15;
      r00 = Load<Align>(input + 0);
      Store<Align>(output + 0, r00);
      r00 = Load<Align>(input + 2);
      r01 = Load<Align>(input + 34);
      Store<Align>(output + 2, r01);
      r01 = Load<Align>(input + 4);
      r02 = Load<Align>(input + 36);
      Store<Align>(output + 4, r02);
      r02 = Load<Align>(input + 6);
      r03 = Load<Align>(input + 38);
      Store<Align>(output + 6, r03);
      r03 = Load<Align>(input + 8);
      r04 = Load<Align>(input + 40);
      Store<Align>(output + 8, r04);
      r04 = Load<Align>(input + 10);
      r05 = Load<Align>(input + 42);
      Store<Align>(output + 10, r05);
      r05 = Load<Align>(input + 12);
      r06 = Load<Align>(input + 44);
      Store<Align>(output + 12, r06);
      r06 = Load<Align>(input + 14);
      r07 = Load<Align>(input + 46);
      Store<Align>(output + 14, r07);
      r07 = Load<Align>(input + 16);
      r08 = Load<Align>(input + 48);
      Store<Align>(output + 16, r08);
      r08 = Load<Align>(input + 18);
      r09 = Load<Align>(input + 58);
      Store<Align>(output + 18, r09);
      r09 = Load<Align>(input + 20);
      r10 = Load<Align>(input + 60);
      Store<Align>(output + 20, r10);
      r10 = Load<Align>(input + 22);
      r11 = Load<Align>(input + 62);
      Store<Align>(output + 22, r11);
      r11 = Load<Align>(input + 24);
      r12 = Load<Align>(input + 56);
      Store<Align>(output + 24, r12);
      r12 = Load<Align>(input + 26);
      r13 = Load<Align>(input + 50);
      Store<Align>(output + 26, r13);
      r13 = Load<Align>(input + 28);
      r14 = Load<Align>(input + 52);
      Store<Align>(output + 28, r14);
      r14 = Load<Align>(input + 30);
      r15 = Load<Align>(input + 54);
      Store<Align>(output + 30, r15);
      r15 = Load<Align>(input + 32);
      StoreStridedSets<Align>(output + 32, 2, r15, r08, r09, r10, r11, r12, r13, r14, r07, r04, r05, r06, r03, r00, r01, r02);
    }
    else
    {
      static_assert(VV == 4);

      // Conveniently with the loads and shuffles ordered as such, we fit within the x64 register space.
      VT r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
      LoadStridedSets<Align>(input, 4, r0, r1, r2, r3, r4);
      r5 = InsertTopHalf(r2, r0);
      r2 = InsertTopHalf(r4, r2);
      LoadStridedSets<Align>(input + 20, 4, r6, r7, r8, r9);
      r0 = InsertTopHalf(r0, r9);
      r4 = InsertTopHalf(r9, r4);
      r9 = LoadForTangle<Align>(input + 36);
      LoadStridedSets<Align>(input + 40, 4, r10, r11, r12, r13, r14);
      r15 = InsertTopHalf(r14, r12);
      r12 = InsertTopHalf(r12, r14);
      r14 = LoadForTangle<Align>(input + 60);
      StoreStridedSets<Align>(output, 4, r0, r9, r10, r11, r12, r14, r15, r13, r4, r6, r7, r8, r2, r3, r5, r1);
    }
  }
  else if constexpr (Type == TangleType::ForwardLargest)
  {
    // ApPoOmNnMiJjKlLkIaBbCdDcEhHgGeFf -> AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPp
    //  (note that pPo, lLk, dDc, and hHg are reversed from the above)
    if constexpr (VV == 2)
    {
      VT r00, r01, r02, r03, r04, r05, r06, r07, r08, r09, r10, r11, r12, r13, r14, r15;
      r00 = Load<Align>(input + 0);
      Store<Align>(output + 0, r00);
      r00 = Load<Align>(input + 2);
      r01 = Load<Align>(input + 34);
      Store<Align>(output + 2, r01);
      r01 = Load<Align>(input + 4);
      r02 = Load<Align>(input + 36);
      Store<Align>(output + 4, r02);
      r02 = Load<Align>(input + 6);
      r03 = Load<Align>(input + 38);
      Store<Align>(output + 6, r03);
      r03 = Load<Align>(input + 8);
      r04 = Load<Align>(input + 40);
      Store<Align>(output + 8, r04);
      r04 = Load<Align>(input + 10);
      r05 = Load<Align>(input + 46);
      Store<Align>(output + 10, r05);
      r05 = Load<Align>(input + 12);
      r06 = Load<Align>(input + 44);
      Store<Align>(output + 12, r06);
      r06 = Load<Align>(input + 14);
      r07 = Load<Align>(input + 42);
      Store<Align>(output + 14, r07);
      r07 = Load<Align>(input + 16);
      r08 = Load<Align>(input + 48);
      Store<Align>(output + 16, r08);
      r08 = Load<Align>(input + 18);
      r09 = Load<Align>(input + 58);
      Store<Align>(output + 18, r09);
      r09 = Load<Align>(input + 20);
      r10 = Load<Align>(input + 60);
      Store<Align>(output + 20, r10);
      r10 = Load<Align>(input + 22);
      r11 = Load<Align>(input + 62);
      Store<Align>(output + 22, r11);
      r11 = Load<Align>(input + 24);
      r12 = Load<Align>(input + 56);
      Store<Align>(output + 24, r12);
      r12 = Load<Align>(input + 26);
      r13 = Load<Align>(input + 54);
      Store<Align>(output + 26, r13);
      r13 = Load<Align>(input + 28);
      r14 = Load<Align>(input + 52);
      Store<Align>(output + 28, r14);
      r14 = Load<Align>(input + 30);
      r15 = Load<Align>(input + 50);
      Store<Align>(output + 30, r15);
      r15 = Load<Align>(input + 32);
      StoreStridedSets<Align>(output + 32, 2, r15, r08, r09, r10, r11, r14, r13, r12, r07, r04, r05, r06, r03, r02, r01, r00);
    }
    else
    {
      static_assert(VV == 4);
      VT r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
      LoadStridedSets<Align>(input, 4, r0, r1, r2, r3, r4);
      r5 = InsertTopHalf(r2, r1);
      r1 = InsertTopHalf(r1, r0);
      r2 = InsertTopHalf(r4, r2);
      LoadStridedSets<Align>(input + 20, 4, r6, r7, r8);
      r9 = InsertTopHalf(r7, r8);
      r7 = InsertTopHalf(r8, r7);
      r8 = LoadForTangle<Align>(input + 32);
      r0 = InsertTopHalf(r0, r8);
      r4 = InsertTopHalf(r8, r4);
      LoadStridedSets<Align>(input + 36, 4, r10, r11, r12);
      r8 = InsertTopHalf(r11, r12);
      r11 = InsertTopHalf(r12, r11);
      LoadStridedSets<Align>(input + 48, 4, r13, r14, r15);
      r12 = InsertTopHalf(r14, r13);
      r14 = InsertTopHalf(r15, r14);
      r13 = InsertTopHalf(r13, r15);
      r15 = LoadForTangle<Align>(input + 60);
      StoreStridedSets<Align>(output, 4, r0, r10, r8, r11, r13, r15, r14, r12, r4, r6, r9, r7, r2, r3, r5, r1);
    }
  }
  else if constexpr (Type == TangleType::InverseInner)
  {
    // AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPp -> AoPpOmNnMiJjKkLlIaBbCcDdEgHhGeFf
    if constexpr (VV == 2)
    {
      VT r00, r01, r02, r03, r04, r05, r06, r07, r08, r09, r10, r11, r12, r13, r14, r15;
      r00 = Load<Align>(input + 0);
      Store<Align>(output + 0, r00);
      r00 = Load<Align>(input + 2);
      r01 = Load<Align>(input + 58);
      Store<Align>(output + 2, r01);
      r01 = Load<Align>(input + 4);
      r15 = Load<Align>(input + 60);
      Store<Align>(output + 4, r15);
      r02 = Load<Align>(input + 6);
      r15 = Load<Align>(input + 62);
      Store<Align>(output + 6, r15);
      r03 = Load<Align>(input + 8);
      r15 = Load<Align>(input + 56);
      Store<Align>(output + 8, r15);
      r04 = Load<Align>(input + 10);
      r15 = Load<Align>(input + 50);
      Store<Align>(output + 10, r15);
      r05 = Load<Align>(input + 12);
      r15 = Load<Align>(input + 52);
      Store<Align>(output + 12, r15);
      r06 = Load<Align>(input + 14);
      r15 = Load<Align>(input + 54);
      Store<Align>(output + 14, r15);
      r07 = Load<Align>(input + 16);
      r15 = Load<Align>(input + 48);
      Store<Align>(output + 16, r15);
      r08 = Load<Align>(input + 18);
      r15 = Load<Align>(input + 34);
      Store<Align>(output + 18, r15);
      r09 = Load<Align>(input + 20);
      r15 = Load<Align>(input + 36);
      Store<Align>(output + 20, r15);
      r10 = Load<Align>(input + 22);
      r15 = Load<Align>(input + 38);
      Store<Align>(output + 22, r15);
      r11 = Load<Align>(input + 24);
      r15 = Load<Align>(input + 40);
      Store<Align>(output + 24, r15);
      r12 = Load<Align>(input + 26);
      r15 = Load<Align>(input + 42);
      Store<Align>(output + 26, r15);
      r13 = Load<Align>(input + 28);
      r15 = Load<Align>(input + 44);
      Store<Align>(output + 28, r15);
      r14 = Load<Align>(input + 30);
      r15 = Load<Align>(input + 46);
      Store<Align>(output + 30, r15);
      r15 = Load<Align>(input + 32);
      StoreStridedSets<Align>(output + 32, 2, r15, r00, r01, r02, r03, r04, r05, r06, r07, r12, r13, r14, r11, r08, r09, r10);
    }
    else
    {
      static_assert(VV == 4);
      VT r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
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
      r15 = LoadForTangle<Align>(input + 60);
      StoreStridedSets<Align>(output, 4, r0, r15, r13, r14, r9, r10, r11, r12, r6, r1, r2, r3, r4, r8, r7, r5);
    }
  }
  else if constexpr (Type == TangleType::InverseLargest)
  {
    // AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPp -> ApPoOmNnMiJjKlLkIaBbCdDcEhHgGeFf
    if constexpr (VV == 2)
    {
      VT r00, r01, r02, r03, r04, r05, r06, r07, r08, r09, r10, r11, r12, r13, r14, r15;
      r15 = Load<Align>(input + 0);
      Store<Align>(output + 0, r15);
      r00 = Load<Align>(input + 2);
      r15 = Load<Align>(input + 62);
      Store<Align>(output + 2, r15);
      r01 = Load<Align>(input + 4);
      r15 = Load<Align>(input + 60);
      Store<Align>(output + 4, r15);
      r02 = Load<Align>(input + 6);
      r15 = Load<Align>(input + 58);
      Store<Align>(output + 6, r15);
      r03 = Load<Align>(input + 8);
      r15 = Load<Align>(input + 56);
      Store<Align>(output + 8, r15);
      r04 = Load<Align>(input + 10);
      r15 = Load<Align>(input + 50);
      Store<Align>(output + 10, r15);
      r05 = Load<Align>(input + 12);
      r15 = Load<Align>(input + 52);
      Store<Align>(output + 12, r15);
      r06 = Load<Align>(input + 14);
      r15 = Load<Align>(input + 54);
      Store<Align>(output + 14, r15);
      r07 = Load<Align>(input + 16);
      r15 = Load<Align>(input + 48);
      Store<Align>(output + 16, r15);
      r08 = Load<Align>(input + 18);
      r15 = Load<Align>(input + 34);
      Store<Align>(output + 18, r15);
      r09 = Load<Align>(input + 20);
      r15 = Load<Align>(input + 36);
      Store<Align>(output + 20, r15);
      r10 = Load<Align>(input + 22);
      r15 = Load<Align>(input + 38);
      Store<Align>(output + 22, r15);
      r11 = Load<Align>(input + 24);
      r15 = Load<Align>(input + 40);
      Store<Align>(output + 24, r15);
      r12 = Load<Align>(input + 26);
      r15 = Load<Align>(input + 46);
      Store<Align>(output + 26, r15);
      r13 = Load<Align>(input + 28);
      r15 = Load<Align>(input + 44);
      Store<Align>(output + 28, r15);
      r14 = Load<Align>(input + 30);
      r15 = Load<Align>(input + 42);
      Store<Align>(output + 30, r15);
      r15 = Load<Align>(input + 32);
      StoreStridedSets<Align>(output + 32, 2, r15, r00, r01, r02, r03, r06, r05, r04, r07, r14, r13, r12, r11, r08, r09, r10);
    }
    else
    {
      static_assert(VV == 4);
      VT r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
      LoadStridedSets<Align>(input, 4, r0, r1, r2, r3);
      r4 = InsertTopHalf(r2, r3);
      r2 = InsertTopHalf(r3, r2);
      LoadStridedSets<Align>(input + 16, 4, r5, r6, r7);
      r3 = InsertTopHalf(r7, r5);
      r8 = LoadForTangle<Align>(input + 28);
      r5 = InsertTopHalf(r5, r8);
      r8 = InsertTopHalf(r8, r7);
      LoadStridedSets<Align>(input + 32, 4, r9, r10, r11, r12);
      r7 = InsertTopHalf(r11, r12);
      r11 = InsertTopHalf(r12, r11);
      r13 = LoadForTangle<Align>(input + 48);
      r12 = InsertTopHalf(r9, r0);
      r9 = InsertTopHalf(r13, r9);
      LoadStridedSets<Align>(input + 56, 4, r14, r15);
      r0 = InsertTopHalf(r0, r15);
      r15 = InsertTopHalf(r15, r14);
      r13 = InsertTopHalf(r14, r13);
      r14 = LoadForTangle<Align>(input + 52);
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