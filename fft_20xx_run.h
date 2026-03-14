// Load a set of w1/w2 radix-4 angles
template <std::floating_point F>
static auto LoadAnglesRadix4(const F *tableData, size_t angleIndex, size_t angleCount) 
  -> auto
{ 
  auto i0 = angleCount * 2;
  auto r0 = i0 - angleCount / 2;
  auto i1 = angleCount * 4;
  auto r1 = i1 - angleCount;

  // Special handling for 8-angle indexing, as they're arranged differently for alignment reasons.
  //  The default math would index the "angleCount == 8" angles from offset 12, but they actually start at 8.
  //  This masks off the bottom bits of 12 to turn it into an 8.
  r0 &= ~7_z;

  auto wr1 = LoadA(tableData + r0 + angleIndex);
  auto wi1 = LoadA(tableData + i0 + angleIndex);
  auto wr2 = LoadA(tableData + r1 + angleIndex);
  auto wi2 = LoadA(tableData + i1 + angleIndex);

  if constexpr (UseFMAOptimization)
    { return std::tuple{ wr1, wi1, wr2, wi2 }; } // FMA optimization doesn't need 3 angle values so we can return now.
  else
  {
    // w3 can be calculated via a complex multiply of w1 and w2.
    return std::tuple{ wr1, wi1, wr2, wi2, NMAdd(wi1, wi2, Mul(wr1, wr2)), MAdd(wr1, wi2, Mul(wi1, wr2)) };
  }
}

// Load a standard radix-2 angle set
template <std::floating_point F>
static auto LoadAngleRadix2(const F *tableData, size_t angleIndex, size_t angleCount) -> Pair<Vec<F>, Vec<F>>
{
  auto w1p = tableData + angleCount * 3 / 2 + angleIndex;
  return { LoadA(w1p), LoadA(w1p + angleCount / 2) };
}


// Load a radix-2 angle from the four-angle area of the table
template <std::floating_point F>
static auto LoadAngleRadix2FromCount4(const F *ptr, size_t angleIndex) -> Pair<Vec<F>, Vec<F>>
  { return { Load4(ptr + angleIndex), Load4(ptr + 4 + angleIndex) }; }

// Load radix-4 angles from the four-angle area of the table
template <std::floating_point F>
static auto LoadAnglesRadix4FromCount4(const F *ptr, size_t angleIndex)
{
  auto wr1 = Load4(ptr + angleIndex);
  auto wi1 = Load4(ptr + 4 + angleIndex);
  auto wr2 = Load4(ptr + 8 + angleIndex);
  auto wi2 = Load4(ptr + 16 + angleIndex);
  
  if constexpr (UseFMAOptimization)
    { return std::tuple{ wr1, wi1, wr2, wi2 }; }
  else
    { return std::tuple{ wr1, wi1, wr2, wi2, NMAdd(wi1, wi2, Mul(wr1, wr2)), MAdd(wr1, wi2, Mul(wi1, wr2)) }; }
}

// We may need to load half-vector (i.e. 4-element instead of 8) angles from the four-angle area of the table.
template <typename F>
static auto LoadAnglesRadix4FromCount4Half(const F *ptr, size_t angleIndex)
{ 
  static_assert(VecSize<Vec<F>> == 8);

  auto wr1 = LoadHalf(ptr + angleIndex);
  auto wi1 = LoadHalf(ptr + 4 + angleIndex);
  auto wr2 = LoadHalf(ptr + 8 + angleIndex);
  auto wi2 = LoadHalf(ptr + 16 + angleIndex);

  if constexpr (UseFMAOptimization)
    { return std::tuple{ wr1, wi1, wr2, wi2 }; }
  else
    { return std::tuple{ wr1, wi1, wr2, wi2, NMAdd(wi1, wi2, Mul(wr1, wr2)), MAdd(wr1, wi2, Mul(wi1, wr2)) }; }
}


// Do a standard Radix-2 butterfly
template <Dir D, typename V>
void Radix2Butterfly(V &r0, V &i0, V &r1, V &i1, V wr, V wi)
{
  if constexpr (!UseFMAOptimization)
  {
    // Without native FMA instructions the best play is to pre-multiply r1 and i1 per standard DIT.
    V rr1, ii1;
    if constexpr (D == Dir::Forward)
    {
      rr1 = NMAdd(i1, wi, Mul(r1, wr));
      ii1 = MAdd(i1, wr, Mul(r1, wi));
    }
    else
    {
      rr1 = MAdd(r1, wr, Mul(i1, wi));
      ii1 = NMAdd(r1, wi, Mul(i1, wr));
    }

    r1 = Sub(r0, rr1);
    i1 = Sub(i0, ii1);
    r0 = Add(r0, rr1);
    i0 = Add(i0, ii1);
  }
  else
  {
    // With FMA instructions we can get a little fancier.
    V tr0, ti0;
    if constexpr (D == Dir::Forward)
    {
      // Rather than premultiply r1/i1, we can do the full calculation of r0/i0 in a pair of FMA instructions each.
      tr0 = FMAdd(r1, wr, FNMAdd(i1, wi, r0)); 
      ti0 = FMAdd(r1, wi, FMAdd(i1, wr, i0));
    }
    else
    {
      tr0 = FMAdd(i1, wi, FMAdd(r1, wr, r0));
      ti0 = FMAdd(i1, wr, FNMAdd(r1, wi, i0));
    }

    // Algebraically, x0' = (x0 + w*x1) and x1' = (x0 - w*x1). We have x0 and x0' and it turns out with some
    //  substitution, this becomes x1' = (2*x0 - x0'), which can be done in another pair of FMA instructions, which
    //  ends up being 6 instructions total, which is less work than the above version (and more accurate!)
    const auto two = Set1<V>(2);
    r1 = FMSub(two, r0, tr0);
    i1 = FMSub(two, i0, ti0);
    r0 = tr0;
    i0 = ti0;
  }
}

// Special-case Radix 4 butterfly in which there are no rotations (besides the 90 degree rotation which doesn't require
//  a multiply).
template <Dir D, typename V>
void Radix4ButterflyNoRotation(V &r0, V &i0, V &r1, V &i1, V &r2,  V &i2, V &r3, V &i3)
{
  // Do this effectively as a set of radix 2 steps.
  auto tr0 = Add(r0, r1);
  auto ti0 = Add(i0, i1);
  auto tr1 = Sub(r0, r1);
  auto ti1 = Sub(i0, i1);
  auto tr2 = Add(r2, r3);
  auto ti2 = Add(i2, i3);
  auto tr3 = Sub(r2, r3);
  auto ti3 = Sub(i2, i3);

  // The second radix 2 step (tr/i3 is rotated 90 degrees, and in a different direction depending on whether this is a
  //  forward or reverse FFT)
  r0 = Add(tr0, tr2);
  i0 = Add(ti0, ti2);
  r2 = Sub(tr0, tr2);
  i2 = Sub(ti0, ti2);
  if constexpr (D == Dir::Forward)
  {
    r1 = Add(tr1, ti3);
    i1 = Sub(ti1, tr3);
    r3 = Sub(tr1, ti3);
    i3 = Add(ti1, tr3);
  }
  else
  {
    r1 = Sub(tr1, ti3);
    i1 = Add(ti1, tr3);
    r3 = Add(tr1, ti3);
    i3 = Sub(ti1, tr3);
  }
}


// Standard Radix-4 butterly (without any FMA support)
template <Dir D, typename V> requires (!UseFMAOptimization)
void Radix4Butterfly(V &r0, V &i0, V &r1, V &i1, V &r2, V &i2, V &r3, V &i3, std::tuple<V, V, V, V, V, V> ws)
{
  auto [wr1, wi1, wr2, wi2, wr3, wi3] = ws;

  // Do the initial DIT pre-rotate of the inputs (x0 never rotates as its angle is always 0)
  if constexpr (D == Dir::Forward)
  {
    auto t = NMAdd(i1, wi1, Mul(r1, wr1));
    i1 = MAdd(i1, wr1, Mul(r1, wi1));
    r1 = t;

    t = NMAdd(i2, wi2, Mul(r2, wr2));
    i2 = MAdd(i2, wr2, Mul(r2, wi2));
    r2 = t;
    t = NMAdd(i3, wi3, Mul(r3, wr3));
    i3 = MAdd(i3, wr3, Mul(r3, wi3));
    r3 = t;
  }
  else
  {
    auto t = MAdd(r1, wr1, Mul(i1, wi1));
    i1 = NMAdd(r1, wi1, Mul(i1, wr1));
    r1 = t;
    t = MAdd(r2, wr2, Mul(i2, wi2));
    i2 = NMAdd(r2, wi2, Mul(i2, wr2));
    r2 = t;
    t = MAdd(r3, wr3, Mul(i3, wi3));
    i3 = NMAdd(r3, wi3, Mul(i3, wr3));
    r3 = t;
  }
    
  auto tr0 = Add(r0, r1);
  auto ti0 = Add(i0, i1);
  auto tr1 = Sub(r0, r1);
  auto ti1 = Sub(i0, i1);

  auto tr2 = Add(r2, r3);
  auto ti2 = Add(i2, i3);
  auto tr3 = Sub(r2, r3);
  auto ti3 = Sub(i2, i3);

  r0 = Add(tr0, tr2);
  i0 = Add(ti0, ti2);
  r2 = Sub(tr0, tr2);
  i2 = Sub(ti0, ti2);

  // x3 rotates a different direction depending on FFT vs IFFT
  if constexpr (D == Dir::Forward)
  {
    r1 = Add(tr1, ti3);
    i1 = Sub(ti1, tr3);
    r3 = Sub(tr1, ti3);
    i3 = Add(ti1, tr3);
  }
  else
  {
    r1 = Sub(tr1, ti3);
    i1 = Add(ti1, tr3);
    r3 = Add(tr1, ti3);
    i3 = Sub(ti1, tr3);
  }
}

// Do a radix-4 butterfly with FMA support. This is not standard radix-4 but is instead two radix-2 steps stacked on
//  top of each other in a trenchcoat. As such, there is no need for a third angle (w3).
template <Dir D, typename V> requires (UseFMAOptimization)
void Radix4Butterfly(V &r0, V &i0, V &r1, V &i1, V &r2, V &i2, V &r3, V &i3, std::tuple<V, V, V, V> ws)
{
  const auto [wr1, wi1, wr2, wi2] = ws;
  const auto two = Set1<V>(2);

  // Like with the radix-2 butterfly, we can get fancy here by doing each radix 2 step as 6 FMA instructions.
  V tr0, ti0;
  if constexpr (D == Dir::Forward)
  {
    tr0 = FMAdd(r1, wr1, FNMAdd(i1, wi1, r0));
    ti0 = FMAdd(r1, wi1, FMAdd(i1, wr1, i0));
  }
  else
  {
    tr0 = FMAdd(i1, wi1, FMAdd(r1, wr1, r0));
    ti0 = FMAdd(i1, wr1, FNMAdd(r1, wi1, i0));
  }
    
  auto tr1 = FMSub(two, r0, tr0);
  auto ti1 = FMSub(two, i0, ti0);

  V tr2, ti2;
  if constexpr (D == Dir::Forward)
  {
    tr2 = FMAdd(r3, wr1, FNMAdd(i3, wi1, r2));
    ti2 = FMAdd(r3, wi1, FMAdd(i3, wr1, i2));
  }
  else
  {
    tr2 = FMAdd(i3, wi1, FMAdd(r3, wr1, r2));
    ti2 = FMAdd(i3, wr1, FNMAdd(r3, wi1, i2));
  }

  auto tr3 = FMSub(two, r2, tr2);
  auto ti3 = FMSub(two, i2, ti2);

  // Now start the second radix-2 step. As is standard in a radix-4 scenario, x3 has a 90 degree rotate in it.
  if constexpr (D == Dir::Forward)
  {
    r0 = FMAdd(tr2, wr2, FNMAdd(ti2, wi2, tr0));
    i0 = FMAdd(tr2, wi2, FMAdd(ti2, wr2, ti0));
  }
  else
  {
    r0 = FMAdd(ti2, wi2, FMAdd(tr2, wr2, tr0));
    i0 = FMAdd(ti2, wr2, FNMAdd(tr2, wi2, ti0));
  }

  r2 = FMSub(two, tr0, r0);
  i2 = FMSub(two, ti0, i0);

  if constexpr (D == Dir::Forward)
  {
    r1 = FMAdd(tr3, wi2, FMAdd(ti3, wr2, tr1));
    i3 = FMAdd(tr3, wr2, FNMAdd(ti3, wi2, ti1));
    r3 = FMSub(two, tr1, r1);
    i1 = FMSub(two, ti1, i3);
  }
  else
  {
    r3 = FMAdd(ti3, wr2, FNMAdd(tr3, wi2, tr1));
    i1 = FMAdd(ti3, wi2, FMAdd(tr3, wr2, ti1));
    r1 = FMSub(two, tr1, r3);
    i3 = FMSub(two, ti1, i1);
  }
}


// The first radix4 has a special case for a 4-element vector for the AVX case where we have an 8-element float vector
//  but the 16-element FFT doesn't work with that size.
enum class ForceSize : bool
{
  No,
  Four,
};


// Do the first batch of Radix4 steps. These need to:
//  - Load and deinterleave the real/imaginary values for N (minimum 4) consecutive sets of butterflies spaced 1/4th
//    of the FFT length apart
//  - Do the IFFT 1/N scale if applicable
//  - Do those butterflies
//  - Do effectively a 4x4 matrix transpose of them (which covers the first pair of bits that need reversing) and store
//    that result (the transpose and store are done in one because NEON has some fancy load/store instructions that
//    make it cleaner)
template <Dir D, ForceSize Force = ForceSize::No, std::floating_point F>
FORCE_INLINE void DoFirstRadix4s(const F *input, F *output, size_t radix4Count)
{
  using V = Vec<F>;
  constexpr auto N = (Force == ForceSize::Four) ? 4u : VecSize<V>;

  if constexpr (N > 4u && Force == ForceSize::No)
  {
    if (radix4Count == 4u)
      { return DoFirstRadix4s<D, ForceSize::Four>(input, output, radix4Count); }
  }

  [[maybe_unused]] F inverse;
  if constexpr (D == Dir::Inverse)
    { inverse = F(0.25) / F(radix4Count); }

  constexpr auto Step = std::max(4u, N);
  for (auto i = 0_z; i < radix4Count; i += Step)
  {
    auto ix0 = i;
    auto ix2 = ix0 + radix4Count; // i2/i1 intentionally out of order here
    auto ix1 = ix2 + radix4Count;
    auto ix3 = ix1 + radix4Count;

    if constexpr (N >= 4)
    {
      // Load the 4 sets of values (either as halves or as full values depending on whether we've forced the size)
      auto [r0, i0, r1, i1, r2, i2, r3, i3] = 
        [&]
        {
          if constexpr (Force == ForceSize::Four)
          {
            static_assert(N == 4);
            auto [rr0, ii0] = LoadDeinterleavedHalf(input + ix0 * 2);
            auto [rr1, ii1] = LoadDeinterleavedHalf(input + ix1 * 2);
            auto [rr2, ii2] = LoadDeinterleavedHalf(input + ix2 * 2);
            auto [rr3, ii3] = LoadDeinterleavedHalf(input + ix3 * 2);
            return std::tuple{ rr0, ii0, rr1, ii1, rr2, ii2, rr3, ii3 };
          }
          else
          {
            auto [rr0, ii0] = LoadDeinterleaved(input + ix0 * 2);
            auto [rr1, ii1] = LoadDeinterleaved(input + ix1 * 2);
            auto [rr2, ii2] = LoadDeinterleaved(input + ix2 * 2);
            auto [rr3, ii3] = LoadDeinterleaved(input + ix3 * 2);
            return std::tuple{ rr0, ii0, rr1, ii1, rr2, ii2, rr3, ii3 };
          }
        }();

      // IFFT needs to scale the coefficients
      if constexpr (D == Dir::Inverse)
        { Scale(inverse, r0, i0, r1, i1, r2, i2, r3, i3); }

      Radix4ButterflyNoRotation<D>(r0, i0, r1, i1, r2, i2, r3, i3);

      // Store the real and imaginary values side by side (in SIMD vectors). Note that for 8-element vectors this is
      //  doing two pairs of 4x4 transposes (transposing the first 4s of each vector and the last 4s separately)
      Store4x4Transposed(
        output + ix0 * 2,
        output + ix1 * 2,
        output + ix2 * 2,
        output + ix3 * 2,
        r0,
        r1,
        r2,
        r3);

      Store4x4Transposed(
        output + ix0 * 2 + N,
        output + ix1 * 2 + N,
        output + ix2 * 2 + N,
        output + ix3 * 2 + N,
        i0,
        i1,
        i2,
        i3);
    }
    else
    {
      // If N == 2 we can't do a 4x4 transpose using a single quad of vectors, so we'll need two.
      static_assert(N == 2);

      auto ip0 = input + ix0 * 2;
      auto ip1 = input + ix1 * 2;
      auto ip2 = input + ix2 * 2;
      auto ip3 = input + ix3 * 2;
      auto [r0a, i0a] = LoadDeinterleaved(ip0);
      auto [r1a, i1a] = LoadDeinterleaved(ip1);
      auto [r2a, i2a] = LoadDeinterleaved(ip2);
      auto [r3a, i3a] = LoadDeinterleaved(ip3);

      if constexpr (D == Dir::Inverse)
        { Scale(inverse, r0a, i0a, r1a, i1a, r2a, i2a, r3a, i3a); }

      Radix4ButterflyNoRotation<D>(r0a, i0a, r1a, i1a, r2a, i2a, r3a, i3a);

      auto op0a = output + ix0 * 2;
      auto op1a = output + ix1 * 2;
      auto op2a = output + ix2 * 2;
      auto op3a = output + ix3 * 2;

      auto op0b = op0a + 2 * N;
      auto op1b = op1a + 2 * N;
      auto op2b = op2a + 2 * N;
      auto op3b = op3a + 2 * N;

      // Store the upper-left of the 4x4 transpose
      Store2x2Transposed(op0a, op1a, r0a, r1a);
      Store2x2Transposed(op0a + N, op1a + N, i0a, i1a);
      
      auto [r0b, i0b] = LoadDeinterleaved(ip0 + 2 * N);
      auto [r1b, i1b] = LoadDeinterleaved(ip1 + 2 * N);

      // Store what was "lower left" to "upper right", transposed
      Store2x2Transposed(op0b, op1b, r2a, r3a);
      Store2x2Transposed(op0b + N, op1b + N, i2a, i3a);

      auto [r2b, i2b] = LoadDeinterleaved(ip2 + 2 * N);
      auto [r3b, i3b] = LoadDeinterleaved(ip3 + 2 * N);

      if constexpr (D == Dir::Inverse)
      {
        Scale(inverse, r0b, i0b, r1b, i1b, r2b, i2b, r3b, i3b);
      }

      Radix4ButterflyNoRotation<D>(r0b, i0b, r1b, i1b, r2b, i2b, r3b, i3b);

      // upper-right to lower-left
      Store2x2Transposed(op2a, op3a, r0b, r1b);
      Store2x2Transposed(op2a + N, op3a + N, i0b, i1b);
      
      // lower-right transposed in position
      Store2x2Transposed(op2b, op3b, r2b, r3b);
      Store2x2Transposed(op2b + N, op3b + N, i2b, i3b);
    }
  }
}


template <Dir D, ForceSize Force = ForceSize::No, std::floating_point F>
FORCE_INLINE void DoFirstRadix4sNoDeinterleave(const F *input, F *output, size_t radix4Count)
{
  using V = Vec<F>;
  constexpr auto N = (Force == ForceSize::Four) ? 4u : VecSize<V>;

  if constexpr (N > 4u && Force == ForceSize::No)
  {
    if (radix4Count == 4u)
      { return DoFirstRadix4sNoDeinterleave<D, ForceSize::Four>(input, output, radix4Count); }
  }

  [[maybe_unused]] F inverse;
  if constexpr (D == Dir::Inverse)
    { inverse = F(0.25) / F(radix4Count); }

  constexpr auto Step = std::max(4u, N);
  for (auto i = 0_z; i < radix4Count; i += Step)
  {
    auto ix0 = i;
    auto ix2 = ix0 + radix4Count; // i2/i1 intentionally out of order here
    auto ix1 = ix2 + radix4Count;
    auto ix3 = ix1 + radix4Count;

    if constexpr (N >= 4)
    {
      // Load the 4 sets of values (either as halves or as full values depending on whether we've forced the size)
      auto [r0, i0, r1, i1, r2, i2, r3, i3] = 
        [&]
        {
          if constexpr (Force == ForceSize::Four)
          {
            static_assert(N == 4);
            auto rr0 = LoadHalf(input + ix0 * 2);
            auto ii0 = LoadHalf(input + ix0 * 2 + N);
            auto rr1 = LoadHalf(input + ix1 * 2);
            auto ii1 = LoadHalf(input + ix1 * 2 + N);
            auto rr2 = LoadHalf(input + ix2 * 2);
            auto ii2 = LoadHalf(input + ix2 * 2 + N);
            auto rr3 = LoadHalf(input + ix3 * 2);
            auto ii3 = LoadHalf(input + ix3 * 2 + N);
            return std::tuple{ rr0, ii0, rr1, ii1, rr2, ii2, rr3, ii3 };
          }
          else
          {
            auto rr0 = Load(input + ix0 * 2);
            auto ii0 = Load(input + ix0 * 2 + N);
            auto rr1 = Load(input + ix1 * 2);
            auto ii1 = Load(input + ix1 * 2 + N);
            auto rr2 = Load(input + ix2 * 2);
            auto ii2 = Load(input + ix2 * 2 + N);
            auto rr3 = Load(input + ix3 * 2);
            auto ii3 = Load(input + ix3 * 2 + N);
            return std::tuple{ rr0, ii0, rr1, ii1, rr2, ii2, rr3, ii3 };
          }
        }();

      // IFFT needs to scale the coefficients
      if constexpr (D == Dir::Inverse)
        { Scale(inverse, r0, i0, r1, i1, r2, i2, r3, i3); }

      Radix4ButterflyNoRotation<D>(r0, i0, r1, i1, r2, i2, r3, i3);

      // Store the real and imaginary values side by side (in SIMD vectors). Note that for 8-element vectors this is
      //  doing two pairs of 4x4 transposes (transposing the first 4s of each vector and the last 4s separately)
      Store4x4Transposed(
        output + ix0 * 2,
        output + ix1 * 2,
        output + ix2 * 2,
        output + ix3 * 2,
        r0,
        r1,
        r2,
        r3);

      Store4x4Transposed(
        output + ix0 * 2 + N,
        output + ix1 * 2 + N,
        output + ix2 * 2 + N,
        output + ix3 * 2 + N,
        i0,
        i1,
        i2,
        i3);
    }
    else
    {
      // If N == 2 we can't do a 4x4 transpose using a single quad of vectors, so we'll need two.
      static_assert(N == 2);

      auto ip0 = input + ix0 * 2;
      auto ip1 = input + ix1 * 2;
      auto ip2 = input + ix2 * 2;
      auto ip3 = input + ix3 * 2;
      auto r0a = Load(ip0);
      auto i0a = Load(ip0 + N);
      auto r1a = Load(ip1);
      auto i1a = Load(ip1 + N);
      auto r2a = Load(ip2);
      auto i2a = Load(ip2 + N);
      auto r3a = Load(ip3);
      auto i3a = Load(ip3 + N);

      if constexpr (D == Dir::Inverse)
        { Scale(inverse, r0a, i0a, r1a, i1a, r2a, i2a, r3a, i3a); }

      Radix4ButterflyNoRotation<D>(r0a, i0a, r1a, i1a, r2a, i2a, r3a, i3a);

      auto op0a = output + ix0 * 2;
      auto op1a = output + ix1 * 2;
      auto op2a = output + ix2 * 2;
      auto op3a = output + ix3 * 2;

      auto op0b = op0a + 2 * N;
      auto op1b = op1a + 2 * N;
      auto op2b = op2a + 2 * N;
      auto op3b = op3a + 2 * N;

      // Store the upper-left of the 4x4 transpose
      Store2x2Transposed(op0a, op1a, r0a, r1a);
      Store2x2Transposed(op0a + N, op1a + N, i0a, i1a);
      
      auto r0b = Load(ip0 + 2 * N);
      auto i0b = Load(ip0 + 3 * N);
      auto r1b = Load(ip1 + 2 * N);
      auto i1b = Load(ip1 + 3 * N);

      // Store what was "lower left" to "upper right", transposed
      Store2x2Transposed(op0b, op1b, r2a, r3a);
      Store2x2Transposed(op0b + N, op1b + N, i2a, i3a);

      auto r2b = Load(ip2 + 2 * N);
      auto i2b = Load(ip2 + 3 * N);
      auto r3b = Load(ip3 + 2 * N);
      auto i3b = Load(ip3 + 3 * N);

      if constexpr (D == Dir::Inverse)
      {
        Scale(inverse, r0b, i0b, r1b, i1b, r2b, i2b, r3b, i3b);
      }

      Radix4ButterflyNoRotation<D>(r0b, i0b, r1b, i1b, r2b, i2b, r3b, i3b);

      // upper-right to lower-left
      Store2x2Transposed(op2a, op3a, r0b, r1b);
      Store2x2Transposed(op2a + N, op3a + N, i0b, i1b);
      
      // lower-right transposed in position
      Store2x2Transposed(op2b, op3b, r2b, r3b);
      Store2x2Transposed(op2b + N, op3b + N, i2b, i3b);
    }
  }
}


// This is the standard transposed radix 4 set. Excepting the first radix 4s, any double-bit flip sets need to involve
//  4 separate sets of 4-vector butterflies to swap around. Effectively it's doing a 4x4 transpose of SIMD sets 
//  (instead of transposing *within* a SIMD set like the first radix 4 step does)
template <Dir D, std::floating_point F, typename Ws>
void DoTransposedRadix4Set(F *data,  size_t i, size_t groupMask, size_t groupSize, size_t angleCount, Ws ws)
{
  using V = Vec<F>;
  constexpr auto N = VecSize<V>;

  auto ix0a = (i & groupMask) | ((i & ~groupMask) << 2);
  auto ix2a = ix0a + groupSize; // i2/i1 intentionally out of order here
  auto ix1a = ix2a + groupSize;
  auto ix3a = ix1a + groupSize;

  auto r0a = Load(data + ix0a * 2);
  auto i0a = Load(data + ix0a * 2 + N);
  auto r1a = Load(data + ix1a * 2);
  auto i1a = Load(data + ix1a * 2 + N);
  auto r2a = Load(data + ix2a * 2);
  auto i2a = Load(data + ix2a * 2 + N);
  auto r3a = Load(data + ix3a * 2);
  auto i3a = Load(data + ix3a * 2 + N);

  Radix4Butterfly<D>(r0a, i0a, r1a, i1a, r2a, i2a, r3a, i3a, ws);

  // There's register pressure here if we're not careful so in general we're going to store values as soon as we can
  //  (so the registers get reused). Since this first set stays in place we can put it right back where it goes.
  Store(data + ix0a * 2, r0a);
  Store(data + ix0a * 2 + N, i0a);

  auto ix0b = ix0a + angleCount;
  auto ix1b = ix1a + angleCount;
  auto ix2b = ix2a + angleCount;
  auto ix3b = ix3a + angleCount;
  auto r0b = Load(data + ix0b * 2);
  auto i0b = Load(data + ix0b * 2 + N);

  // We just loaded the first element of the second set which is also where the second result of the first set gets
  //  stored, so store that result right away to free up more register space. Any future loads/stores will also be
  //  paired like this.
  Store(data + ix0b * 2, r1a);
  Store(data + ix0b * 2 + N, i1a);

  auto r1b = Load(data + ix1b * 2);
  auto i1b = Load(data + ix1b * 2 + N);
  auto r2b = Load(data + ix2b * 2);
  auto i2b = Load(data + ix2b * 2 + N);
  auto r3b = Load(data + ix3b * 2);
  auto i3b = Load(data + ix3b * 2 + N);
  Radix4Butterfly<D>(r0b, i0b, r1b, i1b, r2b, i2b, r3b, i3b, ws);

  Store(data + ix1a * 2, r0b);
  Store(data + ix1a * 2 + N, i0b);
  Store(data + ix1b * 2, r1b);
  Store(data + ix1b * 2 + N, i1b);

  auto ix0c = ix0b + angleCount;
  auto ix1c = ix1b + angleCount;
  auto ix2c = ix2b + angleCount;
  auto ix3c = ix3b + angleCount;
  auto r0c = Load(data + ix0c * 2);
  auto i0c = Load(data + ix0c * 2 + N);
  Store(data + ix0c * 2, r2a);
  Store(data + ix0c * 2 + N, i2a);
  auto r1c = Load(data + ix1c * 2);
  auto i1c = Load(data + ix1c * 2 + N);
  Store(data + ix1c * 2, r2b);
  Store(data + ix1c * 2 + N, i2b);

  auto r2c = Load(data + ix2c * 2);
  auto i2c = Load(data + ix2c * 2 + N);
  auto r3c = Load(data + ix3c * 2);
  auto i3c = Load(data + ix3c * 2 + N);

  Radix4Butterfly<D>(r0c, i0c, r1c, i1c, r2c, i2c, r3c, i3c, ws);

  Store(data + ix2a * 2, r0c);
  Store(data + ix2a * 2 + N, i0c);
  Store(data + ix2b * 2, r1c);
  Store(data + ix2b * 2 + N, i1c);
  Store(data + ix2c * 2, r2c);
  Store(data + ix2c * 2 + N, i2c);

  auto ix0d = ix0c + angleCount;
  auto ix1d = ix1c + angleCount;
  auto ix2d = ix2c + angleCount;
  auto ix3d = ix3c + angleCount;

  auto r0d = Load(data + ix0d * 2);
  auto i0d = Load(data + ix0d * 2 + N);
  Store(data + ix0d * 2, r3a);
  Store(data + ix0d * 2 + N, i3a);
  auto r1d = Load(data + ix1d * 2);
  auto i1d = Load(data + ix1d * 2 + N);
  Store(data + ix1d * 2, r3b);
  Store(data + ix1d * 2 + N, i3b);
  auto r2d = Load(data + ix2d * 2);
  auto i2d = Load(data + ix2d * 2 + N);
  Store(data + ix2d * 2, r3c);
  Store(data + ix2d * 2 + N, i3c);
  auto r3d = Load(data + ix3d * 2);
  auto i3d = Load(data + ix3d * 2 + N);
  Radix4Butterfly<D>(r0d, i0d, r1d, i1d, r2d, i2d, r3d, i3d, ws);
  Store(data + ix3a * 2, r0d);
  Store(data + ix3a * 2 + N, i0d);
  Store(data + ix3b * 2, r1d);
  Store(data + ix3b * 2 + N, i1d);
  Store(data + ix3c * 2, r2d);
  Store(data + ix3c * 2 + N, i2d);
  Store(data + ix3d * 2, r3d);
  Store(data + ix3d * 2 + N, i3d);
}


// The tranposed set needed a special case for when there are four angles, specifically for AVX where there are 
//  8-element vectors.
template <Dir D, std::floating_point F, typename Ws>
void DoTransposedRadix4SetForFourAngles(F *data,  size_t i, size_t groupMask, size_t groupSize, Ws ws)
{
  using V = Vec<F>;
  constexpr auto N = VecSize<V>;
  constexpr auto AngleCount = 4u;

  if constexpr (N < 8)
    { return DoTransposedRadix4Set<D>(data, i, groupMask, groupSize, AngleCount, ws); }
  else
  {
    static_assert(N == 8);

    // the "a, b" and "c, d" butterflies are next to each other, so with 8-element vectors we can do each pair of them
    // at once, then just pull them apart into 4-element halves at save time. This is effectively doing the same 4
    // sets of butterflies as the non-specialized version, but as halves of two SIMD sets instead.
    // Like the specialized set, we store as soon as we can.
    auto ix0ab = (i & groupMask) | ((i & ~groupMask) << 2);
    auto ix2ab = ix0ab + groupSize; // i2/i1 intentionally out of order here
    auto ix1ab = ix2ab + groupSize;
    auto ix3ab = ix1ab + groupSize;

    auto r0ab = Load(data + ix0ab * 2);
    auto i0ab = Load(data + ix0ab * 2 + N);
    auto r1ab = Load(data + ix1ab * 2);
    auto i1ab = Load(data + ix1ab * 2 + N);
    auto r2ab = Load(data + ix2ab * 2);
    auto i2ab = Load(data + ix2ab * 2 + N);
    auto r3ab = Load(data + ix3ab * 2);
    auto i3ab = Load(data + ix3ab * 2 + N);

    Radix4Butterfly<D>(r0ab, i0ab, r1ab, i1ab, r2ab, i2ab, r3ab, i3ab, ws);
    
    auto [r0a, r0b] = SplitHalves(r0ab);
    auto [i0a, i0b] = SplitHalves(i0ab);
    auto [r1a, r1b] = SplitHalves(r1ab);
    auto [i1a, i1b] = SplitHalves(i1ab);
    Store(data + ix0ab * 2, r0a);
    Store(data + ix0ab * 2 + N, i0a);
    Store(data + ix0ab * 2 + AngleCount, r1a);
    Store(data + ix0ab * 2 + N + AngleCount, i1a);
    Store(data + ix1ab * 2, r0b);
    Store(data + ix1ab * 2 + N, i0b);
    Store(data + ix1ab * 2 + AngleCount, r1b);
    Store(data + ix1ab * 2 + N + AngleCount, i1b);

    auto ix0cd = ix0ab + N;
    auto ix1cd = ix1ab + N;
    auto ix2cd = ix2ab + N;
    auto ix3cd = ix3ab + N;
    auto r0cd = Load(data + ix0cd * 2);
    auto i0cd = Load(data + ix0cd * 2 + N);

    auto [r2a, r2b] = SplitHalves(r2ab);
    auto [i2a, i2b] = SplitHalves(i2ab);
    auto [r3a, r3b] = SplitHalves(r3ab);
    auto [i3a, i3b] = SplitHalves(i3ab);
    Store(data + ix0cd * 2, r2a);
    Store(data + ix0cd * 2 + N, i2a);
    Store(data + ix0cd * 2 + AngleCount, r3a);
    Store(data + ix0cd * 2 + N + AngleCount, i3a);

    auto r1cd = Load(data + ix1cd * 2);
    auto i1cd = Load(data + ix1cd * 2 + N);
    Store(data + ix1cd * 2, r2b);
    Store(data + ix1cd * 2 + N, i2b);
    Store(data + ix1cd * 2 + AngleCount, r3b);
    Store(data + ix1cd * 2 + N + AngleCount, i3b);

    auto r2cd = Load(data + ix2cd * 2);
    auto i2cd = Load(data + ix2cd * 2 + N);
    auto r3cd = Load(data + ix3cd * 2);
    auto i3cd = Load(data + ix3cd * 2 + N);

    Radix4Butterfly<D>(r0cd, i0cd, r1cd, i1cd, r2cd, i2cd, r3cd, i3cd, ws);

    auto [r0c, r0d] = SplitHalves(r0cd);
    auto [i0c, i0d] = SplitHalves(i0cd);
    auto [r1c, r1d] = SplitHalves(r1cd);
    auto [i1c, i1d] = SplitHalves(i1cd);
    auto [r2c, r2d] = SplitHalves(r2cd);
    auto [i2c, i2d] = SplitHalves(i2cd);
    auto [r3c, r3d] = SplitHalves(r3cd);
    auto [i3c, i3d] = SplitHalves(i3cd);

    Store(data + ix2ab * 2, r0c);
    Store(data + ix2ab * 2 + N, i0c);
    Store(data + ix2ab * 2 + AngleCount, r1c);
    Store(data + ix2ab * 2 + N + AngleCount, i1c);
    Store(data + ix2cd * 2, r2c);
    Store(data + ix2cd * 2 + N, i2c);
    Store(data + ix2cd * 2 + AngleCount, r3c);
    Store(data + ix2cd * 2 + N + AngleCount, i3c);
    Store(data + ix3ab * 2, r0d);
    Store(data + ix3ab * 2 + N, i0d);
    Store(data + ix3ab * 2 + AngleCount, r1d);
    Store(data + ix3ab * 2 + N + AngleCount, i1d);
    Store(data + ix3cd * 2, r2d);
    Store(data + ix3cd * 2 + N, i2d);
    Store(data + ix3cd * 2 + AngleCount, r3d);
    Store(data + ix3cd * 2 + N + AngleCount, i3d);
  }
}

// If there is no native FMA instruction we need to generate the third radix 4 angle. 
//  Conveniently, cos(3w) + i*sin(3w) == (cos(w) + i*sin(w)) * (cos(2w) + i*sin(2w))
constexpr bool RequiresThirdW = !UseFMAOptimization;
template<typename V>
auto GenerateW3(V wr1, V wi1, V wr2, V wi2)
{
  if constexpr (RequiresThirdW)
  {
    // This takes a third angle which can be synthe
    return Pair{ NMAdd(wi1, wi2, Mul(wr1, wr2)), MAdd(wr1, wi2, Mul(wi1, wr2)) };
  }
  else
  {
    // No angles needed but this function has to return a pair of *something* so here we go, something that will get
    //  completely optimized away.
    struct Empty {};
    return Pair{ Empty{}, Empty{} };
  }
}


// The second pass of radix4s has four angles in the set. This needs special-casing, both because AVX might have 8
//  elements in a vector and thus the halves should be repeated, and also because the 4- and 8-angle sections are
//  stored specially in the angle array (to allow for then to always be loaded in an aligned manner)
template <Dir D, std::floating_point F>
auto DoTransposeRadix4sForFourAngles(
  const F *tableData,
  F *data,
  size_t blockSize,
  size_t radix4Count,
  size_t groupSize) -> size_t
{
  constexpr auto AngleCount = 4u;
  using V = Vec<F>;
  constexpr auto N = VecSize<V>;

  const auto groupMask = groupSize - 1;

  if constexpr (N >= 4)
  {
    // With N >= 4 we don't need an outer loop at all - we load the angles once and keep them alive through the whole
    //  process. Easy peasy.
    std::ignore = blockSize;

    auto ws = LoadAnglesRadix4FromCount4(tableData, 0u);
    for (auto butterflyOffset = 0_z; butterflyOffset < radix4Count; butterflyOffset += 16u)
      { DoTransposedRadix4SetForFourAngles<D>(data, butterflyOffset, groupMask, groupSize, ws); }
  }
  else
  {
    static_assert(N == 2);

    // If N == 2, we need to do the angles 2 at a time so keep the outer loop like normal.
    for (auto angleIndex = 0u; angleIndex < AngleCount; angleIndex += N)
    {
      auto ws = LoadAnglesRadix4FromCount4(tableData, angleIndex);
      for (auto block = 0_z; block < radix4Count; block += blockSize)
      {

        for (auto butterflyOffset = block; butterflyOffset < block + blockSize; butterflyOffset += 16u)
          { DoTransposedRadix4SetForFourAngles<D>(data + 2 * angleIndex, butterflyOffset, groupMask, groupSize, ws); }
      }
    }
  }

  return groupSize / 4;
}


// This does every other double-bit-reversing radix 4 until less than two unreversed bits remain.
template <Dir D, std::floating_point F>
auto DoRemainingTransposeRadix4Iterations(
  const F *tableData,
  F *data,
  size_t blockSize,
  size_t radix4Count,
  size_t angleCount,
  size_t groupSize,
  uint32_t unreversedBitCount,
  uint32_t totalReverseBitCount) -> size_t
{
  using V = Vec<F>;
  constexpr auto N = VecSize<V>;

  while (unreversedBitCount < totalReverseBitCount - 1)
  {
    const auto groupMask = groupSize - 1;

    // As with most of the loops here we break things into blocks - the reason is that it is in our interest to only 
    //  load the angles once (and, if needed, compute w3 once), but at larger sizes it becomes *more* important to try
    //  to stick within data that can fit snugly within the CPU cache - so better, in those cases, to re-load/compute
    //  angles a few times and stick to smaller blocks.
    for (auto blockOffset = 0_z; blockOffset < radix4Count; blockOffset += blockSize)
    {
      for (auto angleIndex = 0_z; angleIndex < angleCount; angleIndex += N)
      {
        auto ws = LoadAnglesRadix4(tableData, angleIndex, angleCount);
        for (auto butterflyOffset = blockOffset; butterflyOffset < blockOffset + blockSize; butterflyOffset += angleCount * 4)
        {
          auto i = butterflyOffset + angleIndex;
          DoTransposedRadix4Set<D>(data, i, groupMask, groupSize, angleCount, ws);
        }
      }
    }

    unreversedBitCount += 2;
    angleCount *= 4;
    groupSize /= 4;
  }

  return groupSize;
}


// This is another special case butterfly, this time it's one of the single-bit-flip cases (this one is for the power-
//  of-four length FFTs).
template <Dir D, std::floating_point F, typename Ws>
void DoSingleFlipRadix4ButterflyForPow4FFT(F *data, size_t ix0,  size_t groupSize, Ws ws)
{
  using V = Vec<F>;
  constexpr auto N = VecSize<V>;

  // This one is nice because it only requires doing a single butterfly per flip!
  auto d0 = data + ix0 * 2;
  auto d2 = d0 + groupSize * 2; // d1/d2 intentionally in reverse order here
  auto d1 = d2 + groupSize * 2;
  auto d3 = d1 + groupSize * 2;

  auto r0 = Load(d0);
  auto i0 = Load(d0 + N);
  auto r1 = Load(d1);
  auto i1 = Load(d1 + N);
  auto r2 = Load(d2);
  auto i2 = Load(d2 + N);
  auto r3 = Load(d3);
  auto i3 = Load(d3 + N);

  Radix4Butterfly<D>(r0, i0, r1, i1, r2, i2, r3, i3, ws);

  // The bits being flipped here happen to be between i1 and i2, so write these out with i1 and i2 in opposite order.
  Store(d0, r0);
  Store(d0 + N, i0);
  Store(d1, r2);
  Store(d1 + N, i2);
  Store(d2, r1);
  Store(d2 + N, i1);
  Store(d3, r3);
  Store(d3 + N, i3);
}


// This is a special case of the above, specifically for a 64-element FFT, because it's a combo of the single-bit-flip
//  *and* the four-angle cases.
template <Dir D, std::floating_point F, typename Ws>
void DoSingleFlipRadix4ButterflyFor64Element(F *data, size_t ix0, Ws ws)
{
  using V = Vec<F>;
  constexpr auto N = VecSize<V>;
  constexpr auto GroupSize = 4u;

  if constexpr (N < 8)
    { return DoSingleFlipRadix4ButterflyForPow4FFT<D>(data, ix0, GroupSize, ws); }
  else
  {
    static_assert(N == 8);

    // For the 4-angle N == 8 case, we're going to do half-loads and half saves and do two butterflies at once, since
    //  the butterfly element spacing is 4 elements.
    constexpr auto Offset = 32u;

    auto d0 = data + ix0 * 2;
    auto d2 = d0 + GroupSize;
    auto d1 = d0 + 4 * GroupSize;
    auto d3 = d1 + GroupSize;

    auto r0 = LoadHalves(d0, Offset);
    auto i0 = LoadHalves(d0 + N, Offset);
    auto r1 = LoadHalves(d1, Offset);
    auto i1 = LoadHalves(d1 + N, Offset);
    auto r2 = LoadHalves(d2, Offset);
    auto i2 = LoadHalves(d2 + N, Offset);
    auto r3 = LoadHalves(d3, Offset);
    auto i3 = LoadHalves(d3 + N, Offset);

    Radix4Butterfly<D>(r0, i0, r1, i1, r2, i2, r3, i3, ws);

    // Like the standard case, we're swapping i1 and i2 here.
    StoreHalves(d0, Offset, r0);
    StoreHalves(d0 + N, Offset, i0);
    StoreHalves(d1, Offset, r2);
    StoreHalves(d1 + N, Offset, i2);
    StoreHalves(d2, Offset, r1);
    StoreHalves(d2 + N, Offset, i1);
    StoreHalves(d3, Offset, r3);
    StoreHalves(d3 + N, Offset, i3);
  }
}


// This does the standard single-flip radix 4 for power-of-four length FFTs (multiple butterflies)
template <Dir D, std::floating_point F>
void DoSingleFlipRadix4ForPow4FFT(
  const F *tableData,
  F *data, 
  size_t groupSize, 
  size_t angleCount, 
  size_t radix4Count,
  size_t blockSize)
{
  using V = Vec<F>;
  constexpr auto N = VecSize<V>;

  if (radix4Count > 16u)
  {
    const auto groupMask = groupSize - 1;
    for (auto block = 0_z; block < radix4Count; block += blockSize)
    {
      for (auto angleIndex = 0_z; angleIndex < angleCount; angleIndex += N)
      {
        auto ws = LoadAnglesRadix4(tableData, angleIndex, angleCount);
        for (auto butterflyOffset = block; butterflyOffset < block + blockSize; butterflyOffset += angleCount)
        {
          auto i = butterflyOffset + angleIndex;
          auto ix0 = (i & groupMask) | ((i & ~groupMask) << 2);
          DoSingleFlipRadix4ButterflyForPow4FFT<D>(data, ix0, groupSize, ws);
        }
      }
    }
  }
  else
  {
    // Special case of the above for 64-element FFTs (because this is also the 4-angle case)
    constexpr auto FFTLength = 64u;
    constexpr auto ButterflyStep = 16u * std::max(1u, (N / 4));
    constexpr auto AngleCount = 4u;

    assert(radix4Count == FFTLength / 4u);
    for (auto angleIndex = 0u; angleIndex < AngleCount; angleIndex += N)
    {
      const auto ws = LoadAnglesRadix4FromCount4(tableData, angleIndex);

      for (auto butterflyIndex = 0_z; butterflyIndex < FFTLength; butterflyIndex += ButterflyStep)
        { DoSingleFlipRadix4ButterflyFor64Element<D>(data + angleIndex * 2, butterflyIndex, ws); }
    }
  }
}


template <Dir D, std::floating_point F, typename Ws>
void DoSingleFlipRadix4ButterflyForNonPow4FFT(F *data, size_t ix0a, size_t groupSize, size_t angleCount, Ws ws)
{
  using V = Vec<F>;
  constexpr auto N = VecSize<V>;

  // This one is swapping between two sets of radix 4s, like:
  //   A B C D  ->  A E C G
  //   E F G H      B F D H
  // (effectively this is two side-by-side 2x2 transposes)
        
  auto ix2a = ix0a + groupSize; // i2/i1 intentionally out of order here
  auto ix1a = ix2a + groupSize;
  auto ix3a = ix1a + groupSize;

  auto r0a = Load(data + ix0a * 2);
  auto i0a = Load(data + ix0a * 2 + N);
  auto r1a = Load(data + ix1a * 2);
  auto i1a = Load(data + ix1a * 2 + N);
  auto r2a = Load(data + ix2a * 2);
  auto i2a = Load(data + ix2a * 2 + N);
  auto r3a = Load(data + ix3a * 2);
  auto i3a = Load(data + ix3a * 2 + N);

  Radix4Butterfly<D>(r0a, i0a, r1a, i1a, r2a, i2a, r3a, i3a, ws);

  Store(data + ix0a * 2, r0a);
  Store(data + ix0a * 2 + N, i0a);
  Store(data + ix2a * 2, r2a);
  Store(data + ix2a * 2 + N, i2a);

  auto ix0b = ix0a + angleCount;
  auto ix1b = ix1a + angleCount;
  auto ix2b = ix2a + angleCount;
  auto ix3b = ix3a + angleCount;

  auto r0b = Load(data + ix0b * 2);
  auto i0b = Load(data + ix0b * 2 + N);

  Store(data + ix0b * 2, r1a);
  Store(data + ix0b * 2 + N, i1a);

  auto r2b = Load(data + ix2b * 2);
  auto i2b = Load(data + ix2b * 2 + N);

  Store(data + ix2b * 2, r3a);
  Store(data + ix2b * 2 + N, i3a);

  auto r1b = Load(data + ix1b * 2);
  auto i1b = Load(data + ix1b * 2 + N);
  auto r3b = Load(data + ix3b * 2);
  auto i3b = Load(data + ix3b * 2 + N);

  Radix4Butterfly<D>(r0b, i0b, r1b, i1b, r2b, i2b, r3b, i3b, ws);

  Store(data + ix1a * 2, r0b);
  Store(data + ix1a * 2 + N, i0b);
  Store(data + ix1b * 2, r1b);
  Store(data + ix1b * 2 + N, i1b);
  Store(data + ix3a * 2, r2b);
  Store(data + ix3a * 2 + N, i2b);
  Store(data + ix3b * 2, r3b);
  Store(data + ix3b * 2 + N, i3b);
}

// There's another single-flip case, and that's for non-power-of-four length FFTs.
template <Dir D, std::floating_point F>
void DoSingleFlipRadix4ForNonPow4FFT(
  const F *tableData,
  F *data, 
  size_t groupSize, 
  size_t angleCount, 
  size_t radix4Count,
  size_t blockSize)
{
  using V = Vec<F>;
  constexpr auto N = VecSize<V>;

  if (radix4Count > 32u)
  {
    const auto groupMask = groupSize - 1;
    for (auto block = 0_z; block < radix4Count; block += blockSize)
    {
      for (auto angleIndex = 0_z; angleIndex < angleCount; angleIndex += N)
      {
        auto ws = LoadAnglesRadix4(tableData, angleIndex, angleCount);
        for (auto butterflyOffset = block; butterflyOffset < block + blockSize; butterflyOffset += 2 * angleCount)
        {
          auto i = butterflyOffset + angleIndex;
          auto ix0a = (i & groupMask) | ((i & ~groupMask) << 2);
          DoSingleFlipRadix4ButterflyForNonPow4FFT<D>(data, ix0a, groupSize, angleCount, ws);
        }
      }
    }
  }
  else
  {
    // As with some of the others, there's a special case of the above necessary: this time it's for 128-element FFTs,
    //  as the single-flip and four-angle cases overlap.

    constexpr auto FFTLength = 128u;
    constexpr auto Radix4Count = FFTLength / 4u;
    constexpr auto ButterflyStep = FFTLength / 4u;
    constexpr auto AngleCount = 4u;
    constexpr auto GroupSize = Radix4Count / 4;

    assert(angleCount == AngleCount);
    assert(radix4Count == Radix4Count);
    assert(groupSize == GroupSize);

    if constexpr (N <= 4)
    {
      for (auto angleIndex = 0u; angleIndex < AngleCount; angleIndex += N)
      {
        auto ws = LoadAnglesRadix4FromCount4(tableData, angleIndex);

        for (auto butterflyOffset = 0_z; butterflyOffset < FFTLength; butterflyOffset += ButterflyStep)
        {
          auto ix0a = butterflyOffset + angleIndex;
          DoSingleFlipRadix4ButterflyForNonPow4FFT<D>(data, ix0a, GroupSize, AngleCount, ws);
        }
      }
    }
    else
    {
      // Special case for when N == 8
      static_assert(N == 8);

      auto ws = LoadAnglesRadix4FromCount4(tableData, 0u);

      for (auto butterflyIndex = 0u; butterflyIndex < FFTLength; butterflyIndex += ButterflyStep)
      {
        auto d0 = data + butterflyIndex * 2;
        auto d2 = d0 + GroupSize * 2;
        auto d1 = d2 + GroupSize * 2;
        auto d3 = d1 + GroupSize * 2;
      
        // The two butterflies we need to do the swaps within live side-by-side in memory, so we can do them as a single
        //  SIMD radix 4 butterfly and then split the halves and transpose them.
        auto r0ab = Load(d0);
        auto i0ab = Load(d0+ N);
        auto r1ab = Load(d1);
        auto i1ab = Load(d1+ N);
        auto r2ab = Load(d2);
        auto i2ab = Load(d2+ N);
        auto r3ab = Load(d3);
        auto i3ab = Load(d3+ N);

        Radix4Butterfly<D>(r0ab, i0ab, r1ab, i1ab, r2ab, i2ab, r3ab, i3ab, ws);

        auto [r0a, r0b] = SplitHalves(r0ab);
        auto [i0a, i0b] = SplitHalves(i0ab);
        auto [r1a, r1b] = SplitHalves(r1ab);
        auto [i1a, i1b] = SplitHalves(i1ab);
        auto [r2a, r2b] = SplitHalves(r2ab);
        auto [i2a, i2b] = SplitHalves(i2ab);
        auto [r3a, r3b] = SplitHalves(r3ab);
        auto [i3a, i3b] = SplitHalves(i3ab);

        constexpr auto Half = N / 2;

        Store(d0, r0a);
        Store(d0 + Half, r1a);
        Store(d0 + N, i0a);
        Store(d0 + Half + N, i1a);
        Store(d1, r0b);
        Store(d1 + Half, r1b);
        Store(d1 + N, i0b);
        Store(d1 + Half + N, i1b);
        Store(d2, r2a);
        Store(d2 + Half, r3a);
        Store(d2 + N, i2a);
        Store(d2 + Half + N, i3a);
        Store(d3, r2b);
        Store(d3 + Half, r3b);
        Store(d3 + N, i2b);
        Store(d3 + Half + N, i3b);
      }
    }
  }
}


// Do a single no-bit-flip radix 2 butterfly
template <Dir D, std::floating_point F>
void DoNoFlipRadix2Butterfly(F *data, size_t ix0, size_t ix1, Vec<F> wr, Vec<F> wi)
{
  using V = Vec<F>;
  constexpr auto N = VecSize<V>;

  auto r0 = Load(data + ix0 * 2);
  auto i0 = Load(data + ix0 * 2 + N);
  auto r1 = Load(data + ix1 * 2);
  auto i1 = Load(data + ix1 * 2 + N);

  Radix2Butterfly<D>(r0, i0, r1, i1, wr, wi);

  Store(data + ix0 * 2, r0);
  Store(data + ix0 * 2 + N, i0);
  Store(data + ix1 * 2, r1);
  Store(data + ix1 * 2 + N, i1);
}


// Do the no-bit-flip radix 2 pass (this will occur in a non-power-of-four FFT immediately after the last radix-4 bit
//  flip pass completes)

template <Dir D, std::floating_point F>
void DoNoFlipRadix2(
  const F *tableData,
  F *data, 
  size_t groupSize, 
  size_t radix4Count,
  size_t blockCount)
{
  using V = Vec<F>;
  constexpr auto N = VecSize<V>;

  if (radix4Count > 8u)
  {
    const auto radix2Count = radix4Count * 2u;
    const auto groupMask = groupSize - 1;
    for (auto block = 0_z; block < radix2Count; block += blockCount)
    {
      for (auto angleIndex = 0_z; angleIndex < groupSize; angleIndex += N)
      {
        auto [wr, wi] = LoadAngleRadix2(tableData, angleIndex, groupSize);

        for (auto butterflyOffset = block; butterflyOffset < block + blockCount; butterflyOffset += groupSize)
        {
          auto i = angleIndex + butterflyOffset;
          auto ix0 = (i & groupMask) | ((i & ~groupMask) << 1);
          auto ix1 = ix0 + groupSize;
          DoNoFlipRadix2Butterfly<D>(data, ix0, ix1, wr, wi);
        }
      }
    }
  }
  else
  {
    assert(radix4Count == 8u); 
    constexpr auto FFTLength = 32u;
    constexpr auto GroupSize = 4u;
    if constexpr (N == 8)
    {
      // As is tradition around here, the above needs a special case: this time, 32-element FFTs, which have to do some
      //  special handling for N == 8 vector lengths (plus it's the four-angle case, which is itself a special case)
      auto [wr, wi] = LoadAngleRadix2FromCount4(tableData, 0u);

      for (auto butterflyOffset = 0u; butterflyOffset < FFTLength; butterflyOffset += 16u)
      {
        auto ix0 = butterflyOffset * 2;
        auto ix1 = ix0 + GroupSize;

        auto o0 = data + ix0;
        auto o1 = data + ix1;
        constexpr auto Offset = 2 * N;
        auto r0 = LoadHalves(o0, Offset);
        auto i0 = LoadHalves(o0 + N, Offset);
        auto r1 = LoadHalves(o1, Offset);
        auto i1 = LoadHalves(o1 + N, Offset);

        Radix2Butterfly<D>(r0, i0, r1, i1, wr, wi);
        StoreHalves(o0, Offset, r0);
        StoreHalves(o0 + N, Offset, i0);
        StoreHalves(o1, Offset, r1);
        StoreHalves(o1 + N, Offset, i1);
      }
    }
    else
    {
      for (auto angleIndex = 0u; angleIndex < 4u; angleIndex += N)
      {
        auto [wr1, wi1] = LoadAngleRadix2FromCount4(tableData, angleIndex);

        for (auto butterflyOffset = 0u; butterflyOffset < FFTLength; butterflyOffset += 8u)
        {
          auto ix0 = butterflyOffset + angleIndex;
          auto ix1 = ix0 + GroupSize; // i2/i1 intentionally out of order here
          DoNoFlipRadix2Butterfly<D>(data, ix0, ix1, wr1, wi1);
        }
      }
    }
  }
}


// Do a no-bit-flip radix 4 pass.
template <Dir D, std::floating_point F>
void DoNoFlipRadix4(
  const F *tableData,
  F *data, 
  size_t groupSize, 
  size_t radix4Count,
  size_t blockCount)
{
  using V = Vec<F>;
  constexpr auto N = VecSize<V>;

  const auto groupMask = groupSize - 1;
  for (auto block = 0_z; block < radix4Count; block += blockCount)
  {
    for (auto angleIndex = 0_z; angleIndex < groupSize; angleIndex += N)
    {
      auto ws = LoadAnglesRadix4(tableData, angleIndex, groupSize);
      for (auto butterflyOffset = block; butterflyOffset < block + blockCount; butterflyOffset += groupSize)
      {
        auto i = angleIndex + butterflyOffset;
        auto ix0 = (i & groupMask) | ((i & ~groupMask) << 2);
        auto ix1 = ix0 + groupSize; // i2/i1 *not* reversed this time around! We finally made it!
        auto ix2 = ix1 + groupSize;
        auto ix3 = ix2 + groupSize;

        auto r0 = Load(data + ix0 * 2);
        auto i0 = Load(data + ix0 * 2 + N);
        auto r1 = Load(data + ix1 * 2);
        auto i1 = Load(data + ix1 * 2 + N);
        auto r2 = Load(data + ix2 * 2);
        auto i2 = Load(data + ix2 * 2 + N);
        auto r3 = Load(data + ix3 * 2);
        auto i3 = Load(data + ix3 * 2 + N);

        Radix4Butterfly<D>(r0, i0, r1, i1, r2, i2, r3, i3, ws);

        Store(data + ix0 * 2, r0);
        Store(data + ix0 * 2 + N, i0);
        Store(data + ix1 * 2, r1);
        Store(data + ix1 * 2 + N, i1);
        Store(data + ix2 * 2, r2);
        Store(data + ix2 * 2 + N, i2);
        Store(data + ix3 * 2, r3);
        Store(data + ix3 * 2 + N, i3);
      }
    }
  }
}


// The final radix-4 pass loads alllll the angles, and also re-interleaves the data as it is the final output.
template <Dir D, std::floating_point F>
void DoFinalReinterleaveRadix4(
  const F *tableData,
  F *data,
  size_t groupSize)
{
  using V = Vec<F>;

  if (groupSize > 4u)
  {
    constexpr auto N = VecSize<V>;
    for (auto angleIndex = 0_z; angleIndex < groupSize; angleIndex += N)
    {
      auto ws = LoadAnglesRadix4(tableData, angleIndex, groupSize);

      auto ix0 = angleIndex;
      auto ix1 = ix0 + groupSize; // i2/i1 *not* reversed this time around
      auto ix2 = ix1 + groupSize;
      auto ix3 = ix2 + groupSize;

      auto r0 = Load(data + ix0 * 2);
      auto i0 = Load(data + ix0 * 2 + N);
      auto r1 = Load(data + ix1 * 2);
      auto i1 = Load(data + ix1 * 2 + N);
      auto r2 = Load(data + ix2 * 2);
      auto i2 = Load(data + ix2 * 2 + N);
      auto r3 = Load(data + ix3 * 2);
      auto i3 = Load(data + ix3 * 2 + N);

      Radix4Butterfly<D>(r0, i0, r1, i1, r2, i2, r3, i3, ws);
      StoreInterleaved(data + ix0 * 2, r0, i0);
      StoreInterleaved(data + ix1 * 2, r1, i1);
      StoreInterleaved(data + ix2 * 2, r2, i2);
      StoreInterleaved(data + ix3 * 2, r3, i3);
    }
  }
  else
  {
    assert(groupSize == 4u);
    // For a 16-element FFT, the final Reinterleave is *also* the four-angle case.
    constexpr auto N = std::min(4u, VecSize<V>);

    constexpr auto DoLoad = 
      [](auto *ptr)
      {
        if constexpr (VecSize<V> == 8u)
          { return LoadHalf(ptr); }
        else
          { return Load(ptr); }
      };
      

    for (auto angleIndex = 0u; angleIndex < 4u; angleIndex += N)
    {
      const auto ws = 
        [&]
        {
          if constexpr (VecSize<V> == 8u)
            { return LoadAnglesRadix4FromCount4Half(tableData, angleIndex); }
          else
            { return LoadAnglesRadix4FromCount4(tableData, angleIndex); }
        }();

      auto ix0 = angleIndex;
      auto ix1 = ix0 + groupSize; // i2/i1 *not* reversed this time around
      auto ix2 = ix1 + groupSize;
      auto ix3 = ix2 + groupSize;

      auto r0 = DoLoad(data + ix0 * 2);
      auto i0 = DoLoad(data + ix0 * 2 + N);
      auto r1 = DoLoad(data + ix1 * 2);
      auto i1 = DoLoad(data + ix1 * 2 + N);
      auto r2 = DoLoad(data + ix2 * 2);
      auto i2 = DoLoad(data + ix2 * 2 + N);
      auto r3 = DoLoad(data + ix3 * 2);
      auto i3 = DoLoad(data + ix3 * 2 + N);

      Radix4Butterfly<D>(r0, i0, r1, i1, r2, i2, r3, i3, ws);
      StoreInterleaved(data + ix0 * 2, r0, i0);
      StoreInterleaved(data + ix1 * 2, r1, i1);
      StoreInterleaved(data + ix2 * 2, r2, i2);
      StoreInterleaved(data + ix3 * 2, r3, i3);
    }
  }
}


// FFTs smaller than 16 just get done as scalar math.
template <Dir D, std::floating_point F>
void TinyFFT(const Complex<F> *input, Complex<F> *output, std::size_t fftLength)
{
  switch (fftLength)
  {
  case 1:
    output[0] = input[0];
    break;

  case 2:
    {
      auto x0 = input[0];
      auto x1 = input[1];
      output[0] = x0 + x1;
      output[1] = x0 - x1;
    }
    break;

  case 4:
    {
      auto x0 = input[0];
      auto x1 = input[1];
      auto x2 = input[2];
      auto x3 = input[3];

      auto sum02 = x0 + x2;
      auto diff02 = x0 - x2;
      auto sum13 = x1 + x3;
      auto iDiff13 = x1 - x3;
      if constexpr(D == Dir::Forward)
        { iDiff13 = { iDiff13.i, -iDiff13.r }; }
      else
        { iDiff13 = { -iDiff13.i, iDiff13.r }; }

      output[0] = sum02 + sum13;
      output[1] = diff02 + iDiff13;
      output[2] = sum02 - sum13;
      output[3] = diff02 - iDiff13;
    }
    break;

  case 8:
    {
      for (auto i = 0u; i < 4u; i++)
      {
        auto t = input[i] + input[i + 4];
        output[i + 4] = input[i] - input[i + 4];
        output[i] = t;
      }

      if constexpr (D == Dir::Forward)
      {
        output[6] = { output[6].i, -output[6].r };
        output[7] = { output[7].i, -output[7].r };
      }
      else
      {
        output[6] = { -output[6].i, output[6].r };
        output[7] = { -output[7].i, output[7].r };
      }

      for (auto i = 0u; i < 8u; i += 4)
      {
        auto t = output[i] + output[i + 2];
        output[i + 2] = output[i] - output[i + 2];
        output[i] = t;

        t = output[i + 1] + output[i + 3];
        output[i + 3] = output[i + 1] - output[i + 3];
        output[i + 1] = t;
      }

      // Now do the bit reverse ordering
      std::swap(output[1], output[4]);
      std::swap(output[3], output[6]);

      if constexpr (D == Dir::Forward)
      {
        output[5] = 
          {
            // (r * s) - (i * -s), (r * -s) + (i * s) -> (r + i)s, (i - r)*s
            (output[5].r + output[5].i) * InvSqrt2<F>,
            (output[5].i - output[5].r) * InvSqrt2<F>,
          };
        output[6] = { output[6].i, -output[6].r };
        output[7] = 
          {
            // (r * -s) - (i * -s), (r * -s) + (i * -s) -> (i - r)s, -(r + i)s
            (output[7].i - output[7].r) * InvSqrt2<F>,
            -(output[7].r + output[7].i) * InvSqrt2<F>,
          };
      }
      else
      {
        output[5] = 
          {
            // (r * s) - (i * s), (r * s) + (i * s) -> (r - i)*s, (r + i)*s
            (output[5].r - output[5].i) * InvSqrt2<F>,
            (output[5].i + output[5].r) * InvSqrt2<F>,
          };
        output[6] = { -output[6].i, output[6].r };
        output[7] = 
          {
            // (r * -s) - (i * s), (r * s) + (i * -s) -> -(r + i)s, (r - i)s
            -(output[7].r + output[7].i) * InvSqrt2<F>,
            (output[7].r - output[7].i) * InvSqrt2<F>,
          };
      }
      for (auto i = 0u; i < 4u; i++)
      {
        auto t = output[i] + output[i + 4];
        output[i + 4] = output[i] - output[i + 4];
        output[i] = t;
      }
    }
    break;
  }

  // Do the IFFT scaling (at the end, this time) if needed.
  if constexpr (D == Dir::Inverse)
  {
    const auto scale = F(1) / F(fftLength);
    for (auto i = 0u; i < fftLength; i++)
      { output[i] *= scale; }
  }
}


constexpr auto SmallestPiecewiseComplexFFT = 16u;

template <Dir D, std::floating_point F>
void RunComplexFFTMiddleStages(const F *tableData, F *output, std::size_t fftLength)
{
  assert(fftLength >= SmallestPiecewiseComplexFFT);
  
  const auto radix4Count = fftLength / 4u;
  const auto totalReverseBitCount = Log2Floor(fftLength) / 2u;

  // For larger FFT sizes we're going to process in blocks - this is much more cache-friendly (even if it requires re-
  //  loading the same angles multiple times, which we otherwise want to avoid)
  const auto blockSize = std::min(2048_z, radix4Count);

  // The initial group is special-cased because the 4-count angles load differently from the data tables.
  auto groupSize = radix4Count / 4;
  if (fftLength >= 256)
  {
    groupSize = DoTransposeRadix4sForFourAngles<D>(tableData, output, blockSize, radix4Count, groupSize);

  // Now do every remaining radix 4 transposing pass. This function returns the remaining group size (which also
  //  doubles as the number of angles we need (which is not true in the first half of the FFT).
    constexpr auto InitialUnreversedBitCount = 4u; // first 2 passes reversed 2 bits each
    constexpr auto InitialAngleCount = 16u; // first pass 1 angle, second pass 4, so...
    groupSize = DoRemainingTransposeRadix4Iterations<D>(
      tableData, 
      output, 
      blockSize, 
      radix4Count, 
      InitialAngleCount, 
      groupSize, 
      InitialUnreversedBitCount, 
      totalReverseBitCount);
  }

  if (IsPowerOfFourAssumingIsPowerOfTwo(fftLength))
  {
    // There's a single remaining bit flip to do.
    if ((totalReverseBitCount & 1) != 0)
      { DoSingleFlipRadix4ForPow4FFT<D>(tableData, output, groupSize, groupSize, radix4Count, blockSize); }
  }
  else
  {
    // Like above, there's a single-bit flip to do, but for a non-power-of-four FFT the swizzle pattern is different.
    if ((totalReverseBitCount & 1) != 0)
      { DoSingleFlipRadix4ForNonPow4FFT<D>(tableData, output, groupSize, groupSize / 2, radix4Count, blockSize); }

    // now do a single radix 2 pass since this isn't a power-of-four length FFT.
    const auto radix2GroupSize = groupSize * 2;
    const auto blockCount = std::max(radix2GroupSize, blockSize);
    DoNoFlipRadix2<D>(tableData, output, radix2GroupSize, radix4Count, blockCount);
  }
  
  // The group size counts down during the bit-reversing process and now counts back up.
  groupSize *= 4;

  // Now we can blast through the remaining radix 4 passes (none of which need to do any bit-reverse transposition
  //  anymore, as we're already in the final position)
  while (groupSize < radix4Count)
  {
    auto blockCount = std::max(groupSize, blockSize);
    DoNoFlipRadix4<D>(tableData, output, groupSize, radix4Count, blockCount);
    groupSize *= 4;
  }
}


// Here's the full DIT radix 4 skeleton!
template <Dir D, std::floating_point F>
void RunComplexFFT(const F *tableData, const F *input, F *output, std::size_t fftLength)
{
  // Handle any special case FFT lengths
  if (fftLength < SmallestPiecewiseComplexFFT)
    { return TinyFFT<D>(reinterpret_cast<const Complex<F> *>(input), reinterpret_cast<Complex<F> *>(output), fftLength); }

  const auto radix4Count = fftLength / 4u;

  // First pass has no rotations (besides a -90). This is the only stage that uses "input" directly (everything else
  //  works in-place in "output", and also is where the 1/N scale happens for an IFFT.
  //  This pass *also* does a vector-sized deinterleave of the real and imaginary components, allowing the intermediate
  //  passes to load real/imaginary components directly.
  DoFirstRadix4s<D>(input, output, radix4Count);

  RunComplexFFTMiddleStages<D>(tableData, output, fftLength);

  // The final pass is distinct as it needs to re-interleave the real and imaginary components.
  DoFinalReinterleaveRadix4<D>(tableData, output, radix4Count);
}


template <std::floating_point F>
void RunComplexFFTForFwdRealTransform(const F *tableData, const F *input, F *output, std::size_t fftLength)
{
  assert(fftLength >= SmallestPiecewiseComplexFFT);

  const auto radix4Count = fftLength / 4u;
  DoFirstRadix4s<Dir::Forward>(input, output, radix4Count);
  RunComplexFFTMiddleStages<Dir::Forward>(tableData, output, fftLength);
  DoNoFlipRadix4<Dir::Forward>(tableData, output, radix4Count, radix4Count, radix4Count);
}


template <std::floating_point F>
void RunComplexFFTForInvRealTransform(const F *tableData, const F *input, F *output, std::size_t fftLength)
{
  assert(fftLength >= SmallestPiecewiseComplexFFT);

  const auto radix4Count = fftLength / 4u;
  DoFirstRadix4sNoDeinterleave<Dir::Inverse>(input, output, radix4Count);
  RunComplexFFTMiddleStages<Dir::Inverse>(tableData, output, fftLength);
  DoFinalReinterleaveRadix4<Dir::Inverse>(tableData, output, radix4Count);
}


template <typename F>
inline void DoForwardRealAdjustment(const F *tableData, size_t angleIndex, size_t angleCount, Vec<F> &r0, Vec<F> &i0, Vec<F> &r1, Vec<F> &i1)
{
  auto half = Set1<Vec<F>>(F(0.5f));

  auto evenR = Add(r0, r1);
  auto evenI = Sub(i0, i1); // treat as conjugate

  // oddR, 90 degree rotated
  auto oddR = Add(i0, i1);
  auto oddI = Sub(r1, r0);

  auto [wr, wi] = LoadAngleRadix2(tableData, angleIndex, angleCount);
  r0 = Mul(half, NMAdd(oddI, wi, MAdd(oddR, wr, evenR)));
  i0 = Mul(half, MAdd(oddR, wi, MAdd(oddI, wr, evenI)));

  r1 = Sub(evenR, r0);
  i1 = Sub(i0, evenI); // conjugate again
}

template <typename F>
void RunRealFwdFFT(const F *tableData, const F *input, F *output, size_t fftLength)
{
  if (fftLength < 2 * SmallestPiecewiseComplexFFT)
  {
    // Small complex FFTs are simple enough that they're hardcoded and we don't have a "real transform" variant of 
    //  them. Instead, just run it as a full-sized complex transform and pull out the half (plus 1) of the results we
    //  care about.

    #if COMPILER_GCC
      #pragma GCC diagnostic push
      #pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
    #endif
    F data[SmallestPiecewiseComplexFFT];
    assert(fftLength <= SmallestPiecewiseComplexFFT);

    for (auto i = 0_z; i < fftLength; i++)
    { 
      data[2 * i] = input[i];
      data[2 * i + 1] = 0;
    }

    RunComplexFFT<Dir::Forward>(tableData, data, data, fftLength);

    #if COMPILER_GCC
      #pragma GCC diagnostic pop
    #endif

    std::ranges::copy(data, data + fftLength, output);
    output[1] = data[fftLength];
    return;
  }

  const auto complexFFTLength = fftLength / 2;
  auto cplxOut = reinterpret_cast<Complex<F> *>(output);

  // A real-valued-input FFT can be calculated as an 1/2 length FFT of complex values. This takes advantage of FFTs
  //  being linear (i.e. FFT(sequenceA) + FFT(sequenceB) == FFT(elementwiseAddition(sequenceA, sequenceB))).
  // Thus, if we treat the real numbers as if every pair of them were a complex number (real number sequence
  //  [a b c d ...] pretending to be [(a + b*i), (c + d*i) ...]), the FFT of that is the same as:
  //    FFT(Even-Indexed Real Values) + FFT(i*(Odd-Indexed Real Values))
  //  (note that the odd values are multiplied by "i" to make the complex number sum).
  if constexpr (IsScalar)
    { RunComplexFFT<Dir::Forward>(tableData, input, output, complexFFTLength); }
  else
    { RunComplexFFTForFwdRealTransform(tableData, input, output, complexFFTLength); }

  constexpr auto ImaginaryComponentOffset = IsScalar ? 1u : VecSize<Vec<F>>;
  // To get the true result out, then, we treat the even/odd values as if we were doing a decimation-in-time FFT.
  //  We have half as many (complex) values as the actual length of the FFT, but we only *need* half (plus one) -
  //  for real-valued FFTs, the second half of the values are a conjugate mirror of the first:
  //     RealFFT(k) == Conjugate(RealFFT(N - k))
  // Two notes about this:
  //  - the indexing is modulo, which matters where k == 0 (RealFFT(N) becomes RealFFT(0))
  //  - The values at k == 0 and k == N/2 have to match their conjugates, which means they *must* be real values
  //
  // We need half + 1 output values (which is more space than in the output array), but since FFT(0) and FFT(N/2) are
  //  real numbers, we can pack them together as a special case in output[0] and output[1] respectively (they're not
  //  truly a complex pair like the rest of the output). This is pretty standard.

  // Do that now: pack the DC and Nyquist terms (0 and N/2) into the first "complex" output. This also allows us to
  //  skip k == 0 in the loop and avoid the modulo indexing issue.
  auto x0r = output[0];
  auto x0i = output[ImaginaryComponentOffset];

  // Calculate the dc/nyquist/middle values up front - they're going to effectively get garbage written to them
  //  during the processing, but we can just set them to these values when we're done.
  auto dc = x0r + x0i;
  auto nyquist = x0r - x0i;
  auto middle = Complex { output[complexFFTLength], -output[complexFFTLength + ImaginaryComponentOffset] };

  // For the rest, use the conjugate property to calculate the real values effectively from the outside in.

  const auto angleCount = complexFFTLength;

  if constexpr (IsScalar)
  {
    for (auto k = 1u; k < complexFFTLength / 2; k++)
    {
      auto x0 = cplxOut[k];
      auto x1Conjugate = Conjugate(cplxOut[complexFFTLength - k]);
      auto even = (x0 + x1Conjugate) * F(0.5);
      auto odd = Rot90<Dir::Forward>((x0 - x1Conjugate) * F(0.5));

      auto angle = 
        Complex<F> 
        {
          tableData[angleCount * 3 / 2 + k],
          tableData[angleCount * 3 / 2 + k + angleCount / 2],
        };
      auto oddRotated = Rot<Dir::Forward>(odd, angle);
      cplxOut[k] = even + oddRotated;
      cplxOut[complexFFTLength - k] = Conjugate(even - oddRotated);
    }
  }
  else
  {
    Vec<F> prevR1In;
    Vec<F> prevI1In;

    Vec<F> prevR1Out;
    Vec<F> prevI1Out;
    
    // Prime the pump by doing the first transpose
    {
      auto r0 = Load(output);
      auto i0 = Load(output + VecSize<Vec<F>>);
      auto [r1, i1] = LoadFirstRightSideForForwardRealTransform(output + fftLength - VecSize<Vec<F>> * 2, prevR1In, prevI1In);
      DoForwardRealAdjustment(tableData, 0u, angleCount, r0, i0, r1, i1);
      StoreInterleaved(output, r0, i0);
      prevR1Out = r1;
      prevI1Out = i1;
    }

    for (auto k = VecSize<Vec<F>>; k < complexFFTLength / 2; k += VecSize<Vec<F>>)
    {
      auto r0 = Load(output + 2 * k);
      auto i0 = Load(output + 2 * k + VecSize<Vec<F>>);

      auto [r1, i1] = LoadRightSideForForwardRealTransform(output + fftLength - (2 * k) - 2*VecSize<Vec<F>>, prevR1In, prevI1In);
      DoForwardRealAdjustment(tableData, k, angleCount, r0, i0, r1, i1);
      StoreInterleaved(output + 2 * k, r0, i0);
      StoreRightSideForForwardRealTransform(output + fftLength - 2 * k, r1, i1, prevR1Out, prevI1Out);
    }

    StoreFinalRightSideForForwardRealTransform(output + fftLength / 2, prevR1Out, prevI1Out); 
  }

  cplxOut[0] = { dc, nyquist };
  cplxOut[complexFFTLength / 2] = middle;
}


template <typename F>
inline void DoInverseRealAdjustment(const F *tableData, size_t angleIndex, size_t angleCount, Vec<F> &r0, Vec<F> &i0, Vec<F> &r1, Vec<F> &i1)
{
  auto [wr, wi] = LoadAngleRadix2(tableData, angleIndex, angleCount);

  auto half = Set1<Vec<F>>(F(0.5f));

  auto evenR = Add(r0, r1);
  auto evenI = Sub(i0, i1); // treat as conjugate

  auto oddR = Sub(r0, r1);
  auto oddI = Add(i0, i1);

  r0 = Mul(half, NMAdd(wr, oddI, MAdd(wi, oddR, evenR)));
  i0 = Mul(half, MAdd(wi, oddI,  MAdd(wr, oddR, evenI)));

  r1 = Sub(evenR, r0);
  i1 = Sub(i0, evenI);
}


template <typename F>
void RunRealInvFFT(const F *tableData, const F *input, F *output, size_t fftLength)
{
  constexpr auto N = VecSize<Vec<F>>;

  if (fftLength < 2 * SmallestPiecewiseComplexFFT)
  {
    #if COMPILER_GCC
      #pragma GCC diagnostic push
      #pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
    #endif

    // Small complex FFTs are simple enough that they're hardcoded and we don't have a "real transform" variant of 
    //  them. Instead, just run it as a full-sized complex transform and pull out the half (plus 1) of the results we
    //  care about.
    F data[SmallestPiecewiseComplexFFT];
    assert(fftLength <= SmallestPiecewiseComplexFFT);

    data[0] = input[0];
    data[1] = 0;
    data[fftLength] = input[1];
    data[fftLength + 1] = 0;

    for(auto k = 1_z; k < fftLength / 2; k++)
    {
      data[2 * k + 0] = input[2 * k + 0];
      data[2 * k + 1] = input[2 * k + 1];

      data[2 * fftLength - 2 * k + 0] = input[2 * k + 0];
      data[2 * fftLength - 2 * k + 1] = -input[2 * k + 1];
    }

    RunComplexFFT<Dir::Inverse>(tableData, data, data, fftLength);

    #if COMPILER_GCC
      #pragma GCC diagnostic pop
    #endif

    for (auto i = 0_z; i < fftLength; i++)
      { output[i] = data[2 * i]; }
    return;
  }

  const auto complexFFTLength = fftLength / 2;

  const auto angleCount = complexFFTLength;

  auto o0r = (input[0] + input[1]) * F(0.5);
  auto o0i = (input[0] - input[1]) * F(0.5);

  auto middleR = input[complexFFTLength];
  auto middleI = -input[complexFFTLength + 1];

  // Undo what the real forward FFT does before we run the complex IFFT
  if constexpr (IsScalar)
  {
    auto cplxIn = reinterpret_cast<const Complex<F> *>(input);
    auto cplxOut = reinterpret_cast<Complex<F> *>(output);

    // Undo what the real forward FFT does before we run the complex IFFT
    for (auto k = 1u; k < complexFFTLength / 2; k++)
    {
      auto x0 = cplxIn[k];
      auto x1Conjugate = Conjugate(cplxIn[complexFFTLength - k]);
      auto even = (x0 + x1Conjugate) * F(0.5);

      auto angle = 
        Complex<F> 
        {
          tableData[angleCount * 3 / 2 + k],
          tableData[angleCount * 3 / 2 + k + angleCount / 2],
        };

      auto oddUnRotated = Rot<Dir::Inverse>((x0 - x1Conjugate) * F(0.5), angle);
      auto odd = Rot90<Dir::Inverse>(oddUnRotated);

      cplxOut[k] = even + odd;
      cplxOut[complexFFTLength - k] = Conjugate(even - odd);
    }
  }
  else
  {
    Vec<F> prevX1AIn;
    Vec<F> prevX1BIn;

    Vec<F> prevR1Out;
    Vec<F> prevI1Out;

    // Prime the pump by doing the first transpose
    {
      auto [r0, i0] = LoadDeinterleaved(input);
      auto [r1, i1] = LoadFirstRightSideForInverseRealTransform(input + fftLength - VecSize<Vec<F>> * 2, prevX1AIn, prevX1BIn);
      DoInverseRealAdjustment(tableData, 0u, angleCount, r0, i0, r1, i1);
      Store(output, r0);
      Store(output + N, i0);
      prevR1Out = r1;
      prevI1Out = i1;
    }

    for (auto k = VecSize<Vec<F>>; k < complexFFTLength / 2; k += VecSize<Vec<F>>)
    {
      auto [r0, i0] = LoadDeinterleaved(input + 2 * k);
      auto [r1, i1] = LoadRightSideForInverseRealTransform(input + fftLength - (2 * k) - 2*VecSize<Vec<F>>, prevX1AIn, prevX1BIn);
      DoInverseRealAdjustment(tableData, k, angleCount, r0, i0, r1, i1);
      Store(output + 2 * k, r0);
      Store(output + 2 * k + N, i0);
      StoreRightSideForInverseRealTransform(output + fftLength - 2 * k, r1, i1, prevR1Out, prevI1Out);
    }

    StoreFinalRightSideForInverseRealTransform(output + fftLength / 2, prevR1Out, prevI1Out); 
  }

  // Restore the special-case values that got clobbered by the SIMD loop
  constexpr auto ImaginaryComponentOffset = IsScalar ? 1u : N;

  output[0] = o0r;
  output[ImaginaryComponentOffset] = o0i;
  output[complexFFTLength] = middleR;
  output[complexFFTLength + ImaginaryComponentOffset] = middleI;

  if constexpr (IsScalar)
    { RunComplexFFT<Dir::Inverse>(tableData, output, output, complexFFTLength); }
  else
    { RunComplexFFTForInvRealTransform(tableData, output, output, complexFFTLength); }
}
