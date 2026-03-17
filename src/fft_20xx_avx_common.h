template <typename T> concept f32 = std::same_as<float, T> || std::same_as<__m128, T> || std::same_as<__m256, T>;
template <typename T> concept simd_vec = std::same_as<std::remove_cvref_t<T>, __m256> || std::same_as<std::remove_cvref_t<T>, __m256d>;
template <typename T> concept simd_half_vec = std::same_as<std::remove_cvref_t<T>, __m128> || std::same_as<std::remove_cvref_t<T>, __m128d>;

template <typename T> using Vec = std::conditional_t<f32<T>, __m256, __m256d>;

template <typename> constexpr uint32_t VecSize = 0;

template <> constexpr uint32_t VecSize<Vec<float>> = 8u;
template <> constexpr uint32_t VecSize<Vec<double>> = 4u;
template <> constexpr uint32_t VecSize<__m128> = 4u;

constexpr bool IsScalar = false;

#if SIMD_FMA
  constexpr bool UseFMAOptimization = true;
#else
  constexpr bool UseFMAOptimization = false;
#endif

template <std::floating_point F>
static auto LoadA(const F *ptr) -> Vec<F>
{
  if constexpr (f32<F>)
    { return _mm256_load_ps(ptr); }
  else
    { return _mm256_load_pd(ptr); }
}

template <std::floating_point F>
static auto Load(const F *ptr) -> Vec<F>
{
  if constexpr (f32<F>)
    { return _mm256_loadu_ps(ptr); }
  else
    { return _mm256_loadu_pd(ptr); }
}

static auto LoadHalf(const float *ptr) -> __m128
  { return _mm_loadu_ps(ptr); }

static auto SplitHalves(__m256 v) -> Pair<__m128, __m128>
  { return { _mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1) }; }

static auto LoadHalves(const float *ptrA, size_t offset)
  { return _mm256_set_m128(_mm_loadu_ps(ptrA + offset), _mm_loadu_ps(ptrA)); }

template <std::floating_point F>
static auto Load4(const F *ptr) -> Vec<F>
{
  if constexpr (f32<F>)
  {
    auto half = _mm_load_ps(ptr); // aligned load
    return _mm256_set_m128(half, half);
  }
  else
    { return _mm256_load_pd(ptr); }
}

template <std::floating_point F>
static auto LoadDeinterleaved(const F *ptr) -> Pair<Vec<F>, Vec<F>>
{
  auto a = Load(ptr);
  auto b = Load(ptr + VecSize<Vec<F>>);

  if constexpr (f32<F>)
  {
      // (A a B b | C c D d) (E e F f | G g H h) -> (A a B b | E e F f)
    auto firstHalves = _mm256_permute2f128_ps(a, b, 0x20);

    // (A a B b | C c D d) (E e F f | G g H h) -> (C c D d | G g H h)
    auto secondHalves = _mm256_permute2f128_ps(a, b, 0x31);

    return
      {
        // (A a B b | E e F f) (C c D d | G g H h) -> (A B C D | E F G H)
        _mm256_shuffle_ps(firstHalves, secondHalves, _MM_SHUFFLE(2, 0, 2, 0)),

        // (A a B b | E e F f) (C c D d | G g H h) -> (a b c d | e f g h)
        _mm256_shuffle_ps(firstHalves, secondHalves, _MM_SHUFFLE(3, 1, 3, 1)),
      };
  }
  else
  {
    // (A a | B b) (C c | D d) -> (A a | C c)
    auto firstHalves = _mm256_permute2f128_pd(a, b, 0x20);

    // (A a | B b) (C c | D d) -> (B b | D d)
    auto secondHalves = _mm256_permute2f128_pd(a, b, 0x31);

    return
      {
        // (A a | C c) (B b | D d) -> (A B | C D)
        _mm256_shuffle_pd(firstHalves, secondHalves, 0b0'0'0'0),

        // (A a | C c) (B b | D d) -> (a b | c d)
        _mm256_shuffle_pd(firstHalves, secondHalves, 0b1'1'1'1),
      };
  }
}

static auto LoadDeinterleavedHalf(const float *ptr) -> Pair<__m128, __m128>
{
  auto t0 = _mm_loadu_ps(ptr);
  auto t1 = _mm_loadu_ps(ptr + 4u);
  return { _mm_shuffle_ps(t0, t1, _MM_SHUFFLE(2, 0, 2, 0)), _mm_shuffle_ps(t0, t1, _MM_SHUFFLE(3, 1, 3, 1)) };
}


static auto StoreHalves(float *ptr, size_t offset, __m256 v)
  { _mm256_storeu2_m128(ptr + offset, ptr, v); }

template <std::floating_point F>
static void Store4x4Transposed(F *ptrA, F *ptrB, F *ptrC, F *ptrD, Vec<F> a, Vec<F> b, Vec<F> c, Vec<F> d)
{
  if constexpr (f32<F>)
  {
    // 4x4 transform on both halves of the AVX vector at once (this is the _MM_TRANSPOSE4 macro but with _mm256 instead)
    auto t0 = _mm256_shuffle_ps(a, b, 0x44);
    auto t2 = _mm256_shuffle_ps(a, b, 0xEE);
    auto t1 = _mm256_shuffle_ps(c, d, 0x44);
    auto t3 = _mm256_shuffle_ps(c, d, 0xEE);

    a = _mm256_shuffle_ps(t0, t1, 0x88);
    b = _mm256_shuffle_ps(t0, t1, 0xDD);
    c = _mm256_shuffle_ps(t2, t3, 0x88);
    d = _mm256_shuffle_ps(t2, t3, 0xDD);

    _mm256_storeu_ps(ptrA, a);
    _mm256_storeu_ps(ptrB, b);
    _mm256_storeu_ps(ptrC, c);
    _mm256_storeu_ps(ptrD, d);
  }
  else
  {
    // For f64 we have to transpose cross-lane, start by swapping all the 2x2s
    auto t0 = _mm256_unpacklo_pd(a, b);
    auto t1 = _mm256_unpackhi_pd(a, b);
    auto t2 = _mm256_unpacklo_pd(c, d);
    auto t3 = _mm256_unpackhi_pd(c, d);

    // Now swap the upper-right and lower-left 2x2 chunks
    a = _mm256_permute2f128_pd(t0, t2, 0x20);
    b = _mm256_permute2f128_pd(t1, t3, 0x20);
    c = _mm256_permute2f128_pd(t0, t2, 0x31);
    d = _mm256_permute2f128_pd(t1, t3, 0x31);

    _mm256_storeu_pd(ptrA, a);
    _mm256_storeu_pd(ptrB, b);
    _mm256_storeu_pd(ptrC, c);
    _mm256_storeu_pd(ptrD, d);
  }
}

static void Store4x4Transposed(float *ptrA, float *ptrB, float *ptrC, float *ptrD, __m128 a, __m128 b, __m128 c, __m128 d)
{
  _MM_TRANSPOSE4_PS(a, b, c, d);
  _mm_storeu_ps(ptrA, a);
  _mm_storeu_ps(ptrB, b);
  _mm_storeu_ps(ptrC, c);
  _mm_storeu_ps(ptrD, d);
}

template <std::floating_point F>
static void Store(F *ptr, Vec<F> a)
{
  if constexpr (f32<F>)
    { _mm256_storeu_ps(ptr, a); }
  else
    { _mm256_storeu_pd(ptr, a); }
}

// Need a half-store for f32s
static void Store(float *ptr, __m128 a)
{
  _mm_storeu_ps(ptr, a);
}

template <std::floating_point F>
static void StoreInterleaved(F *ptr, Vec<F> a, Vec<F> b)
{
  if constexpr (f32<F>)
  {
    // (A B C D E F G H) (a b c d e f g h) -> (A a B b E e F f)
    auto loHalves = _mm256_unpacklo_ps(a, b);

    // (A B C D E F G H) (a b c d e f g h) -> (C c D d G g H h)
    auto hiHalves = _mm256_unpackhi_ps(a, b);

    // (A a B b E e F f) (C c D d G g H h) -> (A a B b C c D d)
    a = _mm256_permute2f128_ps(loHalves, hiHalves, 0x20);

    // (A a B b E e F f) (C c D d G g H h) -> (E e F f G g H h)
    b = _mm256_permute2f128_ps(loHalves, hiHalves, 0x31);

    _mm256_storeu_ps(ptr, a);
    _mm256_storeu_ps(ptr + 8, b);
  }
  else
  {
    // (A B C D) (a b c d) -> (A a C c)
    auto loHalves = _mm256_unpacklo_pd(a, b);

    // (A B C D) (a b c d) -> (B b D d)
    auto hiHalves = _mm256_unpackhi_pd(a, b);

    // (A a C c) (B b D d) -> (A a B b)
    a = _mm256_permute2f128_pd(loHalves, hiHalves, 0x20);

    // (A a C c) (B b D d) -> (C c D d)
    b = _mm256_permute2f128_pd(loHalves, hiHalves, 0x31);

    _mm256_storeu_pd(ptr, a);
    _mm256_storeu_pd(ptr + 4, b);
  }
}

static void StoreInterleaved(float *ptr, __m128 a, __m128 b)
{
  auto aa = _mm_unpacklo_ps(a, b);
  auto bb = _mm_unpackhi_ps(a, b);
  _mm_storeu_ps(ptr, aa);
  _mm_storeu_ps(ptr + 4, bb);
}

template <typename V>
static auto Set1(std::conditional_t<f32<V>, float, double> f) -> V
{
  if constexpr (f32<V>)
  {
    if constexpr (std::same_as<V, __m128>)
      { return _mm_set1_ps(f); }
    else
      { return _mm256_set1_ps(f); }
  }
  else
    { return _mm256_set1_pd(f); }
}

template <simd_vec V>
static V Add(V a, V b)
{
  if constexpr (f32<V>)
    { return _mm256_add_ps(a, b); }
  else
    { return _mm256_add_pd(a, b); }
}

static __m128 Add(__m128 a, __m128 b)
  { return _mm_add_ps(a, b); }

template <simd_vec V>
static V Sub(V a, V b)
{
  if constexpr (f32<V>)
    { return _mm256_sub_ps(a, b); }
  else
    { return _mm256_sub_pd(a, b); }
}

static __m128 Sub(__m128 a, __m128 b)
  { return _mm_sub_ps(a, b); }

template <simd_vec V>
static V Mul(V a, V b)
{
  if constexpr (f32<V>)
    { return _mm256_mul_ps(a, b); }
  else
    { return _mm256_mul_pd(a, b); }
}

#if SIMD_FMA
  template <simd_vec V>
  static V FMAdd(V a, V b, V c)
  {
    if constexpr (f32<V>)
      { return _mm256_fmadd_ps(a, b, c); }
    else
      { return _mm256_fmadd_pd(a, b, c); }
  }

  static __m128 FMAdd(__m128 a, __m128 b, __m128 c)
    { return _mm_fmadd_ps(a, b, c); }

  template <simd_vec V>
  static V FMSub(V a, V b, V c)
  {
    if constexpr (f32<V>)
      { return _mm256_fmsub_ps(a, b, c); }
    else
      { return _mm256_fmsub_pd(a, b, c); }
  }

  static __m128 FMSub(__m128 a, __m128 b, __m128 c)
    { return _mm_fmsub_ps(a, b, c); }

  template <simd_vec V>
  static V FNMAdd(V a, V b, V c) // -(a * b) + c
  {
    if constexpr (f32<V>)
      { return _mm256_fnmadd_ps(a, b, c); }
    else
      { return _mm256_fnmadd_pd(a, b, c); }
  }

  static __m128 FNMAdd(__m128 a, __m128 b, __m128 c)
    { return _mm_fnmadd_ps(a, b, c); }

  template <typename V>
  static V MAdd(V a, V b, V c)
    { return FMAdd(a, b, c); }

  template <typename V>
  static V NMAdd(V a, V b, V c) // -(a * b) + c
    { return FNMAdd(a, b, c); }
#else
  static __m128 Mul(__m128 a, __m128 b)
    { return _mm_mul_ps(a, b); }

  template <typename V>
  static V MAdd(V a, V b, V c)
    { return Add(Mul(a, b), c); }

  template <typename V>
  static V NMAdd(V a, V b, V c)
    { return Sub(c, Mul(a, b)); }
#endif

template <std::floating_point F, std::same_as<Vec<F>> ... Vs>
static void Scale(F scale, Vs &...vs)
{
  if constexpr (f32<F>)
  {
    auto s = _mm256_set1_ps(scale);
    ((vs = _mm256_mul_ps(s, vs)), ...);
  }
  else
  {
    auto s = _mm256_set1_pd(scale);
    ((vs = _mm256_mul_pd(s, vs)), ...);
  }
}

template <std::floating_point F, std::same_as<__m128> ... Vs>
static void Scale(F scale, Vs &...vs)
{
  auto s = _mm_set1_ps(scale);
  ((vs = _mm_mul_ps(s, vs)), ...);
}


// Forward Real Transform /////////////////////////////////////////////////////////////////////////////////////////////

template <typename F>
inline auto LoadFirstRightSideForForwardRealTransform(const F *input, Vec<F> &prevR, Vec<F> &prevI) ->
  Pair<Vec<F>, Vec<F>>
{
  if constexpr (f32<Vec<F>>)
  {
    // Variable naming: the _ denotes the break in the AVX lanes, and capital R and I are for real, imaginary
    //  'X' is used in variables where we don't care about what's in that position.

    // target: X b c d | e f g h

    // load: i h g f | e d c b
    auto ihgf_edcb_R = _mm256_loadu_ps(input);
    auto ihgf_edcb_I = _mm256_loadu_ps(input + 8u);

    // We need the "i" from these in the next iteration so store them
    prevR = ihgf_edcb_R;
    prevI = ihgf_edcb_I;

    // Swap the sides
    auto Xdcb_Xhgf_R = _mm256_permute2f128_ps(ihgf_edcb_R, ihgf_edcb_R, 0b0000'0001);
    auto Xdcb_Xhgf_I = _mm256_permute2f128_ps(ihgf_edcb_I, ihgf_edcb_I, 0b0000'0001);

    // Now reverse the 3 values we care about in each half
    auto Xbcd_Xfgh_R = _mm256_shuffle_ps(Xdcb_Xhgf_R, Xdcb_Xhgf_R, _MM_SHUFFLE(1, 2, 3, 0));
    auto Xbcd_Xfgh_I = _mm256_shuffle_ps(Xdcb_Xhgf_I, Xdcb_Xhgf_I, _MM_SHUFFLE(1, 2, 3, 0));

    // Finally, blend in the "e" from the loaded value
    return
      {
        _mm256_blend_ps(ihgf_edcb_R, Xbcd_Xfgh_R, 0b11101110),
        _mm256_blend_ps(ihgf_edcb_I, Xbcd_Xfgh_I, 0b11101110),
      };
  }
  else
  {
    // load: e d | c b
    // target: X b | c d
    auto ed_cb_R = _mm256_loadu_pd(input);
    auto ed_cb_I = _mm256_loadu_pd(input + 4u);
    prevR = ed_cb_R;
    prevI = ed_cb_I;
    auto Xb_Xd_R = _mm256_permute2f128_pd(ed_cb_R, ed_cb_R, 0b0000'0001);
    auto Xb_Xd_I = _mm256_permute2f128_pd(ed_cb_I, ed_cb_I, 0b0000'0001);

    return
      {
        _mm256_blend_pd(ed_cb_R, Xb_Xd_R, 0b1010),
        _mm256_blend_pd(ed_cb_I, Xb_Xd_I, 0b1010),
      };
  }
}

template <typename F>
inline auto LoadRightSideForForwardRealTransform(const F *input, Vec<F> &prevR, Vec<F> &prevI) ->
  Pair<Vec<F>, Vec<F>>
{
  if constexpr (f32<Vec<F>>)
  {
    // This loads the next-earliest 8 complex elements from the list from input, reverses it, then shifts one element
    //  in from prev

    // target: i j k l | m n o p

    // prev = i h g f | e d c b
    // load   q p o n | m l k j
    auto qpon_mlkj_R = _mm256_loadu_ps(input);
    auto qpon_mlkj_I = _mm256_loadu_ps(input + 8u);

    // Swap halves: m l k j | q p o n
    auto Xklj_Xpon_R = _mm256_permute2f128_ps(qpon_mlkj_R, qpon_mlkj_R, 0b0000'0001);
    auto Xklj_Xpon_I = _mm256_permute2f128_ps(qpon_mlkj_I, qpon_mlkj_I, 0b0000'0001);

    // Combine the lower half of prev with the upper half of what we loaded (this gets 'i' and 'm' into position in
    //  the same vector to blend in to complete the shifted value)
    //  i h g f | m l k j
    auto iXXX_mXXX_R = _mm256_blend_ps(prevR, qpon_mlkj_R, 0xF0);
    auto iXXX_mXXX_I = _mm256_blend_ps(prevI, qpon_mlkj_I, 0xF0);

    // Stash the raw values we loaded for the next go-around, we'll need the 'q' value from it
    prevR = qpon_mlkj_R;
    prevI = qpon_mlkj_I;

    // Rearrange the half-swapped values (keep first element of each lane in place and reverse the other 3 so they're
    //  in order)
    //  m j k l | q n o p
    auto Xjkl_Xnop_R = _mm256_shuffle_ps(Xklj_Xpon_R, Xklj_Xpon_R, _MM_SHUFFLE(1, 2, 3, 0));
    auto Xjkl_Xnop_I = _mm256_shuffle_ps(Xklj_Xpon_I, Xklj_Xpon_I, _MM_SHUFFLE(1, 2, 3, 0));

    // Finally blend in the lowest element from our combined prev into each lane:
    //  i j k l | m n o p
    return
      {
        _mm256_blend_ps(iXXX_mXXX_R, Xjkl_Xnop_R, 0b11101110),
        _mm256_blend_ps(iXXX_mXXX_I, Xjkl_Xnop_I, 0b11101110),
      };
  }
  else
  {
    // prev:   e d | c b
    // cur:    i h | g f
    // target: e f | g h
    auto ih_gf_R = _mm256_loadu_pd(input);
    auto ih_gf_I = _mm256_loadu_pd(input + 4u);

    auto Xf_Xh_R = _mm256_permute2f128_pd(ih_gf_R, ih_gf_R, 0b0000'0001);
    auto Xf_Xh_I = _mm256_permute2f128_pd(ih_gf_I, ih_gf_I, 0b0000'0001);
    auto eX_gX_R = _mm256_blend_pd(prevR, ih_gf_R, 0b0100);
    auto eX_gX_I = _mm256_blend_pd(prevI, ih_gf_I, 0b0100);

    prevR = ih_gf_R;
    prevI = ih_gf_I;

    return
      {
        _mm256_blend_pd(eX_gX_R, Xf_Xh_R, 0b1010),
        _mm256_blend_pd(eX_gX_I, Xf_Xh_I, 0b1010),
      };
  }
}

template <typename F>
inline void StoreRightSideForForwardRealTransform(F *output, Vec<F> r1, Vec<F> i1, Vec<F> &prevR, Vec<F> &prevI)
{
  if constexpr (f32<F>)
  {
    // prev: A B C D | E F G H || a b c d | e f g h  (REAL, imaginary)
    // cur:  I J K L | M N O P || i j k l | m n o p

    // target: I i H h | G g F f || E e D d | C c B b

    // blend:
    //  I B C D | E F G H || i b c d | e f g h
    auto ibcd_efgh_R = _mm256_blend_ps(r1, prevR, 0b11111110);
    auto ibcd_efgh_I = _mm256_blend_ps(i1, prevI, 0b11111110);

    // Now keep the values computed in this loop for next time (we're using 'i' this time but we'll need 'jklmnop'
    //  next
    prevR = r1;
    prevI = i1;

    // Interleave:
    //  I i B b | E e F f || C c D d | G g H h
    //  (now we don't have real/imaginary separate vectors so using the letters to denote the real, imag pairs
    auto ib_ef = _mm256_unpacklo_ps(ibcd_efgh_R, ibcd_efgh_I);
    auto cd_gh = _mm256_unpackhi_ps(ibcd_efgh_R, ibcd_efgh_I);

    // Blend: C c B b | G g F f
    auto cb_gf = _mm256_blend_ps(cd_gh, ib_ef, 0b11001100);

    // Blend: A a D d | E e H h
    auto id_eh = _mm256_blend_ps(ib_ef, cd_gh, 0b11001100);

    // Perm: E e H h | C c B b
    auto eh_cb = _mm256_permute2f128_ps(cb_gf, id_eh, 0b0000'0011);

    auto ib_gf = _mm256_blend_ps(id_eh, cb_gf, 0b11111100);

    // Blend: E e D d | C c B b  :D
    auto ed_cb = _mm256_blend_ps(eh_cb, cd_gh, 0b00001100);

    // Blend: I i H h | G g F f  :D
    auto ih_gf = _mm256_blend_ps(eh_cb, ib_gf, 0b11110011);

    _mm256_storeu_ps(output, ih_gf);
    _mm256_storeu_ps(output + 8u, ed_cb);
  }
  else
  {
    // prev: A B | C D || a b | c d
    // cur: E F | G H || e f | g h
    // target: E e | D d || C c B b

    auto XXCc = _mm256_unpacklo_pd(prevR, prevI);
    auto BbDd = _mm256_unpackhi_pd(prevR, prevI);
    prevR = r1;
    prevI = i1;

    auto EeXX = _mm256_unpacklo_pd(r1, i1);
    auto CcBb = _mm256_permute2f128_pd(XXCc, BbDd, 0b0010'0001);
    auto EeDd = _mm256_blend_pd(EeXX, BbDd, 0b1100);
    _mm256_storeu_pd(output, EeDd);
    _mm256_storeu_pd(output + 4u, CcBb);
  }
}

template <typename F>
inline void StoreFinalRightSideForForwardRealTransform(F *output, Vec<F> r, Vec<F> i)
{
  if constexpr (f32<F>)
  {
    // cur: A B C D | E F G H || a b c d | e f g h (REAL, imaginary)

    // Target:
    //  X x H h | G g F f || E e D d | C c B b

    // Interleave:
    //  A a B b | E e F f || C c D d | G g H h
    auto ab_ef = _mm256_unpacklo_ps(r, i);
    auto cd_gh = _mm256_unpackhi_ps(r, i);

    // Blend: C c B b | G g F f
    auto cb_gf = _mm256_blend_ps(cd_gh, ab_ef, 0b11001100);

    // Blend: A a D d | E e H h
    auto ad_eh = _mm256_blend_ps(ab_ef, cd_gh, 0b11001100);

    // Perm: E e H h | C c B b
    auto eh_cb = _mm256_permute2f128_ps(cb_gf, ad_eh, 0b0000'0011);

    // Blend: E e H h | G g F f  :D
    auto Xh_gf = _mm256_blend_ps(eh_cb, cb_gf, 0xF0);

    // Blend: E e D d | C c B b  :D
    auto ed_cb = _mm256_blend_ps(eh_cb, cd_gh, 0b00001100);

    _mm256_storeu_ps(output, Xh_gf);
    _mm256_storeu_ps(output + 8u, ed_cb);
  }
  else
  {
    // Cur:    A B | C D || a b | c d
    // Target: X x | D d || C c | B b
    auto BbDd = _mm256_unpackhi_pd(r, i);
    auto AaCc = _mm256_unpacklo_pd(r, i);
    auto CcBb = _mm256_permute2f128_pd(BbDd, AaCc, 0b0000'0011);
    _mm256_storeu_pd(output, BbDd);
    _mm256_storeu_pd(output + 4u, CcBb);
  }
}


// Inverse Real Transform /////////////////////////////////////////////////////////////////////////////////////////////

template <typename F>
inline auto LoadFirstRightSideForInverseRealTransform(const F *input, Vec<F> &prev, Vec<F> &)  // only need one prev
  -> Pair<Vec<F>, Vec<F>>
{
  if constexpr (f32<Vec<F>>)
  {
    // Load:   I i H h | G g F f || E e D d | C c B b
    // Target: X B C D | E F G H || X b c d | e f g h

    auto IiHh_GgFf = _mm256_loadu_ps(input);
    prev = IiHh_GgFf;
    auto EeDd_CcBb = _mm256_loadu_ps(input + 8u);

    auto CcBb_EeDd = _mm256_permute2f128_ps(EeDd_CcBb, EeDd_CcBb, 0b0000'0001);
    auto EeDd_IiHh = _mm256_insertf128_ps(EeDd_CcBb, _mm256_castps256_ps128(IiHh_GgFf), 1);

    auto CcBb_GgFf = _mm256_blend_ps(CcBb_EeDd, IiHh_GgFf, 0b11110000);
    auto XXBb_EeFf = _mm256_blend_ps(IiHh_GgFf, CcBb_EeDd, 0b00111100);
    auto CcDd_GgHh = _mm256_blend_ps(CcBb_GgFf, EeDd_IiHh, 0b11001100);

    auto XBCD_EFGH = _mm256_shuffle_ps(XXBb_EeFf, CcDd_GgHh, _MM_SHUFFLE(2, 0, 2, 0));
    auto Xbcd_efgh = _mm256_shuffle_ps(XXBb_EeFf, CcDd_GgHh, _MM_SHUFFLE(3, 1, 3, 1));
    return { XBCD_EFGH, Xbcd_efgh };
  }
  else
  {
    // Cur: E e | D d || C c | B b
    // Target: X B | C D || X b | c d
    auto Ee_Dd = _mm256_loadu_pd(input);
    auto Cc_Bb = _mm256_loadu_pd(input + 4u);

    prev = Ee_Dd;

    auto Bb_Cc = _mm256_permute2f128_pd(Cc_Bb, Cc_Bb, 0b0000'0001);
    auto XB_DC = _mm256_shuffle_pd(Ee_Dd, Bb_Cc, 0b0000);
    auto Xb_dc = _mm256_shuffle_pd(Ee_Dd, Bb_Cc, 0b1111);
    auto XB_CD = _mm256_shuffle_pd(XB_DC, XB_DC, 0b0110);
    auto Xb_cd = _mm256_shuffle_pd(Xb_dc, Xb_dc, 0b0110);
    return { XB_CD, Xb_cd };
  }
}

template <typename F>
inline auto LoadRightSideForInverseRealTransform(const F *input, Vec<F> &prev, Vec<F> &)  // only need one prev
  -> Pair<Vec<F>, Vec<F>>
{
  if constexpr (f32<Vec<F>>)
  {
    // Prev: I i H h | G g F f
    // Cur: Q q P p | O o N n || M m L l | K k J j
    // Target: I J K L | M N O P || i j k l | m n o p
    auto QqPp_OoNn = _mm256_loadu_ps(input);
    auto MmLl_KkJj = _mm256_loadu_ps(input + 8u);

    auto KkJj_MmLl = _mm256_permute2f128_ps(MmLl_KkJj, MmLl_KkJj, 0b0000'0001);
    auto MmLl_QqPp = _mm256_insertf128_ps(MmLl_KkJj, _mm256_castps256_ps128(QqPp_OoNn), 1);
    auto IiHh_OoNn = _mm256_blend_ps(prev, QqPp_OoNn, 0b11110000);
    prev = QqPp_OoNn;

    auto KkJj_OoNn = _mm256_blend_ps(KkJj_MmLl, QqPp_OoNn, 0b11110000);
    auto IiJj_MmNn = _mm256_blend_ps(IiHh_OoNn, KkJj_MmLl, 0b00111100);
    auto KkLl_OoPp = _mm256_blend_ps(KkJj_OoNn, MmLl_QqPp, 0b11001100);

    auto IJKL_MNOP = _mm256_shuffle_ps(IiJj_MmNn, KkLl_OoPp, _MM_SHUFFLE(2, 0, 2, 0));
    auto ijkl_mnop = _mm256_shuffle_ps(IiJj_MmNn, KkLl_OoPp, _MM_SHUFFLE(3, 1, 3, 1));
    return { IJKL_MNOP, ijkl_mnop };
  }
  else
  {
    // Prev: E e | D d
    // Cur: I i | H h || G g | F f
    // Target: E F | G H || e f | g h
    auto Ii_Hh = _mm256_loadu_pd(input);
    auto Gg_Ff = _mm256_loadu_pd(input + 4u);
    auto Ff_Gg = _mm256_permute2f128_pd(Gg_Ff, Gg_Ff, 0b0000'0001);
    auto Ee_Hh = _mm256_blend_pd(prev, Ii_Hh, 0b1100);
    prev = Ii_Hh;

    auto EF_HG = _mm256_shuffle_pd(Ee_Hh, Ff_Gg, 0b0000);
    auto ef_hg = _mm256_shuffle_pd(Ee_Hh, Ff_Gg, 0b1111);
    auto EF_GH = _mm256_shuffle_pd(EF_HG, EF_HG, 0b0110);
    auto ef_gh = _mm256_shuffle_pd(ef_hg, ef_hg, 0b0110);
    return { EF_GH, ef_gh };
  }
}

template <typename F>
inline void StoreRightSideForInverseRealTransform(F *output, Vec<F> r1, Vec<F> i1, Vec<F> &prevR, Vec<F> &prevI)
{
  if constexpr (f32<F>)
  {
    // Prev: ABCD|EFGH
    // Cur: IJKL|MNOP
    // Target: IHGF|EDCB
    auto Xfgh_Xbcd_R = _mm256_permute2f128_ps(prevR, prevR, 0b0000'0001);
    auto Xfgh_Xbcd_I = _mm256_permute2f128_ps(prevI, prevI, 0b0000'0001);

    auto iXXX_eXXX_R = _mm256_blend_ps(r1, prevR, 0b1111'0000);
    auto iXXX_eXXX_I = _mm256_blend_ps(i1, prevI, 0b1111'0000);
    prevR = r1;
    prevI = i1;

    auto Xhgf_Xdcb_R = _mm256_shuffle_ps(Xfgh_Xbcd_R, Xfgh_Xbcd_R, _MM_SHUFFLE(1, 2, 3, 0));
    auto Xhgf_Xdcb_I = _mm256_shuffle_ps(Xfgh_Xbcd_I, Xfgh_Xbcd_I, _MM_SHUFFLE(1, 2, 3, 0));

    auto ihgf_edcb_R = _mm256_blend_ps(iXXX_eXXX_R, Xhgf_Xdcb_R, 0b11101110);
    auto ihgf_edcb_I = _mm256_blend_ps(iXXX_eXXX_I, Xhgf_Xdcb_I, 0b11101110);

    _mm256_storeu_ps(output, ihgf_edcb_R);
    _mm256_storeu_ps(output + 8u, ihgf_edcb_I);
  }
  else
  {
    // Prev: AB|CD
    // Cur: EF|GH
    // Target: ED|CB
    auto cd_ab_R = _mm256_permute2f128_pd(prevR, prevR, 0b0000'0001);
    auto cd_ab_I = _mm256_permute2f128_pd(prevI, prevI, 0b0000'0001);

    auto ef_cd_R = _mm256_blend_pd(r1, prevR, 0b1100);
    auto ef_cd_I = _mm256_blend_pd(i1, prevI, 0b1100);

    prevR = r1;
    prevI = i1;

    auto ed_cb_R = _mm256_shuffle_pd(ef_cd_R, cd_ab_R, 0b1010);
    auto ed_cb_I = _mm256_shuffle_pd(ef_cd_I, cd_ab_I, 0b1010);

    _mm256_storeu_pd(output, ed_cb_R);
    _mm256_storeu_pd(output + 4u, ed_cb_I);
  }
}

template <typename F>
inline void StoreFinalRightSideForInverseRealTransform(F *output, Vec<F> r1, Vec<F> i1)
{
  if constexpr (f32<F>)
  {
    // Cur: ABCD|EFGH
    // Target: XHGF|EDCB
    auto Xfgh_Xbcd_R = _mm256_permute2f128_ps(r1, r1, 0b0000'0001);
    auto Xfgh_Xbcd_I = _mm256_permute2f128_ps(i1, i1, 0b0000'0001);

    auto Xhgf_Xdcb_R = _mm256_shuffle_ps(Xfgh_Xbcd_R, Xfgh_Xbcd_R, _MM_SHUFFLE(1, 2, 3, 0));
    auto Xhgf_Xdcb_I = _mm256_shuffle_ps(Xfgh_Xbcd_I, Xfgh_Xbcd_I, _MM_SHUFFLE(1, 2, 3, 0));

    auto Xhgf_edcb_R = _mm256_blend_ps(r1, Xhgf_Xdcb_R, 0b11101110);
    auto Xhgf_edcb_I = _mm256_blend_ps(i1, Xhgf_Xdcb_I, 0b11101110);

    _mm256_storeu_ps(output, Xhgf_edcb_R);
    _mm256_storeu_ps(output + 8u, Xhgf_edcb_I);
  }
  else
  {
    // Cur: AB|CD
    // Target: XD|CB

    auto cd_ab_R = _mm256_permute2f128_pd(r1, r1, 0b0000'0001);
    auto cd_ab_I = _mm256_permute2f128_pd(i1, i1, 0b0000'0001);

    auto Xd_cb_R = _mm256_shuffle_pd(r1, cd_ab_R, 0b1010);
    auto Xd_cb_I = _mm256_shuffle_pd(i1, cd_ab_I, 0b1010);

    _mm256_storeu_pd(output, Xd_cb_R);
    _mm256_storeu_pd(output + 4u, Xd_cb_I);
  }
}


#include "fft_20xx_run.h"