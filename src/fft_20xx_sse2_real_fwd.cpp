#ifdef _MSC_VER
  #pragma warning (disable: 4711) // function <x> selected for automatic inline expansion
#endif

#include "fft_20xx_common.h"
#include "fft_20xx_table.h"
#include "fft_20xx_tiny_real.h"

static_assert(CPU_X86 || CPU_X64);

static_assert(SIMD_SSE2);

#if COMPILER_GCC
  #pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

namespace FFT20XX::SSE2
{
  template <std::floating_point F>
  bool IsSIMDAligned(const F *ptr)
    { return (uintptr_t(ptr) & (sizeof(__m128) - 1)) == 0; }

  constexpr auto MinimumRealFFTLength = 32;

  constexpr auto VecSize = 4u;

  template <Alignment A>
  [[nodiscard]] FORCE_INLINE auto Load(const float *p)
  {
    if constexpr (A == Alignment::Aligned)
      { return _mm_load_ps(p); }
    else
      { return _mm_loadu_ps(p); }
  }



  template <Alignment A>
  [[nodiscard]] FORCE_INLINE auto Store(float *p, __m128 v)
  {
    if constexpr (A == Alignment::Aligned)
      { return _mm_store_ps(p, v); }
    else
      { return _mm_storeu_ps(p, v); }
  }



  [[nodiscard]] FORCE_INLINE auto Set1(float v)
  {
    return _mm_set1_ps(v);
  }



  [[nodiscard]] FORCE_INLINE auto Mul(auto a, auto b)
  {
    return _mm_mul_ps(a, b);
  }



  template <Alignment A, size_t N>
  [[nodiscard]] FORCE_INLINE auto LoadStridedSets(const float *base, size_t stride)
  {
    return [&]<size_t... I>(std::index_sequence<I...>)
      {
        return std::tuple{Load<A>(base + I * stride)...};
      }(std::make_index_sequence<N>());
  }



  template <Alignment A, typename...V>
  FORCE_INLINE void StoreStridedSets(float *base, size_t stride, V...v)
  {
    auto tup = std::tuple{v...};
    [&]<size_t... I>(std::index_sequence<I...>)
      {
        (Store<A>(base + I * stride, std::get<I>(tup)), ...);
      }(std::index_sequence_for<V...>());
  }



  FORCE_INLINE void DITRealInputRadix2Pair(auto &a, auto &b, auto &c, auto &d)
  {
    auto t = _mm_add_ps(a, c);
    auto t2 = _mm_add_ps(b, d);
    c = _mm_sub_ps(a, c);
    d = _mm_sub_ps(d, b);
    a = t;
    b = t2;
  }



  FORCE_INLINE void DITRealOutputRadix2PairInv(auto &a, auto &b, auto &c, auto &d)
  {
    auto t = _mm_add_ps(a, c);
    auto t2 = _mm_sub_ps(b, d);
    c = _mm_sub_ps(a, c);
    d = _mm_add_ps(d, b);
    a = t;
    b = t2;
  }



  FORCE_INLINE void DITComplexRadix2(auto &aR, auto &aI, auto &bR, auto &bI, auto wR, auto wI)
  {
    // Rotate b * w
    auto t1 = _mm_mul_ps(bR, wR);
    auto t2 = _mm_mul_ps(bI, wI);
    bR = _mm_mul_ps(bR, wI);
    bI = _mm_mul_ps(bI, wR);
    t1 = _mm_sub_ps(t1, t2);
    bI = _mm_add_ps(bR, bI);
    bR = t1;

    t1 = _mm_add_ps(aR, bR);
    t2 = _mm_add_ps(aI, bI);
    bR = _mm_sub_ps(aR, bR);
    bI = _mm_sub_ps(bI, aI);
    aR = t1;
    aI = t2;
  }



  FORCE_INLINE void DIFComplexRadix2Inv(auto &aR, auto &aI, auto &bR, auto &bI, auto wR, auto wI)
  {
    auto t1 = _mm_add_ps(aR, bR);
    auto t2 = _mm_sub_ps(aI, bI);
    bR = _mm_sub_ps(aR, bR);
    bI = _mm_add_ps(aI, bI);
    aR = t1;
    aI = t2;

    // Rotate b * conj(w)
    t1 = _mm_mul_ps(bR, wR);
    t2 = _mm_mul_ps(bI, wI);
    bR = _mm_mul_ps(bR, wI);
    bI = _mm_mul_ps(bI, wR);
    t1 = _mm_add_ps(t1, t2);
    bI = _mm_sub_ps(bI, bR);
    bR = t1;
  }



  FORCE_INLINE void DITRealInputRadix4Pair(auto &a, auto &b, auto &c, auto &d, auto &e, auto &f, auto &g, auto &h, auto invSqrt2)
  {
    __m128 s;
    if constexpr (std::floating_point<decltype(invSqrt2)>)
    {
      s = _mm_set1_ps(invSqrt2);
    }
    else
    {
      s = invSqrt2;
    }

    auto t = _mm_add_ps(a, e);
    e = _mm_sub_ps(a, e);
    a = t;

    t = _mm_add_ps(b, f);
    f = _mm_sub_ps(b, f);
    b = t;

    t = _mm_add_ps(g, c);
    g = _mm_sub_ps(g, c);
    c = t;

    t = _mm_add_ps(h, d);
    h = _mm_sub_ps(h, d);
    d = t;

    // Second radix 2
    t = _mm_add_ps(a, c);
    c = _mm_sub_ps(a, c);
    a = t;

    t = _mm_add_ps(d, b);
    d = _mm_sub_ps(d, b);
    b = t;

    t = _mm_mul_ps(s, _mm_add_ps(f, h));
    h = _mm_mul_ps(s, _mm_sub_ps(h, f));
    f = t;

    t = _mm_sub_ps(e, f);
    e = _mm_add_ps(e, f);
    f = _mm_add_ps(h, g);
    h = _mm_sub_ps(h, g);
    g = t;
  }



  void DIFRealOutputRadix4PairInv(auto &a, auto &b, auto &c, auto &d, auto &e, auto &f, auto &g, auto &h, auto invSqrt2)
  {
    __m128 s;
    if constexpr (std::floating_point<decltype(invSqrt2)>)
    {
      s = _mm_set1_ps(invSqrt2);
    }
    else
    {
      s = invSqrt2;
    }

    auto t = _mm_sub_ps(f, h);
    h = _mm_add_ps(f, h);
    f = _mm_sub_ps(e, g);
    e = _mm_add_ps(e, g);
    g = t;

    t = _mm_mul_ps(s, _mm_sub_ps(f, h));
    h = _mm_mul_ps(s, _mm_add_ps(f, h));
    f = t;

    t = _mm_add_ps(a, c);
    c = _mm_sub_ps(a, c);
    a = t;

    t = _mm_sub_ps(b, d);
    d = _mm_add_ps(b, d);
    b = t;

    t = _mm_add_ps(a, e);
    e = _mm_sub_ps(a, e);
    a = t;

    t = _mm_add_ps(b, f);
    f = _mm_sub_ps(b, f);
    b = t;

    t = _mm_sub_ps(c, g);
    g = _mm_add_ps(c, g);
    c = t;

    t = _mm_sub_ps(d, h);
    h = _mm_add_ps(d, h);
    d = t;
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
    auto w0r,
    auto w0i,
    auto w1r,
    auto w1i,
    auto w2r,
    auto w2i)
  {
    // Rotate b by w0, c by w1, and d by w2
    __m128 wr, wi;
    if constexpr (std::floating_point<decltype(w1r)>)
    {
      // Rotate c by w1
      wr = _mm_set1_ps(w1r);
      wi = _mm_set1_ps(w1i);
    }
    else
    {
      wr = w1r;
      wi = w1i;
    }
    auto t1 = _mm_mul_ps(cR, wr);
    auto t2 = _mm_mul_ps(cI, wi);
    cR = _mm_mul_ps(cR, wi);
    cI = _mm_mul_ps(cI, wr);
    t1 = _mm_sub_ps(t1, t2);
    cI = _mm_add_ps(cR, cI);
    cR = t1;

    if constexpr (std::floating_point<decltype(w0r)>)
    {
      wr = _mm_set1_ps(w0r);
      wi = _mm_set1_ps(w0i);
    }
    else
    {
      wr = w0r;
      wi = w0i;
    }

    // Do the a/c butterfly while that loads
    t1 = _mm_add_ps(aR, cR);
    t2 = _mm_add_ps(aI, cI);
    cR = _mm_sub_ps(aR, cR);
    cI = _mm_sub_ps(aI, cI);
    aR = t1;
    aI = t2;

    // Rotate b by w0
    t1 = _mm_mul_ps(bR, wr);
    t2 = _mm_mul_ps(bI, wi);
    bR = _mm_mul_ps(bR, wi);
    bI = _mm_mul_ps(bI, wr);

      // Load w2 in the middle of this
    if constexpr (std::floating_point<decltype(w2r)>)
    {
      wr = _mm_set1_ps(w2r);
      wi = _mm_set1_ps(w2i);
    }
    else
    {
      wr = w2r;
      wi = w2i;
    }
    t1 = _mm_sub_ps(t1, t2);
    bI = _mm_add_ps(bR, bI);
    bR = t1;

    // now rotate d by w0
    t1 = _mm_mul_ps(dR, wr);
    t2 = _mm_mul_ps(dI, wi);
    dR = _mm_mul_ps(dR, wi);
    dI = _mm_mul_ps(dI, wr);
    t1 = _mm_sub_ps(t1, t2);
    dI = _mm_add_ps(dR, dI);
    dR = t1;

    t1 = _mm_add_ps(bR, dR);
    t2 = _mm_add_ps(bI, dI);
    dR = _mm_sub_ps(dR, bR); // actually -dR
    dI = _mm_sub_ps(bI, dI);
    bR = t1;
    bI = t2;

    t1 = _mm_add_ps(aR, bR);
    t2 = _mm_add_ps(aI, bI);
    bR = _mm_sub_ps(aR, bR);
    bI = _mm_sub_ps(bI, aI);
    aR = t1;
    aI = t2;

    t1 = _mm_add_ps(cR, dI);
    cR = _mm_sub_ps(cR, dI);
    dI = _mm_add_ps(dR, cI);
    cI = _mm_sub_ps(dR, cI);
    dR = t1;
  };



  FORCE_INLINE void DIFComplexRadix4Inv(
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
    auto w2i)
  {
    // This is effectively the opposite of the above Radix4.
    auto t1 = _mm_sub_ps(dR, cR);
    cR = _mm_add_ps(dR, cR);
    dR = _mm_add_ps(dI, cI);
    cI = _mm_sub_ps(dI, cI);
    dI = t1;

    t1 = _mm_add_ps(aR, bR);
    bR = _mm_sub_ps(aR, bR);
    aR = t1;
    t1 = _mm_sub_ps(aI, bI);
    bI = _mm_add_ps(aI, bI);
    aI = t1;

    t1 = _mm_sub_ps(bR, dR);
    dR = _mm_add_ps(bR, dR);
    bR = t1;
    t1 = _mm_add_ps(bI, dI);
    dI = _mm_sub_ps(bI, dI);
    bI = t1;

    t1 = _mm_add_ps(aR, cR);
    cR = _mm_sub_ps(aR, cR);
    aR = t1;
    t1 = _mm_add_ps(aI, cI);
    cI = _mm_sub_ps(aI, cI);
    aI = t1;

    __m128 wr, wi;
    if constexpr (std::floating_point<decltype(w0r)>)
    {
      wr = _mm_set1_ps(w0r);
      wi = _mm_set1_ps(w0i);
    }
    else
    {
      wr = w0r;
      wi = w0i;
    }


    t1 = _mm_mul_ps(bR, wr);
    auto t2 = _mm_mul_ps(bI, wi);
    bR = _mm_mul_ps(bR, wi);
    bI = _mm_mul_ps(bI, wr);

    if constexpr (std::floating_point<decltype(w2r)>)
    {
      wr = _mm_set1_ps(w2r);
      wi = _mm_set1_ps(w2i);
    }
    else
    {
      wr = w2r;
      wi = w2i;
    }

    t1 = _mm_add_ps(t1, t2);
    bI = _mm_sub_ps(bI, bR);
    bR = t1;

    t1 = _mm_mul_ps(dR, wr);
    t2 = _mm_mul_ps(dI, wi);
    dR = _mm_mul_ps(dR, wi);
    dI = _mm_mul_ps(dI, wr);

    if constexpr (std::floating_point<decltype(w1r)>)
    {
      wr = _mm_set1_ps(w1r);
      wi = _mm_set1_ps(w1i);
    }
    else
    {
      wr = w1r;
      wi = w1i;
    }

    t1 = _mm_add_ps(t1, t2);
    dI = _mm_sub_ps(dI, dR);
    dR = t1;


    t1 = _mm_mul_ps(cR, wr);
    t2 = _mm_mul_ps(cI, wi);
    cR = _mm_mul_ps(cR, wi);
    cI = _mm_mul_ps(cI, wr);
    t1 = _mm_add_ps(t1, t2);
    cI = _mm_sub_ps(cI, cR);
    cR = t1;
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
    auto w0R,
    auto w0I,
    auto w1RA,
    auto w1IA)
  {
    // For the final real DIT we need a second w1 angle that is just 0.707, -0.707 (but the complex w1 values remain
    //  the same)
    auto w1RB = _mm_move_ss(w1RA, _mm_set_ss(InvSqrt2<float>));
    auto w1IB = _mm_move_ss(w1IA, _mm_set_ss(-InvSqrt2<float>));

    // ef *= angle0;
    auto t1 = _mm_mul_ps(e, w0R);
    auto t2 = _mm_mul_ps(f, w0I);
    e = _mm_mul_ps(e, w0I);
    f = _mm_mul_ps(f, w0R);
    t1 = _mm_sub_ps(t1, t2);
    f = _mm_add_ps(e, f);
    e = t1;

    // gh *= angle0;
    t1 = _mm_mul_ps(g, w0R);
    t2 = _mm_mul_ps(h, w0I);
    g = _mm_mul_ps(g, w0I);
    h = _mm_mul_ps(h, w0R);
    t1 = _mm_sub_ps(t1, t2);
    h = _mm_add_ps(g, h);
    g = t1;

    t1 = _mm_add_ps(a, e);
    e = _mm_sub_ps(a, e);
    a = t1;
    t2 = _mm_add_ps(f, b);
    f = _mm_sub_ps(f, b);
    b = t2;

    t1 = _mm_add_ps(c, g);
    g = _mm_sub_ps(g, c); // Do -g here so that we can leverage it for the 90 degree rotation later
    c = t1;
    t2 = _mm_add_ps(h, d);
    h = _mm_sub_ps(h, d);
    d = t2;

    // Need to do another swap (this time of b0 and c0, again for the "real" radix's different ordering)
    t1 = _mm_move_ss(c, b);
    b =  _mm_move_ss(b, c);
    c = t1;

    // cd *= angle1;
    t1 = _mm_mul_ps(c, w1RA);
    t2 = _mm_mul_ps(d, w1IA);
    c = _mm_mul_ps(c, w1IA);
    d = _mm_mul_ps(d, w1RA);
    t1 = _mm_sub_ps(t1, t2);
    d = _mm_add_ps(c, d);
    c = t1;

    // gh *= {-angle1.imag(), -angle1.real()};
    //  note that this takes advantage of g being stored as -g above
    t1 = _mm_mul_ps(g, w1RB);
    t2 = _mm_mul_ps(h, w1IB);
    g = _mm_mul_ps(g, w1IB);
    h = _mm_mul_ps(h, w1RB);
    t1 = _mm_sub_ps(t1, t2);
    h = _mm_add_ps(g, h);
    g = t1;

    t1 = _mm_add_ps(a, c);
    c = _mm_sub_ps(a, c);
    a = t1;
    t2 = _mm_add_ps(d, b);
    d = _mm_sub_ps(d, b);
    b = t2;

    t1 = _mm_add_ps(e, h);
    t2 = _mm_sub_ps(e, h);
    e = t1;
    h = _mm_sub_ps(g, f);
    f = _mm_add_ps(f, g);
    g = t2;
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
    auto w0R,
    auto w0I,
    auto w1RA,
    auto w1IA)
  {
      // For the final real DIT we need a second w1 angle that is just 0.707, -0.707 (but the complex w1 values remain
    //  the same)
    auto w1RB = _mm_move_ss(w1RA, _mm_set_ss(InvSqrt2<float>));
    auto w1IB = _mm_move_ss(w1IA, _mm_set_ss(-InvSqrt2<float>));

    auto t1 = _mm_sub_ps(e, g);
    e = _mm_add_ps(e, g);
    g = _mm_add_ps(f, h);
    f = _mm_sub_ps(f, h);
    h = t1;

    t1 = _mm_add_ps(a, c);
    c = _mm_sub_ps(a, c);
    a = t1;
    t1 = _mm_sub_ps(b, d);
    d = _mm_add_ps(b, d);
    b = t1;

    t1 = _mm_mul_ps(g, w1RB);
    auto t2 = _mm_mul_ps(h, w1IB);
    g = _mm_mul_ps(g, w1IB);
    h = _mm_mul_ps(h, w1RB);
    t1 = _mm_add_ps(t1, t2);
    h = _mm_sub_ps(h, g);
    g = t1;

    t1 = _mm_mul_ps(c, w1RA);
    t2 = _mm_mul_ps(d, w1IA);
    c = _mm_mul_ps(c, w1IA);
    d = _mm_mul_ps(d, w1RA);
    t1 = _mm_add_ps(t1, t2);
    d = _mm_sub_ps(d, c);
    c = t1;

    t1 = _mm_move_ss(c, b);
    b =  _mm_move_ss(b, c);
    c = t1;

    t1 = _mm_sub_ps(c, g);
    g = _mm_add_ps(c, g);
    c = t1;
    t1 = _mm_sub_ps(d, h);
    h = _mm_add_ps(d, h);
    d = t1;

    t1 = _mm_add_ps(a, e);
    e = _mm_sub_ps(a, e);
    a = t1;
    t1 = _mm_sub_ps(b, f);
    f = _mm_add_ps(b, f);
    b = t1;

    t1 = _mm_mul_ps(g, w0R);
    t2 = _mm_mul_ps(h, w0I);
    g = _mm_mul_ps(g, w0I);
    h = _mm_mul_ps(h, w0R);
    t1 = _mm_add_ps(t1, t2);
    h = _mm_sub_ps(h, g);
    g = t1;

    t1 = _mm_mul_ps(e, w0R);
    t2 = _mm_mul_ps(f, w0I);
    e = _mm_mul_ps(e, w0I);
    f = _mm_mul_ps(f, w0R);
    t1 = _mm_add_ps(t1, t2);
    f = _mm_sub_ps(f, e);
    e = t1;
  }



  [[nodiscard]] FORCE_INLINE auto Transpose4(auto a, auto b, auto c, auto d)
  {
    _MM_TRANSPOSE4_PS(a, b, c, d);
    return std::tuple { a, b, c, d };
  }



  FORCE_INLINE void SwapRealComponentsForLastDoubleFlip(auto &aceg0, auto &bdfh0)
  {
    auto t1 = _mm_shuffle_ps(aceg0, bdfh0, 0b01'00'01'00);
    bdfh0 = _mm_shuffle_ps(aceg0, bdfh0, 0b11'10'11'10);
    aceg0 = t1;
  }



  [[nodiscard]] FORCE_INLINE auto ComplexMultiply(auto aR, auto aI, auto bR, auto bI)
  {
    return std::tuple
      {
        _mm_sub_ps(_mm_mul_ps(aR, bR), _mm_mul_ps(aI, bI)),
        _mm_add_ps(_mm_mul_ps(aR, bI), _mm_mul_ps(aI, bR)),
      };
  }



  [[nodiscard]] FORCE_INLINE auto Deinterleave4(auto a, auto b)
  {
    return std::tuple
      {
        _mm_shuffle_ps(a, b, 0b10'00'10'00),
        _mm_shuffle_ps(a, b, 0b11'01'11'01),
      };
  }



  [[nodiscard]] FORCE_INLINE auto Interleave4(auto a, auto b)
  {
    return std::tuple
      {
        _mm_unpacklo_ps(a, b),
        _mm_unpackhi_ps(a, b),
      };
  }



  template <Alignment A, TangleType Type>
  void LargeTanglePairs(const float *input, float *output, size_t subLength)
  {
    auto [prevTopA, prevTopB] = LoadStridedSets<A, 2>(input, 4);
    auto prevLowA = Load<A>(input + subLength - 8);

    // This first section gets us into the condition where the loop expects
    {
      auto prevLowB = Load<A>(input + subLength - 4);
      __m128 newTopA, newTopB;
      if constexpr (Type == TangleType::InverseLargest)
      {
        newTopA = _mm_shuffle_ps(prevTopA, prevLowB, 0b11'10'01'00);
        newTopB = _mm_shuffle_ps(prevLowB, prevLowA, 0b11'10'01'00);
      }
      else
      {
        newTopA = _mm_shuffle_ps(prevTopA, prevLowA, 0b11'10'01'00);
        newTopB = prevLowB;
      }
      StoreStridedSets<A>(output, 4, newTopA, newTopB);
      auto [topA, topB] = LoadStridedSets<A, 2>(input + 8, 4);

      __m128 newLowA, newLowB;
      if constexpr (Type == TangleType::ForwardLargest)
      {
        newLowA = _mm_shuffle_ps(topA, prevTopB, 0b11'10'01'00);
        newLowB = _mm_shuffle_ps(prevTopB, prevTopA, 0b11'10'01'00);
      }
      else
      {
        newLowA = _mm_shuffle_ps(topA, prevTopA, 0b11'10'01'00);
        newLowB = prevTopB;
      }

      StoreStridedSets<A>(output + subLength - 8, 4, newLowA, newLowB);
      auto [lowA, lowB] = LoadStridedSets<A, 2>(input + subLength - 16, 4);
      if constexpr (Type == TangleType::ForwardLargest)
      {
        newTopA = _mm_shuffle_ps(prevLowA, lowB, 0b11'10'01'00);
        newTopB = _mm_shuffle_ps(lowB, lowA, 0b11'10'01'00);
      }
      else
      {
        newTopA = _mm_shuffle_ps(prevLowA, lowA, 0b11'10'01'00);
        newTopB = lowB;
      }
      StoreStridedSets<A>(output + 8, 4, newTopA, newTopB);
      prevTopA = topA;
      prevTopB = topB;
      prevLowA = lowA;
    }

    // Step by 16s through (doing 2 sets of 4 pairs at a time, half of which flip in
    //  Shuffle mode and half of which don't)
    for (auto i = size_t(16);;)
    {
      auto [topA, topB] = LoadStridedSets<A, 2>(input + i, 4);

      __m128 newLowA, newLowB;
      if constexpr (Type == TangleType::InverseLargest)
      {
        newLowA = _mm_shuffle_ps(topA, prevTopB, 0b11'10'01'00);
        newLowB = _mm_shuffle_ps(prevTopB, prevTopA, 0b11'10'01'00);
      }
      else
      {
        newLowA = _mm_shuffle_ps(topA, prevTopA, 0b11'10'01'00);
        newLowB = prevTopB;
      }

      StoreStridedSets<A>(output + subLength - i, 4, newLowA, newLowB);

      auto [lowA, lowB] = LoadStridedSets<A, 2>(input + subLength - i - 8, 4);

      __m128 newTopA, newTopB;
      if constexpr (Type == TangleType::InverseLargest)
      {
        newTopA = _mm_shuffle_ps(prevLowA, lowB, 0b11'10'01'00);
        newTopB = _mm_shuffle_ps(lowB, lowA, 0b11'10'01'00);
      }
      else
      {
        newTopA = _mm_shuffle_ps(prevLowA, lowA, 0b11'10'01'00);
        newTopB = lowB;
      }
      StoreStridedSets<A>(output + i, 4, newTopA, newTopB);

      std::tie(prevTopA, prevTopB) = LoadStridedSets<A, 2>(input + i + 8, 4);

      if constexpr (Type == TangleType::ForwardLargest)
      {
        newLowA = _mm_shuffle_ps(prevTopA, topB, 0b11'10'01'00);
        newLowB = _mm_shuffle_ps(topB, topA, 0b11'10'01'00);
      }
      else
      {
        newLowA = _mm_shuffle_ps(prevTopA, topA, 0b11'10'01'00);
        newLowB = topB;
      }

      StoreStridedSets<A>(output + subLength - i - 8, 4, newLowA, newLowB);

      prevLowA = Load<A>(input + subLength - i - 16);
      auto prevLowB = Load<A>(input + subLength - i - 12);

      if constexpr (Type == TangleType::ForwardLargest)
      {
        newTopA = _mm_shuffle_ps(lowA, prevLowB, 0b11'10'01'00);
        newTopB = _mm_shuffle_ps(prevLowB, prevLowA, 0b11'10'01'00);
      }
      else
      {
        newTopA = _mm_shuffle_ps(lowA, prevLowA, 0b11'10'01'00);
        newTopB = prevLowB;
      }

      StoreStridedSets<A>(output + i + 8, 4, newTopA, newTopB);

      i += 16;
      if (i >= subLength / 2)
        { break; }
    }

    // The very middle value needs to be written now.
    {
      __m128 finalA, finalB;
      if constexpr (Type == TangleType::InverseLargest)
      {
        finalA = _mm_shuffle_ps(prevLowA, prevTopB, 0b11'10'01'00);
        finalB = _mm_shuffle_ps(prevTopB, prevTopA, 0b11'10'01'00);
      }
      else
      {
        finalA = _mm_shuffle_ps(prevLowA, prevTopA, 0b11'10'01'00);
        finalB = prevTopB;
      }

      StoreStridedSets<A>(output + subLength / 2, 4, finalA, finalB);
    }
  }



  template <Alignment A>
  void Copy4Pairs(const float *input, float *output)
  {
    auto [Aa, Bb] = LoadStridedSets<A, 2>(input, 4);
    StoreStridedSets<A>(output, 4, Aa, Bb);
  }



  template <Alignment A, TangleType Type>
  void TangleFinal4Pairs(const float *input, float *output)
  {
    if constexpr (Type == TangleType::ForwardLargest)
    {
      //  AbBa -> AaBb
      auto [Ab, Ba] = LoadStridedSets<A, 2>(input, 4);
      auto Aa = _mm_shuffle_ps(Ab, Ba, 0b11'10'01'00);
      auto Bb = _mm_shuffle_ps(Ba, Ab, 0b11'10'01'00);
      StoreStridedSets<A>(output, 4, Aa, Bb);
    }
    else
    {
      static_assert(Type == TangleType::InverseLargest);
      // AaBb -> AbBa
      auto [Aa, Bb] = LoadStridedSets<A, 2>(input, 4);
      auto Ab = _mm_shuffle_ps(Aa, Bb, 0b11'10'01'00);
      auto Ba = _mm_shuffle_ps(Bb, Aa, 0b11'10'01'00);
      StoreStridedSets<A>(output, 4, Ab, Ba);
    }
  }



  template <Alignment A, TangleType Type>
  void Tangle8Pairs(const float *input, float *output)
  {
    if constexpr (Type == TangleType::ForwardInner)
    {
      // AcDdCaBb -> AaBbCcDd
      auto [Ac, Ca] = LoadStridedSets<A, 2>(input, 8);
      auto Aa = _mm_shuffle_ps(Ac, Ca, 0b11'10'01'00);
      auto Cc = _mm_shuffle_ps(Ca, Ac, 0b11'10'01'00);
      auto [Dd, Bb] = LoadStridedSets<A, 2>(input + 4, 8);
      StoreStridedSets<A>(output, 4, Aa, Bb, Cc, Dd);
    }
    else if constexpr (Type == TangleType::ForwardLargest)
    {
      // AdDcCaBb -> AaBbCcDd
      auto [Ad, Dc, Ca, Bb] = LoadStridedSets<A, 4>(input, 4);
      auto Aa = _mm_shuffle_ps(Ad, Ca, 0b11'10'01'00);
      auto Dd = _mm_shuffle_ps(Dc, Ad, 0b11'10'01'00);
      auto Cc = _mm_shuffle_ps(Ca, Dc, 0b11'10'01'00);
      StoreStridedSets<A>(output, 4, Aa, Bb, Cc, Dd);
    }
    else
    {
      auto [Aa, Bb, Cc, Dd] = LoadStridedSets<A, 4>(input, 4);

      if constexpr (Type == TangleType::InverseInner)
      {
        // AaBbCcDd -> AcDdCaBb
        auto Ac = _mm_shuffle_ps(Aa, Cc, 0b11'10'01'00);
        auto Ca = _mm_shuffle_ps(Cc, Aa, 0b11'10'01'00);
        StoreStridedSets<A>(output, 4, Ac, Dd, Ca, Bb);
      }
      else
      {
        // AaBbCcDd -> AdDcCaBb
        static_assert(Type == TangleType::InverseLargest);
        auto Ad = _mm_shuffle_ps(Aa, Dd, 0b11'10'01'00);
        auto Dc = _mm_shuffle_ps(Dd, Cc, 0b11'10'01'00);
        auto Ca = _mm_shuffle_ps(Cc, Aa, 0b11'10'01'00);
        StoreStridedSets<A>(output, 4, Ad, Dc, Ca, Bb);
      }
    }
  }



  template <Alignment A, TangleType Type>
  void Tangle16Pairs(const float *input, float *output)
  {
    if constexpr (Type == TangleType::ForwardInner)
    {
      // AgHhGeFfEaBbCcDd -> AaBbCcDdEeFfGgHh
      auto [Ag, Hh, Ge] = LoadStridedSets<A, 3>(input, 4);
      auto Gg = _mm_shuffle_ps(Ge, Ag, 0b11'10'01'00);
      auto [Ff, Ea] = LoadStridedSets<A, 2>(input + 12, 4);
      auto Aa = _mm_shuffle_ps(Ag, Ea, 0b11'10'01'00);
      auto Ee = _mm_shuffle_ps(Ea, Ge, 0b11'10'01'00);
      auto [Bb, Cc, Dd] = LoadStridedSets<A, 3>(input + 20, 4);

      StoreStridedSets<A>(output, 4, Aa, Bb, Cc, Dd, Ee, Ff, Gg, Hh);
    }
    else if constexpr (Type == TangleType::ForwardLargest)
    {
      // AhHgGeFfEaBbCdDc -> AaBbCcDdEeFfGgHh
      auto [Ah, Hg, Ge] = LoadStridedSets<A, 3>(input, 4);
      auto Hh = _mm_shuffle_ps(Hg, Ah, 0b11'10'01'00);
      auto Gg = _mm_shuffle_ps(Ge, Hg, 0b11'10'01'00);
      auto [Ff, Ea] = LoadStridedSets<A, 2>(input + 12, 4);
      auto Ee = _mm_shuffle_ps(Ea, Ge, 0b11'10'01'00);
      auto Aa = _mm_shuffle_ps(Ah, Ea, 0b11'10'01'00);
      auto [Bb, Cd, Dc] = LoadStridedSets<A, 3>(input + 20, 4);
      auto Cc = _mm_shuffle_ps(Cd, Dc, 0b11'10'01'00);
      auto Dd = _mm_shuffle_ps(Dc, Cd, 0b11'10'01'00);
      StoreStridedSets<A>(output, 4, Aa, Bb, Cc, Dd, Ee, Ff, Gg, Hh);
    }
    else
    {
      auto [Aa, Bb, Cc, Dd, Ee, Ff, Gg, Hh] = LoadStridedSets<A, 8>(input, 4);

      if constexpr (Type == TangleType::InverseInner)
      {
        // AaBbCcDdEeFfGgHh -> AgHhGeFfEaBbCcDd

        auto Ag = _mm_shuffle_ps(Aa, Gg, 0b11'10'01'00);
        auto Ge = _mm_shuffle_ps(Gg, Ee, 0b11'10'01'00);
        auto Ea = _mm_shuffle_ps(Ee, Aa, 0b11'10'01'00);
        StoreStridedSets<A>(output, 4, Ag, Hh, Ge, Ff, Ea, Bb, Cc, Dd);
      }
      else
      {
        // AaBbCcDdEeFfGgHh -> AhHgGeFfEaBbCdDc
        auto Ah = _mm_shuffle_ps(Aa, Hh, 0b11'10'01'00);
        auto Hg = _mm_shuffle_ps(Hh, Gg, 0b11'10'01'00);
        auto Ge = _mm_shuffle_ps(Gg, Ee, 0b11'10'01'00);
        auto Ea = _mm_shuffle_ps(Ee, Aa, 0b11'10'01'00);
        auto Cd = _mm_shuffle_ps(Cc, Dd, 0b11'10'01'00);
        auto Dc = _mm_shuffle_ps(Dd, Cc, 0b11'10'01'00);
        StoreStridedSets<A>(output, 4, Ah, Hg, Ge, Ff, Ea, Bb, Cd, Dc);
      }
    }
  }



  // Swaps 64 elements (32 pairs)
  template <Alignment A, TangleType Type>
  void Tangle32Pairs(const float *input, float *output)
  {
    if constexpr (Type == TangleType::ForwardInner)
    {
      // AoPpOmNnMiJjKkLlIaBbCcDdEgHhGeFf -> AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPp
      // Conveniently with the loads and shuffles ordered as such, we fit within the x64 register space.
      auto [Ao, Pp, Om, Nn, Mi] = LoadStridedSets<A, 5>(input, 4);
      auto Oo = _mm_shuffle_ps(Om, Ao, 0b11'10'01'00);
      auto Mm = _mm_shuffle_ps(Mi, Om, 0b11'10'01'00);
      auto [Jj, Kk, Ll, Ia] = LoadStridedSets<A, 4>(input + 20, 4);
      auto Aa = _mm_shuffle_ps(Ao, Ia, 0b11'10'01'00);
      auto Ii = _mm_shuffle_ps(Ia, Mi, 0b11'10'01'00);
      auto [Bb, Cc, Dd, Eg, Hh, Ge] = LoadStridedSets<A, 6>(input + 36, 4);
      auto Gg = _mm_shuffle_ps(Ge, Eg, 0b11'10'01'00);
      auto Ee = _mm_shuffle_ps(Eg, Ge, 0b11'10'01'00);
      auto Ff = Load<A>(input + 60);

      StoreStridedSets<A>(output, 4, Aa, Bb, Cc, Dd, Ee, Ff, Gg, Hh, Ii, Jj, Kk, Ll, Mm, Nn, Oo, Pp);
    }
    else if constexpr (Type == TangleType::ForwardLargest)
    {
      // ApPoOmNnMiJjKlLkIaBbCdDcEhHgGeFf -> AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPp
      //  (note that pPo, lLk, dDc, and hHg are reversed from the above)
      auto [Ap, Po, Om, Nn, Mi] = LoadStridedSets<A, 5>(input, 4);
      auto Oo = _mm_shuffle_ps(Om, Po, 0b11'10'01'00);
      auto Pp = _mm_shuffle_ps(Po, Ap, 0b11'10'01'00);
      auto Mm = _mm_shuffle_ps(Mi, Om, 0b11'10'01'00);
      auto [Jj, Kl, Lk] = LoadStridedSets<A, 3>(input + 20, 4);
      auto Kk = _mm_shuffle_ps(Kl, Lk, 0b11'10'01'00);
      auto Ll = _mm_shuffle_ps(Lk, Kl, 0b11'10'01'00);
      auto Ia = Load<A>(input + 32);
      auto Aa = _mm_shuffle_ps(Ap, Ia, 0b11'10'01'00);
      auto Ii = _mm_shuffle_ps(Ia, Mi, 0b11'10'01'00);
      auto [Bb, Cd, Dc] = LoadStridedSets<A, 3>(input + 36, 4);
      auto Cc = _mm_shuffle_ps(Cd, Dc, 0b11'10'01'00);
      auto Dd = _mm_shuffle_ps(Dc, Cd, 0b11'10'01'00);
      auto [Eh, Hg, Ge] = LoadStridedSets<A, 3>(input + 48, 4);
      auto Hh = _mm_shuffle_ps(Hg, Eh, 0b11'10'01'00);
      auto Gg = _mm_shuffle_ps(Ge, Hg, 0b11'10'01'00);
      auto Ee = _mm_shuffle_ps(Eh, Ge, 0b11'10'01'00);
      auto Ff = Load<A>(input + 60);
      StoreStridedSets<A>(output, 4, Aa, Bb, Cc, Dd, Ee, Ff, Gg, Hh, Ii, Jj, Kk, Ll, Mm, Nn, Oo, Pp);
    }
    else if constexpr (Type == TangleType::InverseInner)
    {
      // AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPp -> AoPpOmNnMiJjKkLlIaBbCcDdEgHhGeFf
      auto [Aa, Bb, Cc, Dd, Ee, Ff, Gg] = LoadStridedSets<A, 7>(input, 4);
      auto Ge = _mm_shuffle_ps(Gg, Ee, 0b11'10'01'00);
      auto Eg = _mm_shuffle_ps(Ee, Gg, 0b11'10'01'00);
      auto [Hh, Ii] = LoadStridedSets<A, 2>(input + 28, 4);
      auto Ia = _mm_shuffle_ps(Ii, Aa, 0b11'10'01'00);
      auto [Jj, Kk, Ll, Mm] = LoadStridedSets<A, 4>(input + 36, 4);
      auto Mi = _mm_shuffle_ps(Mm, Ii, 0b11'10'01'00);
      auto [Nn, Oo] = LoadStridedSets<A, 2>(input + 52, 4);
      auto Ao = _mm_shuffle_ps(Aa, Oo, 0b11'10'01'00);
      auto Om = _mm_shuffle_ps(Oo, Mm, 0b11'10'01'00);
      auto Pp = Load<A>(input + 60);
      StoreStridedSets<A>(output, 4, Ao, Pp, Om, Nn, Mi, Jj, Kk, Ll, Ia, Bb, Cc, Dd, Eg, Hh, Ge, Ff);
    }
    else if constexpr (Type == TangleType::InverseLargest)
    {
      // AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPp -> ApPoOmNnMiJjKlLkIaBbCdDcEhHgGeFf
      auto [Aa, Bb, Cc, Dd] = LoadStridedSets<A, 4>(input, 4);
      auto Cd = _mm_shuffle_ps(Cc, Dd, 0b11'10'01'00);
      auto Dc = _mm_shuffle_ps(Dd, Cc, 0b11'10'01'00);
      auto [Ee, Ff, Gg] = LoadStridedSets<A, 3>(input + 16, 4);
      auto Ge = _mm_shuffle_ps(Gg, Ee, 0b11'10'01'00);
      auto Hh = Load<A>(input + 28);
      auto Eh = _mm_shuffle_ps(Ee, Hh, 0b11'10'01'00);
      auto Hg = _mm_shuffle_ps(Hh, Gg, 0b11'10'01'00);
      auto [Ii, Jj, Kk, Ll] = LoadStridedSets<A, 4>(input + 32, 4);
      auto Kl = _mm_shuffle_ps(Kk, Ll, 0b11'10'01'00);
      auto Lk = _mm_shuffle_ps(Ll, Kk, 0b11'10'01'00);
      auto Mm = Load<A>(input + 48);
      auto Ia = _mm_shuffle_ps(Ii, Aa, 0b11'10'01'00);
      auto Mi = _mm_shuffle_ps(Mm, Ii, 0b11'10'01'00);
      auto [Nn, Oo, Pp] = LoadStridedSets<A, 3>(input + 52, 4);
      auto Ap = _mm_shuffle_ps(Aa, Pp, 0b11'10'01'00);
      auto Po = _mm_shuffle_ps(Pp, Oo, 0b11'10'01'00);
      auto Om = _mm_shuffle_ps(Oo, Mm, 0b11'10'01'00);
      StoreStridedSets<A>(output, 4, Ap, Po, Om, Nn, Mi, Jj, Kl, Lk, Ia, Bb, Cd, Dc, Eh, Hg, Ge, Ff);
    }
  }



  #include "fft_20xx_algo_real_fwd.h"
  #include "fft_20xx_algo_real_inv.h"
}


using namespace FFT20XX;
using namespace fft_20xx::c;
extern "C"
{
  void fft_real_f32_sse2(const fft_table_f32 *table, const float *input, float *output, size_t fftLength)
  {
    auto t = reinterpret_cast<const FFTTableImpl<float> *>(table);
    COMMON_ASSERTS();
    SSE2::RealForwardFFT<Alignment::Unaligned>(t->RealData(), input, output, fftLength);
  }



  void ifft_real_f32_sse2(const fft_table_f32 *table, const float *input, float *output, size_t fftLength)
  {
    auto t = reinterpret_cast<const FFTTableImpl<float> *>(table);
    COMMON_ASSERTS();
    SSE2::RealInverseFFT<Alignment::Unaligned>(t->RealData(), input, output, fftLength);
  }
}
