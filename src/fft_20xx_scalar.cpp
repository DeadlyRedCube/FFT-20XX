#include "fft_20xx_common.h"
#include "fft_20xx_table.h"

using namespace fft_20xx;
using namespace fft_20xx::c;

#define USE_FMA 0
namespace FFT20XX::Scalar
{
  template <typename T, uint32_t N>
  struct ScalarVec
  {
    using Type = T;
    static constexpr auto Length = N;

    T data[N];

    T &operator[](std::integral auto i)
      { return data[i]; }

    const T &operator[](std::integral auto i) const
      { return data[i]; }
  };

  constexpr uint32_t StandardSize = 8u;

  template <typename T> concept f32 = std::same_as<float, T>
    || std::same_as<ScalarVec<float, StandardSize>, T>
    || std::same_as<ScalarVec<float, StandardSize / 2>, T>;

  template <typename T> using Vec = ScalarVec<T, StandardSize>;

  template <typename> constexpr uint32_t VecSize = StandardSize;

  constexpr bool IsScalar = true;
  constexpr bool UseFMAOptimization = false;

  template <std::floating_point F>
  static auto LoadA(const F *ptr) -> Vec<F>
  {
    Vec<F> r;
    for (auto i = 0u; i < StandardSize; i++)
      { r[i] = ptr[i]; }
    return r;
  }
  template <std::floating_point F>
  static auto Load(const F *ptr) -> Vec<F>
    { return LoadA(ptr); }

  template <std::floating_point F>
  static auto LoadHalf(const F *ptr)
  {
    ScalarVec<F, StandardSize / 2> r;
    for (auto i = 0u; i < StandardSize / 2; i++)
      { r[i] = ptr[i]; }
    return r;
  }

  template <typename F>
  static auto SplitHalves(Vec<F> v)
  {
    Pair<ScalarVec<F, StandardSize / 2>, ScalarVec<F, StandardSize / 2>> r;
    for (auto i = 0u; i < StandardSize / 2; i++)
    {
      r.first[i] = v[i];
      r.second[i] = v[i + StandardSize / 2];
    }
    return r;
  }

  template <typename F>
  static auto LoadHalves(const F *ptr, size_t offset)
  {
    Vec<F> r;
    for (auto i = 0u; i < StandardSize / 2; i++)
    {
      r[i] = ptr[i];
      r[i + StandardSize / 2] = ptr[i + offset];
    }
    return r;
  }


  template <std::floating_point F>
  static auto Load4(const F *ptr) -> Vec<F>
  {
    Vec<F> r;
    for (auto i = 0u; i < StandardSize; i++)
      { r[i] = ptr[i & 3]; }
    return r;
  }

  template <std::floating_point F>
  static auto LoadDeinterleaved(const F *ptr) -> Pair<Vec<F>, Vec<F>>
  {
    Pair<Vec<F>, Vec<F>> r;
    for (auto i = 0u; i < StandardSize; i++)
    {
      r.first[i] = ptr[2 * i + 0];
      r.second[i] = ptr[2 * i + 1];
    }
    return r;
  }

  template <std::floating_point F>
  static auto LoadDeinterleavedHalf(const F *ptr)
  {
    Pair<ScalarVec<F, StandardSize / 2>, ScalarVec<F, StandardSize / 2>> r;
    for (auto i = 0u; i < StandardSize / 2; i++)
    {
      r.first[i] = ptr[2 * i + 0];
      r.second[i] = ptr[2 * i + 1];
    }
    return r;
  }


  template <std::floating_point F>
  static auto StoreHalves(F *ptr, size_t offset, Vec<F> v)
  {
    for (auto i = 0u; i < StandardSize / 2; i++)
    {
      ptr[i] = v[i];
      ptr[i + offset] = v[i + StandardSize / 2];
    }
  }

  template <std::floating_point F, typename V>
  static void Store(F *ptr, V a)
  {
    for (auto i = 0u; i < V::Length; i++)
      { ptr[i] = a[i]; }
  }

  template <std::floating_point F, typename V>
  static void Store4x4Transposed(F *ptrA, F *ptrB, F *ptrC, F *ptrD, V a, V b, V c, V d)
  {
    for (auto x = 0u; x < V::Length; x += 4u)
    {
      std::swap(a[x + 1], b[x + 0]);
      std::swap(a[x + 2], c[x + 0]);
      std::swap(a[x + 3], d[x + 0]);
      std::swap(b[x + 2], c[x + 1]);
      std::swap(b[x + 3], d[x + 1]);
      std::swap(c[x + 3], d[x + 2]);
    }

    Store(ptrA, a);
    Store(ptrB, b);
    Store(ptrC, c);
    Store(ptrD, d);
  }

  template <std::floating_point F, typename V>
  static void StoreInterleaved(F *ptr, V a, V b)
  {
    for (auto i = 0u; i < V::Length; i++)
    {
      ptr[2 * i + 0] = a[i];
      ptr[2 * i + 1] = b[i];
    }
  }

  template <typename V>
  static V Add(V a, V b)
  {
    V r;
    for (auto i = 0u; i < V::Length; i++)
      { r[i] = a[i] + b[i]; }
    return r;
  }

  template <typename V>
  static V Sub(V a, V b)
  {
    V r;
    for (auto i = 0u; i < V::Length; i++)
      { r[i] = a[i] - b[i]; }
    return r;
  }

  template <typename V>
  static V Mul(V a, V b)
  {
    V r;
    for (auto i = 0u; i < V::Length; i++)
      { r[i] = a[i] * b[i]; }
    return r;
  }

  template <typename V, std::floating_point F>
  static V Mul(V a, F b)
  {
    for (auto i = 0u; i < V::Length; i++)
      { a[i] *= b; };
    return a;
  }

  template <std::floating_point F, typename ... Vs>
  static void Scale(F scale, Vs &...vs)
    { ((vs = Mul(vs, scale)), ...); }

  template <typename V>
  static V MAdd(V a, V b, V c)
    { return Add(Mul(a, b), c); }

  template <typename V>
  static V NMAdd(V a, V b, V c)
    { return Sub(c, Mul(a, b)); }

  #include "fft_20xx_run.h"

}

using namespace FFT20XX;

namespace
{
  auto BitRev(std::integral auto v)
  {
    if constexpr (sizeof(v) == sizeof(uint64_t))
    {
      v = ((v >>  1) & 0x55555555'55555555) | ((v & 0x55555555'55555555) << 1);
      v = ((v >>  2) & 0x33333333'33333333) | ((v & 0x33333333'33333333) << 2);
      v = ((v >>  4) & 0x0f0f0f0f'0f0f0f0f) | ((v & 0x0f0f0f0f'0f0f0f0f) << 4);
      v = ((v >>  8) & 0x00ff00ff'00ff00ff) | ((v & 0x00ff00ff'00ff00ff) << 8);
      v = ((v >> 16) & 0x0000ffff'0000ffff) | ((v & 0x0000ffff'0000ffff) << 16);
      v = (v >> 32) | (v << 32);
    }
    else
    {
      static_assert(sizeof(v) == sizeof(uint32_t));
      v = ((v >>  1) & 0x55555555) | ((v & 0x55555555) << 1);
      v = ((v >>  2) & 0x33333333) | ((v & 0x33333333) << 2);
      v = ((v >>  4) & 0x0f0f0f0f) | ((v & 0x0f0f0f0f) << 4);
      v = ((v >>  8) & 0x00ff00ff) | ((v & 0x00ff00ff) << 8);
      v = (v >> 16) | (v << 16);
    }

    return v;
  }

  template <std::floating_point F>
  FORCE_INLINE void DoRadix4NoRotation(
    F &aR,
    F &aI,
    F &bR,
    F &bI,
    F &cR,
    F &cI,
    F &dR,
    F &dI)
  {
    auto t = aR + cR;
    cR = aR - cR;
    aR = t;
    t = aI + cI;
    cI = aI - cI;
    aI = t;

    t = bR + dR;
    dR = bR - dR;
    bR = t;
    t = bI + dI;
    dI = bI - dI;
    bI = t;

    t = aR + bR;
    bR = aR - bR;
    aR = t;
    t = aI + bI;
    bI = aI - bI;
    aI = t;

    // rot90(d)
    t = cR + dI;
    dI = cR - dI;
    cR = t;
    t = cI - dR;
    dR = cI + dR;
    cI = t;
    std::swap(dR, dI);
  }

  template <std::floating_point F>
  FORCE_INLINE void DoRadix4FirstRotation(
    F &aR,
    F &aI,
    F &bR,
    F &bI,
    F &cR,
    F &cI,
    F &dR,
    F &dI)
  {
    // rot90(c)
    auto t = aR + cI;
    cI = aR - cI; // actually cR
    aR = t;
    t = aI - cR;
    cR = aI + cR;
    aI = t;
    std::swap(cR, cI);

    // rot90(d)
    t = bR + dI;
    dI = bR - dI;
    bR = t;
    t = bI - dR;
    dR = bI + dR;
    bI = t;
    std::swap(dR, dI);

    // rot45(b)
    t = bR + bI;
    bI = bI - bR;
    bR = InvSqrt2<F> * t;
    bI = InvSqrt2<F> * bI;
    t = aR + bR;
    bR = aR - bR;
    aR = t;
    t = aI + bI;
    bI = aI - bI;
    aI = t;

    // rot135(d)
    t = dI - dR;
    dI = dI + dR; // This is actually -dI
    dR = InvSqrt2<F> * t;
    dI = InvSqrt2<F> * dI;

    t = cR + dR;
    dR = cR - dR;
    cR = t;
    t = cI - dI; // Because dI is -dI, addition is subtraction!
    dI = cI + dI;
    cI = t;
  }

  template <std::floating_point F>
  void Radix4Or8InitialStepScalar(const F *input, F *output, const size_t length)
  {
    if (!IsPowerOfFourAssumingIsPowerOfTwo(length))
    {
      // Double the stride to skip over complex pairs.
      const auto doubleStride = length / 4;
      // RADIX 8, YO
      for (auto i = 0_z; i < doubleStride; i += 2)
      {
        auto aR = input[doubleStride*0 + i];
        auto aI = input[doubleStride*0 + i + 1];
        auto bR = input[doubleStride*1 + i];
        auto bI = input[doubleStride*1 + i + 1];
        auto cR = input[doubleStride*2 + i];
        auto cI = input[doubleStride*2 + i + 1];
        auto dR = input[doubleStride*3 + i];
        auto dI = input[doubleStride*3 + i + 1];
        auto eR = input[doubleStride*4 + i];
        auto eI = input[doubleStride*4 + i + 1];
        auto fR = input[doubleStride*5 + i];
        auto fI = input[doubleStride*5 + i + 1];
        auto gR = input[doubleStride*6 + i];
        auto gI = input[doubleStride*6 + i + 1];
        auto hR = input[doubleStride*7 + i];
        auto hI = input[doubleStride*7 + i + 1];

        // First Radix 2:
        auto t = aR + eR;
        eR = aR - eR;
        aR = t;
        t = aI + eI;
        eI = aI - eI;
        aI = t;

        t = bR + fR;
        fR = bR - fR;
        bR = t;
        t = bI + fI;
        fI = bI - fI;
        bI = t;

        t = cR + gR;
        gR = cR - gR;
        cR = t;
        t = cI + gI;
        gI = cI - gI;
        cI = t;

        t = dR + hR;
        hR = dR - hR;
        dR = t;
        t = dI + hI;
        hI = dI - hI;
        dI = t;

        // Second Radix 2:
        t = aR + cR;
        cR = aR - cR;
        aR = t;
        t = aI + cI;
        cI = aI - cI;
        aI = t;

        t = bR + dR;
        dR = bR - dR;
        bR = t;
        t = bI + dI;
        dI = bI - dI;
        bI = t;

        // rot90(g)
        t = eR + gI;
        gI = eR - gI;
        eR = t;
        t = eI - gR;
        gR = eI + gR;
        eI = t;
        std::swap(gR, gI);

        // rot90(h)
        t = fR + hI;
        hI = fR - hI;
        fR = t;
        t = fI - hR;
        hR = fI + hR;
        fI = t;
        std::swap(hR, hI);

        // Third Radix 2:
        t = aR + bR;
        bR = aR - bR;
        aR = t;
        t = aI + bI;
        bI = aI - bI;
        aI = t;

        // rot90(d)
        t = cR + dI;
        dI = cR - dI;
        cR = t;
        t = cI - dR;
        dR = cI + dR;
        cI = t;
        std::swap(dR, dI);

        // rot45(f)
        t = fR + fI;
        fI = fI - fR;
        fR = InvSqrt2<F> * t;
        fI = InvSqrt2<F> * fI;
        t = eR + fR;
        fR = eR - fR;
        eR = t;
        t = eI + fI;
        fI = eI - fI;
        eI = t;

        //  rot135(f)
        t = hI - hR;
        hI = hI + hR; // This is actually -hI
        hR = InvSqrt2<F> * t;
        hI = InvSqrt2<F> * hI;

        t = gR + hR;
        hR = gR - hR;
        gR = t;
        t = gI - hI; // Because hI is -hI, addition is subtraction!
        hI = gI + hI;
        gI = t;

        output[doubleStride*0 + i] = aR;
        output[doubleStride*0 + i + 1] = aI;
        output[doubleStride*1 + i] = bR;
        output[doubleStride*1 + i + 1] = bI;
        output[doubleStride*2 + i] = cR;
        output[doubleStride*2 + i + 1] = cI;
        output[doubleStride*3 + i] = dR;
        output[doubleStride*3 + i + 1] = dI;
        output[doubleStride*4 + i] = eR;
        output[doubleStride*4 + i + 1] = eI;
        output[doubleStride*5 + i] = fR;
        output[doubleStride*5 + i + 1] = fI;
        output[doubleStride*6 + i] = gR;
        output[doubleStride*6 + i + 1] = gI;
        output[doubleStride*7 + i] = hR;
        output[doubleStride*7 + i + 1] = hI;
      }
    }
    else
    {
      const auto doubleStride = length / 2;
      for (auto i = 0_z; i < doubleStride; i += 2)
      {
        auto aR = input[doubleStride*0 + i];
        auto aI = input[doubleStride*0 + i + 1];
        auto bR = input[doubleStride*1 + i];
        auto bI = input[doubleStride*1 + i + 1];
        auto cR = input[doubleStride*2 + i];
        auto cI = input[doubleStride*2 + i + 1];
        auto dR = input[doubleStride*3 + i];
        auto dI = input[doubleStride*3 + i + 1];

        DoRadix4NoRotation(aR, aI, bR, bI, cR, cI, dR, dI);

        output[doubleStride*0 + i] = aR;
        output[doubleStride*0 + i + 1] = aI;
        output[doubleStride*1 + i] = bR;
        output[doubleStride*1 + i + 1] = bI;
        output[doubleStride*2 + i] = cR;
        output[doubleStride*2 + i + 1] = cI;
        output[doubleStride*3 + i] = dR;
        output[doubleStride*3 + i + 1] = dI;
      }
    }
  }



  template <std::floating_point F>
  void DoRadix4(
    F &aR,
    F &aI,
    F &bR,
    F &bI,
    F &cR,
    F &cI,
    F &dR,
    F &dI,
    F w0R,
    F w0I,
    F w1R,
    F w1I,
    F w2R,
    F w2I)
  {
    auto t = bR * w0R - bI * w0I;
    bI = bR * w0I + bI * w0R;
    bR = t;

    t = cR * w1R - cI * w1I;
    cI = cR * w1I + cI * w1R;
    cR = t;

    t = dR * w2R - dI * w2I;
    dI = dR * w2I + dI * w2R;
    dR = t;

    t = aR + cR;
    cR = aR - cR;
    aR = t;
    t = aI + cI;
    cI = aI - cI;
    aI = t;

    t = bR + dR;
    dR = bR - dR;
    bR = t;
    t = bI + dI;
    dI = bI - dI;
    bI = t;

    t = aR + bR;
    bR = aR - bR;
    aR = t;
    t = aI + bI;
    bI = aI - bI;
    aI = t;

    // rot90(d)
    t = cR + dI;
    dI = cR - dI;
    cR = t;
    t = cI - dR;
    dR = cI + dR;
    cI = t;
    std::swap(dR, dI);
  }

  template <std::floating_point F>
  auto LoadSingleAngle(const F *tableData, size_t angleIndex, size_t angleCount)
  {
    // !!! do something different here?
    auto i1 = angleCount * 4;
    auto r1 = (angleCount == 4) ? 8 : (angleCount * 3);

    // Special handling for 8-angle indexing, as they're arranged differently for alignment reasons.
    //  The default math would index the "angleCount == 8" angles from offset 12, but they actually start at 8.
    //  This masks off the bottom bits of 12 to turn it into an 8.
    auto r0 = (angleCount == 4) ? 0 : ((r1 >> 1) & ~7_z);
    auto i0 = (angleCount == 4) ? 4 : (i1 >> 1);

    auto wr1 = tableData[r0 + angleIndex];
    auto wi1 = tableData[i0 + angleIndex];
    auto wr2 = tableData[r1 + angleIndex];
    auto wi2 = tableData[i1 + angleIndex];

    #if USE_FMA
      return std::tuple{wr1, wi1, wr2, wi2};
    #else
      // Third angle calculated based on the first two.
      return std::tuple{wr2, wi2, wr1, wi1, wr1 * wr2 - wi1 * wi2, wr1 * wi2 + wi1 * wr2};
    #endif
  }

  template <std::floating_point F>
  void NoFlipRadix4PassScalar(const F *tableData, F *data, const size_t length, const size_t radix4Stride)
  {
    const auto doubleStride = radix4Stride * 2;

    // The first set of stride works with no rotations (besides the 90)
    {
      for (auto i = 0_z; i < doubleStride; i += 2)
      {
        auto aR = data[doubleStride*0 + i];
        auto aI = data[doubleStride*0 + i + 1];
        auto bR = data[doubleStride*1 + i];
        auto bI = data[doubleStride*1 + i + 1];
        auto cR = data[doubleStride*2 + i];
        auto cI = data[doubleStride*2 + i + 1];
        auto dR = data[doubleStride*3 + i];
        auto dI = data[doubleStride*3 + i + 1];

        DoRadix4NoRotation(aR, aI, bR, bI, cR, cI, dR, dI);

        data[doubleStride*0 + i] = aR;
        data[doubleStride*0 + i + 1] = aI;
        data[doubleStride*1 + i] = bR;
        data[doubleStride*1 + i + 1] = bI;
        data[doubleStride*2 + i] = cR;
        data[doubleStride*2 + i + 1] = cI;
        data[doubleStride*3 + i] = dR;
        data[doubleStride*3 + i + 1] = dI;
      }
    }

    // The second set has 90 then 45 (so 45, 90, 135 as R4)
    {
      const auto sectionStart = data + (radix4Stride * 4) * 2;

      for (auto i = 0_z; i < doubleStride; i += 2)
      {
        auto aR = sectionStart[doubleStride*0 + i];
        auto aI = sectionStart[doubleStride*0 + i + 1];
        auto bR = sectionStart[doubleStride*1 + i];
        auto bI = sectionStart[doubleStride*1 + i + 1];
        auto cR = sectionStart[doubleStride*2 + i];
        auto cI = sectionStart[doubleStride*2 + i + 1];
        auto dR = sectionStart[doubleStride*3 + i];
        auto dI = sectionStart[doubleStride*3 + i + 1];

        DoRadix4FirstRotation(aR, aI, bR, bI, cR, cI, dR, dI);

        sectionStart[doubleStride*0 + i] = aR;
        sectionStart[doubleStride*0 + i + 1] = aI;
        sectionStart[doubleStride*1 + i] = bR;
        sectionStart[doubleStride*1 + i + 1] = bI;
        sectionStart[doubleStride*2 + i] = cR;
        sectionStart[doubleStride*2 + i + 1] = cI;
        sectionStart[doubleStride*3 + i] = dR;
        sectionStart[doubleStride*3 + i + 1] = dI;
      }
    }

    // The rest need to load angles
    const auto sectionCount = length / (radix4Stride * 4);
    for (auto sectionIndex = 2_z; sectionIndex < sectionCount; sectionIndex++)
    {
      // !!! on x64 do this as a bit reverse increment instead
      const auto angleIndex = BitRev(sectionIndex) >> (1 + std::countl_zero(sectionCount));

      const auto [w0R, w0I, w1R, w1I, w2R, w2I]
        = LoadSingleAngle(tableData, angleIndex, sectionCount);

      const auto sectionStart = data + (sectionIndex * radix4Stride * 4) * 2;

      for (auto i = 0_z; i < doubleStride; i += 2)
      {
        auto aR = sectionStart[doubleStride*0 + i];
        auto aI = sectionStart[doubleStride*0 + i + 1];
        auto bR = sectionStart[doubleStride*1 + i];
        auto bI = sectionStart[doubleStride*1 + i + 1];
        auto cR = sectionStart[doubleStride*2 + i];
        auto cI = sectionStart[doubleStride*2 + i + 1];
        auto dR = sectionStart[doubleStride*3 + i];
        auto dI = sectionStart[doubleStride*3 + i + 1];

        DoRadix4(aR, aI, bR, bI, cR, cI, dR, dI, w0R, w0I, w1R, w1I, w2R, w2I);

        sectionStart[doubleStride*0 + i] = aR;
        sectionStart[doubleStride*0 + i + 1] = aI;
        sectionStart[doubleStride*1 + i] = bR;
        sectionStart[doubleStride*1 + i + 1] = bI;
        sectionStart[doubleStride*2 + i] = cR;
        sectionStart[doubleStride*2 + i + 1] = cI;
        sectionStart[doubleStride*3 + i] = dR;
        sectionStart[doubleStride*3 + i + 1] = dI;
      }
    }
  }


  template <std::floating_point F>
  void SingleFlipRadix4PassScalar(const F *tableData, F *data, const size_t length, const size_t radix4Stride)
  {
     // There's a different flip pattern for power of 2 and power of 4 single flips
    const auto doubleStride = radix4Stride * 2;
    if (IsPowerOfFourAssumingIsPowerOfTwo(length))
    {
      // For power of four lengths, the flip happens completely within the butterfly (b and c flip). 1024 is a good
      //  test size for this case

      // The first set of stride works with no rotations (besides the 90)
      {
        for (auto i = 0_z; i < doubleStride; i += 2)
        {
          auto aR = data[doubleStride*0 + i];
          auto aI = data[doubleStride*0 + i + 1];
          auto bR = data[doubleStride*1 + i];
          auto bI = data[doubleStride*1 + i + 1];
          auto cR = data[doubleStride*2 + i];
          auto cI = data[doubleStride*2 + i + 1];
          auto dR = data[doubleStride*3 + i];
          auto dI = data[doubleStride*3 + i + 1];

          DoRadix4NoRotation(aR, aI, bR, bI, cR, cI, dR, dI);

          // Write the middle two (c & b) out of order (that's the bit flip). The other two loops do the same.
          data[doubleStride*0 + i] = aR;
          data[doubleStride*0 + i + 1] = aI;
          data[doubleStride*1 + i] = cR;
          data[doubleStride*1 + i + 1] = cI;
          data[doubleStride*2 + i] = bR;
          data[doubleStride*2 + i + 1] = bI;
          data[doubleStride*3 + i] = dR;
          data[doubleStride*3 + i + 1] = dI;
        }
      }

      // The second set has 90 then 45 (so 45, 90, 135 as R4)
      {
        const auto sectionStart = data + (radix4Stride * 4) * 2;

        for (auto i = 0_z; i < doubleStride; i += 2)
        {
          auto aR = sectionStart[doubleStride*0 + i];
          auto aI = sectionStart[doubleStride*0 + i + 1];
          auto bR = sectionStart[doubleStride*1 + i];
          auto bI = sectionStart[doubleStride*1 + i + 1];
          auto cR = sectionStart[doubleStride*2 + i];
          auto cI = sectionStart[doubleStride*2 + i + 1];
          auto dR = sectionStart[doubleStride*3 + i];
          auto dI = sectionStart[doubleStride*3 + i + 1];

          DoRadix4FirstRotation(aR, aI, bR, bI, cR, cI, dR, dI);

          sectionStart[doubleStride*0 + i] = aR;
          sectionStart[doubleStride*0 + i + 1] = aI;
          sectionStart[doubleStride*1 + i] = cR;
          sectionStart[doubleStride*1 + i + 1] = cI;
          sectionStart[doubleStride*2 + i] = bR;
          sectionStart[doubleStride*2 + i + 1] = bI;
          sectionStart[doubleStride*3 + i] = dR;
          sectionStart[doubleStride*3 + i + 1] = dI;
        }
      }

      const auto sectionCount = length / (radix4Stride * 4);
      for (auto sectionIndex = 2_z; sectionIndex < sectionCount; sectionIndex++)
      {
        const auto angleIndex = BitRev(sectionIndex) >> (1 + std::countl_zero(sectionCount));
        auto [w0R, w0I, w1R, w1I, w2R, w2I] = LoadSingleAngle(tableData, angleIndex, sectionCount);

        const auto sectionStart = data + (sectionIndex * radix4Stride * 4) * 2;

        for (auto i = 0_z; i < doubleStride; i += 2)
        {
          auto aR = sectionStart[doubleStride*0 + i];
          auto aI = sectionStart[doubleStride*0 + i + 1];
          auto bR = sectionStart[doubleStride*1 + i];
          auto bI = sectionStart[doubleStride*1 + i + 1];
          auto cR = sectionStart[doubleStride*2 + i];
          auto cI = sectionStart[doubleStride*2 + i + 1];
          auto dR = sectionStart[doubleStride*3 + i];
          auto dI = sectionStart[doubleStride*3 + i + 1];

          DoRadix4(aR, aI, bR, bI, cR, cI, dR, dI, w0R, w0I, w1R, w1I, w2R, w2I);

          sectionStart[doubleStride*0 + i] = aR;
          sectionStart[doubleStride*0 + i + 1] = aI;
          sectionStart[doubleStride*1 + i] = cR;
          sectionStart[doubleStride*1 + i + 1] = cI;
          sectionStart[doubleStride*2 + i] = bR;
          sectionStart[doubleStride*2 + i + 1] = bI;
          sectionStart[doubleStride*3 + i] = dR;
          sectionStart[doubleStride*3 + i + 1] = dI;
        }
      }
    }
    else
    {
      // For non-power-of-four lengths, the flip requires a second butterfly as well. The good news is, it's the *next
      //  sequential* butterfly, which means for the first two butterflies they're still both nice and special-case
      //  (first one is no-rotation, second one is InvSqrt2-only)
      // Then (using the first row as a1/b1/c1/d1 and the second as a2/b2/c2/d2) swap b2/a1 and d2/c1 (the rest stay)
      // 2048 is a good test size for this case.

      // The first two sections are the no-rotation and simple-rotation cases so special case them for efficiency.
      {
        const auto sectionStart0 = data;
        const auto sectionStart1 = sectionStart0 + (radix4Stride * 4) * 2;

        for (auto i = 0_z; i < doubleStride; i += 2)
        {
          auto aR = sectionStart0[doubleStride*0 + i];
          auto aI = sectionStart0[doubleStride*0 + i + 1];
          auto bR = sectionStart0[doubleStride*1 + i];
          auto bI = sectionStart0[doubleStride*1 + i + 1];
          auto cR = sectionStart0[doubleStride*2 + i];
          auto cI = sectionStart0[doubleStride*2 + i + 1];
          auto dR = sectionStart0[doubleStride*3 + i];
          auto dI = sectionStart0[doubleStride*3 + i + 1];

          DoRadix4NoRotation(aR, aI, bR, bI, cR, cI, dR, dI);

          // These can be stored back in-place
          sectionStart0[doubleStride*0 + i] = aR;
          sectionStart0[doubleStride*0 + i + 1] = aI;
          sectionStart0[doubleStride*2 + i] = cR;
          sectionStart0[doubleStride*2 + i + 1] = cI;

          // Now load the two values that get replaced with section 0 values and store over them
          aR = sectionStart1[doubleStride*0 + i];
          aI = sectionStart1[doubleStride*0 + i + 1];

          sectionStart1[doubleStride*0 + i] = bR;
          sectionStart1[doubleStride*0 + i + 1] = bI;

          cR = sectionStart1[doubleStride*2 + i];
          cI = sectionStart1[doubleStride*2 + i + 1];
          sectionStart1[doubleStride*2 + i] = dR;
          sectionStart1[doubleStride*2 + i + 1] = dI;

          // Load the remaining 2
          bR = sectionStart1[doubleStride*1 + i];
          bI = sectionStart1[doubleStride*1 + i + 1];

          dR = sectionStart1[doubleStride*3 + i];
          dI = sectionStart1[doubleStride*3 + i + 1];

          DoRadix4FirstRotation(aR, aI, bR, bI, cR, cI, dR, dI);

          sectionStart0[doubleStride*1 + i] = aR;
          sectionStart0[doubleStride*1 + i + 1] = aI;
          sectionStart1[doubleStride*1 + i] = bR;
          sectionStart1[doubleStride*1 + i + 1] = bI;
          sectionStart0[doubleStride*3 + i] = cR;
          sectionStart0[doubleStride*3 + i + 1] = cI;
          sectionStart1[doubleStride*3 + i] = dR;
          sectionStart1[doubleStride*3 + i + 1] = dI;
        }
      }
      const auto sectionCount = length / (radix4Stride * 4);
      for (auto sectionIndex = 2_z; sectionIndex < sectionCount; sectionIndex += 2)
      {
        // !!! Without native bit reverse, do this as a bitrev increment instead
        const auto angleIndex0 = BitRev(sectionIndex) >> (1 + std::countl_zero(sectionCount));
        auto [w0R0, w0I0, w1R0, w1I0, w2R0, w2I0] = LoadSingleAngle(tableData, angleIndex0, sectionCount);
        const auto angleIndex1 = angleIndex0 + (sectionCount >> 1);
        auto [w0R1, w0I1, w1R1, w1I1, w2R1, w2I1] = LoadSingleAngle(tableData, angleIndex1, sectionCount);

        const auto sectionStart0 = data + (sectionIndex * radix4Stride * 4) * 2;
        const auto sectionStart1 = sectionStart0 + (radix4Stride * 4) * 2;

        for (auto i = 0_z; i < doubleStride; i += 2)
        {
          auto aR = sectionStart0[doubleStride*0 + i];
          auto aI = sectionStart0[doubleStride*0 + i + 1];
          auto bR = sectionStart0[doubleStride*1 + i];
          auto bI = sectionStart0[doubleStride*1 + i + 1];
          auto cR = sectionStart0[doubleStride*2 + i];
          auto cI = sectionStart0[doubleStride*2 + i + 1];
          auto dR = sectionStart0[doubleStride*3 + i];
          auto dI = sectionStart0[doubleStride*3 + i + 1];

          DoRadix4(aR, aI, bR, bI, cR, cI, dR, dI, w0R0, w0I0, w1R0, w1I0, w2R0, w2I0);

          // These can be stored back in-place
          sectionStart0[doubleStride*0 + i] = aR;
          sectionStart0[doubleStride*0 + i + 1] = aI;
          sectionStart0[doubleStride*2 + i] = cR;
          sectionStart0[doubleStride*2 + i + 1] = cI;

          // Now load the two values that get replaced with section 0 values and store over them
          aR = sectionStart1[doubleStride*0 + i];
          aI = sectionStart1[doubleStride*0 + i + 1];

          sectionStart1[doubleStride*0 + i] = bR;
          sectionStart1[doubleStride*0 + i + 1] = bI;

          cR = sectionStart1[doubleStride*2 + i];
          cI = sectionStart1[doubleStride*2 + i + 1];
          sectionStart1[doubleStride*2 + i] = dR;
          sectionStart1[doubleStride*2 + i + 1] = dI;

          // Load the remaining 2
          bR = sectionStart1[doubleStride*1 + i];
          bI = sectionStart1[doubleStride*1 + i + 1];

          dR = sectionStart1[doubleStride*3 + i];
          dI = sectionStart1[doubleStride*3 + i + 1];

          DoRadix4(aR, aI, bR, bI, cR, cI, dR, dI, w0R1, w0I1, w1R1, w1I1, w2R1, w2I1);

          sectionStart0[doubleStride*1 + i] = aR;
          sectionStart0[doubleStride*1 + i + 1] = aI;
          sectionStart1[doubleStride*1 + i] = bR;
          sectionStart1[doubleStride*1 + i + 1] = bI;
          sectionStart0[doubleStride*3 + i] = cR;
          sectionStart0[doubleStride*3 + i + 1] = cI;
          sectionStart1[doubleStride*3 + i] = dR;
          sectionStart1[doubleStride*3 + i + 1] = dI;
        }
      }
    }
  }

  template <std::floating_point F>
  void DoubleFlipRadix4PassScalar(
    const F *tableData,
    F *data,
    const size_t length,
    const size_t radix4Stride)
  {
    const auto sectionCount = length / (radix4Stride * 4);
    const auto flipStride = sectionCount;

    const auto shift = std::countl_zero(length - 1);
    const auto angleIndexFlipInsert = BitRev(flipStride) >> shift;

    for (auto groupStartIndex = 0_z; groupStartIndex < length; groupStartIndex += flipStride * 4)
    {
      for (auto butterflySetStartIndex = 0_z; butterflySetStartIndex < flipStride; butterflySetStartIndex += radix4Stride * 4)
      {
        // we need 4 sets of 3 angles each
        const auto angleIndex0 = (BitRev(groupStartIndex) >> shift) + butterflySetStartIndex;
        const auto angleIndex2 = angleIndex0 | angleIndexFlipInsert;
        const auto angleIndex1 = angleIndex0 | (angleIndexFlipInsert >> 1);
        const auto angleIndex3 = angleIndex1 | angleIndex2;

        const auto [w0R0, w0I0, w1R0, w1I0, w2R0, w2I0] = LoadSingleAngle(tableData, angleIndex0, sectionCount);
        const auto [w0R2, w0I2, w1R2, w1I2, w2R2, w2I2] = LoadSingleAngle(tableData, angleIndex2, sectionCount);
        const auto [w0R1, w0I1, w1R1, w1I1, w2R1, w2I1] = LoadSingleAngle(tableData, angleIndex1, sectionCount);
        const auto [w0R3, w0I3, w1R3, w1I3, w2R3, w2I3] = LoadSingleAngle(tableData, angleIndex3, sectionCount);

        // Use the partially-bit-reversed "groupStartIndex + butterflySetStartIndex" to get the angle
        for (auto butterflyStartIndex = 0_z; butterflyStartIndex < radix4Stride; butterflyStartIndex++)
        {
          auto i = groupStartIndex + butterflySetStartIndex + butterflyStartIndex;

          auto aR0 = data[2 * (i + flipStride * 0 + radix4Stride * 0) + 0];
          auto aI0 = data[2 * (i + flipStride * 0 + radix4Stride * 0) + 1];
          auto bR0 = data[2 * (i + flipStride * 0 + radix4Stride * 1) + 0];
          auto bI0 = data[2 * (i + flipStride * 0 + radix4Stride * 1) + 1];
          auto cR0 = data[2 * (i + flipStride * 0 + radix4Stride * 2) + 0];
          auto cI0 = data[2 * (i + flipStride * 0 + radix4Stride * 2) + 1];
          auto dR0 = data[2 * (i + flipStride * 0 + radix4Stride * 3) + 0];
          auto dI0 = data[2 * (i + flipStride * 0 + radix4Stride * 3) + 1];

          DoRadix4(aR0, aI0, bR0, bI0, cR0, cI0, dR0, dI0, w0R0, w0I0, w1R0, w1I0, w2R0, w2I0);

          // Now store all of the '0's, swapping in 'a's as we go
          data[2 * (i + flipStride * 0 + radix4Stride * 0) + 0] = aR0;
          data[2 * (i + flipStride * 0 + radix4Stride * 0) + 1] = aI0;

          auto aR1 = data[2 * (i + flipStride * 2 + radix4Stride * 0) + 0];
          auto aI1 = data[2 * (i + flipStride * 2 + radix4Stride * 0) + 1];
          data[2 * (i + flipStride * 2 + radix4Stride * 0) + 0] = bR0;
          data[2 * (i + flipStride * 2 + radix4Stride * 0) + 1] = bI0;

          auto aR2 = data[2 * (i + flipStride * 1 + radix4Stride * 0) + 0];
          auto aI2 = data[2 * (i + flipStride * 1 + radix4Stride * 0) + 1];
          data[2 * (i + flipStride * 1 + radix4Stride * 0) + 0] = cR0;
          data[2 * (i + flipStride * 1 + radix4Stride * 0) + 1] = cI0;

          auto aR3 = data[2 * (i + flipStride * 3 + radix4Stride * 0) + 0];
          auto aI3 = data[2 * (i + flipStride * 3 + radix4Stride * 0) + 1];
          data[2 * (i + flipStride * 3 + radix4Stride * 0) + 0] = dR0;
          data[2 * (i + flipStride * 3 + radix4Stride * 0) + 1] = dI0;

          // At this point we have 6 loaded sets (a1, a2, a3)

          auto bR1 = data[2 * (i + flipStride * 2 + radix4Stride * 1) + 0];
          auto bI1 = data[2 * (i + flipStride * 2 + radix4Stride * 1) + 1];
          auto cR1 = data[2 * (i + flipStride * 2 + radix4Stride * 2) + 0];
          auto cI1 = data[2 * (i + flipStride * 2 + radix4Stride * 2) + 1];
          auto dR1 = data[2 * (i + flipStride * 2 + radix4Stride * 3) + 0];
          auto dI1 = data[2 * (i + flipStride * 2 + radix4Stride * 3) + 1];

          // Now there are 12 loaded (a1, b1, c1, d1, a2, a3)
          DoRadix4(aR1, aI1, bR1, bI1, cR1, cI1, dR1, dI1, w0R1, w0I1, w1R1, w1I1, w2R1, w2I1);

          // Store everything (Swapping in bs as we go)
          data[2 * (i + flipStride * 0 + radix4Stride * 1) + 0] = aR1;
          data[2 * (i + flipStride * 0 + radix4Stride * 1) + 1] = aI1;

          data[2 * (i + flipStride * 2 + radix4Stride * 1) + 0] = bR1;
          data[2 * (i + flipStride * 2 + radix4Stride * 1) + 1] = bI1;

          auto bR2 = data[2 * (i + flipStride * 1 + radix4Stride * 1) + 0];
          auto bI2 = data[2 * (i + flipStride * 1 + radix4Stride * 1) + 1];
          data[2 * (i + flipStride * 1 + radix4Stride * 1) + 0] = cR1;
          data[2 * (i + flipStride * 1 + radix4Stride * 1) + 1] = cI1;

          auto bR3 = data[2 * (i + flipStride * 3 + radix4Stride * 1) + 0];
          auto bI3 = data[2 * (i + flipStride * 3 + radix4Stride * 1) + 1];
          data[2 * (i + flipStride * 3 + radix4Stride * 1) + 0] = dR1;
          data[2 * (i + flipStride * 3 + radix4Stride * 1) + 1] = dI1;

          // Now there are 8 loaded (a2, b2, a3, b3)

          auto cR2 = data[2 * (i + flipStride * 1 + radix4Stride * 2) + 0];
          auto cI2 = data[2 * (i + flipStride * 1 + radix4Stride * 2) + 1];
          auto dR2 = data[2 * (i + flipStride * 1 + radix4Stride * 3) + 0];
          auto dI2 = data[2 * (i + flipStride * 1 + radix4Stride * 3) + 1];

          // Back up to 12 (a2, b2, c2, d2, a3, b3)
          DoRadix4(aR2, aI2, bR2, bI2, cR2, cI2, dR2, dI2, w0R2, w0I2, w1R2, w1I2, w2R2, w2I2);

          data[2 * (i + flipStride * 0 + radix4Stride * 2) + 0] = aR2;
          data[2 * (i + flipStride * 0 + radix4Stride * 2) + 1] = aI2;
          data[2 * (i + flipStride * 2 + radix4Stride * 2) + 0] = bR2;
          data[2 * (i + flipStride * 2 + radix4Stride * 2) + 1] = bI2;
          data[2 * (i + flipStride * 1 + radix4Stride * 2) + 0] = cR2;
          data[2 * (i + flipStride * 1 + radix4Stride * 2) + 1] = cI2;

          auto cR3 = data[2 * (i + flipStride * 3 + radix4Stride * 2) + 0];
          auto cI3 = data[2 * (i + flipStride * 3 + radix4Stride * 2) + 1];
          data[2 * (i + flipStride * 3 + radix4Stride * 2) + 0] = dR2;
          data[2 * (i + flipStride * 3 + radix4Stride * 2) + 1] = dI2;

          auto dR3 = data[2 * (i + flipStride * 3 + radix4Stride * 3) + 0];
          auto dI3 = data[2 * (i + flipStride * 3 + radix4Stride * 3) + 1];

          // Now just the '3's are left (8 sets)

          DoRadix4(aR3, aI3, bR3, bI3, cR3, cI3, dR3, dI3, w0R3, w0I3, w1R3, w1I3, w2R3, w2I3);

          data[2 * (i + flipStride * 0 + radix4Stride * 3) + 0] = aR3;
          data[2 * (i + flipStride * 0 + radix4Stride * 3) + 1] = aI3;
          data[2 * (i + flipStride * 2 + radix4Stride * 3) + 0] = bR3;
          data[2 * (i + flipStride * 2 + radix4Stride * 3) + 1] = bI3;
          data[2 * (i + flipStride * 1 + radix4Stride * 3) + 0] = cR3;
          data[2 * (i + flipStride * 1 + radix4Stride * 3) + 1] = cI3;
          data[2 * (i + flipStride * 3 + radix4Stride * 3) + 0] = dR3;
          data[2 * (i + flipStride * 3 + radix4Stride * 3) + 1] = dI3;
        }
      }
    }
  }

  template <std::floating_point F>
  void LastDoubleFlipRadix4PassScalar(const F *tableData, F *data, const size_t length)
  {
    const auto flipStride = length / 4;

    for (auto butterflySetStartIndex = 0_z; butterflySetStartIndex < flipStride; butterflySetStartIndex += 4)
    {
      // we need 4 sets of 3 angles each
      // in SSE/AVX this could be 3 vectors of 4 angles (duplicated for AVX cross lane), then use temp space
      //  to shuffle the correct 3 elements in to all lanes, but that's 6 vectors, so we may instead need to just
      //  stash these on the stack and load them (theoretically from L1 directly). Hopefully that's good enough!
      //  (also it's possible the other implementation already has to do similar, I don't know for sure)

      const auto angleIndex0 = butterflySetStartIndex;
      const auto angleIndex2 = angleIndex0 | 2;
      const auto angleIndex1 = angleIndex0 | 1;
      const auto angleIndex3 = angleIndex0 | 3;

      const auto [w0R0, w0I0, w1R0, w1I0, w2R0, w2I0] = LoadSingleAngle(tableData, angleIndex0, flipStride);
      const auto [w0R2, w0I2, w1R2, w1I2, w2R2, w2I2] = LoadSingleAngle(tableData, angleIndex2, flipStride);
      const auto [w0R1, w0I1, w1R1, w1I1, w2R1, w2I1] = LoadSingleAngle(tableData, angleIndex1, flipStride);
      const auto [w0R3, w0I3, w1R3, w1I3, w2R3, w2I3] = LoadSingleAngle(tableData, angleIndex3, flipStride);

      // Use the partially-bit-reversed "groupStartIndex + butterflySetStartIndex" to get the angle

      auto i = butterflySetStartIndex;

      auto aR0 = data[2 * (i + flipStride * 0 + 0) + 0];
      auto aI0 = data[2 * (i + flipStride * 0 + 0) + 1];
      auto bR0 = data[2 * (i + flipStride * 0 + 1) + 0];
      auto bI0 = data[2 * (i + flipStride * 0 + 1) + 1];
      auto cR0 = data[2 * (i + flipStride * 0 + 2) + 0];
      auto cI0 = data[2 * (i + flipStride * 0 + 2) + 1];
      auto dR0 = data[2 * (i + flipStride * 0 + 3) + 0];
      auto dI0 = data[2 * (i + flipStride * 0 + 3) + 1];

      DoRadix4(aR0, aI0, bR0, bI0, cR0, cI0, dR0, dI0, w0R0, w0I0, w1R0, w1I0, w2R0, w2I0);

      // Now store all of the '0's, swapping in 'a's as we go
      data[2 * (i + flipStride * 0 + 0) + 0] = aR0;
      data[2 * (i + flipStride * 0 + 0) + 1] = aI0;

      auto aR1 = data[2 * (i + flipStride * 2 + 0) + 0];
      auto aI1 = data[2 * (i + flipStride * 2 + 0) + 1];
      data[2 * (i + flipStride * 2 + 0) + 0] = bR0;
      data[2 * (i + flipStride * 2 + 0) + 1] = bI0;

      auto aR2 = data[2 * (i + flipStride * 1 + 0) + 0];
      auto aI2 = data[2 * (i + flipStride * 1 + 0) + 1];
      data[2 * (i + flipStride * 1 + 0) + 0] = cR0;
      data[2 * (i + flipStride * 1 + 0) + 1] = cI0;

      auto aR3 = data[2 * (i + flipStride * 3 + 0) + 0];
      auto aI3 = data[2 * (i + flipStride * 3 + 0) + 1];
      data[2 * (i + flipStride * 3 + 0) + 0] = dR0;
      data[2 * (i + flipStride * 3 + 0) + 1] = dI0;

      // At this point we have 6 loaded sets (a1, a2, a3)

      auto bR1 = data[2 * (i + flipStride * 2 + 1) + 0];
      auto bI1 = data[2 * (i + flipStride * 2 + 1) + 1];
      auto cR1 = data[2 * (i + flipStride * 2 + 2) + 0];
      auto cI1 = data[2 * (i + flipStride * 2 + 2) + 1];
      auto dR1 = data[2 * (i + flipStride * 2 + 3) + 0];
      auto dI1 = data[2 * (i + flipStride * 2 + 3) + 1];

      // Now there are 12 loaded (a1, b1, c1, d1, a2, a3)
      DoRadix4(aR1, aI1, bR1, bI1, cR1, cI1, dR1, dI1, w0R1, w0I1, w1R1, w1I1, w2R1, w2I1);

      // Store everything (Swapping in bs as we go)
      data[2 * (i + flipStride * 0 + 1) + 0] = aR1;
      data[2 * (i + flipStride * 0 + 1) + 1] = aI1;

      data[2 * (i + flipStride * 2 + 1) + 0] = bR1;
      data[2 * (i + flipStride * 2 + 1) + 1] = bI1;

      auto bR2 = data[2 * (i + flipStride * 1 + 1) + 0];
      auto bI2 = data[2 * (i + flipStride * 1 + 1) + 1];
      data[2 * (i + flipStride * 1 + 1) + 0] = cR1;
      data[2 * (i + flipStride * 1 + 1) + 1] = cI1;

      auto bR3 = data[2 * (i + flipStride * 3 + 1) + 0];
      auto bI3 = data[2 * (i + flipStride * 3 + 1) + 1];
      data[2 * (i + flipStride * 3 + 1) + 0] = dR1;
      data[2 * (i + flipStride * 3 + 1) + 1] = dI1;

      // Now there are 8 loaded (a2, b2, a3, b3)

      auto cR2 = data[2 * (i + flipStride * 1 + 2) + 0];
      auto cI2 = data[2 * (i + flipStride * 1 + 2) + 1];
      auto dR2 = data[2 * (i + flipStride * 1 + 3) + 0];
      auto dI2 = data[2 * (i + flipStride * 1 + 3) + 1];

      // Back up to 12 (a2, b2, c2, d2, a3, b3)
      DoRadix4(aR2, aI2, bR2, bI2, cR2, cI2, dR2, dI2, w0R2, w0I2, w1R2, w1I2, w2R2, w2I2);

      data[2 * (i + flipStride * 0 + 2) + 0] = aR2;
      data[2 * (i + flipStride * 0 + 2) + 1] = aI2;
      data[2 * (i + flipStride * 2 + 2) + 0] = bR2;
      data[2 * (i + flipStride * 2 + 2) + 1] = bI2;
      data[2 * (i + flipStride * 1 + 2) + 0] = cR2;
      data[2 * (i + flipStride * 1 + 2) + 1] = cI2;

      auto cR3 = data[2 * (i + flipStride * 3 + 2) + 0];
      auto cI3 = data[2 * (i + flipStride * 3 + 2) + 1];
      data[2 * (i + flipStride * 3 + 2) + 0] = dR2;
      data[2 * (i + flipStride * 3 + 2) + 1] = dI2;

      auto dR3 = data[2 * (i + flipStride * 3 + 3) + 0];
      auto dI3 = data[2 * (i + flipStride * 3 + 3) + 1];

      // Now just the '3's are left (8 sets)

      DoRadix4(aR3, aI3, bR3, bI3, cR3, cI3, dR3, dI3, w0R3, w0I3, w1R3, w1I3, w2R3, w2I3);

      data[2 * (i + flipStride * 0 + 3) + 0] = aR3;
      data[2 * (i + flipStride * 0 + 3) + 1] = aI3;
      data[2 * (i + flipStride * 2 + 3) + 0] = bR3;
      data[2 * (i + flipStride * 2 + 3) + 1] = bI3;
      data[2 * (i + flipStride * 1 + 3) + 0] = cR3;
      data[2 * (i + flipStride * 1 + 3) + 1] = cI3;
      data[2 * (i + flipStride * 3 + 3) + 0] = dR3;
      data[2 * (i + flipStride * 3 + 3) + 1] = dI3;
    }
  }

  constexpr auto SmallestPiecewiseComplexFFT = 16u; // !!! this constant already exists in run.h
  constexpr auto SIMDSetSize = 1u;

  template <std::floating_point F>
  void MiddleOutFFT(
    const F *tableData,
    const F *input,
    F *output,
    size_t length)
  {
    // !!! This minimum length should be declared by the implementation (AVX wants a larger one, for instance)
    if (length < SmallestPiecewiseComplexFFT)
    {
      // Special cases (64 may also want to be one in AVX since everything could *almost* fit in registers!)
      // !!! call TinyFFT
      Scalar::RunComplexFFT<Dir::Forward>(tableData, input, output, length);
      return;
    }

    const auto bitCount = sizeof(length) * 8 - std::countl_zero(length) - 1;
    const auto bitReverseCount = bitCount >> 1;

    // Note that this folds the radix 2 pass into the first radix 4/8 pass for non-power-of-four FFTs, so it's actually
    //  one less than would traditionally be the case (and ends up matching the bit reverse count)

    // This one *does* want to take into account the extra bit reverse though.
    //const auto radix4BitReverseStepCount = (bitReverseCount + 1) >> 1;

    // Step 0: Either a radix 4 or 8 (depending on if pow4) pass with no rotation
    Radix4Or8InitialStepScalar(input, output, length);

    auto radix4Stride = length >> (IsPowerOfFourAssumingIsPowerOfTwo(length) ? 4 : 5);

    // The remainder of the first half of the steps (rounded down) have no bit flips to them, they're just regular boring
    //  radix 4 passes.
    for (const auto endStride = (1_z << bitReverseCount); radix4Stride >= endStride; radix4Stride >>= 2)
      { NoFlipRadix4PassScalar(tableData, output, length, radix4Stride); }

    if ((bitReverseCount & 1) != 0)
    {
      // If there is an odd number of bit reverses needed, then the first reverse is going to be a single bit flip.
      SingleFlipRadix4PassScalar(tableData, output, length, radix4Stride);
      radix4Stride >>= 2;
    }

    // Loop until the stride is no longer larger than the SIMD set size - in this case we want the special-case ones.
    for (; radix4Stride > SIMDSetSize; radix4Stride >>= 2)
      { DoubleFlipRadix4PassScalar(tableData, output, length, radix4Stride); }

    if constexpr (SIMDSetSize > 4u)
    {
      static_assert(SIMDSetSize <= 8u);
      // This needs a special-case because the second-to-last pass has a stride that is smaller than the SIMD set.
      // !!! SecondToLastDoubleFlipRadix4PassScalar(tableData, output, length);
    }

    // Step N - 1:
    //  The final flip (Which has to transpose within a SIMD set)
    LastDoubleFlipRadix4PassScalar(tableData, output, length);
  }
}

extern "C"
{
  void fft_cplx_interleaved_f32_scalar(const fft_table_f32 *table, const float *input, float *output, size_t fftLength)
  {
    [[maybe_unused]] auto t = reinterpret_cast<const FFTTableImpl<float> *>(table);
    COMMON_ASSERTS();

    MiddleOutFFT(t->Data(), input, output, fftLength);
  }


  void fft_cplx_interleaved_f64_scalar(const fft_table_f64 *table, const double *input, double *output, size_t fftLength)
  {
    [[maybe_unused]] auto t = reinterpret_cast<const FFTTableImpl<double> *>(table);
    COMMON_ASSERTS();
    Scalar::RunComplexFFT<Dir::Forward>(t->Data(), input, output, fftLength);
  }


  void ifft_cplx_interleaved_f32_scalar(const fft_table_f32 *table, const float *input, float *output, size_t fftLength)
  {
    [[maybe_unused]] auto t = reinterpret_cast<const FFTTableImpl<float> *>(table);
    COMMON_ASSERTS();
    Scalar::RunComplexFFT<Dir::Inverse>(t->Data(), input, output, fftLength);
  }


  void ifft_cplx_interleaved_f64_scalar(const fft_table_f64 *table, const double *input, double *output, size_t fftLength)
  {
    [[maybe_unused]] auto t = reinterpret_cast<const FFTTableImpl<double> *>(table);
    COMMON_ASSERTS();
    Scalar::RunComplexFFT<Dir::Inverse>(t->Data(), input, output, fftLength);
  }

  void fft_real_f32_scalar(const fft_table_f32 *table, const float *input, float *output, size_t fftLength)
  {
    auto t = reinterpret_cast<const FFTTableImpl<float> *>(table);
    COMMON_ASSERTS();
    Scalar::RunRealFwdFFT(t->Data(), input, output, fftLength);
  }

  void fft_real_f64_scalar(const fft_table_f64 *table, const double *input, double *output, size_t fftLength)
  {
    auto t = reinterpret_cast<const FFTTableImpl<double> *>(table);
    COMMON_ASSERTS();
    Scalar::RunRealFwdFFT(t->Data(), input, output, fftLength);
  }

  void ifft_real_f32_scalar(const fft_table_f32 *table, const float *input, float *output, size_t fftLength)
  {
    auto t = reinterpret_cast<const FFTTableImpl<float> *>(table);
    COMMON_ASSERTS();
    Scalar::RunRealInvFFT(t->Data(), input, output, fftLength);
  }

  void ifft_real_f64_scalar(const fft_table_f64 *table, const double *input, double *output, size_t fftLength)
  {
    auto t = reinterpret_cast<const FFTTableImpl<double> *>(table);
    COMMON_ASSERTS();
    Scalar::RunRealInvFFT(t->Data(), input, output, fftLength);
  }
}