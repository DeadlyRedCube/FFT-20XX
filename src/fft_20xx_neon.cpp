#include "fft_20xx_common.h"
#include "fft_20xx_table.h"

static_assert(SIMD_NEON);

#include <arm_neon.h>

namespace FFT20XX::NEON
{
  template <typename T> concept f32 = std::same_as<float, T> || std::same_as<float32x4_t, T>;
  template <typename T> concept simd_vec = std::same_as<std::remove_cvref_t<T>, float32x4_t> || std::same_as<std::remove_cvref_t<T>, float64x2_t>;
  template <typename T> using Vec = std::conditional_t<f32<T>, float32x4_t, float64x2_t>;

  template <typename> constexpr uint32_t VecSize = 0;

  template <> constexpr uint32_t VecSize<Vec<float>> = 4u;
  template <> constexpr uint32_t VecSize<Vec<double>> = 2u;

  constexpr bool IsScalar = false;
  constexpr bool UseFMAOptimization = false;

  template <std::floating_point F>
  static auto LoadA(const F *ptr) -> Vec<F>
  {
    if constexpr (f32<F>)
      { return vld1q_f32(ptr); }
    else
      { return vld1q_f64(ptr); }
  }
  template <std::floating_point F>
  static auto Load(const F *ptr) -> Vec<F>
    { return LoadA(ptr); }

  template <std::floating_point F>
  static auto Load4(const F *ptr) -> Vec<F>
    { return LoadA(ptr); }

  template <std::floating_point F>
  static auto LoadDeinterleaved(const F *ptr) -> Pair<Vec<F>, Vec<F>>
  {
    if constexpr (f32<F>)
    {
      auto d = vld2q_f32(ptr);
      return { d.val[0], d.val[1] };
    }
    else
    {
      auto d = vld2q_f64(ptr);
      return { d.val[0], d.val[1] };
    }
  }

  template <std::floating_point F>
  static void Store2x2Transposed(F *ptrA, F *ptrB, Vec<F> a, Vec<F> b)
  {
    auto ta = vtrn1q_f64(a, b);
    auto tb = vtrn2q_f64(a, b);
    vst1q_f64(ptrA, ta);
    vst1q_f64(ptrB, tb);
  }

  template <std::floating_point F>
  static void Store4x4Transposed(F *ptrA, F *ptrB, F *ptrC, F *ptrD, Vec<F> a, Vec<F> b, Vec<F> c, Vec<F> d)
  {
    static_assert(f32<F>);
    auto ta = vtrn1q_f32(a, b);
    auto tb = vtrn2q_f32(a, b);
    auto tc = vtrn1q_f32(c, d);
    auto td = vtrn2q_f32(c, d);

    a = vreinterpretq_f32_f64(vtrn1q_f64(vreinterpretq_f64_f32(ta), vreinterpretq_f64_f32(tc)));
    b = vreinterpretq_f32_f64(vtrn1q_f64(vreinterpretq_f64_f32(tb), vreinterpretq_f64_f32(td)));
    c = vreinterpretq_f32_f64(vtrn2q_f64(vreinterpretq_f64_f32(ta), vreinterpretq_f64_f32(tc)));
    d = vreinterpretq_f32_f64(vtrn2q_f64(vreinterpretq_f64_f32(tb), vreinterpretq_f64_f32(td)));
    vst1q_f32(ptrA, a);
    vst1q_f32(ptrB, b);
    vst1q_f32(ptrC, c);
    vst1q_f32(ptrD, d);
  }

  template <std::floating_point F>
  static void Store(F *ptr, Vec<F> a)
  {
    if constexpr (f32<F>)
      { vst1q_f32(ptr, a); }
    else
      { vst1q_f64(ptr, a); }
  }

  template <std::floating_point F>
  static void StoreInterleaved(F *ptr, Vec<F> a, Vec<F> b)
  {
    if constexpr (f32<F>)
      { vst2q_f32(ptr, (float32x4x2_t{a, b})); }
    else
      { vst2q_f64(ptr, (float64x2x2_t{a, b})); }
  }

  template <std::floating_point F, std::same_as<Vec<F>> ... Vs>
  static void Scale(F scale, Vs &...vs)
  {
    if constexpr (f32<F>)
      { ((vs = vmulq_n_f32(vs, scale)), ...); }
    else
      { ((vs = vmulq_n_f64(vs, scale)), ...); }
  }

  template <simd_vec V>
  static V Add(V a, V b)
  {
    if constexpr (f32<V>)
      { return vaddq_f32(a, b); }
    else
      { return vaddq_f64(a, b); }
  }

  template <simd_vec V>
  static V Sub(V a, V b)
  {
    if constexpr (f32<V>)
      { return vsubq_f32(a, b); }
    else
      { return vsubq_f64(a, b); }
  }

  template <simd_vec V>
  static V Mul(V a, V b)
  {
    if constexpr (f32<V>)
      { return vmulq_f32(a, b); }
    else
      { return vmulq_f64(a, b); }
  }

  template <simd_vec V>
  static auto Set1(std::conditional_t<f32<V>, float, double> f) -> V
  {
    if constexpr (f32<V>)
      { return vdupq_n_f32(f); }
    else
      { return vdupq_n_f64(f); }
  }

  template <simd_vec V>
  static V MAdd(V a, V b, V c)
  {
    if constexpr (f32<V>)
      { return vmlaq_f32(c, b, a); }
    else
      { return vfmaq_f64(c, b, a); }
  }

  template <simd_vec V>
  static V NMAdd(V a, V b, V c)
  {
    if constexpr (f32<V>)
      { return vmlsq_f32(c, b, a); }
    else
      { return vfmsq_f64(c, b, a); }
  }

  // Forward Real Transform ///////////////////////////////////////////////////////////////////////////////////////////

  template <typename F>
  inline auto LoadFirstRightSideForForwardRealTransform(const F *input, Vec<F> &prevR, Vec<F> &prevI) ->
    Pair<Vec<F>, Vec<F>>
  {
    if constexpr (f32<Vec<F>>)
    {
      // Load:   e d c b
      // Target: X b c d  (X mean doesn't matter)
      auto t = vld1q_f32_x2(input);
      auto EDCB = t.val[0];
      auto edcb = t.val[1];
      prevR = EDCB;
      prevI = edcb;

      auto BEDC = vextq_f32(EDCB, EDCB, 3);
      auto bedc = vextq_f32(edcb, edcb, 3);
      auto EBCD = vtrn1q_f32(EDCB, BEDC);
      auto ebcd = vtrn1q_f32(edcb, bedc);
      return { EBCD, ebcd };
    }
    else
    {
      // Load: c b
      auto t = vld1q_f64_x2(input);
      prevR = t.val[0];
      prevI = t.val[1];

      // Target: X b
      return { prevR, prevI };
    }
  }

  template <typename F>
  inline auto LoadRightSideForForwardRealTransform(const F *input, Vec<F> &prevR, Vec<F> &prevI) ->
    Pair<Vec<F>, Vec<F>>
  {
    if constexpr (f32<Vec<F>>)
    {
      // Prev:   e d c b
      // Cur:    i h g f
      // Target: e f g h

      auto t = vld1q_f32_x2(input);
      auto IHGF = t.val[0];
      auto ihgf = t.val[1];

      auto HGFE = vextq_f32(IHGF, prevR, 1);
      auto hgfe = vextq_f32(ihgf, prevI, 1);

      prevR = IHGF;
      prevI = ihgf;

      auto GHEF = vrev64q_f32(HGFE);
      auto ghef = vrev64q_f32(hgfe);

      auto EFGH = vextq_f32(GHEF, GHEF, 2);
      auto efgh = vextq_f32(ghef, ghef, 2);

      return { EFGH, efgh };
    }
    else
    {
      // Prev:  c b
      // Cur:   e d
      // Target: c d

      auto t = vld1q_f64_x2(input);
      auto ED = t.val[0];
      auto ed = t.val[1];

      auto CD = vcopyq_laneq_f64(prevR, 1, ED, 1);
      auto cd = vcopyq_laneq_f64(prevI, 1, ed, 1);

      prevR = ED;
      prevI = ed;

      return { CD, cd };
    }
  }

  template <typename F>
  inline void StoreRightSideForForwardRealTransform(F *output, Vec<F> r1, Vec<F> i1, Vec<F> &prevR, Vec<F> &prevI)
  {
    if constexpr (f32<F>)
    {
      // Prev:    A B C D | a b c d (REAL, imaginary)
      // Current: E F G H | e f g h
      // Target:  E e D d | C c B b

      auto BCDE = vextq_f32(prevR, r1, 1);
      auto bcde = vextq_f32(prevI, i1, 1);
      prevR = r1;
      prevI = i1;
      auto CBED = vrev64q_f32(BCDE);
      auto cbed = vrev64q_f32(bcde);
      auto EDCB = vextq_f32(CBED, CBED, 2);
      auto edcb = vextq_f32(cbed, cbed, 2);
      vst2q_f32(output, (float32x4x2_t{EDCB, edcb}));
    }
    else
    {
      // Prev: A B | a b
      // Current: C D | c d
      // Target: C c | B b
      auto Cc = vcopyq_laneq_f64(r1, 1, i1, 0);
      auto Bb = vcopyq_laneq_f64(prevI, 0, prevR, 1);
      prevR = r1;
      prevI = i1;
      vst1q_f64_x2(output, (float64x2x2_t{Cc, Bb}));
    }
  }

  template <typename F>
  inline void StoreFinalRightSideForForwardRealTransform(F *output, Vec<F> r1, Vec<F> i1)
  {
    if constexpr (f32<F>)
    {
      // final: X N O P | X n o p  (X means "don't care")
      // target: X X P p | O o N n
      auto NXPO = vrev64q_f32(r1);
      auto nXpo = vrev64q_f32(i1);
      auto XPON = vextq_f32(NXPO, NXPO, 1);
      auto Xpon = vextq_f32(nXpo, nXpo, 1);
      vst2q_f32(output, (float32x4x2_t{XPON, Xpon}));
    }
    else
    {
      // Final: X N | X n
      // Target : X X | N n
      vst2q_f64(output, (float64x2x2_t{r1, i1}));
    }
  }

  // Inverse Real Transform ///////////////////////////////////////////////////////////////////////////////////////////

  template <typename F>
  inline auto LoadFirstRightSideForInverseRealTransform(const F *input, Vec<F> &prevR, Vec<F> &prevI) ->
    Pair<Vec<F>, Vec<F>>
  {
    if constexpr (f32<Vec<F>>)
    {
      // Load: E D C B | e d c b (auto-deinterleaved)
      // Target: X B C D | X b c d
      auto EDCBedcb = vld2q_f32(input);
      auto EDCB = EDCBedcb.val[0];
      auto edcb = EDCBedcb.val[1];
      prevR = EDCB;
      prevI = edcb;
      auto BEDC = vextq_f32(EDCB, EDCB, 3);
      auto bedc = vextq_f32(edcb, edcb, 3);
      auto EBCD = vtrn1q_f32(EDCB, BEDC);
      auto ebcd = vtrn1q_f32(edcb, bedc);

      return { EBCD, ebcd };
    }
    else
    {
      // Load: C B c b  (auto-deinterleaved)
      // Target: X B | X b
      auto CBcb = vld2q_f64(input);
      prevR = CBcb.val[0];
      prevI = CBcb.val[1];
      return { prevR, prevI };
    }
  }

  template <typename F>
  inline auto LoadRightSideForInverseRealTransform(const F *input, Vec<F> &prevR, Vec<F> &prevI) ->
    Pair<Vec<F>, Vec<F>>
  {
    if constexpr (f32<Vec<F>>)
    {
      // Prev: E X X X | e X X X
      // Cur: I H G F | i h g f  (auto-deinterleaved)
      // Target: E F G H | e f g h

      auto IHGFihgf = vld2q_f32(input);
      auto IHGF = IHGFihgf.val[0];
      auto ihgf = IHGFihgf.val[1];

      auto HGFE = vextq_f32(IHGF, prevR, 1);
      auto hgfe = vextq_f32(ihgf, prevI, 1);

      prevR = IHGF;
      prevI = ihgf;

      auto GHEF = vrev64q_f32(HGFE);
      auto ghef = vrev64q_f32(hgfe);

      auto EFGH = vextq_f32(GHEF, GHEF, 2);
      auto efgh = vextq_f32(ghef, ghef, 2);

      return { EFGH, efgh };
    }
    else
    {
      // Prev: C B | c b
      // Cur: E D | e d (auto de-interleaved)
      auto EDed = vld2q_f64(input);
      // Target: C D | c d
      auto CD = vcopyq_laneq_f64(prevR, 1, EDed.val[0], 1);
      auto cd = vcopyq_laneq_f64(prevI, 1, EDed.val[1], 1);

      prevR = EDed.val[0];
      prevI = EDed.val[1];
      return { CD, cd };
    }
  }

  template <typename F>
  inline void StoreRightSideForInverseRealTransform(F *output, Vec<F> r1, Vec<F> i1, Vec<F> &prevR, Vec<F> &prevI)
  {
    if constexpr (f32<F>)
    {
      // prev: A B C D
      // cur: E F G H
      // Target: E D C B
      auto BCDE = vextq_f32(prevR, r1, 1);
      auto bcde = vextq_f32(prevI, i1, 1);
      prevR = r1;
      prevI = i1;
      auto CBED = vrev64q_f32(BCDE);
      auto cbed = vrev64q_f32(bcde);
      auto EDCB = vextq_f32(CBED, CBED, 2);
      auto edcb = vextq_f32(cbed, cbed, 2);
      vst1q_f32_x2(output, (float32x4x2_t{EDCB, edcb}));
    }
    else
    {
      // Prev: A B
      // Cur: C D
      // Target: C B
      auto CB = vcopyq_laneq_f64(r1, 1, prevR, 1);
      auto cb = vcopyq_laneq_f64(i1, 1, prevI, 1);
      prevR = r1;
      prevI = i1;
      vst1q_f64_x2(output, (float64x2x2_t{CB, cb}));
    }
  }

  template <typename F>
  inline void StoreFinalRightSideForInverseRealTransform(F *output, Vec<F> r1, Vec<F> i1)
  {
    if constexpr (f32<F>)
    {
      // final: X N O P
      // target: X P O N
      auto NXPO = vrev64q_f32(r1);
      auto nXpo = vrev64q_f32(i1);
      auto XPON = vextq_f32(NXPO, NXPO, 1);
      auto Xpon = vextq_f32(nXpo, nXpo, 1);
      vst1q_f32_x2(output, (float32x4x2_t{XPON, Xpon}));
    }
    else
    {
      // Conveniently this one is fine as is: X N | X n
      vst1q_f64_x2(output, (float64x2x2_t{r1, i1}));
    }
  }

  #include "fft_20xx_run.h"
}


using namespace FFT20XX;
using namespace fft_20xx::c;
extern "C"
{
  void fft_cplx_interleaved_f32_neon(const fft_table_f32 *table, const float *input, float *output, size_t fftLength)
  {
    [[maybe_unused]] auto t = reinterpret_cast<const FFTTableImpl<float> *>(table);
    COMMON_ASSERTS();
    NEON::RunComplexFFT<Dir::Forward>(t->Data(), input, output, fftLength);
  }


  void fft_cplx_interleaved_f64_neon(const fft_table_f64 *table, const double *input, double *output, size_t fftLength)
  {
    [[maybe_unused]] auto t = reinterpret_cast<const FFTTableImpl<double> *>(table);
    COMMON_ASSERTS();
    NEON::RunComplexFFT<Dir::Forward>(t->Data(), input, output, fftLength);
  }


  void ifft_cplx_interleaved_f32_neon(const fft_table_f32 *table, const float *input, float *output, size_t fftLength)
  {
    [[maybe_unused]] auto t = reinterpret_cast<const FFTTableImpl<float> *>(table);
    COMMON_ASSERTS();
    NEON::RunComplexFFT<Dir::Inverse>(t->Data(), input, output, fftLength);
  }


  void ifft_cplx_interleaved_f64_neon(const fft_table_f64 *table, const double *input, double *output, size_t fftLength)
  {
    [[maybe_unused]] auto t = reinterpret_cast<const FFTTableImpl<double> *>(table);
    COMMON_ASSERTS();
    NEON::RunComplexFFT<Dir::Inverse>(t->Data(), input, output, fftLength);
  }

  void fft_real_f32_neon(const fft_table_f32 *table, const float *input, float *output, size_t fftLength)
  {
    auto t = reinterpret_cast<const FFTTableImpl<float> *>(table);
    COMMON_ASSERTS();
    NEON::RunRealFwdFFT(t->Data(), input, output, fftLength);
  }

  void fft_real_f64_neon(const fft_table_f64 *table, const double *input, double *output, size_t fftLength)
  {
    auto t = reinterpret_cast<const FFTTableImpl<double> *>(table);
    COMMON_ASSERTS();
    NEON::RunRealFwdFFT(t->Data(), input, output, fftLength);
  }

  void ifft_real_f32_neon(const fft_table_f32 *table, const float *input, float *output, size_t fftLength)
  {
    auto t = reinterpret_cast<const FFTTableImpl<float> *>(table);
    COMMON_ASSERTS();
    NEON::RunRealInvFFT(t->Data(), input, output, fftLength);
  }

  void ifft_real_f64_neon(const fft_table_f64 *table, const double *input, double *output, size_t fftLength)
  {
    auto t = reinterpret_cast<const FFTTableImpl<double> *>(table);
    COMMON_ASSERTS();
    NEON::RunRealInvFFT(t->Data(), input, output, fftLength);
  }
}