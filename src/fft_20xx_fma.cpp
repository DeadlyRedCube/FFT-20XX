#include "fft_20xx_common.h"
#include "fft_20xx_table.h"

static_assert(CPU_X86 || CPU_X64);

static_assert(SIMD_AVX);
static_assert(SIMD_FMA);

#if COMPILER_GCC
  #pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

#if COMPILER_CLANG
  #pragma clang diagnostic ignored "-Wunused-function"
  #pragma clang diagnostic ignored "-Wunneeded-internal-declaration"
#endif

namespace FFT20XX::FMA
{
  #include "fft_20xx_avx_common.h"
}

using namespace FFT20XX;
using namespace fft_20xx::c;

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

FORCE_INLINE void DoRadix4NoRotation(
  __m256 &aR,
  __m256 &aI,
  __m256 &bR,
  __m256 &bI,
  __m256 &cR,
  __m256 &cI,
  __m256 &dR,
  __m256 &dI)
{
  // a' = a + c, c' = a - c
  auto t = _mm256_add_ps(aR, cR);
  cR = _mm256_sub_ps(aR, cR);
  aR = t;
  t = _mm256_add_ps(aI, cI);
  cI = _mm256_sub_ps(aI, cI);
  aI = t;

  // b' = b + d, d' = b - d
  t = _mm256_add_ps(bR, dR);
  dR = _mm256_sub_ps(bR, dR);
  bR = t;
  t = _mm256_add_ps(bI, dI);
  dI = _mm256_sub_ps(bI, dI);
  bI = t;

  // a' = a + b, b' = a - b
  t = _mm256_add_ps(aR, bR);
  bR = _mm256_sub_ps(aR, bR);
  aR = t;
  t = _mm256_add_ps(aI, bI);
  bI = _mm256_sub_ps(aI, bI);
  aI = t;

  // c' = c + {dI, -dR}, d' = c - {dI, -dR}
  t = _mm256_add_ps(cR, dI);
  dI = _mm256_sub_ps(cR, dI); // actually dR
  cR = t;
  t = _mm256_sub_ps(cI, dR);
  dR = _mm256_add_ps(cI, dR); // actually dI
  cI = t;
  std::swap(dR, dI);

}

FORCE_INLINE void DoRadix4FirstRotation(
  __m256 &aR,
  __m256 &aI,
  __m256 &bR,
  __m256 &bI,
  __m256 &cR,
  __m256 &cI,
  __m256 &dR,
  __m256 &dI,
  const __m256 s)
{
  // a' = a + {cI, -cR}, c' = a - {cI, -cR}
  auto t = _mm256_add_ps(aR, cI);
  cI = _mm256_sub_ps(aR, cI); // actually cR
  aR = t;
  t = _mm256_sub_ps(aI, cR);
  cR = _mm256_add_ps(aI, cR);
  aI = t;
  std::swap(cR, cI);

  // b' = b + {dI, -dR}, d' = b - {dI, -dR}
  t = _mm256_add_ps(bR, dI);
  dI = _mm256_sub_ps(bR, dI); // actually cR
  bR = t;
  t = _mm256_sub_ps(bI, dR);
  dR = _mm256_add_ps(bI, dR);
  bI = t;
  std::swap(dR, dI);


  // a' = a + b*{s, -s}, b' = a - b*{s, -s}
  t = _mm256_add_ps(bR, bI);
  bI = _mm256_sub_ps(bI, bR);
  bR = t;
  t = _mm256_fmadd_ps(bR, s, aR);
  bR = _mm256_fnmadd_ps(bR, s, aR);
  aR = t;
  t = _mm256_fmadd_ps(bI, s, aI);
  bI = _mm256_fnmadd_ps(bI, s, aI);
  aI = t;

  // c' = c + d*{-s, -s}, d'= c - d*{-s, -s}
  t = _mm256_sub_ps(dI, dR);
  dI = _mm256_add_ps(dR, dI);
  dR = t;
  t = _mm256_fmadd_ps(s, dR, cR);
  dR = _mm256_fnmadd_ps(s, dR, cR);
  cR = t;
  t = _mm256_fnmadd_ps(s, dI, cI);
  dI = _mm256_fmadd_ps(s, dI, cI);
  cI = t;
}

FORCE_INLINE void DoRadix4(
  __m256 &aR,
  __m256 &aI,
  __m256 &bR,
  __m256 &bI,
  __m256 &cR,
  __m256 &cI,
  __m256 &dR,
  __m256 &dI,
  const __m256 w0R,
  const __m256 w0I,
  const __m256 w1R,
  const __m256 w1I,
  const __m256 two)
{
  // c' = a - c*w0
  auto t = _mm256_fnmadd_ps(cR, w0R, aR);
  t = _mm256_fmadd_ps(cI, w0I, t);
  cI = _mm256_fnmadd_ps(cI, w0R, aI);
  cI = _mm256_fnmadd_ps(cR, w0I, cI);
  cR = t;

  // a' = 2*a - c'
  aR = _mm256_fmsub_ps(two, aR, cR);
  aI = _mm256_fmsub_ps(two, aI, cI);

  // d' = b - d*w0
  t = _mm256_fnmadd_ps(dR, w0R, bR);
  t = _mm256_fmadd_ps(dI, w0I, t);
  dI = _mm256_fnmadd_ps(dI, w0R, bI);
  dI = _mm256_fnmadd_ps(dR, w0I, dI);
  dR = t;

  // b' = 2*b - d'
  bR = _mm256_fmsub_ps(two, bR, dR);
  bI = _mm256_fmsub_ps(two, bI, dI);

  // b' = a - b*w1
  t = _mm256_fnmadd_ps(bR, w1R, aR);
  t = _mm256_fmadd_ps(bI, w1I, t);
  bI = _mm256_fnmadd_ps(bI, w1R, aI);
  bI = _mm256_fnmadd_ps(bR, w1I, bI);
  bR = t;

  // a' = 2*a - b'
  aR = _mm256_fmsub_ps(two, aR, bR);
  aI = _mm256_fmsub_ps(two, aI, bI);

  // d' = c - d*{w1I, -w1R}
  t = _mm256_fnmadd_ps(dI, w1R, cR);
  t = _mm256_fnmadd_ps(dR, w1I, t);
  dI = _mm256_fnmadd_ps(dI, w1I, cI);
  dI = _mm256_fmadd_ps(dR, w1R, dI);
  dR = t;

  // c' = 2*c - d'
  cR = _mm256_fmsub_ps(two, cR, dR);
  cI = _mm256_fmsub_ps(two, cI, dI);
}


template <std::floating_point F>
void Radix4Or8InitialStepAVX(const std::complex<F> *input, std::complex<F> *output, const size_t length)
{
  constexpr auto LoadDeinterleaved =
    [](const auto &src)
    {
      auto r = _mm256_loadu_ps(reinterpret_cast<const float *>(&src));
      auto i = _mm256_loadu_ps(reinterpret_cast<const float *>(&src) + 8);
      auto t = _mm256_permute2f128_ps(r, i, 0x20);
      i = _mm256_permute2f128_ps(r, i, 0x31);
      r = _mm256_shuffle_ps(t, i, _MM_SHUFFLE(2, 0, 2, 0));
      i = _mm256_shuffle_ps(t, i, _MM_SHUFFLE(3, 1, 3, 1));
      return std::tuple{r, i};
    };


  if (!IsPowerOfFourAssumingIsPowerOfTwo(length))
  {
    const auto stride = length / 8;
    // RADIX 8, YO
    for (auto i = 0_z; i < stride; i += 8)
    {
      // !!! in the world where we don't need to deinterleave, we can skip loading aI initially and get aR fully
      //  written out then load aI and push it all the way through
      #if 0
        auto aR = input[stride*0 + i].real();
        auto bR = input[stride*1 + i].real();
        auto cR = input[stride*2 + i].real();
        auto dR = input[stride*3 + i].real();
        auto eR = input[stride*4 + i].real();
        auto fR = input[stride*5 + i].real();
        auto gR = input[stride*6 + i].real();
        auto hR = input[stride*7 + i].real();

        // A', E' = A +- E
        auto t = aR + eR;
        eR = aR - eR;
        aR = t;

        // B', F' = B =- F
        t = bR + fR;
        fR = bR - fR;
        bR = t;

        // C', G' = C +- G
        t = cR + gR;
        gR = cR - gR;
        cR = t;

        // D', H' = D +- H
        t = dR + hR;
        hR = dR - hR;
        dR = t;

        // A'', C'' = A' +- C'
        t = aR + cR;
        cR = aR - cR;
        aR = t;

        // B'', D'' = B' +- D'
        t = bR + dR;
        dR = bR - dR;
        bR = t;

        // Finally, A''', B''' = A'' +- B''
        t = aR + bR;
        bR = aR - bR;
        output[stride*0 + i].real(t); // Store A'''R
        output[stride*1 + i].real(bR); // Store B'''R

        // We've freed up 2 registers out of our initial 8 so now we have space to load the imaginary components and
        //  finish out the process
        auto aI = input[stride*0 + i].imag();
        auto bI = input[stride*1 + i].imag();
        auto cI = input[stride*2 + i].imag();
        auto dI = input[stride*3 + i].imag();
        auto eI = input[stride*4 + i].imag();
        auto fI = input[stride*5 + i].imag();
        auto gI = input[stride*6 + i].imag();
        auto hI = input[stride*7 + i].imag();

        // A', E' = A +- E
        t = aI + eI;
        eI = aI - eI;
        aI = t;

        // B', F' = B =- F
        t = bI + fI;
        fI = bI - fI;
        bI = t;

        // C', G' = C +- G
        t = cI + gI;
        gI = cI - gI;
        cI = t;

        // D', H' = D +- H
        t = dI + hI;
        hI = dI - hI;
        dI = t;

        // A'', C'' = A' +- C'
        t = aI + cI;
        cI = aI - cI;
        aI = t;

        // B'', D'' = B' +- D'
        t = bI + dI;
        dI = bI - dI;
        bI = t;

        // Finally, A''', B''' = A'' +- B''
        t = aI + bI;
        bI = aI - bI;
        output[stride*0 + i].imag(t); // Store A'''I
        output[stride*1 + i].imag(bI); // Store B'''I

        // At this point we have 12 registers loaded (A & B are done) as 6 complex pairs:
        //  C'', D'', E', F', G', H'

        // C''', D''' = C'' +- r90(D'')
        t = cR + dI;
        dI = cR - dI; // This is actually D'''R
        output[stride*2 + i].real(t);
        output[stride*3 + i].real(dI);

        t = cI - dR;
        dI = cI + dR;
        output[stride*2 + i].imag(t);
        output[stride*3 + i].imag(dI);

        // E'', G'' = E' +- r90(G'')
        t = eR + gI;
        gI = eR - gI; // This is actually G''R but stored here temporarily
        eR = t;

        t = eI - gR;
        gR = eI + gR;
        std::swap(gR, gI); // "reassign" registers
        eI = t;

        // F'', H'' = F' +- r90(H')
        t = fR + hI;
        hI = fR - hI; // Like with E/G this is H''R
        fR = t;

        t = fI - hR;
        hR = fI + hR;
        std::swap(hR, hI);
        fI = t;

        // Now we have 12 registers loaded so we can finally load this one angle value (this is 'S' in the comments), to
        //  have 13 total (which leaves 3 temp registers of which we only need 2)
        constexpr auto InvSqrt2 = F(0.7071067811865475244);

        // E''', F''' = E'' +- r45(F'')
        //   r45(x) = { S(xR + xI), S(xI - xR) }
        t = InvSqrt2*(fR + fI);
        fI = InvSqrt2*(fI - fR);
        fR = t;

        t = eR + fR; // with FMA this is fma(InvSqrt2, fR, fma(InvSqrt2, fI, eR))
        fR = eR - fR; // with FMA this is fms(Two, eR, t)
        eR = t;

        t = eI + fI;
        fI = eI - fI;

        output[stride*4 + i] = { eR, t };
        output[stride*5 + i] = { fR, fI };

        // G''', H''' = G'' +- r135(H''')
        //   r135(x) = { S(xI - xR), -S(xR + xI) }
        t = InvSqrt2*(hI - hR);
        hI = -InvSqrt2*(hR + hI);
        hR = t;

        t = gR + hR;
        hR = gR - hR;
        gR = t;

        t = gI + hI;
        hI = gI - hI;

        output[stride*6 + i] = { gR, t };
        output[stride*7 + i] = { hR, hI };
      #else
        // load with deinterleave
        auto [aR, aI] = LoadDeinterleaved(input[stride * 0 + i]);
        auto [cR, cI] = LoadDeinterleaved(input[stride * 2 + i]);
        auto [eR, eI] = LoadDeinterleaved(input[stride * 4 + i]);
        auto [gR, gI] = LoadDeinterleaved(input[stride * 6 + i]);

        // First Radix2: A+-E, B+-F, C+-G, D+-H
        // Second Radix2: A' +- C', B' +- D', E' +- r90(G'), F' +- r90(H')
        // Third Radix 2: A'' +- B'', C'' +- r90(D''), E'' +- r45(F''), G'' +- r135(H'')

        // A', E' = A +- E
        auto t = _mm256_add_ps(aR, eR);
        eR = _mm256_sub_ps(aR, eR);
        aR = t;
        t = _mm256_add_ps(aI, eI);
        eI = _mm256_sub_ps(aI, eI);
        aI = t;

        // C', G' = C +- G
        t = _mm256_add_ps(cR, gR);
        gR = _mm256_sub_ps(cR, gR);
        cR = t;
        t = _mm256_add_ps(cI, gI);
        gI = _mm256_sub_ps(cI, gI);
        cI = t;

        // A'', C'' = A' +- C'
        t = _mm256_add_ps(aR, cR);
        cR = _mm256_sub_ps(aR, cR);
        aR = t;
        t = _mm256_add_ps(aI, cI);
        cI = _mm256_sub_ps(aI, cI);
        aI = t;

        // E'', G'' = E' + r90(G')
        t = _mm256_add_ps(eR, gI);
        gI = _mm256_sub_ps(eR, gI);
        eR = t;
        t = _mm256_sub_ps(eI, gR);
        gR = _mm256_add_ps(eI, gR);
        eI = t;
        std::swap(gR, gI);

        // Currently loaded: A'', C'', E'', G'' (8 vectors). Load b and f (12 vectors total)
        auto [bR, bI] = LoadDeinterleaved(input[stride * 1 + i]);
        auto [fR, fI] = LoadDeinterleaved(input[stride * 5 + i]);

        // B', F' = B +- F
        t = _mm256_add_ps(bR, fR);
        fR = _mm256_sub_ps(bR, fR);
        bR = t;
        t = _mm256_add_ps(bI, fI);
        fI = _mm256_sub_ps(bI, fI);
        bI = t;

        // Okay now we finally need all 16 loaded so now we get tricky
        // !!! This actually sucks because we need to deinterleave the vectors to load them which requires a single
        //  temporary register, which pushes us over into 17 :|
        auto [dR, dI] = LoadDeinterleaved(input[stride * 3 + i]);
        auto [hR, hI] = LoadDeinterleaved(input[stride * 7 + i]);

        // D', H' = D += H. If I didn't have to do the deinterleave there's a trick (from a nudge by Ben Self) here
        //  that looks like:
        //    h = h - d;
        //    d = d + d;
        //    d = d - h;
        // ...that would do the same math without adding temps, but I've already hit 17 SIMD values so whatever. Keep
        //  the temp.
        t = _mm256_add_ps(dR, hR);
        hR = _mm256_sub_ps(dR, hR);
        dR = t;
        t = _mm256_add_ps(dI, hI);
        hI = _mm256_sub_ps(dI, hI);
        dI = t;

        // B'', D'' = B' += D'
        t = _mm256_add_ps(bR, dR);
        dR = _mm256_sub_ps(bR, dR);
        bR = t;
        t = _mm256_add_ps(bI, dI);
        dI = _mm256_sub_ps(bI, dI);
        bI = t;

        // A''', B''' = A'' += B''
        t = _mm256_add_ps(aR, bR);
        bR = _mm256_sub_ps(aR, bR);
        aR = t;
        t = _mm256_add_ps(aI, bI);
        bI = _mm256_sub_ps(aI, bI);
        aI = t;

        // Store A''', B''' - now we have 14 registers again:
        //  C'', D'', E'', F', G'', H'

        // Note that as we store things, we're not re-interleaving the real/imaginary components as the remaining
        //  passes can rely on them being separate.
        _mm256_storeu_ps(reinterpret_cast<float *>(&output[stride*0 + i]), aR);
        _mm256_storeu_ps(reinterpret_cast<float *>(&output[stride*0 + i]) + 8, aI);
        _mm256_storeu_ps(reinterpret_cast<float *>(&output[stride*1 + i]), bR);
        _mm256_storeu_ps(reinterpret_cast<float *>(&output[stride*1 + i]) + 8, bI);

        // C''', D''' = C'' +- r90(D'')
        t = _mm256_add_ps(cR, dI);
        dI = _mm256_sub_ps(cR, dI);
        cR = t;
        t = _mm256_sub_ps(cI, dR);
        dR = _mm256_add_ps(cI, dR);
        cI = t;
        std::swap(dR, dI);

        _mm256_storeu_ps(reinterpret_cast<float *>(&output[stride*2 + i]), cR);
        _mm256_storeu_ps(reinterpret_cast<float *>(&output[stride*2 + i]) + 8, cI);
        _mm256_storeu_ps(reinterpret_cast<float *>(&output[stride*3 + i]), dR);
        _mm256_storeu_ps(reinterpret_cast<float *>(&output[stride*3 + i]) + 8, dI);

        // Now we have 8 registers loaded so we can finally load this one angle value (this is 'S' in the comments),
        //  (plus, in the FMA case, a "2.0") to have 14 total (which leaves 2 temp registers)
        const auto InvSqrt2 = _mm256_set1_ps(F(0.7071067811865475244));

        #define USE_FMA 1

        #if USE_FMA
          const auto Two = _mm256_set1_ps(F(2.0));
        #endif

        // Now we have E'', F', G'', H'

        // F'', H'' = F' +- r90(H')
        t = _mm256_add_ps(fR, hI);
        hI = _mm256_sub_ps(fR, hI);
        fR = t;
        t = _mm256_sub_ps(fI, hR);
        hR = _mm256_add_ps(fI, hR);
        fI = t;
        std::swap(hR, hI);

        // E''', F''' = E'' +- r45(F'') ... r45(x) = {xR * S - xI * -S, xR * -s + xI * S}
        // Note: with FMA instructions there's more efficient way to do this but this is good enough for testing
        #if USE_FMA // FMA path
          t = _mm256_add_ps(fR, fI);
          fI = _mm256_sub_ps(fI, fR);
          fR = _mm256_fnmadd_ps(InvSqrt2, t, eR);
          fI = _mm256_fnmadd_ps(InvSqrt2, fI, eI);
          eR = _mm256_fmsub_ps(Two, eR, fR);
          eI = _mm256_fmsub_ps(Two, eI, fI);
        #else
          t = _mm256_add_ps(fR, fI);
          fI = _mm256_sub_ps(fI, fR);
          fR = _mm256_mul_ps(InvSqrt2, t);
          fI = _mm256_mul_ps(InvSqrt2, fI);
          t  = _mm256_add_ps(eR, fR);
          fR = _mm256_sub_ps(eR, fR);
          eR = t;
          t = _mm256_add_ps(eI, fI);
          fI = _mm256_sub_ps(eI, fI);
          eI = t;
        #endif

        _mm256_storeu_ps(reinterpret_cast<float *>(&output[stride*4 + i]), eR);
        _mm256_storeu_ps(reinterpret_cast<float *>(&output[stride*4 + i]) + 8, eI);
        _mm256_storeu_ps(reinterpret_cast<float *>(&output[stride*5 + i]), fR);
        _mm256_storeu_ps(reinterpret_cast<float *>(&output[stride*5 + i]) + 8, fI);

        // Finally, G''' and H''' = G'' +- r135(H'')
        //   r135(x) = { S(xI - xR), -S(xR + xI) }
        #if USE_FMA
          t = _mm256_sub_ps(hI, hR);
          hI = _mm256_add_ps(hR, hI);
          hR = _mm256_fnmadd_ps(InvSqrt2, t, gR);
          hI = _mm256_fmadd_ps(InvSqrt2, hI, gI);
          gR = _mm256_fmsub_ps(Two, gR, hR);
          gI = _mm256_fmsub_ps(Two, gI, hI);
        #else
          t = _mm256_sub_ps(hI, hR);
          hI = _mm256_add_ps(hR, hI);
          hR = _mm256_mul_ps(InvSqrt2, t);
          hI = _mm256_mul_ps(InvSqrt2, hI);
          t = _mm256_add_ps(gR, hR);
          hR = _mm256_sub_ps(gR, hR);
          gR = t;
          t = _mm256_sub_ps(gI, hI);
          hI = _mm256_add_ps(gI, hI);
          gI = t;
        #endif

        _mm256_storeu_ps(reinterpret_cast<float *>(&output[stride*6 + i]), gR);
        _mm256_storeu_ps(reinterpret_cast<float *>(&output[stride*6 + i]) + 8, gI);
        _mm256_storeu_ps(reinterpret_cast<float *>(&output[stride*7 + i]), hR);
        _mm256_storeu_ps(reinterpret_cast<float *>(&output[stride*7 + i]) + 8, hI);
      #endif
    }
  }
  else
  {
    // radix 4, yo. Way easier than the radix 8 case above.
    const auto stride = length / 4;
    for (auto i = 0_z; i < stride; i += 8)
    {
      auto [aR, aI] = LoadDeinterleaved(input[stride * 0 + i]);
      auto [bR, bI] = LoadDeinterleaved(input[stride * 1 + i]);
      auto [cR, cI] = LoadDeinterleaved(input[stride * 2 + i]);
      auto [dR, dI] = LoadDeinterleaved(input[stride * 3 + i]);

      DoRadix4NoRotation(aR, aI, bR, bI, cR, cI, dR, dI);

      _mm256_storeu_ps(reinterpret_cast<float *>(&output[stride * 0 + i]), aR);
      _mm256_storeu_ps(reinterpret_cast<float *>(&output[stride * 0 + i]) + 8, aI);
      _mm256_storeu_ps(reinterpret_cast<float *>(&output[stride * 1 + i]), bR);
      _mm256_storeu_ps(reinterpret_cast<float *>(&output[stride * 1 + i]) + 8, bI);
      _mm256_storeu_ps(reinterpret_cast<float *>(&output[stride * 2 + i]), cR);
      _mm256_storeu_ps(reinterpret_cast<float *>(&output[stride * 2 + i]) + 8, cI);
      _mm256_storeu_ps(reinterpret_cast<float *>(&output[stride * 3 + i]), dR);
      _mm256_storeu_ps(reinterpret_cast<float *>(&output[stride * 3 + i]) + 8, dI);
    }
  }
}

template <std::floating_point F>
void DoRadix4(
  std::complex<F> &a,
  std::complex<F> &b,
  std::complex<F> &c,
  std::complex<F> &d,
  const std::complex<F> firstAngle,
  const std::complex<F> secondAngle,
  const std::complex<F> thirdAngle)
{
  b *= firstAngle;
  c *= secondAngle;
  d *= thirdAngle;

  std::tie(a, c) = std::tuple{ a + c, a - c };
  std::tie(b, d) = std::tuple{ b + d, b - d };

  std::tie(a, b) = std::tuple{ a + b, a - b };

  d = { d.imag(), -d.real() };
  std::tie(c, d) = std::tuple{ c + d, c - d };
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
    return std::tuple{wr1, wi1, wr2, wi2, wr1 * wr2-wi1 * wi2, wr1 * wi2 + wi1 * wr2};
  #endif
}

template <std::floating_point F>
auto LoadSingleAngleAVX(const F *tableData, size_t angleIndex, size_t angleCount)
{
  #if USE_FMA
    auto [w0R, w0I, w1R, w1I] = LoadSingleAngle(tableData, angleIndex, angleCount);
    return std::tuple{_mm256_set1_ps(w0R), _mm256_set1_ps(w0I), _mm256_set1_ps(w1R), _mm256_set1_ps(w1I) };
  #else
    // Third angle calculated based on the first two.
    return std::tuple{wr1, wi1, wr2, wi2, wr1 * wr2-wi1 * wi2, wr1 * wi2 + wi1 * wr2};
  #endif
}


template <std::floating_point F>
auto LoadAnglesAVX(const F *tableData, size_t angleIndex, size_t angleCount)
{
  // !!! do something different here?
  auto i1 = angleCount * 4;
  auto r1 = (angleCount == 4) ? 8 : (angleCount * 3);

  // Special handling for 8-angle indexing, as they're arranged differently for alignment reasons.
  //  The default math would index the "angleCount == 8" angles from offset 12, but they actually start at 8.
  //  This masks off the bottom bits of 12 to turn it into an 8.
  auto r0 = (angleCount == 4) ? 0 : ((r1 >> 1) & ~7_z);
  auto i0 = (angleCount == 4) ? 4 : (i1 >> 1);

  auto wr1 = _mm256_load_ps(&tableData[r0 + angleIndex]);
  auto wi1 = _mm256_load_ps(&tableData[i0 + angleIndex]);
  auto wr2 = _mm256_load_ps(&tableData[r1 + angleIndex]);
  auto wi2 = _mm256_load_ps(&tableData[i1 + angleIndex]);

  #if USE_FMA
    return std::tuple{wr1, wi1, wr2, wi2};
  #else
    // Third angle calculated based on the first two.
    return std::tuple{wr1, wi1, wr2, wi2, wr1 * wr2-wi1 * wi2, wr1 * wi2 + wi1 * wr2};
  #endif
}


template <std::floating_point F>
void NoFlipRadix4PassAVX(const F *tableData, std::complex<F> *data, const size_t length, const size_t radix4Stride)
{
  // The first set of stride works with no rotations (besides the 90)
  {
    for (auto i = 0_z; i < radix4Stride; i += 8)
    {
      auto aR = _mm256_loadu_ps(reinterpret_cast<float *>(&data[radix4Stride * 0 + i]));
      auto aI = _mm256_loadu_ps(reinterpret_cast<float *>(&data[radix4Stride * 0 + i]) + 8);
      auto bR = _mm256_loadu_ps(reinterpret_cast<float *>(&data[radix4Stride * 1 + i]));
      auto bI = _mm256_loadu_ps(reinterpret_cast<float *>(&data[radix4Stride * 1 + i]) + 8);
      auto cR = _mm256_loadu_ps(reinterpret_cast<float *>(&data[radix4Stride * 2 + i]));
      auto cI = _mm256_loadu_ps(reinterpret_cast<float *>(&data[radix4Stride * 2 + i]) + 8);
      auto dR = _mm256_loadu_ps(reinterpret_cast<float *>(&data[radix4Stride * 3 + i]));
      auto dI = _mm256_loadu_ps(reinterpret_cast<float *>(&data[radix4Stride * 3 + i]) + 8);

      DoRadix4NoRotation(aR, aI, bR, bI, cR, cI, dR, dI);

      _mm256_storeu_ps(reinterpret_cast<float *>(&data[radix4Stride * 0 + i]), aR);
      _mm256_storeu_ps(reinterpret_cast<float *>(&data[radix4Stride * 0 + i]) + 8, aI);
      _mm256_storeu_ps(reinterpret_cast<float *>(&data[radix4Stride * 1 + i]), bR);
      _mm256_storeu_ps(reinterpret_cast<float *>(&data[radix4Stride * 1 + i]) + 8, bI);
      _mm256_storeu_ps(reinterpret_cast<float *>(&data[radix4Stride * 2 + i]), cR);
      _mm256_storeu_ps(reinterpret_cast<float *>(&data[radix4Stride * 2 + i]) + 8, cI);
      _mm256_storeu_ps(reinterpret_cast<float *>(&data[radix4Stride * 3 + i]), dR);
      _mm256_storeu_ps(reinterpret_cast<float *>(&data[radix4Stride * 3 + i]) + 8, dI);
    }
  }
  // The second set has 90 then 45 (so 45, 90, 135 as R4)
  {
    const auto s = _mm256_set1_ps(F(0.7071067811865475244));
    const auto sectionStart = data + radix4Stride * 4;
    for (auto i = 0_z; i < radix4Stride; i += 8)
    {
      auto aR = _mm256_loadu_ps(reinterpret_cast<float *>(&sectionStart[radix4Stride * 0 + i]));
      auto aI = _mm256_loadu_ps(reinterpret_cast<float *>(&sectionStart[radix4Stride * 0 + i]) + 8);
      auto bR = _mm256_loadu_ps(reinterpret_cast<float *>(&sectionStart[radix4Stride * 1 + i]));
      auto bI = _mm256_loadu_ps(reinterpret_cast<float *>(&sectionStart[radix4Stride * 1 + i]) + 8);
      auto cR = _mm256_loadu_ps(reinterpret_cast<float *>(&sectionStart[radix4Stride * 2 + i]));
      auto cI = _mm256_loadu_ps(reinterpret_cast<float *>(&sectionStart[radix4Stride * 2 + i]) + 8);
      auto dR = _mm256_loadu_ps(reinterpret_cast<float *>(&sectionStart[radix4Stride * 3 + i]));
      auto dI = _mm256_loadu_ps(reinterpret_cast<float *>(&sectionStart[radix4Stride * 3 + i]) + 8);

      DoRadix4FirstRotation(aR, aI, bR, bI, cR, cI, dR, dI, s);

      _mm256_storeu_ps(reinterpret_cast<float *>(&sectionStart[radix4Stride * 0 + i]), aR);
      _mm256_storeu_ps(reinterpret_cast<float *>(&sectionStart[radix4Stride * 0 + i]) + 8, aI);
      _mm256_storeu_ps(reinterpret_cast<float *>(&sectionStart[radix4Stride * 1 + i]), bR);
      _mm256_storeu_ps(reinterpret_cast<float *>(&sectionStart[radix4Stride * 1 + i]) + 8, bI);
      _mm256_storeu_ps(reinterpret_cast<float *>(&sectionStart[radix4Stride * 2 + i]), cR);
      _mm256_storeu_ps(reinterpret_cast<float *>(&sectionStart[radix4Stride * 2 + i]) + 8, cI);
      _mm256_storeu_ps(reinterpret_cast<float *>(&sectionStart[radix4Stride * 3 + i]), dR);
      _mm256_storeu_ps(reinterpret_cast<float *>(&sectionStart[radix4Stride * 3 + i]) + 8, dI);
    }
  }

  // The rest need to load angles
  const auto sectionCount = length / (radix4Stride * 4);
  const auto two = _mm256_set1_ps(2.0f);
  for (auto sectionIndex = 2_z; sectionIndex < sectionCount; sectionIndex++)
  {
    // !!! do this as a bit-reversed increment instead of this
    const auto angleIndex = BitRev(sectionIndex) >> (1 + std::countl_zero(sectionCount));

    const auto [w0R, w0I, w1R, w1I] = LoadSingleAngleAVX(tableData, angleIndex, sectionCount);

    const auto sectionStart = data + sectionIndex * radix4Stride * 4;

    for (auto i = 0_z; i < radix4Stride; i += 8)
    {
      auto aR = _mm256_loadu_ps(reinterpret_cast<float *>(&sectionStart[radix4Stride * 0 + i]));
      auto aI = _mm256_loadu_ps(reinterpret_cast<float *>(&sectionStart[radix4Stride * 0 + i]) + 8);
      auto bR = _mm256_loadu_ps(reinterpret_cast<float *>(&sectionStart[radix4Stride * 1 + i]));
      auto bI = _mm256_loadu_ps(reinterpret_cast<float *>(&sectionStart[radix4Stride * 1 + i]) + 8);
      auto cR = _mm256_loadu_ps(reinterpret_cast<float *>(&sectionStart[radix4Stride * 2 + i]));
      auto cI = _mm256_loadu_ps(reinterpret_cast<float *>(&sectionStart[radix4Stride * 2 + i]) + 8);
      auto dR = _mm256_loadu_ps(reinterpret_cast<float *>(&sectionStart[radix4Stride * 3 + i]));
      auto dI = _mm256_loadu_ps(reinterpret_cast<float *>(&sectionStart[radix4Stride * 3 + i]) + 8);

      DoRadix4(aR, aI, bR, bI, cR, cI, dR, dI, w0R, w0I, w1R, w1I, two);

      _mm256_storeu_ps(reinterpret_cast<float *>(&sectionStart[radix4Stride * 0 + i]), aR);
      _mm256_storeu_ps(reinterpret_cast<float *>(&sectionStart[radix4Stride * 0 + i]) + 8, aI);
      _mm256_storeu_ps(reinterpret_cast<float *>(&sectionStart[radix4Stride * 1 + i]), bR);
      _mm256_storeu_ps(reinterpret_cast<float *>(&sectionStart[radix4Stride * 1 + i]) + 8, bI);
      _mm256_storeu_ps(reinterpret_cast<float *>(&sectionStart[radix4Stride * 2 + i]), cR);
      _mm256_storeu_ps(reinterpret_cast<float *>(&sectionStart[radix4Stride * 2 + i]) + 8, cI);
      _mm256_storeu_ps(reinterpret_cast<float *>(&sectionStart[radix4Stride * 3 + i]), dR);
      _mm256_storeu_ps(reinterpret_cast<float *>(&sectionStart[radix4Stride * 3 + i]) + 8, dI);
    }
  }
}

template <std::floating_point F>
void SingleFlipRadix4PassAVX(const F *tableData, std::complex<F> *data, const size_t length, const size_t radix4Stride)
{
  // !!! AVX will need a special case for length == 128 (where radix4Stride == 4)

  const auto sectionCount = length / (radix4Stride * 4);
  const auto two = _mm256_set1_ps(2.0f);

   // There's a different flip pattern for power of 2 and power of 4 single flips
  if (IsPowerOfFourAssumingIsPowerOfTwo(length))
  {
    // For power of four lengths, the flip happens completely within the butterfly (b and c flip). 1024 is a good
    //  test size for this case
    for (auto sectionIndex = 0_z; sectionIndex < sectionCount; sectionIndex++)
    {
      const auto angleIndex = BitRev(sectionIndex) >> (1 + std::countl_zero(sectionCount));
      auto [w0R, w0I, w1R, w1I] = LoadSingleAngleAVX(tableData, angleIndex, sectionCount);
      const auto sectionStart = data + sectionIndex * radix4Stride * 4;

      for (auto i = 0_z; i < radix4Stride; i += 8)
      {
        auto aR = _mm256_loadu_ps(reinterpret_cast<float *>(&sectionStart[radix4Stride * 0 + i]));
        auto aI = _mm256_loadu_ps(reinterpret_cast<float *>(&sectionStart[radix4Stride * 0 + i]) + 8);
        auto bR = _mm256_loadu_ps(reinterpret_cast<float *>(&sectionStart[radix4Stride * 1 + i]));
        auto bI = _mm256_loadu_ps(reinterpret_cast<float *>(&sectionStart[radix4Stride * 1 + i]) + 8);
        auto cR = _mm256_loadu_ps(reinterpret_cast<float *>(&sectionStart[radix4Stride * 2 + i]));
        auto cI = _mm256_loadu_ps(reinterpret_cast<float *>(&sectionStart[radix4Stride * 2 + i]) + 8);
        auto dR = _mm256_loadu_ps(reinterpret_cast<float *>(&sectionStart[radix4Stride * 3 + i]));
        auto dI = _mm256_loadu_ps(reinterpret_cast<float *>(&sectionStart[radix4Stride * 3 + i]) + 8);

        DoRadix4(aR, aI, bR, bI, cR, cI, dR, dI, w0R, w0I, w1R, w1I, two);

        // Write the middle two out of order (that's the bit flip)
        _mm256_storeu_ps(reinterpret_cast<float *>(&sectionStart[radix4Stride * 0 + i]), aR);
        _mm256_storeu_ps(reinterpret_cast<float *>(&sectionStart[radix4Stride * 0 + i]) + 8, aI);
        _mm256_storeu_ps(reinterpret_cast<float *>(&sectionStart[radix4Stride * 1 + i]), cR);
        _mm256_storeu_ps(reinterpret_cast<float *>(&sectionStart[radix4Stride * 1 + i]) + 8, cI);
        _mm256_storeu_ps(reinterpret_cast<float *>(&sectionStart[radix4Stride * 2 + i]), bR);
        _mm256_storeu_ps(reinterpret_cast<float *>(&sectionStart[radix4Stride * 2 + i]) + 8, bI);
        _mm256_storeu_ps(reinterpret_cast<float *>(&sectionStart[radix4Stride * 3 + i]), dR);
        _mm256_storeu_ps(reinterpret_cast<float *>(&sectionStart[radix4Stride * 3 + i]) + 8, dI);
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
    for (auto sectionIndex = 0_z; sectionIndex < sectionCount; sectionIndex += 2)
    {
      const auto angleIndex0 = BitRev(sectionIndex) >> (1 + std::countl_zero(sectionCount));
      auto [w0R0, w0I0, w1R0, w1I0] = LoadSingleAngleAVX(tableData, angleIndex0, sectionCount);

      // on non-ARM can do this as a bit-reversed increment
      const auto angleIndex1 = BitRev(sectionIndex + 1) >> (1 + std::countl_zero(sectionCount));
      auto [w0R1, w0I1, w1R1, w1I1] = LoadSingleAngleAVX(tableData, angleIndex1, sectionCount);

      const auto sectionStart0 = data + sectionIndex * radix4Stride * 4;
      const auto sectionStart1 = sectionStart0 + radix4Stride * 4;

      for (auto i = 0_z; i < radix4Stride; i += 8)
      {
        auto a0R = _mm256_loadu_ps(reinterpret_cast<float *>(&sectionStart0[radix4Stride * 0 + i]));
        auto a0I = _mm256_loadu_ps(reinterpret_cast<float *>(&sectionStart0[radix4Stride * 0 + i]) + 8);
        auto b0R = _mm256_loadu_ps(reinterpret_cast<float *>(&sectionStart0[radix4Stride * 1 + i]));
        auto b0I = _mm256_loadu_ps(reinterpret_cast<float *>(&sectionStart0[radix4Stride * 1 + i]) + 8);
        auto c0R = _mm256_loadu_ps(reinterpret_cast<float *>(&sectionStart0[radix4Stride * 2 + i]));
        auto c0I = _mm256_loadu_ps(reinterpret_cast<float *>(&sectionStart0[radix4Stride * 2 + i]) + 8);
        auto d0R = _mm256_loadu_ps(reinterpret_cast<float *>(&sectionStart0[radix4Stride * 3 + i]));
        auto d0I = _mm256_loadu_ps(reinterpret_cast<float *>(&sectionStart0[radix4Stride * 3 + i]) + 8);

        DoRadix4(a0R, a0I, b0R, b0I, c0R, c0I, d0R, d0I, w0R0, w0I0, w1R0, w1I0, two);

        // These can be stored back in-place
        _mm256_storeu_ps(reinterpret_cast<float *>(&sectionStart0[radix4Stride * 0 + i]), a0R);
        _mm256_storeu_ps(reinterpret_cast<float *>(&sectionStart0[radix4Stride * 0 + i]) + 8, a0I);
        _mm256_storeu_ps(reinterpret_cast<float *>(&sectionStart0[radix4Stride * 2 + i]), c0R);
        _mm256_storeu_ps(reinterpret_cast<float *>(&sectionStart0[radix4Stride * 2 + i]) + 8, c0I);

        // Now load the two values that get replaced with section 0 values and store over them
        auto a1R = _mm256_loadu_ps(reinterpret_cast<float *>(&sectionStart1[radix4Stride * 0 + i]));
        auto a1I = _mm256_loadu_ps(reinterpret_cast<float *>(&sectionStart1[radix4Stride * 0 + i]) + 8);
        _mm256_storeu_ps(reinterpret_cast<float *>(&sectionStart1[radix4Stride * 0 + i]), b0R);
        _mm256_storeu_ps(reinterpret_cast<float *>(&sectionStart1[radix4Stride * 0 + i]) + 8, b0I);

        auto c1R = _mm256_loadu_ps(reinterpret_cast<float *>(&sectionStart1[radix4Stride * 2 + i]));
        auto c1I = _mm256_loadu_ps(reinterpret_cast<float *>(&sectionStart1[radix4Stride * 2 + i]) + 8);
        _mm256_storeu_ps(reinterpret_cast<float *>(&sectionStart1[radix4Stride * 2 + i]), d0R);
        _mm256_storeu_ps(reinterpret_cast<float *>(&sectionStart1[radix4Stride * 2 + i]) + 8, d0I);

        // Load the remaining 2
        auto b1R = _mm256_loadu_ps(reinterpret_cast<float *>(&sectionStart1[radix4Stride * 1 + i]));
        auto b1I = _mm256_loadu_ps(reinterpret_cast<float *>(&sectionStart1[radix4Stride * 1 + i]) + 8);
        auto d1R = _mm256_loadu_ps(reinterpret_cast<float *>(&sectionStart1[radix4Stride * 3 + i]));
        auto d1I = _mm256_loadu_ps(reinterpret_cast<float *>(&sectionStart1[radix4Stride * 3 + i]) + 8);

        DoRadix4(a1R, a1I, b1R, b1I, c1R, c1I, d1R, d1I, w0R1, w0I1, w1R1, w1I1, two);

        _mm256_storeu_ps(reinterpret_cast<float *>(&sectionStart0[radix4Stride * 1 + i]), a1R);
        _mm256_storeu_ps(reinterpret_cast<float *>(&sectionStart0[radix4Stride * 1 + i]) + 8, a1I);
        _mm256_storeu_ps(reinterpret_cast<float *>(&sectionStart0[radix4Stride * 3 + i]), c1R);
        _mm256_storeu_ps(reinterpret_cast<float *>(&sectionStart0[radix4Stride * 3 + i]) + 8, c1I);
        _mm256_storeu_ps(reinterpret_cast<float *>(&sectionStart1[radix4Stride * 1 + i]), b1R);
        _mm256_storeu_ps(reinterpret_cast<float *>(&sectionStart1[radix4Stride * 1 + i]) + 8, b1I);
        _mm256_storeu_ps(reinterpret_cast<float *>(&sectionStart1[radix4Stride * 3 + i]), d1R);
        _mm256_storeu_ps(reinterpret_cast<float *>(&sectionStart1[radix4Stride * 3 + i]) + 8, d1I);
      }
    }
  }
}


template <std::floating_point F>
void DoubleFlipRadix4PassAVX(
  const F *tableData,
  std::complex<F> *data,
  const size_t length,
  const size_t radix4Stride)
{
  // This one requires loading four butterflies and transposing them

  // There are length / groupStep groups
  // each butterfly set is radix4Stride*4 long
  // Each group contains flipStride*4 elements
  const auto sectionCount = length / (radix4Stride * 4);
  const auto flipStride = sectionCount;

  const auto shift = std::countl_zero(length - 1);
  const auto angleIndexFlipInsert = BitRev(flipStride) >> shift;

  const auto two = _mm256_set1_ps(2.0f);
  for (auto groupStartIndex = 0_z; groupStartIndex < length; groupStartIndex += flipStride * 4)
  {
    for (auto butterflySetStartIndex = 0_z; butterflySetStartIndex < flipStride; butterflySetStartIndex += radix4Stride * 4)
    {
      // we need 4 sets of 3 angles each
      // in SSE/AVX this could be 3 vectors of 4 angles (duplicated for AVX cross lane), then use temp space
      //  to shuffle the correct 3 elements in to all lanes, but that's 6 vectors, so we may instead need to just
      //  stash these on the stack and load them (theoretically from L1 directly). Hopefully that's good enough!
      //  (also it's possible the other implementation already has to do similar, I don't know for sure)

      const auto angleIndex0 = (BitRev(groupStartIndex) >> shift) + butterflySetStartIndex;
      const auto angleIndex2 = angleIndex0 | angleIndexFlipInsert;
      const auto angleIndex1 = angleIndex0 | (angleIndexFlipInsert >> 1);
      const auto angleIndex3 = angleIndex1 | angleIndex2;

      const auto [w0R0, w0I0, w1R0, w1I0] = LoadSingleAngle(tableData, angleIndex0, sectionCount);
      const auto [w0R2, w0I2, w1R2, w1I2] = LoadSingleAngle(tableData, angleIndex2, sectionCount);
      const auto [w0R1, w0I1, w1R1, w1I1] = LoadSingleAngle(tableData, angleIndex1, sectionCount);
      const auto [w0R3, w0I3, w1R3, w1I3] = LoadSingleAngle(tableData, angleIndex3, sectionCount);

      // Use the partially-bit-reversed "groupStartIndex + butterflySetStartIndex" to get the angle
      for (auto butterflyStartIndex = 0_z; butterflyStartIndex < radix4Stride; butterflyStartIndex += 8)
      {
        auto i = groupStartIndex + butterflySetStartIndex + butterflyStartIndex;

        // Load the first 8 sets (that's 9 total, counting "two")
        auto a0R = _mm256_loadu_ps(reinterpret_cast<float *>(&data[i + flipStride * 0 + radix4Stride * 0]));
        auto a0I = _mm256_loadu_ps(reinterpret_cast<float *>(&data[i + flipStride * 0 + radix4Stride * 0]) + 8);
        auto b0R = _mm256_loadu_ps(reinterpret_cast<float *>(&data[i + flipStride * 0 + radix4Stride * 1]));
        auto b0I = _mm256_loadu_ps(reinterpret_cast<float *>(&data[i + flipStride * 0 + radix4Stride * 1]) + 8);
        auto c0R = _mm256_loadu_ps(reinterpret_cast<float *>(&data[i + flipStride * 0 + radix4Stride * 2]));
        auto c0I = _mm256_loadu_ps(reinterpret_cast<float *>(&data[i + flipStride * 0 + radix4Stride * 2]) + 8);
        auto d0R = _mm256_loadu_ps(reinterpret_cast<float *>(&data[i + flipStride * 0 + radix4Stride * 3]));
        auto d0I = _mm256_loadu_ps(reinterpret_cast<float *>(&data[i + flipStride * 0 + radix4Stride * 3]) + 8);

        {
          auto w0R0V = _mm256_set1_ps(w0R0);
          auto w0I0V = _mm256_set1_ps(w0I0);
          auto w1R0V = _mm256_set1_ps(w1R0);
          auto w1I0V = _mm256_set1_ps(w1I0);
          DoRadix4(a0R, a0I, b0R, b0I, c0R, c0I, d0R, d0I, w0R0V, w0I0V, w1R0V, w1I0V, two);
        }

        // Now store all of the '0's, swapping in 'a's as we go
        _mm256_storeu_ps(reinterpret_cast<float *>(&data[i + flipStride * 0 + radix4Stride * 0]), a0R);
        _mm256_storeu_ps(reinterpret_cast<float *>(&data[i + flipStride * 0 + radix4Stride * 0]) + 8, a0I);

        auto a1R = _mm256_loadu_ps(reinterpret_cast<float *>(&data[i + flipStride * 2 + radix4Stride * 0]));
        auto a1I = _mm256_loadu_ps(reinterpret_cast<float *>(&data[i + flipStride * 2 + radix4Stride * 0]) + 8);
        _mm256_storeu_ps(reinterpret_cast<float *>(&data[i + flipStride * 2 + radix4Stride * 0]), b0R);
        _mm256_storeu_ps(reinterpret_cast<float *>(&data[i + flipStride * 2 + radix4Stride * 0]) + 8, b0I);

        auto a2R = _mm256_loadu_ps(reinterpret_cast<float *>(&data[i + flipStride * 1 + radix4Stride * 0]));
        auto a2I = _mm256_loadu_ps(reinterpret_cast<float *>(&data[i + flipStride * 1 + radix4Stride * 0]) + 8);
        _mm256_storeu_ps(reinterpret_cast<float *>(&data[i + flipStride * 1 + radix4Stride * 0]), c0R);
        _mm256_storeu_ps(reinterpret_cast<float *>(&data[i + flipStride * 1 + radix4Stride * 0]) + 8, c0I);

        auto a3R = _mm256_loadu_ps(reinterpret_cast<float *>(&data[i + flipStride * 3 + radix4Stride * 0]));
        auto a3I = _mm256_loadu_ps(reinterpret_cast<float *>(&data[i + flipStride * 3 + radix4Stride * 0]) + 8);
        _mm256_storeu_ps(reinterpret_cast<float *>(&data[i + flipStride * 3 + radix4Stride * 0]), d0R);
        _mm256_storeu_ps(reinterpret_cast<float *>(&data[i + flipStride * 3 + radix4Stride * 0]) + 8, d0I);

        // At this point we have 6 loaded sets (a1, a2, a3)

        auto b1R = _mm256_loadu_ps(reinterpret_cast<float *>(&data[i + flipStride * 2 + radix4Stride * 1]));
        auto b1I = _mm256_loadu_ps(reinterpret_cast<float *>(&data[i + flipStride * 2 + radix4Stride * 1]) + 8);
        auto c1R = _mm256_loadu_ps(reinterpret_cast<float *>(&data[i + flipStride * 2 + radix4Stride * 2]));
        auto c1I = _mm256_loadu_ps(reinterpret_cast<float *>(&data[i + flipStride * 2 + radix4Stride * 2]) + 8);
        auto d1R = _mm256_loadu_ps(reinterpret_cast<float *>(&data[i + flipStride * 2 + radix4Stride * 3]));
        auto d1I = _mm256_loadu_ps(reinterpret_cast<float *>(&data[i + flipStride * 2 + radix4Stride * 3]) + 8);

        // Now there are 12 loaded (a1, b1, c1, d1, a2, a3)
        {
          auto w0R1V = _mm256_set1_ps(w0R1);
          auto w0I1V = _mm256_set1_ps(w0I1);
          auto w1R1V = _mm256_set1_ps(w1R1);
          auto w1I1V = _mm256_set1_ps(w1I1);
          DoRadix4(a1R, a1I, b1R, b1I, c1R, c1I, d1R, d1I, w0R1V, w0I1V, w1R1V, w1I1V, two);
        }

        // Store everything (Swapping in bs as we go)
        _mm256_storeu_ps(reinterpret_cast<float *>(&data[i + flipStride * 0 + radix4Stride * 1]), a1R);
        _mm256_storeu_ps(reinterpret_cast<float *>(&data[i + flipStride * 0 + radix4Stride * 1]) + 8, a1I);
        _mm256_storeu_ps(reinterpret_cast<float *>(&data[i + flipStride * 2 + radix4Stride * 1]), b1R);
        _mm256_storeu_ps(reinterpret_cast<float *>(&data[i + flipStride * 2 + radix4Stride * 1]) + 8, b1I);

        auto b2R = _mm256_loadu_ps(reinterpret_cast<float *>(&data[i + flipStride * 1 + radix4Stride * 1]));
        auto b2I = _mm256_loadu_ps(reinterpret_cast<float *>(&data[i + flipStride * 1 + radix4Stride * 1]) + 8);
        _mm256_storeu_ps(reinterpret_cast<float *>(&data[i + flipStride * 1 + radix4Stride * 1]), c1R);
        _mm256_storeu_ps(reinterpret_cast<float *>(&data[i + flipStride * 1 + radix4Stride * 1]) + 8, c1I);

        auto b3R = _mm256_loadu_ps(reinterpret_cast<float *>(&data[i + flipStride * 3 + radix4Stride * 1]));
        auto b3I = _mm256_loadu_ps(reinterpret_cast<float *>(&data[i + flipStride * 3 + radix4Stride * 1]) + 8);
        _mm256_storeu_ps(reinterpret_cast<float *>(&data[i + flipStride * 3 + radix4Stride * 1]), d1R);
        _mm256_storeu_ps(reinterpret_cast<float *>(&data[i + flipStride * 3 + radix4Stride * 1]) + 8, d1I);

        // Now there are 8 loaded (a2, b2, a3, b3)

        auto c2R = _mm256_loadu_ps(reinterpret_cast<float *>(&data[i + flipStride * 1 + radix4Stride * 2]));
        auto c2I = _mm256_loadu_ps(reinterpret_cast<float *>(&data[i + flipStride * 1 + radix4Stride * 2]) + 8);
        auto d2R = _mm256_loadu_ps(reinterpret_cast<float *>(&data[i + flipStride * 1 + radix4Stride * 3]));
        auto d2I = _mm256_loadu_ps(reinterpret_cast<float *>(&data[i + flipStride * 1 + radix4Stride * 3]) + 8);

        // Back up to 12 (a2, b2, c2, d2, a3, b3)
        {
          auto w0R2V = _mm256_set1_ps(w0R2);
          auto w0I2V = _mm256_set1_ps(w0I2);
          auto w1R2V = _mm256_set1_ps(w1R2);
          auto w1I2V = _mm256_set1_ps(w1I2);
          DoRadix4(a2R, a2I, b2R, b2I, c2R, c2I, d2R, d2I, w0R2V, w0I2V, w1R2V, w1I2V, two);
        }

        _mm256_storeu_ps(reinterpret_cast<float *>(&data[i + flipStride * 0 + radix4Stride * 2]), a2R);
        _mm256_storeu_ps(reinterpret_cast<float *>(&data[i + flipStride * 0 + radix4Stride * 2]) + 8, a2I);

        _mm256_storeu_ps(reinterpret_cast<float *>(&data[i + flipStride * 2 + radix4Stride * 2]), b2R);
        _mm256_storeu_ps(reinterpret_cast<float *>(&data[i + flipStride * 2 + radix4Stride * 2]) + 8, b2I);

        _mm256_storeu_ps(reinterpret_cast<float *>(&data[i + flipStride * 1 + radix4Stride * 2]), c2R);
        _mm256_storeu_ps(reinterpret_cast<float *>(&data[i + flipStride * 1 + radix4Stride * 2]) + 8, c2I);

        auto c3R = _mm256_loadu_ps(reinterpret_cast<float *>(&data[i + flipStride * 3 + radix4Stride * 2]));
        auto c3I = _mm256_loadu_ps(reinterpret_cast<float *>(&data[i + flipStride * 3 + radix4Stride * 2]) + 8);
        _mm256_storeu_ps(reinterpret_cast<float *>(&data[i + flipStride * 3 + radix4Stride * 2]), d2R);
        _mm256_storeu_ps(reinterpret_cast<float *>(&data[i + flipStride * 3 + radix4Stride * 2]) + 8, d2I);

        auto d3R = _mm256_loadu_ps(reinterpret_cast<float *>(&data[i + flipStride * 3 + radix4Stride * 3]));
        auto d3I = _mm256_loadu_ps(reinterpret_cast<float *>(&data[i + flipStride * 3 + radix4Stride * 3]) + 8);

        // Now just the '3's are left (8 sets)

        {
          auto w0R3V = _mm256_set1_ps(w0R3);
          auto w0I3V = _mm256_set1_ps(w0I3);
          auto w1R3V = _mm256_set1_ps(w1R3);
          auto w1I3V = _mm256_set1_ps(w1I3);
          DoRadix4(a3R, a3I, b3R, b3I, c3R, c3I, d3R, d3I, w0R3V, w0I3V, w1R3V, w1I3V, two);
        }

        _mm256_storeu_ps(reinterpret_cast<float *>(&data[i + flipStride * 0 + radix4Stride * 3]), a3R);
        _mm256_storeu_ps(reinterpret_cast<float *>(&data[i + flipStride * 0 + radix4Stride * 3]) + 8, a3I);

        _mm256_storeu_ps(reinterpret_cast<float *>(&data[i + flipStride * 2 + radix4Stride * 3]), b3R);
        _mm256_storeu_ps(reinterpret_cast<float *>(&data[i + flipStride * 2 + radix4Stride * 3]) + 8, b3I);

        _mm256_storeu_ps(reinterpret_cast<float *>(&data[i + flipStride * 1 + radix4Stride * 3]), c3R);
        _mm256_storeu_ps(reinterpret_cast<float *>(&data[i + flipStride * 1 + radix4Stride * 3]) + 8, c3I);

        _mm256_storeu_ps(reinterpret_cast<float *>(&data[i + flipStride * 3 + radix4Stride * 3]), d3R);
        _mm256_storeu_ps(reinterpret_cast<float *>(&data[i + flipStride * 3 + radix4Stride * 3]) + 8, d3I);
      }
    }
  }
}

template <std::floating_point F>
void SecondToLastDoubleFlipRadix4PassAVX(const F *tableData, std::complex<F> *data, const size_t length)
{
  constexpr auto radix4Stride = 4_z;
  const auto flipStride = length / 16;

  const auto shift = std::countl_zero(length - 1);
  const auto angleIndexFlipInsert = BitRev(flipStride) >> shift;

  for (auto groupStartIndex = 0_z; groupStartIndex < length; groupStartIndex += flipStride * 4)
  {
    for (auto butterflySetStartIndex = 0_z; butterflySetStartIndex < flipStride; butterflySetStartIndex += radix4Stride * 4)
    {
      // we need 4 sets of 3 angles each
      // in SSE/AVX this could be 3 vectors of 4 angles (duplicated for AVX cross lane), then use temp space
      //  to shuffle the correct 3 elements in to all lanes, but that's 6 vectors, so we may instead need to just
      //  stash these on the stack and load them (theoretically from L1 directly). Hopefully that's good enough!
      //  (also it's possible the other implementation already has to do similar, I don't know for sure)

      const auto angleIndex0 = (BitRev(groupStartIndex) >> shift) + butterflySetStartIndex;

      const auto angleIndex2 = angleIndex0 | angleIndexFlipInsert;
      const auto angleIndex1 = angleIndex0 | (angleIndexFlipInsert >> 1);
      const auto angleIndex3 = angleIndex1 | angleIndex2;

      const auto two = _mm256_set1_ps(2.0f);

      // Use the partially-bit-reversed "groupStartIndex + butterflySetStartIndex" to get the angle
      //for (auto butterflyStartIndex = 0_z; butterflyStartIndex < radix4Stride; butterflyStartIndex++)
      {
        auto i = groupStartIndex + butterflySetStartIndex;

        // The pointer offsets between "columns" are 0, 4, 16, and 20 (in floats) because of how the AVX sets are
        //  strided: 8 reals then 8 imaginaries.
        auto aR = _mm256_insertf128_ps(
          _mm256_castps128_ps256(_mm_loadu_ps(reinterpret_cast<float *>(&data[i + flipStride * 0]))),
          _mm_loadu_ps(reinterpret_cast<float *>(&data[i + flipStride * 2])),
          1);

        auto aI = _mm256_insertf128_ps(
          _mm256_castps128_ps256(_mm_loadu_ps(reinterpret_cast<float *>(&data[i + flipStride * 0]) + 8)),
          _mm_loadu_ps(reinterpret_cast<float *>(&data[i + flipStride * 2]) + 8),
          1);

        auto bR = _mm256_insertf128_ps(
          _mm256_castps128_ps256(_mm_loadu_ps(reinterpret_cast<float *>(&data[i + flipStride * 0]) + 4)),
          _mm_loadu_ps(reinterpret_cast<float *>(&data[i + flipStride * 2]) + 4),
          1);

        auto bI = _mm256_insertf128_ps(
          _mm256_castps128_ps256(_mm_loadu_ps(reinterpret_cast<float *>(&data[i + flipStride * 0]) + 4 + 8)),
          _mm_loadu_ps(reinterpret_cast<float *>(&data[i + flipStride * 2]) + 4 + 8),
          1);

        auto cR = _mm256_insertf128_ps(
          _mm256_castps128_ps256(_mm_loadu_ps(reinterpret_cast<float *>(&data[i + flipStride * 0]) + 16)),
          _mm_loadu_ps(reinterpret_cast<float *>(&data[i + flipStride * 2]) + 16),
          1);

        auto cI = _mm256_insertf128_ps(
          _mm256_castps128_ps256(_mm_loadu_ps(reinterpret_cast<float *>(&data[i + flipStride * 0]) + 16 + 8)),
          _mm_loadu_ps(reinterpret_cast<float *>(&data[i + flipStride * 2]) + 16 + 8),
          1);

        auto dR = _mm256_insertf128_ps(
          _mm256_castps128_ps256(_mm_loadu_ps(reinterpret_cast<float *>(&data[i + flipStride * 0]) + 20)),
          _mm_loadu_ps(reinterpret_cast<float *>(&data[i + flipStride * 2]) + 20),
          1);

        auto dI = _mm256_insertf128_ps(
          _mm256_castps128_ps256(_mm_loadu_ps(reinterpret_cast<float *>(&data[i + flipStride * 0]) + 20 + 8)),
          _mm_loadu_ps(reinterpret_cast<float *>(&data[i + flipStride * 2]) + 20 + 8),
          1);

        {
          const auto [w0RLo, w0ILo, w1RLo, w1ILo] = LoadSingleAngle(tableData, angleIndex0, flipStride);
          const auto [w0RHi, w0IHi, w1RHi, w1IHi] = LoadSingleAngle(tableData, angleIndex1, flipStride);
          const auto w0R = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_set1_ps(w0RLo)), _mm_set1_ps(w0RHi), 1);
          const auto w0I = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_set1_ps(w0ILo)), _mm_set1_ps(w0IHi), 1);
          const auto w1R = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_set1_ps(w1RLo)), _mm_set1_ps(w1RHi), 1);
          const auto w1I = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_set1_ps(w1ILo)), _mm_set1_ps(w1IHi), 1);

          DoRadix4(aR, aI, bR, bI, cR, cI, dR, dI, w0R, w0I, w1R, w1I, two);
        }

        _mm256_storeu_ps(reinterpret_cast<float *>(&data[i + flipStride * 0 + 0]), aR);
        _mm256_storeu_ps(reinterpret_cast<float *>(&data[i + flipStride * 0 + 0]) + 8, aI);

        _mm256_storeu_ps(reinterpret_cast<float *>(&data[i + flipStride * 2 + 0]), bR);
        _mm256_storeu_ps(reinterpret_cast<float *>(&data[i + flipStride * 2 + 0]) + 8, bI);

        aR = _mm256_insertf128_ps(
          _mm256_castps128_ps256(_mm_loadu_ps(reinterpret_cast<float *>(&data[i + flipStride * 1]))),
          _mm_loadu_ps(reinterpret_cast<float *>(&data[i + flipStride * 3])),
          1);
        bR = _mm256_insertf128_ps(
          _mm256_castps128_ps256(_mm_loadu_ps(reinterpret_cast<float *>(&data[i + flipStride * 1]) + 4)),
          _mm_loadu_ps(reinterpret_cast<float *>(&data[i + flipStride * 3]) + 4),
          1);
        _mm256_storeu_ps(reinterpret_cast<float *>(&data[i + flipStride * 1]), cR);
        _mm256_storeu_ps(reinterpret_cast<float *>(&data[i + flipStride * 3]), dR);

        aI = _mm256_insertf128_ps(
          _mm256_castps128_ps256(_mm_loadu_ps(reinterpret_cast<float *>(&data[i + flipStride * 1]) + 8)),
          _mm_loadu_ps(reinterpret_cast<float *>(&data[i + flipStride * 3]) + 8),
          1);
        bI = _mm256_insertf128_ps(
          _mm256_castps128_ps256(_mm_loadu_ps(reinterpret_cast<float *>(&data[i + flipStride * 1]) + 4 + 8)),
          _mm_loadu_ps(reinterpret_cast<float *>(&data[i + flipStride * 3]) + 4 + 8),
          1);

        _mm256_storeu_ps(reinterpret_cast<float *>(&data[i + flipStride * 1]) + 8, cI);
        _mm256_storeu_ps(reinterpret_cast<float *>(&data[i + flipStride * 3]) + 8, dI);

        cR = _mm256_insertf128_ps(
          _mm256_castps128_ps256(_mm_loadu_ps(reinterpret_cast<float *>(&data[i + flipStride * 1]) + 16)),
          _mm_loadu_ps(reinterpret_cast<float *>(&data[i + flipStride * 3]) + 16),
          1);

        cI = _mm256_insertf128_ps(
          _mm256_castps128_ps256(_mm_loadu_ps(reinterpret_cast<float *>(&data[i + flipStride * 1]) + 16 + 8)),
          _mm_loadu_ps(reinterpret_cast<float *>(&data[i + flipStride * 3]) + 16 + 8),
          1);

        dR = _mm256_insertf128_ps(
          _mm256_castps128_ps256(_mm_loadu_ps(reinterpret_cast<float *>(&data[i + flipStride * 1]) + 20)),
          _mm_loadu_ps(reinterpret_cast<float *>(&data[i + flipStride * 3]) + 20),
          1);

        dI = _mm256_insertf128_ps(
          _mm256_castps128_ps256(_mm_loadu_ps(reinterpret_cast<float *>(&data[i + flipStride * 1]) + 20 + 8)),
          _mm_loadu_ps(reinterpret_cast<float *>(&data[i + flipStride * 3]) + 20 + 8),
          1);

        {
          const auto [w0RLo, w0ILo, w1RLo, w1ILo] = LoadSingleAngle(tableData, angleIndex2, flipStride);
          const auto [w0RHi, w0IHi, w1RHi, w1IHi] = LoadSingleAngle(tableData, angleIndex3, flipStride);
          const auto w0R = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_set1_ps(w0RLo)), _mm_set1_ps(w0RHi), 1);
          const auto w0I = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_set1_ps(w0ILo)), _mm_set1_ps(w0IHi), 1);
          const auto w1R = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_set1_ps(w1RLo)), _mm_set1_ps(w1RHi), 1);
          const auto w1I = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_set1_ps(w1ILo)), _mm_set1_ps(w1IHi), 1);

          DoRadix4(aR, aI, bR, bI, cR, cI, dR, dI, w0R, w0I, w1R, w1I, two);
        }

        _mm256_storeu_ps(reinterpret_cast<float *>(&data[i + flipStride * 0]) + 16, aR);
        _mm256_storeu_ps(reinterpret_cast<float *>(&data[i + flipStride * 0]) + 16 + 8, aI);

        _mm256_storeu_ps(reinterpret_cast<float *>(&data[i + flipStride * 2]) + 16, bR);
        _mm256_storeu_ps(reinterpret_cast<float *>(&data[i + flipStride * 2]) + 16 + 8, bI);

        _mm256_storeu_ps(reinterpret_cast<float *>(&data[i + flipStride * 1]) + 16, cR);
        _mm256_storeu_ps(reinterpret_cast<float *>(&data[i + flipStride * 1]) + 16 + 8, cI);

        _mm256_storeu_ps(reinterpret_cast<float *>(&data[i + flipStride * 3]) + 16 + 8, dI);
        _mm256_storeu_ps(reinterpret_cast<float *>(&data[i + flipStride * 3]) + 16, dR);
      }
    }
  }
}

template <std::floating_point F>
void LastDoubleFlipRadix4PassAVX(const F *tableData, std::complex<F> *data, const size_t length)
{
  const auto flipStride = length / 4;

  // Each butterfly set has a single butterfly in it, but we're doing 2 at a time because AVX
  const auto two = _mm256_set1_ps(2.0f);
  for (auto butterflySetStartIndex = 0_z; butterflySetStartIndex < flipStride; butterflySetStartIndex += 4 * 2)
  {
    auto i = butterflySetStartIndex;

    // Note that b and c are reversed order
    auto aR = _mm256_loadu_ps(reinterpret_cast<float *>(&data[i + flipStride * 0]));
    auto aI = _mm256_loadu_ps(reinterpret_cast<float *>(&data[i + flipStride * 0]) + 8);
    auto bR = _mm256_loadu_ps(reinterpret_cast<float *>(&data[i + flipStride * 2]));
    auto bI = _mm256_loadu_ps(reinterpret_cast<float *>(&data[i + flipStride * 2]) + 8);
    auto cR = _mm256_loadu_ps(reinterpret_cast<float *>(&data[i + flipStride * 1]));
    auto cI = _mm256_loadu_ps(reinterpret_cast<float *>(&data[i + flipStride * 1]) + 8);
    auto dR = _mm256_loadu_ps(reinterpret_cast<float *>(&data[i + flipStride * 3]));
    auto dI = _mm256_loadu_ps(reinterpret_cast<float *>(&data[i + flipStride * 3]) + 8);

    // Do a 4x4 transform (in each AVX lane) for each of R and I
    auto t0 = _mm256_shuffle_ps(aR, bR, 0x44);
    auto t2 = _mm256_shuffle_ps(aR, bR, 0xEE);
    auto t1 = _mm256_shuffle_ps(cR, dR, 0x44);
    auto t3 = _mm256_shuffle_ps(cR, dR, 0xEE);

    aR = _mm256_shuffle_ps(t0, t1, 0x88);
    bR = _mm256_shuffle_ps(t0, t1, 0xDD);
    cR = _mm256_shuffle_ps(t2, t3, 0x88);
    dR = _mm256_shuffle_ps(t2, t3, 0xDD);

    t0 = _mm256_shuffle_ps(aI, bI, 0x44);
    t2 = _mm256_shuffle_ps(aI, bI, 0xEE);
    t1 = _mm256_shuffle_ps(cI, dI, 0x44);
    t3 = _mm256_shuffle_ps(cI, dI, 0xEE);

    aI = _mm256_shuffle_ps(t0, t1, 0x88);
    bI = _mm256_shuffle_ps(t0, t1, 0xDD);
    cI = _mm256_shuffle_ps(t2, t3, 0x88);
    dI = _mm256_shuffle_ps(t2, t3, 0xDD);

    // Do all 8 radix4s (with 8 unique angles) at once.
    auto [w0R, w0I, w1R, w1I] = LoadAnglesAVX(tableData, butterflySetStartIndex, flipStride);
    DoRadix4(aR, aI, bR, bI, cR, cI, dR, dI, w0R, w0I, w1R, w1I, two);

    const auto StoreInterleaved =
      [](auto *dest, auto a, auto b)
      {
        // (A B C D E F G H) (a b c d e f g h) -> (A a B b E e F f)
        auto AaBbEeFf = _mm256_unpacklo_ps(a, b);

        // (A B C D E F G H) (a b c d e f g h) -> (C c D d G g H h)
        auto CcDdGgHh = _mm256_unpackhi_ps(a, b);

        auto EeFf = _mm256_extractf128_ps(AaBbEeFf, 1);
        auto GgHh = _mm256_extractf128_ps(CcDdGgHh, 1);
        auto AaBb = _mm256_castps256_ps128(AaBbEeFf);
        auto CcDd = _mm256_castps256_ps128(CcDdGgHh);
        _mm_storeu_ps(dest + 0, AaBb);
        _mm_storeu_ps(dest + 4, CcDd);
        _mm_storeu_ps(dest + 8, EeFf);
        _mm_storeu_ps(dest + 12, GgHh);
      };

    StoreInterleaved(reinterpret_cast<float *>(&data[i + flipStride * 0]), aR, aI);
    StoreInterleaved(reinterpret_cast<float *>(&data[i + flipStride * 2]), bR, bI);
    StoreInterleaved(reinterpret_cast<float *>(&data[i + flipStride * 1]), cR, cI);
    StoreInterleaved(reinterpret_cast<float *>(&data[i + flipStride * 3]), dR, dI);
  }
}

/*template <std::floating_point F>
void MiddleOutFFT(
  const F *tableData,
  const std::complex<F> *input,
  std::complex<F> *output,
  size_t length)
{
  if (length < 256)
  {
    // Special cases (64 may also want to be one in AVX since everything could *almost* fit in registers!)
    return;
  }


}*/
}

extern "C"
{
  /*void fft_cplx_interleaved_f32_fma(const fft_table_f32 *table, const float *inputN, float *outputN, size_t fftLength)
  {
    [[maybe_unused]] auto t = reinterpret_cast<const FFTTableImpl<float> *>(table);
    //const auto tableData = t->Data();

    // COMMON_ASSERTS();
    //if (fftLength < 256)
    {
      FMA::RunComplexFFT<Dir::Forward>(t->Data(), inputN, outputN, fftLength);
      return;
    }

    const auto length = fftLength;
    const auto *input = reinterpret_cast<const std::complex<float> *>(inputN);
    auto *output = reinterpret_cast<std::complex<float> *>(outputN);

    const auto bitCount = sizeof(length) * 8 - std::countl_zero(length) - 1;
    const auto bitReverseCount = bitCount >> 1;

    // Note that this folds the radix 2 pass into the first radix 4/8 pass for non-power-of-four FFTs, so it's actually
    //  one less than would traditionally be the case (and ends up matching the bit reverse count)
    const auto radix4StepCount = bitCount >> 1;

    // This one *does* want to take into account the extra bit reverse though.
    const auto radix4BitReverseStepCount = (bitReverseCount + 1) >> 1;

    // Step 0: Either a radix 4 or 8 (depending on if pow4) pass with no rotation
    Radix4Or8InitialStepAVX(input, output, length);

    auto radix4Stride = length >> (IsPowerOfFourAssumingIsPowerOfTwo(length) ? 4 : 5);

    // Steps [1 - radix4StepCount - radix4BitReverseStepCount): standard radix 4 passes
    auto stepIndex = 1_z;
    for (; stepIndex < radix4StepCount - radix4BitReverseStepCount; stepIndex++)
    {
      NoFlipRadix4PassAVX(tableData, output, length, radix4Stride);
      radix4Stride >>= 2;
    }

    // If non-pow-4-length:
    //   Step (radix4StepCount - radix4BitReverseStepCount): single bit flip
    if ((bitReverseCount & 1) != 0)
    {
      SingleFlipRadix4PassAVX(tableData, output, length, radix4Stride);
      stepIndex++;
      radix4Stride >>= 2;
    }

    // All steps until N - (2 if AVX f32, 1 otherwise): double bit flip radix 4 step
    const auto finishStep = radix4StepCount - 2; // !!! -1 if not AVX

    for (;stepIndex < finishStep; stepIndex++)
    {
      DoubleFlipRadix4PassAVX(tableData, output, length, radix4Stride);

      // !!! early return on last one since those values are never used again
      radix4Stride >>= 2;
    }


    // If AVX f32, step N - 2:
    //  Double bit flip radix 4 where half-vectors matter
    // !!! only if AVX
    constexpr auto SIMDSize = 8u;
    if constexpr (SIMDSize > 4u)
    {
      SecondToLastDoubleFlipRadix4PassAVX(tableData, output, length);
    }


    // Step N - 1:
    //  The final weird flipperooni
    LastDoubleFlipRadix4PassAVX(tableData, output, length);
  }*/


  void fft_cplx_interleaved_f64_fma(const fft_table_f64 *table, const double *input, double *output, size_t fftLength)
  {
    [[maybe_unused]] auto t = reinterpret_cast<const FFTTableImpl<double> *>(table);
    COMMON_ASSERTS();
    FMA::RunComplexFFT<Dir::Forward>(t->Data(), input, output, fftLength);
  }


   void ifft_cplx_interleaved_f64_fma(const fft_table_f64 *table, const double *input, double *output, size_t fftLength)
  {
    [[maybe_unused]] auto t = reinterpret_cast<const FFTTableImpl<double> *>(table);
    COMMON_ASSERTS();
    FMA::RunComplexFFT<Dir::Inverse>(t->Data(), input, output, fftLength);
  }
}