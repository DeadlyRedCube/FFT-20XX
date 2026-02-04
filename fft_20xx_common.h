#pragma once

#include <algorithm>
#include <cinttypes>
#include <cmath>

#include "include/fft_20xx.h"

#include "fft_20xx_complex.h"
#include "fft_20xx_table.h"


#ifdef _MSC_VER
  #define FORCE_INLINE __forceinline
  #pragma warning (disable: 5045) // Compiler will insert Spectre mitigation for memory load if /Qspectre switch specified
#else
  #define FORCE_INLINE inline
#endif

#include "fft_20xx_complex.h"

namespace FFT20XX
{
  enum class Dir
  {
    Forward,
    Reverse,
  };

  template <typename F>
  static constexpr auto InvSqrt2 = F(0.7071067811865476);

  template <Dir D, typename F>
  auto Rot(Complex<F> v, Complex<F> w) -> Complex<F>
  {
    if constexpr (D == Dir::Forward)
      { return v * w; }
    else
      {  } // !!! multiply with conjugate (as a single operation), I think
  }

  template <Dir D, typename F>
  auto Rot45(Complex<F> v) -> Complex<F>
  {
    if constexpr (D == Dir::Forward)
      { return { .r = InvSqrt2<F> * (v.r + v.i), .i = InvSqrt2<F> * (v.i - v.r) }; }
    else
      { } // !!!
  }

  template <Dir D, typename F>
  auto Rot90(Complex<F> v) -> Complex<F>
  {
    if constexpr (D == Dir::Forward)
      { return v * -II; }
    else
      { return v * II; }
  }

  template <Dir D, typename F>
  auto Rot135(Complex<F> v) -> Complex<F>
  {
    if constexpr (D == Dir::Forward)
      { return { .r = InvSqrt2<F> * (v.i - v.r), .i = -InvSqrt2<F> * (v.r + v.i) }; }
    else
      { } // !!!
  }
}

