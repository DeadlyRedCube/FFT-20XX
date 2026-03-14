#pragma once

#include "fft_20xx_platform.h"

#include "include/fft_20xx.h"

#include "fft_20xx_complex.h"
#include "fft_20xx_table.h"
#include "fft_20xx_complex.h"

namespace FFT20XX
{
  enum class Dir
  {
    Forward,
    Inverse,
  };

  constexpr size_t operator""_z(unsigned long long l)
  {
    return size_t(l);
  }

  template <typename F>
  static constexpr auto InvSqrt2 = F(0.7071067811865476);

  template <Dir D, typename F>
  auto Rot(Complex<F> v, Complex<F> w) -> Complex<F>
  {
    if constexpr (D == Dir::Forward)
      { return v * w; }
    else
      { return v * Conjugate(w); }
  }

  template <Dir D, typename F>
  auto Rot45(Complex<F> v) -> Complex<F>
  {
    if constexpr (D == Dir::Forward)
      { return { .r = InvSqrt2<F> * (v.r + v.i), .i = InvSqrt2<F> * (v.i - v.r) }; }
    else
      { return { .r = InvSqrt2<F> * (v.r - v.i), .i = InvSqrt2<F> * (v.r + v.i) }; }
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
      { return { .r = -InvSqrt2<F> * (v.r + v.i), .i = InvSqrt2<F> * (v.r - v.i) }; }
  }

  // Using a custom Pair struct instead of std::pair simply to avoid some ABI warnings on GCC (on ARM), which could be
  //  disabled with -Wno-psabi but in practice that's not a great solution.
  template <typename A, typename B>
  struct Pair
  {
    A first;
    B second;
  };
}

