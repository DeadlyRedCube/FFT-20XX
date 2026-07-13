#pragma once

#include <cassert>

#include "fft_20xx_platform.h"

#include "fft_20xx/fft_20xx.h"

#define COMMON_ASSERTS() \
  assert(t != nullptr); \
  assert(input != nullptr); \
  assert(output != nullptr); \
  assert(IsPowerOfTwo(fftLength)); \
  assert(t->MaxFFTLength() >= fftLength)

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

  template <typename F> constexpr auto InvSqrt2 = F(0.7071067811865476);
  template <typename F> constexpr auto CosPiOver8 = F(0.9238795325112867561);
  template <typename F> constexpr auto SinPiOver8 = F(0.3826834323650897717);

  template <std::unsigned_integral I>
  constexpr auto IsPowerOfTwo(I v) -> bool
    { return (v > 0) && (v & (v - 1)) == 0; }

  template <std::unsigned_integral I>
  constexpr auto IsPowerOfFourAssumingIsPowerOfTwo(I v) -> bool
    { return (v & I(0x55555555'55555555)) != 0; }

  template <std::unsigned_integral I>
  constexpr auto Log2Floor(I i) noexcept  -> std::uint32_t
    { return std::uint32_t(std::numeric_limits<I>::digits - 1 - std::countl_zero(i)); }

  struct INegType;
  struct IType
  {
    consteval INegType operator-() const noexcept;
  };

  struct INegType
  {
    consteval IType operator-() const noexcept { return {}; }
  };

  inline consteval INegType IType::operator-() const noexcept { return {}; }

  template <typename F>
  struct Complex
  {
    F r;
    F i;

    friend constexpr auto operator+(Complex a, Complex b) noexcept -> Complex
      { return { .r = a.r + b.r, .i = a.i + b.i }; }

    friend constexpr auto operator-(Complex a, Complex b) noexcept -> Complex
      { return { .r = a.r - b.r, .i = a.i - b.i }; }

    friend constexpr auto operator*(Complex a, Complex b) noexcept -> Complex
      { return { .r = a.r * b.r - a.i * b.i, .i = a.r * b.i + b.r * a.i }; }

    friend constexpr auto operator*(Complex a, IType) noexcept -> Complex
      { return { .r = -a.i, .i = a.r }; }

    friend constexpr auto operator*(Complex a, INegType) noexcept -> Complex
      { return { .r = a.i, .i = -a.r }; }

    friend constexpr auto operator*(IType, Complex a) noexcept -> Complex
      { return { .r = -a.i, .i = a.r }; }

    friend constexpr auto operator*(INegType, Complex a) noexcept -> Complex
      { return { .r = a.i, .i = -a.r }; }

    constexpr auto operator*(F s) noexcept -> Complex
      { return { .r = r * s, .i = i * s }; }

    constexpr auto operator+=(Complex b) noexcept -> Complex &
      { *this = *this + b; return *this; }

    constexpr auto operator-=(Complex b) noexcept -> Complex &
      { *this = *this - b; return *this; }

    constexpr auto operator*=(Complex b) noexcept -> Complex &
      { *this = *this * b; return *this; }

    constexpr auto operator*=(IType) noexcept -> Complex &
      { *this = { .r = -i, .i = r }; return *this; }

    constexpr auto operator*=(INegType) noexcept -> Complex &
      { *this = { .r = i, .i = -r }; return *this; }

    constexpr auto operator*=(F s) noexcept -> Complex &
      { r *= s; i *= s; return *this; }

    auto operator-() const noexcept -> Complex
      { return { .r = -r, .i = -i }; }

    constexpr auto operator==(const Complex &) const noexcept -> bool = default;
    constexpr auto operator<=>(const Complex &) const noexcept = default;
  };

  template <std::floating_point F>
  constexpr Complex<F> Conjugate(Complex<F> c) noexcept
    { return { .r = c.r, .i = -c.i }; }

  constexpr IType II { };

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

  template <typename V>
  struct ComplexVec
  {
    V r;
    V i;
  };

  // Using a custom Pair struct instead of std::pair simply to avoid some ABI warnings on GCC (on ARM), which could be
  //  disabled with -Wno-psabi but in practice that's not a great solution.
  template <typename A, typename B>
  struct Pair
  {
    A first;
    B second;
  };

  enum class Alignment
  {
    Unaligned,
    Aligned,
  };


  // !!! This should go back taway
  enum class DataType
  {
    Complex,
    Real,
  };



  enum class TangleType
  {
    ForwardLargest,
    ForwardInner,
    InverseLargest,
    InverseInner,
  };



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
}