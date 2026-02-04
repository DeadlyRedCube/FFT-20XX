#pragma once

#include <compare>

namespace FFT20XX
{
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

    auto operator-() const noexcept -> Complex
      { return { .r = -r, .i = -i }; }

    constexpr auto operator==(const Complex &) const noexcept -> bool = default;
    constexpr auto operator<=>(const Complex &) const noexcept = default;
  };

  template <std::floating_point F>
  constexpr Complex<F> Conjugate(Complex<F> c) noexcept
    { return { .r = c.r, .i = -c.i }; }

  constexpr IType II { };
};