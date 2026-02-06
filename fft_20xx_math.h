#pragma once

#include "fft_20xx_platform.h"

#include <bit>
#include <cinttypes>
#include <concepts>
#include <limits>

namespace FFT20XX
{
  template <std::unsigned_integral I>
  constexpr auto IsPowerOfTwo(I v) -> bool
    { return (v > 0) && (v & (v - 1)) == 0; }

  template <std::unsigned_integral I>
  constexpr auto Log2Floor(I i) noexcept  -> std::uint32_t
    { return std::uint32_t(std::numeric_limits<I>::digits - 1 - std::countl_zero(i)); }

  template <std::unsigned_integral I>
  constexpr auto BitReverse(I i) -> I
  {
    #if COMPILER_MSVC
      #pragma warning (push)
      #pragma warning (disable: 4310) // cast truncates constant value
    #endif

    // TODO: ARM has a dedicated reverse instruction
    constexpr auto Mask0 = I(0x55555555'55555555);
    constexpr auto Mask1 = I(0x33333333'33333333);
    constexpr auto Mask2 = I(0x0F0F0F0F'0F0F0F0F);
    constexpr auto Mask3 = I(0x00FF00FF'00FF00FF);
    constexpr auto Mask4 = I(0x0000FFFF'0000FFFF);
    constexpr auto Mask5 = I(0x00000000'FFFFFFFF);

    i = ((i >> 1) & Mask0) | ((i & Mask0) << 1);
    i = ((i >> 2) & Mask1) | ((i & Mask1) << 2);
    i = ((i >> 4) & Mask2) | ((i & Mask2) << 4);
    if constexpr (Mask3 != I(-1))
      { i = ((i >> 8) & Mask3) | ((i & Mask3) << 8); }

    if constexpr (Mask4 != I(-1))
      { i = ((i >> 16) & Mask4) | ((i & Mask4) << 16); }

    if constexpr (Mask5 != I(-1))
      { i = ((i >> 32) & Mask5) | ((i & Mask5) << 32); }
    return i;

    #if COMPILER_MSVC
      #pragma warning (pop)
    #endif
  }


  template <std::unsigned_integral I>
  constexpr auto BitReverseIncrement(I v) noexcept -> I
  {
    auto onesShift = std::countl_one(v);
    constexpr auto SignedHighBit = std::numeric_limits<std::make_signed_t<I>>::min();
    auto ones = I(SignedHighBit >> onesShift);
    return v ^ ones;
  }

  template <std::unsigned_integral I>
  constexpr auto BitReverseShiftFromBitCount(std::uint32_t bitCount) noexcept -> std::uint32_t
    { return std::uint32_t(std::numeric_limits<I>::digits - bitCount); }
}
