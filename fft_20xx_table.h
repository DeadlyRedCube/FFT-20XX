#pragma once

#pragma warning (disable: 5045) // Compiler will insert Spectre mitigation for memory load if /Qspectre switch specified

#include <cassert>
#include <cinttypes>
#include <cmath>

#include "fft_20xx_bit_reverse_copy.h"
#include "fft_20xx_complex.h"
#include "fft_20xx_math.h"



namespace FFT20XX
{
  #ifdef _MSC_VER
    #pragma warning (push)
    #pragma warning (disable: 4820) // 'N' bytes padding added after data member 'Foo'
  #endif
  template <typename F>
  class FFTTableImpl
  {
  public:
    FFTTableImpl(std::size_t maxFFTLength)
    : _data{new Complex<F>[maxFFTLength / 2]}
    , _ownsData(true)
    , _maxFFTLength(maxFFTLength)
    {
      assert(IsPowerOfTwo(maxFFTLength));
      BuildTable();
    }


    size_t MaxFFTLength() const noexcept
      { return _maxFFTLength; }

    const Complex<F> *Data() const noexcept
      { return _data; }

  private:
    void BuildTable() noexcept
    {
      const auto bitCount = Log2Floor(_maxFFTLength) - 1;
      auto revUnshifted = std::size_t(0);
      const auto revShift = BitReverseShiftFromBitCount<std::size_t>(bitCount);
      for (auto i = std::size_t(0); i < _maxFFTLength / 2; i += 2)
      {
        auto rev = BitReverse(i) >> revShift;

        constexpr auto NegTwoPi = -6.283185307179586;
        auto angle = NegTwoPi * double(rev) / double(_maxFFTLength);
        _data[i] = { .r = F(std::cos(angle)), .i = F(std::sin(angle)) };

        angle *= 3.0;
        _data[i + 1] = { .r = F(std::cos(angle)), .i = F(std::sin(angle)) };

        revUnshifted = BitReverseIncrement(revUnshifted);
      }
    }

    Complex<F> *_data = nullptr;
    bool _ownsData = false;
    size_t _maxFFTLength = 0;
  };

  #ifdef _MSC_VER
    #pragma warning (pop)
  #endif
}


