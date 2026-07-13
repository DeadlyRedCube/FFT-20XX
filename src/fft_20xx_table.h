#pragma once

#include "fft_20xx/fft_20xx_util.h"
#include "fft_20xx_common.h"



namespace FFT20XX
{
  #if COMPILER_MSVC
    #pragma warning (push)
    #pragma warning (disable: 4820) // 'N' bytes padding added after data member 'Foo'
  #endif
  template <typename F>
  class FFTTableImpl
  {
  public:
    FFTTableImpl(std::size_t maxFFTLength)
    : _maxFFTLength(std::max(size_t(32), maxFFTLength))
    , _data{fft_20xx::allocate_type_simd_aligned<F>(_maxFFTLength * 3 / 2)}
    , _realData{fft_20xx::allocate_type_simd_aligned<F>(_maxFFTLength)}
    , _ownsData(true)
    {
      assert(IsPowerOfTwo(maxFFTLength));
      BuildTable();
    }

    ~FFTTableImpl()
    {
      if (_ownsData)
        { fft_20xx::free_simd_aligned(_data); }
    }

    size_t MaxFFTLength() const noexcept
      { return _maxFFTLength; }

    const F *Data() const noexcept
      { return _data; }

    const F *RealData() const noexcept
      { return _realData; }

  private:
    // The real FFTs have a strange ordering to the angles due to the order that the data moves around through the
    //  algorithm - this flips the index around when creating the angle so that we store the tables in this oddly-
    //  reversed order.
    static size_t DoRealFFTIndexReverse(size_t i)
    {
      auto mask = (i & (i - 1)) >> 1;

      // Turns out, this transformation is called "Gray code"
      // It's possible (via wikipedia) to calculate this with a single "carryless multiply" instruction on
      // Platforms where that's a thing - but I couldn't find a clean way to test that in C++.
      mask ^= mask >> 32;
      mask ^= mask >> 16;
      mask ^= mask >> 8;
      mask ^= mask >> 4;
      mask ^= mask >> 2;
      mask ^= mask >> 1;

      return (i - (mask & 1)) ^ mask;
    }

    void BuildTable() noexcept
    {
      if (_maxFFTLength >= 16u)
      {
        {
          const auto offset = _maxFFTLength * 3 / 4;
          for (auto i = size_t(0); i < _maxFFTLength * 3 / 4; i++)
          {
            constexpr auto NegTwoPi = -6.283185307179586;
            auto angle = NegTwoPi * double(i) / double(_maxFFTLength);
            _data[i + offset] = F(std::cos(angle));
          }
        }

        // Now spread back from max until we reach 16 (8 and 4 are handled specially)
        for (auto realCount = _maxFFTLength / 4; realCount >= 16; realCount /= 2)
        {
          const auto offset = realCount * 3 / 2;
          for (auto i = 0u; i < realCount * 3 / 2; i++)
          {
            _data[i + offset] = _data[(i + offset) * 2];
          }
        }

        // Finally, handle the 4- and 8- angle cases (Which use a full 2x angle count for nice, aligned  SIMD loading)
        for (auto i = 0u; i < 8u; i++)
        {
          _data[8 + i] = _data[24 + 2*i];
          _data[8 + 8 + i] = _data[24 + 8 + 2*i];
        }

        for (auto i = 0u; i < 4u; i++)
        {
          _data[i] = _data[8 + 2*i];
          _data[4 + i] = _data[8 + 8 + 2*i];
        }
      }

      // First 8 elements are completely unused and are just there for padding/ease of indexing.
      _realData[0] = 0.0f;
      _realData[1] = 0.0f;
      _realData[2] = 0.0f;
      _realData[3] = 0.0f;
      _realData[4] = 0.0f;
      _realData[5] = 0.0f;
      _realData[6] = 0.0f;
      _realData[7] = 0.0f;
      for (auto angleCount = 4_z; angleCount <= _maxFFTLength / 4; angleCount *= 2)
      {
        const auto cosOffset = angleCount * 2;
        const auto sinOffset = cosOffset + angleCount;
        for (auto i = 0_z; i < angleCount; i++)
        {
          constexpr auto NegTwoPi = -6.283185307179586;
          auto angle = NegTwoPi * double(DoRealFFTIndexReverse(i)) / double(angleCount * 4);
          _realData[cosOffset + i] = F(std::cos(angle));
          _realData[sinOffset + i] = F(std::sin(angle));
        }
      }
    }

    size_t _maxFFTLength = 0;
    F *_data = nullptr;
    F *_realData = nullptr;
    bool _ownsData = false;
  };

  #if COMPILER_MSVC
    #pragma warning (pop)
  #endif
}


