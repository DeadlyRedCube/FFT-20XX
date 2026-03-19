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

  private:
    void BuildTable() noexcept
    {
      if (_maxFFTLength >= 16u)
      {
        for (auto i = size_t(0); i < _maxFFTLength * 3 / 4; i++)
        {
          constexpr auto NegTwoPi = -6.283185307179586;
          auto angle = NegTwoPi * double(i) / double(_maxFFTLength);
          const auto offset = _maxFFTLength * 3 / 4;
          _data[i + offset] = F(std::cos(angle));
        }

        // Now spread back from max until we reach 16 (8 and 4 are handled specially)
        for (auto realCount = _maxFFTLength / 4; realCount >= 16; realCount /= 2)
        {
          for (auto i = 0u; i < realCount * 3 / 2; i++)
          {
            const auto offset = realCount * 3 / 2;
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
    }

    size_t _maxFFTLength = 0;
    F *_data = nullptr;
    bool _ownsData = false;
  };

  #if COMPILER_MSVC
    #pragma warning (pop)
  #endif
}


