#pragma once

#include "fft_20xx/fft_20xx.hpp"
#include "fft_20xx/fft_20xx_util.hpp"
#include "fft_20xx_common.h"

#include <cstring>


namespace FFT20XX
{
  template <std::floating_point F>
  constexpr auto TableDataOffset = std::max({1_z, sizeof(size_t) / sizeof(F), SIMDAlignment / sizeof(F)});;
  static_assert(TableDataOffset<float> * sizeof(float) >= sizeof(size_t));
  static_assert(TableDataOffset<double> * sizeof(double) >= sizeof(size_t));

  #define FFT20XX_TABLE_INIT(Table, F) \
    inline const F *GetTableData(const fft_20xx::c::Table *t) \
    { \
      assert(IsSIMDAlignedPtr(t)); \
      return reinterpret_cast<const F *>(t) + TableDataOffset<F>; \
    } \
    \
    inline size_t GetTableMaxFFTLength(const fft_20xx::c::Table *t) \
    { \
      size_t r; \
      memcpy(&r, t, sizeof(size_t)); \
      return r; \
    }

  FFT20XX_TABLE_INIT(fft_table_cplx_f32, float);
  FFT20XX_TABLE_INIT(fft_table_cplx_f64, double);
  FFT20XX_TABLE_INIT(fft_table_real_f32, float);
  FFT20XX_TABLE_INIT(fft_table_real_f64, double);

  #undef FFT20XX_TABLE_INIT
}


