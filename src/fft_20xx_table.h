#pragma once

#include "fft_20xx/fft_20xx.hpp"
#include "fft_20xx/fft_20xx_util.hpp"
#include "fft_20xx_common.h"

#include <cstring>


namespace FFT20XX
{
  // Use the maximum supported SIMD alignment here so that if someone serializes the table data to file, the offset
  //  will be correct regardless of the platform (assuming endianness is the same, at least)
  template <std::floating_point F>
  constexpr auto TableDataOffset = std::max({1_z, sizeof(size_t) / sizeof(F), MaxSIMDAlignment / sizeof(F)});
  static_assert(TableDataOffset<float> * sizeof(float) >= sizeof(size_t));
  static_assert(TableDataOffset<double> * sizeof(double) >= sizeof(size_t));

  #define FFT20XX_TABLE_INIT(Table, F, Type) \
    const fft_20xx::c::Table *GetManagedTable##Type(size_t maxSize); \
    \
    inline const F *GetTableData(const fft_20xx::c::Table *t, size_t maxSize) \
    { \
      if (t == nullptr) \
      { \
        t = GetManagedTable##Type(maxSize); \
      } \
      assert(IsSIMDAlignedPtr(t)); \
      return reinterpret_cast<const F *>(t) + TableDataOffset<F>; \
    } \
    \
    inline size_t GetTableMaxFFTLength(const fft_20xx::c::Table * &t, size_t maxSize) \
    { \
      if (t == nullptr) \
      { \
        t = GetManagedTable##Type(maxSize); \
      } \
      size_t r; \
      memcpy(&r, t, sizeof(size_t)); \
      return r; \
    }

  FFT20XX_TABLE_INIT(fft_table_cplx_f32, float, CplxF32);
  FFT20XX_TABLE_INIT(fft_table_cplx_f64, double, CplxF64);
  FFT20XX_TABLE_INIT(fft_table_real_f32, float, RealF32);
  FFT20XX_TABLE_INIT(fft_table_real_f64, double, RealF64);

  #undef FFT20XX_TABLE_INIT
}


