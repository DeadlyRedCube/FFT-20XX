#include "fft_20xx_table.h"

#include "fft_20xx/fft_20xx_util.h"
#include "fft_20xx_common.h"

namespace
{
  using namespace FFT20XX;

  constexpr auto MinTableFFTLength = 32_z;

  size_t GetComplexTableElementCount(size_t maxFFTLength)
  {
    return std::max(MinTableFFTLength, maxFFTLength) * 3 / 2;
  }

  size_t GetRealTableElementCount(size_t maxFFTLength)
  {
    return std::max(MinTableFFTLength, maxFFTLength);
  }

  template <std::floating_point F>
  void InitializeComplexTable(F *table, size_t maxFFTLength)
  {
    maxFFTLength = std::max(MinTableFFTLength, maxFFTLength);

    {
      const auto offset = maxFFTLength * 3 / 4;
      for (auto i = size_t(0); i < maxFFTLength * 3 / 4; i++)
      {
        constexpr auto NegTwoPi = -6.283185307179586;
        auto angle = NegTwoPi * double(i) / double(maxFFTLength);
        table[i + offset] = F(std::cos(angle));
      }
    }

    // Now spread back from max until we reach 16 (8 and 4 are handled specially)
    for (auto realCount = maxFFTLength / 4; realCount >= 16; realCount /= 2)
    {
      const auto offset = realCount * 3 / 2;
      for (auto i = 0u; i < realCount * 3 / 2; i++)
      {
        table[i + offset] = table[(i + offset) * 2];
      }
    }

    // Finally, handle the 4- and 8- angle cases (Which use a full 2x angle count for nice, aligned  SIMD loading)
    for (auto i = 0u; i < 8u; i++)
    {
      table[8 + i] = table[24 + 2*i];
      table[8 + 8 + i] = table[24 + 8 + 2*i];
    }

    for (auto i = 0u; i < 4u; i++)
    {
      table[i] = table[8 + 2*i];
      table[4 + i] = table[8 + 8 + 2*i];
    }
  }



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



  template <std::floating_point F>
  void InitializeRealTable(F *table, size_t maxFFTLength)
  {
    maxFFTLength = std::max(MinTableFFTLength, maxFFTLength);

    // First 8 elements are completely unused and are just there for padding/ease of indexing.
    table[0] = 0.0f;
    table[1] = 0.0f;
    table[2] = 0.0f;
    table[3] = 0.0f;
    table[4] = 0.0f;
    table[5] = 0.0f;
    table[6] = 0.0f;
    table[7] = 0.0f;
    for (auto angleCount = 4_z; angleCount <= maxFFTLength / 4; angleCount *= 2)
    {
      const auto cosOffset = angleCount * 2;
      const auto sinOffset = cosOffset + angleCount;
      for (auto i = 0_z; i < angleCount; i++)
      {
        constexpr auto NegTwoPi = -6.283185307179586;
        auto angle = NegTwoPi * double(DoRealFFTIndexReverse(i)) / double(angleCount * 4);
        table[cosOffset + i] = F(std::cos(angle));
        table[sinOffset + i] = F(std::sin(angle));
      }
    }
  }

  template <std::floating_point F>
  const F *AllocateAndInitializeComplexTable(size_t maxFFTLength)
  {
    F *t = fft_20xx::allocate_type_simd_aligned<F>(GetComplexTableElementCount(maxFFTLength) + TableDataOffset<F>);
    memcpy(t, &maxFFTLength, sizeof(size_t));
    InitializeComplexTable(t + TableDataOffset<F>, maxFFTLength);
    return t;
  }

  template <std::floating_point F>
  const F *AllocateAndInitializeRealTable(size_t maxFFTLength)
  {
    F *t = fft_20xx::allocate_type_simd_aligned<F>(GetRealTableElementCount(maxFFTLength) + TableDataOffset<F>);
    memcpy(t, &maxFFTLength, sizeof(size_t));
    InitializeRealTable(t + TableDataOffset<F>, maxFFTLength);
    return t;
  }
}

extern "C"
{
  namespace fft_20xx::c
  {
    size_t get_fft_table_cplx_f32_byte_count(size_t maxFFTLength)
      { return (GetComplexTableElementCount(maxFFTLength) + 1) * sizeof(float) + TableDataOffset<float>; }

    size_t get_fft_table_cplx_f64_byte_count(size_t maxFFTLength)
      { return (GetComplexTableElementCount(maxFFTLength) + 1) * sizeof(double) + TableDataOffset<double>; }

    size_t get_fft_table_real_f32_byte_count(size_t maxFFTLength)
      { return (GetRealTableElementCount(maxFFTLength) + 1) * sizeof(float) + TableDataOffset<float>; }

    size_t get_fft_table_real_f64_byte_count(size_t maxFFTLength)
      { return (GetRealTableElementCount(maxFFTLength) + 1) * sizeof(double) + TableDataOffset<double>; }

    const fft_table_cplx_f32 *allocate_fft_table_cplx_f32(size_t maxFFTLength)
      { return reinterpret_cast<const fft_table_cplx_f32 *>(AllocateAndInitializeComplexTable<float>(maxFFTLength)); }

    void free_fft_table_cplx_f32(const fft_table_cplx_f32 *table)
      { free_simd_aligned(const_cast<fft_table_cplx_f32 *>(table)); }

    const fft_table_cplx_f64 *allocate_fft_table_cplx_f64(size_t maxFFTLength)
      { return reinterpret_cast<const fft_table_cplx_f64 *>(AllocateAndInitializeComplexTable<double>(maxFFTLength)); }

    void free_fft_table_cplx_f64(const fft_table_cplx_f64 *table)
      { free_simd_aligned(const_cast<fft_table_cplx_f64 *>(table)); }


    const fft_table_real_f32 *allocate_fft_table_real_f32(size_t maxFFTLength)
      { return reinterpret_cast<const fft_table_real_f32 *>(AllocateAndInitializeRealTable<float>(maxFFTLength)); }

    void free_fft_table_real_f32(const fft_table_real_f32 *table)
      { free_simd_aligned(const_cast<fft_table_real_f32 *>(table)); }

    const fft_table_real_f64 *allocate_fft_table_real_f64(size_t maxFFTLength)
      { return reinterpret_cast<const fft_table_real_f64 *>(AllocateAndInitializeRealTable<double>(maxFFTLength)); }

    void free_fft_table_real_f64(const fft_table_real_f64 *table)
      { free_simd_aligned(const_cast<fft_table_real_f64 *>(table)); }
  }
}
