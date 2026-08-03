#include "fft_20xx_table.h"

#include "fft_20xx/fft_20xx_util.hpp"
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
      for (auto i = 0_z; i < maxFFTLength * 3 / 4; i++)
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
      for (auto i = 0_z; i < realCount * 3 / 2; i++)
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
  size_t DoRealFFTIndexReverse(size_t i)
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
    memset(t, 0, TableDataOffset<F> * sizeof(F));
    memcpy(t, &maxFFTLength, sizeof(size_t));
    InitializeComplexTable(t + TableDataOffset<F>, maxFFTLength);
    return t;
  }

  template <std::floating_point F>
  const F *AllocateAndInitializeRealTable(size_t maxFFTLength)
  {
    F *t = fft_20xx::allocate_type_simd_aligned<F>(GetRealTableElementCount(maxFFTLength) + TableDataOffset<F>);
    memset(t, 0, TableDataOffset<F> * sizeof(F));
    memcpy(t, &maxFFTLength, sizeof(size_t));
    InitializeRealTable(t + TableDataOffset<F>, maxFFTLength);
    return t;
  }
}

namespace fft_20xx
{
  template <typename F>
  fft_table_cplx<F>::fft_table_cplx(size_t max_length)
  {
    static_assert(std::is_floating_point_v<F>);

    if constexpr (std::is_same_v<F, float>)
      { _table = c::allocate_fft_table_cplx_f32(max_length); }
    else
      { _table = c::allocate_fft_table_cplx_f64(max_length); }
  }

  template <typename F>
  fft_table_cplx<F>::fft_table_cplx(size_t max_length, void *ptr, size_t byte_count)
  : _owns_memory(false)
  {
    auto *t = [&]()
      {
        if constexpr (std::same_as<F, float>)
          { return c::make_table_from_memory_cplx_f32(max_length, ptr, byte_count); }
        else
          { return c::make_table_from_memory_cplx_f64(max_length, ptr, byte_count); }
      }();

    _table = t;
  }

  template <typename F>
  fft_table_cplx<F>::~fft_table_cplx() { release(); }

  template <typename F>
  void fft_table_cplx<F>::release()
  {
    if (_owns_memory)
    {
      if constexpr (std::is_same_v<F, float>)
        { c::free_fft_table_cplx_f32(_table); }
      else
        { c::free_fft_table_cplx_f64(_table); }
    }
  }

  template <typename F>
  fft_table_cplx<F>::fft_table_cplx(fft_table_cplx &&other) noexcept
    : _table{std::exchange(other._table, nullptr)}
    { }

  template <typename F>
  fft_table_cplx<F> &fft_table_cplx<F>::operator=(fft_table_cplx &&other) noexcept
  {
    if (&other != this)
    {
      release();
      _table = std::exchange(other._table, nullptr);
    }

    return *this;
  }

  template class fft_table_cplx<float>;
  template class fft_table_cplx<double>;

  template <typename F>
  fft_table_real<F>::fft_table_real(size_t max_length)
  {
    static_assert(std::is_floating_point_v<F>);

    if constexpr (std::is_same_v<F, float>)
      { _table = c::allocate_fft_table_real_f32(max_length); }
    else
      { _table = c::allocate_fft_table_real_f64(max_length); }
  }

  template <typename F>
  fft_table_real<F>::fft_table_real(size_t max_length, void *ptr, size_t byte_count)
  : _owns_memory(false)
  {
    auto *t = [&]()
      {
        if constexpr (std::same_as<F, float>)
          { return c::make_table_from_memory_real_f32(max_length, ptr, byte_count); }
        else
          { return c::make_table_from_memory_real_f64(max_length, ptr, byte_count); }
      }();

    _table = t;
  }

  template <typename F>
  fft_table_real<F>::~fft_table_real() { release(); }

  template <typename F>
  void fft_table_real<F>::release()
  {
    if (_owns_memory)
    {
      if constexpr (std::is_same_v<F, float>)
        { c::free_fft_table_real_f32(_table); }
      else
        { c::free_fft_table_real_f64(_table); }
    }
  }

  template <typename F>
  fft_table_real<F>::fft_table_real(fft_table_real &&other) noexcept
    : _table{std::exchange(other._table, nullptr)}
    { }

  template <typename F>
  fft_table_real<F> &fft_table_real<F>::operator=(fft_table_real &&other) noexcept
  {
    if (&other != this)
    {
      release();
      _table = std::exchange(other._table, nullptr);
    }

    return *this;
  }

  template class fft_table_real<float>;
  template class fft_table_real<double>;
}

extern "C"
{
  namespace fft_20xx::c
  {
    size_t get_fft_table_cplx_f32_byte_count_aligned(size_t maxFFTLength)
      { return GetComplexTableElementCount(maxFFTLength) * sizeof(float) + TableDataOffset<float>; }

    size_t get_fft_table_cplx_f64_byte_count_aligned(size_t maxFFTLength)
      { return GetComplexTableElementCount(maxFFTLength) * sizeof(double) + TableDataOffset<double>; }

    size_t get_fft_table_real_f32_byte_count_aligned(size_t maxFFTLength)
      { return GetRealTableElementCount(maxFFTLength) * sizeof(float) + TableDataOffset<float>; }

    size_t get_fft_table_real_f64_byte_count_aligned(size_t maxFFTLength)
      { return GetRealTableElementCount(maxFFTLength) * sizeof(double) + TableDataOffset<double>; }




    size_t get_fft_table_cplx_f32_byte_count_unaligned(size_t maxFFTLength)
      { return get_fft_table_cplx_f32_byte_count_aligned(maxFFTLength) + SIMDAlignment; }

    size_t get_fft_table_cplx_f64_byte_count_unaligned(size_t maxFFTLength)
      { return get_fft_table_cplx_f64_byte_count_aligned(maxFFTLength) + SIMDAlignment; }

    size_t get_fft_table_real_f32_byte_count_unaligned(size_t maxFFTLength)
      { return get_fft_table_real_f32_byte_count_aligned(maxFFTLength) + SIMDAlignment; }

    size_t get_fft_table_real_f64_byte_count_unaligned(size_t maxFFTLength)
      { return get_fft_table_real_f64_byte_count_aligned(maxFFTLength) + SIMDAlignment; }


    const fft_table_cplx_f32 *make_table_from_memory_cplx_f32(size_t maxFFTLength, void *memory, size_t byteCount)
    {
      auto alignedRequiredByteCount = get_fft_table_cplx_f32_byte_count_aligned(maxFFTLength);
      void *tableStart = SIMDAlignPtr(memory);
      assert(PtrByteOffset(memory, tableStart) >= 0);
      byteCount -= size_t(tableStart);
      assert(byteCount >= alignedRequiredByteCount);
      if (byteCount < alignedRequiredByteCount)
        { return nullptr; }

      auto t = static_cast<float *>(tableStart);
      memset(t, 0, TableDataOffset<float> * sizeof(float));
      memcpy(t, &maxFFTLength, sizeof(size_t));
      InitializeComplexTable(t + TableDataOffset<float>, maxFFTLength);
      return reinterpret_cast<const fft_table_cplx_f32 *>(t);
    }

    const fft_table_cplx_f64 *make_table_from_memory_cplx_f64(size_t maxFFTLength, void *memory, size_t byteCount)
    {
      auto alignedRequiredByteCount = get_fft_table_cplx_f64_byte_count_aligned(maxFFTLength);
      void *tableStart = SIMDAlignPtr(memory);
      assert(PtrByteOffset(memory, tableStart) >= 0);
      byteCount -= size_t(tableStart);
      assert(byteCount >= alignedRequiredByteCount);
      if (byteCount < alignedRequiredByteCount)
        { return nullptr; }

      auto t = static_cast<double *>(tableStart);
      memset(t, 0, TableDataOffset<double> * sizeof(double));
      memcpy(t, &maxFFTLength, sizeof(size_t));
      InitializeComplexTable(t + TableDataOffset<double>, maxFFTLength);
      return reinterpret_cast<const fft_table_cplx_f64 *>(t);
    }



    const fft_table_real_f32 *make_table_from_memory_real_f32(size_t maxFFTLength, void *memory, size_t byteCount)
    {
      auto alignedRequiredByteCount = get_fft_table_real_f32_byte_count_aligned(maxFFTLength);
      void *tableStart = SIMDAlignPtr(memory);
      assert(PtrByteOffset(memory, tableStart) >= 0);
      byteCount -= size_t(tableStart);
      assert(byteCount >= alignedRequiredByteCount);
      if (byteCount < alignedRequiredByteCount)
        { return nullptr; }

      auto t = static_cast<float *>(tableStart);
      memset(t, 0, TableDataOffset<float> * sizeof(float));
      memcpy(t, &maxFFTLength, sizeof(size_t));
      InitializeRealTable(t + TableDataOffset<float>, maxFFTLength);
      return reinterpret_cast<const fft_table_real_f32 *>(t);
    }

    const fft_table_real_f64 *make_table_from_memory_real_f64(size_t maxFFTLength, void *memory, size_t byteCount)
    {
      auto alignedRequiredByteCount = get_fft_table_real_f64_byte_count_aligned(maxFFTLength);
      void *tableStart = SIMDAlignPtr(memory);
      assert(PtrByteOffset(memory, tableStart) >= 0);
      byteCount -= size_t(tableStart);
      assert(byteCount >= alignedRequiredByteCount);
      if (byteCount < alignedRequiredByteCount)
        { return nullptr; }

      auto t = static_cast<double *>(tableStart);
      memset(t, 0, TableDataOffset<double> * sizeof(double));
      memcpy(t, &maxFFTLength, sizeof(size_t));
      InitializeRealTable(t + TableDataOffset<double>, maxFFTLength);
      return reinterpret_cast<const fft_table_real_f64 *>(t);
    }



    const fft_table_cplx_f32 *allocate_fft_table_cplx_f32(size_t maxFFTLength)
      { return reinterpret_cast<const fft_table_cplx_f32 *>(AllocateAndInitializeComplexTable<float>(maxFFTLength)); }

    void free_fft_table_cplx_f32(const fft_table_cplx_f32 *table)
      { free_simd_aligned(table); }



    const fft_table_cplx_f64 *allocate_fft_table_cplx_f64(size_t maxFFTLength)
      { return reinterpret_cast<const fft_table_cplx_f64 *>(AllocateAndInitializeComplexTable<double>(maxFFTLength)); }

    void free_fft_table_cplx_f64(const fft_table_cplx_f64 *table)
      { free_simd_aligned(table); }



    const fft_table_real_f32 *allocate_fft_table_real_f32(size_t maxFFTLength)
      { return reinterpret_cast<const fft_table_real_f32 *>(AllocateAndInitializeRealTable<float>(maxFFTLength)); }

    void free_fft_table_real_f32(const fft_table_real_f32 *table)
      { free_simd_aligned(table); }



    const fft_table_real_f64 *allocate_fft_table_real_f64(size_t maxFFTLength)
      { return reinterpret_cast<const fft_table_real_f64 *>(AllocateAndInitializeRealTable<double>(maxFFTLength)); }

    void free_fft_table_real_f64(const fft_table_real_f64 *table)
      { free_simd_aligned(table); }
  }
}
