#ifdef _MSC_VER
  #pragma warning (push)
  #pragma warning (disable: 4710) // function <x> not inlined
  #pragma warning (disable: 4711) // function <x> selected for automatic inline expansion
#endif

#include <atomic>
#include <memory>
#include <mutex>

#ifdef _MSC_VER
  #pragma warning(pop)
#endif

#include "fft_20xx_table.h"

#include "fft_20xx/fft_20xx_util.hpp"
#include "fft_20xx_common.h"


namespace
{
  using namespace FFT20XX;

  constexpr auto MinTableFFTLength = 32_z;

  enum class TableType
  {
    Complex,
    Real,
  };

  template <TableType Type>
  size_t GetTableElementCount(size_t maxFFTLength)
  {
    if constexpr (Type == TableType::Complex)
    {
      return std::max(MinTableFFTLength, maxFFTLength) * 3 / 2;
    }
    else
    {
      return std::max(MinTableFFTLength, maxFFTLength);
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


  template <TableType Type, std::floating_point F>
  void InitializeTable(F *table, size_t maxFFTLength)
  {
    if constexpr (Type == TableType::Complex)
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
    else
    {
      static_assert(Type == TableType::Real);

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
  }


  template <TableType Type, std::floating_point F>
  using CTableType = std::conditional_t<
    Type == TableType::Complex,
    std::conditional_t<std::same_as<F, float>, fft_20xx::c::fft_table_cplx_f32, fft_20xx::c::fft_table_cplx_f64>,
    std::conditional_t<std::same_as<F, float>, fft_20xx::c::fft_table_real_f32, fft_20xx::c::fft_table_real_f64>>;


  template <TableType Type, std::floating_point F>
  const CTableType<Type, F> *AllocateAndInitializeTable(size_t maxFFTLength)
  {
    F *t = fft_20xx::allocate_trivial_type_simd_aligned<F>(GetTableElementCount<Type>(maxFFTLength) + TableDataOffset<F>);
    if (t == nullptr)
    {
      return nullptr;
    }

    memset(t, 0, TableDataOffset<F> * sizeof(F));
    memcpy(t, &maxFFTLength, sizeof(size_t));
    InitializeTable<Type>(t + TableDataOffset<F>, maxFFTLength);
    return reinterpret_cast<CTableType<Type, F> *>(t);
  }


  template <TableType Type, std::floating_point F>
  size_t GetTableByteCountAligned(size_t maxFFTLength)
    { return GetTableElementCount<Type>(maxFFTLength) * sizeof(F) + TableDataOffset<F>; }


  // These get an extra SIMDAlignment amount of bytes so that we can offset the pointer to get something aligned.
  //  (Technically this is allocating a byte more than necessary but, in practice, it's going to round to pages
  //  anyway, so it's unlikely to actually matter)
  template <TableType Type, std::floating_point F>
  size_t GetTableByteCountUnaligned(size_t maxFFTLength)
    { return GetTableByteCountAligned<Type, F>(maxFFTLength) + SIMDAlignment; }


  template <TableType Type, std::floating_point F>
  const CTableType<Type, F> *MakeTableFromMemory(size_t maxFFTLength, void *memory, size_t byteCount)
  {
    auto alignedRequiredByteCount = GetTableByteCountAligned<Type, F>(maxFFTLength);
    void *tableStart = SIMDAlignPtr(memory);
    assert(PtrByteOffset(memory, tableStart) >= 0);
    byteCount -= PtrByteOffset(memory, tableStart);
    assert(byteCount >= alignedRequiredByteCount);
    if (byteCount < alignedRequiredByteCount)
      { return nullptr; }

    auto t = static_cast<F *>(tableStart);
    memset(t, 0, TableDataOffset<F> * sizeof(F));
    memcpy(t, &maxFFTLength, sizeof(size_t));
    InitializeTable<Type>(t + TableDataOffset<F>, maxFFTLength);
    return reinterpret_cast<CTableType<Type, F> *>(t);
  }


  template <TableType Type, std::floating_point F>
  struct ManagedTable
  {
    ManagedTable(size_t maxFFTLength)
      : table {AllocateAndInitializeTable<Type, F>(maxFFTLength)}
      { }

    ~ManagedTable()
    {
      fft_20xx::free_aligned(table);
    }

    const CTableType<Type, F> *table;
  };


  template <TableType Type, std::floating_point F>
  struct TableManager
  {
    static const CTableType<Type, F> *GetTable(size_t requiredFFTLength)
    {
      auto shared = s_table.load(std::memory_order_acquire);
      size_t curFFTLength = 0;
      if (shared != nullptr)
      {
        memcpy(&curFFTLength, shared->table, sizeof(size_t));
      }

      if (curFFTLength < requiredFFTLength)
      {
        std::unique_lock lock {s_mutex};
        shared = s_table.load(std::memory_order_acquire);
        if (shared != nullptr)
        {
          memcpy(&curFFTLength, shared->table, sizeof(size_t));
        }

        if (curFFTLength < requiredFFTLength)
        {
          shared = std::make_shared<ManagedTable<Type, F>>(requiredFFTLength);
          s_table.store(shared, std::memory_order_release);
        }
      }

      return shared->table;
    }

    static inline std::mutex s_mutex;
    static inline std::atomic<std::shared_ptr<ManagedTable<Type, F>>> s_table;
  };
}


namespace FFT20XX
{
  const fft_20xx::c::fft_table_cplx_f32 *GetManagedTableCplxF32(size_t maxSize)
    { return TableManager<TableType::Complex, float>::GetTable(maxSize); }

  const fft_20xx::c::fft_table_cplx_f64 *GetManagedTableCplxF64(size_t maxSize)
    { return TableManager<TableType::Complex, double>::GetTable(maxSize); }

  const fft_20xx::c::fft_table_real_f32 *GetManagedTableRealF32(size_t maxSize)
    { return TableManager<TableType::Real, float>::GetTable(maxSize); }

  const fft_20xx::c::fft_table_real_f64 *GetManagedTableRealF64(size_t maxSize)
    { return TableManager<TableType::Real, double>::GetTable(maxSize); }
}


extern "C"
{
  namespace fft_20xx::c
  {
    size_t get_fft_table_cplx_f32_byte_count_aligned(size_t maxFFTLength)
      { return GetTableByteCountAligned<TableType::Complex, float>(maxFFTLength); }

    size_t get_fft_table_cplx_f64_byte_count_aligned(size_t maxFFTLength)
      { return GetTableByteCountAligned<TableType::Complex, double>(maxFFTLength); }

    size_t get_fft_table_real_f32_byte_count_aligned(size_t maxFFTLength)
      { return GetTableByteCountAligned<TableType::Real, float>(maxFFTLength); }

    size_t get_fft_table_real_f64_byte_count_aligned(size_t maxFFTLength)
      { return GetTableByteCountAligned<TableType::Real, double>(maxFFTLength); }



    // These get an extra SIMDAlignment amount of bytes so that we can offset the pointer to get something aligned.
    //  (Technically this is allocating a byte more than necessary but, in practice, it's going to round to pages
    //  anyway, so it's unlikely to actually matter)
    size_t get_fft_table_cplx_f32_byte_count_unaligned(size_t maxFFTLength)
      { return GetTableByteCountUnaligned<TableType::Complex, float>(maxFFTLength); }

    size_t get_fft_table_cplx_f64_byte_count_unaligned(size_t maxFFTLength)
      { return GetTableByteCountUnaligned<TableType::Complex, double>(maxFFTLength); }

    size_t get_fft_table_real_f32_byte_count_unaligned(size_t maxFFTLength)
      { return GetTableByteCountUnaligned<TableType::Real, float>(maxFFTLength); }

    size_t get_fft_table_real_f64_byte_count_unaligned(size_t maxFFTLength)
      { return GetTableByteCountUnaligned<TableType::Real, double>(maxFFTLength); }


    const fft_table_cplx_f32 *make_table_from_memory_cplx_f32(size_t maxFFTLength, void *memory, size_t byteCount)
      { return MakeTableFromMemory<TableType::Complex, float>(maxFFTLength, memory, byteCount); }

    const fft_table_cplx_f64 *make_table_from_memory_cplx_f64(size_t maxFFTLength, void *memory, size_t byteCount)
      { return MakeTableFromMemory<TableType::Complex, double>(maxFFTLength, memory, byteCount); }

    const fft_table_real_f32 *make_table_from_memory_real_f32(size_t maxFFTLength, void *memory, size_t byteCount)
      { return MakeTableFromMemory<TableType::Real, float>(maxFFTLength, memory, byteCount); }

    const fft_table_real_f64 *make_table_from_memory_real_f64(size_t maxFFTLength, void *memory, size_t byteCount)
      { return MakeTableFromMemory<TableType::Real, double>(maxFFTLength, memory, byteCount); }



    const fft_table_cplx_f32 *allocate_fft_table_cplx_f32(size_t maxFFTLength)
      { return AllocateAndInitializeTable<TableType::Complex, float>(maxFFTLength); }

    void free_fft_table_cplx_f32(const fft_table_cplx_f32 *table)
      { free_aligned(table); }



    const fft_table_cplx_f64 *allocate_fft_table_cplx_f64(size_t maxFFTLength)
      { return AllocateAndInitializeTable<TableType::Complex, double>(maxFFTLength); }

    void free_fft_table_cplx_f64(const fft_table_cplx_f64 *table)
      { free_aligned(table); }



    const fft_table_real_f32 *allocate_fft_table_real_f32(size_t maxFFTLength)
      { return AllocateAndInitializeTable<TableType::Real, float>(maxFFTLength); }

    void free_fft_table_real_f32(const fft_table_real_f32 *table)
      { free_aligned(table); }



    const fft_table_real_f64 *allocate_fft_table_real_f64(size_t maxFFTLength)
      { return AllocateAndInitializeTable<TableType::Real, double>(maxFFTLength); }

    void free_fft_table_real_f64(const fft_table_real_f64 *table)
      { free_aligned(table); }
  }
}
