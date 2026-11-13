#ifndef FFT_20XX_HPP_INCLUDE_GUARD
  #define FFT_20XX_HPP_INCLUDE_GUARD

  #include <cassert>
  #include <complex>
  #include <utility>

  #define FFT_20XX_INCLUDING_FROM_CPP
  #include "fft_20xx/fft_20xx.h"
  #undef FFT_20XX_INCLUDING_FROM_CPP

  #if __has_include(<concepts>)
    #include <concepts>
  #endif

  #ifdef __cpp_concepts
    #define FFT_20XX_FLOAT_TYPE std::floating_point
  #else
    #define FFT_20XX_FLOAT_TYPE typename
  #endif
  #if defined(_MSC_VER)
    #pragma warning (push)
    #pragma warning (disable: 4514) // unreferenced inline function has been removed
    #pragma warning (disable: 4820) // N bytes padding added after data member '<name>'
  #endif

  namespace fft_20xx
  {
    #ifdef __cpp_using_enum
      // If we can, make this more strongly-typed (and then "using" it so that fft_20xx::cplx and ::real are easy to
      // use)
      enum class type
      {
        cplx,
        real,
      };

      using enum type;
    #else
      // Use a basic enum for this instead
      enum type
      {
        cplx,
        real,
      };
    #endif


    template <type T, FFT_20XX_FLOAT_TYPE F> class fft_table;

    // float array FFT functions

    template <type T, FFT_20XX_FLOAT_TYPE F> void fft(const fft_table<T, F> &, const F *input, F *output, size_t length);
    template <type T, FFT_20XX_FLOAT_TYPE F> void fft(const fft_table<T, F> &, F *data, size_t length);
    template <type T, FFT_20XX_FLOAT_TYPE F> void ifft(const fft_table<T, F> &, const F *input, F *output, size_t length);
    template <type T, FFT_20XX_FLOAT_TYPE F> void ifft(const fft_table<T, F> &, F *data, size_t length);

    // Float array FFT functions with no specified table

    template <type T, FFT_20XX_FLOAT_TYPE F> void fft(const F *input, F *output, size_t length);
    template <type T, FFT_20XX_FLOAT_TYPE F> void fft(F *data, size_t length);
    template <type T, FFT_20XX_FLOAT_TYPE F> void ifft(const F *input, F *output, size_t length);
    template <type T, FFT_20XX_FLOAT_TYPE F> void ifft(F *data, size_t length);


    // Complex FFTs can also take std::complex inputs/output arrays

    template <FFT_20XX_FLOAT_TYPE F>
    void fft(const fft_table<type::cplx, F> &, const std::complex<F> *input, std::complex<F> *output, size_t length);

    template <FFT_20XX_FLOAT_TYPE F>
    void fft(const fft_table<type::cplx, F> &, std::complex<F> *data, size_t length);

    template <FFT_20XX_FLOAT_TYPE F>
    void ifft(const fft_table<type::cplx, F> &, const std::complex<F> *input, std::complex<F> *output, size_t length);

    template <FFT_20XX_FLOAT_TYPE F>
    void ifft(const fft_table<type::cplx, F> &, std::complex<F> *data, size_t length);

    // Complex FFTs that take std::complex, with no specified table

    template <FFT_20XX_FLOAT_TYPE F>
    void fft(const std::complex<F> *input, std::complex<F> *output, size_t length);

    template <FFT_20XX_FLOAT_TYPE F>
    void fft(std::complex<F> *data, size_t length);

    template <FFT_20XX_FLOAT_TYPE F>
    void ifft(const std::complex<F> *input, std::complex<F> *output, size_t length);

    template <FFT_20XX_FLOAT_TYPE F>
    void ifft(std::complex<F> *data, size_t length);


    // The fft_table class wraps the c table, with optional ownership (and proper cleanup on destruction)
    template <type T, FFT_20XX_FLOAT_TYPE F> class fft_table
    {
    public:
      using c_table_t = std::conditional_t<
        (T == type::cplx),
        std::conditional_t<std::is_same_v<F, float>, c::fft_table_cplx_f32, c::fft_table_cplx_f64>,
        std::conditional_t<std::is_same_v<F, float>, c::fft_table_real_f32, c::fft_table_real_f64>>;

      static size_t get_required_alignment();
      static size_t get_byte_count_aligned(size_t max_fft_length);
      static size_t get_byte_count_unaligned(size_t max_fft_length);

      // Allocate a new complex FFT table for FFTs up to the given maximum length.
      fft_table(size_t max_length);

      // Use the given block of memory (with its given size) to create a new table that supports the given max length.
      fft_table(size_t max_length, void *ptr, size_t byte_count);

      // These tables cannot be copied
      fft_table(const fft_table &) = delete;
      fft_table &operator=(const fft_table &) = delete;

      // They are move-compatible, however
      fft_table(fft_table &&other) noexcept;
      fft_table &operator=(fft_table &&other) noexcept;
      ~fft_table();

      // Get a pointer to the C version of the table
      const c_table_t *c_table() const
        { assert(_table != nullptr); return _table; }

      // Implicit conversion to the C version of the table.
      operator const c_table_t *() const
        { return c_table(); }

    private:
      void release();
      const c_table_t *_table = nullptr;
      bool _owns_memory = false;
    };



    // Inline function definitions follow



    template <type T, FFT_20XX_FLOAT_TYPE F> void fft(const fft_table<T, F> &table, const F *input, F *output, size_t length)
    {
      static_assert(std::is_floating_point_v<F>, "Expected a floating point type for the data parameters");

      if constexpr (T == type::cplx)
      {
        if constexpr (std::same_as<F, float>)
          return c::fft_cplx_f32(table.c_table(), input, output, length);
        else
          return c::fft_cplx_f64(table.c_table(), input, output, length);
      }
      else
      {
        if constexpr (std::same_as<F, float>)
          return c::fft_real_f32(table.c_table(), input, output, length);
        else
          return c::fft_real_f64(table.c_table(), input, output, length);
      }
    }

    template <type T, FFT_20XX_FLOAT_TYPE F> void ifft(const fft_table<T, F> &table, const F *input, F *output, size_t length)
    {
      static_assert(std::is_floating_point_v<F>, "Expected a floating point type for the data parameters");

      if constexpr (T == type::cplx)
      {
        if constexpr (std::same_as<F, float>)
          return c::ifft_cplx_f32(table.c_table(), input, output, length);
        else
          return c::ifft_cplx_f64(table.c_table(), input, output, length);
      }
      else
      {
        if constexpr (std::same_as<F, float>)
          return c::ifft_real_f32(table.c_table(), input, output, length);
        else
          return c::ifft_real_f64(table.c_table(), input, output, length);
      }
    }

    template <type T, FFT_20XX_FLOAT_TYPE F> void fft(const fft_table<T, F> &table, F *data, size_t length)
      { fft(table, data, data, length);}

    template <type T, FFT_20XX_FLOAT_TYPE F> void ifft(const fft_table<T, F> &table, F *data, size_t length)
      { ifft(table, data, data, length);}

    template <type T, FFT_20XX_FLOAT_TYPE F> void fft(const F *input, F *output, size_t length)
    {
      static_assert(std::is_floating_point_v<F>, "Expected a floating point type for the data parameters");

      if constexpr (T == type::cplx)
      {
        if constexpr (std::same_as<F, float>)
          return c::fft_cplx_f32(nullptr, input, output, length);
        else
          return c::fft_cplx_f64(nullptr, input, output, length);
      }
      else
      {
        if constexpr (std::same_as<F, float>)
          return c::fft_real_f32(nullptr, input, output, length);
        else
          return c::fft_real_f64(nullptr, input, output, length);
      }
    }

    template <type T, FFT_20XX_FLOAT_TYPE F> void ifft(const F *input, F *output, size_t length)
    {
      static_assert(std::is_floating_point_v<F>, "Expected a floating point type for the data parameters");

      if constexpr (T == type::cplx)
      {
        if constexpr (std::same_as<F, float>)
          return c::ifft_cplx_f32(nullptr, input, output, length);
        else
          return c::ifft_cplx_f64(nullptr, input, output, length);
      }
      else
      {
        if constexpr (std::same_as<F, float>)
          return c::ifft_real_f32(nullptr, input, output, length);
        else
          return c::ifft_real_f64(nullptr, input, output, length);
      }
    }

    template <type T, FFT_20XX_FLOAT_TYPE F> void fft(F *data, size_t length)
      { fft(data, data, length);}

    template <type T, FFT_20XX_FLOAT_TYPE F> void ifft(F *data, size_t length)
      { ifft(data, data, length);}

    template <FFT_20XX_FLOAT_TYPE F>
    void fft(const fft_table<type::cplx, F> &table, const std::complex<F> *input, std::complex<F> *output, size_t length)
      { fft(table, reinterpret_cast<const F *>(input), reinterpret_cast<F *>(output), length); }

    template <FFT_20XX_FLOAT_TYPE F>
    void fft(const fft_table<type::cplx, F> &table, std::complex<F> *data, size_t length)
      { fft(table, reinterpret_cast<F *>(data), length); }

    template <FFT_20XX_FLOAT_TYPE F>
    void ifft(const fft_table<type::cplx, F> &table, const std::complex<F> *input, std::complex<F> *output, size_t length)
      { ifft(table, reinterpret_cast<const F *>(input), reinterpret_cast<F *>(output), length); }

    template <FFT_20XX_FLOAT_TYPE F>
    void ifft(const fft_table<type::cplx, F> &table, std::complex<F> *data, size_t length)
      { ifft(table, reinterpret_cast<F *>(data), length); }

    template <FFT_20XX_FLOAT_TYPE F>
    void fft(const std::complex<F> *input, std::complex<F> *output, size_t length)
      { fft(reinterpret_cast<const F *>(input), reinterpret_cast<F *>(output), length); }

    template <FFT_20XX_FLOAT_TYPE F>
    void fft(std::complex<F> *data, size_t length)
      { fft(reinterpret_cast<F *>(data), length); }

    template <FFT_20XX_FLOAT_TYPE F>
    void ifft(const std::complex<F> *input, std::complex<F> *output, size_t length)
      { ifft(reinterpret_cast<const F *>(input), reinterpret_cast<F *>(output), length); }

    template <FFT_20XX_FLOAT_TYPE F>
    void ifft(std::complex<F> *data, size_t length)
      { ifft(reinterpret_cast<F *>(data), length); }

    template <type T, std::floating_point F> size_t fft_table<T, F>::get_byte_count_aligned(size_t max_fft_length)
    {
      if constexpr (T == type::cplx)
      {
        if constexpr (std::is_same_v<F, float>)
          return c::get_fft_table_cplx_f32_byte_count_aligned(max_fft_length);
        else
          return c::get_fft_table_cplx_f64_byte_count_aligned(max_fft_length);
      }
      else
      {
        if constexpr (std::is_same_v<F, float>)
          return c::get_fft_table_real_f32_byte_count_aligned(max_fft_length);
        else
          return c::get_fft_table_real_f64_byte_count_aligned(max_fft_length);
      }
    }

    template <type T, std::floating_point F> size_t fft_table<T, F>::get_required_alignment()
      { return c::get_simd_alignment(); }

    template <type T, std::floating_point F> size_t fft_table<T, F>::get_byte_count_unaligned(size_t max_fft_length)
    {
      if constexpr (T == type::cplx)
      {
        if constexpr (std::is_same_v<F, float>)
          return c::get_fft_table_cplx_f32_byte_count_unaligned(max_fft_length);
        else
          return c::get_fft_table_cplx_f64_byte_count_unaligned(max_fft_length);
      }
      else
      {
        if constexpr (std::is_same_v<F, float>)
          return c::get_fft_table_real_f32_byte_count_unaligned(max_fft_length);
        else
          return c::get_fft_table_real_f64_byte_count_unaligned(max_fft_length);
      }
    }

    // Allocate a new complex FFT table for FFTs up to the given maximum length.
    template <type T, FFT_20XX_FLOAT_TYPE F>
    fft_table<T, F>::fft_table(size_t max_length)
    {
      static_assert(std::is_floating_point_v<F>);

      if constexpr (T == type::cplx)
      {
        if constexpr (std::is_same_v<F, float>)
          _table = c::allocate_fft_table_cplx_f32(max_length);
        else
          _table = c::allocate_fft_table_cplx_f64(max_length);
      }
      else
      {
        if constexpr (std::is_same_v<F, float>)
          _table = c::allocate_fft_table_real_f32(max_length);
        else
          _table = c::allocate_fft_table_real_f64(max_length);
      }
    }

    // Use the given block of memory (with its given size) to create a new
    template <type T, FFT_20XX_FLOAT_TYPE F>
    fft_table<T, F>::fft_table(size_t max_length, void *ptr, size_t byte_count)
    {
      auto *t = [&]()
        {
          if constexpr (T == type::cplx)
          {
            if constexpr (std::same_as<F, float>)
              return c::make_table_from_memory_cplx_f32(max_length, ptr, byte_count);
            else
              return c::make_table_from_memory_cplx_f64(max_length, ptr, byte_count);
          }
          else
          {
            if constexpr (std::same_as<F, float>)
              return c::make_table_from_memory_real_f32(max_length, ptr, byte_count);
            else
              return c::make_table_from_memory_real_f64(max_length, ptr, byte_count);
          }
        }();

      _table = t;
    }

    template <type T, FFT_20XX_FLOAT_TYPE F>
    fft_table<T, F>::fft_table(fft_table &&other) noexcept
      : _table{std::exchange(other._table, nullptr)}
      , _owns_memory{std::exchange(other._owns_memory, false)}
      { }

    template <type T, FFT_20XX_FLOAT_TYPE F>
    fft_table<T, F> &fft_table<T, F>::operator=(fft_table &&other) noexcept
    {
      if (&other != this)
      {
        release();
        _table = std::exchange(other._table, nullptr);
        _owns_memory = std::exchange(other._owns_memory, false);
      }

      return *this;
    }

    template <type T, FFT_20XX_FLOAT_TYPE F>
    fft_table<T, F>::~fft_table() { release(); }

    template <type T, FFT_20XX_FLOAT_TYPE F>
    void fft_table<T, F>::release()
    {
      if (_owns_memory)
      {
        if constexpr (T == type::cplx)
        {
          if constexpr (std::is_same_v<F, float>)
            c::free_fft_table_cplx_f32(_table);
          else
            c::free_fft_table_cplx_f64(_table);
        }
        else
        {
          if constexpr (std::is_same_v<F, float>)
            c::free_fft_table_real_f32(_table);
          else
            c::free_fft_table_real_f64(_table);
        }
      }
    }
  }

  #undef FFT_20XX_FLOAT_TYPE

  #if defined(_MSC_VER)
    #pragma warning (pop)
  #endif
#endif