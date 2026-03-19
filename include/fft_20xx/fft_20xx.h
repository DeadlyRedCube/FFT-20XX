#ifndef FFT_20XX_INCLUDE_GUARD
  #define FFT_20XX_INCLUDE_GUARD


  #if defined(_MSC_VER)
    #pragma warning (push)
    #pragma warning (disable: 4514) // unreferenced inline function has been removed
  #endif

  #include <inttypes.h>
  #include <stddef.h>

  // C's "float/double _Complex" and C++'s "std::complex<float/double>" are guaranteed to have the same layout, so
  //  use the correct language-specific version.
  #ifdef __cplusplus
    #include <complex>
    #include <utility>


    extern "C"
    {
      namespace fft_20xx::c
      {
        // Use C++ complex type as the native type (it is compatible with the C-style complex types)
        using fft_20xx_cplx_f32 = std::complex<float>;
        using fft_20xx_cplx_f64 = std::complex<double>;

  #else
    #if defined (__STDC_VERSION__)
      #if __STDC_VERSION__ >= 199901L
        // C99 has built-in complex number support, so use that

        #if defined(_MSC_VER) && !defined(__clang__)
          typedef _Fcomplex fft_20xx_cplx_f32;
          typedef _Dcomplex fft_20xx_cplx_f64;
        #else
          typedef float _Complex fft_20xx_cplx_f32;
          typedef double _Complex fft_20xx_cplx_f64;
        #endif

        #define FFT_20XX_HAS_C99_COMPLEX // define this temporarily so we know not to make a custom type
      #endif
    #endif

    #if !defined(FFT_20XX_HAS_C99_COMPLEX)
      // No built-in complex type was detected so use a simple custom struct
      typedef struct fft_20xx_cplx_f32
      {
        float r;
        float i;
      } fft_20xx_cplx_f32;

      typedef struct fft_20xx_cplx_f64
      {
        double r;
        double i;
      } fft_20xx_cplx_f64;
    #else
      #undef FFT_20XX_HAS_C99_COMPLEX
    #endif
  #endif

    typedef struct fft_table_f32 fft_table_f32;
    typedef struct fft_table_f64 fft_table_f64;

    const fft_table_f32 *allocate_fft_table_f32(size_t length);
    void free_fft_table_f32(const fft_table_f32 *table);

    const fft_table_f64 *allocate_fft_table_f64(size_t max_length);
    void free_fft_table_f64(const fft_table_f64 *table);

    // TODO:
    // size_t get_required_table_byte_count_f32(size_t max_length);
    // const fft_table_f32 *initialize_table_f32(void *memory, size_t max_length);
    // size_t get_required_table_byte_count_f64(size_t max_length);
    // const fft_table_f64 *initialize_table_f64(void *memory, size_t max_length);

    void fft_cplx_f32(const fft_table_f32 *table, const fft_20xx_cplx_f32 *input, fft_20xx_cplx_f32 *output, size_t length);
    void fft_cplx_f64(const fft_table_f64 *table, const fft_20xx_cplx_f64 *input, fft_20xx_cplx_f64 *output, size_t length);

    void fft_cplx_interleaved_f32(const fft_table_f32 *table, const float *input, float *output, size_t length);
    void fft_cplx_interleaved_f64(const fft_table_f64 *table, const double *input, double *output, size_t length);

    void ifft_cplx_f32(const fft_table_f32 *table, const fft_20xx_cplx_f32 *input, fft_20xx_cplx_f32 *output, size_t length);
    void ifft_cplx_f64(const fft_table_f64 *table, const fft_20xx_cplx_f64 *input, fft_20xx_cplx_f64 *output, size_t length);

    void ifft_cplx_interleaved_f32(const fft_table_f32 *table, const float *input, float *output, size_t length);
    void ifft_cplx_interleaved_f64(const fft_table_f64 *table, const double *input, double *output, size_t length);

    void fft_real_f32(const fft_table_f32 *table, const float *input, float *output, size_t length);
    void fft_real_f64(const fft_table_f64 *table, const double *input, double *output, size_t length);

    void ifft_real_f32(const fft_table_f32 *table, const float *input, float *output, size_t length);
    void ifft_real_f64(const fft_table_f64 *table, const double *input, double *output, size_t length);
  #ifdef __cplusplus
      } // namespace
    } // extern "C"


    namespace fft_20xx
    {
      template <typename F>
      class fft_table
      {
      public:
        using c_table_t = std::conditional_t<std::is_same_v<F, float>, c::fft_table_f32, c::fft_table_f64>;

        fft_table(size_t max_length)
        {
          static_assert(std::is_floating_point_v<F>);

          if constexpr (std::is_same_v<F, float>)
            { _table = c::allocate_fft_table_f32(max_length); }
          else
            { _table = c::allocate_fft_table_f64(max_length); }
        }

        // !!! todo: non-memory-owning version

        ~fft_table()
        {
          if constexpr (std::is_same_v<F, float>)
            { c::free_fft_table_f32(_table); }
          else
            { c::free_fft_table_f64(_table); }
        }

        fft_table(const fft_table &) = delete;
        fft_table &operator=(const fft_table &) = delete;
        fft_table(fft_table &&other) noexcept
          : _table{std::exchange(other._table, nullptr)}
          { }

        fft_table &operator=(fft_table &&other)
        {
          if (&other != this)
          {
            if constexpr (std::is_same_v<F, float>)
              { c::free_fft_table_f32(_table); }
            else
              { c::free_fft_table_f64(_table); }
            _table = std::exchange(other._table, nullptr);
          }

          return *this;
        }

        const c_table_t *c_table() const
          { return _table; }

        operator const c_table_t *() const
          { return _table; }

      private:
        const c_table_t *_table = nullptr;
      };

      inline void fft_cplx(const fft_table<float> &table, const std::complex<float> *input, std::complex<float> *output, size_t length)
        { return c::fft_cplx_f32(table.c_table(), input, output, length); }
      inline void fft_cplx(const fft_table<double> &table, const std::complex<double> *input, std::complex<double> *output, size_t length)
        { return c::fft_cplx_f64(table.c_table(), input, output, length); }

      inline void fft_cplx(const fft_table<float> &table, const float *input, float *output, size_t length)
        { return c::fft_cplx_interleaved_f32(table.c_table(), input, output, length); }
      inline void fft_cplx(const fft_table<double> &table, const double *input, double *output, size_t length)
        { return c::fft_cplx_interleaved_f64(table.c_table(), input, output, length); }

      inline void fft_cplx(const fft_table<float> &table, std::complex<float> *data, size_t length)
        { return c::fft_cplx_f32(table.c_table(), data, data, length); }
      inline void fft_cplx(const fft_table<double> &table, std::complex<double> *data, size_t length)
        { return c::fft_cplx_f64(table.c_table(), data, data, length); }

      inline void fft_cplx(const fft_table<float> &table, float *data, size_t length)
        { return c::fft_cplx_interleaved_f32(table.c_table(), data, data, length); }
      inline void fft_cplx(const fft_table<double> &table, double *data, size_t length)
        { return c::fft_cplx_interleaved_f64(table.c_table(), data, data, length); }

      inline void ifft_cplx(const fft_table<float> &table, const std::complex<float> *input, std::complex<float> *output, size_t length)
        { return c::ifft_cplx_f32(table.c_table(), input, output, length); }
      inline void ifft_cplx(const fft_table<double> &table, const std::complex<double> *input, std::complex<double> *output, size_t length)
        { return c::ifft_cplx_f64(table.c_table(), input, output, length); }

      inline void ifft_cplx(const fft_table<float> &table, const float *input, float *output, size_t length)
        { return c::ifft_cplx_interleaved_f32(table.c_table(), input, output, length); }
      inline void ifft_cplx(const fft_table<double> &table, const double *input, double *output, size_t length)
        { return c::ifft_cplx_interleaved_f64(table.c_table(), input, output, length); }

      inline void ifft_cplx(const fft_table<float> &table, std::complex<float> *data, size_t length)
        { return c::ifft_cplx_f32(table.c_table(), data, data, length); }
      inline void ifft_cplx(const fft_table<double> &table, std::complex<double> *data, size_t length)
        { return c::ifft_cplx_f64(table.c_table(), data, data, length); }

      inline void ifft_cplx(const fft_table<float> &table, float *data, size_t length)
        { return c::ifft_cplx_interleaved_f32(table.c_table(), data, data, length); }
      inline void ifft_cplx(const fft_table<double> &table, double *data, size_t length)
        { return c::ifft_cplx_interleaved_f64(table.c_table(), data, data, length); }



      inline void fft_real(const fft_table<float> &table, const float *input, float *output, size_t length)
        { return c::fft_real_f32(table.c_table(), input, output, length); }
      inline void fft_real(const fft_table<double> &table, const double *input, double *output, size_t length)
        { return c::fft_real_f64(table.c_table(), input, output, length); }

      inline void fft_real(const fft_table<float> &table, float *data, size_t length)
        { return c::fft_real_f32(table.c_table(), data, data, length); }
      inline void fft_real(const fft_table<double> &table, double *data, size_t length)
        { return c::fft_real_f64(table.c_table(), data, data, length); }



      inline void ifft_real(const fft_table<float> &table, const float *input, float *output, size_t length)
        { return c::ifft_real_f32(table.c_table(), input, output, length); }
      inline void ifft_real(const fft_table<double> &table, const double *input, double *output, size_t length)
        { return c::ifft_real_f64(table.c_table(), input, output, length); }

      inline void ifft_real(const fft_table<float> &table, float *data, size_t length)
        { return c::ifft_real_f32(table.c_table(), data, data, length); }
      inline void ifft_real(const fft_table<double> &table, double *data, size_t length)
        { return c::ifft_real_f64(table.c_table(), data, data, length); }
    }
  #endif

  #if defined(_MSC_VER)
    #pragma warning (pop)
  #endif
#endif