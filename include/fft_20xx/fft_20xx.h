#ifndef FFT_20XX_INCLUDE_GUARD
  #define FFT_20XX_INCLUDE_GUARD

  #if defined(_MSC_VER)
    #pragma warning (push)
    #pragma warning (disable: 4514) // unreferenced inline function has been removed
  #endif

  #include <inttypes.h>
  #include <stddef.h>

  #ifdef __cplusplus
    #include <complex>
    #include <utility>

    extern "C"
    {
      namespace fft_20xx::c
      {
  #endif

    typedef struct fft_table_cplx_f32 fft_table_cplx_f32;
    typedef struct fft_table_cplx_f64 fft_table_cplx_f64;
    typedef struct fft_table_real_f32 fft_table_real_f32;
    typedef struct fft_table_real_f64 fft_table_real_f64;

    const fft_table_cplx_f32 *allocate_fft_table_cplx_f32(size_t length);
    void free_fft_table_cplx_f32(const fft_table_cplx_f32 *table);

    const fft_table_cplx_f64 *allocate_fft_table_cplx_f64(size_t max_length);
    void free_fft_table_cplx_f64(const fft_table_cplx_f64 *table);

    const fft_table_real_f32 *allocate_fft_table_real_f32(size_t length);
    void free_fft_table_real_f32(const fft_table_real_f32 *table);

    const fft_table_real_f64 *allocate_fft_table_real_f64(size_t max_length);
    void free_fft_table_real_f64(const fft_table_real_f64 *table);

    // TODO:
    // size_t get_required_table_byte_count_f32(size_t max_length);
    // const fft_table_f32 *initialize_table_f32(void *memory, size_t max_length);
    // size_t get_required_table_byte_count_f64(size_t max_length);
    // const fft_table_f64 *initialize_table_f64(void *memory, size_t max_length);

    void fft_cplx_f32(const fft_table_cplx_f32 *table, const float *input, float *output, size_t length);
    void fft_cplx_f64(const fft_table_cplx_f64 *table, const double *input, double *output, size_t length);

    void ifft_cplx_f32(const fft_table_cplx_f32 *table, const float *input, float *output, size_t length);
    void ifft_cplx_f64(const fft_table_cplx_f64 *table, const double *input, double *output, size_t length);

    void fft_real_f32(const fft_table_real_f32 *table, const float *input, float *output, size_t length);
    void fft_real_f64(const fft_table_real_f64 *table, const double *input, double *output, size_t length);

    void ifft_real_f32(const fft_table_real_f32 *table, const float *input, float *output, size_t length);
    void ifft_real_f64(const fft_table_real_f64 *table, const double *input, double *output, size_t length);
  #ifdef __cplusplus
      } // namespace
    } // extern "C"


    namespace fft_20xx
    {
      template <typename F>
      class fft_table_cplx
      {
      public:
        using c_table_t = std::conditional_t<std::is_same_v<F, float>, c::fft_table_cplx_f32, c::fft_table_cplx_f64>;

        fft_table_cplx(size_t max_length)
        {
          static_assert(std::is_floating_point_v<F>);

          if constexpr (std::is_same_v<F, float>)
            { _table = c::allocate_fft_table_cplx_f32(max_length); }
          else
            { _table = c::allocate_fft_table_cplx_f64(max_length); }
        }

        // !!! todo: non-memory-owning version

        ~fft_table_cplx()
        {
          if constexpr (std::is_same_v<F, float>)
            { c::free_fft_table_cplx_f32(_table); }
          else
            { c::free_fft_table_cplx_f64(_table); }
        }

        fft_table_cplx(const fft_table_cplx &) = delete;
        fft_table_cplx &operator=(const fft_table_cplx &) = delete;
        fft_table_cplx(fft_table_cplx &&other) noexcept
          : _table{std::exchange(other._table, nullptr)}
          { }

        fft_table_cplx &operator=(fft_table_cplx &&other)
        {
          if (&other != this)
          {
            if constexpr (std::is_same_v<F, float>)
              { c::free_fft_table_cplx_f32(_table); }
            else
              { c::free_fft_table_cplx_f64(_table); }
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

      template <typename F>
      class fft_table_real
      {
      public:
        using c_table_t = std::conditional_t<std::is_same_v<F, float>, c::fft_table_real_f32, c::fft_table_real_f64>;

        fft_table_real(size_t max_length)
        {
          static_assert(std::is_floating_point_v<F>);

          if constexpr (std::is_same_v<F, float>)
            { _table = c::allocate_fft_table_real_f32(max_length); }
          else
            { _table = c::allocate_fft_table_real_f64(max_length); }
        }

        // !!! todo: non-memory-owning version

        ~fft_table_real()
        {
          if constexpr (std::is_same_v<F, float>)
            { c::free_fft_table_real_f32(_table); }
          else
            { c::free_fft_table_real_f64(_table); }
        }

        fft_table_real(const fft_table_real &) = delete;
        fft_table_real &operator=(const fft_table_real &) = delete;
        fft_table_real(fft_table_real &&other) noexcept
          : _table{std::exchange(other._table, nullptr)}
          { }

        fft_table_real &operator=(fft_table_real &&other)
        {
          if (&other != this)
          {
            if constexpr (std::is_same_v<F, float>)
              { c::free_fft_table_real_f32(_table); }
            else
              { c::free_fft_table_real_f64(_table); }
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

      inline void fft_cplx(const fft_table_cplx<float> &table, const std::complex<float> *input, std::complex<float> *output, size_t length)
        { return c::fft_cplx_f32(table.c_table(), reinterpret_cast<const float *>(input), reinterpret_cast<float *>(output), length); }
      inline void fft_cplx(const fft_table_cplx<double> &table, const std::complex<double> *input, std::complex<double> *output, size_t length)
        { return c::fft_cplx_f64(table.c_table(), reinterpret_cast<const double *>(input), reinterpret_cast<double *>(output), length); }

      inline void fft_cplx(const fft_table_cplx<float> &table, const float *input, float *output, size_t length)
        { return c::fft_cplx_f32(table.c_table(), input, output, length); }
      inline void fft_cplx(const fft_table_cplx<double> &table, const double *input, double *output, size_t length)
        { return c::fft_cplx_f64(table.c_table(), input, output, length); }

      inline void fft_cplx(const fft_table_cplx<float> &table, std::complex<float> *data, size_t length)
        { return c::fft_cplx_f32(table.c_table(), reinterpret_cast<float *>(data), reinterpret_cast<float *>(data), length); }
      inline void fft_cplx(const fft_table_cplx<double> &table, std::complex<double> *data, size_t length)
        { return c::fft_cplx_f64(table.c_table(), reinterpret_cast<double *>(data), reinterpret_cast<double *>(data), length); }

      inline void fft_cplx(const fft_table_cplx<float> &table, float *data, size_t length)
        { return c::fft_cplx_f32(table.c_table(), data, data, length); }
      inline void fft_cplx(const fft_table_cplx<double> &table, double *data, size_t length)
        { return c::fft_cplx_f64(table.c_table(), data, data, length); }

      inline void ifft_cplx(const fft_table_cplx<float> &table, const std::complex<float> *input, std::complex<float> *output, size_t length)
        { return c::ifft_cplx_f32(table.c_table(), reinterpret_cast<const float *>(input), reinterpret_cast<float *>(output), length); }
      inline void ifft_cplx(const fft_table_cplx<double> &table, const std::complex<double> *input, std::complex<double> *output, size_t length)
        { return c::ifft_cplx_f64(table.c_table(), reinterpret_cast<const double *>(input), reinterpret_cast<double *>(output), length); }

      inline void ifft_cplx(const fft_table_cplx<float> &table, const float *input, float *output, size_t length)
        { return c::ifft_cplx_f32(table.c_table(), input, output, length); }
      inline void ifft_cplx(const fft_table_cplx<double> &table, const double *input, double *output, size_t length)
        { return c::ifft_cplx_f64(table.c_table(), input, output, length); }

      inline void ifft_cplx(const fft_table_cplx<float> &table, std::complex<float> *data, size_t length)
        { return c::ifft_cplx_f32(table.c_table(), reinterpret_cast<float *>(data), reinterpret_cast<float *>(data), length); }
      inline void ifft_cplx(const fft_table_cplx<double> &table, std::complex<double> *data, size_t length)
        { return c::ifft_cplx_f64(table.c_table(), reinterpret_cast<double *>(data), reinterpret_cast<double *>(data), length); }

      inline void ifft_cplx(const fft_table_cplx<float> &table, float *data, size_t length)
        { return c::ifft_cplx_f32(table.c_table(), data, data, length); }
      inline void ifft_cplx(const fft_table_cplx<double> &table, double *data, size_t length)
        { return c::ifft_cplx_f64(table.c_table(), data, data, length); }



      inline void fft_real(const fft_table_real<float> &table, const float *input, float *output, size_t length)
        { return c::fft_real_f32(table.c_table(), input, output, length); }
      inline void fft_real(const fft_table_real<double> &table, const double *input, double *output, size_t length)
        { return c::fft_real_f64(table.c_table(), input, output, length); }

      inline void fft_real(const fft_table_real<float> &table, float *data, size_t length)
        { return c::fft_real_f32(table.c_table(), data, data, length); }
      inline void fft_real(const fft_table_real<double> &table, double *data, size_t length)
        { return c::fft_real_f64(table.c_table(), data, data, length); }



      inline void ifft_real(const fft_table_real<float> &table, const float *input, float *output, size_t length)
        { return c::ifft_real_f32(table.c_table(), input, output, length); }
      inline void ifft_real(const fft_table_real<double> &table, const double *input, double *output, size_t length)
        { return c::ifft_real_f64(table.c_table(), input, output, length); }

      inline void ifft_real(const fft_table_real<float> &table, float *data, size_t length)
        { return c::ifft_real_f32(table.c_table(), data, data, length); }
      inline void ifft_real(const fft_table_real<double> &table, double *data, size_t length)
        { return c::ifft_real_f64(table.c_table(), data, data, length); }
    }
  #endif

  #if defined(_MSC_VER)
    #pragma warning (pop)
  #endif
#endif