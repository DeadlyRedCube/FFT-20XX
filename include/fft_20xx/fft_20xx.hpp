#ifndef FFT_20XX_HPP_INCLUDE_GUARD
  #define FFT_20XX_HPP_INCLUDE_GUARD

  #define FFT_20XX_INCLUDING_FROM_CPP
  #include "fft_20xx/fft_20xx.h"
  #undef FFT_20XX_INCLUDING_FROM_CPP

  #if defined(_MSC_VER)
    #pragma warning (push)
    #pragma warning (disable: 4514) // unreferenced inline function has been removed
    #pragma warning (disable: 4820) // N bytes padding added after data member '<name>'
  #endif

  namespace fft_20xx
  {
    template <typename F> class fft_table_cplx;
    template <typename F> class fft_table_real;

    inline void fft_cplx(const fft_table_cplx<float> &, const std::complex<float> *input, std::complex<float> *output, size_t length);
    inline void fft_cplx(const fft_table_cplx<float> &, const float *input, float *output, size_t length);
    inline void fft_cplx(const fft_table_cplx<float> &, std::complex<float> *data, size_t length);
    inline void fft_cplx(const fft_table_cplx<float> &, float *data, size_t length);

    inline void fft_cplx(const fft_table_cplx<double> &, const std::complex<double> *input, std::complex<double> *output, size_t length);
    inline void fft_cplx(const fft_table_cplx<double> &, const double *input, double *output, size_t length);
    inline void fft_cplx(const fft_table_cplx<double> &, std::complex<double> *data, size_t length);
    inline void fft_cplx(const fft_table_cplx<double> &, double *data, size_t length);

    inline void ifft_cplx(const fft_table_cplx<float> &, const std::complex<float> *input, std::complex<float> *output, size_t length);
    inline void ifft_cplx(const fft_table_cplx<float> &, const float *input, float *output, size_t length);
    inline void ifft_cplx(const fft_table_cplx<float> &, std::complex<float> *data, size_t length);
    inline void ifft_cplx(const fft_table_cplx<float> &, float *data, size_t length);

    inline void ifft_cplx(const fft_table_cplx<double> &, const std::complex<double> *input, std::complex<double> *output, size_t length);
    inline void ifft_cplx(const fft_table_cplx<double> &, const double *input, double *output, size_t length);
    inline void ifft_cplx(const fft_table_cplx<double> &, std::complex<double> *data, size_t length);
    inline void ifft_cplx(const fft_table_cplx<double> &, double *data, size_t length);



    inline void fft_real(const fft_table_real<float> &table, const float *input, float *output, size_t length);
    inline void fft_real(const fft_table_real<float> &table, float *data, size_t length);

    inline void fft_real(const fft_table_real<double> &table, const double *input, double *output, size_t length);
    inline void fft_real(const fft_table_real<double> &table, double *data, size_t length);



    inline void ifft_real(const fft_table_real<float> &table, const float *input, float *output, size_t length);
    inline void ifft_real(const fft_table_real<float> &table, float *data, size_t length);

    inline void ifft_real(const fft_table_real<double> &table, const double *input, double *output, size_t length);
    inline void ifft_real(const fft_table_real<double> &table, double *data, size_t length);


    template <typename F> class fft_table_cplx
    {
    public:
      using c_table_t = std::conditional_t<std::is_same_v<F, float>, c::fft_table_cplx_f32, c::fft_table_cplx_f64>;

      // Allocate a new complex FFT table for FFTs up to the given maximum length.
      fft_table_cplx(size_t max_length);

      // Use the given block of memory (with its given size) to create a new  
      fft_table_cplx(size_t max_length, void *ptr, size_t byte_count);

      fft_table_cplx(const fft_table_cplx &) = delete;
      fft_table_cplx &operator=(const fft_table_cplx &) = delete;
      fft_table_cplx(fft_table_cplx &&other) noexcept;
      fft_table_cplx &operator=(fft_table_cplx &&other) noexcept;
      ~fft_table_cplx();

      const c_table_t *c_table() const
        { assert(_table != nullptr); return _table; }

      operator const c_table_t *() const
        { return c_table(); }

    private:
      void release();
      const c_table_t *_table = nullptr;
      bool _owns_memory = false;
    };

    template <typename F>
    class fft_table_real
    {
    public:
      using c_table_t = std::conditional_t<std::is_same_v<F, float>, c::fft_table_real_f32, c::fft_table_real_f64>;

      fft_table_real(size_t max_length);
      fft_table_real(size_t max_length, void *ptr, size_t byte_count);

      fft_table_real(const fft_table_real &) = delete;
      fft_table_real &operator=(const fft_table_real &) = delete;
      fft_table_real(fft_table_real &&other) noexcept;
      fft_table_real &operator=(fft_table_real &&other) noexcept;
      ~fft_table_real();

      const c_table_t *c_table() const
        { assert(_table != nullptr); return _table; }

      operator const c_table_t *() const
        { return c_table(); }

    private:
      void release();
      const c_table_t *_table = nullptr;
      bool _owns_memory = false;
    };

    // Inline function definitions


    inline void fft_cplx(
        const fft_table_cplx<float> &table,
        const std::complex<float> *input,
        std::complex<float> *output,
        size_t length)
      { return c::fft_cplx_f32(table.c_table(), reinterpret_cast<const float *>(input), reinterpret_cast<float *>(output), length); }
    inline void fft_cplx(
        const fft_table_cplx<double> &table,
        const std::complex<double> *input,
        std::complex<double> *output,
        size_t length)
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

    inline void ifft_cplx(
        const fft_table_cplx<float> &table,
        const std::complex<float> *input,
        std::complex<float> *output,
        size_t length)
      { return c::ifft_cplx_f32(table.c_table(), reinterpret_cast<const float *>(input), reinterpret_cast<float *>(output), length); }
    inline void ifft_cplx(
        const fft_table_cplx<double> &table,
        const std::complex<double> *input,
        std::complex<double> *output,
        size_t length)
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

  #if defined(_MSC_VER)
    #pragma warning (pop)
  #endif
#endif