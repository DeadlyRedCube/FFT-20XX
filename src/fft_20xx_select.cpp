#include "fft_20xx/fft_20xx.h"
#include "fft_20xx/fft_20xx_util.h"

#include "fft_20xx_common.h"

using namespace FFT20XX;

#if (COMPILER_GCC || COMPILER_CLANG) && !OS_WINDOWS && CPU_X86_ANY
  #include <cpuid.h>
#endif

extern "C"
{
  using namespace fft_20xx::c;

#define MAKE_C_OVERLOADS(suffix) \
  void fft_cplx_interleaved_f32##suffix(const fft_table_f32 *table, const float *input, float *output, size_t fftLength); \
  void fft_cplx_interleaved_f64##suffix(const fft_table_f64 *table, const double *input, double *output, size_t fftLength); \
  \
  void fft_cplx_f32##suffix(const fft_table_f32 *table, const std::complex<float> *input, std::complex<float> *output, size_t fftLength) \
    { fft_cplx_interleaved_f32##suffix(table, reinterpret_cast<const float *>(input), reinterpret_cast<float *>(output), fftLength); } \
  \
  void fft_cplx_f64##suffix(const fft_table_f64 *table, const std::complex<double> *input, std::complex<double> *output, size_t fftLength) \
    { fft_cplx_interleaved_f64##suffix(table, reinterpret_cast<const double *>(input), reinterpret_cast<double *>(output), fftLength); } \
  \
  void ifft_cplx_interleaved_f32##suffix(const fft_table_f32 *table, const float *input, float *output, size_t fftLength); \
  void ifft_cplx_interleaved_f64##suffix(const fft_table_f64 *table, const double *input, double *output, size_t fftLength); \
  \
  void ifft_cplx_f32##suffix(const fft_table_f32 *table, const std::complex<float> *input, std::complex<float> *output, size_t fftLength) \
    { ifft_cplx_interleaved_f32##suffix(table, reinterpret_cast<const float *>(input), reinterpret_cast<float *>(output), fftLength); } \
  \
  void ifft_cplx_f64##suffix(const fft_table_f64 *table, const std::complex<double> *input, std::complex<double> *output, size_t fftLength) \
    { ifft_cplx_interleaved_f64##suffix(table, reinterpret_cast<const double *>(input), reinterpret_cast<double *>(output), fftLength); } \
  \
  void fft_real_f32##suffix(const fft_table_f32 *table, const float *input, float *output, size_t fftLength); \
  void fft_real_f64##suffix(const fft_table_f64 *table, const double *input, double *output, size_t fftLength); \
  \
  void ifft_real_f32##suffix(const fft_table_f32 *table, const float *input, float *output, size_t fftLength); \
  void ifft_real_f64##suffix(const fft_table_f64 *table, const double *input, double *output, size_t fftLength); \
  \

  MAKE_C_OVERLOADS()
  MAKE_C_OVERLOADS(_scalar)

  #if CPU_X86 || CPU_X64
    MAKE_C_OVERLOADS(_sse2);
    MAKE_C_OVERLOADS(_avx);
    MAKE_C_OVERLOADS(_fma);
  #elif SIMD_NEON
    MAKE_C_OVERLOADS(_neon);
  #endif
}


extern "C"
{
  #if SIMD_AVX
    #if SIMD_FMA
      #define SELECT(name, ...) name##_fma(__VA_ARGS__)
    #else
      do \
      { \
        using namespace fft_20xx; \
        if ((get_simd_support() & (simd_support_flags::fma | simd_support_flags::avx)) == (simd_support_flags::fma | simd_support_flags::avx)) \
          { name##_fma(__VA_ARGS__); } \
        else \
          { name##_avx(__VA_ARGS__); } \
      } while (false)
    #endif
  #elif SIMD_SSE2
    #define SELECT(name, ...) \
      do \
      { \
        using namespace fft_20xx; \
        if ((get_simd_support() & (simd_support_flags::fma | simd_support_flags::avx)) == (simd_support_flags::fma | simd_support_flags::avx)) \
          { name##_fma(__VA_ARGS__); } \
        else if ((get_simd_support() & simd_support_flags::avx) == simd_support_flags::avx) \
          { name##_avx(__VA_ARGS__); } \
        else \
          { name##_sse2(__VA_ARGS__); } \
      } while (false)
  #elif CPU_X86 || CPU_X64
    #define SELECT(name, ...) \
      do \
      { \
        using namespace fft_20xx; \
        if ((get_simd_support() & (simd_support_flags::fma | simd_support_flags::avx)) \
            == (simd_support_flags::fma | simd_support_flags::avx)) \
          { name##_fma(__VA_ARGS__); } \
        else if ((get_simd_support() & simd_support_flags::avx) == simd_support_flags::avx) \
          { name##_avx(__VA_ARGS__); } \
        else if ((get_simd_support() & simd_support_flags::sse2) == simd_support_flags::sse2) \
          { name##_sse2(__VA_ARGS__); } \
        else \
          { name##_scalar(__VA_ARGS__); } \
      } while (false)
  #elif SIMD_NEON
    #define SELECT(name, ...) name##_neon(__VA_ARGS__)
  #else
    #define SELECT(name, ...) name##_scalar(__VA_ARGS__)
  #endif

  void fft_cplx_interleaved_f32(const fft_table_f32 *table, const float *input, float *output, size_t length)
    { SELECT(fft_cplx_interleaved_f32, table, input, output, length); }

  void fft_cplx_interleaved_f64(const fft_table_f64 *table, const double *input, double *output, size_t length)
    { SELECT(fft_cplx_interleaved_f64, table, input, output, length); }

  void ifft_cplx_interleaved_f32(const fft_table_f32 *table, const float *input, float *output, size_t length)
    { SELECT(ifft_cplx_interleaved_f32, table, input, output, length); }

  void ifft_cplx_interleaved_f64(const fft_table_f64 *table, const double *input, double *output, size_t length)
    { SELECT(ifft_cplx_interleaved_f64, table, input, output, length); }

  void fft_real_f32(const fft_table_f32 *table, const float *input, float *output, size_t length)
    { SELECT(fft_real_f32, table, input, output, length); }

  void ifft_real_f32(const fft_table_f32 *table, const float *input, float *output, size_t length)
    { SELECT(ifft_real_f32, table, input, output, length); }

  void fft_real_f64(const fft_table_f64 *table, const double *input, double *output, size_t length)
    { SELECT(fft_real_f64, table, input, output, length); }

  void ifft_real_f64(const fft_table_f64 *table, const double *input, double *output, size_t length)
    { SELECT(ifft_real_f64, table, input, output, length); }
}
