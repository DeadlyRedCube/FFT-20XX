#ifndef FFT_20XX_H_INCLUDE_GUARD
  #define FFT_20XX_H_INCLUDE_GUARD

  #include <inttypes.h>
  #include <stddef.h>

  #ifdef __cplusplus
    #ifndef FFT_20XX_INCLUDING_FROM_CPP
      #error This header is intended for C only. Please include "fft_20xx.hpp" instead.
    #endif
    #include <cassert>
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

    // Allocate and free FFT tables for the various input/output types

    const fft_table_cplx_f32 *allocate_fft_table_cplx_f32(size_t length);
    void free_fft_table_cplx_f32(const fft_table_cplx_f32 *table);

    const fft_table_cplx_f64 *allocate_fft_table_cplx_f64(size_t max_length);
    void free_fft_table_cplx_f64(const fft_table_cplx_f64 *table);

    const fft_table_real_f32 *allocate_fft_table_real_f32(size_t length);
    void free_fft_table_real_f32(const fft_table_real_f32 *table);

    const fft_table_real_f64 *allocate_fft_table_real_f64(size_t max_length);
    void free_fft_table_real_f64(const fft_table_real_f64 *table);

    size_t get_simd_alignment();

    // Get the number of bytes needed to allocate a FFT table of the given float and type that supports the given
    //  maximum length when the allocation is guaranteed to be aligned to the result of get_simd_alignment.

    size_t get_fft_table_cplx_f32_byte_count_aligned(size_t maxFFTLength);
    size_t get_fft_table_cplx_f64_byte_count_aligned(size_t maxFFTLength);
    size_t get_fft_table_real_f32_byte_count_aligned(size_t maxFFTLength);
    size_t get_fft_table_real_f64_byte_count_aligned(size_t maxFFTLength);

    // Get the number of bytes needed to allocate a FFT table of the given float and type that supports the given
    //  maximum length when the allocation is not guaranteed to be aligned.

    size_t get_fft_table_cplx_f32_byte_count_unaligned(size_t maxFFTLength);
    size_t get_fft_table_cplx_f64_byte_count_unaligned(size_t maxFFTLength);
    size_t get_fft_table_real_f32_byte_count_unaligned(size_t maxFFTLength);
    size_t get_fft_table_real_f64_byte_count_unaligned(size_t maxFFTLength);

    // Return a pointer to a table of the corresponding type initialized within the given block of memory.
    //  Will assert (if assertions are enabled) and return nullptr if the given byte count was insufficient.

    const fft_table_cplx_f32 *make_table_from_memory_cplx_f32(size_t maxFFTLength, void *memory, size_t byteCount);
    const fft_table_cplx_f64 *make_table_from_memory_cplx_f64(size_t maxFFTLength, void *memory, size_t byteCount);
    const fft_table_real_f32 *make_table_from_memory_real_f32(size_t maxFFTLength, void *memory, size_t byteCount);
    const fft_table_real_f64 *make_table_from_memory_real_f64(size_t maxFFTLength, void *memory, size_t byteCount);

    // Run a complex FFT of the given power-of-two length. Note that `input` and `output` are allowed to be the same
    //  pointer, which means the calculation will run in-place.
    // The input and output are both packed complex values (alternating real and imaginary components)

    void fft_cplx_f32(const fft_table_cplx_f32 *table, const float *input, float *output, size_t length);
    void fft_cplx_f64(const fft_table_cplx_f64 *table, const double *input, double *output, size_t length);

    // Run a complex inverse FFT of the given power-of-two length. Note that `input` and `output` are allowed to be the
    //  same pointer, which means the calculation will run in-place.
    // The input and output are both packed complex values (alternating real and imaginary components)

    void ifft_cplx_f32(const fft_table_cplx_f32 *table, const float *input, float *output, size_t length);
    void ifft_cplx_f64(const fft_table_cplx_f64 *table, const double *input, double *output, size_t length);

    // Run a real-input FFT of the given power-of-two length. Note that `input` and `output` are allowed to be the same
    //  pointer, which means the calculation will run in-place.
    // The input is real (non-complex) sample values.
    // The output will be the first half (plus one) of the complex results of an FFT for the given inputs (the last
    //  half outputs are always complex conjugates of the first half). The DC and nyquist values (complex outputs
    //  [0] and [N/2]) are both real, and packed into the first two elements (as [0] and [1] respectively) of the
    //  output. The remaining values are all complex pairs.

    void fft_real_f32(const fft_table_real_f32 *table, const float *input, float *output, size_t length);
    void fft_real_f64(const fft_table_real_f64 *table, const double *input, double *output, size_t length);

    // Run a real-input FFT of the given power-of-two length. Note that `input` and `output` are allowed to be the same
    //  pointer, which means the calculation will run in-place.
    // The input is in the same configuration of the output of the forward real FFT functions.
    // The output is all real (non-complex) sample values.

    void ifft_real_f32(const fft_table_real_f32 *table, const float *input, float *output, size_t length);
    void ifft_real_f64(const fft_table_real_f64 *table, const double *input, double *output, size_t length);
  #ifdef __cplusplus
      } // namespace
    } // extern "C"
  #endif
#endif