#ifndef FFT_20XX_INCLUDE_GUARD
  #define FFT_20XX_INCLUDE_GUARD

  #include <inttypes.h>
  #include <stddef.h>

  // C's "float/double _Complex" and C++'s "std::complex<float/double>" are guaranteed to have the same layout, so
  //  use the correct language-specific version.
  #ifdef __cplusplus
    #include <complex>

    #define FFT_20XX_F32_COMPLEX std::complex<float>
    #define FFT_20XX_F64_COMPLEX std::complex<double>

    extern "C"
    {
  #else
    #include <complex.h>

    #if defined(_MSC_VER) && !defined(__clang__)
      #define FFT_20XX_F32_COMPLEX _Fcomplex
      #define FFT_20XX_F64_COMPLEX _Dcomplex
    #else
      #define FFT_20XX_F32_COMPLEX float _Complex
      #define FFT_20XX_F64_COMPLEX double _Complex
    #endif
  #endif

    struct FFTTable32;
    struct FFTTable64;

    const struct FFTTable32 *AllocateFFTTable32(size_t maxFFTLength);
    void FreeFFTTable32(const struct FFTTable32 *table);

    const struct FFTTable64 *AllocateFFTTable64(size_t maxFFTLength);
    void FreeFFTTable64(const struct FFTTable64 *table);

    // TODO:
    //size_t GetRequiredFFTTableMemorySize(size_t maxFFTLength);
    //const FFTTable *InitializeFFTTable(void *memory, size_t maxFFTLength);

    void FFTComplex32(const struct FFTTable32 *table, const FFT_20XX_F32_COMPLEX *input, FFT_20XX_F32_COMPLEX *output, size_t fftLength);
    void FFTComplex64(const struct FFTTable64 *table, const FFT_20XX_F64_COMPLEX *input, FFT_20XX_F64_COMPLEX *output, size_t fftLength);

    void FFTComplex32F(const struct FFTTable32 *table, const float *input, float *output, size_t fftLength);
    void FFTComplex64F(const struct FFTTable64 *table, const double *input, double *output, size_t fftLength);

    void FFTComplex32InPlace(const struct FFTTable32 *table, float *data, size_t fftLength);
    void FFTComplex64InPlace(const struct FFTTable64 *table, double *data, size_t fftLength);

    // TODO: Inverse FFT

    // TODO: Real-valued FFT

    // TODO: Inverse real


    // TODO: Explicit Scalar, AVX, FMA, SSE, NEON, AARCH64 variants



    // TODO: C and C++ complex versions?
  #ifdef __cplusplus
    }
  #endif

  #undef FFT_20XX_F32_COMPLEX
  #undef FFT_20XX_F64_COMPLEX
#endif