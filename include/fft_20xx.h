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

    void FFTComplex32Interleaved(const struct FFTTable32 *table, const float *input, float *output, size_t fftLength);
    void FFTComplex64Interleaved(const struct FFTTable64 *table, const double *input, double *output, size_t fftLength);

    void FFTComplex32InPlace(const struct FFTTable32 *table, float *data, size_t fftLength);
    void FFTComplex64InPlace(const struct FFTTable64 *table, double *data, size_t fftLength);

    void IFFTComplex32(const struct FFTTable32 *table, const FFT_20XX_F32_COMPLEX *input, FFT_20XX_F32_COMPLEX *output, size_t fftLength);
    void IFFTComplex64(const struct FFTTable64 *table, const FFT_20XX_F64_COMPLEX *input, FFT_20XX_F64_COMPLEX *output, size_t fftLength);

    void IFFTComplex32Interleaved(const struct FFTTable32 *table, const float *input, float *output, size_t fftLength);
    void IFFTComplex64Interleaved(const struct FFTTable64 *table, const double *input, double *output, size_t fftLength);

    // TODO: IFFT Complex In-Place

    // TODO: Real

    // TODO: Explicit Scalar, AVX, FMA, SSE, NEON, AARCH64 variants
    void FFTComplex32_Scalar(const struct FFTTable32 *table, const FFT_20XX_F32_COMPLEX *input, FFT_20XX_F32_COMPLEX *output, size_t fftLength);
    void FFTComplex64_Scalar(const struct FFTTable64 *table, const FFT_20XX_F64_COMPLEX *input, FFT_20XX_F64_COMPLEX *output, size_t fftLength);
    void FFTComplex32Interleaved_Scalar(const struct FFTTable32 *table, const float *input, float *output, size_t fftLength);
    void FFTComplex64Interleaved_Scalar(const struct FFTTable64 *table, const double *input, double *output, size_t fftLength);
    void IFFTComplex32_Scalar(const struct FFTTable32 *table, const FFT_20XX_F32_COMPLEX *input, FFT_20XX_F32_COMPLEX *output, size_t fftLength);
    void IFFTComplex64_Scalar(const struct FFTTable64 *table, const FFT_20XX_F64_COMPLEX *input, FFT_20XX_F64_COMPLEX *output, size_t fftLength);
    void IFFTComplex32Interleaved_Scalar(const struct FFTTable32 *table, const float *input, float *output, size_t fftLength);
    void IFFTComplex64Interleaved_Scalar(const struct FFTTable64 *table, const double *input, double *output, size_t fftLength);

    #if defined(_M_X64) || defined(__x86_64__) || defined(_M_IX86) || defined(__i386__)
      void FFTComplex32_SSE2(const struct FFTTable32 *table, const FFT_20XX_F32_COMPLEX *input, FFT_20XX_F32_COMPLEX *output, size_t fftLength);
      void FFTComplex64_SSE2(const struct FFTTable64 *table, const FFT_20XX_F64_COMPLEX *input, FFT_20XX_F64_COMPLEX *output, size_t fftLength);
      void FFTComplex32Interleaved_SSE2(const struct FFTTable32 *table, const float *input, float *output, size_t fftLength);
      void FFTComplex64Interleaved_SSE2(const struct FFTTable64 *table, const double *input, double *output, size_t fftLength);
      void IFFTComplex32_SSE2(const struct FFTTable32 *table, const FFT_20XX_F32_COMPLEX *input, FFT_20XX_F32_COMPLEX *output, size_t fftLength);
      void IFFTComplex64_SSE2(const struct FFTTable64 *table, const FFT_20XX_F64_COMPLEX *input, FFT_20XX_F64_COMPLEX *output, size_t fftLength);
      void IFFTComplex32Interleaved_SSE2(const struct FFTTable32 *table, const float *input, float *output, size_t fftLength);
      void IFFTComplex64Interleaved_SSE2(const struct FFTTable64 *table, const double *input, double *output, size_t fftLength);

      void FFTComplex32_AVX(const struct FFTTable32 *table, const FFT_20XX_F32_COMPLEX *input, FFT_20XX_F32_COMPLEX *output, size_t fftLength);
      void FFTComplex64_AVX(const struct FFTTable64 *table, const FFT_20XX_F64_COMPLEX *input, FFT_20XX_F64_COMPLEX *output, size_t fftLength);
      void FFTComplex32Interleaved_AVX(const struct FFTTable32 *table, const float *input, float *output, size_t fftLength);
      void FFTComplex64Interleaved_AVX(const struct FFTTable64 *table, const double *input, double *output, size_t fftLength);
      void IFFTComplex32_AVX(const struct FFTTable32 *table, const FFT_20XX_F32_COMPLEX *input, FFT_20XX_F32_COMPLEX *output, size_t fftLength);
      void IFFTComplex64_AVX(const struct FFTTable64 *table, const FFT_20XX_F64_COMPLEX *input, FFT_20XX_F64_COMPLEX *output, size_t fftLength);
      void IFFTComplex32Interleaved_AVX(const struct FFTTable32 *table, const float *input, float *output, size_t fftLength);
      void IFFTComplex64Interleaved_AVX(const struct FFTTable64 *table, const double *input, double *output, size_t fftLength);

      void FFTComplex32_FMA(const struct FFTTable32 *table, const FFT_20XX_F32_COMPLEX *input, FFT_20XX_F32_COMPLEX *output, size_t fftLength);
      void FFTComplex64_FMA(const struct FFTTable64 *table, const FFT_20XX_F64_COMPLEX *input, FFT_20XX_F64_COMPLEX *output, size_t fftLength);
      void FFTComplex32Interleaved_FMA(const struct FFTTable32 *table, const float *input, float *output, size_t fftLength);
      void FFTComplex64Interleaved_FMA(const struct FFTTable64 *table, const double *input, double *output, size_t fftLength);
      void IFFTComplex32_FMA(const struct FFTTable32 *table, const FFT_20XX_F32_COMPLEX *input, FFT_20XX_F32_COMPLEX *output, size_t fftLength);
      void IFFTComplex64_FMA(const struct FFTTable64 *table, const FFT_20XX_F64_COMPLEX *input, FFT_20XX_F64_COMPLEX *output, size_t fftLength);
      void IFFTComplex32Interleaved_FMA(const struct FFTTable32 *table, const float *input, float *output, size_t fftLength);
      void IFFTComplex64Interleaved_FMA(const struct FFTTable64 *table, const double *input, double *output, size_t fftLength);

    void FFTReal32_FMA(const FFTTable32 *table, const float *input, float *output, size_t fftLength);
    void FFTReal64_FMA(const FFTTable64 *table, const double *input, double *output, size_t fftLength);

    void IFFTReal32_FMA(const FFTTable32 *table, const float *input, float *output, size_t fftLength);
    void IFFTReal64_FMA(const FFTTable64 *table, const double *input, double *output, size_t fftLength);
    #endif

    #if defined(__ARM_NEON)
      void FFTComplex32_NEON(const struct FFTTable32 *table, const FFT_20XX_F32_COMPLEX *input, FFT_20XX_F32_COMPLEX *output, size_t fftLength);
      void FFTComplex64_NEON(const struct FFTTable64 *table, const FFT_20XX_F64_COMPLEX *input, FFT_20XX_F64_COMPLEX *output, size_t fftLength);
      void FFTComplex32Interleaved_NEON(const struct FFTTable32 *table, const float *input, float *output, size_t fftLength);
      void FFTComplex64Interleaved_NEON(const struct FFTTable64 *table, const double *input, double *output, size_t fftLength);
      void IFFTComplex32_NEON(const struct FFTTable32 *table, const FFT_20XX_F32_COMPLEX *input, FFT_20XX_F32_COMPLEX *output, size_t fftLength);
      void IFFTComplex64_NEON(const struct FFTTable64 *table, const FFT_20XX_F64_COMPLEX *input, FFT_20XX_F64_COMPLEX *output, size_t fftLength);
      void IFFTComplex32Interleaved_NEON(const struct FFTTable32 *table, const float *input, float *output, size_t fftLength);
      void IFFTComplex64Interleaved_NEON(const struct FFTTable64 *table, const double *input, double *output, size_t fftLength);
    #endif
  #ifdef __cplusplus
    }
  #endif

  #undef FFT_20XX_F32_COMPLEX
  #undef FFT_20XX_F64_COMPLEX
#endif