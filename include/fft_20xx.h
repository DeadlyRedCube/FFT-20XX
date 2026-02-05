#ifndef FFT_20XX_INCLUDE_GUARD
  #define FFT_20XX_INCLUDE_GUARD

  #include <inttypes.h>
  #include <stddef.h>

  #ifdef __cplusplus
    extern "C"
    {
  #endif

    struct FFTTable32;
    struct FFTTable64;

    const FFTTable32 *AllocateFFTTable32(size_t maxFFTLength);
    void FreeFFTTable32(const FFTTable32 *table);

    const FFTTable64 *AllocateFFTTable64(size_t maxFFTLength);
    void FreeFFTTable64(const FFTTable64 *table);

    // TODO:
    //size_t GetRequiredFFTTableMemorySize(size_t maxFFTLength);
    //const FFTTable *InitializeFFTTable(void *memory, size_t maxFFTLength);

    void FFTComplex32(const FFTTable32 *table, const float *input, float *output, size_t fftLength);
    void FFTComplex64(const FFTTable64 *table, const double *input, double *output, size_t fftLength);

    void FFTComplex32InPlace(const FFTTable32 *table, float *data, size_t fftLength);
    void FFTComplex64InPlace(const FFTTable64 *table, double *data, size_t fftLength);

    // TODO: Inverse FFT

    // TODO: Real-valued FFT

    // TODO: Inverse real


    // TODO: Explicit Scalar, AVX, FMA, SSE, NEON, AARCH64 variants



    // TODO: C and C++ complex versions?
  #ifdef __cplusplus
    }
  #endif
#endif