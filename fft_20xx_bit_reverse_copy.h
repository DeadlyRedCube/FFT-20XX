#pragma once

#include "fft_20xx_complex.h"


namespace FFT20XX
{
  template <std::floating_point F>
  void BitReverseCopy(const Complex<F> *input, Complex<F> *output, std::size_t fftLength);
}
