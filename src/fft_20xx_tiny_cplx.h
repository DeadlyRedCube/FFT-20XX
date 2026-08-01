#pragma once

#include "fft_20xx_common.h"

namespace FFT20XX
{
  constexpr auto TinyFFTThreshold = 16u;

  // FFTs smaller than 16 just get done as scalar math.
  template <Dir D, std::floating_point F>
  void RunTinyFFT(const F *inputS, F *outputS, std::size_t fftLength)
  {
    auto input = reinterpret_cast<const Complex<F> *>(inputS);
    auto output = reinterpret_cast<Complex<F> *>(outputS);
    switch (fftLength)
    {
    case 1:
      output[0] = input[0];
      break;

    case 2:
      {
        auto x0 = input[0];
        auto x1 = input[1];
        output[0] = x0 + x1;
        output[1] = x0 - x1;
      }
      break;

    case 4:
      {
        auto x0 = input[0];
        auto x1 = input[1];
        auto x2 = input[2];
        auto x3 = input[3];

        auto sum02 = x0 + x2;
        auto diff02 = x0 - x2;
        auto sum13 = x1 + x3;
        auto iDiff13 = x1 - x3;
        if constexpr(D == Dir::Forward)
          { iDiff13 = { iDiff13.i, -iDiff13.r }; }
        else
          { iDiff13 = { -iDiff13.i, iDiff13.r }; }

        output[0] = sum02 + sum13;
        output[1] = diff02 + iDiff13;
        output[2] = sum02 - sum13;
        output[3] = diff02 - iDiff13;
      }
      break;

    case 8:
      {
        for (auto i = 0u; i < 4u; i++)
        {
          auto t = input[i] + input[i + 4];
          output[i + 4] = input[i] - input[i + 4];
          output[i] = t;
        }

        if constexpr (D == Dir::Forward)
        {
          output[6] = { output[6].i, -output[6].r };
          output[7] = { output[7].i, -output[7].r };
        }
        else
        {
          output[6] = { -output[6].i, output[6].r };
          output[7] = { -output[7].i, output[7].r };
        }

        for (auto i = 0u; i < 8u; i += 4)
        {
          auto t = output[i] + output[i + 2];
          output[i + 2] = output[i] - output[i + 2];
          output[i] = t;

          t = output[i + 1] + output[i + 3];
          output[i + 3] = output[i + 1] - output[i + 3];
          output[i + 1] = t;
        }

        // Now do the bit reverse ordering
        std::swap(output[1], output[4]);
        std::swap(output[3], output[6]);

        if constexpr (D == Dir::Forward)
        {
          output[5] =
            {
              // (r * s) - (i * -s), (r * -s) + (i * s) -> (r + i)s, (i - r)*s
              (output[5].r + output[5].i) * InvSqrt2<F>,
              (output[5].i - output[5].r) * InvSqrt2<F>,
            };
          output[6] = { output[6].i, -output[6].r };
          output[7] =
            {
              // (r * -s) - (i * -s), (r * -s) + (i * -s) -> (i - r)s, -(r + i)s
              (output[7].i - output[7].r) * InvSqrt2<F>,
              -(output[7].r + output[7].i) * InvSqrt2<F>,
            };
        }
        else
        {
          output[5] =
            {
              // (r * s) - (i * s), (r * s) + (i * s) -> (r - i)*s, (r + i)*s
              (output[5].r - output[5].i) * InvSqrt2<F>,
              (output[5].i + output[5].r) * InvSqrt2<F>,
            };
          output[6] = { -output[6].i, output[6].r };
          output[7] =
            {
              // (r * -s) - (i * s), (r * s) + (i * -s) -> -(r + i)s, (r - i)s
              -(output[7].r + output[7].i) * InvSqrt2<F>,
              (output[7].r - output[7].i) * InvSqrt2<F>,
            };
        }
        for (auto i = 0u; i < 4u; i++)
        {
          auto t = output[i] + output[i + 4];
          output[i + 4] = output[i] - output[i + 4];
          output[i] = t;
        }
      }
      break;
    }

    // Do the IFFT scaling (at the end, this time) if needed.
    if constexpr (D == Dir::Inverse)
    {
      const auto scale = F(1) / F(fftLength);
      for (auto i = 0u; i < fftLength; i++)
        { output[i] *= scale; }
    }
  }
}