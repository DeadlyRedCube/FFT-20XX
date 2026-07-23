#pragma once

namespace FFT20XX
{
  template <std::floating_point F>
  void TinyRealFwd(const F *input, F *output, size_t fftLength)
  {
    using std::tie;
    using tup = std::tuple<F, F>;

    // assert(fftLength < 32);

    switch (fftLength)
    {
    case 1:
      output[0] = input[0];
      return;

    case 2:
      {
        auto a = input[0];
        auto b = input[1];
        output[0] = a + b;
        output[1] = a - b;
      }
      return;

    case 4:
      {
        // Single radix 2 plus the fixup at the end.
        auto a2 = input[0] + input[2];
        auto b2 = input[1] + input[3];
        output[2] = input[0] - input[2];
        output[3] = input[3] - input[1];
        output[0] = a2 + b2;
        output[1] = a2 - b2;
      }
      return;

    case 8:
      {
        // first radix 2
        auto [a, e] = tup{input[0] + input[4], input[0] - input[4]};
        auto [b, f] = tup{input[1] + input[5], input[1] - input[5]};
        auto [c, g] = tup{input[2] + input[6], input[6] - input[2]};
        auto [d, h] = tup{input[3] + input[7], input[7] - input[3]};

        // second radix 2
        tie(a, c) = tup{a + c, a - c};
        tie(b, d) = tup{b + d, d - b};

        constexpr auto s = FFT20XX::InvSqrt2<F>;
        tie(f, h) = tup{s*(f + h), s*(h - f)};

        tie(e, f) = tup{e + f, e - f};
        tie(g, h) = tup{g + h, h - g};

        // Write these out with a single bit reverse
        output[0] = a + b;
        output[1] = a - b;
        output[2] = e;
        output[3] = g;
        output[4] = c;
        output[5] = d;
        output[6] = f;
        output[7] = h;
      }
      return;

    case 16:
      {
        // First R2
        auto [a, k] = tup{input[0] + input[8], input[0] - input[8]};
        auto [b, l] = tup{input[1] + input[9], input[1] - input[9]};
        auto [c, m] = tup{input[2] + input[10], input[2] - input[10]};
        auto [d, n] = tup{input[3] + input[11], input[3] - input[11]};
        auto [e, o] = tup{input[12] + input[4], input[12] - input[4]};
        auto [f, p] = tup{input[13] + input[5], input[13] - input[5]};
        auto [g, q] = tup{input[14] + input[6], input[14] - input[6]};
        auto [h, r] = tup{input[15] + input[7], input[15] - input[7]};

        // Second R2
        tie(a, e) = tup{a + e, a - e};
        tie(b, f) = tup{b + f, b - f};
        tie(c, g) = tup{g + c, g - c};
        tie(d, h) = tup{h + d, h - d};

        constexpr auto Z0 = InvSqrt2<F>;
        tie(m, q) = tup{Z0*(q + m), Z0*(q - m)};
        tie(n, r) = tup{Z0*(r + n), Z0*(r - n)};
        tie(k, m) = tup{k + m, k - m};
        tie(l, n) = tup{l + n, l - n};
        tie(o, q) = tup{q + o, q - o};
        tie(p, r) = tup{r + p, r - p};
        std::swap(o, m);
        std::swap(p, n);

        // Third R2
        constexpr auto Z1 = F(0.9238795325112867561); // cos(pi/8)
        constexpr auto Z2 = F(0.3826834323650897717); // sin(pi/8)

        tie(a, c) = tup{a + c, a - c};
        tie(b, d) = tup{d + b, d - b};

        // angle0 = z0, -z0
        tie(f, h) = tup{Z0*(f + h), Z0*(h - f)};
        tie(e, f) = tup{e + f, e - f};
        tie(g, h) = tup{h + g, h - g};
        std::swap(g, f);

        // angle1 = z1, -z2
        tie(l, n) = tup{l*Z1 + n*Z2, n*Z1 - l*Z2};
        tie(k, l) = tup{k + l, k - l};
        tie(m, n) = tup{n + m, n - m};
        std::swap(m, l);

        // angle2 = z2, -z1
        tie(p, r) = tup{p*Z2 + r*Z1, r*Z2 - p*Z1};
        tie(o, p) = tup{o + p, o - p};
        tie(q, r) = tup{r + q, r - q};
        std::swap(q, p);

        output[0] = a + b;
        output[1] = a - b;
        output[2] = k;
        output[3] = l;
        output[4] = e;
        output[5] = f;
        output[6] = o;
        output[7] = p;
        output[8] = c;
        output[9] = d;
        output[10] = q;
        output[11] = r;
        output[12] = g;
        output[13] = h;
        output[14] = m;
        output[15] = n;
      }
      return;
    }
  }

  // This is literally just the opposite of the above routine
  template <std::floating_point F>
  void TinyRealInv(const F *input, F *output, size_t fftLength)
  {
    using std::tie;
    using tup = std::tuple<F, F>;

    assert(fftLength < 32);

    switch (fftLength)
    {
    case 1:
      output[0] = input[0];
      return;

    case 2:
      {
        auto a = input[0];
        auto b = input[1];
        output[0] = (a + b) * 0.5f;
        output[1] = (a - b) * 0.5f;
      }
      return;

    case 4:
      {
        auto a = (input[0] + input[1]) * 0.5f;
        auto b = (input[0] - input[1]) * 0.5f;
        auto c = input[2];
        auto d = input[3];

        output[0] = (a + c) * 0.5f;
        output[1] = (b - d) * 0.5f;
        output[2] = (a - c) * 0.5f;
        output[3] = (b + d) * 0.5f;
      }
      return;

    case 8:
      {
        // first radix 2
        auto a = (input[0] + input[1]) * 0.5f;
        auto b = (input[0] - input[1]) * 0.5f;
        auto e = input[2];
        auto g = input[3];
        auto c = input[4];
        auto d = input[5];
        auto f = input[6];
        auto h = input[7];

        tie(e, f) = tup{e + f, e - f};
        tie(g, h) = tup{g - h, g + h};

        constexpr auto s = FFT20XX::InvSqrt2<F>;
        tie(f, h) = tup{s*(f - h), s*(f + h)};

        tie(a, c) = tup{a + c, a - c};
        tie(b, d) = tup{b - d, b + d};

        // Final radix2
        tie(output[0], output[4]) = tup{(a + e) * 0.25f, (a - e) * 0.25f};
        tie(output[1], output[5]) = tup{(b + f) * 0.25f, (b - f) * 0.25f};
        tie(output[2], output[6]) = tup{(c - g) * 0.25f, (c + g) * 0.25f};
        tie(output[3], output[7]) = tup{(d - h) * 0.25f, (d + h) * 0.25f};
      }
      return;

    case 16:
      {
        auto a = (input[0] + input[1]) * 0.5f;
        auto b = (input[0] - input[1]) * 0.5f;
        auto k= input[2];
        auto l= input[3];
        auto e= input[4];
        auto f= input[5];
        auto o= input[6];
        auto p= input[7];
        auto c= input[8];
        auto d= input[9];
        auto q= input[10];
        auto r= input[11];
        auto g= input[12];
        auto h= input[13];
        auto m= input[14];
        auto n= input[15];

        constexpr auto Z0 = InvSqrt2<F>;
        constexpr auto Z1 = F(0.9238795325112867561); // cos(pi/8)
        constexpr auto Z2 = F(0.3826834323650897717); // sin(pi/8)

        // First
        std::swap(q, p);
        tie(o, p) = tup{o + p, o - p};
        tie(q, r) = tup{q - r, q + r};
        tie(p, r) = tup{p*Z2 - r*Z1, r*Z2 + p*Z1};

        // angle1 = z1, -z2
        std::swap(m, l);
        tie(k, l) = tup{k + l, k - l};
        tie(m, n) = tup{m - n, m + n};
        tie(l, n) = tup{l*Z1 - n*Z2, n*Z1 + l*Z2};

        // angle0 = z0, -z0
        std::swap(g, f);
        tie(e, f) = tup{e + f, e - f};
        tie(g, h) = tup{g - h, g + h};
        tie(f, h) = tup{Z0*(f - h), Z0*(h + f)};

        tie(a, c) = tup{a + c, a - c};
        tie(b, d) = tup{b - d, b + d};

        // Second R2
        std::swap(p, n);
        std::swap(o, m);

        tie(a, e) = tup{a + e, a - e};
        tie(b, f) = tup{b + f, b - f};
        tie(c, g) = tup{c - g, c + g};
        tie(d, h) = tup{d - h, d + h};

        tie(k, m) = tup{k + m, k - m};
        tie(l, n) = tup{l + n, l - n};
        tie(o, q) = tup{o - q, o + q};
        tie(p, r) = tup{p - r, p + r};
        tie(m, q) = tup{Z0*(m - q), Z0*(m + q)};
        tie(n, r) = tup{Z0*(n - r), Z0*(n + r)};

        // Third R2
        tie(output[0], output[8]) = tup{(a + k) * F(0.125), (a - k) * F(0.125)};
        tie(output[1], output[9]) = tup{(b + l) * F(0.125), (b - l) * F(0.125)};
        tie(output[2], output[10]) = tup{(c + m) * F(0.125), (c - m) * F(0.125)};
        tie(output[3], output[11]) = tup{(d + n) * F(0.125), (d - n) * F(0.125)};
        tie(output[4], output[12]) = tup{(e - o) * F(0.125), (e + o) * F(0.125)};
        tie(output[5], output[13]) = tup{(f - p) * F(0.125), (f + p) * F(0.125)};
        tie(output[6], output[14]) = tup{(g - q) * F(0.125), (g + q) * F(0.125)};
        tie(output[7], output[15]) = tup{(h - r) * F(0.125), (h + r) * F(0.125)};
      }
      return;
    }
  }
}