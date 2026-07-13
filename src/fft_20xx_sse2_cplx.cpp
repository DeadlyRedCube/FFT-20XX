#ifdef _MSC_VER
  #pragma warning (disable: 4711) // function <x> selected for automatic inline expansion
#endif

#include "fft_20xx_common.h"
#include "fft_20xx_table.h"
#include "fft_20xx_tiny.h"

static_assert(CPU_X86 || CPU_X64);

static_assert(SIMD_SSE2);

#if COMPILER_GCC
  #pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

namespace FFT20XX::SSE2
{
  template <typename T> concept f32 = std::same_as<float, T> || std::same_as<__m128, T>;
  template <typename T> concept simd_vec = std::same_as<std::remove_cvref_t<T>, __m128> || std::same_as<std::remove_cvref_t<T>, __m128d>;
  template <typename T> using Vec = std::conditional_t<f32<T>, __m128, __m128d>;

  template <typename> constexpr uint32_t VecSize = 0;

  template <> constexpr uint32_t VecSize<Vec<float>> = 4u;
  template <> constexpr uint32_t VecSize<Vec<double>> = 2u;
  template <> constexpr uint32_t VecSize<float> = 4u;
  template <> constexpr uint32_t VecSize<double> = 2u;

  template <typename T> constexpr uint32_t ComplexVecSize = 2 * VecSize<T>;

  constexpr bool IsScalar = false;
  constexpr bool UseFMAOptimization = false;

  template <std::floating_point F>
  static auto LoadA(const F *ptr) -> Vec<F>
  {
    if constexpr (f32<F>)
      { return _mm_load_ps(ptr); }
    else
      { return _mm_load_pd(ptr); }
  }
  template <std::floating_point F>
  static auto Load(const F *ptr) -> Vec<F>
  {
    if constexpr (f32<F>)
      { return _mm_loadu_ps(ptr); }
    else
      { return _mm_loadu_pd(ptr); }
  }

  template <std::floating_point F>
  static auto Load4(const F *ptr) -> Vec<F>
    { return Load(ptr); }

  template <std::floating_point F>
  static auto LoadDeinterleaved(const F *ptr) -> Pair<Vec<F>, Vec<F>>
  {
    if constexpr (f32<F>)
    {
      auto t0 = _mm_loadu_ps(ptr);
      auto t1 = _mm_loadu_ps(ptr + 4u);
      return { _mm_shuffle_ps(t0, t1, _MM_SHUFFLE(2, 0, 2, 0)), _mm_shuffle_ps(t0, t1, _MM_SHUFFLE(3, 1, 3, 1)) };
    }
    else
    {
      auto t0 = _mm_loadu_pd(ptr);
      auto t1 = _mm_loadu_pd(ptr + 2u);
      return { _mm_shuffle_pd(t0, t1, _MM_SHUFFLE2(0, 0)), _mm_shuffle_pd(t0, t1, _MM_SHUFFLE2(1, 1)) };
    }
  }

  template <std::floating_point F>
  static void Store2x2Transposed(F *ptrA, F *ptrB, Vec<F> a, Vec<F> b)
  {
    auto aa = _mm_shuffle_pd(a, b, _MM_SHUFFLE2(0, 0));
    auto bb = _mm_shuffle_pd(a, b, _MM_SHUFFLE2(1, 1));
    _mm_storeu_pd(ptrA, aa);
    _mm_storeu_pd(ptrB, bb);
  }

  template <std::floating_point F>
  static void Store4x4Transposed(F *ptrA, F *ptrB, F *ptrC, F *ptrD, Vec<F> a, Vec<F> b, Vec<F> c, Vec<F> d)
  {
    static_assert(f32<F>);
    _MM_TRANSPOSE4_PS(a, b, c, d);
    _mm_storeu_ps(ptrA, a);
    _mm_storeu_ps(ptrB, b);
    _mm_storeu_ps(ptrC, c);
    _mm_storeu_ps(ptrD, d);
  }

  template <std::floating_point F>
  static void Store(F *ptr, Vec<F> a)
  {
    if constexpr (f32<F>)
      { _mm_storeu_ps(ptr, a); }
    else
      { _mm_storeu_pd(ptr, a); }
  }

  template <std::floating_point F>
  static void StoreInterleaved(F *ptr, Vec<F> a, Vec<F> b)
  {
    if constexpr (f32<F>)
    {
      auto aa = _mm_unpacklo_ps(a, b);
      auto bb = _mm_unpackhi_ps(a, b);
      _mm_storeu_ps(ptr, aa);
      _mm_storeu_ps(ptr + 4, bb);
    }
    else
    {
      auto aa = _mm_shuffle_pd(a, b, _MM_SHUFFLE2(0, 0));
      auto bb = _mm_shuffle_pd(a, b, _MM_SHUFFLE2(1, 1));
      _mm_storeu_pd(ptr, aa);
      _mm_storeu_pd(ptr + 2, bb);
    }
  }

  template <typename V, std::floating_point F>
  static auto Set1(F f) -> V
  {
    if constexpr (f32<F>)
      { return _mm_set1_ps(f); }
    else
      { return _mm_set1_pd(f); }
  }

  template <std::floating_point F, std::same_as<Vec<F>> ... Vs>
  static void Scale(F scale, Vs &...vs)
  {
    if constexpr (f32<F>)
    {
      auto s = _mm_set1_ps(scale);
      ((vs = _mm_mul_ps(s, vs)), ...);
    }
    else
    {
      auto s = _mm_set1_pd(scale);
      ((vs = _mm_mul_pd(s, vs)), ...);
    }
  }

  template <simd_vec V>
  static V Add(V a, V b)
  {
    if constexpr (f32<V>)
      { return _mm_add_ps(a, b); }
    else
      { return _mm_add_pd(a, b); }
  }

  template <simd_vec V>
  static V Sub(V a, V b)
  {
    if constexpr (f32<V>)
      { return _mm_sub_ps(a, b); }
    else
      { return _mm_sub_pd(a, b); }
  }

  template <simd_vec V>
  static V Mul(V a, V b)
  {
    if constexpr (f32<V>)
      { return _mm_mul_ps(a, b); }
    else
      { return _mm_mul_pd(a, b); }
  }

  template <simd_vec V>
  static V MAdd(V a, V b, V c)
    { return Add(Mul(a, b), c); }

  template <simd_vec V>
  static V NMAdd(V a, V b, V c)
    { return Sub(c, Mul(a, b)); }

  // Forward Real Transform ///////////////////////////////////////////////////////////////////////////////////////////

  template <typename F>
  inline auto LoadFirstRightSideForForwardRealTransform(const F *input, Vec<F> &prevR, Vec<F> &prevI)
    -> Pair<Vec<F>, Vec<F>>
  {
    if constexpr (f32<Vec<F>>)
    {
      prevR = _mm_loadu_ps(input);
      prevI = _mm_loadu_ps(input + VecSize<Vec<F>>);

      return
        {
          _mm_shuffle_ps(prevR, prevR, _MM_SHUFFLE(1, 2, 3, 0)),
          _mm_shuffle_ps(prevI, prevI, _MM_SHUFFLE(1, 2, 3, 0)),
        };
    }
    else
    {
      prevR = _mm_loadu_pd(input);
      prevI = _mm_loadu_pd(input + VecSize<Vec<F>>);

      return { prevR, prevI };
    }
  }

  template <typename F>
  inline auto LoadRightSideForForwardRealTransform(const F *input, Vec<F> &prevR, Vec<F> &prevI)
    -> Pair<Vec<F>, Vec<F>>
  {
    if constexpr (f32<Vec<F>>)
    {
      auto tr1 = _mm_loadu_ps(input);
      auto ti1 = _mm_loadu_ps(input + VecSize<Vec<F>>);

      #if SIMD_SSE4_1
        auto r1 = _mm_blend_ps(prevR, tr1, 0b1110);
        auto i1 = _mm_blend_ps(prevI, ti1, 0b1110);
      #elif 0
        auto mask = _mm_set_ps(
          std::bit_cast<float>(0xFFFFFFFF),
          std::bit_cast<float>(0xFFFFFFFF),
          std::bit_cast<float>(0xFFFFFFFF),
          std::bit_cast<float>(0));
        auto r1 = _mm_or_ps(_mm_andnot_ps(mask, prevR), _mm_and_ps(mask, tr1));
        auto i1 = _mm_or_ps(_mm_andnot_ps(mask, prevI), _mm_and_ps(mask, ti1));
      #else
        auto r1 = _mm_move_ss(tr1, prevR);
        auto i1 = _mm_move_ss(ti1, prevI);
      #endif

      prevR = tr1;
      prevI = ti1;

      return
        {
          _mm_shuffle_ps(r1, r1, _MM_SHUFFLE(1, 2, 3, 0)),
          _mm_shuffle_ps(i1, i1, _MM_SHUFFLE(1, 2, 3, 0)),
        };
    }
    else
    {
      auto tr1 = _mm_loadu_pd(input);
      auto ti1 = _mm_loadu_pd(input + VecSize<Vec<F>>);

      Pair r { _mm_shuffle_pd(prevR, tr1, _MM_SHUFFLE2(1, 0)), _mm_shuffle_pd(prevI, ti1, _MM_SHUFFLE2(1, 0)) };
      prevR = tr1;
      prevI = ti1;

      return r;
    }
  }

  template <typename F>
  inline void StoreRightSideForForwardRealTransform(F *output, Vec<F> r1, Vec<F> i1, Vec<F> &prevR, Vec<F> &prevI)
  {
    if constexpr (f32<F>)
    {
      // Prev: A B C D | a b c d (REAL, imaginary)
      // Current: E F G H | e f g h
      // Target: E e D d | C c B b

      auto ab = _mm_unpacklo_ps(prevR, prevI);
      auto cd = _mm_unpackhi_ps(prevR, prevI);
      prevR = r1;
      prevI = i1;

      auto ef = _mm_unpacklo_ps(prevR, prevI);

      auto ed = _mm_shuffle_ps(ef, cd, _MM_SHUFFLE(3, 2, 1, 0));
      auto cb = _mm_shuffle_ps(cd, ab, _MM_SHUFFLE(3, 2, 1, 0));
      _mm_storeu_ps(output, ed);
      _mm_storeu_ps(output + 4u, cb);
    }
    else
    {
      // Prev: A B | a b
      // Current: C D | c d
      // Target: C c | B b
      auto b = _mm_unpackhi_pd(prevR, prevI);
      prevR = r1;
      prevI = i1;

      auto c = _mm_unpacklo_pd(r1, i1);
      _mm_storeu_pd(output, c);
      _mm_storeu_pd(output + 2u, b);
    }
  }

  template <typename F>
  inline void StoreFinalRightSideForForwardRealTransform(F *output, Vec<F> r1, Vec<F> i1)
  {
    if constexpr (f32<F>)
    {
      // final: X N O P | X n o p  (X means "don't care")
      // target: X X P p | O o N n

      auto NOno = _mm_shuffle_ps(r1, i1, _MM_SHUFFLE(2, 1, 2, 1));
      auto xxPp = _mm_unpackhi_ps(r1, i1);
      auto OoNn = _mm_shuffle_ps(NOno, NOno, _MM_SHUFFLE(2, 0, 3, 1));
      _mm_storeu_ps(output, xxPp);
      _mm_storeu_ps(output + 4u, OoNn);
    }
    else
    {
      // Final: X N | X n
      // Target : X X | N n  (don't need to write the X X portion)
      _mm_storeu_pd(output + 2u, _mm_unpackhi_pd(r1, i1));
    }
  }

  // Inverse Real Transform ///////////////////////////////////////////////////////////////////////////////////////////

  template <typename F>
  inline auto LoadFirstRightSideForInverseRealTransform(const F *input, Vec<F> &prev, Vec<F> &) // only need one prev
    -> Pair<Vec<F>, Vec<F>>
  {
    if constexpr (f32<Vec<F>>)
    {
      auto ed = _mm_loadu_ps(input);
      auto cb = _mm_loadu_ps(input + 4u);
      prev = ed;

      auto cd = _mm_shuffle_ps(cb, ed, _MM_SHUFFLE(3, 2, 1, 0));
      return { _mm_shuffle_ps(cb, cd, _MM_SHUFFLE(2, 0, 2, 0)), _mm_shuffle_ps(cb, cd, _MM_SHUFFLE(3, 1, 3, 1)) };
    }
    else
    {
      auto c = _mm_loadu_pd(input);
      auto b = _mm_loadu_pd(input + 2u);
      prev = c;
      return { _mm_shuffle_pd(c, b, _MM_SHUFFLE2(0, 0)), _mm_shuffle_pd(c, b, _MM_SHUFFLE2(1, 1)) };
    }
  }

  template <typename F>
  inline auto LoadRightSideForInverseRealTransform(const F *input, Vec<F> &prev, Vec<F> &) // only need one prev
    -> Pair<Vec<F>, Vec<F>>
  {
    if constexpr (f32<Vec<F>>)
    {
      // Prev: E e D d
      // Cur: I i H h | G g F f
      // Target: E F G H | e f g h
      auto ih = _mm_loadu_ps(input);
      auto gf = _mm_loadu_ps(input + 4u);

      auto ef = _mm_shuffle_ps(prev, gf, _MM_SHUFFLE(3, 2, 1, 0));
      prev = ih;
      auto gh = _mm_shuffle_ps(gf, ih, _MM_SHUFFLE(3, 2, 1, 0));

      auto EFGH = _mm_shuffle_ps(ef, gh, _MM_SHUFFLE(2, 0, 2, 0));
      auto efgh = _mm_shuffle_ps(ef, gh, _MM_SHUFFLE(3, 1, 3, 1));

      return { EFGH, efgh };
    }
    else
    {
      // Prev: C c
      // Cur: E e | D d
      // Target: C D | c d
      auto e = _mm_loadu_pd(input);
      auto d = _mm_loadu_pd(input + 2u);

      auto CD = _mm_shuffle_pd(prev, d, _MM_SHUFFLE2(0, 0));
      auto cd = _mm_shuffle_pd(prev, d, _MM_SHUFFLE2(1, 1));
      prev = e;
      return { CD, cd };
    }
  }

  template <typename F>
  inline void StoreRightSideForInverseRealTransform(F *output, Vec<F> r1, Vec<F> i1, Vec<F> &prevR, Vec<F> &prevI)
  {
    if constexpr (f32<F>)
    {
      // Unfortunately, if we're not storing these interleaved, the blending logic is uglier (especially before there's
      //  a bespoke blend instruction)
      #if SIMD_SSE4_1
        // If we have a blend instruction
        auto tr1 = _mm_blend_ps(r1, prevR, 0b1110);
        auto ti1 = _mm_blend_ps(i1, prevI, 0b1110);
        prevR = r1;
        prevI = i1;
        tr1 = _mm_shuffle_ps(tr1, tr1, _MM_SHUFFLE(1, 2, 3, 0));
        ti1 = _mm_shuffle_ps(ti1, ti1, _MM_SHUFFLE(1, 2, 3, 0));

        _mm_storeu_ps(output, tr1);
        _mm_storeu_ps(output + 4u, ti1);
      #elif 1
        // If we have a blend instruction
        auto tr1 = _mm_move_ss(prevR, r1); // _mm_blend_ps(r1, prevR, 0b1110);
        auto ti1 = _mm_move_ss(prevI, i1); //_mm_blend_ps(i1, prevI, 0b1110);
        prevR = r1;
        prevI = i1;
        tr1 = _mm_shuffle_ps(tr1, tr1, _MM_SHUFFLE(1, 2, 3, 0));
        ti1 = _mm_shuffle_ps(ti1, ti1, _MM_SHUFFLE(1, 2, 3, 0));

        _mm_storeu_ps(output, tr1);
        _mm_storeu_ps(output + 4u, ti1);
      #else
        // prev: A B C D | a b c d
        // cur: E F G H | e f g h
        // Target: E D C B | e d c b

        // Most efficient here is to actually interleave, shuffle to combine, then deinterleave
        //  (other option would be to emulate the above blend instructions with `or(and, andnot)` but that's
        //  considerably more instructions and no shorter dependency chains)
        auto ab = _mm_unpacklo_ps(prevR, prevI);
        auto cd = _mm_unpackhi_ps(prevR, prevI);
        prevR = r1;
        prevI = i1;
        auto ef = _mm_unpacklo_ps(r1, i1);


        auto ed = _mm_shuffle_ps(ef, cd, _MM_SHUFFLE(3, 2, 1, 0));
        auto cb = _mm_shuffle_ps(cd, ab, _MM_SHUFFLE(3, 2, 1, 0));

        auto tr1 = _mm_shuffle_ps(ed, cb, _MM_SHUFFLE(2, 0, 2, 0));
        auto ti1 = _mm_shuffle_ps(ed, cb, _MM_SHUFFLE(3, 1, 3, 1));

        _mm_storeu_ps(output, tr1);
        _mm_storeu_ps(output + 4u, ti1);
      #endif
    }
    else
    {
      // Prev: A B
      // Cur: C D
      // Target: C B
      auto tr1 = _mm_shuffle_pd(r1, prevR, _MM_SHUFFLE2(1, 0));
      auto ti1 = _mm_shuffle_pd(i1, prevI, _MM_SHUFFLE2(1, 0));
      prevR = r1;
      prevI = i1;
      _mm_storeu_pd(output, tr1);
      _mm_storeu_pd(output + 2u, ti1);
    }
  }

  template <typename F>
  inline void StoreFinalRightSideForInverseRealTransform(F *output, Vec<F> r1, Vec<F> i1)
  {
    if constexpr (f32<F>)
    {
      // final: X N O P | X n o p  (X means "don't care")
      // target: X P O N | X p o n
      _mm_storeu_ps(output, _mm_shuffle_ps(r1, r1, _MM_SHUFFLE(1, 2, 3, 0)));
      _mm_storeu_ps(output + 4u, _mm_shuffle_ps(i1, i1, _MM_SHUFFLE(1, 2, 3, 0)));
    }
    else
    {
      // Conveniently this one is fine as is: X N | X n
      _mm_store_pd(output, r1);
      _mm_store_pd(output + 2u, i1);
    }
  }

  #include "fft_20xx_run.h"
}

using namespace FFT20XX;

namespace FFT20XX::SSE2New
{
  template <typename T> concept f32 = std::same_as<float, T> || std::same_as<__m128, T>;

  template <typename T> concept simd_vec = std::same_as<std::remove_cvref_t<T>, __m128> || std::same_as<std::remove_cvref_t<T>, __m128d>;
  template <typename T> using Vec = std::conditional_t<f32<T>, __m128, __m128d>;

  template <typename> constexpr uint32_t VecSize = 0;

  template <> constexpr uint32_t VecSize<Vec<float>> = 4u;
  template <> constexpr uint32_t VecSize<Vec<double>> = 2u;
  template <> constexpr uint32_t VecSize<float> = 4u;
  template <> constexpr uint32_t VecSize<double> = 2u;

  template <typename T>
  struct Angles
  {
    T w0;
    T w1;
    T w2;
  };


  template <std::floating_point F>
  bool IsSIMDAligned(const F *ptr)
    { return (uintptr_t(ptr) & (sizeof(__m128) - 1)) == 0; }


  template <std::floating_point F>
  FORCE_INLINE auto Set1(F f) -> Vec<F>
  {
    if constexpr (f32<F>)
      { return _mm_set1_ps(f); }
    else
      { return _mm_set1_pd(f); }
  }



  template <Alignment Align, std::floating_point F>
  FORCE_INLINE auto Load(const F *ptr) -> Vec<F>
  {
    if constexpr (f32<F>)
    {
      if constexpr (Align == Alignment::Aligned)
        { return _mm_load_ps(ptr); }
      else
        { return _mm_loadu_ps(ptr); }
    }
    else
    {
      if constexpr (Align == Alignment::Aligned)
        { return _mm_load_pd(ptr); }
      else
        { return _mm_loadu_pd(ptr); }
    }
  }

  template <Alignment Align, std::floating_point F>
  FORCE_INLINE void Store(F *ptr, Vec<F> v)
  {
    if constexpr (f32<F>)
    {
      if constexpr (Align == Alignment::Aligned)
        { return _mm_store_ps(ptr, v); }
      else
        { return _mm_storeu_ps(ptr, v); }
    }
    else
    {
      if constexpr (Align == Alignment::Aligned)
        { return _mm_store_pd(ptr, v); }
      else
        { return _mm_storeu_pd(ptr, v); }
    }
  }



  template <Alignment Align, std::floating_point F>
  FORCE_INLINE auto LoadUnpackedComplex(const F *ptr) -> ComplexVec<Vec<F>>
    { return { Load<Align>(ptr), Load<Align>(ptr + VecSize<F>) }; }


  template <Alignment Align, std::floating_point F>
  FORCE_INLINE void StoreUnpackedComplex(F *ptr, ComplexVec<Vec<F>> v)
  {
    Store<Align>(ptr, v.r);
    Store<Align>(ptr + VecSize<F>, v.i);
  }



  template <Alignment Align, std::floating_point F>
  FORCE_INLINE auto LoadPackedComplex(const F *ptr) -> ComplexVec<Vec<F>>
  {
    auto t0 = Load<Align>(ptr);
    auto t1 = Load<Align>(ptr + VecSize<F>);
    if constexpr (f32<F>)
      { return { _mm_shuffle_ps(t0, t1, _MM_SHUFFLE(2, 0, 2, 0)), _mm_shuffle_ps(t0, t1, _MM_SHUFFLE(3, 1, 3, 1)) }; }
    else
      { return { _mm_shuffle_pd(t0, t1, _MM_SHUFFLE2(0, 0)), _mm_shuffle_pd(t0, t1, _MM_SHUFFLE2(1, 1)) }; }
  }



  template <Alignment Align, std::floating_point F>
  FORCE_INLINE void StorePackedComplex(F *ptr, ComplexVec<Vec<F>> v)
  {
    if constexpr (f32<F>)
    {
      auto aa = _mm_unpacklo_ps(v.r, v.i);
      auto bb = _mm_unpackhi_ps(v.r, v.i);
      Store<Align>(ptr, aa);
      Store<Align>(ptr + 4, bb);
    }
    else
    {
      auto aa = _mm_shuffle_pd(v.r, v.i, _MM_SHUFFLE2(0, 0));
      auto bb = _mm_shuffle_pd(v.r, v.i, _MM_SHUFFLE2(1, 1));
      Store<Align>(ptr, aa);
      Store<Align>(ptr + 2, bb);
    }
  }





  template <simd_vec V>
  FORCE_INLINE void AddSub(ComplexVec<V> &a, ComplexVec<V> &b)
  {
    if constexpr (f32<V>)
    {
      auto t = _mm_add_ps(a.r, b.r);
      b.r = _mm_sub_ps(a.r, b.r);
      a.r = t;
      t = _mm_add_ps(a.i, b.i);
      b.i = _mm_sub_ps(a.i, b.i);
      a.i = t;
    }
    else
    {
      auto t = _mm_add_pd(a.r, b.r);
      b.r = _mm_sub_pd(a.r, b.r);
      a.r = t;
      t = _mm_add_pd(a.i, b.i);
      b.i = _mm_sub_pd(a.i, b.i);
      a.i = t;
    }
  }



  template <Dir D, simd_vec V>
  FORCE_INLINE void AddSubRot45(ComplexVec<V> &a, ComplexVec<V> &b)
  {
    if constexpr (f32<V>)
    {
      const auto s = _mm_set1_ps(InvSqrt2<float>);
      if constexpr (D == Dir::Forward)
      {
        auto t = _mm_add_ps(b.r, b.i);
        b.i = _mm_sub_ps(b.i, b.r);
        b.r = _mm_mul_ps(s, t);
        b.i = _mm_mul_ps(s, b.i);
      }
      else
      {
        auto t = _mm_sub_ps(b.r, b.i);
        b.i = _mm_add_ps(b.i, b.r);
        b.r = _mm_mul_ps(s, t);
        b.i = _mm_mul_ps(s, b.i);
      }

      auto t = _mm_add_ps(a.r, b.r);
      b.r = _mm_sub_ps(a.r, b.r);
      a.r = t;
      t = _mm_add_ps(a.i, b.i);
      b.i = _mm_sub_ps(a.i, b.i);
      a.i = t;
    }
    else
    {
      const auto s = _mm_set1_pd(InvSqrt2<double>);
      if constexpr (D == Dir::Forward)
      {
        auto t = _mm_add_pd(b.r, b.i);
        b.i = _mm_sub_pd(b.i, b.r);
        b.r = _mm_mul_pd(s, t);
        b.i = _mm_mul_pd(s, b.i);
      }
      else
      {
        auto t = _mm_sub_pd(b.r, b.i);
        b.i = _mm_add_pd(b.i, b.r);
        b.r = _mm_mul_pd(s, t);
        b.i = _mm_mul_pd(s, b.i);
      }

      auto t = _mm_add_pd(a.r, b.r);
      b.r = _mm_sub_pd(a.r, b.r);
      a.r = t;
      t = _mm_add_pd(a.i, b.i);
      b.i = _mm_sub_pd(a.i, b.i);
      a.i = t;
    }
  }



  template <Dir D, simd_vec V>
  FORCE_INLINE void AddSubRot90(ComplexVec<V> &a, ComplexVec<V> &b)
  {
    if constexpr (f32<V>)
    {
      if constexpr (D == Dir::Forward)
      {
        auto t = _mm_add_ps(a.r, b.i);
        b.i = _mm_sub_ps(a.r, b.i);
        a.r = t;
        t = _mm_sub_ps(a.i, b.r);
        b.r = _mm_add_ps(a.i, b.r);
        a.i = t;
      }
      else
      {
        auto t = _mm_sub_ps(a.r, b.i);
        b.i = _mm_add_ps(a.r, b.i);
        a.r = t;
        t = _mm_add_ps(a.i, b.r);
        b.r = _mm_sub_ps(a.i, b.r);
        a.i = t;
      }
    }
    else
    {
      if constexpr (D == Dir::Forward)
      {
        auto t = _mm_add_pd(a.r, b.i);
        b.i = _mm_sub_pd(a.r, b.i);
        a.r = t;
        t = _mm_sub_pd(a.i, b.r);
        b.r = _mm_add_pd(a.i, b.r);
        a.i = t;
      }
      else
      {
        auto t = _mm_sub_pd(a.r, b.i);
        b.i = _mm_add_pd(a.r, b.i);
        a.r = t;
        t = _mm_add_pd(a.i, b.r);
        b.r = _mm_sub_pd(a.i, b.r);
        a.i = t;
      }
    }
    std::swap(b.r, b.i);
  }



  template <Dir D, simd_vec V>
  FORCE_INLINE void AddSubRot135(ComplexVec<V> &a, ComplexVec<V> &b)
  {
    if constexpr (f32<V>)
    {
      const auto s = _mm_set1_ps(InvSqrt2<float>);
      if constexpr (D == Dir::Forward)
      {
        auto t = _mm_sub_ps(b.i, b.r);
        b.i = _mm_add_ps(b.i, b.r); // This is actually -b.i
        b.r = _mm_mul_ps(s, t);
        b.i = _mm_mul_ps(s, b.i);

        t = _mm_add_ps(a.r, b.r);
        b.r = _mm_sub_ps(a.r, b.r);
        a.r = t;
        t = _mm_sub_ps(a.i, b.i); // Because b.i is -b.i, addition is subtraction!
        b.i = _mm_add_ps(a.i, b.i);
        a.i = t;
      }
      else
      {
        auto t = _mm_add_ps(b.i, b.r); // This is actually -b.r
        b.i = _mm_sub_ps(b.r, b.i);
        b.r = _mm_mul_ps(s, t);
        b.i = _mm_mul_ps(s, b.i);

        t = _mm_sub_ps(a.r, b.r); // because b.r is actually -b.r, swap the addition/subtraction
        b.r = _mm_add_ps(a.r, b.r);
        a.r = t;
        t = _mm_add_ps(a.i, b.i);
        b.i = _mm_sub_ps(a.i, b.i);
        a.i = t;
      }
    }
    else
    {
      const auto s = _mm_set1_pd(InvSqrt2<double>);
      if constexpr (D == Dir::Forward)
      {
        auto t = _mm_sub_pd(b.i, b.r);
        b.i = _mm_add_pd(b.i, b.r); // This is actually -b.i
        b.r = _mm_mul_pd(s, t);
        b.i = _mm_mul_pd(s, b.i);

        t = _mm_add_pd(a.r, b.r);
        b.r = _mm_sub_pd(a.r, b.r);
        a.r = t;
        t = _mm_sub_pd(a.i, b.i); // Because b.i is -b.i, addition is subtraction!
        b.i = _mm_add_pd(a.i, b.i);
        a.i = t;
      }
      else
      {
        auto t = _mm_add_pd(b.i, b.r); // This is actually -b.r
        b.i = _mm_sub_pd(b.r, b.i);
        b.r = _mm_mul_pd(s, t);
        b.i = _mm_mul_pd(s, b.i);

        t = _mm_sub_pd(a.r, b.r); // because b.r is actually -b.r, swap the addition/subtraction
        b.r = _mm_add_pd(a.r, b.r);
        a.r = t;
        t = _mm_add_pd(a.i, b.i);
        b.i = _mm_sub_pd(a.i, b.i);
        a.i = t;
      }
    }
  }


  template <simd_vec V>
  [[nodiscard]] FORCE_INLINE auto ComplexMultiply(ComplexVec<V> a, ComplexVec<V> b) -> ComplexVec<V>
  {
    if constexpr (f32<V>)
    {
      auto t = _mm_sub_ps(_mm_mul_ps(a.r, b.r), _mm_mul_ps(a.i, b.i));
      a.i = _mm_add_ps(_mm_mul_ps(a.r, b.i), _mm_mul_ps(a.i, b.r));
      a.r = t;
      return a;
    }
    else
    {
      auto t = _mm_sub_pd(_mm_mul_pd(a.r, b.r), _mm_mul_pd(a.i, b.i));
      a.i = _mm_add_pd(_mm_mul_pd(a.r, b.i), _mm_mul_pd(a.i, b.r));
      a.r = t;
      return a;
    }
  }

  template <simd_vec V>
  [[nodiscard]] FORCE_INLINE auto ScalarMultiply(ComplexVec<V> a, V b) -> ComplexVec<V>
  {
    if constexpr (f32<V>)
      { return { _mm_mul_ps(a.r, b), _mm_mul_ps(a.i, b) }; }
    else
      { return { _mm_mul_pd(a.r, b), _mm_mul_pd(a.i, b) }; }
  }

  template <Dir D, simd_vec V>
  FORCE_INLINE void Rotate(ComplexVec<V> &a, ComplexVec<V> w)
  {
    if constexpr (f32<V>)
    {
      if constexpr (D == Dir::Forward)
      {
        auto t = _mm_sub_ps(_mm_mul_ps(a.r, w.r), _mm_mul_ps(a.i, w.i));
        a.i = _mm_add_ps(_mm_mul_ps(a.r, w.i), _mm_mul_ps(a.i, w.r));
        a.r = t;
      }
      else
      {
        auto t = _mm_add_ps(_mm_mul_ps(a.r, w.r), _mm_mul_ps(a.i, w.i));
        a.i = _mm_sub_ps(_mm_mul_ps(a.i, w.r), _mm_mul_ps(a.r, w.i));
        a.r = t;
      }
    }
    else
    {
      if constexpr (D == Dir::Forward)
      {
        auto t = _mm_sub_pd(_mm_mul_pd(a.r, w.r), _mm_mul_pd(a.i, w.i));
        a.i = _mm_add_pd(_mm_mul_pd(a.r, w.i), _mm_mul_pd(a.i, w.r));
        a.r = t;
      }
      else
      {
        auto t = _mm_add_pd(_mm_mul_pd(a.r, w.r), _mm_mul_pd(a.i, w.i));
        a.i = _mm_sub_pd(_mm_mul_pd(a.i, w.r), _mm_mul_pd(a.r, w.i));
        a.r = t;
      }
    }
  }


  template <Dir D, simd_vec V, std::floating_point F>
  FORCE_INLINE void Rotate(ComplexVec<V> &a, Complex<F> w)
  {
    if constexpr (f32<V>)
    {
      auto wR = _mm_set1_ps(w.r);
      auto wI = _mm_set1_ps(w.i);
      if constexpr (D == Dir::Forward)
      {
        auto t = _mm_sub_ps(_mm_mul_ps(a.r, wR), _mm_mul_ps(a.i, wI));
        a.i = _mm_add_ps(_mm_mul_ps(a.r, wI), _mm_mul_ps(a.i, wR));
        a.r = t;
      }
      else
      {
        auto t = _mm_add_ps(_mm_mul_ps(a.r, wR), _mm_mul_ps(a.i, wI));
        a.i = _mm_sub_ps(_mm_mul_ps(a.i, wR), _mm_mul_ps(a.r, wI));
        a.r = t;
      }
    }
    else
    {
      auto wR = _mm_set1_pd(w.r);
      auto wI = _mm_set1_pd(w.i);
      if constexpr (D == Dir::Forward)
      {
        auto t = _mm_sub_pd(_mm_mul_pd(a.r, wR), _mm_mul_pd(a.i, wI));
        a.i = _mm_add_pd(_mm_mul_pd(a.r, wI), _mm_mul_pd(a.i, wR));
        a.r = t;
      }
      else
      {
        auto t = _mm_add_pd(_mm_mul_pd(a.r, wR), _mm_mul_pd(a.i, wI));
        a.i = _mm_sub_pd(_mm_mul_pd(a.i, wR), _mm_mul_pd(a.r, wI));
        a.r = t;
      }
    }
  }



  FORCE_INLINE void Transpose4x4(
    ComplexVec<__m128> &a,
    ComplexVec<__m128> &b,
    ComplexVec<__m128> &c,
    ComplexVec<__m128> &d)
  {
    _MM_TRANSPOSE4_PS(a.r, b.r, c.r, d.r);
    _MM_TRANSPOSE4_PS(a.i, b.i, c.i, d.i);
  }

  FORCE_INLINE void Transpose2x2(
    ComplexVec<__m128d> &a,
    ComplexVec<__m128d> &b)
  {
    auto t = _mm_shuffle_pd(a.r, b.r, _MM_SHUFFLE2(0, 0));
    b.r = _mm_shuffle_pd(a.r, b.r, _MM_SHUFFLE2(1, 1));
    a.r = t;

    t = _mm_shuffle_pd(a.i, b.i, _MM_SHUFFLE2(0,0));
    b.i = _mm_shuffle_pd(a.i, b.i, _MM_SHUFFLE2(1, 1));
    a.i = t;
  }

  #include "fft_20xx_algo_cplx.h"
}


using namespace FFT20XX;
using namespace fft_20xx::c;
extern "C"
{
  void fft_cplx_interleaved_f32_sse2(const fft_table_f32 *table, const float *input, float *output, size_t fftLength)
  {
    [[maybe_unused]] auto t = reinterpret_cast<const FFTTableImpl<float> *>(table);
    COMMON_ASSERTS();
    if (SSE2New::IsSIMDAligned(input) && SSE2New::IsSIMDAligned(output))
      { SSE2New::RunComplex<Dir::Forward, DataType::Complex, Alignment::Aligned>(t->Data(), input, output, fftLength); }
    else
      { SSE2New::RunComplex<Dir::Forward, DataType::Complex, Alignment::Unaligned>(t->Data(), input, output, fftLength); }
  }


  void fft_cplx_interleaved_f64_sse2(const fft_table_f64 *table, const double *input, double *output, size_t fftLength)
  {
    [[maybe_unused]] auto t = reinterpret_cast<const FFTTableImpl<double> *>(table);
    COMMON_ASSERTS();
    if (SSE2New::IsSIMDAligned(input) && SSE2New::IsSIMDAligned(output))
      { SSE2New::RunComplex<Dir::Forward, DataType::Complex, Alignment::Aligned>(t->Data(), input, output, fftLength); }
    else
      { SSE2New::RunComplex<Dir::Forward, DataType::Complex, Alignment::Unaligned>(t->Data(), input, output, fftLength); }
  }


  void ifft_cplx_interleaved_f32_sse2(const fft_table_f32 *table, const float *input, float *output, size_t fftLength)
  {
    [[maybe_unused]] auto t = reinterpret_cast<const FFTTableImpl<float> *>(table);
    COMMON_ASSERTS();
    if (SSE2New::IsSIMDAligned(input) && SSE2New::IsSIMDAligned(output))
      { SSE2New::RunComplex<Dir::Inverse, DataType::Complex, Alignment::Aligned>(t->Data(), input, output, fftLength); }
    else
      { SSE2New::RunComplex<Dir::Inverse, DataType::Complex, Alignment::Unaligned>(t->Data(), input, output, fftLength); }
  }


  void ifft_cplx_interleaved_f64_sse2(const fft_table_f64 *table, const double *input, double *output, size_t fftLength)
  {
    [[maybe_unused]] auto t = reinterpret_cast<const FFTTableImpl<double> *>(table);
    COMMON_ASSERTS();
    if (SSE2New::IsSIMDAligned(input) && SSE2New::IsSIMDAligned(output))
      { SSE2New::RunComplex<Dir::Inverse, DataType::Complex, Alignment::Aligned>(t->Data(), input, output, fftLength); }
    else
      { SSE2New::RunComplex<Dir::Inverse, DataType::Complex, Alignment::Unaligned>(t->Data(), input, output, fftLength); }
  }

  // !!! move these out as they get implemented

  void fft_real_f64_sse2(const fft_table_f64 *table, const double *input, double *output, size_t fftLength)
  {
    auto t = reinterpret_cast<const FFTTableImpl<double> *>(table);
    COMMON_ASSERTS();
    SSE2::RunRealFwdFFT(t->Data(), input, output, fftLength);
  }

  void ifft_real_f64_sse2(const fft_table_f64 *table, const double *input, double *output, size_t fftLength)
  {
    auto t = reinterpret_cast<const FFTTableImpl<double> *>(table);
    COMMON_ASSERTS();
    SSE2::RunRealInvFFT(t->Data(), input, output, fftLength);
  }
}
