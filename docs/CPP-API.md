# FFT-20XX C++ API

This documents the C++ API for FFT-20XX. If you're using C, please refer to the [C documentation](C-API.md)

#  Contents
- [Overview](#overview)
  - [Example: Without Tables](#example-without-tables)
  - [Example: Allocated Table](#exmaple-allocated-table)
  - [Example: User-Managed Table](#example-user-managed-table)
- [Type](#type)
- [Tables](#tables)
- [FFT Functions](#fft-functions)
- [Utilities](#utilities)
  - [Aligned Allocation](#aligned-allocation)
  - [Testing SIMD Support](#testing-simd-support)
  - [Flushing Subnormals](#flushing-subnormals)

&nbsp;

&nbsp;

# Overview

FFT-20XX was designed to be simple to use. The processing functions make use of read-only [FFT Tables](#table-types),
which can be used in a few different ways, depending on use case:
- FFT-20XX can internally manage the table memory and allocate larger tables as-needed: [Example](#example-without-tables)
- The user can pre-make a table with the maximum required FFT length at a time of their choosing:
  [Example](#example-allocated-table)
- The user can manage their own memory allocation(s) and fill that memory with table data: [Example](#example-user-managed-table)
  - This last one is useful if the platform has special allocation requirements/restrictions.

&nbsp;

&nbsp;

## Example: Without Tables

Using FFT-20XX without tables is straightforward:
```cpp
#include "fft_20xx/fft_20xx.hpp"

void run_cplx_fft(const float *input, float *output, size_t length)
{
  // No table specified.
  fft_20xx::fft<cplx>(input, output, length);
}

void run_cplx_fft_in_place(float *data, size_t length)
{
  fft_20xx::fft<cplx>(data, length);
}
```

&nbsp;

&nbsp;

## Example: Allocated Table

```cpp
#include "fft_20xx/fft_20xx.h"

constexpr size_t max_supported_fft_length = 2048;

// Initialize a table for complex 32-bit-float FFTs that can
// handle up to the max length.
fft_20xx::fft_table<cplx, float> g_table{max_supported_fft_length};

void run_cplx_fft(const float *input, float *output, size_t length)
{
  // Note that here the `type` does not need to be specified directly,
  // as the table type implies the FFT type.
  fft_20xx::fft(g_table, input, output, length);
}
```

&nbsp;

&nbsp;

## Example: User-Managed Table

```cpp
#include "fft_20xx/fft_20xx.h"

// A wildly oversized block of memory, just for demonstration.
uint8_t giantBlockOfMemory[1024*1024];

// Initialize this table, using the giant block of memory (which should be large enough to hold our max size)
constexpr size_t max_supported_fft_length = 2048;
fft_20xx::fft_table<cplx, float> g_table{
  max_supported_fft_length,
  giantBlockOfMemory,
  std::size(giantBlockOfMemory)};

void run_cplx_fft(const float *input, float *output, size_t length)
{
  fft_20xx::fft(g_table, input, output, length);
}
```

# Type

---
## `type`

An enumeration of the two types of FFT: complex or real. Used as template parameters for the FFT functions as well as
the table

If Compiling with C++20:
```cpp
enum class type
{
  cplx,
  real,
}

// Allow using fft_20xx::cplx and fft_20xx::real directly.
using enum type;
```

Otherwise, in C++17 and earlier:
```cpp
enum type
{
  cplx,
  real,
}
```

&nbsp;

&nbsp;


# Tables

The C++ API has a `fft_table` type that encompasses all of the available FFT table types:

---
### `fft_table`:
```cpp
template <type T, /*floating-point*/ F>
class fft_table
{
public:
  using c_table_t = /* Equivalent C-API table type*/;

  static size_t get_required_alignment();
  static size_t get_byte_count_aligned(size_t max_fft_length);
  static size_t get_byte_count_unaligned(size_t max_fft_length);

  fft_table(size_t max_length);
  fft_table(size_t max_length, void *ptr, size_t byte_count);

  fft_table(const fft_table &) = delete;
  fft_table &operator=(const fft_table &) = delete;
  fft_table(fft_table &&other) noexcept;
  fft_table &operator=(fft_table &&other) noexcept;

  ~fft_table();

  const c_table_t *c_table() const;
  operator const c_table_t *() const;
}
```

&nbsp;

&nbsp;

---
### `fft_table::get_required_alignment`

Static function that gets the required alignment for a SIMD-aligned user-specified memory pointer.
```cpp
static size_t get_required_alignment()
```

### Return Value:
- Returns the system's SIMD alignment.
&nbsp;

&nbsp;

---
### `fft_table::get_byte_count_aligned`

Returns the count of bytes needed to initialize an FFT table with the given maximum length, assuming the memory block
is correctly SIMD-aligned.

```cpp
static size_t get_byte_count_aligned(size_t max_fft_length)
```

### Parameters:
- `max_fft_length`: The maximum FFT length that the table would need to support. **Must be a power of 2.**

### Return value:
- Returns the number of bytes needed for an FFT table at a SIMD-aligned address.

&nbsp;

&nbsp;

---
### `fft_table::get_byte_count_unaligned`

Returns the count of bytes needed to initialize an FFT table with the given maximum length, assuming the memory block
is *not* SIMD-aligned.

```cpp
static size_t get_byte_count_unaligned(size_t max_fft_length)
```

### Parameters:
- `max_length`: The maximum FFT length that the table would need to support. **Must be a power of 2.**

### Return Value:
- Returns the number of bytes needed for an FFT table at an unaligned address.


&nbsp;

&nbsp;

---
### `fft_table::(constructor)`

Overloads:

1.
   ```cpp
   fft_table(size_t max_length)
   ```
2.
   ```cpp
   fft_table(size_t max_length, void *ptr, size_t byte_count)
   ```
3.
   ```cpp
   fft_table(fft_table &&other) noexcept
   ```
&ensp;**Overload 1**: Allocate and initialize an internal table of the given max FFT length

&ensp;**Overload 2**: Initialize a table of the given max FFT length, using the supplied memory block.

&ensp;**Overload 3**: move construct from another instance of `fft_table`.

### Parameters:
- `max_length`: The maximum FFT length that the table needs to support. **Must be a power of 2.**
- `ptr`: A pointer to user-managed memory to use for the table.
- `byte_count`: The number of bytes in the supplied block of user-managed memory.

> ***NOTE***: When constructing with user-managed memory, no allocations or frees will be performed.
  Otherwise, the constructor will allocate a table, and the corresponding [destructor](#fft_tabledestructor) will free it.

&nbsp;

&nbsp;

---
### `fft_table::(destructor)`

If the table was constructed without user-managed memory, this will free the internally-allocated table.

&nbsp;

&nbsp;

---
### `fft_table::operator=`

Standard move-assignment operator.
```cpp
fft_table &operator=(fft_table &&other) noexcept
```

&nbsp;

&nbsp;


---
### `fft_table::c_table`

Get a pointer to the C API version of this table (for passing into the C API functions, if needed).

```cpp
const c_table_t *c_table() const
```

### Return Value:
- A pointer to the internal C API table.

&nbsp;

&nbsp;

---
### `fft_table::operator const c_table_t *`

Allow auto-conversion to a pointer to the C API version of the table. Equivalent to calling [`c_table()`](fft_tablec_table)

```cpp
operator const c_table_t *() const
```

### Return Value:
- A pointer to the internal C API table.


&nbsp;

&nbsp;

# FFT Functions

The functions in this section perform the actual FFT (or inverse FFT) computations. Note that, in all cases:
- There are table-less overloads that allow FFT-20XX to internally manage a (global,
  thread-safe) table allocated for the given FFT type.
- if a table is specified, no memory will be allocated or freed during the runtime of the function.
- The `input` and `output` pointers can be the same pointer, in which case the (I)FFT will be performed in-place.
  - There are also single-pointer overloads that do the (I)FFT in place.
- No additional scratch memory is needed (the only mutable memory required is the output array).

&nbsp;

&nbsp;

---
## `fft`

Perform an FFT of the given length.

If C++20 is being used, the `/* floating_point*/` declarations will use the [`std::floating_point` concept](https://en.cppreference.com/cpp/concepts/floating_point), otherwise it will be `typename` and a `static_assert` is used to enforce that the type is
floating-point instead.

If this is a `real`-type FFT, it writes the first half of the complex
frequency-domain samples, with the DC and Nyquist terms (which are both always real-valued) packed into the first two
floats, like so:
>  `DC`, `Nyquist`, `real1`, `imag1`, `real2`, `imag2`, ...

Overloads:

1.
   ```cpp
   template <type T, /* floating_point */ F>
   void fft(const fft_table<T, F> &table, const F *input, F *output, size_t length)
   ```
2.
   ```cpp
   template <type T, /* floating_point */ F>
   void fft(const fft_table<T, F> &, F *data, size_t length)
   ```
3.
   ```cpp
   template <type T, /* floating_point */ F>
   void fft(const F *input, F *output, size_t length)
   ```
4.
   ```cpp
   template <type T, /* floating_point */ F>
   void fft(F *data, size_t length)
   ```
5.
   ```cpp
   template <FFT_20XX_FLOAT_TYPE F>
   void fft(const fft_table<type::cplx, F> &,
            const std::complex<F> *input,
            std::complex<F> *output,
            size_t length)
   ```
6.
   ```cpp
   template <FFT_20XX_FLOAT_TYPE F>
   void fft(const fft_table<type::cplx, F> &, std::complex<F> *data, size_t length)
   ```
7.
   ```cpp
   template <FFT_20XX_FLOAT_TYPE F>
   void fft(const std::complex<F> *input, std::complex<F> *output, size_t length);
   ```
8.
   ```cpp
   template <FFT_20XX_FLOAT_TYPE F>
   void fft(std::complex<F> *data, size_t length)
   ```

&ensp;**Overload 1**: Run an FFT of the given type (real or complex), given a table and separate input and output pointers.

&ensp;**Overload 2**: Same as 1, except only one pointer (`data`) is provided, which is both the input *and* output.

&ensp;**Overloads 3 & 4**: Same as 1 & 2, except with no table specified (so the table is managed internally by FFT-20XX)

&ensp;**Overload 5**: Same as 1, except it takes `std::complex` pointers instead of standard float parameters (and thus
cannot be used for `real`-type FFTs).

&ensp;**Overload 6**: same as 2, except it takes a `std::complex` `data` pointer, which is both the input
and the output.

&ensp;**Overloads 7 & 8**: Same as 5 & 6, except with no table specified (so the table is managed internally by FFT-20XX).

### Parameters:
- `table`: The FFT table to use. Must have been created with a maximum length of at least `length`.
- `input`: The time-domain input samples to use.
  - If this is a `real`-type FFT or it's complex and takes a `std::complex` array, there are expected to be `length`
    entries in this array.
  - Otherwise, for `cplx`-type FFTs with floating-point input arrays, there are expected to be `length*2` entries.
- `output`: The frequency-domain output samples.
  - The length of this array is expected to match that of `input`.
  - If this is a `real`-type FFT, there are effectively `length/2 + 1` samples in this output:
    - The first float is the DC value (corresponding to `output[0]` of the full, complex-valued FFT)
    - The second float is the Nyquist value (corresponding to `output[length]` of the full, complex-valued FFT)
    - The remaining values are **packed complex pairs** corresponding to the frequencies below the Nyquist frequency
      (corresponding to the complex values in `output[2]` through `output[length-1]` of the full, complex-valued FFT)
- `data`: The samples that will be time-domain before calling `fft` and frequency-domain after.
  - Like `input`, if this is a `real`-type FFT or it's complex and takes a `std::complex` array, there are expected to be `length`
    entries in this array.
  - Otherwise, for `cplx`-type FFTs with floating-point input arrays, there are expected to be `length*2` entries.
- `length`: The length of the FFT to perform. **Must be a power of 2!**


&nbsp;

&nbsp;

---
## `ifft`

Perform an inverse FFT (IFFT) of the given length.

If C++20 is being used, the `/* floating_point*/` declarations will use the [`std::floating_point` concept](https://en.cppreference.com/cpp/concepts/floating_point), otherwise it will be `typename` and a `static_assert` is used to enforce that the type is
floating-point instead.

Overloads:

1.
   ```cpp
   template <type T, /* floating_point */ F>
   void ifft(const fft_table<T, F> &table, const F *input, F *output, size_t length)
   ```
2.
   ```cpp
   template <type T, /* floating_point */ F>
   void ifft(const fft_table<T, F> &, F *data, size_t length)
   ```
3.
   ```cpp
   template <type T, /* floating_point */ F>
   void ifft(const F *input, F *output, size_t length)
   ```
4.
   ```cpp
   template <type T, /* floating_point */ F>
   void ifft(F *data, size_t length)
   ```
5.
   ```cpp
   template <FFT_20XX_FLOAT_TYPE F>
   void ifft(const fft_table<type::cplx, F> &,
            const std::complex<F> *input,
            std::complex<F> *output,
            size_t length)
   ```
6.
   ```cpp
   template <FFT_20XX_FLOAT_TYPE F>
   void ifft(const fft_table<type::cplx, F> &, std::complex<F> *data, size_t length)
   ```
7.
   ```cpp
   template <FFT_20XX_FLOAT_TYPE F>
   void ifft(const std::complex<F> *input, std::complex<F> *output, size_t length);
   ```
8.
   ```cpp
   template <FFT_20XX_FLOAT_TYPE F>
   void ifft(std::complex<F> *data, size_t length)
   ```
&ensp;**Overload 1**: Run an inverse FFT of the given type (real or complex), given a table and separate input and output pointers.

&ensp;**Overload 2**: Same as 1, except only one pointer (`data`) is provided, which is both the input *and* output.

&ensp;**Overloads 3 & 4**: Same as 1 & 2, except with no table specified (so the table is managed internally by FFT-20XX)

&ensp;**Overload 5**: Same as 1, except it takes `std::complex` pointers instead of standard float parameters (and thus
cannot be used for `real`-type FFTs).

&ensp;**Overload 6**: same as 2, except it takes a `std::complex` `data` pointer, which is both the input
and the output.

&ensp;**Overloads 7 & 8**: Same as 5 & 6, except with no table specified (so the table is managed internally by FFT-20XX).

### Parameters:
- `table`: The FFT table to use. Must have been created with a maximum length of at least `length`.
- `input`: The frequency-domain input samples to use.
  - If this is a `real`-type FFT or it's complex and takes a `std::complex` array, there are expected to be `length`
    entries in this array.
  - Otherwise, for `cplx`-type FFTs with floating-point input arrays, there are expected to be `length*2` entries.
  - The format of this array matches the format of the output of the corresponding [`fft`](#fft) function.
- `output`: The time-domain output samples.
  - The length of this array is expected to match that of `input`.
- `data`: The samples that will be frequency-domain before calling `ifft` and time-domain after.
  - Like `input`, if this is a `real`-type FFT or it's complex and takes a `std::complex` array, there are expected to be `length`
    entries in this array.
  - Otherwise, for `cplx`-type FFTs with floating-point input arrays, there are expected to be `length*2` entries.
- `length`: The length of the FFT to perform. **Must be a power of 2!**

&nbsp;

&nbsp;

# Utilities

The following utilities are in **`fft_20xx/fft_20xx_util.hpp`**, so include that to use these.

As with the entries above, these are all in the `fft_20xx` namespace.

&nbsp;

&nbsp;

## Aligned Allocation

These utility functions support the allocation and deallocation of *aligned* memory (useful for, for instance,
allocating memory for use with SIMD functions).

&nbsp;

&nbsp;

---
### `allocate_bytes_aligned`

Allocates a given number of bytes, with a given memory alignment.

```cpp
void *allocate_bytes_aligned(size_t byte_count, size_t pow_2_alignment)
```

### Parameters:
- `byte_count`: The number of bytes to allocate. **Must be an integral multiple of `pow_2_alignment`**.
- `pow_2_alignment`: The required alignment of the allocation. **Must be a power of two**.

### Return Value:
- On success, returns a pointer to the beginning of the newly-allocated memory.
- On failure, returns a null pointer.

> ***NOTE***: To avoid memory leaks, the returned pointer must be freed using [`free_aligned`](#free_aligned)

&nbsp;

&nbsp;

---
### `allocate_bytes_simd_aligned`

Allocates a given number of bytes, aligned to the system's maximum SIMD vector size. Useful for if you know you want
your memory to be SIMD-aligned but don't know the precise alignment of the SIMD types on the target platform.

```cpp
void *allocate_bytes_simd_aligned(size_t byte_count)
```

### Parameters:
- `byte_count`: The number of bytes to allocate. **Must be an integral multiple of `pow_2_alignment`**.

### Return Value:
- On success, returns a pointer to the beginning of the newly-allocated memory.
- On failure, returns a null pointer.

> ***NOTE***: To avoid memory leaks, the returned pointer must be freed using [`free_aligned`](#free_aligned)

&nbsp;

&nbsp;

---
### `allocate_trivial_type_simd_aligned`

Allocates a given number of elements of type `T`, aligned to the system's maximum SIMD vector size. Useful for allocating
a block of (for example) `float`s to be SIMD-aligned but don't know the precise alignment of the SIMD types on the target
platform.

> **NOTE**: The given type T must be:
> * Trivially constructible
> * Trivially destructible
>
> Otherwise, this will fail to compiile due to a `static_assert` (because constructors/destructors are not handled).

```cpp
template <typename T>
T *allocate_bytes_simd_aligned(size_t element_count)
```

### Parameters:
- `element_count`: The number of elements of type `T` to allocate. **Must be an integral multiple of
  the system's SIMD alignment (divided by `sizeof(T)`)**.

### Return Value:
- On success, returns a pointer to the beginning of the newly-allocated memory.
- On failure, returns a null pointer.

> ***NOTE***: To avoid memory leaks, the returned pointer must be freed using [`free_aligned`](#free_aligned)

&nbsp;

&nbsp;

---
### `free_aligned`

Frees memory that was allocated by [allocate_bytes_aligned](#allocate_bytes_aligned),
[fft_20xx_allocate_bytes_simd_aligned](#fft_20xx_allocate_bytes_simd_aligned),
or [fft_20xx_allocate_trivial_type_simd_aligned](#fft_20xx_allocate_trivial_type_simd_aligned).

```cpp
void free_aligned(const void *memory)
```

### Parameters:
- `memory`: A pointer to memory that was allocated with one of
  [allocate_bytes_aligned](#allocate_bytes_aligned),
  [fft_20xx_allocate_bytes_simd_aligned](#fft_20xx_allocate_bytes_simd_aligned),
  or [fft_20xx_allocate_trivial_type_simd_aligned](#fft_20xx_allocate_trivial_type_simd_aligned).

> ***NOTE***: It is safe to call this with a null pointer.

&nbsp;

&nbsp;

## Testing SIMD Support

---
### `get_simd_support`

On x86 and x64 processors, does a runtime check to determine which SIMD modes are supported by the current machine.

```cpp
simd_support_flags get_simd_support()
```
> ***NOTE***: This function only exists on x86 or x64 platforms.

### Return Value:
  - Returns a [`simd_support_flags`](#simd_support_flags) value that is a combination of all of the supported SIMD
    types that FFT-20XX knows about.

&nbsp;

&nbsp;

---
### `simd_support_flags`

A set of flags for the various types of SIMD support that FFT-20XX knows about.

```cpp
enum class simd_support_flags
{
  none = c::FFT_20XX_SIMD_SUPPORT_FLAG_NONE,
  sse2 = c::FFT_20XX_SIMD_SUPPORT_FLAG_SSE2,
  sse3 = c::FFT_20XX_SIMD_SUPPORT_FLAG_SSE3,
  ssse3 = c::FFT_20XX_SIMD_SUPPORT_FLAG_SSSE3,
  sse4_1 = c::FFT_20XX_SIMD_SUPPORT_FLAG_SSE4_1,
  sse4_2 = c::FFT_20XX_SIMD_SUPPORT_FLAG_SSE4_2,
  avx = c::FFT_20XX_SIMD_SUPPORT_FLAG_AVX,
  fma = c::FFT_20XX_SIMD_SUPPORT_FLAG_FMA,
  avx2 = c::FFT_20XX_SIMD_SUPPORT_FLAG_AVX2,
};
```

> ***NOTE***: These have bitwise operators (`&`, `&=`, `|`, `|=`, `^`, `^=`, and `~`) to allow them to be used as flags
> directly:
> ```cpp
> bool has_avx_support(fft_20xx::simd_support_flags flags)
> {
>   return (flags & simd_support_flags::avx) != simd_support_flags::none;
> }
> ```

&nbsp;

&nbsp;

## Flushing Subnormals

&nbsp;

&nbsp;

---
### `set_flush_subnormals_to_zero`

On x86, x64, or ARM, this function enables or disables flushing [subnormal floating-point values](https://en.wikipedia.org/wiki/Subnormal_number) to 0 **on the current thread**, which can improve performance dramatically (at the cost of very tiny
numbers rounding to 0).

```cpp
void set_flush_subnormals_to_zero(bool flush_to_zero)
```

### Parameters:
- `flush_to_zero`: Set this to `true` to set the CPU to flush to 0 on the current thread. Set this to `false` to revert
  to standard non-flushing behavior.
