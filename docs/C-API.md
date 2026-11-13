# FFT-20XX C API

This documents the C API for FFT-20XX. If you're using C++, please refer to the [C++ documentation](CPP-API.md)

#  Contents
- [Overview](#overview)
  - [Example: Without Tables](#example-without-tables)
  - [Example: Allocated Table](#exmaple-allocated-table)
  - [Example: User-Managed Table](#example-user-managed-table)
- [Tables](#tables)
  - [Table Types](#table-types)
  - [Table Allocation](#table-allocation)
  - [Tables From User Memory](#tables-from-user-memory)
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
#include "fft_20xx/fft_20xx.h"

void run_cplx_fft(const float *input, float *output, size_t length)
{
  // Pass null for the table to let FFT-20XX manage it internally.
  fft_20xx_cplx_f32(NULL, input, output, length);
}
```

&nbsp;

&nbsp;

## Example: Allocated Table

```cpp
#include "fft_20xx/fft_20xx.h"

fft_table_cplx_f32 *g_table = NULL;

// Call this at, say, program startup to set up the table
void initialize_table(size_t max_length)
{
  g_table = allocate_fft_table_cplx_f32(max_length);
}

// Call this at, say, program shutdown to free the table
void free_table()
{
  free_fft_table_cplx_f32(g_table);
  g_table = NULL;
}

void run_cplx_fft(const float *input, float *output, size_t length)
{
  assert(g_table != NULL);
  fft_20xx_cplx_f32(g_table, input, output, length);
}
```

&nbsp;

&nbsp;

## Example: User-Managed Table

```cpp
#include "fft_20xx/fft_20xx.h"

void *g_table_mem = NULL;
fft_table_cplx_f32 *g_table = NULL;

// Call this at, say, program startup to set up the table
void initialize_table(size_t max_length)
{
  // Since malloc does not necessarily SIMD-align its result, use the unaligned
  // variant.
  size_t mem_size = get_fft_table_cplx_f32_byte_count_unaligned(max_length);

  // Can use a custom allocator instead of malloc here
  g_table_mem = malloc(mem_size);

  // Build the table from the above memory block.
  g_table = make_table_from_memory_cplx_f32(max_length, g_table_mem, mem_size);
}

// Call this at, say, program shutdown to free the table
void free_table()
{
  // We just need to free the allocated memory (do not free g_table directly,
  // make sure to use the pointer that was returned from the allocator).
  free(g_table_mem);

  g_table = NULL;
  g_table_mem = NULL;
}

void run_cplx_fft(const float *input, float *output, size_t length)
{
  assert(g_table != NULL);
  fft_20xx_cplx_f32(g_table, input, output, length);
}
```

&nbsp;
# Tables

FFT-20XX uses **FFT tables**, which contain precomputed data used to speed up the computation of the FFT itself.


&nbsp;

&nbsp;

## Table Types

There are four types of FFT table that can be used. These are opaque types that are created via the various table
allocation/initialization functions, and used by the FFT routines themselves.

| Name                 | Description                                               |
| -------------------- | --------------------------------------------------------- |
| `fft_table_cplx_f32` | For use with complex-input FFTs with 32-bit float values. |
| `fft_table_cplx_f64` | For use with complex-input FFTs with 64-bit float values. |
| `fft_table_real_f32` | For use with real-input FFTs with 32-bit float values.    |
| `fft_table_real_f64` | For use with real-input FFTs with 64-bit float values.    |

&nbsp;

&nbsp;

## Table Allocation

If you do not want to allocate your own memory (e.g. using some alternative allocation strategy) but still want to
control when FFT tables get allocated and freed, these functions can be used to allocate and free tables for their
corresponding FFT functions.

&nbsp;

&nbsp;

---
### `allocate_fft_table_<type>_<float>`

Allocate an FFT table that supports FFTs up to the given maximum length.

```cpp
const fft_table_cplx_f32 *allocate_fft_table_cplx_f32(size_t max_length)
const fft_table_cplx_f64 *allocate_fft_table_cplx_f64(size_t max_length)
const fft_table_real_f32 *allocate_fft_table_real_f32(size_t max_length)
const fft_table_real_f64 *allocate_fft_table_real_f64(size_t max_length)
```
### Parameters:
- `max_length`: The maximum FFT length this table needs to support. **Must be a power of 2.**

### Return Value:
- On success, returns a pointer to the allocated FFT table.
- On failure, returns a null pointer.

> ***NOTE***: Memory allocated via these functions must be freed by a call to the corresponding
> [`free_fft_table_<type>_<float>`](#free_fft_table_type_float) function.

&nbsp;

&nbsp;

---
### `free_fft_table_<type>_<float>`

Free the FFT table memory that was allocated by the corresponding [`allocate_fft_table_<type>_<float>`](#allocate_fft_table_type_float)
function.

```cpp
void free_fft_table_cplx_f32(const fft_table_cplx_f32 *table)
void free_fft_table_cplx_f64(const fft_table_cplx_f64 *table)
void free_fft_table_real_f32(const fft_table_real_f32 *table)
void free_fft_table_real_f64(const fft_table_real_f64 *table)
```

### Parameters:
- `table`: The table whose memory needs to be freed. This pointer must have come from a corresponding
  call to `allocate_fft_table_<type>_<float>`.

> ***NOTE***: `table` may be a null pointer (to reduce special-casing), in which case this function does nothing.

&nbsp;

&nbsp;

## Tables From User Memory

This section contains functions used to fill a user-managed block of memory with an FFT table (rather than allowing
FFT-20XX to handle the allocations itself).

&nbsp;

&nbsp;

---
### `get_fft_table_<type>_<float>_byte_count_aligned`

Returns the count of bytes needed to initialize an FFT table with the given maximum length, assuming the memory block
is correctly SIMD-aligned.

```cpp
size_t get_fft_table_cplx_f32_byte_count_aligned(size_t max_length)
size_t get_fft_table_cplx_f64_byte_count_aligned(size_t max_length)
size_t get_fft_table_real_f32_byte_count_aligned(size_t max_length)
size_t get_fft_table_real_f64_byte_count_aligned(size_t max_length)
```
### Parameters:
- `max_length`: The maximum FFT length that the table would need to support. **Must be a power of 2.**

### Return Value:
- Returns the number of bytes needed for an FFT table at a SIMD-aligned address.

&nbsp;

&nbsp;

---
### `get_fft_table_<type>_<float>_byte_count_unaligned`

Returns the count of bytes needed to initialize an FFT table with the given maximum length, assuming the memory block
is *not* SIMD-aligned.

```cpp
size_t get_fft_table_cplx_f32_byte_count_unaligned(size_t max_length)
size_t get_fft_table_cplx_f64_byte_count_unaligned(size_t max_length)
size_t get_fft_table_real_f32_byte_count_unaligned(size_t max_length)
size_t get_fft_table_real_f64_byte_count_unaligned(size_t max_length)
```
### Parameters:
- `max_length`: The maximum FFT length that the table would need to support. **Must be a power of 2.**

### Return Value:
- Returns the number of bytes needed for an FFT table at an unaligned address.

&nbsp;

&nbsp;

---
### `make_table_from_memory_<type>_<float>`

Initialize an FFT table of the given type at a given memory location.

```cpp
const fft_table_cplx_f32 *make_table_from_memory_cplx_f32(
  size_t max_length,
  void *memory,
  size_t byte_count)

const fft_table_cplx_f64 *make_table_from_memory_cplx_f64(
  size_t max_length,
  void *memory,
  size_t byte_count)

const fft_table_real_f32 *make_table_from_memory_real_f32(
  size_t max_length,
  void *memory,
  size_t byte_count)

const fft_table_real_f64 *make_table_from_memory_real_f64(
  size_t max_length,
  void *memory,
  size_t byte_count)
```
### Parameters:
- `max_length`: The max FFT length that this table needs to support. **Must be a power of 2.**
- `memory`: A pointer to the memory to be filled with the FFT table.
- `byte_count`: The amount of bytes available at the memory location.

### Return Value:
- On success, returns a pointer to the start of the constructed table. This will be a pointer to somewhere within the
  given memory block (but is not necessarily at its start, due to SIMD alignment).
- On failure, returns a null pointer (and, in debug builds, will assert).

> ***NOTE***: If `memory` is a pointer with system SIMD alignment, `byte_count` must be at least as large as the value
  return by the corresponding [`get_fft_table_<type>_<float>_byte_count_aligned`](#get_fft_table_type_float_byte_count_aligned)
  function.
>
> If it is *not* SIMD-aligned, then `byte_count` must be at least as large as the value
  return by the corresponding [`get_fft_table_<type>_<float>_byte_count_unaligned`](#get_fft_table_type_float_byte_count_unaligned)
  function.


&nbsp;

&nbsp;

# FFT Functions

The functions in this section perform the actual FFT (or inverse FFT) computations. Note that, in all cases:
- A null pointer can be passed to the first parameter (`table`), to allow FFT-20XX to internally manage a (global,
  thread-safe) table allocated for the given FFT type.
- If a table *is* passed in, no memory will be allocated or freed during the runtime of the function.
- The `input` and `output` pointers can be the same pointer, in which case the (I)FFT will be performed in-place.
- No additional scratch memory is needed (the only mutable memory required is the `output` array).

&nbsp;

&nbsp;

---
### `fft_cplx_<float>`

Perform an FFT of the given length, reading complex-valued time-domain samples and writing complex-valued
frequency-domain samples.

```cpp
void fft_cplx_f32(const fft_table_cplx_f32 *table, const float  *input, float  *output, size_t length);
void fft_cplx_f64(const fft_table_cplx_f64 *table, const double *input, double *output, size_t length);
```

### Parameters
- `table`: The FFT table to use. Must have been created with a maximum length of at least `length`.
  - If a null pointer is passed, FFT-20XX will internally manage a (global, thread-safe) table for the given FFT type,
    which may trigger an allocation if the internal table is not large enough to support the given length.
  - If a non-null pointer is passed, this function will not allocate or free any memory.
- `input`: The input time-domain samples to use.
  - Expected to be **packed complex values** (That is, it is "real0, imaginary0, real1, imaginary1, ...")
  - There should be `2 * length` floats at the given address (as there are 2 floats per complex input).
- `output`: The output frequency-domain samples.
  - Like `input`, this is expected to be packed complex values with a length of `2 * length`.
  - This pointer is allowed to alias to `input` (i.e. they can point to the same place), in which case, the FFT will
    be performed in-place.
- `length`: The length of the FFT to perform.
  - **Must be a power of 2!**
&nbsp;

&nbsp;

---
### `ifft_cplx_<float>`

Perform an inverse FFT (IFFT) of the given length, reading complex-valued frequency-domain samples and writing complex-valued
time-domain samples.

```cpp
void ifft_cplx_f32(const fft_table_cplx_f32 *table, const float  *input, float  *output, size_t length);
void ifft_cplx_f64(const fft_table_cplx_f64 *table, const double *input, double *output, size_t length);
```
### Parameters
- `table`: The FFT table to use. Must have been created with a maximum length of at least `length`.
  - If a null pointer is passed, FFT-20XX will internally manage a (global, thread-safe) table for the given FFT type,
    which may trigger an allocation if the internal table is not large enough to support the given length.
  - If a non-null pointer is passed, this function will not allocate or free any memory.
- `input`: The input frequency-domain samples to use.
  - Expected to be **packed complex values** (That is, it is "real0, imaginary0, real1, imaginary1, ...")
  - There should be `2 * length` floats at the given address (as there are 2 floats per complex input).
- `output`: The output time-domain samples.
  - Like `input`, this is expected to be packed complex values with a length of `2 * length`.
  - This pointer is allowed to alias to `input` (i.e. they can point to the same place), in which case, the IFFT will
    be performed in-place.
- `length`: The length of the IFFT to perform.
  - **Must be a power of 2!**
&nbsp;

&nbsp;

---
### `fft_real_<float>`

Perform an FFT of the given length, reading real-valued time-domain samples and writing the first half of the complex
frequency-domain samples, with the DC and Nyquist terms (which are both always real-valued) packed into the first two
floats, like so:
>  `DC`, `Nyquist`, `real1`, `imag1`, `real2`, `imag2`, ...

This discards the latter half of the FFT results, as with real FFTs that information is redundant (they are just
[complex conjugates](https://en.wikipedia.org/wiki/Complex_conjugate) of the complex values in the first half).

```cpp
void fft_real_f32(const fft_table_real_f32 *table, const float  *input, float  *output, size_t length);
void fft_real_f64(const fft_table_real_f64 *table, const double *input, double *output, size_t length);
```
### Parameters
- `table`: The FFT table to use. Must have been created with a maximum length of at least `length`.
  - If a null pointer is passed, FFT-20XX will internally manage a (global, thread-safe) table for the given FFT type,
    which may trigger an allocation if the internal table is not large enough to support the given length.
  - If a non-null pointer is passed, this function will not allocate or free any memory.
- `input`: The input time-domain samples to use.
  - Expected to be **real values** (That is, each float is a sample, unlike the complex-valued FFT functions)
  - There should be `length` floats at the given address (as there is one float per input sample).
- `output`: The output frequency-domain samples.
  - Like with `input`, there should be `length` floats at the given address.
  - There are effectively `length/2 + 1` samples in this output:
    - The first float is the DC value (corresponding to `output[0]` of the full, complex-valued FFT)
    - The second float is the Nyquist value (corresponding to `output[length]` of the full, complex-valued FFT)
    - The remaining values are **packed complex pairs** corresponding to the frequencies below the Nyquist frequency
      (corresponding to the complex values in `output[2]` through `output[length-1]` of the full, complex-valued FFT)
  - This pointer is allowed to alias to `input` (i.e. they can point to the same place), in which case, the FFT will
    be performed in-place.
- `length`: The length of the FFT to perform.
  - **Must be a power of 2!**
&nbsp;

&nbsp;

---
### `ifft_real_<float>`

Perform an inverse FFT (IFFT) of the given length, reading the "half-complex" frequency-domain samples (in the format
output by the [fft_real_<float>](#fft_real_float) functions) and writing real-valued time-domain samples.

```cpp
void ifft_real_f32(const fft_table_real_f32 *table, const float  *input, float  *output, size_t length);
void ifft_real_f64(const fft_table_real_f64 *table, const double *input, double *output, size_t length);
```
### Parameters
- `table`: The FFT table to use. Must have been created with a maximum length of at least `length`.
  - If a null pointer is passed, FFT-20XX will internally manage a (global, thread-safe) table for the given FFT type,
    which may trigger an allocation if the internal table is not large enough to support the given length.
  - If a non-null pointer is passed, this function will not allocate or free any memory.
- `input`: The input frequency-domain samples to use.
  - There should be `length` floats at the given address.
  - There are effectively `length/2 + 1` samples in this input:
    - The first float is the DC value (corresponding to `input[0]` of the full, complex-valued IFFT)
    - The second float is the Nyquist value (corresponding to `input[length]` of the full, complex-valued IFFT)
    - The remaining values are **packed complex pairs** corresponding to the frequencies below the Nyquist frequency
      (corresponding to the complex values in `input[2]` through `input[length-1]` of the full, complex-valued IFFT)
- `output`: The output time-domain samples to use.
  - Like with `input`, there should be `length` floats at the given address, as there is one float per output sample.
  - This pointer is allowed to alias to `input` (i.e. they can point to the same place), in which case, the FFT will
    be performed in-place.
- `length`: The length of the FFT to perform.
  - **Must be a power of 2!**

&nbsp;

&nbsp;

# Utilities

The following utilities are in **`fft_20xx/fft_20xx_util.h`**, so include that to use these.

&nbsp;

&nbsp;

## Aligned Allocation

These utility functions support the allocation and deallocation of *aligned* memory (useful for, for instance,
allocating memory for use with SIMD functions).
&nbsp;

&nbsp;

---
### `fft_20xx_allocate_bytes_aligned`

Allocates a given number of bytes, with a given memory alignment.

```cpp
void *fft_20xx_allocate_bytes_aligned(size_t byte_count, size_t pow_2_alignment)
```

### Parameters:
- `byte_count`: The number of bytes to allocate. **Must be an integral multiple of `pow_2_alignment`**.
- `pow_2_alignment`: The required alignment of the allocation. **Must be a power of two**.

### Return Value:
- On success, returns a pointer to the beginning of the newly-allocated memory.
- On failure, returns a null pointer.

> ***NOTE***: To avoid memory leaks, the returned pointer must be freed using [`fft_20xx_free_aligned`](#fft_20xx_free_aligned)

&nbsp;

&nbsp;

---
### `fft_20xx_allocate_bytes_simd_aligned`

Allocates a given number of bytes, aligned to the system's maximum SIMD vector size. Useful for if you know you want
your memory to be SIMD-aligned but don't know the precise alignment of the SIMD types on the target platform.

```cpp
void *fft_20xx_allocate_bytes_simd_aligned(size_t byte_count)
```

### Parameters:
- `byte_count`: The number of bytes to allocate. **Must be an integral multiple of the system's SIMD alignment**.

### Return Value:
- On success, returns a pointer to the beginning of the newly-allocated memory.
- On failure, returns a null pointer.

> ***NOTE***: To avoid memory leaks, the returned pointer must be freed using [`fft_20xx_free_aligned`](#fft_20xx_free_aligned)

&nbsp;

&nbsp;

---
### `fft_20xx_free_aligned`

Frees memory that was allocated by [fft_20xx_allocate_bytes_aligned](#fft_20xx_allocate_bytes_aligned) or
[fft_20xx_allocate_bytes_simd_aligned](#fft_20xx_allocate_bytes_simd_aligned).
```cpp
void fft_20xx_free_aligned(const void *memory)
```

### Parameters:
- `memory`: A pointer to memory that was allocated with
  [`fft_20xx_allocate_bytes_aligned`](#fft_20xx_allocate_bytes_aligned) or
  [fft_20xx_allocate_bytes_simd_aligned](#fft_20xx_allocate_bytes_simd_aligned).

> ***NOTE***: It is safe to call this with a null pointer.

&nbsp;

&nbsp;

## Testing SIMD Support

---
### `fft_20xx_get_simd_support`

On x86 and x64 processors, does a runtime check to determine which SIMD modes are supported by the current machine.

```cpp
fft_20xx_simd_support_flags_t fft_20xx_get_simd_support()
```
> ***NOTE***: This function only exists on x86 or x64 platforms.

### Return Value:
  - Returns a [`fft_20xx_simd_support_flags_t`](#fft_20xx_simd_support_flags_t) value that is a combination of all of the supported SIMD
    types that FFT-20XX knows about.

&nbsp;

&nbsp;

---
### `fft_20xx_simd_support_flags_t`

A set of flags for the various types of SIMD support that FFT-20XX knows about.

```cpp
typedef enum fft_20xx_simd_support_flags_t
{
FFT_20XX_SIMD_SUPPORT_FLAG_NONE    = 0x00,

FFT_20XX_SIMD_SUPPORT_FLAG_SSE2    = 0x01,
FFT_20XX_SIMD_SUPPORT_FLAG_SSE3    = 0x02,
FFT_20XX_SIMD_SUPPORT_FLAG_SSSE3   = 0x04,
FFT_20XX_SIMD_SUPPORT_FLAG_SSE4_1  = 0x08,
FFT_20XX_SIMD_SUPPORT_FLAG_SSE4_2  = 0x10,
FFT_20XX_SIMD_SUPPORT_FLAG_AVX     = 0x20,
FFT_20XX_SIMD_SUPPORT_FLAG_FMA     = 0x40,
FFT_20XX_SIMD_SUPPORT_FLAG_AVX2    = 0x80,
} fft_20xx_simd_support_flags_t;
```

&nbsp;

&nbsp;

## Flushing Subnormals

---
### `fft_20xx_set_flush_subnormals_to_zero`

On x86, x64, or ARM, this function enables or disables flushing [subnormal floating-point values](https://en.wikipedia.org/wiki/Subnormal_number) to 0 **on the current thread**, which can improve performance dramatically (at the cost of very tiny
numbers rounding to 0).

```cpp
void fft_20xx_set_flush_subnormals_to_zero(bool flush_to_zero)
```

### Parameters:
- `flush_to_zero`: Set this to `true` to set the CPU to flush to 0 on the current thread. Set this to `false` to revert
  to standard non-flushing behavior.
