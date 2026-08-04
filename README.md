# FFT-20XX

`FFT-20XX` is a fast, SIMD-accelerated, easy-to-use FFT library for C and C++.

# Contents

- [Why?](#why)
- [Dependencies](#dependencies)
- [Usage](#usage)
  - [C](#c)
  - [C++](#c-1)
- [The Algorithm](#the-algorithm)

### Why?

There are a bazillion FFT libraries out there, why one more? Well, I wanted one that:

- Runs the FFT (or inverse FFT) in place (i.e. with the same input and output memory)
- Puts the frequency-domain values in the correct order (i.e. does the proper bit-reverse)
- Requires no additional scratch memory (only a single, constant table that can be created up-front with a given maximum FFT length) so there is no need for extra per-thread allocations (when using the library on multiple threads)
- Is quite fast: it is optimized for SSE2, AVX (with and without FMA instructions), and AARCH64 NEON and performs favorably relative to other open-source FFT libraries
- Also it was an interesting project to work on

This library handles all of this. The only limitation is that it only supports power-of-two-length FFTs (rather than arbitrary sizes).

### Dependencies

As packaged, this builds with CMake and requires a C++20-compatible compiler (tested in MSVC, Clang, and G++). There are no additional dependencies.

### Usage

The library is straightforward to use, simply include the header for your language (`fft_20xx/fft_20xx.h` for C and `fft_20xx/fft_20xx.hpp` for C++, create a table that handles what you need, and then call the functions with your inputs!

(More-comprehensive documentation coming at some point)

#### C

Here is a simple C example of using the library to do a 32-bit float real forward FFT:

```c
#include "fft_20xx/fft_20xx.h"

void example_real_fft(const float *input, float *output, size_t fftLength)
{
  // In practice this can be done once, up-front, using the maximum required FFT length
  fft_table_real_f32 *table = allocate_fft_table_real_f32(fftLength);

  // Now run the forward FFT
  fft_real_f32(table, input, output, fftLength);

  // Clean up the table when done
  free_fft_table_f32(table);
}
```

If you want to allocate your own memory for the table (due to some specific memory requirement setup), there are the following functions:
- `get_simd_alignment` Get the alignment required for the FFT table (it is at least the SIMD alignment for the current CPU)
- `get_fft_table_cplx_f32_byte_count_aligned` (also variants for `cplx_f64`, `real_f32`, and `real_f64`): This gets the number of bytes needed when allocating a buffer aligned to the result of `get_simd_alignment`.
- `get_fft_table_cplx_f32_byte_count_unaligned` (and variants): Get the number of bytes needed for the table when a specific alignment cannot be guaranteed.
- `make_table_from_memory_cplx_f32` (and variants): Given a block of memory with a given size, initialize the memory within to have the table's data. Returns a typed pointer to the start of the table's memory.

#### C++

Here is an equivalent example to the above, but for C++:

```c++
#include "fft_20xx/fft_20xx.hpp"

void example_real_fft(const float *input, float *output, size_t fftLength)
{
  // In practice this can be done once, up-front, using the maximum required FFT length
  auto table = fft_20xx::fft_table<fft_20xx::real, float>{fftLength);

  // Run the forward FFT
  fft_20xx::fft(table, input, output, fftLength);

  // In this case, no need to clean up the table, we're using RIAA up in here!
}
```
Like the C API, there is functionality to allow you to handle the allocation yourself:
- `fft_table<type, float_type>::get_required_alignment` will get the expected alignment for table.
- `fft_table<type, float_type>::get_byte_count_aligned` gets the number of bytes needed when allocating a buffer aligned to the result of the above function.
- `fft_table<type, float_type>::get_byte_count_unaligned` gets the number of bytes needed when allocating a buffer when a specific alignment cannot be guaranteed.
- then initialize a `fft_table<type, float_type>` with the max FFT length along with the preallocated buffer and its size.

### The Algorithm

(This section will be expanded at some point with more details. Possibly too many more details)

The core of the algorithm used here was to do a Radix-4 FFT, but halfway through processing to start doing the bit reversals on the fly (2 bits at a time, which effectively makes each radix-4 set a 4x4 matrix transpose), so that no pre- or post-sorting pass is necessary. 
This is also done entirely in-place, unlike many other libraries that require scratch space to do the sorting.

No generative AI was used in the creation of this project.
