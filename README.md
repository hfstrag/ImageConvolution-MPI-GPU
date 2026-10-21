# ImageConvolution-MPI-GPU

A C/CUDA project for convolution of 8-bit images, with several implementations
for comparing sequential execution, blocking and non-blocking MPI
communications, and GPU computation.

## Contents

- `MPI_parallelization/parallel_blocking.c`: distributed version using
  blocking exchanges of halo rows.
- `MPI_parallelization/parallelnon_blocking.c`: distributed version using
  non-blocking MPI communications and computation overlapped with communication.
- `MPI_parallelization/convo.c`: Sun Raster image reading and writing, filters,
  and sequential reference computation.
- `GPU_parallelization/gpu_convol_v0.cu`: CUDA convolution, compared against
  its CPU reference computation.
- `images/`: sample images in Sun Raster (`.ras`) format.
- `MPI_parallelization/Makefile` and `GPU_parallelization/Makefile`: build and
  benchmark targets.

## Requirements

- A system compatible with the compilers and tools used by the project (the
  source code relies on Unix/POSIX interfaces).
- GNU Make and a C compiler.
- For MPI: an MPI implementation providing `mpicc` and `mpirun`.
- For the MPI integration tests: Python 3 (standard library only).
- For CUDA: the CUDA toolkit (`nvcc`), a compatible NVIDIA GPU, and a suitable
  driver. The CUDA Makefile targets the `sm_60` architecture.

The programs read 8-bit Sun Raster images with a standard RGB color map. The
images provided in `images/` are examples.

## Building

From the project root:

```sh
make -C MPI_parallelization
make -C GPU_parallelization
```

Each command can be run independently if only the MPI or CUDA version is
needed.

## Running with MPI

The MPI executables take an image, a filter number, and a number of iterations:

```sh
mpirun -np 4 MPI_parallelization/parallel_blocking \
  images/femme10.ras 0 10

mpirun -np 4 MPI_parallelization/parallelnon_blocking \
  images/femme10.ras 0 10
```

The available filters are:

| Number | Filter |
| --- | --- |
| `0` | Average of the 9 pixels |
| `1` | Average with a higher weight for the center pixel |
| `2` | Laplacian edge detection |
| `3` | Edge detection using the maximum gradient |
| `4` | Median filter |

The image height must be divisible by the number of MPI processes. The
non-blocking version also requires at least two rows per process. The MPI
result is written to the current directory as
`post-convolution_filter<FILTER>_nbIter<ITERATIONS>.ras`.

## Running with CUDA

The CUDA executable takes the image path, the number of iterations, and
optionally the CUDA block dimensions (width, then height):

```sh
GPU_parallelization/gpu_convol_v0 \
  images/femme10.ras 10 16 16
```

The CUDA filter is fixed in the code: it is a weighted average with a higher
weight for the center pixel. The default block dimensions are `16 × 16`. The
program compares the GPU result with the CPU result and displays their
computation times. The processed image is saved in a user-specific temporary
subdirectory under `/tmp`, with a name beginning with `post-convolution_`.

## Benchmarking

The Makefiles provide `bench` targets, parameter-specific targets (`bench_p`
and `bench_iter` for MPI; `bench_block` and `bench_iter` for CUDA), and a
`parse` target that writes a CSV file to `results/`. The benchmark targets
expect the images listed in the Makefile configuration to be present in their
working directory. The MPI benchmark is also configured for a compute
environment using `OAR_NODEFILE` and specific Open MPI options; it may need to
be adapted for other infrastructures.

## Cleaning

From the project root:

```sh
make -C MPI_parallelization clean
make -C GPU_parallelization clean
```

These targets remove the executables and generated results from the relevant
directories.

## Tests

Run the CPU unit tests from the project root:

```sh
make test
```

Run the MPI integration tests (requires MPI and Python 3):

```sh
make test-mpi
```

The unit tests cover all five CPU filters, convolution border behavior, and
command-line number parsing. The MPI integration tests generate a small
temporary Sun Raster image, then compare both MPI implementations against a
reference result for every filter. The tests do not modify the sample images
or leave generated test images in the repository. The MPI integration tests
are also run by GitHub Actions on pushes and pull requests. CUDA correctness
checking is performed by the GPU executable at runtime; automated CUDA tests
require access to a compatible NVIDIA GPU and are not part of the standard
CI job.
