/*
 * University of Lille
 * Convolution calculation on an image: V0
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>      /* for rint() */
#include <string.h>    /* for memcpy() */
#include <sys/time.h>  /* for timing */
#include <libgen.h>    /* for basename() */
#include <sys/stat.h>  /* for mkdir() */
#include <unistd.h>    /* for getuid()   */
#include <sys/types.h> /* for getpwuid() */
#include <pwd.h>       /* for getpwuid() */
#include <stdbool.h>

#include "rasterfile.h"

#define MAX(a, b) ((a > b) ? a : b)

/**
 * \struct Raster
 * Structure describing an image in Sun Raster format
 */

typedef struct
{
  struct rasterfile file;                        ///< Header Sun Raster image
  unsigned char red[256], green[256], blue[256]; ///< Color palette
  unsigned char *data;                           ///< Pointer to the image
} Raster;

double my_gettimeofday()
{
  struct timeval tmp_time;
  gettimeofday(&tmp_time, NULL);
  return tmp_time.tv_sec + (tmp_time.tv_usec * 1.0e-6L);
}

/**
 * Returns the integer quotient greater or equal to "a/b".
 * (source: CUDA SDK 4.1)
 */

static int iDivUp(int a, int b)
{
  return ((a % b != 0) ? (a / b + 1) : (a / b));
}

/**
 * Error handling and debugging in CUDA.
 */

#ifdef CUDA_DEBUG
#define CUDA_SYNC_ERROR()                                          \
  {                                                                \
    cudaError_t sync_error;                                        \
    cudaDeviceSynchronize();                                       \
    sync_error = cudaGetLastError();                               \
    if (sync_error != cudaSuccess)                                 \
    {                                                              \
      fprintf(stderr, "[CUDA SYNC ERROR at %s:%d -> %s]\n",        \
              __FILE__, __LINE__, cudaGetErrorString(sync_error)); \
      exit(EXIT_FAILURE);                                          \
    }                                                              \
  }
#else /* #ifdef CUDA_DEBUG */
#define CUDA_SYNC_ERROR()
#endif /* #ifdef CUDA_DEBUG */

#define CUDA_ERROR(cuda_call)                                 \
  {                                                           \
    cudaError_t error = cuda_call;                            \
    if (error != cudaSuccess)                                 \
    {                                                         \
      fprintf(stderr, "[CUDA ERROR at %s:%d -> %s]\n",        \
              __FILE__, __LINE__, cudaGetErrorString(error)); \
      exit(EXIT_FAILURE);                                     \
    }                                                         \
    CUDA_SYNC_ERROR();                                        \
  }

/**
 * This function converts a LINUX integer to a SUN integer (endianness).
 *
 * \param i: pointer to the integer to convert
 */

void swap(int *i)
{
  unsigned char s[4], *n;
  memcpy(s, i, 4);
  n = (unsigned char *)i;
  n[0] = s[3];
  n[1] = s[2];
  n[2] = s[1];
  n[3] = s[0];
}

/**
 * \brief Reading an image in Sun Rasterfile format.
 *
 * Upon return from this function, the structure r is filled with the data
 * associated with the image. The r.file field contains the image header
 * information (dimensions, encoding, etc.).
 * The r.data field is a pointer, allocated by the read_rasterfile() function,
 * and which contains the image. This space must be freed after use.
 *
 * \param name: image filename
 * \param r: Raster structure that contains the image loaded into memory
 */

void read_rasterfile(char *name, Raster *r)
{
  FILE *f;
  int i, h, w, w2;

  if ((f = fopen(name, "r")) == NULL)
  {
    fprintf(stderr, "Error when reading the file %s\n", name);
    exit(1);
  }
  if (fread(&(r->file), sizeof(struct rasterfile), 1, f) < 1)
  {
    fprintf(stderr, "Error in fread() ar %s:%d\n", __FILE__, __LINE__);
  };
  swap(&(r->file.ras_magic));
  swap(&(r->file.ras_width));
  swap(&(r->file.ras_height));
  swap(&(r->file.ras_depth));
  swap(&(r->file.ras_length));
  swap(&(r->file.ras_type));
  swap(&(r->file.ras_maptype));
  swap(&(r->file.ras_maplength));

  if ((r->file.ras_depth != 8) || (r->file.ras_type != RT_STANDARD) ||
      (r->file.ras_maptype != RMT_EQUAL_RGB))
  {
    fprintf(stderr, "Unsuitable palette\n");
    exit(1);
  }

  /* component of the palette */
  if (fread(&(r->red), r->file.ras_maplength / 3, 1, f) < 1)
  {
    fprintf(stderr, "Error in fread() ar %s:%d\n", __FILE__, __LINE__);
  };
  if (fread(&(r->green), r->file.ras_maplength / 3, 1, f) < 1)
  {
    fprintf(stderr, "Error in fread() ar %s:%d\n", __FILE__, __LINE__);
  };
  if (fread(&(r->blue), r->file.ras_maplength / 3, 1, f) < 1)
  {
    fprintf(stderr, "Error in fread() ar %s:%d\n", __FILE__, __LINE__);
  };

  if ((r->data = (unsigned char *)malloc(r->file.ras_width * r->file.ras_height)) == NULL)
  {
    fprintf(stderr, "Memory allocation error\n");
    exit(1);
  }

  /* Format Sun Rasterfile: "The width of a scan line is always a multiple of 16 bits, padded when necessary."
   * (see: http://netghost.narod.ru/gff/graphics/summary/sunras.htm) */
  h = r->file.ras_height;
  w = r->file.ras_width;
  w2 = ((w + 1) & ~1); /* multiple of 2 greater or equal */
  //  printf("In read_rasterfile(): h=%d w=%d w2=%d\n", h, w, w2);
  for (i = 0; i < h; i++)
  {
    if (fread(r->data + i * w, w, 1, f) < 1)
    {
      fprintf(stderr, "Error in fread() ar %s:%d\n", __FILE__, __LINE__);
    }
    if (w2 - w > 0)
    {
      fseek(f, w2 - w, SEEK_CUR);
    }
  }

  fclose(f);
}

/**
 * Save an image in Sun Rasterfile format
 */

void save_rasterfile(char *name, Raster *r)
{
  FILE *f;
  int i, h, w, w2;

  if ((f = fopen(name, "w")) == NULL)
  {
    fprintf(stderr, "Error when writing the file %s\n", name);
    exit(1);
  }

  h = r->file.ras_height;
  w = r->file.ras_width;

  /* header: */
  swap(&(r->file.ras_magic));
  swap(&(r->file.ras_width));
  swap(&(r->file.ras_height));
  swap(&(r->file.ras_depth));
  swap(&(r->file.ras_length));
  swap(&(r->file.ras_type));
  swap(&(r->file.ras_maptype));
  swap(&(r->file.ras_maplength));
  fwrite(&(r->file), sizeof(struct rasterfile), 1, f);

  /* component of the palette */
  fwrite(&(r->red), 256, 1, f);
  fwrite(&(r->green), 256, 1, f);
  fwrite(&(r->blue), 256, 1, f);

  /* Format Sun Rasterfile: "The width of a scan line is always a multiple of 16 bits, padded when necessary."
   * (see: http://netghost.narod.ru/gff/graphics/summary/sunras.htm) */
  w2 = ((w + 1) & ~1); /* multiple of 2 greater or equal */
  //  printf("In read_rasterfile(): h=%d w=%d w2=%d\n", h, w, w2);
  for (i = 0; i < h; i++)
  {
    if (fwrite(r->data + i * w, w, 1, f) < 1)
    {
      fprintf(stderr, "Error in fwrite() ar %s:%d\n", __FILE__, __LINE__);
    }
    if (w2 - w > 0)
    { /* padding */
      unsigned char zeros[1] = {0};
      if (w2 - w != 1)
      {
        fprintf(stderr, "Error in save_rasterfile(): w2-w != 1 \n");
      }
      if (fwrite(zeros, w2 - w, 1, f) < 1)
      {
        fprintf(stderr, "Error in fwrite() ar %s:%d\n", __FILE__, __LINE__);
      }
    }
  }

  /* re-order bytes in original order so that
     save_rasterfile() can be called multiple
     times with the same 'r' variable */
  swap(&(r->file.ras_magic));
  swap(&(r->file.ras_width));
  swap(&(r->file.ras_height));
  swap(&(r->file.ras_depth));
  swap(&(r->file.ras_length));
  swap(&(r->file.ras_type));
  swap(&(r->file.ras_maptype));
  swap(&(r->file.ras_maplength));

  fclose(f);
}

/**
 * Convolution of an image using a predefined filter
 * \param buf_src: pointer to the source image
 * \param buf_tgt: pointer to the targer image
 * \param nbl, nbc: image dimensions
 */

void convolution(unsigned char buf_src[], unsigned char buf_tgt[], int nbl, int nbc)
{
  int numerator, denominator;
  int central_weight;
  int i, j;

  for (i = 0; i < nbl; i++)
  {
    for (j = 0; j < nbc; j++)
    {

      /*** CONVOL_AVG2 averaging filter (with more weight in the center).
       * Remark: on the edges, the average is computed with the existing pixels only. */
      denominator = 0.0f;
      numerator = 0.0f;
      if (i < nbl - 1)
      {
        numerator += (int)buf_src[(i + 1) * nbc + j];
        ++denominator;
        if (j > 0)
        {
          numerator += (int)buf_src[(i + 1) * nbc + j - 1];
          ++denominator;
        }
        if (j < nbc - 1)
        {
          numerator += (int)buf_src[(i + 1) * nbc + j + 1];
          ++denominator;
        }
      }
      if (j > 0)
      {
        numerator += (int)buf_src[(i)*nbc + j - 1];
        ++denominator;
      }
      if (j < nbc - 1)
      {
        numerator += (int)buf_src[(i)*nbc + j + 1];
        ++denominator;
      }
      if (i > 0)
      {
        numerator += (int)buf_src[(i - 1) * nbc + j];
        ++denominator;
        if (j > 0)
        {
          numerator += (int)buf_src[(i - 1) * nbc + j - 1];
          ++denominator;
        }
        if (j < nbc - 1)
        {
          numerator += (int)buf_src[(i - 1) * nbc + j + 1];
          ++denominator;
        }
      }
      central_weight = denominator * 0.5f; /* central weight = 50% other weights */
      numerator += central_weight * (int)buf_src[(i)*nbc + j];
      denominator += central_weight;

      buf_tgt[i * nbc + j] = numerator / denominator;
    } /* for j */
  } /* for i */
}

/**
 * Performs 'nbiter' convolutions and returns the pointer to the
 * buffer containing the result of the last convolution.
 */

unsigned char *multiple_convolutions(unsigned char buf[], unsigned char buf_aux[], int nbl, int nbc, int nbiter)
{
  int n, i;
  unsigned char *bufs[2] = {buf, buf_aux};

  for (n = 0; n < nbiter; n++)
  {
    i = n % 2;
    convolution(bufs[i], bufs[1 - i], nbl, nbc);
  } /* for n */

  return bufs[n % 2];
}

/**
 * Convolution on the GPU: one thread per pixel (i, j),
 * x for the columns and y for the lines.
 */

__global__ void convol_kernel(unsigned char buf_src[], unsigned char buf_tgt[], int nbl, int nbc)
{
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int numerator = 0, denominator = 0, central_weight;

  if (i >= nbl || j >= nbc)
    return;

  for (int di = -1; di <= 1; di++)
  {
    for (int dj = -1; dj <= 1; dj++)
    {
      int k = i + di, l = j + dj;
      if ((di != 0 || dj != 0) && k >= 0 && k < nbl && l >= 0 && l < nbc)
      {
        numerator += buf_src[k * nbc + l];
        denominator++;
      }
    }
  }

  central_weight = denominator / 2;
  numerator += central_weight * buf_src[i * nbc + j];
  denominator += central_weight;

  buf_tgt[i * nbc + j] = numerator / denominator;
}

/**
 * Performs 'nbiter' convolutions on the GPU card and returns the pointer to the
 * buffer containing the result of the last convolution.
 */

unsigned char *gpu_multiple_convolutions(unsigned char buf[],
                                         unsigned char buf_aux[],
                                         int nbl,
                                         int nbc,
                                         int nbiter,
                                         int blockDim_x,
                                         int blockDim_y)
{
  size_t size = nbl * nbc * sizeof(unsigned char);
  unsigned char *bufs[2] = {buf, buf_aux};
  unsigned char *d_bufs[2];
  dim3 block(blockDim_x, blockDim_y);
  dim3 grid(iDivUp(nbc, blockDim_x), iDivUp(nbl, blockDim_y));
  int n;

  CUDA_ERROR(cudaMalloc(&d_bufs[0], size));
  CUDA_ERROR(cudaMalloc(&d_bufs[1], size));
  CUDA_ERROR(cudaMemcpy(d_bufs[0], buf, size, cudaMemcpyHostToDevice));

  for (n = 0; n < nbiter; n++)
  {
    convol_kernel<<<grid, block>>>(d_bufs[n % 2], d_bufs[1 - n % 2], nbl, nbc);
  }
  CUDA_ERROR(cudaGetLastError());

  CUDA_ERROR(cudaMemcpy(bufs[n % 2], d_bufs[n % 2], size, cudaMemcpyDeviceToHost));
  CUDA_ERROR(cudaFree(d_bufs[0]));
  CUDA_ERROR(cudaFree(d_bufs[1]));

  return bufs[n % 2];
}

/**
 * User interface
 */

static char usage[] = "Usage : %s <SunRaster image name> <nbiter> [<blockDim_x>] [<blockDim_y>]\n";

/*
 * Main part
 */

int main(int argc, char *argv[])
{

  /* Variables relating to the image */
  Raster r;
  int w, h; /* number of lines and columns of the image */
  struct passwd *pass;
  char output_name[128] = "";
  char *base_name = basename(argv[1]);
  char dir_name[32] = "";

  /* Variables related to the image processing */
  int nbiter; /* number of iterations */

  /* Variables related to timing */
  double start, end;

  /* Buffers: */
  unsigned char *buf, *buf_aux, *buf_res = NULL;
  unsigned char *cpu_buf, *cpu_buf_aux, *cpu_buf_res = NULL;

  /* GPU block dimensions: */
  int blockDim_x = 16, blockDim_y = 16;

  if (argc < 3)
  {
    fprintf(stderr, usage, argv[0]);
    return 1;
  }

  /* Parameter entry */
  nbiter = atoi(argv[2]);
  if (argc > 3)
    blockDim_x = atoi(argv[3]);
  if (argc > 4)
    blockDim_y = atoi(argv[4]);

  /* Reading the Raster file */
  read_rasterfile(argv[1], &r);
  h = r.file.ras_height;
  w = r.file.ras_width;
  printf("Image %s : %dx%d\n", argv[1], h, w);
  printf("Blocks : %dx%d \tnbiter : %d\n", blockDim_x, blockDim_y, nbiter);

  /* Preparing the final save: */
  pass = getpwuid(getuid());
  sprintf(dir_name, "/tmp/%s", pass->pw_name);
  // printf("dir_name = %s\n", dir_name);
  mkdir(dir_name, S_IRWXU);
  sprintf(output_name, "%s/post-convolution_%s", dir_name, base_name);
  // printf("output_name = %s\n", output_name);

  /* Memory allocation: */
  buf = (unsigned char *)malloc(h * w * sizeof(unsigned char));
  buf_aux = (unsigned char *)malloc(h * w * sizeof(unsigned char));
  cpu_buf = (unsigned char *)malloc(h * w * sizeof(unsigned char));
  cpu_buf_aux = (unsigned char *)malloc(h * w * sizeof(unsigned char));

  /* Storing the 'r.data' image in 'buf': */
  memcpy(buf, r.data, h * w * sizeof(unsigned char));

  /* CUDA context initialization, outside of the timing */
  CUDA_ERROR(cudaFree(0));

  /* Start of timing */
  start = my_gettimeofday();

  /* The convolutions */
  buf_res = gpu_multiple_convolutions(buf, buf_aux, h, w, nbiter, blockDim_x, blockDim_y);

  /* End of timing */
  end = my_gettimeofday();
  printf("Total GPU time: %g second(s) \tNb convolutions/s : %g\n",
         end - start, nbiter / (end - start));

  /* Comparison with CPU execution: */
  {
    int i, j;
    bool same = true;

    printf("Checking the results with CPU calculations: ");
    fflush(stdout);
    memcpy(cpu_buf, r.data, h * w * sizeof(unsigned char));

    start = my_gettimeofday();
    cpu_buf_res = multiple_convolutions(cpu_buf, cpu_buf_aux, h, w, nbiter);
    end = my_gettimeofday();

    for (i = 0; i < h; i++)
    {
      for (j = 0; j < w; j++)
      {
        if (fabsf(cpu_buf_res[i * w + j] - buf_res[i * w + j]) > 0.01f)
        {
          fprintf(stderr, "\n cpu_buf_res[%d,%d]=%d differs from buf_res[%d,%d]=%d ",
                  i, j, cpu_buf_res[i * w + j], i, j, buf_res[i * w + j]);
          same = false;
        }
      }
    }
    printf((same ? " Ok!\n" : "\n -> Ko! \n"));

    printf("Total CPU time: %g second(s) \tNb convolutions/s : %g\n",
           end - start, nbiter / (end - start));
  }

  /* Storing the final result in 'r.data': */
  memcpy(r.data, buf_res, h * w * sizeof(unsigned char));

  /* Saving the Raster file */
  {
    char output_name_nbIter[256];
    sprintf(output_name_nbIter, "%s_nbIter%d.ras", output_name, nbiter);
    save_rasterfile(output_name_nbIter, &r);
  }

  /* Free memory: */
  free(buf);
  free(buf_aux);
  free(cpu_buf);
  free(cpu_buf_aux);

  return 0;
}
