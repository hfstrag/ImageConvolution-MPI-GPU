#include <mpi.h>

#define main main_seq
#include "convo.c"
#undef main

int main(int argc, char *argv[]) {

  /* Variables relating to the image */
  Raster r;
  int    w, h;	/* number of lines and columns of the image */

  /* Variables related to the image processing */
  int 	 filter;		/* filter number */
  int 	 nbiter;		/* number of iterations */

  /* Variables related to timing */
  double start, end;

  /* Variables related to MPI */
  int    my_rank;	/* process rank */
  int    p;		/* number of processes */
  int    up, down;	/* ranks of the neighbors above and below */
  int    nbl;		/* number of lines per process */
  int    dims[2];	/* h and w sent by process 0 */
  unsigned char *local;	/* my lines + 2 ghost lines */

  /* Loop variables */
  int 	i;
  int    local_ok;


  /* Initialization */
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  if (argc != 4) {
    if (my_rank == 0) fprintf( stderr, usage, argv[0]);
    MPI_Finalize();
    return 1;
  }

  /* Parameter entry */
  if (!parse_filter_arg(argv[2], &filter) ||
      !parse_nonnegative_int(argv[3], &nbiter)) {
    if (my_rank == 0) {
      fprintf(stderr, "Error: filter must be 0-4 and nbiter a non-negative integer.\n");
      fprintf(stderr, usage, argv[0]);
    }
    MPI_Finalize();
    return 1;
  }

  /* Reading the Raster file */
  if (my_rank == 0) {
    read_rasterfile( argv[1], &r);
    dims[0] = r.file.ras_height;
    dims[1] = r.file.ras_width;
  }
  MPI_Bcast(dims, 2, MPI_INT, 0, MPI_COMM_WORLD);
  h = dims[0];
  w = dims[1];

  if (h % p != 0) {
    if (my_rank == 0) fprintf(stderr, "The height (%d) must be divisible by %d\n", h, p);
    MPI_Finalize();
    return 1;
  }

  nbl = h / p;
  local = malloc((nbl + 2) * w);
  local_ok = local != NULL;
  {
    int all_ok;
    MPI_Allreduce(&local_ok, &all_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (!all_ok) {
      if (my_rank == 0) fprintf(stderr, "Error: unable to allocate MPI image buffers.\n");
      free(local);
      if (my_rank == 0) free(r.data);
      MPI_Finalize();
      return 1;
    }
  }

  up   = (my_rank == 0)     ? MPI_PROC_NULL : my_rank - 1;
  down = (my_rank == p - 1) ? MPI_PROC_NULL : my_rank + 1;

  /* Start of timing */
  start = my_gettimeofday();

  /* Distribution of the lines */
  MPI_Scatter(r.data, nbl * w, MPI_UNSIGNED_CHAR,
              local + w, nbl * w, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

  /* The convolution */
  for(i=0 ; i < nbiter ; i++){
    /* Ghost lines exchange */
    MPI_Sendrecv(local + w, w, MPI_UNSIGNED_CHAR, up, 0,
                 local + (nbl + 1) * w, w, MPI_UNSIGNED_CHAR, down, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(local + nbl * w, w, MPI_UNSIGNED_CHAR, down, 0,
                 local, w, MPI_UNSIGNED_CHAR, up, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    /* Duplicate boundary rows for the processes on the image edges. */
    if (up == MPI_PROC_NULL) {
      memcpy(local, local + w, w);
    }
    if (down == MPI_PROC_NULL) {
      memcpy(local + (nbl + 1) * w, local + nbl * w, w);
    }

    convolution(filter, local, nbl + 2, w);
  } /* for i */

  /* Gathering of the lines */
  MPI_Gather(local + w, nbl * w, MPI_UNSIGNED_CHAR,
             r.data, nbl * w, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

  /* End of timing */
  end = my_gettimeofday();

  if (my_rank == 0) {
    printf("Total calculation time: %g second(s) \n", end - start);

    /* Saving the Raster file */
    {
      char output_name[100] = "";
      sprintf(output_name, "post-convolution_filter%d_nbIter%d.ras", filter, nbiter);
      save_rasterfile(output_name, &r);
    }
  }

  /* Termination */
  free(local);
  MPI_Finalize();

  return 0;
}
