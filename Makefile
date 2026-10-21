.PHONY: all mpi cuda test test-cpu test-mpi clean

all: mpi cuda

mpi:
	$(MAKE) -C MPI_parallelization

cuda:
	$(MAKE) -C GPU_parallelization

test: test-cpu

test-cpu:
	$(MAKE) -C MPI_parallelization test-cpu

test-mpi:
	$(MAKE) -C MPI_parallelization all
	python3 tests/test_mpi.py

clean:
	$(MAKE) -C MPI_parallelization clean
	$(MAKE) -C GPU_parallelization clean
