# ================================================================== #
#  Makefile — Lab 2 Convolution MPI                                  #
#  Université de Lille / Grid5000 chiclet cluster                    #
# ================================================================== #

# ------------------------------------------------------------------ #
#  Tools                                                             #
# ------------------------------------------------------------------ #
MPICC   = mpicc
CFLAGS  = -O3 -Wall -march=native -mtune=native -ffast-math
LIBS    = -lm

# ------------------------------------------------------------------ #
#  Binaries                                                          #
# ------------------------------------------------------------------ #
BIN_BLO    = parallel_blocking
BIN_NONBLO = parallelnon_blocking

# ------------------------------------------------------------------ #
#  Benchmark parameters                                              #
# ------------------------------------------------------------------ #
IMAGES  = femme10.ras Sukhothai_4080x6132.ras Sukhothai_4096x6156.ras
FILTER  = 0

# bench_iter : p fixé
P_FIXED    = 32

RESDIR     = results
MPIRUN     = mpirun --mca pml ^ucx -machinefile $$OAR_NODEFILE

# ------------------------------------------------------------------ #
#  Compilation                                                       #
# ------------------------------------------------------------------ #
.PHONY: all clean bench bench_p bench_iter parse plot

all: $(BIN_BLO) $(BIN_NONBLO)

$(BIN_BLO): parallel_blocking.c convo.c rasterfile.h
	$(MPICC) $(CFLAGS) -o $@ parallel_blocking.c $(LIBS)

$(BIN_NONBLO): parallelnon_blocking.c convo.c rasterfile.h
	$(MPICC) $(CFLAGS) -o $@ parallelnon_blocking.c $(LIBS)

# ------------------------------------------------------------------ #
#  Result directory                                                  #
# ------------------------------------------------------------------ #
$(RESDIR):
	mkdir -p $(RESDIR)

# ------------------------------------------------------------------ #
#  bench_p (Adaptatif selon l'image)                                 #
# ------------------------------------------------------------------ #
bench_p: all $(RESDIR)
	@echo "=== bench_p (Adaptatif selon l'image) ==="
	@for img in $(IMAGES); do \
		[ -f "$$img" ] || { echo "WARNING: $$img not found"; continue; }; \
		img_base=$$(basename $$img .ras); \
		if [ "$$img_base" = "femme10" ]; then \
			cur_p_list="1 2 4 8 16 32"; cur_iter=1000000; \
		else \
			cur_p_list="8 16 32"; cur_iter=2000; \
		fi; \
		for p in $$cur_p_list; do \
			echo "--- Blocking    image=$$img_base p=$$p nbiter=$$cur_iter ---" ; \
			$(MPIRUN) -np $$p ./$(BIN_BLO) $$img $(FILTER) $$cur_iter \
				> $(RESDIR)/blo__$${img_base}__p$${p}__iter$${cur_iter}.txt 2>&1 ; \
			echo "--- NonBlocking image=$$img_base p=$$p nbiter=$$cur_iter ---" ; \
			$(MPIRUN) -np $$p ./$(BIN_NONBLO) $$img $(FILTER) $$cur_iter \
				> $(RESDIR)/nonblo__$${img_base}__p$${p}__iter$${cur_iter}.txt 2>&1 ; \
		done \
	done
	@echo "=== bench_p done. ==="

# ------------------------------------------------------------------ #
#  bench_iter (Adaptatif selon l'image)                              #
# ------------------------------------------------------------------ #
bench_iter: all $(RESDIR)
	@echo "=== bench_iter (Adaptatif selon l'image) ==="
	@for img in $(IMAGES); do \
		[ -f "$$img" ] || { echo "WARNING: $$img not found"; continue; }; \
		img_base=$$(basename $$img .ras); \
		if [ "$$img_base" = "femme10" ]; then \
			cur_iter_list="10 100 1000 10000 100000 1000000"; \
		else \
			cur_iter_list="10 50 100 500 1000 5000"; \
		fi; \
		for nbiter in $$cur_iter_list; do \
			echo "--- Blocking    image=$$img_base p=$(P_FIXED) nbiter=$$nbiter ---" ; \
			$(MPIRUN) -np $(P_FIXED) ./$(BIN_BLO) $$img $(FILTER) $$nbiter \
				> $(RESDIR)/blo__$${img_base}__p$(P_FIXED)__iter$${nbiter}.txt 2>&1 ; \
			echo "--- NonBlocking image=$$img_base p=$(P_FIXED) nbiter=$$nbiter ---" ; \
			$(MPIRUN) -np $(P_FIXED) ./$(BIN_NONBLO) $$img $(FILTER) $$nbiter \
				> $(RESDIR)/nonblo__$${img_base}__p$(P_FIXED)__iter$${nbiter}.txt 2>&1 ; \
		done \
	done
	@echo "=== bench_iter done. ==="

# ------------------------------------------------------------------ #
#  bench — run both                                                  #
# ------------------------------------------------------------------ #
bench: bench_p bench_iter

# ------------------------------------------------------------------ #
#  parse — extract times from result files, write CSV                #
# ------------------------------------------------------------------ #
parse: $(RESDIR)
	@echo "algo,image,p,nbiter,time_s" > $(RESDIR)/results.csv
	@for f in $(RESDIR)/blo__*.txt $(RESDIR)/nonblo__*.txt; do \
		[ -f "$$f" ] || continue ; \
		base=$$(basename $$f .txt) ; \
		algo=$$(echo $$base | awk -F'__' '{print $$1}') ; \
		img=$$(echo $$base | awk -F'__' '{print $$2}') ; \
		p=$$(echo $$base | awk -F'__' '{print $$3}' | sed 's/p//') ; \
		nbiter=$$(echo $$base | awk -F'__' '{print $$4}' | sed 's/iter//') ; \
		time=$$(grep "Total calculation time" $$f | awk '{print $$4}') ; \
		if [ -n "$$time" ]; then \
			echo "$$algo,$$img,$$p,$$nbiter,$$time" >> $(RESDIR)/results.csv ; \
		else \
			echo "WARNING: no time found in $$f" >&2 ; \
		fi ; \
	done
	@echo "=== CSV written to $(RESDIR)/results.csv ==="

# ------------------------------------------------------------------ #
#  clean                                                             #
# ------------------------------------------------------------------ #
clean:
	rm -f $(BIN_BLO) $(BIN_NONBLO) post-convolution_*.ras
	rm -rf $(RESDIR) fig*.pdf fig*.png