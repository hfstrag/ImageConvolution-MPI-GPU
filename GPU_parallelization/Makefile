# ================================================================== #
#  Makefile — Lab 2 Convolution GPU                                  #
# ================================================================== #

NVCC    = nvcc
# Baseline équitable : on compile le code CPU hôte avec les mêmes flags que MPI
NVFLAGS = -arch=sm_60 -Wno-deprecated-gpu-targets -O3 -Xcompiler "-O3 -march=native -mtune=native -ffast-math"
LIBS    = -lm
BIN     = gpu_convol_v0

IMAGES  = femme10.ras Sukhothai_4080x6132.ras Sukhothai_4096x6156.ras

# bench_block
BX_LIST = 1 2 4 8 16 32 64 128 256 512 1024
BY_LIST = 1 2 4 8 16 32

# bench_iter
BX_FIXED = 16
BY_FIXED = 16

RESDIR  = results

.PHONY: all run clean bench bench_block bench_iter parse

all: $(BIN)

$(BIN): $(BIN).cu rasterfile.h
	$(NVCC) $(NVFLAGS) -o $@ $< $(LIBS)

$(RESDIR):
	mkdir -p $(RESDIR)

# ------------------------------------------------------------------ #
#  bench_block (Adaptatif selon l'image)                             #
# ------------------------------------------------------------------ #
bench_block: all $(RESDIR)
	@echo "=== bench_block (Adaptatif selon l'image) ==="
	@for img in $(IMAGES); do \
		[ -f "$$img" ] || { echo "WARNING: $$img not found"; continue; }; \
		img_base=$$(basename $$img .ras); \
		if [ "$$img_base" = "femme10" ]; then \
			cur_iter=10000; \
		else \
			cur_iter=200; \
		fi; \
		for by in $(BY_LIST); do \
			for bx in $(BX_LIST); do \
				[ $$((bx * by)) -le 1024 ] || continue ; \
				out=$(RESDIR)/gpu__$${img_base}__b$${bx}x$${by}__iter$${cur_iter}.txt ; \
				echo "--- image=$$img_base block=$${bx}x$${by} nbiter=$$cur_iter ---" ; \
				./$(BIN) $$img $$cur_iter $$bx $$by > $$out 2>&1 \
					|| echo "    FAILED: $$(head -n 1 $$out)" ; \
			done ; \
		done \
	done
	@echo "=== bench_block done. ==="

# ------------------------------------------------------------------ #
#  bench_iter (Adaptatif selon l'image)                              #
# ------------------------------------------------------------------ #
bench_iter: all $(RESDIR)
	@echo "=== bench_iter (Adaptatif selon l'image) ==="
	@for img in $(IMAGES); do \
		[ -f "$$img" ] || { echo "WARNING: $$img not found"; continue; }; \
		img_base=$$(basename $$img .ras); \
		if [ "$$img_base" = "femme10" ]; then \
			cur_iter_list="1 10 100 1000 10000 100000"; \
		else \
			cur_iter_list="1 10 50 100 500 1000 5000"; \
		fi; \
		for nbiter in $$cur_iter_list; do \
			out=$(RESDIR)/gpu__$${img_base}__b$(BX_FIXED)x$(BY_FIXED)__iter$${nbiter}.txt ; \
			echo "--- image=$$img_base block=$(BX_FIXED)x$(BY_FIXED) nbiter=$$nbiter ---" ; \
			./$(BIN) $$img $$nbiter $(BX_FIXED) $(BY_FIXED) > $$out 2>&1 \
				|| echo "    FAILED: $$(head -n 1 $$out)" ; \
		done \
	done
	@echo "=== bench_iter done. ==="

bench: bench_block bench_iter

parse: $(RESDIR)
	@echo "image,bx,by,nbiter,gpu_time_s,cpu_time_s,check" > $(RESDIR)/results.csv
	@for f in $(RESDIR)/gpu__*.txt; do \
		[ -f "$$f" ] || continue ; \
		base=$$(basename $$f .txt) ; \
		img=$$(echo $$base | awk -F'__' '{print $$2}') ; \
		grep -q "Total GPU time" $$f || { echo "WARNING: $$f failed" >&2; continue; } ; \
		awk -v img="$$img" -v OFS=, \
			'/^Blocks/ {split($$3, b, "x"); n = $$6} \
			 /Total GPU time/ {g = $$4} \
			 /Total CPU time/ {c = $$4} \
			 /Ok/ {ok = 1} \
			 END {print img, b[1], b[2], n, g, c, (g == "" ? "error" : ok ? "ok" : "KO")}' $$f ; \
	done | sort -t, -k1,1 -k2,2n -k3,3n -k4,4n >> $(RESDIR)/results.csv
	@echo "=== CSV written to $(RESDIR)/results.csv ==="

clean:
	rm -f $(BIN)
	rm -rf $(RESDIR)
	rm -f /tmp/$$(whoami)/post-convolution_*