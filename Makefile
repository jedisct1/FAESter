CC = gcc
CFLAGS = -O3 -Wall -Wextra
AVX512_FLAGS = -mavx512f -maes -march=native

# Platform detection
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

# Check for Apple Silicon and skip AVX512 flags if detected
ifeq ($(UNAME_M),arm64)
    # Apple Silicon - cannot use AVX512
    AVX512_FLAGS =
endif

# Add platform-specific flags
ifeq ($(UNAME_S),Linux)
    # Add Linux-specific flags
    PLATFORM_FLAGS = -pthread
else
    PLATFORM_FLAGS =
endif

# Combine all flags
ALL_FLAGS = $(CFLAGS) $(AVX512_FLAGS) $(PLATFORM_FLAGS)

SRC_DIR = src
TEST_DIR = test
SRCS = $(SRC_DIR)/faester_avx512.c
SRCS_OPT = $(SRC_DIR)/faester_avx512_optimized.c
AREION_SRCS = $(TEST_DIR)/areion.c
HEADERS = $(SRC_DIR)/faester_avx512.h $(SRC_DIR)/untrinsics/untrinsics.h $(SRC_DIR)/untrinsics/untrinsics_avx512.h
HEADERS_OPT = $(SRC_DIR)/faester_avx512_optimized.h $(SRC_DIR)/untrinsics/untrinsics.h $(SRC_DIR)/untrinsics/untrinsics_avx512.h
AREION_HEADERS = $(TEST_DIR)/areion.h

.PHONY: all clean bench run compare

all: benchmark benchmark_advanced avalanche avalanche_detailed crypto_properties diffusion_test compare_permutations benchmark_compare benchmark_optimized faester_zen4

bench: benchmark benchmark_advanced

benchmark: benchmark.c $(SRCS) $(HEADERS)
	$(CC) $(ALL_FLAGS) -o $@ $< $(SRCS) -lm

benchmark_advanced: benchmark_advanced.c $(SRCS) $(HEADERS)
	$(CC) $(ALL_FLAGS) -o $@ $< $(SRCS) -lm

avalanche: avalanche.c $(SRCS) $(HEADERS)
	$(CC) $(ALL_FLAGS) -o $@ $< $(SRCS) -lm

avalanche_detailed: avalanche_detailed.c $(SRCS) $(HEADERS)
	$(CC) $(ALL_FLAGS) -o $@ $< $(SRCS) -lm

crypto_properties: crypto_properties.c $(SRCS) $(HEADERS)
	$(CC) $(ALL_FLAGS) -o $@ $< $(SRCS) -lm

diffusion_test: diffusion_test.c $(SRCS) $(HEADERS)
	$(CC) $(ALL_FLAGS) -o $@ $< $(SRCS) -lm

compare_permutations: compare_permutations.c $(SRCS) $(AREION_SRCS) $(HEADERS) $(AREION_HEADERS)
	$(CC) $(ALL_FLAGS) -o $@ $< $(SRCS) $(AREION_SRCS) -lm

benchmark_compare: benchmark_compare.c $(SRCS) $(AREION_SRCS) $(HEADERS) $(AREION_HEADERS)
	$(CC) $(ALL_FLAGS) -o $@ $< $(SRCS) $(AREION_SRCS) -lm

benchmark_optimized: benchmark_optimized.c $(AREION_SRCS) $(AREION_HEADERS)
	$(CC) $(ALL_FLAGS) -o $@ $< $(AREION_SRCS) -lm

# For Zen 4, we add special optimization flags
ZEN4_FLAGS = -march=znver4 -mtune=znver4 -funroll-loops -ftree-vectorize

# If znver4 is not supported (e.g., older GCC), fall back to znver3 or generic
ZEN4_FLAGS := $(shell if $(CC) -march=znver4 -E - </dev/null >/dev/null 2>&1; then echo "$(ZEN4_FLAGS)"; \
              elif $(CC) -march=znver3 -E - </dev/null >/dev/null 2>&1; then echo "-march=znver3 -mtune=znver3 -funroll-loops -ftree-vectorize"; \
              else echo "-funroll-loops -ftree-vectorize"; fi)

faester_zen4: $(SRC_DIR)/faester_avx512_zen4.c $(HEADERS)
	$(CC) $(ALL_FLAGS) $(ZEN4_FLAGS) -o faester_zen4 -DFAESTER_ZEN4 benchmark.c $(SRC_DIR)/faester_avx512_zen4.c -lm

clean:
	rm -f benchmark benchmark_advanced avalanche avalanche_detailed crypto_properties diffusion_test compare_permutations benchmark_compare benchmark_optimized faester_zen4

run: benchmark
	./benchmark

run_advanced: benchmark_advanced
	./benchmark_advanced

run_avalanche: avalanche
	./avalanche

run_avalanche_detailed: avalanche_detailed
	./avalanche_detailed

run_crypto: crypto_properties
	./crypto_properties

run_diffusion: diffusion_test
	./diffusion_test

run_compare: compare_permutations
	./compare_permutations

run_bench_compare: benchmark_compare
	./benchmark_compare

run_optimized: benchmark_optimized
	./benchmark_optimized

run_zen4: faester_zen4
	./faester_zen4