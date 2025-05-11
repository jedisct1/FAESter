#ifndef FAESTER_AVX512_H
#define FAESTER_AVX512_H

#include <stdint.h>

#ifdef __AVX512F__
#    include <immintrin.h>
#else
#    include "untrinsics/untrinsics_avx512.h"
#endif

// 2048-bit permutation (16 × 128-bit blocks)
typedef struct {
    __m512i blocks[4]; // 4 × 512 bits = 2048 bits
} faester_state_t;

// Initialize the permutation state with input data
void faester_init(faester_state_t *state, const uint8_t *input);

// Apply the permutation to the state
void faester_permute(faester_state_t *state, int rounds);

// Extract output from the permutation state
void faester_extract(const faester_state_t *state, uint8_t *output);

// Number of recommended rounds
#define FAESTER_RECOMMENDED_ROUNDS 8

#endif // faester_PERM_AVX512_H
