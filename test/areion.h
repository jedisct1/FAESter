#ifndef AREION_H
#define AREION_H

#include <stdint.h>
#if defined(__AES__) && defined(__AVX__)
#    include <immintrin.h>
#else
#    include "../src/untrinsics/untrinsics_avx512.h"
#endif

// AREION-512 state (4 x 128-bit blocks = 512 bits)
typedef struct {
    __m128i blocks[4];
} areion512_state_t;

// Initialize the AREION-512 state with input data
void areion512_init(areion512_state_t *state, const uint8_t *input);

// Apply the AREION-512 permutation (15 rounds)
// This follows the reference implementation from the paper
void areion512_permute(areion512_state_t *state);

// Extract output from the AREION-512 state
void areion512_extract(const areion512_state_t *state, uint8_t *output);

// Direct permutation function matching the reference implementation
// Permutes input directly to output without intermediate state
void permute_areion_512(__m128i dst[4], const __m128i src[4]);

#endif // AREION_H
