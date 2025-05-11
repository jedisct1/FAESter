#ifndef FAESTER_AVX512_OPTIMIZED_H
#define FAESTER_AVX512_OPTIMIZED_H

#include <stdint.h>
#include "faester_avx512.h" // Include the original header which has the type definitions

// We don't redefine types here, we use the ones from the original header
// This file is just to mark that we're using the optimized implementation

// Only the implementation of this function changes in the optimized version:
void faester_permute(faester_state_t *state, int rounds);

#endif // FAESTER_AVX512_OPTIMIZED_H