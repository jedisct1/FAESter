#include "areion.h"
#include <string.h>

/* Round Constants - Exactly as in the reference implementation */
static const uint32_t RC[24*4] = {
    0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344,
    0xa4093822, 0x299f31d0, 0x082efa98, 0xec4e6c89,
    0x452821e6, 0x38d01377, 0xbe5466cf, 0x34e90c6c,
    0xc0ac29b7, 0xc97c50dd, 0x3f84d5b5, 0xb5470917,
    0x9216d5d9, 0x8979fb1b, 0xd1310ba6, 0x98dfb5ac,
    0x2ffd72db, 0xd01adfb7, 0xb8e1afed, 0x6a267e96,
    0xba7c9045, 0xf12c7f99, 0x24a19947, 0xb3916cf7,
    0x801f2e28, 0x58efc166, 0x36920d87, 0x1574e690,
    0xa458fea3, 0xf4933d7e, 0x0d95748f, 0x728eb658,
    0x718bcd58, 0x82154aee, 0x7b54a41d, 0xc25a59b5,
    0x9c30d539, 0x2af26013, 0xc5d1b023, 0x286085f0,
    0xca417918, 0xb8db38ef, 0x8e79dcb0, 0x603a180e,
    0x6c9e0e8b, 0xb01e8a3e, 0xd71577c1, 0xbd314b27,
    0x78af2fda, 0x55605c60, 0xe65525f3, 0xaa55ab94,
    0x57489862, 0x63e81440, 0x55ca396a, 0x2aab10b6,
    0xb4cc5c34, 0x1141e8ce, 0xa15486af, 0x7c72e993,
    0xb3ee1411, 0x636fbc2a, 0x2ba9c55d, 0x741831f6,
    0xce5c3e16, 0x9b87931e, 0xafd6ba33, 0x6c24cf5c,
    0x7a325381, 0x28958677, 0x3b8f4898, 0x6b4bb9af,
    0xc4bfe81b, 0x66282193, 0x61d809cc, 0xfb21a991,
    0x487cac60, 0x5dec8032, 0xef845d5d, 0xe98575b1,
    0xdc262302, 0xeb651b88, 0x23893e81, 0xd396acc5,
    0xf6d6ff38, 0x3f442392, 0xe0b4482a, 0x48420040,
    0x69c8f04a, 0x9e1f9b5e, 0x21c66842, 0xf6e96c9a
};

/* Macros for accessing round constants */
#define RC0(i) _mm_setr_epi32(RC[(i)*4+0], RC[(i)*4+1], RC[(i)*4+2], RC[(i)*4+3])
#define RC1(i) _mm_setr_epi32(0, 0, 0, 0)

/* Round Function for the 512-bit permutation - Exactly as in the reference */
#define Round_Function_512(x0, x1, x2, x3, i) do { \
    x1 = _mm_aesenc_si128(x0, x1); \
    x3 = _mm_aesenc_si128(x2, x3); \
    x0 = _mm_aesenclast_si128(x0, RC1(i)); \
    x2 = _mm_aesenc_si128(_mm_aesenclast_si128(x2, RC0(i)), RC1(i)); \
} while(0)

/* 512-bit permutation - Exactly as in the reference */
#define perm512(x0, x1, x2, x3) do { \
    Round_Function_512(x0, x1, x2, x3, 0); \
    Round_Function_512(x1, x2, x3, x0, 1); \
    Round_Function_512(x2, x3, x0, x1, 2); \
    Round_Function_512(x3, x0, x1, x2, 3); \
    Round_Function_512(x0, x1, x2, x3, 4); \
    Round_Function_512(x1, x2, x3, x0, 5); \
    Round_Function_512(x2, x3, x0, x1, 6); \
    Round_Function_512(x3, x0, x1, x2, 7); \
    Round_Function_512(x0, x1, x2, x3, 8); \
    Round_Function_512(x1, x2, x3, x0, 9); \
    Round_Function_512(x2, x3, x0, x1, 10); \
    Round_Function_512(x3, x0, x1, x2, 11); \
    Round_Function_512(x0, x1, x2, x3, 12); \
    Round_Function_512(x1, x2, x3, x0, 13); \
    Round_Function_512(x2, x3, x0, x1, 14); \
} while(0)

// Helper for loading data into __m128i
static inline __m128i load_128(const void *ptr) {
    return _mm_loadu_si128((const __m128i*)ptr);
}

// Helper for storing __m128i data
static inline void store_128(void *ptr, __m128i v) {
    _mm_storeu_si128((__m128i*)ptr, v);
}

// Initialize the AREION-512 state with input data
void areion512_init(areion512_state_t *state, const uint8_t *input) {
    state->blocks[0] = load_128(input);
    state->blocks[1] = load_128(input + 16);
    state->blocks[2] = load_128(input + 32);
    state->blocks[3] = load_128(input + 48);
}

// Extract output from the AREION-512 state
void areion512_extract(const areion512_state_t *state, uint8_t *output) {
    store_128(output, state->blocks[0]);
    store_128(output + 16, state->blocks[1]);
    store_128(output + 32, state->blocks[2]);
    store_128(output + 48, state->blocks[3]);
}

// Direct permutation function that exactly matches the reference implementation
void permute_areion_512(__m128i dst[4], const __m128i src[4]) {
    __m128i x0 = src[0];
    __m128i x1 = src[1];
    __m128i x2 = src[2];
    __m128i x3 = src[3];
    
    perm512(x0, x1, x2, x3);
    
    dst[0] = x0;
    dst[1] = x1;
    dst[2] = x2;
    dst[3] = x3;
}

// Implementation of AREION-512 permutation using our state structure
// Uses the reference implementation's function internally
void areion512_permute(areion512_state_t *state) {
    __m128i src[4], dst[4];
    
    // Copy from state to local array
    src[0] = state->blocks[0];
    src[1] = state->blocks[1];
    src[2] = state->blocks[2];
    src[3] = state->blocks[3];
    
    // Use the reference permutation function
    permute_areion_512(dst, src);
    
    // Copy back to state
    state->blocks[0] = dst[0];
    state->blocks[1] = dst[1];
    state->blocks[2] = dst[2];
    state->blocks[3] = dst[3];
}