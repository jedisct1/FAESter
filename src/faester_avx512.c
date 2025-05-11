#include "faester_avx512.h"
#include <string.h>

// Round constants derived from binary expansion of π
static const uint32_t RC[32 * 4] = {
    0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344, 0xa4093822, 0x299f31d0, 0x082efa98, 0xec4e6c89,
    0x452821e6, 0x38d01377, 0xbe5466cf, 0x34e90c6c, 0xc0ac29b7, 0xc97c50dd, 0x3f84d5b5, 0xb5470917,
    0x9216d5d9, 0x8979fb1b, 0xd1310ba6, 0x98dfb5ac, 0x2ffd72db, 0xd01adfb7, 0xb8e1afed, 0x6a267e96,
    0xba7c9045, 0xf12c7f99, 0x24a19947, 0xb3916cf7, 0x801f2e28, 0x58efc166, 0x36920d87, 0x1574e690,
    0xa458fea3, 0xf4933d7e, 0x0d95748f, 0x728eb658, 0x718bcd58, 0x82154aee, 0x7b54a41d, 0xc25a59b5,
    0x9c30d539, 0x2af26013, 0xc5d1b023, 0x286085f0, 0xca417918, 0xb8db38ef, 0x8e79dcb0, 0x603a180e,
    0x6c9e0e8b, 0xb01e8a3e, 0xd71577c1, 0xbd314b27, 0x78af2fda, 0x55605c60, 0xe65525f3, 0xaa55ab94,
    0x57489862, 0x63e81440, 0x55ca396a, 0x2aab10b6, 0xb4cc5c34, 0x1141e8ce, 0xa15486af, 0x7c72e993
};

#define RC_BLOCK(i, offset)                                                                     \
    _mm512_setr_epi32(RC[((i + offset) % 32) * 4 + 0], RC[((i + offset) % 32) * 4 + 1],         \
                      RC[((i + offset) % 32) * 4 + 2], RC[((i + offset) % 32) * 4 + 3],         \
                      RC[((i + offset + 1) % 32) * 4 + 0], RC[((i + offset + 1) % 32) * 4 + 1], \
                      RC[((i + offset + 1) % 32) * 4 + 2], RC[((i + offset + 1) % 32) * 4 + 3], \
                      RC[((i + offset + 2) % 32) * 4 + 0], RC[((i + offset + 2) % 32) * 4 + 1], \
                      RC[((i + offset + 2) % 32) * 4 + 2], RC[((i + offset + 2) % 32) * 4 + 3], \
                      RC[((i + offset + 3) % 32) * 4 + 0], RC[((i + offset + 3) % 32) * 4 + 1], \
                      RC[((i + offset + 3) % 32) * 4 + 2], RC[((i + offset + 3) % 32) * 4 + 3])

// Helper for loading data into __m512i
static inline __m512i
load_512(const void *ptr)
{
    return _mm512_loadu_si512((const __m512i *) ptr);
}

// Helper for storing __m512i data
static inline void
store_512(void *const ptr, const __m512i v)
{
    _mm512_storeu_si512((__m512i *) ptr, v);
}

// Initialize the permutation state with input data
void
faester_init(faester_state_t *state, const uint8_t *input)
{
    state->blocks[0] = load_512(input);
    state->blocks[1] = load_512(input + 64);
    state->blocks[2] = load_512(input + 128);
    state->blocks[3] = load_512(input + 192);
}

// Extract output from the permutation state
void
faester_extract(const faester_state_t *state, uint8_t *output)
{
    store_512(output, state->blocks[0]);
    store_512(output + 64, state->blocks[1]);
    store_512(output + 128, state->blocks[2]);
    store_512(output + 192, state->blocks[3]);
}

// Permutation
void
faester_permute(faester_state_t *state, const int rounds)
{
    __m512i a, b, c, d;
    __m512i t0, t1, t2, t3;

    a = state->blocks[0];
    b = state->blocks[1];
    c = state->blocks[2];
    d = state->blocks[3];

    for (int r = 0; r < rounds; r++) {
        // Round constants from π
        const __m512i rc0 = RC_BLOCK(r, 0);
        const __m512i rc1 = RC_BLOCK(r, 8);
        const __m512i rc2 = RC_BLOCK(r, 16);
        const __m512i rc3 = RC_BLOCK(r, 24);

        // Step 1: AES operations
        // Pre-combine keys with round constants
        __m512i enhanced_b = _mm512_xor_si512(b, rc0);
        __m512i enhanced_c = _mm512_xor_si512(c, rc1);
        __m512i enhanced_d = _mm512_xor_si512(d, rc2);
        __m512i enhanced_a = _mm512_xor_si512(a, rc3);

        // Directly apply AESENC with pre-enhanced keys for maximum throughput
        t0 = _mm512_aesenc_epi128(a, enhanced_b);
        t1 = _mm512_aesenc_epi128(b, enhanced_c);
        t2 = _mm512_aesenc_epi128(c, enhanced_d);
        t3 = _mm512_aesenc_epi128(d, enhanced_a);

        // Step 2: Mixing across 128-bit lanes within 512-bit blocks
        a = _mm512_shuffle_i32x4(t0, t0, 0x4E); // Shuffle 128-bit lanes: 0,1,2,3 -> 2,3,0,1
        b = _mm512_shuffle_i32x4(t1, t1, 0x93); // Shuffle 128-bit lanes: 0,1,2,3 -> 1,0,3,2
        c = _mm512_shuffle_i32x4(t2, t2, 0x39); // Shuffle 128-bit lanes: 0,1,2,3 -> 3,2,1,0
        d = _mm512_shuffle_i32x4(t3, t3, 0xC6); // Shuffle 128-bit lanes: 0,1,2,3 -> 0,2,1,3

        // Step 3: Mixing between 512-bit blocks
        t0 = a;
        t1 = b;
        t2 = c;
        t3 = d;

        // Rotations
        __m512i shift1 = _mm512_srli_epi32(t1, 8);
        __m512i shift3 = _mm512_slli_epi32(t3, 24);
        a = _mm512_xor_si512(t0, _mm512_or_si512(shift1, shift3)); // Mix with parts of b and d

        __m512i shift2 = _mm512_srli_epi32(t2, 8);
        __m512i shift0 = _mm512_slli_epi32(t0, 24);
        b = _mm512_xor_si512(t1, _mm512_or_si512(shift2, shift0)); // Mix with parts of c and a

        __m512i shift3b = _mm512_srli_epi32(t3, 8);
        __m512i shift1b = _mm512_slli_epi32(t1, 24);
        c = _mm512_xor_si512(t2, _mm512_or_si512(shift3b, shift1b)); // Mix with parts of d and b

        __m512i shift0b = _mm512_srli_epi32(t0, 8);
        __m512i shift2b = _mm512_slli_epi32(t2, 24);
        d = _mm512_xor_si512(t3, _mm512_or_si512(shift0b, shift2b)); // Mix with parts of a and c

        // Step 4: Second batch of AESENCs for additional diffusion
        // Pre-computed combined constants
        __m512i combined0 = _mm512_xor_si512(d, rc3);
        __m512i combined1 = _mm512_xor_si512(a, rc0);
        __m512i combined2 = _mm512_xor_si512(b, rc1);
        __m512i combined3 = _mm512_xor_si512(c, rc2);

        // Apply AESENC directly with pre-combined keys
        a = _mm512_aesenc_epi128(a, combined0);
        b = _mm512_aesenc_epi128(b, combined1);
        c = _mm512_aesenc_epi128(c, combined2);
        d = _mm512_aesenc_epi128(d, combined3);
    }

    state->blocks[0] = a;
    state->blocks[1] = b;
    state->blocks[2] = c;
    state->blocks[3] = d;
}
