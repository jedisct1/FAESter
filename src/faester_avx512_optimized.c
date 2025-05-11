#include "faester_avx512_optimized.h"
#include <string.h>

// Rename the original permute function
#undef faester_permute  // Undefine any previous definition to avoid conflicts

// Round constants derived from binary expansion of π - precomputed for all lanes at once
// This simplifies and speeds up constant loading - a major optimization
static const uint32_t RC[8][16] __attribute__((aligned(64))) = {
    // Precomputed constants for offset 0, 1, 2, 3 (for rc0)
    {
        0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344,  // i+0
        0xa4093822, 0x299f31d0, 0x082efa98, 0xec4e6c89,  // i+1
        0x452821e6, 0x38d01377, 0xbe5466cf, 0x34e90c6c,  // i+2
        0xc0ac29b7, 0xc97c50dd, 0x3f84d5b5, 0xb5470917   // i+3
    },
    // Precomputed constants for offset 4, 5, 6, 7 (for rc0)
    {
        0x9216d5d9, 0x8979fb1b, 0xd1310ba6, 0x98dfb5ac,  // i+4
        0x2ffd72db, 0xd01adfb7, 0xb8e1afed, 0x6a267e96,  // i+5
        0xba7c9045, 0xf12c7f99, 0x24a19947, 0xb3916cf7,  // i+6
        0x801f2e28, 0x58efc166, 0x36920d87, 0x1574e690   // i+7
    },
    // Precomputed constants for offset 8, 9, 10, 11 (for rc1)
    {
        0xa458fea3, 0xf4933d7e, 0x0d95748f, 0x728eb658,  // i+8
        0x718bcd58, 0x82154aee, 0x7b54a41d, 0xc25a59b5,  // i+9
        0x9c30d539, 0x2af26013, 0xc5d1b023, 0x286085f0,  // i+10
        0xca417918, 0xb8db38ef, 0x8e79dcb0, 0x603a180e   // i+11
    },
    // Precomputed constants for offset 12, 13, 14, 15 (for rc1)
    {
        0x6c9e0e8b, 0xb01e8a3e, 0xd71577c1, 0xbd314b27,  // i+12
        0x78af2fda, 0x55605c60, 0xe65525f3, 0xaa55ab94,  // i+13
        0x57489862, 0x63e81440, 0x55ca396a, 0x2aab10b6,  // i+14
        0xb4cc5c34, 0x1141e8ce, 0xa15486af, 0x7c72e993   // i+15
    },
    // Precomputed constants for offset 16, 17, 18, 19 (for rc2)
    {
        0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344,  // Repeat of earlier constants (mod 32)
        0xa4093822, 0x299f31d0, 0x082efa98, 0xec4e6c89, 
        0x452821e6, 0x38d01377, 0xbe5466cf, 0x34e90c6c,
        0xc0ac29b7, 0xc97c50dd, 0x3f84d5b5, 0xb5470917
    },
    // Precomputed constants for offset 20, 21, 22, 23 (for rc2)
    {
        0x9216d5d9, 0x8979fb1b, 0xd1310ba6, 0x98dfb5ac,  // Repeat of earlier constants (mod 32)
        0x2ffd72db, 0xd01adfb7, 0xb8e1afed, 0x6a267e96,
        0xba7c9045, 0xf12c7f99, 0x24a19947, 0xb3916cf7,
        0x801f2e28, 0x58efc166, 0x36920d87, 0x1574e690
    },
    // Precomputed constants for offset 24, 25, 26, 27 (for rc3)
    {
        0xa458fea3, 0xf4933d7e, 0x0d95748f, 0x728eb658,  // Repeat of earlier constants (mod 32)
        0x718bcd58, 0x82154aee, 0x7b54a41d, 0xc25a59b5,
        0x9c30d539, 0x2af26013, 0xc5d1b023, 0x286085f0,
        0xca417918, 0xb8db38ef, 0x8e79dcb0, 0x603a180e
    },
    // Precomputed constants for offset 28, 29, 30, 31 (for rc3)
    {
        0x6c9e0e8b, 0xb01e8a3e, 0xd71577c1, 0xbd314b27,  // Repeat of earlier constants (mod 32)
        0x78af2fda, 0x55605c60, 0xe65525f3, 0xaa55ab94,
        0x57489862, 0x63e81440, 0x55ca396a, 0x2aab10b6,
        0xb4cc5c34, 0x1141e8ce, 0xa15486af, 0x7c72e993
    }
};

// Precompute shuffle masks for faster 128-bit lane shuffling
static const int SHUFFLE_MASK_1 = 0x4E; // Shuffle: 0,1,2,3 -> 2,3,0,1
static const int SHUFFLE_MASK_2 = 0x93; // Shuffle: 0,1,2,3 -> 1,0,3,2
static const int SHUFFLE_MASK_3 = 0x39; // Shuffle: 0,1,2,3 -> 3,2,1,0
static const int SHUFFLE_MASK_4 = 0xC6; // Shuffle: 0,1,2,3 -> 0,2,1,3

// Fast load RC constant
static inline __m512i load_rc(int r, int offset) {
    int index = (r + offset) % 8; // Wrap around after 8 rounds (matches the constant array)
    return _mm512_loadu_si512((const __m512i*)RC[index]);
}

// Helper for loading data into __m512i
static inline __m512i load_512(const void *ptr) {
    return _mm512_loadu_si512((const __m512i*)ptr);
}

// Helper for storing __m512i data
static inline void store_512(void *const ptr, const __m512i v) {
    _mm512_storeu_si512((__m512i*)ptr, v);
}

// Initialize the permutation state with input data
void faester_init(faester_state_t *state, const uint8_t *input) {
    state->blocks[0] = load_512(input);
    state->blocks[1] = load_512(input + 64);
    state->blocks[2] = load_512(input + 128);
    state->blocks[3] = load_512(input + 192);
}

// Extract output from the permutation state
void faester_extract(const faester_state_t *state, uint8_t *output) {
    store_512(output, state->blocks[0]);
    store_512(output + 64, state->blocks[1]);
    store_512(output + 128, state->blocks[2]);
    store_512(output + 192, state->blocks[3]);
}

// Optimized permutation for Zen 4
void faester_permute(faester_state_t *state, const int rounds) {
    // Load state into registers
    __m512i a = state->blocks[0];
    __m512i b = state->blocks[1];
    __m512i c = state->blocks[2];
    __m512i d = state->blocks[3];
    __m512i t0, t1;

    for (int r = 0; r < rounds; r++) {
        // OPTIMIZATION 1: Faster constant loading with precomputed constants 
        // Requires fewer registers and calculations
        const __m512i rc0 = load_rc(r, 0);
        const __m512i rc1 = load_rc(r, 2);
        const __m512i rc2 = load_rc(r, 4);
        const __m512i rc3 = load_rc(r, 6);

        // OPTIMIZATION 2: Reduce variable count in first stage of AES operations
        // Directly use enhanced values to reduce register pressure
        t0 = _mm512_aesenc_epi128(a, _mm512_xor_si512(b, rc0));
        t1 = _mm512_aesenc_epi128(b, _mm512_xor_si512(c, rc1));
        c = _mm512_aesenc_epi128(c, _mm512_xor_si512(d, rc2)); 
        d = _mm512_aesenc_epi128(d, _mm512_xor_si512(a, rc3));
        
        // Use t0, t1 to hold values while c,d already processed
        a = t0;
        b = t1;

        // OPTIMIZATION 3: Static shuffle masks and fused operations
        // Shuffle operations - use the precomputed constants
        t0 = _mm512_shuffle_i32x4(a, a, SHUFFLE_MASK_1);
        t1 = _mm512_shuffle_i32x4(b, b, SHUFFLE_MASK_2);
        a = _mm512_shuffle_i32x4(c, c, SHUFFLE_MASK_3);
        b = _mm512_shuffle_i32x4(d, d, SHUFFLE_MASK_4);
        
        // Store shuffled values in appropriate registers
        c = t0;
        d = t1;

        // OPTIMIZATION 4: Reduce temporary variables in mixing operations
        // Maintain register contents in a-d as much as possible
        // Save original values before mixing
        t0 = a; // Store 'a' for later use
        
        // Mix state - do this in-place when possible to reduce register pressure
        t1 = _mm512_or_si512(_mm512_srli_epi32(b, 8), _mm512_slli_epi32(d, 24));
        a = _mm512_xor_si512(a, t1);
        
        t1 = _mm512_or_si512(_mm512_srli_epi32(c, 8), _mm512_slli_epi32(t0, 24));
        b = _mm512_xor_si512(b, t1);
        
        t1 = _mm512_or_si512(_mm512_srli_epi32(d, 8), _mm512_slli_epi32(b, 24));
        c = _mm512_xor_si512(c, t1);
        
        t1 = _mm512_or_si512(_mm512_srli_epi32(t0, 8), _mm512_slli_epi32(c, 24));
        d = _mm512_xor_si512(d, t1);
        
        // OPTIMIZATION 5: Fuse operations in second AES round
        // Apply final AES round - reusing rc values from before
        a = _mm512_aesenc_epi128(a, _mm512_xor_si512(d, rc3));
        b = _mm512_aesenc_epi128(b, _mm512_xor_si512(a, rc0));
        c = _mm512_aesenc_epi128(c, _mm512_xor_si512(b, rc1));
        d = _mm512_aesenc_epi128(d, _mm512_xor_si512(c, rc2));
    }

    // Store results
    state->blocks[0] = a;
    state->blocks[1] = b;
    state->blocks[2] = c;
    state->blocks[3] = d;
}