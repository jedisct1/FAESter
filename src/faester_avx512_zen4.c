#include "faester_avx512.h"
#include <string.h>

// Round constants derived from binary expansion of π - precomputed for all lanes at once
// This simplifies and speeds up constant loading - a major optimization
static const uint32_t RC[8][16] __attribute__((aligned(64))) = {
    // Precomputed constants for offset 0, 1, 2, 3
    {
        0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344,  // i+0
        0xa4093822, 0x299f31d0, 0x082efa98, 0xec4e6c89,  // i+1
        0x452821e6, 0x38d01377, 0xbe5466cf, 0x34e90c6c,  // i+2
        0xc0ac29b7, 0xc97c50dd, 0x3f84d5b5, 0xb5470917   // i+3
    },
    // Precomputed constants for offset 4, 5, 6, 7
    {
        0x9216d5d9, 0x8979fb1b, 0xd1310ba6, 0x98dfb5ac,  // i+4
        0x2ffd72db, 0xd01adfb7, 0xb8e1afed, 0x6a267e96,  // i+5
        0xba7c9045, 0xf12c7f99, 0x24a19947, 0xb3916cf7,  // i+6
        0x801f2e28, 0x58efc166, 0x36920d87, 0x1574e690   // i+7
    },
    // Precomputed constants for offset 8, 9, 10, 11
    {
        0xa458fea3, 0xf4933d7e, 0x0d95748f, 0x728eb658,  // i+8
        0x718bcd58, 0x82154aee, 0x7b54a41d, 0xc25a59b5,  // i+9
        0x9c30d539, 0x2af26013, 0xc5d1b023, 0x286085f0,  // i+10
        0xca417918, 0xb8db38ef, 0x8e79dcb0, 0x603a180e   // i+11
    },
    // Precomputed constants for offset 12, 13, 14, 15
    {
        0x6c9e0e8b, 0xb01e8a3e, 0xd71577c1, 0xbd314b27,  // i+12
        0x78af2fda, 0x55605c60, 0xe65525f3, 0xaa55ab94,  // i+13
        0x57489862, 0x63e81440, 0x55ca396a, 0x2aab10b6,  // i+14
        0xb4cc5c34, 0x1141e8ce, 0xa15486af, 0x7c72e993   // i+15
    },
    // Precomputed constants for offset 16, 17, 18, 19
    {
        0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344,  // Repeat of earlier constants (mod 32)
        0xa4093822, 0x299f31d0, 0x082efa98, 0xec4e6c89, 
        0x452821e6, 0x38d01377, 0xbe5466cf, 0x34e90c6c,
        0xc0ac29b7, 0xc97c50dd, 0x3f84d5b5, 0xb5470917
    },
    // Precomputed constants for offset 20, 21, 22, 23
    {
        0x9216d5d9, 0x8979fb1b, 0xd1310ba6, 0x98dfb5ac,  // Repeat of earlier constants (mod 32)
        0x2ffd72db, 0xd01adfb7, 0xb8e1afed, 0x6a267e96,
        0xba7c9045, 0xf12c7f99, 0x24a19947, 0xb3916cf7,
        0x801f2e28, 0x58efc166, 0x36920d87, 0x1574e690
    },
    // Precomputed constants for offset 24, 25, 26, 27
    {
        0xa458fea3, 0xf4933d7e, 0x0d95748f, 0x728eb658,  // Repeat of earlier constants (mod 32)
        0x718bcd58, 0x82154aee, 0x7b54a41d, 0xc25a59b5,
        0x9c30d539, 0x2af26013, 0xc5d1b023, 0x286085f0,
        0xca417918, 0xb8db38ef, 0x8e79dcb0, 0x603a180e
    },
    // Precomputed constants for offset 28, 29, 30, 31
    {
        0x6c9e0e8b, 0xb01e8a3e, 0xd71577c1, 0xbd314b27,  // Repeat of earlier constants (mod 32)
        0x78af2fda, 0x55605c60, 0xe65525f3, 0xaa55ab94,
        0x57489862, 0x63e81440, 0x55ca396a, 0x2aab10b6,
        0xb4cc5c34, 0x1141e8ce, 0xa15486af, 0x7c72e993
    }
};

// Helper for loading data into __m512i
static inline __m512i load_512(const void *ptr) {
    return _mm512_loadu_si512((const __m512i *) ptr);
}

// Helper for storing __m512i data
static inline void store_512(void *const ptr, const __m512i v) {
    _mm512_storeu_si512((__m512i *) ptr, v);
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
// This version uses a completely different approach with two key optimizations:
// 1. Process two rounds at once to amortize memory operations
// 2. Use precomputed round constants to minimize instruction count
void faester_permute(faester_state_t *state, const int rounds) {
    // Load state into registers
    __m512i a = state->blocks[0];
    __m512i b = state->blocks[1];
    __m512i c = state->blocks[2];
    __m512i d = state->blocks[3];
    
    // Fast round constant access to improve cache locality
    const __m512i *rc_table = (__m512i*)RC;
    
    // OPTIMIZATION: Process rounds in pairs to reduce memory ops and improve pipelining
    int r = 0;
    while (r + 1 < rounds) {
        // Process two rounds at once
        // Round r
        // Load constants for round r
        const __m512i rc0 = rc_table[r % 8];
        const __m512i rc1 = rc_table[(r+2) % 8];
        const __m512i rc2 = rc_table[(r+4) % 8];
        const __m512i rc3 = rc_table[(r+6) % 8];
        
        // First AES operation
        __m512i t0 = _mm512_aesenc_epi128(a, _mm512_xor_si512(b, rc0));
        __m512i t1 = _mm512_aesenc_epi128(b, _mm512_xor_si512(c, rc1));
        __m512i t2 = _mm512_aesenc_epi128(c, _mm512_xor_si512(d, rc2));
        __m512i t3 = _mm512_aesenc_epi128(d, _mm512_xor_si512(a, rc3));
        
        // Shuffling - fused to use fewer registers
        a = _mm512_shuffle_i32x4(t0, t0, 0x4E);
        b = _mm512_shuffle_i32x4(t1, t1, 0x93);
        c = _mm512_shuffle_i32x4(t2, t2, 0x39);
        d = _mm512_shuffle_i32x4(t3, t3, 0xC6);
        
        // Mixing - straight through implementation to minimize register pressure
        t0 = a;
        // Mix 1: Combine b and d parts
        __m512i mix_bd = _mm512_or_si512(_mm512_srli_epi32(b, 8), _mm512_slli_epi32(d, 24));
        a = _mm512_xor_si512(a, mix_bd);
        
        // Mix 2: Combine c and original a parts
        __m512i mix_ca = _mm512_or_si512(_mm512_srli_epi32(c, 8), _mm512_slli_epi32(t0, 24));
        b = _mm512_xor_si512(b, mix_ca);
        
        // Mix 3: Combine d and new b parts
        __m512i mix_db = _mm512_or_si512(_mm512_srli_epi32(d, 8), _mm512_slli_epi32(b, 24));
        c = _mm512_xor_si512(c, mix_db);
        
        // Mix 4: Combine original a and new c parts
        __m512i mix_ac = _mm512_or_si512(_mm512_srli_epi32(t0, 8), _mm512_slli_epi32(c, 24));
        d = _mm512_xor_si512(d, mix_ac);
        
        // Precompute combined values for second AES
        __m512i combined0 = _mm512_xor_si512(d, rc3);
        __m512i combined1 = _mm512_xor_si512(a, rc0);
        __m512i combined2 = _mm512_xor_si512(b, rc1);
        __m512i combined3 = _mm512_xor_si512(c, rc2);
        
        // Second AES
        a = _mm512_aesenc_epi128(a, combined0);
        b = _mm512_aesenc_epi128(b, combined1);
        c = _mm512_aesenc_epi128(c, combined2);
        d = _mm512_aesenc_epi128(d, combined3);
        
        // Round r+1
        // Load constants for round r+1
        const __m512i rc0b = rc_table[(r+1) % 8];
        const __m512i rc1b = rc_table[(r+3) % 8];
        const __m512i rc2b = rc_table[(r+5) % 8];
        const __m512i rc3b = rc_table[(r+7) % 8];
        
        // Third AES operation (first of next round)
        t0 = _mm512_aesenc_epi128(a, _mm512_xor_si512(b, rc0b));
        t1 = _mm512_aesenc_epi128(b, _mm512_xor_si512(c, rc1b));
        t2 = _mm512_aesenc_epi128(c, _mm512_xor_si512(d, rc2b));
        t3 = _mm512_aesenc_epi128(d, _mm512_xor_si512(a, rc3b));
        
        // Shuffling - again with minimal register use
        a = _mm512_shuffle_i32x4(t0, t0, 0x4E);
        b = _mm512_shuffle_i32x4(t1, t1, 0x93);
        c = _mm512_shuffle_i32x4(t2, t2, 0x39);
        d = _mm512_shuffle_i32x4(t3, t3, 0xC6);
        
        // Mixing - with new values (no need to store original a this time)
        // Mix 1: Combine b and d parts for next round
        mix_bd = _mm512_or_si512(_mm512_srli_epi32(b, 8), _mm512_slli_epi32(d, 24));
        t0 = a; // Save a for reuse in mix 2 and 4
        a = _mm512_xor_si512(a, mix_bd);
        
        // Mix 2: Combine c and original a parts
        mix_ca = _mm512_or_si512(_mm512_srli_epi32(c, 8), _mm512_slli_epi32(t0, 24));
        b = _mm512_xor_si512(b, mix_ca);
        
        // Mix 3: Combine d and new b parts
        mix_db = _mm512_or_si512(_mm512_srli_epi32(d, 8), _mm512_slli_epi32(b, 24));
        c = _mm512_xor_si512(c, mix_db);
        
        // Mix 4: Combine original a and new c parts
        mix_ac = _mm512_or_si512(_mm512_srli_epi32(t0, 8), _mm512_slli_epi32(c, 24));
        d = _mm512_xor_si512(d, mix_ac);
        
        // Last AES of double-round - direct computation again
        combined0 = _mm512_xor_si512(d, rc3b);
        combined1 = _mm512_xor_si512(a, rc0b);
        combined2 = _mm512_xor_si512(b, rc1b);
        combined3 = _mm512_xor_si512(c, rc2b);
        
        a = _mm512_aesenc_epi128(a, combined0);
        b = _mm512_aesenc_epi128(b, combined1);
        c = _mm512_aesenc_epi128(c, combined2);
        d = _mm512_aesenc_epi128(d, combined3);
        
        // Increment round counter by 2
        r += 2;
    }
    
    // Handle odd number of rounds if necessary
    if (r < rounds) {
        // Process final round
        const __m512i rc0 = rc_table[r % 8];
        const __m512i rc1 = rc_table[(r+2) % 8];
        const __m512i rc2 = rc_table[(r+4) % 8];
        const __m512i rc3 = rc_table[(r+6) % 8];
        
        // First AES
        __m512i t0 = _mm512_aesenc_epi128(a, _mm512_xor_si512(b, rc0));
        __m512i t1 = _mm512_aesenc_epi128(b, _mm512_xor_si512(c, rc1));
        __m512i t2 = _mm512_aesenc_epi128(c, _mm512_xor_si512(d, rc2));
        __m512i t3 = _mm512_aesenc_epi128(d, _mm512_xor_si512(a, rc3));
        
        // Shuffling
        a = _mm512_shuffle_i32x4(t0, t0, 0x4E);
        b = _mm512_shuffle_i32x4(t1, t1, 0x93);
        c = _mm512_shuffle_i32x4(t2, t2, 0x39);
        d = _mm512_shuffle_i32x4(t3, t3, 0xC6);
        
        // Mixing
        t0 = a; // Save original a
        
        // Mix operations
        __m512i mix_bd = _mm512_or_si512(_mm512_srli_epi32(b, 8), _mm512_slli_epi32(d, 24));
        a = _mm512_xor_si512(a, mix_bd);
        
        __m512i mix_ca = _mm512_or_si512(_mm512_srli_epi32(c, 8), _mm512_slli_epi32(t0, 24));
        b = _mm512_xor_si512(b, mix_ca);
        
        __m512i mix_db = _mm512_or_si512(_mm512_srli_epi32(d, 8), _mm512_slli_epi32(b, 24));
        c = _mm512_xor_si512(c, mix_db);
        
        __m512i mix_ac = _mm512_or_si512(_mm512_srli_epi32(t0, 8), _mm512_slli_epi32(c, 24));
        d = _mm512_xor_si512(d, mix_ac);
        
        // Second AES
        __m512i combined0 = _mm512_xor_si512(d, rc3);
        __m512i combined1 = _mm512_xor_si512(a, rc0);
        __m512i combined2 = _mm512_xor_si512(b, rc1);
        __m512i combined3 = _mm512_xor_si512(c, rc2);
        
        a = _mm512_aesenc_epi128(a, combined0);
        b = _mm512_aesenc_epi128(b, combined1);
        c = _mm512_aesenc_epi128(c, combined2);
        d = _mm512_aesenc_epi128(d, combined3);
    }
    
    // Store results
    state->blocks[0] = a;
    state->blocks[1] = b;
    state->blocks[2] = c;
    state->blocks[3] = d;
}