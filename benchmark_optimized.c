#define _GNU_SOURCE  /* Define this first, before any includes */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <math.h>
#include "test/areion.h"

#ifdef __APPLE__
#include <mach/mach_time.h>
#include <mach/mach.h>
#include <mach/thread_policy.h>
#include <mach/thread_act.h>
#include <pthread.h>
#elif defined(__linux__)
#include <x86intrin.h>
#include <sched.h>
#include <pthread.h>
#else
#include <x86intrin.h>
#endif

// ANSI color codes for prettier output
#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"

// Include the state struct definition
typedef struct {
    __m512i blocks[4]; // 4 × 512 bits = 2048 bits
} faester_state_t;

// Define the recommended rounds
#define FAESTER_RECOMMENDED_ROUNDS 8

// Function prototypes for original faester
void faester_original_init(faester_state_t *state, const uint8_t *input);
void faester_original_permute(faester_state_t *state, int rounds);
void faester_original_extract(const faester_state_t *state, uint8_t *output);

// Function prototypes for optimized faester
void faester_optimized_init(faester_state_t *state, const uint8_t *input);
void faester_optimized_permute(faester_state_t *state, int rounds);
void faester_optimized_extract(const faester_state_t *state, uint8_t *output);

// Platform-specific cycle counter
#ifdef __APPLE__
// On macOS, we return time in nanoseconds
static uint64_t rdtsc(void) {
    static mach_timebase_info_data_t timebase_info;
    if (timebase_info.denom == 0) {
        mach_timebase_info(&timebase_info);
    }
    
    // Get the absolute time
    uint64_t time = mach_absolute_time();
    
    // Convert to nanoseconds
    return time * timebase_info.numer / timebase_info.denom;
}

// Pin thread to a specific core on macOS
static void pin_to_core(int core_id) {
    thread_affinity_policy_data_t policy = { core_id };
    thread_port_t mach_thread = pthread_mach_thread_np(pthread_self());
    thread_policy_set(mach_thread, THREAD_AFFINITY_POLICY, 
                     (thread_policy_t)&policy, THREAD_AFFINITY_POLICY_COUNT);
}
#else
// x86 specific cycle counter
static uint64_t rdtsc(void) {
    unsigned int lo, hi;
    __asm__ volatile ("rdtsc" : "=a" (lo), "=d" (hi));
    return ((uint64_t)hi << 32) | lo;
}

// Pin thread to a specific core on Linux
static void pin_to_core(int core_id) {
#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#else
    // On other platforms, provide a stub implementation
    (void)core_id; // Avoid unused parameter warning
#endif
}
#endif

// Memory barrier to prevent reordering
static inline void memory_barrier(void) {
    __asm__ volatile("" ::: "memory");
}

// Function to estimate CPU frequency
static double estimate_cpu_freq_ghz(void) {
    uint64_t start, end;
    struct timespec t_start, t_end;
    
    // Do multiple measurements and take the maximum
    double max_ghz = 0.0;
    
    for (int i = 0; i < 10; i++) {
        clock_gettime(CLOCK_MONOTONIC, &t_start);
        memory_barrier();
        start = rdtsc();
        memory_barrier();
        
        // On macOS, rdtsc returns ns, so we need a longer wait time
        #ifdef __APPLE__
        // Busy-wait for about 250ms
        uint64_t wait_ns = 250000000;
        #else
        // Busy-wait for about 100ms in cycles (assuming ~3GHz)
        uint64_t wait_ns = 300000000;
        #endif
        
        uint64_t target = start + wait_ns;
        while (rdtsc() < target) {
            // Prevent optimization
            memory_barrier();
        }
        
        memory_barrier();
        end = rdtsc();
        memory_barrier();
        clock_gettime(CLOCK_MONOTONIC, &t_end);
        
        double elapsed_sec = (t_end.tv_sec - t_start.tv_sec) + 
                            (t_end.tv_nsec - t_start.tv_nsec) / 1e9;
        double cycles = end - start;
        
        double current_ghz;
        
        #ifdef __APPLE__
        // On macOS, rdtsc returns ns, so convert to equivalent cycles
        // using the ratio of nanoseconds to seconds
        current_ghz = 1.0; // 1 GHz (1 cycle per ns)
        // Prevent unused variable warnings
        (void)elapsed_sec;
        (void)cycles;
        #else
        // On x86, rdtsc returns actual cycles
        current_ghz = cycles / (elapsed_sec * 1e9);
        #endif
        
        if (current_ghz > max_ghz) {
            max_ghz = current_ghz;
        }
    }
    
    #ifdef __APPLE__
    // For Apple Silicon M1/M2, use a reasonable approximation
    // based on the chip's known performance characteristics
    char *arch = getenv("ARCH");
    if (arch && strstr(arch, "arm64")) {
        return 3.2; // A reasonable approximation for M1/M2
    }
    #endif
    
    return max_ghz;
}

// Statistical functions
typedef struct {
    double min;
    double max;
    double mean;
    double median;
    double stddev;
} stats_t;

static stats_t calculate_stats(uint64_t measurements[], int count) {
    stats_t stats = {0};
    
    if (count <= 0) return stats;
    
    // Sort measurements for median
    uint64_t* sorted = malloc(count * sizeof(uint64_t));
    memcpy(sorted, measurements, count * sizeof(uint64_t));
    
    for (int i = 0; i < count; i++) {
        for (int j = i + 1; j < count; j++) {
            if (sorted[i] > sorted[j]) {
                uint64_t temp = sorted[i];
                sorted[i] = sorted[j];
                sorted[j] = temp;
            }
        }
    }
    
    // Calculate statistics
    stats.min = sorted[0];
    stats.max = sorted[count-1];
    
    double sum = 0;
    for (int i = 0; i < count; i++) {
        sum += sorted[i];
    }
    stats.mean = sum / count;
    
    // Median
    if (count % 2 == 0) {
        stats.median = (sorted[count/2 - 1] + sorted[count/2]) / 2.0;
    } else {
        stats.median = sorted[count/2];
    }
    
    // Standard deviation
    double variance_sum = 0;
    for (int i = 0; i < count; i++) {
        double diff = sorted[i] - stats.mean;
        variance_sum += diff * diff;
    }
    stats.stddev = sqrt(variance_sum / count);
    
    free(sorted);
    return stats;
}

#define BENCHMARK_ITERATIONS 10000
#define WARMUP_ITERATIONS 1000

// Original FAESTER implementation (just the functions we need for benchmarking) 
// Round constants derived from binary expansion of π
static const uint32_t RC_ORIGINAL[32 * 4] = {
    0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344, 0xa4093822, 0x299f31d0, 0x082efa98, 0xec4e6c89,
    0x452821e6, 0x38d01377, 0xbe5466cf, 0x34e90c6c, 0xc0ac29b7, 0xc97c50dd, 0x3f84d5b5, 0xb5470917,
    0x9216d5d9, 0x8979fb1b, 0xd1310ba6, 0x98dfb5ac, 0x2ffd72db, 0xd01adfb7, 0xb8e1afed, 0x6a267e96,
    0xba7c9045, 0xf12c7f99, 0x24a19947, 0xb3916cf7, 0x801f2e28, 0x58efc166, 0x36920d87, 0x1574e690,
    0xa458fea3, 0xf4933d7e, 0x0d95748f, 0x728eb658, 0x718bcd58, 0x82154aee, 0x7b54a41d, 0xc25a59b5,
    0x9c30d539, 0x2af26013, 0xc5d1b023, 0x286085f0, 0xca417918, 0xb8db38ef, 0x8e79dcb0, 0x603a180e,
    0x6c9e0e8b, 0xb01e8a3e, 0xd71577c1, 0xbd314b27, 0x78af2fda, 0x55605c60, 0xe65525f3, 0xaa55ab94,
    0x57489862, 0x63e81440, 0x55ca396a, 0x2aab10b6, 0xb4cc5c34, 0x1141e8ce, 0xa15486af, 0x7c72e993
};

#define RC_BLOCK_ORIGINAL(i, offset)                                                                     \
    _mm512_setr_epi32(RC_ORIGINAL[((i + offset) % 32) * 4 + 0], RC_ORIGINAL[((i + offset) % 32) * 4 + 1],         \
                      RC_ORIGINAL[((i + offset) % 32) * 4 + 2], RC_ORIGINAL[((i + offset) % 32) * 4 + 3],         \
                      RC_ORIGINAL[((i + offset + 1) % 32) * 4 + 0], RC_ORIGINAL[((i + offset + 1) % 32) * 4 + 1], \
                      RC_ORIGINAL[((i + offset + 1) % 32) * 4 + 2], RC_ORIGINAL[((i + offset + 1) % 32) * 4 + 3], \
                      RC_ORIGINAL[((i + offset + 2) % 32) * 4 + 0], RC_ORIGINAL[((i + offset + 2) % 32) * 4 + 1], \
                      RC_ORIGINAL[((i + offset + 2) % 32) * 4 + 2], RC_ORIGINAL[((i + offset + 2) % 32) * 4 + 3], \
                      RC_ORIGINAL[((i + offset + 3) % 32) * 4 + 0], RC_ORIGINAL[((i + offset + 3) % 32) * 4 + 1], \
                      RC_ORIGINAL[((i + offset + 3) % 32) * 4 + 2], RC_ORIGINAL[((i + offset + 3) % 32) * 4 + 3])

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
void faester_original_init(faester_state_t *state, const uint8_t *input)
{
    state->blocks[0] = load_512(input);
    state->blocks[1] = load_512(input + 64);
    state->blocks[2] = load_512(input + 128);
    state->blocks[3] = load_512(input + 192);
}

// Extract output from the permutation state
void faester_original_extract(const faester_state_t *state, uint8_t *output)
{
    store_512(output, state->blocks[0]);
    store_512(output + 64, state->blocks[1]);
    store_512(output + 128, state->blocks[2]);
    store_512(output + 192, state->blocks[3]);
}

// Original permutation
void faester_original_permute(faester_state_t *state, const int rounds)
{
    __m512i a, b, c, d;
    __m512i t0, t1, t2, t3;

    a = state->blocks[0];
    b = state->blocks[1];
    c = state->blocks[2];
    d = state->blocks[3];

    for (int r = 0; r < rounds; r++) {
        // Round constants from π
        const __m512i rc0 = RC_BLOCK_ORIGINAL(r, 0);
        const __m512i rc1 = RC_BLOCK_ORIGINAL(r, 8);
        const __m512i rc2 = RC_BLOCK_ORIGINAL(r, 16);
        const __m512i rc3 = RC_BLOCK_ORIGINAL(r, 24);

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

// Optimized FAESTER implementation
// Round constants derived from binary expansion of π - precomputed for all lanes at once
static const uint32_t RC_OPTIMIZED[8][16] __attribute__((aligned(64))) = {
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
    return _mm512_loadu_si512((const __m512i*)RC_OPTIMIZED[index]);
}

// Initialize the permutation state with input data
void faester_optimized_init(faester_state_t *state, const uint8_t *input) {
    state->blocks[0] = load_512(input);
    state->blocks[1] = load_512(input + 64);
    state->blocks[2] = load_512(input + 128);
    state->blocks[3] = load_512(input + 192);
}

// Extract output from the permutation state
void faester_optimized_extract(const faester_state_t *state, uint8_t *output) {
    store_512(output, state->blocks[0]);
    store_512(output + 64, state->blocks[1]);
    store_512(output + 128, state->blocks[2]);
    store_512(output + 192, state->blocks[3]);
}

// Optimized permutation for Zen 4
// This version uses a completely different approach with two key optimizations:
// 1. Process two rounds at once to amortize memory operations
// 2. Use precomputed round constants to minimize instruction count
void faester_optimized_permute(faester_state_t *state, const int rounds) {
    // Load state into registers
    __m512i a = state->blocks[0];
    __m512i b = state->blocks[1];
    __m512i c = state->blocks[2];
    __m512i d = state->blocks[3];

    // Fast round constant access to improve cache locality
    const __m512i *rc_table = (__m512i*)RC_OPTIMIZED;

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

// AREION implementation is already included from the header

// Function pointer type for initialization, permutation, and extraction
typedef void (*init_fn_t)(void* state, const uint8_t* input);
typedef void (*permute_fn_t)(void* state, int rounds);
typedef void (*extract_fn_t)(const void* state, uint8_t* output);

// Benchmark a permutation function
static stats_t benchmark_permutation(
    const char* name, 
    size_t state_size,
    size_t input_size,
    init_fn_t init_fn,
    permute_fn_t permute_fn,
    extract_fn_t extract_fn,
    int rounds,
    double *throughput) {
    
    uint8_t *input = calloc(input_size, 1);
    uint8_t *output = calloc(input_size, 1);
    void *state = calloc(state_size, 1);
    
    // Initialize input with some data
    for (size_t i = 0; i < input_size; i++) {
        input[i] = (uint8_t)i;
    }
    
    // Initialize state
    init_fn(state, input);
    
    // Warmup phase
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        permute_fn(state, rounds);
    }
    
    // Prepare for measurement
    uint64_t *measurements = malloc(BENCHMARK_ITERATIONS * sizeof(uint64_t));
    
    // Actual benchmark
    printf("Benchmarking %s (%d rounds)...\n", name, rounds);
    
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
        memory_barrier();
        uint64_t start = rdtsc();
        memory_barrier();
        
        permute_fn(state, rounds);
        
        memory_barrier();
        uint64_t end = rdtsc();
        memory_barrier();
        
        measurements[i] = end - start;
    }
    
    // Extract output for verification
    extract_fn(state, output);
    
    // Verify output is not all zeros (simple check)
    int nonzero_count = 0;
    for (size_t i = 0; i < input_size; i++) {
        if (output[i] != 0) {
            nonzero_count++;
        }
    }
    printf("  Output verification: %zu non-zero bytes out of %zu\n", nonzero_count, input_size);
    
    // Calculate statistics
    stats_t stats = calculate_stats(measurements, BENCHMARK_ITERATIONS);
    
    // Calculate throughput
    double bytes = (double)input_size;
    double cpu_freq_ghz = estimate_cpu_freq_ghz();
    
    #ifdef __APPLE__
    // On macOS, our cycle counts are actually in nanoseconds
    // So we calculate GB/s directly
    double ns_per_byte = stats.median / bytes;
    double bytes_per_second = 1e9 / ns_per_byte;
    *throughput = bytes_per_second / 1e9; // GB/s
    #else
    // On x86, we have actual cycle counts
    double cycles_per_byte = stats.median / bytes;
    *throughput = (cpu_freq_ghz * 1e9) / (cycles_per_byte * 1e9);
    #endif
    
    free(measurements);
    free(state);
    free(input);
    free(output);
    
    return stats;
}

// Wrappers for FAESTER original
static void faester_original_init_wrapper(void* state, const uint8_t* input) {
    faester_original_init((faester_state_t*)state, input);
}

static void faester_original_permute_wrapper(void* state, int rounds) {
    faester_original_permute((faester_state_t*)state, rounds);
}

static void faester_original_extract_wrapper(const void* state, uint8_t* output) {
    faester_original_extract((const faester_state_t*)state, output);
}

// Wrappers for FAESTER optimized
static void faester_optimized_init_wrapper(void* state, const uint8_t* input) {
    faester_optimized_init((faester_state_t*)state, input);
}

static void faester_optimized_permute_wrapper(void* state, int rounds) {
    faester_optimized_permute((faester_state_t*)state, rounds);
}

static void faester_optimized_extract_wrapper(const void* state, uint8_t* output) {
    faester_optimized_extract((const faester_state_t*)state, output);
}

// Wrappers for AREION
static void areion_init_wrapper(void* state, const uint8_t* input) {
    areion512_init((areion512_state_t*)state, input);
}

static void areion_permute_wrapper(void* state, int rounds) {
    (void)rounds; // Unused, AREION always uses 15 rounds
    areion512_permute((areion512_state_t*)state);
}

static void areion_extract_wrapper(const void* state, uint8_t* output) {
    areion512_extract((const areion512_state_t*)state, output);
}

// Print side-by-side benchmark comparison
static void print_benchmark_comparison(stats_t original_stats, double original_throughput,
                                      stats_t optimized_stats, double optimized_throughput,
                                      stats_t areion_stats, double areion_throughput,
                                      int rounds) {
    double cpu_freq_ghz = estimate_cpu_freq_ghz();
    
    printf("\n");
    printf(ANSI_COLOR_CYAN "=================================================================================================================\n" ANSI_COLOR_RESET);
    printf(ANSI_COLOR_CYAN "       FAESTER (ORIGINAL vs OPTIMIZED) vs AREION PERFORMANCE COMPARISON       \n" ANSI_COLOR_RESET);
    printf(ANSI_COLOR_CYAN "=================================================================================================================\n" ANSI_COLOR_RESET);
    printf("\n");
    
    // Print CPU info
    printf("CPU Frequency: %.2f GHz\n", cpu_freq_ghz);
    printf("FAESTER Rounds: %d, AREION Rounds: 15 (fixed)\n\n", rounds);
    
    // Print permutation stats
    printf("┌────────────────────────┬─────────────────────┬─────────────────────┬─────────────────────┐\n");
    printf("│ Metric                 │ FAESTER ORIGINAL    │ FAESTER OPTIMIZED   │ AREION              │\n");
    printf("├────────────────────────┼─────────────────────┼─────────────────────┼─────────────────────┤\n");
    
    #ifdef __APPLE__
    printf("│ Min time (ns)          │ %-17.2f │ %-17.2f │ %-17.2f │\n", 
           original_stats.min, optimized_stats.min, areion_stats.min);
    printf("│ Max time (ns)          │ %-17.2f │ %-17.2f │ %-17.2f │\n", 
           original_stats.max, optimized_stats.max, areion_stats.max);
    printf("│ Mean time (ns)         │ %-17.2f │ %-17.2f │ %-17.2f │\n", 
           original_stats.mean, optimized_stats.mean, areion_stats.mean);
    printf("│ Median time (ns)       │ %-17.2f │ %-17.2f │ %-17.2f │\n", 
           original_stats.median, optimized_stats.median, areion_stats.median);
    printf("│ StdDev (ns)            │ %-17.2f │ %-17.2f │ %-17.2f │\n", 
           original_stats.stddev, optimized_stats.stddev, areion_stats.stddev);
    
    printf("│ Time per byte (ns)     │ %-17.2f │ %-17.2f │ %-17.2f │\n", 
           original_stats.median / 256.0, optimized_stats.median / 256.0, areion_stats.median / 64.0);
    
    // On macOS, convert to equivalent cycles per byte
    double original_equiv_cycles = (original_stats.median / 256.0) * cpu_freq_ghz;
    double optimized_equiv_cycles = (optimized_stats.median / 256.0) * cpu_freq_ghz;
    double areion_equiv_cycles = (areion_stats.median / 64.0) * cpu_freq_ghz;
    
    printf("│ Equiv. cycles per byte │ %-17.2f │ %-17.2f │ %-17.2f │\n", 
           original_equiv_cycles, optimized_equiv_cycles, areion_equiv_cycles);
    #else
    printf("│ Min cycles             │ %-17.2f │ %-17.2f │ %-17.2f │\n", 
           original_stats.min, optimized_stats.min, areion_stats.min);
    printf("│ Max cycles             │ %-17.2f │ %-17.2f │ %-17.2f │\n", 
           original_stats.max, optimized_stats.max, areion_stats.max);
    printf("│ Mean cycles            │ %-17.2f │ %-17.2f │ %-17.2f │\n", 
           original_stats.mean, optimized_stats.mean, areion_stats.mean);
    printf("│ Median cycles          │ %-17.2f │ %-17.2f │ %-17.2f │\n", 
           original_stats.median, optimized_stats.median, areion_stats.median);
    printf("│ StdDev cycles          │ %-17.2f │ %-17.2f │ %-17.2f │\n", 
           original_stats.stddev, optimized_stats.stddev, areion_stats.stddev);
    
    printf("│ Cycles per byte        │ %-17.2f │ %-17.2f │ %-17.2f │\n", 
           original_stats.median / 256.0, optimized_stats.median / 256.0, areion_stats.median / 64.0);
    #endif
    
    printf("│ Throughput (GB/s)      │ %-17.2f │ %-17.2f │ %-17.2f │\n", 
           original_throughput, optimized_throughput, areion_throughput);
    
    // Calculate relative metrics
    double original_per_bit = original_stats.median / (256.0 * 8); // time per bit
    double optimized_per_bit = optimized_stats.median / (256.0 * 8); // time per bit
    double areion_per_bit = areion_stats.median / (64.0 * 8); // time per bit
    
    double original_vs_areion = areion_per_bit / original_per_bit;
    double optimized_vs_areion = areion_per_bit / optimized_per_bit;
    double improvement = original_stats.median / optimized_stats.median;
    
    printf("│ Relative bit efficiency│ %-17.2f │ %-17.2f │ %-17s │\n", 
           original_vs_areion, optimized_vs_areion, "1.00 (reference)");
    printf("│ Speedup vs Original    │ %-17s │ %-17.2fx │ %-17s │\n", 
           "1.00x (reference)", improvement, "-");
    
    printf("└────────────────────────┴─────────────────────┴─────────────────────┴─────────────────────┘\n\n");
    
    // Improvement summary
    printf("Performance Improvement:\n");
    printf("- FAESTER optimized is %.2fx faster than the original implementation\n", improvement);
    
    if (optimized_vs_areion > 1.0) {
        printf("- FAESTER optimized is %.2fx more efficient per bit than AREION\n", optimized_vs_areion);
        printf("- FAESTER optimized is now FASTER than AREION when adjusted for state size!\n");
    } else if (optimized_vs_areion < 1.0) {
        printf("- AREION is still %.2fx more efficient per bit than optimized FAESTER\n", 1.0 / optimized_vs_areion);
        if (optimized_vs_areion > original_vs_areion) {
            printf("- But the gap has narrowed from %.2fx to %.2fx\n", 1.0 / original_vs_areion, 1.0 / optimized_vs_areion);
        }
    } else {
        printf("- FAESTER optimized and AREION now have equal bit-level efficiency\n");
    }
    
    // Side-by-side throughput comparison 
    if (optimized_throughput > areion_throughput) {
        printf("- FAESTER optimized throughput (%.2f GB/s) exceeds AREION (%.2f GB/s)\n",
               optimized_throughput, areion_throughput);
    } else if (optimized_throughput < areion_throughput) {
        printf("- AREION throughput (%.2f GB/s) exceeds FAESTER optimized (%.2f GB/s)\n", 
               areion_throughput, optimized_throughput);
    } else {
        printf("- FAESTER optimized and AREION have similar throughput (%.2f GB/s)\n", 
               optimized_throughput);
    }
    
    printf("\n");
}

int main(int argc, char *argv[]) {
    // Parse command line arguments
    int faester_rounds = FAESTER_RECOMMENDED_ROUNDS;
    if (argc > 1) {
        faester_rounds = atoi(argv[1]);
        if (faester_rounds <= 0) {
            faester_rounds = FAESTER_RECOMMENDED_ROUNDS;
        }
    }
    
    // Pin to core 1 to avoid system processes on core 0
    pin_to_core(1);
    
    // Print header
    printf(ANSI_COLOR_CYAN "==================================================================\n" ANSI_COLOR_RESET);
    printf(ANSI_COLOR_CYAN "       FAESTER (ORIGINAL vs OPTIMIZED) vs AREION BENCHMARK        \n" ANSI_COLOR_RESET);
    printf(ANSI_COLOR_CYAN "==================================================================\n" ANSI_COLOR_RESET);
    printf("\n");
    
    // Estimate CPU frequency
    double cpu_freq_ghz = estimate_cpu_freq_ghz();
    printf("Estimated CPU frequency: %.2f GHz\n\n", cpu_freq_ghz);
    
    // Benchmark FAESTER Original
    double original_throughput;
    stats_t original_stats = benchmark_permutation(
        "FAESTER Original", 
        sizeof(faester_state_t), 
        256,
        faester_original_init_wrapper, 
        faester_original_permute_wrapper, 
        faester_original_extract_wrapper,
        faester_rounds, 
        &original_throughput
    );
    
    // Benchmark FAESTER Optimized 
    double optimized_throughput;
    stats_t optimized_stats = benchmark_permutation(
        "FAESTER Optimized", 
        sizeof(faester_state_t), 
        256,
        faester_optimized_init_wrapper, 
        faester_optimized_permute_wrapper, 
        faester_optimized_extract_wrapper,
        faester_rounds, 
        &optimized_throughput
    );
    
    // Benchmark AREION
    double areion_throughput;
    stats_t areion_stats = benchmark_permutation(
        "AREION", 
        sizeof(areion512_state_t), 
        64,
        areion_init_wrapper, 
        areion_permute_wrapper, 
        areion_extract_wrapper,
        15, // AREION always uses 15 rounds
        &areion_throughput
    );
    
    // Print comparison
    print_benchmark_comparison(
        original_stats, original_throughput,
        optimized_stats, optimized_throughput,
        areion_stats, areion_throughput,
        faester_rounds
    );
    
    return 0;
}