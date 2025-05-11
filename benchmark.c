#define _GNU_SOURCE  /* Define this first, before any includes */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include "src/faester_avx512.h"

#ifdef __APPLE__
#include <mach/mach_time.h>
#elif defined(__linux__)
#include <x86intrin.h>
#include <sched.h>
#include <pthread.h>
#else
#include <x86intrin.h>
#endif

// Cycles measurement functions
#ifdef __APPLE__
// macOS specific time measurement with conversion to nanoseconds
static inline uint64_t rdtsc(void) {
    static mach_timebase_info_data_t timebase_info;
    if (timebase_info.denom == 0) {
        mach_timebase_info(&timebase_info);
    }
    
    // Get the absolute time
    uint64_t time = mach_absolute_time();
    
    // Convert to nanoseconds
    return time * timebase_info.numer / timebase_info.denom;
}
#else
// x86 specific cycle counter
static inline uint64_t rdtsc(void) {
    return __rdtsc();
}
#endif

// Add memory barrier to prevent reordering
static inline void memory_barrier(void) {
    __asm__ volatile("" ::: "memory");
}

// Function to estimate CPU frequency
static double estimate_cpu_freq_ghz(void) {
    uint64_t start, end;
    struct timespec t_start, t_end;
    
    clock_gettime(CLOCK_MONOTONIC, &t_start);
    memory_barrier();
    start = rdtsc();
    
    // Busy-wait for about 100ms
    uint64_t wait_time;
    #ifdef __APPLE__
    wait_time = 100000000; // 100ms in ns
    #else
    wait_time = 300000000; // 100ms assuming ~3GHz CPU
    #endif
    
    uint64_t target = start + wait_time;
    while (rdtsc() < target) {
        memory_barrier();
    }
    
    end = rdtsc();
    memory_barrier();
    clock_gettime(CLOCK_MONOTONIC, &t_end);
    
    double elapsed_sec = (t_end.tv_sec - t_start.tv_sec) +
                         (t_end.tv_nsec - t_start.tv_nsec) / 1e9;
    double cycles = end - start;

    #ifdef __APPLE__
    // On macOS, our time is already in nanoseconds
    // Avoid unused variable warnings
    (void)elapsed_sec;
    (void)cycles;
    return 1.0; // Use 1 GHz as reference (1 cycle per ns)
    #else
    // On x86, convert actual cycles to frequency
    return cycles / (elapsed_sec * 1e9);
    #endif
}

#define BENCHMARK_ITERATIONS 10000
#define WARMUP_ITERATIONS 1000

int main(void) {
    // Estimate CPU frequency
    double cpu_freq_ghz = estimate_cpu_freq_ghz();
    
    // Prepare input and output buffers with volatile to prevent optimization
    volatile uint8_t input[256];
    volatile uint8_t output[256];
    faester_state_t state;
    
    // Initialize input with some data
    for (int i = 0; i < 256; i++) {
        ((uint8_t*)input)[i] = (uint8_t)i;
    }
    
    // Initialize state
    faester_init(&state, (const uint8_t*)input);
    
    printf("Faester Permutation Benchmark\n");
    printf("-----------------------------\n");
    printf("Estimated CPU frequency: %.2f GHz\n\n", cpu_freq_ghz);
    
    // Warmup phase - this ensures CPU is at full speed and caches are warm
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        faester_permute(&state, FAESTER_RECOMMENDED_ROUNDS);
    }
    
    // Benchmark with different round counts
    int rounds[] = {4, 6, 8, 10, 12};
    for (size_t r = 0; r < sizeof(rounds)/sizeof(rounds[0]); r++) {
        int round_count = rounds[r];
        
        // Reset state for this benchmark
        faester_init(&state, (const uint8_t*)input);
        
        // Run benchmark in batches to improve measurement accuracy
        uint64_t min_cycles = UINT64_MAX;
        uint64_t max_cycles = 0;
        uint64_t total_cycles = 0;
        
        for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
            // Add memory barriers before and after measurement to prevent reordering
            memory_barrier();
            uint64_t start = rdtsc();
            memory_barrier();
            
            // Run the operation we're measuring
            faester_permute(&state, round_count);
            
            memory_barrier();
            uint64_t end = rdtsc();
            memory_barrier();
            
            uint64_t cycles = end - start;
            total_cycles += cycles;
            
            if (cycles < min_cycles) min_cycles = cycles;
            if (cycles > max_cycles) max_cycles = cycles;
        }
        
        // Extract output for verification
        faester_extract(&state, (uint8_t*)output);
        
        // Calculate average cycles
        double avg_cycles = (double)total_cycles / BENCHMARK_ITERATIONS;
        
        #ifdef __APPLE__
        printf("Rounds: %2d | Average time: %10.2f ns | ", round_count, avg_cycles);
        printf("Time per byte: %6.2f ns | ", avg_cycles / 256.0);
        printf("Equivalent cycles/byte: %6.2f (at %.2f GHz)\n", 
               (avg_cycles / 256.0) * cpu_freq_ghz, cpu_freq_ghz);
        #else
        printf("Rounds: %2d | Average cycles: %10.2f | ", round_count, avg_cycles);
        printf("Cycles per byte: %6.2f | ", avg_cycles / 256.0);
        printf("Throughput: %6.2f GB/s\n", 
               (cpu_freq_ghz * 1e9) / ((avg_cycles / 256.0) * 1e9));
        #endif
    }
    
    // Verify output is not all zeros (simple check)
    int nonzero_count = 0;
    for (int i = 0; i < 256; i++) {
        if (((uint8_t*)output)[i] != 0) {
            nonzero_count++;
        }
    }
    
    printf("\nOutput verification: %d non-zero bytes out of 256\n", nonzero_count);
    
    return 0;
}