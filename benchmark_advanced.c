#define _GNU_SOURCE  /* Define this first, before any includes */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <math.h>
#include "src/faester_avx512.h"

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

// Cycles measurement functions
#ifdef __APPLE__
// On macOS, we return time in nanoseconds, which we'll later
// convert to "equivalent cycles" based on measured CPU frequency
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
    return __rdtsc();
}

// Pin thread to a specific core on Linux
static void pin_to_core(int core_id) {
#ifdef __linux__
    /* Need to define cpu_set_t and related macros if _GNU_SOURCE wasn't defined early enough */
    #ifndef CPU_ZERO
    typedef unsigned long int cpu_set_t;
    #define CPU_SETSIZE 1024
    #define __NCPUBITS (8 * sizeof(unsigned long))
    #define CPU_SET(cpu, cpusetp) ((*(cpusetp)) |= (1UL << ((cpu) % __NCPUBITS)))
    #define CPU_ZERO(cpusetp) (*(cpusetp) = 0)
    #endif
    
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

#define BENCHMARK_ITERATIONS 10000
#define WARMUP_ITERATIONS 1000
#define MIN_TIME_SEC 2.0

// Function to estimate CPU frequency
static double estimate_cpu_freq_ghz(void) {
    uint64_t start, end;
    struct timespec t_start, t_end;
    
    // Do multiple measurements and take the maximum (most accurate)
    double max_ghz = 0.0;
    
    for (int i = 0; i < 10; i++) {
        clock_gettime(CLOCK_MONOTONIC, &t_start);
        start = rdtsc();
        
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
            __asm__ volatile("" ::: "memory");
        }
        
        end = rdtsc();
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

int main(void) {
    // Pin to a specific core (use core 1 to avoid system processes on core 0)
    pin_to_core(1);
    
    // Estimate CPU frequency
    double cpu_freq_ghz = estimate_cpu_freq_ghz();
    printf("Estimated CPU frequency: %.2f GHz\n\n", cpu_freq_ghz);
    
    // Prepare input and output buffers
    uint8_t input[256] = {0};
    uint8_t output[256] = {0};
    faester_state_t state;
    
    // Initialize input with some data
    for (int i = 0; i < 256; i++) {
        input[i] = (uint8_t)i;
    }
    
    printf("Faester Permutation Benchmark\n");
    printf("============================\n");
    
    // Benchmark with different round counts
    int rounds[] = {4, 6, 8, 10, 12};
    
    for (size_t r = 0; r < sizeof(rounds)/sizeof(rounds[0]); r++) {
        int round_count = rounds[r];
        
        printf("\nBenchmarking with %d rounds:\n", round_count);
        printf("---------------------------\n");
        
        // Warmup phase
        faester_init(&state, input);
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            faester_permute(&state, round_count);
        }
        
        // Prepare for measurement
        uint64_t *measurements = malloc(BENCHMARK_ITERATIONS * sizeof(uint64_t));
        if (!measurements) {
            fprintf(stderr, "Memory allocation failed\n");
            return 1;
        }
        
        // Actual benchmark
        faester_init(&state, input);
        
        int iterations = 0;
        double total_time = 0;
        struct timespec t_start, t_current;
        clock_gettime(CLOCK_MONOTONIC, &t_start);
        
        while (iterations < BENCHMARK_ITERATIONS) {
            uint64_t start = rdtsc();
            faester_permute(&state, round_count);
            uint64_t end = rdtsc();
            
            measurements[iterations++] = end - start;
            
            // Check if we've run long enough
            if (iterations % 100 == 0) {
                clock_gettime(CLOCK_MONOTONIC, &t_current);
                total_time = (t_current.tv_sec - t_start.tv_sec) + 
                             (t_current.tv_nsec - t_start.tv_nsec) / 1e9;
                if (total_time >= MIN_TIME_SEC && iterations >= 1000) {
                    break;
                }
            }
        }
        
        // Calculate statistics
        stats_t stats = calculate_stats(measurements, iterations);
        
        // Extract output for verification
        faester_extract(&state, output);
        
        // Display results
        printf("Iterations: %d (%.2f seconds)\n", iterations, total_time);

        #ifdef __APPLE__
        printf("Nanoseconds per permutation (2048 bits):\n");
        #else
        printf("Cycles per permutation (2048 bits):\n");
        #endif

        printf("  Minimum: %.2f\n", stats.min);
        printf("  Maximum: %.2f\n", stats.max);
        printf("  Mean:    %.2f\n", stats.mean);
        printf("  Median:  %.2f\n", stats.median);
        printf("  StdDev:  %.2f\n", stats.stddev);
        
        // Calculate cycles per byte
        double bytes = 256.0; // 2048 bits = 256 bytes

        #ifdef __APPLE__
        printf("\nNanoseconds per byte:\n");
        #else
        printf("\nCycles per byte:\n");
        #endif

        printf("  Minimum: %.2f\n", stats.min / bytes);
        printf("  Maximum: %.2f\n", stats.max / bytes);
        printf("  Mean:    %.2f\n", stats.mean / bytes);
        printf("  Median:  %.2f\n", stats.median / bytes);
        printf("  StdDev:  %.2f\n", stats.stddev / bytes);
        
        // Calculate throughput
        printf("\nThroughput:\n");

        #ifdef __APPLE__
        // On macOS, our cycle counts are actually in nanoseconds
        // So we calculate GB/s directly
        double ns_per_byte = stats.median / bytes;
        double bytes_per_second = 1e9 / ns_per_byte;
        double gb_per_second = bytes_per_second / 1e9;

        // Convert ns/byte to "equivalent cycles/byte" using estimated CPU freq
        double equiv_cycles_per_byte = ns_per_byte * cpu_freq_ghz;

        printf("  GB/s:    %.2f\n", gb_per_second);
        printf("  Equivalent Cycles/byte: %.2f (at %.2f GHz)\n", 
               equiv_cycles_per_byte, cpu_freq_ghz);
        #else
        // On x86, we have actual cycle counts
        double cycles_per_byte = stats.median / bytes;
        printf("  GB/s:    %.2f\n", (cpu_freq_ghz * 1e9) / (cycles_per_byte * 1e9));
        printf("  Cycles/byte: %.2f\n", cycles_per_byte);
        #endif
        
        free(measurements);
    }
    
    // Verify output is not all zeros (simple check)
    int nonzero_count = 0;
    for (int i = 0; i < 256; i++) {
        if (output[i] != 0) {
            nonzero_count++;
        }
    }
    
    printf("\nOutput verification: %d non-zero bytes (should be close to 256 for good diffusion)\n", 
           nonzero_count);
    
    return 0;
}