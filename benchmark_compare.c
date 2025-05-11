#define _GNU_SOURCE  /* Define this first, before any includes */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <math.h>
#include "src/faester_avx512.h"
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
    return __rdtsc();
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

// Benchmarking function for FAESTER
static stats_t benchmark_faester(int rounds, double *throughput) {
    uint8_t input[256] = {0};
    uint8_t output[256] = {0};
    faester_state_t state;
    
    // Initialize input with some data
    for (int i = 0; i < 256; i++) {
        input[i] = (uint8_t)i;
    }
    
    // Warmup phase
    faester_init(&state, input);
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        faester_permute(&state, rounds);
    }
    
    // Prepare for measurement
    uint64_t *measurements = malloc(BENCHMARK_ITERATIONS * sizeof(uint64_t));
    
    // Actual benchmark
    faester_init(&state, input);
    
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
        memory_barrier();
        uint64_t start = rdtsc();
        memory_barrier();
        
        faester_permute(&state, rounds);
        
        memory_barrier();
        uint64_t end = rdtsc();
        memory_barrier();
        
        measurements[i] = end - start;
    }
    
    // Extract output for verification
    faester_extract(&state, output);
    
    // Calculate statistics
    stats_t stats = calculate_stats(measurements, BENCHMARK_ITERATIONS);
    
    // Calculate throughput
    double bytes = 256.0; // 2048 bits = 256 bytes
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
    return stats;
}

// Benchmarking function for AREION
static stats_t benchmark_areion(double *throughput) {
    uint8_t input[64] = {0};
    uint8_t output[64] = {0};
    areion512_state_t state;
    
    // Initialize input with some data
    for (int i = 0; i < 64; i++) {
        input[i] = (uint8_t)i;
    }
    
    // Warmup phase
    areion512_init(&state, input);
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        areion512_permute(&state);
    }
    
    // Prepare for measurement
    uint64_t *measurements = malloc(BENCHMARK_ITERATIONS * sizeof(uint64_t));
    
    // Actual benchmark
    areion512_init(&state, input);
    
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
        memory_barrier();
        uint64_t start = rdtsc();
        memory_barrier();
        
        areion512_permute(&state);
        
        memory_barrier();
        uint64_t end = rdtsc();
        memory_barrier();
        
        measurements[i] = end - start;
    }
    
    // Extract output for verification
    areion512_extract(&state, output);
    
    // Calculate statistics
    stats_t stats = calculate_stats(measurements, BENCHMARK_ITERATIONS);
    
    // Calculate throughput
    double bytes = 64.0; // 512 bits = 64 bytes
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
    return stats;
}

// Print side-by-side benchmark comparison
static void print_benchmark_comparison(stats_t faester_stats, double faester_throughput,
                                      stats_t areion_stats, double areion_throughput,
                                      int faester_rounds) {
    double cpu_freq_ghz = estimate_cpu_freq_ghz();
    
    printf("\n");
    printf(ANSI_COLOR_CYAN "=============================================================\n" ANSI_COLOR_RESET);
    printf(ANSI_COLOR_CYAN "       FAESTER vs AREION PERFORMANCE COMPARISON       \n" ANSI_COLOR_RESET);
    printf(ANSI_COLOR_CYAN "=============================================================\n" ANSI_COLOR_RESET);
    printf("\n");
    
    // Print CPU info
    printf("CPU Frequency: %.2f GHz\n", cpu_freq_ghz);
    printf("FAESTER Rounds: %d, AREION Rounds: 15 (fixed)\n\n", faester_rounds);
    
    // Print permutation stats
    printf("┌────────────────────────┬─────────────────────┬─────────────────────┐\n");
    printf("│ Metric                 │ FAESTER (2048 bits) │ AREION (512 bits)   │\n");
    printf("├────────────────────────┼─────────────────────┼─────────────────────┤\n");
    
    #ifdef __APPLE__
    printf("│ Min time (ns)          │ %-17.2f │ %-17.2f │\n", 
           faester_stats.min, areion_stats.min);
    printf("│ Max time (ns)          │ %-17.2f │ %-17.2f │\n", 
           faester_stats.max, areion_stats.max);
    printf("│ Mean time (ns)         │ %-17.2f │ %-17.2f │\n", 
           faester_stats.mean, areion_stats.mean);
    printf("│ Median time (ns)       │ %-17.2f │ %-17.2f │\n", 
           faester_stats.median, areion_stats.median);
    printf("│ StdDev (ns)            │ %-17.2f │ %-17.2f │\n", 
           faester_stats.stddev, areion_stats.stddev);
    
    printf("│ Time per byte (ns)     │ %-17.2f │ %-17.2f │\n", 
           faester_stats.median / 256.0, areion_stats.median / 64.0);
    
    // On macOS, convert to equivalent cycles per byte
    double faester_equiv_cycles = (faester_stats.median / 256.0) * cpu_freq_ghz;
    double areion_equiv_cycles = (areion_stats.median / 64.0) * cpu_freq_ghz;
    
    printf("│ Equiv. cycles per byte │ %-17.2f │ %-17.2f │\n", 
           faester_equiv_cycles, areion_equiv_cycles);
    #else
    printf("│ Min cycles             │ %-17.2f │ %-17.2f │\n", 
           faester_stats.min, areion_stats.min);
    printf("│ Max cycles             │ %-17.2f │ %-17.2f │\n", 
           faester_stats.max, areion_stats.max);
    printf("│ Mean cycles            │ %-17.2f │ %-17.2f │\n", 
           faester_stats.mean, areion_stats.mean);
    printf("│ Median cycles          │ %-17.2f │ %-17.2f │\n", 
           faester_stats.median, areion_stats.median);
    printf("│ StdDev cycles          │ %-17.2f │ %-17.2f │\n", 
           faester_stats.stddev, areion_stats.stddev);
    
    printf("│ Cycles per byte        │ %-17.2f │ %-17.2f │\n", 
           faester_stats.median / 256.0, areion_stats.median / 64.0);
    #endif
    
    printf("│ Throughput (GB/s)      │ %-17.2f │ %-17.2f │\n", 
           faester_throughput, areion_throughput);
    
    // Calculate relative metrics
    double faester_per_bit = faester_stats.median / (256.0 * 8); // time per bit
    double areion_per_bit = areion_stats.median / (64.0 * 8); // time per bit
    double relative_efficiency = areion_per_bit / faester_per_bit;
    
    printf("│ Relative bit efficiency│ %-17.2f │ %-17s │\n", 
           relative_efficiency, "1.00 (reference)");
    
    printf("└────────────────────────┴─────────────────────┴─────────────────────┘\n\n");
    
    // Normalized comparison (per bit)
    printf("Normalized Comparison (per bit):\n");
    if (relative_efficiency > 1.0) {
        printf("FAESTER is %.2fx more efficient per bit than AREION\n", relative_efficiency);
    } else if (relative_efficiency < 1.0) {
        printf("AREION is %.2fx more efficient per bit than FAESTER\n", 1.0 / relative_efficiency);
    } else {
        printf("FAESTER and AREION have similar bit-level efficiency\n");
    }
    
    // State size vs. speed comparison
    printf("\nState size vs. performance tradeoff:\n");
    printf("- FAESTER: 2048-bit state (4x larger), ");
    if (faester_throughput > areion_throughput) {
        printf("%.2fx higher throughput\n", faester_throughput / areion_throughput);
    } else {
        printf("%.2fx lower throughput\n", areion_throughput / faester_throughput);
    }
    
    // Application recommendations
    printf("\nApplication recommendations:\n");
    if (relative_efficiency > 1.2) {
        printf("- For maximum security and efficiency: FAESTER\n");
        printf("- For constrained environments with limited memory: AREION\n");
    } else if (relative_efficiency < 0.8) {
        printf("- For maximum throughput: AREION\n");
        printf("- For applications requiring larger state size: FAESTER\n");
    } else {
        printf("- Both permutations offer similar efficiency with different state sizes\n");
        printf("- Choose based on your security requirements and available memory\n");
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
    printf(ANSI_COLOR_CYAN "=======================================================\n" ANSI_COLOR_RESET);
    printf(ANSI_COLOR_CYAN "       FAESTER vs AREION PERFORMANCE BENCHMARK         \n" ANSI_COLOR_RESET);
    printf(ANSI_COLOR_CYAN "=======================================================\n" ANSI_COLOR_RESET);
    printf("\n");
    
    // Estimate CPU frequency
    double cpu_freq_ghz = estimate_cpu_freq_ghz();
    printf("Estimated CPU frequency: %.2f GHz\n\n", cpu_freq_ghz);
    
    // Benchmark FAESTER
    printf("Benchmarking FAESTER (%d rounds)...\n", faester_rounds);
    double faester_throughput;
    stats_t faester_stats = benchmark_faester(faester_rounds, &faester_throughput);
    
    // Benchmark AREION
    printf("Benchmarking AREION (15 rounds)...\n");
    double areion_throughput;
    stats_t areion_stats = benchmark_areion(&areion_throughput);
    
    // Print comparison
    print_benchmark_comparison(faester_stats, faester_throughput,
                              areion_stats, areion_throughput,
                              faester_rounds);
    
    return 0;
}