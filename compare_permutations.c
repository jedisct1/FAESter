#define _GNU_SOURCE  /* Define this first, before any includes */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>
#include "src/faester_avx512.h"
#include "test/areion.h"

// ANSI color codes for prettier output
#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"

// Test parameters
#define FAESTER_BUFFER_SIZE 256  // Faester: 2048 bits = 256 bytes
#define AREION_BUFFER_SIZE 64    // Areion: 512 bits = 64 bytes
#define NUM_TESTS 1000          // Number of tests for each property
#define NUM_SAMPLES 1000        // Number of samples for statistical tests

// Enum for the permutation type
typedef enum {
    PERM_FAESTER,
    PERM_AREION
} permutation_type_t;

// Helper functions
static void fill_random(uint8_t *buffer, size_t size) {
    for (size_t i = 0; i < size; i++) {
        buffer[i] = rand() & 0xFF;
    }
}

static int hamming_weight(const uint8_t *data, size_t size) {
    int weight = 0;
    for (size_t i = 0; i < size; i++) {
        uint8_t byte = data[i];
        for (int j = 0; j < 8; j++) {
            weight += (byte >> j) & 1;
        }
    }
    return weight;
}

static int hamming_distance(const uint8_t *data1, const uint8_t *data2, size_t size) {
    int distance = 0;
    for (size_t i = 0; i < size; i++) {
        uint8_t xor_byte = data1[i] ^ data2[i];
        for (int j = 0; j < 8; j++) {
            distance += (xor_byte >> j) & 1;
        }
    }
    return distance;
}

static void print_hex(const uint8_t *data, size_t len, size_t max_bytes) {
    for (size_t i = 0; i < len && i < max_bytes; i++) {
        printf("%02x", data[i]);
        if ((i + 1) % 32 == 0) printf("\n");
        else if ((i + 1) % 8 == 0) printf(" ");
    }
    if (len > max_bytes) printf("...");
    printf("\n");
}

// Chi-square test for randomness
static double chi_square_test(const uint8_t *data, size_t size) {
    int frequencies[256] = {0};
    for (size_t i = 0; i < size; i++) {
        frequencies[data[i]]++;
    }
    
    double expected = (double)size / 256.0;
    double chi_square = 0.0;
    
    for (int i = 0; i < 256; i++) {
        double diff = frequencies[i] - expected;
        chi_square += (diff * diff) / expected;
    }
    
    return chi_square;
}

// Entropy calculation (Shannon entropy in bits)
static double calculate_entropy(const uint8_t *data, size_t size) {
    int frequencies[256] = {0};
    for (size_t i = 0; i < size; i++) {
        frequencies[data[i]]++;
    }
    
    double entropy = 0.0;
    for (int i = 0; i < 256; i++) {
        if (frequencies[i] > 0) {
            double p = (double)frequencies[i] / size;
            entropy -= p * log2(p);
        }
    }
    
    return entropy;
}

// Apply the permutation based on type
static void apply_permutation(permutation_type_t type, void *state, uint8_t *input, uint8_t *output, int rounds) {
    if (type == PERM_FAESTER) {
        faester_state_t *faester_state = (faester_state_t *)state;
        faester_init(faester_state, input);
        faester_permute(faester_state, rounds);
        faester_extract(faester_state, output);
    } else { // PERM_AREION
        areion512_state_t *areion_state = (areion512_state_t *)state;
        areion512_init(areion_state, input);
        areion512_permute(areion_state); // Areion always uses 15 rounds
        areion512_extract(areion_state, output);
    }
}

// Test avalanche effect
static double test_avalanche(permutation_type_t type, int rounds) {
    size_t buffer_size = (type == PERM_FAESTER) ? FAESTER_BUFFER_SIZE : AREION_BUFFER_SIZE;
    int total_bits = buffer_size * 8;
    
    uint8_t input[FAESTER_BUFFER_SIZE] = {0}; // Use larger buffer for both
    uint8_t modified[FAESTER_BUFFER_SIZE] = {0};
    uint8_t output_orig[FAESTER_BUFFER_SIZE] = {0};
    uint8_t output_mod[FAESTER_BUFFER_SIZE] = {0};
    
    // Allocate state based on permutation type
    void *state_orig, *state_mod;
    if (type == PERM_FAESTER) {
        state_orig = malloc(sizeof(faester_state_t));
        state_mod = malloc(sizeof(faester_state_t));
    } else {
        state_orig = malloc(sizeof(areion512_state_t));
        state_mod = malloc(sizeof(areion512_state_t));
    }
    
    printf("Testing avalanche effect with %s (%d rounds)...\n", 
           (type == PERM_FAESTER) ? "FAESTER" : "AREION",
           (type == PERM_FAESTER) ? rounds : 15);
    
    // Statistics
    double total_avg_flips = 0.0;
    int total_tests = 0;
    
    // For each bit in the input
    for (int bit_pos = 0; bit_pos < total_bits; bit_pos++) {
        int total_flips = 0;
        
        for (int test = 0; test < NUM_TESTS; test++) {
            // Generate random input
            fill_random(input, buffer_size);
            
            // Create modified input with one bit flipped
            memcpy(modified, input, buffer_size);
            int byte_pos = bit_pos / 8;
            int bit_in_byte = bit_pos % 8;
            modified[byte_pos] ^= (1 << bit_in_byte);
            
            // Process both inputs
            apply_permutation(type, state_orig, input, output_orig, rounds);
            apply_permutation(type, state_mod, modified, output_mod, rounds);
            
            // Calculate Hamming distance
            int distance = hamming_distance(output_orig, output_mod, buffer_size);
            total_flips += distance;
        }
        
        // Update statistics
        double avg_flips = (double)total_flips / NUM_TESTS;
        total_avg_flips += avg_flips;
        total_tests++;
        
        // Progress indicator every 10% of bits
        if (bit_pos % (total_bits / 10) == 0 || bit_pos == total_bits - 1) {
            printf("  Processed %d/%d bits (%.1f%%)\r", bit_pos + 1, total_bits, 
                   (bit_pos + 1) * 100.0 / total_bits);
            fflush(stdout);
        }
    }
    printf("\n");
    
    // Calculate overall statistics
    double average_flips = total_avg_flips / total_tests;
    double ideal_flips = total_bits * 0.5; // Ideal: half the bits should flip
    double avalanche_quality = average_flips / ideal_flips;
    
    printf("Avalanche Effect Results:\n");
    printf("  Average bit flips: %.2f out of %d bits (%.2f%%)\n", 
           average_flips, total_bits, (average_flips * 100.0) / total_bits);
    printf("  Ideal bit flips: %.2f (50%%)\n", ideal_flips);
    printf("  Avalanche quality: %.4f (1.0 is perfect)\n\n", avalanche_quality);
    
    // Clean up
    free(state_orig);
    free(state_mod);
    
    return avalanche_quality;
}

// Test diffusion completeness
static int test_diffusion(permutation_type_t type, int rounds) {
    size_t buffer_size = (type == PERM_FAESTER) ? FAESTER_BUFFER_SIZE : AREION_BUFFER_SIZE;
    int total_bits = buffer_size * 8;
    
    uint8_t input[FAESTER_BUFFER_SIZE] = {0};
    uint8_t modified[FAESTER_BUFFER_SIZE] = {0};
    uint8_t output_orig[FAESTER_BUFFER_SIZE] = {0};
    uint8_t output_mod[FAESTER_BUFFER_SIZE] = {0};
    
    // Allocate state based on permutation type
    void *state_orig, *state_mod;
    if (type == PERM_FAESTER) {
        state_orig = malloc(sizeof(faester_state_t));
        state_mod = malloc(sizeof(faester_state_t));
    } else {
        state_orig = malloc(sizeof(areion512_state_t));
        state_mod = malloc(sizeof(areion512_state_t));
    }
    
    printf("Testing diffusion completeness with %s (%d rounds):\n", 
           (type == PERM_FAESTER) ? "FAESTER" : "AREION",
           (type == PERM_FAESTER) ? rounds : 15);
    
    // Create a bit dependency matrix (input_bit × output_bit)
    int num_samples = 10; // Multiple samples per input bit 
    double **dependency_matrix = (double **)malloc(total_bits * sizeof(double *));
    for (int i = 0; i < total_bits; i++) {
        dependency_matrix[i] = (double *)calloc(total_bits, sizeof(double));
    }
    
    // For each input bit
    for (int input_bit = 0; input_bit < total_bits; input_bit++) {
        for (int sample = 0; sample < num_samples; sample++) {
            // Generate random input
            fill_random(input, buffer_size);
            
            // Create modified input with one bit flipped
            memcpy(modified, input, buffer_size);
            int byte_pos = input_bit / 8;
            int bit_pos = input_bit % 8;
            modified[byte_pos] ^= (1 << bit_pos);
            
            // Process both inputs
            apply_permutation(type, state_orig, input, output_orig, rounds);
            apply_permutation(type, state_mod, modified, output_mod, rounds);
            
            // Calculate which output bits were affected
            for (int output_bit = 0; output_bit < total_bits; output_bit++) {
                int out_byte_pos = output_bit / 8;
                int out_bit_pos = output_bit % 8;
                
                bool orig_bit = (output_orig[out_byte_pos] >> out_bit_pos) & 1;
                bool mod_bit = (output_mod[out_byte_pos] >> out_bit_pos) & 1;
                
                if (orig_bit != mod_bit) {
                    dependency_matrix[input_bit][output_bit] += 1.0 / num_samples;
                }
            }
        }
        
        if (input_bit % (total_bits / 10) == 0 || input_bit == total_bits - 1) {
            printf("  Processed %d/%d input bits (%.1f%%)\r", input_bit + 1, total_bits,
                   (input_bit + 1) * 100.0 / total_bits);
            fflush(stdout);
        }
    }
    printf("\n");
    
    // Analyze the diffusion completeness using appropriate thresholds
    // For cryptographic permutations, we expect each bit to change with ~50% probability
    // So we'll count a significant influence as >20% probability of flipping
    double significant_influence_threshold = 0.2;
    
    // For each input bit, count output bits it significantly influences
    int *significant_output_bits = calloc(total_bits, sizeof(int));
    for (int i = 0; i < total_bits; i++) {
        for (int j = 0; j < total_bits; j++) {
            if (dependency_matrix[i][j] >= significant_influence_threshold) {
                significant_output_bits[i]++;
            }
        }
    }
    
    // For each output bit, count input bits that significantly influence it
    int *significant_input_bits = calloc(total_bits, sizeof(int));
    for (int j = 0; j < total_bits; j++) {
        for (int i = 0; i < total_bits; i++) {
            if (dependency_matrix[i][j] >= significant_influence_threshold) {
                significant_input_bits[j]++;
            }
        }
    }
    
    // Calculate statistics
    int min_outputs_influenced = total_bits;
    int max_outputs_influenced = 0;
    double avg_outputs_influenced = 0.0;
    
    int min_inputs_influencing = total_bits;
    int max_inputs_influencing = 0;
    double avg_inputs_influencing = 0.0;
    
    for (int i = 0; i < total_bits; i++) {
        // Update statistics for outputs influenced
        if (significant_output_bits[i] < min_outputs_influenced) {
            min_outputs_influenced = significant_output_bits[i];
        }
        if (significant_output_bits[i] > max_outputs_influenced) {
            max_outputs_influenced = significant_output_bits[i];
        }
        avg_outputs_influenced += significant_output_bits[i];
        
        // Update statistics for inputs influencing
        if (significant_input_bits[i] < min_inputs_influencing) {
            min_inputs_influencing = significant_input_bits[i];
        }
        if (significant_input_bits[i] > max_inputs_influencing) {
            max_inputs_influencing = significant_input_bits[i];
        }
        avg_inputs_influencing += significant_input_bits[i];
    }
    
    avg_outputs_influenced /= total_bits;
    avg_inputs_influencing /= total_bits;
    
    // Calculate percentage of total bits
    double min_outputs_influenced_pct = 100.0 * (double)min_outputs_influenced / total_bits;
    double max_outputs_influenced_pct = 100.0 * (double)max_outputs_influenced / total_bits;
    double avg_outputs_influenced_pct = 100.0 * avg_outputs_influenced / total_bits;
    
    double min_inputs_influencing_pct = 100.0 * (double)min_inputs_influencing / total_bits;
    double max_inputs_influencing_pct = 100.0 * (double)max_inputs_influencing / total_bits;
    double avg_inputs_influencing_pct = 100.0 * avg_inputs_influencing / total_bits;
    
    // Print results
    printf("  Diffusion results with significance threshold: %.2f\n", significant_influence_threshold);
    
    printf("  Output bits significantly influenced by each input bit:\n");
    printf("    Min: %d (%.2f%%)\n", min_outputs_influenced, min_outputs_influenced_pct);
    printf("    Max: %d (%.2f%%)\n", max_outputs_influenced, max_outputs_influenced_pct);
    printf("    Avg: %.2f (%.2f%%)\n", avg_outputs_influenced, avg_outputs_influenced_pct);
    
    printf("  Input bits significantly influencing each output bit:\n");
    printf("    Min: %d (%.2f%%)\n", min_inputs_influencing, min_inputs_influencing_pct);
    printf("    Max: %d (%.2f%%)\n", max_inputs_influencing, max_inputs_influencing_pct);
    printf("    Avg: %.2f (%.2f%%)\n", avg_inputs_influencing, avg_inputs_influencing_pct);
    
    // Calculate avalanche uniformity
    double avalanche_score = 0.0;
    int total_pairs = total_bits * total_bits;
    double expected_prob = 0.5;
    double variance_sum = 0.0;
    
    for (int i = 0; i < total_bits; i++) {
        for (int j = 0; j < total_bits; j++) {
            double diff = dependency_matrix[i][j] - expected_prob;
            variance_sum += diff * diff;
        }
    }
    
    double mean_variance = variance_sum / total_pairs;
    double std_deviation = sqrt(mean_variance);
    
    printf("  Avalanche uniformity statistics:\n");
    printf("    Standard deviation from ideal 50%%: %.6f\n", std_deviation);
    
    // Calculate actual diffusion score
    double influence_score = avg_outputs_influenced_pct;
    double influenced_score = avg_inputs_influencing_pct;
    double uniformity_score = 100.0 * (1.0 - std_deviation * 2.0); // Lower std_dev is better
    
    if (uniformity_score < 0.0) uniformity_score = 0.0;
    
    // Combine scores with appropriate weights
    double diffusion_score = (influence_score * 0.4 + influenced_score * 0.4 + uniformity_score * 0.2);
    
    printf("  Diffusion component scores:\n");
    printf("    Influence coverage: %.2f/100\n", influence_score);
    printf("    Dependency coverage: %.2f/100\n", influenced_score);
    printf("    Avalanche uniformity: %.2f/100\n", uniformity_score);
    printf("    Overall diffusion score: %.2f/100\n\n", diffusion_score);
    
    // Clean up
    for (int i = 0; i < total_bits; i++) {
        free(dependency_matrix[i]);
    }
    free(dependency_matrix);
    free(significant_output_bits);
    free(significant_input_bits);
    free(state_orig);
    free(state_mod);
    
    return (int)(diffusion_score + 0.5);
}

// Test statistical properties of permutation output
static double test_statistical_properties(permutation_type_t type, int rounds) {
    size_t buffer_size = (type == PERM_FAESTER) ? FAESTER_BUFFER_SIZE : AREION_BUFFER_SIZE;
    
    uint8_t input[FAESTER_BUFFER_SIZE] = {0};
    uint8_t output[FAESTER_BUFFER_SIZE] = {0};
    
    // Allocate state based on permutation type
    void *state;
    if (type == PERM_FAESTER) {
        state = malloc(sizeof(faester_state_t));
    } else {
        state = malloc(sizeof(areion512_state_t));
    }
    
    printf("Testing statistical properties with %s (%d rounds)...\n", 
           (type == PERM_FAESTER) ? "FAESTER" : "AREION",
           (type == PERM_FAESTER) ? rounds : 15);
    
    // Statistical accumulators
    double sum_entropy = 0.0;
    double sum_chi_square = 0.0;
    
    for (int sample = 0; sample < NUM_SAMPLES; sample++) {
        // Generate random input
        fill_random(input, buffer_size);
        
        // Apply permutation
        apply_permutation(type, state, input, output, rounds);
        
        // Calculate statistical properties
        double entropy = calculate_entropy(output, buffer_size);
        double chi_square = chi_square_test(output, buffer_size);
        
        // Accumulate statistics
        sum_entropy += entropy;
        sum_chi_square += chi_square;
        
        if ((sample + 1) % (NUM_SAMPLES / 10) == 0 || sample == NUM_SAMPLES - 1) {
            printf("  Processed %d/%d samples (%.1f%%)\r", sample + 1, NUM_SAMPLES,
                   (sample + 1) * 100.0 / NUM_SAMPLES);
            fflush(stdout);
        }
    }
    printf("\n");
    
    // Calculate averages
    double avg_entropy = sum_entropy / NUM_SAMPLES;
    double avg_chi_square = sum_chi_square / NUM_SAMPLES;
    
    // Ideal values
    double ideal_entropy = 8.0; // 8 bits per byte is maximum entropy
    double ideal_chi_square = 255.0; // For 256 possible byte values
    
    // Calculate percentage of ideal
    double entropy_pct = (avg_entropy / ideal_entropy) * 100.0;
    double chi_square_pct = 100.0 - (avg_chi_square / (4 * ideal_chi_square)) * 100.0;
    
    // Calculate overall statistical score (0-100)
    double stat_score = (entropy_pct + chi_square_pct) / 2.0;
    
    printf("  Statistical test results:\n");
    printf("  - Average entropy: %.6f bits/byte (%.2f%% of ideal)\n", 
           avg_entropy, entropy_pct);
    printf("  - Average chi-square: %.6f (%.2f%% goodness)\n", 
           avg_chi_square, chi_square_pct);
    printf("  - Overall statistical score: %.2f/100\n\n", stat_score);
    
    free(state);
    return stat_score;
}

// Simple test of basic properties (zero input, single bit, etc.)
static void demonstrate_basic_properties(permutation_type_t type, int rounds) {
    size_t buffer_size = (type == PERM_FAESTER) ? FAESTER_BUFFER_SIZE : AREION_BUFFER_SIZE;
    
    uint8_t input[FAESTER_BUFFER_SIZE] = {0};
    uint8_t output[FAESTER_BUFFER_SIZE] = {0};
    
    // Allocate state based on permutation type
    void *state;
    if (type == PERM_FAESTER) {
        state = malloc(sizeof(faester_state_t));
    } else {
        state = malloc(sizeof(areion512_state_t));
    }
    
    printf("\n");
    printf(ANSI_COLOR_CYAN "============================================\n" ANSI_COLOR_RESET);
    printf(ANSI_COLOR_CYAN "     BASIC PROPERTY DEMONSTRATION: %s     \n" ANSI_COLOR_RESET, 
           (type == PERM_FAESTER) ? "FAESTER" : "AREION");
    printf(ANSI_COLOR_CYAN "============================================\n" ANSI_COLOR_RESET);
    printf("\n");
    
    // Test 1: Zero input
    memset(input, 0, buffer_size);
    printf("Test 1: Zero input\n");
    printf("Input:  ");
    print_hex(input, buffer_size, 32);
    
    apply_permutation(type, state, input, output, rounds);
    
    printf("Output: ");
    print_hex(output, buffer_size, 32);
    
    int nonzero_bytes = 0;
    for (int i = 0; i < buffer_size; i++) {
        if (output[i] != 0) nonzero_bytes++;
    }
    printf("Non-zero bytes: %d/%d (%.2f%%)\n\n", 
           nonzero_bytes, buffer_size, (double)nonzero_bytes/buffer_size*100.0);
    
    // Test 2: Single bit set (first bit)
    memset(input, 0, buffer_size);
    input[0] = 1; // Set first bit
    printf("Test 2: Single bit set (first bit)\n");
    printf("Input:  ");
    print_hex(input, buffer_size, 32);
    
    apply_permutation(type, state, input, output, rounds);
    
    printf("Output: ");
    print_hex(output, buffer_size, 32);
    
    nonzero_bytes = 0;
    for (int i = 0; i < buffer_size; i++) {
        if (output[i] != 0) nonzero_bytes++;
    }
    printf("Non-zero bytes: %d/%d (%.2f%%)\n\n", 
           nonzero_bytes, buffer_size, (double)nonzero_bytes/buffer_size*100.0);
    
    // Test 3: Single bit set (middle)
    memset(input, 0, buffer_size);
    input[buffer_size/2] = 128; // Set middle bit
    printf("Test 3: Single bit set (middle bit)\n");
    printf("Input:  ");
    print_hex(input, buffer_size, 32);
    
    apply_permutation(type, state, input, output, rounds);
    
    printf("Output: ");
    print_hex(output, buffer_size, 32);
    
    nonzero_bytes = 0;
    for (int i = 0; i < buffer_size; i++) {
        if (output[i] != 0) nonzero_bytes++;
    }
    printf("Non-zero bytes: %d/%d (%.2f%%)\n\n", 
           nonzero_bytes, buffer_size, (double)nonzero_bytes/buffer_size*100.0);
    
    // Test 4: Random input with entropy measurement
    fill_random(input, buffer_size);
    printf("Test 4: Random input\n");
    printf("Input:  ");
    print_hex(input, buffer_size, 32);
    printf("Input entropy: %.6f bits/byte\n", calculate_entropy(input, buffer_size));
    
    apply_permutation(type, state, input, output, rounds);
    
    printf("Output: ");
    print_hex(output, buffer_size, 32);
    printf("Output entropy: %.6f bits/byte\n\n", calculate_entropy(output, buffer_size));
    
    free(state);
}

// Print a summary assessment of the cryptographic properties
static void print_summary(permutation_type_t type, double avalanche_score, 
                          int diffusion_score, double statistical_score) {
    printf("\n");
    printf(ANSI_COLOR_CYAN "============================================\n" ANSI_COLOR_RESET);
    printf(ANSI_COLOR_CYAN "       %s PROPERTY SUMMARY       \n" ANSI_COLOR_RESET,
           (type == PERM_FAESTER) ? "FAESTER" : "AREION");
    printf(ANSI_COLOR_CYAN "============================================\n" ANSI_COLOR_RESET);
    printf("\n");
    
    // Print Avalanche score
    printf("Avalanche Effect: ");
    if (avalanche_score >= 0.99) {
        printf(ANSI_COLOR_GREEN "Excellent (%.4f)\n" ANSI_COLOR_RESET, avalanche_score);
    } else if (avalanche_score >= 0.95) {
        printf(ANSI_COLOR_GREEN "Good (%.4f)\n" ANSI_COLOR_RESET, avalanche_score);
    } else if (avalanche_score >= 0.90) {
        printf(ANSI_COLOR_YELLOW "Adequate (%.4f)\n" ANSI_COLOR_RESET, avalanche_score);
    } else {
        printf(ANSI_COLOR_RED "Poor (%.4f)\n" ANSI_COLOR_RESET, avalanche_score);
    }
    
    // Print diffusion score
    printf("Diffusion Completeness: ");
    if (diffusion_score >= 90) {
        printf(ANSI_COLOR_GREEN "Excellent (%d/100)\n" ANSI_COLOR_RESET, diffusion_score);
    } else if (diffusion_score >= 80) {
        printf(ANSI_COLOR_GREEN "Good (%d/100)\n" ANSI_COLOR_RESET, diffusion_score);
    } else if (diffusion_score >= 70) {
        printf(ANSI_COLOR_YELLOW "Adequate (%d/100)\n" ANSI_COLOR_RESET, diffusion_score);
    } else {
        printf(ANSI_COLOR_RED "Poor (%d/100)\n" ANSI_COLOR_RESET, diffusion_score);
    }
    
    // Print statistical score
    printf("Statistical Properties: ");
    if (statistical_score >= 95) {
        printf(ANSI_COLOR_GREEN "Excellent (%.1f/100)\n" ANSI_COLOR_RESET, statistical_score);
    } else if (statistical_score >= 90) {
        printf(ANSI_COLOR_GREEN "Good (%.1f/100)\n" ANSI_COLOR_RESET, statistical_score);
    } else if (statistical_score >= 85) {
        printf(ANSI_COLOR_YELLOW "Adequate (%.1f/100)\n" ANSI_COLOR_RESET, statistical_score);
    } else {
        printf(ANSI_COLOR_RED "Poor (%.1f/100)\n" ANSI_COLOR_RESET, statistical_score);
    }
    
    // Print overall assessment
    double overall_score = (avalanche_score * 100 + diffusion_score + statistical_score) / 3;
    printf("\nOverall Cryptographic Quality: ");
    
    if (overall_score >= 90) {
        printf(ANSI_COLOR_GREEN "Excellent (%.1f/100)\n" ANSI_COLOR_RESET, overall_score);
    } else if (overall_score >= 80) {
        printf(ANSI_COLOR_GREEN "Good (%.1f/100)\n" ANSI_COLOR_RESET, overall_score);
    } else if (overall_score >= 70) {
        printf(ANSI_COLOR_YELLOW "Adequate (%.1f/100)\n" ANSI_COLOR_RESET, overall_score);
    } else {
        printf(ANSI_COLOR_RED "Poor (%.1f/100)\n" ANSI_COLOR_RESET, overall_score);
    }
    printf("\n");
}

// Print comparison table between Faester and Areion
static void print_comparison(double faester_avalanche, int faester_diffusion, double faester_statistical,
                            double areion_avalanche, int areion_diffusion, double areion_statistical) {
    printf("\n");
    printf(ANSI_COLOR_CYAN "============================================================\n" ANSI_COLOR_RESET);
    printf(ANSI_COLOR_CYAN "       FAESTER vs AREION CRYPTOGRAPHIC PROPERTY COMPARISON       \n" ANSI_COLOR_RESET);
    printf(ANSI_COLOR_CYAN "============================================================\n" ANSI_COLOR_RESET);
    printf("\n");
    
    printf("┌────────────────────────┬────────────────┬────────────────┬────────────┐\n");
    printf("│ Property               │ FAESTER        │ AREION         │ Difference │\n");
    printf("├────────────────────────┼────────────────┼────────────────┼────────────┤\n");
    
    // Avalanche Effect
    double avalanche_diff = fabs(faester_avalanche - areion_avalanche);
    const char *avalanche_better = (faester_avalanche > areion_avalanche) ? "FAESTER" : "AREION";
    if (avalanche_diff < 0.01) avalanche_better = "EQUAL";
    
    printf("│ Avalanche Effect       │ %-12.4f  │ %-12.4f  │ %-8.4f  │\n", 
           faester_avalanche, areion_avalanche, avalanche_diff);
    
    // Diffusion Completeness
    int diffusion_diff = abs(faester_diffusion - areion_diffusion);
    const char *diffusion_better = (faester_diffusion > areion_diffusion) ? "FAESTER" : "AREION";
    if (diffusion_diff < 3) diffusion_better = "EQUAL";
    
    printf("│ Diffusion Completeness │ %-12d  │ %-12d  │ %-8d  │\n", 
           faester_diffusion, areion_diffusion, diffusion_diff);
    
    // Statistical Properties
    double statistical_diff = fabs(faester_statistical - areion_statistical);
    const char *statistical_better = (faester_statistical > areion_statistical) ? "FAESTER" : "AREION";
    if (statistical_diff < 1.0) statistical_better = "EQUAL";
    
    printf("│ Statistical Properties │ %-12.1f  │ %-12.1f  │ %-8.1f  │\n", 
           faester_statistical, areion_statistical, statistical_diff);
    
    // Overall Score
    double faester_overall = (faester_avalanche * 100 + faester_diffusion + faester_statistical) / 3;
    double areion_overall = (areion_avalanche * 100 + areion_diffusion + areion_statistical) / 3;
    double overall_diff = fabs(faester_overall - areion_overall);
    const char *overall_better = (faester_overall > areion_overall) ? "FAESTER" : "AREION";
    if (overall_diff < 1.0) overall_better = "EQUAL";
    
    printf("│ Overall Score          │ %-12.1f  │ %-12.1f  │ %-8.1f  │\n", 
           faester_overall, areion_overall, overall_diff);
    
    printf("└────────────────────────┴────────────────┴────────────────┴────────────┘\n\n");
    
    printf("Assessment:\n");
    printf(" - Avalanche Effect: %s %s\n", 
           avalanche_better, (avalanche_better == "EQUAL") ? "performs equally" : "performs better");
    printf(" - Diffusion Completeness: %s %s\n", 
           diffusion_better, (diffusion_better == "EQUAL") ? "performs equally" : "performs better");
    printf(" - Statistical Properties: %s %s\n", 
           statistical_better, (statistical_better == "EQUAL") ? "performs equally" : "performs better");
    printf(" - Overall: %s %s\n\n", 
           overall_better, (overall_better == "EQUAL") ? "performs equally" : "performs better");
    
    // Additional important comparison note about state size
    printf("NOTE: FAESTER operates on a 2048-bit state (256 bytes), while AREION\n");
    printf("operates on a 512-bit state (64 bytes). This means FAESTER has a\n");
    printf("state that is 4 times larger than AREION.\n\n");
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
    
    // AREION is fixed at 15 rounds
    int areion_rounds = 15;
    
    // Initialize random seed
    srand(time(NULL));
    
    // Test header
    printf(ANSI_COLOR_CYAN "=======================================================\n" ANSI_COLOR_RESET);
    printf(ANSI_COLOR_CYAN "       FAESTER vs AREION CRYPTOGRAPHIC COMPARISON      \n" ANSI_COLOR_RESET);
    printf(ANSI_COLOR_CYAN "=======================================================\n" ANSI_COLOR_RESET);
    printf("\n");
    printf("Testing permutations:\n");
    printf("- FAESTER with %d rounds (state size: %d bytes / %d bits)\n", 
           faester_rounds, FAESTER_BUFFER_SIZE, FAESTER_BUFFER_SIZE * 8);
    printf("- AREION with %d rounds (state size: %d bytes / %d bits)\n", 
           areion_rounds, AREION_BUFFER_SIZE, AREION_BUFFER_SIZE * 8);
    printf("Number of tests: %d\n", NUM_TESTS);
    printf("\n");
    
    // Test basic properties
    printf("====================== BASIC PROPERTIES ======================\n\n");
    demonstrate_basic_properties(PERM_FAESTER, faester_rounds);
    demonstrate_basic_properties(PERM_AREION, areion_rounds);
    
    // Track scores for summary
    double faester_avalanche, areion_avalanche;
    int faester_diffusion, areion_diffusion;
    double faester_statistical, areion_statistical;
    
    // Test avalanche effect
    printf("====================== AVALANCHE EFFECT ======================\n\n");
    faester_avalanche = test_avalanche(PERM_FAESTER, faester_rounds);
    areion_avalanche = test_avalanche(PERM_AREION, areion_rounds);
    
    // Test diffusion completeness
    printf("====================== DIFFUSION COMPLETENESS ======================\n\n");
    faester_diffusion = test_diffusion(PERM_FAESTER, faester_rounds);
    areion_diffusion = test_diffusion(PERM_AREION, areion_rounds);
    
    // Test statistical properties
    printf("====================== STATISTICAL PROPERTIES ======================\n\n");
    faester_statistical = test_statistical_properties(PERM_FAESTER, faester_rounds);
    areion_statistical = test_statistical_properties(PERM_AREION, areion_rounds);
    
    // Print individual summaries
    print_summary(PERM_FAESTER, faester_avalanche, faester_diffusion, faester_statistical);
    print_summary(PERM_AREION, areion_avalanche, areion_diffusion, areion_statistical);
    
    // Print side-by-side comparison
    print_comparison(faester_avalanche, faester_diffusion, faester_statistical,
                    areion_avalanche, areion_diffusion, areion_statistical);
    
    return 0;
}