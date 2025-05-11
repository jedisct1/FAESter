#define _GNU_SOURCE  /* Define this first, before any includes */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>
#include "src/faester_avx512.h"

// Test parameters
#define BUFFER_SIZE 256  // Size in bytes (2048 bits)
#define NUM_TESTS 1000   // Number of tests for each property
#define NUM_SAMPLES 1000 // Number of samples for statistical tests

// ANSI color codes for prettier output
#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"

// Helper functions
static void fill_random(uint8_t *buffer, size_t size) {
    for (size_t i = 0; i < size; i++) {
        buffer[i] = rand() & 0xFF;
    }
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

// Chi-square test for randomness
static double chi_square_test(const uint8_t *data, size_t size) {
    // Count byte frequencies
    int frequencies[256] = {0};
    for (size_t i = 0; i < size; i++) {
        frequencies[data[i]]++;
    }
    
    // Calculate chi-square statistic
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
    // Count byte frequencies
    int frequencies[256] = {0};
    for (size_t i = 0; i < size; i++) {
        frequencies[data[i]]++;
    }
    
    // Calculate entropy
    double entropy = 0.0;
    for (int i = 0; i < 256; i++) {
        if (frequencies[i] > 0) {
            double p = (double)frequencies[i] / size;
            entropy -= p * log2(p);
        }
    }
    
    return entropy;
}

// Calculate the serial correlation coefficient
static double serial_correlation(const uint8_t *data, size_t size) {
    if (size <= 1) return 0.0;
    
    double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0;
    double sum_x2 = 0.0, sum_y2 = 0.0;
    
    for (size_t i = 0; i < size - 1; i++) {
        double x = data[i];
        double y = data[i + 1];
        
        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_x2 += x * x;
        sum_y2 += y * y;
    }
    
    double n = size - 1;
    double numerator = n * sum_xy - sum_x * sum_y;
    double denominator = sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y));
    
    if (denominator == 0.0) return 0.0;
    return numerator / denominator;
}

// Monte Carlo estimation of Pi to test randomness
// If the output is random, the estimate of Pi should converge to the true value
static double monte_carlo_pi(const uint8_t *data, size_t size) {
    // Need at least 4 bytes per sample
    int samples = size / 4;
    int inside_circle = 0;
    
    for (int i = 0; i < samples; i++) {
        // Use 2 bytes to create x coordinate (0.0 to 1.0)
        double x = ((data[i*4] << 8) | data[i*4 + 1]) / 65535.0;
        
        // Use 2 bytes to create y coordinate (0.0 to 1.0)
        double y = ((data[i*4 + 2] << 8) | data[i*4 + 3]) / 65535.0;
        
        // Check if point is inside unit circle
        if (x*x + y*y <= 1.0) {
            inside_circle++;
        }
    }
    
    // Pi ≈ 4 * (points inside circle) / (total points)
    return 4.0 * inside_circle / samples;
}

// Tests for bit bias (should be close to 0.5 for each bit position)
static void bit_bias_test(const uint8_t *data, size_t size, double *max_bias, double *avg_bias) {
    int total_bits = size * 8;
    int *bit_counts = calloc(total_bits, sizeof(int));
    
    if (!bit_counts) {
        fprintf(stderr, "Memory allocation failed\n");
        *max_bias = *avg_bias = 1.0; // To indicate failure
        return;
    }
    
    // Count set bits at each position
    for (size_t i = 0; i < size; i++) {
        uint8_t byte = data[i];
        for (int j = 0; j < 8; j++) {
            if ((byte >> j) & 1) {
                bit_counts[i*8 + j]++;
            }
        }
    }
    
    // Calculate bias statistics
    *max_bias = 0.0;
    double sum_bias = 0.0;
    
    for (int i = 0; i < total_bits; i++) {
        double bias = fabs(((double)bit_counts[i] / NUM_SAMPLES) - 0.5);
        sum_bias += bias;
        if (bias > *max_bias) {
            *max_bias = bias;
        }
    }
    
    *avg_bias = sum_bias / total_bits;
    free(bit_counts);
}

// Test diffusion completeness (how many rounds until all output bits depend on all input bits)
static int test_diffusion_completeness(int rounds) {
    uint8_t input[BUFFER_SIZE] = {0};
    uint8_t modified[BUFFER_SIZE];
    uint8_t output_orig[BUFFER_SIZE];
    uint8_t output_mod[BUFFER_SIZE];
    
    faester_state_t state_orig, state_mod;
    
    printf("Testing diffusion completeness with %d rounds:\n", rounds);
    
    // Create a bit dependency matrix (input_bit × output_bit)
    int total_bits = BUFFER_SIZE * 8;
    int *dependency_matrix = calloc(total_bits * total_bits, sizeof(int));
    
    if (!dependency_matrix) {
        fprintf(stderr, "Memory allocation failed\n");
        return 0;
    }
    
    // For each input bit
    for (int input_bit = 0; input_bit < total_bits; input_bit++) {
        // Generate random input
        fill_random(input, BUFFER_SIZE);
        
        // Create modified input with one bit flipped
        memcpy(modified, input, BUFFER_SIZE);
        int byte_pos = input_bit / 8;
        int bit_pos = input_bit % 8;
        modified[byte_pos] ^= (1 << bit_pos);
        
        // Run permutation on both inputs
        faester_init(&state_orig, input);
        faester_init(&state_mod, modified);
        
        faester_permute(&state_orig, rounds);
        faester_permute(&state_mod, rounds);
        
        faester_extract(&state_orig, output_orig);
        faester_extract(&state_mod, output_mod);
        
        // Calculate which output bits were affected
        for (int output_bit = 0; output_bit < total_bits; output_bit++) {
            int out_byte_pos = output_bit / 8;
            int out_bit_pos = output_bit % 8;
            
            bool orig_bit = (output_orig[out_byte_pos] >> out_bit_pos) & 1;
            bool mod_bit = (output_mod[out_byte_pos] >> out_bit_pos) & 1;
            
            if (orig_bit != mod_bit) {
                dependency_matrix[input_bit * total_bits + output_bit] = 1;
            }
        }
        
        if (input_bit % 100 == 0 || input_bit == total_bits - 1) {
            printf("  Processed %d/%d input bits\r", input_bit + 1, total_bits);
            fflush(stdout);
        }
    }
    printf("\n");
    
    // Analyze the diffusion completeness
    int min_dependencies = total_bits;
    int max_dependencies = 0;
    double avg_dependencies = 0.0;
    int fully_diffused_input_bits = 0;
    
    // For each input bit, count how many output bits depend on it
    for (int input_bit = 0; input_bit < total_bits; input_bit++) {
        int dependencies = 0;
        for (int output_bit = 0; output_bit < total_bits; output_bit++) {
            dependencies += dependency_matrix[input_bit * total_bits + output_bit];
        }
        
        avg_dependencies += dependencies;
        if (dependencies < min_dependencies) min_dependencies = dependencies;
        if (dependencies > max_dependencies) max_dependencies = dependencies;
        if (dependencies >= total_bits * 0.99) fully_diffused_input_bits++;
    }
    avg_dependencies /= total_bits;
    
    double diffusion_percentage = (double)fully_diffused_input_bits / total_bits * 100.0;
    
    printf("  Diffusion results with %d rounds:\n", rounds);
    printf("  - Min output bits affected by one input bit: %d (%.2f%%)\n", 
           min_dependencies, (double)min_dependencies / total_bits * 100.0);
    printf("  - Max output bits affected by one input bit: %d (%.2f%%)\n", 
           max_dependencies, (double)max_dependencies / total_bits * 100.0);
    printf("  - Avg output bits affected by one input bit: %.2f (%.2f%%)\n", 
           avg_dependencies, avg_dependencies / total_bits * 100.0);
    printf("  - Input bits that affect >99%% of output bits: %d (%.2f%%)\n", 
           fully_diffused_input_bits, diffusion_percentage);
    
    // Calculate summary score (0-100)
    double diffusion_score = diffusion_percentage * (avg_dependencies / total_bits);
    
    free(dependency_matrix);
    return (int)(diffusion_score + 0.5);
}

// Test for the strict avalanche criterion (a single bit change should change each output bit with p=0.5)
static double test_strict_avalanche(int rounds) {
    uint8_t input[BUFFER_SIZE];
    uint8_t modified[BUFFER_SIZE];
    uint8_t output_orig[BUFFER_SIZE];
    uint8_t output_mod[BUFFER_SIZE];
    
    faester_state_t state_orig, state_mod;
    
    printf("Testing strict avalanche criterion with %d rounds...\n", rounds);
    
    // For each input bit position, we'll count how often each output bit changes
    int total_bits = BUFFER_SIZE * 8;
    unsigned long *bit_change_counts = calloc(total_bits * total_bits, sizeof(unsigned long));
    
    if (!bit_change_counts) {
        fprintf(stderr, "Memory allocation failed\n");
        return 0.0;
    }
    
    // For each input bit
    for (int input_bit = 0; input_bit < total_bits; input_bit++) {
        for (int test = 0; test < NUM_TESTS; test++) {
            // Generate random input
            fill_random(input, BUFFER_SIZE);
            
            // Create modified input with one bit flipped
            memcpy(modified, input, BUFFER_SIZE);
            int byte_pos = input_bit / 8;
            int bit_pos = input_bit % 8;
            modified[byte_pos] ^= (1 << bit_pos);
            
            // Run permutation on both inputs
            faester_init(&state_orig, input);
            faester_init(&state_mod, modified);
            
            faester_permute(&state_orig, rounds);
            faester_permute(&state_mod, rounds);
            
            faester_extract(&state_orig, output_orig);
            faester_extract(&state_mod, output_mod);
            
            // Count which output bits changed
            for (int output_bit = 0; output_bit < total_bits; output_bit++) {
                int out_byte_pos = output_bit / 8;
                int out_bit_pos = output_bit % 8;
                
                bool orig_bit = (output_orig[out_byte_pos] >> out_bit_pos) & 1;
                bool mod_bit = (output_mod[out_byte_pos] >> out_bit_pos) & 1;
                
                if (orig_bit != mod_bit) {
                    bit_change_counts[input_bit * total_bits + output_bit]++;
                }
            }
        }
        
        if (input_bit % 100 == 0 || input_bit == total_bits - 1) {
            printf("  Processed %d/%d input bits\r", input_bit + 1, total_bits);
            fflush(stdout);
        }
    }
    printf("\n");
    
    // Calculate statistics on the avalanche effect
    double max_deviation = 0.0;
    double avg_deviation = 0.0;
    int count = 0;
    
    for (int input_bit = 0; input_bit < total_bits; input_bit++) {
        for (int output_bit = 0; output_bit < total_bits; output_bit++) {
            unsigned long changes = bit_change_counts[input_bit * total_bits + output_bit];
            double probability = (double)changes / NUM_TESTS;
            double deviation = fabs(probability - 0.5);
            
            avg_deviation += deviation;
            if (deviation > max_deviation) {
                max_deviation = deviation;
            }
            count++;
        }
    }
    avg_deviation /= count;
    
    printf("  Strict avalanche criterion results with %d rounds:\n", rounds);
    printf("  - Maximum deviation from ideal (0.5): %.6f\n", max_deviation);
    printf("  - Average deviation from ideal (0.5): %.6f\n", avg_deviation);
    
    // Calculate SAC score (1.0 is perfect, 0.0 is worst)
    double sac_score = 1.0 - (2.0 * avg_deviation);
    
    free(bit_change_counts);
    return sac_score;
}

// Test statistical properties of the permutation output
static void test_statistical_properties(int rounds) {
    uint8_t input[BUFFER_SIZE];
    uint8_t output[BUFFER_SIZE];
    faester_state_t state;
    
    printf("Testing statistical properties with %d rounds...\n", rounds);
    
    // Statistical accumulators
    double sum_entropy = 0.0;
    double sum_chi_square = 0.0;
    double sum_correlation = 0.0;
    double sum_pi_error = 0.0;
    double max_bias = 0.0;
    double avg_bias = 0.0;
    double sum_bias_max = 0.0;
    double sum_bias_avg = 0.0;
    
    for (int sample = 0; sample < NUM_SAMPLES; sample++) {
        // Generate random input
        fill_random(input, BUFFER_SIZE);
        
        // Apply permutation
        faester_init(&state, input);
        faester_permute(&state, rounds);
        faester_extract(&state, output);
        
        // Calculate statistical properties
        double entropy = calculate_entropy(output, BUFFER_SIZE);
        double chi_square = chi_square_test(output, BUFFER_SIZE);
        double correlation = serial_correlation(output, BUFFER_SIZE);
        double pi_approx = monte_carlo_pi(output, BUFFER_SIZE);
        double pi_error = fabs(pi_approx - M_PI) / M_PI;
        
        // Bit bias for this sample
        bit_bias_test(output, BUFFER_SIZE, &max_bias, &avg_bias);
        
        // Accumulate statistics
        sum_entropy += entropy;
        sum_chi_square += chi_square;
        sum_correlation += fabs(correlation);
        sum_pi_error += pi_error;
        sum_bias_max += max_bias;
        sum_bias_avg += avg_bias;
        
        if ((sample + 1) % 100 == 0 || sample == NUM_SAMPLES - 1) {
            printf("  Processed %d/%d samples\r", sample + 1, NUM_SAMPLES);
            fflush(stdout);
        }
    }
    printf("\n");
    
    // Calculate averages
    double avg_entropy = sum_entropy / NUM_SAMPLES;
    double avg_chi_square = sum_chi_square / NUM_SAMPLES;
    double avg_correlation = sum_correlation / NUM_SAMPLES;
    double avg_pi_error = sum_pi_error / NUM_SAMPLES;
    double avg_bias_max = sum_bias_max / NUM_SAMPLES;
    double avg_bias_avg = sum_bias_avg / NUM_SAMPLES;
    
    // Ideal values for a truly random sequence
    double ideal_entropy = 8.0; // 8 bits per byte is maximum entropy
    double ideal_chi_square = 255.0; // For 256 possible byte values
    // We don't directly use these ideal values in calculations below,
    // but they're kept here as reference
    // double ideal_correlation = 0.0;
    // double ideal_pi_error = 0.0;
    // double ideal_bias = 0.0;
    
    // Calculate percentage of ideal
    double entropy_pct = (avg_entropy / ideal_entropy) * 100.0;
    double chi_square_pct = 100.0 - (avg_chi_square / (4 * ideal_chi_square)) * 100.0;
    double correlation_pct = (1.0 - avg_correlation) * 100.0;
    double pi_error_pct = (1.0 - avg_pi_error) * 100.0;
    double bias_pct = (1.0 - avg_bias_avg * 2.0) * 100.0;
    
    // Calculate overall statistical score (0-100)
    double stat_score = (entropy_pct + chi_square_pct + correlation_pct + pi_error_pct + bias_pct) / 5.0;
    
    printf("  Statistical test results with %d rounds:\n", rounds);
    printf("  - Average entropy: %.6f bits/byte (%.2f%% of ideal)\n", 
           avg_entropy, entropy_pct);
    printf("  - Average chi-square: %.6f (%.2f%% goodness)\n", 
           avg_chi_square, chi_square_pct);
    printf("  - Average serial correlation: %.6f (%.2f%% independence)\n", 
           avg_correlation, correlation_pct);
    printf("  - Average Monte Carlo Pi error: %.6f (%.2f%% accuracy)\n", 
           avg_pi_error, pi_error_pct);
    printf("  - Average maximum bit bias: %.6f\n", avg_bias_max);
    printf("  - Average bit bias: %.6f (%.2f%% unbiased)\n", 
           avg_bias_avg, bias_pct);
    printf("  - Overall statistical score: %.2f/100\n", stat_score);
}

// Test for collision resistance properties
static void test_collision_resistance(int rounds) {
    uint8_t input1[BUFFER_SIZE];
    uint8_t input2[BUFFER_SIZE];
    uint8_t output1[BUFFER_SIZE];
    uint8_t output2[BUFFER_SIZE];
    
    faester_state_t state1, state2;
    
    printf("Testing collision resistance with %d rounds...\n", rounds);
    
    // Statistics for Hamming distances
    int min_distance = BUFFER_SIZE * 8;
    int max_distance = 0;
    double sum_distance = 0.0;
    
    // Generate pairs of inputs with varying Hamming distances and compare outputs
    for (int test = 0; test < NUM_TESTS; test++) {
        // Generate first random input
        fill_random(input1, BUFFER_SIZE);
        
        // Create second input with specified number of bits different
        memcpy(input2, input1, BUFFER_SIZE);
        
        // Flip 1-10 random bits
        int bits_to_flip = (test % 10) + 1;
        for (int i = 0; i < bits_to_flip; i++) {
            int bit_position = rand() % (BUFFER_SIZE * 8);
            int byte_pos = bit_position / 8;
            int bit_pos = bit_position % 8;
            input2[byte_pos] ^= (1 << bit_pos);
        }
        
        // Process both inputs
        faester_init(&state1, input1);
        faester_init(&state2, input2);
        
        faester_permute(&state1, rounds);
        faester_permute(&state2, rounds);
        
        faester_extract(&state1, output1);
        faester_extract(&state2, output2);
        
        // Calculate Hamming distance between outputs
        int distance = hamming_distance(output1, output2, BUFFER_SIZE);
        
        // Update statistics
        sum_distance += distance;
        if (distance < min_distance) min_distance = distance;
        if (distance > max_distance) max_distance = distance;
        
        if ((test + 1) % 100 == 0 || test == NUM_TESTS - 1) {
            printf("  Processed %d/%d tests\r", test + 1, NUM_TESTS);
            fflush(stdout);
        }
    }
    printf("\n");
    
    // Calculate average distance
    double avg_distance = sum_distance / NUM_TESTS;
    double ideal_distance = BUFFER_SIZE * 4; // 50% of bits should differ
    double distance_pct = (avg_distance / ideal_distance) * 100.0;
    
    printf("  Collision resistance results with %d rounds:\n", rounds);
    printf("  - Minimum Hamming distance: %d (%.2f%%)\n", 
           min_distance, (double)min_distance / (BUFFER_SIZE * 8) * 100.0);
    printf("  - Maximum Hamming distance: %d (%.2f%%)\n", 
           max_distance, (double)max_distance / (BUFFER_SIZE * 8) * 100.0);
    printf("  - Average Hamming distance: %.2f (%.2f%% of ideal)\n", 
           avg_distance, distance_pct);
    
    // Test for simple differential patterns
    printf("  - Testing for differential patterns...\n");
    
    int diff_pattern_count = 0;
    
    // Generate 100 different single-bit input differences
    for (int test = 0; test < 100; test++) {
        fill_random(input1, BUFFER_SIZE);
        memcpy(input2, input1, BUFFER_SIZE);
        
        // Flip one bit
        int bit_pos = test % (BUFFER_SIZE * 8);
        input2[bit_pos / 8] ^= (1 << (bit_pos % 8));
        
        // Process inputs with multiple different inputs
        for (int variant = 0; variant < 10; variant++) {
            // Modify both inputs in the same way (preserving their difference)
            if (variant > 0) {
                int mod_bit = rand() % (BUFFER_SIZE * 8);
                input1[mod_bit / 8] ^= (1 << (mod_bit % 8));
                input2[mod_bit / 8] ^= (1 << (mod_bit % 8));
            }
            
            faester_init(&state1, input1);
            faester_init(&state2, input2);
            
            faester_permute(&state1, rounds);
            faester_permute(&state2, rounds);
            
            faester_extract(&state1, output1);
            faester_extract(&state2, output2);
            
            // Look for patterns in the output difference
            int output_diff_pattern = 0;
            
            // Count the number of bytes where the same bit position differs
            for (int i = 0; i < BUFFER_SIZE; i++) {
                uint8_t diff = output1[i] ^ output2[i];
                if (diff == 0) continue;
                
                // Check if only one bit differs in this byte
                if ((diff & (diff - 1)) == 0) {
                    output_diff_pattern++;
                }
            }
            
            // If more than 25% of the differing bytes have single-bit differences,
            // we consider this a potential differential pattern
            if (output_diff_pattern > BUFFER_SIZE / 8) {
                diff_pattern_count++;
            }
        }
    }
    
    printf("    Potential differential patterns detected: %d/1000 tests\n", diff_pattern_count);
}

// Print a summary assessment of the cryptographic properties
static void print_summary(double sac_score, int diffusion_score) {
    printf("\n");
    printf(ANSI_COLOR_CYAN "============================================\n" ANSI_COLOR_RESET);
    printf(ANSI_COLOR_CYAN "       CRYPTOGRAPHIC PROPERTY SUMMARY       \n" ANSI_COLOR_RESET);
    printf(ANSI_COLOR_CYAN "============================================\n" ANSI_COLOR_RESET);
    printf("\n");
    
    // Print SAC score
    printf("Strict Avalanche Criterion (SAC): ");
    if (sac_score >= 0.99) {
        printf(ANSI_COLOR_GREEN "Excellent (%.4f)\n" ANSI_COLOR_RESET, sac_score);
    } else if (sac_score >= 0.95) {
        printf(ANSI_COLOR_GREEN "Good (%.4f)\n" ANSI_COLOR_RESET, sac_score);
    } else if (sac_score >= 0.90) {
        printf(ANSI_COLOR_YELLOW "Adequate (%.4f)\n" ANSI_COLOR_RESET, sac_score);
    } else {
        printf(ANSI_COLOR_RED "Poor (%.4f)\n" ANSI_COLOR_RESET, sac_score);
    }
    
    // Print diffusion score
    printf("Diffusion Completeness: ");
    if (diffusion_score >= 95) {
        printf(ANSI_COLOR_GREEN "Excellent (%d%%)\n" ANSI_COLOR_RESET, diffusion_score);
    } else if (diffusion_score >= 85) {
        printf(ANSI_COLOR_GREEN "Good (%d%%)\n" ANSI_COLOR_RESET, diffusion_score);
    } else if (diffusion_score >= 75) {
        printf(ANSI_COLOR_YELLOW "Adequate (%d%%)\n" ANSI_COLOR_RESET, diffusion_score);
    } else {
        printf(ANSI_COLOR_RED "Poor (%d%%)\n" ANSI_COLOR_RESET, diffusion_score);
    }
    
    // Print overall assessment
    double overall_score = (sac_score * 100 + diffusion_score) / 2;
    printf("\nOverall Cryptographic Quality: ");
    
    if (overall_score >= 95) {
        printf(ANSI_COLOR_GREEN "Excellent (%.1f/100)\n" ANSI_COLOR_RESET, overall_score);
    } else if (overall_score >= 85) {
        printf(ANSI_COLOR_GREEN "Good (%.1f/100)\n" ANSI_COLOR_RESET, overall_score);
    } else if (overall_score >= 75) {
        printf(ANSI_COLOR_YELLOW "Adequate (%.1f/100)\n" ANSI_COLOR_RESET, overall_score);
    } else {
        printf(ANSI_COLOR_RED "Poor (%.1f/100)\n" ANSI_COLOR_RESET, overall_score);
    }
    printf("\n");
}

// Function to demonstrate basic permutation properties
static void demonstrate_basic_properties(int rounds) {
    uint8_t input[BUFFER_SIZE] = {0};
    uint8_t output[BUFFER_SIZE];
    faester_state_t state;
    
    printf("\n");
    printf(ANSI_COLOR_CYAN "============================================\n" ANSI_COLOR_RESET);
    printf(ANSI_COLOR_CYAN "          BASIC PROPERTY DEMONSTRATION      \n" ANSI_COLOR_RESET);
    printf(ANSI_COLOR_CYAN "============================================\n" ANSI_COLOR_RESET);
    printf("\n");
    
    // Test 1: Zero input
    memset(input, 0, BUFFER_SIZE);
    printf("Test 1: Zero input\n");
    printf("Input:  ");
    print_hex(input, BUFFER_SIZE, 32);
    
    faester_init(&state, input);
    faester_permute(&state, rounds);
    faester_extract(&state, output);
    
    printf("Output: ");
    print_hex(output, BUFFER_SIZE, 32);
    
    int nonzero_bytes = 0;
    for (int i = 0; i < BUFFER_SIZE; i++) {
        if (output[i] != 0) nonzero_bytes++;
    }
    printf("Non-zero bytes: %d/%d (%.2f%%)\n\n", 
           nonzero_bytes, BUFFER_SIZE, (double)nonzero_bytes/BUFFER_SIZE*100.0);
    
    // Test 2: Single bit set
    memset(input, 0, BUFFER_SIZE);
    input[0] = 1; // Set first bit
    printf("Test 2: Single bit set (first bit)\n");
    printf("Input:  ");
    print_hex(input, BUFFER_SIZE, 32);
    
    faester_init(&state, input);
    faester_permute(&state, rounds);
    faester_extract(&state, output);
    
    printf("Output: ");
    print_hex(output, BUFFER_SIZE, 32);
    
    nonzero_bytes = 0;
    for (int i = 0; i < BUFFER_SIZE; i++) {
        if (output[i] != 0) nonzero_bytes++;
    }
    printf("Non-zero bytes: %d/%d (%.2f%%)\n\n", 
           nonzero_bytes, BUFFER_SIZE, (double)nonzero_bytes/BUFFER_SIZE*100.0);
    
    // Test 3: Single bit set (middle)
    memset(input, 0, BUFFER_SIZE);
    input[BUFFER_SIZE/2] = 128; // Set middle bit
    printf("Test 3: Single bit set (middle bit)\n");
    printf("Input:  ");
    print_hex(input, BUFFER_SIZE, 32);
    
    faester_init(&state, input);
    faester_permute(&state, rounds);
    faester_extract(&state, output);
    
    printf("Output: ");
    print_hex(output, BUFFER_SIZE, 32);
    
    nonzero_bytes = 0;
    for (int i = 0; i < BUFFER_SIZE; i++) {
        if (output[i] != 0) nonzero_bytes++;
    }
    printf("Non-zero bytes: %d/%d (%.2f%%)\n\n", 
           nonzero_bytes, BUFFER_SIZE, (double)nonzero_bytes/BUFFER_SIZE*100.0);
    
    // Test 4: Alternating bits
    for (int i = 0; i < BUFFER_SIZE; i++) {
        input[i] = (i % 2 == 0) ? 0xAA : 0x55; // Alternating patterns
    }
    printf("Test 4: Alternating bit pattern\n");
    printf("Input:  ");
    print_hex(input, BUFFER_SIZE, 32);
    
    faester_init(&state, input);
    faester_permute(&state, rounds);
    faester_extract(&state, output);
    
    printf("Output: ");
    print_hex(output, BUFFER_SIZE, 32);
    
    // Test 5: Random input with entropy measurement
    fill_random(input, BUFFER_SIZE);
    printf("Test 5: Random input\n");
    printf("Input:  ");
    print_hex(input, BUFFER_SIZE, 32);
    printf("Input entropy: %.6f bits/byte\n", calculate_entropy(input, BUFFER_SIZE));
    
    faester_init(&state, input);
    faester_permute(&state, rounds);
    faester_extract(&state, output);
    
    printf("Output: ");
    print_hex(output, BUFFER_SIZE, 32);
    printf("Output entropy: %.6f bits/byte\n\n", calculate_entropy(output, BUFFER_SIZE));
}

int main(int argc, char *argv[]) {
    // Parse command line arguments
    int rounds = FAESTER_RECOMMENDED_ROUNDS;
    if (argc > 1) {
        rounds = atoi(argv[1]);
        if (rounds <= 0) {
            rounds = FAESTER_RECOMMENDED_ROUNDS;
        }
    }
    
    // Initialize random seed
    srand(time(NULL));
    
    printf(ANSI_COLOR_CYAN "================================================\n" ANSI_COLOR_RESET);
    printf(ANSI_COLOR_CYAN "       FAESTER CRYPTOGRAPHIC PROPERTY TESTS     \n" ANSI_COLOR_RESET);
    printf(ANSI_COLOR_CYAN "================================================\n" ANSI_COLOR_RESET);
    printf("\n");
    printf("Testing permutation with %d rounds\n", rounds);
    printf("Buffer size: %d bytes (%d bits)\n", BUFFER_SIZE, BUFFER_SIZE * 8);
    printf("Number of tests: %d\n", NUM_TESTS);
    printf("\n");
    
    // Demonstrate basic properties
    demonstrate_basic_properties(rounds);
    
    // Test diffusion completeness
    printf("\n");
    int diffusion_score = test_diffusion_completeness(rounds);
    
    // Test strict avalanche criterion
    printf("\n");
    double sac_score = test_strict_avalanche(rounds);
    
    // Test statistical properties
    printf("\n");
    test_statistical_properties(rounds);
    
    // Test collision resistance
    printf("\n");
    test_collision_resistance(rounds);
    
    // Print summary
    print_summary(sac_score, diffusion_score);
    
    return 0;
}