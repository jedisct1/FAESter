#define _GNU_SOURCE  /* Define this first, before any includes */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include "src/faester_avx512.h"

// Number of tests to perform for each bit
#define NUM_TESTS 1000

// Function to calculate Hamming weight (number of bits set)
// (Not used in current code but kept for potential future use)
static inline int hamming_weight(const uint8_t *data, size_t len) {
    int count = 0;
    for (size_t i = 0; i < len; i++) {
        uint8_t byte = data[i];
        // Count bits set in this byte using bit manipulation
        while (byte) {
            count += (byte & 1);
            byte >>= 1;
        }
    }
    return count;
}

// Function to calculate Hamming distance between two buffers
static inline int hamming_distance(const uint8_t *data1, const uint8_t *data2, size_t len) {
    int count = 0;
    for (size_t i = 0; i < len; i++) {
        uint8_t xor_byte = data1[i] ^ data2[i];
        // Count bits set in the XOR (different bits)
        while (xor_byte) {
            count += (xor_byte & 1);
            xor_byte >>= 1;
        }
    }
    return count;
}

// Create a modified copy of input with a single bit flipped at the specified position
static void flip_bit(const uint8_t *input, uint8_t *modified, size_t len, int bit_pos) {
    // Copy input to modified
    memcpy(modified, input, len);

    // Calculate byte position and bit within byte
    size_t byte_pos = bit_pos / 8;
    int bit_in_byte = bit_pos % 8;

    // Flip the specified bit
    if (byte_pos < len) {
        modified[byte_pos] ^= (1 << bit_in_byte);
    }
}

int main() {
    uint8_t input[256] = {0};
    uint8_t modified_input[256] = {0};
    uint8_t output1[256] = {0};
    uint8_t output2[256] = {0};
    
    faester_state_t state1, state2;
    
    // Stats variables
    int total_bits = 256 * 8; // Total bits in output (2048)
    int *bit_flip_counts = calloc(total_bits, sizeof(int));
    double *bit_probabilities = calloc(total_bits, sizeof(double));
    // Ideal probability for each bit flip is 0.5
    double ideal_prob = 0.5;
    double chi_square = 0.0;
    double max_deviation = 0.0;
    double min_prob = 1.0, max_prob = 0.0;
    double average_prob = 0.0;
    
    if (!bit_flip_counts || !bit_probabilities) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }
    
    // Set up initial input (we'll use randomized data for better testing)
    srand(time(NULL));
    for (int i = 0; i < 256; i++) {
        input[i] = rand() & 0xFF;
    }
    
    printf("Faester Avalanche Effect Analysis\n");
    printf("================================\n\n");
    
    printf("Testing with %d random samples per bit...\n", NUM_TESTS);
    
    // Test statistics variables
    double total_avg_flips = 0.0;
    int total_tests = 0;
    int bit_flips_per_test[NUM_TESTS * total_bits];
    int flip_index = 0;
    
    // For each bit in the input
    for (int bit_pos = 0; bit_pos < total_bits; bit_pos++) {
        // Perform multiple tests with different random inputs
        int total_flips = 0;
        
        for (int test = 0; test < NUM_TESTS; test++) {
            // Randomize input for each test (keep some bits from previous)
            for (int i = 0; i < 256; i++) {
                if (rand() % 4 == 0) { // 25% chance to change a byte
                    input[i] = rand() & 0xFF;
                }
            }
            
            // Create modified input with one bit flipped
            flip_bit(input, modified_input, 256, bit_pos);
            
            // Process both inputs
            faester_init(&state1, input);
            faester_init(&state2, modified_input);
            
            faester_permute(&state1, FAESTER_RECOMMENDED_ROUNDS);
            faester_permute(&state2, FAESTER_RECOMMENDED_ROUNDS);
            
            faester_extract(&state1, output1);
            faester_extract(&state2, output2);
            
            // Calculate Hamming distance (number of bit differences)
            int distance = hamming_distance(output1, output2, 256);
            total_flips += distance;
            
            // Store individual test result for distribution analysis
            bit_flips_per_test[flip_index++] = distance;
        }
        
        // Update statistics
        double avg_flips = (double)total_flips / NUM_TESTS;
        total_avg_flips += avg_flips;
        total_tests++;
        
        // Calculate probability of each bit flipping
        for (int bit = 0; bit < total_bits; bit++) {
            double prob = (double)bit_flip_counts[bit] / NUM_TESTS;
            bit_probabilities[bit] = prob;
            average_prob += prob;
            
            if (prob < min_prob) min_prob = prob;
            if (prob > max_prob) max_prob = prob;
            
            double deviation = fabs(prob - ideal_prob);
            if (deviation > max_deviation) max_deviation = deviation;
            
            // Chi-square calculation
            if (prob > 0.0 && prob < 1.0) { // Avoid division by zero
                chi_square += ((prob - ideal_prob) * (prob - ideal_prob)) / (prob * (1.0 - prob));
            }
        }
        
        // Progress indicator every 256 bits
        if (bit_pos % 256 == 255) {
            printf("Processed %d bits (%.1f%%)\n", bit_pos + 1, (bit_pos + 1) * 100.0 / total_bits);
        }
    }
    
    // Calculate overall statistics
    average_prob /= total_bits;
    double average_flips = total_avg_flips / total_tests;
    double ideal_flips = total_bits * 0.5; // Ideal: half the bits should flip
    double avalanche_quality = average_flips / ideal_flips;
    
    // Calculate the standard deviation of bit flips
    double sum_squared_diff = 0.0;
    for (int i = 0; i < flip_index; i++) {
        double diff = bit_flips_per_test[i] - average_flips;
        sum_squared_diff += diff * diff;
    }
    double stddev = sqrt(sum_squared_diff / flip_index);
    
    // Create a histogram of bit flips
    int histogram[65] = {0}; // We'll group by 32 bits (0-31, 32-63, etc.)
    for (int i = 0; i < flip_index; i++) {
        int bin = bit_flips_per_test[i] / 32;
        if (bin >= 65) bin = 64;
        histogram[bin]++;
    }
    
    // Print results
    printf("\nAvalanche Effect Results\n");
    printf("======================\n");
    printf("Input size: %d bytes (%d bits)\n", 256, total_bits);
    printf("Output size: %d bytes (%d bits)\n", 256, total_bits);
    printf("Number of rounds: %d\n", FAESTER_RECOMMENDED_ROUNDS);
    printf("Tests per bit: %d\n", NUM_TESTS);
    printf("Total tests: %d\n\n", total_tests * NUM_TESTS);
    
    printf("Average bit flips: %.2f out of %d bits (%.2f%%)\n", 
           average_flips, total_bits, (average_flips * 100.0) / total_bits);
    printf("Ideal bit flips: %.2f (50%%)\n", ideal_flips);
    printf("Avalanche quality: %.4f (1.0 is perfect)\n", avalanche_quality);
    printf("Standard deviation: %.2f bits\n", stddev);
    printf("Chi-square: %.4f\n\n", chi_square);
    
    // Print distribution of bit flips
    printf("Bit flip distribution:\n");
    for (int i = 0; i < 65; i++) {
        if (histogram[i] > 0) {
            int lower = i * 32;
            int upper = (i + 1) * 32 - 1;
            if (i == 64) {
                printf("  > 2048 bits: %d tests (%.2f%%)\n", 
                       histogram[i], histogram[i] * 100.0 / (total_tests * NUM_TESTS));
            } else {
                printf("  %4d-%-4d bits: %d tests (%.2f%%)\n", 
                       lower, upper, histogram[i], histogram[i] * 100.0 / (total_tests * NUM_TESTS));
            }
        }
    }
    
    // Clean up
    free(bit_flip_counts);
    free(bit_probabilities);
    
    return 0;
}