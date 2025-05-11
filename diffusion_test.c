#define _GNU_SOURCE  /* Define this first, before any includes */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include "src/faester_avx512.h"

#define BUFFER_SIZE 256  // Size in bytes (2048 bits)
#define NUM_TESTS 100    // Number of test samples per input bit

// ANSI color codes for prettier output
#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"

static void fill_random(uint8_t *buffer, size_t size) {
    for (size_t i = 0; i < size; i++) {
        buffer[i] = rand() & 0xFF;
    }
}

// Calculate Hamming distance between two buffers
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

// Test diffusion using multiple methods to get a comprehensive picture
void test_diffusion(int rounds) {
    printf(ANSI_COLOR_CYAN "================================================\n" ANSI_COLOR_RESET);
    printf(ANSI_COLOR_CYAN "        FAESTER DIFFUSION ANALYSIS (ROUNDS: %d)   \n" ANSI_COLOR_RESET, rounds);
    printf(ANSI_COLOR_CYAN "================================================\n" ANSI_COLOR_RESET);
    
    uint8_t input[BUFFER_SIZE];
    uint8_t modified[BUFFER_SIZE];
    uint8_t output_orig[BUFFER_SIZE];
    uint8_t output_mod[BUFFER_SIZE];
    
    faester_state_t state_orig, state_mod;
    
    printf("Testing with %d samples per input bit\n", NUM_TESTS);
    
    int total_bits = BUFFER_SIZE * 8;
    
    // 1. Create a bit dependency matrix
    printf("\n1. BIT DEPENDENCY ANALYSIS\n");
    
    // Allocate and initialize matrix
    double **dependency_matrix = (double **)malloc(total_bits * sizeof(double *));
    for (int i = 0; i < total_bits; i++) {
        dependency_matrix[i] = (double *)calloc(total_bits, sizeof(double));
    }
    
    // For each input bit
    for (int input_bit = 0; input_bit < total_bits; input_bit++) {
        if (input_bit % 100 == 0 || input_bit == total_bits - 1) {
            printf("Processing input bit %d/%d...\r", input_bit + 1, total_bits);
            fflush(stdout);
        }
        
        // Run multiple tests per input bit
        for (int test = 0; test < NUM_TESTS; test++) {
            // Generate random input
            fill_random(input, BUFFER_SIZE);
            
            // Create modified input with one bit flipped
            memcpy(modified, input, BUFFER_SIZE);
            int byte_pos = input_bit / 8;
            int bit_pos = input_bit % 8;
            modified[byte_pos] ^= (1 << bit_pos);
            
            // Process both inputs
            faester_init(&state_orig, input);
            faester_init(&state_mod, modified);
            
            faester_permute(&state_orig, rounds);
            faester_permute(&state_mod, rounds);
            
            faester_extract(&state_orig, output_orig);
            faester_extract(&state_mod, output_mod);
            
            // Update dependency matrix
            for (int output_bit = 0; output_bit < total_bits; output_bit++) {
                int out_byte_pos = output_bit / 8;
                int out_bit_pos = output_bit % 8;
                
                bool orig_bit = (output_orig[out_byte_pos] >> out_bit_pos) & 1;
                bool mod_bit = (output_mod[out_byte_pos] >> out_bit_pos) & 1;
                
                if (orig_bit != mod_bit) {
                    dependency_matrix[input_bit][output_bit] += 1.0 / NUM_TESTS;
                }
            }
        }
    }
    printf("\n");
    
    // Analyze bit dependencies
    // For each input bit, analyze how many output bits it influences
    int *input_bit_influence = (int *)calloc(total_bits, sizeof(int));
    int *output_bit_influenced = (int *)calloc(total_bits, sizeof(int));
    
    // Thresholds for counting significant influence (flippping probability > 0.2)
    double significant_influence_threshold = 0.2;
    
    // Count output bits significantly influenced by each input bit
    for (int i = 0; i < total_bits; i++) {
        for (int j = 0; j < total_bits; j++) {
            if (dependency_matrix[i][j] >= significant_influence_threshold) {
                input_bit_influence[i]++;
            }
        }
    }
    
    // Count input bits that significantly influence each output bit
    for (int j = 0; j < total_bits; j++) {
        for (int i = 0; i < total_bits; i++) {
            if (dependency_matrix[i][j] >= significant_influence_threshold) {
                output_bit_influenced[j]++;
            }
        }
    }
    
    // Calculate statistics for influence
    int min_influence = total_bits, max_influence = 0;
    double avg_influence = 0.0;
    
    for (int i = 0; i < total_bits; i++) {
        if (input_bit_influence[i] < min_influence) min_influence = input_bit_influence[i];
        if (input_bit_influence[i] > max_influence) max_influence = input_bit_influence[i];
        avg_influence += input_bit_influence[i];
    }
    avg_influence /= total_bits;
    
    // Calculate statistics for being influenced
    int min_influenced = total_bits, max_influenced = 0;
    double avg_influenced = 0.0;
    
    for (int j = 0; j < total_bits; j++) {
        if (output_bit_influenced[j] < min_influenced) min_influenced = output_bit_influenced[j];
        if (output_bit_influenced[j] > max_influenced) max_influenced = output_bit_influenced[j];
        avg_influenced += output_bit_influenced[j];
    }
    avg_influenced /= total_bits;
    
    // Print results of bit dependency analysis
    printf("Bit dependency statistics (influence threshold: %.2f):\n", significant_influence_threshold);
    printf("  Input bits significantly influencing output bits:\n");
    printf("    Min: %d (%.2f%%)\n", min_influence, 100.0 * min_influence / total_bits);
    printf("    Max: %d (%.2f%%)\n", max_influence, 100.0 * max_influence / total_bits);
    printf("    Avg: %.2f (%.2f%%)\n", avg_influence, 100.0 * avg_influence / total_bits);
    printf("\n");
    printf("  Output bits significantly influenced by input bits:\n");
    printf("    Min: %d (%.2f%%)\n", min_influenced, 100.0 * min_influenced / total_bits);
    printf("    Max: %d (%.2f%%)\n", max_influenced, 100.0 * max_influenced / total_bits);
    printf("    Avg: %.2f (%.2f%%)\n", avg_influenced, 100.0 * avg_influenced / total_bits);
    
    // Calculate the diffusion score based on a more reasonable criterion
    double input_influence_score = avg_influence / total_bits;
    double output_influenced_score = avg_influenced / total_bits;
    
    double bit_dependency_score = (input_influence_score + output_influenced_score) / 2.0 * 100.0;
    
    printf("\nBit dependency score: %.2f/100\n", bit_dependency_score);
    
    // 2. Hamming distance analysis
    printf("\n2. HAMMING DISTANCE ANALYSIS\n");
    
    // For single bit changes
    double avg_hamming_distance = 0.0;
    int num_samples = 1000;
    
    for (int i = 0; i < num_samples; i++) {
        // Generate random input
        fill_random(input, BUFFER_SIZE);
        
        // Flip a random bit
        int bit_to_flip = rand() % total_bits;
        int byte_pos = bit_to_flip / 8;
        int bit_pos = bit_to_flip % 8;
        
        memcpy(modified, input, BUFFER_SIZE);
        modified[byte_pos] ^= (1 << bit_pos);
        
        // Process both inputs
        faester_init(&state_orig, input);
        faester_init(&state_mod, modified);
        
        faester_permute(&state_orig, rounds);
        faester_permute(&state_mod, rounds);
        
        faester_extract(&state_orig, output_orig);
        faester_extract(&state_mod, output_mod);
        
        // Calculate Hamming distance
        int distance = hamming_distance(output_orig, output_mod, BUFFER_SIZE);
        avg_hamming_distance += distance;
    }
    avg_hamming_distance /= num_samples;
    
    double hamming_distance_ratio = avg_hamming_distance / total_bits;
    double hamming_distance_score = 0.0;
    
    // Ideal ratio is 0.5 (50% of bits flipped)
    // Calculate score based on deviation from 0.5
    hamming_distance_score = (1.0 - fabs(hamming_distance_ratio - 0.5) * 4.0) * 100.0;
    if (hamming_distance_score < 0.0) hamming_distance_score = 0.0;
    
    printf("Single bit change results:\n");
    printf("  Average Hamming distance: %.2f (%.2f%% of total bits)\n", 
           avg_hamming_distance, 100.0 * avg_hamming_distance / total_bits);
    printf("  Ideal distance: %.2f (50%% of total bits)\n", total_bits * 0.5);
    printf("\nHamming distance score: %.2f/100\n", hamming_distance_score);
    
    // 3. Diffusion speed test
    printf("\n3. DIFFUSION SPEED ANALYSIS\n");
    
    // Round-wise diffusion measure
    int rounds_to_test = rounds;
    if (rounds_to_test > 16) rounds_to_test = 16; // Cap to avoid excessive testing
    
    double *diffusion_by_round = (double *)calloc(rounds_to_test + 1, sizeof(double));
    
    num_samples = 100;
    for (int i = 0; i < num_samples; i++) {
        // Generate random input
        fill_random(input, BUFFER_SIZE);
        
        // Flip a random bit
        int bit_to_flip = rand() % total_bits;
        int byte_pos = bit_to_flip / 8;
        int bit_pos = bit_to_flip % 8;
        
        memcpy(modified, input, BUFFER_SIZE);
        modified[byte_pos] ^= (1 << bit_pos);
        
        // Test diffusion after each round
        for (int r = 0; r <= rounds_to_test; r++) {
            // Process both inputs
            faester_init(&state_orig, input);
            faester_init(&state_mod, modified);
            
            faester_permute(&state_orig, r);
            faester_permute(&state_mod, r);
            
            faester_extract(&state_orig, output_orig);
            faester_extract(&state_mod, output_mod);
            
            // Calculate Hamming distance
            int distance = hamming_distance(output_orig, output_mod, BUFFER_SIZE);
            diffusion_by_round[r] += distance;
        }
    }
    
    // Calculate the average for each round
    for (int r = 0; r <= rounds_to_test; r++) {
        diffusion_by_round[r] /= num_samples;
    }
    
    // Determine how many rounds it takes to reach 90% of full diffusion
    double full_diffusion = diffusion_by_round[rounds_to_test];
    int rounds_to_90_percent = 0;
    
    for (int r = 0; r <= rounds_to_test; r++) {
        if (diffusion_by_round[r] >= 0.9 * full_diffusion) {
            rounds_to_90_percent = r;
            break;
        }
    }
    
    // Print round-wise diffusion
    printf("Diffusion by round:\n");
    for (int r = 0; r <= rounds_to_test; r++) {
        printf("  Round %2d: %.2f bits (%.2f%%)\n", 
               r, diffusion_by_round[r], 100.0 * diffusion_by_round[r] / total_bits);
    }
    
    double diffusion_speed_score = 0.0;
    
    // Calculate diffusion speed score - better if full diffusion is reached in fewer rounds
    if (rounds_to_90_percent <= 0) {
        diffusion_speed_score = 0.0;
    } else {
        diffusion_speed_score = 100.0 * (1.0 - (double)rounds_to_90_percent / rounds_to_test);
    }
    
    printf("\nRounds to reach 90%% of full diffusion: %d\n", rounds_to_90_percent);
    printf("Diffusion speed score: %.2f/100\n", diffusion_speed_score);
    
    // Calculate overall diffusion score
    double overall_diffusion_score = (bit_dependency_score + hamming_distance_score + diffusion_speed_score) / 3.0;
    
    printf("\n" ANSI_COLOR_CYAN "OVERALL DIFFUSION SCORE: %.2f/100" ANSI_COLOR_RESET "\n", overall_diffusion_score);
    
    char *assessment;
    if (overall_diffusion_score >= 90.0) {
        assessment = ANSI_COLOR_GREEN "EXCELLENT" ANSI_COLOR_RESET;
    } else if (overall_diffusion_score >= 80.0) {
        assessment = ANSI_COLOR_GREEN "GOOD" ANSI_COLOR_RESET;
    } else if (overall_diffusion_score >= 70.0) {
        assessment = ANSI_COLOR_YELLOW "ADEQUATE" ANSI_COLOR_RESET;
    } else {
        assessment = ANSI_COLOR_RED "POOR" ANSI_COLOR_RESET;
    }
    
    printf("Assessment: %s\n", assessment);
    
    // Clean up
    for (int i = 0; i < total_bits; i++) {
        free(dependency_matrix[i]);
    }
    free(dependency_matrix);
    free(input_bit_influence);
    free(output_bit_influenced);
    free(diffusion_by_round);
}

int main(int argc, char *argv[]) {
    // Parse command line arguments
    int rounds = 8;
    if (argc > 1) {
        rounds = atoi(argv[1]);
        if (rounds <= 0) rounds = 8;
    }
    
    // Initialize random seed
    srand(time(NULL));
    
    // Run diffusion tests
    test_diffusion(rounds);
    
    return 0;
}