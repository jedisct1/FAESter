#define _GNU_SOURCE  /* Define this first, before any includes */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include "src/faester_avx512.h"

// Number of tests to perform for each input bit
#define NUM_TESTS 100

// Size of the input/output in bytes (256 bytes = 2048 bits)
#define BUFFER_SIZE 256

// Structure to hold a bit-dependency matrix
typedef struct {
    // For each input bit, store the probability of each output bit changing
    double matrix[BUFFER_SIZE * 8][BUFFER_SIZE * 8];
    double row_averages[BUFFER_SIZE * 8];     // Average for each input bit
    double column_averages[BUFFER_SIZE * 8];  // Average for each output bit
    double overall_average;                   // Overall average across all bits
} bit_dependency_t;

// Flip a specific bit in a byte array
static void flip_bit(uint8_t *data, int bit_position) {
    int byte_pos = bit_position / 8;
    int bit_pos = bit_position % 8;
    data[byte_pos] ^= (1 << bit_pos);
}

// Check if a specific bit is set
static int is_bit_set(const uint8_t *data, int bit_position) {
    int byte_pos = bit_position / 8;
    int bit_pos = bit_position % 8;
    return (data[byte_pos] & (1 << bit_pos)) ? 1 : 0;
}

// Fill a buffer with random data
static void randomize_buffer(uint8_t *buffer, size_t size) {
    for (size_t i = 0; i < size; i++) {
        buffer[i] = rand() & 0xFF;
    }
}

// Calculate Hamming weight (number of bits set)
static int hamming_weight(const uint8_t *data, size_t size) {
    int weight = 0;
    for (size_t i = 0; i < size; i++) {
        uint8_t byte = data[i];
        while (byte) {
            weight += (byte & 1);
            byte >>= 1;
        }
    }
    return weight;
}

// Calculate Hamming distance between two buffers
static int hamming_distance(const uint8_t *data1, const uint8_t *data2, size_t size) {
    int distance = 0;
    for (size_t i = 0; i < size; i++) {
        uint8_t xor_byte = data1[i] ^ data2[i];
        while (xor_byte) {
            distance += (xor_byte & 1);
            xor_byte >>= 1;
        }
    }
    return distance;
}

// Initialize the bit dependency matrix to zeros
static void init_bit_dependency(bit_dependency_t *bd) {
    memset(bd->matrix, 0, sizeof(bd->matrix));
    memset(bd->row_averages, 0, sizeof(bd->row_averages));
    memset(bd->column_averages, 0, sizeof(bd->column_averages));
    bd->overall_average = 0.0;
}

// Calculate statistics for the bit dependency matrix
static void calculate_statistics(bit_dependency_t *bd) {
    int total_bits = BUFFER_SIZE * 8;
    double total_sum = 0.0;
    
    // Calculate row averages (input bits)
    for (int i = 0; i < total_bits; i++) {
        double row_sum = 0.0;
        for (int j = 0; j < total_bits; j++) {
            row_sum += bd->matrix[i][j];
        }
        bd->row_averages[i] = row_sum / total_bits;
        total_sum += row_sum;
    }
    
    // Calculate column averages (output bits)
    for (int j = 0; j < total_bits; j++) {
        double col_sum = 0.0;
        for (int i = 0; i < total_bits; i++) {
            col_sum += bd->matrix[i][j];
        }
        bd->column_averages[j] = col_sum / total_bits;
    }
    
    // Calculate overall average
    bd->overall_average = total_sum / (total_bits * total_bits);
}

// Export the bit dependency matrix to a CSV file for visualization
static void export_matrix_to_csv(const bit_dependency_t *bd, const char *filename) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Failed to open file %s for writing\n", filename);
        return;
    }
    
    int total_bits = BUFFER_SIZE * 8;
    
    // Write header row
    fprintf(fp, "Input Bit");
    for (int j = 0; j < total_bits; j++) {
        fprintf(fp, ",Out_%d", j);
    }
    fprintf(fp, ",Average\n");
    
    // Write matrix data
    for (int i = 0; i < total_bits; i++) {
        fprintf(fp, "In_%d", i);
        for (int j = 0; j < total_bits; j++) {
            fprintf(fp, ",%.6f", bd->matrix[i][j]);
        }
        fprintf(fp, ",%.6f\n", bd->row_averages[i]);
    }
    
    // Write column averages
    fprintf(fp, "Average");
    for (int j = 0; j < total_bits; j++) {
        fprintf(fp, ",%.6f", bd->column_averages[j]);
    }
    fprintf(fp, ",%.6f\n", bd->overall_average);
    
    fclose(fp);
    printf("Exported bit dependency matrix to %s\n", filename);
}

// Run the avalanche test and build the bit dependency matrix
static void run_avalanche_test(bit_dependency_t *bd, int rounds) {
    const int total_bits = BUFFER_SIZE * 8;
    const int total_tests = total_bits * NUM_TESTS;
    
    uint8_t original[BUFFER_SIZE];
    uint8_t modified[BUFFER_SIZE];
    uint8_t output_orig[BUFFER_SIZE];
    uint8_t output_mod[BUFFER_SIZE];
    faester_state_t state_orig, state_mod;
    
    int test_num = 0;
    
    // Statistics for overall bit flip distribution
    int *flip_counts = calloc(total_bits + 1, sizeof(int));
    if (!flip_counts) {
        fprintf(stderr, "Memory allocation failed\n");
        return;
    }
    
    printf("Running %d avalanche tests with %d rounds...\n", total_tests, rounds);
    
    // For each input bit
    for (int input_bit = 0; input_bit < total_bits; input_bit++) {
        // Run multiple tests with different random inputs
        for (int test = 0; test < NUM_TESTS; test++) {
            // Generate random input
            randomize_buffer(original, BUFFER_SIZE);
            
            // Create a copy with one bit flipped
            memcpy(modified, original, BUFFER_SIZE);
            flip_bit(modified, input_bit);
            
            // Process both inputs through Faester
            faester_init(&state_orig, original);
            faester_init(&state_mod, modified);
            
            faester_permute(&state_orig, rounds);
            faester_permute(&state_mod, rounds);
            
            faester_extract(&state_orig, output_orig);
            faester_extract(&state_mod, output_mod);
            
            // Calculate total bit flips
            int bit_flips = hamming_distance(output_orig, output_mod, BUFFER_SIZE);
            if (bit_flips <= total_bits) {
                flip_counts[bit_flips]++;
            }
            
            // Update the bit dependency matrix
            for (int output_bit = 0; output_bit < total_bits; output_bit++) {
                int orig_bit_val = is_bit_set(output_orig, output_bit);
                int mod_bit_val = is_bit_set(output_mod, output_bit);
                
                if (orig_bit_val != mod_bit_val) {
                    // This output bit flipped, increment its count
                    bd->matrix[input_bit][output_bit] += 1.0 / NUM_TESTS;
                }
            }
            
            // Progress indicator
            test_num++;
            if (test_num % 1000 == 0 || test_num == total_tests) {
                printf("\rProgress: %.1f%% (%d/%d tests)", 
                       (double)test_num * 100.0 / total_tests, test_num, total_tests);
                fflush(stdout);
            }
        }
    }
    printf("\nTests completed.\n");
    
    // Calculate overall statistics
    calculate_statistics(bd);
    
    // Calculate ideal values
    double ideal_probability = 0.5;
    double ideal_flips = total_bits * ideal_probability;
    
    // Calculate overall bit flip statistics
    double avg_flips = 0.0;
    int total_count = 0;
    for (int i = 0; i <= total_bits; i++) {
        if (flip_counts[i] > 0) {
            avg_flips += i * flip_counts[i];
            total_count += flip_counts[i];
        }
    }
    avg_flips /= total_count;
    
    // Calculate standard deviation
    double variance = 0.0;
    for (int i = 0; i <= total_bits; i++) {
        if (flip_counts[i] > 0) {
            double diff = i - avg_flips;
            variance += diff * diff * flip_counts[i];
        }
    }
    variance /= total_count;
    double stddev = sqrt(variance);
    
    // Print summary statistics
    printf("\nAvalanche Test Summary (Rounds: %d)\n", rounds);
    printf("================================\n");
    printf("Average bit flips: %.2f out of %d (%.2f%%)\n", 
           avg_flips, total_bits, avg_flips * 100.0 / total_bits);
    printf("Ideal bit flips: %.0f (50%%)\n", ideal_flips);
    printf("Standard deviation: %.2f bits\n", stddev);
    printf("Overall bit flip probability: %.6f (ideal: 0.5)\n", bd->overall_average);
    
    // Print distribution of bit flips in histogram form
    printf("\nBit Flip Distribution:\n");
    
    // Find the maximum count for scaling
    int max_count = 0;
    for (int i = 0; i <= total_bits; i++) {
        if (flip_counts[i] > max_count) {
            max_count = flip_counts[i];
        }
    }
    
    // Group into bins of 32 bits for better visualization
    int bin_size = 32;
    int num_bins = (total_bits + bin_size - 1) / bin_size;
    
    for (int bin = 0; bin < num_bins; bin++) {
        int start = bin * bin_size;
        int end = start + bin_size - 1;
        if (end > total_bits) end = total_bits;
        
        int bin_count = 0;
        for (int i = start; i <= end; i++) {
            bin_count += flip_counts[i];
        }
        
        double percentage = (double)bin_count * 100.0 / total_count;
        
        // Create visual histogram bar (scaled to 50 chars width)
        int bar_width = (int)(percentage / 2);
        if (bar_width > 50) bar_width = 50;
        
        printf("%4d-%-4d bits: %7d (%.2f%%) ", start, end, bin_count, percentage);
        for (int i = 0; i < bar_width; i++) {
            printf("█");
        }
        printf("\n");
    }
    
    free(flip_counts);
}

// Find the minimum and maximum values in the matrix
static void find_matrix_extremes(const bit_dependency_t *bd, 
                                double *min_val, double *max_val, 
                                int *min_i, int *min_j, int *max_i, int *max_j) {
    int total_bits = BUFFER_SIZE * 8;
    *min_val = 1.0;
    *max_val = 0.0;
    
    for (int i = 0; i < total_bits; i++) {
        for (int j = 0; j < total_bits; j++) {
            if (bd->matrix[i][j] < *min_val) {
                *min_val = bd->matrix[i][j];
                *min_i = i;
                *min_j = j;
            }
            if (bd->matrix[i][j] > *max_val) {
                *max_val = bd->matrix[i][j];
                *max_i = i;
                *max_j = j;
            }
        }
    }
}

// Calculate chi-square measure of the bit dependency matrix
static double calculate_chi_square(const bit_dependency_t *bd) {
    int total_bits = BUFFER_SIZE * 8;
    double ideal_probability = 0.5;
    double chi_square = 0.0;
    
    for (int i = 0; i < total_bits; i++) {
        for (int j = 0; j < total_bits; j++) {
            double observed = bd->matrix[i][j];
            double expected = ideal_probability;
            
            if (observed > 0.0 && observed < 1.0) {
                double diff = observed - expected;
                chi_square += (diff * diff) / (observed * (1.0 - observed));
            }
        }
    }
    
    return chi_square;
}

int main(int argc, char *argv[]) {
    // Initialize random seed
    srand(time(NULL));
    
    // Parse command line arguments
    int rounds = FAESTER_RECOMMENDED_ROUNDS;
    if (argc > 1) {
        rounds = atoi(argv[1]);
        if (rounds <= 0) {
            rounds = FAESTER_RECOMMENDED_ROUNDS;
        }
    }
    
    printf("Faester Avalanche Analysis (Detailed)\n");
    printf("=====================================\n");
    printf("Buffer size: %d bytes (%d bits)\n", BUFFER_SIZE, BUFFER_SIZE * 8);
    printf("Number of rounds: %d\n", rounds);
    printf("Tests per input bit: %d\n", NUM_TESTS);
    printf("Total bits tested: %d\n", BUFFER_SIZE * 8);
    printf("Total tests: %d\n\n", BUFFER_SIZE * 8 * NUM_TESTS);
    
    // Allocate the bit dependency matrix
    bit_dependency_t *bd = malloc(sizeof(bit_dependency_t));
    if (!bd) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }
    
    // Initialize the matrix
    init_bit_dependency(bd);
    
    // Run the avalanche test
    run_avalanche_test(bd, rounds);
    
    // Find min and max values
    double min_val, max_val;
    int min_i, min_j, max_i, max_j;
    find_matrix_extremes(bd, &min_val, &max_val, &min_i, &min_j, &max_i, &max_j);
    
    printf("\nBit Dependency Matrix Stats:\n");
    printf("Min Probability: %.6f (Input bit %d -> Output bit %d)\n", min_val, min_i, min_j);
    printf("Max Probability: %.6f (Input bit %d -> Output bit %d)\n", max_val, max_i, max_j);
    printf("Average Probability: %.6f (ideal: 0.5)\n", bd->overall_average);
    
    // Calculate chi-square
    double chi_square = calculate_chi_square(bd);
    printf("Chi-square: %.6f\n", chi_square);
    
    // Find inputs and outputs with most deviation from ideal
    double max_row_dev = 0.0, max_col_dev = 0.0;
    int max_row_idx = 0, max_col_idx = 0;
    
    for (int i = 0; i < BUFFER_SIZE * 8; i++) {
        double row_dev = fabs(bd->row_averages[i] - 0.5);
        double col_dev = fabs(bd->column_averages[i] - 0.5);
        
        if (row_dev > max_row_dev) {
            max_row_dev = row_dev;
            max_row_idx = i;
        }
        
        if (col_dev > max_col_dev) {
            max_col_dev = col_dev;
            max_col_idx = i;
        }
    }
    
    printf("\nMaximum deviations from ideal (0.5):\n");
    printf("Input bit %d: %.6f (deviation: %.6f)\n", 
           max_row_idx, bd->row_averages[max_row_idx], max_row_dev);
    printf("Output bit %d: %.6f (deviation: %.6f)\n", 
           max_col_idx, bd->column_averages[max_col_idx], max_col_dev);
    
    // Export the matrix for further analysis
    export_matrix_to_csv(bd, "faester_avalanche_matrix.csv");
    
    // Clean up
    free(bd);
    
    return 0;
}