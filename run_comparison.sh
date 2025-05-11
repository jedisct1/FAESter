#!/bin/bash

# Ensure we're in the correct directory
cd "$(dirname "$0")"

# Make sure everything is built
make compare_permutations benchmark_compare

# Create output directory
mkdir -p comparisons

# Run the cryptographic property comparison
echo "Running cryptographic property comparison..."
./compare_permutations > comparisons/crypto_comparison.txt

# Run the performance comparison
echo "Running performance comparison..."
./benchmark_compare > comparisons/performance_comparison.txt

echo "Done! Results saved to comparisons/ directory."
echo ""
echo "Cryptographic comparison: comparisons/crypto_comparison.txt"
echo "Performance comparison: comparisons/performance_comparison.txt"