# Faester Cryptographic Permutation

Faester is a high-performance 2048-bit cryptographic permutation implemented using AES-NI and AVX-512 instructions. This repository contains the implementation of Faester along with various cryptographic analysis tools, benchmarks, and a comparison with AREION (a 512-bit permutation).

## Building

The project can be built using make:

```bash
make
```

This will build all the benchmarking and analysis tools.

## Cryptographic Analysis Tools

### Basic Benchmarking

Simple benchmarking of the Faester permutation with cycles per byte measurements:

```bash
make run
```

Advanced benchmarking with detailed statistical analysis:

```bash
make run_advanced
```

### Cryptographic Property Testing

Test for avalanche effect (basic version):

```bash
make run_avalanche
```

Detailed avalanche effect analysis with bit dependency matrix:

```bash
make run_avalanche_detailed
```

Comprehensive cryptographic property testing:

```bash
make run_crypto
```

Detailed diffusion analysis:

```bash
make run_diffusion
```

## Comparison with AREION

This repository also includes tools to compare Faester with AREION, a 512-bit permutation.

### Cryptographic Property Comparison

Compare the cryptographic properties of Faester and AREION:

```bash
make run_compare
```

This will run a series of tests on both permutations and provide a side-by-side comparison of:
- Avalanche effect
- Diffusion completeness
- Statistical properties
- Overall cryptographic quality

### Performance Comparison

Compare the performance characteristics of Faester and AREION:

```bash
make run_bench_compare
```

This benchmark provides a detailed performance comparison including:
- Cycles per permutation
- Cycles per byte
- Throughput (GB/s)
- Relative bit-level efficiency
- State size vs. performance tradeoffs

## Notes on State Sizes

- Faester operates on a 2048-bit state (256 bytes)
- AREION operates on a 512-bit state (64 bytes)

When comparing the two permutations, it's important to consider this difference in state size when evaluating security properties and performance characteristics.

## Platform Support

The code should work on both x86 platforms (with AVX-512 and AES-NI) and Apple Silicon (using emulation through untrinsics). The build system automatically detects the platform and adjusts compilation flags accordingly.