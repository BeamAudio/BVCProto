# BVC - C++ Implementation

This directory contains the optimized, high-performance implementation of the Beam Vocal Codec.

## Architecture

The codec employs advanced parallelization strategies for both encoding and decoding:

### Encoder (Producer-Consumer)
1.  **Producer (Main Thread):** Reads audio, performs LPC Analysis, Pitch Detection, and Frame Classification.
2.  **Workers (Thread Pool):** Execute Greedy Matching Pursuit in parallel to find sparse atoms.
3.  **Consumer (Writer Thread):** Writes the Entropy-coded bitstream.

### Decoder (Parallel Reconstruction)
1.  **Parser:** Reads the bitstream sequentially.
2.  **Parallel Reconstruction:** Uses **OpenMP** to reconstruct the sparse excitation signal (Dictionary Atoms) in parallel for all frames.
3.  **Synthesis:** Performs Overlap-Add and **Direct Form I** IIR Filtering sequentially to ensure stability and prevent state discontinuities.

## Source Layout

*   **`core/BVCEncoder.cpp`**: Main encoding pipeline with thread pool management.
*   **`core/BVCDecoder.cpp`**: Parallel decoding logic with Direct Form I synthesis filtering.
*   **`core/MatchingPursuit.cpp`**: Optimized Greedy Matching Pursuit algorithm.
*   **`core/Dictionary.cpp`**: Generates **Perceptually Tuned** dictionaries (concentrated 50-400Hz) with **Impulse Atoms** for transients. Includes a 256MB LRU cache.
*   **`core/Entropy.cpp`**: Static Huffman coding.
*   **`core/LPC.cpp`**: Linear Predictive Coding analysis and custom filter implementations.

## Build Instructions

**Windows (Powershell):**
```powershell
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

**Linux / macOS:**
```bash
mkdir build
cd build
cmake ..
make -j
```