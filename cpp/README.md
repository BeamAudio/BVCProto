# BVC - C++ Implementation

This directory contains the optimized, high-performance implementation of the Beam Vocal Codec.

## Architecture

The codec is built as a **Streaming Producer-Consumer Pipeline**:
1.  **Producer (Main Thread):** Reads audio in chunks, performs Serial Analysis (LPC, Filtering), and pushes jobs to a queue.
2.  **Workers (Thread Pool):** Pop jobs and perform the heavy Matching Pursuit algorithm in parallel.
3.  **Consumer (Writer Thread):** Re-orders finished frames and writes the Entropy-coded bitstream to disk.

## Source Layout

*   **`core/BVCEncoder.cpp`**: Main encoding pipeline and logic.
*   **`core/BVCDecoder.cpp`**: Decoding logic with Post-Filtering.
*   **`core/MatchingPursuit.cpp`**: The sparse approximation engine. Optimized with manual loop unrolling and robust solvers.
*   **`core/Dictionary.cpp`**: Generates Gabor/DCT dictionaries on-the-fly with an LRU cache to save memory.
*   **`core/Entropy.cpp`**: Static Huffman coding for lossless compression of atoms.
*   **`core/LPC.cpp`**: Linear Predictive Coding analysis and filtering (FIR/IIR).

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