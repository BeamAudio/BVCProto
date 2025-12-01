# Beam Vocal Codec (BVC)

BVC is a next-generation, low-bitrate parametric speech codec designed for high efficiency and low latency. It utilizes a novel hybrid Matching Pursuit algorithm with a frequency-optimized Gabor dictionary to achieve high-quality speech reconstruction at bitrates as low as 24-48 kbps.

## Key Features

*   **High Quality:** Achieves ~15dB SNR with accurate spectral envelope reconstruction (LSD < 0.05 dB).
*   **Low Bitrate:** Uses Entropy Coding (Huffman) and sparse representation to minimize data size (~48kbps for high fidelity).
*   **Efficiency:** 
    *   **Streaming Architecture:** Constant low memory usage (< 64MB RAM) regardless of file size.
    *   **Fast:** Multithreaded (OpenMP) and Vectorized (SIMD) C++ implementation.
*   **Robustness:** Advanced VAD (Voice Activity Detection) and dynamic frame merging logic.

## Project Structure

*   **`cpp/`**: The production-grade C++ implementation.
    *   `core/`: The codec engine (LPC, Dictionary, Matching Pursuit, Entropy Coding).
    *   `cli/`: The command-line interface source.
*   **`python/`**: The reference Python implementation (for research/prototyping).
*   **`samples/`**: Directory for input/output audio files.
*   **`tools/`**: Analysis scripts (`BVC_Inspector.py`, `BVC_Perceptual.py`) for evaluating codec performance.

## Getting Started

### Building the C++ Codec (Recommended)

**Requirements:** CMake 3.10+, C++17 Compiler (MSVC, GCC, Clang).

1.  Navigate to the `cpp` directory.
2.  Create a build directory: `mkdir build && cd build`
3.  Configure and build:
    ```bash
    cmake ..
    cmake --build . --config Release
    ```
4.  The executable `bvc` (or `bvc.exe`) will be generated in `Release/`.

### Using the CLI

**Encode:**
```bash
./bvc encode input.wav output.bvc
```

**Decode:**
```bash
./bvc decode input.bvc output.wav
```

### Python Reference
To run the Python version (slower, but good for understanding the algorithm):
```bash
cd python
python BVC_CLI.py encode ../samples/input.wav ../samples/out_py.bvc
```

## License
[MIT / Apache 2.0 - Insert License Here]
