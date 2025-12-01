# Beam Vocal Codec (BVC)

BVC is a next-generation, high-fidelity parametric speech codec designed for efficiency and perceptual quality. It utilizes a novel hybrid Matching Pursuit algorithm with a perceptually tuned Gabor dictionary to achieve high-quality speech reconstruction.

## Key Features

*   **High Quality:** Achieves > 10.5 dB SNR with accurate spectral envelope reconstruction (LSD < 0.06 dB).
*   **Bitrate:** High Fidelity mode operates at ~90-100 kbps using Entropy Coding (Huffman) and sparse representation.
*   **Efficiency:** 
    *   **Real-Time Performance:** Optimized C++ implementation runs at **~1.0x RTF (Encoding)** and **~0.5x RTF (Decoding)** on standard CPUs.
    *   **Streaming Architecture:** Uses an optimized LRU Dictionary Cache (< 300MB RAM) for speed.
*   **Perceptual Tuning:** Dictionary atoms are concentrated in the speech power range (50-400Hz) for maximum fidelity.
*   **Robustness:** Advanced VAD (Voice Activity Detection), Pitch-Synchronous Merging, and Direct Form I synthesis filtering.

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
