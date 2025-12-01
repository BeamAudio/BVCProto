#pragma once
#include <cstdint>
#include <string>
#include <stdexcept>

namespace BVC {
    constexpr double PI = 3.14159265358979323846;
    constexpr float PI_F = 3.14159265358979323846f;

    enum class FrameMode : uint8_t {
        SILENCE = 0,
        VOICED = 1,
        UNVOICED = 2
    };

    constexpr char FILE_MAGIC[] = "RBVC";
    constexpr uint8_t FILE_VERSION = 1;

    struct CodecConfig {
        int base_frame_size = 256;
        int overlap_samples = 32;
        int lpc_order = 16;
        int max_merge_frames = 32; // Moderate limit to balance memory and quality
        int default_num_freqs = 768; // Reduced from 1024 but increased from 512 to preserve quality
        int sample_rate = 44100;
        float silence_threshold_rms = 0.002f;

        float lar_max = 10.0f;
        float lar_min = -10.0f;
        int lar_bits = 10;
        float min_db = -100.0f;
        float max_db = 0.0f;
        int energy_bits = 8;
    };

    class BVException : public std::runtime_error {
    public:
        explicit BVException(const std::string& msg) : std::runtime_error("BVC Error: " + msg) {}
    };
}
