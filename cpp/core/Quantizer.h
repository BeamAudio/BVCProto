#pragma once
#include "Common.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <utility>
#include <cstdint>

namespace BVC {
    class Quantizer {
    private:
        float lar_q_scale_;
        float energy_q_scale_;

    public:
        explicit Quantizer(const CodecConfig& cfg);

        std::vector<int16_t> quantize_lpc(const std::vector<float>& k_coeffs);
        std::vector<float> dequantize_lpc(const std::vector<int16_t>& q_indices);

        uint8_t quantize_energy(float energy);
        float dequantize_energy(uint8_t q);

        std::pair<float, std::vector<int8_t>> quantize_atom_coeffs_frame(const std::vector<float>& coeffs);
        
        uint8_t quantize_gain_val(float gain);
        float dequantize_gain_val(uint8_t q_gain);

    private:
        CodecConfig config_;
    };
}
