#include "Quantizer.h"

namespace BVC {
    Quantizer::Quantizer(const CodecConfig& cfg) : config_(cfg) {
        lar_q_scale_ = static_cast<float>((1 << config_.lar_bits) - 1);
        energy_q_scale_ = static_cast<float>((1 << config_.energy_bits) - 1);
    }

    std::vector<int16_t> Quantizer::quantize_lpc(const std::vector<float>& k_coeffs) {
        std::vector<int16_t> q(k_coeffs.size());
        for (size_t i = 0; i < k_coeffs.size(); ++i) {
            float k = std::max(-0.995f, std::min(0.995f, k_coeffs[i]));
            float lar = std::log((1.0f + k) / (1.0f - k));
            lar = std::max(config_.lar_min, std::min(config_.lar_max, lar));
            float norm = (lar - config_.lar_min) / (config_.lar_max - config_.lar_min);
            q[i] = static_cast<int16_t>(std::round(norm * lar_q_scale_));
        }
        return q;
    }

    std::vector<float> Quantizer::dequantize_lpc(const std::vector<int16_t>& q_indices) {
        std::vector<float> k(q_indices.size());
        for (size_t i = 0; i < q_indices.size(); ++i) {
            float norm = static_cast<float>(q_indices[i]) / lar_q_scale_;
            float lar = norm * (config_.lar_max - config_.lar_min) + config_.lar_min;
            k[i] = std::tanh(lar / 2.0f);
        }
        return k;
    }

    uint8_t Quantizer::quantize_energy(float energy) {
        float db = 10.0f * std::log10(energy + 1e-10f);
        db = std::max(config_.min_db, std::min(config_.max_db, db));
        float norm = (db - config_.min_db) / (config_.max_db - config_.min_db);
        int q = static_cast<int>(std::round(norm * energy_q_scale_));
        return static_cast<uint8_t>(std::max(0, std::min(static_cast<int>(energy_q_scale_), q)));
    }

    float Quantizer::dequantize_energy(uint8_t q) {
        float norm = static_cast<float>(q) / energy_q_scale_;
        float db = norm * (config_.max_db - config_.min_db) + config_.min_db;
        return std::pow(10.0f, db / 10.0f);
    }

    std::pair<float, std::vector<int8_t>> Quantizer::quantize_atom_coeffs_frame(const std::vector<float>& coeffs) {

        if (coeffs.empty()) return {0.0f, {}};

    

        float max_val = 0.0f;

        for (float c : coeffs) max_val = std::max(max_val, std::abs(c));

    

        if (max_val < 1e-9f) return {0.0f, std::vector<int8_t>(coeffs.size(), 0)};

    

        std::vector<int8_t> q_coeffs(coeffs.size());

        for (size_t i = 0; i < coeffs.size(); ++i) {

            float norm = coeffs[i] / max_val;

            q_coeffs[i] = static_cast<int8_t>(std::round(norm * 127.0f));

        }

        return {max_val, q_coeffs};

    }

    uint8_t Quantizer::quantize_gain_val(float gain) {
        if (gain < 1e-9f) return 0;
        float db = 20.0f * std::log10(gain);
        // Range: -60dB to +40dB (Gain 0.001 to 100.0)
        db = std::max(-60.0f, std::min(40.0f, db));
        
        // Map -60..40 to 1..255
        // Range = 100dB
        // Steps = 254
        // Factor = 2.54
        int q = 1 + static_cast<int>((db + 60.0f) * 2.54f);
        return static_cast<uint8_t>(std::max(1, std::min(255, q)));
    }

    float Quantizer::dequantize_gain_val(uint8_t q_gain) {
        if (q_gain == 0) return 0.0f;
        float norm = (float)(q_gain - 1) / 2.54f; // 0..100
        float db = -60.0f + norm;
        return std::pow(10.0f, db / 20.0f);
    }
}
