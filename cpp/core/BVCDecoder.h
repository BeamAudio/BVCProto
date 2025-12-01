#pragma once
#include "Common.h"
#include "Quantizer.h"
#include "Entropy.h"
#include <vector>

namespace BVC {
    class BVCDecoder {
    private:
        CodecConfig config_;
        Quantizer quantizer_;
        EntropyCoder entropy_;
        std::vector<float> y_prev_history_; // History for DF-I synthesis
        std::vector<float> overlap_buf_;
        
    public:
        explicit BVCDecoder(CodecConfig cfg);
        void decode_file(const std::string& input_bvc, const std::string& output_wav);
        
    private:
        std::vector<float> apply_post_filter(const std::vector<float>& input, const std::vector<float>& lpc_coeffs);
    };
}
