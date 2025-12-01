#pragma once
#include "Common.h"
#include "Quantizer.h"
#include "MatchingPursuit.h"
#include "Pipeline.h"
#include "Entropy.h"
#include <vector>
#include <thread>
#include <atomic>

namespace BVC {
    struct FrameJob {
        uint32_t frame_index;
        FrameMode mode;
        int merge_count;
        std::vector<float> residual;
        int total_len;
        std::vector<int16_t> q_lar;
        uint8_t q_energy;
        std::vector<MP::Atom> atoms;
        float atom_gain;
        std::vector<int8_t> q_atom_coeffs;
    };

    class BVCEncoder {
    private:
        CodecConfig config_;
        Quantizer quantizer_;
        EntropyCoder entropy_;
        std::vector<float> lpc_state_;
        
    public:
        explicit BVCEncoder(CodecConfig cfg);
        void encode_file(const std::string& input_wav, const std::string& output_bvc);
        
    private:
        std::tuple<FrameMode, int> analyze_frame(const std::vector<float>& sig, int start_idx, int total_frames);
        
        // Worker function for MP threads
        void mp_worker(BoundedQueue<FrameJob>& input_q, BoundedQueue<FrameJob>& output_q);
    };
}
