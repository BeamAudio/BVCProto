#include "BVCDecoder.h"
#include "LPC.h"
#include "Dictionary.h"
#include "MatchingPursuit.h"
#include "MathUtils.h"
#include <fstream>
#include <iostream>
#include <cstring>

namespace BVC {
    BVCDecoder::BVCDecoder(CodecConfig cfg) : config_(cfg), quantizer_(cfg) {
        y_prev_history_.resize(config_.lpc_order, 0.0f);
        overlap_buf_.resize(config_.overlap_samples, 0.0f);
    }

    void BVCDecoder::decode_file(const std::string& input_bvc, const std::string& output_wav) {
        std::ifstream in(input_bvc, std::ios::binary);
        if (!in) throw BVException("Cannot open BVC file");
        
        char magic[4];
        in.read(magic, 4);
        if(std::strncmp(magic, FILE_MAGIC, 4) != 0) throw BVException("Invalid Magic");
        
        uint8_t ver;
        in.read((char*)&ver, 1);
        
        uint32_t fs, num_frames, orig_len;
        in.read((char*)&fs, 4);
        in.read((char*)&num_frames, 4);
        in.read((char*)&orig_len, 4);
        
        config_.sample_rate = fs;
        
        struct DecodedFrame {
            FrameMode mode;
            int merge;
            std::vector<int16_t> q_lar;
            std::vector<MP::Atom> atoms;
            std::vector<float> excitation; // To be computed in parallel
        };
        
        std::vector<DecodedFrame> frames(num_frames);
        
        // 1. Parse File (Sequential)
        for(uint32_t i=0; i<num_frames; ++i) {
            DecodedFrame& df = frames[i];
            uint8_t flags, q_energy;
            uint16_t n_atoms;
            df.q_lar.resize(config_.lpc_order);
            
            in.read((char*)&flags, 1);
            in.read((char*)&q_energy, 1);
            in.read((char*)&n_atoms, 2);
            in.read((char*)df.q_lar.data(), config_.lpc_order * sizeof(int16_t));
            
            df.mode = static_cast<FrameMode>(flags >> 6);
            df.merge = flags & 0x3F;
            if(df.merge==0) df.merge=1;
            
            if (n_atoms > 0) {
                float gain;
                in.read((char*)&gain, 4);
                
                uint16_t huff_bytes;
                in.read((char*)&huff_bytes, 2);
                
                std::vector<uint8_t> huff_data(huff_bytes);
                in.read((char*)huff_data.data(), huff_bytes);
                
                BitReader br(huff_data.data(), huff_bytes);
                df.atoms.reserve(n_atoms);
                
                for(int k=0; k<n_atoms; ++k) {
                    uint16_t idx = entropy_.decode_atom_index(br);
                    
                    int8_t qc = entropy_.decode_coefficient(br);
                    float val = (float)qc * (gain / 127.0f);
                    df.atoms.push_back({idx, val});
                }
            }
        }
        
        // 2. Generate Excitation (Parallel)
        #pragma omp parallel for schedule(dynamic)
        for(int i=0; i<(int)num_frames; ++i) {
            DecodedFrame& df = frames[i];
            int core_len = df.merge * config_.base_frame_size;
            int total_len = core_len + config_.overlap_samples;
            
            df.excitation.assign(total_len, 0.0f);
            
            if (!df.atoms.empty()) {
                // No Gram needed for reconstruction
                auto dict = Dictionary::get(total_len, fs, config_.default_num_freqs, df.mode, false);
                for(const auto& atom : df.atoms) {
                    if (atom.idx < dict->n_atoms) {
                         const float* atom_data = &dict->D_flat[atom.idx * total_len];
                         // Vectorize this addition?
                         for(int s=0; s<total_len; ++s) df.excitation[s] += atom.value * atom_data[s];
                    }
                }
            }
        }
        
        // 3. Synthesis Filter & Overlap-Add (Sequential)
        std::vector<float> output_buffer;
        output_buffer.reserve(orig_len > 0 ? orig_len : num_frames * 256);
        
        for(uint32_t i=0; i<num_frames; ++i) {
            DecodedFrame& df = frames[i];
            int core_len = df.merge * config_.base_frame_size;
            
            // Overlap Add Input
            for(int s=0; s<config_.overlap_samples; ++s) {
                df.excitation[s] += overlap_buf_[s];
            }
            for(int s=0; s<config_.overlap_samples; ++s) {
                overlap_buf_[s] = df.excitation[core_len + s];
            }
            
            auto k_quant = quantizer_.dequantize_lpc(df.q_lar);
            auto a_quant = LPC::rc_to_lpc(k_quant);
            
            // Filter only valid part
            std::vector<float> valid_res(df.excitation.begin(), df.excitation.begin() + core_len);
            std::vector<float> synth = LPC::synthesis_df1(a_quant, valid_res, y_prev_history_);
            
            // Update history
            if (synth.size() >= config_.lpc_order) {
                std::copy(synth.end() - config_.lpc_order, synth.end(), y_prev_history_.begin());
            } else {
                int remaining = config_.lpc_order - (int)synth.size();
                for(int j=0; j<remaining; ++j) y_prev_history_[j] = y_prev_history_[j + synth.size()];
                for(size_t j=0; j<synth.size(); ++j) y_prev_history_[remaining + j] = synth[j];
            }
            
            output_buffer.insert(output_buffer.end(), synth.begin(), synth.end());
            
            // Free memory eagerly
            std::vector<float>().swap(df.excitation);
            std::vector<MP::Atom>().swap(df.atoms);
            
            if (i % 100 == 0) std::cout << "\rDecoded: " << (i * 100 / num_frames) << "%" << std::flush;
        }
        
        // De-emphasis
        if (!output_buffer.empty()) {
            float last = 0.0f;
            for(size_t s=0; s<output_buffer.size(); ++s) {
                output_buffer[s] = output_buffer[s] + 0.97f * last;
                last = output_buffer[s];
            }
        }
        
        // Write WAV
        std::ofstream wav(output_wav, std::ios::binary);
        wav.write("RIFF", 4);
        uint32_t total_bytes = output_buffer.size() * 2;
        uint32_t wav_size = 36 + total_bytes;
        wav.write((char*)&wav_size, 4);
        wav.write("WAVE", 4);
        wav.write("fmt ", 4);
        uint32_t fmt_len = 16;
        wav.write((char*)&fmt_len, 4);
        uint16_t fmt = 1, ch = 1;
        wav.write((char*)&fmt, 2);
        wav.write((char*)&ch, 2);
        wav.write((char*)&fs, 4);
        uint32_t br = fs * 2;
        wav.write((char*)&br, 4);
        uint16_t ba = 2, bits = 16;
        wav.write((char*)&ba, 2);
        wav.write((char*)&bits, 2);
        wav.write("data", 4);
        wav.write((char*)&total_bytes, 4);
        
        std::vector<int16_t> final_pcm(output_buffer.size());
        for(size_t s=0; s<output_buffer.size(); ++s) {
            float v = std::max(-1.0f, std::min(1.0f, output_buffer[s]));
            final_pcm[s] = (int16_t)(v * 32767.0f);
        }
        wav.write((char*)final_pcm.data(), final_pcm.size()*2);
        
        std::cout << "\nDecoding Done." << std::endl;
    }
}
