#include "BVCEncoder.h"
#include "LPC.h"
#include "Dictionary.h"
#include "MatchingPursuit.h"
#include "MathUtils.h"
#include <fstream>
#include <iostream>
#include <cmath>
#include <tuple>
#include <vector>
#include <algorithm>
#include <map>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace BVC {
    BVCEncoder::BVCEncoder(CodecConfig cfg) : config_(cfg), quantizer_(cfg) {
        lpc_state_.resize(config_.lpc_order, 0.0f);
    }

    std::tuple<FrameMode, int> BVCEncoder::analyze_frame(const std::vector<float>& sig, int idx, int num_frames) {
        int N = config_.base_frame_size;
        
        auto get_stats = [&](int i) {
            int start = i * N;
            float sum_sq = 0;
            float zcr_cnt = 0;
            float max_val = 0;
            
            // Unrolled loop
            int k = 0;
            for(; k + 4 <= N; k += 4) {
                float s0 = sig[start+k];
                float s1 = sig[start+k+1];
                float s2 = sig[start+k+2];
                float s3 = sig[start+k+3];
                
                sum_sq += s0*s0 + s1*s1 + s2*s2 + s3*s3;
                max_val = std::max({max_val, std::abs(s0), std::abs(s1), std::abs(s2), std::abs(s3)});
                
                if (k>0) {
                    if ((s0 >= 0) != (sig[start+k-1] >= 0)) zcr_cnt++;
                    if ((s1 >= 0) != (s0 >= 0)) zcr_cnt++;
                    if ((s2 >= 0) != (s1 >= 0)) zcr_cnt++;
                    if ((s3 >= 0) != (s2 >= 0)) zcr_cnt++;
                }
            }
            for(; k<N; ++k) {
                float s = sig[start+k];
                sum_sq += s*s;
                max_val = std::max(max_val, std::abs(s));
                if (k>0 && (sig[start+k] >= 0) != (sig[start+k-1] >= 0)) zcr_cnt++;
            }
            
            float rms = std::sqrt(sum_sq / N);
            return std::make_tuple(rms, zcr_cnt/N, max_val/(rms+1e-9f));
        };
        
        auto [rms, zcr, crest] = get_stats(idx);
        FrameMode mode;
        int merge = 1;
        
        if (rms < config_.silence_threshold_rms) {
            mode = FrameMode::SILENCE;
            for(int k=1; k < config_.max_merge_frames && (idx+k) < num_frames; ++k) {
                auto [next_rms, _, __] = get_stats(idx+k);
                if (next_rms < config_.silence_threshold_rms * 2.5f) merge++;
                else break;
            }
        } else {
            if (zcr > 0.35f || crest > 6.0f) mode = FrameMode::UNVOICED;
            else mode = FrameMode::VOICED;
            
            if (mode == FrameMode::VOICED) {
                // Pitch analysis for current frame
                size_t avail_samps = sig.size() - (idx * N);
                float current_pitch = MathUtils::estimate_pitch(&sig[idx*N], avail_samps, config_.sample_rate);

                for(int k=1; k < config_.max_merge_frames && (idx+k) < num_frames; ++k) {
                    auto [next_rms, next_zcr, _] = get_stats(idx+k);
                    float ratio = next_rms / (rms + 1e-9f);
                    
                    bool conditions_met = (ratio > 0.4f && ratio < 2.5f && next_zcr < 0.4f && std::abs(next_zcr - zcr) < 0.2f);
                    
                    if (conditions_met) {
                        // Check Pitch Tolerance (3%)
                        size_t next_avail = sig.size() - ((idx+k) * N);
                        float next_pitch = MathUtils::estimate_pitch(&sig[(idx+k)*N], next_avail, config_.sample_rate);
                        
                        if (current_pitch > 0.0f && next_pitch > 0.0f) {
                             float diff = std::abs(next_pitch - current_pitch);
                             if (diff / current_pitch > 0.03f) {
                                 conditions_met = false; // Pitch deviated too much
                             }
                        } else {
                             // If we can't detect pitch reliably, do not merge to be safe (improve SNR)
                             conditions_met = false;
                        }
                    }

                    if (conditions_met) {
                        merge++;
                    } else {
                        break;
                    }
                }
            }
        }
        return {mode, merge};
    }

    // The heavy lifting worker (MP)
    void BVCEncoder::mp_worker(BoundedQueue<FrameJob>& input_q, BoundedQueue<FrameJob>& output_q) {
        while (true) {
            auto opt_job = input_q.pop();
            if (!opt_job) break; // Queue finished

            FrameJob job = std::move(*opt_job);

            if (job.mode != FrameMode::SILENCE && (job.q_energy > 0 || job.mode == FrameMode::VOICED)) {
                float energy = quantizer_.dequantize_energy(job.q_energy);

                // For voiced frames that were quantized to 0, calculate actual residual energy
                if (job.mode == FrameMode::VOICED && job.q_energy == 0) {
                    energy = MathUtils::dot_product(job.residual.data(), job.residual.data(), job.total_len);
                }

                // Use appropriate threshold
                float threshold = std::max(energy * 0.05f, 1e-8f); // Higher threshold for efficiency but not zero
                int max_atoms = (job.mode == FrameMode::UNVOICED) ? 128 : (80 * std::min(job.merge_count, 4)); // Use reasonable limit

                // Get dictionary
                auto dict = Dictionary::get(job.total_len, config_.sample_rate, config_.default_num_freqs, job.mode);

                job.atoms = MP::matching_pursuit(job.residual, dict, max_atoms, threshold);

                if (!job.atoms.empty()) {
                    std::vector<float> coeffs;
                    coeffs.reserve(job.atoms.size());
                    for(auto& a : job.atoms) coeffs.push_back(a.value);
                    auto result = quantizer_.quantize_atom_coeffs_frame(coeffs);
                    job.atom_gain = result.first;
                    job.q_atom_coeffs = std::move(result.second);
                }
            }

            // Clear heavy data before sending to writer to reduce memory usage
            std::vector<float>().swap(job.residual);

            output_q.push(std::move(job));
        }
    }

    void BVCEncoder::encode_file(const std::string& input_wav, const std::string& output_bvc) {
        std::ifstream wav(input_wav, std::ios::binary);
        if (!wav) throw BVException("Cannot open input file");
        
        wav.seekg(0, std::ios::end);
        size_t file_size = wav.tellg();
        wav.seekg(44, std::ios::beg); 
        
        size_t num_samples_total = (file_size - 44) / 2;
        int N = config_.base_frame_size;
        int overlap = config_.overlap_samples;
        
        // Calculate required buffer size based on max merge frames but with reasonable limits
        int max_frame_samples = config_.max_merge_frames * N + overlap;
        int buffer_capacity = std::min(max_frame_samples * 2, 32768); // Use up to 32k samples max for buffer

        std::vector<float> buffer;
        buffer.reserve(buffer_capacity);
        
        // Pre-emphasis state
        float pre_emp_state = 0.0f;
        
        // Streaming Pipeline Setup - Reduce queue sizes to limit memory usage
        BoundedQueue<FrameJob> job_queue(16);  // Smaller queue to limit memory
        BoundedQueue<FrameJob> write_queue(16);  // Smaller queue to limit memory

        // Limit number of threads to reduce memory overhead
        unsigned int n_threads = std::min(std::thread::hardware_concurrency(), 4U); // Cap at 4 threads
        if (n_threads == 0) n_threads = 1;  // Minimum 1 thread
        std::vector<std::thread> workers;
        for(unsigned int t=0; t<n_threads; ++t) {
            workers.emplace_back(&BVCEncoder::mp_worker, this, std::ref(job_queue), std::ref(write_queue));
        }
        
        std::thread writer_thread([&]() {
            std::ofstream out(output_bvc, std::ios::binary);
            out.write(FILE_MAGIC, 4);
            out.write((const char*)&FILE_VERSION, 1);
            uint32_t fs_u32 = config_.sample_rate;
            uint32_t orig_len = (uint32_t)num_samples_total;

            std::streampos header_pos = out.tellp();
            uint32_t frame_count_placeholder = 0;
            out.write((char*)&fs_u32, 4);
            out.write((char*)&frame_count_placeholder, 4);
            out.write((char*)&orig_len, 4);

            // Use a smaller max size for pending frames to limit memory
            std::map<uint32_t, FrameJob> pending_frames;
            uint32_t next_frame_idx = 0;
            uint32_t total_frames_written = 0;

            while (true) {
                // Process any ready frames first
                while (pending_frames.count(next_frame_idx)) {
                    auto it = pending_frames.find(next_frame_idx);
                    const auto& job = it->second;

                    uint8_t flags = (static_cast<uint8_t>(job.mode) << 6) | (job.merge_count & 0x3F);
                    out.write((char*)&flags, 1);
                    out.write((char*)&job.q_energy, 1);
                    uint16_t n_atoms = (uint16_t)job.atoms.size();
                    out.write((char*)&n_atoms, 2);
                    out.write((char*)job.q_lar.data(), config_.lpc_order * sizeof(int16_t));

                    if (n_atoms > 0) {
                        out.write((char*)&job.atom_gain, 4);

                        // Huffman Encode Atoms
                        BitWriter bw;
                        for(size_t k=0; k<job.atoms.size(); ++k) {
                            entropy_.encode_atom_index(bw, job.atoms[k].idx);
                            entropy_.encode_coefficient(bw, job.q_atom_coeffs[k]);
                        }
                        bw.flush();

                        uint16_t huff_bytes = (uint16_t)bw.data().size();
                        out.write((char*)&huff_bytes, 2);
                        bw.dump_to(out);
                    }

                    pending_frames.erase(it);
                    next_frame_idx++;
                    total_frames_written++;
                    if (total_frames_written % 50 == 0) std::cout << "\rFrames: " << total_frames_written << " " << std::flush;
                }

                // Check for new jobs with a timeout to avoid blocking indefinitely
                auto opt_job = write_queue.pop();
                if (!opt_job) break; // Queue closed means all jobs are done

                if (opt_job->frame_index == next_frame_idx) {
                    // Process immediately if it's the next expected frame
                    const auto& job = *opt_job;

                    uint8_t flags = (static_cast<uint8_t>(job.mode) << 6) | (job.merge_count & 0x3F);
                    out.write((char*)&flags, 1);
                    out.write((char*)&job.q_energy, 1);
                    uint16_t n_atoms = (uint16_t)job.atoms.size();
                    out.write((char*)&n_atoms, 2);
                    out.write((char*)job.q_lar.data(), config_.lpc_order * sizeof(int16_t));

                    if (n_atoms > 0) {
                        out.write((char*)&job.atom_gain, 4);

                        // Huffman Encode Atoms
                        BitWriter bw;
                        for(size_t k=0; k<job.atoms.size(); ++k) {
                            entropy_.encode_atom_index(bw, job.atoms[k].idx);
                            
                            entropy_.encode_coefficient(bw, job.q_atom_coeffs[k]);
                        }
                        bw.flush();

                        uint16_t huff_bytes = (uint16_t)bw.data().size();
                        out.write((char*)&huff_bytes, 2);
                        bw.dump_to(out);
                    }

                    next_frame_idx++;
                    total_frames_written++;
                    if (total_frames_written % 50 == 0) std::cout << "\rFrames: " << total_frames_written << " " << std::flush;
                } else {
                    // Store non-sequential frames in the pending map
                    pending_frames[opt_job->frame_index] = std::move(*opt_job);
                    // Limit pending frames to prevent memory buildup
                    if (pending_frames.size() > 100) { // Increase limit to ensure all frames are processed
                        std::this_thread::sleep_for(std::chrono::milliseconds(1)); // Brief pause
                    }
                }
            }

            // After main loop exits, write any remaining frames in sequence
            while (!pending_frames.empty()) {
                if (pending_frames.count(next_frame_idx)) {
                    auto it = pending_frames.find(next_frame_idx);
                    const auto& job = it->second;

                    uint8_t flags = (static_cast<uint8_t>(job.mode) << 6) | (job.merge_count & 0x3F);
                    out.write((char*)&flags, 1);
                    out.write((char*)&job.q_energy, 1);
                    uint16_t n_atoms = (uint16_t)job.atoms.size();
                    out.write((char*)&n_atoms, 2);
                    out.write((char*)job.q_lar.data(), config_.lpc_order * sizeof(int16_t));

                    if (n_atoms > 0) {
                        out.write((char*)&job.atom_gain, 4);

                        // Huffman Encode Atoms
                        BitWriter bw;
                        for(size_t k=0; k<job.atoms.size(); ++k) {
                            entropy_.encode_atom_index(bw, job.atoms[k].idx);
                            
                            entropy_.encode_coefficient(bw, job.q_atom_coeffs[k]);
                        }
                        bw.flush();

                        uint16_t huff_bytes = (uint16_t)bw.data().size();
                        out.write((char*)&huff_bytes, 2);
                        bw.dump_to(out);
                    }

                    pending_frames.erase(it);
                    next_frame_idx++;
                    total_frames_written++;
                    if (total_frames_written % 50 == 0) std::cout << "\rFinal Frames: " << total_frames_written << " " << std::flush;
                } else {
                    // If the next expected frame isn't available, increment to check again
                    next_frame_idx++;
                }
            }

            out.seekp(header_pos);
            out.write((char*)&fs_u32, 4);
            out.write((char*)&total_frames_written, 4);
            std::cout << "\nDone. Total Blocks: " << total_frames_written << std::endl;
        });
        
        // PRODUCER LOOP
        std::cout << "Encoding Stream..." << std::endl;
        uint32_t frame_counter = 0;
        size_t samples_processed_total = 0;
        bool eof_reached = false;
        
        // Read chunk buffer - smaller buffer to reduce memory usage
        std::vector<int16_t> pcm_chunk(2048); 
        
        while (samples_processed_total < num_samples_total || !buffer.empty()) {
            // 1. Top up buffer
            while (buffer.size() < buffer_capacity && !eof_reached) {
                size_t needed = buffer_capacity - buffer.size();
                size_t to_read = std::min(needed, pcm_chunk.size());
                
                // Check file remaining
                size_t file_rem = num_samples_total - (static_cast<size_t>(wav.tellg()) - 44)/2;
                if (file_rem == 0) {
                    eof_reached = true;
                    // Padding at end
                    size_t pad_needed = (N - (num_samples_total % N)) % N;
                    buffer.insert(buffer.end(), pad_needed + overlap, 0.0f);
                    break;
                }
                to_read = std::min(to_read, file_rem);
                
                wav.read(reinterpret_cast<char*>(pcm_chunk.data()), to_read * 2);
                size_t read_count = wav.gcount() / 2;
                
                if (read_count == 0) {
                    eof_reached = true;
                    break;
                }
                
                // Pre-emphasis on the fly
                for(size_t k=0; k<read_count; ++k) {
                    float s = pcm_chunk[k] / 32768.0f;
                    float filtered = s - 0.97f * pre_emp_state;
                    pre_emp_state = s;
                    buffer.push_back(filtered);
                }
            }
            
            // Check if we have enough data for at least one frame analysis
            // Minimal needed is N samples to analyze 1 frame.
            if (buffer.empty()) break;
            
            // If we are at EOF, we might process whatever is left.
            // analyze_frame expects buffer to have data starting at 'idx'. 
            // We will always treat buffer[0] as start of current frame.
            
            // 2. Analyze & Decide Merge
            // We need to know how many frames are available in buffer to merge.
            // Max merge is bounded by config and buffer size.
            
            int frames_in_buffer = (int)(buffer.size() / N); // Approximate available frames
            if (frames_in_buffer < 1) break; // Should only happen at very end
            
            auto [mode, merge_count] = analyze_frame(buffer, 0, frames_in_buffer);
            
            int core_len = merge_count * N;
            int needed_len = core_len + overlap;
            
            // If we don't have enough data for the desired merge, reduce merge count
            if (buffer.size() < needed_len) {
                // Reduce merge count to fit available data
                int available_merge = (int)(buffer.size() - overlap) / N;
                if (available_merge > 0) {
                    merge_count = std::min(merge_count, available_merge);
                    core_len = merge_count * N;
                    needed_len = core_len + overlap;
                } else {
                    // If not enough for even one frame, we reached end of file
                    if (eof_reached && buffer.size() < N) break;
                    // Otherwise try to get more data in the next iteration
                    continue;
                }
            }
            
            // 3. Create Job
            FrameJob job;
            job.frame_index = frame_counter++;
            job.mode = mode;
            job.merge_count = merge_count;
            job.total_len = needed_len;

            // Copy only the needed data
            std::vector<float> raw_audio(buffer.begin(), buffer.begin() + needed_len);
            std::vector<float> audio_core(raw_audio.begin(), raw_audio.begin() + core_len);

            // 4. LPC & Filter
            std::vector<float> k_raw;
            LPC::compute_lpc(audio_core, config_.lpc_order, config_.sample_rate, k_raw);

            job.q_lar = quantizer_.quantize_lpc(k_raw);
            auto k_quant = quantizer_.dequantize_lpc(job.q_lar);
            auto a_quant = LPC::rc_to_lpc(k_quant);

            std::vector<float> input_core(raw_audio.begin(), raw_audio.begin() + core_len);
            auto [res_core, state_after_core] = LPC::lfilter(a_quant, {1.0f}, input_core, lpc_state_);

            std::vector<float> input_ovl(raw_audio.begin() + core_len, raw_audio.end());
            auto [res_ovl, _] = LPC::lfilter(a_quant, {1.0f}, input_ovl, state_after_core);

            lpc_state_ = state_after_core;

            job.residual = std::move(res_core);
            job.residual.insert(job.residual.end(), res_ovl.begin(), res_ovl.end());

            // Apply windowing
            for(int k=0; k<overlap; ++k) {
                float t = (float)k;
                float w_fade_in = (frame_counter==1) ? 1.0f : std::pow(std::sin(0.5f * PI_F * t / overlap), 2.0f);
                float w_fade_out = std::pow(std::cos(0.5f * PI_F * t / overlap), 2.0f);

                if (frame_counter > 1 && k < overlap) job.residual[k] *= w_fade_in;
                job.residual[job.total_len - overlap + k] *= w_fade_out;
            }

            float res_energy = MathUtils::dot_product(job.residual.data(), job.residual.data(), job.total_len);
            job.q_energy = quantizer_.quantize_energy(res_energy);

            // Clear raw_audio to free memory before pushing job
            std::vector<float>().swap(audio_core);
            std::vector<float>().swap(input_core);
            std::vector<float>().swap(input_ovl);
            std::vector<float>().swap(raw_audio);

            job_queue.push(std::move(job));
            
            // 5. Advance Buffer
            // Remove core_len samples from front. Overlap samples remain for next frame?
            // NO. In raw audio streaming, the next frame starts after `core_len`.
            // The `overlap` samples needed for THIS frame are the start of the NEXT frame.
            // So we strictly remove `core_len` samples.
            
            if (core_len <= buffer.size()) {
                buffer.erase(buffer.begin(), buffer.begin() + core_len);
            } else {
                buffer.clear();
            }
            samples_processed_total += core_len;
        }
        
        job_queue.finish();
        for(auto& t : workers) if(t.joinable()) t.join();
        write_queue.finish();
        if(writer_thread.joinable()) writer_thread.join();
    }
}
