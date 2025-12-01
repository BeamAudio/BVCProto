#include "Dictionary.h"
#include "MathUtils.h"
#include <cmath>
#include <algorithm>
#include <iostream>

namespace BVC {
    // Static member init
    std::map<std::tuple<int, int, int>, Dictionary::CacheItem> Dictionary::g_cache;
    std::list<std::tuple<int, int, int>> Dictionary::g_lru_list;
    size_t Dictionary::g_current_memory = 0;
    std::mutex Dictionary::g_mutex;

    void Dictionary::clear_cache() {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_cache.clear();
        g_lru_list.clear();
        g_current_memory = 0;
    }

    std::shared_ptr<DictionaryEntry> Dictionary::get(int N, int fs, int num_freqs, FrameMode mode, bool compute_gram) {
        auto key = std::make_tuple(N, static_cast<int>(mode), num_freqs);
        
        {
            std::lock_guard<std::mutex> lock(g_mutex);
            auto it = g_cache.find(key);
            if (it != g_cache.end()) {
                auto entry = it->second.entry;
                if (compute_gram && !entry->has_gram) {
                    // Miss
                } else {
                    g_lru_list.erase(it->second.lru_it);
                    g_lru_list.push_front(key);
                    it->second.lru_it = g_lru_list.begin();
                    return entry;
                }
            }
        }

        // 2. Generate Dictionary
        auto entry = std::make_shared<DictionaryEntry>();
        entry->length = N;
        
        // Optimization: Cap global atoms to 1024 for speed
        size_t max_atoms = 1024;
        if (mode == FrameMode::UNVOICED) max_atoms = 512; // Reduce unvoiced atoms
        
        entry->D_flat.reserve(max_atoms * N);
        entry->atom_freqs_hz.reserve(max_atoms);

        std::vector<float> t(N);
        MathUtils::vector_fill_t(t);

        std::vector<float> env(N);
        std::vector<float> atom_buf(N);

        auto add_atom = [&](const std::vector<float>& a, float freq_hz) {
            entry->D_flat.insert(entry->D_flat.end(), a.begin(), a.end());
            entry->atom_freqs_hz.push_back(freq_hz);
            entry->n_atoms++;
        };

        if (mode == FrameMode::UNVOICED) {
            std::vector<float> scales = {2.0f, 4.0f, 8.0f};
            int shift_step = 8;
            
            for (float s : scales) {
                for (int u = 0; u < N; u += shift_step) {
                    MathUtils::vector_gaussian_env(t, (float)u, s, env);
                    
                    std::vector<float> freqs_hz;
                    freqs_hz.push_back(0.0f);
                    for(int k=0; k<8; ++k) {
                        float f_start = 2500.0f;
                        float f_end = fs/2.0f - 500.0f;
                        freqs_hz.push_back(f_start + k * (f_end - f_start) / 7.0f); 
                    }

                    for (float f_hz : freqs_hz) {
                        float w = 2.0f * PI_F * f_hz / fs;
                        float norm_sq;
                        MathUtils::vector_cos_mod(t, w, env, atom_buf, norm_sq);
                        
                        if (norm_sq > 1e-6f) {
                            float inv_norm = 1.0f / std::sqrt(norm_sq);
                            for(int i=0; i<N; ++i) atom_buf[i] *= inv_norm;
                            add_atom(atom_buf, f_hz);
                        }
                    }
                }
            }

            // DCT Part
            std::vector<int> k_dct;
            for(int k=1; k<N; k+=2) {
                if (k_dct.size() >= 64) break;
                k_dct.push_back(k);
            }
            
            for (int k : k_dct) {
                float norm_sq = 0.0f;
                for(int n=0; n<N; ++n) {
                    float val = std::cos(PI_F * k * (2.0f * n + 1.0f) / (2.0f * N));
                    atom_buf[n] = val;
                    norm_sq += val * val;
                }

                if (norm_sq > 1e-6f) {
                    float inv_norm = 1.0f / std::sqrt(norm_sq);
                    for(int i=0; i<N; ++i) atom_buf[i] *= inv_norm;
                    float f_hz = (float)k * fs / (2.0f * N);
                    add_atom(atom_buf, f_hz);
                }
            }

        } else if (mode == FrameMode::VOICED) {
            // 1. Impulses (Transients/Phase correction)
            for (int u = 0; u < N; u += 4) {
                std::vector<float> atom(N, 0.0f);
                if (u < N) atom[u] = 1.0f;
                add_atom(atom, 0.0f);
            }

            float min_s = 32.0f;
            int num_scales = 5;
            std::vector<float> scales;
            for(int i=0; i<num_scales; ++i) {
                float s = min_s * std::pow(2.0f, i);
                if(s < N) scales.push_back(s);
            }
            if(scales.empty()) scales.push_back(N/2.0f);

            int shift_step = std::max(64, N/4);
            
            float min_f = 50.0f;
            float max_f = fs / 2.0f - 100.0f;
            int nf = std::max(32, num_freqs / 2); 
            
            std::vector<float> freqs_hz_list;
            // Perceptual Tuning: Concentrate 70% of atoms in 50-400Hz range
            int nf_low = (int)(nf * 0.7f); 
            if (nf_low < 2) nf_low = nf; // Safety
            int nf_high = nf - nf_low;
            float split_f = 400.0f;

            // Low Range (50 - 400 Hz)
            for(int fi=0; fi<nf_low; ++fi) {
                float t = static_cast<float>(fi) / std::max(1, nf_low - 1);
                float hz = min_f * std::pow(split_f/min_f, t);
                freqs_hz_list.push_back(hz);
            }
            
            // High Range (400 - Max Hz)
            if (nf_high > 0) {
                for(int fi=1; fi<=nf_high; ++fi) {
                    float t = static_cast<float>(fi) / nf_high;
                    float hz = split_f * std::pow(max_f/split_f, t);
                    freqs_hz_list.push_back(hz);
                }
            }

            for (float s : scales) {
                for (int u = 0; u < N; u += shift_step) {
                    MathUtils::vector_gaussian_env(t, (float)u, s, env);
                    int idx = std::min(N-1, std::max(0, u));
                    if (env[idx] <= 0.01f) continue;

                    for(float hz : freqs_hz_list) {
                        float w = 2.0f * PI_F * hz / fs;
                        float norm_sq;

                        // Cosine
                        MathUtils::vector_cos_mod(t, w, env, atom_buf, norm_sq);
                        if (norm_sq > 1e-6f) {
                            float inv_norm = 1.0f / std::sqrt(norm_sq);
                            for(int i=0; i<N; ++i) atom_buf[i] *= inv_norm;
                            add_atom(atom_buf, hz);
                        }
                        
                        // Sine
                        MathUtils::vector_sin_mod(t, w, env, atom_buf, norm_sq);
                        if (norm_sq > 1e-6f) {
                            float inv_norm = 1.0f / std::sqrt(norm_sq);
                            for(int i=0; i<N; ++i) atom_buf[i] *= inv_norm;
                            add_atom(atom_buf, hz);
                        }
                    }
                }
            }
        }
        
        // Handle Fallback
        if (entry->n_atoms == 0) {
            entry->D_flat.resize(N, 0.0f);
            entry->atom_freqs_hz.resize(1, 0.0f);
            entry->n_atoms = 1;
        }

        // Decimate if too large
        if (entry->n_atoms > max_atoms) {
            size_t step = entry->n_atoms / max_atoms + 1;
            size_t write_atom_idx = 0;
            
            for(size_t read_atom_idx = 0; read_atom_idx < entry->n_atoms; read_atom_idx += step) {
                if (read_atom_idx != write_atom_idx) {
                    std::copy(
                        entry->D_flat.begin() + read_atom_idx * N,
                        entry->D_flat.begin() + read_atom_idx * N + N,
                        entry->D_flat.begin() + write_atom_idx * N
                    );
                    entry->atom_freqs_hz[write_atom_idx] = entry->atom_freqs_hz[read_atom_idx];
                }
                write_atom_idx++;
            }
            
            entry->n_atoms = write_atom_idx;
            entry->D_flat.resize(entry->n_atoms * N);
            entry->atom_freqs_hz.resize(entry->n_atoms);
        }

        // Gram Matrix
        if (compute_gram && entry->n_atoms <= 1024) {
            entry->has_gram = true;
            entry->G_flat.resize(entry->n_atoms * entry->n_atoms);
            #pragma omp parallel for schedule(dynamic)
            for(int i=0; i < (int)entry->n_atoms; ++i) {
                for(int j=i; j < (int)entry->n_atoms; ++j) {
                    float dot = MathUtils::dot_product(&entry->D_flat[i*N], &entry->D_flat[j*N], N);
                    entry->G_flat[i * entry->n_atoms + j] = dot;
                    entry->G_flat[j * entry->n_atoms + i] = dot;
                }
            }
        } else {
            entry->has_gram = false;
        }

        // 3. Add to Cache
        {
            std::lock_guard<std::mutex> lock(g_mutex);
            
            size_t new_size = entry->D_flat.size() * sizeof(float) 
                            + entry->G_flat.size() * sizeof(float)
                            + entry->atom_freqs_hz.size() * sizeof(float);
            
            while (g_current_memory + new_size > MAX_MEMORY_BYTES && !g_lru_list.empty()) {
                auto last = g_lru_list.back();
                auto it = g_cache.find(last);
                if (it != g_cache.end()) {
                    g_current_memory -= it->second.size_bytes;
                    g_cache.erase(it);
                }
                g_lru_list.pop_back();
            }
            
            if (g_current_memory + new_size <= MAX_MEMORY_BYTES) {
                g_lru_list.push_front(key);
                g_cache[key] = {entry, new_size, g_lru_list.begin()};
                g_current_memory += new_size;
            }
        }
        
        return entry;
    }
}