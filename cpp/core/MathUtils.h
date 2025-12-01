#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include "Common.h"

// Check for OpenMP
#ifdef _OPENMP
#include <omp.h>
#endif

namespace BVC {
    class MathUtils {
    public:
        static float dot_product(const float* a, const float* b, size_t n) {
            float sum = 0.0f;
            size_t i = 0;
            // Unroll 8x for better pipelining
            for (; i + 8 <= n; i += 8) {
                sum += a[i] * b[i] 
                     + a[i+1] * b[i+1] 
                     + a[i+2] * b[i+2] 
                     + a[i+3] * b[i+3]
                     + a[i+4] * b[i+4] 
                     + a[i+5] * b[i+5] 
                     + a[i+6] * b[i+6] 
                     + a[i+7] * b[i+7];
            }
            for (; i < n; ++i) sum += a[i] * b[i];
            return sum;
        }

        static std::vector<float> get_hann_window(int N) {
            std::vector<float> w(N);
            const float factor = 2.0f * BVC::PI_F / (N - 1);
            for (int i = 0; i < N; ++i) {
                w[i] = 0.5f * (1.0f - std::cos(factor * i));
            }
            return w;
        }

        // Vectorized generation helpers to help compiler auto-vectorization
        static void vector_fill_t(std::vector<float>& t) {
            for(size_t i=0; i<t.size(); ++i) t[i] = static_cast<float>(i);
        }

        static void vector_gaussian_env(const std::vector<float>& t, float u, float s, std::vector<float>& out) {
            const float inv_s = 1.0f / s;
            const float neg_half = -0.5f;
            size_t n = t.size();
            for(size_t i=0; i<n; ++i) {
                float val = (t[i] - u) * inv_s;
                out[i] = std::exp(neg_half * val * val);
            }
        }

        static void vector_cos_mod(const std::vector<float>& t, float w, const std::vector<float>& env, std::vector<float>& out, float& norm_sq) {
            size_t n = t.size();
            float sum = 0.0f;
            
            // Pre-calculate constants for recurrence if needed, but standard cos is safer for precision.
            // Rely on compiler auto-vectorization with fast math.
            // MSVC Pragma to ignore aliasing
            #pragma loop(ivdep)
            for(size_t i=0; i<n; ++i) {
                float val = env[i] * std::cos(w * t[i]);
                out[i] = val;
                sum += val * val;
            }
            norm_sq = sum;
        }

        static void vector_sin_mod(const std::vector<float>& t, float w, const std::vector<float>& env, std::vector<float>& out, float& norm_sq) {
            size_t n = t.size();
            float sum = 0.0f;
            
            #pragma loop(ivdep)
            for(size_t i=0; i<n; ++i) {
                float val = env[i] * std::sin(w * t[i]);
                out[i] = val;
                sum += val * val;
            }
            norm_sq = sum;
        }

        static float estimate_pitch(const float* sig, size_t len, int sample_rate) {
            const float min_freq = 60.0f;
            const float max_freq = 600.0f;
            
            int min_lag = sample_rate / (int)max_freq;
            int max_lag = sample_rate / (int)min_freq;
            
            if (len < (size_t)max_lag + min_lag) {
                max_lag = (int)len / 2;
                if (max_lag < min_lag) return 0.0f;
            }

            float max_corr = 0.0f;
            int best_lag = 0;

            // Calculate energy of the stationary part (approximation)
            // We strictly need normalized cross-correlation at lag k:
            // R(k) = sum(x[i] * x[i+k]) / sqrt(sum(x[i]^2) * sum(x[i+k]^2))
            
            // Optimization: Calculate base energy once?
            // The window shifts, so energy terms change.
            // For efficiency in this loop, we can approximate or just compute explicitly.
            // Let's compute explicitly but efficiently.
            
            // Limit search range to reasonable window (e.g. correlation over 512 samples)
            // to keep it fast.
            int corr_len = std::min((int)len - max_lag, 512);

            float signal_energy = 0.0f;
            for(int i=0; i<corr_len; ++i) signal_energy += sig[i] * sig[i];
            
            if (signal_energy < 1e-9f) return 0.0f; // Silence

            for (int lag = min_lag; lag <= max_lag; ++lag) {
                float dot = 0.0f;
                float lag_energy = 0.0f;
                
                // Vectorizable loop
                for (int i = 0; i < corr_len; ++i) {
                    float s1 = sig[i];
                    float s2 = sig[i + lag];
                    dot += s1 * s2;
                    lag_energy += s2 * s2;
                }

                float norm = std::sqrt(signal_energy * lag_energy);
                float correlation = dot / (norm + 1e-9f);

                if (correlation > max_corr) {
                    max_corr = correlation;
                    best_lag = lag;
                }
            }

            if (max_corr > 0.4f) { // Threshold for voicing
                return (float)sample_rate / best_lag;
            }
            
            return 0.0f;
        }
    };
}