#include "LPC.h"
#include "MathUtils.h"
#include <cmath>
#include <algorithm>
#include <iostream>

namespace BVC {
    void LPC::compute_lpc(const std::vector<float>& sig, int order, int sample_rate, std::vector<float>& k_out) {
        int n = sig.size();
        std::vector<float> r(order + 1);
        
        auto w = MathUtils::get_hann_window(n);
        std::vector<float> xw(n);
        for(int i=0; i<n; ++i) xw[i] = sig[i] * w[i];
        
        float gamma_base = std::exp(-0.5f * (2.0f * PI_F * 60.0f / (float)sample_rate)); 
        
        for(int l=0; l<=order; ++l) {
            double acc = 0.0;
            for(int i=0; i < n - l; ++i) acc += xw[i] * xw[i+l];
            r[l] = static_cast<float>(acc) * std::pow(gamma_base, (float)l);
        }
        
        std::vector<float> a(order + 1, 0.0f);
        a[0] = 1.0f;
        k_out.resize(order);
        double e = r[0] + 1e-12;
        
        for(int i=0; i<order; ++i) {
            double acc = 0.0;
            for(int j=0; j<=i; ++j) acc += a[j] * r[i+1-j];
            
            float ki = static_cast<float>(-acc / e);
            if (ki > 0.99f) ki = 0.99f;
            if (ki < -0.99f) ki = -0.99f;
            k_out[i] = ki;
            
            std::vector<float> a_new = a;
            for(int j=0; j<=i; ++j) {
                a_new[j+1] = a[j+1] + ki * a[i-j];
            }
            a = a_new;
            e *= (1.0 - ki*ki);
        }
    }

    std::vector<float> LPC::rc_to_lpc(const std::vector<float>& k) {
        int order = k.size();
        std::vector<float> a(order + 1, 0.0f);
        a[0] = 1.0f;
        for(int i=0; i<order; ++i) {
            float ki = k[i];
            std::vector<float> a_prev = a;
            for(int j=0; j<=i; ++j) {
                a[j+1] = a_prev[j+1] + ki * a_prev[i-j];
            }
        }
        return a;
    }

    std::vector<float> LPC::synthesis_df1(
        const std::vector<float>& a,
        const std::vector<float>& x,
        std::vector<float>& y_prev_history
    ) {
        int n_x = x.size();
        int order = a.size() - 1; // a[0] is 1.0
        
        std::vector<float> y(n_x);
        
        for (int n = 0; n < n_x; ++n) {
            float val = x[n]; // Assume b[0] = 1.0
            
            for (int k = 1; k <= order; ++k) {
                float prev_y;
                if (n - k >= 0) {
                    prev_y = y[n - k];
                } else {
                    // Access from y_prev_history. 
                    // y_prev_history[0] corresponds to y[-order], y_prev_history[order-1] corresponds to y[-1]
                    // Index mapping: n-k (negative) -> order + (n-k) (0 to order-1)
                    // Example: n=0, k=1 => index order - 1 (y[-1])
                    // Example: n=0, k=order => index 0 (y[-order])
                    prev_y = y_prev_history[order + (n - k)];
                }
                val -= a[k] * prev_y;
            }
            y[n] = val;
        }
        return y;
    }

    std::pair<std::vector<float>, std::vector<float>> LPC::lfilter(
        const std::vector<float>& b, 
        const std::vector<float>& a, 
        const std::vector<float>& x, 
        const std::vector<float>& zi
    ) {
        // Standard Difference Equation Implementation
        // a[0]*y[n] = b[0]*x[n] + b[1]*x[n-1] + ... - a[1]*y[n-1] - ...
        // Usually a[0] = 1.
        // zi stores the 'delay' values.
        // For filter order M, zi has size M.
        // zi[0] corresponds to the term for lag 1 (x[n-1], y[n-1]), zi[1] for lag 2...
        
        size_t n_in = x.size();
        size_t order = std::max(a.size(), b.size()) - 1;
        
        // Normalize coefficients if a[0] != 1
        float a0 = a[0];
        std::vector<float> b_norm = b;
        std::vector<float> a_norm = a;
        if (std::abs(a0 - 1.0f) > 1e-9) {
             for(auto& v : b_norm) v /= a0;
             for(auto& v : a_norm) v /= a0;
        }
        
        // Pad to match order
        while(b_norm.size() <= order) b_norm.push_back(0.0f);
        while(a_norm.size() <= order) a_norm.push_back(0.0f);

        std::vector<float> y(n_in);
        std::vector<float> state = zi; 
        if(state.size() < order) state.resize(order, 0.0f);
        
        for(size_t n=0; n<n_in; ++n) {
            // Calculate output
            // The state[0] holds the accumulated history terms for the current step
            float val = b_norm[0] * x[n] + state[0];
            y[n] = val;
            
            // Update state (shift and add new terms)
            // state[k] = b[k+1]*x[n] - a[k+1]*y[n] + state[k+1]
            for(size_t k=0; k<order-1; ++k) {
                state[k] = b_norm[k+1] * x[n] - a_norm[k+1] * y[n] + state[k+1];
            }
            // Last state element
            state[order-1] = b_norm[order] * x[n] - a_norm[order] * y[n];
        }
        
        return {y, state};
    }
}