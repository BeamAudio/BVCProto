#pragma once
#include <vector>
#include "Common.h"

namespace BVC {
    class LPC {
    public:
        // Compute LPC coefficients (reflection coeffs) from signal
        static void compute_lpc(const std::vector<float>& sig, int order, int sample_rate, std::vector<float>& k_out);
        
        // Convert reflection coeffs to LPC coeffs (1, a1, a2...)
        static std::vector<float> rc_to_lpc(const std::vector<float>& k);
        static std::pair<std::vector<float>, std::vector<float>> lfilter(
            const std::vector<float>& b, 
            const std::vector<float>& a, 
            const std::vector<float>& x, 
            const std::vector<float>& zi
        );
        static std::vector<float> synthesis_df1(
            const std::vector<float>& a,
            const std::vector<float>& x,
            std::vector<float>& y_prev_history
        );
    };
}
