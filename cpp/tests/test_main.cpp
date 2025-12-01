#include "TestUtils.h"
#include "../core/LPC.h"
#include "../core/Quantizer.h"
#include "../core/BVCEncoder.h"
#include <vector>
#include <iostream>

void test_quantizer() {
    std::cout << "--- Test Quantizer ---" << std::endl;
    BVC::CodecConfig cfg;
    BVC::Quantizer q(cfg);

    // Test Energy
    float e = 0.001f; // -30dB
    uint8_t qe = q.quantize_energy(e);
    float de = q.dequantize_energy(qe);
    // -30dB maps to something. Check roundtrip error.
    // 8 bits over 100dB range -> ~0.4dB resolution.
    // 10log10(de/e) should be small.
    float err_db = std::abs(10*std::log10(de/e));
    TestUtils::assert_true(err_db < 0.5f, "Energy Quantization Roundtrip");

    // Test LAR
    std::vector<float> k = {0.5f, -0.8f, 0.0f};
    auto qk = q.quantize_lpc(k);
    auto dk = q.dequantize_lpc(qk);
    for(size_t i=0; i<k.size(); ++i) {
        TestUtils::assert_close(k[i], dk[i], 0.05f, "LAR Quantization Roundtrip index " + std::to_string(i));
    }
}

void test_filter_logic() {
    std::cout << "--- Test Filter Logic (lfilter) ---" << std::endl;
    // Simple FIR: y[n] = x[n] + 0.5*x[n-1]
    std::vector<float> b = {1.0f, 0.5f};
    std::vector<float> a = {1.0f};
    std::vector<float> x = {1.0f, 1.0f, 1.0f}; 
    std::vector<float> zi = {0.0f}; // Initial state x[-1]=0
    
    // Expected:
    // y[0] = 1*1 + 0.5*0 = 1.0
    // y[1] = 1*1 + 0.5*1 = 1.5
    // y[2] = 1*1 + 0.5*1 = 1.5
    // Final state: x[2]=1
    
    auto [y, next_state] = BVC::LPC::lfilter(b, a, x, zi);
    
    TestUtils::assert_close(y[0], 1.0f, 1e-5f, "FIR y[0]");
    TestUtils::assert_close(y[1], 1.5f, 1e-5f, "FIR y[1]");
    TestUtils::assert_close(y[2], 1.5f, 1e-5f, "FIR y[2]");
    TestUtils::assert_close(next_state[0], 0.5f, 1e-5f, "FIR State");

    // Simple IIR: y[n] = x[n] + 0.5*y[n-1] (Decay)
    // a=[1.0, -0.5] (Note: scipy uses a[0]y[n] + a[1]y[n-1] = ... -> y[n] = ... - a[1]y[n-1])
    // So if we want y[n] = x[n] + 0.5y[n-1], then a[1] must be -0.5.
    b = {1.0f};
    a = {1.0f, -0.5f};
    x = {1.0f, 0.0f, 0.0f};
    zi = {0.0f}; // Initial state y[-1]=0, or rather, the internal scaled state
    
    // Expected:
    // y[0] = 1 + 0.5*0 = 1.0
    // y[1] = 0 + 0.5*1 = 0.5
    // y[2] = 0 + 0.5*0.5 = 0.25
    
    auto [y_iir, next_state_iir] = BVC::LPC::lfilter(b, a, x, zi);
    TestUtils::assert_close(y_iir[0], 1.0f, 1e-5f, "IIR y[0]");
    TestUtils::assert_close(y_iir[1], 0.5f, 1e-5f, "IIR y[1]");
    TestUtils::assert_close(y_iir[2], 0.25f, 1e-5f, "IIR y[2]");
    TestUtils::assert_close(next_state_iir[0], 0.125f, 1e-5f, "IIR State");
}

int main() {
    test_quantizer();
    test_filter_logic();
    std::cout << "All Tests Passed!" << std::endl;
    return 0;
}
