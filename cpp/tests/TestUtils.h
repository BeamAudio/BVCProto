#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <cmath>

namespace TestUtils {
    void assert_true(bool condition, const std::string& msg) {
        if (!condition) {
            std::cerr << "[FAIL] " << msg << std::endl;
            exit(1);
        } else {
            std::cout << "[PASS] " << msg << std::endl;
        }
    }

    void assert_close(float a, float b, float tol, const std::string& msg) {
        if (std::abs(a - b) > tol) {
            std::cerr << "[FAIL] " << msg << " (Expected " << b << ", got " << a << ")" << std::endl;
            exit(1);
        } else {
            std::cout << "[PASS] " << msg << std::endl;
        }
    }
}
