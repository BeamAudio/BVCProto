#pragma once
#include "Dictionary.h"
#include <vector>
#include <cstdint>

namespace BVC {
    namespace MP {
        struct Atom {
            uint16_t idx;
            float value;
        };

        std::vector<Atom> matching_pursuit(const std::vector<float>& residual, 
                                           const std::shared_ptr<DictionaryEntry>& dict, 
                                           int max_atoms, float threshold);
    }
}
