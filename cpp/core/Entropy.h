#pragma once
#include <vector>
#include <cstdint>
#include "BitStream.h"

namespace BVC {
    class EntropyCoder {
    public:
        struct Node {
            int16_t symbol; // -1 for internal
            int left = -1;
            int right = -1;
        };
        
        struct Code {
            uint32_t bits;
            int len;
        };

        EntropyCoder();

        // Atom Index: Huffman coding based on "Voice PDF" (favoring low frequencies)
        void encode_atom_index(BitWriter& writer, uint16_t idx);
        uint16_t decode_atom_index(BitReader& reader);

        // Atom Coefficient: Huffman coding based on Laplacian distribution
        void encode_coefficient(BitWriter& writer, int8_t coeff);
        int8_t decode_coefficient(BitReader& reader);

    private:
        // Index Tree
        std::vector<Node> idx_tree_;
        std::vector<Code> idx_codes_;
        
        // Coeff Tree
        std::vector<Node> coef_tree_;
        std::vector<Code> coef_codes_;

        void build_static_models();
    };
}
