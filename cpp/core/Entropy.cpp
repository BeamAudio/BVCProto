#include "Entropy.h"
#include <cmath>
#include <algorithm>
#include <queue>
#include <map>

namespace BVC {

    EntropyCoder::EntropyCoder() {
        build_static_models();
    }

    struct ProbItem {
        int symbol;
        double prob;
        
        bool operator>(const ProbItem& other) const {
            return prob > other.prob; // Min-heap priority queue
        }
    };
    
    // Helper to build tree from probabilities
    static void build_tree(const std::vector<double>& probs, std::vector<EntropyCoder::Node>& tree, std::vector<EntropyCoder::Code>& codes) {
        struct HeapNode {
            int node_idx;
            double prob;
            bool operator>(const HeapNode& rhs) const { return prob > rhs.prob; }
        };

        std::priority_queue<HeapNode, std::vector<HeapNode>, std::greater<HeapNode>> pq;
        
        // Leaves
        tree.clear();
        tree.reserve(probs.size() * 2);
        codes.assign(probs.size(), {0, 0});

        for(size_t i=0; i<probs.size(); ++i) {
            if(probs[i] > 0.0) {
                tree.push_back({(int16_t)i, -1, -1});
                pq.push({(int)(tree.size()-1), probs[i]});
            }
        }

        if (pq.empty()) return;

        while(pq.size() > 1) {
            auto n1 = pq.top(); pq.pop();
            auto n2 = pq.top(); pq.pop();
            
            tree.push_back({-1, n1.node_idx, n2.node_idx});
            int new_idx = (int)tree.size() - 1;
            pq.push({new_idx, n1.prob + n2.prob});
        }

        // Generate codes by traversing
        int root = pq.top().node_idx;
        
        // Stack: node_idx, current_code, depth
        struct State { int n; uint32_t code; int len; };
        std::vector<State> stack;
        stack.push_back({root, 0, 0});
        
        while(!stack.empty()) {
            auto s = stack.back(); stack.pop_back();
            const auto& node = tree[s.n];
            
            if(node.symbol != -1) {
                // Leaf
                codes[node.symbol] = {s.code, s.len};
            } else {
                // Internal
                if(node.left != -1) stack.push_back({node.left, s.code, s.len + 1}); // Left = 0
                if(node.right != -1) stack.push_back({node.right, s.code | (1u << s.len), s.len + 1}); // Right = 1
            }
        }
        
        // Keep tree for decoder. Root is last element.
        // Optimization: Move root to index 0 or store root index?
        // We'll store root at the end, decode starts from back.
    }

    void EntropyCoder::build_static_models() {
        // 1. Atom Index Model (Voice PDF)
        // Favor indices where (idx % 512) is small (Low Frequency).
        // Size 4096.
        std::vector<double> idx_probs(4096);
        for(int i=0; i<4096; ++i) {
            int freq_idx = i % 512;
            // Model: 1 / (f + C)
            idx_probs[i] = 1.0 / (freq_idx + 10.0);
        }
        // Normalize? Huffman build doesn't strictly require sum=1 but good practice.
        // build_tree handles absolute weights fine.
        
        build_tree(idx_probs, idx_tree_, idx_codes_);

        // 2. Coefficient Model (Laplacian)
        // int8_t range: -128 to 127. Mapped to 0..255 for array index.
        std::vector<double> coef_probs(256);
        for(int i=0; i<256; ++i) {
            int val = (int8_t)i; // Cast back to signed
            // Laplacian: exp(-|x|/b)
            coef_probs[i] = std::exp(-std::abs(val) / 10.0);
        }
        build_tree(coef_probs, coef_tree_, coef_codes_);
    }

    void EntropyCoder::encode_atom_index(BitWriter& writer, uint16_t idx) {
        if(idx >= idx_codes_.size()) idx = 0; // Safety fallback
        Code c = idx_codes_[idx];
        writer.write_bits(c.bits, c.len);
    }

    uint16_t EntropyCoder::decode_atom_index(BitReader& reader) {
        // Traverse tree
        // Root is last node
        int curr = (int)idx_tree_.size() - 1;
        while(idx_tree_[curr].symbol == -1) {
            uint32_t bit = reader.read_bits(1);
            if(bit == 0) curr = idx_tree_[curr].left;
            else curr = idx_tree_[curr].right;
            
            if(curr == -1) return 0; // Error
        }
        return (uint16_t)idx_tree_[curr].symbol;
    }

    void EntropyCoder::encode_coefficient(BitWriter& writer, int8_t coeff) {
        uint8_t sym = (uint8_t)coeff;
        Code c = coef_codes_[sym];
        writer.write_bits(c.bits, c.len);
    }

    int8_t EntropyCoder::decode_coefficient(BitReader& reader) {
        int curr = (int)coef_tree_.size() - 1;
        while(coef_tree_[curr].symbol == -1) {
            uint32_t bit = reader.read_bits(1);
            if(bit == 0) curr = coef_tree_[curr].left;
            else curr = coef_tree_[curr].right;
            
            if(curr == -1) return 0;
        }
        return (int8_t)coef_tree_[curr].symbol;
    }
}
