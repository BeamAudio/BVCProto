#pragma once
#include <vector>
#include <memory>
#include <tuple>
#include <list>
#include <map>
#include <mutex>
#include "Common.h"

namespace BVC {
struct DictionaryEntry {
    int n_atoms = 0;
    int length = 0;
    std::vector<float> D_flat; // Flattened Dictionary [n_atoms, length]
    std::vector<float> G_flat; // Flattened Gram Matrix [n_atoms, n_atoms]
    std::vector<float> atom_freqs_hz; // Center frequency of each atom
    bool has_gram = false;
};

    class Dictionary {
    public:
        static std::shared_ptr<DictionaryEntry> get(int N, int fs, int num_freqs, FrameMode mode, bool compute_gram = true);
        static void clear_cache();
        
    private:
        // Cache internals
        struct CacheItem {
            std::shared_ptr<DictionaryEntry> entry;
            size_t size_bytes;
            std::list<std::tuple<int, int, int>>::iterator lru_it;
        };

        static std::map<std::tuple<int, int, int>, CacheItem> g_cache;
        static std::list<std::tuple<int, int, int>> g_lru_list;
        static size_t g_current_memory;
        static const size_t MAX_MEMORY_BYTES = 256 * 1024 * 1024; // 256 MB Cache Limit
        static std::mutex g_mutex;
    };
}