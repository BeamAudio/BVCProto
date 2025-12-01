#pragma once
#include <vector>
#include <cstdint>
#include <iostream>
#include <fstream>

namespace BVC {
    class BitWriter {
    private:
        std::vector<uint8_t> buffer_;
        uint64_t accumulator_ = 0;
        int bit_count_ = 0;

    public:
        void write_bits(uint32_t value, int bits) {
            // Mask value just in case
            value &= (1u << bits) - 1;
            
            accumulator_ |= (static_cast<uint64_t>(value) << bit_count_);
            bit_count_ += bits;

            while (bit_count_ >= 8) {
                buffer_.push_back(static_cast<uint8_t>(accumulator_ & 0xFF));
                accumulator_ >>= 8;
                bit_count_ -= 8;
            }
        }

        // Flush remaining bits (padding with 0)
        void flush() {
            if (bit_count_ > 0) {
                buffer_.push_back(static_cast<uint8_t>(accumulator_ & 0xFF));
                accumulator_ = 0;
                bit_count_ = 0;
            }
        }

        const std::vector<uint8_t>& data() const { return buffer_; }
        void clear() { buffer_.clear(); accumulator_ = 0; bit_count_ = 0; }
        
        // Helper to dump to ofstream
        void dump_to(std::ofstream& out) {
            flush();
            out.write(reinterpret_cast<const char*>(buffer_.data()), buffer_.size());
            clear();
        }
    };

    class BitReader {
    private:
        const uint8_t* data_;
        size_t size_;
        size_t byte_pos_ = 0;
        uint64_t accumulator_ = 0;
        int bit_count_ = 0;

    public:
        BitReader(const uint8_t* data, size_t size) : data_(data), size_(size) {}

        uint32_t read_bits(int bits) {
            while (bit_count_ < bits) {
                if (byte_pos_ < size_) {
                    accumulator_ |= (static_cast<uint64_t>(data_[byte_pos_++]) << bit_count_);
                    bit_count_ += 8;
                } else {
                    // EOF simulation: just return zeros if we run out (or handle error)
                    bit_count_ += 8; // Pretend we read zeros
                }
            }

            uint32_t val = static_cast<uint32_t>(accumulator_ & ((1u << bits) - 1));
            accumulator_ >>= bits;
            bit_count_ -= bits;
            return val;
        }
        
        bool eof() const {
            return byte_pos_ >= size_ && bit_count_ < 8;
        }
    };
}
