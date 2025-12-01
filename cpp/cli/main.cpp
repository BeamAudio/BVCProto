#include <iostream>
#include "../core/BVCEncoder.h"
#include "../core/BVCDecoder.h"
#include "../core/Common.h"

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cout << "BVC CLI (Refactored Core)\nUsage: bvc encode <in.wav> <out.bvc>\n       bvc decode <in.bvc> <out.wav>\n";
        return 1;
    }

    std::string cmd = argv[1];
    BVC::CodecConfig config;
    
    for(int i=4; i<argc; ++i) {
        if(std::string(argv[i]) == "--merge" && i+1 < argc) {
            config.max_merge_frames = std::min(std::stoi(argv[++i]), 63); // Cap at 63
        }
        if(std::string(argv[i]) == "--freqs" && i+1 < argc) {
            config.default_num_freqs = std::min(std::stoi(argv[++i]), 1024); // Cap at 1024
        }
    }

    try {
        if (cmd == "encode") {
            BVC::BVCEncoder encoder(config);
            encoder.encode_file(argv[2], argv[3]);
        } else if (cmd == "decode") {
            BVC::BVCDecoder decoder(config);
            decoder.decode_file(argv[2], argv[3]);
        }
    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

