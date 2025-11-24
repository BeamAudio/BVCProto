import argparse
import numpy as np
from scipy.io import wavfile
from BVC import BVC_GLPC
import os
import sys

def ensure_mono(sig):
    if len(sig.shape) > 1:
        print("Warning: Input is stereo, converting to mono.")
        return np.mean(sig, axis=1)
    return sig

def float_to_int16(sig):
    return np.clip(sig * 32767, -32768, 32767).astype(np.int16)

def int16_to_float(sig):
    return sig.astype(np.float32) / 32768.0

def cmd_encode(args):
    input_path = args.input
    output_path = args.output
    
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found.")
        sys.exit(1)
        
    print(f"Reading {input_path}...")
    fs, audio = wavfile.read(input_path)
    
    # Convert to float mono
    audio = int16_to_float(audio)
    audio = ensure_mono(audio)
    
    # Config
    quantize = not args.lossless
    config = {}
    if quantize:
        config['LAR_BITS'] = args.bits
        
    print(f"Encoding with BVC (fs={fs}, quantize={quantize}, lpc={args.lpc}, max_merge={args.max_merge}, num_freqs={args.num_freqs})...")
    codec = BVC_GLPC(fs, quantize=quantize, lpc_order=args.lpc, quantizer_config=config, max_merge=args.max_merge, num_freqs=args.num_freqs)
    
    frames = codec.process(audio)
    
    print(f"Saving to {output_path}...")
    codec.save_to_file(output_path, frames)
    
    # Stats
    file_size = os.path.getsize(output_path)
    duration = len(audio) / fs
    bitrate = (file_size * 8) / duration / 1000.0
    print(f"Done! Size: {file_size} bytes. Bitrate: {bitrate:.2f} kbps")

def cmd_decode(args):
    input_path = args.input
    output_path = args.output
    
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found.")
        sys.exit(1)
        
    print(f"Loading {input_path}...")
    # We need a dummy instance to load, fs will be overwritten
    codec = BVC_GLPC(44100) 
    frames = codec.load_from_file(input_path)
    
    print(f"Decoding {len(frames)} frames...")
    audio = codec.decode(frames)
    
    print(f"Decoded audio length: {len(audio)} samples ({len(audio)/codec.fs:.3f} seconds)")
    audio_int16 = float_to_int16(audio)
    print(f"After int16 conversion: {len(audio_int16)} samples")
    
    print(f"Saving to {output_path}...")
    wavfile.write(output_path, codec.fs, audio_int16)
    print("Done!")

def main():
    parser = argparse.ArgumentParser(description="Beam Vocal Codec (BVC) CLI Tool")
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Encode Command
    parser_encode = subparsers.add_parser('encode', help='Encode WAV to BVC')
    parser_encode.add_argument('input', help='Input WAV file')
    parser_encode.add_argument('output', help='Output BVC file')
    parser_encode.add_argument('bits', nargs='?', type=int, default=10, help='Quantization bits for LAR (default: 10)')
    parser_encode.add_argument('max_merge', nargs='?', type=int, default=32, help='Maximum number of frames to merge (default: 32)')
    parser_encode.add_argument('num_freqs', nargs='?', type=int, default=64, help='Number of frequencies (num_freqs) (default: 64)')
    parser_encode.add_argument('--lpc', type=int, default=16, help='LPC Order (default: 16)')
    parser_encode.add_argument('--lossless', action='store_true', help='Disable quantization (Lossless mode)')
    
    # Decode Command
    parser_decode = subparsers.add_parser('decode', help='Decode BVC to WAV')
    parser_decode.add_argument('input', help='Input BVC file')
    parser_decode.add_argument('output', help='Output WAV file')
    
    args = parser.parse_args()
    
    if args.command == 'encode':
        cmd_encode(args)
    elif args.command == 'decode':
        cmd_decode(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
