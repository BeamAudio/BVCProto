import struct
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

class BVCInspector:
    def __init__(self, filename):
        self.filename = filename
        self.frames = []
        self.fs = 0
        self.version = 0
        self.quantize = False
        
    def inspect(self):
        print(f"=== Inspecting: {self.filename} ===")
        file_size = os.path.getsize(self.filename)
        print(f"File Size: {file_size} bytes")
        
        with open(self.filename, 'rb') as f:
            # --- Header ---
            magic = f.read(4)
            if magic != b'RBVC':
                print("ERROR: Invalid Magic Bytes")
                return
            
            version_byte = f.read(1)
            if len(version_byte) == 0:
                print("Format: Legacy (No Version)")
                f.seek(4)
                self.version = 1
                self.fs, num_frames = struct.unpack('<II', f.read(8))
            else:
                self.version = struct.unpack('B', version_byte)[0]
                self.fs, num_frames = struct.unpack('<II', f.read(8))
                
            self.quantize = (self.version == 1)
            mode_str = "Lossy" if self.quantize else "Lossless"
            
            print(f"Format Version: {self.version} ({mode_str})")
            print(f"Sample Rate: {self.fs} Hz")
            print(f"Total Frames: {num_frames}")
            
            # --- Frame Loop ---
            total_samples = 0
            total_atoms = 0
            
            for i in range(num_frames):
                frame_start_pos = f.tell()
                
                flags_byte = f.read(1)
                if not flags_byte: break
                flags = struct.unpack('B', flags_byte)[0]
                
                mode = (flags >> 4) & 0x0F
                merge = flags & 0x0F
                
                # Energy
                if self.version == 1:
                    q_energy = struct.unpack('B', f.read(1))[0]
                    energy_val = f"Q={q_energy}"
                else:
                    q_energy = struct.unpack('<f', f.read(4))[0]
                    energy_val = f"{q_energy:.4e}"
                
                n_atoms = struct.unpack('<H', f.read(2))[0]
                
                # LAR
                lar_bytes = 0
                if mode == 0:
                    if self.version == 1:
                        f.read(32) # 16 * 2
                        lar_bytes = 32
                    else:
                        f.read(64) # 16 * 4
                        lar_bytes = 64
                
                # Atoms
                f.read(6 * n_atoms)
                
                frame_end_pos = f.tell()
                frame_bytes = frame_end_pos - frame_start_pos
                
                duration_samples = merge * 256
                total_samples += duration_samples
                total_atoms += n_atoms
                
                self.frames.append({
                    'index': i,
                    'mode': mode,
                    'merge': merge,
                    'n_atoms': n_atoms,
                    'bytes': frame_bytes,
                    'energy_str': energy_val
                })
                
            duration_sec = total_samples / self.fs
            avg_bitrate = (file_size * 8) / duration_sec / 1000.0
            
            print(f"\n--- Statistics ---")
            print(f"Total Duration: {duration_sec:.3f} s")
            print(f"Average Bitrate: {avg_bitrate:.2f} kbps")
            print(f"Total Atoms: {total_atoms}")
            print(f"Avg Atoms/Frame: {total_atoms/num_frames:.1f}")
            
    def plot_analysis(self):
        if not self.frames:
            print("No frames to plot.")
            return
            
        indices = [f['index'] for f in self.frames]
        merges = [f['merge'] for f in self.frames]
        modes = [f['mode'] for f in self.frames]
        atoms = [f['n_atoms'] for f in self.frames]
        bytes_per_frame = [f['bytes'] for f in self.frames]
        
        # Time axis (approximate, since frames vary in length)
        # To be precise, we should accumulate duration
        times = []
        curr_t = 0
        for m in merges:
            times.append(curr_t)
            curr_t += (m * 256) / self.fs
        
        plt.figure(figsize=(10, 8))
        
        # 1. Frame Size (Merge Count)
        plt.subplot(3, 1, 1)
        plt.step(times, merges, where='post', color='blue', label='Merge Count')
        plt.ylabel('Frame Size (x256)')
        plt.title('Adaptive Frame Sizing')
        plt.grid(True, alpha=0.3)
        
        # Overlay Mode (Transient vs Normal)
        # We can color background? Or just scatter points
        transient_indices = [i for i, m in enumerate(modes) if m == 1]
        if transient_indices:
            t_trans = [times[i] for i in transient_indices]
            y_trans = [merges[i] for i in transient_indices]
            plt.plot(t_trans, y_trans, 'rx', label='Transient Mode')
        plt.legend()
        
        # 2. Atoms per Frame
        plt.subplot(3, 1, 2)
        plt.bar(times, atoms, width=0.01, color='green', alpha=0.6)
        plt.ylabel('Num Atoms')
        plt.title('Sparsity (Atoms per Frame)')
        plt.grid(True, alpha=0.3)
        
        # 3. Bitrate (Bytes per Frame)
        plt.subplot(3, 1, 3)
        plt.plot(times, bytes_per_frame, 'k-', linewidth=1)
        plt.ylabel('Bytes')
        plt.xlabel('Time (s)')
        plt.title('Bitrate Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = self.filename.replace('.rbvc', '_analysis.png')
        plt.savefig(output_file)
        print(f"Saved analysis plot to: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python BVC_Inspector.py <file.rbvc>")
    else:
        inspector = BVCInspector(sys.argv[1])
        inspector.inspect()
        inspector.plot_analysis()
