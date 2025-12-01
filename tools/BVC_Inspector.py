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
                # Try to read 12 bytes (current format), fall back if short
                header_bytes = f.read(12)
                if len(header_bytes) == 12:
                    self.fs, num_frames, orig_len = struct.unpack('<III', header_bytes)
                elif len(header_bytes) == 8:
                    self.fs, num_frames = struct.unpack('<II', header_bytes)
                else:
                    print("ERROR: Invalid Header")
                    return
                
            self.quantize = (self.version == 1)
            mode_str = "Lossy" if self.quantize else "Lossless"
            
            print(f"Format Version: {self.version} ({mode_str})")
            print(f"Sample Rate: {self.fs} Hz")
            print(f"Total Frames: {num_frames}")
            
            # --- Frame Loop ---
            print(f"\n{'Idx':<5} {'Mode':<5} {'Mrg':<4} {'Energy':<12} {'Atoms':<6} {'Bytes':<6} {'Kbps':<8}")
            print("-" * 55)
            
            total_samples = 0
            total_atoms = 0
            
            # Stats collectors
            mode_counts = {0: 0, 1: 0, 2: 0, 3: 0}
            atom_list = []
            bitrate_list = []
            merge_list = []
            
            for i in range(num_frames):
                frame_start_pos = f.tell()
                
                flags_byte = f.read(1)
                if not flags_byte: break
                flags = struct.unpack('B', flags_byte)[0]
                
                mode = (flags >> 6) & 0x03
                merge = flags & 0x3F
                if merge == 0: merge = 1
                
                mode_map = {0: "SIL", 1: "VCD", 2: "UNV"}
                mode_name = mode_map.get(mode, "???")
                mode_counts[mode] = mode_counts.get(mode, 0) + 1
                merge_list.append(merge)
                
                # Energy
                if self.version == 1:
                    q_energy = struct.unpack('B', f.read(1))[0]
                    energy_val = f"Q={q_energy}"
                else:
                    q_energy = struct.unpack('<f', f.read(4))[0]
                    energy_val = f"{q_energy:.2e}"
                
                n_atoms = struct.unpack('<H', f.read(2))[0]
                atom_list.append(n_atoms)
                
                # LAR (Read unconditionally)
                lar_bytes = 0
                if self.version == 1:
                    f.read(32) # 16 * 2 bytes (short)
                    lar_bytes = 32
                else:
                    f.read(64) # 16 * 4 bytes (float)
                    lar_bytes = 64
                
                # Atoms
                atoms_bytes = 0
                if n_atoms > 0:
                    if self.version == 1:
                        # Quantized: Gain (4 bytes) + n_atoms * (2 byte idx + 1 byte coeff)
                        f.read(4) # Gain
                        f.read(3 * n_atoms)
                        atoms_bytes = 4 + (3 * n_atoms)
                    else:
                        # Lossless: n_atoms * (2 byte idx + 4 byte coef)
                        f.read(6 * n_atoms)
                        atoms_bytes = 6 * n_atoms
                
                frame_end_pos = f.tell()
                frame_bytes = frame_end_pos - frame_start_pos
                
                duration_samples = merge * 256
                total_samples += duration_samples
                total_atoms += n_atoms
                
                # Derived Stats
                frame_dur_sec = duration_samples / self.fs
                if frame_dur_sec > 0:
                    frame_kbps = (frame_bytes * 8) / frame_dur_sec / 1000.0
                else:
                    frame_kbps = 0
                bitrate_list.append(frame_kbps)
                
                print(f"{i:<5} {mode_name:<5} {merge:<4} {energy_val:<12} {n_atoms:<6} {frame_bytes:<6} {frame_kbps:<8.2f}")
                
                self.frames.append({
                    'index': i,
                    'mode': mode,
                    'merge': merge,
                    'n_atoms': n_atoms,
                    'bytes': frame_bytes,
                    'energy_str': energy_val
                })
                
            duration_sec = total_samples / self.fs
            avg_bitrate = (file_size * 8) / duration_sec / 1000.0 if duration_sec > 0 else 0
            
            raw_pcm_bytes = total_samples * 2 # 16-bit mono
            compression_ratio = raw_pcm_bytes / file_size if file_size > 0 else 0
            
            print(f"\n=== Detailed Statistics ===")
            print(f"Total Duration:   {duration_sec:.3f} s")
            print(f"Raw PCM Size:     {raw_pcm_bytes / 1024:.2f} KB")
            print(f"Compressed Size:  {file_size / 1024:.2f} KB")
            print(f"Compression Ratio: {compression_ratio:.2f}:1")
            
            print(f"\n--- Mode Distribution ---")
            total_f = sum(mode_counts.values())
            if total_f > 0:
                print(f"  Silence (SIL):  {mode_counts.get(0,0)} ({mode_counts.get(0,0)/total_f*100:.1f}%)")
                print(f"  Voiced (VCD):   {mode_counts.get(1,0)} ({mode_counts.get(1,0)/total_f*100:.1f}%)")
                print(f"  Unvoiced (UNV): {mode_counts.get(2,0)} ({mode_counts.get(2,0)/total_f*100:.1f}%)")
            
            print(f"\n--- Merging Stats ---")
            if merge_list:
                print(f"  Avg Merge: {np.mean(merge_list):.2f} frames")
                print(f"  Max Merge: {np.max(merge_list)} frames")
            
            print(f"\n--- Atom Stats ---")
            if atom_list:
                print(f"  Total Atoms: {total_atoms}")
                print(f"  Min Atoms/Frame: {np.min(atom_list)}")
                print(f"  Max Atoms/Frame: {np.max(atom_list)}")
                print(f"  Avg Atoms/Frame: {np.mean(atom_list):.2f}")
                print(f"  Atom Density:    {total_atoms/duration_sec:.1f} atoms/sec")

            print(f"\n--- Bitrate Stats ---")
            if bitrate_list:
                print(f"  Average Bitrate: {avg_bitrate:.2f} kbps")
                print(f"  Min Frame Bitrate: {np.min(bitrate_list):.2f} kbps")
                print(f"  Max Frame Bitrate: {np.max(bitrate_list):.2f} kbps")
            
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
        base, _ = os.path.splitext(self.filename)
        output_file = f"{base}_analysis.png"
        plt.savefig(output_file)
        print(f"Saved analysis plot to: {output_file}")
        plt.close()

        # --- New Performance Plots ---
        plt.figure(figsize=(12, 8))
        
        # 1. Bitrate Histogram
        plt.subplot(2, 2, 1)
        plt.hist(bitrate_list, bins=20, color='skyblue', edgecolor='black')
        plt.title('Bitrate Distribution')
        plt.xlabel('kbps')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)

        # 2. Frame Size Histogram
        plt.subplot(2, 2, 2)
        plt.hist(merges, bins=np.arange(1, 66)-0.5, color='lightgreen', edgecolor='black')
        plt.title('Frame Size Distribution')
        plt.xlabel('Merge Count')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)

        # 3. Atoms vs Energy Scatter
        plt.subplot(2, 2, 3)
        # Parse energy strings back to floats? Or use atoms
        # energy_str is "Q=..." or float.
        # Let's assume correlation with atoms is enough
        plt.scatter(merges, atoms, alpha=0.5, c=modes, cmap='viridis')
        plt.title('Atoms vs Frame Size (Color=Mode)')
        plt.xlabel('Merge Count')
        plt.ylabel('Atom Count')
        plt.grid(True, alpha=0.3)
        
        # 4. Mode Pie Chart
        plt.subplot(2, 2, 4)
        # Recalculate counts for safety
        labels = ['Silence', 'Voiced', 'Unvoiced']
        sizes = [modes.count(0), modes.count(1), modes.count(2)]
        # Filter zero sizes
        pie_labels = [l for l, s in zip(labels, sizes) if s > 0]
        pie_sizes = [s for s in sizes if s > 0]
        
        if pie_sizes:
            plt.pie(pie_sizes, labels=pie_labels, autopct='%1.1f%%', startangle=140, colors=['lightgray', 'lightgreen', 'salmon'])
            plt.title('Mode Distribution')
        
        plt.tight_layout()
        output_file_perf = f"{base}_performance.png"
        plt.savefig(output_file_perf)
        print(f"Saved performance plots to: {output_file_perf}")
        plt.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python BVC_Inspector.py <file.rbvc>")
    else:
        inspector = BVCInspector(sys.argv[1])
        inspector.inspect()
        inspector.plot_analysis()
