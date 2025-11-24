import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.io.wavfile
from BVC import BVC_GLPC
import os
import shutil

# --- IEEE Plotting Style ---
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'axes.grid': True,
    'image.cmap': 'inferno'
})

class MasterTestSuite:
    def __init__(self):
        self.fs = 44100
        self.results_dir = "results"
        self.setup_directories()

    def setup_directories(self):
        if os.path.exists(self.results_dir):
            shutil.rmtree(self.results_dir)
        os.makedirs(os.path.join(self.results_dir, "sweeps"))
        os.makedirs(os.path.join(self.results_dir, "synthetic"))
        # Removed 'complex' as we will do focused tests

    # --- Signal Generation ---
    def generate_complex_signal(self):
        """Generates the 5-second dynamic test signal."""
        total_samples = 5 * self.fs
        sig = np.zeros(total_samples)
        t = np.arange(total_samples) / self.fs
        
        # 1. Steady Tone (0.5 - 1.5s)
        idx_start = int(0.5 * self.fs)
        idx_end = int(1.5 * self.fs)
        t_seg = t[idx_start:idx_end]
        sig[idx_start:idx_end] += 0.5 * np.sin(2 * np.pi * 440 * t_seg)
        
        # 2. Fast Chirp (1.5 - 2.0s)
        idx_start = int(1.5 * self.fs)
        idx_end = int(2.0 * self.fs)
        t_seg = t[idx_start:idx_end] - 1.5
        sig[idx_start:idx_end] += 0.5 * signal.chirp(t_seg, f0=20, f1=self.fs/2 - 100, t1=0.5, method='linear')
        
        # 3. Impulse Train (2.0 - 3.0s)
        idx_start = int(2.0 * self.fs)
        idx_end = int(3.0 * self.fs)
        for i in range(10):
            pos = idx_start + int(i * 0.1 * self.fs)
            if pos + 50 < len(sig):
                impulse = np.exp(-np.linspace(0, 20, 50)) * np.random.randn(50)
                sig[pos:pos+50] += 2.0 * impulse
                
        # 4. Speech-like (3.0 - 4.0s)
        idx_start = int(3.0 * self.fs)
        idx_end = int(4.0 * self.fs)
        t_seg = t[idx_start:idx_end]
        carrier = np.random.randn(len(t_seg))
        b, a = signal.butter(4, [500/(self.fs/2), 3000/(self.fs/2)], btype='band')
        speech_noise = signal.lfilter(b, a, carrier)
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t_seg)
        sig[idx_start:idx_end] += 0.4 * speech_noise * envelope
        
        sig += 0.001 * np.random.randn(len(sig))
        return sig

    def compute_snr(self, ref, test):
        min_len = min(len(ref), len(test))
        ref = ref[:min_len]
        test = test[:min_len]
        
        max_val = max(np.max(np.abs(ref)), np.max(np.abs(test)))
        if max_val > 0:
            ref_norm = ref / max_val
            test_norm = test / max_val
        else:
            return 0.0 
            
        noise = ref_norm - test_norm
        signal_power = np.sum(ref_norm**2)
        noise_power = np.sum(noise**2)
        
        if noise_power < 1e-10: return 100.0
        if signal_power < 1e-10: return 0.0
        
        snr = 10 * np.log10(signal_power / noise_power)
        
        if np.isnan(snr) or np.isinf(snr) or snr > 100: return 100.0
        if snr < -20: return -20.0
        
        return snr

    def compute_lsd(self, ref, test):
        min_len = min(len(ref), len(test))
        ref = ref[:min_len]
        test = test[:min_len]
        
        max_val = max(np.max(np.abs(ref)), np.max(np.abs(test)))
        if max_val > 0:
            ref = ref / max_val
            test = test / max_val
            
        f, t, S_ref = signal.spectrogram(ref, self.fs, nperseg=512, noverlap=256)
        _, _, S_test = signal.spectrogram(test, self.fs, nperseg=512, noverlap=256)
        
        log_S_ref = 10 * np.log10(np.abs(S_ref)**2 + 1e-9)
        log_S_test = 10 * np.log10(np.abs(S_test)**2 + 1e-9)
        
        diff = (log_S_ref - log_S_test)**2
        lsd = np.mean(np.sqrt(np.mean(diff, axis=0)))
        
        if np.isnan(lsd) or np.isinf(lsd): return 20.0 
        if lsd > 20.0: return 20.0
        
        return lsd

    def compute_bitrate(self, codec, frames, duration):
        temp_file = os.path.join(self.results_dir, "temp_bitrate.rbvc")
        codec.save_to_file(temp_file, frames)
        size_bytes = os.path.getsize(temp_file)
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        bits = size_bytes * 8
        kbps = (bits / duration) / 1000.0
        return kbps

    # --- New "Sweet Spot" Analysis ---
    def run_sweet_spot_analysis(self):
        print("Running Conclusive Sweet Spot Analysis...")
        
        # Use full synthetic signal for better average statistics
        sig = self.generate_complex_signal()
        duration = len(sig) / self.fs
        
        # 1. Dictionary Size Sweep (Key for quality)
        print("  1. Sweeping Dictionary Size...")
        dict_sizes = [64, 128, 256, 512]
        results_dict = {'kbps': [], 'snr': [], 'lsd': []}
        
        for d in dict_sizes:
            codec = BVC_GLPC(self.fs, quantize=True, quantizer_config={'LAR_BITS': 10}, num_freqs=d, max_merge=8)
            frames = codec.process(sig)
            recon = codec.decode(frames)
            results_dict['kbps'].append(self.compute_bitrate(codec, frames, duration))
            results_dict['snr'].append(self.compute_snr(sig, recon))
            results_dict['lsd'].append(self.compute_lsd(sig, recon))
            
        self._plot_curve(results_dict['kbps'], results_dict['snr'], 'SNR', dict_sizes, 'Dict Size', 'dict_snr.png')
        self._plot_curve(results_dict['kbps'], results_dict['lsd'], 'LSD', dict_sizes, 'Dict Size', 'dict_lsd.png')

        # 2. Merge Limit Sweep (Key for compression)
        print("  2. Sweeping Merge Limit...")
        merge_limits = [1, 4, 8, 16, 32]
        results_merge = {'kbps': [], 'snr': [], 'lsd': []}
        
        for m in merge_limits:
            codec = BVC_GLPC(self.fs, quantize=True, quantizer_config={'LAR_BITS': 10}, num_freqs=128, max_merge=m)
            frames = codec.process(sig)
            recon = codec.decode(frames)
            results_merge['kbps'].append(self.compute_bitrate(codec, frames, duration))
            results_merge['snr'].append(self.compute_snr(sig, recon))
            results_merge['lsd'].append(self.compute_lsd(sig, recon))
            
        self._plot_curve(results_merge['kbps'], results_merge['snr'], 'SNR', merge_limits, 'Merge Limit', 'merge_snr.png')
        
        # 3. Quantization Bits Sweep (Key for fidelity trade-off)
        print("  3. Sweeping Quantization Bits...")
        bits_list = [6, 8, 10, 12, 14]
        results_bits = {'kbps': [], 'snr': [], 'lsd': []}
        
        for b in bits_list:
            codec = BVC_GLPC(self.fs, quantize=True, quantizer_config={'LAR_BITS': b}, num_freqs=128, max_merge=8)
            frames = codec.process(sig)
            recon = codec.decode(frames)
            results_bits['kbps'].append(self.compute_bitrate(codec, frames, duration))
            results_bits['snr'].append(self.compute_snr(sig, recon))
            results_bits['lsd'].append(self.compute_lsd(sig, recon))
            
        self._plot_curve(results_bits['kbps'], results_bits['snr'], 'SNR', bits_list, 'Quant Bits', 'bits_snr.png')

    def _plot_curve(self, x_data, y_data, y_label, param_values, param_name, filename):
        plt.figure()
        plt.plot(x_data, y_data, 'o-', linewidth=2)
        for i, txt in enumerate(param_values):
            plt.annotate(f"{param_name}={txt}", (x_data[i], y_data[i]), xytext=(5, 5), textcoords='offset points')
        
        plt.xlabel('Bitrate (kbps)')
        plt.ylabel(f'{y_label} (dB)')
        plt.title(f'{y_label} vs Bitrate (Varying {param_name})')
        plt.grid(True)
        plt.savefig(os.path.join(self.results_dir, "sweeps", filename))
        plt.close()

    def run_synthetic_batch(self):
        print("Running Experiment C: Synthetic Dataset Batch...")
        # Simply check if codec crashes and generates basic plots
        # Reduced scope for speed
        pass 

    def run_frame_merging_visualization(self):
        print("Running Frame Merging Logic Visualization...")
        
        # Generate a signal with distinct switching regimes
        T = 2.0
        t = np.arange(int(T * self.fs)) / self.fs
        sig = np.zeros_like(t)
        
        # 0.0-0.4s: Silence
        # 0.4-0.8s: Stable Tone (Voiced-like)
        sig[int(0.4*self.fs):int(0.8*self.fs)] = 0.5 * np.sin(2 * np.pi * 440 * t[int(0.4*self.fs):int(0.8*self.fs)])
        
        # 0.8-1.2s: White Noise (Unvoiced-like)
        sig[int(0.8*self.fs):int(1.2*self.fs)] = 0.3 * np.random.randn(int(0.4*self.fs))
        
        # 1.2-1.6s: Fast Chirp (Transient-like)
        t_seg = t[int(1.2*self.fs):int(1.6*self.fs)] - 1.2
        sig[int(1.2*self.fs):int(1.6*self.fs)] = 0.5 * signal.chirp(t_seg, f0=100, f1=1000, t1=0.4)
        
        # 1.6-2.0s: Mixed/Tone again
        sig[int(1.6*self.fs):] = 0.5 * np.sin(2 * np.pi * 220 * t[int(1.6*self.fs):])
        
        # Process
        codec = BVC_GLPC(self.fs, quantize=True, max_merge=16)
        frames = codec.process(sig)
        
        # --- Visualization ---
        plt.figure(figsize=(12, 8))
        plt.suptitle("Frame Merging & Mode Decision Analysis")
        
        # Subplot 1: Waveform & Modes
        plt.subplot(2, 1, 1)
        plt.plot(t, sig, 'k-', alpha=0.3, linewidth=1)
        plt.ylabel("Amplitude")
        plt.title("Waveform with Frame Boundaries (Color=Mode)")
        
        curr_time = 0
        colors = {0: 'gray', 1: 'green', 2: 'red'} # Silence, Voiced, Unvoiced
        labels = {0: 'Silence', 1: 'Voiced', 2: 'Unvoiced'}
        added_labels = set()
        
        for f in frames:
            mode = f['mode']
            merge = f['merge']
            duration = (merge * 256) / self.fs
            
            c = colors.get(mode, 'blue')
            lbl = labels.get(mode, 'Unknown') if mode not in added_labels else None
            if lbl: added_labels.add(mode)
            
            # Draw box/shading for the frame
            plt.axvspan(curr_time, curr_time + duration, color=c, alpha=0.2, label=lbl)
            # Draw boundary line
            plt.axvline(curr_time + duration, color='k', linestyle=':', alpha=0.5, linewidth=0.5)
            
            curr_time += duration
            
        plt.legend(loc='upper right')
        plt.xlim(0, T)
        
        # Subplot 2: Merge Count Step Plot
        plt.subplot(2, 1, 2)
        
        times = []
        merges = []
        modes = []
        curr_time = 0
        
        for f in frames:
            merge = f['merge']
            duration = (merge * 256) / self.fs
            times.extend([curr_time, curr_time + duration])
            merges.extend([merge, merge])
            modes.append(f['mode'])
            curr_time += duration
            
        plt.plot(times, merges, 'b-', linewidth=2)
        plt.fill_between(times, merges, alpha=0.2, color='blue')
        plt.ylabel("Merge Count (Frames)")
        plt.xlabel("Time (s)")
        plt.title("Frame Merging Decision over Time")
        plt.grid(True)
        plt.xlim(0, T)
        plt.ylim(0, codec.max_merge + 2)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(self.results_dir, "vis_frame_merging.png"))
        plt.close()

    def run_real_world_visualization(self):
        print("Running Real-World Visualization (Waveform & Spectrogram)...")
        
        # Try to load real file, else synthetic
        filename = "Recording63.wav"
        if os.path.exists(filename):
            print(f"  Using real file: {filename}")
            fs, sig_int16 = scipy.io.wavfile.read(filename)
            # Convert to float mono
            if len(sig_int16.shape) > 1:
                sig = np.mean(sig_int16, axis=1)
            else:
                sig = sig_int16
            sig = sig.astype(np.float32) / 32768.0
            # Trim to 2 seconds for clear plotting if too long
            if len(sig) > 2 * fs:
                sig = sig[int(0.5*fs):int(2.5*fs)]
        else:
            print("  Using synthetic complex signal.")
            sig = self.generate_complex_signal()
            # Trim
            sig = sig[:int(2.0*self.fs)]
            
        # Process
        # Use high fidelity settings
        codec = BVC_GLPC(self.fs, quantize=True, quantizer_config={'LAR_BITS': 12}, num_freqs=256, max_merge=8)
        frames = codec.process(sig)
        recon = codec.decode(frames)
        
        # Metrics
        snr = self.compute_snr(sig, recon)
        lsd = self.compute_lsd(sig, recon)
        bitrate = self.compute_bitrate(codec, frames, len(sig)/self.fs)
        print(f"  Result Metrics -> SNR: {snr:.2f} dB, LSD: {lsd:.2f} dB, Bitrate: {bitrate:.2f} kbps")
        
        # Plotting
        min_len = min(len(sig), len(recon))
        t = np.arange(min_len) / self.fs
        sig = sig[:min_len]
        recon = recon[:min_len]
        
        plt.figure(figsize=(12, 10))
        plt.suptitle(f"Real-World Analysis: SNR={snr:.1f}dB | LSD={lsd:.2f} | {bitrate:.0f} kbps", fontsize=14)
        
        # 1. Input Waveform
        plt.subplot(2, 2, 1)
        plt.plot(t, sig, 'k-', linewidth=0.5)
        plt.title("Original Waveform")
        plt.ylabel("Amplitude")
        plt.xlabel("Time (s)")
        plt.grid(True, alpha=0.3)
        
        # 2. Output Waveform
        plt.subplot(2, 2, 2)
        plt.plot(t, recon, 'b-', linewidth=0.5)
        plt.title("Reconstructed Waveform")
        plt.xlabel("Time (s)")
        plt.grid(True, alpha=0.3)
        
        # 3. Input Spectrogram
        plt.subplot(2, 2, 3)
        plt.specgram(sig, NFFT=1024, Fs=self.fs, noverlap=512, cmap='inferno', vmin=-100)
        plt.title("Original Spectrogram")
        plt.ylabel("Frequency (Hz)")
        plt.xlabel("Time (s)")
        
        # 4. Output Spectrogram
        plt.subplot(2, 2, 4)
        plt.specgram(recon, NFFT=1024, Fs=self.fs, noverlap=512, cmap='inferno', vmin=-100)
        plt.title("Reconstructed Spectrogram")
        plt.ylabel("Frequency (Hz)")
        plt.xlabel("Time (s)")
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(self.results_dir, "vis_real_world.png"))
        plt.close()

    def run_all(self):
        print("=== Starting Master Test Suite ===")
        self.run_real_world_visualization()
        self.run_frame_merging_visualization()
        # self.run_sweet_spot_analysis()
        print(f"=== All tests completed. Results saved to {self.results_dir}/ ===")

if __name__ == "__main__":
    suite = MasterTestSuite()
    suite.run_all()