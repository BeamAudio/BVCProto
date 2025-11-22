import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
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
        os.makedirs(os.path.join(self.results_dir, "complex"))
        os.makedirs(os.path.join(self.results_dir, "synthetic"))

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

    def get_synthetic_dataset(self):
        duration = 1.0
        t = np.arange(int(self.fs * duration)) / self.fs
        signals = {}
        
        # Pure Tone
        signals["Pure_Tone_440Hz"] = np.sin(2 * np.pi * 440 * t)
        
        # Harmonic Series
        sig = np.zeros_like(t)
        for k in range(1, 6):
            sig += (1.0/k) * np.sin(2 * np.pi * 220 * k * t)
        signals["Harmonic_Series_220Hz"] = sig / np.max(np.abs(sig))
        
        # Chirp
        signals["Linear_Chirp"] = signal.chirp(t, f0=20, f1=self.fs/2 - 100, t1=duration, method='linear')
        
        # Transients
        sig = np.zeros_like(t)
        for i in range(10):
            pos = int((0.05 + i*0.1) * self.fs)
            if pos + 50 < len(sig):
                sig[pos:pos+50] += np.exp(-np.linspace(0, 10, 50)) * np.random.randn(50)
        signals["Transients"] = sig
        
        # Silence
        signals["Silence"] = np.zeros_like(t) + 0.0001 * np.random.randn(len(t))
        
        return signals

    def compute_snr(self, ref, test):
        min_len = min(len(ref), len(test))
        ref = ref[:min_len]
        test = test[:min_len]
        noise = ref - test
        signal_power = np.sum(ref**2)
        noise_power = np.sum(noise**2)
        if noise_power < 1e-10: return 100.0
        return 10 * np.log10(signal_power / noise_power)

    # --- Experiments ---
    def run_parameter_sweeps(self):
        print("Running Experiment A: Parameter Sweeps...")
        # Signal for sweeps (complex but short)
        sig = self.generate_complex_signal()[:int(1.0*self.fs)] 
        
        # 1. SNR vs LPC Order (Lossless)
        lpc_orders = [8, 12, 16, 20, 24, 32]
        snrs = []
        for order in lpc_orders:
            codec = BVC_GLPC(self.fs, quantize=False, lpc_order=order)
            frames = codec.process(sig)
            decoded = codec.decode(frames)
            snrs.append(self.compute_snr(sig, decoded))
            
        plt.figure()
        plt.plot(lpc_orders, snrs, 'o-')
        plt.xlabel('LPC Order')
        plt.ylabel('SNR (dB)')
        plt.title('SNR vs LPC Order (Lossless)')
        plt.savefig(os.path.join(self.results_dir, "sweeps", "plot_snr_vs_lpc.png"))
        plt.close()
        
        # 2. SNR vs Bits (Lossy)
        bits_list = [6, 8, 10, 12, 14]
        snrs = []
        for bits in bits_list:
            codec = BVC_GLPC(self.fs, quantize=True, quantizer_config={'LAR_BITS': bits})
            frames = codec.process(sig)
            decoded = codec.decode(frames)
            snrs.append(self.compute_snr(sig, decoded))
            
        plt.figure()
        plt.plot(bits_list, snrs, 's-', color='orange')
        plt.xlabel('LAR Quantization Bits')
        plt.ylabel('SNR (dB)')
        plt.title('SNR vs Quantization Bits')
        plt.savefig(os.path.join(self.results_dir, "sweeps", "plot_snr_vs_bits.png"))
        plt.close()

    def run_complex_analysis(self):
        print("Running Experiment B: Complex Signal Analysis...")
        sig = self.generate_complex_signal()
        codec = BVC_GLPC(self.fs, quantize=True, quantizer_config={'LAR_BITS': 10})
        frames = codec.process(sig)
        decoded = codec.decode(frames)
        
        # Spectrogram & Frame Analysis
        plt.figure(figsize=(12, 8))
        
        plt.subplot(3, 1, 1)
        plt.specgram(sig, NFFT=1024, Fs=self.fs, noverlap=512)
        plt.title('Original Spectrogram')
        
        plt.subplot(3, 1, 2)
        plt.specgram(decoded, NFFT=1024, Fs=self.fs, noverlap=512)
        plt.title('Reconstructed Spectrogram')
        
        plt.subplot(3, 1, 3)
        plt.plot(np.arange(len(sig))/self.fs, sig, 'k-', alpha=0.3)
        curr_sample = 0
        for f in frames:
            length = f['merge'] * 256
            color = 'r' if f['mode'] == 1 else 'b'
            alpha = 0.5 if f['mode'] == 1 else 0.1
            plt.axvline(x=curr_sample/self.fs, color=color, alpha=alpha)
            curr_sample += length
        plt.title('Frame Boundaries (Blue=Normal, Red=Transient)')
        plt.xlim(0, 5.0)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "complex", "analysis_complex.png"))
        plt.close()

    def run_synthetic_batch(self):
        print("Running Experiment C: Synthetic Dataset Batch...")
        signals = self.get_synthetic_dataset()
        codec = BVC_GLPC(self.fs, quantize=True, quantizer_config={'LAR_BITS': 10})
        
        for name, sig in signals.items():
            frames = codec.process(sig)
            decoded = codec.decode(frames)
            
            min_len = min(len(sig), len(decoded))
            sig = sig[:min_len]
            decoded = decoded[:min_len]
            
            # Increased figure height for better spacing
            plt.figure(figsize=(10, 15))
            
            # Spectrograms
            Pxx, freq, t, _ = plt.specgram(sig, NFFT=1024, Fs=self.fs, noverlap=512)
            plt.clf() # Clear to plot properly in subplots
            
            # 1. Original Spectrogram
            plt.subplot(5, 1, 1)
            plt.specgram(sig, NFFT=1024, Fs=self.fs, noverlap=512)
            plt.title(f'{name}: Original')
            
            # 2. Reconstructed Spectrogram
            plt.subplot(5, 1, 2)
            Pxx_rec, _, _, _ = plt.specgram(decoded, NFFT=1024, Fs=self.fs, noverlap=512)
            plt.title(f'{name}: Reconstructed')
            
            # 3. Difference
            plt.subplot(5, 1, 3)
            min_t = min(Pxx.shape[1], Pxx_rec.shape[1])
            diff = 10 * np.log10(Pxx_rec[:, :min_t] + 1e-9) - 10 * np.log10(Pxx[:, :min_t] + 1e-9)
            plt.imshow(diff, aspect='auto', origin='lower', 
                       extent=[0, 1.0, 0, self.fs/2], vmin=-20, vmax=20, cmap='seismic')
            plt.colorbar(label='dB')
            plt.title('Spectral Difference')

            # 4. Time Domain
            plt.subplot(5, 1, 4)
            t_axis = np.arange(len(sig)) / self.fs
            plt.plot(t_axis, sig, 'k-', alpha=0.5, label='Original')
            plt.plot(t_axis, decoded, 'r--', alpha=0.5, label='Reconstructed')
            plt.legend()
            plt.title('Time Domain Waveform')
            plt.xlabel('Time (s)')
            
            # 5. Frame Size / Mode Visualization
            plt.subplot(5, 1, 5)
            
            # Reconstruct time axis for frames
            frame_times = []
            frame_sizes = []
            curr_time = 0
            
            for f in frames:
                # Duration in seconds
                duration = (f['merge'] * 256) / self.fs 
                frame_times.append(curr_time)
                frame_sizes.append(f['merge'] * 256)
                
                # Add end point for step plot
                frame_times.append(curr_time + duration)
                frame_sizes.append(f['merge'] * 256)
                
                curr_time += duration
                
            plt.plot(frame_times, frame_sizes, 'b-', linewidth=2)
            plt.fill_between(frame_times, frame_sizes, alpha=0.3, color='blue')
            plt.title('Dynamic Frame Sizes (Codec Adaptation)')
            plt.ylabel('Frame Size (samples)')
            plt.xlabel('Time (s)')
            plt.grid(True, alpha=0.3)
            
            plt.subplots_adjust(hspace=0.6)
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, "synthetic", f"analysis_{name}.png"))
            plt.close()

    def run_all(self):
        print("=== Starting Master Test Suite ===")
        self.run_parameter_sweeps()
        self.run_complex_analysis()
        self.run_synthetic_batch()
        print(f"=== All tests completed. Results saved to {self.results_dir}/ ===")

if __name__ == "__main__":
    suite = MasterTestSuite()
    suite.run_all()
