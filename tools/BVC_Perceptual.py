import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import sys
import os

def load_wav_mono(filename):
    fs, audio = wavfile.read(filename)
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    return fs, audio.astype(np.float32) / 32768.0

def compute_snr(ref, test):
    # Align lengths
    min_len = min(len(ref), len(test))
    ref = ref[:min_len]
    test = test[:min_len]
    
    noise = ref - test
    signal_power = np.sum(ref**2)
    noise_power = np.sum(noise**2)
    
    if noise_power < 1e-9:
        return 100.0 # Infinite SNR
        
    return 10 * np.log10(signal_power / noise_power)

def compute_segsnr(ref, test, fs, frame_len_ms=20):
    min_len = min(len(ref), len(test))
    ref = ref[:min_len]
    test = test[:min_len]
    
    frame_len = int(fs * frame_len_ms / 1000)
    num_frames = min_len // frame_len
    
    segsnr_acc = 0.0
    count = 0
    
    for i in range(num_frames):
        start = i * frame_len
        end = start + frame_len
        r_frame = ref[start:end]
        t_frame = test[start:end]
        
        signal_power = np.sum(r_frame**2)
        noise_power = np.sum((r_frame - t_frame)**2)
        
        if signal_power > 1e-9 and noise_power > 1e-9:
            snr = 10 * np.log10(signal_power / noise_power)
            # Clamp SNR to realistic range [-10, 35] db to avoid outliers
            snr = max(-10.0, min(35.0, snr))
            segsnr_acc += snr
            count += 1
            
    if count == 0: return 0.0
    return segsnr_acc / count

def compute_lsd(ref, test, fs):
    min_len = min(len(ref), len(test))
    ref = ref[:min_len]
    test = test[:min_len]
    
    # Spectrogram settings
    nperseg = 512
    noverlap = 256
    
    f, t, S_ref = from_scipy_spectrogram(ref, fs, nperseg, noverlap)
    _, _, S_test = from_scipy_spectrogram(test, fs, nperseg, noverlap)
    
    # Log-spectra
    log_S_ref = 10 * np.log10(np.abs(S_ref)**2 + 1e-9)
    log_S_test = 10 * np.log10(np.abs(S_test)**2 + 1e-9)
    
    diff = (log_S_ref - log_S_test)**2
    lsd = np.mean(np.sqrt(np.mean(diff, axis=0)))
    
    return lsd

def from_scipy_spectrogram(x, fs, nperseg, noverlap):
    from scipy.signal import spectrogram
    return spectrogram(x, fs, nperseg=nperseg, noverlap=noverlap)

def compute_spectrogram(sig, fs, nperseg=512, noverlap=256):
    f, t, Sxx = from_scipy_spectrogram(sig, fs, nperseg, noverlap)
    return f, t, 10 * np.log10(Sxx + 1e-9)

def main():
    if len(sys.argv) < 3:
        print("Usage: python BVC_Perceptual.py <original.wav> <decoded.wav>")
        sys.exit(1)
        
    orig_file = sys.argv[1]
    dec_file = sys.argv[2]
    
    if not os.path.exists(orig_file) or not os.path.exists(dec_file):
        print("Error: File(s) not found.")
        sys.exit(1)
        
    print(f"Comparing:\n  Ref: {orig_file}\n  Test: {dec_file}")
    
    fs1, sig1 = load_wav_mono(orig_file)
    fs2, sig2 = load_wav_mono(dec_file)
    
    if fs1 != fs2:
        print(f"Warning: Sample rate mismatch ({fs1} vs {fs2})")
        
    # --- 1. Metrics ---
    snr = compute_snr(sig1, sig2)
    segsnr = compute_segsnr(sig1, sig2, fs1)
    lsd = compute_lsd(sig1, sig2, fs1)
    
    print(f"\nMetrics:")
    print(f"  Global SNR:    {snr:.2f} dB")
    print(f"  Segmental SNR: {segsnr:.2f} dB")
    print(f"  LSD:           {lsd:.2f} dB")
    
    # --- 2. Waveform Comparison ---
    min_len = min(len(sig1), len(sig2))
    t_axis = np.arange(min_len) / fs1
    
    plt.figure(figsize=(12, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(t_axis, sig1[:min_len], 'b', alpha=0.7, label='Original')
    plt.plot(t_axis, sig2[:min_len], 'r', alpha=0.7, label='Decoded')
    plt.title(f'Waveform Comparison (SNR={snr:.2f} dB)')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # --- 3. Spectrograms ---
    f1, t1, S1 = compute_spectrogram(sig1, fs1)
    f2, t2, S2 = compute_spectrogram(sig2, fs2)
    
    plt.subplot(3, 1, 2)
    plt.pcolormesh(t1, f1, S1, shading='gouraud', cmap='magma', vmin=-80, vmax=0)
    plt.title('Original Spectrogram')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar(format='%+2.0f dB')
    
    plt.subplot(3, 1, 3)
    plt.pcolormesh(t2, f2, S2, shading='gouraud', cmap='magma', vmin=-80, vmax=0)
    plt.title('Decoded Spectrogram')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.colorbar(format='%+2.0f dB')
    
    plt.tight_layout()
    
    base, _ = os.path.splitext(dec_file)
    output_plot = f"{base}_perceptual.png"
    plt.savefig(output_plot)
    print(f"\nSaved perceptual plots to: {output_plot}")

if __name__ == "__main__":
    main()
