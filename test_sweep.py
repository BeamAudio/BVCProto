import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from BVC import BVC_GLPC
import os

def compute_snr(ref, test):
    min_len = min(len(ref), len(test))
    ref = ref[:min_len]
    test = test[:min_len]
    
    noise = ref - test
    signal_pow = np.sum(ref**2)
    noise_pow = np.sum(noise**2)
    
    if noise_pow == 0: return 100.0
    return 10 * np.log10(signal_pow / noise_pow)

def compute_lsd(ref, test, fs=44100):
    min_len = min(len(ref), len(test))
    ref = ref[:min_len]
    test = test[:min_len]
    
    # STFT parameters
    nperseg = 2048
    noverlap = 1024
    
    from scipy.signal import stft
    f, t, Zxx_ref = stft(ref, fs, nperseg=nperseg, noverlap=noverlap)
    f, t, Zxx_test = stft(test, fs, nperseg=nperseg, noverlap=noverlap)
    
    # Log Power Spectra
    P_ref = np.log10(np.abs(Zxx_ref)**2 + 1e-10)
    P_test = np.log10(np.abs(Zxx_test)**2 + 1e-10)
    
    # LSD
    diff = (P_ref - P_test)**2
    lsd = np.mean(np.sqrt(np.mean(diff, axis=0)))
    
    return lsd

def main():
    input_file = "Recording63.wav"
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    print(f"Loading {input_file}...")
    fs, audio = wavfile.read(input_file)
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    audio = audio.astype(np.float32) / 32768.0

    # Sweep Parameters
    max_merge_values = [4, 8, 16, 32]
    snr_results = []
    lsd_results = []
    
    print("Starting Sweep...")
    
    for mm in max_merge_values:
        print(f"Testing max_merge = {mm}...")
        
        # Initialize Codec
        codec = BVC_GLPC(fs=fs, quantize=True, lpc_order=16)
        codec.max_merge = mm # Override max_merge
        
        # Encode
        frames = codec.process(audio)
        
        # Decode
        recon = codec.decode(frames)
        
        # Metrics
        snr = compute_snr(audio, recon)
        lsd = compute_lsd(audio, recon, fs)
        
        snr_results.append(snr)
        lsd_results.append(lsd)
        
        print(f"  -> SNR: {snr:.2f} dB, LSD: {lsd:.2f}")

    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Upper Frame Limit (max_merge)')
    ax1.set_ylabel('SNR (dB)', color=color)
    ax1.plot(max_merge_values, snr_results, color=color, marker='o', label='SNR')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True)

    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('LSD', color=color)  
    ax2.plot(max_merge_values, lsd_results, color=color, marker='s', linestyle='--', label='LSD')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('BVC Performance vs Frame Limit (Recording63.wav)')
    fig.tight_layout()  
    
    output_plot = "artifacts/sweep_results.png"
    os.makedirs("artifacts", exist_ok=True)
    plt.savefig(output_plot)
    print(f"Plot saved to {output_plot}")

if __name__ == "__main__":
    main()
