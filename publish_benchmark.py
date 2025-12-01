import os
import sys
import subprocess
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import wavfile
from scipy import signal

# --- Configuration ---
SAMPLE_FILE = "samples/Recording63.wav"
OUTPUT_DIR = "benchmark_results"
CPP_BUILD_DIR = "cpp/build/Release" # Adjust for Linux/Mac if needed
CPP_EXE = os.path.join(CPP_BUILD_DIR, "bvc.exe") if os.name == 'nt' else os.path.join(CPP_BUILD_DIR, "bvc")

# Ensure output dir exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- Metrics Functions ---

def load_wav(filename):
    fs, audio = wavfile.read(filename)
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    return fs, audio.astype(np.float32) / 32768.0

def compute_snr(ref, test):
    min_len = min(len(ref), len(test))
    ref = ref[:min_len]
    test = test[:min_len]
    noise = ref - test
    s = np.sum(ref**2)
    n = np.sum(noise**2)
    if n < 1e-9: return 100.0
    return 10 * np.log10(s / n)

def compute_segsnr(ref, test, fs, frame_ms=20):
    min_len = min(len(ref), len(test))
    ref = ref[:min_len]
    test = test[:min_len]
    N = int(fs * frame_ms / 1000)
    segsnr = 0
    count = 0
    for i in range(0, min_len - N, N):
        r = ref[i:i+N]
        t = test[i:i+N]
        s = np.sum(r**2)
        n = np.sum((r-t)**2)
        if s > 1e-9 and n > 1e-9:
            snr = 10 * np.log10(s/n)
            snr = max(-10, min(35, snr))
            segsnr += snr
            count += 1
    return segsnr / max(1, count)

def compute_lsd(ref, test, fs):
    f, t, S_ref = signal.spectrogram(ref, fs, nperseg=512, noverlap=256)
    _, _, S_test = signal.spectrogram(test, fs, nperseg=512, noverlap=256)
    
    # Align time dimensions
    min_t = min(S_ref.shape[1], S_test.shape[1])
    S_ref = S_ref[:, :min_t]
    S_test = S_test[:, :min_t]
    
    log_ref = 10*np.log10(np.abs(S_ref)**2 + 1e-9)
    log_test = 10*np.log10(np.abs(S_test)**2 + 1e-9)
    diff = (log_ref - log_test)**2
    lsd = np.mean(np.sqrt(np.mean(diff, axis=0)))
    return lsd

# --- Benchmarking ---

def run_benchmark():
    print("=== Starting BVC Comprehensive Benchmark ===")
    
    # 1. Encoding Speed & Bitrate
    bvc_out = os.path.join(OUTPUT_DIR, "bench.bvc")
    wav_out = os.path.join(OUTPUT_DIR, "bench.wav")
    
    start_t = time.time()
    try:
        cmd = [CPP_EXE, "encode", SAMPLE_FILE, bvc_out]
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL)
    except Exception as e:
        print(f"Error running encoder: {e}")
        return
    enc_time = time.time() - start_t
    
    # 2. Decoding Speed
    start_t = time.time()
    try:
        cmd = [CPP_EXE, "decode", bvc_out, wav_out]
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL)
    except Exception as e:
        print(f"Error running decoder: {e}")
        return
    dec_time = time.time() - start_t
    
    # 3. Analysis
    fs, sig_ref = load_wav(SAMPLE_FILE)
    _, sig_test = load_wav(wav_out)
    
    duration = len(sig_ref) / fs
    file_size = os.path.getsize(bvc_out)
    bitrate = (file_size * 8) / duration / 1000.0
    
    snr = compute_snr(sig_ref, sig_test)
    segsnr = compute_segsnr(sig_ref, sig_test, fs)
    lsd = compute_lsd(sig_ref, sig_test, fs)
    
    rtf_enc = enc_time / duration
    rtf_dec = dec_time / duration
    
    print(f"\n--- Results ---")
    print(f"Bitrate:       {bitrate:.2f} kbps")
    print(f"Global SNR:    {snr:.2f} dB")
    print(f"Segmental SNR: {segsnr:.2f} dB")
    print(f"LSD:           {lsd:.2f} dB")
    print(f"Enc RTF:       {rtf_enc:.4f}x (Speed: {1/rtf_enc:.1f}x)")
    print(f"Dec RTF:       {rtf_dec:.4f}x (Speed: {1/rtf_dec:.1f}x)")
    
    # --- Plotting ---
    plt.style.use('bmh') # Better visuals
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"BVC Performance Analysis\n{bitrate:.1f} kbps | SNR: {snr:.1f}dB | SegSNR: {segsnr:.1f}dB | LSD: {lsd:.2f}dB", fontsize=14)
    
    # 1. Waveform Compare
    ax1 = plt.subplot(3, 1, 1)
    t = np.arange(len(sig_ref)) / fs
    ax1.plot(t, sig_ref, 'k', alpha=0.5, linewidth=0.8, label='Original')
    # Align check: simple shift is not done here, assumes sample-accurate
    min_n = min(len(t), len(sig_test))
    ax1.plot(t[:min_n], sig_test[:min_n], 'r--', alpha=0.6, linewidth=0.8, label='Decoded')
    ax1.set_title("Waveform Comparison")
    ax1.set_ylabel("Amplitude")
    ax1.legend()
    ax1.set_xlim(0, duration)
    
    # 2. Spectrograms (Side by Side)
    f, t_spec, S_ref = signal.spectrogram(sig_ref, fs, nperseg=1024, noverlap=512)
    _, t_spec_test, S_test = signal.spectrogram(sig_test[:len(sig_ref)], fs, nperseg=1024, noverlap=512)

    # Make sure dimensions match
    min_time_frames = min(S_ref.shape[1], S_test.shape[1])
    S_ref = S_ref[:, :min_time_frames]
    S_test = S_test[:, :min_time_frames]
    t_spec = t_spec[:min_time_frames]  # Trim time axis to match

    ax2 = plt.subplot(3, 2, 3)
    ax2.pcolormesh(t_spec, f, 10*np.log10(S_ref+1e-9), cmap='inferno', vmin=-80, vmax=0, shading='gouraud')
    ax2.set_title("Original Spectrogram")
    ax2.set_ylabel("Frequency (Hz)")

    ax3 = plt.subplot(3, 2, 4)
    im = ax3.pcolormesh(t_spec, f, 10*np.log10(S_test+1e-9), cmap='inferno', vmin=-80, vmax=0, shading='gouraud')
    ax3.set_title("Decoded Spectrogram")
    ax3.set_ylabel("Frequency (Hz)")
    
    # 3. Error / Residual
    ax4 = plt.subplot(3, 1, 3)
    # Compute Error signal
    error = sig_ref[:min_n] - sig_test[:min_n]
    ax4.plot(t[:min_n], error, 'gray', linewidth=0.5)
    ax4.set_title("Reconstruction Error (Residual)")
    ax4.set_xlabel("Time (s)")
    ax4.set_xlim(0, duration)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = os.path.join(OUTPUT_DIR, "bvc_benchmark_report.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\nReport saved to: {plot_path}")
    plt.close()

    # --- Bitrate Pie Chart ---
    # We can't get per-frame stats easily from here without parsing BVC file again
    # But we can do a simple bar chart of metrics if we had comparison data.
    # For now, just the summary report is great.

if __name__ == "__main__":
    run_benchmark()
