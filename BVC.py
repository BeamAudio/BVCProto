import numpy as np
import scipy.signal as signal
import struct
import io
from numba import jit, prange
from joblib import Parallel, delayed, cpu_count

# ==============================================================================
# GLOBAL CACHE & WORKER FUNCTIONS
# ==============================================================================

_DICT_CACHE = {}
_GRAM_CACHE = {}
_HANN_CACHE = {}

# Constants for Modes
MODE_SILENCE = 0
MODE_VOICED = 1
MODE_UNVOICED = 2

def _get_hann_window_global(N):
    if N not in _HANN_CACHE:
        _HANN_CACHE[N] = np.hanning(N).astype(np.float32)
    return _HANN_CACHE[N]

@jit(nopython=True, fastmath=True, cache=True)
def _autocorr_fast(x, w, order, lag_window):
    n = len(x)
    xw = x * w
    r = np.zeros(order + 1, dtype=np.float64)
    for lag in range(order + 1):
        acc = 0.0
        for i in range(n - lag):
            acc += xw[i] * xw[i + lag]
        r[lag] = acc
    return r * lag_window

@jit(nopython=True, fastmath=True, cache=True)
def _levinson_durbin(r, order):
    a = np.zeros(order + 1, dtype=np.float64)
    k = np.zeros(order, dtype=np.float64)
    a[0] = 1.0
    e = r[0] + 1e-12
    
    for i in range(order):
        acc = 0.0
        for j in range(i + 1):
            acc += a[j] * r[i + 1 - j]
        
        ki = -acc / e
        # Strict clipping for stability
        if ki > 0.99: ki = 0.99
        if ki < -0.99: ki = -0.99
        k[i] = ki
        
        a_prev = a.copy()
        for j in range(i + 1):
            a[j + 1] = a_prev[j + 1] + ki * a_prev[i - j]
        e = e * (1.0 - ki * ki)
        
    return a, k

@jit(nopython=True, fastmath=True, cache=True)
def _rc_to_lpc(k):
    order = len(k)
    a = np.zeros(order + 1, dtype=np.float64)
    a[0] = 1.0
    for i in range(order):
        ki = k[i]
        a_prev = a.copy()
        for j in range(i + 1):
            a[j + 1] = a_prev[j + 1] + ki * a_prev[i - j]
    return a

@jit(nopython=True, fastmath=True, cache=True)
def _mp_fast_loop(work, D, G, n_atoms, threshold):
    if D.shape[1] == 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float32)

    correlations = D.T @ work 
    indices = np.zeros(n_atoms, dtype=np.int32)
    coeffs = np.zeros(n_atoms, dtype=np.float32)
    current_energy = np.sum(work**2)
    count = 0
    
    for i in range(n_atoms):
        best_idx = np.argmax(np.abs(correlations))
        coeff = correlations[best_idx]
        
        if abs(coeff) < 1e-6: break
        
        indices[i] = best_idx
        coeffs[i] = coeff
        count += 1
        
        correlations -= coeff * G[:, best_idx]
        
        current_energy -= coeff**2
        if current_energy < threshold: break
        
    return indices[:count], coeffs[:count]

def _get_gabor_dict_global(N, fs, num_freqs, mode):
    """
    Generates Gabor Dictionary based on Mode:
    - VOICED: Harmonic/Tonal structure (Longer atoms, Log Freqs)
    - UNVOICED: Noise/Transient structure (Short atoms, Dense Shifts)
    - SILENCE: Empty (should not be called usually)
    """
    cache_key = (N, mode, num_freqs)
    if cache_key in _DICT_CACHE:
        return _DICT_CACHE[cache_key], _GRAM_CACHE[cache_key]
    
    t = np.arange(N, dtype=np.float32)
    
    # --- Dictionary Definition ---
    
    if mode == MODE_UNVOICED:
        # UNVOICED / TRANSIENT
        # Hybrid Dictionary: Short Gabor Atoms + DCT Basis
        
        # 1. Short Gabor Atoms (Clicks/Transients)
        scales = np.array([2, 4, 8], dtype=np.float32) 
        shifts = np.arange(0, N, 4) 
        
        # 2. High Freq Modulated Atoms (Fricatives)
        freqs_hz = np.linspace(2500, fs/2 - 500, 8)
        freqs = 2 * np.pi * freqs_hz / fs
        freqs = np.concatenate(([0], freqs))
        
        # 3. DCT Basis (Stationary Noise/Texture)
        # Discrete Cosine Transform basis vectors
        k_dct = np.arange(1, N, 2) # Skip DC, sparse coverage
        if len(k_dct) > 128: k_dct = k_dct[:128] # Limit size
        
        # We will generate DCT atoms in the generation loop below
        # But first, setup standard params for Gabor part
        
    elif mode == MODE_VOICED:
        # ... (rest is same) ...
        # VOICED / TONAL
        # Emphasis on frequency resolution
        min_s = 64 # Minimum scale slightly larger for voiced
        num_scales = 5
        scales = min_s * (2 ** np.arange(num_scales)) 
        scales = scales[scales < N]
        if len(scales) == 0: scales = np.array([N/2])
        
        shifts = np.arange(0, N, max(16, N//16)) # Coarser shift
        
        # Logarithmic Frequency spacing
        # Dense packing in speech range (100Hz - 4kHz)
        min_f = 50
        max_f = fs / 2 - 100
        if num_freqs < 64: num_freqs = 64 # Safety lower bound
        
        freqs_hz = np.logspace(np.log10(min_f), np.log10(max_f), num_freqs)
        freqs = 2 * np.pi * freqs_hz / fs
        
    else:
        # Silence or unknown
        D = np.zeros((N, 0), dtype=np.float32)
        G = np.zeros((0, 0), dtype=np.float32)
        _DICT_CACHE[cache_key] = D
        _GRAM_CACHE[cache_key] = G
        return D, G

    # --- Generation ---
    atoms_list = []
    
    if mode == MODE_UNVOICED:
        # A. Gabor Part
        valid_envelopes = []
        for s in scales:
            for u in shifts:
                env = np.exp(-0.5 * ((t - u) / s)**2)
                if env.max() > 0.01:
                    valid_envelopes.append(env)
        Envelopes = np.array(valid_envelopes)
        
        for f in freqs:
            mod = np.cos(f * t)
            for env in Envelopes:
                atom = env * mod
                norm = np.sqrt(np.sum(atom**2))
                if norm > 1e-6:
                    atoms_list.append(atom / norm)
                    
        # B. DCT Part
        # DCT-II: cos(pi * k * (2n+1) / (2N))
        for k in k_dct:
            atom = np.cos(np.pi * k * (2*t + 1) / (2*N))
            norm = np.sqrt(np.sum(atom**2))
            if norm > 1e-6:
                atoms_list.append(atom / norm)

    elif mode == MODE_VOICED:
        # 1. Generate Envelopes
        valid_envelopes = []
        for s in scales:
            for u in shifts:
                env = np.exp(-0.5 * ((t - u) / s)**2)
                if env.max() > 0.01:
                    valid_envelopes.append(env)
        
        Envelopes = np.array(valid_envelopes)
        
        # 2. Modulate
        Cos_Mod = np.cos(freqs[:, None] * t[None, :])
        Sin_Mod = np.sin(freqs[:, None] * t[None, :])
        
        batch_size = 50
        for i in range(0, len(Envelopes), batch_size):
            batch_env = Envelopes[i:i+batch_size]
            
            for env in batch_env:
                atoms_c = env * Cos_Mod
                atoms_s = env * Sin_Mod
                
                norms_c = np.sqrt(np.sum(atoms_c**2, axis=1))
                norms_s = np.sqrt(np.sum(atoms_s**2, axis=1))
                
                valid_c = norms_c > 1e-6
                if np.any(valid_c):
                    atoms_list.append(atoms_c[valid_c] / norms_c[valid_c, None])
                
                valid_s = norms_s > 1e-6
                if np.any(valid_s):
                    atoms_list.append(atoms_s[valid_s] / norms_s[valid_s, None])

    if len(atoms_list) > 0:
        # Flatten
        D = np.array(atoms_list).T if mode == MODE_UNVOICED else np.vstack(atoms_list).T
    else:
        D = np.zeros((N, 1), dtype=np.float32)

    D = D.astype(np.float32)
    
    # --- Size Limit & Downsampling ---
    limit = 4096 if mode == MODE_VOICED else 2048
    if D.shape[1] > limit:
        step = D.shape[1] // limit + 1
        D = D[:, ::step]
        
    G = D.T @ D
    
    _DICT_CACHE[cache_key] = D
    _GRAM_CACHE[cache_key] = G
    return D, G

def _least_squares_refinement(work, D, indices):
    """
    Refines coefficients for selected atoms using Least Squares.
    """
    if len(indices) == 0:
        return np.array([], dtype=np.float32)
        
    D_sub = D[:, indices]
    c_new, _, _, _ = np.linalg.lstsq(D_sub, work, rcond=None)
    return c_new.astype(np.float32)

def _mp_worker_batch(tasks):
    """
    Worker function for batched parallel Matching Pursuit.
    tasks: List of tuples (masked_residual, total_len, mode, q_energy, merge_count, num_freqs, fs)
    """
    results = []
    
    for task in tasks:
        masked_residual, total_len, mode, q_energy, merge_count, num_freqs, fs = task
        
        if mode == MODE_SILENCE or q_energy == 0:
            results.append([])
            continue

        D, G = _get_gabor_dict_global(total_len, fs, num_freqs, mode=mode)
        
        if D.shape[1] == 0:
            results.append([])
            continue

        res_energy = np.sum(masked_residual**2)
        # Threshold relaxed to 0.05 (95% energy preservation)
        # Pre-emphasis ensures this energy loss is not just high-freq loss
        threshold = res_energy * 0.05 
        
        # Atoms allowance: keeping for speed
        max_atoms = 16 if mode == MODE_UNVOICED else (10 * merge_count)
        
        indices, coeffs = _mp_fast_loop(
            masked_residual.astype(np.float32), D, G, max_atoms, threshold
        )
        
        # --- Least Squares Optimization (Boosts SNR) ---
        if len(indices) > 0:
            coeffs = _least_squares_refinement(masked_residual.astype(np.float32), D, indices)
        
        atoms_list = []
        for idx, c in zip(indices, coeffs):
            atoms_list.append((idx, c))
        results.append(atoms_list)
        
    return results

class EncodedFrames(list):
    def __init__(self, frames=None, original_length=None):
        super().__init__(frames if frames is not None else [])
        self.original_length = original_length

class BVC_GLPC:
    def __init__(self, fs=44100, quantize=False, lpc_order=16, quantizer_config=None, max_merge=64, num_freqs=256):
        self.fs = fs
        self.base_N = 256
        self.max_merge = max_merge
        if self.max_merge > 127:
            raise ValueError("max_merge cannot exceed 127")
        self.num_freqs = num_freqs # Requested dictionary resolution
        self.quantize = quantize
        self.lpc_order = lpc_order
        self.overlap = 32 
        # Bandwidth expansion
        self.gamma = np.exp(-0.5 * (2.0 * np.pi * 60.0 / self.fs)) 
        self.lag_window = self.gamma ** np.arange(self.lpc_order + 1)
        
        if self.quantize:
            from BVCQuantizer import BVCQuantizer
            self.quantizer = BVCQuantizer(config=quantizer_config)
        else:
            self.quantizer = None

    def _quantize_energy(self, energy):
        return self.quantizer.quantize_energy(energy) if self.quantizer else energy

    def _dequantize_energy(self, q):
        return self.quantizer.dequantize_energy(q) if self.quantizer else q

    def _quantize_lpc(self, k_coeffs):
        return self.quantizer.quantize_lpc(k_coeffs) if self.quantizer else k_coeffs

    def _dequantize_lpc(self, q_indices):
        return self.quantizer.dequantize_lpc(q_indices) if self.quantizer else q_indices
    
    def _rc_to_lpc_instance(self, k):
        return _rc_to_lpc(k)

    def process(self, sig):
        sig = sig.astype(np.float32)
        
        # Pre-emphasis (boost highs for better LPC/MP modeling)
        sig = np.append(sig[0], sig[1:] - 0.97 * sig[:-1])
        
        N = self.base_N
        
        pad_len = (N - (len(sig) % N)) % N
        sig = np.pad(sig, (0, pad_len + self.overlap))
        
        num_frames = (len(sig) - self.overlap) // N
        frames_matrix = sig[:num_frames*N].reshape(num_frames, N)
        
        zi_pre = np.zeros(self.lpc_order, dtype=np.float32)
        
        mp_tasks = []
        frame_metadata = [] 
        
        i = 0
        while i < num_frames:
            # --- 1. Mode Detection ---
            curr_frame = frames_matrix[i]
            energy = np.sum(curr_frame**2)
            
            # Zero Crossing Rate
            zcr = np.sum(np.abs(np.diff(np.signbit(curr_frame)))) / N
            
            # Crest Factor
            peak = np.max(np.abs(curr_frame))
            rms = np.sqrt(energy/N)
            crest = peak / (rms + 1e-9)
            
            # Classification
            if rms < 0.002: # Slightly higher silence threshold
                mode = MODE_SILENCE
                # Aggressive merging for silence
                merge_count = 1
                for k in range(1, min(self.max_merge, num_frames - i)):
                     next_rms = np.sqrt(np.sum(frames_matrix[i+k]**2)/N)
                     if next_rms < 0.005: # Stay in silence
                         merge_count += 1
                     else:
                         break
            else:
                # High ZCR or High Crest -> Unvoiced
                # ZCR > 0.35 is a typical threshold for fricatives
                if zcr > 0.35 or crest > 6.0:
                    mode = MODE_UNVOICED
                    merge_count = 1 
                else:
                    mode = MODE_VOICED
                    merge_count = 1
                    # Attempt merge for Voiced segments
                    for k in range(1, min(self.max_merge, num_frames - i)):
                        next_rms = np.sqrt(np.sum(frames_matrix[i+k]**2)/N)
                        ratio = next_rms / (rms + 1e-9)
                        next_zcr = np.sum(np.abs(np.diff(np.signbit(frames_matrix[i+k])))) / N
                        
                        # Relaxed merging criteria for Perceptual/Bitrate optimization
                        # Allow 0.4 - 2.5 amplitude variance
                        if (0.4 < ratio < 2.5) and (next_zcr < 0.4) and (abs(next_zcr - zcr) < 0.2): 
                            merge_count += 1
                        else:
                            break
            
            core_len = merge_count * N
            total_len = core_len + self.overlap
            start_idx = i * N
            raw_audio = sig[start_idx : start_idx + total_len]
            
            # --- 2. LPC ---
            audio_core = raw_audio[:core_len]
            w = _get_hann_window_global(len(audio_core))
            r = _autocorr_fast(audio_core, w, self.lpc_order, self.lag_window)
            _, k_raw = _levinson_durbin(r, self.lpc_order)
            
            q_lar = self._quantize_lpc(k_raw)
            k_quant = self._dequantize_lpc(q_lar)
            a_quant = _rc_to_lpc(k_quant)
            
            # --- 3. Inverse Filter ---
            res_core, zi_next = signal.lfilter(a_quant, [1.0], raw_audio[:core_len], zi=zi_pre)
            res_overlap, _ = signal.lfilter(a_quant, [1.0], raw_audio[core_len:], zi=zi_next.copy())
            residual = np.concatenate([res_core, res_overlap])
            zi_pre = zi_next
            
            # --- 4. Windowing ---
            window = np.ones(total_len, dtype=np.float32)
            t_fade = np.arange(self.overlap)
            window[-self.overlap:] = np.cos(0.5 * np.pi * t_fade / self.overlap)**2
            if i > 0:
                window[:self.overlap] = np.sin(0.5 * np.pi * t_fade / self.overlap)**2
            
            masked_residual = residual * window
            res_energy = np.sum(masked_residual**2)
            q_energy = self._quantize_energy(res_energy)
            
            frame_metadata.append({
                'mode': mode,
                'merge': merge_count,
                'q_energy': q_energy,
                'q_lar': q_lar
            })
            
            mp_tasks.append((masked_residual, total_len, mode, q_energy, merge_count, self.num_freqs, self.fs))
            
            i += merge_count
            
        # --- Phase 2: Batched Parallel Execution ---
        batch_size = 64 
        task_batches = [mp_tasks[i:i + batch_size] for i in range(0, len(mp_tasks), batch_size)]
        
        batch_results = Parallel(n_jobs=-1, verbose=0)(
            delayed(_mp_worker_batch)(batch) for batch in task_batches
        )
        
        flat_atoms = [atoms for batch in batch_results for atoms in batch]
        
        # --- Phase 3: Assembly ---
        encoded_frames = []
        for meta, atoms in zip(frame_metadata, flat_atoms):
            meta['atoms'] = atoms
            encoded_frames.append(meta)
            
        return EncodedFrames(encoded_frames, original_length=len(sig) - pad_len - self.overlap)

    def save_to_file(self, filename, encoded_frames):
        with open(filename, 'wb') as f:
            f.write(b'RBVC')
            version = 1 if self.quantize else 0
            f.write(struct.pack('B', version))
            orig_len = getattr(encoded_frames, 'original_length', 0) or 0
            f.write(struct.pack('<III', self.fs, len(encoded_frames), orig_len))
            
            for frame in encoded_frames:
                m = frame['mode']
                mg = frame['merge']
                if mg > 63: mg = 63 
                
                flags = (m << 6) | (mg & 0x3F)
                f.write(struct.pack('B', flags))
                
                if self.quantize:
                    f.write(struct.pack('B', frame['q_energy']))
                else:
                    f.write(struct.pack('<f', frame['q_energy']))
                
                n_atoms = len(frame['atoms'])
                f.write(struct.pack('<H', n_atoms))
                
                if self.quantize:
                    f.write(struct.pack('<16h', *frame['q_lar']))
                else:
                    f.write(struct.pack('<16f', *frame['q_lar']))
                
                # Atom Coefficients Quantization
                if self.quantize and n_atoms > 0:
                    # Extract coeffs
                    indices = [a[0] for a in frame['atoms']]
                    coeffs = np.array([a[1] for a in frame['atoms']], dtype=np.float32)
                    
                    # Quantize
                    gain, q_coeffs = self.quantizer.quantize_atom_coeffs_frame(coeffs)
                    
                    # Write Gain (float32 for now, could be float16)
                    f.write(struct.pack('<f', gain))
                    
                    # Write Indices (ushort) + Coeffs (int8)
                    # Interleaved? Or block? Block is better for compression but let's stick to sequence
                    for idx, qc in zip(indices, q_coeffs):
                        f.write(struct.pack('<Hb', idx, qc))
                        
                elif not self.quantize and n_atoms > 0:
                    # Lossless (float)
                    for idx, coef in frame['atoms']:
                        f.write(struct.pack('<Hf', idx, coef))

    def load_from_file(self, filename):
        with open(filename, 'rb') as f:
            magic = f.read(4)
            if magic != b'RBVC':
                raise ValueError("Invalid File Format")
            
            version_byte = f.read(1)
            if len(version_byte) == 0:
                f.seek(4)
                version = 1 
                fs, num_frames = struct.unpack('<II', f.read(8))
            else:
                version = struct.unpack('B', version_byte)[0]
                header_data = f.read(12)
                if len(header_data) == 12:
                    fs, num_frames, orig_len = struct.unpack('<III', header_data)
                elif len(header_data) == 8:
                    fs, num_frames = struct.unpack('<II', header_data)
                    orig_len = 0
                else:
                    raise ValueError("Invalid Header")
            
            self.fs = fs
            self.quantize = (version == 1)
            if self.quantize and self.quantizer is None:
                from BVCQuantizer import BVCQuantizer
                self.quantizer = BVCQuantizer()
            
            frames = []
            for _ in range(num_frames):
                flags_byte = f.read(1)
                if not flags_byte: break
                flags = struct.unpack('B', flags_byte)[0]
                
                mode = (flags >> 6) & 0x03
                merge = flags & 0x3F
                if merge == 0: merge = 1 
                
                if version == 1:
                    q_energy = struct.unpack('B', f.read(1))[0]
                else:
                    q_energy = struct.unpack('<f', f.read(4))[0]
                
                n_atoms = struct.unpack('<H', f.read(2))[0]
                
                if version == 1:
                    q_lar = list(struct.unpack('<16h', f.read(32)))
                else:
                    q_lar = list(struct.unpack('<16f', f.read(64)))
                
                atoms = []
                
                if n_atoms > 0:
                    if version == 1: # Quantized
                        # Read Gain
                        gain = struct.unpack('<f', f.read(4))[0]
                        # Read Atoms
                        for _ in range(n_atoms):
                            idx, qc = struct.unpack('<Hb', f.read(3))
                            # Dequantize immediately
                            coef = float(qc) * (gain / 127.0)
                            atoms.append((idx, coef))
                    else: # Lossless
                        for _ in range(n_atoms):
                            idx, coef = struct.unpack('<Hf', f.read(6))
                            atoms.append((idx, coef))
                    
                if len(q_lar) > 0:
                    if version == 1:
                        q_lar_array = np.array(q_lar, dtype=np.int16)
                    else:
                        q_lar_array = np.array(q_lar, dtype=np.float32)
                else:
                    q_lar_array = np.array([], dtype=np.float32)

                frames.append({
                    'mode': mode,
                    'merge': merge,
                    'q_energy': q_energy,
                    'q_lar': q_lar_array,
                    'atoms': atoms
                })
                
            return EncodedFrames(frames, original_length=orig_len)

    def decode(self, encoded_frames):
        zi = np.zeros(self.lpc_order, dtype=np.float32)
        total_samples = sum(f['merge'] * self.base_N for f in encoded_frames)
        residual_buffer = np.zeros(total_samples + self.overlap, dtype=np.float32)
        
        current_pos = 0
        output_audio = []
        
        for frame in encoded_frames:
            mode = frame['mode']
            merge = frame['merge']
            
            k_quant = self._dequantize_lpc(frame['q_lar'])
            a_quant = _rc_to_lpc(k_quant)
            
            core_len = merge * self.base_N
            total_len = core_len + self.overlap
            
            excitation_block = np.zeros(total_len, dtype=np.float32)
            
            # Use global dict gen
            D, _ = _get_gabor_dict_global(total_len, self.fs, self.num_freqs, mode=mode)
            
            # Reconstruct atoms
            if D.shape[1] > 0:
                for idx, coef in frame['atoms']:
                    if idx < D.shape[1]:
                        excitation_block += coef * D[:, idx]
            
            end_pos = current_pos + total_len
            if end_pos <= len(residual_buffer):
                residual_buffer[current_pos : end_pos] += excitation_block
            
            valid_residual = residual_buffer[current_pos : current_pos + core_len]
            
            synth_block, zi = signal.lfilter([1.0], a_quant, valid_residual, zi=zi)
            output_audio.append(synth_block)
            current_pos += core_len
            
        output = np.concatenate(output_audio)
        
        # De-emphasis
        output = signal.lfilter([1], [1, -0.97], output)
        
        if hasattr(encoded_frames, 'original_length') and encoded_frames.original_length:
            if encoded_frames.original_length < len(output):
                output = output[:encoded_frames.original_length]
                
        return output
