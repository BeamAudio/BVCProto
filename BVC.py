import numpy as np
import scipy.signal as signal
import struct
import io
from numba import jit

class BVC_GLPC:
    def __init__(self, fs=44100, quantize=False, lpc_order=16, quantizer_config=None):
        """
        Initialize BVC_GLPC codec.
        
        Args:
            fs: Sample rate (Hz)
            quantize: If True, apply quantization (lossy compression).
            lpc_order: Order of LPC filter (default 16).
            quantizer_config: Dict of quantization parameters (passed to BVCQuantizer).
        """
        self.fs = fs
        self.base_N = 256
        self.max_merge = 8
        self.quantize = quantize
        
        # --- Caches ---
        self.dictionaries = {}
        self.gram_matrices = {}
        self._hann_cache = {}
        
        # --- Parameters ---
        self.lpc_order = lpc_order
        self.overlap = 32 # Overlap for OLA (samples)
        
        # --- Quantizer (Optional) ---
        if self.quantize:
            from BVCQuantizer import BVCQuantizer
            self.quantizer = BVCQuantizer(config=quantizer_config)
        else:
            self.quantizer = None
        
        # --- Signal Processing ---
        # Bandwidth expansion for stability (Lag Window)
        self.gamma = np.exp(-0.5 * (2.0 * np.pi * 60.0 / self.fs)) 
        self.lag_window = self.gamma ** np.arange(self.lpc_order + 1)

    def _get_hann_window(self, N):
        if N not in self._hann_cache:
            self._hann_cache[N] = np.hanning(N).astype(np.float32)
        return self._hann_cache[N]

    def _get_gabor_dict(self, N, transient_mode=False):
        """
        VECTORIZED Gabor Dictionary Generation.
        Replaces slow Python loops with NumPy broadcasting.
        """
        cache_key = (N, transient_mode)
        if cache_key in self.dictionaries:
            return self.dictionaries[cache_key], self.gram_matrices[cache_key]
        
        t = np.arange(N, dtype=np.float32)
        atoms_list = []
        
        # --- 1. Define Parameters Arrays ---
        
        if transient_mode:
            # Impulse-like parameters (Short width, dense spacing)
            scales = np.array([1, 2, 4, 8, 16], dtype=np.float32)
            shifts = np.arange(0, N, 4) # Dense shifting
            freqs = np.array([0], dtype=np.float32) # Baseband only for impulses
        else:
            # Tonal parameters (Dyadic scales)
            min_s = 32
            num_scales = 4
            scales = min_s * (2 ** np.arange(num_scales))
            scales = scales[scales < N/2] # Safety
            if len(scales) == 0: scales = np.array([N/4])
            
            shifts = np.arange(0, N, max(8, N//32))
            
            # Logarithmic Frequency spacing
            min_f = 60
            max_f = self.fs / 2 - 500
            num_freqs = 128
            freqs_hz = np.logspace(np.log10(min_f), np.log10(max_f), num_freqs)
            freqs = 2 * np.pi * freqs_hz / self.fs

        # --- 2. Vectorized Generation ---
        
        # Grid 1: Envelopes (Shift vs Scale)
        # shape: (len(scales), len(shifts), N)
        S_grid, U_grid, T_grid = np.meshgrid(scales, shifts, t, indexing='ij')
        # Gaussian Envelope: exp(-0.5 * ((t-u)/s)^2)
        Envelopes = np.exp(-0.5 * ((T_grid - U_grid) / S_grid)**2)
        
        # Filter out weak envelopes
        # Reshape to (Num_Envelopes, N)
        Envelopes = Envelopes.reshape(-1, N)
        
        # NOTE: Hard windowing removed for OLA compatibility.
        # The residual itself will be windowed before MP.
            
        mask = Envelopes.max(axis=1) > 0.01
        Envelopes = Envelopes[mask]
        
        if not transient_mode:
            # Grid 2: Modulation
            # For tonal, we modulate the envelopes
            # This can be memory heavy, so we iterate the envelopes and broadcast freqs
            final_atoms = []
            
            # Pre-compute Cos/Sin tables
            # shape: (Num_Freqs, N)
            Freq_Grid, T_Line = np.meshgrid(freqs, t, indexing='ij')
            Cos_Mod = np.cos(Freq_Grid * T_Line)
            Sin_Mod = np.sin(Freq_Grid * T_Line)
            
            # Apply modulation to envelopes
            # This is an O(N_Env * N_Freq) operation.
            # We do it in chunks to save RAM.
            
            for env in Envelopes:
                # Modulate
                atoms_c = env * Cos_Mod
                atoms_s = env * Sin_Mod
                
                # Normalize and add
                norms_c = np.sqrt(np.sum(atoms_c**2, axis=1))
                norms_s = np.sqrt(np.sum(atoms_s**2, axis=1))
                
                # Add valid cosine atoms
                valid_c = norms_c > 1e-6
                if np.any(valid_c):
                    final_atoms.append(atoms_c[valid_c] / norms_c[valid_c, None])
                    
                # Add valid sine atoms
                valid_s = norms_s > 1e-6
                if np.any(valid_s):
                    final_atoms.append(atoms_s[valid_s] / norms_s[valid_s, None])
            
            if len(final_atoms) > 0:
                D = np.vstack(final_atoms).T
            else:
                D = np.zeros((N, 1), dtype=np.float32)

        else:
            # Transient mode: Just normalize envelopes (no modulation needed for base impulses)
            norms = np.sqrt(np.sum(Envelopes**2, axis=1))
            valid = norms > 1e-6
            D = (Envelopes[valid] / norms[valid, None]).T

        D = D.astype(np.float32)
        
        # Security limit: Don't let dictionary explode
        if D.shape[1] > 2000:
            # Deterministic downsample if too big
            D = D[:, ::2]
            
        G = D.T @ D
        
        self.dictionaries[cache_key] = D
        self.gram_matrices[cache_key] = G
        return D, G

    @staticmethod
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

    @staticmethod
    @jit(nopython=True, fastmath=True, cache=True)
    def _levinson_durbin(r, order):
        # Standard robust Levinson-Durbin recursion
        a = np.zeros(order + 1, dtype=np.float64)
        k = np.zeros(order, dtype=np.float64)
        a[0] = 1.0
        e = r[0] + 1e-12
        
        for i in range(order):
            acc = 0.0
            for j in range(i + 1):
                acc += a[j] * r[i + 1 - j]
            
            ki = -acc / e
            # Aggressive clamping for stability
            if ki > 0.999: ki = 0.999
            if ki < -0.999: ki = -0.999
            k[i] = ki
            
            a_prev = a.copy()
            for j in range(i + 1):
                a[j + 1] = a_prev[j + 1] + ki * a_prev[i - j]
            e = e * (1.0 - ki * ki)
            
        return a, k

    @staticmethod
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

    @staticmethod
    @jit(nopython=True, fastmath=True, cache=True)
    def _mp_fast_loop(work, D, G, n_atoms, threshold):
        """
        Matching Pursuit Core.
        """
        correlations = D.T @ work 
        indices = np.zeros(n_atoms, dtype=np.int32)
        coeffs = np.zeros(n_atoms, dtype=np.float32)
        current_energy = np.sum(work**2)
        count = 0
        
        for i in range(n_atoms):
            best_idx = np.argmax(np.abs(correlations))
            coeff = correlations[best_idx]
            
            # Simple threshold
            if abs(coeff) < 1e-6: break
            
            indices[i] = best_idx
            coeffs[i] = coeff
            count += 1
            
            # Update correlations (Gram-Schmidt style)
            # corr_new = corr_old - coeff * <atom_new, atom_all>
            correlations -= coeff * G[:, best_idx]
            
            current_energy -= coeff**2
            if current_energy < threshold: break
            
        return indices[:count], coeffs[:count]

    # Note: Quantization methods moved to BVCQuantizer.py
    # These wrapper methods maintained for backward compatibility
    
    def _quantize_energy(self, energy):
        """Wrapper for energy quantization (delegates to quantizer if enabled)."""
        if self.quantizer:
            return self.quantizer.quantize_energy(energy)
        else:
            return energy  # Lossless mode

    def _dequantize_energy(self, q):
        """Wrapper for energy dequantization."""
        if self.quantizer:
            return self.quantizer.dequantize_energy(q)
        else:
            return q  # Lossless mode

    def _quantize_lpc(self, k_coeffs):
        """Wrapper for LPC quantization (delegates to quantizer if enabled)."""
        if self.quantizer:
            return self.quantizer.quantize_lpc(k_coeffs)
        else:
            return k_coeffs  # Lossless mode

    def _dequantize_lpc(self, q_indices):
        """Wrapper for LPC dequantization."""
        if self.quantizer:
            return self.quantizer.dequantize_lpc(q_indices)
        else:
            return q_indices  # Lossless mode

    def process(self, sig):
        sig = sig.astype(np.float32)
        N = self.base_N
        
        # --- Input Buffering ---
        # Pad to multiple of N, plus extra for OLA
        pad_len = (N - (len(sig) % N)) % N
        # Add extra overlap padding at the end for the last frame's tail
        sig = np.pad(sig, (0, pad_len + self.overlap))
        
        num_frames = (len(sig) - self.overlap) // N
        # Re-calculate exact frames to avoid index errors
        # We iterate by N, but read N + overlap
        
        frames_matrix = sig[:num_frames*N].reshape(num_frames, N)
        
        # Filter State (Must persist!)
        zi_pre = np.zeros(self.lpc_order, dtype=np.float32)
        zi_syn = np.zeros(self.lpc_order, dtype=np.float32)
        
        # Stream of continuous excitation
        full_excitation = []
        
        # Storage for binary packing
        encoded_frames = []
        
        i = 0
        while i < num_frames:
            # 1. Logic: Determine Block Size (Transient Detection)
            # Simplified transient detection for robustness
            curr_frame = frames_matrix[i]
            energy = np.sum(curr_frame**2)
            peak = np.max(np.abs(curr_frame))
            rms = np.sqrt(energy/N)
            crest = peak / (rms + 1e-9)
            
            is_transient = (crest > 6.0) and (energy > 0.001)
            
            merge_count = 1
            if not is_transient:
                # Attempt merge up to max_merge
                # Simple energy similarity check
                for k in range(1, min(self.max_merge, num_frames - i)):
                    next_rms = np.sqrt(np.sum(frames_matrix[i+k]**2)/N)
                    ratio = next_rms / (rms + 1e-9)
                    if 0.5 < ratio < 2.0: # Similar energy
                        merge_count += 1
                    else:
                        break
            
            core_len = merge_count * N
            total_len = core_len + self.overlap
            
            # 2. Extract Audio Block (with Overlap)
            start_idx = i * N
            # Ensure we don't go out of bounds (padding handled above)
            raw_audio = sig[start_idx : start_idx + total_len]
            
            # 3. LPC Analysis
            # Analyze only the CORE part for stability and consistency
            audio_core = raw_audio[:core_len]
            
            # Note: We use Order 16 even for transients to maintain Filter State continuity.
            # For transients, we just use a flatter predictor (pre-emphasis)
            
            if is_transient:
                # "Hard-coded" gentle pre-emphasis coefficients
                # This bypasses the potentially unstable LPC analysis on impulsive noise
                # But we maintain the vector size 16 for the filter state.
                a_quant = np.zeros(self.lpc_order + 1, dtype=np.float32)
                a_quant[0] = 1.0
                a_quant[1] = -0.5 # Mild pre-emphasis
                k_raw = np.zeros(self.lpc_order, dtype=np.float32)
                q_lar = self._quantize_lpc(k_raw)  # Will be zeros or pass-through
                mode_flag = 1 # Transient
            else:
                w = self._get_hann_window(len(audio_core))
                r = self._autocorr_fast(audio_core, w, self.lpc_order, self.lag_window)
                a_raw, k_raw = self._levinson_durbin(r, self.lpc_order)
                
                # Quantize (or pass-through in lossless mode)
                q_lar = self._quantize_lpc(k_raw)
                k_quant = self._dequantize_lpc(q_lar)
                a_quant = self._rc_to_lpc(k_quant)
                mode_flag = 0 # Normal
            
            # 4. Inverse Filtering (Analysis)
            # Crucial: Pass 'zi' (state)
            # We produce the residual (excitation)
            
            # Step A: Filter Core -> updates state for next frame
            res_core, zi_next = signal.lfilter(a_quant, [1.0], raw_audio[:core_len], zi=zi_pre)
            
            # Step B: Filter Overlap -> using a COPY of the state
            res_overlap, _ = signal.lfilter(a_quant, [1.0], raw_audio[core_len:], zi=zi_next.copy())
            
            # Combine
            residual = np.concatenate([res_core, res_overlap])
            
            # Update main state
            zi_pre = zi_next
            
            # 5. Windowing for OLA
            # Apply Trapezoidal/Sine window to residual
            # Fade In: First 'overlap' samples (if not first frame)
            # Fade Out: Last 'overlap' samples
            
            window = np.ones(total_len, dtype=np.float32)
            
            # Fade Out (Tail) - Always apply
            t_fade = np.arange(self.overlap)
            # Cos^2 fade out
            fade_out = np.cos(0.5 * np.pi * t_fade / self.overlap)**2
            window[-self.overlap:] = fade_out
            
            # Fade In (Head) - Apply if not first frame (i > 0)
            if i > 0:
                # Sin^2 fade in
                fade_in = np.sin(0.5 * np.pi * t_fade / self.overlap)**2
                window[:self.overlap] = fade_in
            
            masked_residual = residual * window
            
            # 5. Matching Pursuit on Residual
            # Scale residual for quantization optimization
            res_energy = np.sum(masked_residual**2)
            q_energy = self._quantize_energy(res_energy)
            
            # If silence, skip MP
            atoms_list = []
            recon_residual = np.zeros_like(residual)
            
            if q_energy > 0:
                D, G = self._get_gabor_dict(total_len, transient_mode=is_transient)
                threshold = res_energy * 0.05 # Target 95% reconstruction or limit
                max_atoms = 50 if is_transient else 20 * merge_count
                
                indices, coeffs = self._mp_fast_loop(
                    masked_residual.astype(np.float32), D, G, max_atoms, threshold
                )
                
                for idx, c in zip(indices, coeffs):
                    atoms_list.append((idx, c))
                    recon_residual += c * D[:, idx]
            
            # 6. Store Data
            encoded_frames.append({
                'mode': mode_flag,
                'merge': merge_count,
                'q_energy': q_energy,
                'q_lar': q_lar,
                'atoms': atoms_list
            })
            
            # 7. Synthesis (Local Decoding for state update)
            # This ensures the encoder's filter state matches the decoder's
            # Note: We filter the RECONSTRUCTED residual, not the original
            # synth_local, zi_syn = signal.lfilter([1.0], a_quant, recon_residual, zi=zi_syn)
            
            # full_excitation.append(synth_local)
            
            i += merge_count
            
        # Return compressed object
        return encoded_frames

    def save_to_file(self, filename, encoded_frames):
        """
        Saves to a custom binary format (.rbvc)
        Structure:
        [HEADER]
        - Magic "RBVC": 4 bytes
        - Version: 1 byte (0=lossless, 1=lossy)
        - SampleRate: 4 bytes (int)
        - NumFrames: 4 bytes (int)
        [FRAME_LOOP]
        - Header Byte: (Mode << 4) | MergeCount : 1 byte
        - Energy: 1 byte (lossy) or 4 bytes float (lossless)
        - NumAtoms: 2 bytes (ushort)
        - LAR Data: 16 * 2 bytes (shorts) if lossy, 16 * 4 bytes (floats) if lossless (Only if Mode == 0)
        - Atoms: NumAtoms * (Index(2 bytes) + Coeff(4 bytes float))
        """
        with open(filename, 'wb') as f:
            # Header
            f.write(b'RBVC')
            version = 1 if self.quantize else 0
            f.write(struct.pack('B', version))
            f.write(struct.pack('<II', self.fs, len(encoded_frames)))
            
            for frame in encoded_frames:
                # Pack Flags
                # mode is 1 bit (actually 0 or 1), merge is 3 bits (1-8)
                flags = (frame['mode'] << 4) | (frame['merge'] & 0x0F)
                f.write(struct.pack('B', flags))
                
                # Pack Energy (format depends on quantize mode)
                if self.quantize:
                    f.write(struct.pack('B', frame['q_energy']))
                else:
                    f.write(struct.pack('<f', frame['q_energy']))
                
                # Pack Num Atoms
                n_atoms = len(frame['atoms'])
                f.write(struct.pack('<H', n_atoms))
                
                # Pack LARs (only if Normal Mode)
                if frame['mode'] == 0:
                    if self.quantize:
                        # Quantized: 16 shorts
                        f.write(struct.pack('<16h', *frame['q_lar']))
                    else:
                        # Lossless: 16 floats
                        f.write(struct.pack('<16f', *frame['q_lar']))
                    
                # Pack Atoms
                # Each atom: Index (ushort), Coeff (float) -> 6 bytes
                for idx, coef in frame['atoms']:
                    f.write(struct.pack('<Hf', idx, coef))

    def load_from_file(self, filename):
        with open(filename, 'rb') as f:
            magic = f.read(4)
            if magic != b'RBVC':
                raise ValueError("Invalid File Format")
            
            # Read version to determine format
            version_byte = f.read(1)
            if len(version_byte) == 0:
                # Old format without version byte
                # Rewind and read as old format
                f.seek(4)
                version = 1  # Assume lossy
                fs, num_frames = struct.unpack('<II', f.read(8))
            else:
                version = struct.unpack('B', version_byte)[0]
                fs, num_frames = struct.unpack('<II', f.read(8))
            
            self.fs = fs
            # Update quantize mode based on file
            self.quantize = (version == 1)
            if self.quantize and self.quantizer is None:
                from BVCQuantizer import BVCQuantizer
                self.quantizer = BVCQuantizer()
            
            frames = []
            for _ in range(num_frames):
                flags_byte = f.read(1)
                if not flags_byte: break
                flags = struct.unpack('B', flags_byte)[0]
                
                mode = (flags >> 4) & 0x0F
                merge = flags & 0x0F
                
                # Read energy (format depends on version)
                if version == 1:  # Lossy
                    q_energy = struct.unpack('B', f.read(1))[0]
                else:  # Lossless
                    q_energy = struct.unpack('<f', f.read(4))[0]
                
                n_atoms = struct.unpack('<H', f.read(2))[0]
                
                q_lar = []
                if mode == 0:
                    if version == 1:  # Lossy
                        q_lar = list(struct.unpack('<16h', f.read(32)))
                    else:  # Lossless
                        q_lar = list(struct.unpack('<16f', f.read(64)))
                
                atoms = []
                for _ in range(n_atoms):
                    idx, coef = struct.unpack('<Hf', f.read(6))
                    atoms.append((idx, coef))
                    
                # Convert q_lar to appropriate dtype
                if len(q_lar) > 0:
                    if version == 1:  # Lossy
                        q_lar_array = np.array(q_lar, dtype=np.int16)
                    else:  # Lossless
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
                
            return frames

    def decode(self, encoded_frames):
        # Filter state for synthesis
        zi = np.zeros(self.lpc_order, dtype=np.float32)
        
        # OLA Buffer for Residual
        total_samples = sum(f['merge'] * self.base_N for f in encoded_frames)
        residual_buffer = np.zeros(total_samples + self.overlap, dtype=np.float32)
        
        current_pos = 0
        output_audio = []
        
        for frame in encoded_frames:
            mode = frame['mode']
            merge = frame['merge']
            
            # 1. Reconstruct LPC Filter
            if mode == 1: # Transient
                a_quant = np.zeros(self.lpc_order + 1, dtype=np.float32)
                a_quant[0] = 1.0
                a_quant[1] = -0.5 
            else:
                k_quant = self._dequantize_lpc(frame['q_lar'])
                a_quant = self._rc_to_lpc(k_quant)
            
            # 2. Reconstruct Residual (Excitation)
            core_len = merge * self.base_N
            total_len = core_len + self.overlap
            
            excitation_block = np.zeros(total_len, dtype=np.float32)
            
            is_transient = (mode == 1)
            D, _ = self._get_gabor_dict(total_len, transient_mode=is_transient)
            
            for idx, coef in frame['atoms']:
                if idx < D.shape[1]:
                    excitation_block += coef * D[:, idx]
            
            # 3. Overlap-Add into Residual Buffer
            end_pos = current_pos + total_len
            if end_pos <= len(residual_buffer):
                residual_buffer[current_pos : end_pos] += excitation_block
            
            # 4. Synthesis Filter
            # Extract the valid core part
            valid_residual = residual_buffer[current_pos : current_pos + core_len]
            
            synth_block, zi = signal.lfilter([1.0], a_quant, valid_residual, zi=zi)
            
            output_audio.append(synth_block)
            
            current_pos += core_len
            
        return np.concatenate(output_audio)
