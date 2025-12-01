import numpy as np

class BVCQuantizer:
    """
    Quantization module for BVC_GLPC codec.
    Handles all lossy quantization operations for compression.
    """
    
    def __init__(self, config=None):
        if config is None:
            config = {}
        
        # --- LPC Quantization (Log Area Ratios) ---
        self.LAR_MAX = config.get('LAR_MAX', 10.0)
        self.LAR_MIN = config.get('LAR_MIN', -10.0)
        self.LAR_BITS = config.get('LAR_BITS', 10)
        self.LAR_Q_SCALE = (2**self.LAR_BITS - 1)
        
        # --- Energy Quantization (dB domain) ---
        self.MIN_DB = config.get('MIN_DB', -100.0)
        self.MAX_DB = config.get('MAX_DB', 0.0)
        self.ENERGY_BITS = config.get('ENERGY_BITS', 8)
        self.ENERGY_Q_SCALE = (2**self.ENERGY_BITS - 1)
        
    # ========== LPC Quantization ==========
    
    def quantize_lpc(self, k_coeffs):
        k_safe = np.clip(k_coeffs, -0.995, 0.995)
        lar = np.log((1 + k_safe) / (1 - k_safe))
        lar = np.clip(lar, self.LAR_MIN, self.LAR_MAX)
        norm = (lar - self.LAR_MIN) / (self.LAR_MAX - self.LAR_MIN)
        q_indices = np.round(norm * self.LAR_Q_SCALE).astype(np.int16)
        return q_indices
    
    def dequantize_lpc(self, q_indices):
        norm = q_indices / self.LAR_Q_SCALE
        lar = norm * (self.LAR_MAX - self.LAR_MIN) + self.LAR_MIN
        k = np.tanh(lar / 2.0)
        return k
    
    # ========== Energy Quantization ==========
    
    def quantize_energy(self, energy):
        db = 10 * np.log10(energy + 1e-10)
        db = np.clip(db, self.MIN_DB, self.MAX_DB)
        norm = (db - self.MIN_DB) / (self.MAX_DB - self.MIN_DB)
        q = int(round(norm * self.ENERGY_Q_SCALE))
        return max(0, min(self.ENERGY_Q_SCALE, q))
    
    def dequantize_energy(self, q):
        norm = q / self.ENERGY_Q_SCALE
        db = norm * (self.MAX_DB - self.MIN_DB) + self.MIN_DB
        return 10 ** (db / 10.0)
    
    # ========== Atom Coefficient Quantization (Frame-Based) ==========
    
    def quantize_atom_coeffs_frame(self, coeffs):
        """
        Quantizes a frame of coefficients using 8-bit dynamic range scaling.
        Returns: (gain, q_coeffs_int8)
        """
        if len(coeffs) == 0:
            return 0.0, np.array([], dtype=np.int8)
            
        # Find peak magnitude in this frame
        max_val = np.max(np.abs(coeffs))
        
        if max_val < 1e-9:
            return 0.0, np.zeros(len(coeffs), dtype=np.int8)
            
        # Normalize -1 to 1
        norm = coeffs / max_val
        
        # Quantize to -127..127 (preserving sign)
        q_coeffs = np.round(norm * 127).astype(np.int8)
        
        return max_val, q_coeffs

    def dequantize_atom_coeffs_frame(self, gain, q_coeffs):
        """
        Dequantizes int8 coefficients using the frame gain.
        """
        if len(q_coeffs) == 0:
            return np.array([], dtype=np.float32)
            
        return q_coeffs.astype(np.float32) * (gain / 127.0)