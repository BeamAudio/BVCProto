import numpy as np


class BVCQuantizer:
    """
    Quantization module for BVC_GLPC codec.
    Handles all lossy quantization operations for compression.
    """
    
    def __init__(self, config=None):
        """
        Initialize quantizer with configuration.
        
        Args:
            config: Optional dict with quantization parameters.
                   If None, uses default parameters.
        """
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
        
        # --- Atom Coefficient Quantization ---
        self.ATOM_COEFF_BITS = config.get('ATOM_COEFF_BITS', 32)  # 32 = float32 (no quantization)
        self.ATOM_COEFF_MAX = config.get('ATOM_COEFF_MAX', 10.0)
    
    # ========== LPC Quantization ==========
    
    def quantize_lpc(self, k_coeffs):
        """
        Quantizes Reflection Coefficients using Log-Area Ratios (LAR).
        
        Args:
            k_coeffs: Array of reflection coefficients (floats)
        
        Returns:
            q_indices: Quantized indices (int16 array)
        """
        # 1. RC to LAR: log((1+k)/(1-k))
        k_safe = np.clip(k_coeffs, -0.995, 0.995)
        lar = np.log((1 + k_safe) / (1 - k_safe))
        
        # 2. Clip LAR
        lar = np.clip(lar, self.LAR_MIN, self.LAR_MAX)
        
        # 3. Normalize and Quantize
        norm = (lar - self.LAR_MIN) / (self.LAR_MAX - self.LAR_MIN)
        q_indices = np.round(norm * self.LAR_Q_SCALE).astype(np.int16)
        
        return q_indices
    
    def dequantize_lpc(self, q_indices):
        """
        Dequantizes LAR indices back to reflection coefficients.
        
        Args:
            q_indices: Quantized indices (int16 array)
        
        Returns:
            k_coeffs: Reflection coefficients (float array)
        """
        # 1. De-normalize
        norm = q_indices / self.LAR_Q_SCALE
        lar = norm * (self.LAR_MAX - self.LAR_MIN) + self.LAR_MIN
        
        # 2. LAR to RC: tanh(lar/2)
        k = np.tanh(lar / 2.0)
        return k
    
    # ========== Energy Quantization ==========
    
    def quantize_energy(self, energy):
        """
        Quantizes energy value to log scale (dB).
        
        Args:
            energy: Linear energy value (float)
        
        Returns:
            q: Quantized energy (uint8, 0-255)
        """
        # Convert to dB
        db = 10 * np.log10(energy + 1e-10)
        db = np.clip(db, self.MIN_DB, self.MAX_DB)
        
        # Normalize 0-1
        norm = (db - self.MIN_DB) / (self.MAX_DB - self.MIN_DB)
        
        # Quantize
        q = int(round(norm * self.ENERGY_Q_SCALE))
        return max(0, min(self.ENERGY_Q_SCALE, q))
    
    def dequantize_energy(self, q):
        """
        Dequantizes energy from log scale.
        
        Args:
            q: Quantized energy (uint8)
        
        Returns:
            energy: Linear energy value (float)
        """
        norm = q / self.ENERGY_Q_SCALE
        db = norm * (self.MAX_DB - self.MIN_DB) + self.MIN_DB
        return 10 ** (db / 10.0)
    
    # ========== Atom Coefficient Quantization ==========
    
    def quantize_atom_coeff(self, coeff):
        """
        Quantizes a single atom coefficient.
        Currently uses float32 (no quantization) but can be extended.
        
        Args:
            coeff: Coefficient value (float)
        
        Returns:
            q_coeff: Quantized coefficient (float32 or int)
        """
        if self.ATOM_COEFF_BITS == 32:
            # No quantization, just ensure float32
            return np.float32(coeff)
        else:
            # Future: implement fixed-point quantization
            # For now, just clip and convert
            clipped = np.clip(coeff, -self.ATOM_COEFF_MAX, self.ATOM_COEFF_MAX)
            return np.float32(clipped)
    
    def dequantize_atom_coeff(self, q_coeff):
        """
        Dequantizes atom coefficient.
        
        Args:
            q_coeff: Quantized coefficient
        
        Returns:
            coeff: Dequantized coefficient (float)
        """
        return float(q_coeff)
    
    # ========== Batch Operations ==========
    
    def quantize_atom_coeffs(self, coeffs):
        """Quantize array of atom coefficients."""
        return np.array([self.quantize_atom_coeff(c) for c in coeffs], dtype=np.float32)
    
    def dequantize_atom_coeffs(self, q_coeffs):
        """Dequantize array of atom coefficients."""
        return np.array([self.dequantize_atom_coeff(c) for c in q_coeffs], dtype=np.float32)
