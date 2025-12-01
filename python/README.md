# BVC - Python Implementation

This is the reference implementation of the Beam Vocal Codec, used for algorithmic validation and research.

## Setup

Requires Python 3.8+.

```bash
pip install numpy scipy matplotlib numba joblib
```

## Usage

**CLI Tool:**
```bash
python BVC_CLI.py encode <input.wav> <output.bvc>
python BVC_CLI.py decode <input.bvc> <output.wav>
```

**Library Usage:**
```python
from bvc.BVC import BVC_GLPC

# Initialize
codec = BVC_GLPC(fs=44100, quantize=True)

# Encode
frames = codec.process(audio_data)
codec.save_to_file("out.bvc", frames)

# Decode
frames = codec.load_from_file("out.bvc")
audio = codec.decode(frames)
```

## Testing
Run the master test suite to generate analysis plots:
```bash
python test_suite_master.py
```
