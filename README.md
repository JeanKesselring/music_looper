# Music Looper

Generate perfectly synchronized music loops from any audio file. Automatically detects beats, extracts harmonic content, and assembles loops with zero phase shift at bar boundaries.

## Features

- 🎵 **Beat-Perfect Alignment**: Detects exact drum hits and snaps loop points to beat boundaries
- 🎼 **Smart Harmonic Matching**: Selects bars that sound good together using MFCC + chroma features
- 🔄 **Avoids Repetition**: Penalizes frequently-used bars to create variety
- ✨ **Minimal Crossfade**: Preserves drum timing with 10ms crossfades between bars

## Installation

**Option 1: From GitHub (recommended)**
```bash
pip install git+https://github.com/yourusername/music-looper.git
```

**Option 2: Local development**
```bash
git clone https://github.com/yourusername/music-looper.git
cd music-looper
pip install -e .
```

**Option 3: One-off usage**
```bash
pip install librosa pydub scipy numpy soundfile
# Then use music_looper directly in this directory
```

## Quick Start

**As a command:**
```bash
python -c "from music_looper import perfect_sync_remix; perfect_sync_remix('input.wav', 'output', 120)"
```

**In Python code:**
```python
from music_looper import perfect_sync_remix

# Create a 2-minute loop from any audio file
perfect_sync_remix(
    input_file="drums.wav",
    output_name="my_loop",      # saves as my_loop.wav
    target_length_sec=120       # 2 minutes
)
```

## How It Works

1. **Beat Detection** → Finds exact drum hit positions using librosa
2. **Bar Slicing** → Groups beats into 4/4 bars (drum perfect alignment)
3. **Feature Extraction** → Computes MFCC (timbre) + chroma (pitch) fingerprints
4. **Smart Assembly** → Picks the next bar based on:
   - How similar it sounds (Euclidean distance)
   - Penalty for bars already used (avoid boring repetition)
5. **Export** → Writes beat-aligned, crossfaded WAV file

## Example

```bash
# Create a 3-minute loop from a 10-second drum sample
python -c "from music_looper import perfect_sync_remix; perfect_sync_remix('sample.wav', 'beat_loop', 180)"
```

Output: `beat_loop.wav` (ready to use in your DAW or player)

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Check coverage
pytest tests/ --cov=music_looper
```

## Requirements

- Python 3.8+
- librosa (beat tracking + feature extraction)
- pydub (audio I/O)
- scipy (distance calculations)
- numpy (array operations)

## Notes

- Works best with rhythmic material (drums, percussion, loops)
- Currently assumes 4/4 time signature
- Output format: WAV at source sample rate
