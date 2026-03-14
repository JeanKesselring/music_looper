# Music Looper

A Python library for creating perfectly synchronized music loops with harmonic matching and beat-perfect alignment.

## Features

- **Dynamic Beat Tracking**: Automatically detects drum hits and beat positions
- **Harmonic Matching**: Matches bars based on musical fingerprints (chroma + MFCC)
- **Beat-Perfect Synchronization**: Aligns loops precisely to beat boundaries
- **Playback History Penalty**: Avoids repetitive arrangements
- **Smooth Crossfading**: Minimal crossfade to preserve drum timing

## Installation

```bash
pip install music-looper
```

Or for development:

```bash
git clone <repo-url>
cd music-looper
pip install -e ".[dev]"
```

## Quick Start

```python
from music_looper import perfect_sync_remix

# Create a 120-second remix from a source audio file
perfect_sync_remix(
    input_file="input_audio.wav",
    output_name="output_remix",
    target_length_sec=120
)
```

## How It Works

1. **Beat Detection**: Analyzes the audio to find exact beat positions
2. **Bar Slicing**: Divides the audio into beat-perfect bars (4/4 time)
3. **Feature Extraction**: Computes musical fingerprints using MFCC and chroma features
4. **Harmonic Matching**: Selects the next bar based on:
   - Euclidean distance between feature vectors
   - Playback history penalty (to avoid repetition)
5. **Assembly**: Concatenates bars with minimal crossfade to create the final track

## Requirements

- Python 3.8+
- librosa
- pydub
- scipy
- numpy

## License

MIT
