"""Pytest fixtures for music_looper tests."""

import numpy as np
import pytest
import soundfile


@pytest.fixture(scope="session")
def raw_audio():
    """Generate synthetic audio: 440 Hz sine wave + click track at 120 BPM.

    Returns:
        tuple: (audio_array, sample_rate) where audio_array is float32
    """
    sr = 22050
    duration = 8.0
    num_samples = int(sr * duration)

    # Generate 440 Hz sine wave
    t = np.arange(num_samples) / sr
    y = 0.3 * np.sin(2 * np.pi * 440 * t)

    # Add click track at 120 BPM (every 0.5 seconds = 2 beats per second)
    beat_interval_samples = sr // 2  # 0.5 second intervals
    for i in range(0, num_samples, beat_interval_samples):
        # Add sharp click impulse
        if i < num_samples:
            y[i] += 0.8

    # Normalize and convert to float32
    y = y / (np.max(np.abs(y)) + 1e-6)
    y = y.astype(np.float32)

    return y, sr


@pytest.fixture(scope="session")
def audio_file(raw_audio, tmp_path_factory):
    """Write synthetic audio to a temporary WAV file.

    Args:
        raw_audio: The raw_audio fixture
        tmp_path_factory: pytest's tmp_path_factory

    Returns:
        Path: Path to the created WAV file
    """
    y, sr = raw_audio
    tmp_dir = tmp_path_factory.mktemp("audio")
    audio_path = tmp_dir / "test_clip.wav"

    soundfile.write(str(audio_path), y, sr)
    return audio_path


@pytest.fixture(scope="session")
def short_audio_file(tmp_path_factory):
    """Generate a very short (0.1 second) synthetic audio file.

    Used for testing error handling when audio is too short.

    Args:
        tmp_path_factory: pytest's tmp_path_factory

    Returns:
        Path: Path to the created WAV file
    """
    sr = 22050
    duration = 0.1
    num_samples = int(sr * duration)

    # Very short sine wave, no need for complex structure
    t = np.arange(num_samples) / sr
    y = 0.3 * np.sin(2 * np.pi * 440 * t)
    y = (y / (np.max(np.abs(y)) + 1e-6)).astype(np.float32)

    tmp_dir = tmp_path_factory.mktemp("short_audio")
    audio_path = tmp_dir / "short_clip.wav"

    soundfile.write(str(audio_path), y, sr)
    return audio_path
