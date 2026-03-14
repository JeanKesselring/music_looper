"""Unit tests for extract_musical_fingerprint."""

import numpy as np
import pytest

from music_looper import extract_musical_fingerprint


class TestFingerprintBasics:
    """Test basic behavior of extract_musical_fingerprint."""

    def test_returns_correct_shape(self, raw_audio):
        """Output vector should be 25-dimensional (13 MFCCs + 12 chroma)."""
        y, sr = raw_audio
        # Use a 2-second segment
        start_sample = 0
        end_sample = sr * 2

        result = extract_musical_fingerprint(y, sr, start_sample, end_sample)

        assert result.shape == (25,), f"Expected shape (25,), got {result.shape}"

    def test_output_is_float(self, raw_audio):
        """Output should be float type, not integer."""
        y, sr = raw_audio
        start_sample = 0
        end_sample = sr * 2

        result = extract_musical_fingerprint(y, sr, start_sample, end_sample)

        assert np.issubdtype(result.dtype, np.floating), (
            f"Expected floating point dtype, got {result.dtype}"
        )

    def test_output_is_deterministic(self, raw_audio):
        """Same inputs should produce identical output."""
        y, sr = raw_audio
        start_sample = 0
        end_sample = sr * 2

        result1 = extract_musical_fingerprint(y, sr, start_sample, end_sample)
        result2 = extract_musical_fingerprint(y, sr, start_sample, end_sample)

        assert np.array_equal(result1, result2), "Function is not deterministic"


class TestShortSegmentHandling:
    """Test edge cases for very short audio segments."""

    def test_returns_zeros_for_short_segment(self, raw_audio):
        """Segment shorter than 2048 samples should return zero vector."""
        y, sr = raw_audio
        start_sample = 0
        end_sample = 1000  # < 2048

        result = extract_musical_fingerprint(y, sr, start_sample, end_sample)

        expected = np.zeros(25)
        assert np.array_equal(result, expected), (
            "Expected zero vector for short segment"
        )

    def test_returns_zeros_at_threshold_minus_one(self, raw_audio):
        """Segment with 2047 samples (just below threshold) should return zeros."""
        y, sr = raw_audio
        start_sample = 0
        end_sample = 2047

        result = extract_musical_fingerprint(y, sr, start_sample, end_sample)

        expected = np.zeros(25)
        assert np.array_equal(result, expected), (
            "Expected zero vector at 2047 samples"
        )

    def test_returns_nonzeros_at_threshold(self, raw_audio):
        """Segment with exactly 2048 samples should process normally."""
        y, sr = raw_audio
        start_sample = 0
        end_sample = 2048

        result = extract_musical_fingerprint(y, sr, start_sample, end_sample)

        expected_zeros = np.zeros(25)
        assert not np.array_equal(result, expected_zeros), (
            "Expected non-zero vector at exactly 2048 samples"
        )


class TestNormalization:
    """Test that output is properly normalized."""

    def test_output_is_normalized(self, raw_audio):
        """Output should be normalized to roughly [-1, 1] range."""
        y, sr = raw_audio
        start_sample = sr * 2
        end_sample = sr * 4

        result = extract_musical_fingerprint(y, sr, start_sample, end_sample)

        max_abs = np.max(np.abs(result))
        assert max_abs <= 1.0 + 1e-5, (
            f"Expected normalized output, max absolute value: {max_abs}"
        )


class TestContentSensitivity:
    """Test that the function distinguishes different audio content."""

    def test_different_frequencies_produce_different_fingerprints(self, raw_audio):
        """Fingerprints of different harmonic content should differ."""
        y, sr = raw_audio

        # Extract fingerprint from 440 Hz region
        fp1 = extract_musical_fingerprint(y, sr, sr * 2, sr * 3)

        # Generate a second snippet with a different frequency
        # (Create a modified copy with 880 Hz instead of 440 Hz)
        t = np.arange(len(y)) / sr
        y_alt = 0.3 * np.sin(2 * np.pi * 880 * t)
        y_alt = (y_alt / (np.max(np.abs(y_alt)) + 1e-6)).astype(np.float32)

        fp2 = extract_musical_fingerprint(y_alt, sr, sr * 2, sr * 3)

        assert not np.array_equal(fp1, fp2), (
            "Expected different fingerprints for different frequencies"
        )
