"""Integration tests for perfect_sync_remix."""

import numpy as np
import pytest
import soundfile

from music_looper import perfect_sync_remix, extract_musical_fingerprint


class TestRemixBasics:
    """Test basic functionality of perfect_sync_remix."""

    def test_output_file_is_created(self, audio_file, tmp_path):
        """Output WAV file should be created."""
        output_path = tmp_path / "output"
        perfect_sync_remix(str(audio_file), str(output_path), 5)

        assert (tmp_path / "output.wav").exists(), "Output WAV file was not created"

    def test_remix_returns_none(self, audio_file, tmp_path):
        """Function should return None (no return value)."""
        output_path = tmp_path / "output"
        result = perfect_sync_remix(str(audio_file), str(output_path), 5)

        assert result is None, "Expected None return value"

    def test_output_is_valid_wav(self, audio_file, tmp_path):
        """Output file should be a valid, readable WAV."""
        output_path = tmp_path / "output"
        perfect_sync_remix(str(audio_file), str(output_path), 5)

        # Try to read the output
        wav_path = tmp_path / "output.wav"
        audio_data, sr = soundfile.read(str(wav_path))

        assert audio_data.shape[0] > 0, "Output WAV is empty"
        assert sr > 0, "Invalid sample rate in output"
        assert np.any(audio_data != 0), "Output WAV contains only silence"

    def test_output_duration_approximately_correct(self, audio_file, tmp_path):
        """Output duration should match target length (within 500ms tolerance)."""
        target_sec = 5
        output_path = tmp_path / "output"
        perfect_sync_remix(str(audio_file), str(output_path), target_sec)

        wav_path = tmp_path / "output.wav"
        audio_data, sr = soundfile.read(str(wav_path))
        duration = len(audio_data) / sr

        tolerance = 0.5  # 500ms tolerance
        assert abs(duration - target_sec) < tolerance, (
            f"Duration {duration:.2f}s does not match target {target_sec}s "
            f"(tolerance: {tolerance}s)"
        )


class TestTargetLengths:
    """Test different target lengths."""

    def test_very_short_target_length(self, audio_file, tmp_path):
        """Very short target length should work without crashing."""
        output_path = tmp_path / "output_short"
        perfect_sync_remix(str(audio_file), str(output_path), 1)

        wav_path = tmp_path / "output_short.wav"
        assert wav_path.exists(), "Output WAV file was not created"

        audio_data, sr = soundfile.read(str(wav_path))
        duration = len(audio_data) / sr
        assert duration <= 2.0, "Duration should be short when target is 1 second"


class TestErrorHandling:
    """Test error paths and edge cases."""

    def test_short_input_error_path(self, short_audio_file, tmp_path, capsys):
        """Too-short input should print error and not create output."""
        output_path = tmp_path / "output_error"
        perfect_sync_remix(str(short_audio_file), str(output_path), 5)

        wav_path = tmp_path / "output_error.wav"
        assert not wav_path.exists(), "Output should not be created for too-short input"

        captured = capsys.readouterr()
        assert "too short" in captured.out.lower(), (
            "Expected 'too short' error message in stdout"
        )


class TestAlgorithmicValue:
    """Test that the algorithm produces meaningful results."""

    def test_play_counts_reduce_repetition(self, audio_file, tmp_path):
        """Longer output should have variety, not just looping one bar."""
        target_sec = 10
        output_path = tmp_path / "output_long"
        perfect_sync_remix(str(audio_file), str(output_path), target_sec)

        wav_path = tmp_path / "output_long.wav"
        audio_data, sr = soundfile.read(str(wav_path))

        # Extract fingerprints at different points in the output
        # to check that we're not just repeating the same bar
        fingerprints = []
        window_size = sr * 2  # 2-second windows

        for i in range(0, len(audio_data) - window_size, window_size):
            fp = extract_musical_fingerprint(
                audio_data.astype(np.float32),
                sr,
                i,
                i + window_size,
            )
            fingerprints.append(fp)

        if len(fingerprints) >= 2:
            # Check that at least some fingerprints differ significantly
            distances = []
            for i in range(len(fingerprints) - 1):
                dist = np.linalg.norm(fingerprints[i] - fingerprints[i + 1])
                distances.append(dist)

            avg_distance = np.mean(distances)
            # The exact threshold is arbitrary, but it should be non-zero
            # (indicating some variation in bar selection)
            assert avg_distance > 0.001, (
                f"Expected variation in bar selection, avg distance: {avg_distance}"
            )
