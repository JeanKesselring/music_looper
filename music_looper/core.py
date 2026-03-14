"""Core functionality for music looping with beat-perfect synchronization."""

import warnings

import librosa
import numpy as np
from pydub import AudioSegment
from scipy.spatial import distance


def extract_musical_fingerprint(y, sr, start_sample, end_sample):
    """Extracts Pitch (Chroma) and Texture (MFCC) for harmonic matching.

    Args:
        y: Audio time series
        sr: Sample rate
        start_sample: Start sample index
        end_sample: End sample index

    Returns:
        np.ndarray: Combined feature vector (13 MFCCs + 12 chroma features)
    """
    bar_audio = y[start_sample:end_sample]

    # Check if the slice is too short (edge case protection)
    if len(bar_audio) < 2048:
        return np.zeros(25)  # Dummy vector

    # Use an n_fft that doesn't exceed the signal length
    n_fft = min(2048, len(bar_audio))

    mfccs = librosa.feature.mfcc(y=bar_audio, sr=sr, n_mfcc=13, n_fft=n_fft)

    # chroma_cqt may warn about FFT size for short segments near the threshold
    # This is expected and harmless, so suppress the warning
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*n_fft.*too large.*")
        chroma = librosa.feature.chroma_cqt(y=bar_audio, sr=sr)

    mfcc_norm = np.mean(mfccs, axis=1) / (np.max(np.abs(np.mean(mfccs, axis=1))) + 1e-6)
    chroma_norm = np.mean(chroma, axis=1) / (np.max(np.abs(np.mean(chroma, axis=1))) + 1e-6)

    return np.concatenate((mfcc_norm, chroma_norm))


def perfect_sync_remix(input_file, output_name, target_length_sec):
    """Generate a perfectly synced remix using beat-aligned bars and harmonic matching.

    Args:
        input_file: Path to input audio file
        output_name: Base name for output file (without extension)
        target_length_sec: Target length of remix in seconds
    """
    print("Step 1: Dynamic Beat Tracking (Finding the exact drum hits)...")
    y, sr = librosa.load(input_file)
    original_audio = AudioSegment.from_file(input_file)

    # Get the exact frame of every detected beat
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_samples = librosa.frames_to_samples(beat_frames)

    # Group beats into bars (assuming 4/4 time signature)
    bar_start_samples = beat_samples[::4]

    if len(bar_start_samples) < 3:
        print("Error: Snippet is too short to extract enough full bars.")
        return

    bars_audio = []
    bars_features = []

    print(f"Step 2: Slicing into {len(bar_start_samples)-1} dynamically sized, beat-perfect bars...")

    for i in range(len(bar_start_samples) - 1):
        start_sample = bar_start_samples[i]
        end_sample = bar_start_samples[i+1]

        # Convert exact samples to milliseconds for pydub
        start_ms = (start_sample / sr) * 1000
        end_ms = (end_sample / sr) * 1000

        # Store the perfectly snapped audio slice
        bars_audio.append(original_audio[start_ms:end_ms])

        # Extract the harmonic/timbral fingerprint
        fingerprint = extract_musical_fingerprint(y, sr, start_sample, end_sample)
        bars_features.append(fingerprint)

    total_bars = len(bars_audio)
    target_ms = target_length_sec * 1000

    print("Step 3: Assembling arrangement (Harmonic matching + History penalty)...")
    remixed_track = bars_audio[0]
    current_bar_index = 0
    play_counts = np.zeros(total_bars)
    play_counts[0] += 1

    # Use a very short crossfade (10ms) to avoid "flamming" the perfectly aligned drum hits
    xfade = 10

    while len(remixed_track) < target_ms:
        current_fingerprint = bars_features[current_bar_index]
        next_bar_index = current_bar_index + 1

        if next_bar_index >= total_bars:
            best_match_index = 0
            lowest_cost = float('inf')

            for i, candidate_fingerprint in enumerate(bars_features):
                if i == current_bar_index:
                    continue

                dist = distance.euclidean(current_fingerprint, candidate_fingerprint)
                penalty = play_counts[i] * 1.5
                total_cost = dist + penalty

                if total_cost < lowest_cost:
                    lowest_cost = total_cost
                    best_match_index = i

            next_bar_index = best_match_index

        remixed_track = remixed_track.append(bars_audio[next_bar_index], crossfade=xfade)
        current_bar_index = next_bar_index
        play_counts[current_bar_index] += 1

    print("Step 4: Exporting perfectly synced track...")
    remixed_track = remixed_track.fade_out(4000)[:target_ms]
    remixed_track.export(f"{output_name}.wav", format="wav")
    print(f"Success! Masterpiece saved to {output_name}.wav")
