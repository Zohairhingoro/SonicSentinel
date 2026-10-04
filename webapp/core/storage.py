"""Secure audio file storage, content hashing, fingerprinting, and duplicate detection module."""

import hashlib
import io
from pathlib import Path
from typing import BinaryIO, Optional, Tuple, Union
from django.core.files.uploadedfile import UploadedFile

from audio_events.models import AudioEvent


def _read_raw_bytes(
    file_input: Union[bytes, BinaryIO, UploadedFile, str, Path]
) -> bytes:
    """Helper to read raw bytes payload without destroying stream position."""
    if isinstance(file_input, bytes):
        return file_input
    elif isinstance(file_input, (str, Path)):
        return Path(file_input).read_bytes()
    elif hasattr(file_input, "read"):
        pos = file_input.tell() if hasattr(file_input, "tell") else None
        contents = file_input.read()
        if pos is not None and hasattr(file_input, "seek"):
            file_input.seek(pos)
        return contents
    else:
        raise ValueError("Unsupported file input type.")


def compute_file_hash(
    file_input: Union[bytes, BinaryIO, UploadedFile, str, Path]
) -> str:
    """Compute hex-encoded SHA-256 digest of file payload for exact duplicate detection.

    Args:
        file_input: Raw bytes, file-like stream, UploadedFile, or file path.

    Returns:
        Hex-encoded 64-character SHA-256 hash string.
    """
    hasher = hashlib.sha256()

    if isinstance(file_input, bytes):
        hasher.update(file_input)
    elif isinstance(file_input, (str, Path)):
        p = Path(file_input)
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
    elif hasattr(file_input, "read"):
        pos = file_input.tell() if hasattr(file_input, "tell") else None
        for chunk in iter(lambda: file_input.read(65536), b""):
            hasher.update(chunk)
        if pos is not None and hasattr(file_input, "seek"):
            file_input.seek(pos)
    else:
        raise ValueError("Unsupported file input type for hash calculation.")

    return hasher.hexdigest()


def compute_audio_fingerprint(
    file_input: Union[bytes, BinaryIO, UploadedFile, str, Path],
    num_bins: int = 16,
) -> str:
    """Compute a rough amplitude envelope audio fingerprint for near-duplicate detection.

    Downsamples the audio energy profile into `num_bins` normalized temporal bins,
    quantized into hex energy levels (0-f). Combined with duration estimate signature.

    Args:
        file_input: Raw bytes, file stream, UploadedFile, or file path.
        num_bins: Number of temporal envelope bins (default 16).

    Returns:
        Fingerprint string formatted as `dur:<duration>_env:<hex_bins>`.
    """
    raw_bytes = _read_raw_bytes(file_input)
    if not raw_bytes:
        return ""

    pcm_data = None
    duration = 0.0

    if raw_bytes.startswith(b"RIFF") and b"WAVE" in raw_bytes[:16]:
        try:
            import wave
            buf = io.BytesIO(raw_bytes)
            with wave.open(buf, "rb") as wf:
                n_frames = wf.getnframes()
                fr = wf.getframerate()
                n_channels = wf.getnchannels()
                samp_width = wf.getsampwidth()
                if fr > 0:
                    duration = float(n_frames) / float(fr)
                frames = wf.readframes(n_frames)
                if samp_width == 2:
                    import numpy as np
                    pcm_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
                    if n_channels > 1:
                        pcm_data = pcm_data[::n_channels]
        except Exception:
            pcm_data = None

    if pcm_data is not None and len(pcm_data) > 0:
        import numpy as np
        abs_y = np.abs(pcm_data)
        bin_size = max(1, len(abs_y) // num_bins)
        bin_energies = []
        for i in range(num_bins):
            chunk = abs_y[i * bin_size : (i + 1) * bin_size]
            if len(chunk) > 0:
                bin_energies.append(float(np.mean(chunk)))
            else:
                bin_energies.append(0.0)
        max_energy = max(bin_energies) if max(bin_energies) > 0 else 1.0
        norm_bins = [min(15, int((e / max_energy) * 15)) for e in bin_energies]
        env_hex = "".join(f"{v:x}" for v in norm_bins)
        return f"dur:{round(duration, 1):.1f}_env:{env_hex}"

    chunk_size = max(1, len(raw_bytes) // num_bins)
    chunk_means = []
    for i in range(num_bins):
        chunk = raw_bytes[i * chunk_size : (i + 1) * chunk_size]
        if chunk:
            chunk_means.append(sum(chunk) / len(chunk))
        else:
            chunk_means.append(0.0)
    max_m = max(chunk_means) if max(chunk_means) > 0 else 1.0
    norm_bins = [min(15, int((m / max_m) * 15)) for m in chunk_means]
    env_hex = "".join(f"{v:x}" for v in norm_bins)
    approx_dur = round(len(raw_bytes) / 44100.0, 1)
    return f"dur:{approx_dur:.1f}_env:{env_hex}"


def are_fingerprints_near_duplicate(
    fp1: str, fp2: str, max_hamming_distance: int = 3, max_duration_diff: float = 0.5
) -> bool:
    """Compare two audio fingerprint strings for near-duplicate match.

    Returns True if duration difference <= max_duration_diff and
    envelope hex string Hamming distance <= max_hamming_distance.
    """
    if not fp1 or not fp2:
        return False
    if fp1 == fp2:
        return True

    try:
        parts1 = fp1.split("_env:")
        parts2 = fp2.split("_env:")
        if len(parts1) != 2 or len(parts2) != 2:
            return False

        dur1 = float(parts1[0].replace("dur:", ""))
        dur2 = float(parts2[0].replace("dur:", ""))
        if abs(dur1 - dur2) > max_duration_diff:
            return False

        env1 = parts1[1]
        env2 = parts2[1]
        if len(env1) != len(env2):
            return False

        dist = sum(1 for a, b in zip(env1, env2) if a != b)
        return dist <= max_hamming_distance
    except Exception:
        return False


def check_duplicate_audio(
    file_input: Union[bytes, BinaryIO, UploadedFile, str, Path]
) -> Tuple[Optional[AudioEvent], str]:
    """Check database for an existing AudioEvent record matching the exact SHA-256 hash.

    Args:
        file_input: Uploaded audio file payload or stream.

    Returns:
        Tuple of (existing_audio_event_or_none, calculated_sha256_hash).
    """
    file_hash = compute_file_hash(file_input)
    existing_event = AudioEvent.objects.filter(hash=file_hash).first()
    return existing_event, file_hash


def check_near_duplicate_audio(
    file_input: Union[bytes, BinaryIO, UploadedFile, str, Path],
    limit: int = 50,
    exclude_audio_id: Optional[str] = None,
) -> Tuple[Optional[AudioEvent], str]:
    """Check recent N AudioEvents (ordered by uploaded_at desc) for a suspected near-duplicate audio fingerprint.

    Does not silently block upload — returns the suspected duplicate AudioEvent instance if found,
    allowing caller to flag the new event for manual review and record the duplicate Audio ID.

    Args:
        file_input: Uploaded audio payload or file stream.
        limit: Number N of most recent AudioEvents to compare against (default 50).
        exclude_audio_id: Optional Audio ID string to exclude from comparison.

    Returns:
        Tuple of (suspected_near_duplicate_audio_event_or_none, calculated_fingerprint).
    """
    fingerprint = compute_audio_fingerprint(file_input)
    if not fingerprint:
        return None, ""

    qs = AudioEvent.objects.exclude(audio_fingerprint="").order_by("-uploaded_at")
    if exclude_audio_id:
        qs = qs.exclude(audio_id=exclude_audio_id)

    recent_events = list(qs[:limit])
    for event in recent_events:
        if event.audio_fingerprint and are_fingerprints_near_duplicate(fingerprint, event.audio_fingerprint):
            return event, fingerprint

    return None, fingerprint
