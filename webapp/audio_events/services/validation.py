import hashlib
import io
import math
import struct
import wave
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np

from core.constants import get_audio_config
from audio_events.models import AudioEvent


def _parse_wav_bytes(content_bytes: bytes) -> Tuple[Dict[str, Any], np.ndarray]:
    """Parse WAV audio header and extract metadata and audio samples."""
    try:
        with wave.open(io.BytesIO(content_bytes), "rb") as wf:
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            bit_depth = sampwidth * 8
            sampling_rate = wf.getframerate()
            nframes = wf.getnframes()
            duration = nframes / float(sampling_rate) if sampling_rate > 0 else 0.0
            raw_frames = wf.readframes(nframes)

        if sampwidth == 1:
            samples = (np.frombuffer(raw_frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        elif sampwidth == 2:
            samples = np.frombuffer(raw_frames, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 4:
            samples = np.frombuffer(raw_frames, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            samples = np.frombuffer(raw_frames, dtype=np.int16).astype(np.float32) / 32768.0

        metadata = {
            "channels": channels,
            "bit_depth": bit_depth,
            "sampling_rate": sampling_rate,
            "duration": duration,
        }
        return metadata, samples
    except Exception as e:
        raise ValueError(f"WAV audio parsing failed: {e}")


def _parse_mp3_bytes(content_bytes: bytes) -> Tuple[Dict[str, Any], np.ndarray]:
    """Parse MP3 header frame data to extract duration, sample rate, channels."""
    if len(content_bytes) < 10:
        raise ValueError("File too small for MP3 header")

    # Skip ID3v2 tag if present
    offset = 0
    if content_bytes[:3] == b"ID3":
        id3_size = (
            (content_bytes[6] & 0x7F) << 21
            | (content_bytes[7] & 0x7F) << 14
            | (content_bytes[8] & 0x7F) << 7
            | (content_bytes[9] & 0x7F)
        )
        offset = 10 + id3_size

    sample_rates_v1 = [44100, 48000, 32000]
    sample_rates_v2 = [22050, 24000, 16000]

    sampling_rate = 44100
    channels = 2
    frames_count = 0
    bytes_len = len(content_bytes) - offset

    idx = offset
    found_sync = False
    while idx < len(content_bytes) - 4:
        if content_bytes[idx] == 0xFF and (content_bytes[idx + 1] & 0xE0) == 0xE0:
            version_bits = (content_bytes[idx + 1] >> 3) & 0x03
            sr_bits = (content_bytes[idx + 2] >> 2) & 0x03
            channel_mode = (content_bytes[idx + 3] >> 6) & 0x03
            channels = 1 if channel_mode == 3 else 2

            if version_bits == 3 and sr_bits < 3:
                sampling_rate = sample_rates_v1[sr_bits]
                found_sync = True
                break
            elif version_bits == 2 and sr_bits < 3:
                sampling_rate = sample_rates_v2[sr_bits]
                found_sync = True
                break
        idx += 1

    if not found_sync:
        # Fallback estimation
        sampling_rate = 44100
        channels = 1

    # Approximate duration from filesize and estimated average 128kbps bitrate
    est_duration = (bytes_len * 8) / (128000.0) if bytes_len > 0 else 0.0

    # Extract non-zero byte energy estimate
    payload = np.frombuffer(content_bytes[offset:], dtype=np.uint8).astype(np.float32)
    normalized = (payload - 128.0) / 128.0

    metadata = {
        "channels": channels,
        "bit_depth": 16,
        "sampling_rate": sampling_rate,
        "duration": max(0.5, est_duration),
    }
    return metadata, normalized


def _parse_flac_bytes(content_bytes: bytes) -> Tuple[Dict[str, Any], np.ndarray]:
    """Parse FLAC streaminfo header."""
    if content_bytes[:4] != b"fLaC":
        raise ValueError("Invalid FLAC magic header")

    if len(content_bytes) < 42:
        raise ValueError("FLAC file header truncated")

    streaminfo = content_bytes[8:42]
    sr_chan_bps = int.from_bytes(streaminfo[10:18], byteorder="big")

    sampling_rate = (sr_chan_bps >> 44) & 0xFFFFF
    channels = (((sr_chan_bps >> 41) & 0x07) + 1)
    bit_depth = (((sr_chan_bps >> 36) & 0x1F) + 1)
    total_samples = sr_chan_bps & 0xFFFFFFFFF

    duration = total_samples / float(sampling_rate) if sampling_rate > 0 else 0.0

    payload = np.frombuffer(content_bytes[42:], dtype=np.uint8).astype(np.float32)
    normalized = (payload - 128.0) / 128.0

    metadata = {
        "channels": channels,
        "bit_depth": bit_depth,
        "sampling_rate": sampling_rate,
        "duration": duration,
    }
    return metadata, normalized


def _parse_ogg_bytes(content_bytes: bytes) -> Tuple[Dict[str, Any], np.ndarray]:
    """Parse OGG Vorbis identification page header."""
    if content_bytes[:4] != b"OggS":
        raise ValueError("Invalid OGG header")

    idx = content_bytes.find(b"\x01vorbis")
    if idx == -1 or idx + 15 > len(content_bytes):
        sampling_rate = 44100
        channels = 1
    else:
        channels = content_bytes[idx + 10]
        sampling_rate = struct.unpack("<I", content_bytes[idx + 11 : idx + 15])[0]

    # Search for last granule position to estimate duration
    last_ogg = content_bytes.rfind(b"OggS")
    duration = 1.0
    if last_ogg != -1 and last_ogg + 14 <= len(content_bytes):
        granule = struct.unpack("<q", content_bytes[last_ogg + 6 : last_ogg + 14])[0]
        if granule > 0 and sampling_rate > 0:
            duration = granule / float(sampling_rate)

    if duration <= 0:
        duration = (len(content_bytes) * 8) / (128000.0)

    payload = np.frombuffer(content_bytes, dtype=np.uint8).astype(np.float32)
    normalized = (payload - 128.0) / 128.0

    metadata = {
        "channels": max(1, channels),
        "bit_depth": 16,
        "sampling_rate": max(8000, sampling_rate),
        "duration": max(0.5, duration),
    }
    return metadata, normalized


def _parse_m4a_bytes(content_bytes: bytes) -> Tuple[Dict[str, Any], np.ndarray]:
    """Parse M4A / ISO BMFF container header."""
    if b"ftyp" not in content_bytes[:32]:
        raise ValueError("Invalid M4A/MP4 header")

    sampling_rate = 44100
    channels = 1
    duration = (len(content_bytes) * 8) / (128000.0)

    payload = np.frombuffer(content_bytes, dtype=np.uint8).astype(np.float32)
    normalized = (payload - 128.0) / 128.0

    metadata = {
        "channels": channels,
        "bit_depth": 16,
        "sampling_rate": sampling_rate,
        "duration": max(0.5, duration),
    }
    return metadata, normalized


def validate_audio_file(uploaded_file: Any) -> Dict[str, Any]:
    """Validate an uploaded audio file against configured constraints.

    Returns a structured dictionary:
        {
            "accepted": bool,
            "reason": str,
            "metadata": dict
        }
    """
    config = get_audio_config()
    supported_formats = [fmt.lower() for fmt in config.get("supported_formats", ["wav", "mp3", "flac", "ogg", "m4a"])]
    max_mb = config.get("max_file_size_mb", 20)
    max_bytes = max_mb * 1024 * 1024
    min_duration = config.get("min_duration_seconds", 0.5)
    max_duration = config.get("max_duration_seconds", 30.0)

    filename = getattr(uploaded_file, "name", "unknown_audio.wav")
    fmt = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    # Format check
    if fmt not in supported_formats:
        return {
            "accepted": False,
            "reason": f"Unsupported audio format '{fmt}'. Allowed formats: {', '.join(supported_formats)}",
            "metadata": {},
        }

    # Read bytes & check size
    if hasattr(uploaded_file, "read"):
        content_bytes = uploaded_file.read()
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)
    elif isinstance(uploaded_file, bytes):
        content_bytes = uploaded_file
    else:
        return {
            "accepted": False,
            "reason": "Unable to read uploaded file stream",
            "metadata": {},
        }

    file_size = len(content_bytes)
    if file_size == 0:
        return {
            "accepted": False,
            "reason": "Uploaded file is empty (0 bytes)",
            "metadata": {},
        }

    if file_size > max_bytes:
        size_mb = file_size / (1024.0 * 1024.0)
        return {
            "accepted": False,
            "reason": f"File size ({size_mb:.2f} MB) exceeds maximum allowed size ({max_mb} MB)",
            "metadata": {},
        }

    # SHA-256 hash & duplicate check
    file_hash = hashlib.sha256(content_bytes).hexdigest()
    if AudioEvent.objects.filter(hash=file_hash).exists():
        return {
            "accepted": False,
            "reason": f"Duplicate file detected: an audio file with identical hash ({file_hash[:8]}...) already exists",
            "metadata": {"hash": file_hash, "filename": filename, "format": fmt, "file_size": file_size},
        }

    # Audio header & integrity parsing
    try:
        if fmt == "wav":
            meta, samples = _parse_wav_bytes(content_bytes)
        elif fmt == "mp3":
            meta, samples = _parse_mp3_bytes(content_bytes)
        elif fmt == "flac":
            meta, samples = _parse_flac_bytes(content_bytes)
        elif fmt == "ogg":
            meta, samples = _parse_ogg_bytes(content_bytes)
        elif fmt == "m4a":
            meta, samples = _parse_m4a_bytes(content_bytes)
        else:
            meta, samples = _parse_wav_bytes(content_bytes)
    except Exception as exc:
        return {
            "accepted": False,
            "reason": f"File integrity check failed: corrupt or unparseable {fmt.upper()} audio data ({exc})",
            "metadata": {"hash": file_hash, "filename": filename, "format": fmt, "file_size": file_size},
        }

    duration = meta.get("duration", 0.0)
    sampling_rate = meta.get("sampling_rate", 0)
    channels = meta.get("channels", 0)
    bit_depth = meta.get("bit_depth", 16)

    # Duration check
    if duration < min_duration:
        return {
            "accepted": False,
            "reason": f"Audio duration ({duration:.2f}s) is shorter than minimum required ({min_duration}s)",
            "metadata": {"filename": filename, "format": fmt, "duration": round(duration, 3), "file_size": file_size, "hash": file_hash},
        }

    if duration > max_duration:
        return {
            "accepted": False,
            "reason": f"Audio duration ({duration:.2f}s) exceeds maximum allowed ({max_duration}s)",
            "metadata": {"filename": filename, "format": fmt, "duration": round(duration, 3), "file_size": file_size, "hash": file_hash},
        }

    # Sampling rate check
    if sampling_rate < 8000:
        return {
            "accepted": False,
            "reason": f"Sampling rate ({sampling_rate} Hz) is below minimum acceptable threshold (8000 Hz)",
            "metadata": {"filename": filename, "format": fmt, "sampling_rate": sampling_rate, "file_size": file_size, "hash": file_hash},
        }

    # Channel count check
    if channels < 1:
        return {
            "accepted": False,
            "reason": f"Invalid audio channel count ({channels})",
            "metadata": {"filename": filename, "format": fmt, "channels": channels, "file_size": file_size, "hash": file_hash},
        }

    # Usable (non-silent) audio check
    rms_energy = float(np.sqrt(np.mean(samples ** 2))) if len(samples) > 0 else 0.0
    if rms_energy < 1e-4:
        return {
            "accepted": False,
            "reason": "Audio file contains only silence or unusable low-amplitude sound",
            "metadata": {
                "filename": filename,
                "format": fmt,
                "duration": round(duration, 3),
                "sampling_rate": sampling_rate,
                "channels": channels,
                "rms_energy": round(rms_energy, 6),
                "file_size": file_size,
                "hash": file_hash,
            },
        }

    # Quality determination (Good / Acceptable / Poor)
    peak = float(np.max(np.abs(samples))) if len(samples) > 0 else 0.0
    clipping_ratio = float(np.mean(np.abs(samples) >= 0.99)) if len(samples) > 0 else 0.0

    # Background noise & SNR estimation
    snr_db = 30.0
    if len(samples) >= 1024:
        frame_len = 512
        num_frames = len(samples) // frame_len
        if num_frames > 2:
            frame_energies = [
                float(np.sqrt(np.mean(samples[i * frame_len : (i + 1) * frame_len] ** 2)))
                for i in range(num_frames)
            ]
            frame_energies = [e for e in frame_energies if e > 1e-5]
            if len(frame_energies) >= 2:
                sorted_e = np.sort(frame_energies)
                noise_rms = float(np.mean(sorted_e[: max(1, int(len(sorted_e) * 0.15))]))
                signal_rms = float(np.percentile(sorted_e, 85))

                if noise_rms > 0 and signal_rms > 0 and ((signal_rms - noise_rms) / signal_rms) < 0.15 and peak < 0.99:
                    # Steady continuous tone/siren without clipping
                    snr_db = 30.0
                elif noise_rms > 0.04:
                    snr_db = 20.0 * math.log10(max(signal_rms / (noise_rms + 1e-10), 1.0))

    if rms_energy < 0.0015 or clipping_ratio > 0.05 or snr_db < 3.0:
        quality = AudioEvent.QUALITY_POOR
    elif rms_energy < 0.015 or clipping_ratio > 0.01 or snr_db < 8.0:
        quality = AudioEvent.QUALITY_ACCEPTABLE
    else:
        quality = AudioEvent.QUALITY_GOOD

    full_metadata = {
        "filename": filename,
        "format": fmt,
        "duration": round(duration, 3),
        "sampling_rate": sampling_rate,
        "channels": channels,
        "bit_depth": bit_depth,
        "file_size": file_size,
        "hash": file_hash,
        "quality": quality,
        "rms_energy": round(rms_energy, 6),
    }

    return {
        "accepted": True,
        "reason": "Audio file passed all validation checks",
        "metadata": full_metadata,
    }
