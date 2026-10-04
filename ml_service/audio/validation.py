import io
import json
import os
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO, Dict, Optional, Tuple, Union
import numpy as np

try:
    import soundfile as sf
except ImportError:
    sf = None


def load_audio_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load audio processing configuration parameters from JSON configuration file."""
    if config_path:
        path = Path(config_path)
    else:
        env_dir = os.getenv("CONFIG_DIR")
        if env_dir:
            path = Path(env_dir) / "audio_config.json"
        else:
            path = Path(__file__).resolve().parent.parent.parent / "config" / "audio_config.json"

    if not path.is_file():
        raise FileNotFoundError(f"Audio config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@dataclass
class AudioMetadata:
    format: str
    file_size_bytes: int
    duration_seconds: float
    sample_rate: int
    channels: int
    rms_energy: float
    is_silent: bool
    is_clipped: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert AudioMetadata instance to a dictionary."""
        return asdict(self)


@dataclass
class AudioValidationResult:
    is_valid: bool
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Optional[AudioMetadata] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert AudioValidationResult instance to a dictionary."""
        return {
            "is_valid": self.is_valid,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "metadata": self.metadata.to_dict() if self.metadata else None,
        }


def _decode_audio(audio_bytes: bytes) -> Tuple[np.ndarray, int]:
    """Decode raw audio bytes into float32 numpy array (frames, channels) and sample rate."""
    buffer = io.BytesIO(audio_bytes)

    if sf is not None:
        try:
            data, sr = sf.read(buffer, always_2d=True, dtype="float32")
            return data, sr
        except Exception:
            pass

    buffer.seek(0)
    try:
        with wave.open(buffer, "rb") as wav_file:
            n_channels = wav_file.getnchannels()
            sample_rate = wav_file.getframerate()
            sampwidth = wav_file.getsampwidth()
            n_frames = wav_file.getnframes()
            frames = wav_file.readframes(n_frames)

            if sampwidth == 1:
                data_1d = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
            elif sampwidth == 2:
                data_1d = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            elif sampwidth == 4:
                data_1d = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
            else:
                raise ValueError(f"Unsupported sample width: {sampwidth}")

            if n_channels > 1:
                audio_data = data_1d.reshape(-1, n_channels)
            else:
                audio_data = data_1d.reshape(-1, 1)

            return audio_data, sample_rate
    except Exception as exc:
        raise ValueError(f"Failed to decode audio stream: {str(exc)}") from exc


def validate_audio_file(
    file_input: Union[str, Path, bytes, BinaryIO],
    filename: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    config_path: Optional[Union[str, Path]] = None,
) -> AudioValidationResult:
    """Validate incoming audio file for format, size, duration, channels, integrity, and usable signal.

    Args:
        file_input: Path to audio file, raw bytes, or file-like stream object.
        filename: Optional filename used for format validation if file_input is bytes or stream.
        config: Optional pre-loaded configuration dictionary.
        config_path: Optional custom path to audio_config.json.

    Returns:
        AudioValidationResult with validation status, error code/message if rejected, and metadata.
    """
    if config is None:
        config = load_audio_config(config_path)

    supported_formats = [fmt.lower() for fmt in config.get("supported_formats", ["wav", "mp3", "flac", "ogg", "m4a"])]
    max_duration = float(config.get("max_duration_seconds", 30.0))
    min_duration = float(config.get("min_duration_seconds", 0.5))
    max_file_size_bytes = int(config.get("max_file_size_mb", 20) * 1024 * 1024)
    silence_threshold_rms = float(config.get("silence_threshold_rms", 1e-4))

    resolved_filename = ""
    if isinstance(file_input, (str, Path)):
        p_obj = Path(file_input)
        resolved_filename = filename or p_obj.name
    elif filename:
        resolved_filename = filename

    ext = resolved_filename.rsplit(".", 1)[-1].lower() if "." in resolved_filename else ""

    if not ext or ext not in supported_formats:
        return AudioValidationResult(
            is_valid=False,
            error_code="UNSUPPORTED_FORMAT",
            error_message=f"Unsupported file format '{ext}'. Supported formats: {supported_formats}",
        )

    try:
        if isinstance(file_input, (str, Path)):
            p = Path(file_input)
            if not p.exists():
                return AudioValidationResult(
                    is_valid=False,
                    error_code="FILE_NOT_FOUND",
                    error_message=f"File not found: {file_input}",
                )
            file_size_bytes = p.stat().st_size
            audio_bytes = p.read_bytes()
        elif isinstance(file_input, bytes):
            audio_bytes = file_input
            file_size_bytes = len(audio_bytes)
        elif hasattr(file_input, "read"):
            pos = file_input.tell() if hasattr(file_input, "tell") else None
            audio_bytes = file_input.read()
            file_size_bytes = len(audio_bytes)
            if pos is not None and hasattr(file_input, "seek"):
                file_input.seek(pos)
        else:
            return AudioValidationResult(
                is_valid=False,
                error_code="INVALID_INPUT",
                error_message="Invalid file input type provided.",
            )
    except Exception as exc:
        return AudioValidationResult(
            is_valid=False,
            error_code="FILE_READ_ERROR",
            error_message=f"Failed to read file data: {str(exc)}",
        )

    if file_size_bytes == 0:
        return AudioValidationResult(
            is_valid=False,
            error_code="FILE_EMPTY",
            error_message="Uploaded audio file is empty (0 bytes).",
        )

    if file_size_bytes > max_file_size_bytes:
        max_mb = config.get("max_file_size_mb", 20)
        return AudioValidationResult(
            is_valid=False,
            error_code="FILE_TOO_LARGE",
            error_message=f"File size exceeds limit of {max_mb} MB.",
        )

    try:
        audio_data, sample_rate = _decode_audio(audio_bytes)
    except Exception as exc:
        return AudioValidationResult(
            is_valid=False,
            error_code="FILE_CORRUPT",
            error_message=f"Audio file is corrupted or unreadable: {str(exc)}",
        )

    if audio_data is None or len(audio_data) == 0:
        return AudioValidationResult(
            is_valid=False,
            error_code="FILE_CORRUPT",
            error_message="Audio decoding yielded no audio frames.",
        )

    num_frames, channels = audio_data.shape
    duration_seconds = float(num_frames) / float(sample_rate)

    rms_energy = float(np.sqrt(np.mean(audio_data**2)))
    peak_amplitude = float(np.max(np.abs(audio_data)))
    is_silent = rms_energy < silence_threshold_rms or peak_amplitude < silence_threshold_rms
    is_clipped = peak_amplitude >= 0.999

    metadata = AudioMetadata(
        format=ext,
        file_size_bytes=file_size_bytes,
        duration_seconds=round(duration_seconds, 4),
        sample_rate=int(sample_rate),
        channels=int(channels),
        rms_energy=round(rms_energy, 6),
        is_silent=is_silent,
        is_clipped=is_clipped,
    )

    if duration_seconds < min_duration:
        return AudioValidationResult(
            is_valid=False,
            error_code="TOO_SHORT",
            error_message=f"Audio duration ({duration_seconds:.2f}s) is less than minimum required ({min_duration}s).",
            metadata=metadata,
        )

    if duration_seconds > max_duration:
        return AudioValidationResult(
            is_valid=False,
            error_code="TOO_LONG",
            error_message=f"Audio duration ({duration_seconds:.2f}s) exceeds maximum allowed ({max_duration}s).",
            metadata=metadata,
        )

    if is_silent:
        return AudioValidationResult(
            is_valid=False,
            error_code="NO_USABLE_SIGNAL",
            error_message="Audio file contains silence or no usable acoustic signal.",
            metadata=metadata,
        )

    return AudioValidationResult(
        is_valid=True,
        error_code=None,
        error_message=None,
        metadata=metadata,
    )
