"""Audio validation, preprocessing, and visualization module."""

from .preprocessing import (
    AudioSegment,
    PreprocessedAudio,
    normalize_amplitude,
    pad_or_truncate,
    preprocess_audio,
    reduce_noise,
    resample_audio,
    segment_audio,
    to_mono,
    trim_silence,
)
from .validation import (
    AudioMetadata,
    AudioValidationResult,
    load_audio_config,
    validate_audio_file,
)
from .visualization import (
    analyze_audio_visualization,
    classify_audio_quality,
    detect_clipping,
    estimate_background_noise,
    generate_spectrogram_data,
    generate_waveform_data,
)

__all__ = [
    "AudioMetadata",
    "AudioValidationResult",
    "AudioSegment",
    "PreprocessedAudio",
    "analyze_audio_visualization",
    "classify_audio_quality",
    "detect_clipping",
    "estimate_background_noise",
    "generate_spectrogram_data",
    "generate_waveform_data",
    "load_audio_config",
    "normalize_amplitude",
    "pad_or_truncate",
    "preprocess_audio",
    "reduce_noise",
    "resample_audio",
    "segment_audio",
    "to_mono",
    "trim_silence",
    "validate_audio_file",
]
