import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List


def get_config_dir() -> Path:
    """Return the absolute path to the configuration directory."""
    env_dir = os.getenv("CONFIG_DIR")
    if env_dir:
        return Path(env_dir).resolve()

    # Default location: <repo_root>/config
    repo_root = Path(__file__).resolve().parent.parent.parent
    return repo_root / "config"


def _load_json_file(filename: str) -> Dict[str, Any]:
    config_dir = get_config_dir()
    filepath = config_dir / filename
    if not filepath.is_file():
        raise FileNotFoundError(f"Configuration file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def get_classes() -> Dict[str, Any]:
    """Load and return cached sound class definitions from config/classes.json."""
    return _load_json_file("classes.json")


@lru_cache(maxsize=1)
def get_alert_rules() -> Dict[str, Any]:
    """Load and return cached alert rules from config/alert_rules.json."""
    return _load_json_file("alert_rules.json")


@lru_cache(maxsize=1)
def get_audio_config() -> Dict[str, Any]:
    """Load and return cached audio processing parameters from config/audio_config.json."""
    return _load_json_file("audio_config.json")


def get_categories() -> List[str]:
    """Return the list of mandatory sound categories."""
    return list(get_classes().get("categories", []))


def get_supported_formats() -> List[str]:
    """Return the list of supported audio file formats."""
    return list(get_audio_config().get("supported_formats", []))


def get_severity_for_category(category: str) -> str:
    """Return severity level for a given category per config/alert_rules.json."""
    try:
        rules = get_alert_rules().get("rules", {})
        if category in rules:
            return rules[category].get("severity", "Informational")
        return get_alert_rules().get("default_rule", {}).get("severity", "Medium")
    except Exception:
        severity_map = {
            "Gunshot": "Critical",
            "Panic Scream": "Critical",
            "Person Asking for Help": "Critical",
            "Glass Breaking": "High",
            "Alarm or Siren": "High",
            "Aggression or Violent Conflict": "High",
            "Machinery Fault": "Medium",
            "Vehicle Horn": "Low",
            "Animal Sound": "Informational",
            "Background Noise": "Informational",
        }
        return severity_map.get(category, "Informational")


def reload_configs() -> None:
    """Clear cached configuration data so next access reloads from disk."""
    get_classes.cache_clear()
    get_alert_rules.cache_clear()
    get_audio_config.cache_clear()

