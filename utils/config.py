# utils/config.py

import yaml
from pathlib import Path
import logging


def load_config(path: str = "config.yaml") -> dict:
    """
    Load the YAML config file with full validation and debug logging.

    Parameters:
        path (str): Relative or absolute path to config.yaml

    Returns:
        dict: Parsed configuration dictionary
    """
    config_path = Path(path).resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"❌ config.yaml not found at: {config_path}")

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"❌ Failed to parse YAML file at {config_path} — {e}")

    if not isinstance(config, dict):
        raise TypeError(f"❌ Config file did not parse into a dictionary: {config_path}")

    logging.debug(f"🛠️ Loaded config from: {config_path}")
    logging.debug(f"🔑 Top-level keys: {list(config.keys())}")

    return config
