import yaml
from pathlib import Path
import logging

def load_config(path: str = "config.yaml") -> dict:
    """
    Load the YAML config file with full validation and debug logging.

    Parameters:
        path (str): Relative or absolute path to config.yaml or its parent directory.

    Returns:
        dict: Parsed configuration dictionary
    """
    config_path = Path(path).resolve()

    # If a directory is passed instead of a file, append config.yaml
    if config_path.is_dir():
        config_path = config_path / "config.yaml"

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

def save_config(config: dict, path: str = "config.yaml"):
    """
    Save the configuration dictionary to a YAML file.

    Parameters:
        config (dict): Configuration dictionary to save.
        path (str): Path to save the YAML file to.
    """
    config_path = Path(path).resolve()

    # If a directory is passed, save to config.yaml inside it
    if config_path.is_dir():
        config_path = config_path / "config.yaml"

    with open(config_path, "w") as f:
        yaml.dump(config, f, sort_keys=False)

    logging.info(f"✅ Saved config to {config_path}")
