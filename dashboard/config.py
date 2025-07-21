import yaml
from pathlib import Path

def load_config(project_root):
    config_path = project_root / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def save_config(project_root, updated_config):
    config_path = project_root / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(updated_config, f, sort_keys=False) 