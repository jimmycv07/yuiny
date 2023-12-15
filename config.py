import os
import yaml
import shutil
import argparse

from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, field


class Config:
    """Configuration flags for everything."""
    
    # Consistency Arguments
    # starting_pts: list = field(default_factory=list)
    # target_pts: list = field(default_factory=list)
    model: str = "resnet50"
    model_weight: str = ""
    num_epochs: int = 30
    learning_rate: float = 0.001
    batch_size: int = 128
    train_val_split: float = 0.7
    image_size: int = 240
    log_dir: str = "./logs/resnet50/"
    seed: int = 42
    device: str = "cuda:1"

def load_config(config_file):
    with open(config_file, 'r') as f:
        yaml_config = yaml.safe_load(f)
    config = Config()
    for key, value in yaml_config.items():
        setattr(config, key, value)
    # config.out_path = os.path.join("out", datetime.now().strftime("%Y%m%d-%H%M%S"))
    
    if not os.path.exists(config.log_dir):
        os.mkdir(config.log_dir)
    
    output_path = os.path.join(config.log_dir, os.path.basename(config_file))
    # output_path = os.path.join(config.log_dir, "config.yaml")
    if not os.path.exists(output_path):
        shutil.copyfile(config_file, output_path)
    # import ipdb; ipdb.set_trace()
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
    args = parser.parse_args()
    config = load_config(args.config)
