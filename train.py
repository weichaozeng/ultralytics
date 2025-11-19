import os
import argparse
import torch
import yaml
from ultralytics import YOLO

def prepare_checkpoint(cfg):
    pretrained_weight_path = cfg['weights']
    model_config_path = cfg['model_cfg']

    print(f"--- Preparing checkpoint with new structure: {model_config_path} ---")
    
    project_dir = os.path.join(cfg['project'], cfg['name'])
    os.makedirs(project_dir, exist_ok=True)
    temp_ckpt_path = os.path.join(project_dir, 'temp_transfer_weights.pt')
    
    new_model = YOLO(model_config_path) 
    
    try:
        old_checkpoint = torch.load(pretrained_weight_path, map_location='cpu', weights_only=False)
        old_state_dict = old_checkpoint['model'].state_dict()
    except Exception as e:
        print(f"Error loading old checkpoint weights: {e}")
        return None
        
    new_state_dict = new_model.model.state_dict()
    loaded_keys = []
    
    for name, param in old_state_dict.items():
        if name in new_state_dict and new_state_dict[name].shape == param.shape:
            new_state_dict[name].copy_(param)
            loaded_keys.append(name)

    new_model.model.load_state_dict(new_state_dict, strict=False)
    print(f"Successfully loaded {len(loaded_keys)} matching parameters.")
    
    
    
    return temp_ckpt_path


def main(cfg):
    if os.path.exists(cfg['data']):
        print(f"Datset config file not found at {cfg['data']}")
        return

    tmp_weight_path = prepare_checkpoint(cfg)
    if tmp_weight_path is None:
        return
    
    print("--- Start Training ---")
    model = YOLO(tmp_weight_path)
    cfg.pop('weights', None)
    cfg.pop('model_cfg', None)

    results = model.train(**cfg)

    print("--- End of Traing ---")
    os.remove(tmp_weight_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument('--cfg', type=str, required=True, help='Path to the training configuration YAML file')
    args = parser.parse_args()
    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)

    main(cfg)