from pathlib import Path
from tools.feature_fusion_helper import build_model_for_multi_channels
from ptflops import get_model_complexity_info
from pprint import pprint
import json
import torch


config_file = 'params/paths_zmachine.json'
with open(config_file, 'r') as f:
    config = json.load(f)
pprint(config)

model = build_model_for_multi_channels(
        model_name=config['model_name'],
        encoder_name=config['encoder_name'],
        num_classes=config['num_classes'],
        in_channels=config['in_channels'],
    )
model_dir = Path(config['root_dir']) / config['model_dir'] / config['model_name']
model_file = model_dir / config['model_file']
pretrained_epoch = int(model_file.stem.split('_')[-1])
input_shape = (8, 512, 256)
if model_file.exists():
    model.load_state_dict(torch.load(model_file, weights_only=True))
    print(f"======🌞Loaded model from disk: {model_file.stem}.======")
else:
    raise FileNotFoundError(f"Pretrained model not found at {model_file}")
macs, params = get_model_complexity_info(model, input_shape, as_strings=True,
                                             print_per_layer_stat=False, verbose=False)
GFLOPs = float(macs.split(' ')[0]) * 2
print(f"Input shape: {input_shape}")
print(f"Number of parameters: {params}")
print(f"GFLOPs: {GFLOPs:.2f} GFLOPs")



