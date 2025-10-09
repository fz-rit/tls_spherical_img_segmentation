import json
import numpy as np
import torch
import yaml
from collections import Counter
from matplotlib.colors import ListedColormap
from pathlib import Path


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def get_color_map(config: dict):
    """Get color map from config."""
    label_file = Path(config['root_dir']) / config['label_file']
    with open(label_file, 'r') as f:
        label_json = json.load(f)
    color_map = {label_dict['code']: label_dict["color"] for label_dict in label_json}
    color_map = {k: v for k, v in color_map.items() if k < 18}
    scale = 1/255 if type(color_map[0][0]) == int and any([ci > 1 for ci in color_map[1]]) else 1
    color_list = []
    for i in range(len(color_map)):
        color_ls = [(color_map[i][j] * scale) for j in range(3)]
        color_list.append(color_ls)
    custom_cmap = ListedColormap(color_list)
    return custom_cmap, color_list


def get_pil_palette(config: dict):
    """Get a flat palette for PIL Image from the color map."""
    color_list = get_color_map(config)[1]
    flat_palette = [int(x * 255) for rgb in color_list for x in rgb]
    flat_palette += [0] * (768 - len(flat_palette))
    return flat_palette


def get_label_map(config: dict):
    """Get label map from config."""
    label_file = Path(config['root_dir']) / config['label_file']
    with open(label_file, 'r') as f:
        label_json = json.load(f)
    label_map = {label_dict['code']: label_dict["label"] for label_dict in label_json}
    label_map = {k: v for k, v in label_map.items() if k < 18}
    return label_map


def save_model_locally(model, model_dir, model_name_prefix, dummy_shape, save_onnx=False):
    """Save model locally with optional ONNX export."""
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"----Created directory {model_dir}----")

    save_model_path = model_dir / f'{model_name_prefix}.pth'
    torch.save(model.state_dict(), save_model_path)
    print(f"----Model saved at {save_model_path}----")

    if save_onnx:
        # Create a dummy input with the same shape as your input
        dummy_input = torch.randn(dummy_shape).to('cuda')
        onnx_model_path = model_dir / f'{model_name_prefix}.onnx'
        torch.onnx.export(
            model, 
            dummy_input, 
            onnx_model_path,
            input_names=["input"],
            output_names=["output"],
            opset_version=11,
            do_constant_folding=True
        )
        print(f"----ONNX model saved at {onnx_model_path}----")


def dump_dict_to_yaml(dict_obj, output_path: Path):
    """
    Dump a dictionary to a YAML file.

    Args:
        dict_obj (dict): Dictionary to dump.
        output_path (Path): Path to the output YAML file.
    """
    # Ensure the values are serializable
    serialized_dict = {}
    for key, value in dict_obj.items():
        if isinstance(value, torch.Tensor):
            serialized_dict[key] = value.cpu().numpy().tolist()
        elif isinstance(value, np.ndarray):
            serialized_dict[key] = np.round(value, 4).tolist()
        elif isinstance(value, float):
            serialized_dict[key] = round(float(value), 4)
        elif isinstance(value, list):
            serialized_dict[key] = [round(float(v), 4) if isinstance(v, float) else v for v in value]
        else:
            serialized_dict[key] = value

    with open(output_path, 'w') as f:
        yaml.dump(serialized_dict, f)
    print(f"Evaluation metrics saved to {output_path}")