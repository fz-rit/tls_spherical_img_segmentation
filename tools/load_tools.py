from collections import Counter
# from prepare_dataset import load_data
import json
from matplotlib.colors import ListedColormap
import torch
import yaml
from pathlib import Path
import numpy as np
from prepare_dataset import CONFIG
# config_file = 'params/paths_zmachine_mangrove3d_multichannel.json'
# with open(config_file, 'r') as f:
#     CONFIG = json.load(f)


# def checkout_class_freq(config, num_classes = 5):
#     train_dataset, _, _ = load_data(config)
#     labels_map = train_dataset.labels_map
    

#     all_labels = []
#     for img_patch, mask_patch in train_dataset:
#         # mask_patch is of shape (H, W), containing class indices
#         flat_mask = mask_patch.view(-1)
#         all_labels.extend(flat_mask.tolist())

#     class_counts = Counter(all_labels)
#     total_pixels = sum(class_counts.values())

#     print("Class Frequencies:")
#     class_freq = {}
#     for c in range(num_classes):
#         freq = class_counts.get(c, 0) / total_pixels
#         class_freq[c] = freq
#         print(f"Class {c}-{labels_map[c]}: {freq*100:.2f}% of pixels")



def get_color_map():
    label_file = Path(CONFIG['root_dir']) / CONFIG['label_file']
    with open(label_file, 'r') as f:
        label_json = json.load(f)
    color_map = {label_dict['code']:label_dict["color"] for label_dict in label_json}
    color_map = {k: v for k, v in color_map.items() if k <18}
    scale = 1/255 if type(color_map[0][0]) == int and any([ci>1 for ci in color_map[1]]) else 1
    color_list = []
    for i in range(len(color_map)):
        color_ls = [(color_map[i][j] * scale) for j in range(3)]
        color_list.append(color_ls)
    custom_cmap = ListedColormap(color_list)
    return custom_cmap, color_list


def get_pil_palette():
    """
    Get a flat palette for PIL Image from the color map.
    """
    color_list = get_color_map()[1]
    flat_palette = [int(x * 255) for rgb in color_list for x in rgb]
    flat_palette += [0] * (768 - len(flat_palette))

    return flat_palette


def get_label_map():
    label_file = Path(CONFIG['root_dir']) / CONFIG['label_file']
    with open(label_file, 'r') as f:
        label_json = json.load(f)

    label_map = {label_dict['code']:label_dict["label"] for label_dict in label_json}
    label_map = {k: v for k, v in label_map.items() if k <18}
    return label_map


def save_model_locally(model, model_dir, model_name_prefix, dummy_shape, save_onnx=False):
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
    for key, value in dict_obj.items():
        if isinstance(value, torch.Tensor):
            dict_obj[key] = value.cpu().numpy().tolist()
        elif isinstance(value, np.ndarray):
            dict_obj[key] = np.round(dict_obj[key], 4).tolist()
        elif isinstance(value, float):
            dict_obj[key] = round(float(value), 4)
        elif isinstance(value, list):
            dict_obj[key] = [round(float(v), 4) if isinstance(v, float) else v for v in value]

    with open(output_path, 'w') as f:
        yaml.dump(dict_obj, f)
    print(f"Evaluation metrics saved to {output_path}")

if __name__ == "__main__":
    config = CONFIG
    print(config['ensemble_config'])