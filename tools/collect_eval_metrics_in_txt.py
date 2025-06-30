from pathlib import Path
from pprint import pprint
import re
import pandas as pd

from collections import OrderedDict
from pprint import pprint


def read_selected_metrics(txt_path: Path) -> dict:
    """
    Read selected scalar and list metrics from a metrics text file.

    Returns:
        dict with keys: oAcc, mAcc, mIoU, IoU_per_class
    """
    metrics = {}
    with open(txt_path, 'r') as f:
        for line in f:
            if line.startswith("oAcc:"):
                metrics["oAcc"] = float(line.split(":")[1].strip())
            elif line.startswith("mAcc:"):
                metrics["mAcc"] = float(line.split(":")[1].strip())
            elif line.startswith("mIoU:"):
                metrics["mIoU"] = float(line.split(":")[1].strip())
            elif line.startswith("IoU_per_class:"):
                list_str = line.split(":")[1].strip()
                # Remove brackets and split
                numbers = re.findall(r"[\d\.]+", list_str)
                # metrics["IoU_per_class"] = [float(x) for x in numbers]
                metrics["IoU_class_0"] = float(numbers[0])
                metrics["IoU_class_1"] = float(numbers[1])
                metrics["IoU_class_2"] = float(numbers[2])
                metrics["IoU_class_3"] = float(numbers[3])
                metrics["IoU_class_4"] = float(numbers[4])
                metrics["IoU_class_5"] = float(numbers[5])
            elif line.startswith("error_map_entropy:"):
                metrics["error_map_entropy"] = float(line.split(":")[1].strip())
            elif line.startswith("total_uncertainty_entropy:"):
                metrics["total_uncertainty_entropy"] = float(line.split(":")[1].strip())
            elif line.startswith("var_based_epistemic_entropy:"):
                metrics["var_based_epistemic_entropy"] = float(line.split(":")[1].strip())
            elif line.startswith("mutual_info_entropy:"):
                metrics["mutual_info_entropy"] = float(line.split(":")[1].strip())
            elif line.startswith("total_uncertainty_auprc:"):
                metrics["total_uncertainty_auprc"] = float(line.split(":")[1].strip())
            elif line.startswith("var_based_epistemic_auprc:"):
                metrics["var_based_epistemic_auprc"] = float(line.split(":")[1].strip())
            elif line.startswith("mutual_info_auprc:"):
                metrics["mutual_info_auprc"] = float(line.split(":")[1].strip())
    return metrics


# # matched_file= Path("/home/fzhcis/Downloads/eval_metrics_3_4_5.txt")
# matched_file = Path("/home/fzhcis/mylab/data/point_cloud_segmentation/segmentation_on_unwrapped_image/palau_2024/mangrove3d/run_subset_30/outputs/eval_3_4_5/eval_metrics_3_4_5.txt")
# eval_metrics = read_selected_metrics(matched_file)
# pprint(eval_metrics)


root_dir = Path("/home/fzhcis/data/palau_2024_for_rc/mangrove3d")
matched_files = list(root_dir.rglob("outputs/eval_*/eval_metrics_*.txt"))
matched_files.sort()
pprint(matched_files)
# matched_file = matched_files[0]



# Define desired order of columns
col_names = ["subset_num_col", "channels", "oAcc", "mAcc", "mIoU", 
             "IoU_class_0", "IoU_class_1", "IoU_class_2", "IoU_class_3", "IoU_class_4", "IoU_class_5",
             "error_map_entropy", 
             "total_uncertainty_entropy", 
             "var_based_epistemic_entropy",
             "mutual_info_entropy",
             "total_uncertainty_auprc",
             "var_based_epistemic_auprc", 
             "mutual_info_auprc"]

# Collect data

eval_metrics_list = []
eval_metrics_df = pd.DataFrame(columns=col_names)
for matched_file in matched_files:
    subset_num_col = matched_file.parents[2].name
    channels = matched_file.parents[0].name
    eval_metrics = read_selected_metrics(matched_file)

    # Fill in OrderedDict in desired order
    per_row = OrderedDict()
    per_row["subset_num_col"] = subset_num_col
    per_row["channels"] = channels
    for key in col_names[2:]:  # remaining metric keys
        per_row[key] = eval_metrics.get(key, None)  # use .get in case some are missing

    # Pretty print
    print("per_row:")
    pprint(per_row)
    eval_metrics_list.append(per_row)


eval_metrics_df = pd.DataFrame(eval_metrics_list, columns=col_names)
write_path = root_dir / "eval_metrics_summary.csv"
eval_metrics_df.to_csv(write_path, index=False)

