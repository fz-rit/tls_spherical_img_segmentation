import json
from pprint import pprint

# Replace 'your_file.json' with the path to your JSON file
json_file_path = '/home/fzhcis/mylab/data/point_cloud_segmentation/segmentation_on_unwrapped_image/palau_2024/train_val_test_split.json'

# Load and print the JSON content
with open(json_file_path, 'r') as file:
    data = json.load(file)

pprint(data)