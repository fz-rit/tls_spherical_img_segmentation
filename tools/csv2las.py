import os
import laspy
import numpy as np
import pandas as pd

csv_folder = '/home/fzhcis/mylab/data/point_cloud_segmentation/palau_2024/test/csv'
las_folder = '/home/fzhcis/mylab/data/point_cloud_segmentation/palau_2024/test/las'
os.makedirs(las_folder, exist_ok=True)

csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]

for csv_file in csv_files:
    csv_path = os.path.join(csv_folder, csv_file)
    df = pd.read_csv(csv_path, sep=r'\s+|,', engine='python')

    # Required fields from CSV
    x = df['X'].values
    y = df['Y'].values
    z = df['Z'].values
    intensity = df['Intensity'].astype(np.uint16).values
    classification = df['class_id'].astype(np.uint8).values

    # Create LAS header: Version 1.4, Point Format 6 (minimal modern format)
    header = laspy.LasHeader(point_format=3, version="1.3")
    las = laspy.LasData(header)

    # Assign required fields
    las.x = x
    las.y = y
    las.z = z
    las.intensity = intensity
    las.classification = classification  # Standard field usable in MATLAB

    # Before writing, check the dimensions in the current LAS data
    dims = las.point_format.dimension_names
    print(f"\nFor file '{csv_file}':")
    print("Dimension names:", dims)
    if 'intensity' in dims:
        print("✅ Intensity attribute is present.")
    else:
        print("❌ Intensity attribute is missing.")
    if 'classification' in dims:
        print("✅ Classification attribute is present.")
    else:
        print("❌ Classification attribute is missing.")

    # Save the LAS file
    las_name = os.path.splitext(csv_file)[0] + '.las'
    las_path = os.path.join(las_folder, las_name)
    las.write(las_path)

    print(f"✅ Converted {csv_file} → {las_name} [X, Y, Z, Intensity, Classification]")
