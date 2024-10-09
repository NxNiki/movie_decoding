"""
convert cluster-less or lfp data to .npz files and save in data path:
"""

import os
import re
from typing import List, Union

import numpy as np
import pandas as pd


def convert_csv_to_npz(input_dir: str, output_dir: str, pattern: str) -> None:
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Compile the regular expression pattern
    regex = re.compile(pattern)

    # Iterate over all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".csv") and regex.match(filename):
            # Create the output file name by changing the extension to .npz
            npz_filename = os.path.splitext(filename)[0] + ".npz"
            npz_path = os.path.join(output_dir, npz_filename)

            if os.path.exists(npz_path):
                print(f"skip existing file: {npz_path}")
                continue

            csv_path = os.path.join(input_dir, filename)
            data = pd.read_csv(csv_path)
            np_array = data.to_numpy()

            # Save the NumPy array as a .npz file
            np.savez_compressed(npz_path, np_array)

            print(f"Converted {filename} to {npz_filename} and saved to {output_dir}")


def convert_neural_data(input_dirs: Union[str, List[str]], output_dirs: Union[str, List[str]], pattern: str) -> None:
    if isinstance(input_dirs, str):
        input_dirs = [input_dirs]
        output_dirs = [output_dirs]

    for input_dir, output_dir in zip(input_dirs, output_dirs):
        convert_csv_to_npz(input_dir, output_dir, pattern)


if __name__ == "__main__":
    # Usage example
    input_directory = "/Users/XinNiuAdmin/Library/CloudStorage/Box-Box/Vwani_Movie/Clusterless/562/Experiment6_MovieParadigm_notch_CAR"
    output_directory = "/Users/XinNiuAdmin/Documents/brain_decoding/data/562/notch CAR-quant-neg/time_sleep"
    file_pattern = r"clusterless_.*\.csv"

    convert_neural_data(input_directory, output_directory, file_pattern)
