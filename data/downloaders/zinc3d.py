"""
The molecules are downloaded via the ZINC tranche browser 3D. Downloader is called via:
```sh
python zinc3d.py <uri_file> [<output_file>]
```
Where:	
- `<uri_file>` is the path to the file containing the URIs of the molecules to download.
- `<output_file>` is the path to the file where the downloaded molecules will be saved.
If not provided, the molecules will be saved in `data/zinc3d.sdf`.
If the file already exists, the downloader ask if it should overwrite it.

The URIs file should contain one URL per line.

This script downloads all the .sdf.gz files, extracts them, and concatenates them into
a single .sdf file, which is saved in the output file.
"""

import gzip
import os
import shutil
import requests
import argparse

from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Download molecules from ZINC 3D.")
    parser.add_argument("uri_file", type=str, help="Path to the file containing the URIs of the molecules to download.")
    parser.add_argument("-o", "--output_file", type=str, default="zinc3d.sdf", help="Path to the file where the downloaded molecules will be saved.")
    return parser.parse_args()

if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), "."))
    args = parse_args()
    # Check if the input file exists
    if not os.path.isfile(args.uri_file):
        raise FileNotFoundError(f"File {args.uri_file} not found in this directory: {os.getcwd()}.")
    # Check if the output file exists
    if os.path.isfile(args.output_file):
        if input(f"File {args.output_file} already exists. Overwrite? (y/n): ").lower() != "y":
            print("Aborted.")
            exit()

    # Download the .gz files
    with open(args.uri_file, "r") as f:
        uris = f.read().splitlines()
    
    tempdir = "temp"
    if os.path.exists(tempdir):
        shutil.rmtree(tempdir)
    os.makedirs(tempdir)

    tempfiles = [f"{tempdir}/{args.output_file}_{i}.sdf.gz" for i in range(len(uris))]
    for uri, tempfile in tqdm(zip(uris, tempfiles), desc="Downloading", total=len(uris)):
        response = requests.get(uri)
        response.raise_for_status()
        with open(tempfile, "wb") as f:
            f.write(response.content)


    # Extract the .gz files and append them to the output file
    with open(args.output_file, "wb") as f_out:
        count = 0
        for tempfile in tqdm(tempfiles, desc="Extracting"):
            with gzip.open(tempfile, "rb") as f_in:
                content = f_in.read()
                count += content.count(b"$$$$")
                f_out.write(content)
    shutil.rmtree(tempdir)
    print(f"Downloaded {count} molecules.")

