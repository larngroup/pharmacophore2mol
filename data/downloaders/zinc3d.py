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

import asyncio
import glob
import gzip
import os
import shutil
import aiohttp
import argparse

from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Download molecules from ZINC 3D.")
    parser.add_argument("uri_file", type=str, help="Path to the file containing the URIs of the molecules to download.")
    parser.add_argument("-o", "--output_file", type=str, default="zinc3d.sdf", help="Path to the file where the downloaded molecules will be saved.")
    return parser.parse_args()


async def download_file(session, uri, tempfile, semaphore, max_retries=5):
    """Download a file with retries and adaptive rate limiting."""
    retry_count = 0
    backoff = 1  # Start with 1 second backoff
    while retry_count < max_retries:
        async with semaphore:
            try:
                async with session.get(uri) as response:
                    if response.status == 503:  # Server busy, slow down
                        # print(f"503 received from {uri}, retrying in {backoff}s...")
                        await asyncio.sleep(backoff)
                        backoff *= 2  # Exponential backoff
                        retry_count += 1
                        continue
                    elif response.status != 200:
                        print(f"Error {response.status} downloading {uri}")
                        return None

                    with open(tempfile, "wb") as f:
                        f.write(await response.read())
                    return tempfile  # Success
            except Exception as e:
                print(f"Error downloading {uri}: {e}")
                return None

    print(f"Failed to download {uri} after {max_retries} retries.")
    return None

async def download_all(uris, tempfiles):
    """Download all files concurrently."""
    semaphore = asyncio.Semaphore(20) #so as not to turn this into a DoS attack
    async with aiohttp.ClientSession() as session:
        tasks = [download_file(session, uri, tempfile, semaphore) for uri, tempfile in zip(uris, tempfiles)]
        return await atqdm.gather(*tasks)
    




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


    downloaded_files = asyncio.run(download_all(uris, tempfiles))
    downloaded_files = [f for f in downloaded_files if f]

    # downloaded_files = glob.glob(tempdir + "/*.sdf.gz") # for testing


    # Extract the .gz files and append them to the output file
    with open(tempdir + "/out.sdf", "wb") as f_out:
        count = 0
        for tempfile in tqdm(downloaded_files, desc="Extracting"):
            with gzip.open(tempfile, "rb") as f_in:
                content = f_in.read()
                count += content.count(b"$$$$")
                f_out.write(content)
    shutil.move(tempdir + "/out.sdf", "../" + args.output_file) # just to make things atomic
    shutil.rmtree(tempdir)


    print(f"Downloaded {count} molecules.")