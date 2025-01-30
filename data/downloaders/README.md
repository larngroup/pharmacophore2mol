#This folder contains the downloaders for the datasets used in the project. Each downloader places the outputs in the data folder, but each has its own requirements.

##ZINC 3D
They are downloaded via the ZINC tranche browser 3D. Downloader is called via:
```sh
python zinc3d.py <uri_file> [<output_file>]
```
Where:	
- `<uri_file>` is the path to the file containing the URIs of the molecules to download.
- `<output_file>` is the path to the file where the downloaded molecules will be saved. If not provided, the molecules will be saved in `data/zinc3d.sdf`. If the file already exists, the downloader ask if it should overwrite it.
