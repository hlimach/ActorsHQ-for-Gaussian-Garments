# ActorsHQ for Gaussian Garments
Repository to use actorsHQ dataset with gs2mesh to reconstruct 3D garment mesh.
## Setup
Start by cloning this repository:
```bash
git clone --recursive git@github.com:hlimach/ActorsHQ-for-Gaussian-Garments.git
```

Follow the instructions provided on the official [gs2mesh repo](https://github.com/yanivw12/gs2mesh/tree/main) page for the environment creation & setup. Then, activate the env and download the additional requirement:
```bash
pip install pyacvd
```
## Data Population
While inside the `gs2mesh` repo, you need to setup a directory containing the first frame images for a specific subject sequence from actorsHQ that must be reconstructed. 
```bash
mkdir data
cd data
mkdir custom
```
Inside the custom folder, you must create a folder with your `data_name` and place the **COLMAP compatible format** of the data. More specifically, the custom folder must contain:
```
data_name/
    ├── txt/         
    |   ├── cameras.txt 
    |   ├── images.txt 
    |   └── points3D.txt 
    └── images/
```
Where the images folder contains RGB images that are the **first** frame from **each** camera for a specific subject sequence from actorsHQ. 

**Note:** the COLMAP compatible format txt folder can be ceated by following the instructions provided in the [official actorsHQ repo](https://github.com/synthesiaresearch/humanrf) using the script under `humanrf/actorshq/toolbox/export_colmap.py`. However, please do note that their extraction script uses 0 indexing for `camera_id` within the first loop (line 14). We changed their code for `camera_lines` (ine 20) and `image_lines` (line 24) to `{camera_id+1}` to avoid errors in subsequent COLMAP scripts.

## Sparse COLMAP reconstruction
Once the data directory has been setup, run the `sparse_reconstruction.py` script to get the sparse reconstruction of the subject. This is done because gs2mesh assumes camera extrinsics information is not available, so we use our script instead of relying on their built-in COLMAP pipeline to generate a higher quality sparse mesh and provide gs2mesh with actual camera extrinsics information.

```bash
# optional: --no_gpu to curb GPU usage in COLMAP
python sparse_reconstruction.py --data <data_name>
```

This will generate `sparse/` folder inside the data directory  `data_name`, organized to fit the requirements of running subsequent gs2mesh stages. 

## Mesh Reconstruction
You may choose to either work with the interactive notebook `actorsHQ_gs2mesh.ipynb`, or run the provided file `mesh_reconstruction.py`.
The notebook includes helpful visualizations, and allows masking interactively. It is recommended to use it to debug potential issues.

To run the script `mesh_reconstruction.py`:
```bash
python mesh_reconstruction.py --colmap_name <data_name> --garment_type <gtype> --mesh_output_path <abs_path> --masker_prompt <garment_prompt> --skip_video_extraction --skip_colmap --masker_automask
```

The listed parameters in the table that follows are of importance, please set them up according to the provided guidelines. Further parameter details can be found on the [gs2mesh repo](https://github.com/yanivw12/gs2mesh/tree/main) under [Custom Data](https://github.com/yanivw12/gs2mesh?tab=readme-ov-file#custom-data), which we suggest checking out as well.

| Parameter       | Description                                                                 | Default Value | Required |
|-----------------|-----------------------------------------------------------------------------|---------------|----------|
| `colmap_name`     | Name of the directory containing the dataset. Corresponds to `data_name` from previous stage.                                | `None`        | Yes      |
| `garment_type`     | The garment label to be processed, must be one of [upper, lower, dress], where upper corresponds to tops, sweaters, jackets, etc., lower corresponds to pants, shorts, etc., and dress is self-explanatory.                                   | `None`          | Yes       |
| `mesh_output_path`       | The absolute path to the directory where you require the final(cleaned) garment mesh is to be stored.                                | `None`        | Yes       |
| `masker_prompt`       | Prompt for GroundingDINO to segment out the garment of intrest. A short description (e.g. green_dress) suffices.                                | `None`        | Yes       |


Helpful suggestion: In case the garment mesh is generated but cleaned mesh has 0 vertices, reduce the `TSDF_cleaning_threshold` default value for small garments, and increase for larger ones.

## Mask Generation
Use the script `mask_generation.py` to generate masks based on garment prompt. 
```bash
python mask_generation.py --imgs_dir <abs path> --output_root <abs path> --prompt <garment>
```
Note that the script generates masks only for the portrait cameras from ActorsHQ, which are specified in `config.yaml`. If you wish to generate masks for all cameras, simply add onto the camera list. However, do note that GroundingDINO produces false positives and negatives in horizontal frames, where the garment is minimally visible or completely out of frame, which we were not able to reliably curb.

| Parameter       | Description                                                                 | Default Value | Required |
|-----------------|-----------------------------------------------------------------------------|---------------|----------|
| `imgs_dir`     | The absolute path to the directory where the CamXXX folders are stored (i.e. path/to/folder that contains Cam001, Cam002, ...).                                | `None`        | Yes      |
| `output_root`     | The absolute path to the root of the mask directory where the CamXXX folders are to be created and populated with masks (i.e. path/to/folder/masks that will get populated with Cam001, Cam002, ... folders containing mask images).                                   | `None`          | Yes       |
| `prompt`       | Prompt for GroundingDINO to locate object (garment) of interest. A short description (e.g. green_dress) suffices.                               | `None`        | Yes       |

Further parameters specific to GroundingDINO are set in `config.yaml`, which will most likely not require tuning in our use-case.