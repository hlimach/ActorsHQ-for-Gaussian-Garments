# gs2mesh-for-actorsHQ
Repo that uses actorsHQ dataset with gs2mesh repo to reconstruct 3D meshes. 
## Setup
Clone the [gs2mesh repo](https://github.com/yanivw12/gs2mesh/tree/main) inside this repo by following the instructions provided on their repo for the environment creation & repo setup.
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
Once the data directory has been setup, run the `sparse_reconstruction.py` script to get the sparse reconstruction. Gs2mesh assumes camera extrinsics information is not available, so we use a custom script instead of relying on their built-in COLMAP reconstruction to generate higher quality sparse mesh and provide gs2mesh with actual camera extrinsics information.

```bash
# optional: --no_gpu to curb GPU usage in COLMAP
python sparse_reconstruction.py --data data_name
```

This will generate `sparse/` folder inside the data directory  `data_name`, organized to fit the requirements of running subsequent gs2mesh stages. 


