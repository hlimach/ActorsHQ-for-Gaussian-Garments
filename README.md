# gs2mesh-for-actorsHQ
Repository to use actorsHQ dataset with gs2mesh to reconstruct 3D garment mesh.
## Setup
Start by cloning this repository:
```bash
git clone https://github.com/hlimach/gs2mesh-for-actorsHQ.git
```

Then clone the official [gs2mesh repo](https://github.com/yanivw12/gs2mesh/tree/main) inside this repo by following the instructions provided on their page for the environment creation & setup.
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
python sparse_reconstruction.py --data <data_name>
```

This will generate `sparse/` folder inside the data directory  `data_name`, organized to fit the requirements of running subsequent gs2mesh stages. 

## Mesh Generation
You may choose to either work with the interactive notebook `actorsHQ_gs2mesh.ipynb`, or run the provided file `mesh_reconstruction.py`.
The notebook includes helpful visualizations, and allows masking interactively. It is highly recommended to use it, at least on the first try.

To run the script `mesh_reconstruction.py`:
```bash
python mesh_reconstruction.py --colmap_name <data_name> --garment_type <gtype> --mesh_output_path <abs_path> --masker_prompt <prompt> --skip_video_extraction --skip_colmap --masker_automask
```

The listed parameters in the table that follows are of importance, please set them up according to the provided guidelines. Further parameter details can be found on the [gs2mesh repo](https://github.com/yanivw12/gs2mesh/tree/main) under [Custom Data](https://github.com/yanivw12/gs2mesh?tab=readme-ov-file#custom-data), which we suggest checking out as well.

| Parameter       | Description                                                                 | Default Value | Required |
|-----------------|-----------------------------------------------------------------------------|---------------|----------|
| `colmap_name`     | Name of the directory containing the dataset. Corresponds to `data_name` from previous stage.                                | `None`        | Yes      |
| `garment_type`     | The garment label to be processed, must be one of [upper, lower, dress], where upper corresponds to tops, sweaters, jackets, etc., lower corresponds to pants, shorts, etc., and dress is self-explanatory.                                   | `None`          | Yes       |
| `mesh_output_path`       | The absolute path to the directory where you require the final(cleaned) garment mesh is to be stored.                                | `None`        | Yes       |
| `masker_prompt`       | Prompt for GroundingDINO to segment out the garment of intrest. A short description e.g. green dress) suffices.                                | `None`        | Yes       |
| `TSDF_cleaning_threshold`    | Minimal cluster size for clean mesh. In case the garment mesh is generated but cleaned mesh has 0 vertices, reduce this threshold for small garments, and increase for larger ones.                                      | `100000`      | No       |
