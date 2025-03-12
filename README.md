# ActorsHQ for Gaussian Garments
Repository to use actorsHQ dataset with gs2mesh to reconstruct 3D garment mesh.
## Setup
Start by cloning this repository:
```bash
git clone --recursive git@github.com:hlimach/ActorsHQ-for-Gaussian-Garments.git
```

Follow the instructions provided on the official [gs2mesh repo](https://github.com/yanivw12/gs2mesh/tree/main) page for the environment creation & setup. Then, activate the env and download the additional requirements:
```bash
pip install pyacvd
pip install munch
```

**Add information about defaults.py and input data structure expectations.**

## Data Setup & Mesh Initialization
Run the provided script `mesh_initialization.py`:
```bash
python mesh_initialization.py --subject Actor0X --sequence SequenceX
``` 
This script prepares the custom data folder for gs2mesh using the provided arguments, exports the calibration data in COLMAP compatible format in 'txt' subdir, and runs our sparse COLMAP reconstruction pipeline. 

**Note:** We use our own COLMAP pipeline because gs2mesh assumes camera extrinsics information is not available, so we use our script instead of relying on their built-in COLMAP pipeline to generate a higher quality sparse mesh and provide gs2mesh with actual camera extrinsics information.

<details>
<summary> Parameters (click to expand) </summary>

| Parameter       | Description                                                                 | Default Value | Required |
|-----------------|-----------------------------------------------------------------------------|---------------|----------|
| `--subject`  `-s`   | Subject folder name that contains the sequence folders (e.g. Actor06).                                     | `None`        | Yes      |
| `--sequence`   `-q` | Sequence folder name (e.g. Sequence1).                                  | `None`        | Yes      |
| `--resolution` `-r` | Resolution folder of ActorsHQ images (e.g. 1x).                                         | `4x`    | No       |
| `--ff` | Frame number to use as the first frame for each camera.                                              | 0 | No |
| `--no_gpu` | Whether to use GPU for feature extraction and matching.                                    | False           | No       |

</details>

After a successful run, the `gs2mesh/data/custom/` should contain a `subject_sequence/` subdirectory containing the following:
```
subject_sequence/
    ├── txt/         
    |   ├── cameras.txt 
    |   ├── images.txt 
    |   └── points3D.txt 
    ├── sparse/         
    |   ├── 0/         
    |   |   ├── cameras.txt 
    |   |   ├── cameras.bin 
    |   |   ├── images.txt 
    |   |   ├── images.bin
    |   |   ├── points3D.txt
    |   └── └── points3D.bin
    ├──images/     
    |   ├── Cam001.jpg
    |   ├── Cam002.jpg 
    |   └── ...
```
Where the images folder contains the first frame RGB images from each camera for this subject-sequence. Please do not move/ reorganize this folder, as it is imperative to running the subsequent gs2mesh stages. 

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