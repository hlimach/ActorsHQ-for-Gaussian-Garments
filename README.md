# ActorsHQ for Gaussian Garments
Official supplementary repository for [Gaussian Garments](https://ribosome-rbx.github.io/Gaussian-Garments/): Stage 1 - Garment Initialization. This repository is tailored for ActorsHQ dataset for generating segmented garment mesh using [gs2mesh](https://gs2mesh.github.io/).
## Setup
### Environment
Start by cloning this repository:
```bash
git clone --recursive git@github.com:hlimach/ActorsHQ-for-Gaussian-Garments.git
```

Follow the instructions provided on the official [gs2mesh repo](https://github.com/yanivw12/gs2mesh/tree/main) page for the environment creation & setup. Then, activate the env and install the following additional requirements:
```bash
pip install pyacvd munch gdown smplx
```
### Data
Setup the `defaults.py` file with the necessary data paths. Note that it is assumed in our scripts that your ActorsHQ dataset directory is in the format that it is originally downloaded in.

## Mesh Initialization
Run the provided script `mesh_initialization.py`:
```bash
python mesh_initialization.py --subject Actor0X --sequence SequenceX
``` 
This script prepares the custom data folder for gs2mesh using the provided arguments, exports the calibration data in COLMAP compatible format, and runs our sparse COLMAP reconstruction pipeline. 

**Note:** We use our own COLMAP pipeline because gs2mesh assumes camera extrinsics information is not available, so instead of relying on their built-in COLMAP pipeline, we use our own to generate a higher quality sparse mesh and provide gs2mesh with actual camera extrinsics information.

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

**Helpful suggestions:** 
1. Ensure that the argument `--ff` is set to the number corresponding to the frame where the subject pose is appropriate for reconstruction i.e. hands are away from the body.
2. In general, pick garments that are not occluded by other features (pants not fully visible due to shirt, shirt tucked in, hair occluding part of the top, etc).

After a successful run, the `gs2mesh/data/custom/` directory should contain a `subject_sequence/` subdirectory containing the following:
```
subject_sequence/
    ├── txt/         
    |   ├── cameras.txt 
    |   ├── images.txt 
    |   └── points3D.txt 
    ├── sparse/         
    |   └── 0/         
    |       ├── cameras.txt 
    |       ├── cameras.bin 
    |       ├── images.txt 
    |       ├── images.bin
    |       ├── points3D.txt
    |       └── points3D.bin
    └── images/     
        ├── Cam001.jpg
        ├── Cam002.jpg 
        └── ...
```
Where the images folder contains the specified 'first frame' RGB images from each camera for this subject-sequence. Please do not move/ reorganize this folder, as it is imperative to running the subsequent gs2mesh stages. 

## Mesh Reconstruction
You may choose to either work with the interactive notebook `mesh_reconstruction.ipynb`, or run the provided script `mesh_reconstruction.py`.
The notebook includes helpful visualizations, and allows masking interactively. It is recommended to use it to debug potential issues.

To run the script `mesh_reconstruction.py`:
```bash
python mesh_reconstruction.py --subject Actor0X --sequence SequenceX --garment_type Gtype --masker_prompt Prompt --masker_automask 
```
<details>
<summary> Parameters (click to expand) </summary>

| Parameter       | Description                                                                 | Default Value | Required |
|-----------------|-----------------------------------------------------------------------------|---------------|----------|
| `--subject`  `-s`   | Subject folder name that contains the sequence folders (e.g. Actor06).                                     | `None`        | Yes      |
| `--sequence`   `-q` | Sequence folder name (e.g. Sequence1).                                  | `None`        | Yes      |
| `--garment_type`   `-g`  | The garment label to be processed, must be one of [upper, lower, dress], where upper corresponds to tops, sweaters, jackets, etc., lower corresponds to pants, shorts, etc., and dress is self-explanatory.                                   | `None`          | Yes       |
| `--masker_prompt`       | Prompt for GroundingDINO to segment out the garment of intrest. A short description (e.g. green_dress) suffices.                                | `None`        | Yes       |
| `--masker_automask`       | Internal gs2mesh flag that must be passed to trigger garment segmentation.                                | -        | Yes       |

Further parameter details can be found on the [gs2mesh repo](https://github.com/yanivw12/gs2mesh/tree/main) under [Custom Data](https://github.com/yanivw12/gs2mesh?tab=readme-ov-file#custom-data), which we suggest checking out as well.
</details>

**Helpful suggestions:** 
1. In case the garment mesh is generated but cleaned mesh has 0 vertices, reduce the `--TSDF_cleaning_threshold` default value by 10x for small garments, and increase for larger ones.
2. The stages of gs2mesh are run in the following order: Gaussian Splatting, Rendering, Masking, TSDF. If you only wish to continue your run from a certain stage, you can skip the previous stages using their respective skip flags:
    - Gaussian Splitting `--skip_GS`
    - Rendering `--skip_rendering`
    - Masking `--skip_masking`
    - TSDF `--skip_TSDF`

After a successful run, the `DEFAULTS.output_root` should contain a `subject/` subdirectory containing the following:
```
subject/
    └── stage1/
        ├── point_cloud.ply
        ├── template.obj
        └── sparse/     
            └── points3D.bin
```
These are the minimal output requirements from Stage 1: Garment Initialization, and are imperative to running the subsequent stages of Gaussian Garments. Note that you must place the `template_uv.obj` file in this subdirectory after manually adding garment seams as required by Gaussian Garments.

## Data Preparation
Use the script `data_preparation.py` to setup the ActorsHQ data in a Gaussian Garments compatible format. 
```bash
python data_preparation.py --subject Actor0X --sequence SequenceX --masker_prompt Prompt --gender Gender
```

<details>
<summary> Parameters (click to expand) </summary>

| Parameter       | Description                                                                 | Default Value | Required |
|-----------------|-----------------------------------------------------------------------------|---------------|----------|
| `--subject`  `-s`   | Subject folder name that contains the sequence folders (e.g. Actor06).                                     | `None`        | Yes      |
| `--sequence`   `-q` | Sequence folder name (e.g. Sequence1).                                  | `None`        | Yes      |
| `--masker_prompt` `-p`      | Prompt for GroundingDINO to segment out the garment of intrest. A short description (e.g. green_dress) suffices.                                | `None`        | Yes       |
| `--gender`   `-g`  | Gender of the SMPLX model, must be one of [male, female], corresponding to gender of subject.                                   | `None`          | Yes       |
| `--resolution` `-r` | Resolution folder of ActorsHQ images (e.g. 1x).                                         | `4x`    | No       |

Further parameters specific to GroundingDINO are set in `defaults.py`, which will most likely not require tuning in our use-case.
</details>

After a successful run, the `DEFAULTS.data_root` should contain a subdirectory `subject/sequence`, populated with the following:
```
subject/
    └── sequence/
        ├── Cam001
        |   ├── *rgb_images
        |   ├── *foreground_masks
        |   └── garment_masks     
        ├── Cam002
        ├── ...
        └── smplx
            ├── 000000.pkl
            ├── 000000.ply
            ├── 000001.pkl
            └── ...  
```
Where the folders `rgb_images` and `foreground_masks` are symbolic links to `Defaults.AHQ_data_root`, and `garment_masks` is physically stored at this location with the segmented garment masks for each frame.
The `smplx` subdirectory contains the extracted SMPLX parameters for each frame of this sequence in a `.pkl` file, along with its point cloud in a `.ply` file. 

**Note:** the script only generates masks for the portrait cameras from ActorsHQ, which are specified in `defaults.py`. This is because GroundingDINO produces false positives and negatives in horizontal frames, where the garment is minimally visible or completely out of frame, which we were not able to reliably curb. If you still wish to generate masks for all cameras, simply add onto the camera list.