import os
import shutil
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser
from scipy.spatial.transform import Rotation as R

from defaults import DEFAULTS
gs2mesh_path = Path.cwd() / 'gs2mesh'

def init_parser():
    parser = ArgumentParser("COLMAP data setup and sparse reconstruction arguments.")

    parser.add_argument("--subject", "-s", required=True, type=str, help="Subject folder name that contains the sequence folders (e.g. Actor06).")
    parser.add_argument("--sequence", "-q", required=True, type=str, help="Sequence folder name (e.g. Sequence1).")
    parser.add_argument("--resolution", "-r", default='4x', type=str, help="Resolution folder of ActorsHQ images (e.g. 1x).")
    parser.add_argument("--ff", default=0, type=int, help="Frame number to use as the first frame for each camera.")
    parser.add_argument("--no_gpu", action='store_true')

    return parser


def prepare_gs2mesh_data_folder(args):
    """
    Prepare the data folder for gs2mesh using the provided arguments.

    Returns:
        _root (Path): Path to the root of the data folder for gs2mesh.
        _txt (Path): Path to the 'txt' subfolder of the data folder for gs2mesh.
        _images (Path): Path to the 'images' subfolder of the data folder for gs2mesh.
    """
    _root = gs2mesh_path / 'data' / 'custom' / f'{args.subject}_{args.sequence}'
    _root.mkdir(parents=True, exist_ok=True)

    _txt = _root / 'txt'
    _txt.mkdir(parents=True, exist_ok=True)

    _images = _root / 'images'
    _images.mkdir(parents=True, exist_ok=True)

    print(f"\nData folder for gs2mesh created at: {_root}\n")
    return _root, _txt, _images


def export_first_frames(args, _images):
    """
    Creates a folder subject_sequence and populates it with the COLMAP compatible format of the data. More specifically, an images/ subdir containing the first frame from each camera, and a txt/ subdir containing the calibration data in COLMAP format.

    Parameters:
        args (Namespace): Arguments from the command line.
        _images (Path): Path to the 'images' subfolder of the gs2mesh data folder.
    """

    images_root = DEFAULTS['AHQ_data_root'] / args.subject / args.sequence / args.resolution / 'rgbs'

    print(f"Copying first frames to {_images}")
    for cam in tqdm(images_root.iterdir()):
        cid = cam.name

        ff_src = images_root / cid / f'{cid}_rgb{args.ff:06d}.jpg'
        ff_dest = _images / f'{cid}.jpg'

        if not ff_src.exists():
            continue
        shutil.copy(ff_src, ff_dest)
    
    print(f"First frames copied successfully.\n")


def convert_local_to_colmap(R_local, t_local):
    """
    Convert camera extrinsics from camera-to-world (local) to world-to-camera (global) as used by COLMAP.
    
    Parameters:
      R_local (np.ndarray): 3x3 rotation matrix (camera-to-world)
      t_local (np.ndarray): 3x1 translation vector (camera position in world coords)
    
    Returns:
      R_global (np.ndarray): 3x3 rotation matrix (world-to-camera)
      t_global (np.ndarray): 3x1 translation vector (as expected by COLMAP)
    """
    R_global = R_local.T
    t_global = - R_local.T @ t_local
    return R_global, t_global


def export_colmap_format(args, _txt):
    """
    Export the calibration data in COLMAP compatible format in 'txt' subdir.

    Parameters:
        args (Namespace): Arguments from the command line.
        _txt (Path): Path to the 'txt' subfolder of the gs2mesh data folder.
    """

    input_csv = DEFAULTS['AHQ_data_root'] / args.subject / args.sequence / args.resolution / 'calibration.csv'
    calibration = pd.read_csv(input_csv, index_col=0)

    camera_lines = ""
    image_lines = ""
    cid = 0
    for row in calibration.iloc:
        camera_name = row.name
        image_name = f'{camera_name}.jpg'

        rotvec = np.array([row['rx'], row['ry'], row['rz']])#.astype(np.float32)
        translation = np.array([row['tx'], row['ty'],row['tz']])#.astype(np.float32)
        focal = np.array([row['fx'], row['fy']])#.astype(np.float32) 
        principal = np.array([row['px'], row['py']])#.astype(np.float32) 

        w, h = [int(row['w']), int(row['h'])]

        focal *= np.array([w, h])
        principal *= np.array([w, h])

        rotmat = R.from_rotvec(rotvec).as_matrix()    
        rotmat, translation = convert_local_to_colmap(rotmat, translation)
        quaternion = R.from_matrix(rotmat).as_quat()
        quaternion = np.roll(quaternion, 1)

        image_lines += f"{cid} {quaternion[0]} {quaternion[1]} {quaternion[2]} {quaternion[3]} {translation[0]} {translation[1]} {translation[2]} {cid} {image_name}\n\n"
        camera_lines += f"{cid} PINHOLE {w} {h} {focal[0]} {focal[1]} {principal[0]} {principal[1]}\n"
        cid += 1

    # Write intrinsics to cameras.txt
    with open(_txt / "cameras.txt", "w") as f:
        f.write(camera_lines)

    # Write extrinsics to images.txt
    with open(_txt / "images.txt", "w") as f:
        f.write(image_lines)

    # Write an empty points3D.txt file
    with open(_txt / "points3D.txt", "w") as f:
        f.write("# Empty file...\n")


def run_colmap(_root, _txt, _images, use_gpu=1):
    _sparse = _root / 'sparse'
    _dense = _root / 'dense'
    
    ## Feature extraction
    feat_extracton_cmd = "colmap feature_extractor " +\
                        f"--database_path {_root}/database.db " +\
                        f"--image_path {_images} " +\
                        f"--SiftExtraction.use_gpu {use_gpu} "
    exit_code = os.system(feat_extracton_cmd)
    if exit_code != 0:
        logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ## Feature matching
    feat_matching_cmd = "colmap exhaustive_matcher " +\
                       f"--database_path {_root}/database.db " +\
                       f"--SiftMatching.use_gpu {use_gpu} " +\
                       f"--ExhaustiveMatching.block_size 200 "
    exit_code = os.system(feat_matching_cmd)
    if exit_code != 0:
        logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ## Point triangulation
    triangulate_cmd = "colmap point_triangulator " +\
                      f"--database_path {_root}/database.db " +\
                      f"--image_path {_images} " +\
                      f"--input_path {_txt} " +\
                      f"--output_path {_sparse} "
    _sparse.mkdir(parents=True, exist_ok=True)
    exit_code = os.system(triangulate_cmd)
    if exit_code != 0:
        logging.error(f"Point triangulation failed with code {exit_code}. Exiting.")
        exit(exit_code)
    
    ## Image undistortion
    _dense.mkdir(parents=True, exist_ok=True)
    img_undist_cmd = "colmap image_undistorter " +\
                     f"--image_path {_images} " +\
                     f"--input_path {_sparse} " +\
                     f"--output_path {_dense}"
    exit_code = os.system(img_undist_cmd)
    if exit_code != 0:
        logging.error(f"Image Undistortion failed with code {exit_code}. Exiting.")
        exit(exit_code)

    # Convert the model to .txt files
    os.system(f"colmap model_converter " +\
    f"--input_path {os.path.join(_dense, 'sparse')} " +\
    f"--output_path {os.path.join(_dense, 'sparse')} " +\
    "--output_type TXT")

    # Reorganize to meet gs2mesh requirements, and remove unnecessary outputs
    os.system(f"rm -rf {_sparse}")
    os.makedirs(_sparse, exist_ok=True)
    os.makedirs(os.path.join(_sparse, '0'), exist_ok=True)
    os.system(f"mv {os.path.join(_dense, 'sparse/*')} {os.path.join(_sparse, '0')}")
    os.system(f"rm -rf {_dense}")
    os.system(f"rm -rf {_root}/database.db")


def main():
    """
    Main function to run the COLMAP pipeline for a given subject.
    Prepare the data folder for gs2mesh using the provided arguments.
    Export the calibration data in COLMAP compatible format in 'txt' subdir.
    Runs the sparse COLMAP reconstruction pipeline.
    """
    parser = init_parser()
    args = parser.parse_args()

    _root, _txt, _images = prepare_gs2mesh_data_folder(args)
    export_first_frames(args, _images)
    export_colmap_format(args, _txt)
    
    run_colmap(_root, _txt, _images, use_gpu=int(not args.no_gpu))


if __name__ == "__main__":
    main()