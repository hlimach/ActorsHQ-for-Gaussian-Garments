import os
import json
import logging
from argparse import ArgumentParser

def colmap_parser():
    parser = ArgumentParser("Stage 1: Garment initialization.")
    parser.add_argument("--data", "-d", required=True, type=str, help="Folder name inside ./gs2mesh/data/custom/ that contains the data.")
    parser.add_argument("--no_gpu", action='store_true')
    return parser


def colmap(source_path, use_gpu=1):
    # input data folder
    _images = os.path.join(source_path, "images")
    _txt = os.path.join(source_path, "txt")
    _sparse = os.path.join(source_path, "sparse")
    _dense = os.path.join(source_path, "dense")
    
    ## Feature extraction
    feat_extracton_cmd = "colmap feature_extractor " +\
                        f"--database_path {source_path}/database.db " +\
                        f"--image_path {_images} " +\
                        f"--SiftExtraction.use_gpu {use_gpu} "
    exit_code = os.system(feat_extracton_cmd)
    if exit_code != 0:
        logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ## Feature matching
    feat_matching_cmd = "colmap exhaustive_matcher " +\
                       f"--database_path {source_path}/database.db " +\
                       f"--SiftMatching.use_gpu {use_gpu} " +\
                       f"--ExhaustiveMatching.block_size 200 "
    exit_code = os.system(feat_matching_cmd)
    if exit_code != 0:
        logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ## point triangulate
    triangulate_cmd = "colmap point_triangulator " +\
                      f"--database_path {source_path}/database.db " +\
                      f"--image_path {_images} " +\
                      f"--input_path {_txt} " +\
                      f"--output_path {_sparse} "
    os.makedirs(_sparse, exist_ok=True)
    exit_code = os.system(triangulate_cmd)
    if exit_code != 0:
        logging.error(f"Point triangulation failed with code {exit_code}. Exiting.")
        exit(exit_code)
    
    ### Image undistortion
    ## We need to undistort our images into ideal pinhole intrinsics.
    os.makedirs(_dense, exist_ok=True)
    img_undist_cmd = "colmap image_undistorter " +\
                     f"--image_path {_images} " +\
                     f"--input_path {_sparse} " +\
                     f"--output_path {_dense}"
    exit_code = os.system(img_undist_cmd)
    if exit_code != 0:
        logging.error(f"Image Undistortion failed with code {exit_code}. Exiting.")
        exit(exit_code)
    
    # convert the model to txt
    os.system(f"colmap model_converter " +\
    f"--input_path {os.path.join(_dense, 'sparse')} " +\
    f"--output_path {os.path.join(_dense, 'sparse')} " +\
    "--output_type TXT")

    # reorganize according to gs2mesh requirements 
    # and remove intermediate folders created by colmap
    os.system(f"rm -rf {_sparse}")
    os.makedirs(_sparse, exist_ok=True)
    os.makedirs(os.path.join(_sparse, '0'), exist_ok=True)
    os.system(f"mv {os.path.join(_dense, 'sparse/*')} {os.path.join(_sparse, '0')}")
    os.system(f"rm -rf {_dense}")


def main():
    parser = colmap_parser()
    args = parser.parse_args()

    path = os.path.join("gs2mesh", "data", "custom", args.data)
    if not os.path.exists(path):
        logging.error(f"Data folder {path} does not exist. Exiting.")
        exit(1)
    
    colmap(path, use_gpu=int(not args.no_gpu))


if __name__ == "__main__":
    main()