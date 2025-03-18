# =============================================================================
#  IMPORTANT: 
# =============================================================================
# This script is a modified replica of the run_singple.py script provided in
# the gs2mesh repository. In order to make variable image sizes work, we have 
# implemented patches to the intermediate steps and outputs of gs2mesh. 

# DO NOT change the order execution. The functionality is dependent on it.
# PLEASE follow the guidelines provided for data population and COLMAP  
# reconstruction in our repository before running this script.

# Please note:
# If the cleaned tsdf mesh saving throws an error saying PLY could not be saved
# because it has 0 vertices, please adjust the args.TSDF_cleaning_threshold 
# parameter. For smaller garments such as shorts, a 10x reduction from default
# value is sufficient.

# =============================================================================
#  Imports
# =============================================================================

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pyacvd
import pyvista as pv
import vtk
vtk.vtkObject.GlobalWarningDisplayOff()

import shutil
import types
from sam2.sam2_video_predictor import SAM2VideoPredictor
import open3d as o3d
import sys
from pathlib import Path
from defaults import DEFAULTS

gs2mesh_path = os.path.join(os.getcwd(), "gs2mesh")

# Changes the working directory to gs2mesh
os.chdir(gs2mesh_path)

# Adds gs2mesh to sys.path so Python can find modules inside it
sys.path.insert(0, gs2mesh_path)

from gs2mesh_utils.argument_utils import ArgParser
from gs2mesh_utils.colmap_utils import extract_frames, create_downsampled_colmap_dir, run_colmap
from gs2mesh_utils.eval_utils import create_strings
from gs2mesh_utils.renderer_utils import Renderer
from gs2mesh_utils.stereo_utils import Stereo
from gs2mesh_utils.masker_utils import init_predictor, Masker
from gs2mesh_utils.tsdf_utils import TSDF

device = 'cuda' if torch.cuda.is_available() else 'cpu'
base_dir = os.path.abspath(os.getcwd())

# =============================================================================
#  ActorsHQ garment reconstruction specific parameters and functions
# =============================================================================

def actorshq_params(parser):
    parser.add_argument("--subject", "-s", required=True, type=str, help="Subject folder name that contains the sequence folders (e.g. Actor06).")
    
    parser.add_argument("--sequence", "-q", required=True, type=str, help="Sequence folder name (e.g. Sequence1).")
    
    parser.add_argument("--garment_type", "-g", required=True, type=str, help="The garment label to be processed, must be one of [upper, lower, dress], where upper corresponds to tops, sweaters, jackets, etc., lower corresponds to pants, shorts, etc., and dress is self-explanatory.")
    return parser


def setup_stage1_outputs(subject):
    """
    Creates the stage1 output directory for the given subject and returns the path.
    """
    _stage1 = DEFAULTS['output_root'] / subject / 'stage1'
    _stage1.mkdir(parents=True, exist_ok=True)
    return _stage1


def set_references(garment_type):
    if garment_type.lower() == "upper":
        hor_ref, vert_ref = 1, 0
    elif garment_type.lower() in ["lower", "dress"]:
        hor_ref, vert_ref = 0, 0
    else:
        raise ValueError(f"Invalid garment type: {garment_type}")
    return hor_ref, vert_ref


def segment_custom(self, images_dir):
    """
    Patch on Masker class to handle variable size images data logic
    """
    image_filenames = [
            int(p.split('.')[0]) for p in os.listdir(images_dir)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    image_filenames.sort()
    
    for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state):
        mask = (out_mask_logits[0] > 0.0).cpu().numpy().squeeze(0)
        idx = image_filenames[out_frame_idx]
        output_dir = self.renderer.render_folder_name(idx)
        np.save(os.path.join(output_dir, 'left_mask.npy'), mask)
        plt.imsave(os.path.join(output_dir, 'left_mask.png'), mask)
    plt.close('all')


def run_full_masker(orientation, GD_model, renderer, stereo, args, images_dir):
    subdir_bin = orientation["subdir"]
    predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-large", device=device)
    images_subdir = os.path.join(images_dir, subdir_bin)
    inference_state = predictor.init_state(video_path=images_subdir)
    
    masker = Masker(GD_model, predictor, inference_state, images_subdir, renderer, stereo, args, image_number=orientation["ref"], visualize=False)
    masker.segment_custom = types.MethodType(segment_custom, masker)
    masker.segment_custom(images_subdir)
    return masker

# =============================================================================
#  Run
# =============================================================================

def run_single(args):
    TSDF_voxel_length=args.TSDF_voxel/512
    colmap_dir = os.path.abspath(os.path.join(base_dir,'data',args.dataset_name,args.colmap_name))
    
    strings = create_strings(args)
    h_ref, v_ref = set_references(args.garment_type)
    _stage1 = setup_stage1_outputs(args.subject)

    # =============================================================================
    #  Run Gaussian Splatting
    # =============================================================================
    
    if not args.skip_GS:
        try:
            os.chdir(os.path.join(base_dir, 'third_party', 'gaussian-splatting'))
            iterations_str = ' '.join([str(iteration) for iteration in args.GS_save_test_iterations])
            os.system(f"python train.py -s {colmap_dir} --port {args.GS_port} --model_path {os.path.join(base_dir, 'splatting_output', strings['splatting'], args.colmap_name)} --iterations {args.GS_iterations} --test_iterations {iterations_str} --save_iterations {iterations_str}{' --white_background' if args.GS_white_background else ''}")
            os.chdir(base_dir)
        except:
            os.chdir(base_dir)
            print("ERROR")

    # =============================================================================
    #  Initialize renderer
    # =============================================================================
    
    renderer = Renderer(base_dir, 
                        colmap_dir,
                        strings['output_dir_root'],
                        args,
                        dataset = strings['dataset'], 
                        splatting = strings['splatting'],
                        experiment_name = strings['experiment_name'],
                        device=device)

    # =============================================================================
    #  Prepare renderer
    # =============================================================================
    
    if not args.skip_rendering:
        renderer.prepare_renderer()

    # =============================================================================
    #  Initialize stereo
    # =============================================================================
    
    stereo = Stereo(base_dir, renderer, args, device=device)

    # =============================================================================
    #  Run stereo
    # =============================================================================
    
    if not args.skip_rendering:
        shutil.rmtree(strings['output_dir_root'])
        stereo.run(start=0, visualize=False)

    # =============================================================================
    #  Perform automatic masking
    # =============================================================================
    
    if not args.skip_masking:
        if args.masker_automask:
            GD_model, predictor, inference_state, images_dir = init_predictor(base_dir, renderer, args, device=device) 

            subdir_bins = []
            for img in os.listdir(images_dir):
                if os.path.splitext(img)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]:
                    src_im = os.path.join(images_dir, img)
                    image = plt.imread(src_im)
                    height, width = image.shape[:2]
                    subdir_bin = str(height) + 'x' + str(width)
                    if subdir_bin not in subdir_bins:
                        subdir_bins.append(subdir_bin)
                    os.makedirs(os.path.join(images_dir, subdir_bin), exist_ok=True)
                    shutil.copy(src_im, os.path.join(images_dir, subdir_bin, img))
            
            garment_dict = {
                "vertical": {"subdir": next(d for d in subdir_bins if int(d.split('x')[0]) > int(d.split('x')[1])), "ref": v_ref},
                "horizontal": {"subdir": next(d for d in subdir_bins if int(d.split('x')[0]) < int(d.split('x')[1])), "ref": h_ref},
            }

            masker = run_full_masker(garment_dict["horizontal"], GD_model, renderer, stereo, args, images_dir)
            
            # once masks are generated, we need to store the first frame outputs 
            # to handle overwrite issue by moving the contents of 000 folder to 000_save
            copy_src = renderer.render_folder_name(0)
            copy_dest = copy_src + '_save'
            os.system(f"mkdir {copy_dest}")
            os.system(f"mv {copy_src + '/*'} {copy_dest}")

            masker = run_full_masker(garment_dict["vertical"], GD_model, renderer, stereo, args, images_dir)

            # now, repopulate the original 000 folder with the stored data
            os.system(f"rm -rf {copy_src + '/*'}")
            os.system(f"mv {copy_dest + '/*'} {copy_src}")
            os.system(f"rm -rf {copy_dest}")

            # if the garment is an upper, then the 000 folder does not have any mask
            # since it is skipped in masking. so, a manual copy paste from one of the 
            # later empty frames is done
            if args.garment_type.lower() == 'upper':
                src_path = renderer.render_folder_name(8)
                dest_path = renderer.render_folder_name(0)
                shutil.copy(os.path.join(src_path, 'left_mask.png'), os.path.join(dest_path, 'left_mask.png'))
                shutil.copy(os.path.join(src_path, 'left_mask.npy'), os.path.join(dest_path, 'left_mask.npy'))

        else:
            print("Automask must be enabled for masking in script mode by setting --masker_automask. Skipping.")
            
                
    # =============================================================================
    #  Initialize TSDF
    # =============================================================================
    
    tsdf = TSDF(renderer, stereo, args, strings['TSDF'])

    if not args.skip_TSDF:
        # ================================================================================
        #  Run TSDF. the TSDF class will have an attribute "mesh" with the resulting mesh
        # ================================================================================
        tsdf.run(visualize = False)

        # =============================================================================
        #  Save the original mesh before cleaning
        # =============================================================================
        tsdf.save_mesh()

        # =============================================================================
        #  Cleans the mesh using clustering and saves .ply, then smooth surface remeshing
        # =============================================================================
        
        # original mesh is still available under tsdf.mesh (the cleaned is tsdf.clean_mesh)
        tsdf.clean_mesh()

        # saving the cleaned mesh in desired output directory with required name
        o3d.io.write_triangle_mesh(os.path.join(_stage1, 'point_cloud.ply'), tsdf.clean_mesh)
        print(f"Cleaned mesh saved at: {os.path.join(_stage1, 'point_cloud.ply')}")
        
        # smooth surface remeshing
        mesh = pv.read(os.path.join(renderer.output_dir_root, f'{tsdf.out_name}_cleaned_mesh.ply'))
        clus = pyacvd.Clustering(mesh)
        clus.cluster(8000)
        remesh = clus.create_mesh()

        # saving the smooth mesh in desired output directory with required name
        remesh.save(os.path.join(_stage1, 'template.obj'))
        print(f"Smooth remesh saved at: {os.path.join(_stage1, 'template.obj')}")
    
    os.makedirs(os.path.join(_stage1, 'sparse'), exist_ok=True)
    shutil.copy(f'{colmap_dir}/sparse/0/points3D.bin', f'{_stage1}/sparse/points3D.bin')
    print(f"Points3D.bin copied to: {os.path.join(_stage1, 'sparse', 'points3D.bin')}\n")


# =============================================================================
#  Main driver code with arguments
# =============================================================================

if __name__ == "__main__":
    parser = ArgParser('custom')
    parser = actorshq_params(parser.parser)

    for action in parser._actions:
        if action.dest == "masker_prompt":
            action.required = True

    args = parser.parse_args()
    args.colmap_name = f"{args.subject}_{args.sequence}"
    args.TSDF_use_mask = True
    run_single(args)