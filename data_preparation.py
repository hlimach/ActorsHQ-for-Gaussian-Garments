import os
import yaml
import smplx
import torch
import gdown
import pickle
import trimesh
import zipfile
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision.ops import box_convert

from sam2.sam2_image_predictor import SAM2ImagePredictor
import gs2mesh.third_party.GroundingDINO.groundingdino.util.inference as GD

from defaults import DEFAULTS
device = 'cuda' if torch.cuda.is_available() else 'cpu'
gfile_id = '1DVk3k-eNbVqVCkLhGJhD_e9ILLCwhspR'


def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", "-s", required=True, type=str, help="Subject folder name that contains the sequence folders (e.g. Actor06).")
    parser.add_argument("--sequence", "-q", required=True, type=str, help="Sequence folder name (e.g. Sequence1).")
    parser.add_argument("--resolution", "-r", default='4x', type=str, help="Resolution folder of ActorsHQ images (e.g. 1x).")
    parser.add_argument("--masker_prompt", "-p", required=True, type=str, help="Prompt for GroundingDINO to locate object (garment) of interest.")
    parser.add_argument("--gender", "-g", required=True, type=str, help="Gender of the SMPLX model, must be one of [male, female], corresponding to gender of subject.")
    return parser


def get_input_folder(args):
    """
    Returns the input folder path containing the images.
    """
    return DEFAULTS['AHQ_data_root'] / args.subject / args.sequence / args.resolution


def setup_output_folder(args):
    """
    Creates output directory for storing garment masks.
    """
    _root = DEFAULTS['data_root'] / args.subject / args.sequence
    _root.mkdir(parents=True, exist_ok=True)
    return _root


def generate_masks(args, in_root, out_root):
    """
    Generates garment masks for all the images for the subject-seq, in the 
    same directory structure as the input folder. 
    """

    GD_dir = os.path.join(os.getcwd(), "gs2mesh", 'third_party', 'GroundingDINO')
    GD_model = GD.load_model(os.path.join(GD_dir, 'groundingdino', 'config', 'GroundingDINO_SwinT_OGC.py'), os.path.join(GD_dir, 'weights', 'groundingdino_swint_ogc.pth'))
    predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large", device=device)

    print("\n\nGenerating Garment Masks...\n")

    # iterate over all folders that start with 'Cam'
    for f in in_root.iterdir():
        cam = f.name
        
        if f.is_dir() and cam in DEFAULTS['portrait_cams']:
            
            # create the destination folder
            dest_path = out_root / cam / 'garment_masks'
            dest_path.mkdir(parents=True, exist_ok=True)

            print(f'\n\nMasking {cam} Images.\nStoring masks in {dest_path}')

            # iterate over every JPG image in the folder
            for i in tqdm(sorted(list(f.iterdir()))):

                    img = i.name
                    if img.split('.')[-1] not in ["jpg", "jpeg", "JPG", "JPEG"]:
                        continue

                    # get bounding box from groundingDINO
                    img_path = f / img
                    image_source, gd_image = GD.load_image(img_path)
                    boxes, logits, phrases = GD.predict(
                        model=GD_model,
                        image=gd_image,
                        caption=args.masker_prompt,
                        box_threshold=DEFAULTS['box_threshold'],
                        text_threshold=DEFAULTS['text_threshold']
                    )
                    
                    h, w, _ = image_source.shape
                    if len(boxes) == 0:
                        # Create an empty mask
                        mask = np.zeros((h, w), dtype=np.uint8)
                    else:
                        # convert the bounding box to required format
                        boxes = boxes * torch.Tensor([w, h, w, h])
                        boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()[0]
                    
                        # feed the bounding box to the sam2 image predictor
                        image = Image.open(img_path)
                        predictor.set_image(np.array(image.convert("RGB")))
                        masks, scores, logits = predictor.predict(box=boxes)
                        mask = masks[np.argmax(scores), :, :]
                        predictor.reset_predictor()

                    plt.imsave(dest_path / img, mask)
            
    plt.close('all') 
    print("\nAll Garment Masks Generated Successfully!")     


def symlink_loop(ddir, src_name, out_root):
    for f in ddir.iterdir():
        cam = f.name
        if f.is_dir() and cam.startswith('Cam'):
            
            src = out_root / cam / src_name
            src.parent.mkdir(parents=True, exist_ok=True)

            if src.exists() or src.is_symlink():
                src.unlink()  # Remove existing file or symlink

            if cam in DEFAULTS['portrait_cams']:
                src.symlink_to(f, target_is_directory=True)
                print(f'Created symlink: {src} ----> {f}.')
            

def generate_symlinks(in_root, out_root):
    """
    Generates symlinks for the RGB images and foreground masks as required by the Gaussian Garments pipeline.
    """
    print("\n\nGenerating Symlinks...\n")
    symlink_loop(in_root / 'rgbs', 'rgb_images', out_root)
    symlink_loop(in_root / 'masks', 'foreground_masks', out_root)
    print("\nAll Symlinks Generated Successfully!")


def unpack_smplx(args, out_root):
    # Download the SMPLX model for ActorHQ dataset
    print("\n\nDownloading ActorsHQ SMPLX model...")
    smplx_zip = str(out_root / 'ActorsHQ_smplx.zip')
    gdown.download(f"https://drive.google.com/uc?export=download&id={gfile_id}", smplx_zip, fuzzy=True, quiet=False)

    # Unzip the SMPLX model
    zip_extract = out_root / 'ActorsHQ_smplx'
    zip_extract.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(smplx_zip, 'r') as zip_ref:
        zip_ref.extractall(zip_extract)
    print(f'Successfully extracted SMPLX model to {zip_extract}')

    if args.sequence != 'Sequence1':
        # raise warning
        print("\033[91mOnly Sequence1 is supported for SMPLX model extraction.\033[0m")
        return
    
    # Create output smplx folder
    smplx_dir = out_root / 'smplx'
    smplx_dir.mkdir(parents=True, exist_ok=True)

    # Load the .npz file of subject
    actor_smplx = zip_extract / args.subject / args.sequence / 'smpl_params.npz'
    data = np.load(actor_smplx)

    print(f"\nUnpacking {args.subject} SMPLX model .pkl and .ply files...")

    model = smplx.create(
        model_path=DEFAULTS['aux_root'],  # Path to SMPL-X models
        model_type="smplx", gender=args.gender, use_pca=False, batch_size=1
    ).to(device)

    for frame_id in tqdm(range(data['transl'].shape[0])):
        frame_smplx = {}
        for key in data.files:
            if key != 'betas':
                frame_smplx[key] = data[key][frame_id]
        
        # fill missing keys
        frame_smplx['betas'] = data['betas'][0]
        frame_smplx['leye_pose'] = np.array([0., 0., 0.], dtype=np.float32)
        frame_smplx['reye_pose'] = np.array([0., 0., 0.], dtype=np.float32)
        
        # Save the SMPLX model .pkl file
        pkl_path = smplx_dir / f"{frame_id:06d}.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(frame_smplx, f)

        # convert numpy arrays to torch tensors and pass to SMPL-X model
        params_torch = {key: torch.tensor(value, dtype=torch.float32, device=device).unsqueeze(0) for key, value in frame_smplx.items()}
        output = model(**params_torch)

        # save the .ply file
        vertices = output.vertices.detach().cpu().numpy().squeeze()
        faces = model.faces
        trimesh.Trimesh(vertices=vertices, faces=faces).export(smplx_dir / f"{frame_id:06d}.ply")
    
    print(f"Unpacked successfully! Files stored in {smplx_dir}\n")


def main():
    parser = init_parser()
    args = parser.parse_args()

    in_root = get_input_folder(args)
    out_root = setup_output_folder(args)

    unpack_smplx(args, out_root)
    generate_symlinks(in_root, out_root)
    generate_masks(args, in_root / 'rgbs', out_root)


if __name__ == "__main__":
    main()