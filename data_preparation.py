import os
import yaml
import torch
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.ops import box_convert
from sam2.sam2_image_predictor import SAM2ImagePredictor
import gs2mesh.third_party.GroundingDINO.groundingdino.util.inference as GD

from defaults import DEFAULTS
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", "-s", required=True, type=str, help="Subject folder name that contains the sequence folders (e.g. Actor06).")
    parser.add_argument("--sequence", "-q", required=True, type=str, help="Sequence folder name (e.g. Sequence1).")
    parser.add_argument("--resolution", "-r", default='4x', type=str, help="Resolution folder of ActorsHQ images (e.g. 1x).")
    parser.add_argument("--masker_prompt", "-p", required=True, type=str, help="Prompt for GroundingDINO to locate object (garment) of interest.")
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
    GD_dir = os.path.join(os.getcwd(), "gs2mesh", 'third_party', 'GroundingDINO')
    GD_model = GD.load_model(os.path.join(GD_dir, 'groundingdino', 'config', 'GroundingDINO_SwinT_OGC.py'), os.path.join(GD_dir, 'weights', 'groundingdino_swint_ogc.pth'))
    predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large", device=device)

    in_root = in_root / 'rgbs'

    # iterate over all folders that start with 'Cam'
    for f in in_root.iterdir():
        cam = f.name
        
        if f.is_dir() and cam in DEFAULTS['portrait_cams']:
            
            # create the destination folder
            dest_path = out_root / cam / 'garment_masks'
            dest_path.mkdir(parents=True, exist_ok=True)

            print(f'\n\nMasking {cam} Images.\nStoring masks in {dest_path}.\n')

            # iterate over every JPG image in the folder
            for i in tqdm(f.iterdir()):

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


def symlink_loop(ddir, src_name, out_root):
    for f in ddir.iterdir():
        cam = f.name
        if f.is_dir() and cam.startswith('Cam'):
            
            src = out_root / cam / src_name
            print('src', src)
            assert False
            src.mkdir(parents=True, exist_ok=True)

            if src.exists() or src.is_symlink():
                src.unlink()  # Remove existing file or symlink

            src.symlink_to(f, target_is_directory=True)
            print(f'Created symlink: {src} ----> {f}.')
            

def generate_symlinks(in_root, out_root):
    symlink_loop(in_root / 'rgbs', 'rgb_images', out_root)
    symlink_loop(in_root / 'masks', 'foreground_masks', out_root)


def main():
    parser = init_parser()
    args = parser.parse_args()

    in_root = get_input_folder(args)
    out_root = setup_output_folder(args)

    generate_symlinks(in_root, out_root)
    generate_masks(args, in_root / 'rgbs', out_root)


if __name__ == "__main__":
    main()