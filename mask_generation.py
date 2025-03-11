import os
import yaml
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.ops import box_convert
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image
import gs2mesh.third_party.GroundingDINO.groundingdino.util.inference as GD

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgs_dir", "-in", required=True, type=str, help="The absolute path to the directory where the CamXXX folders are stored (i.e. path/to/folder that contains Cam001, Cam002, ...).")
    parser.add_argument("--output_root", "-out", required=True, type=str, help="The absolute path to the root of the mask directory where the CamXXX folders are to be created and populated with masks (i.e. path/to/folder/masks that will get populated with Cam001, Cam002, ... folders containing mask images).") 
    parser.add_argument("--prompt", "-p", required=True, type=str, help="Prompt for GroundingDINO to locate object (garment) of interest.")
    return parser


def generate_masks(args, config):
    GD_dir = os.path.join(os.getcwd(), "gs2mesh", 'third_party', 'GroundingDINO')
    GD_model = GD.load_model(os.path.join(GD_dir, 'groundingdino', 'config', 'GroundingDINO_SwinT_OGC.py'), os.path.join(GD_dir, 'weights', 'groundingdino_swint_ogc.pth'))
    predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large", device=device)

    # iterate over all folders that start with 'Cam'
    for cam in os.listdir(args.imgs_dir):
        if os.path.isdir(os.path.join(args.imgs_dir, cam)) and cam in config['portrait_cams']:
            
            # create the destination folder
            dest_path = os.path.join(args.output_root, cam)
            os.makedirs(dest_path, exist_ok=True)
            print(f'\n\nMasking {cam} Images.\nStoring masks in {dest_path}.\n')

            # iterate over every JPG image in the folder
            for img in tqdm(os.listdir(os.path.join(args.imgs_dir, cam))):
                if os.path.splitext(img)[-1] not in [".jpg", ".jpeg", ".JPG", ".JPEG"]:
                    continue

                # get bounding box from groundingDINO
                img_path = os.path.join(args.imgs_dir, cam, img)
                image_source, gd_image = GD.load_image(img_path)
                boxes, logits, phrases = GD.predict(
                    model=GD_model,
                    image=gd_image,
                    caption=args.prompt,
                    box_threshold=config['box_threshold'],
                    text_threshold=config['text_threshold']
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

                plt.imsave(os.path.join(dest_path, img), mask)
            
    plt.close('all')      


# =============================================================================
#  Main driver code with arguments
# =============================================================================

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    # load yaml file
    with open('config.yaml', 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    
    generate_masks(args, config)

    