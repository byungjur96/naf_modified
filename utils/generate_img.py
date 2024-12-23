import numpy as np
import os
import pickle
import scipy
from PIL import Image
import argparse
import cv2

normalize_array = lambda image: (image.astype(np.float32) - np.min(image)) / (np.max(image) - np.min(image))
PICKLE_DIR = '/workspace/naf_modified/data'
DDNM_DIR = '/workspace/Diffusion/DDNM/exp/datasets'

parser = argparse.ArgumentParser()
parser.add_argument("--pickle_file", help="Name of pickle file")
parser.add_argument("--img_folder", help="Target forlder name for images to save")
parser.add_argument("--normalize", action="store_true", help="If true, normalize projection image in range [0, 1]")
args = parser.parse_args()

# Open Pickle File
with open(os.path.join(PICKLE_DIR, f"{args.pickle_file}.pickle"), "rb") as handle:
    data = pickle.load(handle)

# If the Directory does not exists, make folder
save_dir = os.path.join(DDNM_DIR, f"{args.img_folder}")
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

for i, img in enumerate(data['train']['projections']):
    if args.normalize:
        img = normalize_array(np.array(img))
    else:
        img = np.array(img)
    img_name = f"proj_{i+1}.png"
    # img = Image.fromarray((img * 255).astype(np.uint8))
    img = (img * 255).astype(np.uint8)

    zoom_factors = (512. / img.shape[0], 512. / img.shape[1])
    
    upsampled_img = cv2.resize(img, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)

    img_to_save = Image.fromarray(upsampled_img)
    img_to_save.save(os.path.join(save_dir, img_name))

print(f"Generate image of {args.pickle_file} with{'' if args.normalize else 'out'} at {save_dir} Done!")