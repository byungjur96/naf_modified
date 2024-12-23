import argparse
import pickle
from PIL import Image
import os
import numpy as np

normalize_array = lambda image: (image.astype(np.float32) - np.min(image)) / (np.max(image) - np.min(image))
PICKLE_DIR = "/workspace/naf_modified/data"
DDNM_DIR = "/workspace/Diffusion/DDNM/exp/image_samples"

parser = argparse.ArgumentParser()
parser.add_argument("--pickle_file", help="Name of pickle file")
parser.add_argument("--img_folder", help="Target forlder name for images to save")
args = parser.parse_args()


with open(os.path.join(PICKLE_DIR, f"{args.pickle_file}.pickle"), "rb") as handle:
    pickle_data = pickle.load(handle)

train_lst = [file for file in os.listdir(os.path.join(DDNM_DIR, args.img_folder)) if file.endswith('.npy')]


train_img = []


for i, file in enumerate(train_lst):
    high = np.max(pickle_data['train']['projections'][i])
    low = np.min(pickle_data['train']['projections'][i])
    # img = Image.open(os.path.join(DDNM_DIR, args.img_folder, file)).convert('L')
    img = np.load(os.path.join(DDNM_DIR, args.img_folder, file)).squeeze()
    # train_img.append((high - low) * (np.array(img).astype(np.float32) / 255.) + low)
    train_img.append((high - low) * normalize_array(np.array(img)) + low)
    
    
pickle_data['train']['projections'] = np.array(train_img)

with open(os.path.join(PICKLE_DIR, f"{args.img_folder}.pickle"), "wb") as handle:
    pickle.dump(pickle_data, handle, pickle.HIGHEST_PROTOCOL)
print(f'{os.path.join(PICKLE_DIR, f"{args.img_folder}.pickle")} Generated!')
    