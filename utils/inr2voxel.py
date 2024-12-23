import torch
import pickle
import argparse
import os.path as osp
import numpy as np

from src.config.configloading import load_config

from src.network import get_network
from src.encoder import get_encoder
from src.render import run_network

def get_voxels(vol_shape):
    """
    Get the voxels.
    """
    n1, n2, n3 = np.array(vol_shape)
    
    dVoxel = np.array([.375, .375, .375])/1000
    sVoxel = np.array([512, 512, 512]) * dVoxel
    
    s1, s2, s3 = sVoxel / 2 - dVoxel / 2

    xyz = np.meshgrid(np.linspace(-s1, s1, n1),
                    np.linspace(-s2, s2, n2),
                    np.linspace(-s3, s3, n3), indexing="ij")
    voxel = np.asarray(xyz).transpose([1, 2, 3, 0])
    return voxel

parser = argparse.ArgumentParser()
parser.add_argument("--idx", default="training", help="Sample Index")
args = parser.parse_args()
cfg = load_config("./config/AAAI_vol_SR.yaml")

network = get_network(cfg["network"]["net_type"])
cfg["network"].pop("net_type", None)
encoder = get_encoder(**cfg["encoder"])
net = network(encoder, **cfg["network"]).to('cuda')

# exp_path = "/workspace/naf_modified/logs/MELA0048_with_LR_pretrain" 
exp_path = '/workspace/naf_modified/logs/Pretrain_MELA0048'
# exp_path = '/workspace/naf_modified/logs/Pretrain_MELA0048_128'

ckpt = torch.load(osp.join(exp_path, "ckpt.tar"))
net.load_state_dict(ckpt["network"])

with open(f'./data/50_MELA{args.idx}_gt512.pickle', "rb") as handle:
    data = pickle.load(handle)
   
volume = get_voxels(np.array([128, 128, 128]))
voxels = torch.tensor(volume, dtype=torch.float32, device='cuda')
print(voxels.shape)
print(voxels[0][0][0])

with torch.no_grad():
    image_pred = run_network(voxels, net, 409600)
print(image_pred.shape)
np.save('./results/Pretrain_128.npy', image_pred.detach().cpu().numpy())