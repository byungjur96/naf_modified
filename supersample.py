import torch
import numpy as np
import yaml
from src.config.configloading import load_config
from src.render import render, run_network

from src.dataset import ConeGeometry
from src.network import get_network
from src.encoder import get_encoder

import matplotlib.pyplot as plt

def get_voxels(geo: ConeGeometry, resolution=512):
        """
        Get the voxels.
        """
        n1, n2, n3 = np.array([resolution, resolution, resolution])
        s1, s2, s3 = geo.sVoxel / 2 - geo.dVoxel / 2

        xyz = np.meshgrid(np.linspace(-s1, s1, n1),
                        np.linspace(-s2, s2, n2),
                        np.linspace(-s3, s3, n3), indexing="ij")
        voxel = np.asarray(xyz).transpose([1, 2, 3, 0])
        return voxel
    
cfg = load_config('./config/thesis_default.yaml')
network = get_network(cfg["network"]["net_type"])
encoder = get_encoder(**cfg["encoder"])

device = 'cuda'

cfg["network"].pop("net_type", None)
net = network(encoder, **cfg["network"]).to(device)

with open('/workspace/naf_modified/config/pickle/gt128.yml', "rb") as handle:
    data128 = yaml.safe_load(handle)

with open('/workspace/naf_modified/config/pickle/gt512.yml', "rb") as handle:
    data512 = yaml.safe_load(handle)
            
geo128 = ConeGeometry(data128)
geo512 = ConeGeometry(data512)

voxels128 = torch.tensor(get_voxels(geo128, 128), dtype=torch.float32, device=device)
voxels512 = torch.tensor(get_voxels(geo512, 512), dtype=torch.float32, device=device)

ckpt = torch.load('/workspace/naf_modified/logs/Pretrain/MELA0050_LR/ckpt.tar', map_location=device)
net.load_state_dict(ckpt["network"])