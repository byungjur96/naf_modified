import os
import os.path as osp
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import imageio.v2 as iio
import numpy as np
import argparse
import wandb
from PIL import Image
import pickle
import yaml
import json
from shutil import copyfile
from tqdm import tqdm
from scipy.ndimage import zoom
import SimpleITK as sitk

from src.config.configloading import load_config
from src.render import render, run_network
from src.trainer import Trainer
from src.loss import calc_mse_loss, calc_regularizer
from src.utils import get_psnr, get_mse, get_psnr_3d, get_ssim_3d, cast_to_image

from src.dataset import ConeGeometry
from src.network import get_network
from src.encoder import get_encoder

def downsample_volume(volume, scale_factor=2):
    if isinstance(scale_factor, (int, float)):
        volume_dim = volume.ndim
        scale_factor = [scale_factor] * volume_dim
    
    original_spacing = volume.GetSpacing()
    original_size = volume.GetSize()

    new_spacing = np.multiply(scale_factor, original_spacing)

    new_size = [
        int(round(original_size[i] * (original_spacing[i] / new_spacing[i])))
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputOrigin(volume.GetOrigin())
    resampler.SetOutputDirection(volume.GetDirection())
    resampler.SetDefaultPixelValue(volume.GetPixelIDValue())

    downsampled_volume = resampler.Execute(volume)

    return downsampled_volume

def averagepooling(volume, scale):
    return volume.reshape(volume.shape[0] // scale, scale, volume.shape[1] // scale, scale, volume.shape[2] // scale, scale).mean(axis=(1, 3, 5))

def save_tensor_as_image(tensor, filename):
    # Remove the extra dimensions
    tensor = tensor.squeeze()  # Now the shape is [512, 512]

    # Normalize the tensor to the range [0, 255]
    tensor = tensor - tensor.min()
    tensor = tensor / tensor.max()
    tensor = (tensor * 255).byte()  # Convert to byte type (0-255)

    # Convert to a NumPy array
    np_array = tensor.cpu().numpy()

    # Create a PIL Image object
    image = Image.fromarray(np_array)

    # Save the image
    image.save(filename)

normalize_array = lambda image: (image.astype(np.float32) - np.min(image)) / (np.max(image) - np.min(image))

PICKLE_CONF_PATH = "/workspace/naf_modified/config/pickle"
VOL_PATH = "/workspace/CVPR25/data/CVPR"

class SlideDataset(Dataset):
    def __init__(self, datapath, geopath, upsample, device):
        super().__init__()
        
        # Load TIGRE GEO Configuration
        with open(geopath, "rb") as handle:
            data = yaml.safe_load(handle)
        print(f"Geometry config from {geopath} Loaded")
        
        # Load GT Volume
        vol_file = sitk.ReadImage(datapath)
        vol_file = self.normalize_volume(vol_file)
        vol = np.transpose(sitk.GetArrayFromImage(vol_file), (1,2,0))
        print(f"Volume from {datapath} Loaded")
        
        # Generate LR Volume
        scale = np.array(data["nVoxel"]) / np.array(data['lrVol'])
        
        if np.all(scale == 1.0):
            print(f"No downsample was done! {data['nVoxel']} is equal to {data['lrVol']}")
            upsampled_data = vol 
        else:
            lr_vol = sitk.GetArrayFromImage(downsample_volume(vol_file, scale_factor=scale))
            lr_vol = np.transpose(lr_vol, (1,2,0))
            print(f"LR is generated into size {lr_vol.shape}")        
            # Upsample LR volume to fit with shape of GT Volume
            if upsample == 'trilinear':
                upsampled_data = zoom(lr_vol, scale, order=1)
            elif upsample == 'cubic':
                upsampled_data = zoom(lr_vol, scale, order=3, prefilter=False)
            elif upsample == 'prefilter':
                upsampled_data = zoom(lr_vol, scale, order=3)
            else:
                print('No upsample method was selected')
                upsampled_data = zoom(lr_vol, scale, order=0)
        
        
        self.gt = torch.tensor(vol, device=device)
        self.volume = torch.tensor(upsampled_data, device=device)
        self.vol_shape = self.volume.shape
        
        self.geo = ConeGeometry(data)
        self.voxels = torch.tensor(self.get_voxels(self.geo), dtype=torch.float32, device=device)
        
    def __len__(self):
        return self.vol_shape[0] + self.vol_shape[1] + self.vol_shape[2]
    
    def __getitem__(self, idx):
        orientation = idx // self.vol_shape[0]
        slide_idx = idx % self.vol_shape[0]
        
        if orientation == 0:
            slide = self.volume[slide_idx,:,:]
        elif orientation == 1:
            slide = self.volume[:,slide_idx,:]
        else:
            slide = self.volume[:,:,slide_idx]
        
        return {
            'slide' : slide,
            'orientation' : orientation,
            'index' : slide_idx
        }
         
    def get_voxels(self, geo: ConeGeometry):
        """
        Get the voxels.
        """
        n1, n2, n3 = np.array(self.vol_shape)
        s1, s2, s3 = geo.sVoxel / 2 - geo.dVoxel / 2

        xyz = np.meshgrid(np.linspace(-s1, s1, n1),
                        np.linspace(-s2, s2, n2),
                        np.linspace(-s3, s3, n3), indexing="ij")
        voxel = np.asarray(xyz).transpose([1, 2, 3, 0])
        return voxel
    
    def normalize_volume(self, volume):
        stats = sitk.StatisticsImageFilter()
        stats.Execute(volume)
        min_value = stats.GetMinimum()
        max_value = stats.GetMaximum()

        normalized_volume = sitk.Cast((volume - min_value) / (max_value - min_value), sitk.sitkFloat32)

        return normalized_volume


def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expName", default="dummy",
                        help="Name of experiment name to be saved.")
    parser.add_argument("--config", default="./config/pretrain/default.yaml",
                        help="configs file path")
    parser.add_argument("--data", default="mela_0257.nii.gz",
                        help="Name of experiment name to be saved.")
    return parser

parser = config_parser()
args = parser.parse_args()

cfg = load_config(args.config)
cfg['exp']['expname'] = args.expName
cfg['exp']['data'] = args.data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BasicTrainer():
    def args2string(self, hp):
        """
        Transfer args to string.
        """
        json_hp = json.dumps(hp, indent=2)
        return "".join("\t" + line for line in json_hp.splitlines(True))
    
    def dict2namespace(self, config):
        namespace = argparse.Namespace()
        for key, value in config.items():
            if isinstance(value, dict):
                new_value = self.dict2namespace(value)
            else:
                new_value = value
            setattr(namespace, key, new_value)
        return namespace
    
    def parallel2single(self, checkpoint):
        new_state_dict = {}
        for k, v in checkpoint.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v  # Remove the "module." prefix
            else:
                new_state_dict[k] = v
        return new_state_dict
    
    def __init__(self, cfg):
        """
        Basic network trainer.
        """
        # Args
        self.global_step = 0
        self.conf = cfg
        
        self.n_fine = cfg["render"]["n_fine"]
        self.epochs = cfg["train"]["epoch"]
        self.batch_size = cfg["train"]["n_batch"]
        self.netchunk = cfg["render"]["netchunk"]
        
        self.i_eval = cfg["log"]["i_eval"]
        self.i_save = cfg["log"]["i_save"]
  
        # Log direcotry
        self.expdir = osp.join(cfg["exp"]["expdir"], cfg["exp"]["expname"])
        self.ckptdir = osp.join(self.expdir, "ckpt.tar")
        self.ckptdir_backup = osp.join(self.expdir, "ckpt_backup.tar")
        self.evaldir = osp.join(self.expdir, "eval")
        os.makedirs(self.evaldir, exist_ok=True)
        with open(osp.join(self.expdir, 'config.yaml'), 'w') as yaml_file:
            yaml.dump(self.conf, yaml_file, default_flow_style=False)

        # Dataset
        train_dset = SlideDataset(
            datapath=osp.join(VOL_PATH, cfg["exp"]["data"]),
            geopath=osp.join(PICKLE_CONF_PATH, cfg["exp"]["conf"]),
            upsample=cfg['exp']['upsample'],
            device=device)
        
        self.train_dloader = DataLoader(train_dset, batch_size=self.batch_size)
        self.vol = train_dset.volume  # Upsampled volume
        self.gt = train_dset.gt
        self.voxels = train_dset.voxels  # Corresponding coord
        
        # Network
        network = get_network(cfg["network"]["net_type"])
        cfg["network"].pop("net_type", None)
        encoder = get_encoder(**cfg["encoder"])
        self.net = network(encoder, **cfg["network"]).to(device)
        grad_vars = list(self.net.parameters())
        self.net_fine = None
        if self.n_fine > 0:
            self.net_fine = network(encoder, **cfg["network"]).to(device)
            grad_vars += list(self.net_fine.parameters())
            
        # Optimizer
        self.optimizer = torch.optim.Adam(params=grad_vars, lr=cfg["train"]["lrate"], betas=(0.9, 0.999))
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=self.optimizer, step_size=cfg["train"]["lrate_step"], gamma=cfg["train"]["lrate_gamma"])

        # Load checkpoints
        self.epoch_start = 0
        if cfg["train"]["resume"] and osp.exists(self.ckptdir):
            print(f"Load checkpoints from {self.ckptdir}.")
            ckpt = torch.load(self.ckptdir)
            self.epoch_start = ckpt["epoch"] + 1
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.global_step = self.epoch_start * len(self.train_dloader)
            self.net.load_state_dict(ckpt["network"])
            if self.n_fine > 0:
                self.net_fine.load_state_dict(ckpt["network_fine"])

        print(f"[Start] exp: {cfg['exp']['expname']}, net: Basic network")

    def start(self):
        """
        Main loop.
        """
        def fmt_loss_str(losses):
            return "".join(", " + k + ": " + f"{losses[k].item():.3g}" for k in losses)
        iter_per_epoch = len(self.train_dloader)
        pbar = tqdm(total= iter_per_epoch * self.epochs, leave=True)
        if self.epoch_start > 0:
            pbar.update(self.epoch_start*iter_per_epoch)

        for idx_epoch in range(self.epoch_start, self.epochs+1):
            # Evaluate
            if (idx_epoch % self.i_eval == 0 or idx_epoch == self.epochs) and self.i_eval > 0 and idx_epoch > 0:
                self.net.eval()
                with torch.no_grad():
                    loss_test = self.eval_step(global_step=self.global_step, idx_epoch=idx_epoch)
                self.net.train()
                tqdm.write(f"[EVAL] epoch: {idx_epoch}/{self.epochs}{fmt_loss_str(loss_test)}")
            
            # Train
            for data in self.train_dloader:
                self.global_step += 1
                # Train
                self.net.train()
                loss_train = self.train_step(data, global_step=self.global_step, idx_epoch=idx_epoch)
                pbar.set_description(f"epoch={idx_epoch}/{self.epochs}, loss={loss_train['loss'].item():.3g}, lr={self.optimizer.param_groups[0]['lr']:.3g}")
                pbar.update(1)
            
            # Save
            if (idx_epoch % self.i_save == 0 or idx_epoch == self.epochs) and self.i_save > 0 and idx_epoch > 0:
                if osp.exists(self.ckptdir):
                    copyfile(self.ckptdir, self.ckptdir_backup)
                tqdm.write(f"[SAVE] epoch: {idx_epoch}/{self.epochs}, path: {self.ckptdir}")
                torch.save(
                    {
                        "epoch": idx_epoch,
                        "network": self.net.state_dict(),
                        "network_fine": self.net_fine.state_dict() if self.n_fine > 0 else None,
                        "optimizer": self.optimizer.state_dict(),
                    },
                    self.ckptdir,
                )

            # Update lrate
            self.lr_scheduler.step()

        tqdm.write(f"Training complete! See logs in {self.expdir}")

    def train_step(self, data, global_step, idx_epoch):
        """
        Training step
        """
        self.optimizer.zero_grad()
        loss_dict = self.compute_loss(data, global_step, idx_epoch)
        loss = loss_dict["loss"]
        loss.backward()
        self.optimizer.step()
        return loss_dict
    
    
    def get_coord(self, orientation, index):
        if orientation == 0:
            coord = self.voxels[index,:,:]
        elif orientation == 1:
            coord = self.voxels[:,index,:]
        else:
            coord = self.voxels[:,:,index]
            
        return coord
        
    def compute_loss(self, data, global_step, idx_epoch):
        loss = {"loss": 0.}
        gt = data['slide']
        pred = torch.stack([
                run_network(
                    self.get_coord(data['orientation'][i].item(), data['index'][i].item()),
                    self.net,
                    self.netchunk
                    ).squeeze()
             for i in range(len(data['orientation']))
        ])
        # print(gt.shape, pred.shape)
        loss['loss'] += torch.mean((gt - pred)**2)
        
        return loss
        

    def eval_step(self, global_step, idx_epoch):
        """
        Evaluation step
        """
        # Evaluate density
        image = self.gt  # 512, 512, 512
        image_pred = run_network(self.voxels, self.net_fine if self.net_fine is not None else self.net, self.netchunk)
        image_pred = image_pred.squeeze()
        loss = {
            "psnr_3d": get_psnr_3d(image_pred, image),
            "ssim_3d": get_ssim_3d(image_pred, image),
        }

        # Logging
        show_slice = 5
        show_step = image.shape[-1]//show_slice
        show_image = image[...,::show_step]
        show_image_pred = image_pred[...,::show_step]
        show = []
        for i_show in range(show_slice):
            show.append(torch.concat([show_image[..., i_show], show_image_pred[..., i_show]], dim=0))
        show_density = torch.concat(show, dim=1)
            
        # Save
        eval_save_dir = osp.join(self.evaldir, f"epoch_{idx_epoch:05d}")
        os.makedirs(eval_save_dir, exist_ok=True)
        np.save(osp.join(eval_save_dir, "image_pred.npy"), image_pred.cpu().detach().numpy())
        iio.imwrite(osp.join(eval_save_dir, "slice_show_row1_gt_row2_pred.png"), (cast_to_image(show_density)*255).astype(np.uint8))
        
        with open(osp.join(eval_save_dir, "stats.txt"), "w") as f: 
            for key, value in loss.items(): 
                f.write("%s: %f\n" % (key, value.item()))

        return loss

trainer = BasicTrainer(cfg)
trainer.start()
        
