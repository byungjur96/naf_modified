import os
import os.path as osp
import torch
import torch.nn.functional as F
import imageio.v2 as iio
import numpy as np
import argparse
import wandb
from PIL import Image

from src.config.configloading import load_config
from src.render import render, run_network
from src.trainer import Trainer
from src.loss import calc_mse_loss, calc_regularizer, calc_adaptive_mse_loss, calc_tv_loss, calc_ssim_loss, calc_2d_tv_loss, calc_l1_loss
from src.utils import get_psnr, get_mse, get_psnr_3d, get_ssim_3d, cast_to_image

normalize_array = lambda image: (image.astype(np.float32) - np.min(image)) / (np.max(image) - np.min(image))

def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config/abdomen_50.yaml",
                        help="configs file path")
    parser.add_argument("--expName", default="training",
                        help="Name of experiment name to be saved.")
    parser.add_argument("--dataName", default="50_MELA0085_ddnm_avg512", help="Name of data.")
    
    parser.add_argument("--pretrain", default="", help="Name of data.")
    
    parser.add_argument("--use_wandb", action="store_true",
                        help="Save experiment information and log into W&B")
    parser.add_argument("--use_sds", action="store_true",
                        help="Load Diffusion Model to use SDS Loss")
    return parser

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

parser = config_parser()
args = parser.parse_args()

cfg = load_config(args.config)
cfg['exp']['expname'] = args.expName
cfg['exp']['datadir'] = f"./data/{args.dataName}.pickle"
if args.pretrain != "":
    cfg['pretrain'] = args.pretrain

cfg['use_wandb'] = args.use_wandb
cfg['use_sds'] = args.use_sds

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BasicTrainer(Trainer):
    def __init__(self):
        """
        Basic network trainer.
        """
        super().__init__(cfg, device)
        print(f"[Start] exp: {cfg['exp']['expname']}, net: Basic network")

    def compute_loss(self, data, global_step, idx_epoch):
        rays = data["rays"].reshape(-1, 8)
        projs = data["projs"].reshape(-1)
        
        ret = render(rays, self.net, self.net_fine, **self.conf["render"])
        projs_pred = ret["acc"]  # 1024
        
        if self.patch_size > 0:
            patch = data["patch_projs"]
            patch_pred = ret["acc"][:self.patch_size**2].reshape(self.patch_size, self.patch_size)
        
        loss = {"loss": 0.}
        
        calc_mse_loss(loss, projs, projs_pred)
        
        if "acc0" in ret:
            loss_coarse = torch.mean((projs-ret["acc0"])**2)
            loss["loss"] += loss_coarse
            loss["loss_coarse"] = loss_coarse
            
        
        if 'ssim' in cfg["loss"] and cfg["loss"]["ssim"] > 0 and self.patch_size > 0:
            calc_ssim_loss(loss, patch, patch_pred, cfg["loss"]["ssim"])
        
        if 'tv_2d' in cfg["loss"] and cfg["loss"]["tv_2d"] > 0 and self.patch_size > 0:
            calc_2d_tv_loss(loss, patch_pred, cfg["loss"]["tv_2d"])
        
        if 'tv_3d' in cfg["loss"] and cfg["loss"]["tv_3d"]["weight"] > 0:
            margin = cfg["loss"]["tv_3d"]["patch"]
            x, y, z = np.random.randint(margin*2, 512-margin*2, size=3)
            subvol_coord = self.eval_dset.voxels[x-margin:x+margin, y-margin:y+margin, z-margin:z+margin]
            subvol = run_network(subvol_coord, self.net_fine if self.net_fine is not None else self.net, self.netchunk).squeeze()
            calc_tv_loss(loss, subvol, cfg["loss"]["tv_3d"]["weight"])
        
        if self.net_fine is not None and self.conf["render"]["n_fine"] > 0:
            calc_adaptive_mse_loss(loss, projs_pred, ret["acc0"], projs)
            
        if 'regularizer' in cfg["loss"] and cfg["loss"]['regularizer']['weight'] != 0:
            # Without downsampling
            self.select_ind = np.random.choice(self.eval_dset.image.shape[0])
            lr_idx = self.select_ind // cfg["loss"]['regularizer']['scale']
            scale = cfg["loss"]['regularizer']['scale']
            gt_slide = self.lr_image[:,:,lr_idx * scale : lr_idx * scale + scale]
            inr_coord = self.eval_dset.voxels[:,:,lr_idx * scale : lr_idx * scale + scale]

            inr_slide = run_network(inr_coord, self.net_fine if self.net_fine is not None else self.net, self.netchunk).squeeze()            
            img = calc_regularizer(loss, gt_slide, inr_slide, cfg["loss"]['regularizer']['weight'], cfg["loss"]['regularizer']['scale'])
            
        # Log
        for ls in loss.keys():
            self.writer.add_scalar(f"train/{ls}", loss[ls].item(), global_step)

        return loss

    def eval_step(self, global_step, idx_epoch):
        """
        Evaluation step
        """
        # Evaluate projection
        select_ind = np.random.choice(len(self.eval_dset))
        projs = self.eval_dset.projs[select_ind]
        rays = self.eval_dset.rays[select_ind].reshape(-1, 8)
        H, W = projs.shape
        projs_pred = []
        for i in range(0, rays.shape[0], self.n_rays):
            projs_pred.append(render(rays[i:i+self.n_rays], self.net, self.net_fine, **self.conf["render"])["acc"])  # 2048 * 128
        projs_pred = torch.cat(projs_pred, 0).reshape(H, W)  # 512x512

        # Evaluate density
        image = self.eval_dset.image # 512, 512, 512
        image_pred = run_network(self.eval_dset.voxels, self.net_fine if self.net_fine is not None else self.net, self.netchunk)
        image_pred = image_pred.squeeze()
        loss = {
            "proj_mse": get_mse(projs_pred, projs),
            "proj_psnr": get_psnr(projs_pred, projs),
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
        show_proj = torch.concat([projs, projs_pred], dim=1)
        
        # if self.use_regularizer:
        #     lr_idx = self.select_ind // cfg["loss"]['regularizer']['scale']
        #     if global_step % 3 == 0:
        #         gt_slide = self.lr_image[:,:,lr_idx]
        #         inr_coord = self.eval_dset.voxels[:,:,self.select_ind]
        #     elif global_step % 3 == 1:
        #         gt_slide = self.lr_image[:,lr_idx,:]
        #         inr_coord = self.eval_dset.voxels[:,self.select_ind,:]
        #     else:
        #         gt_slide = self.lr_image[lr_idx,:,:]
        #         inr_coord = self.eval_dset.voxels[self.select_ind,:,:]
        #     inr_slide = run_network(inr_coord, self.net_fine if self.net_fine is not None else self.net, self.netchunk).squeeze()
        #     inr_slide_resized = F.interpolate(inr_slide.unsqueeze(0).unsqueeze(0), size=(128, 128), mode='bilinear', align_corners=False).squeeze().squeeze()

        #     show_slide = torch.concat([gt_slide, inr_slide_resized], dim=1)

        for ls in loss.keys():
            self.writer.add_scalar(f"eval/{ls}", loss[ls], global_step)
            
        # Save
        eval_save_dir = osp.join(self.evaldir, f"epoch_{idx_epoch:05d}")
        os.makedirs(eval_save_dir, exist_ok=True)
        
        np.save(osp.join(eval_save_dir, "image_pred.npy"), image_pred.cpu().detach().numpy())
        if idx_epoch == self.epochs:
            np.save(osp.join(eval_save_dir, "image_gt.npy"), image.cpu().detach().numpy())
        iio.imwrite(osp.join(eval_save_dir, "slice_show_row1_gt_row2_pred.png"), (cast_to_image(show_density)*255).astype(np.uint8))
        iio.imwrite(osp.join(eval_save_dir, "proj_show_left_gt_right_pred.png"), (cast_to_image(show_proj)*255).astype(np.uint8))
        # if self.use_regularizer:
        #     iio.imwrite(osp.join(eval_save_dir, "slide_show_left_gt_right_pred.png"), (cast_to_image(show_slide)*255).astype(np.uint8))
        
        if self.use_wandb:
            self.loss_table.add_data(*([idx_epoch] + [loss[i].item() for i in loss.keys()]))
            image_dict = {
                "Slide Image" : wandb.Image(show_density),
                "Projection Image" : wandb.Image(show_proj)
            }
            if self.use_regularizer:
                image_dict["Regularization Image"] = wandb.Image(show_slide)
            wandb.log(image_dict, step=idx_epoch)
        with open(osp.join(eval_save_dir, "stats.txt"), "w") as f: 
            for key, value in loss.items(): 
                f.write("%s: %f\n" % (key, value.item()))

        return loss


trainer = BasicTrainer()
trainer.start()
        
