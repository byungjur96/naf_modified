import argparse
import os
import os.path as osp
import json
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from shutil import copyfile
import numpy as np
import yaml

from .dataset import TIGREDataset as Dataset
from .network import get_network
from .encoder import get_encoder

from .diffusion.models import Model as Diffusion

import wandb

class Trainer:
    def __init__(self, cfg, device="cuda"):

        # Args
        self.global_step = 0
        self.conf = cfg
        self.n_fine = cfg["render"]["n_fine"]
        self.epochs = cfg["train"]["epoch"]
        self.i_eval = cfg["log"]["i_eval"]
        self.i_save = cfg["log"]["i_save"]
        self.netchunk = cfg["render"]["netchunk"]
        self.n_rays = cfg["train"]["n_rays"]
        self.load_pretrain = "pretrain" in cfg.keys()
  
        # Log direcotry
        self.expdir = osp.join(cfg["exp"]["expdir"], cfg["exp"]["expname"])
        self.ckptdir = osp.join(self.expdir, "ckpt.tar")
        self.ckptdir_backup = osp.join(self.expdir, "ckpt_backup.tar")
        self.evaldir = osp.join(self.expdir, "eval")
        os.makedirs(self.evaldir, exist_ok=True)
        with open(osp.join(self.expdir, 'config.yaml'), 'w') as yaml_file:
            yaml.dump(self.conf, yaml_file, default_flow_style=False)

        # Dataset
        self.patch_size = cfg['loss']['patch'] if 'patch' in cfg['loss'] else 0
        # If SSIM Loss or TV_2d Loss are not used, no need for patchsampling
        if ('ssim' not in cfg["loss"] or cfg["loss"]["ssim"] == 0) and ('tv_2d' not in cfg["loss"] or cfg["loss"]["tv_2d"] == 0):
            print("No need for patch sampling. Convert into random sampling")
            self.patch_size = 0
        train_dset = Dataset(cfg["exp"]["datadir"], cfg["train"]["n_rays"], "train", device, self.patch_size)
        self.eval_dset = Dataset(cfg["exp"]["datadir"], cfg["train"]["n_rays"], "val", device) if self.i_eval > 0 else None
        self.train_dloader = torch.utils.data.DataLoader(train_dset, batch_size=cfg["train"]["n_batch"])
        self.voxels = self.eval_dset.voxels if self.i_eval > 0 else None
        
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
        if self.load_pretrain:
            pretrain_path = osp.join(cfg["exp"]["expdir"], cfg["pretrain"], "ckpt.tar")
            pretrain_ckpt = torch.load(pretrain_path)
            self.net.load_state_dict(pretrain_ckpt["network"])
            print(f"Pretrain Model Weight from {pretrain_path} Loaded!")
            
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

        # Summary writer
        self.writer = SummaryWriter(self.expdir)
        self.writer.add_text("parameters", self.args2string(cfg), global_step=0)
        self.use_wandb = cfg['use_wandb']
        
        self.use_regularizer = "regularizer" in cfg["loss"] and cfg["loss"]["regularizer"]["weight"] > 0
        if self.use_regularizer:
            def averagepooling(volume, scale):
                return volume.reshape(volume.shape[0] // scale, scale, volume.shape[1] // scale, scale, volume.shape[2] // scale, scale).mean(axis=(1, 3, 5))
            self.lr_image = averagepooling(self.eval_dset.image, 4)
            
        
        if self.use_wandb:
            wandb.init(project="CT Reconstruction", config=self.conf)
            wandb.run.name = cfg['exp']['expname']
            self.loss_table = wandb.Table(columns=["epoch", "proj_mse", "proj_psnr", "psnr_3d", "ssim_3d"])

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
                if self.use_wandb:
                    loss_info = {
                        "loss" : loss_train["loss"],
                        "Learning Rate" : self.optimizer.param_groups[0]['lr']
                    }
                    if self.use_regularizer:
                        loss_info["Regularizer"] = loss_train["regularization"]
                        
                    wandb.log(loss_info, step=idx_epoch)
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
            self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], self.global_step)
            self.lr_scheduler.step()

        tqdm.write(f"Training complete! See logs in {self.expdir}")
        if self.use_wandb:
            wandb.log({"Loss" : self.loss_table})

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
        
    def compute_loss(self, data, global_step, idx_epoch):
        """
        Training step
        """
        raise NotImplementedError()


    def eval_step(self, global_step, idx_epoch):
        """
        Evaluation step
        """
        raise NotImplementedError()
        