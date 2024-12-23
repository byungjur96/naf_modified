import torch
from einops import rearrange
import numpy as np
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure as ssim

def calc_mse_loss(loss, x, y):
    """
    Calculate mse loss.
    """
    # Compute loss
    loss_mse = torch.mean((x-y)**2)
    loss["loss"] += loss_mse
    loss["loss_mse"] = loss_mse
    return loss

def calc_l1_loss(loss, x, y):
    """
    Calculate mse loss.
    """
    # Compute loss
    loss_l1 = torch.mean(abs(x-y))
    loss["loss"] += loss_l1
    loss["loss_l1"] = loss_l1
    return loss

def calc_ssim_loss(loss, x, y, weight=1e-5):
    loss_ssim = 1 - ssim(x.unsqueeze(0), y.unsqueeze(0).unsqueeze(0))  # SSIM returns similarity, we use (1 - SSIM) as loss
    loss["loss"] += (loss_ssim * weight)
    loss["loss_ssim"] = loss_ssim *weight
    
    return loss

def calc_2d_tv_loss(loss, image, weight):
    loss_horizontal = torch.sum(torch.abs(image[:, :-1] - image[:, 1:]))
    loss_vertical = torch.sum(torch.abs(image[:-1, :] - image[1:, :]))
    tv_loss = weight * (loss_horizontal + loss_vertical)
    
    loss["loss"] += tv_loss
    loss["loss_2d_tv"] = tv_loss
    
    return loss

def calc_adaptive_mse_loss(loss, x, x0, y):
    weights = torch.sqrt(torch.abs(x - y)).detach()
    loss_adaptive_mse = torch.mean(weights * torch.square(x0 - y))
    
    loss["loss_adaptive_mse"] = loss_adaptive_mse
    loss["loss"] += loss_adaptive_mse
    return loss

def calc_regularizer(loss, gt, pred, weight, scale):
    """
    Calculate MSE Loss for random slide
    """
    gt = gt.unsqueeze(dim=0)
    pred = pred.unsqueeze(dim=0) # .unsqueeze(dim=0)
    
    # downsampled_pred = F.avg_pool3d(pred, kernel_size=(scale, scale, scale)).squeeze(-1).squeeze(0)
    # reg = torch.mean((gt-downsampled_pred)**2)
    reg = torch.mean((gt-pred)**2)
    
    loss["loss"] += reg * weight
    if "regularization" in loss:
        loss["regularization"] += reg
    else:
        loss["regularization"] = reg
    
    # pred_img = downsampled_pred.detach().cpu().numpy()
    # pred_img = (255 * (pred_img - np.min(pred_img)) / (np.max(pred_img) - np.min(pred_img))).astype(np.uint8)
    
    # gt_img = gt.detach().cpu().numpy()
    # gt_img = (255 * (gt_img - np.min(gt_img)) / (np.max(gt_img) - np.min(gt_img) + 1e-7)).astype(np.uint8)
    
    # img = np.concatenate((gt_img, pred_img), axis=1)
    
    # return img.squeeze(0)
    return loss

def calc_tv_loss(loss, x, k):
    """
    Calculate total variation loss.
    Args:
        x (n1, n2, n3, 1): 3d density field.
        k: relative weight
    """
    n1, n2, n3 = x.shape
    tv_1 = torch.abs(x[1:,1:,1:]-x[:-1,1:,1:]).sum()
    tv_2 = torch.abs(x[1:,1:,1:]-x[1:,:-1,1:]).sum()
    tv_3 = torch.abs(x[1:,1:,1:]-x[1:,1:,:-1]).sum()
    tv = (tv_1+tv_2+tv_3) / (n1*n2*n3)
    loss["loss"] += tv * k
    loss["loss_tv"] = tv * k
    return loss

