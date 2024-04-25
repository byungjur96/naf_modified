import torch
# from einops import rearrange


def calc_mse_loss(loss, x, y):
    """
    Calculate mse loss.
    """
    # Compute loss
    loss_mse = torch.mean((x-y)**2)
    loss["loss"] += loss_mse
    loss["loss_mse"] = loss_mse
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


# num_train_timesteps = 1000
# t_range=[0.02, 0.98]
# min_step = int(num_train_timesteps * t_range[0])
# max_step = int(num_train_timesteps * t_range[1])
# scheduler = None
# def calc_sds_loss(loss, img):
#     num_rays, _ = img.shape
#     h = w = int(num_rays ** (1/2))
#     img = rearrange(img, "(h w) c -> 1 c h w", h-h, w=w)
#     t = torch.randint(min_step, max_step + 1, [1], dtype=torch.long)
    
#     with torch.no_grad():
#         noise = torch.randn_like(img)
#         img_noisy = scheduler.add_noise(img, noise, t)
        
        

