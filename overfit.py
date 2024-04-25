import torch
import numpy as np
import pickle
from tqdm import tqdm

from src.network import get_network
from src.encoder import get_encoder

def run_network(inputs, fn, netchunk):
    """
    Prepares inputs and applies network "fn".
    """
    uvt_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    out_flat = torch.cat([fn(uvt_flat[i:i + netchunk]) for i in range(0, uvt_flat.shape[0], netchunk)], 0)
    out = out_flat.reshape(list(inputs.shape[:-1]) + [out_flat.shape[-1]])
    return out 

def resample_3d(data, ratio):
    if not all(isinstance(r, (int, float)) for r in ratio) or len(ratio) != 3:
        raise ValueError("Ratio must be a tuple of three integers or floats.")
    
    shape = np.array(data.shape)
    new_shape = tuple(int(s * r) for s, r in zip(shape, ratio))
    
    if any(ns <= 0 for ns in new_shape):
        raise ValueError("Invalid resampling ratio results in non-positive shape.")
    
    resampled_data = np.zeros(new_shape)
    
    for i in range(new_shape[0]):
        for j in range(new_shape[1]):
            for k in range(new_shape[2]):
                orig_i = int(i / ratio[0])
                orig_j = int(j / ratio[1])
                orig_k = int(k / ratio[2])
                resampled_data[i, j, k] = data[orig_i, orig_j, orig_k]
                
    return resampled_data

def get_voxels():
        """
        Get the voxels.
        """
        # nVoxel = np.array((128, 128, 128))
        # dVoxel = np.array((2., 2., 2.))/1000
        nVoxel = np.array((256, 256, 256))
        dVoxel = np.array((1., 1., 1.))/1000
        sVoxel = nVoxel * dVoxel
        
        n1, n2, n3 = nVoxel
        s1, s2, s3 = sVoxel / 2 - dVoxel / 2

        xyz = np.meshgrid(np.linspace(-s1, s1, n1),
                        np.linspace(-s2, s2, n2),
                        np.linspace(-s3, s3, n3), indexing="ij")
        voxel = np.asarray(xyz).transpose([1, 2, 3, 0])
        return voxel
    

def main():
    epochs = 500
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    with open("/workspace/naf_cbct/data/MELA0085_HR_100_2.pickle", "rb") as handle:
        data = pickle.load(handle)
    
    print("Network Generating...", end="")
    network = get_network('mlp')
    print("Done!")
    
    print("Encoder Generating...", end="")
    encoder = get_encoder(
        encoding='hashgrid',
        input_dim=3,
        num_levels=16,
        level_dim=2,
        base_resolution=16,
        log2_hashmap_size=19
    )
    print("Done!")
    
    net = network(encoder, num_layers=4, hidden_dim=32, skips=[2], out_dim=1, last_activation='sigmoid', bound=.3)
    net = net.cuda(device)
    print("Full Network Generated!")
    
    optimizer = torch.optim.Adam(params=net.parameters(), lr=0.001, betas=(0.9, 0.999))
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optimizer, step_size=1500, gamma=0.1
    )
    
    # gt = data['image']
    gt = resample_3d(data['image'], (.5, .5, .5))
    
    gt = torch.tensor(gt).cuda(device)
    print(f"GT Volume for prediction generated with size {gt.shape}")
    
    voxel = torch.tensor(get_voxels(), dtype=torch.float32).cuda(device)
    
    print("Start Train...")
    net.train()
    
    pbar = tqdm(total= epochs)
    
    for i in range(epochs):
        image_pred = run_network(voxel, net, 256)  # (128, 128, 128, 1) 4096
        loss = torch.mean((image_pred.squeeze().unsqueeze(0) - gt.unsqueeze(0))**2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        pbar.set_description(f"[Epoch {i+1}/{epochs}]")
        pbar.set_postfix({
            "loss" :loss.item(), 
            "lr":optimizer.param_groups[0]['lr']
        })
        pbar.update(1)
        
    np.save('image_pred_256_small_chunk.npy', image_pred.detach().cpu().numpy())
    

if __name__ == "__main__":
    main()