import pickle
import numpy as np
import os
import argparse

PICKLE_FOLDER = "/workspace/naf_modified/data"
DDNM_FOLDER = "/workspace/Diffusion/DDNM/exp/image_samples"

def main(pickle_file, numpy_file, new_prefix):
    # Load the pickle file
    with open(os.path.join(PICKLE_FOLDER, f"{pickle_file}.pickle"), 'rb') as f:
        data = pickle.load(f)

    # Load the numpy file
    new_projections = np.load(os.path.join(DDNM_FOLDER, numpy_file, 'whole.npy'))
    
    # Check if the number of projections (n) is the same
    old_projections = data['train']['projections']
    if old_projections.shape[0] != new_projections.shape[0]:
        raise ValueError(f"Mismatch in number of projections: Pickle file has {old_projections.shape[0]}, but numpy file has {new_projections.shape[0]}")


    # Adjust new projections based on old projections' min and max values
    adjusted_projections = []
    for old_proj, new_proj in zip(old_projections, new_projections):
        min_val, max_val = old_proj.min(), old_proj.max()
        adjusted_proj = new_proj * (max_val - min_val) + min_val
        adjusted_proj = np.transpose(adjusted_proj, (1,2,0))[:,:,0]
        adjusted_projections.append(adjusted_proj)
    adjusted_projections = np.array(adjusted_projections)

    # Replace the projections in data['train']['projections']
    data['train']['projections'] = adjusted_projections
    
    # Create new filename
    new_filename = os.path.join(PICKLE_FOLDER, f"{pickle_file}_{new_prefix}.pickle")

    
    # Save the modified pickle file
    with open(new_filename, 'wb') as f:
        pickle.dump(data, f)

    print(f"New pickle file saved as: {new_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replace projections in a pickle file with new numpy projections.")
    parser.add_argument('--pickle', default='MELA257_gt512_clamp_100', help="Path to the pickle file")
    parser.add_argument('--proj', required=True, help="Path to the numpy file containing new projections")
    parser.add_argument('--prefix', default='DDNM', help="New prefix for the output pickle file")

    args = parser.parse_args()

    main(args.pickle, args.proj, args.prefix)
