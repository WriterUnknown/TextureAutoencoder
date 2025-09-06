# ori_evolution.py
import os
import numpy as np
import torch
from orientation_autoencoder_cuda_optimized import OrientationAutoencoder
import argparse


def save_ori(quat_array: np.ndarray, path: str):
    """
    Save a quaternion field to a .ori file.
    quat_array: (4, D, H, W)
    Writes in row-major voxel order.
    """
    # move channel to last: (D, H, W, 4)
    vox = np.moveaxis(quat_array, 0, -1)
    D, H, W, _ = vox.shape
    flat = vox.reshape(-1, 4)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(f"# Quaternion field {D}x{H}x{W}\n")
        for q in flat:
            f.write(f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")


def main():
    parser = argparse.ArgumentParser(description="Decode latent evolution to .ori sequence")
    parser.add_argument('--latents', type=str, default='pred_ori_evolution.npy',
                        help='Path to (T, latent_dim) latent npy')
    parser.add_argument('--checkpoint', type=str, default='ori_autoencoder.pth',
                        help='Path to orientation autoencoder weights')
    parser.add_argument('--grid-size', type=int, required=True,
                        help='Spatial grid size used during autoencoder training')
    parser.add_argument('--base-ch', type=int, default=32,
                        help='Base channels used during autoencoder training')
    parser.add_argument('--output-dir', type=str, default='ori_sequence_pred',
                        help='Directory to save .ori files')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load latents
    latents = np.load(args.latents).astype(np.float32)  # shape (T, latent_dim)
    T, latent_dim = latents.shape

    # Build model and load weights
    model = OrientationAutoencoder(
        input_channels=4,
        latent_dim=latent_dim,
        base_channels=args.base_ch,
        grid_size=args.grid_size
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Decode each latent and save .ori
    with torch.no_grad():
        for t in range(T):
            z = torch.from_numpy(latents[t:t+1]).to(device)  # (1, latent_dim)
            x_recon = model.decode(z)  # (1,4,D,H,W)
            toks = x_recon.cpu().numpy().squeeze(0)  # (4,D,H,W)
            out_path = os.path.join(args.output_dir, f'step_{t:03d}.ori')
            save_ori(toks, out_path)
            print(f"Saved {out_path}")

if __name__ == '__main__':
    main()
