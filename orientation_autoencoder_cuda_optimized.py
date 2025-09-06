import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# 1. OrientationAutoencoder definition with flexible grid_size (no longer requiring power-of-2)
class OrientationAutoencoder(nn.Module):
    def __init__(self, input_channels=4, latent_dim=32, base_channels=32, grid_size=40, min_spatial=4):
        super().__init__()
        self.input_channels = input_channels
        self.grid_size = grid_size
        # Compute how many times we can downsample by 2 while staying integer and >= min_spatial
        n_down = 0
        temp = grid_size
        while temp % 2 == 0 and temp // 2 >= min_spatial:
            temp = temp // 2
            n_down += 1
        self.n_down = n_down
        self.final_spatial = temp  # spatial size after downsampling

        # Encoder
        enc_layers = []
        in_ch = input_channels
        out_ch = base_channels
        for _ in range(n_down):
            enc_layers.append(nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=2, padding=1))
            enc_layers.append(nn.BatchNorm3d(out_ch))
            enc_layers.append(nn.ReLU(inplace=True))
            in_ch = out_ch
            out_ch = min(out_ch * 2, 256)
        self.encoder_conv = nn.Sequential(*enc_layers)

        # After encoder_conv, channels = in_ch
        self.final_channels = in_ch
        flattened_size = self.final_channels * (self.final_spatial ** 3)

        # Latent space
        self.fc_enc = nn.Linear(flattened_size, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, flattened_size)

        # Decoder
        dec_layers = []
        in_ch = self.final_channels
        out_ch = in_ch // 2
        for _ in range(n_down):
            # Use ConvTranspose3d to upsample by factor 2
            dec_layers.append(nn.ConvTranspose3d(in_ch, out_ch, kernel_size=3, stride=2,
                                                 padding=1, output_padding=1))
            dec_layers.append(nn.BatchNorm3d(out_ch))
            dec_layers.append(nn.ReLU(inplace=True))
            in_ch = out_ch
            out_ch = max(out_ch // 2, base_channels)
        # Final output conv to map back to input_channels
        dec_layers.append(nn.Conv3d(in_ch, input_channels, kernel_size=3, padding=1))
        self.decoder_conv = nn.Sequential(*dec_layers)

    def encode(self, x):
        # x: [B, input_channels, D, H, W]
        h = self.encoder_conv(x)  # [B, C, d, h, w]
        B = h.shape[0]
        h = h.view(B, -1)
        z = self.fc_enc(h)
        return z

    def decode(self, z):
        B = z.shape[0]
        h = self.fc_dec(z)  # [B, C * d * h * w]
        h = h.view(B, self.final_channels, self.final_spatial,
                   self.final_spatial, self.final_spatial)
        x_recon = self.decoder_conv(h)

        # Normalize quaternion per voxel
        norm = torch.norm(x_recon, dim=1, keepdim=True).clamp(min=1e-6)
        x_recon = x_recon / norm
        return x_recon

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon

# 2. Updated .ori loader

def load_ori(path, grid_size):
    """
    Reads a DAMASK‐MTEX .ori file (with header lines beginning '#'),
    parses quaternion lines, reshapes to [4, D, H, W], and enforces unit norm.
    """
    with open(path, 'r') as f:
        lines = [l for l in f if not l.startswith('#')]
    data = np.loadtxt(lines)  # shape (N_voxels, 4)
    expected = grid_size**3
    if data.shape[0] != expected:
        raise ValueError(f"{path}: expected {expected} voxels but got {data.shape[0]}")
    arr = data.reshape(grid_size, grid_size, grid_size, 4)
    arr = np.moveaxis(arr, -1, 0)  # → (4, D, H, W)
    tensor = torch.from_numpy(arr).float()
    norm = tensor.norm(dim=0, keepdim=True).clamp(min=1e-6)
    return tensor / norm

class RveDataset(Dataset):
    def __init__(self, root_dir, grid_size, cache=False):
        self.paths = []
        self.cache = cache
        self.cache_data = {} if cache else None
        expected = grid_size**3
        for sim in sorted(os.listdir(root_dir)):
            sim_dir = os.path.join(root_dir, sim)
            if not os.path.isdir(sim_dir):
                continue
            for path in sorted(glob.glob(f"{sim_dir}/ori_increment_*.ori")):
                # count non-header lines
                cnt = 0
                try:
                    with open(path, 'r') as f:
                        for l in f:
                            if not l.startswith('#'):
                                cnt += 1
                except Exception as e:
                    print(f"Warning: could not read {path}: {e}")
                    continue
                if cnt != expected:
                    print(f"Skipping {path}: expected {expected} voxels but got {cnt}")
                    continue
                self.paths.append(path)
        if len(self.paths) == 0:
            raise RuntimeError(f"No .ori files found with grid_size={grid_size} under {root_dir}")
        self.grid_size = grid_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        if self.cache:
            if path in self.cache_data:
                tensor = self.cache_data[path]
            else:
                tensor = load_ori(path, self.grid_size)
                self.cache_data[path] = tensor
        else:
            tensor = load_ori(path, self.grid_size)
        return tensor, tensor

if __name__ == "__main__":
    # --- 3. Hyperparams & setup ---
    import argparse
    parser = argparse.ArgumentParser(description="Train Orientation Autoencoder with optimized GPU utilization.")
    parser.add_argument('--data-root', type=str, default=os.path.dirname(os.path.abspath(__file__)), help="Root dir containing sim_*/ folders")
    parser.add_argument('--grid-size', type=int, default=40, help="Grid size (e.g., 40)")
    parser.add_argument('--base-ch', type=int, default=32, help="Base channels")
    parser.add_argument('--latent-dim', type=int, default=32, help="Latent dimension")
    parser.add_argument('--batch-size', type=int, default=4, help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--epochs', type=int, default=100, help="Number of epochs")
    parser.add_argument('--num-workers', type=int, default=max(1, os.cpu_count() - 1), help="Number of DataLoader workers")
    parser.add_argument('--cache', action='store_true', help="Cache loaded tensors in memory (may use significant RAM)")
    args = parser.parse_args()

    data_root = args.data_root
    grid_size = args.grid_size
    base_ch = args.base_ch
    latent_dim = args.latent_dim
    batch_size = args.batch_size
    lr = args.lr
    n_epochs = args.epochs
    num_workers = args.num_workers
    cache = args.cache

    model_path = "ori_autoencoder.pth"
    output_dir = "latent_trajectories"
    os.makedirs(output_dir, exist_ok=True)

    # Enable cudnn benchmark for potential speedup on fixed-size inputs
    torch.backends.cudnn.benchmark = True

    # --- 4. DataLoader ---
    dataset = RveDataset(data_root, grid_size, cache=cache)
    print(f"Using grid_size={grid_size}, found {len(dataset)} valid .ori files. num_workers={num_workers}, cache={cache}")
    for p in dataset.paths[:5]:
        print("  sample:", p)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, pin_memory=True,
                        persistent_workers=(num_workers > 0))

    # --- 5. Model + training setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OrientationAutoencoder(
        input_channels=4,
        latent_dim=latent_dim,
        base_channels=base_ch,
        grid_size=grid_size
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    # Use new amp API
    scaler = torch.amp.GradScaler()

    # Optional quick shape test
    # with torch.no_grad():
    #     x_test = torch.randn(1, 4, grid_size, grid_size, grid_size).to(device)
    #     out_test = model(x_test)
    #     print("Test output shape:", out_test.shape)

    # --- 6. Training loop with mixed precision ---
    for ep in range(1, n_epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in loader:
            # Transfer to GPU with non_blocking
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad()
            # Use new autocast API
            with torch.amp.autocast(device_type=device.type):
                recon = model(x)
                loss = criterion(recon, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * x.size(0)
        avg_loss = total_loss / len(dataset)
        print(f"[Epoch {ep:02d}/{n_epochs}] Loss: {avg_loss:.6f}")

    # --- 7. Save model & extract latents ---
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    model.eval()
    with torch.no_grad():
        for sim in sorted(os.listdir(data_root)):
            sim_dir = os.path.join(data_root, sim)
            if not os.path.isdir(sim_dir):
                continue
            codes = []
            for file in sorted(glob.glob(f"{sim_dir}/ori_increment_*.ori")):
                try:
                    x = load_ori(file, grid_size).unsqueeze(0).to(device, non_blocking=True)
                except ValueError:
                    continue
                with torch.amp.autocast(device_type=device.type):
                    z = model.encode(x)
                codes.append(z.cpu().squeeze(0).numpy())
            if len(codes) == 0:
                continue
            traj = np.stack(codes, axis=0)
            out_path = os.path.join(output_dir, f"{sim}_ori_latent.npy")
            np.save(out_path, traj)
            print(f"Saved latent trajectory for {sim} → {out_path}")

    print("All done.")
