import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import numpy as np
import random

class GridEncoder(nn.Module):
    def __init__(self, in_channels=7, out_dim=64):
        """
        in_channels=7 for the 7-channel BEV grid
        out_dim=64 is the size of the final CNN embedding
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=7, stride=2, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=7, stride=2, padding=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=7, stride=2, padding=2)
        self.fc = nn.Linear(64*4*4, out_dim)  # after 3 conv layers: shape ~ (64,4,4)

    def forward(self, x):
        # x shape: (B, 7, 40, 40)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # Flatten
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x  # shape (B, out_dim)

def make_mlp(input_dim, hidden_dim=64, output_dim=32):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
        nn.ReLU()
    )

class LaserEncoder(nn.Module):
    def __init__(self, in_dim=8, out_dim=32):
        super().__init__()
        self.mlp = make_mlp(in_dim, 64, out_dim)
    def forward(self, x):
        # x shape: (B, 8)
        return self.mlp(x)

class TargetEncoder(nn.Module):
    def __init__(self, in_dim=10, out_dim=32):  # des_info is 10-dim since run 32 (+densities)
        super().__init__()
        self.mlp = make_mlp(in_dim, 64, out_dim)
    def forward(self, x):
        # x shape: (B, 6)
        return self.mlp(x)

class WaypointsEncoder(nn.Module):
    def __init__(self, in_dim=20, out_dim=32):
        super().__init__()
        self.mlp = make_mlp(in_dim, 64, out_dim)
    def forward(self, x):
        # x shape: (B, 10)
        return self.mlp(x)

class FusionTransformer(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=2):
        """
        A small Transformer encoder.
        d_model=64: internal dimensionality for each token
        nhead=4: multi-head attention
        num_layers=2: stack 2 encoder layers
        """
        super().__init__()
        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Because we have 4 tokens, each "token" must be of dimension d_model
        # We'll need small linear layers to up/down project the 32-d to 64-d or vice versa
    def forward(self, tokens):
        """
        tokens shape: (batch_size, seq_len=4, d_model=64)
          or we might fix the shape with some padding approach.

        The PyTorch transformer expects shape: (seq_len, batch_size, d_model).
        So we must transpose first.
        """
        # (B, 4, 64) -> (4, B, 64)
        x = tokens.transpose(0, 1)
        x = self.transformer(x)  # (4, B, 64)
        x = x.transpose(0, 1)    # back to (B, 4, 64)
        return x

class UpProject(nn.Module):
    def __init__(self, in_dim=32, out_dim=64):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
    def forward(self, x):
        return self.lin(x)

class SACActor(nn.Module):
    """
    Given the four raw inputs, produce a mean and log_std for 2D continuous action.
    """
    def __init__(self):
        super().__init__()
        # Encoders
        self.grid_encoder = GridEncoder(in_channels=7, out_dim=64)
        self.laser_encoder = LaserEncoder(in_dim=8, out_dim=32)
        self.target_encoder = TargetEncoder(in_dim=8, out_dim=32)
        self.wp_encoder = WaypointsEncoder(in_dim=20, out_dim=32)

        # Up-project for the 3 smaller MLPs
        self.up_laser = UpProject(in_dim=32, out_dim=64)
        self.up_target = UpProject(in_dim=32, out_dim=64)
        self.up_wp = UpProject(in_dim=32, out_dim=64)

        # Transformer to fuse
        self.fusion = FusionTransformer(d_model=64, nhead=4, num_layers=2)

        # After transformer we might pool or just flatten the 4 tokens
        self.post_fusion_dim = 64

        # Final MLP to produce mean and log_std
        # fused_mean (64) + a_ref (2) → 66
        self.mlp = nn.Sequential(
            nn.Linear(self.post_fusion_dim + 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.delta_head = nn.Linear(64, 2)
        self.gate_head  = nn.Linear(64, 1)
        self.log_std_head = nn.Linear(64, 2)

    def forward(self, grid, laser, target, wps, a_ref):
        """
        grid:  (B, 7, 40, 40)
        laser: (B, 8)
        target:(B, 6)
        wps:   (B, 10)
        """
        # Encode
        grid_emb = self.grid_encoder(grid)        # (B, 64)
        laser_emb = self.laser_encoder(laser)     # (B, 32)
        target_emb = self.target_encoder(target)  # (B, 32)
        wp_emb = self.wp_encoder(wps)             # (B, 32)

        # Up-project the 3 smaller
        laser64 = self.up_laser(laser_emb)    # (B,64)
        targ64 = self.up_target(target_emb)   # (B,64)
        wp64   = self.up_wp(wp_emb)           # (B,64)

        # Combine into tokens => shape (B, 4, 64)
        tokens = torch.stack([grid_emb, laser64, targ64, wp64], dim=1)
        # Fusion
        fused = self.fusion(tokens)  # (B, 4, 64)

        # Let's do a simple "mean pool" over the 4 tokens
        fused_mean = fused.mean(dim=1)  # (B, 64)

        x = torch.cat([fused_mean, a_ref], dim=-1)

        # Pass into final MLP
        hidden = self.mlp(x)   # (B, 64)

        delta    = self.delta_head(hidden)                # (B,2)
        alpha    = torch.sigmoid(self.gate_head(hidden))  # (B,1)
        log_std  = torch.clamp(self.log_std_head(hidden), -20, 2)

        mean = a_ref + (1 - alpha) * delta

        return mean, log_std


class SACCritic(nn.Module):
    def __init__(self):
        super().__init__()
        # We can either reuse the same encoders or define new ones
        self.grid_encoder = GridEncoder(in_channels=7, out_dim=64)
        self.laser_encoder = LaserEncoder(in_dim=8, out_dim=32)
        self.target_encoder = TargetEncoder(in_dim=8, out_dim=32)
        self.wp_encoder = WaypointsEncoder(in_dim=20, out_dim=32)

        self.up_laser = UpProject(32, 64)
        self.up_target = UpProject(32, 64)
        self.up_wp = UpProject(32, 64)

        self.fusion = FusionTransformer(d_model=64, nhead=4, num_layers=2)

        # We'll produce a final embedding, then we also need to incorporate action
        # Let's do it after the transformer
        self.post_fusion_dim = 64
        self.post_fusion_mlp = nn.Sequential(
            nn.Linear(self.post_fusion_dim + 2, 128),  # +2 for (accel, steer)
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Q-value
        )

    def forward(self, grid, laser, target, wps, action):
        """
        action shape: (B, 2) for (accel, steer)
        returns Q-value: (B, 1)
        """
        # Encode
        grid_emb = self.grid_encoder(grid)        # (B, 64)
        laser_emb = self.laser_encoder(laser)     # (B, 32)
        target_emb = self.target_encoder(target)  # (B, 32)
        wp_emb = self.wp_encoder(wps)             # (B, 32)

        # Up-project
        l64 = self.up_laser(laser_emb)
        t64 = self.up_target(target_emb)
        w64 = self.up_wp(wp_emb)

        # Fusion tokens
        tokens = torch.stack([grid_emb, l64, t64, w64], dim=1)
        fused = self.fusion(tokens)  # (B, 4, 64)
        fused_mean = fused.mean(dim=1)

        # Concat action
        x = torch.cat([fused_mean, action], dim=-1)  # (B, 66)

        q_value = self.post_fusion_mlp(x)  # (B,1)
        return q_value


class ReplayBuffer:
    def __init__(self, capacity=int(1e6)):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, obs, action, ref_action, reward, next_obs, next_ref_action, done):
        self._push_single(obs, action, ref_action, reward, next_obs, next_ref_action, done)

    def _push_single(self, obs, action, ref_action, reward, next_obs, next_ref_action, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (obs, action, ref_action, reward, next_obs, next_ref_action, done)
        self.position = (self.position + 1) % self.capacity

    def push_batch(self, batch):
        """
        batch: Iterable[tuple(obs, act, a_ref, rew, next_obs, done)]
        """
        for transition in batch:
            self._push_single(*transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs_batch, act_batch, ref_action_batch, rew_batch, next_obs_batch, next_ref_action_batch, done_batch = [], [], [], [], [], [], []

        for (obs, action, ref_action, reward, next_obs, next_ref_action, done) in batch:
            obs_batch.append(obs)
            act_batch.append(action)
            ref_action_batch.append(ref_action)
            rew_batch.append([reward])
            next_obs_batch.append(next_obs)
            next_ref_action_batch.append(next_ref_action)
            done_batch.append([done])

        return (obs_batch,
                np.array(act_batch),
                np.array(ref_action_batch),
                np.array(rew_batch),
                next_obs_batch,
                np.array(next_ref_action_batch),
                np.array(done_batch))

    def __len__(self):
        return len(self.buffer)