import numpy as np
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from SAC import GridEncoder, LaserEncoder, TargetEncoder, WaypointsEncoder, UpProject
from frenet_planner import CAND_FEAT_DIM, CAND_FEAT_DIM_V2, CAND_FEAT_DIM_V3  # noqa: F401 (re-export)

# per-neighbor features (vehicles within a 60 m sensing radius, nearest 15):
# [dx/70, dy/70, sin(rel_heading), cos(rel_heading), v/16,
#  dist_to_conflict_box/70, approach_speed/16,
#  accel_est/8 (speed trend — "is that car braking for me?"),
#  maneuver one-hot (straight, left, right),
#  rl_flag]
# The first 11 are observable and go to BOTH actor and critic. The maneuver
# one-hot is the "turn signal" — whether the neighbor's route actually
# conflicts with ours. The TTA-derived pair (urgency/arrival delta) was
# removed in run 20: run 13 proved it contributes nothing beyond the raw
# d/v it is computed from. rl_flag is critic-only global information.
NBR_FEAT_DIM = 12
ACTOR_NBR_FEAT_DIM = 11
# Plan-sharing (G7, optional): +4 dims per neighbour — the neighbour's
# committed-plan positions 1.5 s and 3.0 s ahead, in the ego frame
# (candidate-level V2V; frenet_planner.predict_plan_points has the source
# rules). Token layout becomes [11 shared | 4 plan | rl_flag]: the plan dims
# are shared with the actor, rl_flag stays critic-only and LAST either way.
NBR_PLAN_DIM = 4
PLAN_SHARE_TS = (1.5, 3.0)
# Priority broadcast (G8, optional): +2 dims per neighbour — delta-ETA to the
# conflict area (the EXACT ordering the ETA-IDM yields by; IDM.eta_priority_of)
# and a binary "they precede me". Common-knowledge right-of-way instead of
# pairwise inference. Layout [11 shared | plan? | prio? | rl_flag].
NBR_PRIO_DIM = 2


class TokenFusion(nn.Module):
    """
    Self-attention fusion over a variable-length token sequence.
    dropout=0 so that PPO's stored log-probs match recomputation exactly.
    """
    def __init__(self, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                        dim_feedforward=128, dropout=0.0,
                                        batch_first=True)
        self.tr = TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, tokens, key_padding_mask=None):
        # tokens: (B, S, 64); key_padding_mask: (B, S) True = ignore
        return self.tr(tokens, src_key_padding_mask=key_padding_mask)


class ObsEncoder(nn.Module):
    """
    Shared observation branch: [BEV] / laser / destination / waypoints ->
    4 (or 3) tokens. use_bev=False drops the CNN branch entirely — the run-18
    information ablation showed the BEV is redundant given the neighbor
    tokens / lasers / candidates, and the CNN is the largest single compute
    item of the network forward pass.
    """
    def __init__(self, use_bev=True):
        super().__init__()
        self.use_bev = use_bev
        if use_bev:
            self.grid_encoder = GridEncoder(in_channels=7, out_dim=64)
        self.laser_encoder = LaserEncoder(in_dim=8, out_dim=32)
        self.target_encoder = TargetEncoder(in_dim=10, out_dim=32)  # +3 density dims (run 32)
        self.wp_encoder = WaypointsEncoder(in_dim=20, out_dim=32)
        self.up_laser = UpProject(32, 64)
        self.up_target = UpProject(32, 64)
        self.up_wp = UpProject(32, 64)

    def forward(self, grid, laser, target, wps):
        toks = []
        if self.use_bev:
            toks.append(self.grid_encoder(grid))
        toks.append(self.up_laser(self.laser_encoder(laser)))
        toks.append(self.up_target(self.target_encoder(target)))
        toks.append(self.up_wp(self.wp_encoder(wps)))
        return torch.stack(toks, dim=1)  # (B, 4 or 3, 64)


class MAPPOActor(nn.Module):
    """
    Decentralized actor: scores each Frenet candidate trajectory.
    Observation tokens, neighbor-vehicle tokens (locally observable motion
    features only — right-of-way must be inferred, not given), and candidate
    tokens are fused with self-attention; a shared head turns each candidate
    token into a logit. Invalid candidates are masked to -inf.
    """
    def __init__(self, cand_feat_dim=CAND_FEAT_DIM, nbr_feat_dim=None,
                 use_bev=True, plan_sharing=False, priority_broadcast=False):
        super().__init__()
        if nbr_feat_dim is None:
            nbr_feat_dim = (ACTOR_NBR_FEAT_DIM
                            + (NBR_PLAN_DIM if plan_sharing else 0)
                            + (NBR_PRIO_DIM if priority_broadcast else 0))
        self.plan_sharing = plan_sharing
        self.priority_broadcast = priority_broadcast
        self.nbr_in_dim = nbr_feat_dim
        # candidate token width: 20 = v1/G7 planner, CAND_FEAT_DIM_V2 (23) =
        # v2 planner with safety feats. Weights of one width do not load
        # into the other — the planner generation is part of the model
        self.cand_in_dim = cand_feat_dim
        self.obs_encoder = ObsEncoder(use_bev=use_bev)
        self.cand_encoder = nn.Sequential(
            nn.Linear(cand_feat_dim, 64), nn.ReLU(),
            nn.Linear(64, 64),
        )
        self.nbr_encoder = nn.Sequential(
            nn.Linear(nbr_feat_dim, 64), nn.ReLU(),
            nn.Linear(64, 64),
        )
        self.fusion = TokenFusion()
        self.score = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, grid, laser, target, wps, cand, cand_mask, nbr, nbr_mask):
        """
        cand:      (B, K, F) candidate features
        cand_mask: (B, K) bool, True = selectable
        nbr:       (B, M, 7) observable neighbor features (zero-padded)
        nbr_mask:  (B, M) bool, True = real neighbor
        returns logits (B, K)
        """
        obs_tokens = self.obs_encoder(grid, laser, target, wps)      # (B,4,64)
        nbr_tokens = self.nbr_encoder(nbr)                           # (B,M,64)
        cand_tokens = self.cand_encoder(cand)                        # (B,K,64)
        tokens = torch.cat([obs_tokens, nbr_tokens, cand_tokens], dim=1)
        pad = torch.cat([
            torch.zeros(obs_tokens.shape[:2], dtype=torch.bool, device=nbr_mask.device),
            ~nbr_mask,
            torch.zeros(cand_tokens.shape[:2], dtype=torch.bool, device=nbr_mask.device),
        ], dim=1)
        fused = self.fusion(tokens, key_padding_mask=pad)
        n_front = obs_tokens.size(1) + nbr_tokens.size(1)
        logits = self.score(fused[:, n_front:, :]).squeeze(-1)       # (B,K)
        logits = logits.masked_fill(~cand_mask, -1e9)
        return logits


class MAPPOCritic(nn.Module):
    """
    Centralized value function (CTDE): the ego observation tokens are fused
    with tokens of all nearby vehicles (global information available during
    training only). Padded neighbor slots are excluded via key_padding_mask.
    """
    def __init__(self, nbr_feat_dim=None, use_bev=True, plan_sharing=False,
                 priority_broadcast=False):
        super().__init__()
        if nbr_feat_dim is None:
            nbr_feat_dim = (NBR_FEAT_DIM
                            + (NBR_PLAN_DIM if plan_sharing else 0)
                            + (NBR_PRIO_DIM if priority_broadcast else 0))
        self.plan_sharing = plan_sharing
        self.priority_broadcast = priority_broadcast
        self.nbr_in_dim = nbr_feat_dim
        self.obs_encoder = ObsEncoder(use_bev=use_bev)
        self.nbr_encoder = nn.Sequential(
            nn.Linear(nbr_feat_dim, 64), nn.ReLU(),
            nn.Linear(64, 64),
        )
        self.fusion = TokenFusion()
        self.value_head = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, grid, laser, target, wps, nbr, nbr_mask):
        """
        nbr:      (B, M, F) neighbor features (zero-padded)
        nbr_mask: (B, M) bool, True = real neighbor
        returns value (B, 1)
        """
        obs_tokens = self.obs_encoder(grid, laser, target, wps)      # (B,4,64)
        nbr_tokens = self.nbr_encoder(nbr)                           # (B,M,64)
        tokens = torch.cat([obs_tokens, nbr_tokens], dim=1)
        pad = torch.cat([
            torch.zeros(obs_tokens.shape[:2], dtype=torch.bool, device=nbr_mask.device),
            ~nbr_mask,
        ], dim=1)                                                    # True = ignore
        fused = self.fusion(tokens, key_padding_mask=pad)
        # mean-pool over the always-valid observation tokens
        pooled = fused[:, :obs_tokens.size(1), :].mean(dim=1)
        return self.value_head(pooled)


class RolloutBuffer:
    """
    On-policy buffer for dynamically spawning/despawning agents.
    Transitions are grouped per agent "track"; GAE is computed per track when
    it finishes (agent done, or bootstrapped at update time). Each transition
    covers `substeps` env steps (the replan window), so the per-transition
    discount is gamma**substeps.
    """
    def __init__(self, gamma=0.99, lam=0.95):
        self.gamma = gamma
        self.lam = lam
        self.tracks = {}     # agent key -> list of transition dicts
        self.ready = []      # finished transitions with 'adv' / 'ret'

    def add(self, key, transition):
        self.tracks.setdefault(key, []).append(transition)

    def finish(self, key, last_value):
        seq = self.tracks.pop(key, [])
        adv, next_v = 0.0, float(last_value)
        for tr in reversed(seq):
            disc = self.gamma ** max(tr['substeps'], 1)
            nonterm = 0.0 if tr['done'] else 1.0
            delta = tr['reward'] + disc * next_v * nonterm - tr['value']
            adv = delta + disc * self.lam * nonterm * adv
            tr['adv'] = adv
            tr['ret'] = adv + tr['value']
            next_v = tr['value']
        self.ready.extend(seq)

    def total_size(self):
        return len(self.ready) + sum(len(s) for s in self.tracks.values())

    def open_keys(self):
        return list(self.tracks.keys())

    def tensors(self, device):
        d = self.ready

        def stk(key, dtype):
            # from_numpy avoids the extra copy torch.tensor would make
            return torch.from_numpy(np.stack([t[key] for t in d])).to(device=device, dtype=dtype)

        out = {
            'grid':      stk('grid', torch.float32),
            'laser':     stk('laser', torch.float32),
            'target':    stk('target', torch.float32),
            'wps':       stk('wps', torch.float32),
            'cand':      stk('cand', torch.float32),
            'cand_mask': stk('cand_mask', torch.bool),
            'nbr':       stk('nbr', torch.float32),
            'nbr_mask':  stk('nbr_mask', torch.bool),
            'action':    stk('action', torch.long),
            'logp':      stk('logp', torch.float32),
            'adv':       stk('adv', torch.float32),
            'ret':       stk('ret', torch.float32),
        }
        return out

    def clear(self):
        self.tracks.clear()
        self.ready.clear()
