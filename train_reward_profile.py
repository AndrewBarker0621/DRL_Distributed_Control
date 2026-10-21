"""Single-knob reward-weight PROFILE trainer (H20 line).

Three independent profiles, one knob touched at a time (never two):

  progress  (PG): 0.125  0.25  0.5  1.0  2.0  4.0  8.0  16.0   (8 points)
  delay     (DL): 0      0.25  0.5  1.0  2.0  5.0  10.0  20.0  (8 points)
  collision (CL): 12.5   25    50   100  200                   (5 points)

Every other reward weight stays at the certified fleet defaults
(collision 50, goal 20, side 0.2, wrong_lane 0.5). All lines share the
all-default center point (PG1 = DL1 = CL50 = runNEUT). The CL line is
the risk-aversion axis: fleet-wide collision -80 failed twice (runs
28/42), but a minority style against a FROZEN fleet is a different
question — this is its first single-knob portrait.

Protocol per model — identical to runAGGR/runNEUT/runCONS (H17/H18), so
existing anchors are REUSED by copy instead of retrained:
  from scratch, 240k steps, warmup 25%, arrival curriculum 200->1400
  veh/h, rl_ratio_range=(1.0, 0.8) (pure-RL warmup, IDM metered in to
  20% after it), hero_frac 0.5 (trainable style cohort co-trains with
  the frozen final fleet), plan-sharing G7, seed 7.
PROVENANCE NOTE: the anchors were launched (2026-08-10 session) by
scratchpad scripts train_style_pair.py / train_neut.py passing exactly
these arguments, including rl_ratio_range=(1.0, 0.8). That differs from
the older H10-v2 delay pair (runH10aS/bS), which trained at CONSTANT
rl_ratio 0.8 — do not conflate the two protocols; this file is now the
committed authority for the trio's protocol. If you want to re-verify
anchor identity empirically, run with --no-reuse (retrains all 16
points, +4 x ~2 h) and diff PG1 against the copied runNEUT fingerprint.
Reused anchors: PG0.25 <- runCONS, PG1 & DL1 & CL50 <- runNEUT,
PG4 <- runAGGR. New trainings: 5 (PG) + 7 (DL) + 4 (CL) = 16 x ~1.5-2 h.

After each model, an optional fingerprint battery runs in its own
homogeneous env (80% model + 20% IDM): demands {800,1200,1400} x eval
seeds {11,22,33,44,55} x 300 s -> ONE CSV ROW PER SEED appended to
model/profile_results.csv (safe to re-run; finished rows are skipped).
Per-seed rows are deliberate: the archived H12/H17/H18 findings show
short-window outcome noise is the same order as adjacent-grid-point
effects, so the profile analysis must pool at read time and report the
seed spread as its noise floor — never trust a pooled point estimate.

Usage:
  python3 train_reward_profile.py --plan        # print the run list, no work
  python3 train_reward_profile.py               # train + test everything missing
  python3 train_reward_profile.py --only PG     # just the progress line
  python3 train_reward_profile.py --only DL,CL  # delay + collision lines
  python3 train_reward_profile.py --no-test     # training only
  python3 train_reward_profile.py --test-only   # fingerprints only (models must exist)
Interrupt-safe: finished models (model/run<TAG>_actor.pt) are skipped on
relaunch, so the sweep can be stopped and resumed freely.
"""
import argparse
import csv
import os
import random
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)
# environment.IE writes ./log/episode_reward_terms.csv without creating
# the directory; a fresh checkout/sync (which excludes log/) crashes in
# seconds without this (first NCI submission died exactly there)
os.makedirs('log', exist_ok=True)

TRAIN_SEED = 7
EVAL_SEEDS = (11, 22, 33, 44, 55)
EVAL_STEPS = 3000            # 300 s per fingerprint run
EVAL_DEMANDS = (800.0, 1200.0, 1400.0)
RESULTS_CSV = 'model/profile_results.csv'

PROFILES = {
    'PG': ('progress',  [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]),
    'DL': ('delay',     [0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]),
    'CL': ('collision', [12.5, 25.0, 50.0, 100.0, 200.0]),
}
# Training-seed replicates (H22 follow-up).  Same weights, same protocol,
# different TRAINING seed: PG1 x {8,9} gives the same-weight policy noise
# floor (3 pairs with the seed-7 original); the rest re-test the island
# suspects (DL0.5, PG2, PG16) and the two claimed effect points (CL200,
# PG0.25 = CONS).  Saved as model/run<TAG>_s<seed>_*.pt.
REPLICATES = [('PG', 1.0, 8), ('PG', 1.0, 9), ('DL', 0.5, 8), ('PG', 2.0, 8),
              ('PG', 16.0, 8), ('CL', 200.0, 8), ('PG', 0.25, 8)]
# already trained under the identical protocol+seed -> copy, don't burn compute
REUSE = {'PG0.25': 'CONS', 'PG1': 'NEUT', 'PG4': 'AGGR',
         'DL1': 'NEUT', 'CL50': 'NEUT'}


def tag_of(knob, w):
    return f"{knob}{w:g}"


def actor_path(tag):
    return f"model/run{tag}_actor.pt"


def train_one(knob, w, tag=None, seed=TRAIN_SEED):
    """Train one profile point from scratch under the H17/H18 protocol."""
    from train_MAPPO import main
    term = PROFILES[knob][0]
    tag = tag or tag_of(knob, w)
    print(f"==== TRAIN {tag} {{{term}: {w}}} seed {seed} ====", flush=True)
    main(num_steps=240_000,
         warmup_frac=0.25,
         demand_mode='arrival',
         arrival_range=(200.0, 1400.0),
         rl_ratio_range=(1.0, 0.8),
         hero_mode=True,
         hero_frac=0.5,
         hero_weights={term: w},
         fleet_from=('model/final_actor.pt', 'model/final_critic.pt'),
         plan_sharing=True,
         seed=seed,
         send_mail=False)
    shutil.copy('mappo_actor.pt', f'model/run{tag}_actor.pt')
    shutil.copy('mappo_critic.pt', f'model/run{tag}_critic.pt')
    shutil.copy('mappo_actor_loss.csv', f'model/run{tag}_actor_loss.csv')
    print(f"==== {tag} DONE -> model/run{tag}_actor.pt ====", flush=True)


def reuse_one(tag, src):
    for part in ('actor.pt', 'critic.pt', 'actor_loss.csv'):
        s = f'model/run{src}_{part}'
        if os.path.exists(s):
            shutil.copy(s, f'model/run{tag}_{part}')
    print(f"==== {tag} REUSED from run{src} ====", flush=True)


# ---------------------------------------------------------------- fingerprint
def fingerprint_run(net, demand, seed, planner_kwargs=None):
    """One homogeneous eval run (80% model + 20% IDM); returns stats.
    planner_kwargs selects the planner generation (v2 models: train_v2.py)."""
    import numpy as np
    import torch
    from rule import load_config_yaml
    from vehicle import VehicleManager
    from veh_generatort import generate_veh
    from Laser import create_multiple_rectangles
    from environment import IE
    from frenet_planner import FrenetPlanner, track_action
    from MAPPO_trainer import obs_tensors, build_neighbors

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    ic = load_config_yaml('./config/intersection_config.yaml')
    vc = load_config_yaml('./config/veh_config.yaml')
    ec = load_config_yaml('./config/env_config.yaml')
    ol, dl = np.array(ic['origin_lines']), np.array(ic['destination_lines'])
    oyw = np.array([np.pi, np.pi/2, 0, 3*np.pi/2])
    dyw = np.array([0, 3*np.pi/2, np.pi, np.pi/2])
    pz = np.concatenate([np.array(ic[k]).reshape((4, -1)) for k in
        ('prohibited_zones_center', 'prohibited_zones_length',
         'prohibited_zones_width', 'prohibited_zones_heading')], axis=1)
    m = VehicleManager(vc, ic, ec, create_multiple_rectangles(pz),
                       np.arange(0, 2*np.pi, np.pi/4), np.arange(0.1, 1.01, 0.1))
    # use_bev=False actors never read the BEV grid; skip rendering it
    # (train_MAPPO does the same — ~45% wall-clock on obs generation)
    m.ablate_bev = True

    def spawn(rl):
        s = generate_veh(ol, dl, oyw, dyw, vc['veh_length'], vc['veh_width'],
                         vc['init_vel'], m, rl_flag=rl)
        if s is None:
            return False
        m.add_vehicle(*s)
        return True

    spawn(True)
    env = IE(m, vc, ec, idm_yield_to_rl=True, idm_eta_priority=True,
             idm_gap_accept=True)
    pl = FrenetPlanner(ec, vc, **(planner_kwargs or {}))
    lo = np.array([ec['acc_lim']['min'], ec['ste_lim']['min']])
    hi = np.array([ec['acc_lim']['max'], ec['ste_lim']['max']])
    dt = ec['dt']; last = -3; backlog = 0.0
    speeds = []; g = c = t = 0
    with torch.no_grad():
        for step in range(EVAL_STEPS):
            env.current_step = step
            for v in m.vehicles.values():
                if not v.rl_flag:
                    continue
                if (getattr(v, 'plan_traj', None) is None) or (v.plan_k >= 2):
                    p = pl.generate(v, m)
                    gg, ll, tt, ww = obs_tensors(v, 'cpu')
                    nbr, nm = build_neighbors(v, m, plan_sharing=True)
                    a = int(torch.argmax(net(gg, ll, tt, ww,
                        torch.tensor(p['feats']).unsqueeze(0),
                        torch.tensor(p['mask']).unsqueeze(0),
                        torch.tensor(nbr[:, :net.nbr_in_dim]).unsqueeze(0),
                        torch.tensor(nm).unsqueeze(0)), dim=-1).item())
                    v.plan_traj = p['trajs'][a]; v.plan_k = 0
            for v in m.vehicles.values():
                if not v.rl_flag:
                    continue
                acc, ste = track_action(v.state, v.plan_traj, v.plan_k, dt,
                                        v.size[0])
                v.acc, v.ste = np.clip([acc, ste], lo, hi)
            env.step()
            for v in m.vehicles.values():
                if v.rl_flag:
                    speeds.append(v.state[3])
            rm = []
            for v in m.vehicles.values():
                if v.rl_flag:
                    v.plan_k += 1
                if v.goal or v.collision or v.outside or v.alive < 0:
                    if v.rl_flag:
                        if v.goal:
                            g += 1
                        elif v.collision:
                            c += 1
                        else:
                            t += 1
                    rm.append(v.id)
            for vid in rm:
                m.remove_vehicle(vid)
            backlog += np.random.poisson(demand * dt / 3600.0)
            if backlog >= 1 and len(m.vehicles) < 18 and step - last >= 3:
                want_rl = (sum(1 for x in m.vehicles.values() if x.rl_flag)
                           / max(len(m.vehicles), 1)) < 0.8
                if spawn(want_rl):
                    backlog -= 1; last = step
    import numpy as np
    sp = np.array(speeds)
    return sp, g, c, t


def fingerprint(tag, knob, w):
    """One CSV row PER (demand, seed) — pool at analysis time, and use
    the seed spread as the noise floor (H12/H17/H18 lesson)."""
    import numpy as np
    import torch
    from MAPPO import MAPPOActor

    torch.set_num_threads(int(os.environ.get('TORCH_THREADS', '3')))
    done = set()
    if os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV) as f:
            for row in csv.DictReader(f):
                done.add((row['tag'], row['demand'], row['seed']))
    header_needed = not os.path.exists(RESULTS_CSV)
    net = MAPPOActor(use_bev=False, plan_sharing=True)
    net.load_state_dict(torch.load(actor_path(tag), map_location='cpu'))
    net.eval()
    with open(RESULTS_CSV, 'a', newline='') as f:
        wcsv = csv.writer(f)
        if header_needed:
            wcsv.writerow(['tag', 'knob', 'weight', 'demand', 'seed', 'mean',
                           'p50', 'p90', 'stopped_pct', 'goals', 'cols', 'tos'])
        for D in EVAL_DEMANDS:
            for seed in EVAL_SEEDS:
                if (tag, f'{D:g}', str(seed)) in done:
                    print(f"FP {tag} d{D:g} s{seed}: already recorded, skip",
                          flush=True)
                    continue
                sp, g, c, t = fingerprint_run(net, D, seed)
                row = [tag, knob, f'{w:g}', f'{D:g}', seed, f'{sp.mean():.3f}',
                       f'{np.percentile(sp, 50):.2f}',
                       f'{np.percentile(sp, 90):.2f}',
                       f'{(sp < 0.5).mean()*100:.2f}', g, c, t]
                wcsv.writerow(row); f.flush()
                print(f"FP {tag} d{D:g} s{seed}: mean {row[5]} p50 {row[6]} "
                      f"p90 {row[7]} stopped {row[8]}% | {g}/{c}/{t}",
                      flush=True)


# ----------------------------------------------------------------------- main
def run_list(only=None):
    picked = set(only.split(',')) if only else set(PROFILES)
    bad = picked - set(PROFILES)
    if bad:
        sys.exit(f"--only: unknown profile(s) {sorted(bad)}; "
                 f"choose from {sorted(PROFILES)} (comma-separated)")
    jobs = []
    for knob, (_, weights) in PROFILES.items():
        if knob not in picked:
            continue
        for w in weights:
            jobs.append((knob, w, tag_of(knob, w), TRAIN_SEED))
    return jobs


def replicate_list():
    return [(knob, w, f'{tag_of(knob, w)}_s{seed}', seed)
            for knob, w, seed in REPLICATES]


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', action='store_true', help='print run list, do nothing')
    ap.add_argument('--only', help="profile line(s), comma-separated: PG, DL, CL "
                                   "(e.g. --only PG  or  --only DL,CL)")
    ap.add_argument('--no-test', action='store_true', help='skip fingerprints')
    ap.add_argument('--test-only', action='store_true', help='fingerprints only')
    ap.add_argument('--no-reuse', action='store_true',
                    help='ignore the anchor REUSE map and train every point '
                         'fresh (anchor-identity verification, +4 trainings)')
    ap.add_argument('--replicates', action='store_true',
                    help='run ONLY the training-seed replicate list '
                         '(noise floor + island/effect re-tests)')
    args = ap.parse_args()

    reuse = {} if args.no_reuse else REUSE
    jobs = replicate_list() if args.replicates else run_list(args.only)
    if args.plan:
        for knob, w, tag, seed in jobs:
            term = PROFILES[knob][0]
            state = ('trained' if os.path.exists(actor_path(tag)) else
                     f'reuse run{reuse[tag]}' if tag in reuse else 'TRAIN 240k')
            print(f"{tag:10s} {{{term}: {w:g}}} seed {seed}  ->  {state}")
        sys.exit(0)

    for knob, w, tag, seed in jobs:
        if not os.path.exists(actor_path(tag)):
            if args.test_only:
                print(f"==== {tag} missing, --test-only set: skip ====", flush=True)
                continue
            if tag in reuse:
                reuse_one(tag, reuse[tag])
            else:
                train_one(knob, w, tag=tag, seed=seed)
        else:
            print(f"==== {tag} already trained, skip ====", flush=True)
        if not args.no_test:
            fingerprint(tag, knob, w)
    print('ALL DONE', flush=True)
