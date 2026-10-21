"""Planner-generation v2 trainer (H23, 2026-10).

v2 = FrenetPlanner(soft_collision=True, col_hard_horizon=0.2,
                   safety_feats=True)  ->  23-dim candidate tokens.
Mask-3 (predicted collision) no longer deletes candidates except conflicts
inside the first replan window; every candidate instead carries predicted
min clearance / time-to-conflict / relative heading of the conflicting
neighbour, and the collision reward prices the choice.

Training protocol is IDENTICAL to train_reward_profile.train_one (the
H17-H22 protocol): from scratch, 240k steps, warmup 25%, Poisson arrival
200->1400 veh/h, rl_ratio_range (1.0, 0.8), hero_frac 0.5 against the
frozen G7 `final` fleet, plan-sharing G7.  DUAL PLANNER: the fleet keeps
the v1 planner (its candidate set, hard mask and 20-dim features are
byte-identical to every earlier run); only the trainable cohort sees v2.
The single experimental variable vs the v1 profile models is therefore
the trainable vehicle's expression space.

Outputs: model/v2/run<TAG>_s<SEED>_{actor,critic}.pt, _actor_loss.csv and
a .json provenance record; fingerprints -> model/v2/profile_results.csv
(same homogeneous 80/20 battery as the v1 profiles, v2 planner for all
RL vehicles, one row per demand x eval seed).

  python3 train_v2.py --tag NEUT --seed 7
  python3 train_v2.py --tag CL200 --seed 7 --weights '{"collision": 200}'
  python3 train_v2.py --tag PG0.25 --seed 8 --weights '{"progress": 0.25}'
  python3 train_v2.py --tag NEUT --seed 7 --test-only       # battery only
  python3 train_v2.py --tag SMOKE --seed 7 --steps 2000 --no-test
"""
import argparse
import csv
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)
os.makedirs('log', exist_ok=True)

OUT_DIR = 'model/v2'
RESULTS_CSV = f'{OUT_DIR}/profile_results.csv'
V2_PLANNER = dict(soft_collision=True, col_hard_horizon=0.2, safety_feats=True)
# v3 pilot (H26, 2026-10-17, models archived in model/v3pilot/): v2 +
# scene-relative follow/yield/go/pass candidates, 4 s horizon.  Three
# planner defects found in the pilot (RESULT.md H26): mis-timed yield/go,
# 4 s profiles cutting per-step control authority, violent pass manoeuvres.
V3PILOT_PLANNER = dict(soft_collision=True, col_hard_horizon=0.2, safety_feats=True,
                       gen_v3=True, planner_horizon=4.0)
# v3 (H26 fixed, 2026-10-18): same extras / K 49 / 28 feats, arrival-time
# yield/go solve + feasibility, pass bounded by 2 m/s^2, manoeuvre profiles
# complete in 3 s (v1/v2 control authority) while the collision layer and
# safety features look 4 s ahead
V3_PLANNER = dict(V3PILOT_PLANNER, profile_horizon=3.0, v3_fix=True)
GENS = {'v2': V2_PLANNER, 'v3': V3_PLANNER, 'v3pilot': V3PILOT_PLANNER}


def set_generation(gen):
    """Select the planner generation: output dir model/<gen>/, planner kwargs."""
    global OUT_DIR, RESULTS_CSV
    OUT_DIR = f'model/{gen}'
    RESULTS_CSV = f'{OUT_DIR}/profile_results.csv'
    return GENS[gen]
# battery grid; FP_EVAL_SEEDS="66,77,88,99" / FP_EVAL_DEMANDS="1400" override
# (used by nci/eval_v3.pbs to add evaluation seeds to already-trained models;
# rows already in profile_results.csv are skipped, so re-runs only add)
EVAL_SEEDS = tuple(int(x) for x in os.environ.get('FP_EVAL_SEEDS', '11,22,33,44,55').split(','))
EVAL_STEPS = 3000
EVAL_DEMANDS = tuple(float(x) for x in os.environ.get('FP_EVAL_DEMANDS', '800,1200,1400').split(','))
PROTOCOL = dict(num_steps=240_000, warmup_frac=0.25, demand_mode='arrival',
                arrival_range=(200.0, 1400.0), rl_ratio_range=(1.0, 0.8),
                hero_mode=True, hero_frac=0.5,
                fleet_from=('model/final_actor.pt', 'model/final_critic.pt'),
                plan_sharing=True, send_mail=False)


KNOB_TERM = {'PG': 'progress', 'DL': 'delay', 'CL': 'collision'}


def weights_from_tag(tag):
    """NEUT -> {}; PG0.25 -> {'progress': 0.25}; CL200 -> {'collision': 200.0};
    same tag grammar as train_reward_profile.tag_of, so a tag can never
    disagree with its weights (NCI jobs pass only TAG/SEED)."""
    import re
    if tag == 'NEUT':
        return {}
    mt = re.fullmatch(r'(PG|DL|CL)([0-9.]+)', tag)
    if not mt:
        raise SystemExit(f"--weights required: tag '{tag}' is not NEUT or <PG|DL|CL><value>")
    return {KNOB_TERM[mt.group(1)]: float(mt.group(2))}


def paths(tag, seed):
    base = f'{OUT_DIR}/run{tag}_s{seed}'
    return (f'{base}_actor.pt', f'{base}_critic.pt', f'{base}_actor_loss.csv',
            f'{base}.json')


def train_one(tag, seed, weights, steps=None, planner=None):
    from train_MAPPO import main
    planner = dict(V2_PLANNER if planner is None else planner)
    os.makedirs(OUT_DIR, exist_ok=True)
    kw = dict(PROTOCOL)
    if steps:
        kw['num_steps'] = steps
    print(f"==== TRAIN {os.path.basename(OUT_DIR)} {tag} weights={weights} seed {seed} "
          f"({kw['num_steps']} steps) planner={planner} ====", flush=True)
    t0 = time.time()
    main(hero_weights=dict(weights), seed=seed, **kw, **planner)
    a, c, l, j = paths(tag, seed)
    shutil.copy('mappo_actor.pt', a)
    shutil.copy('mappo_critic.pt', c)
    shutil.copy('mappo_actor_loss.csv', l)
    # training-phase per-episode reward terms (hero + fleet rows): the
    # fingerprint battery below re-creates IE and truncates this file, so
    # keep the training copy next to the weights (H23 used it for the
    # hero/fleet collision curves)
    if os.path.exists('log/episode_reward_terms.csv'):
        shutil.copy('log/episode_reward_terms.csv', f'{OUT_DIR}/run{tag}_s{seed}_episodes.csv')
    with open(j, 'w') as f:
        json.dump({'tag': tag, 'seed': seed, 'weights': weights,
                   'planner': planner, 'generation': os.path.basename(OUT_DIR),
                   'protocol': {k: v for k, v in kw.items()},
                   'wall_s': round(time.time() - t0)},
                  f, indent=1)
    print(f"==== {os.path.basename(OUT_DIR)} {tag}_s{seed} DONE -> {a} ({(time.time() - t0) / 60:.0f} min) ====",
          flush=True)


def fingerprint(tag, seed, weights, planner=None):
    import numpy as np
    import torch
    from MAPPO import MAPPOActor
    from frenet_planner import FrenetPlanner
    from train_reward_profile import fingerprint_run

    planner = dict(V2_PLANNER if planner is None else planner)
    # fingerprint_run passes planner kwargs straight to FrenetPlanner, which
    # takes `horizon`, not `planner_horizon`
    pkw = {('horizon' if k == 'planner_horizon' else k): v for k, v in planner.items()}
    from rule import load_config_yaml
    feat_dim = FrenetPlanner(load_config_yaml('./config/env_config.yaml'),
                             load_config_yaml('./config/veh_config.yaml'), **pkw).feat_dim
    torch.set_num_threads(int(os.environ.get('TORCH_THREADS', '3')))
    a, _, _, _ = paths(tag, seed)
    done = set()
    if os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV) as f:
            for row in csv.DictReader(f):
                done.add((row['tag'], row['seed'], row['demand'], row['eval_seed']))
    header_needed = not os.path.exists(RESULTS_CSV)
    net = MAPPOActor(cand_feat_dim=feat_dim, use_bev=False, plan_sharing=True)
    net.load_state_dict(torch.load(a, map_location='cpu'))
    net.eval()
    with open(RESULTS_CSV, 'a', newline='') as f:
        w = csv.writer(f)
        if header_needed:
            w.writerow(['tag', 'seed', 'weights', 'demand', 'eval_seed', 'mean',
                        'p50', 'p90', 'stopped_pct', 'goals', 'cols', 'tos'])
        for D in EVAL_DEMANDS:
            for es in EVAL_SEEDS:
                if (tag, str(seed), f'{D:g}', str(es)) in done:
                    print(f"FP v2 {tag}_s{seed} d{D:g} s{es}: already recorded, skip", flush=True)
                    continue
                sp, g, c, t = fingerprint_run(net, D, es, planner_kwargs=pkw)
                row = [tag, seed, json.dumps(weights, sort_keys=True), f'{D:g}', es,
                       f'{sp.mean():.3f}', f'{np.percentile(sp, 50):.2f}',
                       f'{np.percentile(sp, 90):.2f}', f'{(sp < 0.5).mean() * 100:.2f}',
                       g, c, t]
                w.writerow(row); f.flush()
                print(f"FP v2 {tag}_s{seed} d{D:g} s{es}: mean {row[5]} p50 {row[6]} "
                      f"p90 {row[7]} stopped {row[8]}% | {g}/{c}/{t}", flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True, help='model tag, e.g. NEUT, CL200, PG0.25')
    ap.add_argument('--seed', type=int, default=7, help='TRAINING seed')
    ap.add_argument('--weights', default=None,
                    help='JSON partial reward-weight overrides, e.g. \'{"collision": 200}\'; '
                         'default: derived from --tag (NEUT, PG<x>, DL<x>, CL<x>)')
    ap.add_argument('--steps', type=int, default=None, help='override 240k (smoke tests)')
    ap.add_argument('--no-test', action='store_true', help='skip the fingerprint battery')
    ap.add_argument('--test-only', action='store_true', help='battery only (model must exist)')
    ap.add_argument('--gen', choices=sorted(GENS), default='v2',
                    help='planner generation: v2 (soft mask + safety feats), v3 (+ follow/yield/go/pass, '
                         '3 s manoeuvres / 4 s look-ahead, fixed yield/go/pass), v3pilot (the 2026-10-17 '
                         'pilot semantics, model/v3pilot/*)')
    args = ap.parse_args()
    planner = set_generation(args.gen)
    weights = json.loads(args.weights) if args.weights is not None else weights_from_tag(args.tag)
    print(f"gen {args.gen} tag {args.tag} seed {args.seed} -> weights {weights} | out {OUT_DIR}", flush=True)
    actor_pt = paths(args.tag, args.seed)[0]
    if os.path.exists(actor_pt) and not args.test_only:
        print(f"==== {actor_pt} exists, skip training ====", flush=True)
    elif not args.test_only:
        train_one(args.tag, args.seed, weights, steps=args.steps, planner=planner)
    if not args.no_test:
        fingerprint(args.tag, args.seed, weights, planner=planner)
    print('ALL DONE', flush=True)
