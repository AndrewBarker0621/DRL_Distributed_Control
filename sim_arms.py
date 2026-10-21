"""Three-arm simulations for the planner-v2 reward grid (H24 spec section 2).

For every trained v2 model (tag, training seed) x demand x arm, one 20-min
(12000-step) run with a fixed eval seed (identical background spawn stream
across models = paired design):

  arm      RL share  RL composition                        IDM
  homo80   0.8       probe 100 %                           20 %
  mix20    0.8       probe 25 % (=20 % of all), neut 75 %  20 %
  mix60    0.8       probe 75 % (=60 % of all), neut 25 %  20 %

probe = the tested model <weights-dir>/run<TAG>_s<SEED>_actor.pt (v2 planner,
23-dim candidate tokens, argmax).  neut = the NEUT medium: every neut
vehicle is assigned at spawn one of run NEUT_s{7,8,9,10} uniformly at random
(own seeded RNG, so the global spawn stream stays byte-identical to a
behavior_probe-style loop); the seed it got is stored per vehicle.

The loop is behavior_probe.simulate (vehicle registration by object key
v.vkey because ids are recycled, IDM vehicles removed INSIDE env.step() and
flushed by alive-set diff with outcome `exit`, final-step flush guarded by
hasattr(v, 'vkey')) with all RL forwards of a step batched per network like
MAPPO_trainer.train_mappo (plans generated first, neighbour tokens built
from the pre-update plans, one forward per network).

Outputs (append + skip-completed on (tag, seed, demand, arm, eval_seed)):
  <model-dir>/traj/<TAG>_s<SEED>_<arm>_d<DEMAND>_e<EVALSEED>.npz
        rows float32 (N,9): step, key, cls_code, x, y, h, v, L, W
        outcomes int64 (M,3): key, cls_code, out_code
        neut_seed int64 (M,): NEUT seed of neut vehicles, -1 otherwise
        meta: json string (tag/seed/demand/arm/eval_seed/steps/dt/...)
        cls codes probe 0, neut 1, idm 2; out codes goal 0, collision 1,
        timeout 2, exit 3, alive 4 (as behavior_probe)
  <model-dir>/sim_vehicles.csv   one row per finished vehicle
  <model-dir>/sim_runs.csv       one row per run
  <model-dir>/sim_parts/         per-run CSV fragments written by worker
                                 processes, merged by the parent (the parent
                                 is the ONLY writer of the two CSVs)

Usage:
  python3 sim_arms.py --plan                             # list runs + state
  python3 sim_arms.py --models ALL --workers 2           # full battery
  python3 sim_arms.py --models NEUT --seeds 7 --arms mix20 --demands 1400 \
      --steps 600 --no-require-all-neut --model-dir /tmp/h24_sim/model
  python3 sim_arms.py --merge                            # merge leftover parts
Environment: TORCH_THREADS (default 1 — the per-step batches are tiny, and
several workers may run next to trainings).
"""
import argparse
import csv
import glob
import json
import os
import random
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)
# environment.IE truncates ./log/episode_reward_terms.csv on construction
os.makedirs('log', exist_ok=True)

import numpy as np           # noqa: E402

ARMS = {
    'homo80': dict(rl_ratio=0.8, probe_frac=1.00),
    'mix20':  dict(rl_ratio=0.8, probe_frac=0.25),
    'mix60':  dict(rl_ratio=0.8, probe_frac=0.75),
}
# the 13 distinct grid models (H24 section 1), NEUT first
GRID_TAGS = ['NEUT',
             'CL12.5', 'CL25', 'CL100', 'CL200',
             'PG0.125', 'PG0.25', 'PG4', 'PG16',
             'DL0', 'DL0.25', 'DL5', 'DL20']
CLS_CODE = {'probe': 0, 'neut': 1, 'idm': 2}
OUT_CODE = {'goal': 0, 'collision': 1, 'timeout': 2, 'exit': 3, 'alive': 4}
HARD_BRAKE = -3.0      # m/s^2, as behavior_probe
REPLAN_EVERY = 2
MAX_POP = 18
V2_PLANNER = dict(soft_collision=True, col_hard_horizon=0.2, safety_feats=True)
# planner generation of the probe / medium networks (--gen); 'v2' keeps the
# H24/H25 behaviour byte-identical, 'v3' = the fixed v3 planner of H26/H27
PLANNER_GEN = 'v2'
_FEAT_DIM = {}


def planner_kwargs(gen=None):
    gen = gen or PLANNER_GEN
    if gen == 'v2':
        return dict(V2_PLANNER)
    from train_v2 import GENS
    kw = dict(GENS[gen])
    if 'planner_horizon' in kw:
        kw['horizon'] = kw.pop('planner_horizon')
    return kw


def feat_dim_of(gen=None):
    gen = gen or PLANNER_GEN
    if gen not in _FEAT_DIM:
        from rule import load_config_yaml
        from frenet_planner import FrenetPlanner
        _FEAT_DIM[gen] = FrenetPlanner(load_config_yaml('./config/env_config.yaml'),
                                       load_config_yaml('./config/veh_config.yaml'),
                                       **planner_kwargs(gen)).feat_dim
    return _FEAT_DIM[gen]

VEH_HEADER = ['tag', 'seed', 'arm', 'demand', 'eval_seed', 'cls', 'neut_seed',
              'outcome', 'life_s', 'mean_speed', 'p90_speed', 'stopped_frac',
              'stop_episodes', 'hard_brakes_per_min']
RUN_HEADER = ['tag', 'seed', 'arm', 'demand', 'eval_seed', 'steps',
              'probe_goals', 'probe_cols', 'probe_tos',
              'neut_goals', 'neut_cols', 'neut_tos', 'idm_exits', 'wall_s']


# ------------------------------------------------------------------ naming
def run_name(tag, seed, arm, demand, es):
    return f'{tag}_s{seed}_{arm}_d{demand:g}_e{es}'


def run_key(tag, seed, arm, demand, es):
    return (str(tag), str(seed), str(arm), f'{float(demand):g}', str(es))


def actor_path(weights_dir, tag, seed):
    return os.path.join(weights_dir, f'run{tag}_s{seed}_actor.pt')


def paths_of(model_dir):
    return dict(traj=os.path.join(model_dir, 'traj'),
                parts=os.path.join(model_dir, 'sim_parts'),
                veh=os.path.join(model_dir, 'sim_vehicles.csv'),
                runs=os.path.join(model_dir, 'sim_runs.csv'))


# ------------------------------------------------------------------ models
def load_actor(path):
    import torch
    from MAPPO import MAPPOActor
    net = MAPPOActor(cand_feat_dim=feat_dim_of(), use_bev=False,
                     plan_sharing=True)
    net.load_state_dict(torch.load(path, map_location='cpu'))
    net.eval()
    return net


def resolve_neut_seeds(medium_dir, medium_tag, wanted):
    """Split the requested medium seeds into (available, missing).  The medium
    ("neut" in all outputs for continuity) is run<medium_tag>_s<seed> in
    medium_dir: NEUT s7-10 of model/v2 in H24/H25, the stage-2 centre models
    of model/v3 in H27."""
    have, miss = [], []
    for s in wanted:
        (have if os.path.exists(actor_path(medium_dir, medium_tag, s))
         else miss).append(s)
    return have, miss


# ------------------------------------------------------------------ simulation
def simulate(probe_net, neut_nets, demand, eval_seed, steps, rl_ratio,
             probe_frac, log_every=0):
    """One run. Returns (records, frames, dt):
    records: vkey -> (stats dict, outcome)   stats carries cls / neut_seed
    frames:  per step, list of (key, cls, x, y, h, speed, L, W)."""
    import torch
    from rule import load_config_yaml
    from vehicle import VehicleManager
    from veh_generatort import generate_veh
    from Laser import create_multiple_rectangles
    from environment import IE
    from frenet_planner import FrenetPlanner, track_action
    from MAPPO_trainer import build_neighbors, dict2arrays

    neut_seeds = sorted(neut_nets)
    if probe_frac < 1.0 and not neut_seeds:
        raise RuntimeError('arm needs neut vehicles but no NEUT weights loaded')

    random.seed(eval_seed); np.random.seed(eval_seed); torch.manual_seed(eval_seed)
    # the neut-seed draw has its OWN stream so the global spawn/class stream
    # (random + np.random) is identical to a behavior_probe-style run
    nrng = random.Random(eval_seed * 7919 + 1)
    ic = load_config_yaml('./config/intersection_config.yaml')
    vc = load_config_yaml('./config/veh_config.yaml')
    ec = load_config_yaml('./config/env_config.yaml')
    ol, dl = np.array(ic['origin_lines']), np.array(ic['destination_lines'])
    oyw = np.array([np.pi, np.pi / 2, 0, 3 * np.pi / 2])
    dyw = np.array([0, 3 * np.pi / 2, np.pi, np.pi / 2])
    pz = np.concatenate([np.array(ic[k]).reshape((4, -1)) for k in
        ('prohibited_zones_center', 'prohibited_zones_length',
         'prohibited_zones_width', 'prohibited_zones_heading')], axis=1)
    m = VehicleManager(vc, ic, ec, create_multiple_rectangles(pz),
                       np.arange(0, 2 * np.pi, np.pi / 4),
                       np.arange(0.1, 1.01, 0.1))
    m.ablate_bev = True      # use_bev=False actors never read the grid

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
    pl = FrenetPlanner(ec, vc, **planner_kwargs())
    lo = np.array([ec['acc_lim']['min'], ec['ste_lim']['min']])
    hi = np.array([ec['acc_lim']['max'], ec['ste_lim']['max']])
    dt = ec['dt']; last = -3; backlog = 0.0

    nets = {'probe': probe_net}
    for s, net in neut_nets.items():
        nets[('neut', s)] = net

    nkey = [0]
    stats = {}      # live accumulators, vkey -> dict
    records = {}    # finished: vkey -> (stats dict, outcome)
    frames = []

    def register(v):
        if not hasattr(v, 'vkey'):
            nkey[0] += 1
            v.vkey = nkey[0]
            ns = -1
            if v.rl_flag:
                if random.random() < probe_frac:
                    v.cls = 'probe'; v.net_key = 'probe'
                else:
                    v.cls = 'neut'
                    ns = nrng.choice(neut_seeds)
                    v.net_key = ('neut', ns)
            else:
                v.cls = 'idm'
            stats[v.vkey] = dict(cls=v.cls, neut_seed=ns, life=0, speeds=[],
                                 stopped=0, hard=0, stops=0, in_stop=False)

    def finalize(key, outcome):
        s = stats.pop(key, None)
        if s is not None and s['life'] > 0:
            records[key] = (s, outcome)

    def stack(arrs, dtype=torch.float32):
        return torch.from_numpy(np.stack(arrs)).to(dtype)

    t0 = time.time()
    with torch.inference_mode():
        for step in range(steps):
            env.current_step = step
            for v in m.vehicles.values():
                register(v)
            # ---- decisions: plans first, then one batched forward per net
            groups = {}
            for v in m.vehicles.values():
                if not v.rl_flag:
                    continue
                if (getattr(v, 'plan_traj', None) is None) or (v.plan_k >= REPLAN_EVERY):
                    groups.setdefault(v.net_key, []).append((v, pl.generate(v, m)))
            for net_key, batch in groups.items():
                net = nets[net_key]
                obs = [dict2arrays(v.current_obs) for v, _ in batch]
                nbrs = [build_neighbors(v, m, plan_sharing=True) for v, _ in batch]
                g_t, l_t, t_t, w_t = (stack([o[k] for o in obs]) for k in range(4))
                cand_t = stack([p['feats'] for _, p in batch])
                mask_t = stack([p['mask'] for _, p in batch], torch.bool)
                nbr_t = stack([n for n, _ in nbrs])[..., :net.nbr_in_dim]
                nm_t = stack([mk for _, mk in nbrs], torch.bool)
                acts = torch.argmax(net(g_t, l_t, t_t, w_t, cand_t, mask_t,
                                        nbr_t, nm_t), dim=-1)
                for j, (v, p) in enumerate(batch):
                    v.plan_traj = p['trajs'][int(acts[j])]
                    v.plan_k = 0
            for v in m.vehicles.values():
                if not v.rl_flag:
                    continue
                acc, ste = track_action(v.state, v.plan_traj, v.plan_k, dt,
                                        v.size[0])
                v.acc, v.ste = np.clip([acc, ste], lo, hi)
            env.step()
            env.step_record = []     # IE.record() buffers every vehicle-step

            # ---- per-vehicle accumulators + raw frame
            frame = []
            for v in m.vehicles.values():
                s = stats.get(v.vkey)
                if s is None:
                    continue
                x, y, h, sp = (float(v.state[0]), float(v.state[1]),
                               float(v.state[2]), float(v.state[3]))
                s['life'] += 1
                s['speeds'].append(sp)
                if sp < 0.5:
                    s['stopped'] += 1
                    if not s['in_stop']:
                        s['in_stop'] = True
                        s['stops'] += 1
                else:
                    s['in_stop'] = False
                if s['cls'] != 'idm' and float(v.acc) < HARD_BRAKE:
                    s['hard'] += 1
                frame.append((v.vkey, s['cls'], x, y, h, sp,
                              float(v.size[0]), float(v.size[1])))
            frames.append(frame)

            # ---- lifecycle: RL outcomes here, IDM removed inside env.step
            rm = []
            for v in m.vehicles.values():
                if v.rl_flag:
                    v.plan_k += 1
                if v.goal or v.collision or v.outside or v.alive < 0:
                    finalize(v.vkey, 'goal' if v.goal else
                             ('collision' if v.collision else 'timeout'))
                    rm.append(v.id)
            for vid in rm:
                m.remove_vehicle(vid)
            alive_keys = {v.vkey for v in m.vehicles.values()}
            for key in [k for k in stats if k not in alive_keys]:
                finalize(key, 'exit')
            # ---- Poisson arrivals, RL/IDM share rule as behavior_probe
            backlog += np.random.poisson(demand * dt / 3600.0)
            if backlog >= 1 and len(m.vehicles) < MAX_POP and step - last >= 3:
                want_rl = (sum(1 for x in m.vehicles.values() if x.rl_flag)
                           / max(len(m.vehicles), 1)) < rl_ratio
                if spawn(want_rl):
                    backlog -= 1; last = step
            if log_every and (step + 1) % log_every == 0:
                print(f'    step {step + 1}/{steps} vehs {len(m.vehicles)} '
                      f'finished {len(records)} ({time.time() - t0:.0f} s)',
                      flush=True)
        for v in list(m.vehicles.values()):
            # a vehicle spawned in the very last step was never registered
            # (register runs at the top of the NEXT step): nothing to flush
            if hasattr(v, 'vkey'):
                finalize(v.vkey, 'alive')
    return records, frames, dt


def dump_traj(path, records, frames, meta):
    """Raw-trajectory archive in the behavior_probe.dump_traj layout plus
    neut_seed and meta; written atomically (tmp + rename)."""
    n = sum(len(f) for f in frames)
    rows = np.empty((n, 9), np.float32)
    i = 0
    for step, frame in enumerate(frames):
        for (key, cls, x, y, h, sp, L, W) in frame:
            rows[i] = (step, key, CLS_CODE[cls], x, y, h, sp, L, W)
            i += 1
    keys = sorted(records)
    outs = np.array([(k, CLS_CODE[records[k][0]['cls']], OUT_CODE[records[k][1]])
                     for k in keys], np.int64).reshape(-1, 3)
    nseed = np.array([records[k][0]['neut_seed'] for k in keys], np.int64)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        np.savez_compressed(f, rows=rows, outcomes=outs, neut_seed=nseed,
                            meta=np.array(json.dumps(meta, sort_keys=True)))
    os.replace(tmp, path)


def vehicle_rows(tag, seed, arm, demand, es, records, dt):
    rows = []
    for key in sorted(records):
        s, outcome = records[key]
        sp = np.array(s['speeds'])
        rows.append([
            tag, seed, arm, f'{demand:g}', es, s['cls'], s['neut_seed'], outcome,
            f'{s["life"] * dt:.1f}',
            f'{sp.mean():.3f}', f'{np.percentile(sp, 90):.2f}',
            f'{s["stopped"] / s["life"]:.4f}', s['stops'],
            (f'{s["hard"] / (s["life"] * dt) * 60:.3f}'
             if s['cls'] != 'idm' else ''),
        ])
    return rows


def run_row(tag, seed, arm, demand, es, steps, records, wall):
    out = {'probe': [0, 0, 0], 'neut': [0, 0, 0]}
    idm_exits = 0
    for s, outcome in records.values():
        if s['cls'] in out and outcome in ('goal', 'collision', 'timeout'):
            out[s['cls']][('goal', 'collision', 'timeout').index(outcome)] += 1
        elif s['cls'] == 'idm' and outcome == 'exit':
            idm_exits += 1
    return [tag, seed, arm, f'{demand:g}', es, steps,
            *out['probe'], *out['neut'], idm_exits, f'{wall:.1f}']


def run_job(job, cfg, cache):
    """Simulate one (tag, seed, arm, demand, es); write the npz; return
    (vehicle rows, run row).  `cache` keeps loaded nets across jobs."""
    import torch
    torch.set_num_threads(int(os.environ.get('TORCH_THREADS', '1')))
    tag, seed, arm, demand, es = job
    spec = ARMS[arm]
    wpath = actor_path(cfg['weights_dir'], tag, seed)
    if wpath not in cache:
        cache[wpath] = load_actor(wpath)
    probe_net = cache[wpath]
    neut_nets = {}
    if spec['probe_frac'] < 1.0:
        for s in cfg['neut_seeds']:
            p = actor_path(cfg['medium_dir'], cfg['medium_tag'], s)
            if p not in cache:
                cache[p] = load_actor(p)
            neut_nets[s] = cache[p]
    name = run_name(tag, seed, arm, demand, es)
    print(f'SIM {name}: start (steps {cfg["steps"]}, neut seeds '
          f'{cfg["neut_seeds"] if neut_nets else "-"})', flush=True)
    t0 = time.time()
    records, frames, dt = simulate(probe_net, neut_nets, demand, es,
                                   cfg['steps'], spec['rl_ratio'],
                                   spec['probe_frac'],
                                   log_every=cfg.get('log_every', 0))
    wall = time.time() - t0
    meta = dict(tag=tag, seed=seed, demand=demand, arm=arm, eval_seed=es,
                steps=cfg['steps'], dt=dt, rl_ratio=spec['rl_ratio'],
                probe_frac=spec['probe_frac'],
                neut_seeds=(sorted(neut_nets) if neut_nets else []),
                neut_seeds_missing=cfg['neut_seeds_missing'],
                probe_weights=wpath, cls_code=CLS_CODE, out_code=OUT_CODE,
                medium_tag=cfg['medium_tag'], medium_dir=cfg['medium_dir'],
                planner=PLANNER_GEN, wall_s=round(wall, 1),
                created=time.strftime('%Y-%m-%d %H:%M:%S'))
    dump_traj(os.path.join(cfg['traj_dir'], name + '.npz'), records, frames,
              meta)
    return (vehicle_rows(tag, seed, arm, demand, es, records, dt),
            run_row(tag, seed, arm, demand, es, cfg['steps'], records, wall))


# ------------------------------------------------------------------ csv io
def append_rows(path, header, rows):
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'a', newline='') as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerows(rows)
        f.flush()


def write_fragment(path, header, rows):
    tmp = path + '.tmp'
    with open(tmp, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    os.replace(tmp, path)


def read_done(runs_csv):
    done = set()
    if os.path.exists(runs_csv):
        with open(runs_csv) as f:
            for r in csv.DictReader(f):
                done.add(run_key(r['tag'], r['seed'], r['arm'], r['demand'],
                                 r['eval_seed']))
    return done


_RUN_RE = None


def traj_done(traj_dir):
    """Keys of the runs whose npz archive exists (--no-csv mode)."""
    import re
    global _RUN_RE
    if _RUN_RE is None:
        _RUN_RE = re.compile(r'^(?P<tag>.+)_s(?P<seed>\d+)_(?P<arm>' + '|'.join(ARMS) +
                             r')_d(?P<demand>[\d.]+)_e(?P<es>\d+)\.npz$')
    done = set()
    for f in glob.glob(os.path.join(traj_dir, '*.npz')):
        mt = _RUN_RE.match(os.path.basename(f))
        if mt:
            done.add(run_key(mt['tag'], mt['seed'], mt['arm'], mt['demand'], mt['es']))
    return done


def fmt_run(r):
    return (f'SIM {r[0]}_s{r[1]} {r[2]} d{r[3]} e{r[4]}: '
            f'probe {r[6]}/{r[7]}/{r[8]} neut {r[9]}/{r[10]}/{r[11]} '
            f'idm_exits {r[12]} wall {r[13]} s')


def merge_fragments(P, done, names=None):
    """Move complete fragments (runs part present, written last) into the
    main CSVs.  Returns the run rows merged.  Parent-only."""
    merged = []
    if names is None:
        names = [os.path.basename(f)[:-len('_runs.csv')]
                 for f in sorted(glob.glob(os.path.join(P['parts'], '*_runs.csv')))]
    for name in names:
        rp = os.path.join(P['parts'], name + '_runs.csv')
        vp = os.path.join(P['parts'], name + '_vehicles.csv')
        if not (os.path.exists(rp) and os.path.exists(vp)):
            continue
        with open(rp) as f:
            rrows = list(csv.reader(f))[1:]
        with open(vp) as f:
            vrows = list(csv.reader(f))[1:]
        for r in rrows:
            k = run_key(r[0], r[1], r[2], r[3], r[4])
            if k in done:
                print(f'MERGE {name}: already in sim_runs.csv, fragment dropped',
                      flush=True)
                continue
            if not os.path.exists(os.path.join(P['traj'], name + '.npz')):
                print(f'MERGE {name}: npz missing, fragment dropped (re-run)',
                      flush=True)
                continue
            append_rows(P['veh'], VEH_HEADER, vrows)
            append_rows(P['runs'], RUN_HEADER, [r])
            done.add(k)
            merged.append(r)
        os.remove(vp); os.remove(rp)
    return merged


# ------------------------------------------------------------------ driver
def parse_list(s, conv):
    return [conv(x) for x in s.split(',') if x.strip()]


def build_jobs(args):
    tags = GRID_TAGS if args.models == 'ALL' else parse_list(args.models, str)
    seeds = parse_list(args.seeds, int)
    arms = parse_list(args.arms, str)
    for a in arms:
        if a not in ARMS:
            sys.exit(f'unknown arm {a}; choose from {sorted(ARMS)}')
    demands = parse_list(args.demands, float)
    ess = parse_list(args.eval_seeds, int)
    jobs = []
    for tag in tags:
        for seed in seeds:
            for D in demands:
                for arm in arms:
                    for es in ess:
                        jobs.append((tag, seed, arm, D, es))
    return jobs


def job_state(job, done, weights_dir, neut_ok):
    tag, seed, arm, D, es = job
    if run_key(tag, seed, arm, D, es) in done:
        return 'done'
    if not os.path.exists(actor_path(weights_dir, tag, seed)):
        return 'MISSING weights'
    if ARMS[arm]['probe_frac'] < 1.0 and not neut_ok:
        return 'MISSING neut medium'
    return 'TODO'


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--models', default='ALL',
                    help='comma list of tags (NEUT, CL200, ...) or ALL (13-tag grid)')
    ap.add_argument('--seeds', default='7,8,9,10', help='TRAINING seeds')
    ap.add_argument('--arms', default='homo80,mix20,mix60')
    ap.add_argument('--demands', default='800,1200,1400')
    ap.add_argument('--steps', type=int, default=12000)
    ap.add_argument('--minutes', type=float, default=None,
                    help='run length in simulated minutes (overrides --steps; dt from env_config)')
    ap.add_argument('--gen', default='v2',
                    help='planner generation of the networks: v2 (H24/H25) or v3 (H27, fixed planner)')
    ap.add_argument('--eval-seeds', '--eval-seed', dest='eval_seeds', default='11',
                    help='eval seed(s), comma list (default 11)')
    ap.add_argument('--workers', type=int, default=1,
                    help='N>1: N worker processes, each taking the next undone '
                         'run and writing CSV fragments the parent merges')
    ap.add_argument('--model-dir', default='model',
                    help='outputs: <model-dir>/traj, sim_vehicles.csv, sim_runs.csv')
    ap.add_argument('--weights-dir', default='model/v2',
                    help='where run<TAG>_s<SEED>_actor.pt live (the REAL repo '
                         'model/v2 even when --model-dir is a scratch dir)')
    ap.add_argument('--neut-seeds', '--medium-seeds', dest='neut_seeds', default='7,8,9,10',
                    help='training seeds of the medium models (NEUT s7-10 in H24/H25; '
                         'the centre seeds in H27)')
    ap.add_argument('--medium-tag', default='NEUT',
                    help='tag of the medium models (H27: the centre tag, e.g. CL25)')
    ap.add_argument('--medium-dir', default=None,
                    help='where run<medium-tag>_s<seed>_actor.pt live (default --weights-dir; '
                         'H27: model/v3)')
    ap.add_argument('--no-csv', action='store_true',
                    help='do not touch sim_vehicles.csv / sim_runs.csv: workers write npz + '
                         'fragments only (several PBS jobs may share one --model-dir); a run '
                         'counts as done when its npz exists; merge later with --merge')
    ap.add_argument('--require-all-neut', dest='require_all_neut',
                    action='store_true', default=True,
                    help='abort if any NEUT medium seed is missing (default)')
    ap.add_argument('--no-require-all-neut', dest='require_all_neut',
                    action='store_false',
                    help='drop missing NEUT seeds with a warning (smoke tests)')
    ap.add_argument('--plan', action='store_true', help='list runs and state')
    ap.add_argument('--merge', action='store_true',
                    help='only merge leftover sim_parts fragments, then exit')
    ap.add_argument('--log-every', type=int, default=0,
                    help='print an in-run progress line every N steps')
    # worker mode (internal): --job TAG:SEED:ARM:DEMAND:EVALSEED runs exactly
    # one job and writes fragments; --neut-missing carries the parent's list
    # of dropped NEUT seeds into the run meta
    ap.add_argument('--job', default=None, help=argparse.SUPPRESS)
    ap.add_argument('--neut-missing', default='', help=argparse.SUPPRESS)
    args = ap.parse_args()

    global PLANNER_GEN
    PLANNER_GEN = args.gen
    if args.minutes is not None:
        from rule import load_config_yaml
        args.steps = int(round(args.minutes * 60.0 / load_config_yaml('./config/env_config.yaml')['dt']))
    medium_dir = args.medium_dir or args.weights_dir
    P = paths_of(args.model_dir)
    neut_have, neut_miss = resolve_neut_seeds(medium_dir, args.medium_tag,
                                              parse_list(args.neut_seeds, int))
    cfg = dict(weights_dir=args.weights_dir, traj_dir=P['traj'], steps=args.steps,
               neut_seeds=neut_have, medium_tag=args.medium_tag, medium_dir=medium_dir,
               neut_seeds_missing=sorted(set(neut_miss)
                                         | set(parse_list(args.neut_missing, int))),
               log_every=args.log_every, no_csv=args.no_csv)

    # ---- worker mode: one job -> npz + fragments, no shared CSV writes
    if args.job:
        tag, seed, arm, D, es = args.job.split(':')
        job = (tag, int(seed), arm, float(D), int(es))
        name = run_name(*job)
        os.makedirs(P['parts'], exist_ok=True)
        vrows, rrow = run_job(job, cfg, {})
        write_fragment(os.path.join(P['parts'], name + '_vehicles.csv'),
                       VEH_HEADER, vrows)
        write_fragment(os.path.join(P['parts'], name + '_runs.csv'),
                       RUN_HEADER, [rrow])
        print(fmt_run(rrow), flush=True)
        return

    if neut_miss:
        print(f'WARNING: medium {args.medium_tag} seeds {neut_miss} missing in '
              f'{medium_dir} (have {neut_have})', flush=True)
    jobs = build_jobs(args)
    done = read_done(P['runs'])
    if args.no_csv:
        done |= traj_done(P['traj'])
    print(f'planner generation {PLANNER_GEN} (cand feat dim {feat_dim_of()}), run length '
          f'{args.steps} steps, medium {args.medium_tag} s{neut_have} from {medium_dir}', flush=True)
    neut_ok = bool(neut_have) and not (neut_miss and args.require_all_neut)
    states = {j: job_state(j, done, args.weights_dir, neut_ok) for j in jobs}

    if args.plan:
        for j in jobs:
            print(f'{run_name(*j):40s} {states[j]}')
        n = len(jobs)
        print(f'{n} runs: {sum(s == "done" for s in states.values())} done, '
              f'{sum(s == "TODO" for s in states.values())} todo, '
              f'{sum(s.startswith("MISSING") for s in states.values())} missing',
              flush=True)
        return

    if (not args.merge and neut_miss and args.require_all_neut
            and any(ARMS[j[2]]['probe_frac'] < 1.0 for j in jobs)):
        sys.exit(f'ABORT: medium {args.medium_tag} seeds {neut_miss} missing and '
                 f'--require-all-neut is on (pass --no-require-all-neut for '
                 f'a smoke test)')
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(P['traj'], exist_ok=True)
    os.makedirs(P['parts'], exist_ok=True)
    for r in merge_fragments(P, done):
        print('MERGED leftover ' + fmt_run(r), flush=True)
    if args.merge:
        print('merge done', flush=True)
        return

    todo = []
    for j in jobs:
        st = states[j]
        if st == 'done':
            print(f'SIM {run_name(*j)}: recorded, skip', flush=True)
        elif st != 'TODO':
            print(f'SKIP {run_name(*j)}: {st}', flush=True)
        else:
            todo.append(j)
    total = len(todo)
    print(f'{total} runs to simulate ({len(jobs) - total} skipped), '
          f'workers {args.workers}', flush=True)
    if not todo:
        print('ALL DONE', flush=True)
        return

    t_start = time.time()
    walls = []

    def progress(n_done, wall):
        walls.append(wall)
        remain = total - n_done
        est = remain * (sum(walls) / len(walls)) / max(args.workers, 1)
        print(f'PROGRESS {n_done}/{total} done | last {wall:.0f} s | elapsed '
              f'{(time.time() - t_start) / 60:.1f} min | est. remaining '
              f'{est / 60:.1f} min', flush=True)

    if args.workers <= 1:
        cache = {}
        for i, j in enumerate(todo):
            t0 = time.time()
            vrows, rrow = run_job(j, cfg, cache)
            if args.no_csv:
                name = run_name(*j)
                write_fragment(os.path.join(P['parts'], name + '_vehicles.csv'), VEH_HEADER, vrows)
                write_fragment(os.path.join(P['parts'], name + '_runs.csv'), RUN_HEADER, [rrow])
            else:
                append_rows(P['veh'], VEH_HEADER, vrows)
                append_rows(P['runs'], RUN_HEADER, [rrow])
            done.add(run_key(*j))
            print(fmt_run(rrow), flush=True)
            progress(i + 1, time.time() - t0)
        print('ALL DONE', flush=True)
        return

    # ---- parallel: N worker processes, parent merges fragments; each
    # worker's stdout goes to <model-dir>/sim_logs/<run>.log
    log_dir = os.path.join(args.model_dir, 'sim_logs')
    os.makedirs(log_dir, exist_ok=True)
    base = [sys.executable, '-u', os.path.abspath(__file__),
            '--model-dir', args.model_dir, '--weights-dir', args.weights_dir,
            '--steps', str(args.steps), '--no-require-all-neut', '--gen', PLANNER_GEN,
            '--medium-tag', args.medium_tag, '--medium-dir', medium_dir,
            '--neut-seeds', ','.join(str(s) for s in neut_have),
            '--neut-missing', ','.join(str(s) for s in neut_miss),
            '--log-every', str(args.log_every)]
    queue = list(todo)
    running = {}      # Popen -> (job, t0, log path)
    n_done = 0
    try:
        while queue or running:
            while queue and len(running) < args.workers:
                j = queue.pop(0)
                name = run_name(*j)
                logp = os.path.join(log_dir, name + '.log')
                lf = open(logp, 'w')
                pr = subprocess.Popen(base + ['--job', ':'.join(
                    [j[0], str(j[1]), j[2], f'{j[3]:g}', str(j[4])])],
                    stdout=lf, stderr=subprocess.STDOUT, cwd=HERE)
                pr._logfile = lf
                running[pr] = (j, time.time(), logp)
                print(f'LAUNCH {name} (pid {pr.pid}, log {logp})', flush=True)
            time.sleep(1.0)
            for pr in [p for p in running if p.poll() is not None]:
                j, t0, logp = running.pop(pr)
                pr._logfile.close()
                name = run_name(*j)
                if pr.returncode != 0:
                    print(f'FAILED {name}: exit {pr.returncode}, see {logp}',
                          flush=True)
                    continue
                if args.no_csv:
                    # fragments stay for a later --merge; done = npz on disk
                    if not os.path.exists(os.path.join(P['traj'], name + '.npz')):
                        print(f'FAILED {name}: no npz written, see {logp}', flush=True)
                        continue
                    n_done += 1
                    print(f'SIM {name}: npz written (fragments kept, merge with --merge)', flush=True)
                    progress(n_done, time.time() - t0)
                    continue
                merged = merge_fragments(P, done, names=[name])
                if not merged:
                    print(f'FAILED {name}: no fragment merged, see {logp}',
                          flush=True)
                    continue
                n_done += 1
                print(fmt_run(merged[0]), flush=True)
                progress(n_done, time.time() - t0)
    except KeyboardInterrupt:
        print('interrupted: terminating workers', flush=True)
        for pr in running:
            pr.terminate()
        for pr in running:
            try:
                pr.wait(timeout=10)
            except Exception:
                pr.kill()
            pr._logfile.close()
        raise
    print('ALL DONE' if n_done == total else
          f'DONE with {total - n_done} failed runs', flush=True)


if __name__ == '__main__':
    main()
