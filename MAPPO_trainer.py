import copy
import math
import random

import numpy as np
import torch
from torch.distributions import Categorical

from MAPPO import (RolloutBuffer, NBR_FEAT_DIM, ACTOR_NBR_FEAT_DIM,
                   NBR_PLAN_DIM, PLAN_SHARE_TS, NBR_PRIO_DIM)
from IDM import eta_priority_of
from frenet_planner import (track_action, dist_to_conflict_box, approach_speed,
                            predict_plan_points,
                            build_reference_path)
from veh_generatort import generate_veh

MAX_NEIGHBORS = 15
SENSE_RADIUS = 60.0   # vehicles beyond this range are not observable as tokens


def dict2arrays(obs_dict):
    return (obs_dict["rl_bev"], obs_dict["rl_lasers"],
            obs_dict["rl_des_info"], obs_dict["rl_waypoints"])


def build_neighbors(v, manager, max_nb=MAX_NEIGHBORS, plan_sharing=False,
                    priority_broadcast=False):
    """
    Nearest-vehicle features in the ego frame, zero-padded to a fixed slot
    count. The first ACTOR_NBR_FEAT_DIM features are locally observable
    (position, motion, distance/approach toward the conflict box — enough to
    infer arrival order, i.e. implicit right-of-way) and are shared with the
    actor; the trailing rl_flag is global info for the centralized critic only.

    plan_sharing (G7, optional): inserts NBR_PLAN_DIM dims per neighbour —
    its committed-plan positions at PLAN_SHARE_TS horizons, in the ego frame
    (candidate-level V2V; predict_plan_points has the source rules). Layout
    [11 shared | 4 plan | rl_flag]: plan dims are actor-visible, rl_flag
    stays last and critic-only in both layouts.
    """
    x, y, th = v.state[0], v.state[1], v.state[2]
    r2 = SENSE_RADIUS ** 2
    others = [o for o in manager.vehicles.values()
              if o.id != v.id
              and (o.state[0] - x) ** 2 + (o.state[1] - y) ** 2 <= r2]
    others.sort(key=lambda o: (o.state[0] - x) ** 2 + (o.state[1] - y) ** 2)
    others = others[:max_nb]

    width = (NBR_FEAT_DIM + (NBR_PLAN_DIM if plan_sharing else 0)
             + (NBR_PRIO_DIM if priority_broadcast else 0))
    feats = np.zeros((max_nb, width), dtype=np.float32)
    if priority_broadcast:
        eta_e = eta_priority_of(v, manager)
        eta_e = 15.0 if eta_e is None else eta_e   # past conflict: no claim
    mask = np.zeros(max_nb, dtype=bool)
    c, s = math.cos(-th), math.sin(-th)
    for i, o in enumerate(others):
        dx, dy = o.state[0] - x, o.state[1] - y
        rel_h = o.state[2] - th
        row = [(dx * c - dy * s) / 70.0,
               (dx * s + dy * c) / 70.0,
               math.sin(rel_h), math.cos(rel_h),
               o.state[3] / 16.0,
               dist_to_conflict_box(o.state[0], o.state[1]) / 70.0,
               approach_speed(o.state) / 16.0,
               float(np.clip(getattr(o, 'accel_est', 0.0) / 8.0, -1.5, 1.5)),
               1.0 if o.maneuver == 0 else 0.0,
               1.0 if o.maneuver == -1 else 0.0,
               1.0 if o.maneuver == 1 else 0.0]
        if plan_sharing:
            for px, py in predict_plan_points(o, PLAN_SHARE_TS):
                pdx, pdy = px - x, py - y
                row += [(pdx * c - pdy * s) / 70.0, (pdx * s + pdy * c) / 70.0]
        if priority_broadcast:
            eta_o = eta_priority_of(o, manager)
            eta_o = 15.0 if eta_o is None else eta_o
            row += [float(np.clip((eta_o - eta_e) / 10.0, -1.5, 1.5)),
                    1.0 if eta_o < eta_e else 0.0]
        row.append(1.0 if o.rl_flag else 0.0)
        feats[i] = row
        mask[i] = True
    return feats, mask


def obs_tensors(v, device):
    g, l, ta, w = dict2arrays(v.current_obs)
    return (torch.tensor(g, dtype=torch.float32, device=device).unsqueeze(0),
            torch.tensor(l, dtype=torch.float32, device=device).unsqueeze(0),
            torch.tensor(ta, dtype=torch.float32, device=device).unsqueeze(0),
            torch.tensor(w, dtype=torch.float32, device=device).unsqueeze(0))


def _stack(arrs, device, dtype=torch.float32):
    return torch.from_numpy(np.stack(arrs)).to(device=device, dtype=dtype)


@torch.inference_mode()
def values_of(vehicles, manager, critic, device):
    """Batched critic values for a list of vehicles (rollout bootstrap)."""
    obs = [dict2arrays(v.current_obs) for v in vehicles]
    nbrs = [build_neighbors(v, manager,
                            plan_sharing=getattr(critic, 'plan_sharing', False),
                            priority_broadcast=getattr(critic, 'priority_broadcast', False))
            for v in vehicles]
    g_t, l_t, t_t, w_t = (_stack([o[k] for o in obs], device) for k in range(4))
    nbr_t = _stack([n for n, _ in nbrs], device)
    nm_t = _stack([m for _, m in nbrs], device, torch.bool)
    return critic(g_t, l_t, t_t, w_t, nbr_t, nm_t).squeeze(-1).cpu().tolist()


def ppo_update(actor, critic, actor_opt, critic_opt, data, device,
               epochs=4, minibatch=64, clip=0.2, ent_coef=0.01, vf_coef=0.5,
               max_grad_norm=0.5):
    n = data['action'].shape[0]
    adv = data['adv']
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    actor_losses, critic_losses = [], []
    for _ in range(epochs):
        # CPU randperm: reproducible from the seeded CPU RNG and safe on MPS
        idx = torch.randperm(n).to(device)
        for start in range(0, n, minibatch):
            mb = idx[start:start + minibatch]
            logits = actor(data['grid'][mb], data['laser'][mb], data['target'][mb],
                           data['wps'][mb], data['cand'][mb], data['cand_mask'][mb],
                           data['nbr'][mb][..., :getattr(actor, 'nbr_in_dim', ACTOR_NBR_FEAT_DIM)],
                           data['nbr_mask'][mb])
            dist = Categorical(logits=logits)
            logp = dist.log_prob(data['action'][mb])
            ratio = torch.exp(logp - data['logp'][mb])
            surr1 = ratio * adv[mb]
            surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv[mb]
            pg_loss = -torch.min(surr1, surr2).mean()
            entropy = dist.entropy().mean()

            value = critic(data['grid'][mb], data['laser'][mb], data['target'][mb],
                           data['wps'][mb], data['nbr'][mb], data['nbr_mask'][mb]).squeeze(-1)
            v_loss = ((value - data['ret'][mb]) ** 2).mean()

            loss = pg_loss - ent_coef * entropy + vf_coef * v_loss
            actor_opt.zero_grad()
            critic_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
            actor_opt.step()
            critic_opt.step()

            actor_losses.append(pg_loss.detach())
            critic_losses.append(v_loss.detach())
    return (torch.stack(actor_losses).cpu().tolist(),
            torch.stack(critic_losses).cpu().tolist())


def train_mappo(env, planner, intersection_config, veh_config, env_config,
                actor, critic, actor_optimizer, critic_optimizer,
                num_steps=40_000,
                replan_every=2,
                rollout_size=512,
                ppo_epochs=4,
                minibatch=64,
                gamma=0.99,
                lam=0.95,
                clip=0.2,
                ent_coef=0.01,
                vf_coef=0.5,
                device="cpu",
                vehs_range=(4, 14),
                rl_ratio_range=(0.25, 0.6),
                spawn_cooldown=3,
                warmup_frac=0.0,
                checkpoint_every=25,
                demand_mode='maintain',
                arrival_range=(200.0, 800.0),
                commit_weight=0.0,          # plan-commitment shaping (run 43):
                                            # pre-weighted penalty per voluntary
                                            # plan flip, charged via v.commit_pen
                max_pop=18,
                reset_state_prob=0.0,
                hero_mode=False,            # heterogeneous training: ONE hero
                                            # vehicle learns with its own
                                            # reward_weights while the rest of
                                            # the RL fleet runs FROZEN via
                                            # fleet_actor (argmax); the buffer
                                            # collects hero decisions only
                fleet_actor=None,
                hero_weights=None,
                hero_count=1,               # simultaneous heroes on the map
                                            # (all share the trainable policy)
                hero_frac=None,             # alternative to hero_count: each
                                            # fresh RL spawn becomes a hero
                                            # with this probability (fleet-
                                            # share semantics, e.g. 0.1)
                fleet_planner=None):        # v2 dual planner (H23): the frozen
                                            # fleet plans with THIS planner
                                            # (its own generation, so the
                                            # medium stays byte-identical);
                                            # heroes use `planner`. None =
                                            # everyone uses `planner`
    """
    MAPPO training loop with parameter sharing over all RL vehicles (CTDE).
    Every `replan_every` env steps each RL vehicle regenerates its Frenet
    candidates and picks one; between decisions the selected trajectory is
    tracked by the pure-pursuit controller. Rewards inside a window are
    accumulated with per-substep discounting; GAE then uses gamma**substeps.

    Curriculum: for the first `warmup_frac` of training the population is
    held at vehs_range[0] vehicles with rl_ratio_range[0] RL share (e.g.
    (2, 1.0) = two RL vehicles alone learning to drive the route); after the
    warmup both quantities ramp linearly to vehs_range[1] / rl_ratio_range[1]
    — with a starting share of 1.0 this meters IDM traffic in gradually.
    Each step, if the population is below the current target, one vehicle is
    spawned (at most one per `spawn_cooldown` steps); it is RL-controlled
    when the current RL share is below the scheduled ratio.

    Every `checkpoint_every` PPO updates the weights are saved to
    model/mappo_{actor,critic}_ckpt.pt (rolling overwrite).
    """
    origin_lines = np.array(intersection_config['origin_lines'])
    destination_lines = np.array(intersection_config['destination_lines'])
    origin_yaws = np.array([np.pi, np.pi / 2, 0, 3 * np.pi / 2])
    destination_yaws = np.array([0, 3 * np.pi / 2, np.pi, np.pi / 2])
    veh_length = veh_config['veh_length']
    veh_width = veh_config['veh_width']
    init_vel = veh_config['init_vel']

    min_action = np.array([env_config['acc_lim']['min'], env_config['ste_lim']['min']])
    max_action = np.array([env_config['acc_lim']['max'], env_config['ste_lim']['max']])
    dt = env_config['dt']

    actor.to(device)
    critic.to(device)

    # On Apple MPS the big-batch PPO update is ~6x faster than CPU, but
    # small-batch rollout forwards drown in dispatch/sync overhead. Hybrid:
    # canonical models (and Adam state) stay on MPS for updates; rollout
    # inference runs on CPU mirrors refreshed after every update, so outside
    # ppo_update the mirrors are always identical to the canonical models.
    use_mirror = str(device).startswith("mps")
    if use_mirror:
        actor_inf = copy.deepcopy(actor).to("cpu")
        critic_inf = copy.deepcopy(critic).to("cpu")
        inf_device = "cpu"
    else:
        actor_inf, critic_inf = actor, critic
        inf_device = device

    buffer = RolloutBuffer(gamma=gamma, lam=lam)
    pending = {}          # veh_id -> open transition (current replan window)
    actor_loss_history, critic_loss_history = [], []
    reward_history = []

    def close_pending(vid, done):
        tr = pending.pop(vid, None)
        if tr is not None:
            tr['done'] = done
            buffer.add(vid, tr)

    total_steps = 0
    n_updates = 0
    last_spawn_step = -spawn_cooldown
    spawn_backlog = 0.0   # arrival-demand mode: arrivals waiting to enter

    def _ensure_hero():
        # up to hero_count heroes on the map (all driven by the trainable
        # policy, parameter-shared); tag only FRESH spawns (first 0.5 s of
        # life) so hero episodes start clean at the map edge.
        # hero_frac mode instead rolls each fresh RL spawn ONCE with
        # probability hero_frac (fleet-share semantics).
        alive0 = env.agents_manager.alive
        if hero_frac is not None:
            for vv in env.agents_manager.vehicles.values():
                if (vv.rl_flag and not getattr(vv, 'hero_rolled', False)
                        and vv.alive >= alive0 - 5):
                    vv.hero_rolled = True
                    if random.random() < hero_frac:
                        vv.hero = True
                        vv.reward_weights = dict(hero_weights or {})
            return
        n_heroes = sum(1 for vv in env.agents_manager.vehicles.values()
                       if getattr(vv, 'hero', False))
        if n_heroes >= hero_count:
            return
        for vv in env.agents_manager.vehicles.values():
            if n_heroes >= hero_count:
                break
            if vv.rl_flag and not getattr(vv, 'hero', False) and vv.alive >= alive0 - 5:
                vv.hero = True
                vv.reward_weights = dict(hero_weights or {})
                n_heroes += 1

    hero_stats = {'goal': 0, 'collision': 0, 'timeout': 0}

    while total_steps < num_steps:
        if hero_mode:
            _ensure_hero()
        # 1) decision + low-level control for every RL vehicle.
        # Planning (numpy) stays per vehicle; the actor/critic forwards are
        # batched over all vehicles replanning this step — vehicle states do
        # not change inside this phase, so this is equivalent to the previous
        # per-vehicle batch-1 calls (only the sampling RNG stream differs).
        plan_batch = []
        for v in env.agents_manager.vehicles.values():
            if not v.rl_flag:
                continue
            if (v.id not in pending) or (pending[v.id]['substeps'] >= replan_every):
                close_pending(v.id, done=False)
                # hero tagging (_ensure_hero) runs before this phase, so a
                # fresh spawn is already classified at its first plan
                pl = (fleet_planner
                      if (fleet_planner is not None and hero_mode
                          and not getattr(v, 'hero', False))
                      else planner)
                plan_batch.append((v, pl.generate(v, env.agents_manager)))

        if hero_mode and plan_batch:
            fleet_batch = [(v, p) for v, p in plan_batch if not getattr(v, 'hero', False)]
            plan_batch = [(v, p) for v, p in plan_batch if getattr(v, 'hero', False)]
            if fleet_batch:
                fobs = [dict2arrays(v.current_obs) for v, _ in fleet_batch]
                fnbrs = [build_neighbors(v, env.agents_manager,
                                         plan_sharing=getattr(fleet_actor, 'plan_sharing', False),
                                         priority_broadcast=getattr(fleet_actor, 'priority_broadcast', False))
                         for v, _ in fleet_batch]
                fg, fl, ft, fw = (_stack([o[k] for o in fobs], 'cpu') for k in range(4))
                fc = _stack([p['feats'] for _, p in fleet_batch], 'cpu')
                fm = _stack([p['mask'] for _, p in fleet_batch], 'cpu', torch.bool)
                fn = _stack([n for n, _ in fnbrs], 'cpu')
                fnm = _stack([m for _, m in fnbrs], 'cpu', torch.bool)
                with torch.inference_mode():
                    flogits = fleet_actor(fg, fl, ft, fw, fc, fm,
                                          fn[..., :getattr(fleet_actor, 'nbr_in_dim', ACTOR_NBR_FEAT_DIM)],
                                          fnm)
                facts = torch.argmax(flogits, dim=-1)
                for j, (v, plan) in enumerate(fleet_batch):
                    v.plan_traj = plan['trajs'][int(facts[j])]
                    v.plan_k = 0

        if plan_batch:
            obs = [dict2arrays(v.current_obs) for v, _ in plan_batch]
            nbrs = [build_neighbors(v, env.agents_manager,
                                    plan_sharing=getattr(actor_inf, 'plan_sharing', False),
                                    priority_broadcast=getattr(actor_inf, 'priority_broadcast', False))
                    for v, _ in plan_batch]
            g_t, l_t, t_t, w_t = (_stack([o[k] for o in obs], inf_device) for k in range(4))
            cand_t = _stack([p['feats'] for _, p in plan_batch], inf_device)
            mask_t = _stack([p['mask'] for _, p in plan_batch], inf_device, torch.bool)
            nbr_t = _stack([n for n, _ in nbrs], inf_device)
            nm_t = _stack([m for _, m in nbrs], inf_device, torch.bool)

            with torch.inference_mode():
                logits = actor_inf(g_t, l_t, t_t, w_t, cand_t, mask_t,
                                   nbr_t[..., :getattr(actor_inf, 'nbr_in_dim', ACTOR_NBR_FEAT_DIM)],
                                   nm_t)
                values = critic_inf(g_t, l_t, t_t, w_t, nbr_t, nm_t).squeeze(-1)
                # sample on CPU: one device sync per decision step, and the
                # categorical draw stays off accelerator RNG
                dist = Categorical(logits=logits.cpu())
                actions = dist.sample()
                logps = dist.log_prob(actions)
                values = values.cpu()

            for j, (v, plan) in enumerate(plan_batch):
                a = int(actions[j])
                if commit_weight > 0.0:
                    # voluntary-flip penalty: charged only when the previous
                    # grid cell is STILL mask-legal (forced replans are free)
                    # and neither choice is a stop/emergency candidate
                    stop0 = len(plan['mask']) - 3
                    prev = getattr(v, 'commit_prev', None)
                    if a < stop0 and prev is not None and plan['mask'][prev[0]]:
                        d_f = plan['feats'][a][12] * 3.0
                        v_f = plan['feats'][a][13] * 16.0
                        v.commit_pen = commit_weight * (abs(d_f - prev[1]) / 3.0
                                                        + abs(v_f - prev[2]) / 16.0)
                    v.commit_prev = (None if a >= stop0 else
                                     (a, plan['feats'][a][12] * 3.0,
                                      plan['feats'][a][13] * 16.0))
                v.plan_traj = plan['trajs'][a]
                v.plan_k = 0
                g, l, ta, w = obs[j]
                nbr, nbr_mask = nbrs[j]
                pending[v.id] = {
                    'grid': g, 'laser': l, 'target': ta, 'wps': w,
                    'cand': plan['feats'], 'cand_mask': plan['mask'],
                    'nbr': nbr, 'nbr_mask': nbr_mask,
                    'action': a, 'logp': float(logps[j]), 'value': float(values[j]),
                    'reward': 0.0, 'substeps': 0,
                }

        for v in env.agents_manager.vehicles.values():
            if not v.rl_flag:
                continue
            acc, ste = track_action(v.state, v.plan_traj, v.plan_k, dt, v.size[0])
            v.acc, v.ste = np.clip([acc, ste], min_action, max_action)

        # 2) env step (IDM vehicles + kinematics + new obs + rewards)
        obs_batch, r_batch, d_batch, _ = env.step()
        total_steps += 1
        env.current_step = total_steps

        # 3) accumulate rewards, handle finished agents
        i = 0
        to_remove = []
        for v in env.agents_manager.vehicles.values():
            if not v.rl_flag:
                continue
            tr = pending.get(v.id)
            if tr is not None:
                tr['reward'] += (gamma ** tr['substeps']) * float(r_batch[i, 0])
                tr['substeps'] += 1
            v.plan_k += 1
            if d_batch[i, 0]:
                if (not hero_mode) or getattr(v, 'hero', False):
                    # timeout is a truncation, not a true terminal state: the
                    # episode was cut by the alive budget, so bootstrap the tail
                    # with V(s_T) instead of 0 (HANDOFF §4.5). Collision/goal/
                    # outside remain genuine terminals.
                    truncated = (v.alive < 0) and not (v.goal or v.collision or v.outside)
                    close_pending(v.id, done=not truncated)
                    last_v = (values_of([v], env.agents_manager, critic_inf, inf_device)[0]
                              if truncated else 0.0)
                    buffer.finish(v.id, last_value=last_v)
                    if hero_mode:
                        hero_stats['goal' if v.goal else
                                   ('collision' if v.collision else 'timeout')] += 1
                to_remove.append(v.id)
            i += 1
        reward_history.append(float(np.mean(r_batch)) if len(r_batch) else 0.0)

        if to_remove:
            for vid in to_remove:
                env.agents_manager.remove_vehicle(vid)
            env.record_saver()

        # 4) traffic generation: either Poisson arrival demand (veh/h, the
        # density is emergent — standard traffic-sim protocol) or the legacy
        # maintain-a-population mode. Both ramp over the curriculum.
        progress = total_steps / num_steps
        ramp = 0.0 if progress < warmup_frac else \
            (progress - warmup_frac) / max(1.0 - warmup_frac, 1e-9)
        rl_target = rl_ratio_range[0] + (rl_ratio_range[1] - rl_ratio_range[0]) * ramp
        vehicles = env.agents_manager.vehicles
        if demand_mode == 'arrival':
            lam = arrival_range[0] + (arrival_range[1] - arrival_range[0]) * ramp
            spawn_backlog += np.random.poisson(lam * dt / 3600.0)
            n_target = None
            want_spawn = (spawn_backlog >= 1) and (len(vehicles) < max_pop)
        else:
            n_target = round(vehs_range[0] + (vehs_range[1] - vehs_range[0]) * ramp)
            want_spawn = len(vehicles) < n_target
        if want_spawn and (total_steps - last_spawn_step >= spawn_cooldown):
            n_rl = sum(1 for veh in vehicles.values() if veh.rl_flag)
            # ">= 0.999": an all-RL population already sits exactly at ratio
            # 1.0, which would otherwise flip every spawn to IDM
            want_rl = (n_rl / max(len(vehicles), 1)) < rl_target or rl_target >= 0.999
            spawn = generate_veh(
                origin_lines, destination_lines, origin_yaws, destination_yaws,
                veh_length, veh_width, init_vel, env.agents_manager, rl_flag=want_rl)
            if spawn is not None:   # None = every approach lane is occupied; retry next step
                new_origin, new_destination, new_size, new_rl_flag = spawn
                vid = env.agents_manager.add_vehicle(new_origin, new_destination, new_size, new_rl_flag)
                last_spawn_step = total_steps
                if demand_mode == 'arrival':
                    spawn_backlog -= 1
                if new_rl_flag and reset_state_prob > 0 and random.random() < reset_state_prob:
                    # dead-zone reset curriculum: teleport this spawn to a
                    # stopped, misaligned mid-route pose (the argmax trap
                    # diagnosed at light traffic: stalled mid-turn, heading
                    # 25-40 deg off the reference) so the policy learns escape
                    nv = env.agents_manager.vehicles[vid]
                    ref = build_reference_path(nv)
                    nv.frenet_ref = ref
                    for _ in range(6):
                        s = random.uniform(0.35, 0.75) * ref.s[-1]
                        px, py, pth = (float(a) for a in ref.sample(np.array([s])))
                        lat = random.uniform(-1.5, 1.5)
                        px += lat * -math.sin(pth)
                        py += lat * math.cos(pth)
                        if not (5 <= px <= 135 and 5 <= py <= 135):
                            continue
                        clear = all((o.state[0]-px)**2 + (o.state[1]-py)**2 > 8.0**2
                                    for o in env.agents_manager.vehicles.values() if o.id != vid)
                        if clear:
                            pth += random.uniform(-0.7, 0.7)
                            nv.state = np.array([px, py, pth, 0.0])
                            nv.zone_id_update()
                            nv.his_traj = [nv.state.copy()]
                            env.agents_manager.get_single_obs(nv)
                            break

        # 5) PPO update once enough decisions were collected
        if buffer.total_size() >= rollout_size:
            for vid in list(pending.keys()):
                close_pending(vid, done=False)
            open_ks = buffer.open_keys()
            last_vals = dict.fromkeys(open_ks, 0.0)
            live = [(k, env.agents_manager.vehicles.get(k)) for k in open_ks]
            live = [(k, v) for k, v in live if v is not None and v.rl_flag]
            if live:
                vals = values_of([v for _, v in live], env.agents_manager, critic_inf, inf_device)
                last_vals.update({k: val for (k, _), val in zip(live, vals)})
            for key in open_ks:
                buffer.finish(key, last_value=last_vals[key])

            data = buffer.tensors(device)
            a_losses, c_losses = ppo_update(
                actor, critic, actor_optimizer, critic_optimizer, data, device,
                epochs=ppo_epochs, minibatch=minibatch, clip=clip,
                ent_coef=ent_coef, vf_coef=vf_coef)
            actor_loss_history.extend(a_losses)
            critic_loss_history.extend(c_losses)
            buffer.clear()
            if use_mirror:
                actor_inf.load_state_dict(actor.state_dict())
                critic_inf.load_state_dict(critic.state_dict())
            n_updates += 1
            if checkpoint_every and (n_updates % checkpoint_every == 0):
                torch.save(actor.state_dict(), "model/mappo_actor_ckpt.pt")
                torch.save(critic.state_dict(), "model/mappo_critic_ckpt.pt")
            n_rl_now = sum(1 for veh in env.agents_manager.vehicles.values() if veh.rl_flag)
            print(f"step {total_steps}: PPO update on {data['action'].shape[0]} decisions | "
                  f"pg_loss {np.mean(a_losses):.4f} | v_loss {np.mean(c_losses):.4f} | "
                  f"mean step reward {np.mean(reward_history[-1000:]):.3f} | "
                  f"curriculum: {len(env.agents_manager.vehicles)} vehs "
                  + (f"(demand {lam:.0f} veh/h, backlog {spawn_backlog:.0f})"
                     if demand_mode == 'arrival' else f"(target {n_target})")
                  + f", {n_rl_now} RL (ratio target {rl_target:.2f})"
                  + (f" | hero g/c/t {hero_stats['goal']}/{hero_stats['collision']}"
                     f"/{hero_stats['timeout']}" if hero_mode else ""))

    return actor, critic, actor_loss_history, critic_loss_history, reward_history
