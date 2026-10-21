# DRL_Distributed_Control

Deep Reinforcement Learning for Communication-free Distributed Control of Autonomous Vehicles in the Unstructured Intersection.

This repository releases the **NEUT model, planner generation v3** — a simplified version of the
model used in the study — together with the minimal code needed to load it and run it in the
intersection simulator.

## What is in the model

* **Scenario**: a four-arm unsignalised, lane-free intersection; vehicles arrive as a Poisson
  stream (200–1400 veh/h) on 12 origin–destination routes; a share of the vehicles is driven by
  the learned policy, the rest by a gap-acceptance IDM.
* **Decision model**: every 0.2 s a Frenet candidate planner proposes 49 trajectory candidates
  (5 lateral offsets × 7 target speeds, 11 scene-relative follow / yield / go / pass manoeuvres,
  3 stop candidates incl. an always-available emergency stop); a masked actor network scores the
  candidates from the ego state, the laser scan, the neighbour tokens (with the neighbours'
  committed plans, no explicit communication) and 28 per-candidate features (trajectory geometry
  + predicted minimum clearance, time-to-conflict, relative heading, manoeuvre type).
  Infeasible candidates (corridor, kinematics, predicted collision within 0.2 s) are masked.
* **Training**: Masked MAPPO, 240k environment steps, curriculum from 200 to 1400 veh/h,
  hero-mode training against a frozen fleet; reward = progress − delay − collision (weights 1 / 1 / 50,
  "NEUT" = neutral weights).
* **Planner v3** (`frenet_planner.py`, `FrenetPlanner(gen_v3=True, horizon=4.0, profile_horizon=3.0, v3_fix=True, ...)`):
  soft collision layer, 4 s look-ahead with 3 s manoeuvre profiles, arrival-time yield / go candidates.

## Files

| path | content |
|---|---|
| `model/v3/runNEUT_s<seed>_actor.pt` / `_critic.pt` | the released NEUT v3 policies (5 training seeds, 7–11) |
| `model/v3/profile_results.csv` | evaluation battery of the released models (see below) |
| `model/final_actor.pt`, `model/final_critic.pt` | the frozen fleet policy used during training (planner v1) |
| `frenet_planner.py` | candidate planner (v1 / v2 / v3 generations) |
| `MAPPO.py`, `MAPPO_trainer.py` | masked actor / critic and the PPO loop |
| `environment.py`, `vehicle.py`, `veh_generatort.py`, `IDM.py`, `rule.py`, `kinematic.py`, `Laser.py`, `BEV.py`, `waypoint.py` | simulator |
| `train_MAPPO.py`, `train_v2.py` | training entry points (`train_v2.py --gen v3 --tag NEUT --seed 7`) |
| `train_reward_profile.py`, `sim_arms.py` | evaluation battery and long-run simulations |
| `config/*.yaml` | intersection, vehicle and environment parameters |

## Quick start

```bash
pip install -r requirements.txt            # numpy, torch, pyyaml, shapely, matplotlib, pillow, gym 0.26
# evaluation battery of one released model: homogeneous 80 % policy + 20 % IDM,
# demands 800 / 1200 / 1400 veh/h x 5 evaluation seeds x 300 s -> model/v3/profile_results.csv
python3 train_v2.py --gen v3 --tag NEUT --seed 10 --test-only
# a 10-minute simulation at 1400 veh/h with trajectory export (-> runs/traj/*.npz, runs/sim_*.csv)
python3 sim_arms.py --gen v3 --models NEUT --seeds 10 --arms homo80 --demands 1400 --minutes 10 \
    --weights-dir model/v3 --model-dir runs
# train a new NEUT v3 policy from scratch (240k steps, ~30 min on 8 CPU cores)
python3 train_v2.py --gen v3 --tag NEUT --seed 12 --no-test
```

`TORCH_THREADS` sets the number of CPU threads (default 3).

## Released models: evaluation battery

Homogeneous 80 % policy / 20 % IDM, Poisson demand, 5 evaluation seeds × 300 s per demand,
pooled over the 5 training seeds (success = reached the destination; the rest are collisions or timeouts):

| demand (veh/h) | success rate % | collision % | timeout % |
|---|---|---|---|
| 800 | 90.7 | 9.1 | 0.3 |
| 1200 | 88.4 | 11.4 | 0.1 |
| 1400 | 87.0 | 12.9 | 0.1 |
| all demands | 88.4 | 11.5 | 0.1 |

Per training seed (all demands): 84.5, 85.4, 91.2, 93.2, 87.4 % — seed 10 is the strongest single model.

## Notes

* The policy is communication-free at run time: each vehicle only observes neighbours' states and
  their last committed plan positions (as a vehicle would from on-board perception), no messages are exchanged.
* Collision detection is evaluated for policy-driven vehicles; an IDM vehicle involved in a collision keeps driving.
* License: MIT (see `LICENSE`).
* `mail.py` is a no-op unless `MAIL_APP_PASSWORD` / `MAIL_USER` / `MAIL_TO` are set.
