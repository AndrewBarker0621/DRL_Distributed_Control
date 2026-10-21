import numpy as np
import torch

from environment import IE
from rule import load_config_yaml
from vehicle import VehicleManager
from veh_generatort import generate_veh
from Laser import create_multiple_rectangles

from frenet_planner import FrenetPlanner
from MAPPO import MAPPOActor, MAPPOCritic
from MAPPO_trainer import train_mappo
from plot_saver import save_loss_plot


def main(num_steps=120_000,
         replan_every=2,               # RL decision every 0.2 s (dt = 0.1 s)
         rollout_size=512,
         lr=3e-4,
         vehs_range=(2, 18),           # curriculum: total vehicles, warmup hold -> end
         rl_ratio_range=(1.0, 0.6),    # mixed era (run 32+): warmup pure RL, then meter
                                       # ETA-priority IDM in; (1.0, 1.0) = pure self-play
         warmup_frac=0.25,             # first quarter: 2 RL vehicles alone, then ramp density
         demand_mode='arrival',        # 'arrival' = Poisson demand in veh/h (density emergent);
                                       # 'maintain' = legacy population clamp via vehs_range
         arrival_range=(200.0, 800.0), # veh/h over the curriculum (arrival mode)
         max_pop=18,                   # hard population cap (arrival mode)
         reset_state_prob=0.0,         # fraction of RL spawns teleported to stopped
                                       # misaligned mid-route poses (dead-zone curriculum)
         spawn_cooldown=3,             # min steps between spawns
         resume_from=None,             # (actor_path, critic_path) to continue training
         idm_yield_to_rl=True,         # IDM car-follows the RL vehicle ahead (False = original blind IDM)
         idm_eta_priority=True,        # run 32+: IDM yields to ALL by predicted arrival order (ETA)
         idm_gap_accept=True,          # run 45+ default: gap-acceptance IDM (conflict filter +
                                       # usable-gap test, no entrance stalling); False = pure ETA
                                       # ordering (runs 32-44). Takes precedence over eta
         commit_weight=0.0,            # plan-commitment shaping (run 43 trial): penalty per
                                       # voluntary plan flip; 0 = off (standing default)
         hero_mode=False,              # heterogeneous training: one hero vehicle learns with
                                       # hero_weights; the rest of the RL fleet runs FROZEN
                                       # from fleet_from (argmax). Pair with resume_from to
                                       # warm-start the hero from the fleet policy
         hero_weights=None,            # partial reward-weight overrides, e.g. {'goal': 40.0}
         hero_count=1,                 # simultaneous heroes (shared trainable policy)
         hero_frac=None,               # fleet-share heroes: P(fresh RL spawn is hero), overrides count
         fleet_from=('model/final_actor.pt', 'model/final_critic.pt'),
         seed=7,                       # training seed (multi-seed runs are part of the eval protocol)
         priority_broadcast=False,     # G8 (optional): neighbor tokens carry delta-ETA priority
                                       # (the exact ordering ETA-IDM yields by) + precede flag
         plan_sharing=False,           # G7 (optional): neighbor tokens carry the neighbor's
                                       # committed-plan positions 1.5/3.0 s ahead (candidate-level
                                       # V2V, +4 dims). Changes net shapes — G6 weights incompatible
         ablate_bev=False,             # zero the BEV observation (information ablation, arch kept)
         use_bev=False,                # default off since run 19: battery-identical to the BEV
                                       # model, -67% params, ~45% faster wall clock; True restores it
         soft_collision=False,         # v2 planner (H23): collision layer soft except the first
         col_hard_horizon=0.2,         #   col_hard_horizon s; see FrenetPlanner
         safety_feats=False,           # v2 planner: +3 safety dims per candidate -> actor
                                       #   cand width 23; v1 weights incompatible. In hero mode
                                       #   the frozen fleet keeps a v1 planner (dual planner)
         gen_v3=False,                 # v3 planner (H26): scene-relative follow/yield/go/pass
                                       #   candidates (+11 candidates, +5 feature dims -> 28)
         planner_horizon=None,         # candidate horizon in s (None = planner default 3.0;
                                       #   v3 uses 4.0)
         profile_horizon=None,         # manoeuvre duration <= horizon (v3: 3.0 with a 4 s
                                       #   look-ahead); None = horizon
         v3_fix=False,                 # v3 (fixed): arrival-time yield/go + bounded pass
         send_mail=True):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    # the `random` module drives vehicle generation and was previously never
    # seeded in this entry point (spawn streams were irreproducible)
    import random as _random
    _random.seed(seed)
    # M1 Pro: the default of 10 intra-op threads (incl. the 2 efficiency
    # cores) is ~40% slower than 3 threads on this model's small CPU convs.
    # TORCH_THREADS overrides for other hosts (e.g. an NCI ncpus allocation)
    import os as _os
    torch.set_num_threads(int(_os.environ.get('TORCH_THREADS', '3')))

    intersection_config = load_config_yaml('./config/intersection_config.yaml')
    veh_config = load_config_yaml('./config/veh_config.yaml')
    env_config = load_config_yaml('./config/env_config.yaml')

    origin_lines = np.array(intersection_config['origin_lines'])
    destination_lines = np.array(intersection_config['destination_lines'])
    origin_yaws = np.array([np.pi, np.pi / 2, 0, 3 * np.pi / 2])
    destination_yaws = np.array([0, 3 * np.pi / 2, np.pi, np.pi / 2])
    prohibited_zones_center = np.array(intersection_config['prohibited_zones_center']).reshape((4, -1))
    prohibited_zones_length = np.array(intersection_config['prohibited_zones_length']).reshape((4, -1))
    prohibited_zones_width = np.array(intersection_config['prohibited_zones_width']).reshape((4, -1))
    prohibited_zones_heading = np.array(intersection_config['prohibited_zones_heading']).reshape((4, -1))

    # PPO minibatch fwd+bwd on this M1 Pro: ~22 ms on MPS vs ~130 ms on CPU
    # (outputs/grads agree to ~1e-7; categorical sampling stays on CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"training device: {device}")

    prohibited_zone_params = np.concatenate(
        (prohibited_zones_center, prohibited_zones_length,
         prohibited_zones_width, prohibited_zones_heading), axis=1)
    prohibited_zone_recs = create_multiple_rectangles(prohibited_zone_params)
    laser_angles = np.arange(0, 2 * np.pi, np.pi / 4)
    way_point_interval = np.arange(0.1, 1.01, 0.1)

    agents_manager = VehicleManager(
        veh_config=veh_config,
        intersection_config=intersection_config,
        env_config=env_config,
        prohibited_zone_recs=prohibited_zone_recs,
        laser_angles=laser_angles,
        way_point_interval=way_point_interval,
    )
    new_origin, new_destination, new_size, new_rl_flag = generate_veh(
        origin_lines, destination_lines, origin_yaws, destination_yaws,
        veh_config['veh_length'], veh_config['veh_width'], veh_config['init_vel'],
        agents_manager, rl_flag=True)
    agents_manager.add_vehicle(new_origin, new_destination, new_size, new_rl_flag)

    agents_manager.ablate_bev = ablate_bev or (not use_bev)
    env = IE(vehicles_manager=agents_manager, veh_config=veh_config, env_config=env_config,
             idm_yield_to_rl=idm_yield_to_rl, idm_eta_priority=idm_eta_priority,
             idm_gap_accept=idm_gap_accept)
    pkw = dict(soft_collision=soft_collision, col_hard_horizon=col_hard_horizon,
               safety_feats=safety_feats, gen_v3=gen_v3)
    if planner_horizon is not None:
        pkw['horizon'] = planner_horizon
    if profile_horizon is not None:
        pkw['profile_horizon'] = profile_horizon
    if v3_fix:
        pkw['v3_fix'] = True
    planner = FrenetPlanner(env_config, veh_config, **pkw)
    planner_v2 = soft_collision or safety_feats or gen_v3
    gen_name = 'v3' if gen_v3 else ('v2' if planner_v2 else 'v1')
    print(f"planner: {gen_name} (soft collision {soft_collision}, hard floor {col_hard_horizon} s, "
          f"safety feats {safety_feats}, v3 extras {planner.n_extra}, horizon {planner.T} s, profile {planner.Tp} s)"
          f" | K {planner.K} | cand feat dim {planner.feat_dim}")

    actor = MAPPOActor(cand_feat_dim=planner.feat_dim, use_bev=use_bev,
                       plan_sharing=plan_sharing,
                       priority_broadcast=priority_broadcast)
    critic = MAPPOCritic(use_bev=use_bev, plan_sharing=plan_sharing,
                         priority_broadcast=priority_broadcast)
    if resume_from is not None:
        # continue training from saved weights (optimizer state starts fresh)
        actor.load_state_dict(torch.load(resume_from[0], map_location="cpu"))
        critic.load_state_dict(torch.load(resume_from[1], map_location="cpu"))
        print(f"resumed weights from {resume_from[0]} / {resume_from[1]}")
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=lr)

    fleet_actor = None
    fleet_planner = None
    if hero_mode:
        # the fleet is a v1/G7 model (cand width 20) whatever the hero uses
        fleet_actor = MAPPOActor(use_bev=use_bev, plan_sharing=plan_sharing,
                                 priority_broadcast=priority_broadcast)
        fleet_actor.load_state_dict(torch.load(fleet_from[0], map_location="cpu"))
        fleet_actor.eval()
        if planner_v2:
            # dual planner: the medium keeps the v1 candidate set, hard
            # mask and 20-dim features — byte-identical to every H17-H22 run
            fleet_planner = FrenetPlanner(env_config, veh_config)
        print(f"hero mode: fleet frozen from {fleet_from[0]}, hero_weights={hero_weights}"
              f"{', fleet planner v1 (dual planner)' if fleet_planner is not None else ''}")

    trained_actor, trained_critic, actor_losses, critic_losses, rewards = train_mappo(
        env=env,
        planner=planner,
        intersection_config=intersection_config,
        veh_config=veh_config,
        env_config=env_config,
        actor=actor,
        critic=critic,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        num_steps=num_steps,
        replan_every=replan_every,
        rollout_size=rollout_size,
        device=device,
        vehs_range=vehs_range,
        rl_ratio_range=rl_ratio_range,
        spawn_cooldown=spawn_cooldown,
        warmup_frac=warmup_frac,
        demand_mode=demand_mode,
        arrival_range=arrival_range,
        max_pop=max_pop,
        reset_state_prob=reset_state_prob,
        commit_weight=commit_weight,
        hero_mode=hero_mode,
        fleet_actor=fleet_actor,
        hero_weights=hero_weights,
        hero_count=hero_count,
        hero_frac=hero_frac,
        fleet_planner=fleet_planner,
    )

    torch.save(trained_actor.state_dict(), "mappo_actor.pt")
    torch.save(trained_critic.state_dict(), "mappo_critic.pt")
    np.savetxt("mappo_actor_loss.csv", actor_losses, delimiter=",")
    np.savetxt("mappo_critic_loss.csv", critic_losses, delimiter=",")
    np.savetxt("mappo_step_reward.csv", rewards, delimiter=",")
    save_loss_plot(actor_losses, critic_losses, filename="mappo_losses.png")
    env.close()

    if send_mail:
        from mail import send_email_notification
        send_email_notification(
            subject="MAPPO Task Completed",
            body="Your MAPPO training task has been completed successfully.",
            to_email=None,
        )
    print('-------------------------------------------------')


if __name__ == "__main__":
    main()
