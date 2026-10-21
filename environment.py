import gym
import numpy as np
from waypoint import generate_waypoints
import math
from gym import spaces
from IDM import IDM_update
import json



def oncoming_lane_penalty(state):
    """
    -1 while driving in the oncoming half of a corridor arm, else 0.
    Lane convention (from intersection_config origin/destination lines):
    westbound y in [60,70], eastbound y in [70,80], northbound x in [60,70],
    southbound x in [70,80]. Applies only in the arms (never inside the
    conflict box), only when roughly axis-aligned (within 60 degrees), and
    with a 0.5 m centerline margin so brushing the line is not punished.
    """
    x, y, th = state[0], state[1], state[2]
    if 60 <= y <= 80 and (x > 80 or x < 60):        # horizontal arm
        c = math.cos(th)
        if (c < -0.5 and y > 70.5) or (c > 0.5 and y < 69.5):
            return -1.0
    elif 60 <= x <= 80 and (y < 60 or y > 80):      # vertical arm
        s = math.sin(th)
        if (s > 0.5 and x > 70.5) or (s < -0.5 and x < 69.5):
            return -1.0
    return 0.0


class IE(gym.Env):
    def __init__(self, vehicles_manager, veh_config, env_config, idm_yield_to_rl=False,
                 idm_eta_priority=False, idm_gap_accept=False):
        super(IE, self).__init__()

        # False = original IDM (blind to RL vehicles); True = IDM additionally
        # car-follows the RL vehicle ahead (IDM.priority_deceleration_yield_rl)
        self.idm_yield_to_rl = idm_yield_to_rl
        # ETA-ordered conflict priority for IDM (yields to ALL vehicles by
        # predicted arrival order, lane-density adjusted; IDM.py has details)
        self.idm_eta_priority = idm_eta_priority
        # gap-acceptance IDM (movement-conflict filter + usable-gap test);
        # takes precedence over eta_priority when both are set
        self.idm_gap_accept = idm_gap_accept
        self.veh_config = veh_config
        # Initialize environment parameters
        self.dt = env_config['dt']
        self.vel_lim = env_config['vel_lim']
        self.front_back_lim = env_config['front_back_lim']
        self.side_lim = env_config['side_lim']

        self.acc_lim = env_config['acc_lim']
        self.ste_lim = env_config['ste_lim']
        self.action_space = spaces.Box(low=np.array([self.acc_lim['min'], self.ste_lim['min']]), high=np.array([self.acc_lim['max'], self.ste_lim['max']]), dtype=np.float64)

        # Initialize agent and vehicle data
        self.agents_manager = vehicles_manager

        # Terminate-related parameters
        self.current_step = 0

        self.step_record = []
        self.record()

        # per-episode reward-term instrumentation (one row per finished RL
        # vehicle); truncated at env construction so each run starts clean
        self.terms_path = "./log/episode_reward_terms.csv"
        with open(self.terms_path, "w") as f:
            f.write("end_step,veh_id,maneuver,outcome,steps,"
                    "progress,delay,collision,side,wrong_lane,goal,commit,total\n")

    def record(self):
        for veh in self.agents_manager.vehicles.values():  
            record = {
                "veh_id": veh.id,
                "step": self.current_step,
                "rl_flag": 1 if veh.rl_flag else 0,
                "pos_x": veh.state[0],
                "pos_y": veh.state[1],
                "heading": veh.state[2],
                "velocity": veh.state[3],
                "des_x": veh.destination[0],
                "des_y": veh.destination[1],
                "length": veh.size[0],
                "width":  veh.size[1],
            }
            self.step_record.append(record)
    
    def record_saver(self):
        # dump-then-clear: each file holds only the steps since the previous
        # dump (video.py reads any such segment); keeping the full history in
        # memory and re-dumping it made writes O(total steps) each episode
        if not self.step_record:
            return
        try:
            with open(f"./data/step_{self.current_step}.json", "w", newline="") as f:
                json.dump(self.step_record, f)
        except FileNotFoundError:
            # no ./data dir (the NCI sync excludes it deliberately): step
            # recording only feeds local video rendering, and a 240k-step
            # training would shed tens of thousands of these tiny files —
            # drop the segment instead of crashing. Buffer still clears.
            pass
        self.step_record = []

    def eval_record_saver(self):
        if not self.step_record:
            return
        with open(f"./data/EVAL_step_{self.current_step}.json", "w", newline="") as f:
            json.dump(self.step_record, f)
        self.step_record = []
    
    def get_laser_reward(self, lasers):
        penalty = 0

        for i in [0, 4]:
            if lasers[i] < self.front_back_lim:
                penalty += (lasers[i] - self.front_back_lim)

        for i in [2, 6]:
            if lasers[i] < self.side_lim:
                penalty += (lasers[i] - self.side_lim)

        for i in [1, 3, 5, 7]:
            if lasers[i] < (self.side_lim + 3):
                penalty += (lasers[i] - self.side_lim - 3)

        return penalty

    def get_reward(self, v):
        """
        Sparse safety/goal terms + a single dense progress term.
        Progress is measured along the vehicle's Frenet reference route (arc
        length s), which is monotone along the intended route and consistent
        with the candidate action space — unlike the old L1 distance to the
        destination point, which kept a permanent lateral residual now that
        success only requires exiting the map inside the destination zone.
        Smoothness is guaranteed by construction (jerk-bounded polynomial
        candidates), so it is no longer rewarded here.
        """
        s_cur, _ = v.frenet_ref.project(v.state[0], v.state[1])
        pre_s = getattr(v, 'pre_s', None)
        progress_reward = (s_cur - pre_s) if pre_s is not None else 0.0

        # run 13 tried -0.06 to make yielding stops cheaper; timeouts exploded
        # (10-veh argmax: 33.5% -> 50.9%) — waiting must stay this expensive
        delay_penalty = -0.1

        # NOTE: fault-weighted collision (0.4x for the box's earlier entrant)
        # was tried in run 17 and REVERTED — battery success dropped 50.4->38.7
        # (~2.5 sigma): discounting the right-of-way holder's crashes rewards
        # claiming the box early, deepening congestion. Keep the flat -50.
        collision_penalty = -1 if v.collision else 0

        # NOTE: walls stay included in the proximity penalty — run 5 excluded
        # them and both collisions (16%->25%) and right-turn timeouts got
        # WORSE; the wall component does real keep-off-the-walls shaping
        side_penalty = self.get_laser_reward(v.lasers)
        # proximity risk scales with motion: full strength from 5 m/s, floored
        # at 0.4 when slow/stopped. The floor keeps a keep-your-distance
        # deterrent (run 3 with a zero floor cut queueing cost but collision
        # rate rose 20%->35%); the scaling still fixes run 2's inversion where
        # waiting out a jam accumulated side -97 and scored worse than crashing.
        side_scale = min(max(v.state[3] / 5.0, 0.4), 1.0)

        goal_bonus = 1 if v.goal else 0

        wrong_lane = oncoming_lane_penalty(v.state)

        # weighted terms, kept separate for instrumentation; summing them in
        # this order is float-identical to the previous inline expression
        # heterogeneous agents (hetero_env): a vehicle may carry its own
        # weight overrides in v.reward_weights (dict, partial — missing keys
        # fall back to the fleet defaults below). The HERO agent trains with
        # e.g. a doubled goal weight while the frozen fleet keeps the
        # certified defaults.
        w = getattr(v, 'reward_weights', None) or {}
        terms = {
            'progress': w.get('progress', 1.0) * progress_reward,
            'delay': w.get('delay', 1.0) * delay_penalty,
            'collision': w.get('collision', 50.0) * collision_penalty,   # -80 tried TWICE and
            # reverted twice. Run 28: collisions 4.0->2.8 but timeouts
            # 2.8->6.9, net DOWN. Run 42 (retest after the intent-aware mask
            # killed timeouts): transit inflated +0.7-1.4 s everywhere and at
            # 1200 veh/h BOTH failure modes worsened (collisions 4.4->5.8,
            # timeouts 0.2->1.5) — hesitation lengthens box occupancy, which
            # BREEDS conflicts at high demand. The caution axis is not a
            # tunable win at any tested operating point; -50 is final.
            'side': w.get('side', 0.2) * side_scale * side_penalty,
            'wrong_lane': w.get('wrong_lane', 0.5) * wrong_lane,
            'goal': w.get('goal', 20.0) * goal_bonus,
            # plan-commitment shaping (run 43): the trainer charges a small
            # pre-weighted penalty when the vehicle VOLUNTARILY abandons its
            # broadcast plan (same grid cell still mask-legal but a different
            # one chosen) — G7 plan sharing is only worth what the plans are
            # worth (measured flip rate before shaping: 28.1% of replans,
            # 31.3% near the box). Zero for IDM, evaluation, forced replans
            # and emergency/stop choices; the trainer owns the weight.
            'commit': -getattr(v, 'commit_pen', 0.0),
        }
        v.commit_pen = 0.0
        total_reward = sum(terms.values())

        acc = getattr(v, 'reward_terms', None)
        if acc is None:
            acc = {k: 0.0 for k in terms}
            v.reward_terms = acc
        for k, val in terms.items():
            acc[k] += val

        rl_controler = side_penalty < 0
        return total_reward, s_cur, rl_controler

    def log_episode(self, v):
        """Append one csv row with the episode's accumulated reward terms."""
        if v.collision:
            outcome = 'collision'
        elif v.goal:
            outcome = 'goal'
        elif v.outside:
            outcome = 'outside'
        else:
            outcome = 'timeout'
        maneuver = {0: 'straight', -1: 'left', 1: 'right'}[v.maneuver]
        acc = getattr(v, 'reward_terms', {})
        vals = [acc.get(k, 0.0) for k in ('progress', 'delay', 'collision', 'side', 'wrong_lane', 'goal', 'commit')]
        steps = self.agents_manager.alive - v.alive
        with open(self.terms_path, "a") as f:
            f.write(f"{self.current_step},{v.id},{maneuver},{outcome},{steps},"
                    + ",".join(f"{x:.3f}" for x in vals)
                    + f",{sum(vals):.3f}\n")

    def step(self):
        obs_batch, reward_batch, done_batch = [], [], []
        # current obs
        for v in self.agents_manager.vehicles.values():
            if v.rl_flag:
                obs_batch.append(v.current_obs)

        # IDM-vehicles action update
        IDM_update(manager=self.agents_manager, yield_to_rl=self.idm_yield_to_rl,
                   eta_priority=self.idm_eta_priority, gap_accept=self.idm_gap_accept)
        # all vehicles update
        self.agents_manager.act_all(self.dt)
        # update record
        self.record()

        # next obs
        for v in self.agents_manager.vehicles.values():
            if v.rl_flag:
                v = self.agents_manager.get_single_obs(v)       
                dead = True if v.alive < 0 else False

                done_i = v.goal or dead or v.collision or v.outside
                # print('GOAL in step', self.current_step)
                r_i, v.pre_s, v.rl_controler = self.get_reward(v)
                if done_i:
                    self.log_episode(v)
                reward_batch.append([r_i])
                done_batch.append([done_i])

        # IDM lifecycle sweep: non-RL vehicles are not in the done batch and
        # were previously never removed at all — cars that left the map kept
        # orbiting their waypoint forever (pure pursuit with the target inside
        # the min turning radius, the "circling" seen in replays) and these
        # ghosts silently ate curriculum population slots (~4 of 14 at high
        # density). RL removal stays with the trainer (rollout bookkeeping).
        idm_gone = [v.id for v in self.agents_manager.vehicles.values()
                    if (not v.rl_flag)
                    and (v.alive < 0 or self.agents_manager.is_outside(v))]
        for vid in idm_gone:
            self.agents_manager.remove_vehicle(vid)

        trunc = False
        return obs_batch, np.array(reward_batch), np.array(done_batch), trunc
        
    
