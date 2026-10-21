import random
import numpy as np
from kinematic import kinematic_Update, simple_Update, inverse_kinematic_Update
from waypoint import generate_first_waypoint
import heapq
from Laser import rect_corners_np, laser_scan, rects_overlap_any
import math
from BEV import create_bev_grid
from waypoint import generate_waypoints
from frenet_planner import (ref_waypoints, dist_to_conflict_box, approach_speed,
                            build_reference_path)
from IDM import _lane_half_key

# exit half of each destination arm (lane convention, see HANDOFF):
# arm0 exit y in [70,80] -> half 1; arm1 exit x in [70,80] -> 1;
# arm2 exit y in [60,70] -> 0; arm3 exit x in [60,70] -> 0
EXIT_HALF = {0: 1, 1: 1, 2: 0, 3: 0}


class Vehicle:
    def __init__(self, vehicle_id, origin, destination, size, prio, rl_flag, alive):
        self.id = vehicle_id
        self.rl_flag = rl_flag
        # priority is the distance to the center, smaller means higher
        self.priority = prio   # rl agent and agent starts from des section has inf priority

        self.origin = np.array(origin, dtype=float)  # (x, y, theta, velocity)
        self.destination = np.array(destination, dtype=float)  # (x, y, theta)
        self.size = np.array(size, dtype=float)  # (length, width)
        self.state = self.origin.copy()  # Current dynamic state
        '''
        one zone includes two sections (one origin section and one destination section in parallel)
        '''
        self.zone_id = zone_assign(self.state)  # 0: right, 1: bottom, 2: left, 3: top, -1: conflict zone
        self.origin_zone_id = zone_assign(self.origin)  # 0: right, 1: bottom, 2: left, 3: top, -1: conflict zone
        self.destination_zone_id = zone_assign(self.destination)  # 0: right, 1: bottom, 2: left, 3: top, -1: conflict zone
        self.maneuver = maneuver_assign(self.origin_zone_id, self.destination_zone_id)  # 0: straight, -1: turn left, 1: turn right
        
        self.IDM_config = None

        self.acc = 0
        self.ste = 0
        # coor to current obs
        self.ref_action = np.array([0,0])
        # coor to next obs
        self.next_ref_action = None

        self.rl_controler = False

        self.his_traj = []           
        self.max_history_len = 12    # store up to 15 states
        self.record_history()

        self.init_dis = np.abs(self.destination[:2] - self.origin[:2]).sum()
        self.pre_dis = self.init_dis

        self.alive = alive
        self.collision = False
        self.goal = False
        self.outside = False
        self.reward = 0

        self.self_bbs = None
        self.other_bbs = None
        self.lasers = None
        self.current_obs = None
        # per-step acceleration estimate (m/s^2), updated in act(); feeds the
        # neighbor-token speed-trend feature ("is that car braking for me?")
        self.accel_est = 0.0

    def act(self, dt, method='auto'):
        v_before = self.state[3]
        if method == 'kinematic':
            self.state = kinematic_Update(self.state, self.acc, self.ste, dt, self.size[0])
        elif method == 'inverse':
            waypoint = generate_first_waypoint(self.state, self.destination, self.zone_id, self.destination_zone_id, self.maneuver, 0.1)
            # acc is from IDM
            _, self.ste = inverse_kinematic_Update(self.state, waypoint, 14, 18, self.size[0], dt)
            [self.acc, self.ste] = np.clip([self.acc, self.ste], [-8, -0.3142], [8, 0.3142])
            self.state = kinematic_Update(self.state, self.acc, self.ste, dt, self.size[0])
        elif method == 'ref_pursuit':
            # IDM turning vehicles: pure pursuit on the same kinematically
            # feasible reference route the RL planner uses (straight approach
            # -> setback arc -> parallel exit). The legacy 'inverse' path via
            # waypoint.py's origin-based Bezier cuts the prohibited corner and
            # swings oddly; it is kept above behind manager.idm_ref_steering.
            # Longitudinal control (self.acc) still comes from IDM untouched.
            ref = getattr(self, 'frenet_ref', None)
            if ref is None:
                ref = build_reference_path(self)
                self.frenet_ref = ref
            s0, _ = ref.project(self.state[0], self.state[1])
            look = max(4.0, 0.6 * self.state[3])
            tx, ty, _ = ref.sample(np.array([s0 + look]))
            dx, dy = float(tx[0]) - self.state[0], float(ty[0]) - self.state[1]
            dist = math.hypot(dx, dy)
            alpha = (math.atan2(dy, dx) - self.state[2] + math.pi) % (2 * math.pi) - math.pi
            self.ste = math.atan2(2.0 * 0.8 * self.size[0] * math.sin(alpha), max(dist, 1e-3))
            [self.acc, self.ste] = np.clip([self.acc, self.ste], [-8, -0.3142], [8, 0.3142])
            self.state = kinematic_Update(self.state, self.acc, self.ste, dt, self.size[0])
        
        # non-negtive velocity
        self.state[3] = max(0, self.state[3])
        self.accel_est = (self.state[3] - v_before) / dt

        self.zone_id_update()
        self.alive -= 1
        
        if self.rl_flag:
            self.record_history()

    def zone_id_update(self):
        self.zone_id = zone_assign(self.state)
    
    def record_history(self):
        """
        Stores the current (x, y, theta, velocity) in history. 
        If length exceeds max_history_len, remove the oldest entry.
        """
        # Append a copy of the current state to history
        self.his_traj.append(self.state.copy())
        # print((self.his_traj))
        # Keep only the last max_history_len entries
        if len(self.his_traj) > self.max_history_len:
            self.his_traj.pop(0)
        
    def shortest_angle_diff(self):
        # Calculate the direct difference
        difference = (self.destination[2] - self.origin[2]) % (2 * np.pi)
        
        # Adjust to always take the shortest path
        if difference > np.pi:
            difference -= 2 * np.pi
        return difference
        
    def add_manually_history(self, state):
        '''
        Manually add a state to the vehicle's trajectory history.
        state` should be a list or array-like: [x, y, theta, velocity]
        '''
        self.his_traj.append(state.copy())
        if len(self.his_traj) > self.max_history_len:
            self.his_traj.pop(0)
        # vehicle.add_to_history([x, y, theta, velocity])


class VehicleManager:
    def __init__(self, veh_config, intersection_config, env_config, prohibited_zone_recs, laser_angles, way_point_interval):
        self.vehicles = {}
        # --- ID config ---
        self.rl_vehicle_ids = set()
        self.max_vehicles = veh_config['max_vehs']
        self.available_ids = list(range(self.max_vehicles))
        heapq.heapify(self.available_ids)
        # --- Init priority ---
        self.next_vehicle_prio = 0
        
        self.IDM_a_max = veh_config['IDM_config']['a_max']
        self.IDM_desired_velocity = veh_config['IDM_config']['desired_velocity']
        self.IDM_delta = veh_config['IDM_config']['delta']
        self.IDM_s0 = veh_config['IDM_config']['s0']
        self.IDM_T = veh_config['IDM_config']['T']
        self.IDM_b = veh_config['IDM_config']['b']
        self.alive = veh_config['alive']
        self.prohibited_zones = np.array(intersection_config['prohibited_zones'])
        self.BEV_size = env_config['BEV']['grid_size']
        self.BEV_resolution = env_config['BEV']['resolution']

        self.laser_length = veh_config['laser_length']
        # static prohibited-zone corner arrays for the vectorized laser/collision
        self._zone_corners = self.prohibited_zones.astype(float)

        self.laser_angles = laser_angles
        self.way_point_interval = way_point_interval
        self.prohibited_zone_recs = prohibited_zone_recs

    def add_vehicle(self, origin, destination, size, rl_flag):
        if not self.available_ids:
            raise RuntimeError("Maximum number of vehicles reached.")
        """Adds a new vehicle to the manager."""
        new_id = heapq.heappop(self.available_ids)
        # print(new_id)
        if rl_flag:
            self.vehicles[new_id] = Vehicle(new_id, origin, destination, size, np.inf, rl_flag, self.alive)
            # init bbs and obs
            self.vehicles[new_id].self_bbs = self.get_bbs_to_self(self.vehicles[new_id])
            self.vehicles[new_id] = self.get_single_obs(self.vehicles[new_id])
            # self.rl_vehicle_ids.add(new_id)
        else:
            self.vehicles[new_id] = Vehicle(new_id, origin, destination, size, self.next_vehicle_prio, rl_flag, self.alive)
            # --- Set priority ---
            self.next_vehicle_prio += 0.001
            # --- Config IDM ---
            self.vehicles[new_id].IDM_config = self.set_IDM_config()
        return new_id

    def remove_vehicle(self, vehicle_id):
        """Removes a vehicle from the manager by its ID."""
        self.vehicles.pop(vehicle_id, None)
        # self.rl_vehicle_ids.discard(vehicle_id)
        heapq.heappush(self.available_ids, vehicle_id)
    
    def get_other_bbs(self, v):
        """Corner arrays (N+4, 4, 2) of all other vehicles plus prohibited zones."""
        rects = [
            rect_corners_np(other_v.state[0], other_v.state[1], other_v.size[0], other_v.size[1], other_v.state[2])
            for other_v in self.vehicles.values() if other_v.id != v.id
        ]
        if rects:
            return np.concatenate([np.array(rects), self._zone_corners], axis=0)
        return self._zone_corners

    def get_bbs_to_self(self, v):
        return rect_corners_np(v.state[0], v.state[1], v.size[0], v.size[1], v.state[2])

    def get_laser(self, v):
        other_bbs = self.get_other_bbs(v)
        angles_aligned = v.state[2] - self.laser_angles
        dist_all = laser_scan(v.state[0:2], self.laser_length, angles_aligned, other_bbs)
        # distance from center to own bumper along each ray (exit point), so
        # readings measure from the vehicle body instead of its center
        dist_self = laser_scan(v.state[0:2], 10, angles_aligned, v.self_bbs[None, :, :], zero_if_inside=False)
        return (dist_all - dist_self), other_bbs

    def get_relative_des(self, v):
        dx = v.destination[0] - v.state[0]
        dy = v.destination[1] - v.state[1]
        # Rotate by -ego_heading so that ego faces local +x
        cos_theta = math.cos(-v.state[2])
        sin_theta = math.sin(-v.state[2])
        x_rel = dx * cos_theta - dy * sin_theta
        y_rel = dx * sin_theta + dy * cos_theta
        # Relative heading = heading_dest - heading_ego (then wrap to [-pi, pi))
        heading_rel = v.destination[2] - v.state[2]
        heading_rel = (heading_rel + math.pi) % (2 * math.pi) - math.pi
        # observable right-of-way cues: distance to the conflict box and speed
        # toward the intersection center (arrival order is inferable from
        # these). The bearing-to-destination sin/cos pair was removed in
        # run 20 (pure derivative of x_rel/y_rel); ego speed added — it
        # previously appeared nowhere in the observation directly.
        d_box = dist_to_conflict_box(v.state[0], v.state[1]) / 70.0
        appr = approach_speed(v.state) / 16.0
        # density features (run 32): current lane half, destination exit lane
        # half, and conflict-box interior occupancy — congestion context the
        # per-neighbor tokens only carry implicitly
        cur_key = _lane_half_key(v.state[0], v.state[1])
        tgt_key = (v.destination_zone_id, EXIT_HALF.get(v.destination_zone_id, 0))
        cur_d = tgt_d = box_d = 0
        for o in self.vehicles.values():
            if o.id == v.id:
                continue
            ok = _lane_half_key(o.state[0], o.state[1])
            if ok is None and 60 <= o.state[0] <= 80 and 60 <= o.state[1] <= 80:
                box_d += 1
            if ok is not None and ok == cur_key:
                cur_d += 1
            if ok is not None and ok == tgt_key:
                tgt_d += 1
        return [x_rel, y_rel, math.sin(heading_rel), math.cos(heading_rel),
                d_box, appr, v.state[3] / 16.0,
                cur_d / 8.0, tgt_d / 8.0, box_d / 4.0]
    
    def get_waypoints(self, v, t_interval):
        waypoints = generate_waypoints(v, t_interval)
        return waypoints
    
    def collision_detector(self, v):
        return rects_overlap_any(v.self_bbs, v.other_bbs)

    def reached_destination(self, v):
        if v.zone_id == v.destination_zone_id:
            return (v.state[0] > 140) or (v.state[0] < 0) or (v.state[1] > 140) or (v.state[1] < 0)

    def is_outside(self, v):
        return not (0 <= v.state[0] <= 140 and 0 <= v.state[1] <= 140)

    def normalize_obs(self, obs, laser_max=20.0, distance_max=70.0):
        # ----- Normalize BEV -----
        obs["rl_bev"][2] /= 255.0  # other vehicle vel parallel
        obs["rl_bev"][3] /= 255.0  # other vehicle vel perpendicular
        obs["rl_bev"][5] /= 255.0  # ego velocity history

        # ----- Normalize laser -----
        obs["rl_lasers"] = obs["rl_lasers"] / laser_max

        # ----- Normalize des_info (waypoints are already ego-frame /70) -----
        obs["rl_des_info"][0:2] = obs["rl_des_info"][0:2] / distance_max
        return obs
        
    def get_single_obs(self, v):
        # refresh own bounding box to the current pose (used by laser + collision)
        v.self_bbs = self.get_bbs_to_self(v)
        if getattr(self, 'ablate_bev', False):
            # information ablation (run 18): BEV zeroed, architecture unchanged
            bev = np.zeros((self.BEV_size, self.BEV_size, 7), dtype=np.uint8)
        else:
            bev = create_bev_grid(
                vehicles=self.vehicles,
                ego_id=v.id,
                prohibited_zones=self.prohibited_zones,
                grid_size=self.BEV_size,
                resolution=self.BEV_resolution
            )
        v.lasers, v.other_bbs = self.get_laser(v)
        des_info = self.get_relative_des(v)
        # lookahead points from the Frenet reference route (ego frame, /70),
        # consistent with the candidate trajectories the actor chooses from
        waypoints = ref_waypoints(v)
        # collision detection
        collision = self.collision_detector(v)
        goal = self.reached_destination(v)
        outside = self.is_outside(v)

        v.collision = collision
        v.goal = goal
        v.outside = outside

        unnorm_obs = {
            "rl_bev": np.transpose(bev, (2,0,1)).astype(np.float32),
            "rl_lasers": v.lasers.astype(np.float32),
            "rl_des_info": np.array(des_info, dtype=np.float32),
            "rl_waypoints": np.array(waypoints, dtype=np.float32).flatten()
        }
        current_obs = self.normalize_obs(unnorm_obs)
        v.current_obs = current_obs
        # NOTE: alive is decremented in act() only; decrementing here too made
        # RL vehicles age twice per step, which made the goal unreachable
        return v
            
    def act_all(self, dt):
        for v in self.vehicles.values():
            if not v.rl_flag:
                if v.maneuver == 0:
                    v.act(dt, method='kinematic')
                elif getattr(self, 'idm_ref_steering', True):
                    # reference-path pure pursuit over the whole route
                    # (default since 2026-07-21; False = legacy Bezier turns)
                    v.act(dt, method='ref_pursuit')
                else:
                    if (v.zone_id == -1) or (v.zone_id == v.destination_zone_id):
                        v.act(dt, method='inverse')
                    else:
                        v.act(dt, method='kinematic')
            else:
                v.act(dt, method='kinematic')

    def rl_offset(self, offset):
        # only if there is one rl_agent
        self.rl_agent.origin[2]+= random.uniform(offset['min'], offset['max'])
        self.rl_agent.state = self.rl_agent.origin.copy()

    def set_IDM_config(self):
        """Sample one vehicle's randomized IDM parameters (stable per vehicle)."""
        return np.array([self.IDM_a_max + random.uniform(-0.2, 0.2),
                         self.IDM_desired_velocity + random.uniform(-2, 2),
                         self.IDM_delta + random.randint(-2, 2),
                         self.IDM_s0 + random.uniform(-0.5, 0.5),
                         self.IDM_T + random.uniform(-0.2, 0.2),
                         self.IDM_b + random.uniform(0, 0.5)])


def zone_assign(state):
    x, y = state[:2]
    if 60 <= y <= 80:
        if x > 80:
            return 0
        elif x < 60:
            return 2
    elif 60 <= x <= 80:
        if y < 60:
            return 1
        elif y > 80:
            return 3
    return -1
    

def maneuver_assign(origin_zone, destination_zone):
    '''
    0 -> go straight
    -1 -> turn left
    1 -> turn right
    '''
    if (destination_zone == -1) or (origin_zone == -1):
        return 0   # straight
    elif origin_zone == destination_zone:
        return 0
    else:
        if origin_zone == 0:  # from right
            if destination_zone == 2:
                return 0  # straight
            elif destination_zone == 1:
                return -1  # left
            elif destination_zone == 3:
                return 1  # right
        elif origin_zone == 1:  # from bottom
            if destination_zone == 3:
                return 0  # straight
            elif destination_zone == 2:
                return -1  # left
            elif destination_zone == 0:
                return 1  # right
        elif origin_zone == 2:  # from left
            if destination_zone == 0:
                return 0  # straight
            elif destination_zone == 3:
                return -1  # left
            elif destination_zone == 1:
                return 1  # right
        elif origin_zone == 3:  # from top
            if destination_zone == 1:
                return 0  # straight
            elif destination_zone == 0:
                return -1  # left
            elif destination_zone == 2:
                return 1  # right
