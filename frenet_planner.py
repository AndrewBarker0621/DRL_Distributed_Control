import numpy as np

# feature vector per candidate:
# 6 sub-sampled trajectory points in ego frame (12) +
# [d_f/3, v_f/16, dv/16, peak_acc/8, peak_curvature*10, jerk_lat, jerk_lon, emergency_flag]
# (a 21st min-clearance feature was tried in run 16 and reverted: battery
#  result within the noise band of run 15, mildly negative — under the HARD
#  collision mask, where admissible clearances barely differ; see v2)
CAND_FEAT_DIM = 20
# Planner generation v2 (2026-10, H23): with safety_feats=True three
# per-candidate safety features are APPENDED (the 20 legacy dims keep their
# positions, so downstream index use such as feats[:, 13] stays valid).
# All three come from the SAME neighbour rollout mask-3 uses (constant
# speed along each neighbour's own route, two-circle footprint):
#   [20] predicted minimum body clearance over neighbours x horizon,
#        clip(-5, 20) m / 10            (negative = circle footprints overlap)
#   [21] time to first predicted conflict (exactly the mask-3 test, margin
#        included) / horizon            (1 = none within the horizon)
#   [22] cos(relative heading) of the neighbour that sets the min clearance
#        (1 same direction, 0 crossing, -1 oncoming; 0 when no neighbour)
# Paired with soft_collision=True (mask-3 no longer deletes, except inside
# the first replan window) these give the policy a visible, selectable
# safety dimension that the reward can price. Changing F = new generation:
# v1 (G7) weights expect 20.
CAND_SAFETY_DIM = 3
CAND_FEAT_DIM_V2 = CAND_FEAT_DIM + CAND_SAFETY_DIM
CLEAR_CLIP = (-5.0, 20.0)
CLEAR_SCALE = 10.0
# Planner generation v3 (2026-10-17, H26): SCENE-RELATIVE candidates inserted
# between the grid and the stop candidates (the emergency stop stays LAST):
#   follow x len(follow_headways)  quintic longitudinal to a time headway behind
#          the leader on MY route (lane-free "same path": ahead within
#          leader_window, |Δd| < leader_lat, heading aligned with my route)
#   yield  x len(yield_gaps)       velocity-keeping to reach the conflict point
#          Δt AFTER the crossing vehicle has cleared it
#   go     x len(go_gaps)          velocity-keeping to reach it Δt BEFORE the
#          crossing vehicle arrives
#   pass   x 2*len(pass_clearances) keep speed, lateral quintic to pass the
#          reference vehicle (leader, else conflict vehicle) with a body
#          clearance, either side
# The conflict point is where the neighbour's predicted route (same constant-
# speed route model as mask-3) first comes within conflict_lat of my route.
# Each extra candidate adds CAND_V3_DIM feature dims: kind one-hot
# (follow, yield, go, pass) + its parameter (headway or gap /3 s, clearance /3 m).
# Extras without a scene partner keep a neutral trajectory and are masked out.
CAND_V3_DIM = 5
CAND_FEAT_DIM_V3 = CAND_FEAT_DIM_V2 + CAND_V3_DIM   # 28


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class ReferencePath:
    """
    Arc-length parameterized reference line built from the vehicle's intended
    route (straight line or the same cubic Bezier used by waypoint.py).
    Extended backward/forward so projection and sampling stay well-defined
    near spawn and past the destination.
    """
    def __init__(self, points, step=0.5, extend_back=6.0, extend_fwd=40.0):
        pts = np.asarray(points, dtype=float)
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        keep = np.concatenate([[True], seg > 1e-6])
        pts = pts[keep]

        t0 = pts[1] - pts[0]
        t0 = t0 / np.linalg.norm(t0)
        t1 = pts[-1] - pts[-2]
        t1 = t1 / np.linalg.norm(t1)
        pts = np.vstack([pts[0] - t0 * extend_back, pts, pts[-1] + t1 * extend_fwd])

        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s_raw = np.concatenate([[0.0], np.cumsum(seg)])

        n = int(s_raw[-1] / step) + 2
        self.s = np.linspace(0.0, s_raw[-1], n)
        self.x = np.interp(self.s, s_raw, pts[:, 0])
        self.y = np.interp(self.s, s_raw, pts[:, 1])
        hx = np.gradient(self.x)
        hy = np.gradient(self.y)
        self.theta = np.unwrap(np.arctan2(hy, hx))

    def project(self, x, y):
        """Return the Frenet (s, d) of a cartesian point."""
        d2 = (self.x - x) ** 2 + (self.y - y) ** 2
        i = int(np.argmin(d2))
        tx, ty = np.cos(self.theta[i]), np.sin(self.theta[i])
        dxp, dyp = x - self.x[i], y - self.y[i]
        s = self.s[i] + dxp * tx + dyp * ty
        d = -dxp * ty + dyp * tx
        return s, d

    def sample(self, s):
        """Return (x, y, theta) arrays at the given arc lengths."""
        s = np.clip(s, 0.0, self.s[-1])
        x = np.interp(s, self.s, self.x)
        y = np.interp(s, self.s, self.y)
        th = np.interp(s, self.s, self.theta)
        return x, y, th

    def heading_at(self, s):
        return float(np.interp(np.clip(s, 0.0, self.s[-1]), self.s, self.theta))


# cardinal travel direction when coming FROM each origin zone
_DIR_IN = {0: (-1.0, 0.0), 1: (0.0, 1.0), 2: (1.0, 0.0), 3: (0.0, -1.0)}
# cardinal travel direction when exiting INTO each destination zone
_DIR_OUT = {0: (1.0, 0.0), 1: (0.0, -1.0), 2: (-1.0, 0.0), 3: (0.0, 1.0)}

ARC_SETBACK = 6.0    # the turn arc starts this far BEFORE the conflict-box edge,
                     # giving long vehicles room to finish a left turn inside
                     # their own destination lane instead of the oncoming one
R_MIN, R_MAX = 11.0, 24.0
MAP_OVERRUN = 10.0   # how far the route continues past the map boundary


def _approach(p0, A, din, n=100):
    """
    Tangent-continuous approach from the spawn point to A: a cubic Bezier with
    both handles along the travel direction, so any lateral shift (spawn lane
    clipped off the walls) blends in smoothly instead of leaving a heading
    kink where it meets the straight/arc segment that follows.
    """
    p0 = np.asarray(p0, dtype=float)
    A = np.asarray(A, dtype=float)
    g = max(np.linalg.norm(A - p0) / 3.0, 1.0)
    t = np.linspace(0.0, 1.0, n)[:, None]
    p1, p2 = p0 + g * din, A - g * din
    return ((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1
            + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * A)


def build_reference_path(v):
    """
    Static, kinematically feasible route that stays on the road cross.
    Success in this env means crossing the map boundary inside the destination
    zone, so the route does NOT aim at the exact destination point:
      straight: approach (lateral clipped off the walls) -> straight lane
                keeping, parallel to the corridor, out past the boundary
      turn:     straight approach -> constant-radius 90-degree arc inside the
                conflict box -> straight run parallel to the exit corridor,
                out past the boundary
    The old waypoint.py Bezier (origin-based control points) cuts through the
    prohibited corner on left turns and is not used here.
    """
    p0 = np.asarray(v.origin[:2], dtype=float)
    p3 = np.asarray(v.destination[:2], dtype=float)
    if (v.origin_zone_id == -1) or (v.origin_zone_id == v.destination_zone_id):
        t = np.linspace(0.0, 1.0, 400)[:, None]
        return ReferencePath((1 - t) * p0 + t * p3)

    din = np.asarray(_DIR_IN[v.origin_zone_id])

    lat_axis = 0 if din[1] != 0 else 1          # coordinate index perpendicular to din
    lon_axis = 1 - lat_axis
    lat_in = float(np.clip(p0[lat_axis], 62.0, 78.0))

    if v.maneuver == 0:
        # lane keeping parallel to the corridor, across the map and beyond
        E = np.zeros(2)
        E[lat_axis] = lat_in
        E[lon_axis] = 60.0 if din[lon_axis] > 0 else 80.0
        far = E + din * (140.0 - 60.0 + MAP_OVERRUN)
        return ReferencePath(np.vstack([_approach(p0, E, din), far[None, :]]))

    dout = np.asarray(_DIR_OUT[v.destination_zone_id])
    # left turn if din x dout is positive (CCW)
    turn = 1.0 if (din[0] * dout[1] - din[1] * dout[0]) > 0 else -1.0
    n_arc = np.array([-din[1], din[0]]) * turn   # unit vector from E toward arc center

    # left turns curve toward the near corner while still outside the box, so
    # keep the arc start 4 m off that corner's wall
    if n_arc[lat_axis] < 0:
        lat_in = max(lat_in, 64.0)
    else:
        lat_in = min(lat_in, 76.0)

    # arc starts ARC_SETBACK before the box edge (= "entry line moved back"),
    # which lets the 90-degree arc exit inside the vehicle's own destination
    # lane instead of the oncoming lane
    E = np.zeros(2)
    E[lat_axis] = lat_in
    E[lon_axis] = (60.0 - ARC_SETBACK) if din[lon_axis] > 0 else (80.0 + ARC_SETBACK)

    # aim the arc exit at the destination lane, 2.5 m off that lane's edges;
    # exit lateral = E[lon_axis] +/- R
    dest_lat_axis = 0 if dout[1] != 0 else 1
    dest_lat = float(p3[dest_lat_axis])
    band_lo = 60.0 if dest_lat < 70.0 else 70.0
    target = float(np.clip(dest_lat, band_lo + 2.5, band_lo + 7.5))
    R = float(np.clip(abs(target - E[lon_axis]), R_MIN, R_MAX))

    center = E + R * n_arc
    a0 = np.arctan2((E - center)[1], (E - center)[0])
    phi = a0 + turn * np.linspace(0.0, np.pi / 2, 80)
    arc = center[None, :] + R * np.stack([np.cos(phi), np.sin(phi)], axis=1)
    Q = arc[-1]

    # straight run parallel to the exit corridor until past the map boundary
    exit_lon_axis = 1 - dest_lat_axis
    map_exit = 140.0 + MAP_OVERRUN if dout[exit_lon_axis] > 0 else -MAP_OVERRUN
    dist = (map_exit - Q[exit_lon_axis]) * dout[exit_lon_axis]
    Q2 = Q + max(dist, 2.0) * dout

    # smooth approach ends 10 m before the arc, aligned with the lane direction
    A = E - 10.0 * din
    pts = np.vstack([_approach(p0, A, din), arc, Q2[None, :]])
    return ReferencePath(pts)


def dist_to_conflict_box(x, y):
    """Euclidean distance to the conflict box [60,80]^2 (0 inside)."""
    dx = max(60.0 - x, 0.0, x - 80.0)
    dy = max(60.0 - y, 0.0, y - 80.0)
    return float(np.hypot(dx, dy))


def predict_plan_points(o, ts, dt=0.1):
    """
    World-frame future positions (len(ts), 2) of a vehicle at horizons `ts`.

    Plan-sharing source of truth (G7 observation): an RL vehicle reports
    sample points of its COMMITTED candidate trajectory (plan_traj, offset by
    the substeps already executed). A vehicle without a live plan — IDM, an
    RL vehicle before its first replan, or one whose plan has expired (a
    recovery-layer takeover stops replanning, so plan_k outgrows the horizon)
    — is predicted along its own reference route at current speed, the same
    model the intent-aware collision mask uses, with the same >60-degree
    misalignment fallback to straight-line extrapolation.
    """
    ts = np.asarray(ts, dtype=float)
    out = np.empty((len(ts), 2))
    traj = getattr(o, 'plan_traj', None)
    k = getattr(o, 'plan_k', None)
    if traj is not None and k is not None and k < len(traj):
        idx = np.minimum(k + np.round(ts / dt).astype(int), len(traj) - 1)
        out[:] = np.asarray(traj)[idx, :2]
        return out
    ref = getattr(o, 'frenet_ref', None)
    if ref is None:
        ref = build_reference_path(o)
        o.frenet_ref = ref
    s0, d0 = ref.project(o.state[0], o.state[1])
    if np.cos(o.state[2] - ref.heading_at(s0)) > 0.5:
        rx, ry, rh = ref.sample(s0 + max(o.state[3], 0.0) * ts)
        out[:, 0] = rx - d0 * np.sin(rh)
        out[:, 1] = ry + d0 * np.cos(rh)
    else:
        out[:, 0] = o.state[0] + o.state[3] * ts * np.cos(o.state[2])
        out[:, 1] = o.state[1] + o.state[3] * ts * np.sin(o.state[2])
    return out


def approach_speed(state):
    """
    Speed component toward the intersection center (70, 70).
    Positive = approaching, negative = leaving. Together with the distance to
    the conflict box this makes arrival order (priority) inferable from
    observable quantities only.
    """
    x, y, th, v = state
    bearing = np.arctan2(70.0 - y, 70.0 - x)
    return float(v * np.cos(th - bearing))


def time_to_conflict_box(state, cap=20.0):
    """
    Bounded time-to-arrival at the conflict box, from observable state only.
    0 when already inside; `cap` when stopped or moving away (semantically
    "not arriving"). Feeds the urgency / arrival-order neighbor features.
    """
    d = dist_to_conflict_box(state[0], state[1])
    if d <= 0.0:
        return 0.0
    a = approach_speed(state)
    if a <= 0.1:
        return cap
    return min(d / a, cap)


def ref_waypoints(v, n=10, spacing=5.0):
    """
    n points sampled along the vehicle's Frenet reference route ahead of its
    current position, in the ego frame, normalized by /70. Replaces the old
    waypoint.py Bezier lookahead (which cuts the prohibited corner on left
    turns and disagrees with the candidate trajectories).
    """
    ref = getattr(v, 'frenet_ref', None)
    if ref is None:
        ref = build_reference_path(v)
        v.frenet_ref = ref
    x, y, th = float(v.state[0]), float(v.state[1]), float(v.state[2])
    s0, _ = ref.project(x, y)
    s = s0 + spacing * np.arange(1, n + 1)
    px, py, _ = ref.sample(s)
    c, si = np.cos(-th), np.sin(-th)
    ex = (px - x) * c - (py - y) * si
    ey = (px - x) * si + (py - y) * c
    return (np.stack([ex, ey], axis=1) / 70.0).astype(np.float32).flatten()


def track_action(state, traj, k, dt, veh_length, lookahead=3):
    """
    Low-level tracking of a selected candidate: acceleration from the
    trajectory's velocity profile, steering from pure pursuit toward a
    lookahead point. Caller clips to action limits.
    """
    x, y, th, vel = state
    H = len(traj)
    ki = min(k, H - 1)
    li = min(k + lookahead, H - 1)

    acc = (traj[ki, 3] - vel) / dt

    dx, dy = traj[li, 0] - x, traj[li, 1] - y
    Ld = np.hypot(dx, dy)
    if Ld < 0.3:
        return acc, 0.0
    alpha = wrap_angle(np.arctan2(dy, dx) - th)
    L = 0.8 * veh_length
    ste = np.arctan2(2.0 * L * np.sin(alpha), Ld)
    return acc, ste


class FrenetPlanner:
    """
    Werling-style candidate generation in the Frenet frame of the vehicle's
    reference route:
      - lateral: quintic d(t) from (d0, d0_dot, 0) to (d_f, 0, 0)
      - longitudinal: quartic velocity-keeping s(t) toward target speed v_f
    plus constant-deceleration stop candidates along the route (lateral
    return to the centerline): soft stops in `stop_decel_set` pass the normal
    collision mask and let the policy express short gentle stops in queues
    (the quartic family's minimum travel is ~3 m at low speed, which does not
    fit a 3-4 m queue gap — run-23 jams showed 27% of decisions forced to
    emergency-only). The final, hardest one (`emergency_decel`) is exempt
    from the collision mask so the action set is never empty.
    """
    def __init__(self, env_config, veh_config,
                 d_set=(-3.0, -1.5, 0.0, 1.5, 3.0),
                 # 18 m/s tier added for the flow push (run 30): the quartic's
                 # peak accel 1.5*dv/T exceeds the 8 m/s^2 limit below ~2.3 m/s,
                 # so the kinematic mask auto-gates it to already-moving cars
                 v_set=(0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0),
                 horizon=3.0,
                 corridor_margin=1.0,
                 centerline_mask=True,      # forbid candidates entering the
                 centerline_margin=0.5,     # oncoming half while in an arm
                 centerline_grace=6,        # candidate points exempt from the
                                            # centerline check (escape window,
                                            # x dt = seconds): a vehicle on the
                                            # line must CLEAR it within this
                                            # window. 12 (1.2 s) tried in run
                                            # 38 and REJECTED: battery tied,
                                            # hugging 2.14%->3.75% — the longer
                                            # window just re-opens line dwell
                 centerline_body_clear=8.0, # beyond this distance from the box
                                            # the whole BODY must clear the
                                            # centerline (margin -width/2, kills
                                            # line-hugging); near the box the
                                            # forgiving margin above applies so
                                            # turn entries stay legal. None =
                                            # center-based rule everywhere (run 33)
                 stop_decel_set=(1.5, 3.0),
                 emergency_decel=5.0,
                 collision_margin=0.5,      # 0.3 until run 34: too thin for
                                            # 8-14 m/s hot entries (gen v2)
                 same_dir_margin=None,      # B.2 platoon densification
                                            # (RUNBOOK S7.17): 0.3 gives
                                            # same-direction pairs the tight
                                            # 0.6 m pair total while crossing
                                            # keeps 1.0 m — part of the +12%
                                            # capacity package; user keeps
                                            # the uniform default (None)
                 collision_horizon=3.0,
                 intent_aware_mask=True,    # neighbours predicted along their
                                            # OWN route, not straight-line
                 plan_aware_mask=False,     # run 47, REJECTED: predicting RL
                                            # neighbours by their COMMITTED plan
                                            # couples every mask to every plan —
                                            # two yielding vehicles both see the
                                            # other braking, both unmask
                                            # aggressive candidates, both commit
                                            # simultaneously (v_loss 187, x4
                                            # collisions even after adaptation).
                                            # The decoupled constant-speed route
                                            # model is a STABLE common prior,
                                            # not an inaccuracy. Kept as switch
                 max_pred_vehs=20,
                 soft_collision=False,      # v2 (H23): mask-3 stops DELETING
                                            # predicted conflicts. Only a
                                            # conflict inside the first
                                            # col_hard_horizon seconds is still
                                            # hard-removed (one replan window —
                                            # the next decision cannot undo it);
                                            # everything else stays selectable,
                                            # is exposed through safety_feats
                                            # and is priced by the collision
                                            # reward. Corridor and kinematic
                                            # layers stay hard. The H22 bank
                                            # showed the hard layer left a
                                            # median 0.02 m clearance spread
                                            # among admissible candidates, so
                                            # no reward weight could buy a
                                            # safety preference
                 col_hard_horizon=0.2,
                 safety_feats=False,        # v2: append CAND_SAFETY_DIM dims
                 gen_v3=False,              # v3 (H26): scene-relative extras
                 follow_headways=(1.0, 1.5, 2.5),
                 yield_gaps=(1.0, 2.0),
                 go_gaps=(1.0, 2.0),
                 pass_clearances=(1.0, 2.0),
                 leader_window=40.0, leader_lat=2.0, leader_cos=0.9,
                 conflict_lat=2.5, conflict_horizon=6.0, conflict_reach=60.0,
                 v3_fix=False,            # H26 v3 (fixed): arrival-time yield/go solve + feasibility,
                                          # pass manoeuvre bounded by pass_lat_acc. False keeps
                                          # the pilot-v3 candidate semantics (model/v3/*)
                 pass_lat_acc=2.0,        # m/s^2 peak lateral acceleration of a pass manoeuvre
                 profile_horizon=None):   # manoeuvre duration Tp <= horizon (None = horizon):
                                          # speed/lateral profiles complete at Tp and hold, the
                                          # collision layer / features still look `horizon` ahead
        self.dt = env_config['dt']
        self.T = horizon
        self.H = int(round(horizon / self.dt))
        self.Tp = float(horizon) if profile_horizon is None else float(min(profile_horizon, horizon))
        self.v3_fix = bool(v3_fix)
        self.pass_lat_acc = float(pass_lat_acc)
        self.d_set = np.asarray(d_set, dtype=float)
        self.v_set = np.asarray(v_set, dtype=float)
        # all constant-decel candidates; the last (hardest) is the exempt one
        self.decel_set = tuple(stop_decel_set) + (emergency_decel,)
        self.gen_v3 = bool(gen_v3)
        self.follow_headways = tuple(follow_headways) if gen_v3 else ()
        self.yield_gaps = tuple(yield_gaps) if gen_v3 else ()
        self.go_gaps = tuple(go_gaps) if gen_v3 else ()
        self.pass_clearances = tuple(pass_clearances) if gen_v3 else ()
        self.leader_window, self.leader_lat, self.leader_cos = leader_window, leader_lat, leader_cos
        self.conflict_lat, self.conflict_horizon, self.conflict_reach = conflict_lat, conflict_horizon, conflict_reach
        self.n_extra = (len(self.follow_headways) + len(self.yield_gaps) + len(self.go_gaps)
                        + 2 * len(self.pass_clearances))
        self.K = len(self.d_set) * len(self.v_set) + self.n_extra + len(self.decel_set)

        self.acc_max = env_config['acc_lim']['max']
        self.ste_max = env_config['ste_lim']['max']
        self.corridor_margin = corridor_margin
        self.centerline_mask = centerline_mask
        self.centerline_margin = centerline_margin
        self.centerline_grace = centerline_grace
        self.centerline_body_clear = centerline_body_clear
        self.emergency_decel = emergency_decel
        self.collision_margin = collision_margin
        self.same_dir_margin = same_dir_margin
        # constant-velocity conflicts are checked over this window only.
        # 1.5 s was tried (run 8): it raises the mean selectable candidates
        # 12->16 but at true 14-vehicle density whole-run goal DROPPED
        # 39%->30% — the conservative filter earns its keep. Default stays at
        # the full horizon; the knob remains for future experiments.
        self.collision_horizon = collision_horizon
        self.intent_aware_mask = intent_aware_mask
        self.plan_aware_mask = plan_aware_mask
        self.max_pred_vehs = max_pred_vehs
        self.soft_collision = soft_collision
        self.col_hard_horizon = col_hard_horizon
        self.safety_feats = safety_feats
        self.feat_dim = (CAND_FEAT_DIM + (CAND_SAFETY_DIM if safety_feats else 0)
                         + (CAND_V3_DIM if gen_v3 else 0))

        self.t = np.arange(1, self.H + 1) * self.dt
        # 6 evenly spaced sample indices used for the feature vector
        self.feat_idx = (np.arange(1, 7) * self.H) // 6 - 1

        # quintic boundary-condition matrix for the lateral polynomial (over Tp)
        T = self.Tp
        self._lat_A = np.array([
            [T ** 3, T ** 4, T ** 5],
            [3 * T ** 2, 4 * T ** 3, 5 * T ** 4],
            [6 * T, 12 * T ** 2, 20 * T ** 3],
        ])

    def _kappa_max(self, veh_length):
        # steady-state curvature limit of kinematic.py's bicycle model at ste_max
        wb = 0.8 * veh_length
        lr = 0.6 * wb
        beta = np.arctan(0.6 * np.tan(self.ste_max))
        return 0.95 * np.sin(beta) / lr

    def get_ref(self, v):
        ref = getattr(v, 'frenet_ref', None)
        if ref is None:
            ref = build_reference_path(v)
            v.frenet_ref = ref
        return ref

    # ------------------------------------------------------------------ v3
    def _predict_route(self, o, times):
        """Neighbour positions (len(times), 2) under the mask-3 model: constant
        speed along its OWN reference route, straight-line fallback when its
        heading is >60 deg off the route tangent."""
        ref = self.get_ref(o)
        s0, d0 = ref.project(o.state[0], o.state[1])
        sp = max(float(o.state[3]), 0.0)
        if np.cos(o.state[2] - ref.heading_at(s0)) > 0.5:
            rx, ry, rh = ref.sample(s0 + sp * times)
            return np.stack([rx - d0 * np.sin(rh), ry + d0 * np.cos(rh)], axis=1)
        dxy = np.array([np.cos(o.state[2]), np.sin(o.state[2])])
        # o.state may be a plain list (IDM vehicles) — never index it with a tuple
        p0 = np.asarray(o.state[:2], dtype=float)
        return p0[None, :] + (sp * times)[:, None] * dxy[None, :]

    def _v3_extras(self, ref, v, others, s0, d0, s_dot0, d_dot0, d_e, dd_e, add_e):
        """Scene-relative candidates (follow / yield / go / pass) in the order
        of the constructor lists.  Returns dict of arrays of length n_extra:
        s, d, dd (n,H) profiles, v_lon, a_lon, a_lat (n,H), df, vf, kind (n,4),
        param (n,), valid (n,) and the scene summary for last_debug."""
        t, T, H = self.t, self.T, self.H
        n = self.n_extra
        L_self, W_self = float(v.size[0]), float(v.size[1])
        # manoeuvre duration Tp (<= T): profiles complete at Tp and hold
        Tp = self.Tp
        tl = np.minimum(t, Tp)
        tau = tl / Tp
        hold = t > Tp
        prof = 3 * tau ** 2 - 2 * tau ** 3
        s_int = np.where(hold, t - 0.5 * Tp, Tp * (tau ** 3 - 0.5 * tau ** 4))
        a_prof = np.where(hold, 0.0, (6 * tau - 6 * tau ** 2) / Tp)
        # neutral longitudinal (keep speed) and lateral (-> 0) profiles
        s_keep = s0 + s_dot0 * t
        v_keep = np.full(H, s_dot0)
        a_zero = np.zeros(H)

        # ---- scene: leader on my route, nearest crossing conflict
        leader, lead_s, lead_d = None, None, None
        conflict, c_s, c_in, c_out = None, None, None, None
        tc = np.arange(1, int(round(self.conflict_horizon / self.dt)) + 1) * self.dt
        ss = s0 + np.arange(0.0, self.conflict_reach, 0.5)
        rx, ry, _ = ref.sample(ss)
        route_pts = np.stack([rx, ry], axis=1)                              # (M,2)
        best_lead = np.inf
        best_c = np.inf
        for o in others:
            so, do = ref.project(o.state[0], o.state[1])
            aligned = np.cos(o.state[2] - ref.heading_at(so)) > self.leader_cos
            ahead = so - s0
            if aligned and 0.0 < ahead < self.leader_window and abs(do - d0) < self.leader_lat:
                if ahead < best_lead:
                    best_lead, leader, lead_s, lead_d = ahead, o, so, do
                continue
            if (o.state[0] - v.state[0]) ** 2 + (o.state[1] - v.state[1]) ** 2 > (self.conflict_reach + 10.0) ** 2:
                continue
            P = self._predict_route(o, tc)                                  # (Hc,2)
            dist = np.sqrt(((P[:, None, :] - route_pts[None, :, :]) ** 2).sum(-1))   # (Hc,M)
            hit = dist < self.conflict_lat
            if not hit.any():
                continue
            ti, mj = np.nonzero(hit)
            s_c = float(ss[mj.min()])
            if s_c - s0 < 2.0 or s_c >= best_c:
                continue
            best_c, conflict = s_c, o
            c_s, c_in, c_out = s_c, float(tc[ti.min()]), float(tc[ti.max()])

        S = np.tile(s_keep, (n, 1)); V = np.tile(v_keep, (n, 1)); A = np.zeros((n, H))
        D = np.tile(d_e, (n, 1)); DD = np.tile(dd_e, (n, 1)); ADD = np.tile(add_e, (n, 1))
        df = np.zeros(n); vf = np.full(n, s_dot0); kind = np.zeros((n, 4)); param = np.zeros(n)
        valid = np.zeros(n, dtype=bool)
        i = 0
        # ---- follow: quintic s(t) to a time headway behind the leader
        for hw in self.follow_headways:
            kind[i, 0] = 1.0; param[i] = hw / 3.0
            if leader is not None:
                v_l = max(float(leader.state[3]), 0.0)
                gap = 0.5 * L_self + 0.5 * float(leader.size[0]) + 1.0
                sT = lead_s + v_l * Tp - hw * v_l - gap
                if sT > s0 + 0.5:
                    coef = np.linalg.solve(self._lat_A, np.array([sT - s0 - s_dot0 * Tp, v_l - s_dot0, 0.0]))
                    a3, a4, a5 = coef
                    s_f = (s0 + s_dot0 * tl + a3 * tl ** 3 + a4 * tl ** 4 + a5 * tl ** 5
                           + v_l * np.maximum(t - Tp, 0.0))
                    v_f = np.where(hold, v_l, s_dot0 + 3 * a3 * tl ** 2 + 4 * a4 * tl ** 3 + 5 * a5 * tl ** 4)
                    a_f = np.where(hold, 0.0, 6 * a3 * tl + 12 * a4 * tl ** 2 + 20 * a5 * tl ** 3)
                    if v_f.min() > -0.1:
                        S[i], V[i], A[i] = s_f, np.maximum(v_f, 0.0), a_f
                        vf[i] = v_l; valid[i] = True
            i += 1
        # ---- yield / go: velocity-keeping family toward an arrival-time target
        L_m = 0.5 * L_self + 1.0
        for gaps, col, sign in ((self.yield_gaps, 1, +1), (self.go_gaps, 2, -1)):
            for g in gaps:
                kind[i, col] = 1.0; param[i] = g / 3.0
                if conflict is not None:
                    c_in_e, c_out_e = c_in, c_out
                    if self.v3_fix:
                        # the band test is on the other vehicle's CENTRE: its
                        # front enters the band half a length earlier and its
                        # rear leaves half a length later (plus my half width)
                        ext = (0.5 * float(conflict.size[0]) + 0.5 * W_self) / max(float(conflict.state[3]), 0.5)
                        c_in_e, c_out_e = max(c_in - ext, 0.0), c_out + ext
                    if sign > 0:
                        t_a, s_a = c_out_e + g, c_s - L_m
                    else:
                        t_a, s_a = max(c_in_e - g, 0.3), c_s + L_m
                    if self.v3_fix:
                        # fixed v3: be at s_a at time t_a following the SAME smoothstep
                        # speed profile the grid uses (s_dot0 -> v_t over Tp, then
                        # hold): solve the position equation for dv.  The pilot
                        # form v_t = distance / t_a took v_t for the average
                        # speed, which the smooth transition never delivers (a
                        # 'yield' from 12 m/s over 2 s overshot the conflict
                        # point by ~7 m), see RESULT.md H26.
                        if t_a <= Tp:
                            ta = t_a / Tp
                            den = Tp * (ta ** 3 - 0.5 * ta ** 4)
                        else:
                            den = t_a - 0.5 * Tp
                        dv = (s_a - s0 - s_dot0 * t_a) / max(den, 1e-3)
                        v_t = float(np.clip(s_dot0 + dv, 0.0, self.v_set.max()))
                    else:
                        # pilot v3 (model/v3/*): average-speed target
                        v_t = float(np.clip((s_a - s0) / t_a, 0.0, self.v_set.max()))
                    dv = v_t - s_dot0
                    S[i] = s0 + s_dot0 * t + dv * s_int
                    V[i] = s_dot0 + dv * prof
                    A[i] = dv * a_prof
                    feasible = True
                    if self.v3_fix:
                        # after clipping: a yield that still reaches the conflict
                        # point before the other vehicle has left it (or a go that
                        # cannot clear it in time) is not that manoeuvre -> masked
                        if t_a <= Tp:
                            s_at = s0 + s_dot0 * t_a + (dv * Tp) * (ta ** 3 - 0.5 * ta ** 4)
                        else:
                            s_at = s0 + s_dot0 * Tp + 0.5 * dv * Tp + v_t * (t_a - Tp)
                        feasible = (s_at <= s_a + 0.5) if sign > 0 else (s_at >= s_a - 0.5)
                    vf[i] = v_t; valid[i] = bool(feasible)
                i += 1
        # ---- pass: keep speed, lateral quintic beside the reference vehicle.
        #      The lateral manoeuvre is completed by the time I draw level with
        #      it (t_pass), then held; only meaningful when the reference
        #      vehicle sits inside my lateral band (a crossing vehicle 15 m to
        #      the side cannot be "passed beside").
        refveh = leader if leader is not None else conflict
        pass_ok = False
        if refveh is not None:
            s_r, d_ref = ref.project(refveh.state[0], refveh.state[1])
            v_r = max(float(refveh.state[3]), 0.0) if refveh is leader else 0.0
            pass_ok = abs(d_ref - d0) <= (self.d_set.max() - self.d_set.min())
            t_pass = float(np.clip((s_r - s0) / max(s_dot0 - v_r, 0.5), 1.0, Tp))
        for c in self.pass_clearances:
            for side in (+1.0, -1.0):
                kind[i, 3] = 1.0; param[i] = c / 3.0
                if pass_ok:
                    d_pass = float(np.clip(d_ref + side * (0.5 * W_self + 0.5 * float(refveh.size[1]) + c),
                                           self.d_set.min(), self.d_set.max()))
                    Tl = t_pass
                    if self.v3_fix:
                        # fixed v3: a quintic lateral shift of |dd| peaks at
                        # 5.77*|dd|/Tl^2 lateral acceleration; the pilot's 1 s
                        # floor allowed ~17 m/s^2 jerks the tracker could not
                        # follow (vehicles drifted into the corner zones).  Take
                        # at least the time a pass_lat_acc manoeuvre needs; if
                        # that exceeds Tp the pass is not available.
                        T_min = float(np.sqrt(5.7735 * abs(d_pass - d0) / self.pass_lat_acc))
                        if T_min > Tp:
                            i += 1
                            continue
                        Tl = float(np.clip(max(t_pass, T_min), 1.0, Tp))
                    A_l = np.array([[Tl ** 3, Tl ** 4, Tl ** 5],
                                    [3 * Tl ** 2, 4 * Tl ** 3, 5 * Tl ** 4],
                                    [6 * Tl, 12 * Tl ** 2, 20 * Tl ** 3]])
                    a3, a4, a5 = np.linalg.solve(A_l, np.array([d_pass - d0 - d_dot0 * Tl, -d_dot0, 0.0]))
                    tl = np.minimum(t, Tl)
                    D[i] = d0 + d_dot0 * tl + a3 * tl ** 3 + a4 * tl ** 4 + a5 * tl ** 5
                    DD[i] = np.where(t <= Tl, d_dot0 + 3 * a3 * tl ** 2 + 4 * a4 * tl ** 3 + 5 * a5 * tl ** 4, 0.0)
                    ADD[i] = np.where(t <= Tl, 6 * a3 * tl + 12 * a4 * tl ** 2 + 20 * a5 * tl ** 3, 0.0)
                    df[i] = d_pass; valid[i] = True
                i += 1
        scene = dict(leader=None if leader is None else int(leader.id),
                     conflict=None if conflict is None else int(conflict.id),
                     conflict_s=c_s, conflict_t_in=c_in, conflict_t_out=c_out)
        return dict(s=S, v=V, a=A, d=D, dd=DD, add=ADD, df=df, vf=vf, kind=kind,
                    param=param, valid=valid, scene=scene)

    def generate(self, v, manager):
        """
        Returns dict with:
          trajs: (K, H, 4) float32 [x, y, heading, speed]
          feats: (K, self.feat_dim) float32: the 20 legacy dims (trajectory
                 points in ego frame + shape descriptors), plus the 3 v2
                 safety dims when safety_feats=True
          mask:  (K,) bool, True = selectable
        """
        ref = self.get_ref(v)
        x, y, th, vel = [float(s) for s in v.state]
        s0, d0 = ref.project(x, y)
        dth = wrap_angle(th - ref.heading_at(s0))
        s_dot0 = max(vel * np.cos(dth), 0.0)
        d_dot0 = vel * np.sin(dth)

        t, T, H = self.t, self.T, self.H
        tau = t / T
        Tp = self.Tp
        tl = np.minimum(t, Tp)                                       # profile time (held after Tp)
        tau_p = tl / Tp
        hold = t > Tp
        Nd, Nv = len(self.d_set), len(self.v_set)

        # ---- longitudinal: quartic velocity keeping (smoothstep speed profile)
        dv = self.v_set - s_dot0                                     # (Nv,)
        prof = 3 * tau_p ** 2 - 2 * tau_p ** 3                       # (H,)
        # displacement integral of the smoothstep: Tp*(tau^3 - tau^4/2) up to Tp,
        # then the terminal speed is held -> (t - Tp/2)
        s_int = np.where(hold, t - 0.5 * Tp, Tp * (tau_p ** 3 - 0.5 * tau_p ** 4))
        a_prof = np.where(hold, 0.0, (6 * tau_p - 6 * tau_p ** 2) / Tp)
        v_lon = s_dot0 + dv[:, None] * prof[None, :]                 # (Nv,H)
        s_lon = s0 + s_dot0 * t[None, :] + dv[:, None] * s_int[None, :]
        a_lon = dv[:, None] * a_prof[None, :]

        # ---- lateral: quintic to each terminal offset
        B = np.stack([
            self.d_set - d0 - d_dot0 * Tp,
            np.full(Nd, -d_dot0),
            np.zeros(Nd),
        ])                                                           # (3,Nd)
        coef = np.linalg.solve(self._lat_A, B)                       # (3,Nd)
        a3, a4, a5 = coef[0][:, None], coef[1][:, None], coef[2][:, None]
        d_lat = d0 + d_dot0 * tl[None, :] + a3 * tl ** 3 + a4 * tl ** 4 + a5 * tl ** 5   # (Nd,H)
        dd_lat = np.where(hold, 0.0, d_dot0 + 3 * a3 * tl ** 2 + 4 * a4 * tl ** 3 + 5 * a5 * tl ** 4)
        add_lat = np.where(hold, 0.0, 6 * a3 * tl + 12 * a4 * tl ** 2 + 20 * a5 * tl ** 3)

        # ---- constant-deceleration stops along the route, lateral -> 0
        #      (soft stops + the mask-exempt emergency as the last entry)
        dec = np.asarray(self.decel_set)[:, None]                            # (Ne,1)
        v_e = np.clip(s_dot0 - dec * t[None, :], 0.0, None)                  # (Ne,H)
        ts = s_dot0 / dec
        s_e = np.where(t[None, :] < ts,
                       s0 + s_dot0 * t[None, :] - 0.5 * dec * t[None, :] ** 2,
                       s0 + s_dot0 ** 2 / (2 * dec))
        zero_i = int(np.argmin(np.abs(self.d_set - 0.0)))
        d_e, dd_e, add_e = d_lat[zero_i], dd_lat[zero_i], add_lat[zero_i]

        # ---- combine to cartesian
        bx, by, bth = ref.sample(s_lon.ravel())
        bx, by, bth = bx.reshape(Nv, H), by.reshape(Nv, H), bth.reshape(Nv, H)
        nx, ny = -np.sin(bth), np.cos(bth)                            # (Nv,H)

        X = bx[None, :, :] + d_lat[:, None, :] * nx[None, :, :]      # (Nd,Nv,H)
        Y = by[None, :, :] + d_lat[:, None, :] * ny[None, :, :]
        HD = bth[None, :, :] + np.arctan2(
            np.broadcast_to(dd_lat[:, None, :], (Nd, Nv, H)),
            np.maximum(np.broadcast_to(v_lon[None, :, :], (Nd, Nv, H)), 1e-3))
        SP = np.hypot(np.broadcast_to(v_lon[None, :, :], (Nd, Nv, H)),
                      np.broadcast_to(dd_lat[:, None, :], (Nd, Nv, H)))

        Ne = len(self.decel_set)
        ex, ey, eth = ref.sample(s_e.ravel())
        ex, ey, eth = ex.reshape(Ne, H), ey.reshape(Ne, H), eth.reshape(Ne, H)
        enx, eny = -np.sin(eth), np.cos(eth)
        Xe = ex + d_e[None, :] * enx
        Ye = ey + d_e[None, :] * eny
        HDe = eth + np.arctan2(np.broadcast_to(dd_e[None, :], (Ne, H)),
                               np.maximum(v_e, 1e-3))
        SPe = np.hypot(v_e, np.broadcast_to(dd_e[None, :], (Ne, H)))

        if self.gen_v3:
            # ---- v3 scene-relative extras, inserted between grid and stops
            others_v3 = [o for o in manager.vehicles.values() if o.id != v.id]
            ex3 = self._v3_extras(ref, v, others_v3, s0, d0, s_dot0, d_dot0, d_e, dd_e, add_e)
            n3 = self.n_extra
            gx, gy, gth = ref.sample(ex3['s'].ravel())
            gx, gy, gth = gx.reshape(n3, H), gy.reshape(n3, H), gth.reshape(n3, H)
            Xx = gx + ex3['d'] * (-np.sin(gth))
            Yx = gy + ex3['d'] * np.cos(gth)
            HDx = gth + np.arctan2(ex3['dd'], np.maximum(ex3['v'], 1e-3))
            SPx = np.hypot(ex3['v'], ex3['dd'])
            X = np.vstack([X.reshape(-1, H), Xx, Xe])                     # (K,H)
            Y = np.vstack([Y.reshape(-1, H), Yx, Ye])
            HD = np.vstack([HD.reshape(-1, H), HDx, HDe])
            SP = np.vstack([SP.reshape(-1, H), SPx, SPe])
        else:
            ex3 = None
            X = np.vstack([X.reshape(-1, H), Xe])                         # (K,H)
            Y = np.vstack([Y.reshape(-1, H), Ye])
            HD = np.vstack([HD.reshape(-1, H), HDe])
            SP = np.vstack([SP.reshape(-1, H), SPe])
        K = self.K

        # ---- mask 1: stay on the road cross (prohibited corner zones) and,
        #      in the ARMS, stay in your own half of the carriageway. The
        #      intersection plaza [60,80]^2 is exempt: every turn has to cross
        #      the centreline in there. Lane convention (left-hand traffic,
        #      see HANDOFF): westbound y<70, eastbound y>70, northbound x<70,
        #      southbound x>70; the legal half follows the candidate's own
        #      heading, so a vehicle mid-turn in an arm is judged by where it
        #      is actually pointing.
        m = self.corridor_margin
        ok_pts = (((X >= 60 + m) & (X <= 80 - m)) | ((Y >= 60 + m) & (Y <= 80 - m)))
        if self.centerline_mask:
            cosH, sinH = np.cos(HD), np.sin(HD)
            in_plaza = (X >= 60) & (X <= 80) & (Y >= 60) & (Y <= 80)
            h_arm = (Y >= 60) & (Y <= 80) & ~in_plaza      # horizontal arms
            v_arm = (X >= 60) & (X <= 80) & ~in_plaza      # vertical arms
            # run 37: deep in the arms the whole BODY must stay in its half
            # (a 2 m-wide vehicle centred 0.4 m inside the line still straddles
            # it — 99% of visible "crossings" in the 1600 replay were such
            # line-hugging, center-legal frames); near the box the forgiving
            # +0.5 m margin stays so setback-arc turn entries survive.
            if self.centerline_body_clear is not None:
                dbx = np.maximum(np.maximum(60.0 - X, 0.0), X - 80.0)
                dby = np.maximum(np.maximum(60.0 - Y, 0.0), Y - 80.0)
                far = np.hypot(dbx, dby) > self.centerline_body_clear
                cm = np.where(far, -v.size[1] / 2.0, self.centerline_margin)
            else:
                cm = self.centerline_margin
            bad_h = h_arm & (((cosH < -0.5) & (Y > 70 + cm))     # westbound in north half
                             | ((cosH > 0.5) & (Y < 70 - cm)))   # eastbound in south half
            bad_v = v_arm & (((sinH > 0.5) & (X > 70 + cm))      # northbound in east half
                             | ((sinH < -0.5) & (X < 70 - cm)))  # southbound in west half
            bad = bad_h | bad_v
            # escape grace: a vehicle already hugging/over the line keeps
            # candidates that CLEAR it within the window — without this, every
            # moving candidate's early points inherit the illegal position and
            # the vehicle would be forced into an emergency stop on the line.
            bad[:, :self.centerline_grace] = False
            ok_pts &= ~bad
        mask_corridor = ok_pts.all(axis=1)

        # ---- mask 2: kinematic feasibility (curvature + acceleration)
        dhd = np.abs(wrap_angle(np.diff(HD, axis=1)))
        ds = np.maximum(SP[:, :-1] * self.dt, 1e-6)
        kappa = np.where(ds > 0.05, dhd / ds, 0.0)
        acc_fd = np.diff(np.concatenate([np.full((K, 1), vel), SP], axis=1), axis=1) / self.dt
        mask_kin = ((kappa.max(axis=1) <= self._kappa_max(v.size[0]))
                    & (np.abs(acc_fd).max(axis=1) <= self.acc_max * 0.98))

        # ---- mask 3: predicted collision (two-circle footprint approximation).
        #      Neighbour rollout is INTENT-AWARE since run 34: each neighbour
        #      advances at its current speed along its OWN reference route
        #      (constant lateral offset), so a turning neighbour is predicted
        #      to turn instead of ghost-driving straight through the corner.
        #      This is consistent with the intent one-hot already shared in
        #      the observation (V2V framing). Straight-line fallback per
        #      neighbour when its heading is >60 deg off the route tangent
        #      (recovering/misaligned vehicles) or via intent_aware_mask=False.
        mask_col = np.ones(K, dtype=bool)        # v1 verdict (always computed)
        mask_col_applied = mask_col              # what actually enters `mask`
        need_safety = self.safety_feats or self.soft_collision
        # v2 per-candidate safety quantities; defaults = no neighbour at all
        clear_min = np.full(K, 60.0)
        t_hit = np.full(K, self.T)
        cos_rel = np.zeros(K)
        others = [o for o in manager.vehicles.values() if o.id != v.id]
        if others:
            others.sort(key=lambda o: (o.state[0] - x) ** 2 + (o.state[1] - y) ** 2)
            others = others[:self.max_pred_vehs]
            ov = np.array([max(o.state[3], 0.0) for o in others])
            oL = np.array([o.size[0] for o in others])
            oW = np.array([o.size[1] for o in others])

            Hc = min(int(round(self.collision_horizon / self.dt)), H)
            N = len(others)
            centers_o = np.empty((N, Hc, 2))
            dirs_o = np.empty((N, Hc, 2))                                        # per-step tangent
            for i, o in enumerate(others):
                if self.plan_aware_mask:
                    # committed-plan rollout (speed-aware) for neighbours with
                    # a live plan; expired plans (plan_k past the horizon,
                    # e.g. recovery takeovers) fall through to the route model
                    traj_o = getattr(o, 'plan_traj', None)
                    kk = getattr(o, 'plan_k', None)
                    if traj_o is not None and kk is not None and kk < len(traj_o):
                        idx = np.minimum(kk + np.arange(1, Hc + 1), len(traj_o) - 1)
                        centers_o[i, :, 0] = traj_o[idx, 0]
                        centers_o[i, :, 1] = traj_o[idx, 1]
                        dirs_o[i, :, 0] = np.cos(traj_o[idx, 2])
                        dirs_o[i, :, 1] = np.sin(traj_o[idx, 2])
                        continue
                use_ref = False
                if self.intent_aware_mask:
                    ref = self.get_ref(o)
                    s0, d0 = ref.project(o.state[0], o.state[1])
                    if np.cos(o.state[2] - ref.heading_at(s0)) > 0.5:
                        use_ref = True
                if use_ref:
                    rx, ry, rh = ref.sample(s0 + ov[i] * t[:Hc])
                    centers_o[i, :, 0] = rx - d0 * np.sin(rh)
                    centers_o[i, :, 1] = ry + d0 * np.cos(rh)
                    dirs_o[i, :, 0] = np.cos(rh)
                    dirs_o[i, :, 1] = np.sin(rh)
                else:
                    dxy = np.array([np.cos(o.state[2]), np.sin(o.state[2])])
                    centers_o[i] = o.state[:2] + (ov[i] * t[:Hc])[:, None] * dxy
                    dirs_o[i] = dxy
            off_o = (oL / 4)[:, None, None] * dirs_o                             # (N,Hc,2)
            # per-PAIR margin, folded into r_o so it can differ by neighbour:
            # same-direction pairs (within ~45 deg of the ego heading) get the
            # tight platoon total (2 x same_dir_margin = 0.6 m, the pre-run-34
            # value), crossing conflicts keep the full 2 x collision_margin
            margins = np.full(N, 2.0 * self.collision_margin)
            if self.same_dir_margin is not None:
                oth_now = np.array([o.state[2] for o in others])
                margins[np.cos(oth_now - v.state[2]) > 0.7] = 2.0 * self.same_dir_margin
            r_o = 0.5 * np.hypot(oL / 2, oW) + margins                           # (N,)

            ce = np.stack([X[:, :Hc], Y[:, :Hc]], axis=2)                        # (K,Hc,2)
            dir_e = np.stack([np.cos(HD[:, :Hc]), np.sin(HD[:, :Hc])], axis=2)   # (K,Hc,2)
            off_e = (v.size[0] / 4) * dir_e
            r_e = 0.5 * np.hypot(v.size[0] / 2, v.size[1])   # pair margin lives in r_o

            rr2 = (r_e + r_o)[None, :, None] ** 2                                # (1,N,1)
            hit = np.zeros((K, N, Hc), dtype=bool)
            d2min = np.full((K, N, Hc), np.inf) if need_safety else None
            for a in (ce + off_e, ce - off_e):
                for b in (centers_o + off_o, centers_o - off_o):
                    d2 = ((a[:, None, :, :] - b[None, :, :, :]) ** 2).sum(-1)    # (K,N,Hc)
                    hit |= d2 < rr2
                    if need_safety:
                        np.minimum(d2min, d2, out=d2min)
            mask_col = ~hit.any(axis=(1, 2))
            mask_col[-1] = True  # emergency candidate exempt from collision mask
            if self.soft_collision:
                # v2: hard-remove only conflicts inside the first replan
                # window; later ones are the policy's call (see safety feats)
                n_hard = int(round(self.col_hard_horizon / self.dt))
                mask_col_applied = ~hit[:, :, :n_hard].any(axis=(1, 2))
                mask_col_applied[-1] = True
            else:
                mask_col_applied = mask_col
            if need_safety:
                # body clearance of the two-circle footprints WITHOUT the pair
                # margin: 0 = circles touch, <0 = circle overlap
                r_o_raw = 0.5 * np.hypot(oL / 2, oW)                             # (N,)
                gap = np.sqrt(d2min) - (r_e + r_o_raw)[None, :, None]            # (K,N,Hc)
                gap_n = gap.min(axis=2)                                          # (K,N)
                n_star = gap_n.argmin(axis=1)                                    # (K,)
                clear_min = gap_n[np.arange(K), n_star]
                hit_t = hit.any(axis=1)                                          # (K,Hc)
                first = np.where(hit_t.any(axis=1), hit_t.argmax(axis=1), Hc)
                t_hit = np.where(first < Hc, (first + 1) * self.dt, self.T)
                oth_h = np.array([o.state[2] for o in others])
                cos_rel = np.cos(oth_h[n_star] - th)

        mask = mask_corridor & mask_kin & mask_col_applied
        if ex3 is not None:
            # v3 extras without a scene partner (or infeasible) are never selectable
            g0 = Nd * Nv
            mask[g0:g0 + self.n_extra] &= ex3['valid']
        mask[-1] = True  # guaranteed fallback
        self.last_debug = {
            "corridor": mask_corridor, "kinematic": mask_kin, "collision": mask_col,
            "collision_applied": mask_col_applied,
            "max_kappa": kappa.max(axis=1), "kappa_lim": self._kappa_max(v.size[0]),
        }
        if need_safety:
            self.last_debug.update({"clear": clear_min, "t_hit": t_hit, "cos_rel": cos_rel})
        if ex3 is not None:
            self.last_debug.update({"v3_valid": ex3['valid'], "v3_kind": ex3['kind'],
                                    "v3_scene": ex3['scene']})

        # ---- candidate features in ego frame
        rel_x = X[:, self.feat_idx] - x
        rel_y = Y[:, self.feat_idx] - y
        c, s = np.cos(-th), np.sin(-th)
        pts_ego = np.stack([rel_x * c - rel_y * s, rel_x * s + rel_y * c], axis=2) / 40.0  # (K,6,2)

        Ne = len(self.decel_set)
        if ex3 is not None:
            df_all = np.concatenate([np.repeat(self.d_set, Nv), ex3['df'], np.zeros(Ne)])
            vf_all = np.concatenate([np.tile(self.v_set, Nd), ex3['vf'], np.zeros(Ne)])
            a_lat_all = np.vstack([np.repeat(add_lat, Nv, axis=0), ex3['add'],
                                   np.repeat(add_e[None, :], Ne, axis=0)])
            a_lon_all = np.vstack([np.tile(a_lon, (Nd, 1)), ex3['a'],
                                   -np.asarray(self.decel_set)[:, None] * np.ones((1, H))])
        else:
            df_all = np.concatenate([np.repeat(self.d_set, Nv), np.zeros(Ne)])
            vf_all = np.concatenate([np.tile(self.v_set, Nd), np.zeros(Ne)])
            a_lat_all = np.vstack([np.repeat(add_lat, Nv, axis=0),
                                   np.repeat(add_e[None, :], Ne, axis=0)])
            a_lon_all = np.vstack([np.tile(a_lon, (Nd, 1)),
                                   -np.asarray(self.decel_set)[:, None] * np.ones((1, H))])
        jerk_lat = np.clip(np.abs(np.diff(a_lat_all, axis=1)).mean(axis=1) / self.dt / 10.0, 0, 2)
        jerk_lon = np.clip(np.abs(np.diff(a_lon_all, axis=1)).mean(axis=1) / self.dt / 10.0, 0, 2)
        # graded stop-hardness flag: soft stops 0.3/0.6, emergency 1.0
        emerg = np.zeros(K)
        emerg[-Ne:] = np.asarray(self.decel_set) / self.emergency_decel

        parts = [
            pts_ego.reshape(K, -1),
            (df_all / 3.0)[:, None],
            (vf_all / 16.0)[:, None],
            ((vf_all - vel) / 16.0)[:, None],
            (np.abs(acc_fd).max(axis=1) / self.acc_max)[:, None],
            np.clip(kappa.max(axis=1) * 10.0, 0, 2)[:, None],
            jerk_lat[:, None],
            jerk_lon[:, None],
            emerg[:, None],
        ]
        if self.safety_feats:
            parts += [
                (np.clip(clear_min, *CLEAR_CLIP) / CLEAR_SCALE)[:, None],
                (t_hit / self.T)[:, None],
                cos_rel[:, None],
            ]
        if ex3 is not None:
            kind_all = np.zeros((K, 4)); param_all = np.zeros(K)
            g0 = Nd * Nv
            kind_all[g0:g0 + self.n_extra] = ex3['kind']
            param_all[g0:g0 + self.n_extra] = ex3['param']
            parts += [kind_all, param_all[:, None]]
        feats = np.concatenate(parts, axis=1).astype(np.float32)

        trajs = np.stack([X, Y, HD, SP], axis=2).astype(np.float32)   # (K,H,4)
        return {"trajs": trajs, "feats": feats, "mask": mask, "frenet": (s0, d0)}
