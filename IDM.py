import random
import math
import numpy as np


def calculate_path(manager):
    agent_paths = []
    for v in manager.vehicles.values():
        if not v.rl_flag:
            agent_path = [np.cos(v.origin[2]), np.sin(v.origin[2])]
            agent_paths.append(agent_path)

    return agent_paths


def collision_point_Generator(agent_1, agent_2):
    """
    Computes the intersection of the infinite lines determined by each vehicle's
    (x, y) position and heading.
    
    agent_1.state => [x1, y1, theta1, v1]
    agent_2.state => [x2, y2, theta2, v2]
    
    Returns [x_int, y_int] if the lines intersect, or None if parallel.
    """
    x1, y1, theta1, _ = agent_1.state
    x2, y2, theta2, _ = agent_2.state

    if ((x1 - x2) < 1e-6 and (y1 - y2) < 1e-6 and (theta1 - theta2) < 1e-6):
        # Return something to indicate lines coincide
        return None

    # Direction vectors from headings
    dx1 = np.cos(theta1)
    dy1 = np.sin(theta1)
    dx2 = np.cos(theta2)
    dy2 = np.sin(theta2)

    # Solve the system:
    #   (x1, y1) + t * (dx1, dy1) = (x2, y2) + s * (dx2, dy2)
    A = np.array([[dx1, -dx2],
                  [dy1, -dy2]])
    b = np.array([x2 - x1, y2 - y1])

    det = np.linalg.det(A)
    if np.abs(det) < 1e-6:
        # Lines are parallel or nearly so
        return None
    else:
        t_s = np.linalg.solve(A, b)
        t = t_s[0]
        # Intersection point along agent_1's line
        x_int = x1 + dx1 * t
        y_int = y1 + dy1 * t
        return [x_int, y_int]
    

def collision_point_Vector(manager):
    """
    Given a list of vehicle objects, compute the intersection (collision) points
    for every pair (i, j). If parallel or no intersection, store None.
    
    Returns: a 2D list collision_points[i][j] with intersection coords or None.
    """
    vehicle_num = manager.__len__()
    collision_points = [[None for _ in range(vehicle_num)] 
                             for _ in range(vehicle_num)]
    
    for i in range(vehicle_num):
        for j in range(i + 1, vehicle_num):
            cp = collision_point_Generator(manager.vehicles[i], manager.vehicles[j])
            # Store the result symmetrically
            collision_points[i][j] = cp
            collision_points[j][i] = cp
    
    return collision_points



def IDM_update(manager, yield_to_rl=False, eta_priority=False, gap_accept=False):
    delta_v = 1e-5

    for v in manager.vehicles.values():
        # dont consider ego vehicle
        if not v.rl_flag:
            a_max, desired_velocity, delta, s0, T, b = v.IDM_config
            vel = v.state[3]

            # Conflict deceleration
            # if highest priority than acc
            # if someone higher than me, then deacc and wait till i'm the highest
            if gap_accept == 'quad':
                # quadrant-resolved v2: REJECTED in run 46 (worse at every
                # demand even after adaptation, -10.3 pt at 2000) — kept
                # behind this switch for reference
                a_priority, s = priority_deceleration_gap4(v, manager, b)
            elif gap_accept:
                a_priority, s = priority_deceleration_gap(v, manager, b)
            elif eta_priority:
                a_priority, s = priority_deceleration_eta(v, manager, b)
            elif yield_to_rl:
                a_priority, s = priority_deceleration_yield_rl(v, manager, b)
            else:
                a_priority, s = priority_deceleration(v, manager, b)
            
            # Standard IDM acceleration
            s_star = s0 + vel * T + (vel * delta_v) / (2 * np.sqrt(a_max * b))
            acc = a_max * (1 - (vel / desired_velocity) ** delta - (s_star / s) ** 2)
    
            # Adjust acceleration to avoid collision
            acc -= a_priority

            v.acc = max(acc, -b)


def priority_deceleration(vehicle, manager, comfortable_deceleration):
    a_priority = 0
    s = np.inf
    for v in manager.vehicles.values():
        # dont consider ego vehicle
        if not v.rl_flag:
            if v.id != vehicle.id:
                # Check if the vehicle has already passed the collision zone
                if vehicle.zone_id == vehicle.destination_zone_id:
                    continue
                if v.zone_id == v.destination_zone_id:
                    continue

                if vehicle.priority >= v.priority:
                    if vehicle.zone_id == v.zone_id:
                        distance_to_collision_self = np.linalg.norm(np.array(v.state[:2]) - np.array(vehicle.state[:2]))
                    else:
                        distance_to_collision_self = np.linalg.norm(np.array([70, 70]) - np.array(vehicle.state[:2]) - 5)
                    # distance_to_collision_other = np.linalg.norm(cp_vector[id][other_id] - states[other_id,:2])

                    # if distance_to_collision_self <= 0:
                    #     continue
                    
                    required_deceleration = (vehicle.state[3] ** 2) / (2 * distance_to_collision_self)
                    a_priority = max(a_priority, min(required_deceleration, comfortable_deceleration))
                    s = min(s,distance_to_collision_self)

    return a_priority, s


def priority_deceleration_yield_rl(vehicle, manager, comfortable_deceleration):
    """
    Selectable variant of priority_deceleration that ALSO reacts to RL
    vehicles (the original is preserved above and scans non-RL only, so IDM
    drove straight through the RL car ahead — 93% of run-9 collisions were
    IDM-into-RL, 71% same-direction rear-ends).

    Added behavior: an RL vehicle in the same zone, ahead of us and heading
    the same way is treated as a followable leader (gap feeds the IDM
    car-following term and the emergency-deceleration cap). RL vehicles do
    NOT get conflict-box priority: their priority is inf and several are
    always en route, so full yielding would freeze IDM at the box entrance.
    IDM-vs-IDM behavior is identical to the original function.
    """
    a_priority, s = priority_deceleration(vehicle, manager, comfortable_deceleration)
    for v in manager.vehicles.values():
        if (not v.rl_flag) or (v.id == vehicle.id):
            continue
        if vehicle.zone_id != v.zone_id:
            continue
        dx = v.state[0] - vehicle.state[0]
        dy = v.state[1] - vehicle.state[1]
        head = vehicle.state[2]
        ahead = dx * np.cos(head) + dy * np.sin(head)
        same_dir = np.cos(v.state[2] - head) > 0.0
        if ahead <= 0 or not same_dir:
            continue
        gap = np.hypot(dx, dy)
        required_deceleration = (vehicle.state[3] ** 2) / (2 * max(gap, 1e-6))
        a_priority = max(a_priority, min(required_deceleration, comfortable_deceleration))
        s = min(s, gap)
    return a_priority, s


def _dist_to_box(x, y):
    dx = max(60.0 - x, 0.0, x - 80.0)
    dy = max(60.0 - y, 0.0, y - 80.0)
    return math.sqrt(dx * dx + dy * dy)


def _lane_half_key(x, y):
    """(arm, half) of a position on the cross; None inside the box / corners."""
    if 60 <= x <= 80 and 60 <= y <= 80:
        return None
    if 60 <= y <= 80 and (x > 80 or x < 60):
        return (0 if x > 80 else 2, 0 if y < 70 else 1)
    if 60 <= x <= 80 and (y < 60 or y > 80):
        return (1 if y < 60 else 3, 0 if x < 70 else 1)
    return None


def eta_priority_of(v, manager, queue_delay=2.0):
    """
    The negotiation priority (2026-07-28, shared helper): predicted arrival
    time at the conflict area — dist_to_box / approach_speed plus
    queue_delay per queued vehicle ahead in the same lane half. 0.0 inside
    the box (absolute right of way), None once past the conflict (out of
    the negotiation). This is THE quantity the ETA-IDM yields by, and the
    G8 priority broadcast exposes the very same number to RL observations.
    """
    if v.zone_id == v.destination_zone_id:
        return None
    vx, vy, vth, vv = v.state
    dv = _dist_to_box(vx, vy)
    if dv <= 0.0:
        return 0.0
    lk = _lane_half_key(vx, vy)
    n_ahead = 0
    if lk is not None:
        n_ahead = sum(1 for o in manager.vehicles.values()
                      if o.id != v.id
                      and _lane_half_key(o.state[0], o.state[1]) == lk
                      and _dist_to_box(o.state[0], o.state[1]) < dv)
    appr = vv * math.cos(vth - math.atan2(70.0 - vy, 70.0 - vx))
    return dv / max(appr, 1.0) + queue_delay * n_ahead


def priority_deceleration_eta(vehicle, manager, comfortable_deceleration,
                              activation_range=40.0, queue_delay=2.0):
    """
    Third selectable priority model (2026-07-23, user spec): within
    `activation_range` of the conflict box, right of way follows the
    PREDICTED ARRIVAL ORDER at the conflict area — not spawn order (FCFS)
    and not vehicle class. ETA = dist_to_box / approach_speed +
    queue_delay * (queued vehicles ahead in the same lane half), so lane
    density delays effective arrival. Vehicles already inside the box have
    ETA 0. Yielding applies toward ALL vehicles, RL included; same-lane
    car-following also covers every leader. Original FCFS and yield_rl
    variants are preserved above.
    """
    a_priority, s = 0.0, np.inf
    x, y, th, vel = vehicle.state

    # 1) same-lane car-following toward every leader (RL and IDM alike)
    for o in manager.vehicles.values():
        if o.id == vehicle.id or o.zone_id != vehicle.zone_id:
            continue
        fdx, fdy = o.state[0] - x, o.state[1] - y
        if fdx * math.cos(th) + fdy * math.sin(th) <= 0:
            continue
        if math.cos(o.state[2] - th) <= 0:
            continue
        gap = math.hypot(fdx, fdy)
        a_priority = max(a_priority,
                         min(vel * vel / (2 * max(gap, 1e-6)), comfortable_deceleration))
        s = min(s, gap)

    # 2) conflict-area negotiation by predicted arrival order
    if vehicle.zone_id == vehicle.destination_zone_id:
        return a_priority, s          # already past the conflict area
    d_self = _dist_to_box(x, y)
    if d_self > activation_range:
        return a_priority, s

    eta_self = eta_priority_of(vehicle, manager, queue_delay)
    for o in manager.vehicles.values():
        if o.id == vehicle.id:
            continue
        if o.zone_id == o.destination_zone_id:
            continue                   # already past its conflict
        d_o = _dist_to_box(o.state[0], o.state[1])
        if d_o > activation_range:
            continue
        if eta_priority_of(o, manager, queue_delay) < eta_self:  # they arrive first -> yield
            required = vel * vel / (2 * max(d_self, 1.0))
            a_priority = max(a_priority, min(required, comfortable_deceleration))
            s = min(s, d_self)
    return a_priority, s


# ---- movement-conflict matrix (gap-acceptance IDM, 2026-07-29) ----
# Two movements conflict iff their reference paths pass within 3 m of each
# other inside/near the box. Precomputed lazily over the 12 (origin, dest)
# movements from canonical lane-center entries; same-origin pairs are
# non-conflicting here because the car-following term owns that case.
_CONFLICT_CACHE = None


def _movement_conflicts():
    global _CONFLICT_CACHE
    if _CONFLICT_CACHE is None:
        from frenet_planner import build_reference_path
        entries = {0: (150.0, 65.0, math.pi), 1: (65.0, -10.0, math.pi / 2),
                   2: (-10.0, 75.0, 0.0), 3: (75.0, 150.0, -math.pi / 2)}
        dests = {0: (150.0, 75.0), 1: (75.0, -10.0), 2: (-10.0, 65.0), 3: (65.0, 150.0)}

        class _V:
            pass

        paths = {}
        for o in range(4):
            for d in range(4):
                if o == d:
                    continue
                v = _V()
                ex, ey, eh = entries[o]
                v.origin = np.array([ex, ey, eh, 10.0])
                v.origin_zone_id = o
                v.destination_zone_id = d
                v.destination = np.array([*dests[d], 0.0])
                v.maneuver = 0 if d == (o + 2) % 4 else 1
                ref = build_reference_path(v)
                pts = np.stack([ref.x, ref.y], axis=1)
                near = pts[(pts[:, 0] > 52) & (pts[:, 0] < 88)
                           & (pts[:, 1] > 52) & (pts[:, 1] < 88)]
                paths[(o, d)] = near
        cache = {}
        moves = list(paths)
        for m1 in moves:
            for m2 in moves:
                if m1[0] == m2[0]:
                    cache[(m1, m2)] = False      # same approach: following case
                    continue
                a, b = paths[m1], paths[m2]
                d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
                cache[(m1, m2)] = bool(d2.min() < 9.0)   # < 3 m
        _CONFLICT_CACHE = cache
    return _CONFLICT_CACHE


def priority_deceleration_gap(vehicle, manager, comfortable_deceleration,
                              activation_range=40.0, t_clear=3.0):
    """
    Gap-acceptance refinement of the ETA rule (2026-07-29, user spec: an
    IDM waiting at the box entrance should take any usable gap instead of
    waiting out the whole arrival stream). Yield to a vehicle ONLY when
    BOTH hold:
      1) its MOVEMENT genuinely conflicts with ours (path-proximity matrix),
      2) the gap is unusable: it arrives before us AND we would enter the
         conflict area within t_clear seconds behind it.
    Vehicles inside the box count with eta 0 (yield iff we'd arrive within
    t_clear while a CONFLICTING one is still crossing). Same-lane
    car-following toward every leader is kept unchanged; the queue-delay
    term is dropped here — queues resolve through following, and gap
    timing needs the kinematic arrival estimate, not the adjusted one.
    """
    a_priority, s = 0.0, np.inf
    x, y, th, vel = vehicle.state
    for o in manager.vehicles.values():
        if o.id == vehicle.id or o.zone_id != vehicle.zone_id:
            continue
        fdx, fdy = o.state[0] - x, o.state[1] - y
        if fdx * math.cos(th) + fdy * math.sin(th) <= 0:
            continue
        if math.cos(o.state[2] - th) <= 0:
            continue
        gap = math.hypot(fdx, fdy)
        a_priority = max(a_priority,
                         min(vel * vel / (2 * max(gap, 1e-6)), comfortable_deceleration))
        s = min(s, gap)

    if vehicle.zone_id == vehicle.destination_zone_id:
        return a_priority, s
    d_self = _dist_to_box(x, y)
    if d_self > activation_range:
        return a_priority, s

    conflicts = _movement_conflicts()
    my_move = (vehicle.origin_zone_id, vehicle.destination_zone_id)
    appr_e = vel * math.cos(th - math.atan2(70.0 - y, 70.0 - x))
    eta_e = d_self / max(appr_e, 1.0)

    for o in manager.vehicles.values():
        if o.id == vehicle.id or o.zone_id == o.destination_zone_id:
            continue
        if not conflicts.get((my_move, (o.origin_zone_id, o.destination_zone_id)), True):
            continue
        ox, oy, oth, ov = o.state
        d_o = _dist_to_box(ox, oy)
        if d_o > activation_range:
            continue
        if d_o <= 0.0:
            eta_o = 0.0
        else:
            appr_o = ov * math.cos(oth - math.atan2(70.0 - oy, 70.0 - ox))
            eta_o = d_o / max(appr_o, 1.0)
        if eta_o < eta_e < eta_o + t_clear:      # they precede AND gap unusable
            required = vel * vel / (2 * max(d_self, 1.0))
            a_priority = max(a_priority, min(required, comfortable_deceleration))
            s = min(s, d_self)
    return a_priority, s


# ---- quadrant-level gap acceptance (gap v2, 2026-07-29, user spec:
#      CfDCA's 16 zones judged too fine — four 10x10 sub-zones suffice) ----
_QUADS = ((60.0, 70.0, 60.0, 70.0), (70.0, 80.0, 60.0, 70.0),
          (60.0, 70.0, 70.0, 80.0), (70.0, 80.0, 70.0, 80.0))


def _quad_spans(v):
    """{quad_idx: (s_in, s_out)} arc-length window of each box quadrant along
    the vehicle's own reference path; cached per vehicle."""
    sp = getattr(v, '_quad_spans', None)
    if sp is None:
        ref = getattr(v, 'frenet_ref', None)
        if ref is None:
            from frenet_planner import build_reference_path
            ref = build_reference_path(v)
            v.frenet_ref = ref
        sp = {}
        for qi, (x0, x1, y0, y1) in enumerate(_QUADS):
            inside = ((ref.x >= x0) & (ref.x <= x1)
                      & (ref.y >= y0) & (ref.y <= y1))
            if inside.any():
                idx = np.where(inside)[0]
                sp[qi] = (float(ref.s[idx[0]]), float(ref.s[idx[-1]]))
        v._quad_spans = sp
    return sp


def priority_deceleration_gap4(vehicle, manager, comfortable_deceleration,
                               activation_range=40.0, t_clear=1.5):
    """
    Quadrant-resolved gap acceptance (gap v2). Same movement-conflict filter
    and same-lane following as gap v1, but the timing is per SHARED QUADRANT:
    yield iff, for some quadrant both paths cross, the conflicting vehicle
    enters it before us and we would enter within t_clear of it leaving.
    Two conflicting movements whose shared quadrant is time-separated now
    pass concurrently through different corners of the box — the whole-box
    window of gap v1 (t_clear 3.0 on box ENTRY times) blocked that. Pairs
    that conflict by the path matrix but share no quadrant (near-boundary
    geometry) fall back to the v1 box-entry rule.
    """
    a_priority, s = 0.0, np.inf
    x, y, th, vel = vehicle.state
    for o in manager.vehicles.values():
        if o.id == vehicle.id or o.zone_id != vehicle.zone_id:
            continue
        fdx, fdy = o.state[0] - x, o.state[1] - y
        if fdx * math.cos(th) + fdy * math.sin(th) <= 0:
            continue
        if math.cos(o.state[2] - th) <= 0:
            continue
        gap = math.hypot(fdx, fdy)
        a_priority = max(a_priority,
                         min(vel * vel / (2 * max(gap, 1e-6)), comfortable_deceleration))
        s = min(s, gap)

    if vehicle.zone_id == vehicle.destination_zone_id:
        return a_priority, s
    d_self = _dist_to_box(x, y)
    if d_self > activation_range:
        return a_priority, s

    conflicts = _movement_conflicts()
    my_move = (vehicle.origin_zone_id, vehicle.destination_zone_id)
    spans_e = _quad_spans(vehicle)
    s_e, _ = vehicle.frenet_ref.project(x, y)
    v_e = max(vel, 1.0)
    appr_e = vel * math.cos(th - math.atan2(70.0 - y, 70.0 - x))
    eta_e = d_self / max(appr_e, 1.0)

    for o in manager.vehicles.values():
        if o.id == vehicle.id or o.zone_id == o.destination_zone_id:
            continue
        if not conflicts.get((my_move, (o.origin_zone_id, o.destination_zone_id)), True):
            continue
        ox, oy, oth, ov = o.state
        d_o = _dist_to_box(ox, oy)
        if d_o > activation_range:
            continue
        spans_o = _quad_spans(o)
        shared = set(spans_e) & set(spans_o)
        must_yield = False
        if shared:
            s_o, _ = o.frenet_ref.project(ox, oy)
            v_o = max(ov, 1.0)
            for q in shared:
                if s_e > spans_e[q][1] or s_o > spans_o[q][1]:
                    continue                      # one of us already past q
                tin_e = max((spans_e[q][0] - s_e) / v_e, 0.0)
                tin_o = max((spans_o[q][0] - s_o) / v_o, 0.0)
                tout_o = (spans_o[q][1] - s_o) / v_o
                if tin_o <= tin_e < tout_o + t_clear:
                    must_yield = True
                    break
        else:
            # conflicting by path proximity but no common quadrant: v1 rule
            if d_o <= 0.0:
                eta_o = 0.0
            else:
                appr_o = ov * math.cos(oth - math.atan2(70.0 - oy, 70.0 - ox))
                eta_o = d_o / max(appr_o, 1.0)
            must_yield = eta_o < eta_e < eta_o + 3.0
        if must_yield:
            required = vel * vel / (2 * max(d_self, 1.0))
            a_priority = max(a_priority, min(required, comfortable_deceleration))
            s = min(s, d_self)
    return a_priority, s
