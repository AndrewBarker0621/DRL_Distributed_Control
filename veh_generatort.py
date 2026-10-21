import random

from vehicle import maneuver_assign

# maneuver -> sampling weight (0 straight, -1 left, 1 right); None = uniform.
# {0: 0.4, -1: 0.4, 1: 0.2} is the Australian-road profile tested in RUNBOOK
# S7.17 (+12% capacity with the platoon margin); user chose to keep the
# uniform default for benchmark continuity.
TURN_WEIGHTS = None
import numpy as np
import math
from vehicle import Vehicle

def generate_veh(origin_lines, destination_lines, origin_yaws, destination_yaws, vehicle_length, vehicle_width, initial_vel, manager, rl_flag=None):

    if rl_flag is None:
        rl_flag = random.choice([True, False])

    # Vehicle dimensions (length, width)
    length = random.uniform(vehicle_length['min'], vehicle_length['max'])
    width = random.uniform(vehicle_width['min'], vehicle_width['max'])

    # Randomly select initial velocity within the specified range
    velocity = random.uniform(initial_vel['min'], initial_vel['max'])

    # Randomly select a spawn point with enough clear road. Existing vehicles
    # are checked at their CURRENT position (the old check compared against
    # v.origin — birthplaces — so a queue backed up to the map edge was
    # invisible and cars spawned at speed right into its tail; >half of all
    # collisions in the first training run happened within 2 s of spawning).
    # The ahead-gap covers the emergency-braking distance at the spawn speed
    # (5 m/s^2 nominal, margin included). If no approach has room, return
    # None and let the caller skip this spawn tick.
    ahead_gap = 6.0 + velocity ** 2 / 8.0
    max_attempt = 12
    # approach selection: least-loaded entry arm first (uniform random left
    # some approaches empty while others clumped); random tiebreak keeps the
    # demand pattern stochastic
    counts = {s: 0 for s in range(4)}
    for veh in manager.vehicles.values():
        oz = getattr(veh, 'origin_zone_id', None)
        if oz in counts and veh.zone_id == oz:   # still on its entry approach
            counts[oz] += 1
    order = sorted(range(4), key=lambda s: (counts[s], random.random()))
    origin_section_id = None
    for attempt in range(max_attempt):
        section_id = order[attempt % 4]
        origin_point = random_point_in_section(origin_lines[section_id])
        if spawn_clear(section_id, origin_point, manager, ahead_gap):
            origin_section_id = section_id
            break
    if origin_section_id is None:
        return None

    # Destination by maneuver weights (2026-07-28, Australian left-hand
    # traffic: the RIGHT turn crosses the oncoming stream and is the
    # capacity killer, so it gets 20% instead of the uniform 33%;
    # straight/left split the rest evenly). TURN_WEIGHTS = None restores
    # the uniform choice.
    dests = [s for s in range(4) if s != origin_section_id]
    if TURN_WEIGHTS is None:
        destination_section_id = random.choice(dests)
    else:
        w = [TURN_WEIGHTS[maneuver_assign(origin_section_id, d)] for d in dests]
        destination_section_id = random.choices(dests, weights=w, k=1)[0]
    destination_line = destination_lines[destination_section_id]
    destination_point = random_point_in_line(destination_line)
    # des align
    if destination_section_id == 0:
        destination_point[0] = 150
    elif destination_section_id == 1:
        destination_point[1] = -10
    elif destination_section_id == 2:
        destination_point[0] = -10
    else:
        destination_point[1] = 150

    yaw_origin = origin_yaws[origin_section_id]
    yaw_destination = destination_yaws[destination_section_id]

    origin = np.array([origin_point[0], origin_point[1], yaw_origin, velocity])
    destination = np.array([destination_point[0], destination_point[1], yaw_destination])
    size = np.array([length, width])

    return origin, destination, size, rl_flag


def random_point_in_line(line_points):
    # Get the min and max coordinates of the bounding box
    min_x = min(line_points, key=lambda p: p[0])[0]
    max_x = max(line_points, key=lambda p: p[0])[0]
    min_y = min(line_points, key=lambda p: p[1])[1]
    max_y = max(line_points, key=lambda p: p[1])[1]
    
    # Randomly pick a point inside the bounding box (set a boundary)
    if min_x == max_x:
        x = min_x
        y = random.uniform(min_y+1.3, max_y-1.3)
    elif min_y == max_y:
        x = random.uniform(min_x+1.3, max_x-1.3)
        y = min_y
    else:
        x = random.uniform(min_x+1.3, max_x-1.3)
        y = random.uniform(min_y+1.3, max_y-1.3)
    return np.array([x, y])


def random_point_in_section(section_points):
    # Get the min and max coordinates of the bounding box
    min_x = min(section_points, key=lambda p: p[0])[0]
    max_x = max(section_points, key=lambda p: p[0])[0]
    min_y = min(section_points, key=lambda p: p[1])[1]
    max_y = max(section_points, key=lambda p: p[1])[1]

    # Randomly pick a point with a margin from the edges. Origin lines are
    # degenerate on one axis (e.g. x fixed at 140); an inverted uniform range
    # there used to spawn vehicles outside the map, dead on arrival.
    x = min_x if max_x - min_x <= 5.0 else random.uniform(min_x + 2.5, max_x - 2.5)
    y = min_y if max_y - min_y <= 5.0 else random.uniform(min_y + 2.5, max_y - 2.5)

    return np.array([x, y])


def vehicles_overlap(x1, y1, length1, width1,
                     x2, y2, length2, width2):
    """
    Returns True if two vehicles (approximated by bounding circles) overlap.

    (x1, y1): center of vehicle 1
    length1, width1: size of vehicle 1
    (x2, y2): center of vehicle 2
    length2, width2: size of vehicle 2

    We approximate each vehicle by a circle of radius = half the diagonal = sqrt( (L/2)^2 + (W/2)^2 ).
    """
    r1 = math.sqrt((length1 / 2.0)**2 + (width1 / 2.0)**2)
    r2 = math.sqrt((length2 / 2.0)**2 + (width2 / 2.0)**2)

    dist_centers = math.hypot(x2 - x1, y2 - y1)
    return dist_centers < (r1 + r2)


def yaw_selection_in_conflict_zone(origin):
    x, y = origin

    if x == 70 or y == 70:
        id = random.choice([0, 1, 2, 3])
    elif x > 70 and y > 70:   #3
        id = random.choice([0, 1, 2])
    elif x > 70 and y < 70:   #0
        id = random.choice([1, 2, 3])
    elif x < 70 and y < 70:   #1
        id = random.choice([0, 2, 3])
    else:   #2
        id = random.choice([0, 1, 3])

    return id


# travel direction into the map from each origin section
_SPAWN_DIR = {0: (-1.0, 0.0), 1: (0.0, 1.0), 2: (1.0, 0.0), 3: (0.0, -1.0)}


def spawn_clear(origin_section_id, origin_point, manager, ahead_gap,
                lateral_gap=6.0, behind_gap=6.0):
    # lateral_gap 6.0 (was 2.5) since generator v2: the entry half is 10 m
    # wide, so 2.5 m allowed side-by-side spawns that looked glued together
    """
    True if no vehicle currently occupies the spawn lane segment from
    `behind_gap` behind the spawn point to `ahead_gap` ahead of it
    (ahead = the direction the new vehicle will travel). The lateral gate
    keeps parallel exit-lane traffic from blocking spawns.
    """
    dx, dy = _SPAWN_DIR[origin_section_id]
    for v in manager.vehicles.values():
        rx = v.state[0] - origin_point[0]
        ry = v.state[1] - origin_point[1]
        lon = rx * dx + ry * dy          # + = ahead of the spawn point
        lat = -rx * dy + ry * dx
        if abs(lat) <= lateral_gap and -behind_gap <= lon <= ahead_gap:
            return False
    return True
