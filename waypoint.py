import numpy as np


def bezier_curve(t, P0, P1, P2, P3):
    # Convert control points to NumPy arrays for element-wise operations
    P0, P1, P2, P3 = map(np.array, (P0, P1, P2, P3))
    return (1 - t)**3 * P0 + 3 * (1 - t)**2 * t * P1 + 3 * (1 - t) * t**2 * P2 + t**3 * P3


def bezier_curve_simplify(t, P0, P3):
    # Convert control points to NumPy arrays for element-wise operations
    P0, P3 = map(np.array, (P0, P3))
    return (1 - t)* P0 + t * P3


# def generate_waypoints(agent, interval):
#     if agent.maneuver == 0:
#         waypoints = [bezier_curve_simplify(t, agent.origin[0:2], agent.destination[0:2]) for t in interval]

#     else:
#         candidate = np.array([70, 70])
#         waypoints = [bezier_curve(t, agent.origin[0:2], candidate, candidate, agent.destination[0:2]) for t in interval]

#     return waypoints
def generate_first_waypoint(state, destination, zone_id, destination_zone_id, maneuver, t):
    if maneuver == 0:
        waypoints = bezier_curve_simplify(t, state[0:2], destination[0:2])
    elif (zone_id == -1) or (zone_id == destination_zone_id):
        waypoints = bezier_curve_simplify(t, state[0:2], destination[0:2])
    else:
        candidate_1 = get_candidate(zone_id, state)
        candidate_2 = get_candidate(destination_zone_id, destination)
        waypoints = bezier_curve(t, state[0:2], candidate_1, candidate_2, destination[0:2])

    return waypoints


def generate_waypoints(agent, interval):
    if agent.maneuver == 0:
        waypoints = [bezier_curve_simplify(t, agent.state[0:2], agent.destination[0:2]) for t in interval]
    elif (agent.zone_id == -1) or (agent.zone_id == agent.destination_zone_id):
        waypoints = [bezier_curve_simplify(t, agent.state[0:2], agent.destination[0:2]) for t in interval]
    else:
        candidate_1 = get_candidate(agent.zone_id, agent.state)
        candidate_2 = get_candidate(agent.destination_zone_id, agent.destination)
        waypoints = [bezier_curve(t, agent.state[0:2], candidate_1, candidate_2, agent.destination[0:2]) for t in interval]

    return waypoints


def get_candidate(zone_id, state):
    if zone_id == 0:
        return np.array([80, state[1]])
    elif zone_id == 1:
        return np.array([state[0], 60])
    elif zone_id == 2:
        return np.array([60, state[1]])
    elif zone_id == 3:
        return np.array([state[0], 80])
    else:
        return np.array([70, 70])