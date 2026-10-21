import numpy as np
import os
import csv
import yaml
import matplotlib.pyplot as plt
from typing import List, Sequence, Tuple, Union
Vector = Union[np.ndarray, Sequence[float]]
Matrix = Union[np.ndarray, Sequence[Sequence[float]]]
Interval = Union[
    np.ndarray,
    Tuple[Vector, Vector],
    Tuple[Matrix, Matrix],
    Tuple[float, float],
    List[Vector],
    List[Matrix],
    List[float],
]


def load_config_yaml(file_path):
    """Load the YAML configuration file."""
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)
    

def is_In_Range(car_pos, intersection_length, corner_length):
    """
    - car_pos: tuple (x, y) representing the car's current position.
    - intersection_length: float, length of a square intersection.
    - corner_length: float, length of a corner of the intersection.
    """
    # car_pos -> [x,y]
    if (car_pos[0] > corner_length) and (car_pos[0] < (corner_length + intersection_length)):
        if car_pos[1] <= (2 * corner_length + intersection_length) and car_pos[1] >= 0:
            return True
        else:
            return False
    elif (car_pos[1] > corner_length) and (car_pos[1] < (corner_length + intersection_length)):
        if car_pos[0] <= (2 * corner_length + intersection_length) and car_pos[0] >= 0:
            return True
        else:
            return False
    else:
        return False


def get_edges(rect):
    """Get the edge vectors of the rectangle"""
    return [rect[i] - rect[i - 1] for i in range(len(rect))]

def get_normals(edges):
    """Get the normal vectors of the edges"""
    return [np.array([-edge[1], edge[0]]) for edge in edges]

def project(rect, axis):
    """Project the vertices of the rectangle onto the specified axis"""
    return [np.dot(vertex, axis) for vertex in rect]

def is_separated(proj1, proj2):
    """Check if two projections are separated"""
    return max(proj1) < min(proj2) or max(proj2) < min(proj1)

def is_overlapping(rect1, rect2):
    """Check if two arbitrary rectangles overlap using the Separating Axis Theorem"""
    # Get the edges of rect1 and rect2
    edges1 = get_edges(rect1)
    edges2 = get_edges(rect2)
    
    # Get all potential separating axes (the normal direction of each edge)
    normals = get_normals(edges1) + get_normals(edges2)
    
    # Check if projections on any axis are separated
    for axis in normals:
        proj1 = project(rect1, axis)
        proj2 = project(rect2, axis)
        if is_separated(proj1, proj2):
            return False  # If separated on any axis, the rectangles do not overlap
    
    return True  # If not separated on any axis, the rectangles overlap



def collision_sector(braking_distance, current_position, current_orientation, current_length, other_position, other_length):
    # Calculate the angle between the direction of the vehicle and the line connecting the vehicle to the other vehicle
    vector_to_other = other_position - current_position
    angle_to_other = np.arctan2(vector_to_other[1], vector_to_other[0]) - current_orientation
    
    # Calculate the distance to the other vehicle
    distance_to_other = np.linalg.norm(vector_to_other)
    
    # Set the maximum angle between the direction of the vehicle and the edges of the collision sector
    # 60 degree for sector
    max_angle = np.deg2rad(30)
    
    # Check if the other vehicle is within the collision sector
    if distance_to_other <= braking_distance and np.abs(angle_to_other) <= max_angle:
        return True
    else:
        return False
    


def norm(obv):
    # Calculate the minimum and maximum values of the array
    min_val = np.min(obv)
    max_val = np.max(obv)
    
    # Normalize the array
    return (obv - min_val) / (max_val - min_val + 1e-8)


def weight_norm(obv):
    for i in range(13):
        obv[i*6:i*6+2] = obv[i*6:i*6+2] / 140 + 1e-8
        obv[i*6+4:i*6+6] = obv[i*6+4:i*6+6] / 16.7 + 1e-8
    obv[78:110] = obv[78:110] / 140 + 1e-8
    return obv


def orientation_cost(target_orientation, current_orientation):
    # Calculate the minimum and maximum values of the array
    difference = np.abs(np.sin(current_orientation)) - np.abs(np.sin(current_orientation))
    # difference = np.sin(current_orientation) - np.sin(current_orientation)
    return np.abs(difference)


def lmap(v: float, x: Interval, y: Interval) -> float:
    """Linear map of value v with range x to desired range y."""
    return y[0] + (v - x[0]) * (y[1] - y[0]) / (x[1] - x[0])


def penalized_Norm(s, g, w, p):
    diff = s - g
    weighted_norm = np.power(np.sum(w * np.power(np.abs(diff), p)), 1/p)
    p_reward = - weighted_norm

    return p_reward


def coor_Translator(x_ego, y_ego, heading_ego, x_target, y_target):
    # positive x-axis is the direction of the heading of ego vehicle
    
    x_prime = x_target - x_ego
    y_prime = y_target - y_ego

    x_double_prime = x_prime * np.cos(heading_ego) + y_prime * np.sin(heading_ego)
    y_double_prime = - x_prime * np.sin(heading_ego) + y_prime * np.cos(heading_ego)

    return np.array([x_double_prime, y_double_prime])


def angle_Difference_Calculator(x_ego, y_ego, heading_ego, x_target, y_target):

    # Calculate the angle of the line connection in radians
    delta_y = y_target - y_ego
    delta_x = x_target - x_ego
    theta_line = np.arctan2(delta_y, delta_x)

    # Calculate the difference in radians and convert to degrees
    angle_difference_rad = theta_line - heading_ego
    # Normalize the angle difference to the range [-π, π]
    angle_difference_rad = (angle_difference_rad + np.pi) % (2 * np.pi) - np.pi
    # Normalize the angle difference to the range [0, π]
    # angle_difference_rad = np.abs((angle_difference_rad + np.pi) % (2 * np.pi) - np.pi)
    
    # angle_difference_deg = np.degrees(angle_difference_rad)

    return angle_difference_rad


def line_Intersection(A_x, A_y, B_x, B_y):
    """Calculate the intersection point of two lines if exists."""
    # Matrix setup
    matrix = np.array([A_y - A_x, -(B_y - B_x)]).T
    vector = B_x - A_x

    det = np.linalg.det(matrix)
    if abs(det) < 1e-6:  # Consider as zero
        return None  # Lines are parallel or collinear
    
    # Solve for t and s
    inv_matrix = np.linalg.inv(matrix)
    result = inv_matrix.dot(vector)
    t, s = result

    # Check if t and s are within the segment range
    if 0 <= t <= 1 and 0 <= s <= 1:
        intersection = A_x + t * (A_y - A_x)
        return intersection
    else:
        return None

# Return closest_intersection
# def laser_intersection(x_ego, y_ego, laser_theta, rectangle, max_distance=20):
#     """Calculate laser intersections for a given angle in radians."""
    
#     laser_end_x = x_ego + max_distance * np.cos(laser_theta)
#     laser_end_y = y_ego + max_distance * np.sin(laser_theta)
#     laser_line = np.array([[x_ego, y_ego], [laser_end_x, laser_end_y]])

#     closest_intersection = None
#     min_distance = float(20)

#     for rect in rectangle:
#         # Rectangle edges
#         rect_edges = np.array([
#             [rect[0], rect[1]], [rect[1], rect[2]],
#             [rect[2], rect[3]], [rect[3], rect[0]]
#         ])
#         for edge in rect_edges:
#             # print(edge)
#             intersection = line_Intersection(laser_line[0], laser_line[1], edge[0], edge[1])
#             if intersection is not None:
#                 distance = np.hypot(intersection[0] - x_ego, intersection[1] - y_ego)
#                 if distance < min_distance:
#                     min_distance = distance
#                     closest_intersection = intersection
#     if closest_intersection is None or min_distance > max_distance:
#         closest_intersection = (laser_end_x, laser_end_y)
    
#     return closest_intersection


# Return min distance
def laser_intersection(x_ego, y_ego, laser_theta, rectangle, max_distance):
    """Calculate laser intersections for a given angle in radians."""
    
    laser_end_x = x_ego + max_distance * np.cos(laser_theta)
    laser_end_y = y_ego + max_distance * np.sin(laser_theta)
    laser_line = np.array([[x_ego, y_ego], [laser_end_x, laser_end_y]])

    min_distance = float(max_distance)
    
    for rect in rectangle:
        # Rectangle edges
        rect_edges = np.array([
            [rect[0], rect[1]], [rect[1], rect[2]],
            [rect[2], rect[3]], [rect[3], rect[0]]
        ])
        for edge in rect_edges:
            intersection = line_Intersection(laser_line[0], laser_line[1], edge[0], edge[1])
            if intersection is not None:
                distance = np.hypot(intersection[0] - x_ego, intersection[1] - y_ego)
                if distance < min_distance:
                    min_distance = distance
    return min_distance


def radians_to_degrees(radians):
    degrees = radians * (180 / np.pi)
    return degrees


def rectangle_corners(cx, cy, length, width, heading):
    # Calculate direction vectors
    length_dir = (np.cos(heading), np.sin(heading))
    width_dir = (-np.sin(heading), np.cos(heading))
    
    # Half vectors
    half_length = (length_dir[0] * length / 2, length_dir[1] * length / 2)
    half_width = (width_dir[0] * width / 2, width_dir[1] * width / 2)
    
    # Calculate corners
    top_right = (cx + half_length[0] + half_width[0], cy + half_length[1] + half_width[1])
    top_left = (cx + half_length[0] - half_width[0], cy + half_length[1] - half_width[1])
    bottom_right = (cx - half_length[0] + half_width[0], cy - half_length[1] + half_width[1])
    bottom_left = (cx - half_length[0] - half_width[0], cy - half_length[1] - half_width[1])
    
    return (top_right, top_left, bottom_left, bottom_right)


def reward_Plot(reward_data, file_name):
    plt.figure(figsize=(10, 6))
    plt.plot(reward_data, linestyle='-', color='#35678b', label='Episode reward')
    plt.title('Reward for each episode')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    plt.savefig(file_name, format='png', dpi=300)
    plt.close()


def radian_Norm(rad):
    return (rad + np.pi) % (2 * np.pi) - np.pi


def relative_Vel_Calculator(v1, theta1, v2, theta2, x, y):
    """
    Calculate the relative velocity vector.
    
    Parameters:
    v1 (float): Speed of ego vehicle.
    theta1 (float): Heading of ego vehicle in radians.
    v2 (float): Speed of the other vehicle.
    theta2 (float): Heading of the other vehicle in radians.
    x, y: relative position of the other vehicle

    Returns:
    np.array: Relative velocity vector, sign function for ttc
    """
    v1x = v1 * np.cos(theta1)
    v1y = v1 * np.sin(theta1)
    v2x = v2 * np.cos(theta2)
    v2y = v2 * np.sin(theta2)

    determin = (v1x-v2x) * (0-x) + (v1y-v2y) * (0-y)
    return np.array([v2x - v1x, v2y - v1y]), np.sign(determin)


def minmax_norm(action, min_val, max_val):
    return (action - min_val) / (max_val - min_val) * 2 - 1

def minmax_denorm(action_norm, min_val, max_val):
    """
    Denormalize the action array from the range [-1, 1] to original range.
    
    Parameters:
    - action_norm: np.ndarray of shape (n_samples, n_features), the normalized array
    - min_val: np.ndarray of shape (n_features,), minimum values for each feature
    - max_val: np.ndarray of shape (n_features,), maximum values for each feature
    
    Returns:
    - denormalized_action: np.ndarray of shape (n_samples, n_features), the denormalized array
    """
    return action_norm * max_val
    # return (action_norm + 1) / 2 * (max_val - min_val) + min_val


def generate_control_points(P0, P3):
    # Candidate control points that make P0P1 and P2P3 parallel to the axes
    candidates = [
        np.array([P0[0], P3[1]]),  # Aligns with P0's x-coordinate, y within specified range
        np.array([P3[0], P0[1]])   # Aligns with P0's y-coordinate, x within specified range
    ]

    # candidates = [
    #     np.array([P0[0], 70]),  # Aligns with P0's x-coordinate, y within specified range
    #     np.array([70, P0[1]])   # Aligns with P0's y-coordinate, x within specified range
    # ]
    
    # Find the first candidate within the target range to use as control point
    for candidate in candidates:
        if 60 < candidate[0] < 80 and 60 < candidate[1] < 80:
            return candidate, candidate  # P1 and P2 are overlapping here
        else:
            return np.array((P0 + P3) / 2), np.array((P0 + P3) / 2)
    


def bezier_curve(t, P0, P1, P2, P3):
    # Convert control points to NumPy arrays for element-wise operations
    P0, P1, P2, P3 = map(np.array, (P0, P1, P2, P3))
    return (1 - t)**3 * P0 + 3 * (1 - t)**2 * t * P1 + 3 * (1 - t) * t**2 * P2 + t**3 * P3


def calculate_waypoint_progress(origin, destination, current_position):

    # Calculate the total distance using the custom distance formula
    total_distance = np.sum(np.abs(np.array(destination) - np.array(origin)))

    # Calculate the distance covered so far using the custom distance formula
    covered_distance = np.sum(np.abs(np.array(current_position) - np.array(origin)))

    # Calculate the percentage of progress
    if total_distance == 0:
        progress = 0  # Avoid division by zero
    else:
        progress = covered_distance / total_distance

    # Clamp the result between 0.1 and 0.9
    progress = max(0.1, min(progress, 0.9)) if progress < 0.9 else 1

    return progress


def waypoint_dir(origin, destination, current):
    step = calculate_waypoint_progress(origin, destination, current)
    P1, P2 = generate_control_points(current, destination)
    next_point = bezier_curve(step, current, P1, P2, destination)

    return next_point.reshape((1,2))


def sample_des_saver(des, episode, filename):
    if episode <= 1:
        if os.path.exists(filename):
            os.remove(filename)
    else:
        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(des)


if __name__ == "__main__":
    
    x_ego, y_ego, theta_ego = 1, 1, 0  # Ego vehicle at (100, 100) with a heading of 30 degrees
    x_ctrl, y_ctrl = 2, 2

    relative_pos = coor_Translator(x_ego, y_ego, theta_ego, x_ctrl, y_ctrl)
    angle_difference = angle_Difference_Calculator(x_ego, y_ego, theta_ego, x_ctrl, y_ctrl)

    # # Example usage
    car_position = ((x_ego, y_ego))
    rectangles = (
        ((5, 0), (5, 2), (7, 2), (7, 0)),
        ((3, 2), (4, 2), (4, 4), (3, 4))
    )
    radian_angles = [(i * np.pi / 10 + theta_ego) for i in range(20)]  # Angles in radians
    
    results = {}
    for angle in radian_angles:
        results[angle] = laser_intersection(car_position[0], car_position[1], angle, rectangles)

    # for angle, coord in results.items():
    #     print(f"Angle {angle:.2f} radians: Intersection at {coord}")

    # corner_length = 60
    # zone_center = np.array(((30,30), (110,30), (110,110), (30,110)))
    # zone_rectangles = np.zeros((4,4,2))
    # for i in range(zone_center.shape[0]):
    #     zone_rectangles[i] = rectangle_corners(zone_center[i,0], zone_center[i,1], corner_length, corner_length, 0)
    # for rec in zone_rectangles:
    #     print(rec)
