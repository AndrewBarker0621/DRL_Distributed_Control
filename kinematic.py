import numpy as np

# Define kinematic bicycle model update function
def kinematic_Update(state, acceleration, steering_angle, dt, vehicle_length):
    
    x, y, orientation, velocity = state
    wheelbase = 0.8 * vehicle_length
    lf = wheelbase / 2.5
    lr = 1.5* lf
    
    beta = np.arctan((lr / (lf + lr)) * np.tan(steering_angle))

    # Compute the final state using the discrete time model
    new_x = x + velocity * np.cos(orientation + beta) * dt
    new_y = y + velocity * np.sin(orientation + beta) * dt

    new_orientation = orientation + (velocity / lr) * np.sin(beta) * dt

    # new_velocity = 0 if (velocity + dt * acceleration) < 0 else velocity + dt * acceleration
    new_velocity = velocity + dt * acceleration
    # print(new_x, new_y, new_orientation, new_velocity)
    # return np.array([new_x, new_y, new_orientation%(2*np.pi), new_velocity])
    return [new_x, new_y, new_orientation, new_velocity]


def simple_Update(state, acceleration, dt, origin_id, ori, des, angle_diff):
    
    x, y, orientation, velocity = state
    # Compute the final state using the discrete time model
    new_x = x + velocity * np.cos(orientation) * dt
    new_y = y + velocity * np.sin(orientation) * dt

    if origin_id == 0:
        _rate = (new_x - 79)/(des[0] - 79)
        rate = max(0, min(1, _rate))
        new_orientation = ori[2] + angle_diff * rate if (59 < new_y < 81) else des[2]
    elif origin_id == 1:
        _rate = (new_y - 61)/(des[1] - 61)
        rate = max(0, min(1, _rate))
        new_orientation = ori[2] + angle_diff * rate if (59 < new_x < 81) else des[2]
    elif origin_id == 2:
        _rate = (new_x - 61)/(des[0] - 61)
        rate = max(0, min(1, _rate))
        new_orientation = ori[2] + angle_diff * rate if (59 < new_y < 81) else des[2]
    else:
        _rate = (new_y - 79)/(des[1] - 79)
        rate = max(0, min(1, _rate))
        new_orientation = ori[2] + angle_diff * rate if (59 < new_x < 81) else des[2]

    # new_velocity = 0 if (velocity + dt * acceleration) < 0 else velocity + dt * acceleration
    new_velocity = velocity + dt * acceleration
    # print(new_x, new_y, new_orientation, new_velocity)
    # return np.array([new_x, new_y, new_orientation%(2*np.pi), new_velocity])
    return [new_x, new_y, new_orientation, new_velocity]


def shortest_angle_diff(des, ori):
    # Calculate the direct difference
    difference = (des - ori) % (2 * np.pi)
    
    # Adjust to always take the shortest path
    if difference > np.pi:
        difference -= 2 * np.pi
    # print(difference)
    return difference


def inverse_kinematic_Update(state, waypoint, v_min, v_max, vehicle_length, dt):
    
    x, y, theta, v = state
    x_d, y_d = waypoint

    L = 0.8 * vehicle_length

    dx, dy = x_d - x, y_d - y
    Ld = np.hypot(dx, dy) + 1e-6        
    alpha = np.arctan2(dy, dx) - theta
    alpha = np.arctan2(np.sin(alpha), np.cos(alpha))  

    ste = np.arctan2(2.0 * L * np.sin(alpha), Ld)

    target_v = np.random.uniform(v_min, v_max)
    acc = (target_v - v) / dt

    return acc, ste