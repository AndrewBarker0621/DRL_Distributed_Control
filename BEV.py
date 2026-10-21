import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = None
import math
from shapely.geometry import Polygon, Point
from PIL import Image
import os
from collections import namedtuple

###############################################################################
# Coordinate Transform Helpers
###############################################################################

def world_to_local(x_world, y_world, ego_x, ego_y, ego_heading):
    """
    Convert a world-frame point (x_world, y_world) to the ego's local frame,
    where the ego is at (0,0) and 'up' in local coords matches ego_heading in world.

    We rotate by -(ego_heading - pi/2) so that the ego's heading aligns with local +y.
    """
    dx = x_world - ego_x
    dy = y_world - ego_y
    
    # Rotate so the ego heading aligns with local +y
    theta = -(ego_heading - math.pi/2)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    
    x_local = dx * cos_t - dy * sin_t
    y_local = dx * sin_t + dy * cos_t
    return x_local, y_local


def local_to_grid(x_local, y_local, grid_size=80, resolution=1.0):
    """
    Convert local coords (x_local, y_local) -> (row, col) in an 80×80 grid at 1m/cell.

    The ego is at the center => (row=40, col=40) if grid_size=80.
    local +y is "up" => row decreases as y_local increases.
    """
    half_size = grid_size / 2.0
    
    col = int((x_local + half_size) / resolution)
    row = int((half_size - y_local) / resolution)
    return row, col


###############################################################################
# Polygon Transform & Rasterization
###############################################################################

def polygon_world_to_local(polygon_pts, ego_x, ego_y, ego_heading):
    """
    Transform a list of polygon vertices (in world coords) into the ego's local frame.
    Return a shapely Polygon in local coordinates.
    """
    local_pts = []
    for (wx, wy) in polygon_pts:
        lx, ly = world_to_local(wx, wy, ego_x, ego_y, ego_heading)
        local_pts.append((lx, ly))
    return Polygon(local_pts)


def rasterize_polygon(
    bev_grid, 
    local_poly, 
    channel_occ, 
    channel_vel=None, 
    velocity_value=0, 
    channel_vel_perp=None,
    perpendicular_velocity_value=0,
    grid_size=80
):
    """
    For each cell in bev_grid (a CuPy array), check if the cell-center is inside local_poly
    (a CPU-based shapely Polygon). If so:
      - Mark bev_grid[row, col, channel_occ] = 1 (occupancy)
      - Optionally mark bev_grid[row, col, channel_vel] = velocity_value (if channel_vel is not None).
    """
    half_size = grid_size / 2.0
    
    for row in range(grid_size):
        for col in range(grid_size):
            # Center of this cell in local coords
            x_local = (col + 0.5) - half_size
            y_local = half_size - (row + 0.5)
            
            pt = Point(x_local, y_local)  # shapely point on CPU
            if local_poly.contains(pt):
                bev_grid[row, col, channel_occ] = 1
                if channel_vel is not None:
                    bev_grid[row, col, channel_vel] = velocity_value
                if channel_vel_perp is not None:
                    bev_grid[row, col, channel_vel_perp] = perpendicular_velocity_value

###############################################################################
# Vehicle Polygon
###############################################################################

def vehicle_polygon_local(vehicle_state, vehicle_size, ego_state):
    """
    Build a shapely Polygon representing the bounding box of a vehicle
    in the ego’s local coordinate system.

    vehicle_state: (x_world, y_world, heading_world, velocity)
    vehicle_size:  (length, width)
    ego_state:     (x_ego, y_ego, heading_ego, v_ego)
    """
    x_v, y_v, heading_v, _ = vehicle_state
    L, W = vehicle_size
    
    half_L = L / 2.0
    half_W = W / 2.0
    
    # Corners in vehicle's local ref (front = +x)
    corners_vehicle_local = [
        ( half_L,  half_W),
        ( half_L, -half_W),
        (-half_L, -half_W),
        (-half_L,  half_W)
    ]
    
    cos_v = math.cos(heading_v)
    sin_v = math.sin(heading_v)
    
    # Rotate + translate corners into world
    world_corners = []
    for (cx, cy) in corners_vehicle_local:
        wx = x_v + (cx*cos_v - cy*sin_v)
        wy = y_v + (cx*sin_v + cy*cos_v)
        world_corners.append((wx, wy))
    
    # Convert world_corners to ego-local
    ego_x, ego_y, ego_h, _ = ego_state
    local_corners = []
    for (wx, wy) in world_corners:
        lx, ly = world_to_local(wx, wy, ego_x, ego_y, ego_h)
        local_corners.append((lx, ly))
    
    return Polygon(local_corners)


###############################################################################
# Rasterize Prohibited Zones
###############################################################################

def rasterize_prohibited_zones(
    bev_grid, 
    prohibited_zones, 
    ego_state, 
    grid_size=80
):
    """
    Rasterize each prohibited zone polygon into channel 0.
    """
    for poly_pts in prohibited_zones:
        local_poly = polygon_world_to_local(
            polygon_pts=poly_pts, 
            ego_x=ego_state[0], 
            ego_y=ego_state[1], 
            ego_heading=ego_state[2]
        )
        rasterize_polygon(
            bev_grid, 
            local_poly, 
            channel_occ=0,  # channel 0 for prohibited zone occupancy
            grid_size=grid_size
        )


###############################################################################
# Rasterize Other Vehicles
###############################################################################

def rasterize_other_vehicles(
    bev_grid, 
    other_vehicles, 
    ego_state, 
    grid_size=80, 
    channel_occ=1, 
    channel_vel=2,
    channel_vel_perp=3  # new channel for perpendicular relative velocity
):
    """
    For each other vehicle, rasterize:
      - Occupancy in channel_occ,
      - Forward velocity (relative to ego heading) in channel_vel,
      - Perpendicular (lateral) velocity in channel_vel_perp.
      
    Velocities are clamped to [MIN_REL_VEL, MAX_REL_VEL] and scaled to [0..255].
    """
    MIN_REL_VEL = -32.0
    MAX_REL_VEL =  32.0
    
    _, _, theta_e, v_e = ego_state

    for veh in other_vehicles:        
        # Compute the vehicle's polygon in the ego-local frame.
        poly_local = vehicle_polygon_local(veh.state, veh.size, ego_state)
        
        # Retrieve the vehicle's state
        _, _, theta_o, v_o = veh.state

        # Forward relative velocity (along ego heading)
        v_rel = (v_o * math.cos(theta_o - theta_e)) - v_e
        
        # Perpendicular relative velocity (lateral component)
        v_rel_perp = v_o * math.sin(theta_o - theta_e)
        
        # Clamp and scale the forward component.
        v_clamped = max(MIN_REL_VEL, min(MAX_REL_VEL, v_rel))
        v_scaled = int((v_clamped - MIN_REL_VEL) / (MAX_REL_VEL - MIN_REL_VEL) * 255)
        
        # Clamp and scale the perpendicular component.
        v_perp_clamped = max(MIN_REL_VEL, min(MAX_REL_VEL, v_rel_perp))
        v_perp_scaled = int((v_perp_clamped - MIN_REL_VEL) / (MAX_REL_VEL - MIN_REL_VEL) * 255)
        
        # Rasterize the polygon in three channels:
        # occupancy, forward velocity, and perpendicular velocity.
        rasterize_polygon(
            bev_grid=bev_grid,
            local_poly=poly_local,
            channel_occ=channel_occ,
            channel_vel=channel_vel,
            velocity_value=v_scaled,
            channel_vel_perp=channel_vel_perp,  # new channel for lateral velocity
            perpendicular_velocity_value=v_perp_scaled,
            grid_size=grid_size
        )


###############################################################################
# Rasterize Ego History
###############################################################################

def rasterize_ego_history(
    bev_grid, 
    ego_history, 
    ego_state,
    vehicle_size, 
    grid_size=80, 
    channel_occ=4, 
    channel_vel=5
):
    """
    Rasterize the bounding box of the ego vehicle for each historical state,
    aligned to the *current* heading. That is, we treat the *last* state in
    ego_history as the "current ego" reference frame.
    
    - channel_occ: channel to mark occupancy (0=free, 1=occupied).
    - channel_vel: channel to store velocity (scaled 0..255).
    """
    MIN_REL_VEL = -32.0
    MAX_REL_VEL =  32.0
    
    _, _, theta_e, v_e = ego_state

    for past_state in ego_history:
        # "vehicle_polygon_local" transforms a 'vehicle' state 
        # into the local frame of "ego_state"
        # We pass in the *current_ego_state* as the "ego_state" parameter
        # so that every historical box is aligned to the CURRENT heading.
        poly_local = vehicle_polygon_local(
            vehicle_state=past_state,     # The historical state to place
            vehicle_size=vehicle_size,    # The vehicle bounding box
            ego_state=ego_state   # The *current* ego reference
        )
        
        #    v_rel = (v_o cos(theta_o - theta_e)) - v_e
        _, _, theta_o, v_o = past_state

        # scalar projection in ego's heading
        v_rel = (v_o * math.cos(theta_o - theta_e)) - v_e
        
        # Clamp and scale to [0..255]
        v_clamped = max(MIN_REL_VEL, min(MAX_REL_VEL, v_rel))
        # shift and scale
        v_scaled = int( (v_clamped - MIN_REL_VEL) / (MAX_REL_VEL - MIN_REL_VEL) * 255 )
        
        # Rasterize the bounding box into the occupancy channel + velocity channel
        rasterize_polygon(
            bev_grid=bev_grid,
            local_poly=poly_local,
            channel_occ=channel_occ,
            channel_vel=channel_vel,
            velocity_value=v_scaled,
            grid_size=grid_size
        )


###############################################################################
# Mark Target
###############################################################################

def mark_target(
    bev_grid, 
    target_coord, 
    ego_state, 
    channel=6, 
    grid_size=80, 
    resolution=1.0
):
    """
    Mark the ego's goal location in channel=6 (1 in that cell).
    """
    ego_x, ego_y, ego_heading, _ = ego_state
    tx, ty = target_coord
    lx, ly = world_to_local(tx, ty, ego_x, ego_y, ego_heading)
    row, col = local_to_grid(lx, ly, grid_size, resolution)
    if 0 <= row < grid_size and 0 <= col < grid_size:
        bev_grid[row, col, channel] = 1
        

###############################################################################
# Vectorized rasterization
#
# Every BEV shape is a rectangle (vehicle boxes, prohibited zones, history
# boxes), so instead of testing 1600 shapely Points per polygon we transform
# all cell centers into each rectangle's edge basis with two dot products and
# apply a strict-interior box test. Identical output to the shapely path
# (verified cell-for-cell at resolution=1), ~2 orders of magnitude faster.
#
# NOTE: unlike the old rasterize_polygon, cell centers honor `resolution`
# (the old code ignored it, so the grid covered only grid_size meters and the
# target channel — which did honor resolution — was spatially misaligned).
###############################################################################

_CENTER_CACHE = {}


def _grid_centers(grid_size, resolution):
    key = (grid_size, resolution)
    if key not in _CENTER_CACHE:
        half = grid_size / 2.0
        xs = ((np.arange(grid_size) + 0.5) - half) * resolution
        ys = (half - (np.arange(grid_size) + 0.5)) * resolution
        X, Y = np.meshgrid(xs, ys)          # X[row, col], Y[row, col] in meters
        _CENTER_CACHE[key] = (X, Y)
    return _CENTER_CACHE[key]


def _rect_mask(X, Y, corners):
    """
    Strict-interior test of all cell centers against one rectangle given by
    its 4 local-frame corners in order (matches shapely Polygon.contains,
    which excludes the boundary).
    """
    c0 = corners[0]
    e1 = corners[1] - corners[0]
    e2 = corners[3] - corners[0]
    dx, dy = X - c0[0], Y - c0[1]
    u = dx * e1[0] + dy * e1[1]
    v = dx * e2[0] + dy * e2[1]
    return (u > 0) & (u < e1 @ e1) & (v > 0) & (v < e2 @ e2)


def _corners_world_to_local(world_corners, ego_state):
    ego_x, ego_y, ego_h = ego_state[0], ego_state[1], ego_state[2]
    theta = -(ego_h - math.pi / 2)
    c, s = math.cos(theta), math.sin(theta)
    pts = np.asarray(world_corners, dtype=float) - [ego_x, ego_y]
    return pts @ np.array([[c, s], [-s, c]])


def _vehicle_corners_local(vehicle_state, vehicle_size, ego_state):
    x_v, y_v, heading_v = vehicle_state[0], vehicle_state[1], vehicle_state[2]
    hl, hw = vehicle_size[0] / 2.0, vehicle_size[1] / 2.0
    cv, sv = math.cos(heading_v), math.sin(heading_v)
    local = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]])
    world = local @ np.array([[cv, sv], [-sv, cv]]) + [x_v, y_v]
    return _corners_world_to_local(world, ego_state)


def _scale_vel(v_rel):
    MIN_REL_VEL, MAX_REL_VEL = -32.0, 32.0
    v_clamped = max(MIN_REL_VEL, min(MAX_REL_VEL, v_rel))
    return int((v_clamped - MIN_REL_VEL) / (MAX_REL_VEL - MIN_REL_VEL) * 255)


def _rect_masks_np(X, Y, corners):
    """
    Batched version of _rect_mask: corners (R, 4, 2) in the local frame ->
    (R, G, G) boolean masks. Elementwise operations match _rect_mask exactly.
    """
    c0 = corners[:, 0, :]                                     # (R,2)
    e1 = corners[:, 1, :] - c0
    e2 = corners[:, 3, :] - c0
    dx = X[None, :, :] - c0[:, 0, None, None]
    dy = Y[None, :, :] - c0[:, 1, None, None]
    u = dx * e1[:, 0, None, None] + dy * e1[:, 1, None, None]
    v = dx * e2[:, 0, None, None] + dy * e2[:, 1, None, None]
    l1 = (e1[:, 0] * e1[:, 0] + e1[:, 1] * e1[:, 1])[:, None, None]
    l2 = (e2[:, 0] * e2[:, 0] + e2[:, 1] * e2[:, 1])[:, None, None]
    return (u > 0) & (u < l1) & (v > 0) & (v < l2)


def _vehicles_corners_world(states, sizes):
    """
    Batched version of _vehicle_corners_local's world-corner step:
    states (N, >=3), sizes (N, 2) -> world corners (N, 4, 2), same corner
    order and float operations as the scalar helper.
    """
    hl, hw = sizes[:, 0] / 2.0, sizes[:, 1] / 2.0
    cv, sv = np.cos(states[:, 2]), np.sin(states[:, 2])
    lx = np.stack([hl, hl, -hl, -hl], axis=1)                 # (N,4)
    ly = np.stack([hw, -hw, -hw, hw], axis=1)
    wx = lx * cv[:, None] - ly * sv[:, None] + states[:, 0, None]
    wy = lx * sv[:, None] + ly * cv[:, None] + states[:, 1, None]
    return np.stack([wx, wy], axis=2)


def _corners_world_to_local_np(world_corners, ego_state):
    """Batched version of _corners_world_to_local for (..., 2) corner arrays."""
    theta = -(ego_state[2] - math.pi / 2)
    c, s = math.cos(theta), math.sin(theta)
    dx = world_corners[..., 0] - ego_state[0]
    dy = world_corners[..., 1] - ego_state[1]
    return np.stack([dx * c - dy * s, dx * s + dy * c], axis=-1)


def create_bev_grid(
    vehicles,
    ego_id,
    prohibited_zones,
    grid_size=80,
    resolution=1.0
):
    """
    Construct a (grid_size x grid_size x 7) uint8 BEV grid covering
    grid_size*resolution meters, ego-centric and heading-aligned:
      Channel 0: prohibited zones
      Channel 1: other vehicles occupancy
      Channel 2: other vehicles forward relative velocity (scaled 0..255)
      Channel 3: other vehicles lateral relative velocity (scaled 0..255)
      Channel 4: ego history occupancy
      Channel 5: ego history velocity (scaled 0..255)
      Channel 6: target cell
    """
    other_vehicles = []
    for v in vehicles.values():
        if v.id == ego_id:
            ego_vehicle = v
            ego_history = v.his_traj.copy()
            target_coord = v.destination[:2].copy()
        else:
            other_vehicles.append(v)

    bev_grid = np.zeros((grid_size, grid_size, 7), dtype=np.uint8)
    ego_state = ego_vehicle.state
    theta_e, v_e = ego_state[2], ego_state[3]
    X, Y = _grid_centers(grid_size, resolution)

    # gather every rectangle (zones, other vehicles, ego history), then run
    # the corner transforms and interior tests in one batched pass
    zones = np.asarray(prohibited_zones, dtype=float)          # (Z,4,2)
    n_z, n_o, n_h = len(zones), len(other_vehicles), len(ego_history)
    world = [zones]
    if n_o:
        o_states = np.array([v.state for v in other_vehicles])
        o_sizes = np.array([v.size for v in other_vehicles])
        world.append(_vehicles_corners_world(o_states, o_sizes))
    if n_h:
        h_states = np.array(ego_history)
        h_sizes = np.broadcast_to(np.asarray(ego_vehicle.size, dtype=float), (n_h, 2))
        world.append(_vehicles_corners_world(h_states, h_sizes))
    corners = _corners_world_to_local_np(np.concatenate(world, axis=0), ego_state)
    masks = _rect_masks_np(X, Y, corners)

    # 1) prohibited zones (channel 0)
    for mask in masks[:n_z]:
        bev_grid[mask, 0] = 1

    # 2) other vehicles (channels 1-3); per-vehicle write order preserved so
    #    later vehicles still win on overlapping cells
    for i, veh in enumerate(other_vehicles):
        mask = masks[n_z + i]
        theta_o, v_o = veh.state[2], veh.state[3]
        bev_grid[mask, 1] = 1
        bev_grid[mask, 2] = _scale_vel(v_o * math.cos(theta_o - theta_e) - v_e)
        bev_grid[mask, 3] = _scale_vel(v_o * math.sin(theta_o - theta_e))

    # 3) ego history (channels 4-5)
    for i, past_state in enumerate(ego_history):
        mask = masks[n_z + n_o + i]
        bev_grid[mask, 4] = 1
        bev_grid[mask, 5] = _scale_vel(past_state[3] * math.cos(past_state[2] - theta_e) - v_e)

    # 4) target (channel 6)
    mark_target(bev_grid, target_coord, ego_state, channel=6,
                grid_size=grid_size, resolution=resolution)

    return bev_grid


def create_bev_grid_reference(
    vehicles,
    ego_id,
    prohibited_zones,
    grid_size=80,
    resolution=1.0
):
    """
    Original shapely-based implementation, kept for equivalence testing of
    the vectorized create_bev_grid (identical at resolution=1; the old raster
    ignored resolution, which the vectorized version fixes).
    """
    other_vehicles = []
    for v in vehicles.values():
        if v.id == ego_id:
            ego_vehicle = v
            ego_history = v.his_traj.copy()
            target_coord = v.destination[:2].copy()
        else:
            other_vehicles.append(v)

    # 7 channels total
    bev_grid = np.zeros((grid_size, grid_size, 7), dtype=np.uint8)
    
    # Ego's current state: (x, y, heading, velocity)
    ego_state = ego_vehicle.state
    
    # 1) Rasterize prohibited zones (channel=0)
    rasterize_prohibited_zones(
        bev_grid=bev_grid,
        prohibited_zones=prohibited_zones,
        ego_state=ego_state,
        grid_size=grid_size
    )
    
    # 2) Rasterize other vehicles (channels 1 & 2 & 3)
    rasterize_other_vehicles(
        bev_grid=bev_grid,
        other_vehicles=other_vehicles, 
        ego_state=ego_state, 
        grid_size=grid_size,
        channel_occ=1,   # occupancy
        channel_vel=2,    # velocity
        channel_vel_perp=3
    )
    
    # 3) Rasterize ego history (channels 4 & 5)
    #    We'll treat the last entry in ego_history as "current" for the local frame
    rasterize_ego_history(
        bev_grid=bev_grid,
        ego_history=ego_history,
        ego_state=ego_state,
        vehicle_size=ego_vehicle.size,
        grid_size=grid_size,
        channel_occ=4,
        channel_vel=5
    )
    
    # 4) Mark target (channel=6)
    mark_target(
        bev_grid=bev_grid,
        target_coord=target_coord,
        ego_state=ego_state,
        channel=6,
        grid_size=grid_size,
        resolution=resolution
    )
    
    return bev_grid


def save_bev_channels_as_images(bev_array, output_dir="bev_channels", scale=4):
    """
    Saves each channel of bev_array (H, W, C) as a separate 8-bit grayscale image.
    
    Parameters:
    -----------
    bev_array : numpy.ndarray
        A 3D NumPy array with shape (H, W, C).
    output_dir : str
        Directory to save the output images.
    scale : int
        Upscale factor for the output images (e.g., 4 => 4x larger in each dimension).
    """
    # bev_array is already a NumPy array, so no need to convert from CuPy.
    
    # Make output directory if it doesn't exist.
    os.makedirs(output_dir, exist_ok=True)
    
    num_channels = bev_array.shape[2]
    for c in range(num_channels):
        channel_data = bev_array[..., c]  # shape: (H, W)
        
        # Example: if the channel represents occupancy (with values {0,1}),
        # scale it to the full 0..255 range.
        if c in [0, 1, 4, 6]:  # adjust these channel indices as needed
            channel_data = channel_data * 255
        
        # Convert to uint8 for image saving.
        channel_img = channel_data.astype(np.uint8)
        
        # Create a PIL image in grayscale mode ('L').
        img = Image.fromarray(channel_img, mode='L')
        
        # Optionally resize the image.
        if scale > 1:
            width, height = img.size
            new_size = (width * scale, height * scale)
            # Use NEAREST neighbor to preserve blocky appearance.
            img = img.resize(new_size, resample=Image.NEAREST)
        
        # Save the image.
        img.save(os.path.join(output_dir, f"channel_{c}.png"))


# def save_bev_channels_as_images(bev_cupy_array, output_dir="bev_channels", scale=4):
#     """
#     Saves each channel of bev_cupy_array (H,W,C) as a separate 8-bit grayscale image.
    
#     Parameters:
#     -----------
#     bev_cupy_array : cupy.ndarray
#         A 3D CuPy array with shape (H, W, C).
#     output_dir : str
#         Directory to save the output images.
#     scale : int
#         Upscale factor for the output images (e.g., 4 => 4x larger in each dimension).
#     """
#     # 1) Copy from CuPy -> NumPy
#     bev_cpu = bev_cupy_array.get()  # shape: (H, W, Channels)
    
#     # Make output directory
#     os.makedirs(output_dir, exist_ok=True)
    
#     # 2) Iterate over channels
#     num_channels = bev_cpu.shape[2]
#     for c in range(num_channels):
#         channel_data = bev_cpu[..., c]  # shape: (H, W)
        
#         # Example: if channel represents occupancy {0,1}, scale to 0..255
#         if c in [0,1,3,5]:  # example channels for occupancy
#             channel_data = channel_data * 255
        
#         # Convert to uint8 for image saving
#         channel_img = channel_data.astype(np.uint8)
        
#         # 3) Create a PIL image
#         img = Image.fromarray(channel_img, mode='L')
        
#         # 4) Optionally resize the image to make it larger
#         if scale > 1:
#             # width, height = original shape
#             width, height = img.size
#             new_size = (width * scale, height * scale)
#             # Use NEAREST neighbor so each pixel just gets replicated
#             # (that way occupancy still appears blocky rather than blurred).
#             img = img.resize(new_size, resample=Image.NEAREST)
        
#         # 5) Save
#         img.save(os.path.join(output_dir, f"channel_{c}.png"))
