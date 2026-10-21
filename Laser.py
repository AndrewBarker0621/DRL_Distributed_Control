import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = None
import math
from shapely.geometry import Polygon, Point, LineString, MultiPolygon
from shapely.affinity import rotate, translate
from PIL import Image
import os
from collections import namedtuple
from shapely.validation import explain_validity


def create_rectangle(center_x, center_y, width, height, heading):

    # Create an axis-aligned rectangle centered at the origin.
    rect = Polygon([
        (-width / 2, -height / 2),
        (width / 2, -height / 2),
        (width / 2, height / 2),
        (-width / 2, height / 2)
    ])
    
    # Rotate the rectangle by 'heading' degrees (set use_radians=True if using radians)
    rotated_rect = rotate(rect, heading, origin=(0, 0), use_radians=True)
    
    # Translate the rectangle to the desired center.
    final_rect = translate(rotated_rect, xoff=center_x, yoff=center_y)

    return final_rect
    
    # Convert the coordinates to a NumPy array.
    # Note: The last point repeats the first; slice if you only need the unique corners.
    coords = np.array(final_rect.exterior.coords[:-1])
    return coords


def create_multiple_rectangles(rectangles_params):
    """
    Given a list of parameters for rectangles, returns a MultiPolygon.
    Each parameter is a tuple: (center, width, height, heading)
    """
    recs = [create_rectangle(*params) for params in rectangles_params]

    return recs


###############################################################################
# Vectorized ray casting / collision (replaces the per-ray shapely path)
###############################################################################

def rect_corners_np(center_x, center_y, width, height, heading):
    """
    Corner array (4,2) of the same rectangle create_rectangle builds:
    axis-aligned (+-width/2, +-height/2) rotated by heading, then translated.
    """
    hw, hh = width / 2.0, height / 2.0
    base = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]])
    c, s = math.cos(heading), math.sin(heading)
    return base @ np.array([[c, s], [-s, c]]) + np.array([center_x, center_y])


def point_in_any_rect(p, rects):
    """Strict-interior test of point p (2,) against rects (N,4,2)."""
    c0 = rects[:, 0]
    e1 = rects[:, 1] - rects[:, 0]
    e2 = rects[:, 3] - rects[:, 0]
    dp = p - c0
    u = (dp * e1).sum(axis=1)
    v = (dp * e2).sum(axis=1)
    return bool((((u > 0) & (u < (e1 * e1).sum(axis=1))
                  & (v > 0) & (v < (e2 * e2).sum(axis=1)))).any())


def laser_scan(center, laser_range, angles, rects, zero_if_inside=True):
    """
    Vectorized ray casting: distance from center to the first boundary
    crossing along each angle, capped at laser_range (0 for every ray when the
    center lies inside an obstacle, matching the shapely clipped-segment
    semantics). rects: (N,4,2) corner arrays.
    """
    angles = np.asarray(angles, dtype=float)
    center = np.asarray(center, dtype=float)
    rects = np.asarray(rects, dtype=float)
    if rects.size == 0:
        return np.full(angles.shape[0], float(laser_range))
    if zero_if_inside and point_in_any_rect(center, rects):
        return np.zeros(angles.shape[0])

    a = rects.reshape(-1, 2)                                  # (M,2) edge starts
    e = (np.roll(rects, -1, axis=1) - rects).reshape(-1, 2)   # (M,2) edge vectors
    ac = a - center
    d = np.stack([np.cos(angles), np.sin(angles)], axis=1)    # (K,2)

    # ray: center + t*d ; edge: a + s*e  ->  t = cross(ac,e)/cross(d,e), s = cross(ac,d)/cross(d,e)
    denom = d[:, 0:1] * e[None, :, 1] - d[:, 1:2] * e[None, :, 0]          # (K,M)
    cross_ae = ac[:, 0] * e[:, 1] - ac[:, 1] * e[:, 0]                     # (M,)
    cross_ad = ac[None, :, 0] * d[:, 1:2] - ac[None, :, 1] * d[:, 0:1]     # (K,M)
    with np.errstate(divide='ignore', invalid='ignore'):
        t = cross_ae[None, :] / denom
        s = cross_ad / denom
    valid = (np.abs(denom) > 1e-12) & (s >= 0.0) & (s <= 1.0) & (t >= 0.0) & (t <= laser_range)
    return np.where(valid, t, float(laser_range)).min(axis=1)


def rects_overlap_any(rect, rects):
    """
    Vectorized SAT: does rectangle `rect` (4,2) touch/overlap ANY of `rects`
    (N,4,2)? Inclusive of boundary contact, matching shapely intersects.
    """
    rect = np.asarray(rect, dtype=float)
    rects = np.asarray(rects, dtype=float)
    if rects.size == 0:
        return False

    def edge_axes(r):
        edges = np.roll(r, -1, axis=-2) - r
        normals = np.stack([-edges[..., 1], edges[..., 0]], axis=-1)
        return normals[..., :2, :]        # 2 unique axes per rectangle

    axes1 = edge_axes(rect)                                   # (2,2)
    axesN = edge_axes(rects)                                  # (N,2,2)

    p_own_1 = rect @ axes1.T                                  # (4,2)
    p_oth_1 = rects @ axes1.T                                 # (N,4,2)
    ov1 = ((p_oth_1.min(axis=1) <= p_own_1.max(axis=0))
           & (p_own_1.min(axis=0) <= p_oth_1.max(axis=1)))    # (N,2)

    p_own_N = np.einsum('nax,cx->nac', axesN, rect)           # (N,2,4)
    p_oth_N = np.einsum('nax,ncx->nac', axesN, rects)         # (N,2,4)
    ov2 = ((p_oth_N.min(axis=2) <= p_own_N.max(axis=2))
           & (p_own_N.min(axis=2) <= p_oth_N.max(axis=2)))    # (N,2)

    return bool((ov1.all(axis=1) & ov2.all(axis=1)).any())


def get_laser_intersection_multi(center, laser_range, angle, multi_recs):

    # Compute the endpoint of the laser (limited range).
    laser_endpoint = (center[0] + laser_range * math.cos(angle),
                        center[1] + laser_range * math.sin(angle))
    laser_line = LineString([center, laser_endpoint])
    inter = laser_line.intersection(multi_recs)

    # Default to the laser endpoint if no valid intersection is found.
    intersection_point = laser_endpoint

    if inter.is_empty:
        # No intersection; use the laser endpoint.
        pass
    elif inter.geom_type == 'Point':
        intersection_point = (inter.x, inter.y)
    elif inter.geom_type == 'MultiPoint':
        points = list(inter.geoms)
        points.sort(key=lambda p: p.distance(Point(center)))
        intersection_point = (points[0].x, points[0].y)
    elif inter.geom_type == 'GeometryCollection':
        pts = [geom for geom in inter.geoms if geom.geom_type == 'Point']
        if pts:
            pts.sort(key=lambda p: p.distance(Point(center)))
            intersection_point = (pts[0].x, pts[0].y)
    elif inter.geom_type == 'LineString':
        # If the intersection is a LineString (e.g., laser grazes an edge),
        # take the endpoint of the line segment that is closest to the center.
        coords = list(inter.coords)
        p0 = Point(coords[0])
        p1 = Point(coords[-1])
        if p0.distance(Point(center)) < p1.distance(Point(center)):
            intersection_point = (p0.x, p0.y)
        else:
            intersection_point = (p1.x, p1.y)
    elif inter.geom_type == 'MultiLineString':
        # For multiple line segments, choose the closest endpoint among all segments.
        best = None
        best_dist = float('inf')
        for line in inter.geoms:
            coords = list(line.coords)
            for pt in [coords[0], coords[-1]]:
                d = Point(pt).distance(Point(center))
                if d < best_dist:
                    best_dist = d
                    best = pt
        if best is not None:
            intersection_point = best
    else:
        # Unhandled geometry type; default to laser endpoint.
        print(f"Unhandled geometry type: {inter.geom_type}; defaulting to laser endpoint.")

    return intersection_point


def get_laser_intersection_single(center, laser_range, angle, multi_recs):

    # Compute the endpoint of the laser (limited range).
    laser_endpoint = (center[0] + laser_range * math.cos(angle),
                        center[1] + laser_range * math.sin(angle))
    laser_line = LineString([center, laser_endpoint])
    inter = laser_line.intersection(multi_recs)

    coords = list(inter.coords)
    p0 = Point(coords[0])
    p1 = Point(coords[-1])
    intersection_point = (p1.x, p1.y)
    # intersection_point = (p0.x, p0.y)

    return intersection_point


def laser_intersections_multi(center, laser_range, angles, recs):
    """
    Given a list of angles, this function returns an array where each row contains:
    [angle, x-coordinate of intersection, y-coordinate of intersection]
    """
    intersections = []
    cleaned_recs = recs.buffer(0)

    for angle in angles:
        pt = get_laser_intersection_multi(center, laser_range, angle, cleaned_recs)
        distance = math.sqrt((pt[0] - center[0])**2 + (pt[1] - center[1])**2)
        intersections.append([angle, pt[0], pt[1], distance])
    return np.array(intersections)


def laser_intersections_single(center, laser_range, angles, recs):
    """
    Given a list of angles, this function returns an array where each row contains:
    [angle, x-coordinate of intersection, y-coordinate of intersection]
    """
    intersections = []
    cleaned_recs = recs.buffer(0)

    for angle in angles:
        pt = get_laser_intersection_single(center, laser_range, angle, cleaned_recs)
        distance = math.sqrt((pt[0] - center[0])**2 + (pt[1] - center[1])**2)
        intersections.append([angle, pt[0], pt[1], distance])
    return np.array(intersections)
