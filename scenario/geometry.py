from shapely.geometry import GeometryCollection, box
from shapely.ops import unary_union


SPECIAL_POINT_CLEARANCE_MARGIN = 0
# SPECIAL_POINT_CLEARANCE_MARGIN = 1e-6


def buffered_obstacle_space(occupancy, agv_radius, buffer_segments):
    """Return the union of occupied cells, inflated by the AGV radius."""
    
    occupied_cells = [box(c, r, c + 1, r + 1) for r, c in zip(*occupancy.nonzero())]
    return unary_union(occupied_cells).buffer(agv_radius, quad_segs=buffer_segments, join_style="round")


def offset_access_points(access_points, agv_radius, margin=SPECIAL_POINT_CLEARANCE_MARGIN):
    """Offset access points away from obstacles by the AGV radius plus a margin, to account for obstacle buffer."""
    clearance = agv_radius + margin
    offsets = {
        "n": (-clearance, 0.0),
        "s": (clearance, 0.0),
        "w": (0.0, -clearance),
        "e": (0.0, clearance),
    }

    offset_points = []
    for row, column, direction in access_points:
        offset_row, offset_column = offsets[direction]
        offset_points.append((row + offset_row, column + offset_column, direction))
    return offset_points


def centered_io_points(io_points):
    """Represent grid-cell IO points by their continuous cell centres."""
    return [(row, column) for row, column in io_points]
    # return [(row + 0.5, column + 0.5) for row, column in io_points]


def continuous_special_points(access_points, io_points, agv_radius,
                              margin=SPECIAL_POINT_CLEARANCE_MARGIN):
    """Convert grid special points to collision-free configurations via offsetting by the AGV radius."""
    offset_access = offset_access_points(access_points, agv_radius, margin)
    # swap row and column for shapely coordinates (x, y)
    continuous_access_points = [(column, row) for row, column, _ in offset_access]
    continuous_io_points = [(column, row) for row, column in centered_io_points(io_points)]
    return continuous_access_points, continuous_io_points
