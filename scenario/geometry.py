from shapely.geometry import GeometryCollection, box
from shapely.ops import unary_union


def buffered_obstacle_space(occupancy, agv_radius, buffer_segments):
    """Return the union of occupied cells, inflated by the AGV radius."""
    
    occupied_cells = [box(c, r, c + 1, r + 1) for r, c in zip(*occupancy.nonzero())]
    return unary_union(occupied_cells).buffer(agv_radius, quad_segs=buffer_segments, join_style="round")
