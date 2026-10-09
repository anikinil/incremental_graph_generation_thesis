from math import floor, hypot, isclose

from shapely.geometry import GeometryCollection, box
from shapely.ops import unary_union


SPECIAL_POINT_CLEARANCE_MARGIN = 1e-6


def amanatides_woo_cells(start, end, height, width):
    """Returns cells crossed by a 2D Amanatides-Woo traversal.

    "A Fast Voxel Traversal Algorithm for Ray Tracing", John Amanatides and
    Andrew Woo (1987)

    https://m4xc.dev/articles/amanatides-and-woo/
    https://gitlab.com/athilenius/fast-voxel-traversal-rs/-/blob/main/src/raycast_2d.rs?ref_type=heads
    """
    
    # NOTE does NOT work for robot_radius = 0

    x0, y0 = start # starting point of the ray
    x1, y1 = end # ending point of the ray
    dx = x1 - x0 # total change in x
    dy = y1 - y0 # total change in y

    current_x = floor(x0) # current cell x-coordinate
    current_y = floor(y0) # current cell y-coordinate
    end_x = floor(x1) # end cell x-coordinate
    end_y = floor(y1) # end cell y-coordinate

    # set x-variables
    if dx > 0:
        step_x = 1 # step of one cell in the positive x direction
        t_delta_x = 1 / dx # how far along the ray we must move for the x-coordinate to change by 1
        t_max_x = (current_x + 1 - x0) / dx # how far along the ray we must move for the first x-boundary to be crossed
    elif dx < 0:
        step_x = -1
        t_delta_x = 1 / -dx
        t_max_x = (current_x - x0) / dx
    else: # dx == 0: the ray is vertical, x-boundaries will never be crossed
        step_x = 0
        t_delta_x = float("inf")
        t_max_x = float("inf")

    # set y-variables
    if dy > 0:
        step_y = 1 # step of one cell in the positive y direction
        t_delta_y = 1 / dy # how far along the ray we must move for the y-coordinate to change by 1
        t_max_y = (current_y + 1 - y0) / dy # how far along the ray we must move for the first y-boundary to be crossed
    elif dy < 0:
        step_y = -1
        t_delta_y = 1 / -dy
        t_max_y = (current_y - y0) / dy
    else: # dy == 0: the ray is horizontal, y-boundaries will never be crossed
        step_y = 0
        t_delta_y = float("inf")
        t_max_y = float("inf")

    cells = []
    seen = set()

    def _add_cell(row, column):
        if 0 <= row < height and 0 <= column < width and (row, column) not in seen:
            cells.append((row, column)) # (y, x) for row, column indexing
            seen.add((row, column))

    _add_cell(current_y, current_x)
    while current_x != end_x or current_y != end_y:
        if isclose(t_max_x, t_max_y): # ray crosses a cell corner
            if step_x == 0:
                current_y += step_y
                t_max_y += t_delta_y
                _add_cell(current_y, current_x)
            elif step_y == 0:
                current_x += step_x
                t_max_x += t_delta_x
                _add_cell(current_y, current_x)
            else:
                # add the two cells touched only at the corner before stepping diagonally
                next_x = current_x + step_x
                next_y = current_y + step_y
                _add_cell(current_y, next_x)
                _add_cell(next_y, current_x)
                current_x, current_y = next_x, next_y
                t_max_x += t_delta_x
                t_max_y += t_delta_y
                _add_cell(current_y, current_x)
        elif t_max_x < t_max_y:
            current_x += step_x
            t_max_x += t_delta_x
            _add_cell(current_y, current_x)
        else:
            current_y += step_y
            t_max_y += t_delta_y
            _add_cell(current_y, current_x)

    # if ray lies on a cell boundary, add the cells on the other side too
    if dy == 0 and isclose(y0, round(y0)):
        for row, column in tuple(cells):
            _add_cell(row - 1, column)
    if dx == 0 and isclose(x0, round(x0)):
        for row, column in tuple(cells):
            _add_cell(row, column - 1)

    return cells

def buffered_edge_cells(vertex1, vertex2, robot_radius, height, width):
    """Returns grid cells crossed by the buffered edge between two vertices."""

    x1, y1 = vertex1
    x2, y2 = vertex2
    dx = x2 - x1
    dy = y2 - y1
    length = hypot(dx, dy) # TODO: necessary?

    if robot_radius == 0: # if robot is a point, only traverse the edge centerline
        return amanatides_woo_cells(vertex1, vertex2, height, width)

    if length == 0:
        return amanatides_woo_cells(vertex1, vertex2, height, width)

    # offset the vertices in the positive and negative directions of the edge normal by the robot's radius
    offset_x = -dy / length * robot_radius
    offset_y = dx / length * robot_radius

    # get the grid cells crossed by the two offset edges using Amanatides-Woo traversal
    cells = amanatides_woo_cells((x1 + offset_x, y1 + offset_y), (x2 + offset_x, y2 + offset_y), height, width)
    cells.extend(amanatides_woo_cells((x1 - offset_x, y1 - offset_y), (x2 - offset_x, y2 - offset_y), height, width))
    return list(dict.fromkeys(cells))


def buffered_obstacle_space(occupancy, robot_radius, num_buffer_segments):
    """Return the union of occupied cells, inflated by the robot's radius."""
    
    occupied_cells = [box(c, r, c + 1, r + 1) for r, c in zip(*occupancy.nonzero())]
    
    if num_buffer_segments == 0: # 0 = 90°
        union = unary_union(occupied_cells).buffer(robot_radius, cap_style=2, join_style=2)
    else:
        union = unary_union(occupied_cells).buffer(robot_radius, quad_segs=num_buffer_segments, join_style="round")
        
    height, width = occupancy.shape
    world = box(0, 0, width, height)
    return union.intersection(world) # clip to world bounds

def offset_access_points(access_points, robot_radius):
    """Offset access points away from obstacles by the robot's radius plus a small margin."""
    clearance = robot_radius + SPECIAL_POINT_CLEARANCE_MARGIN
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


def continuous_special_points(access_points, interaction_points, robot_radius):
    """Convert grid special points to collision-free configurations via offsetting by the robot's radius."""
    offset_access = offset_access_points(access_points, robot_radius)
    # swap row and column for shapely coordinates (x, y)
    continuous_access_points = [(column, row) for row, column, _ in offset_access]
    continuous_interaction_points = [(column, row) for row, column in interaction_points]
    return continuous_access_points, continuous_interaction_points

def navigation_space(occupancy, robot_radius, num_buffer_segments):
    """
    Return the free space and obstacle space as Shapely geometries, with obstacles buffered for robot's radius.
    """
    world = box(0, 0, occupancy.shape[1], occupancy.shape[0])
    obstacle_space = buffered_obstacle_space(occupancy, robot_radius, num_buffer_segments)
    return world.difference(obstacle_space), obstacle_space # free_space, obstacle_space
