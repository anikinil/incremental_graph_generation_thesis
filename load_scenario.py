import numpy as np
import shapely

import matplotlib.pyplot as plt
import geopandas as gpd

import numpy as np
from shapely.geometry import box
from shapely.ops import unary_union

# very small offset for access points to make the algorithms recognize them as part of the free space
ERROR_TOLERANCE = 1e-6

# number of segments used to approximate the round corners of the obstacle buffer (0 = no rounding)
CORNER_SEGMENTS = 1

def load_scenario_as_shapely(iteration = 0):
    scenario = np.load(f"scenario/{iteration}.npz", allow_pickle=True)

    occupancy = scenario["occupancy"]
    occupancy_rate = scenario["occupancy_rate"]
    rows, cols = occupancy.shape

    squares = []
    for r in range(rows):
        for c in range(cols):
            if occupancy[r, c] == 0:
                squares.append(box(c, r, c + 1, r + 1))

    # free_space = unary_union(squares)
    
    # world = box(0, 0, scenario["width"], scenario["height"])

    # # original obstacles
    # obstacle_space = world.difference(free_space)

    # # expand obstacles by AGV radius
    # if CORNER_SEGMENTS == 0:
    #     obstacle_space = obstacle_space.buffer(
    #         scenario["agv_radius"],
    #         join_style=shapely.BufferJoinStyle.mitre
    #     )
    # else:
    #     obstacle_space = obstacle_space.buffer(
    #         scenario["agv_radius"],
    #         resolution=CORNER_SEGMENTS,  # number of segments used to approximate the round corners of the obstacle buffer
    #         join_style=shapely.BufferJoinStyle.round
    #     )
    
    # inner_frame = world.buffer(
    #     -scenario["agv_radius"],
    #     join_style=shapely.BufferJoinStyle.mitre
    # )

    # free_space = world.difference(obstacle_space)
    # free_space = free_space.intersection(inner_frame)
    
    world = box(0, 0, scenario["width"], scenario["height"])

    # build obstacles from occupancy
    grid_free_space = unary_union(squares)
    obstacle_space = world.difference(grid_free_space)

    # inflate obstacles
    if CORNER_SEGMENTS == 0:
        obstacle_space = obstacle_space.buffer(scenario["agv_radius"], join_style=shapely.BufferJoinStyle.mitre)
    else:
        obstacle_space = obstacle_space.buffer(scenario["agv_radius"], resolution=CORNER_SEGMENTS, join_style=shapely.BufferJoinStyle.round)

    inner_frame = world.buffer(-scenario["agv_radius"], join_style=shapely.BufferJoinStyle.mitre)

    # final free space
    free_space = world.difference(obstacle_space)
    free_space = free_space.intersection(inner_frame)

    # free_space = free_space.simplify(0, preserve_topology=True)
    # obstacle_space = obstacle_space.simplify(0, preserve_topology=True)

    # move access points out by agv_radius
    access_points = []
    for y, x, dir in scenario["access_points"]:
        y = np.float64(y)
        x = np.float64(x)
        if dir == "n":
            access_points.append((np.float64(y - scenario["agv_radius"] - ERROR_TOLERANCE), x, dir))
        elif dir == "s":
            access_points.append((np.float64(y + scenario["agv_radius"] + ERROR_TOLERANCE), x, dir))
        elif dir == "e":
            access_points.append((y, np.float64(x + scenario["agv_radius"] + ERROR_TOLERANCE), dir))
        elif dir == "w":
            access_points.append((y, np.float64(x - scenario["agv_radius"] - ERROR_TOLERANCE), dir))

    io_points = [(np.float64(y + 0.5), np.float64(x + 0.5)) for y, x in scenario["io_points"]]

    return (
        {
            "free_space": free_space,
            "obstacle_space": obstacle_space,
            "occupancy": occupancy,
            "occupancy_rate": occupancy_rate,
            "agv_radius": scenario["agv_radius"],
            "io_points": io_points,
            "access_points": access_points,
            "width": scenario["width"],
            "height": scenario["height"],
            "updates": np.array(scenario["updates"], dtype=object)
        }
    )

def load_scenario_as_numpy(iteration = 0):
    scenario = np.load(f"scenario/{iteration}.npz", allow_pickle=True)
    occupancy = scenario["occupancy"]
    occupancy_rate = scenario["occupancy_rate"]
    
    access_points = []
    for y, x, dir in scenario["access_points"]:
        y = np.float64(y)
        x = np.float64(x)
        if dir == "n":
            access_points.append((np.float64(y - scenario["agv_radius"] - ERROR_TOLERANCE), x, dir))
        elif dir == "s":
            access_points.append((np.float64(y + scenario["agv_radius"] + ERROR_TOLERANCE), x, dir))
        elif dir == "e":
            access_points.append((y, np.float64(x + scenario["agv_radius"] + ERROR_TOLERANCE), dir))
        elif dir == "w":
            access_points.append((y, np.float64(x - scenario["agv_radius"] - ERROR_TOLERANCE), dir))

    io_points = [(np.float64(y + 0.5), np.float64(x + 0.5)) for y, x in scenario["io_points"]]

    return (
        {
            "occupancy": occupancy,
            "occupancy_rate": occupancy_rate,
            "agv_radius": scenario["agv_radius"],
            "io_points": io_points,
            "access_points": access_points,
            "width": scenario["width"],
            "height": scenario["height"],
            "updates": np.array(scenario["updates"], dtype=object)
        }
    )

def load_obstacle_collision_zone(iteration = 0):
    """Loads infalted regions of obstacle space only."""
    
    scenario = np.load(f"scenario/{iteration}.npz", allow_pickle=True)
    occupancy = scenario["occupancy"]
    rows, cols = occupancy.shape

    squares = []
    for r in range(rows):
        for c in range(cols):
            if occupancy[r, c] == 0:
                squares.append(box(c, r, c + 1, r + 1))

    world = box(0, 0, scenario["width"], scenario["height"])

    # build obstacles from occupancy
    grid_free_space = unary_union(squares)
    obstacle_space = world.difference(grid_free_space)
    
    # add outer walls to obstacle space
    wall_thickness = scenario["agv_radius"]
    wall_left = box(-wall_thickness, 0, 0, scenario["height"])
    wall_right = box(scenario["width"], 0, scenario["width"] + wall_thickness, scenario["height"])
    wall_bottom = box(0, -wall_thickness, scenario["width"], 0)
    wall_top = box(0, scenario["height"], scenario["width"], scenario["height"] + wall_thickness)
    obstacle_space = unary_union([obstacle_space, wall_left, wall_right, wall_bottom, wall_top])

    # inflate obstacles
    if CORNER_SEGMENTS == 0:
        inflated_obstacle_space = obstacle_space.buffer(scenario["agv_radius"], join_style=shapely.BufferJoinStyle.mitre)
    else:
        inflated_obstacle_space = obstacle_space.buffer(scenario["agv_radius"], resolution=CORNER_SEGMENTS, join_style=shapely.BufferJoinStyle.round)

    # subtract obstacle space from infalted obstacle space
    difference = inflated_obstacle_space.difference(obstacle_space)
    
    return difference