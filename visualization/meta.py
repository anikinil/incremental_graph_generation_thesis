"""Shared presentation metadata for the handcrafted visualizations."""

from artifacts.plot import (
    ACCESS_POINT_COLOR,
    ROBOT_RADIUS_BUFFER_COLOR,
    GRAPH_NODE_COLOR,
    GRID_ZORDER,
    IO_POINT_COLOR,
    OFFSET_VEC_COLOR,
    SIMPLE_OBSTACLE_COLOR,
    SPECIAL_POINT_EDGE_COLOR,
    UNIT_LOAD_COLOR,
    VISIBILITY_EDGE_COLOR,
)
from artifacts.presentation import LEGEND_FONT_SIZE, LEGEND_POSITION, LEGEND_RIGHT


# Typography and layout
FONT_SIZE = LEGEND_FONT_SIZE
SINGLE_FIGURE_SIZE = (9, 5.5)
PRM_FIGURE_SIZE = (17, 6.5)

# Visualization-specific colors
WALL_COLOR = (0, 0, 0, 1)
WALL_THICKNESS = 0.25
RADIUS_OFFSET_COLOR = (0.2, 0.5, 0.2, 1)
ACCESS_POINT_ARROW_LENGTH = 0.42

# PRM-specific styling
PRM_GRAPH_COLOR = (0.28, 0.28, 0.28, 1)
PRM_SAMPLE_COLOR = PRM_GRAPH_COLOR
PRM_EDGE_COLOR = (0.28, 0.28, 0.28, 0.55)
PRM_START_COLOR = (0, 0, 0, 1)
PRM_GOAL_COLOR = (0, 0, 0, 1)
PRM_PATH_COLOR = (0, 0, 0, 1)
PRM_TERMINAL_LABEL_COLOR = "white"
PRM_TERMINAL_LABEL_SIZE = 11

# Interaction points use the same 225%-of-access-point emphasis as result
# plots, while access-point markers and graph nodes retain their own sizes.
INTERACTION_POINT_SIZE = 117
GRAPH_NODE_SIZE_SCALE = 2

# Sampling-based roadmap of trees (SRT) styling
SRT_CONNECTION_COLOR = (0, 0, 0, 0.9)
SRT_TREE_COLORS = (
    (0.20, 0.49, 0.72, 0.9),
    (0.89, 0.40, 0.13, 0.9),
    (0.23, 0.60, 0.42, 0.9),
    (0.76, 0.27, 0.29, 0.9),
    (0.51, 0.39, 0.66, 0.9),
    (0.65, 0.37, 0.23, 0.9),
    (0.82, 0.36, 0.59, 0.9),
    (0.50, 0.50, 0.12, 0.9),
    (0.10, 0.62, 0.62, 0.9),
    (0.30, 0.32, 0.66, 0.9),
    (0.90, 0.57, 0.10, 0.9),
    (0.40, 0.70, 0.17, 0.9),
    (0.72, 0.25, 0.13, 0.9),
    (0.38, 0.24, 0.58, 0.9),
)
