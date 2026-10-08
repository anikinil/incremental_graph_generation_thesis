"""Presentation settings shared by generated and handcrafted figures."""


LEGEND_FONT_SIZE = 24
LEGEND_POSITION = (0.78, 0.5)
LEGEND_RIGHT = 0.76

# Result plots centre their legends beside the centred graph.  The canvas
# boundary is tailored only for algorithm families whose later iterations add
# legend entries; static figures use no unnecessary reserved space.
RESULT_LEGEND_POSITION = (0.78, 0.5)
SCENARIO_LEGEND_POSITION = (0.86, 0.5)
GVD_LEGEND_CANVAS_RIGHT = 1.20
DYNAMIC_VGRAPH_LEGEND_CANVAS_RIGHT = 1.14
DRM_LEGEND_CANVAS_RIGHT = 1.23

# Distance maps use a continuous colour scale in addition to the regular
# categorical legend.  Give each a fixed, non-overlapping right-hand slot.
DISTANCE_MAP_COLORBAR_GAP = 0.01
DISTANCE_MAP_COLORBAR_WIDTH = 0.025
DISTANCE_MAP_LEGEND_POSITION = (0.90, 0.5)
