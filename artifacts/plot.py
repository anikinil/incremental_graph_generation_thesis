
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, PathPatch, Rectangle, Patch
from matplotlib.collections import LineCollection, PatchCollection, QuadMesh
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from algorithms.geometry import buffered_obstacle_space
from artifacts.presentation import (
    DISTANCE_MAP_COLORBAR_GAP,
    DISTANCE_MAP_COLORBAR_WIDTH,
    DISTANCE_MAP_LEGEND_POSITION,
    DRM_LEGEND_CANVAS_RIGHT,
    DYNAMIC_VGRAPH_LEGEND_CANVAS_RIGHT,
    GVD_LEGEND_CANVAS_RIGHT,
    LEGEND_FONT_SIZE,
    LEGEND_RIGHT,
    RESULT_LEGEND_POSITION,
    SCENARIO_LEGEND_POSITION,
)

# color palette

ROBOT_RADIUS_BUFFER_COLOR = (0.8, 0.8, 0.8, 1)      # light grey

SIMPLE_OBSTACLE_COLOR = (0.3, 0.3, 0.3, 1)        # dark grey
UNIT_LOAD_COLOR = (0.5, 0.5, 0.5, 1)              # grey

VORO_SKELETON_COLOR = (0.3, 0.3, 0.5, 0.7)        # blue
AFFECTED_VORO_CELLS_COLOR = (0.8, 0.2, 0.2, 0.7)  # red overlay
AFFECTED_DISTANCES_COLOR = (0.9, 0.1, 0.1, 0.3)   # red
BRIDGE_COLOR = (0.2, 0.5, 0.2, 0.7)               # green

OFFSET_VEC_COLOR = (0.9, 0.2, 0.2, 0.8)  # red, semi-transparent

GRAPH_VERTEX_COLOR = (0.1, 0.4, 0.7, 1)           # blue
GRAPH_EDGE_COLOR = (0.1, 0.4, 0.7, 0.8)           # blue
PRM_EDGE_COLOR = (0.08, 0.34, 0.60, 1)            # darker blue
PRM_EDGE_ALPHA = 0.65

INACTIVE_GRAPH_VERTEX_COLOR = (0.4, 0.4, 0.4, 0.4)  # semi-transparent grey
INACTIVE_GRAPH_EDGE_COLOR = (0.35, 0.35, 0.35, 0.45)  # darker, more visible grey

AFFECTED_GRAPH_VERTEX_COLOR = (0.9, 0.2, 0.2, 1)   # red
AFFECTED_GRAPH_EDGE_COLOR = (0.9, 0.2, 0.2, 0.8)   # red

AFFECTED_INACTIVE_GRAPH_VERTEX_COLOR = (0.9, 0.2, 0.2, 0.2) # red
AFFECTED_INACTIVE_GRAPH_EDGE_COLOR = (0.85, 0.18, 0.18, 0.2) # darker, more visible red

# outline of affected region on the GVD distance map
AFFECTED_OUTLINE_COLOR = (0.9, 0.2, 0.2, 1)       # red

VISIBILITY_EDGE_COLOR = (0.1, 0.4, 0.7, 0.5)      # blue, semi-transparent for better visibility
RESULT_VISIBILITY_EDGE_COLOR = (0.08, 0.34, 0.60, 0.65)  # darker, more visible blue

IO_POINT_COLOR = (0.32, 0.70, 0.28)               # brighter, slightly yellow-green
ACCESS_POINT_COLOR = (0.9, 0.5, 0.1)              # orange
SPECIAL_POINT_EDGE_COLOR = (0.1, 0.1, 0.1, 1)     # dark grey

GRID_COLOR = (0, 0, 0, 1)                         # black

# sizes
# TODO
GRID_LINE_WIDTH = 0.5
UNIT_CELL_GRID_LINE_WIDTH = 0.9

# zorders
# TODO
GRID_ZORDER = 100
UNIT_CELL_GRID_ZORDER = GRID_ZORDER + 1
POINT_ZORDER = UNIT_CELL_GRID_ZORDER + 1
ROADMAP_EDGE_ZORDER = POINT_ZORDER + 1
ROADMAP_NODE_ZORDER = ROADMAP_EDGE_ZORDER + 1

RESULT_ROADMAP_NODE_SIZE = 36
# Access points are just larger than roadmap nodes, leaving a visible ring when
# a node shares their position.  Interaction points remain clearly larger.
RESULT_SPECIAL_POINT_SIZE = 120
RESULT_INTERACTION_POINT_SIZE = int(RESULT_SPECIAL_POINT_SIZE * 1.5)
# Cell-type keys are deliberately larger than point and edge keys, so their
# filled-cell meaning remains readable at thesis-print scale.
OBSTACLE_LEGEND_MARKER_SIZE = 24

# utility

def _obstacle_legend_handle(color, label):
    """Return a square marker for an obstacle-type legend entry."""
    return Line2D(
        [], [], marker="s", linestyle="", markersize=OBSTACLE_LEGEND_MARKER_SIZE,
        markerfacecolor=color, markeredgecolor="none", label=label,
    )


def _free_cell_legend_handle():
    """Return the shared square legend marker for unoccupied workspace cells."""
    return Line2D(
        [], [], marker="s", linestyle="", markersize=OBSTACLE_LEGEND_MARKER_SIZE,
        markerfacecolor="white", markeredgecolor="black", label="Free cell",
    )


class _DistanceMapFreeCellLegendHandle:
    """Marker type rendered as a distance-colour gradient in legends."""


class _DistanceMapFreeCellLegendHandler(HandlerBase):
    """Render a smooth diagonal plasma gradient in a cell-sized legend square."""

    _resolution = 24
    _gradient = (
        np.add.outer(
            np.linspace(0, 1, _resolution),
            np.linspace(0, 1, _resolution),
        ) / 2
    )

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width,
                       height, fontsize, trans):
        del orig_handle, fontsize
        # Legend-handler coordinates are in points; ``trans`` performs the
        # point-to-pixel conversion, so use the shared marker size directly.
        # Do not constrain this to the handler box: a normal legend handler is
        # shorter than a doubled cell key.  Its centred square may extend into
        # the available row spacing, just as a Line2D marker does.
        square_size = OBSTACLE_LEGEND_MARKER_SIZE
        x_position = xdescent + (width - square_size) / 2
        y_position = ydescent + (height - square_size) / 2
        cell_size = square_size / self._resolution
        colours = plt.cm.plasma(self._gradient)
        gradient_cells = [
            Rectangle(
                (x_position + column * cell_size, y_position + row * cell_size),
                cell_size,
                cell_size,
                facecolor=colours[row, column],
                edgecolor="none",
                antialiased=False,
                transform=trans,
            )
            for row in range(self._resolution)
            for column in range(self._resolution)
        ]
        outline = Rectangle(
            (x_position, y_position),
            square_size,
            square_size,
            fill=False,
            edgecolor="black",
            linewidth=1,
            transform=trans,
        )
        return [*gradient_cells, outline]


class _ZoneLegendHandler(HandlerBase):
    """Draw a filled-area key with the requested height scale.

    The handle box already provides the appropriate horizontal length.  Area
    keys can therefore differ in height without lengthening line or arrow
    samples in the same legend.
    """

    def __init__(self, height_scale):
        super().__init__()
        self.height_scale = height_scale

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width,
                       height, fontsize, trans):
        del legend, fontsize
        zone_height = self.height_scale * height
        zone = Rectangle(
            (xdescent, ydescent + (height - zone_height) / 2),
            width,
            zone_height,
            facecolor=orig_handle.get_facecolor(),
            edgecolor=orig_handle.get_edgecolor(),
            linewidth=orig_handle.get_linewidth(),
            linestyle=orig_handle.get_linestyle(),
            hatch=orig_handle.get_hatch(),
            alpha=orig_handle.get_alpha(),
            transform=trans,
            clip_on=False,
        )
        return [zone]


class _BufferZoneLegendPatch(Patch):
    """Legend proxy for the thinner robot-radius buffer swatch."""


_LARGE_ZONE_LEGEND_HANDLER = _ZoneLegendHandler(height_scale=2)
_BUFFER_ZONE_LEGEND_HANDLER = _ZoneLegendHandler(height_scale=1)


def _buffer_zone_legend_handle():
    """Return the shared, thinner legend proxy for robot-radius buffers."""
    return _BufferZoneLegendPatch(
        facecolor=ROBOT_RADIUS_BUFFER_COLOR,
        edgecolor="none",
        label="Robot radius buffer",
    )


def _area_legend_handler_map():
    """Return the standard handlers for enlarged zones and buffer zones."""
    return {
        Patch: _LARGE_ZONE_LEGEND_HANDLER,
        _BufferZoneLegendPatch: _BUFFER_ZONE_LEGEND_HANDLER,
    }


def _register_result_legend(axis, handles=None, labels=None, *,
                            free_cell_handle=None, handler_map=None,
                            cell_entries=None):
    """Register legend content; finalization applies the shared presentation."""
    if handles is None:
        handles, labels = axis.get_legend_handles_labels()

    if cell_entries is None:
        cell_entries = [
            (free_cell_handle or _free_cell_legend_handle(), "Free cell"),
            (_obstacle_legend_handle(SIMPLE_OBSTACLE_COLOR, "Simple obstacle"), "Simple obstacle"),
            (_obstacle_legend_handle(UNIT_LOAD_COLOR, "Unit-load obstacle"), "Unit-load obstacle"),
            (_buffer_zone_legend_handle(), "Robot radius buffer"),
        ]
    entries = list(zip(handles, labels))
    point_labels = ("Access points", "Interaction points")
    point_entries = [
        entry for point_label in point_labels
        for entry in entries if entry[1] == point_label
    ]
    other_entries = [entry for entry in entries if entry[1] not in point_labels]

    unique_entries = {}
    for handle, label in [*cell_entries, *point_entries, *other_entries]:
        unique_entries.setdefault(label, handle)
    result_handler_map = _area_legend_handler_map()
    if handler_map:
        result_handler_map.update(handler_map)
    legend = axis.legend(
        unique_entries.values(),
        unique_entries.keys(),
        markerscale=1,
        handler_map=result_handler_map,
    )
    # This temporary axes legend is replaced by a figure-level legend during
    # finalization.  Excluding it keeps tight_layout independent of how many
    # entries the current iteration has.
    legend.set_in_layout(False)
    legend._result_legend_handles = list(unique_entries.values())
    legend._result_legend_labels = list(unique_entries.keys())
    legend._result_legend_handler_map = result_handler_map
    return legend


def _with_vgraph_edge_legend_proxies(handles, labels):
    """Use readable legend samples without changing roadmap-edge rendering."""
    edge_colors = {
        "Roadmap edges": RESULT_VISIBILITY_EDGE_COLOR,
        "Active roadmap edges": GRAPH_EDGE_COLOR,
        "Inactive roadmap edges": INACTIVE_GRAPH_EDGE_COLOR,
    }
    return [
        Line2D([], [], color=edge_colors[label], linewidth=1.5,
               solid_capstyle="round")
        if label in edge_colors else handle
        for handle, label in zip(handles, labels)
    ]


def _order_legend_entries(handles, labels, preferred_labels):
    """Deduplicate legend entries while placing related roadmap items first."""
    unique_handles = dict(zip(labels, handles))
    ordered_labels = [
        label for label in preferred_labels if label in unique_handles
    ]
    ordered_labels.extend(
        label for label in unique_handles if label not in preferred_labels
    )
    return [unique_handles[label] for label in ordered_labels], ordered_labels


def _with_drm_edge_legend_proxies(handles, labels):
    """Give every DRM edge entry an equally readable legend sample."""
    edge_colors = {
        "Active roadmap edges": GRAPH_EDGE_COLOR,
        "Inactive roadmap edges": INACTIVE_GRAPH_EDGE_COLOR,
        "Affected active roadmap edges": AFFECTED_GRAPH_EDGE_COLOR,
        "Affected inactive roadmap edges": AFFECTED_INACTIVE_GRAPH_EDGE_COLOR,
    }
    return [
        Line2D([], [], color=edge_colors[label], linewidth=1.5,
               solid_capstyle="round")
        if label in edge_colors else handle
        for handle, label in zip(handles, labels)
    ]


def _move_legend_to_result_margin(fig, *, position=RESULT_LEGEND_POSITION,
                                  canvas_right=None,
                                  subplots_right=LEGEND_RIGHT):
    """Move any axes legends into the shared right-hand result margin."""
    handles = []
    labels = []
    handler_map = {}
    for axis in fig.axes:
        legend = axis.get_legend()
        if legend is None:
            continue

        legend_handles = getattr(legend, "legend_handles", None)
        if legend_handles is None:  # Matplotlib versions before 3.7
            legend_handles = legend.legendHandles
        handles.extend(getattr(legend, "_result_legend_handles", legend_handles))
        labels.extend(getattr(
            legend,
            "_result_legend_labels",
            [text.get_text() for text in legend.get_texts()],
        ))
        handler_map.update(getattr(legend, "_result_legend_handler_map", None) or {})
        legend.remove()

    if not handles:
        return

    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=position,
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        markerscale=1,
        handler_map=handler_map or None,
    )
    if canvas_right is not None:
        # ``bbox_inches="tight"`` would otherwise shrink PDFs from the same
        # dynamic algorithm to different widths as their legends change.
        # This transparent marker fixes a family-specific right boundary
        # without enlarging unrelated figures.
        fig.text(
            canvas_right,
            0.5,
            " ",
            transform=fig.transFigure,
            alpha=0,
            ha="left",
            va="center",
        )
    if subplots_right is not None:
        fig.subplots_adjust(right=subplots_right)


def _mask_from_cells(cells, height, width):
    """Returns a boolean mask of shape (height, width) with True at the given cells."""
    mask = np.zeros((height, width), dtype=bool)
    if not cells:
        return mask
    coords = np.asarray(list(cells), dtype=np.intp)
    coords = coords[(coords[:, 0] >= 0) & (coords[:, 0] < height) & (coords[:, 1] >= 0) & (coords[:, 1] < width)]
    mask[coords[:, 0], coords[:, 1]] = True
    return mask



def _build_buffer_patch(occupancy, robot_radius, num_buffer_segments, *, color=ROBOT_RADIUS_BUFFER_COLOR):
    """
    Builds vector patches for polygonal obstacle buffer used by PRM and VGraph.
    """
    # get approximation of buffered obstacle space as a Shapely geometry
    buffer_space = buffered_obstacle_space(occupancy, robot_radius, num_buffer_segments)

    # convert to matplotlib PatchCollection for plotting
    polygons = ([buffer_space] if buffer_space.geom_type == "Polygon" else list(buffer_space.geoms))
    def _polygon_path(polygon):
        vertices = []
        codes = []
        for ring in (polygon.exterior, *polygon.interiors):
            coordinates = np.asarray(ring.coords)
            vertices.extend(coordinates)
            codes.extend([Path.MOVETO] + [Path.LINETO] * (len(coordinates) - 2) + [Path.CLOSEPOLY])
        return Path(vertices, codes)
    patches = [PathPatch(_polygon_path(polygon)) for polygon in polygons]
    return PatchCollection(patches, facecolor=color, edgecolor="none", zorder=0)


def _build_smooth_buffer_patch(occupancy, robot_radius):
    """Build a smooth visual buffer for GVD plots (as they don't rely on polygonal approximation)."""
    rows, columns = np.where(occupancy > 0)
    patches = [
        FancyBboxPatch((column - robot_radius, row - robot_radius), 
                       1 + 2 * robot_radius,
                       1 + 2 * robot_radius,
                       boxstyle=f"round,pad=0,rounding_size={robot_radius}")
        for row, column in zip(rows, columns)
    ]
    return PatchCollection(patches, facecolor=ROBOT_RADIUS_BUFFER_COLOR, edgecolor="none", zorder=0)


def _occupancy_cell_image(occupancy):
    """RGBA array representing occupancy: transparent free cells, dark grey simple obstacles, grey unit loads."""
    vis_map = np.zeros((*occupancy.shape, 4), dtype=np.float32)
    vis_map[occupancy == 1] = SIMPLE_OBSTACLE_COLOR
    vis_map[occupancy == 2] = UNIT_LOAD_COLOR
    return vis_map


def _rgba_quadmesh(ax, rgba, extent, zorder=1):
    """
    Draws an RGBA array as a QuadMesh on the given axes, with the given extent and zorder.
    Used for occupancy and distance-map visualizations, where each cell is a single color.
    """
    height, width = rgba.shape[:2]
    x0, x1, y0, y1 = extent
    xs = np.linspace(x0, x1, width + 1)
    ys = np.linspace(y0, y1, height + 1)

    x_grid, y_grid = np.meshgrid(xs, ys)
    mesh = QuadMesh(np.stack((x_grid, y_grid), axis=-1), shading="flat", antialiased=False, zorder=zorder,)
    mesh.set_facecolor(np.asarray(rgba, dtype=np.float32).reshape(-1, 4))
    mesh.set_edgecolor("none")
    ax.add_collection(mesh)
    return mesh


def _major_ticks(lower, upper, max_ticks=10):
    """Return evenly spaced ticks without overcrowding a continuous plot."""
    step = max(1, int(np.ceil((upper - lower) / max_ticks)))
    return np.arange(lower, upper + 1, step)


def _draw_grid_lines(ax, x_boundaries, y_boundaries, *, linewidth=GRID_LINE_WIDTH, zorder=GRID_ZORDER):
    """Draw a uniform cell-boundary grid above every plot element."""
    x_boundaries = np.asarray(x_boundaries)
    y_boundaries = np.asarray(y_boundaries)
    if len(x_boundaries) < 2 or len(y_boundaries) < 2:
        return
    x_min, x_max = x_boundaries[0], x_boundaries[-1]
    y_min, y_max = y_boundaries[0], y_boundaries[-1]
    segments = [[(x, y_min), (x, y_max)] for x in x_boundaries] + [[(x_min, y), (x_max, y)] for y in y_boundaries]
    ax.add_collection(LineCollection(segments, colors=[GRID_COLOR], linewidths=linewidth, zorder=zorder))


def _setup_continuous_axes(ax, bounds):
    """Apply the shared PRM/VGraph axis labels and unit-cell grid."""
    min_x, min_y, max_x, max_y = bounds
    ax.set_xticks(_major_ticks(min_x, max_x))
    ax.set_yticks(_major_ticks(min_y, max_y))
    ax.grid(False)
    _draw_grid_lines(ax, np.arange(min_x, max_x + 1), np.arange(min_y, max_y + 1))
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(max_y, min_y)
    ax.set_aspect("equal")
    ax.margins(0)


def _draw_continuous_environment(ax, occupancy, bounds, *, robot_radius=None,
                                 num_buffer_segments=None):
    """Draw the occupancy grid plus an optional robot-radius obstacle buffer."""

    min_x, min_y, max_x, max_y = bounds

    if num_buffer_segments is not None:
        buffer = _build_buffer_patch(occupancy, robot_radius, num_buffer_segments)
        if buffer is not None:
            ax.add_collection(buffer)

    _rgba_quadmesh(ax, _occupancy_cell_image(occupancy), extent=(min_x, max_x, min_y, max_y), zorder=1,)
    _setup_continuous_axes(ax, bounds)


def _draw_continuous_special_points(ax, access_points, interaction_points, *, grid_coordinates=False,
                                    marker_size=RESULT_SPECIAL_POINT_SIZE, interaction_marker_size=None):
    """Draw PRM/VGraph-style special points, optionally converting grid coordinates."""
    if interaction_marker_size is None:
        interaction_marker_size = RESULT_INTERACTION_POINT_SIZE

    if grid_coordinates:
        access_points = [(column, row) for row, column, *_ in access_points]
        interaction_points = [(column + 0.5, row + 0.5) for row, column in interaction_points]

    if access_points:
        xs, ys = zip(*access_points)
        ax.scatter(xs, ys, s=marker_size, color=ACCESS_POINT_COLOR,
                   edgecolors="none",
                   zorder=POINT_ZORDER, label="Access points")
    if interaction_points:
        xs, ys = zip(*interaction_points)
        ax.scatter(xs, ys, s=interaction_marker_size, color=IO_POINT_COLOR,
                   edgecolors="none",
                   zorder=POINT_ZORDER, label="Interaction points")


def _draw_scenario_access_points(ax, access_points):
    """Draw scenario access points as circles at their shared cell-edge midpoints."""
    if not access_points:
        return None

    # Scenario.update_access_points() stores access points as (row, column,
    # direction), where row and column already identify the edge midpoint.
    points = np.asarray([(column, row) for row, column, _ in access_points], dtype=float)
    return ax.scatter(
        points[:, 0],
        points[:, 1],
        s=RESULT_SPECIAL_POINT_SIZE,
        color=ACCESS_POINT_COLOR,
        edgecolors="none",
        zorder=POINT_ZORDER,
        label="Access points",
    )


def _draw_grid_special_points(ax, access_points, interaction_points):
    """Scatter access and interaction points stored as grid (row, col[, direction]) tuples."""
    handles = []
    if access_points:
        access = np.asarray([(p[1], p[0]) for p in access_points], dtype=float)
        handles.append(ax.scatter(access[:, 0], access[:, 1], s=RESULT_SPECIAL_POINT_SIZE,
                                  color=ACCESS_POINT_COLOR,
                                  edgecolors="none", zorder=POINT_ZORDER,
                                  label="Access points"))
    if interaction_points:
        io = np.asarray([(p[1], p[0]) for p in interaction_points], dtype=float)
        handles.append(ax.scatter(io[:, 0], io[:, 1], s=RESULT_INTERACTION_POINT_SIZE,
                                  color=IO_POINT_COLOR,
                                  edgecolors="none", zorder=POINT_ZORDER,
                                  label="Interaction points"))
    return handles


def _setup_grid_axes(ax, height, width, scale_factor):
    """Set up GVD axes with fine GVD cells and thicker original unit-cell boundaries."""
    ax.set_xticks(np.arange(1, width, scale_factor))
    ax.set_yticks(np.arange(1, height, scale_factor))
    ax.tick_params(axis="both", which="both", length=0)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(False)
    _draw_grid_lines(ax, np.arange(width + 1), np.arange(height + 1))
    _draw_grid_lines(ax, np.arange(1, width, scale_factor), np.arange(1, height, scale_factor), linewidth=UNIT_CELL_GRID_LINE_WIDTH, zorder=UNIT_CELL_GRID_ZORDER)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal")
    ax.margins(0)


def _draw_affected_outline(ax, affected_cells, height, width):
    """Draw an outline along the boundaries of affected cells (GVD)."""
    LINE_WIDTH = 2
    affected = _mask_from_cells(affected_cells, height, width)
    for row, col in zip(*np.where(affected)):
        if col == 0 or not affected[row, col - 1]:
            ax.plot([col, col], [row, row + 1], color=AFFECTED_OUTLINE_COLOR, linewidth=LINE_WIDTH, zorder=50)
        if col == width - 1 or not affected[row, col + 1]:
            ax.plot([col + 1, col + 1], [row, row + 1], color=AFFECTED_OUTLINE_COLOR, linewidth=LINE_WIDTH, zorder=50)
        if row == 0 or not affected[row - 1, col]:
            ax.plot([col, col + 1], [row, row], color=AFFECTED_OUTLINE_COLOR, linewidth=LINE_WIDTH, zorder=50)
        if row == height - 1 or not affected[row + 1, col]:
            ax.plot([col, col + 1], [row + 1, row + 1], color=AFFECTED_OUTLINE_COLOR, linewidth=LINE_WIDTH, zorder=50)


class BasePlotter:
    """Interface implemented by every plotter."""

    def plot(self, data, scenario, path, title=None):
        """Receives the snapshot passed to ExperimentRecorder.save() as data"""
        fig, ax = self.setup_figure()
        self.draw(ax, data)
        if title:
            self.set_title(fig, ax, title)
        self.finalize(fig, ax)
        fig.savefig(path, format="pdf", bbox_inches="tight")
        plt.close(fig)

    def setup_figure(self):
        fig, ax = plt.subplots()
        return fig, ax

    def draw(self, ax, data):
        raise NotImplementedError

    def set_title(self, fig, ax, title):
        """Result plots intentionally do not display titles."""
        del fig, ax, title

    def finalize(self, fig, ax):
        target = ax[0] if isinstance(ax, (list, tuple)) else ax
        target.set_aspect("equal")
        fig.tight_layout()
        _move_legend_to_result_margin(fig)

class ScenarioPlotter(BasePlotter):
    """Plots the priority color bar, priority field, and occupancy map."""

    def setup_figure(self):
        fig = plt.figure(figsize=(12, 10))
        fig.subplots_adjust(left=0.11, right=0.98, bottom=0.07, top=0.95)
        gs = fig.add_gridspec(
            2,
            2,
            width_ratios=[0.07, 1],
            height_ratios=[1, 1],
            wspace=0.15,
            hspace=0.45,
        )
        ax_cbar = fig.add_subplot(gs[0, 0])
        ax_prob = fig.add_subplot(gs[0, 1])
        ax_occupancy = fig.add_subplot(gs[1, 1])
        return fig, (ax_cbar, ax_prob, ax_occupancy)

    def draw(self, ax, data):
        ax_cbar, ax_prob, ax_occupancy = ax

        occupancy = data.get("occupancy") if isinstance(data, dict) else data
        access_points = data.get("access_points", []) if isinstance(data, dict) else []
        interaction_points = data.get("interaction_points", []) if isinstance(data, dict) else []
        priority_field = data.get("priority_field") if isinstance(data, dict) else None

        height, width = occupancy.shape if occupancy is not None else (0, 0)

        ax_prob.set_title("Priority map")
        ax_occupancy.set_title("Occupancy map")

        # The QuadMesh keeps the priority map vector-based in the PDF.
        mesh = ax_prob.pcolormesh(priority_field, cmap="viridis", shading="flat", zorder=1)
        mesh.set_edgecolor("none")
        mesh.set_antialiased(False)

        _draw_continuous_environment(ax_occupancy, occupancy, bounds=(0, 0, width, height),)
        ax_occupancy.set_anchor("W")

        _setup_continuous_axes(ax_prob, bounds=(0, 0, width, height))
        ax_prob.set_anchor("W")
        _draw_scenario_access_points(ax_occupancy, access_points)
        _draw_continuous_special_points(ax_occupancy, [], interaction_points, grid_coordinates=True)
        _register_result_legend(
            ax_occupancy,
            cell_entries=[
                (_free_cell_legend_handle(), "Free cell"),
                (_obstacle_legend_handle(SIMPLE_OBSTACLE_COLOR, "Obstacle"), "Obstacle"),
            ],
        )

        colorbar = ax_occupancy.figure.colorbar(mesh, cax=ax_cbar, label="Priority")
        colorbar.ax.yaxis.set_ticks_position("left")
        colorbar.ax.yaxis.set_label_position("left")
        colorbar.ax.tick_params(pad=4)

        priority_position = ax_prob.get_position()
        colorbar_position = ax_cbar.get_position()
        ax_cbar.set_position([colorbar_position.x0, priority_position.y0, colorbar_position.width, priority_position.height])

        occupancy_position = ax_occupancy.get_position()
        arrow_x = (priority_position.x0 + priority_position.x1) / 2
        ax_occupancy.figure.add_artist(FancyArrowPatch(
            (arrow_x, priority_position.y0 - 0.04),
            (arrow_x, occupancy_position.y1 + 0.04),
            transform=ax_occupancy.figure.transFigure,
            arrowstyle="->",
            mutation_scale=16,
            linewidth=1.5,
            color="black",
            zorder=50,
            clip_on=False,
        ))

    def finalize(self, fig, ax):
        # The manually positioned colourbar and inter-map arrow must retain
        # their geometry, so extend the saved canvas for the legend instead of
        # shrinking this three-panel figure.
        del ax
        _move_legend_to_result_margin(
            fig,
            position=SCENARIO_LEGEND_POSITION,
            subplots_right=None,
        )


class _GVDGridPlotter(BasePlotter):
    """Base class for GVD-related grid plots (distance map, Voronoi skeleton)."""

    legend_canvas_right = GVD_LEGEND_CANVAS_RIGHT
    
    def _context(self, data):
        """Return the GVD context for plotting."""
        
        occupancy = np.asarray(data["occupancy"])
        robot_radius = data.get("robot_radius", 0)
        scale_factor = data.get("scale_factor", 1)
        access_points = data.get("access_points", [])
        interaction_points = data.get("interaction_points", [])
        iteration = data.get("iteration", 0)
        affected_distance_cells = data.get("affected_distance_cells", set())
        affected_voro_cells = data.get("affected_voro_cells", set())

        height, width = occupancy.shape

        return {
            "occupancy": occupancy,
            "robot_radius": robot_radius,
            "scale_factor": scale_factor,
            "height": height,
            "width": width,
            "access_points": access_points,
            "interaction_points": interaction_points,
            "iteration": iteration,
            "affected_distance_cells": affected_distance_cells,
            "affected_voro_cells": affected_voro_cells
        }

    def setup_figure(self):
        return plt.subplots(figsize=(16, 10))

    def _draw_obstacles(self, ax, occupancy, robot_radius):
        """Draw occupancy cells and the buffer zone."""
        height, width = occupancy.shape

        # obstacle buffer
        buffer_patch = _build_smooth_buffer_patch(occupancy, robot_radius)
        ax.add_collection(buffer_patch)

        # occupancy cells
        occ_image = _occupancy_cell_image(occupancy)
        _rgba_quadmesh(ax, occ_image, extent=(0, width, 0, height), zorder=2)

    def finalize(self, fig, ax):
        fig.tight_layout()
        _move_legend_to_result_margin(
            fig,
            canvas_right=self.legend_canvas_right,
        )


class DistanceMapPlotter(_GVDGridPlotter):
    """Plots distance map, with optional affected cells highlighted."""

    def draw(self, ax, data):
        self._distance_scalar_mappable = None
        ctx = self._context(data)
        occupancy = ctx["occupancy"]
        height = ctx["height"]
        width = ctx["width"]

        self._draw_obstacles(ax, occupancy, ctx["robot_radius"])

        D = data["D"] if isinstance(data, dict) else data
        D = np.asarray(D)

        finite = np.isfinite(D)
        D_colormap = np.zeros((*D.shape, 4))
        max_D = 0
        if np.any(finite):
            max_D = np.max(D[finite])
            D_normalized = np.zeros_like(D, dtype=float)
            D_normalized[finite] = D[finite] / max_D if max_D > 0 else 0
            D_colormap[finite] = plt.cm.plasma(D_normalized[finite])

        D_colormap[..., 3] *= 0.8
        _rgba_quadmesh(ax, D_colormap, extent=(0, width, 0, height), zorder=1)

        if ctx["iteration"] > 0:
            _draw_affected_outline(ax, ctx["affected_distance_cells"], height, width)

        if np.any(finite):
            norm = Normalize(vmin=0, vmax=max_D if max_D > 0 else 1)
            self._distance_scalar_mappable = ScalarMappable(
                cmap=plt.cm.plasma,
                norm=norm,
            )

        _setup_grid_axes(ax, height, width, ctx["scale_factor"])
        _draw_grid_special_points(ax, ctx["access_points"], ctx["interaction_points"])
        _register_result_legend(
            ax,
            free_cell_handle=_DistanceMapFreeCellLegendHandle(),
            handler_map={
                _DistanceMapFreeCellLegendHandle: _DistanceMapFreeCellLegendHandler(),
            },
        )

    def finalize(self, fig, ax):
        """Place a full-height distance scale without resizing the map axes."""
        fig.tight_layout()
        _move_legend_to_result_margin(
            fig,
            position=DISTANCE_MAP_LEGEND_POSITION,
            canvas_right=self.legend_canvas_right,
        )

        if self._distance_scalar_mappable is None:
            return

        map_position = ax.get_position()
        colorbar_axis = fig.add_axes([
            map_position.x1 + DISTANCE_MAP_COLORBAR_GAP,
            map_position.y0,
            DISTANCE_MAP_COLORBAR_WIDTH,
            map_position.height,
        ])
        colorbar = fig.colorbar(self._distance_scalar_mappable, cax=colorbar_axis)
        colorbar.set_label(
            "Squared distance to nearest obstacle",
            fontsize=LEGEND_FONT_SIZE,
        )
        colorbar.ax.tick_params(labelsize=LEGEND_FONT_SIZE)


class VoronoiSkeletonPlotter(_GVDGridPlotter):
    """Plots the Voronoi skeleton, with optional affected cells highlighted."""

    def draw(self, ax, data):
        ctx = self._context(data)

        self._draw_obstacles(ax, ctx["occupancy"], ctx["robot_radius"])

        height = ctx["height"]
        width = ctx["width"]

        voro = np.asarray(data["voro"])

        vis_map = np.zeros((*voro.shape, 4), dtype=np.float32)
        vis_map[voro] = VORO_SKELETON_COLOR

        if ctx["iteration"] > 0:
            vis_map[_mask_from_cells(ctx["affected_distance_cells"], height, width)] = AFFECTED_DISTANCES_COLOR
            vis_map[_mask_from_cells(ctx["affected_voro_cells"], height, width)] = AFFECTED_VORO_CELLS_COLOR

        _rgba_quadmesh(ax, vis_map, extent=(0, width, 0, height), zorder=1)

        _setup_grid_axes(ax, height, width, ctx["scale_factor"])
        _draw_grid_special_points(ax, ctx["access_points"], ctx["interaction_points"])
        _register_result_legend(
            ax,
            [Patch(facecolor=VORO_SKELETON_COLOR, edgecolor="none")],
            ["Voronoi skeleton"],
        )


class VoronoiSkeletonWithBridgeCells(_GVDGridPlotter):
    """Plots the Voronoi skeleton with highlighted bridge cells, plus optional affected cells."""
    
    def draw(self, ax, data):
        ctx = self._context(data)

        occupancy = ctx["occupancy"]
        height = ctx["height"]
        width = ctx["width"]

        self._draw_obstacles(ax, occupancy, ctx["robot_radius"])

        # voronoi skeleton
        voro = np.asarray(data["voro"])

        vis_map = np.zeros((*voro.shape, 4), dtype=np.float32)
        vis_map[voro] = VORO_SKELETON_COLOR

        # changed cells, if this is an incremental iteration
        if ctx["iteration"] > 0:
            vis_map[_mask_from_cells(ctx["affected_distance_cells"], height, width)] = AFFECTED_DISTANCES_COLOR
            vis_map[_mask_from_cells(ctx["affected_voro_cells"], height, width)] = AFFECTED_VORO_CELLS_COLOR

        # bridge cells are a boolean grid
        bridges = np.asarray(data["bridges"], dtype=bool)
        vis_map[bridges] = BRIDGE_COLOR

        _rgba_quadmesh(ax, vis_map, extent=(0, width, 0, height), zorder=1)

        _setup_grid_axes(ax, height, width, ctx["scale_factor"])
        _draw_grid_special_points(ax, ctx["access_points"], ctx["interaction_points"])
        _register_result_legend(ax)


class GraphPlotter(_GVDGridPlotter):
    """Plots the graph overlay on the GVD grid."""

    def draw(self, ax, data):
        ctx = self._context(data)
        occupancy = ctx["occupancy"]
        height, width = occupancy.shape

        self._draw_obstacles(ax, occupancy, ctx["robot_radius"])

        vis_map = np.zeros((height, width, 4), dtype=np.float32)

        if ctx["iteration"] > 0:
            vis_map[_mask_from_cells(ctx["affected_distance_cells"], height, width)] = AFFECTED_DISTANCES_COLOR

        voro = data.get("voro") if isinstance(data, dict) else None
        if voro is not None:
            vis_map[np.asarray(voro)] = VORO_SKELETON_COLOR

        if ctx["iteration"] > 0:
            vis_map[_mask_from_cells(ctx["affected_voro_cells"], height, width)] = AFFECTED_VORO_CELLS_COLOR

        bridges = data.get("bridges") if isinstance(data, dict) else None
        if bridges is not None:
            vis_map[np.asarray(bridges, dtype=bool)] = BRIDGE_COLOR

        _rgba_quadmesh(ax, vis_map, extent=(0, width, 0, height), zorder=1)

        _setup_grid_axes(ax, height, width, ctx["scale_factor"])
        _draw_grid_special_points(ax, ctx["access_points"], ctx["interaction_points"])

        graph = data.get("graph", data) if isinstance(data, dict) else None
        if graph and graph.get("vertices"):
            vertices = graph["vertices"]
            edges = graph["edges"]

            edge_segments = [[(vertices[v1][1], vertices[v1][0]), (vertices[v2][1], vertices[v2][0])] for v1, v2 in edges]
            ax.add_collection(LineCollection(edge_segments, colors=[GRAPH_EDGE_COLOR], linewidths=1.6, capstyle="round", joinstyle="round", zorder=ROADMAP_EDGE_ZORDER, label="Roadmap edges"))

            coords = np.asarray([(x, y) for y, x in vertices.values()], dtype=float)
            ax.scatter(coords[:, 0], coords[:, 1], color=GRAPH_VERTEX_COLOR,
                       edgecolors="none", s=RESULT_ROADMAP_NODE_SIZE,
                       zorder=ROADMAP_NODE_ZORDER, label="Roadmap nodes")

        handles, labels = ax.get_legend_handles_labels()
        if voro is not None:
            handles.append(Patch(facecolor=VORO_SKELETON_COLOR, edgecolor="none"))
            labels.append("Voronoi skeleton")
        if bridges is not None:
            handles.append(Patch(facecolor=BRIDGE_COLOR, edgecolor="none"))
            labels.append("Bridge cells")
        if ctx["iteration"] > 0:
            handles.extend([
                Patch(facecolor=AFFECTED_DISTANCES_COLOR, edgecolor="none"),
                Patch(facecolor=AFFECTED_VORO_CELLS_COLOR, edgecolor="none"),
            ])
            labels.extend(["Changed distance cells", "Affected Voronoi cells"])
        ordered_handles, ordered_labels = _order_legend_entries(
            handles, labels, ("Roadmap nodes", "Roadmap edges"),
        )
        _register_result_legend(ax, ordered_handles, ordered_labels)


class _ContinuousGraphPlotter(BasePlotter):
    """Shared rendering for PRM and visibility-graph snapshots."""

    requires_buffer_segments = True
    legend_canvas_right = None

    def setup_figure(self):
        return plt.subplots(figsize=(16, 12))

    def _context(self, data):
        required = ("occupancy", "robot_radius", "access_points", "interaction_points", "height", "width")
        if self.requires_buffer_segments:
            required += ("num_buffer_segments",)
        missing = [field for field in required if field not in data]
        if missing:
            raise ValueError(f"Continuous graph snapshot missing: {', '.join(missing)}")
        return data

    def _draw_environment(self, ax, context):
        occupancy = np.asarray(context["occupancy"])
        min_x, min_y, max_x, max_y = 0, 0, context["width"], context["height"]

        _draw_continuous_environment(
            ax,
            occupancy,
            bounds=(min_x, min_y, max_x, max_y),
            robot_radius=context["robot_radius"],
            num_buffer_segments=context["num_buffer_segments"],
        )

    def _draw_special_points(self, ax, context):
        _draw_continuous_special_points(
            ax,
            context["access_points"],
            context["interaction_points"],
            marker_size=RESULT_SPECIAL_POINT_SIZE,
            interaction_marker_size=RESULT_INTERACTION_POINT_SIZE,
        )

    def finalize(self, fig, ax):
        fig.tight_layout()
        _move_legend_to_result_margin(
            fig,
            canvas_right=self.legend_canvas_right,
        )


class PRMGraphPlotter(_ContinuousGraphPlotter):
    """Plot a self-contained PRM roadmap snapshot."""

    def draw(self, ax, data):
        context = self._context(data)
        self._draw_environment(ax, context)

        vertices = np.asarray(data["vertices"])
        edges = data["edges"]
        if len(edges):
            ax.add_collection(LineCollection([[vertices[left], vertices[right]] for left, right in edges],
                                             colors=[PRM_EDGE_COLOR], linewidths=1.1, alpha=PRM_EDGE_ALPHA,
                                             capstyle="round", zorder=ROADMAP_EDGE_ZORDER, label="Roadmap edges"))

        adjacency = [set() for _ in range(len(vertices))]
        for left, right in edges:
            adjacency[left].add(right)
            adjacency[right].add(left)
        components = []
        unseen = set(range(len(vertices)))
        while unseen:
            component = {unseen.pop()}
            frontier = list(component)
            while frontier:
                current = frontier.pop()
                for neighbor in adjacency[current] & unseen:
                    unseen.remove(neighbor)
                    component.add(neighbor)
                    frontier.append(neighbor)
            components.append(component)

        for component in components:
            coordinates = vertices[list(component)]
            ax.scatter(coordinates[:, 0], coordinates[:, 1], s=RESULT_ROADMAP_NODE_SIZE, color=GRAPH_VERTEX_COLOR, edgecolors="none", zorder=ROADMAP_NODE_ZORDER, label="Roadmap nodes")

        self._draw_special_points(ax, context)
        handles, labels = ax.get_legend_handles_labels()
        ordered_handles, ordered_labels = _order_legend_entries(
            handles, labels, ("Roadmap nodes", "Roadmap edges"),
        )
        _register_result_legend(ax, ordered_handles, ordered_labels)


class DRMGraphPlotter(_ContinuousGraphPlotter):
    """Plot a DRM snapshot, distinguishing active and inactive roadmap items."""

    legend_canvas_right = DRM_LEGEND_CANVAS_RIGHT

    def draw(self, ax, data):
        context = self._context(data)
        self._draw_environment(ax, context)

        vertices = np.asarray(data["vertices"])
        vertex_present = np.asarray(data.get("vertex_present", np.ones(len(vertices), dtype=bool)), dtype=bool)
        vertex_active = np.asarray(data["vertex_active"], dtype=bool)
        if len(vertex_present) != len(vertices) or len(vertex_active) != len(vertices):
            raise ValueError("DRM snapshot has mismatched vertex labels")

        active_edges = []
        inactive_edges = []
        all_edges = set()
        for left, right in data["edges"]:
            arc = (left, right) if left < right else (right, left)
            all_edges.add(arc)
            if (data["edge_active"].get(arc, False) and vertex_present[left] and vertex_present[right]
                and vertex_active[left] and vertex_active[right]):
                active_edges.append((left, right))
            else:
                inactive_edges.append((left, right))

        if inactive_edges:
            ax.add_collection(LineCollection([[vertices[left], vertices[right]] for left, right in inactive_edges],
                                             colors=[INACTIVE_GRAPH_EDGE_COLOR], linewidths=0.75, capstyle="round", zorder=ROADMAP_EDGE_ZORDER, label="Inactive roadmap edges"))
        if active_edges:
            ax.add_collection(LineCollection([[vertices[left], vertices[right]] for left, right in active_edges],
                                             colors=[GRAPH_EDGE_COLOR], linewidths=1.1, capstyle="round", zorder=ROADMAP_EDGE_ZORDER, label="Active roadmap edges"))

        inactive_vertex_mask = vertex_present & ~vertex_active
        active_vertex_mask = vertex_present & vertex_active
        if np.any(inactive_vertex_mask):
            inactive_vertices = vertices[inactive_vertex_mask]
            ax.scatter(inactive_vertices[:, 0], inactive_vertices[:, 1], s=RESULT_ROADMAP_NODE_SIZE,
                       color=INACTIVE_GRAPH_VERTEX_COLOR, edgecolors="none",
                       zorder=ROADMAP_NODE_ZORDER, label="Inactive roadmap nodes")
        if np.any(active_vertex_mask):
            active_vertices = vertices[active_vertex_mask]
            ax.scatter(active_vertices[:, 0], active_vertices[:, 1], s=RESULT_ROADMAP_NODE_SIZE,
                       color=GRAPH_VERTEX_COLOR, edgecolors="none",
                       zorder=ROADMAP_NODE_ZORDER, label="Active roadmap nodes")

        affected_edges = {
            tuple(sorted(edge)) for edge in data.get("affected_edges", ())
        } & all_edges
        affected_active_edges = [
            edge for edge in affected_edges
            if (data["edge_active"].get(edge, False) and vertex_present[edge[0]] and vertex_present[edge[1]]
                and vertex_active[edge[0]] and vertex_active[edge[1]])
        ]
        affected_inactive_edges = affected_edges.difference(affected_active_edges)
        if affected_active_edges:
            ax.add_collection(LineCollection(
                [[vertices[left], vertices[right]] for left, right in affected_active_edges],
                colors=[AFFECTED_GRAPH_EDGE_COLOR], linewidths=1.1,
                capstyle="round", zorder=ROADMAP_EDGE_ZORDER + 1, label="Affected active roadmap edges",
            ))
        if affected_inactive_edges:
            ax.add_collection(LineCollection(
                [[vertices[left], vertices[right]] for left, right in affected_inactive_edges],
                colors=[AFFECTED_INACTIVE_GRAPH_EDGE_COLOR], linewidths=0.75,
                capstyle="round", zorder=ROADMAP_EDGE_ZORDER + 1, label="Affected inactive roadmap edges",
            ))

        affected_vertices = sorted({
            int(index) for index in data.get("affected_vertices", ())
            if 0 <= int(index) < len(vertices) and vertex_present[int(index)]
        })
        affected_active_vertices = [index for index in affected_vertices if vertex_active[index]]
        affected_inactive_vertices = [index for index in affected_vertices if not vertex_active[index]]
        if affected_active_vertices:
            affected_coordinates = vertices[affected_active_vertices]
            ax.scatter(affected_coordinates[:, 0], affected_coordinates[:, 1], s=RESULT_ROADMAP_NODE_SIZE,
                       color=AFFECTED_GRAPH_VERTEX_COLOR, edgecolors="none",
                       zorder=ROADMAP_NODE_ZORDER + 1, label="Affected active roadmap nodes")
        if affected_inactive_vertices:
            affected_coordinates = vertices[affected_inactive_vertices]
            ax.scatter(affected_coordinates[:, 0], affected_coordinates[:, 1], s=RESULT_ROADMAP_NODE_SIZE,
                       color=AFFECTED_INACTIVE_GRAPH_VERTEX_COLOR, edgecolors="none",
                       zorder=ROADMAP_NODE_ZORDER + 1, label="Affected inactive roadmap nodes")

        self._draw_special_points(ax, context)
        handles, labels = ax.get_legend_handles_labels()
        unique_handles = dict(zip(labels, handles))
        roadmap_legend_order = (
            "Active roadmap nodes",
            "Active roadmap edges",
            "Inactive roadmap nodes",
            "Inactive roadmap edges",
            "Affected active roadmap nodes",
            "Affected active roadmap edges",
            "Affected inactive roadmap nodes",
            "Affected inactive roadmap edges",
        )
        unique_labels = [
            label for label in roadmap_legend_order if label in unique_handles
        ]
        unique_labels.extend(
            label for label in unique_handles if label not in roadmap_legend_order
        )
        _register_result_legend(
            ax,
            _with_drm_edge_legend_proxies(
                [unique_handles[label] for label in unique_labels],
                unique_labels,
            ),
            unique_labels,
        )


class IncrementalKingsGridPlotter(_GVDGridPlotter):
    """Plot an eight-connected occupancy-grid graph and its special points."""

    legend_canvas_right = None

    def draw(self, ax, data):
        ctx = self._context(data)
        occupancy = ctx["occupancy"]
        height = ctx["height"]
        width = ctx["width"]

        self._draw_obstacles(ax, occupancy, ctx["robot_radius"])

        vertices = data["vertices"]
        edges = data["edges"]
        if edges:
            ax.add_collection(LineCollection([[vertices[left]["pos"], vertices[right]["pos"]] for left, right in edges],
                                             colors=[GRAPH_EDGE_COLOR], linewidths=0.65, alpha=0.65, capstyle="round", zorder=ROADMAP_EDGE_ZORDER, label="Roadmap edges",))

        grid_vertices = [vertex["pos"] for vertex in vertices.values() if vertex["type"] == "grid"]
        if grid_vertices:
            vertices = np.asarray(grid_vertices, dtype=float)
            ax.scatter(vertices[:, 0], vertices[:, 1], s=RESULT_ROADMAP_NODE_SIZE,
                       color=GRAPH_VERTEX_COLOR, edgecolors="none",
                       zorder=ROADMAP_NODE_ZORDER, label="Roadmap nodes")

        _setup_grid_axes(ax, height, width, ctx["scale_factor"])
        _draw_grid_special_points(ax, ctx["access_points"], ctx["interaction_points"])

        handles, labels = ax.get_legend_handles_labels()
        ordered_handles, ordered_labels = _order_legend_entries(
            handles, labels, ("Roadmap nodes", "Roadmap edges"),
        )
        _register_result_legend(ax, ordered_handles, ordered_labels)


class VisibilityGraphPlotter(_ContinuousGraphPlotter):
    """Plot a self-contained visibility graph snapshot."""

    def draw(self, ax, data):
        context = self._context(data)
        self._draw_environment(ax, context)

        vertices = data["vertices"]
        edges = data["edges"]
        if edges:
            ax.add_collection(LineCollection(
                [[vertices[left]["pos"], vertices[right]["pos"]] for left, right in edges],
                colors=[RESULT_VISIBILITY_EDGE_COLOR], linewidths=0.45,
                capstyle="round", zorder=ROADMAP_EDGE_ZORDER, label="Roadmap edges",
            ))

        self._draw_special_points(ax, context)
        node_positions = np.asarray(
            [attributes["pos"] for attributes in vertices.values()],
            dtype=float,
        )
        if node_positions.size:
            ax.scatter(
                node_positions[:, 0],
                node_positions[:, 1],
                s=RESULT_ROADMAP_NODE_SIZE,
                color=GRAPH_VERTEX_COLOR,
                edgecolors="none",
                zorder=ROADMAP_NODE_ZORDER,
                label="Roadmap nodes",
            )
        handles, labels = ax.get_legend_handles_labels()
        ordered_handles, ordered_labels = _order_legend_entries(
            handles, labels, ("Roadmap nodes", "Roadmap edges"),
        )
        _register_result_legend(
            ax,
            _with_vgraph_edge_legend_proxies(ordered_handles, ordered_labels),
            ordered_labels,
        )


class DynamicVisibilityGraphPlotter(_ContinuousGraphPlotter):
    """Plot a dynamic visibility graph with active and inactive candidates."""

    requires_buffer_segments = False
    legend_canvas_right = DYNAMIC_VGRAPH_LEGEND_CANVAS_RIGHT

    def __init__(self, plot_buffer_segments=None):
        self.plot_buffer_segments = plot_buffer_segments

    def _draw_environment(self, ax, context):
        """Draw the buffer with an optional visualization-only resolution."""
        plot_context = context.copy()
        # DynamicVGraph uses square buffer corners, so preserve that appearance
        # when no plotting-only override was configured.
        plot_context["num_buffer_segments"] = (
            0 if self.plot_buffer_segments is None else self.plot_buffer_segments
        )
        return super()._draw_environment(ax, plot_context)

    def draw(self, ax, data):
        context = self._context(data)
        self._draw_environment(ax, context)

        edges = data["edges"]
        edge_active = data["edge_active"]
        active_edges = [edge for edge in edges if edge_active.get(edge, False)]
        inactive_edges = [edge for edge in edges if not edge_active.get(edge, False)]
        if active_edges:
            ax.add_collection(LineCollection(
                [[left, right] for left, right in active_edges],
                colors=[GRAPH_EDGE_COLOR], linewidths=0.9,
                capstyle="round", zorder=ROADMAP_EDGE_ZORDER, label="Active roadmap edges",
            ))

        if inactive_edges:
            ax.add_collection(LineCollection(
                [[left, right] for left, right in inactive_edges],
                colors=[INACTIVE_GRAPH_EDGE_COLOR], linewidths=0.55,
                capstyle="round", zorder=ROADMAP_EDGE_ZORDER, label="Inactive roadmap edges",
            ))

        vertices = np.asarray(data["vertices"], dtype=float)
        if vertices.size:
            vertices = vertices.reshape(-1, 2)
            ax.scatter(vertices[:, 0], vertices[:, 1], s=RESULT_ROADMAP_NODE_SIZE,
                       color=GRAPH_VERTEX_COLOR, edgecolors="none",
                       zorder=ROADMAP_NODE_ZORDER, label="Roadmap nodes")

        self._draw_special_points(ax, context)
        handles, labels = ax.get_legend_handles_labels()
        ordered_handles, ordered_labels = _order_legend_entries(
            handles,
            labels,
            ("Roadmap nodes", "Active roadmap edges", "Inactive roadmap edges"),
        )
        _register_result_legend(
            ax,
            _with_vgraph_edge_legend_proxies(
                ordered_handles, ordered_labels,
            ),
            ordered_labels,
        )

# -----------------------------------------------

class PlotManager:
    """Dispatches recorder.save(...) calls to the plotter registered for that data_type."""

    def __init__(self, scenario=None, dynamic_vgraph_plot_buffer_segments=None):
        self.scenario = scenario

        self.plotters = {
            "scenario": ScenarioPlotter(),
            "distance_map": DistanceMapPlotter(),
            "voronoi_skeleton": VoronoiSkeletonPlotter(),
            "voronoi_skeleton_with_bridges": VoronoiSkeletonWithBridgeCells(),
            "graph": GraphPlotter(),
            "prm_graph": PRMGraphPlotter(),
            "drm_graph": DRMGraphPlotter(),
            "incremental_kings_grid": IncrementalKingsGridPlotter(),
            "vgraph": VisibilityGraphPlotter(),
            "dynamic_vgraph": DynamicVisibilityGraphPlotter(
                plot_buffer_segments=dynamic_vgraph_plot_buffer_segments,
            ),
        }

    def plot(self, data_type, data, path, title_info=None):

        plotter = self.plotters.get(data_type)
        if plotter is None:
            raise ValueError(f"No plotter registered for data_type: {data_type}")

        plotter.plot(data=data, scenario=self.scenario, path=path)
