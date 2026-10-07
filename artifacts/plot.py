
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, PathPatch, Rectangle, Patch
from matplotlib.collections import LineCollection, PatchCollection, QuadMesh
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from algorithms.geometry import buffered_obstacle_space

# color palette

AGV_RADIUS_BUFFER_COLOR = (0.8, 0.8, 0.8, 1)      # light grey

SIMPLE_OBSTACLE_COLOR = (0.3, 0.3, 0.3, 1)        # dark grey
UNIT_LOAD_COLOR = (0.5, 0.5, 0.5, 1)              # grey

VORO_SKELETON_COLOR = (0.3, 0.3, 0.5, 0.7)        # blue
AFFECTED_VORO_CELLS_COLOR = (0.8, 0.2, 0.2, 0.7)  # red overlay
AFFECTED_DISTANCES_COLOR = (0.9, 0.1, 0.1, 0.3)   # red
BRIDGE_COLOR = (0.2, 0.5, 0.2, 0.7)               # green

OFFSET_VEC_COLOR = (0.9, 0.2, 0.2, 0.8)  # red, semi-transparent

GRAPH_VERTEX_COLOR = (0.1, 0.4, 0.7, 1)           # blue
GRAPH_EDGE_COLOR = (0.1, 0.4, 0.7, 0.8)           # blue

INACTIVE_GRAPH_VERTEX_COLOR = (0.4, 0.4, 0.4, 0.4)  # semi-transparent grey
INACTIVE_GRAPH_EDGE_COLOR = (0.4, 0.4, 0.4, 0.3)    # semi-transparent grey

AFFECTED_GRAPH_VERTEX_COLOR = (0.9, 0.2, 0.2, 1)   # red
AFFECTED_GRAPH_EDGE_COLOR = (0.9, 0.2, 0.2, 0.8)   # red

AFFECTED_INACTIVE_GRAPH_VERTEX_COLOR = (0.9, 0.2, 0.2, 0.2) # red
AFFECTED_INACTIVE_GRAPH_EDGE_COLOR = (0.9, 0.2, 0.2, 0.1) # red

# outline of affected region on the GVD distance map
AFFECTED_OUTLINE_COLOR = (0.9, 0.2, 0.2, 1)       # red

VISIBILITY_EDGE_COLOR = (0.1, 0.4, 0.7, 0.5)      # blue, semi-transparent for better visibility

IO_POINT_COLOR = (0.2, 0.6, 0.4)                  # green
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

# utility

def _mask_from_cells(cells, height, width):
    """Returns a boolean mask of shape (height, width) with True at the given cells."""
    mask = np.zeros((height, width), dtype=bool)
    if not cells:
        return mask
    coords = np.asarray(list(cells), dtype=np.intp)
    coords = coords[(coords[:, 0] >= 0) & (coords[:, 0] < height) & (coords[:, 1] >= 0) & (coords[:, 1] < width)]
    mask[coords[:, 0], coords[:, 1]] = True
    return mask



def _build_buffer_patch(occupancy, agv_radius, num_buffer_segments, *, color=AGV_RADIUS_BUFFER_COLOR):
    """
    Builds vector patches for polygonal obstacle buffer used by PRM and VGraph.
    """
    # get approximation of buffered obstacle space as a Shapely geometry
    buffer_space = buffered_obstacle_space(occupancy, agv_radius, num_buffer_segments)

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


def _build_smooth_buffer_patch(occupancy, agv_radius):
    """Build a smooth visual buffer for GVD plots (as they don't rely on polygonal approximation)."""
    rows, columns = np.where(occupancy > 0)
    patches = [
        FancyBboxPatch((column - agv_radius, row - agv_radius), 
                       1 + 2 * agv_radius,
                       1 + 2 * agv_radius,
                       boxstyle=f"round,pad=0,rounding_size={agv_radius}")
        for row, column in zip(rows, columns)
    ]
    return PatchCollection(patches, facecolor=AGV_RADIUS_BUFFER_COLOR, edgecolor="none", zorder=0)


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


def _draw_continuous_environment(ax, occupancy, bounds, *, agv_radius=None,
                                 num_buffer_segments=None):
    """Draw the occupancy grid plus an optional AGV-radius obstacle buffer."""

    min_x, min_y, max_x, max_y = bounds

    if num_buffer_segments is not None:
        buffer = _build_buffer_patch(occupancy, agv_radius, num_buffer_segments)
        if buffer is not None:
            ax.add_collection(buffer)

    _rgba_quadmesh(ax, _occupancy_cell_image(occupancy), extent=(min_x, max_x, min_y, max_y), zorder=1,)
    _setup_continuous_axes(ax, bounds)


def _draw_continuous_special_points(ax, access_points, io_points, *, grid_coordinates=False):
    """Draw PRM/VGraph-style special points, optionally converting grid coordinates."""
    if grid_coordinates:
        access_points = [(column, row) for row, column, *_ in access_points]
        io_points = [(column + 0.5, row + 0.5) for row, column in io_points]

    if access_points:
        xs, ys = zip(*access_points)
        ax.scatter(xs, ys, s=20, color=ACCESS_POINT_COLOR,
                   edgecolors=SPECIAL_POINT_EDGE_COLOR, linewidths=0.7,
                   zorder=20, label="Access points")
    if io_points:
        xs, ys = zip(*io_points)
        ax.scatter(xs, ys, s=20, color=IO_POINT_COLOR,
                   edgecolors=SPECIAL_POINT_EDGE_COLOR, linewidths=0.8,
                   zorder=20, label="IO points")


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
        s=42,
        color=ACCESS_POINT_COLOR,
        edgecolors=SPECIAL_POINT_EDGE_COLOR,
        linewidths=0.7,
        zorder=GRID_ZORDER + 2,
        label="Access points",
    )


def _draw_grid_special_points(ax, access_points, io_points):
    """Scatter access and IO-points stored as grid (row, col[, direction]) tuples."""
    handles = []
    if access_points:
        access = np.asarray([(p[1], p[0]) for p in access_points], dtype=float)
        handles.append(ax.scatter(access[:, 0], access[:, 1], s=34,
                                  color=ACCESS_POINT_COLOR,
                                  edgecolors=SPECIAL_POINT_EDGE_COLOR,
                                  linewidths=0.7, zorder=20,
                                  label="Access points"))
    if io_points:
        io = np.asarray([(p[1], p[0]) for p in io_points], dtype=float)
        handles.append(ax.scatter(io[:, 0], io[:, 1], s=48,
                                  color=IO_POINT_COLOR,
                                  edgecolors=SPECIAL_POINT_EDGE_COLOR,
                                  linewidths=0.8, zorder=20,
                                  label="IO points"))
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
        target = ax[0] if isinstance(ax, (list, tuple)) else ax
        target.set_title(title)

    def finalize(self, fig, ax):
        target = ax[0] if isinstance(ax, (list, tuple)) else ax
        target.set_aspect("equal")
        fig.tight_layout()

class ScenarioPlotter(BasePlotter):
    """Plots the probability color bar, probability field, and obstacle layout."""

    def setup_figure(self):
        fig = plt.figure(figsize=(12, 5))
        fig.subplots_adjust(left=0.11, right=0.98, bottom=0.1, top=0.9)
        gs = fig.add_gridspec(1, 3, width_ratios=[0.07, 1, 1], wspace=0.15)
        ax_cbar = fig.add_subplot(gs[0, 0])
        ax_prob = fig.add_subplot(gs[0, 1])
        ax_obst = fig.add_subplot(gs[0, 2])
        return fig, (ax_cbar, ax_prob, ax_obst)

    def draw(self, ax, data):
        ax_cbar, ax_prob, ax_obst = ax

        occupancy = data.get("occupancy") if isinstance(data, dict) else data
        access_points = data.get("access_points", []) if isinstance(data, dict) else []
        io_points = data.get("io_points", []) if isinstance(data, dict) else []
        probability_field = data.get("probability_field") if isinstance(data, dict) else None

        height, width = occupancy.shape if occupancy is not None else (0, 0)

        ax_obst.set_title("Obstacle layout")
        ax_prob.set_title("Probability map")

        # The QuadMesh keeps the probability map vector-based in the PDF.
        mesh = ax_prob.pcolormesh(probability_field, cmap="viridis", shading="flat", zorder=1)
        mesh.set_edgecolor("none")
        mesh.set_antialiased(False)

        _draw_continuous_environment(ax_obst, occupancy, bounds=(0, 0, width, height),)
        # make centre gap available for the arrow.
        ax_obst.yaxis.tick_right()

        _setup_continuous_axes(ax_prob, bounds=(0, 0, width, height))
        _draw_scenario_access_points(ax_obst, access_points)
        _draw_continuous_special_points(ax_obst, [], io_points, grid_coordinates=True)

        colorbar = ax_obst.figure.colorbar(mesh, cax=ax_cbar, label="Probability")
        colorbar.ax.yaxis.set_ticks_position("left")
        colorbar.ax.yaxis.set_label_position("left")
        colorbar.ax.tick_params(pad=4)

        probability_position = ax_prob.get_position()
        colorbar_position = ax_cbar.get_position()
        ax_cbar.set_position([colorbar_position.x0, probability_position.y0, colorbar_position.width, probability_position.height])

        obstacle_position = ax_obst.get_position()
        arrow_y = (probability_position.y0 + probability_position.y1) / 2
        ax_obst.figure.add_artist(FancyArrowPatch(
            (probability_position.x1 + 0.006, arrow_y),
            (obstacle_position.x0 - 0.006, arrow_y),
            transform=ax_obst.figure.transFigure,
            arrowstyle="->",
            mutation_scale=16,
            linewidth=1.5,
            color="black",
            zorder=50,
            clip_on=False,
        ))

    def set_title(self, fig, ax, title):
        fig.suptitle(title)

    def finalize(self, fig, ax):
        pass


class _GVDGridPlotter(BasePlotter):
    """Base class for GVD-related grid plots (distance map, Voronoi skeleton)."""
    
    def _context(self, data):
        """Return the GVD context for plotting."""
        
        occupancy = np.asarray(data["occupancy"])
        agv_radius = data.get("agv_radius", 0)
        scale_factor = data.get("scale_factor", 1)
        access_points = data.get("access_points", [])
        io_points = data.get("io_points", [])
        iteration = data.get("iteration", 0)
        affected_distance_cells = data.get("affected_distance_cells", set())
        affected_voro_cells = data.get("affected_voro_cells", set())

        height, width = occupancy.shape

        return {
            "occupancy": occupancy,
            "agv_radius": agv_radius,
            "scale_factor": scale_factor,
            "height": height,
            "width": width,
            "access_points": access_points,
            "io_points": io_points,
            "iteration": iteration,
            "affected_distance_cells": affected_distance_cells,
            "affected_voro_cells": affected_voro_cells
        }

    def setup_figure(self):
        return plt.subplots(figsize=(16, 10))

    def _draw_obstacles(self, ax, occupancy, agv_radius):
        """Draw occupancy cells and the buffer zone."""
        height, width = occupancy.shape

        # obstacle buffer
        buffer_patch = _build_smooth_buffer_patch(occupancy, agv_radius)
        ax.add_collection(buffer_patch)

        # occupancy cells
        occ_image = _occupancy_cell_image(occupancy)
        _rgba_quadmesh(ax, occ_image, extent=(0, width, 0, height), zorder=2)

    def finalize(self, fig, ax):
        fig.tight_layout()


class DistanceMapPlotter(_GVDGridPlotter):
    """Plots distance map, with optional affected cells highlighted."""

    def draw(self, ax, data):
        ctx = self._context(data)
        occupancy = ctx["occupancy"]
        height = ctx["height"]
        width = ctx["width"]

        self._draw_obstacles(ax, occupancy, ctx["agv_radius"])

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
            sm = ScalarMappable(cmap=plt.cm.plasma, norm=norm)
            fig = ax.figure
            fig.colorbar(sm, ax=ax, label="Squared distance to nearest obstacle")

        _setup_grid_axes(ax, height, width, ctx["scale_factor"])
        _draw_grid_special_points(ax, ctx["access_points"], ctx["io_points"])


class VoronoiSkeletonPlotter(_GVDGridPlotter):
    """Plots the Voronoi skeleton, with optional affected cells highlighted."""

    def draw(self, ax, data):
        ctx = self._context(data)

        self._draw_obstacles(ax, ctx["occupancy"], ctx["agv_radius"])

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
        _draw_grid_special_points(ax, ctx["access_points"], ctx["io_points"])


class VoronoiSkeletonWithBridgeCells(_GVDGridPlotter):
    """Plots the Voronoi skeleton with highlighted bridge cells, plus optional affected cells."""
    
    def draw(self, ax, data):
        ctx = self._context(data)

        occupancy = ctx["occupancy"]
        height = ctx["height"]
        width = ctx["width"]

        self._draw_obstacles(ax, occupancy, ctx["agv_radius"])

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
        _draw_grid_special_points(ax, ctx["access_points"], ctx["io_points"])


class GraphPlotter(_GVDGridPlotter):
    """Plots the graph overlay on the GVD grid."""

    def draw(self, ax, data):
        ctx = self._context(data)
        occupancy = ctx["occupancy"]
        height, width = occupancy.shape

        self._draw_obstacles(ax, occupancy, ctx["agv_radius"])

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
        _draw_grid_special_points(ax, ctx["access_points"], ctx["io_points"])

        graph = data.get("graph", data) if isinstance(data, dict) else None
        if graph and graph.get("vertices"):
            vertices = graph["vertices"]
            edges = graph["edges"]

            edge_segments = [[(vertices[v1][1], vertices[v1][0]), (vertices[v2][1], vertices[v2][0])] for v1, v2 in edges]
            ax.add_collection(LineCollection(edge_segments, colors=[GRAPH_EDGE_COLOR], linewidths=1.6, capstyle="round", joinstyle="round", zorder=10, label="Graph edges"))

            coords = np.asarray([(x, y) for y, x in vertices.values()], dtype=float)
            ax.scatter(coords[:, 0], coords[:, 1], color=GRAPH_VERTEX_COLOR,
                       edgecolors=SPECIAL_POINT_EDGE_COLOR, linewidths=0.6,
                       s=13, zorder=11, label="Graph vertices")

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
        ax.legend(handles, labels, loc="upper right", fontsize="small",
                  frameon=True, framealpha=0.92)


class _ContinuousGraphPlotter(BasePlotter):
    """Shared rendering for PRM and visibility-graph snapshots."""

    requires_buffer_segments = True

    def setup_figure(self):
        return plt.subplots(figsize=(16, 12))

    def _context(self, data):
        required = ("occupancy", "agv_radius", "access_points", "io_points", "height", "width")
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
            agv_radius=context["agv_radius"],
            num_buffer_segments=context["num_buffer_segments"],
        )

    def _draw_special_points(self, ax, context):
        _draw_continuous_special_points(ax, context["access_points"], context["io_points"])

    def finalize(self, fig, ax):
        fig.tight_layout()


class PRMGraphPlotter(_ContinuousGraphPlotter):
    """Plot a self-contained PRM roadmap snapshot."""

    def draw(self, ax, data):
        context = self._context(data)
        self._draw_environment(ax, context)

        vertices = np.asarray(data["vertices"])
        edges = data["edges"]
        if len(edges):
            ax.add_collection(LineCollection([[vertices[left], vertices[right]] for left, right in edges],
                                             colors=[GRAPH_EDGE_COLOR], linewidths=1.1, alpha=0.55,
                                             capstyle="round", zorder=10, label="Graph edges"))

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
            ax.scatter(coordinates[:, 0], coordinates[:, 1], s=5, color=GRAPH_VERTEX_COLOR, edgecolors=SPECIAL_POINT_EDGE_COLOR, linewidths=0.35, zorder=11, label="Graph vertices")

        self._draw_special_points(ax, context)
        handles, labels = ax.get_legend_handles_labels()
        unique_handles = dict(zip(labels, handles))
        ax.legend(unique_handles.values(), unique_handles.keys(), loc="upper right", fontsize="small", frameon=True, framealpha=0.92)


class DRMGraphPlotter(_ContinuousGraphPlotter):
    """Plot a DRM snapshot, distinguishing active and inactive roadmap items."""

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
                                             colors=[INACTIVE_GRAPH_EDGE_COLOR], linewidths=0.75, capstyle="round", zorder=8, label="Inactive roadmap arcs"))
        if active_edges:
            ax.add_collection(LineCollection([[vertices[left], vertices[right]] for left, right in active_edges],
                                             colors=[GRAPH_EDGE_COLOR], linewidths=1.1, capstyle="round", zorder=10, label="Active roadmap arcs"))

        inactive_vertex_mask = vertex_present & ~vertex_active
        active_vertex_mask = vertex_present & vertex_active
        if np.any(inactive_vertex_mask):
            inactive_vertices = vertices[inactive_vertex_mask]
            ax.scatter(inactive_vertices[:, 0], inactive_vertices[:, 1], s=13,
                       color=INACTIVE_GRAPH_VERTEX_COLOR, edgecolors="none",
                       zorder=9, label="Inactive roadmap vertices")
        if np.any(active_vertex_mask):
            active_vertices = vertices[active_vertex_mask]
            ax.scatter(active_vertices[:, 0], active_vertices[:, 1], s=18,
                       color=GRAPH_VERTEX_COLOR, edgecolors=SPECIAL_POINT_EDGE_COLOR,
                       linewidths=0.35, zorder=11, label="Active roadmap vertices")

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
                capstyle="round", zorder=13, label="Affected roadmap arcs",
            ))
        if affected_inactive_edges:
            ax.add_collection(LineCollection(
                [[vertices[left], vertices[right]] for left, right in affected_inactive_edges],
                colors=[AFFECTED_INACTIVE_GRAPH_EDGE_COLOR], linewidths=0.75,
                capstyle="round", zorder=13, label="Affected roadmap arcs",
            ))

        affected_vertices = sorted({
            int(index) for index in data.get("affected_vertices", ())
            if 0 <= int(index) < len(vertices) and vertex_present[int(index)]
        })
        affected_active_vertices = [index for index in affected_vertices if vertex_active[index]]
        affected_inactive_vertices = [index for index in affected_vertices if not vertex_active[index]]
        if affected_active_vertices:
            affected_coordinates = vertices[affected_active_vertices]
            ax.scatter(affected_coordinates[:, 0], affected_coordinates[:, 1], s=18,
                       color=AFFECTED_GRAPH_VERTEX_COLOR, edgecolors=SPECIAL_POINT_EDGE_COLOR,
                       linewidths=0.35, zorder=14, label="Affected roadmap vertices")
        if affected_inactive_vertices:
            affected_coordinates = vertices[affected_inactive_vertices]
            ax.scatter(affected_coordinates[:, 0], affected_coordinates[:, 1], s=13,
                       color=AFFECTED_INACTIVE_GRAPH_VERTEX_COLOR, edgecolors="none",
                       zorder=14, label="Affected roadmap vertices")

        self._draw_special_points(ax, context)
        handles, labels = ax.get_legend_handles_labels()
        unique_handles = dict(zip(labels, handles))
        ax.legend(unique_handles.values(), unique_handles.keys(), loc="upper right", fontsize="small", frameon=True, framealpha=0.92)


class IncrementalKingsGridPlotter(_GVDGridPlotter):
    """Plot an eight-connected occupancy-grid graph and its special points."""

    def draw(self, ax, data):
        ctx = self._context(data)
        occupancy = ctx["occupancy"]
        height = ctx["height"]
        width = ctx["width"]

        self._draw_obstacles(ax, occupancy, ctx["agv_radius"])

        vertices = data["vertices"]
        edges = data["edges"]
        if edges:
            ax.add_collection(LineCollection([[vertices[left]["pos"], vertices[right]["pos"]] for left, right in edges],
                                             colors=[GRAPH_EDGE_COLOR], linewidths=0.65, alpha=0.65, capstyle="round", zorder=10, label="King's-grid edges",))

        grid_vertices = [vertex["pos"] for vertex in vertices.values() if vertex["type"] == "grid"]
        if grid_vertices:
            vertices = np.asarray(grid_vertices, dtype=float)
            ax.scatter(vertices[:, 0], vertices[:, 1], s=9,
                       color=GRAPH_VERTEX_COLOR, edgecolors="none",
                       zorder=11, label="King's-grid vertices")

        _setup_grid_axes(ax, height, width, ctx["scale_factor"])
        _draw_grid_special_points(ax, ctx["access_points"], ctx["io_points"])

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            unique_handles = dict(zip(labels, handles))
            ax.legend(unique_handles.values(), unique_handles.keys(), loc="upper right",
                      fontsize="small", frameon=True, framealpha=0.92)


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
                colors=[VISIBILITY_EDGE_COLOR], linewidths=0.45,
                capstyle="round", zorder=10, label="Visibility edges",
            ))

        self._draw_special_points(ax, context)
        ax.legend(loc="upper right", fontsize="small", frameon=True, framealpha=0.92)


class DynamicVisibilityGraphPlotter(_ContinuousGraphPlotter):
    """Plot a dynamic visibility graph with active and inactive candidates."""

    requires_buffer_segments = False

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

        if inactive_edges:
            ax.add_collection(LineCollection(
                [[left, right] for left, right in inactive_edges],
                colors=[INACTIVE_GRAPH_EDGE_COLOR], linewidths=0.55,
                capstyle="round", zorder=8, label="Inactive visibility edges",
            ))
        if active_edges:
            ax.add_collection(LineCollection(
                [[left, right] for left, right in active_edges],
                colors=[GRAPH_EDGE_COLOR], linewidths=0.9,
                capstyle="round", zorder=10, label="Active visibility edges",
            ))

        vertices = np.asarray(data["vertices"], dtype=float)
        if vertices.size:
            vertices = vertices.reshape(-1, 2)
            ax.scatter(vertices[:, 0], vertices[:, 1], s=15,
                       color=GRAPH_VERTEX_COLOR, edgecolors=SPECIAL_POINT_EDGE_COLOR,
                       linewidths=0.35, zorder=11, label="Visibility vertices")

        self._draw_special_points(ax, context)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            unique_handles = dict(zip(labels, handles))
            ax.legend(unique_handles.values(), unique_handles.keys(), loc="upper right",
                      fontsize="small", frameon=True, framealpha=0.92)

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

        algorithm = title_info.get("algorithm") if title_info else None
        iteration = title_info.get("iteration", 0) if title_info else 0
        display_name = {
            "gvd": "GVD",
            "prm": "PRM",
            "drm": "Dynamic Roadmap",
            "incremental_kings_grid": "Incremental King's grid",
            "vgraph": "Visibility graph",
            "dynamic_vgraph": "Dynamic visibility graph",
            "scenario": "Scenario",
        }.get(algorithm, algorithm or "Scenario")
        print(f"[{display_name} {iteration}] Plotting {data_type}", flush=True)

        title = None
        if title_info:
            title = (
                f"scenario {title_info['scenario_version']:03d} | "
                f"{title_info['algorithm'] or 'scenario'} | "
                f"{data_type}"
            )

        plotter.plot(data=data, scenario=self.scenario, path=path, title=title)
