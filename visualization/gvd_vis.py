def draw_affected_outline(self, ax):
        """Draw an outline exactly along the boundaries of affected cells."""

        affected = np.zeros((self.height, self.width), dtype=bool)
        LINE_WIDTH = 2
        COLOR = (0.1, 0.1, 0.1, 1)

        for row, col in self.affected_distance_cells:
            if 0 <= row < self.height and 0 <= col < self.width:
                affected[row, col] = True

        for row, col in zip(*np.where(affected)):
            # left boundary
            if col == 0 or not affected[row, col - 1]:
                ax.plot([col, col], [row, row + 1], color=COLOR, linewidth=LINE_WIDTH, zorder=50)

            # right boundary
            if col == self.width - 1 or not affected[row, col + 1]:
                ax.plot([col + 1, col + 1], [row, row + 1], color=COLOR, linewidth=LINE_WIDTH, zorder=50)

            # bottom boundary
            if row == 0 or not affected[row - 1, col]:
                ax.plot([col, col + 1], [row, row], color=COLOR, linewidth=LINE_WIDTH, zorder=50)

            # top boundary
            if row == self.height - 1 or not affected[row + 1, col]:
                ax.plot([col, col + 1], [row + 1, row + 1], color=COLOR, linewidth=LINE_WIDTH, zorder=50)

def draw_obstacles(self):

    fig, ax = plt.subplots()
    vis_map = np.zeros((self.height, self.width, 4), dtype=np.float32)

    # buffer zone
    from shapely.geometry import box
    from shapely.ops import unary_union
    occupied = self.occupancy_grid > 0
    ys, xs = np.where(occupied)
    obstacle_polygons = [box(x, y, x + 1, y + 1) for y, x in zip(ys, xs)]
    obstacles = unary_union(obstacle_polygons)
    buffer_zone = obstacles.buffer(self.agv_radius, resolution=BUFFER_RES, join_style=shapely.BufferJoinStyle.round)
    buffer_zone = buffer_zone.intersection(box(0, 0, self.width, self.height))
    polygons = (list(buffer_zone.geoms) if hasattr(buffer_zone, "geoms") else [buffer_zone])
    for poly in polygons:
        vertices = []
        codes = []
        exterior = np.asarray(poly.exterior.coords)
        vertices.extend(exterior)
        codes.extend([Path.MOVETO] + [Path.LINETO] * (len(exterior) - 2) + [Path.CLOSEPOLY])
        for hole in poly.interiors:
            hole = np.asarray(hole.coords)
            vertices.extend(hole)
            codes.extend([Path.MOVETO] + [Path.LINETO] * (len(hole) - 2) + [Path.CLOSEPOLY])
        ax.add_patch(PathPatch( Path(vertices, codes), facecolor=BUFFER_COLOR[:3], alpha=BUFFER_COLOR[3], edgecolor="none", zorder=0))
    
    # occupancy
    vis_map[self.occupancy_grid == 1] = SIMPLE_OBSTACLE_COLOR
    vis_map[self.occupancy_grid == 2] = UNIT_LOAD_COLOR

    ax.imshow(vis_map, origin="lower", extent=(0, self.width, 0, self.height), interpolation="none", zorder=2)

    return fig, ax
    

def visualize_distance_map(self, title=None):
    """Draws obstacles and D"""
    
    fig, ax = self.draw_obstacles()
    vis_map = np.zeros((self.height, self.width, 4), dtype=np.float32)

    # distance map
    
    # D_normalized = self.D / np.max(self.D) if np.max(self.D) > 0 else self.D
    # D_colormap = plt.cm.plasma(D_normalized)
    # vis_map[self.D < np.inf] = D_colormap[self.D < np.inf]
    
    finite = np.isfinite(self.D)

    D_colormap = np.zeros((*self.D.shape, 4))

    if np.any(finite):
        max_D = np.max(self.D[finite])
        D_normalized = np.zeros_like(self.D, dtype=float)
        D_normalized[finite] = self.D[finite] / max_D if max_D > 0 else 0

        D_colormap[finite] = plt.cm.plasma(D_normalized[finite])

    ax.imshow(D_colormap, origin="lower", extent=(0, self.width, 0, self.height), interpolation="none", alpha=0.8, zorder=1)

    ax.imshow(vis_map, origin="lower", extent=(0, self.width, 0, self.height), interpolation="none", alpha=0.8, zorder=1)
    
    # affected cells outline
    if self.iteration > 0:
        self.draw_affected_outline(ax)
    
    # distance map legend
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    # norm = Normalize(vmin=0, vmax=np.max(self.D) if np.max(self.D) > 0 else 1)
    # norm = Normalize(vmin=0, vmax=146 if np.max(self.D) > 0 else 1)
    # sm = ScalarMappable(cmap=plt.cm.plasma, norm=norm)
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="3%", pad=0.1)
    # cbar = fig.colorbar(sm, cax=cax)
    # cbar.set_label('Distance to Nearest Obstacle', rotation=270, labelpad=15)
    
    ax.set_xticks(np.arange(1, self.width, self.scale_factor))
    ax.set_yticks(np.arange(1, self.height, self.scale_factor))
    
    # remove tick marks and labels
    ax.tick_params(axis='both', which='both', length=0) 
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    ax.set_axisbelow(False)
    ax.grid(True, color='black', linestyle='-', linewidth=0.5, zorder=40)
    ax.set_xlim(0, self.width)
    ax.set_ylim(self.height, 0)
    ax.set_aspect('equal')
    ax.margins(0)
    
    # if title:
    #     ax.set_title(title, fontsize=20, pad=5)
            
    fig.savefig(f"distance_map_{self.iteration}_{title}.pdf", format="pdf", bbox_inches="tight")
    
    # filename = f"gvd_steps/gvd_{title}.pdf"
    # fig.savefig(filename, format="pdf", bbox_inches="tight")
    
    plt.show()

def visualize_voro(self, title=None):
    """Draw obstacles and voro"""

    fig, ax = self.draw_obstacles()
    vis_map = np.zeros((self.height, self.width, 4), dtype=np.float32)

    # voro map
    vis_map[self.voro] = VORO_SKELETON_COLOR
    
    # affected cells outline
    if self.iteration > 0:
        # self.draw_affected_outline(ax)
        # affected distance cells
        if self.iteration > 0:
            changed_distance_mask = np.zeros((self.height, self.width), dtype=bool)
            if self.affected_distance_cells:
                cells = np.asarray(list(self.affected_distance_cells), dtype=np.intp)
                if cells.size > 0:
                    cells = cells[(cells[:, 0] >= 0) & (cells[:, 0] < self.height) & (cells[:, 1] >= 0) & (cells[:, 1] < self.width)]
                    changed_distance_mask[cells[:, 0], cells[:, 1]] = True
            vis_map[changed_distance_mask] = AFFECTED_DISTANCES_COLOR
        # affected Voronoi cells
        if self.iteration > 0:
            affected_voro_mask = np.zeros((self.height, self.width), dtype=bool)
            if self.affected_voro_cells:
                cells = np.asarray(list(self.affected_voro_cells), dtype=np.intp)
                if cells.size > 0:
                    cells = cells[(cells[:, 0] >= 0) & (cells[:, 0] < self.height) & (cells[:, 1] >= 0) & (cells[:, 1] < self.width)]
                    affected_voro_mask[cells[:, 0], cells[:, 1]] = True
            vis_map[affected_voro_mask] = AFFECTED_VORO_CELLS_COLOR
    
    ax.imshow(vis_map, origin="lower", extent=(0, self.width, 0, self.height), interpolation="none", zorder=1)
    ax.set_xticks(np.arange(1, self.width, self.scale_factor))
    ax.set_yticks(np.arange(1, self.height, self.scale_factor))
    
    # remove tick marks and labels
    ax.tick_params(axis='both', which='both', length=0) 
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    ax.set_axisbelow(False)
    ax.grid(True, color='black', linestyle='-', linewidth=0.5, zorder=40)
    ax.set_xlim(0, self.width)
    ax.set_ylim(self.height, 0)
    ax.set_aspect('equal')
    ax.margins(0)
    
    # if title:
    #     ax.set_title(title, fontsize=20, pad=5)
            
    fig.savefig(f"voro_map_{self.iteration}_{title}.pdf", format="pdf", bbox_inches="tight")
    
    plt.show()

def visualize_distance_map_and_voro(self, title=None):
    """Draw obstacles, D, and voro as two subplots in one figure."""
    
    fig = plt.figure(figsize=(7, 10))

    gs = fig.add_gridspec(2, 2, width_ratios=[1, 0.05], height_ratios=[1, 1], wspace=0.1, hspace=0.1)

    ax1 = fig.add_subplot(gs[0, 0])  # distance map
    cax = fig.add_subplot(gs[0, 1])  # colorbar
    ax2 = fig.add_subplot(gs[1, 0])  # voro map

    # obstacles

    vis_map = np.zeros((self.height, self.width, 4), dtype=np.float32)

    vis_map[self.occupancy_grid == 1] = SIMPLE_OBSTACLE_COLOR
    vis_map[self.occupancy_grid == 2] = UNIT_LOAD_COLOR

    ax1.imshow(vis_map, origin="lower", extent=(0, self.width, 0, self.height), interpolation="none", zorder=2)
    ax2.imshow(vis_map, origin="lower", extent=(0, self.width, 0, self.height), interpolation="none", zorder=2)
    
    # distance map

    vis_map = np.zeros((self.height, self.width, 4), dtype=np.float32)

    # distance map
    D_normalized = self.D / np.max(self.D) if np.max(self.D) > 0 else self.D
    D_colormap = plt.cm.plasma(D_normalized)
    vis_map[self.D < np.inf] = D_colormap[self.D < np.inf]

    ax1.imshow(vis_map, origin="lower", extent=(0, self.width, 0, self.height), interpolation="none", alpha=0.8, zorder=1)
    
    # affected cells outline
    if self.iteration > 0:
        self.draw_affected_outline(ax1)
    
    # distance map legend
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    norm = Normalize(vmin=0, vmax=np.max(self.D) if np.max(self.D) > 0 else 1)

    sm = ScalarMappable(cmap=plt.cm.plasma, norm=norm)

    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Distance to Nearest Obstacle")
    
    ax1.set_xticks(np.arange(1, self.width, self.scale_factor))
    ax1.set_yticks(np.arange(1, self.height, self.scale_factor))
    
    # remove tick marks and labels
    ax1.tick_params(axis='both', which='both', length=0) 
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    
    ax1.set_axisbelow(False)
    ax1.grid(True, color='black', linestyle='-', linewidth=0.5, zorder=40)
    ax1.set_xlim(0, self.width)
    ax1.set_ylim(self.height, 0)
    ax1.set_aspect('equal')
    ax1.margins(0)
    
    # voro
    
    vis_map = np.zeros((self.height, self.width, 4), dtype=np.float32)

    # voro map
    vis_map[self.voro] = VORO_SKELETON_COLOR
    
    # affected cells outline
    if self.iteration > 0:
        # self.draw_affected_outline(ax)
        # affected distance cells
        if self.iteration > 0:
            changed_distance_mask = np.zeros((self.height, self.width), dtype=bool)
            if self.affected_distance_cells:
                cells = np.asarray(list(self.affected_distance_cells), dtype=np.intp)
                if cells.size > 0:
                    cells = cells[(cells[:, 0] >= 0) & (cells[:, 0] < self.height) & (cells[:, 1] >= 0) & (cells[:, 1] < self.width)]
                    changed_distance_mask[cells[:, 0], cells[:, 1]] = True
            vis_map[changed_distance_mask] = AFFECTED_DISTANCES_COLOR
        # affected Voronoi cells
        if self.iteration > 0:
            affected_voro_mask = np.zeros((self.height, self.width), dtype=bool)
            if self.affected_voro_cells:
                cells = np.asarray(list(self.affected_voro_cells), dtype=np.intp)
                if cells.size > 0:
                    cells = cells[(cells[:, 0] >= 0) & (cells[:, 0] < self.height) & (cells[:, 1] >= 0) & (cells[:, 1] < self.width)]
                    affected_voro_mask[cells[:, 0], cells[:, 1]] = True
            vis_map[affected_voro_mask] = AFFECTED_VORO_CELLS_COLOR
    
    ax2.imshow(vis_map, origin="lower", extent=(0, self.width, 0, self.height), interpolation="none", zorder=1)
    ax2.set_xticks(np.arange(1, self.width, self.scale_factor))
    ax2.set_yticks(np.arange(1, self.height, self.scale_factor))
    
    # remove tick marks and labels
    ax2.tick_params(axis='both', which='both', length=0) 
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    
    ax2.set_axisbelow(False)
    ax2.grid(True, color='black', linestyle='-', linewidth=0.5, zorder=40)
    ax2.set_xlim(0, self.width)
    ax2.set_ylim(self.height, 0)
    ax2.set_aspect('equal')
    ax2.margins(0)
    
    fig.canvas.draw()

    ax1.annotate("", xy=(0.5, 0.48), xytext=(0.5, 0.52), xycoords=fig.transFigure, arrowprops=dict(arrowstyle="->", linewidth=2))
    
    plt.savefig(f"distance_and_voro_map_{self.iteration}_{title}.pdf", format="pdf", bbox_inches="tight")
    
    plt.show()

def draw_grid(self, title=None):
    from matplotlib.patches import Patch
    
    fig, ax = self.draw_obstacles()
            
    vis_map = np.zeros((self.height, self.width, 4), dtype=np.float32)

    # buffer zone
    from shapely.geometry import box
    from shapely.ops import unary_union
    occupied = self.occupancy_grid > 0
    ys, xs = np.where(occupied)
    obstacle_polygons = [box(x, y, x + 1, y + 1) for y, x in zip(ys, xs)]
    obstacles = unary_union(obstacle_polygons)
    buffer_zone = obstacles.buffer(self.agv_radius, resolution=BUFFER_RES, join_style=shapely.BufferJoinStyle.round)
    buffer_zone = buffer_zone.intersection(box(0, 0, self.width, self.height))
    polygons = (list(buffer_zone.geoms) if hasattr(buffer_zone, "geoms") else [buffer_zone])
    for poly in polygons:
        vertices = []
        codes = []
        exterior = np.asarray(poly.exterior.coords)
        vertices.extend(exterior)
        codes.extend([Path.MOVETO] + [Path.LINETO] * (len(exterior) - 2) + [Path.CLOSEPOLY])
        for hole in poly.interiors:
            hole = np.asarray(hole.coords)
            vertices.extend(hole)
            codes.extend([Path.MOVETO] + [Path.LINETO] * (len(hole) - 2) + [Path.CLOSEPOLY])
        ax.add_patch(PathPatch( Path(vertices, codes), facecolor=BUFFER_COLOR[:3], alpha=BUFFER_COLOR[3], edgecolor="none", zorder=0))
    
    # occupancy
    vis_map[self.occupancy_grid == 1] = SIMPLE_OBSTACLE_COLOR
    vis_map[self.occupancy_grid == 2] = UNIT_LOAD_COLOR

    # affected distance cells
    if self.iteration > 0:
        changed_distance_mask = np.zeros((self.height, self.width), dtype=bool)
        if self.affected_distance_cells:
            cells = np.asarray(list(self.affected_distance_cells), dtype=np.intp)
            if cells.size > 0:
                cells = cells[(cells[:, 0] >= 0) & (cells[:, 0] < self.height) & (cells[:, 1] >= 0) & (cells[:, 1] < self.width)]
                changed_distance_mask[cells[:, 0], cells[:, 1]] = True
        vis_map[changed_distance_mask] = AFFECTED_DISTANCES_COLOR

    # Voronoi skeleton
    vis_map[self.voro] = VORO_SKELETON_COLOR

    # affected Voronoi cells
    if self.iteration > 0:
        affected_voro_mask = np.zeros((self.height, self.width), dtype=bool)
        if self.affected_voro_cells:
            cells = np.asarray(list(self.affected_voro_cells), dtype=np.intp)
            if cells.size > 0:
                cells = cells[(cells[:, 0] >= 0) & (cells[:, 0] < self.height) & (cells[:, 1] >= 0) & (cells[:, 1] < self.width)]
                affected_voro_mask[cells[:, 0], cells[:, 1]] = True
        vis_map[affected_voro_mask] = AFFECTED_VORO_CELLS_COLOR

    # bridges are the final cell overlay.
    vis_map[self.special_point_bridges] = BRIDGE_COLOR
    ax.imshow(vis_map, origin="lower", extent=(0, self.width, 0, self.height), interpolation="none", zorder=1)
    
    # grid and axes
    ax.set_xticks(np.arange(1, self.width, self.scale_factor))
    ax.set_yticks(np.arange(1, self.height, self.scale_factor))
    ax.set_axisbelow(False)
    ax.grid(True, color='black', linestyle='-', linewidth=0.5, zorder=40)
    ax.set_xlim(0, self.width)
    ax.set_ylim(self.height, 0)
    ax.set_aspect('equal')
    ax.margins(0)

    # special points
    if self.access_points:
        access = np.asarray([(p[1], p[0]) for p in self.access_points], dtype=float)
        ax.scatter(access[:, 0], access[:, 1], s=30, color="red", zorder=20, label="Access points")
    if self.io_points:
        io = np.asarray([(p[1], p[0]) for p in self.io_points], dtype=float)
        ax.scatter(io[:, 0], io[:, 1], s=50, color="green", zorder=20, label="IO points")

    # legend
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor=AFFECTED_DISTANCES_COLOR, edgecolor="none"))
    handles.append(Patch(facecolor=AFFECTED_VORO_CELLS_COLOR, edgecolor="none"))
    handles.append(Patch(facecolor=BRIDGE_COLOR, edgecolor="none"))
    handles.append(Patch(facecolor=VORO_SKELETON_COLOR, edgecolor="none"))
    labels.append("Changed distance cells")
    labels.append("Affected Voronoi cells")
    labels.append("Bridges")
    labels.append("Voronoi skeleton")
    ax.legend(handles, labels, loc="upper right")
    
    ax.set_title(title)

    return fig, ax

def visualize_gvd(self, title=None):
    fig, ax = self.draw_grid(title=title or "Incremental GVD")
    
    plt.show()

def visualize_graph(self, title=None):
    
    fig, ax = self.draw_grid(title=title or "Graph from Voronoi Diagram")

    edge_segments = []
    for vertex1, vertex2 in self.graph["edges"]:
        y1, x1 = self.graph["vertices"][vertex1]
        y2, x2 = self.graph["vertices"][vertex2]
        edge_segments.append([(x1, y1), (x2, y2)])

    ax.add_collection(LineCollection(edge_segments, colors=EDGE_COLOR, linewidths=3, zorder=10, label="Graph edges"))

    vertices = np.asarray([(x, y) for y, x in self.graph["vertices"].values()], dtype=float)
    ax.scatter(vertices[:, 0], vertices[:, 1], color=VERTEX_COLOR, linewidth=1, s=50, zorder=11, label="Graph vertices")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper right")

    # fig.tight_layout()
    plt.show()