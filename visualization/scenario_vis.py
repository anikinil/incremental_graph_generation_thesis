def _setup_visualization(self, ax, show_grid=True):
    ax.set_xlim(0, self.width)
    ax.set_ylim(self.height, 0)
    ax.set_xticks(np.arange(self.width + 1))
    ax.set_yticks(np.arange(self.height + 1))

    if show_grid:
        ax.grid(which="major", color="black", linewidth=1, alpha=0.5)

    ax.set_aspect("equal", adjustable="box")
    ax.margins(0)

    
def _draw_special_points(self, ax):
    """Draw access points."""

    IO_POINT_COLOR = (1, 0.4, 0.15)
    IO_POINT_EDGE_COLOR = (0, 0, 0)
    
    # x = []
    # y = []
    # u = []
    # v = []
    # for r, c, direction in self.access_points:
    #     if direction == "n":
    #         x.append(c)
    #         y.append(r + 0.3)
    #         u.append(0)
    #         v.append(-0.5)
    #     elif direction == "s":
    #         x.append(c)
    #         y.append(r - 0.3)
    #         u.append(0)
    #         v.append(0.5)
    #     elif direction == "e":
    #         x.append(c - 0.3)
    #         y.append(r)
    #         u.append(0.5)
    #         v.append(0)
    #     elif direction == "w":
    #         x.append(c + 0.3)
    #         y.append(r)
    #         u.append(-0.5)
    #         v.append(0)
    # ax.quiver(x, y, u, v, angles="xy", scale_units="xy", scale=1, color="red", width=0.008, zorder=3)

    if self.io_points:
        rows, cols = zip(*self.io_points)
        ax.scatter(np.asarray(cols) + 0.5, np.asarray(rows) + 0.5, s=100, color=IO_POINT_COLOR, edgecolors=IO_POINT_EDGE_COLOR, linewidths=1.5, zorder=3)
    

def visualize(self, title=""):
    """
    Display and save the probability map and the probability map
    with obstacles side by side.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.collections import PatchCollection
    
    fig = plt.figure(figsize=(17, 4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 0.05, 1], height_ratios=[1], wspace=0.1, hspace=0.1)
    
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 2])]

    # probability field
    ax = axes[0]
    image = ax.imshow(self.probability_field, cmap="viridis", origin="upper", interpolation="nearest", extent=(0, self.width, self.height, 0))
    self._setup_visualization(ax)

    # probability field + obstacles
    ax = axes[1]
    # ax.imshow(self.probability_field, cmap="viridis", origin="upper", interpolation="nearest", extent=(0, self.width, self.height, 0), zorder=0)

    obstacle_cells = np.argwhere(self.occupancy > 0)

    CELL_MARGIN = 0.15
    CELL_SIZE = 1.0 - 2.0 * CELL_MARGIN
    
    SIMPLE_OBSTACLE_COLOR = [0.3, 0.3, 0.3, 1] # dark grey
    UNIT_LOAD_COLOR = [0.5, 0.5, 0.5, 1] # grey

    patches = []
    colors = []

    for r, c in obstacle_cells:
        patches.append(Rectangle((c + CELL_MARGIN, r + CELL_MARGIN), CELL_SIZE, CELL_SIZE))

        if self.occupancy[r, c] == 1:
            colors.append(SIMPLE_OBSTACLE_COLOR)
        else:
            colors.append(UNIT_LOAD_COLOR)

    collection = PatchCollection(patches, facecolor=colors, edgecolor="black", linewidth=1, zorder=2)
    
    # remove tick marks and labels
    ax.tick_params(axis='both', which='both', length=0) 
    ax.set_xticklabels([])
    ax.set_yticklabels([])
            
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    self._setup_visualization(ax, show_grid=True)
    ax.add_collection(collection)
    self._draw_special_points(ax)

    # colorbar on the right, spanning the full height
    cax = fig.add_subplot(gs[0, 1])
    fig.colorbar(image, cax=cax, label="Probability")
    
    # title
    # fig.suptitle(f"Iteration {self.iteration} - {title}", fontsize=16)

    filename = f"probability/{title}_{self.iteration}.pdf"
    print(filename)
    fig.savefig(filename, format="pdf")
    plt.show()
    plt.close(fig)

def save_scenario(self, filename):
    
    np.savez(
        filename,
        seed=self.seed,
        width=self.width,
        height=self.height,
        occupancy=np.array(self.occupancy, dtype=np.uint8),
        occupancy_rate=self.occupancy_rate,
        io_points=np.array(list(self.io_points)),
        io_points_radius=self.IO_points_radius,
        access_points=np.array(list(self.access_points)),
        agv_radius=self.agv_radius,
        updates=np.array(self.updates, dtype=object) if self.updates else np.array([], dtype=object),
    )
    
    self.updates.clear()