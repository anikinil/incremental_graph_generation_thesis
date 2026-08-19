

def visualize(self, title = None):
    _, ax = plt.subplots(figsize=(25, 20))

    width = self.bounds[2] - self.bounds[0]
    height = self.bounds[3] - self.bounds[1]

    BUFFER_COLOR = [0.8, 0.8, 0.8, 1] # light grey
    FREE_SPACE_COLOR = [0, 0, 0, 0]  # transparent
    SIMPLE_OBSTACLE_COLOR = [0.3, 0.3, 0.3, 1] # dark grey
    UNIT_LOAD_COLOR = [0.5, 0.5, 0.5, 1] # grey

    # draw obstacle/free space first (light grey)
    gpd.GeoSeries(self.obstacle_space).plot(ax=ax, color=BUFFER_COLOR, zorder=0)
    gpd.GeoSeries(self.free_space).plot(ax=ax, color=FREE_SPACE_COLOR[0:4], zorder=0)
    
    # draw wall buffer
    ax.add_patch(plt.Rectangle((0, 0), width, self.agv_radius, color=BUFFER_COLOR, zorder=0))
    ax.add_patch(plt.Rectangle((0, height - self.agv_radius), width, self.agv_radius, color=BUFFER_COLOR, zorder=0))
    ax.add_patch(plt.Rectangle((0, 0), self.agv_radius, height, color=BUFFER_COLOR, zorder=0))
    ax.add_patch(plt.Rectangle((width - self.agv_radius, 0), self.agv_radius, height, color=BUFFER_COLOR, zorder=0))

    # access points
    access_x = [p[0] for p in self.access_points]
    access_y = [p[1] for p in self.access_points]
    ax.scatter(access_x, access_y, s=30, color="red", zorder=3, label="Access points", rasterized=True)

    # IO points
    io_x = [p[0] for p in self.io_points]
    io_y = [p[1] for p in self.io_points]
    ax.scatter(io_x, io_y, s=50, color="green", zorder=3, label="IO points", rasterized=True)
    ax.legend(loc='upper right')

    G = self.build_graph()

    # draw occupancy grid on top
    # [free space (transparent), simple obstacle (dark grey), unit load (grey)]
    cmap = ListedColormap([FREE_SPACE_COLOR, SIMPLE_OBSTACLE_COLOR, UNIT_LOAD_COLOR])
    ax.pcolormesh(self.occupancy, cmap=cmap, edgecolors="black", linewidth=0.5, shading="flat", zorder=1, rasterized=True)
    
    # draw edges (color by component)
    segments = [[G.nodes[u]["pos"], G.nodes[v]["pos"]] for u, v in G.edges]
    ax.add_collection(LineCollection(segments, colors=["gray"], alpha=0.5, linewidths=2, zorder=2, rasterized=True))
    
    # for u, v in G.edges:
    #     x1, y1 = G.nodes[u]["pos"]
    #     x2, y2 = G.nodes[v]["pos"]
    #     ax.plot([x1, x2], [y1, y2], "b-", linewidth=1, zorder=2)
    
    # draw nodes
    components = list(nx.connected_components(G))
    colors = plt.cm.tab20(np.linspace(0, 1, len(components)))
    for comp_idx, component in enumerate(components):
        x_coords = [self.nodes[n][0] for n in component]
        y_coords = [self.nodes[n][1] for n in component]
        ax.scatter(x_coords, y_coords, s=15, color=[colors[comp_idx]], zorder=3, edgecolors=[colors[comp_idx]], linewidth=2, rasterized=True)
    
    # # draw nodes, mark top 100 nodes with highest failure ratio
    # failure_ratios = np.zeros(self.num_nodes, dtype=np.float64)
    # for i in range(self.num_nodes):
    #     if self.attempt_count[i] > 0:
    #         failure_ratios[i] = self.failure_count[i] / (self.attempt_count[i] + 1)
    #     else:
    #         failure_ratios[i] = 0.0
    # top_indices = np.argsort(failure_ratios)[-len(failure_ratios)//4:]  # top 25% nodes with highest failure ratio
    # x_coords = [self.nodes[i][0] for i in top_indices]
    # y_coords = [self.nodes[i][1] for i in top_indices]
    # ax.scatter(x_coords, y_coords, s=20, color="yellow", zorder=4, linewidth=1.0, rasterized=True, label="Top 100 failure ratio nodes")
    
    # # mark top 5% with orange
    # top_5_percent = max(1, len(failure_ratios) // 20)
    # top_5_indices = np.argsort(failure_ratios)[-top_5_percent:]
    # x_coords = [self.nodes[i][0] for i in top_5_indices]
    # y_coords = [self.nodes[i][1] for i in top_5_indices]
    # ax.scatter(x_coords, y_coords, s=30, color="orange", zorder=5, linewidth=1.0, rasterized=True, label="Top 5% failure ratio nodes")
    
    # mark last N added nodes
    if self.num_nodes > 0:
        last_nodes = [self.nodes[i] for i in range(max(0, self.num_nodes - 50), self.num_nodes)]
        x_coords = [node[0] for node in last_nodes]
        y_coords = [node[1] for node in last_nodes]
        for i, (x, y) in enumerate(zip(x_coords, y_coords)):
            ax.scatter(x, y, s=10, color="black", zorder=4, linewidth=1.0, rasterized=True)

    # for x in range(width + 1):
    #     ax.axvline(x, color="black", linewidth=1, zorder=4)
    # for y in range(height + 1):
    #     ax.axhline(y, color="black", linewidth=1, zorder=4)
    
    # grid
    ax.set_xticks(np.arange(width + 1))
    ax.set_yticks(np.arange(height + 1))
    ax.grid(True, color="black", linewidth=1, zorder=4)

    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.margins(0)
    ax.invert_yaxis()

    plt.title(title if title else "PRM Roadmap")
    plt.show()

