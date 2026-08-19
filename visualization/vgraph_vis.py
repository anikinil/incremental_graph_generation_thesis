
def visualize(self, title = None):
    _, ax = plt.subplots(figsize=(25, 20))

    LIGHT_GREY = (0.8, 0.8, 0.8)

    # draw obstacle/free space first (light grey)
    gpd.GeoSeries(self.obstacle_space).plot(ax=ax, color=LIGHT_GREY, zorder=0)
    gpd.GeoSeries(self.free_space).plot(ax=ax, color="white", zorder=0)
    
    # draw wall buffer
    ax.add_patch(plt.Rectangle((0, 0), self.width, self.agv_radius, color=LIGHT_GREY, zorder=0))
    ax.add_patch(plt.Rectangle((0, self.height - self.agv_radius), self.width, self.agv_radius, color=LIGHT_GREY, zorder=0))
    ax.add_patch(plt.Rectangle((0, 0), self.agv_radius, self.height, color=LIGHT_GREY, zorder=0))
    ax.add_patch(plt.Rectangle((self.width - self.agv_radius, 0), self.agv_radius, self.height, color=LIGHT_GREY, zorder=0))

    # draw occupancy grid on top
    # [free space (transparent), simple obstacle (dark grey), unit load (grey)]
    cmap = ListedColormap([(0, 0, 0, 0), (0.3, 0.3, 0.3, 1), (0.5, 0.5, 0.5, 1)])
    ax.pcolormesh(self.occupancy, cmap=cmap, edgecolors="black", linewidth=0.5, shading="flat", zorder=1)
    
    # draw visibility graph edges
    for u, v in self.G.edges:
        x1, y1 = self.G.nodes[u]["pos"]
        x2, y2 = self.G.nodes[v]["pos"]
        ax.plot([x1, x2], [y1, y2], "b-", linewidth=1, zorder=2)

    # access points
    access_x = [p[0] for p in self.access_points]
    access_y = [p[1] for p in self.access_points]
    ax.scatter(access_x, access_y, s=30, color="red", zorder=3, label="Access points", rasterized=True)        
    
    # IO points
    io_x = [p[0] for p in self.io_points]
    io_y = [p[1] for p in self.io_points]
    ax.scatter(io_x, io_y, s=50, color="green", zorder=3, label="IO points", rasterized=True)
    ax.legend(loc='upper right')
    
    # grid boundaries
    for x in range(self.width + 1):
        ax.axvline(x, color="black", linewidth=1, zorder=4)
    for y in range(self.height + 1):
        ax.axhline(y, color="black", linewidth=1, zorder=4)

    ax.set_xlim(0, self.width)
    ax.set_ylim(0, self.height)
    ax.margins(0)
    ax.invert_yaxis()

    plt.title(title if title else "Visibility Graph")
    plt.show()

