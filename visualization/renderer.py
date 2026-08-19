import matplotlib.pyplot as plt

class WarehouseRenderer:

    def render(self, state, layers):
        fig, ax = plt.subplots()

        for layer in layers:
            layer.draw(state, ax)

        return fig

class SimpleObstacleLayer:

    def draw(self, state, ax):
        for obstacle in state.simple_obstacles:
            x, y = obstacle.position
            ax.add_patch(plt.Rectangle((x, y), 1, 1, color='gray', alpha=0.5))

class UnitLoadLayer:

    def draw(self, state, ax):
        for unit_load in state.unit_loads:
            x, y = unit_load.position
            ax.add_patch(plt.Rectangle((x, y), 1, 1, color='blue', alpha=0.5))