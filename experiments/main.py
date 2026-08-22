from pathlib import Path
import sys

import numpy as np

from scenario.geometry import continuous_special_points
from scenario.scenario_generation import Scenario
from artifacts.record import ExperimentRecorder

from algorithms.gvd import GVD
from algorithms.prm import PRM
from algorithms.vgraph import VGraph

def main():

    seed = 42
    rng = np.random.default_rng(seed)

    height = 50
    width = 80

    io_points = {
        (int(height * 0.1), 0),
        (height - 1, int(width * 0.7)),
    }

    recorder = ExperimentRecorder(seed=seed)

    recorder.set_algorithm("scenario", use_iterations=False)
    scenario = Scenario(
        rng=rng,
        height=height,
        width=width,
        occupancy_rate=0.1,
        sigma_coarse=3,
        sigma_fine=3,
        noisy_aisles_num=0,
        noisy_aisle_sigma=1,
        io_points=io_points,
        io_points_radius=3,
        simple_obstacle_rate=0.1,
        agv_radius=0.42,
        recorder=recorder,
    )
    scenario.generate()

    continuous_access_points, continuous_io_points = continuous_special_points(scenario.access_points, scenario.io_points, scenario.agv_radius)
    bounds = (0, 0, width, height)

    recorder.set_algorithm("vgraph", use_iterations=False)
    buffer_segments = 3  # Shapely segments per 90-degree buffered corner
    free_space, obstacle_space = scenario.navigation_space(buffer_segments=buffer_segments)
    vgraph = VGraph(
        width=width,
        height=height,
        free_space=free_space,
        obstacle_space=obstacle_space,
        occupancy=scenario.occupancy,
        agv_radius=scenario.agv_radius,
        access_points=continuous_access_points,
        io_points=continuous_io_points,
        buffer_segments=buffer_segments,
        recorder=recorder,
    )
    vgraph.create_visibility_graph()
    vgraph.record_graph()

    recorder.set_algorithm("prm", use_iterations=False)
    buffer_segments = 2  # PRM may use a different buffered-corner approximation
    free_space, obstacle_space = scenario.navigation_space(buffer_segments=buffer_segments)
    prm = PRM(
        rng=np.random.default_rng(seed + 1),
        free_space=free_space,
        obstacle_space=obstacle_space,
        occupancy=scenario.occupancy,
        access_points=continuous_access_points,
        io_points=continuous_io_points,
        bounds=bounds,
        eps=0.1,
        max_dist=5,
        max_neighbors=15,
        agv_radius=scenario.agv_radius,
        buffer_segments=buffer_segments,
        recorder=recorder,
    )
    prm.construction_step(num_nodes=500)
    prm.connect_special_points()
    prm.record_graph()

    scale_factor = 3
    recorder.set_algorithm("gvd", use_iterations=True)
    gvd = GVD(
        rng=rng,
        occupancy_grid=scenario.occupancy,
        scale_factor=scale_factor,
        access_points=scenario.access_points,
        io_points=scenario.io_points,
        agv_radius=scenario.agv_radius,
        recorder=recorder
    )
    gvd.computeDistanceMap()
    gvd.build_graph()

    gvd_iterations = 3
    updates_per_iteration = 10
    for _ in range(gvd_iterations):
        scenario.make_updates(number_of_updates=updates_per_iteration)
        gvd.apply_obstacle_updates(updates=scenario.consume_updates(), new_access_points=scenario.access_points, new_io_points=scenario.io_points)
        gvd.updateDistanceMap()
        gvd.build_graph()
        
    scenario.remove_region(top_left=(10, 10), bottom_right=(20, 20))
    gvd.apply_obstacle_updates(updates=scenario.consume_updates(), new_access_points=scenario.access_points, new_io_points=scenario.io_points)
    gvd.updateDistanceMap()
    gvd.build_graph()

if __name__ == "__main__":
    main()
        
