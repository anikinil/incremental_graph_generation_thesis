from pathlib import Path
import sys

import numpy as np

from algorithms.geometry import continuous_special_points
from scenario.scenario_generation import Scenario
from artifacts.record import ExperimentRecorder

from algorithms.gvd import GVD
from algorithms.prm import PRM
from algorithms.vgraph import VGraph

if __name__ == "__main__":

    seed = 41
    rng = np.random.default_rng(seed)
    
    height = 30
    width = 30
    
    agv_radius = 0.1
    occupancy_rate = 0.4

    io_points = {
        (height * 0.15, 0),
        (height - 1.5, width * 0.75),
    }

    recorder = ExperimentRecorder(seed=seed)

    recorder.set_algorithm("scenario", use_iterations=False, plot=True)

    scenario = Scenario(
        rng=rng,
        height=height,
        width=width,
        occupancy_rate=occupancy_rate,
        sigma_coarse=3,
        sigma_fine=1,
        noisy_aisles_num=0,
        noisy_aisle_sigma=1,
        io_points=io_points,
        io_points_radius=3,
        simple_obstacle_rate=0.1,
        agv_radius=agv_radius,
        recorder=recorder,
    )
    scenario_success = scenario.generate()
    print()

    continuous_access_points, continuous_io_points = continuous_special_points(scenario.access_points, scenario.io_points, scenario.agv_radius)
    bounds = (0, 0, width, height)

    eps = 0.05

    recorder.set_algorithm("prm", use_iterations=False, plot=True)
    num_buffer_segments = 2  # PRM may use a different buffered-corner approximation
    prm = PRM(
        rng=rng,
        occupancy=scenario.occupancy,
        access_points=continuous_access_points,
        io_points=continuous_io_points,
        height=height,
        width=width,
        eps=eps,
        max_dist=5,
        max_neighbors=10,
        agv_radius=scenario.agv_radius,
        num_buffer_segments=num_buffer_segments,
        recorder=recorder,
    )
    prm.construction_step(min_num_vertices=1000, avoid_cycles=False)
    # prm.connect_special_points()
    prm.record_graph()
    print()

    scale_factor = 3
    recorder.set_algorithm("gvd", use_iterations=True, plot=True)
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
        gvd.apply_layout_updates(obstacle_updates=scenario.consume_updates(), new_access_points=scenario.access_points, new_io_points=scenario.io_points)
        gvd.updateDistanceMap()
        gvd.build_graph()
        
    scenario.remove_region(top_left=(10, 10), bottom_right=(20, 20))
    gvd.apply_layout_updates(obstacle_updates=scenario.consume_updates(), new_access_points=scenario.access_points, new_io_points=scenario.io_points)
    gvd.updateDistanceMap()
    gvd.build_graph()
    
    scenario.add_region(top_left=(12, 12), bottom_right=(15, 15))
    gvd.apply_layout_updates(obstacle_updates=scenario.consume_updates(), new_access_points=scenario.access_points, new_io_points=scenario.io_points)
    gvd.updateDistanceMap()
    gvd.build_graph()

        
