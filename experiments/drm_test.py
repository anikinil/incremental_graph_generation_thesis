import networkx as nx

import numpy as np

from artifacts.record import ExperimentRecorder
from algorithms.drm import DRM
from scenario.geometry import continuous_special_points
from scenario.scenario_generation import Scenario


def _apply_scenario_updates(drm, scenario):
    
    access_points, io_points = continuous_special_points(scenario.access_points, scenario.io_points, scenario.agv_radius)
    
    drm.apply_layout_updates(scenario.consume_updates(), new_access_points=access_points, new_io_points=io_points)
    drm.build_graph()

if __name__ == "__main__":

    seed = 42
    rng = np.random.default_rng(seed)

    height = 20
    width = 30

    io_points = {
        (height * 0.15, 0),
        (height - 1.5, width * 0.75),
    }

    recorder = ExperimentRecorder(seed=seed, experiment_name="drm_test")

    recorder.set_algorithm("scenario", use_iterations=False)

    scenario = Scenario(
        rng=rng,
        height=height,
        width=width,
        occupancy_rate=0.35,
        sigma_coarse=3,
        sigma_fine=1,
        noisy_aisles_num=0,
        noisy_aisle_sigma=1,
        io_points=io_points,
        io_points_radius=3,
        simple_obstacle_rate=0.1,
        agv_radius=0.42,
        recorder=recorder,
    )

    scenario.generate()

    access_points, continuous_io_points = continuous_special_points(
        scenario.access_points,
        scenario.io_points,
        scenario.agv_radius,
    )

    recorder.set_algorithm("drm", use_iterations=True)

    drm = DRM(
        rng=rng,
        occupancy=scenario.occupancy,
        access_points=access_points,
        io_points=continuous_io_points,
        bounds=(0, 0, width, height),
        max_dist=4.0,
        max_neighbors=10,
        agv_radius=scenario.agv_radius,
        buffer_segments=2,
        recorder=recorder,
    )
    drm.preprocessing_step(num_vertices=500)

    scenario.make_updates(number_of_updates=10)
    _apply_scenario_updates(drm, scenario)

    scenario.remove_region(top_left=(4, 6), bottom_right=(8, 12))
    _apply_scenario_updates(drm, scenario)

    scenario.add_region(top_left=(10, 18), bottom_right=(11, 19))
    _apply_scenario_updates(drm, scenario)

    scenario.add_region(top_left=(0, 0), bottom_right=(1, 1))
    _apply_scenario_updates(drm, scenario)
