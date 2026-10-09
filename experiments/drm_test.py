import networkx as nx

import numpy as np

from artifacts.record import ExperimentRecorder
from algorithms.drm import DRM
from algorithms.geometry import continuous_special_points
from scenario.scenario_generation import Scenario


def _apply_scenario_updates(drm, scenario, min_num_vertices=None):
    
    access_points, interaction_points = continuous_special_points(scenario.access_points, scenario.interaction_points, scenario.agv_radius)
    
    drm.apply_layout_updates(scenario.consume_updates(), new_access_points=access_points, new_interaction_points=interaction_points, min_num_vertices=min_num_vertices)
    drm.build_graph()

if __name__ == "__main__":

    seed = 41
    rng = np.random.default_rng(seed)

    
    height = 30
    width = 30
    
    agv_radius = 0.1
    
    occupancy_rate = 0.4

    interaction_points = {
        (height * 0.15, 0),
        (height - 1.5, width * 0.75),
    }

    recorder = ExperimentRecorder(seed=seed, experiment_name="drm_test")

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
        interaction_points=interaction_points,
        interaction_points_radius=3,
        simple_obstacle_rate=0.1,
        agv_radius=agv_radius,
        recorder=recorder,
        print_progress=True,
    )

    scenario.generate()

    access_points, continuous_interaction_points = continuous_special_points(
        scenario.access_points,
        scenario.interaction_points,
        scenario.agv_radius,
    )

    recorder.set_algorithm("drm", use_iterations=True, plot=True)

    drm = DRM(
        rng=rng,
        occupancy=scenario.occupancy,
        access_points=access_points,
        interaction_points=continuous_interaction_points,
        height=height,
        width=width,
        max_dist=5,
        max_neighbors=10,
        agv_radius=scenario.agv_radius,
        num_buffer_segments=2,
        recorder=recorder,
        print_progress=True,
    )
    
    # drm.preprocessing_step()
    drm.preprocessing_step(min_num_vertices=1000)
    # drm.preprocessing_step(num_vertices=int(width * height * (1-scenario.occupancy_rate) // 2))


    scenario.make_updates(number_of_updates=30)
    _apply_scenario_updates(drm, scenario)

    scenario.remove_region(top_left=(int(height * 0.15), int(width * 0.25)), bottom_right=(int(height * 0.55), int(width * 0.65)))
    _apply_scenario_updates(drm, scenario)

    scenario.add_region(top_left=(int(height * 0.25), int(width * 0.35)), bottom_right=(int(height * 0.45), int(width * 0.55)))
    _apply_scenario_updates(drm, scenario)

    scenario.add_region(top_left=(0, 0), bottom_right=(1, 1))
    _apply_scenario_updates(drm, scenario)
