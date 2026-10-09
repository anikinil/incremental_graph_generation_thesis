import numpy as np

from scenario.scenario_generation import Scenario
from algorithms.vgraph import VGraph
from artifacts.record import ExperimentRecorder

from algorithms.geometry import continuous_special_points





if __name__ == "__main__":


    seed = 42
    rng = np.random.default_rng(seed)

    height = 15
    width = 20
    
    robot_radius = 0.3
    
    occupancy_rate = 0.3

    interaction_points = {
        (height - 0.3, width * 0.73),
        (height - 0.3, width * 0.78),
        (height - 0.3, width * 0.83),
    }

    recorder = ExperimentRecorder(seed=seed, experiment_name="vgraph_test")

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
        robot_radius=robot_radius,
        recorder=recorder,
        print_progress=True,
        
        # ordered_aisle_widths = (2, 1),
        # ordered_aisle_spacing = (3, 5),
        # ordered_aisle_offset = (0, 0),
        # wall_clearance = (2, 2, 3, 3)
    )
    scenario_success = scenario.generate()
    print()

    continuous_access_points, continuous_interaction_points = continuous_special_points(scenario.access_points, scenario.interaction_points, scenario.robot_radius)
    bounds = (0, 0, width, height)


    recorder.set_algorithm("vgraph", use_iterations=False, plot=True)
    num_buffer_segments = 0
    vgraph = VGraph(
        width=width,
        height=height,
        occupancy=scenario.occupancy,
        robot_radius=scenario.robot_radius,
        access_points=continuous_access_points,
        interaction_points=continuous_interaction_points,
        num_buffer_segments=num_buffer_segments,
        recorder=recorder,
        print_progress=True,
    )
    vgraph.create_visibility_graph()
    vgraph.record_graph()
