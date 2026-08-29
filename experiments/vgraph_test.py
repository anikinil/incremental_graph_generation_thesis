import numpy as np

from scenario.scenario_generation import Scenario
from algorithms.vgraph import VGraph
from artifacts.record import ExperimentRecorder

from scenario.geometry import continuous_special_points





if __name__ == "__main__":


    seed = 42
    rng = np.random.default_rng(seed)

    height = 20
    width = 20
    
    agv_radius = 0.45
    
    occupancy_rate = 0.5

    io_points = {
        (height * 0.18, 0),
        (height - 1.6, width * 0.73),
    }

    recorder = ExperimentRecorder(seed=seed)

    recorder.set_algorithm("scenario", use_iterations=False)
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
        
        # ordered_aisle_widths = (2, 1),
        # ordered_aisle_spacing = (3, 5),
        # ordered_aisle_offset = (0, 0),
        # wall_clearance = (2, 2, 3, 3)
    )
    scenario_success = scenario.generate()
    print()

    continuous_access_points, continuous_io_points = continuous_special_points(scenario.access_points, scenario.io_points, scenario.agv_radius)
    bounds = (0, 0, width, height)


    recorder.set_algorithm("vgraph", use_iterations=False)
    buffer_segments = 0
    vgraph = VGraph(
        width=width,
        height=height,
        occupancy=scenario.occupancy,
        agv_radius=scenario.agv_radius,
        access_points=continuous_access_points,
        io_points=continuous_io_points,
        buffer_segments=buffer_segments,
        recorder=recorder,
    )
    vgraph.create_visibility_graph()
    vgraph.record_graph()