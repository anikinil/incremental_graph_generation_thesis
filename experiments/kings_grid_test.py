import numpy as np

from scenario.scenario_generation import Scenario
from algorithms.kings_graph import IncrementalKingsGraph
from artifacts.record import ExperimentRecorder

    
if __name__ == "__main__":
    seed = 42
    rng = np.random.default_rng(seed)

    height = 25
    width = 30

    io_points = {
        (height * 0.13, 0),
        (height - 1.8, width * 0.78),
    }

    recorder = ExperimentRecorder(seed=seed)
    recorder.set_algorithm("scenario", use_iterations=False)
    
    scenario = Scenario(
        rng=rng,
        height=height,
        width=width,
        occupancy_rate=0.5,
        sigma_coarse=3,
        sigma_fine=1,
        noisy_aisles_num=0,
        noisy_aisle_sigma=1,
        io_points=io_points,
        io_points_radius=3,
        simple_obstacle_rate=0.1,
        agv_radius=0.1,
        recorder=recorder,
        
        # ordered_aisle_widths = (2, 1),
        # ordered_aisle_spacing = (3, 5),
        # ordered_aisle_offset = (0, 0),
        # wall_clearance = (2, 2, 3, 3)
    )
    
    scenario.generate()
    print()

    recorder.set_algorithm("incremental_kings_grid", use_iterations=True)

    kings_grid = IncrementalKingsGraph(
        occupancy=scenario.occupancy,
        access_points=scenario.access_points,
        io_points=scenario.io_points,
        bounds=(0, 0, height, width),
        agv_radius=scenario.agv_radius,
        recorder=recorder,
    )
    kings_grid.build_graph()
    kings_grid.record_graph()

    scenario.make_updates(number_of_updates=10)
    kings_grid.apply_layout_updates_and_update_graph(scenario.consume_updates(), scenario.access_points, scenario.io_points,)
    kings_grid.record_graph()

    scenario.remove_region(top_left=(5, 5), bottom_right=(15, 15))
    kings_grid.apply_layout_updates_and_update_graph(scenario.consume_updates(), scenario.access_points, scenario.io_points,)
    kings_grid.record_graph()


    scenario.add_region(top_left=(10, 8), bottom_right=(12, 11))
    kings_grid.apply_layout_updates_and_update_graph(scenario.consume_updates(), scenario.access_points, scenario.io_points,)
    kings_grid.record_graph()

    
