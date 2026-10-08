
import numpy as np
from algorithms.dynamic_vgraph import DynamicVGraph
from artifacts.record import ExperimentRecorder
from algorithms.geometry import continuous_special_points
from scenario.scenario_generation import Scenario


def _apply_scenario_updates(vgraph, scenario):

    access_points, io_points = continuous_special_points(scenario.access_points, scenario.io_points, scenario.agv_radius)

    vgraph.apply_layout_updates(scenario.consume_updates(), new_access_points=access_points, new_io_points=io_points)


if __name__ == "__main__":
    



    seed = 40
    rng = np.random.default_rng(seed)

    height = 6
    width = 9
    
    agv_radius = 0.3
    
    occupancy_rate = 0.4

    io_points = {
        (height - 0.3, width * 0.73),
        (height - 0.3, width * 0.78),
        (height - 0.3, width * 0.83),
    }

    recorder = ExperimentRecorder(seed=seed, experiment_name="dynamic_vgraph_test", plot_buffer_segments=0)

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

    scenario.generate()

    access_points, continuous_io_points = continuous_special_points(
        scenario.access_points,
        scenario.io_points,
        scenario.agv_radius,
    )

    recorder.set_algorithm("dynamic_vgraph", use_iterations=True, plot=True)

    vgraph = DynamicVGraph(
        height=height,
        width=width,
        occupancy=scenario.occupancy,
        agv_radius=agv_radius,
        access_points=access_points,
        io_points=continuous_io_points,
        recorder=recorder,
    )

    vgraph.preprocessing()
    vgraph.record_graph(phase="preprocessing")
    
    scenario._add_obstacle(cell=(3, 5), type=2)
    scenario.update_access_points()
    scenario._record_scenario()
    
    scenario._remove_obstacle(cell=(5, 2))
    scenario.update_access_points()
    _apply_scenario_updates(vgraph, scenario)

    # scenario.make_updates(number_of_updates=1)
    # _apply_scenario_updates(vgraph, scenario)
    
    # scenario.make_updates(number_of_updates=3)
    # _apply_scenario_updates(vgraph, scenario)


    # scenario.remove_region(top_left=(int(height * 0.15), int(width * 0.25)), bottom_right=(int(height * 0.55), int(width * 0.65)))
    # _apply_scenario_updates(vgraph, scenario)

    # scenario.add_region(top_left=(int(height * 0.25), int(width * 0.35)), bottom_right=(int(height * 0.45), int(width * 0.55)))
    # _apply_scenario_updates(vgraph, scenario)

    # scenario.add_region(top_left=(0, 0), bottom_right=(1, 1))
    # _apply_scenario_updates(vgraph, scenario)
