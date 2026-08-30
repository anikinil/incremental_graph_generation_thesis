
import numpy as np
from algorithms.dynamic_vgraph import DynamicVGraph
from artifacts.record import ExperimentRecorder
from scenario.geometry import continuous_special_points
from scenario.scenario_generation import Scenario


if __name__ == "__main__":
    

    seed = 42
    rng = np.random.default_rng(seed)

    height = 50
    width = 70
    
    agv_radius = 0.45
    
    occupancy_rate = 0.5

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

    recorder.set_algorithm("drm", use_iterations=True)

    vgraph = DynamicVGraph(
        height=height,
        width=width,
        occupancy=scenario.occupancy,
        agv_radius=0,
        buffer_segments=0,
        access_points=access_points,
        io_points=io_points,
    )

    vgraph.preprocessing()
    vgraph.record_graph()