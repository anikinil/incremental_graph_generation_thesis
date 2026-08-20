import numpy as np

from scenario.scenario_generation import Scenario
from save.record import ExperimentRecorder

from algorithms.gvd import GVD

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

    recorder.set_bucket("scenario", use_iterations=False)
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
        simple_obstacle_rate=1,
        agv_radius=0.42,
        recorder=recorder,
    )
    scenario.generate()

    recorder.set_bucket("gvd", use_iterations=True)
    gvd = GVD(
        rng=rng,
        occupancy_grid=scenario.occupancy,
        scale_factor=3,
        access_points=scenario.access_points,
        io_points=scenario.io_points,
        agv_radius=scenario.agv_radius,
        recorder=recorder
    )
    gvd.computeDistanceMap()
    gvd.build_graph()    


if __name__ == "__main__":
    main()
        
