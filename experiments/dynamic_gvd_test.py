import numpy as np

from algorithms.gvd import GVD
from artifacts.record import ExperimentRecorder
from scenario.scenario_generation import Scenario


def _apply_scenario_updates(gvd, scenario):
    """Apply pending scenario changes and rebuild the incremental GVD graph."""
    gvd.apply_layout_updates(
        scenario.consume_updates(),
        new_access_points=scenario.access_points,
        new_interaction_points=scenario.interaction_points,
    )
    gvd.updateDistanceMap()
    gvd.build_graph()


if __name__ == "__main__":
    seed = 42
    rng = np.random.default_rng(seed)

    height = 25
    width = 25
    
    robot_radius = 0.49

    occupancy_rate = 0.4

    interaction_points = {
        (height * 0.15, 0),
        (height - 1.5, width * 0.75),
    }

    recorder = ExperimentRecorder(
        seed=seed,
        experiment_name="dynamic_gvd_test",
        plot_buffer_segments=5,
    )

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
    )

    scenario.generate()

    recorder.set_algorithm("gvd", use_iterations=True, plot=True)

    gvd = GVD(
        rng=rng,
        occupancy_grid=scenario.occupancy,
        scale_factor=3,
        access_points=scenario.access_points,
        interaction_points=scenario.interaction_points,
        robot_radius=robot_radius,
        recorder=recorder,
        print_progress=True,
    )

    gvd.computeDistanceMap()
    gvd.build_graph()

    scenario.make_updates(number_of_updates=30)
    _apply_scenario_updates(gvd, scenario)

    scenario.remove_region(
        top_left=(int(height * 0.15), int(width * 0.25)),
        bottom_right=(int(height * 0.55), int(width * 0.65)),
    )
    _apply_scenario_updates(gvd, scenario)

    scenario.add_region(
        top_left=(int(height * 0.25), int(width * 0.35)),
        bottom_right=(int(height * 0.45), int(width * 0.55)),
    )
    _apply_scenario_updates(gvd, scenario)

    scenario.add_region(top_left=(0, 0), bottom_right=(1, 1))
    _apply_scenario_updates(gvd, scenario)
