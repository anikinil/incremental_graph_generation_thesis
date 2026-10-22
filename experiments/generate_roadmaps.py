from datetime import datetime
import json
from pathlib import Path
import pickle
import shutil
import tomllib

import numpy as np

from algorithms.drm import DRM
from algorithms.dynamic_vgraph import DynamicVGraph
from algorithms.geometry import continuous_special_points
from algorithms.gvd import GVD
from algorithms.kings_graph import IncrementalKingsGraph
from algorithms.prm import PRM
from algorithms.vgraph import VGraph
from artifacts.record import ExperimentRecorder


def load_roadmap_config(toml_path):
    with Path(toml_path).open("rb") as file:
        config = tomllib.load(file)

    return config["experiment"], config["algorithms"]


def load_scenarios(scenario_run_dir):
    scenarios = []

    stage_dirs = sorted(scenario_run_dir.glob("stage_*"))
    if not stage_dirs:
        stage_dirs = sorted(scenario_run_dir.glob("scenario_*"))

    for scenario_dir in stage_dirs:
        path = scenario_dir / "scenario" / "data" / "scenario.pkl"
        if not path.is_file():
            continue
        with path.open("rb") as file:
            scenarios.append((scenario_dir.name, pickle.load(file)))

    if not scenarios:
        raise RuntimeError(f"No saved scenarios found in {scenario_run_dir}")

    return scenarios


def find_scenario_run(scenario_run):
    """Gets the specified scenario run directory or if the time-dir was not specified, finds the most recent one with the same experiment name."""
    scenario_run_dir = Path("artifacts/results") / scenario_run

    if (scenario_run_dir / "stage_001").is_dir() or (scenario_run_dir / "scenario_000").is_dir():
        return scenario_run_dir

    if not scenario_run_dir.is_dir():
        raise FileNotFoundError(f"Scenario experiment does not exist: {scenario_run_dir}")

    # if the time directory not specified, ge the most recent
    timestamp_dirs = []
    for path in scenario_run_dir.iterdir():
        if not path.is_dir():
            continue
        try:
            datetime.strptime(path.name, "%Y-%m-%d_%H-%M-%S")
        except ValueError:
            continue
        timestamp_dirs.append(path)

    if not timestamp_dirs:
        raise RuntimeError(f"No timestamped scenario runs found in {scenario_run_dir}")

    return max(timestamp_dirs, key=lambda path: path.name)


def save_roadmap_metadata(roadmap_run_dir, config_path, scenario_run_dir, experiment_config, algorithm_configs, scenario_count, timestamp):
    shutil.copyfile(config_path, roadmap_run_dir / "input.toml")

    metadata = {
        "created_at": datetime.now().isoformat(),
        "timestamp": timestamp,
        "scenario_run": str(scenario_run_dir.relative_to("artifacts/results")),
        "seed": experiment_config["seed"],
        "input_toml": "input.toml",
        "scenario_count": scenario_count,
        "algorithms": ", ".join(algorithm_configs),
    }
    (roadmap_run_dir / "metadata.json").write_text(json.dumps(metadata, indent=4) + "\n")


def run_static_algorithm(name, params, scenario_data, seed, recorder):
    occupancy = scenario_data["occupancy"]
    access_points = scenario_data["access_points"]
    interaction_points = scenario_data["interaction_points"]
    height = scenario_data["height"]
    width = scenario_data["width"]
    robot_radius = scenario_data["robot_radius"]

    if name == "vgraph":
        access_points, interaction_points = continuous_special_points(access_points, interaction_points, robot_radius)
        roadmap = VGraph(width, height, occupancy, robot_radius, access_points, interaction_points, params.get("num_buffer_segments", 0), recorder, False)
        roadmap.create_visibility_graph()
        roadmap.record_graph()

    elif name == "prm":
        access_points, interaction_points = continuous_special_points(access_points, interaction_points, robot_radius)
        roadmap = PRM(np.random.default_rng(seed), occupancy, access_points, interaction_points, height, width, params["eps"], params["max_dist"], params["max_neighbors"], robot_radius, params.get("num_buffer_segments", 0), recorder, False)
        roadmap.construction_step(height * width * params["min_num_nodes_factor"], params.get("avoid_cycles", False))
        roadmap.record_graph()
        
    elif name == "gvd":
        roadmap = GVD(np.random.default_rng(seed), occupancy, params.get("scale_factor", 1), access_points, interaction_points, robot_radius, recorder, False)
        roadmap.computeDistanceMap()
        
    elif name == "kings_graph":
        roadmap = IncrementalKingsGraph((0, 0, height, width), robot_radius, occupancy, access_points, interaction_points, recorder, False)
        roadmap.build_graph()
        roadmap.record_graph()

    else:
        raise ValueError(f"Unknown roadmap algorithm: {name}")

    print(f"Time: {recorder.get_recorded_time()} seconds")


def get_obstacle_updates(previous_occupancy, occupancy):
    updates = []

    added_cells = np.argwhere((previous_occupancy == 0) & (occupancy != 0))
    for row, col in added_cells:
        updates.append({"action": "add", "cell": (int(row), int(col)), "obstacle_type": int(occupancy[row, col])})

    removed_cells = np.argwhere((previous_occupancy != 0) & (occupancy == 0))
    for row, col in removed_cells:
        updates.append({"action": "remove", "cell": (int(row), int(col))})

    return updates


def run_dynamic_algorithm(name, params, scenarios, seed, recorder, roadmap_run_dir):
    scenario_name, scenario_record = scenarios[0]
    scenario_data = scenario_record["data"]
    recorder.use_scenario(roadmap_run_dir / scenario_name, scenario_data, scenario_record["iteration"])
    recorder.set_algorithm(name, use_iterations=False, plot=False)
    print(f"\nProcessing scenario: {scenario_name}")

    occupancy = scenario_data["occupancy"]
    access_points = scenario_data["access_points"]
    interaction_points = scenario_data["interaction_points"]
    height = scenario_data["height"]
    width = scenario_data["width"]
    robot_radius = scenario_data["robot_radius"]

    if name == "dynamic_vgraph":
        access_points, interaction_points = continuous_special_points(access_points, interaction_points, robot_radius)
        roadmap = DynamicVGraph(height, width, occupancy, robot_radius, access_points, interaction_points, recorder, False)
        roadmap.preprocessing()
        roadmap.record_graph(phase="preprocessing")
        
    elif name == "drm":
        access_points, interaction_points = continuous_special_points(access_points, interaction_points, robot_radius)
        roadmap = DRM(np.random.default_rng(seed), occupancy, access_points, interaction_points, height, width, params["max_dist"], params["max_neighbors"], robot_radius, params.get("num_buffer_segments", 0), recorder, False)
        roadmap.preprocessing_step(height * width * params["min_num_nodes_factor"])

    elif name == "dynamic_gvd":
        roadmap = GVD(np.random.default_rng(seed), occupancy, params.get("scale_factor", 1), access_points, interaction_points, robot_radius, recorder, False)
        roadmap.computeDistanceMap()

    elif name == "dynamic_kings_graph":
        roadmap = IncrementalKingsGraph((0, 0, height, width), robot_radius, occupancy, access_points, interaction_points, recorder, False)
        roadmap.build_graph()
        roadmap.record_graph()

    else:
        raise ValueError(f"Unknown dynamic roadmap algorithm: {name}")
    
    print(f"Time: {recorder.get_recorded_time()} seconds")
    

    # update the dynamic roadmap for every later scenario stage
    previous_scenario_data = scenario_data
    for scenario_name, scenario_record in scenarios[1:]:
        scenario_data = scenario_record["data"]
        recorder.use_scenario(roadmap_run_dir / scenario_name, scenario_data, scenario_record["iteration"])
        print(f"\nProcessing scenario: {scenario_name}")
        obstacle_updates = get_obstacle_updates(previous_scenario_data["occupancy"], scenario_data["occupancy"])

        if name == "dynamic_vgraph":
            access_points, interaction_points = continuous_special_points(scenario_data["access_points"], scenario_data["interaction_points"], scenario_data["robot_radius"])
            roadmap.apply_layout_updates(obstacle_updates, access_points, interaction_points)

        elif name == "drm":
            access_points, interaction_points = continuous_special_points(scenario_data["access_points"], scenario_data["interaction_points"], scenario_data["robot_radius"])
            # roadmap.apply_layout_updates(obstacle_updates, access_points, interaction_points, height * width * params["min_num_nodes_factor"])
            roadmap.apply_layout_updates(obstacle_updates, access_points, interaction_points, None)

        elif name == "dynamic_gvd":
            roadmap.apply_layout_updates(obstacle_updates, scenario_data["access_points"], scenario_data["interaction_points"])
            roadmap.updateDistanceMap()

        elif name == "dynamic_kings_graph":
            roadmap.apply_layout_updates_and_update_graph(obstacle_updates, scenario_data["access_points"], scenario_data["interaction_points"])
            roadmap.record_graph()

        print(f"Time: {recorder.get_recorded_time()} seconds")

        previous_scenario_data = scenario_data


def main():
    config_path = Path("experiments/roadmap_config.toml")
    experiment_config, algorithm_configs = load_roadmap_config(config_path)

    scenario_run_dir = find_scenario_run(experiment_config["scenario_run"])

    scenarios = load_scenarios(scenario_run_dir)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    roadmap_run_dir = scenario_run_dir / "roadmaps" / timestamp
    roadmap_run_dir.mkdir(parents=True, exist_ok=False)

    save_roadmap_metadata(roadmap_run_dir, config_path, scenario_run_dir, experiment_config, algorithm_configs, len(scenarios), timestamp)

    recorder = ExperimentRecorder(seed=experiment_config["seed"], experiment_name=timestamp, save_metadata=False, root=scenario_run_dir / "roadmaps")

    static_algorithms = ["vgraph", "prm", "gvd", "kings_graph"]
    dynamic_algorithms = ["dynamic_vgraph", "drm", "dynamic_gvd", "dynamic_kings_graph"]
    
    for algorithm_name, algorithm_params in algorithm_configs.items():
        if algorithm_name in static_algorithms:
            
            print(f"\n\n-------------------------------------------")
            print(f"Running algorithm: {algorithm_name}")
            for scenario_name, scenario_record in scenarios:
                scenario_data = scenario_record["data"]
                recorder.use_scenario(roadmap_run_dir / scenario_name, scenario_data, scenario_record["iteration"])
                recorder.set_algorithm(algorithm_name, use_iterations=False, plot=False)
                print(f"\nProcessing scenario stage: {scenario_name}")
                run_static_algorithm(algorithm_name, algorithm_params, scenario_data, experiment_config["seed"], recorder)

    for algorithm_name, algorithm_params in algorithm_configs.items():
        if algorithm_name in dynamic_algorithms:

            print(f"\n\n-------------------------------------------")
            print(f"Running algorithm: {algorithm_name}")
            run_dynamic_algorithm(algorithm_name, algorithm_params, scenarios, experiment_config["seed"], recorder, roadmap_run_dir)
            

    print(f"Saved roadmaps to {roadmap_run_dir}")
    
    for _ in range(20):
        print("###############################################")


if __name__ == "__main__":
    main()
