from datetime import datetime
import json
from pathlib import Path
import shutil
import sys
import tomllib

import numpy as np

from logs.record import ExperimentRecorder
from experiments.generate_roadmaps import load_scenarios, run_dynamic_algorithm, run_static_algorithm, save_roadmap_metadata
from experiments.generate_scenarios import _interaction_points_from_warehouse_dimensions, _save_run_metadata, apply_scenario_stage
from scenario.scenario_generation import Scenario


RESULTS_ROOT = Path("artifacts/results")
STATIC_ALGORITHMS = ["vgraph", "prm", "gvd", "kings_graph"]
DYNAMIC_ALGORITHMS = ["dynamic_vgraph", "drm", "dynamic_gvd", "dynamic_kings_graph"]


def _scenario_params(config, warehouse, condition):
    scenario_params = {
        key: value for key, value in config["scenario"].items()
        if key != "stages" }
    
    scenario_params.update({
        key: value for key, value in warehouse.items()
        if key not in ("id", "run_dynamic_vgraph")
    })
    
    scenario_params.update({
        key: value for key, value in condition.items()
        if key != "id"})
    
    scenario_params["interaction_points"] = _interaction_points_from_warehouse_dimensions(scenario_params["height"], scenario_params["width"])

    for key in ("ordered_aisle_widths", "ordered_aisle_spacing", "ordered_aisle_offset"):
        scenario_params[key] = tuple(scenario_params.get(key, (0, 0)))
    scenario_params["wall_clearance"] = tuple(scenario_params.get("wall_clearance", (0, 0, 0, 0)))

    return scenario_params, config["scenario"].get("stages", [])



def _save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2) + "\n")


def _record_scenarios(case_dir, config_path, batch_name, timestamp, seed, warehouse_id, condition_id, scenario_params, stages):
    case_dir.mkdir(parents=True, exist_ok=False)
    experiment_name = f"{batch_name}/seed_{seed}/{warehouse_id}/{condition_id}"
    _save_run_metadata(case_dir, config_path, {"name": experiment_name, "seed": seed}, scenario_params, stages, timestamp)

    metadata_path = case_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["batch"] = {"name": batch_name, "timestamp": timestamp, "seed": seed, "warehouse": warehouse_id, "condition": condition_id}
    _save_json(metadata_path, metadata)

    recorder = ExperimentRecorder(seed=seed, experiment_name=case_dir.name, save_metadata=False, root=case_dir.parent, show_priority_map=False)
    recorder.set_algorithm("scenario", use_iterations=True, plot=False)
    scenario = Scenario(rng=np.random.default_rng(seed), recorder=recorder, print_progress=False, **scenario_params)

    scenario.generate()

    for stage in stages:
        apply_scenario_stage(scenario, stage)
        scenario.update_access_points()
        scenario.iteration += 1
        recorder.new_iteration(scenario.iteration)
        scenario._record_scenario()

    
def _case_algorithms(config, warehouse):
    algorithm_configs = {name: dict(params) for name, params in config["algorithms"].items()}
    skipped_algorithms = {}

    # skipping dynamic_vgraph if the warehouse configuration disables it (memory issues for large warehouses)
    if not warehouse.get("run_dynamic_vgraph", True) and "dynamic_vgraph" in algorithm_configs:
        algorithm_configs.pop("dynamic_vgraph")
        skipped_algorithms["dynamic_vgraph"] = "disabled for this warehouse by the matrix configuration"

    return algorithm_configs, skipped_algorithms


def _record_roadmaps(case_dir, config_path, timestamp, seed, algorithm_configs, skipped_algorithms):
    scenarios = load_scenarios(case_dir)
    roadmap_run_dir = case_dir / "roadmaps" / timestamp
    roadmap_run_dir.mkdir(parents=True, exist_ok=False)
    save_roadmap_metadata(roadmap_run_dir, config_path, case_dir, {"seed": seed}, algorithm_configs, len(scenarios), timestamp)

    metadata_path = roadmap_run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["skipped_algorithms"] = skipped_algorithms
    _save_json(metadata_path, metadata)

    recorder = ExperimentRecorder(seed=seed, experiment_name=timestamp, save_metadata=False, root=case_dir / "roadmaps")

    for algorithm_name, algorithm_params in algorithm_configs.items():
        if algorithm_name not in STATIC_ALGORITHMS:
            continue

        print(f"\n\n-------------------------------------------")
        print(f"Running static algorithm: {algorithm_name}")
        for scenario_name, scenario_record in scenarios:
            scenario_data = scenario_record["data"]
            recorder.use_scenario(roadmap_run_dir / scenario_name, scenario_data, scenario_record["iteration"])
            recorder.set_algorithm(algorithm_name, use_iterations=False, plot=False)
            print(f"\nProcessing scenario stage: {scenario_name}")
            run_static_algorithm(algorithm_name, algorithm_params, scenario_data, seed, recorder)

    for algorithm_name, algorithm_params in algorithm_configs.items():
        if algorithm_name not in DYNAMIC_ALGORITHMS:
            continue

        print(f"\n\n-------------------------------------------")
        print(f"\nRunning dynamic algorithm: {algorithm_name}")
        run_dynamic_algorithm(algorithm_name, algorithm_params, scenarios, seed, recorder, roadmap_run_dir)

    return roadmap_run_dir


def run_evaluation_matrix(config_path):

    config_path = Path(config_path)

    with Path(config_path).open("rb") as file:
        config = tomllib.load(file)

    batch_name = config["batch"]["name"]
    seeds = config["batch"]["seeds"]
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    batch_dir = RESULTS_ROOT / batch_name / timestamp
    batch_dir.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(config_path, batch_dir / "input.toml")

    batch_metadata = {"batch_name": batch_name, "timestamp": timestamp, "seeds": seeds, "cases": []}
    _save_json(batch_dir / "metadata.json", batch_metadata)

    for warehouse in config["warehouses"]:
        for seed in seeds:
            for condition in config["conditions"]:
                warehouse_id = warehouse["id"]
                condition_id = condition["id"]
                case_dir = batch_dir / f"seed_{seed}" / warehouse_id / condition_id
                scenario_params, stages = _scenario_params(config, warehouse, condition)
                algorithm_configs, skipped_algorithms = _case_algorithms(config, warehouse)

                print("\n\n=========================================")
                print(f"\nSeed {seed} | {warehouse_id} | {condition_id}")
                print("\n=========================================")
                
                _record_scenarios(case_dir, config_path, batch_name, timestamp, seed, warehouse_id, condition_id, scenario_params, stages)
                roadmap_run_dir = _record_roadmaps(case_dir, config_path, timestamp, seed, algorithm_configs, skipped_algorithms)

                batch_metadata["cases"].append({
                    "seed": seed,
                    "warehouse": warehouse_id,
                    "condition": condition_id,
                    "path": str(case_dir),
                    "roadmap_run": str(roadmap_run_dir),
                    "algorithms": list(algorithm_configs),
                    "skipped_algorithms": skipped_algorithms,
                })
                _save_json(batch_dir / "metadata.json", batch_metadata)

    print(f"\nSaved results to {batch_dir}")
    return batch_dir


def main():
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("configs/evaluation_matrix_one_obstacle_change.toml")
    run_evaluation_matrix(config_path)


if __name__ == "__main__":
    main()
