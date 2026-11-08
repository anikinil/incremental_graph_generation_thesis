from datetime import datetime
import json
from pathlib import Path
import shutil
import tomllib

import numpy as np

from logs.record import ExperimentRecorder
from scenario.scenario_generation import Scenario


def _interaction_points_from_warehouse_dimensions(height, width):
    """Return interaction points at positions relative to the warehouse dimensions."""
    return {
        (height * 0.1, width * 0.05),
        (height * 0.14, width * 0.05),
        (height * 0.2, width * 0.05),
        
        (height * 0.95, width * 0.75),
        (height * 0.95, width * 0.78),
        (height * 0.95, width * 0.84),
        (height * 0.95, width * 0.93)
    }


def load_scenario_config(toml_path):
    with Path(toml_path).open("rb") as file:
        config = tomllib.load(file)

    experiment_config = config["experiment"]    
    scenario_config = config["scenario"]

    scenario_params = {key: value for key, value in scenario_config.items() if key != "stages"}
    scenario_params["interaction_points"] = _interaction_points_from_warehouse_dimensions(scenario_params["height"], scenario_params["width"])

    for key in ("ordered_aisle_widths", "ordered_aisle_spacing", "ordered_aisle_offset"):
        scenario_params[key] = tuple(scenario_params.get(key, (0, 0)))

    scenario_params["wall_clearance"] = tuple(scenario_params.get("wall_clearance", (0, 0, 0, 0)))

    return experiment_config, scenario_params, scenario_config.get("stages", [])


def _save_run_metadata(run_dir, config_path, experiment_config, scenario_params, stages, timestamp):
    def json_value(value):
        if isinstance(value, dict):
            return {key: json_value(item) for key, item in value.items()}
        if isinstance(value, set):
            try:
                items = sorted(value)
            except TypeError:
                items = sorted(value, key=repr)
            return [json_value(item) for item in items]
        if isinstance(value, (list, tuple)):
            return [json_value(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, np.generic):
            return value.item()
        return value

    shutil.copyfile(config_path, run_dir / "input.toml")

    metadata = {
        "created_at": datetime.now().isoformat(),
        "timestamp": timestamp,
        "experiment_name": experiment_config["name"],
        "description": experiment_config.get("description"),
        "seed": experiment_config["seed"],
        "input_toml": "input.toml",
        "resolved_scenario_parameters": scenario_params,
        "dynamic_stages": stages,
        "expected_scenario_stages": len(stages) + 1,
    }
    (run_dir / "metadata.json").write_text(json.dumps(json_value(metadata), indent=4) + "\n")


def apply_scenario_stage(scenario, stage):
    """Apply one configured scenario update stage."""
    stage_type = stage["type"]

    if stage_type == "add_random_obstacles":
        scenario.add_obstacles_randomly(count=int(stage["percentage"] * scenario.get_total_obstacle_count()))
    elif stage_type == "remove_random_obstacles":
        scenario.remove_obstacles_randomly(count=int(stage["percentage"] * scenario.get_total_obstacle_count()))
    elif stage_type == "change_random_obstacles":
        scenario.change_obstacles_randomly(percentage=stage.get("percentage"), count=stage.get("count"))
    else:
        raise ValueError(f"Unknown scenario stage type: {stage_type}")


def main():
    config_path = Path("configs/scenario_config.toml")
    experiment_config, scenario_params, stages = load_scenario_config(config_path)

    experiment_name = experiment_config["name"]
    seed = experiment_config["seed"]
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    run_dir = (Path("artifacts/results") / experiment_name / timestamp)
    run_dir.mkdir(parents=True, exist_ok=False)

    _save_run_metadata(run_dir, config_path, experiment_config, scenario_params, stages, timestamp)

    recorder = ExperimentRecorder(seed=seed, experiment_name=f"{experiment_name}/{timestamp}", save_metadata=False, show_priority_map=False)
    recorder.set_algorithm("scenario", use_iterations=True, plot=True)

    scenario = Scenario(rng=np.random.default_rng(seed), recorder=recorder, print_progress=False, **scenario_params)

    if not scenario.generate():
        raise RuntimeError("Initial scenario generation failed.")

    for stage in stages:
        apply_scenario_stage(scenario, stage)

        scenario.update_access_points()
        scenario.iteration += 1
        recorder.new_iteration(scenario.iteration)
        scenario._record_scenario()

    print(f"Saved scenarios to {run_dir}")


if __name__ == "__main__":
    main()
