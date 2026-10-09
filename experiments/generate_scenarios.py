
from pathlib import Path
import tomllib

import numpy as np

from scenario.scenario_generation import Scenario


def load_scenario_config(toml_path: str | Path) -> tuple[Scenario, list[dict]]:
    with Path(toml_path).open("rb") as file:
        config = tomllib.load(file)

    experiment_config = config["experiment"]
    
    scenario_config = config["scenario"]

    # scenario pararms without stages
    scenario_params = {
        key: value
        for key, value in scenario_config.items()
        if key != "stages"
    }

    # interaction point placement
    scenario_params["interaction_points"] = {
        tuple(point)
        for point in scenario_params["interaction_points"]
    }

    # params that need to be tuples
    for key in ("ordered_aisle_widths", "ordered_aisle_spacing", "ordered_aisle_offset"):
        scenario_params[key] = tuple(scenario_params.get(key, (0, 0)))

    # wall clearance
    scenario_params["wall_clearance"] = tuple(scenario_params.get("wall_clearance", (0, 0, 0, 0)))

    return experiment_config, scenario_params, scenario_config.get("stages", [])


def main():
    
    experiment_config, scenario_params, stages = load_scenario_config("experiments/scenario_config.toml")

    scenario = Scenario(
        rng=np.random.default_rng(experiment_config["seed"]),
        recorder=None,
        print_progress=False,
        **scenario_params,
    )

    scenario.generate()

    for stage in stages:
        if stage["type"] == "add_random_obstacles":
            scenario.add_obstacles_randomly(count=stage["count"])

        elif stage["type"] == "remove_random_obstacles":
            scenario.remove_obstacles_randomly(count=stage["count"])
            
        else:
            raise ValueError(f"Unknown scenario stage type: {stage['type']}")
        
        




if __name__ == "__main__":
    main()
