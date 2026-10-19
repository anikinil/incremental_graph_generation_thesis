from datetime import datetime
from pathlib import Path
import copy
import csv
import pickle

import numpy as np

from artifacts.plot import PlotManager


class ExperimentRecorder:
    """
    Records experiment data and optionally corresponding plots to disk.
    """

    def __init__(
        self,
        seed=None, # rng seed used in the experiment
        metadata=None, # additional metadata to save in the experiment folder
        experiment_name="", # if empty, will be generated based on timestamp, else gets a specified name
        enabled=True, # false to disable saving and plotting
        plot_buffer_segments=None, # visual buffer resolution for dynamic-vgraph plots only
        save_metadata=True,
        root="artifacts/results",
    ):

        self.root = Path(root)
        self.seed = seed
        self.enabled = enabled

        if experiment_name == "":
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            experiment_name = f"experiment_{timestamp}"

        self.experiment_name = experiment_name
        self.experiment_dir = (self.root / self.experiment_name)

        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        self.algorithm = None
        # ``None`` preserves a caller's per-save plotting choice.  An explicit
        # boolean set through ``set_algorithm`` overrides it for that algorithm.
        self.algorithm_plot = None

        self.use_iterations = True

        self.iteration = 0
        self.scenario_version = -1
        self.scenario_dir = None

        self.scenario = None

        self.metadata = metadata.copy() if metadata else {}

        self.metadata.update({
            "seed": seed,
            "created": datetime.now().isoformat(),
        })

        if save_metadata:
            self._save_metadata()

        self.plot_manager = PlotManager(dynamic_vgraph_plot_buffer_segments=plot_buffer_segments)
        
        self.seconds = None

    def set_algorithm(self, algorithm, use_iterations=True, plot=None):
        """
        Switches the active bucket that subsequent save() calls write into.

        Pass ``plot=True`` or ``plot=False`` to override the plotting choice of
        every artifact subsequently saved for this algorithm.  Leave it as
        ``None`` to retain each save call's existing plotting choice.
        """
        self.algorithm = algorithm
        self.use_iterations = use_iterations
        self.algorithm_plot = plot

    def set_scenario(self, scenario, plot=False, iteration=None):
        """
        Sets the scenario in recorder to later use in plotting of roadmap generation results.
        Saves the scenario to disk and plots scenario (warehouse layout) if plot=True.
        """
        self.scenario = self._snapshot(scenario)
        self.plot_manager.scenario = self.scenario
        self.scenario_version += 1
        self.scenario_dir = self.experiment_dir / f"stage_{self.scenario_version + 1:03d}"
        self.scenario_dir.mkdir(parents=True, exist_ok=True)
        return self.save(data_type="scenario", data=self.scenario, plot=plot, algorithm="scenario", iteration=iteration)

    def use_scenario(self, scenario_dir, scenario, iteration=0):
        scenario_dir = Path(scenario_dir)
        if scenario_dir.resolve().parent != self.experiment_dir.resolve():
            raise ValueError("Scenario directory does not belong to this experiment.")
        if scenario_dir.name.startswith("stage_"):
            scenario_version = int(scenario_dir.name.removeprefix("stage_")) - 1
        else:
            raise ValueError(f"Invalid scenario directory name: {scenario_dir.name}")

        scenario_dir.mkdir(parents=True, exist_ok=True)

        self.scenario = self._snapshot(scenario)
        self.plot_manager.scenario = self.scenario
        self.scenario_dir = scenario_dir
        self.scenario_version = scenario_version
        self.iteration = iteration

    def save(self, data_type, data, plot=False, algorithm=None, iteration=None):
        if not self.enabled:
            return None

        if self.scenario_dir is None:
            raise RuntimeError("A scenario must be recorded before saving artifacts.")

        record_algorithm = self.algorithm if algorithm is None else algorithm
        if not record_algorithm:
            raise RuntimeError("An algorithm must be selected before saving artifacts.")

        # A per-algorithm setting controls artifacts saved for the active
        # algorithm, including the scenario artifact.
        should_plot = (
            self.algorithm_plot
            if record_algorithm == self.algorithm and self.algorithm_plot is not None
            else plot
        )

        base_dir = self.scenario_dir / record_algorithm

        data_dir = base_dir / "data"
        plot_dir = base_dir / "plots"

        data_dir.mkdir(parents=True, exist_ok=True)
        
        if should_plot:
            plot_dir.mkdir(parents=True, exist_ok=True)

        filename = data_type
        sequence = 1
        while (data_dir / f"{filename}.pkl").exists():
            filename = f"{data_type}_{sequence:03d}"
            sequence += 1

        pkl_path = data_dir / f"{filename}.pkl"
        pdf_path = plot_dir / f"{filename}.pdf"

        data_snapshot = self._snapshot(data)
        record_iteration = self.iteration if iteration is None else iteration

        record = {
            "algorithm": record_algorithm,
            "iteration": record_iteration,
            "scenario_version": self.scenario_version,
            "data_type": data_type,
            "timestamp": datetime.now().isoformat(),
            "data": data_snapshot,
        }

        with open(pkl_path, "wb") as f:
            pickle.dump(record, f, protocol=pickle.HIGHEST_PROTOCOL)

        if should_plot:
            self.plot_manager.plot(
                data_type=data_type,
                data=data_snapshot,
                path=pdf_path,
                title_info={
                    "algorithm": record_algorithm,
                    "iteration": record_iteration,
                    "scenario_version": self.scenario_version,
                },
            )

        return {
            "pkl": pkl_path,
            "pdf": pdf_path if should_plot else None,
        }

    def record_time(self, seconds, phase):
        if not self.enabled:
            return None

        if self.scenario_dir is None:
            raise RuntimeError("A scenario must be selected before recording a time.")
        if not self.algorithm:
            raise RuntimeError("An algorithm must be selected before recording a time.")

        self.seconds = seconds

        path = self.experiment_dir / "timings.csv"
        write_header = not path.exists()

        with path.open("a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["algorithm", "scenario", "iteration", "phase", "seconds"])
            if write_header:
                writer.writeheader()
            writer.writerow({
                "algorithm": self.algorithm,
                "scenario": self.scenario_dir.name,
                "iteration": self.iteration,
                "phase": phase,
                "seconds": seconds,
            })
            
    def get_recorded_time(self):
        return self.seconds

    def new_iteration(self, iteration=None):
        """
        Updates the active algorithm iteration metadata.
        """
        if iteration is not None:
            self.iteration = iteration
        else:
            self.iteration += 1

    def _save_metadata(self):
        path = self.experiment_dir / "metadata.pkl"
        with open(path, "wb") as f:
            pickle.dump(self.metadata, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _snapshot(data):
        """Returns a deep copy of data to later save to disk."""
        
        if isinstance(data, np.ndarray):
            return data.copy()

        if isinstance(data, dict):
            return { key: ExperimentRecorder._snapshot(value) for key, value in data.items() }

        if isinstance(data, list):
            return [ExperimentRecorder._snapshot(value) for value in data]

        if isinstance(data, tuple):
            return tuple(ExperimentRecorder._snapshot(value) for value in data)

        if isinstance(data, set):
            return { ExperimentRecorder._snapshot(value) for value in data}

        return copy.deepcopy(data)
