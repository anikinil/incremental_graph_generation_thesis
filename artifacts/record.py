from pathlib import Path
from datetime import datetime
import copy
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
    ):

        self.root = Path("artifacts/results")
        self.seed = seed
        self.enabled = enabled

        if experiment_name == "":
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            experiment_name = f"experiment_{timestamp}"

        self.experiment_name = experiment_name
        self.experiment_dir = (self.root / self.experiment_name)

        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        self.algorithm = None
        self.algorithm_dir = None

        self.use_iterations = True

        self.iteration = 0
        self.step = 0

        self.scenario = None

        self.metadata = metadata.copy() if metadata else {}

        self.metadata.update({
            "seed": seed,
            "created": datetime.now().isoformat(),
        })

        self._save_metadata()

        self.plot_manager = PlotManager()

    def set_algorithm(self, algorithm, use_iterations=True):
        """
        Switches the active bucket that subsequent save() calls write into.
        """
        self.algorithm = algorithm
        self.use_iterations = use_iterations

        self.algorithm_dir = (self.experiment_dir / algorithm)
        self.algorithm_dir.mkdir(parents=True, exist_ok=True)

        self.iteration = 0
        self.step = 0

        self.metadata["algorithm"] = self.algorithm

        self._save_metadata()

    def set_scenario(self, scenario, plot=False):
        """
        Sets the scenario in recorder to later use in plotting of roadmap generation results.
        Saves the scenario to disk and plots scenario (warehouse layout) if plot=True.
        """
        self.scenario = self._snapshot(scenario)
        self.plot_manager.scenario = self.scenario
        self.save(data_type="scenario", data=self.scenario, plot=plot)

    def save(self, data_type, data, plot=False):
        if not self.enabled:
            return None

        if self.algorithm is None:
            base_dir = self.experiment_dir

        elif self.use_iterations:
            base_dir = self._get_iteration_dir()
            filename = f"{self.step:05d}_{data_type}"

        else:
            base_dir = self.algorithm_dir

        data_dir = base_dir / "data"
        plot_dir = base_dir / "plots"

        data_dir.mkdir(parents=True, exist_ok=True)
        
        if plot:
            plot_dir.mkdir(parents=True, exist_ok=True)

        if self.algorithm is not None and self.use_iterations:
            filename = f"{self.step:05d}_{data_type}"
        else:
            filename = data_type

        pkl_path = data_dir / f"{filename}.pkl"
        pdf_path = plot_dir / f"{filename}.pdf"

        data_snapshot = self._snapshot(data)

        record = {
            "algorithm": self.algorithm,
            "iteration": self.iteration if self.use_iterations else None,
            "step": self.step if self.use_iterations else None,
            "data_type": data_type,
            "timestamp": datetime.now().isoformat(),
            "data": data_snapshot,
        }

        with open(pkl_path, "wb") as f:
            pickle.dump(record, f, protocol=pickle.HIGHEST_PROTOCOL)

        if plot:
            self.plot_manager.plot(
                data_type=data_type,
                data=data_snapshot,
                path=pdf_path,
                title_info={
                    "algorithm": self.algorithm,
                    "iteration": self.iteration if self.use_iterations else "-",
                    "step": self.step if self.use_iterations else "-",
                },
            )

        if self.algorithm is not None and self.use_iterations:
            self.step += 1

        return {
            "pkl": pkl_path,
            "pdf": pdf_path if plot else None,
        }

    def new_iteration(self, iteration=None):
        """
        Starts a new iteration for the current algorithm bucket.
        If iteration is None, increments the current iteration by 1.
        """
        if iteration is not None:
            self.iteration = iteration
        else:
            self.iteration += 1
            self.step = 0
        self._get_iteration_dir()

    def _get_iteration_dir(self):
        """Makes sure the directory for the current iteration exists and returns it."""
        path = (self.algorithm_dir / f"iteration_{self.iteration:03d}")
        path.mkdir(parents=True, exist_ok=True)
        return path

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
