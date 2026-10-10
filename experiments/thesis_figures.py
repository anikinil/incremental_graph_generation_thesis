"""Generate the scenario and roadmap figures used in the thesis.

Run this module from the repository root:

    python -m experiments.thesis_figures

Each run writes PDF files only to a timestamped directory below
``artifacts/results/thesis_figures``.  That directory contains
``scenario_with_aisles.pdf``, ``scenario_without_aisles.pdf``, ``vgraph.pdf``,
``dynamic_vgraph_0.pdf``, ``dynamic_vgraph_1.pdf``, ``drm_0.pdf``,
``drm_1.pdf``, ``prm.pdf``, ``gvd_distance_map.pdf``,
``gvd_voronoi_skeleton.pdf``, ``gvd_roadmap.pdf``, ``gvd_connections.pdf``,
``dynamic_gvd_0.pdf``,
``dynamic_gvd_1.pdf``, ``kings_graph.pdf``, ``dynamic_kings_graph_0.pdf``,
and ``dynamic_kings_graph_1.pdf``—with no pickle files, metadata, or
per-scenario subdirectories.
"""

from datetime import datetime
from pathlib import Path

import numpy as np

from algorithms.dynamic_vgraph import DynamicVGraph
from algorithms.drm import DRM
from algorithms.geometry import continuous_special_points
from algorithms.gvd import GVD
from algorithms.kings_graph import IncrementalKingsGraph
from algorithms.prm import PRM
from algorithms.vgraph import VGraph
from artifacts.plot import PlotManager
from scenario.scenario_generation import Scenario


SCENARIO_SEED = 42
VGRAPH_SEED = 42
DYNAMIC_VGRAPH_SEED = 40
DRM_SEED = 41
GVD_SEED = 43
DYNAMIC_GVD_SEED = GVD_SEED
KINGS_GRAPH_SEED = 45
DYNAMIC_KINGS_GRAPH_SEED = 46
FIGURE_DIRECTORY = Path("artifacts/visualization/thesis_figures")
RUN_DIRECTORY = FIGURE_DIRECTORY / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

PRINT_PROGRESS = True

def _lower_left_interaction_points(height, width):
    """Return three interaction points on the lower-left edge."""
    return {
        # (height - 0.3, width * 0.11),
        (height - 0.3, 5),
        (height - 0.3, width * 0.22),
        (height - 0.3, width * 0.33),
    }


def _dynamic_vgraph_test_interaction_points(height, width):
    """Return the interaction point layout used by dynamic_vgraph_test.py."""
    return {
        (height - 0.3, width * 0.70),
        (height - 0.3, width * 0.78),
        (height - 0.3, width * 0.9),
        (height - 0.3, width * 0.98),
    }

# retaining the PRM settings used by the main experiment.
PRM_MIN_NUM_VERTICES = 1000
DRM_MIN_NUM_VERTICES = 1000
DRM_LAYOUT_CHANGE_COUNT = 6
ILLUSTRATIVE_LAYOUT_CHANGE_COUNT = 4


class ThesisFigureRecorder:
    """Recorder-compatible PDF writer used only by this figure-generation script.

    The roadmap implementations already talk to a recorder.  This small adapter
    keeps that interface but forwards only plot requests to ``PlotManager`` so
    thesis figures do not produce serialised experiment data.
    """

    def __init__(self, scenario_filename, *, plot_buffer_segments=None):
        RUN_DIRECTORY.mkdir(parents=True, exist_ok=True)
        self.scenario_filename = scenario_filename
        self.algorithm = None
        self.iteration = 0
        self._scenario_written = False
        self.plot_manager = PlotManager(
            dynamic_vgraph_plot_buffer_segments=plot_buffer_segments,
        )

    def set_algorithm(self, algorithm, use_iterations=True, plot=None):
        """Select the algorithm whose next output is being generated."""
        del use_iterations, plot
        self.algorithm = algorithm

    def set_scenario(self, scenario, plot=False, iteration=None):
        """Retain the current environment and write only its initial scenario PDF."""
        del plot, iteration
        self.plot_manager.scenario = scenario
        if (
            self.scenario_filename is not None
            and self.algorithm == "scenario"
            and not self._scenario_written
        ):
            self._write("scenario", scenario, self.scenario_filename, algorithm="scenario")
            self._scenario_written = True

    def save(self, data_type, data, plot=False, algorithm=None, iteration=None):
        """Write a graph PDF with the filename specified by its algorithm/iteration."""
        del plot
        algorithm = algorithm or self.algorithm
        if algorithm is None:
            raise RuntimeError("An algorithm must be selected before saving a figure.")

        # GVD also emits intermediate distance-map and Voronoi snapshots.  The
        # thesis set needs only its final roadmap graph for each layout.
        if algorithm in {"gvd", "dynamic_gvd"} and data_type != "graph":
            return {"pdf": None, "pkl": None}

        if algorithm == "dynamic_vgraph":
            filename = f"dynamic_vgraph_{self.iteration if iteration is None else iteration}.pdf"
        elif algorithm == "drm":
            filename = f"drm_{self.iteration if iteration is None else iteration}.pdf"
        elif algorithm == "vgraph":
            filename = "vgraph.pdf"
        elif algorithm == "prm":
            filename = "prm.pdf"
        elif algorithm == "gvd":
            filename = "gvd.pdf"
        elif algorithm == "dynamic_gvd":
            filename = f"dynamic_gvd_{self.iteration if iteration is None else iteration}.pdf"
        elif algorithm == "kings_graph":
            filename = "kings_graph.pdf"
        elif algorithm == "dynamic_kings_graph":
            filename = f"dynamic_kings_graph_{self.iteration if iteration is None else iteration}.pdf"
        elif algorithm == "scenario":
            # Scenario updates are represented by the following dynamic-graph
            # PDF, so do not write an additional environment-only figure.
            return {"pdf": None, "pkl": None}
        else:
            filename = f"{algorithm}.pdf"

        return self._write(
            data_type,
            data,
            filename,
            algorithm=algorithm,
            iteration=self.iteration if iteration is None else iteration,
        )

    def new_iteration(self, iteration=None):
        """Match the recorder interface used by ``DynamicVGraph``."""
        self.iteration = self.iteration + 1 if iteration is None else iteration

    def record_time(self, seconds, phase):
        """Ignore timings because this recorder writes thesis figures only."""
        del seconds, phase

    def save_named(self, data_type, data, filename, *, algorithm=None, iteration=None):
        """Write one explicitly named, PDF-only thesis figure."""
        algorithm = algorithm or self.algorithm
        if algorithm is None:
            raise RuntimeError("An algorithm must be selected before saving a figure.")
        return self._write(
            data_type,
            data,
            filename,
            algorithm=algorithm,
            iteration=self.iteration if iteration is None else iteration,
        )

    def _write(self, data_type, data, filename, *, algorithm, iteration=0):
        path = RUN_DIRECTORY / filename
        self.plot_manager.plot(
            data_type=data_type,
            data=data,
            path=path,
            title_info={"algorithm": algorithm, "iteration": iteration},
        )
        # The algorithms use the historical ``pkl`` key for their progress
        # message.  Returning the PDF path keeps that progress output useful.
        return {"pdf": path, "pkl": path}


def _recorder(scenario_filename, *, plot_buffer_segments=None):
    """Create the PDF-only recorder used by a single thesis-figure set."""
    return ThesisFigureRecorder(
        scenario_filename,
        plot_buffer_segments=plot_buffer_segments,
    )


def _generate_scenario(recorder, **scenario_parameters):
    """Generate and record a scenario, failing clearly if it cannot be made."""
    recorder.set_algorithm("scenario", use_iterations=False, plot=True)
    scenario = Scenario(recorder=recorder, print_progress=PRINT_PROGRESS, **scenario_parameters)
    if not scenario.generate():
        raise RuntimeError("Thesis scenario generation did not place every requested obstacle.")
    return scenario


def _large_layout_parameters(rng, *, with_aisles):
    """Return the requested 30 x 60 scenario parameters.

    Wall clearance is intentionally retained in both variants.  The only
    distinction between the two figures is whether the ordered aisles exist.
    """
    height = 20
    width = 30

    return {
        "rng": rng,
        "height": height,
        "width": width,
        "occupancy_rate": 0.3,
        "sigma_coarse": 3,
        "sigma_fine": 1,
        "noisy_aisles_num": 0,
        "noisy_aisle_sigma": 1,
        "ordered_aisle_widths": (2, 1) if with_aisles else (0, 0),
        "ordered_aisle_spacing": (8, 2) if with_aisles else (0, 0),
        "ordered_aisle_offset": (0, 0),
        "wall_clearance": (2, 2, 5, 3) if with_aisles else (0, 0, 0, 0),
        "interaction_points": {
            (int(height * 0.1), 0),
            (int(height * 0.95), int(width * 0.55)),
            (int(height * 0.95), int(width * 0.67)),
            (int(height * 0.95), int(width * 0.79)),
        },
        "interaction_points_radius": 3,
        "simple_obstacle_rate": 1,
        "robot_radius": 0.1,
    }


def generate_large_scenario_figures():
    """Generate the ordered-aisle and no-aisle 30 x 60 scenario figures."""
    for figure_name, with_aisles in (
        ("scenario_with_aisles", True),
        ("scenario_without_aisles", False),
    ):
        recorder = _recorder(f"{figure_name}.pdf")
        _generate_scenario(
            recorder,
            **_large_layout_parameters(
                np.random.default_rng(SCENARIO_SEED),
                with_aisles=with_aisles,
            ),
        )


def _roadmap_scenario_parameters(rng, height, width, occupancy_rate, interaction_points):
    """Return the common compact scenario configuration for roadmap figures."""
    return {
        "rng": rng,
        "height": height,
        "width": width,
        "occupancy_rate": occupancy_rate,
        "sigma_coarse": 3,
        "sigma_fine": 1,
        "noisy_aisles_num": 0,
        "noisy_aisle_sigma": 1,
        "interaction_points": interaction_points,
        "interaction_points_radius": 3,
        "simple_obstacle_rate": 0.1,
        "robot_radius": 0.3,
    }


def generate_vgraph_figure():
    """Generate the requested 15 x 20 static visibility-graph figure."""
    height = 20
    width = 30
    rng = np.random.default_rng(VGRAPH_SEED)
    recorder = _recorder(None)
    scenario = _generate_scenario(
        recorder,
        **_roadmap_scenario_parameters(
            rng,
            height,
            width,
            occupancy_rate=0.3,
            interaction_points=_lower_left_interaction_points(height, width),
        ),
    )

    access_points, interaction_points = continuous_special_points(
        scenario.access_points,
        scenario.interaction_points,
        scenario.robot_radius,
    )
    recorder.set_algorithm("vgraph", use_iterations=False, plot=True)
    graph = VGraph(
        width=width,
        height=height,
        occupancy=scenario.occupancy,
        robot_radius=scenario.robot_radius,
        access_points=access_points,
        interaction_points=interaction_points,
        num_buffer_segments=0,
        recorder=recorder,
        print_progress=PRINT_PROGRESS,
    )
    graph.create_visibility_graph()
    graph.record_graph()


def _apply_dynamic_update(graph, scenario):
    """Apply one recorded scenario update to the dynamic visibility graph."""
    access_points, interaction_points = continuous_special_points(
        scenario.access_points,
        scenario.interaction_points,
        scenario.robot_radius,
    )
    graph.apply_layout_updates(
        scenario.consume_updates(),
        new_access_points=access_points,
        new_interaction_points=interaction_points,
    )


def generate_dynamic_vgraph_figures():
    """Generate the initial dynamic graph and one updated iteration."""
    height = 20
    width = 30
    rng = np.random.default_rng(DYNAMIC_VGRAPH_SEED)
    recorder = _recorder(None, plot_buffer_segments=0)
    scenario = _generate_scenario(
        recorder,
        **_roadmap_scenario_parameters(
            rng,
            height,
            width,
            occupancy_rate=0.4,
            interaction_points=_dynamic_vgraph_test_interaction_points(height, width),
        ),
    )

    access_points, interaction_points = continuous_special_points(
        scenario.access_points,
        scenario.interaction_points,
        scenario.robot_radius,
    )
    recorder.set_algorithm("dynamic_vgraph", use_iterations=True, plot=True)
    graph = DynamicVGraph(
        height=height,
        width=width,
        occupancy=scenario.occupancy,
        robot_radius=scenario.robot_radius,
        access_points=access_points,
        interaction_points=interaction_points,
        recorder=recorder,
        print_progress=PRINT_PROGRESS,
    )
    graph.preprocessing()
    graph.record_graph(phase="preprocessing")

    scenario._add_obstacle(cell=(3, 5), type=2)
    scenario._remove_obstacle(cell=(1, 4))
    scenario.update_access_points()
    scenario._record_scenario()
    _apply_dynamic_update(graph, scenario)


def _interior_cells(scenario, occupied):
    """Return reproducibly ordered, non-interaction-point interior cells of the requested type."""
    candidates = [
        (int(row), int(column))
        for row, column in np.argwhere(scenario.occupancy != 0 if occupied else scenario.occupancy == 0)
        if 0 < row < scenario.height - 1
        and 0 < column < scenario.width - 1
        and (int(row), int(column)) not in scenario.interaction_points
    ]
    # The coprime multipliers spread consecutive edits over the compact layout
    # instead of concentrating them in row-major order.
    return sorted(
        candidates,
        key=lambda cell: ((cell[0] * 11 + cell[1] * 7) % (scenario.height * scenario.width), cell),
    )


def _apply_illustrative_layout_changes(scenario, *, count):
    """Add and remove several obstacles for a clearly changed second layout."""
    removable_cells = _interior_cells(scenario, occupied=True)

    additions = 0
    for cell in _interior_cells(scenario, occupied=False):
        obstacle_type = 1 if additions % 2 == 0 else 2
        if scenario._add_obstacle(cell, type=obstacle_type):
            additions += 1
            if additions == count:
                break

    removals = 0
    for cell in removable_cells:
        if scenario._remove_obstacle(cell):
            removals += 1
            if removals == count:
                break

    if additions != count or removals != count:
        raise RuntimeError("Could not apply every requested illustrative layout edit.")

    scenario.update_access_points()
    scenario._record_scenario()


def _apply_drm_updates(graph, scenario):
    """Update DRM with the obstacle and special-point changes stored by Scenario."""
    access_points, interaction_points = continuous_special_points(
        scenario.access_points,
        scenario.interaction_points,
        scenario.robot_radius,
    )
    graph.apply_layout_updates(
        scenario.consume_updates(),
        new_access_points=access_points,
        new_interaction_points=interaction_points,
    )


def generate_drm_figures():
    """Generate the initial and manually changed 10 x 15 dynamic-roadmap figures."""
    height = 20
    width = 30
    rng = np.random.default_rng(DRM_SEED)
    recorder = _recorder(None)
    scenario = _generate_scenario(
        recorder,
        **_roadmap_scenario_parameters(
            rng,
            height,
            width,
            occupancy_rate=0.3,
            interaction_points=_lower_left_interaction_points(height, width),
        ),
    )

    access_points, interaction_points = continuous_special_points(
        scenario.access_points,
        scenario.interaction_points,
        scenario.robot_radius,
    )
    recorder.set_algorithm("drm", use_iterations=True, plot=True)
    graph = DRM(
        rng=rng,
        occupancy=scenario.occupancy,
        access_points=access_points,
        interaction_points=interaction_points,
        height=height,
        width=width,
        max_dist=5,
        max_neighbors=10,
        robot_radius=scenario.robot_radius,
        num_buffer_segments=5,
        recorder=recorder,
        print_progress=PRINT_PROGRESS,
    )
    graph.preprocessing_step(min_num_vertices=DRM_MIN_NUM_VERTICES)

    _apply_illustrative_layout_changes(scenario, count=DRM_LAYOUT_CHANGE_COUNT)
    _apply_drm_updates(graph, scenario)


def _build_gvd(rng, scenario, recorder):
    """Create a GVD using the scenario's grid-coordinate special points."""
    return GVD(
        rng=rng,
        occupancy_grid=scenario.occupancy,
        scale_factor=1,
        access_points=scenario.access_points,
        interaction_points=scenario.interaction_points,
        robot_radius=scenario.robot_radius,
        recorder=recorder,
        print_progress=PRINT_PROGRESS,
    )


def _gvd_stage_data(gvd, **stage_data):
    """Return a self-contained plot payload for one static GVD stage."""
    return {**gvd._plot_context(), **stage_data}


def _gvd_roadmap_without_special_connections(roadmap):
    """Return the skeleton-derived roadmap before special-point links are added."""
    special_vertex_ids = {
        vertex_id
        for point_vertices in roadmap["special_point_vertices"].values()
        for vertex_id in point_vertices.values()
    }
    return {
        "vertices": {
            vertex_id: position
            for vertex_id, position in roadmap["vertices"].items()
            if vertex_id not in special_vertex_ids
        },
        "edges": {
            edge for edge in roadmap["edges"]
            if special_vertex_ids.isdisjoint(edge)
        },
        "special_point_vertices": {"access": {}, "interaction": {}},
    }


def generate_gvd_figure():
    """Generate the static GVD construction sequence for the thesis."""
    height = 20
    width = 30
    rng = np.random.default_rng(GVD_SEED)
    recorder = _recorder(None)
    scenario = _generate_scenario(
        recorder,
        **_roadmap_scenario_parameters(
            rng,
            height,
            width,
            occupancy_rate=0.25,
            interaction_points=_lower_left_interaction_points(height, width),
        ),
    )

    recorder.set_algorithm("gvd", use_iterations=False, plot=True)
    graph = _build_gvd(rng, scenario, recorder)
    gvd_recorder = graph.recorder
    graph.recorder = None
    try:
        graph.computeDistanceMap()
    finally:
        graph.recorder = gvd_recorder

    recorder.save_named(
        "distance_map",
        _gvd_stage_data(graph, D=graph.D, obst=graph.obst),
        "gvd_distance_map.pdf",
    )
    recorder.save_named(
        "voronoi_skeleton",
        _gvd_stage_data(graph, voro=graph.voro),
        "gvd_voronoi_skeleton.pdf",
    )

    # ``computeDistanceMap`` already builds the completed graph. Render both
    # the pre-connection and completed forms explicitly below.
    recorder.save_named(
        "graph",
        _gvd_stage_data(
            graph,
            graph=_gvd_roadmap_without_special_connections(graph.graph),
            voro=graph.voro,
        ),
        "gvd_graph.pdf",
    )
    recorder.save_named(
        "graph",
        _gvd_stage_data(graph, graph=graph.graph, voro=graph.voro),
        "gvd_roadmap.pdf",
    )


def _apply_gvd_update(graph, scenario):
    """Apply the pending scenario change and update the GVD roadmap."""
    graph.apply_layout_updates(
        scenario.consume_updates(),
        new_access_points=scenario.access_points,
        new_interaction_points=scenario.interaction_points,
    )
    graph.updateDistanceMap()


def _add_gvd_l_shape_above_interaction_points(scenario, *, distance=4):
    """Queue a three-cell unit-load L shape above the lower interaction points.

    The cells are deliberately added to the scenario's existing pending updates.
    The subsequent GVD rebuild therefore treats this cluster and the illustrative
    layout edits as one coupled first update.
    """
    interaction_rows, interaction_columns = zip(*scenario.interaction_points)
    anchor_row = int(max(interaction_rows)) - distance
    anchor_column = int(np.rint(np.median(interaction_columns)))
    l_shape_cells = (
        (anchor_row - 1, anchor_column),
        (anchor_row, anchor_column),
        (anchor_row, anchor_column + 1),
    )

    if any(
        not (0 <= row < scenario.height and 0 <= column < scenario.width)
        or scenario.occupancy[row, column] != 0
        for row, column in l_shape_cells
    ):
        raise RuntimeError("Could not place the GVD L-shaped obstacle cluster.")

    for cell in l_shape_cells:
        if not scenario._add_obstacle(cell, type=2):
            raise RuntimeError("Could not add a GVD L-shaped obstacle cell.")
    scenario.update_access_points(l_shape_cells)


def generate_dynamic_gvd_figures():
    """Generate the initial and changed-layout GVD roadmap figures."""
    height = 20
    width = 30
    rng = np.random.default_rng(DYNAMIC_GVD_SEED)
    recorder = _recorder(None)
    scenario = _generate_scenario(
        recorder,
        **_roadmap_scenario_parameters(
            rng,
            height,
            width,
            occupancy_rate=0.25,
            interaction_points=_lower_left_interaction_points(height, width),
        ),
    )

    recorder.set_algorithm("dynamic_gvd", use_iterations=True, plot=True)
    graph = _build_gvd(rng, scenario, recorder)
    graph.computeDistanceMap()

    _apply_illustrative_layout_changes(
        scenario,
        count=ILLUSTRATIVE_LAYOUT_CHANGE_COUNT,
    )
    _add_gvd_l_shape_above_interaction_points(scenario)
    _apply_gvd_update(graph, scenario)


def _build_kings_graph(scenario, recorder):
    """Create the King's-grid roadmap for the current scenario layout."""
    return IncrementalKingsGraph(
        occupancy=scenario.occupancy,
        access_points=scenario.access_points,
        interaction_points=scenario.interaction_points,
        bounds=(0, 0, scenario.height, scenario.width),
        robot_radius=scenario.robot_radius,
        recorder=recorder,
        print_progress=PRINT_PROGRESS,
    )


def generate_kings_graph_figure():
    """Generate the initial, static King's-graph figure."""
    height = 20
    width = 30
    rng = np.random.default_rng(KINGS_GRAPH_SEED)
    recorder = _recorder(None)
    scenario = _generate_scenario(
        recorder,
        **_roadmap_scenario_parameters(
            rng,
            height,
            width,
            occupancy_rate=0.6,
            interaction_points=_lower_left_interaction_points(height, width),
        ),
    )

    recorder.set_algorithm("kings_graph", use_iterations=False, plot=True)
    graph = _build_kings_graph(scenario, recorder)
    graph.build_graph()
    graph.record_graph()


def _apply_kings_graph_update(graph, scenario):
    """Apply the pending scenario change and save the updated King's graph."""
    graph.apply_layout_updates_and_update_graph(
        scenario.consume_updates(),
        scenario.access_points,
        scenario.interaction_points,
    )
    graph.record_graph()


def generate_dynamic_kings_graph_figures():
    """Generate the initial and changed-layout King's-graph figures."""
    height = 20
    width = 30
    rng = np.random.default_rng(DYNAMIC_KINGS_GRAPH_SEED)
    recorder = _recorder(None)
    scenario = _generate_scenario(
        recorder,
        **_roadmap_scenario_parameters(
            rng,
            height,
            width,
            occupancy_rate=0.6,
            interaction_points=_lower_left_interaction_points(height, width),
        ),
    )

    recorder.set_algorithm("dynamic_kings_graph", use_iterations=True, plot=True)
    graph = _build_kings_graph(scenario, recorder)
    graph.build_graph()
    graph.record_graph()

    _apply_illustrative_layout_changes(
        scenario,
        count=ILLUSTRATIVE_LAYOUT_CHANGE_COUNT,
    )
    _apply_kings_graph_update(graph, scenario)


def generate_prm_figure():
    """Generate a PRM on the same compact scenario used for the VGraph."""
    height = 20
    width = 30
    rng = np.random.default_rng(VGRAPH_SEED)
    recorder = _recorder(None)
    scenario = _generate_scenario(
        recorder,
        **_roadmap_scenario_parameters(
            rng,
            height,
            width,
            occupancy_rate=0.3,
            interaction_points=_lower_left_interaction_points(height, width),
        ),
    )

    access_points, interaction_points = continuous_special_points(
        scenario.access_points,
        scenario.interaction_points,
        scenario.robot_radius,
    )
    recorder.set_algorithm("prm", use_iterations=False, plot=True)
    graph = PRM(
        rng=rng,
        occupancy=scenario.occupancy,
        access_points=access_points,
        interaction_points=interaction_points,
        height=height,
        width=width,
        eps=0.05,
        max_dist=5,
        max_neighbors=10,
        robot_radius=scenario.robot_radius,
        num_buffer_segments=5,
        recorder=recorder,
        print_progress=PRINT_PROGRESS,
    )
    graph.construction_step(min_num_vertices=PRM_MIN_NUM_VERTICES, avoid_cycles=False)
    graph.record_graph()


def main():
    """Generate every thesis figure in a deterministic order."""
    print(f"Saving thesis figures to {RUN_DIRECTORY}")
    generate_large_scenario_figures()
    generate_vgraph_figure()
    generate_dynamic_vgraph_figures()
    generate_drm_figures()
    generate_prm_figure()
    generate_gvd_figure()
    generate_dynamic_gvd_figures()
    generate_kings_graph_figure()
    generate_dynamic_kings_graph_figures()


if __name__ == "__main__":
    main()
