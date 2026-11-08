
import json
from math import ceil
from pathlib import Path
import pickle
import sys

import networkx as nx
import numpy as np


from shapely.geometry import Point, box
from algorithms.geometry import buffered_obstacle_space


TASK_NUM = 40
SEED = 8599
CLEARANCE_SAMPLE_SPACING = 0.1 # clearance samples are sampled every 0.1 units along the roadmap edges

########## evaluation ###########

def _sample_two_leg_task(roadmap, rng):
    """Sample one interaction -> access -> interaction task."""
    interaction_points = roadmap["interaction_points"]
    access_points = roadmap["access_points"]

    source = interaction_points[rng.integers(len(interaction_points))]
    access = access_points[rng.integers(len(access_points))]

    # avoid sampling the same interaction point as source and destination, unless there is only one interaction point
    if len(interaction_points) == 1:
        destination = source
    else:
        destinations = [point for point in interaction_points if point != source]
        destination = destinations[rng.integers(len(destinations))]

    return source, access, destination

def _evaluate_path_length(roadmap, source, access, destination):
    graph = roadmap["graph"]
    source, access, destination = source["node"], access["node"], destination["node"]

    outbound_length = nx.shortest_path_length(graph, source, access, weight="weight")
    inbound_length = nx.shortest_path_length(graph, access, destination, weight="weight")

    return {
        "task_route_length": outbound_length + inbound_length
    }
    
def _evaluate_route_redundancy(roadmap, source, access, destination):
    graph = roadmap["graph"]

    outbound_path = nx.shortest_path(graph, source, access, weight="weight")
    inbound_path = nx.shortest_path(graph, access, destination, weight="weight")
    task_path = outbound_path[:-1] + inbound_path
    bridge_edges = set(nx.bridges(graph))

    task_length = 0
    non_bridge_length = 0
    for left, right in nx.utils.pairwise(task_path):
        edge_length = graph.edges[left, right]["weight"]
        task_length += edge_length
        if (left, right) not in bridge_edges and (right, left) not in bridge_edges:
            non_bridge_length += edge_length

    return {
        "non_bridge_path_percentage": non_bridge_length / task_length if task_length else 0
    }

def _evaluate_task_route_clearance(roadmap, source, access, destination):
    pass # TODO

def _calculate_free_area(roadmap):
    occupancy = roadmap["occupancy"]
    robot_radius = roadmap["robot_radius"]

    height, width = occupancy.shape
    r = robot_radius

    # warehouse area
    robot_centre_world = box(0, 0, width, height)
    # obstacles inflated by r (circle approximated by a polygon with 16 segments)
    inflated_obstacles = buffered_obstacle_space(occupancy, robot_radius=r, num_buffer_segments=16)
    # free space area
    free_area = robot_centre_world.difference(inflated_obstacles).area

    return free_area

def _evaluate_geometric_roadmap_density(roadmap):
    free_area = _calculate_free_area(roadmap)
    edges = roadmap["graph"].edges(data=True)
    total_length = sum(attributes["weight"] for _, _, attributes in edges)
    return total_length / free_area

def _evaluate_data_structure_density(roadmap):
    free_area = _calculate_free_area(roadmap)
    num_edges = roadmap["graph"].number_of_edges()
    num_nodes = roadmap["graph"].number_of_nodes()
    return (num_edges + num_nodes) / free_area


def evaluate_tasks(roadmap):
    """Evaluate the properties of a number of randomly sampled tasks."""
    rng = np.random.default_rng(SEED)
    results = []

    for _ in range(TASK_NUM):
        source, access, destination = _sample_two_leg_task(roadmap, rng)
        result = {
            "task": {
                "source_interaction": source["position"],
                "access": access["position"],
                "destination_interaction": destination["position"]
            },
            "path_length": _evaluate_path_length(roadmap, source, access, destination),
            # "clearance": _evaluate_task_route_clearance(roadmap, source, access, destination),
            "route_redundancy": _evaluate_route_redundancy(roadmap, source["node"], access["node"], destination["node"])
        }
        results.append(result)

    return results


def average_task_metrics(task_results):
    """Calculate the average metrics for a set of task results."""
    task_route_lengths = [result["path_length"]["task_route_length"] for result in task_results]
    # mean_clearances = [result["clearance"]["mean_clearance"] for result in task_results]
    non_bridge_path_percentages = [result["route_redundancy"]["non_bridge_path_percentage"] for result in task_results]

    return {
        "average_task_route_length": float(np.mean(task_route_lengths)),
        # "average_task_mean_clearance": float(np.mean(mean_clearances)),
        "average_non_bridge_path_percentage": float(np.mean(non_bridge_path_percentages))
    }


def evaluate_roadmap(roadmap):
    """Evaluate the non-task-specific roadmap properties."""
    return {
        "geometric_roadmap_density": _evaluate_geometric_roadmap_density(roadmap),
        "data_structure_density": _evaluate_data_structure_density(roadmap)
    }

########## build graph ###########

def _build_graph(record):
    """Convert one saved roadmap record into a NetworkX graph with edge lengths."""

    def _add_weighted_edges(graph, edges):
        """Add roadmap edges to the graph and use edge lengths as their weights."""
        for left, right in edges:
            left_position = np.asarray(graph.nodes[left]["pos"], dtype=float)
            right_position = np.asarray(graph.nodes[right]["pos"], dtype=float)
            graph.add_edge(left, right, weight=float(np.linalg.norm(right_position - left_position)))
        
    data = record["data"]
    data_type = record["data_type"]
    graph = nx.Graph()

    if data_type == "graph": # gvd graph
        saved_graph = data["graph"]
        scale_factor = float(data.get("scale_factor", 1))
        for node, position in saved_graph["nodes"].items():
            row, column = position
            # inverted order because the GVD uses (row, column) instead of (x, y) when saving graphs
            graph.add_node(node, pos=((column - 1) / scale_factor, (row - 1) / scale_factor))
        _add_weighted_edges(graph, saved_graph["edges"])

    elif isinstance(data["nodes"], dict): # visibility graph and king's graph
        for node, attributes in data["nodes"].items():
            graph.add_node(node, pos=tuple(attributes["pos"]))
        _add_weighted_edges(graph, data["edges"])

    elif data_type == "drm_graph": # drm
        for index, position in enumerate(data["nodes"]):
            if data["node_present"][index] and data["node_active"][index]:
                graph.add_node(index, pos=tuple(position))
        active_edges = [edge for edge in data["edges"] if data["edge_active"][edge]]
        _add_weighted_edges(graph, active_edges)

    elif data_type == "dynamic_vgraph": #roadmap_run dynamic vgraph
        active_edges = [edge for edge in data["edges"] if data["edge_active"][edge]]
        # add only nodes that are part of active edges to the graph
        active_nodes = {node for edge in active_edges for node in edge}
        for node in active_nodes:
            graph.add_node(node, pos=tuple(node))
        _add_weighted_edges(graph, active_edges)

    elif data_type == "prm_graph": # prm graph
        for index, position in enumerate(data["nodes"]):
            graph.add_node(index, pos=tuple(position))
        _add_weighted_edges(graph, data["edges"])

    return graph


def _special_nodes(record, graph):
    """Return the separately saved access and interaction graph nodes."""

    def _find_special_nodes(graph, points):
        """Find graph nodes at the separately saved special-point positions."""
        nodes = list(graph.nodes)
        positions = np.asarray([graph.nodes[node]["pos"] for node in nodes], dtype=float)
        special_nodes = []
        for point in points:
            position = np.asarray(point[:2], dtype=float)
            special_nodes.append(nodes[np.argmin(np.linalg.norm(positions - position, axis=1))])
        return special_nodes
    
    data = record["data"]

    graph_data = data["graph"] if record["data_type"] == "graph" else data
    if "special_point_nodes" in graph_data:
        mapping = graph_data["special_point_nodes"]
        return list(mapping["access"].values()), list(mapping["interaction"].values())

    access_nodes = _find_special_nodes(graph, data["access_points"])
    interaction_nodes = _find_special_nodes(graph, data["interaction_points"])
    return access_nodes, interaction_nodes


def _special_points(graph, special_nodes):
    """Return each special point's graph node and warehouse position."""
    special_points = []
    for special_node in special_nodes:
        position = tuple(graph.nodes[special_node]["pos"])
        special_points.append({
            "node": special_node,
            "position": position,
        })
    return special_points


########## loading and saving ###########


def _load_roadmap_data(roadmap_run):
    """Load every roadmap graph together with its scenario occupancy context."""
    graph_data_types = {
        "vgraph",
        "dynamic_vgraph",
        "prm_graph",
        "drm_graph",
        "incremental_kings_grid",
        "graph",
    }
    roadmaps = []
    scenario_run = roadmap_run.parent.parent
    robot_radius = None

    for stage_dir in sorted(roadmap_run.glob("stage_*")):
        scenario_path = scenario_run / stage_dir.name / "scenario" / "data" / "scenario.pkl"

        with scenario_path.open("rb") as file:
            scenario_record = pickle.load(file)

        scenario_data = scenario_record["data"]
        occupancy = scenario_data["occupancy"]
        if robot_radius is None: # get robot radius from the first stage
            robot_radius = float(scenario_data["robot_radius"])

        for algorithm_dir in sorted(path for path in stage_dir.iterdir() if path.is_dir()):
            records = []
            for path in algorithm_dir.glob("data/*.pkl"):
                with path.open("rb") as file:
                    record = pickle.load(file)
                if record["data_type"] in graph_data_types:
                    records.append(record)
            # sort records by iteration and timestamp, then take the last one
            record = max(records, key=lambda item: (item["iteration"], item["timestamp"]))
            graph = _build_graph(record)
            access_nodes, interaction_nodes = _special_nodes(record, graph)
            roadmaps.append({
                "name": f"{stage_dir.name}/{algorithm_dir.name}",
                "stage": stage_dir.name,
                "algorithm": algorithm_dir.name,
                "graph": graph,
                "occupancy": occupancy,
                "robot_radius": robot_radius,
                "access_points": _special_points(graph, access_nodes),
                "interaction_points": _special_points(graph, interaction_nodes),
            })

    return roadmaps

def _json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


########### main ###########

def main():
    roadmap_run = Path(sys.argv[1])
    roadmap_data = _load_roadmap_data(roadmap_run)

    output = {}
    for roadmap in roadmap_data:
        task_results = evaluate_tasks(roadmap)
        roadmap_results = evaluate_roadmap(roadmap)
        output[roadmap["name"]] = {
            "tasks": task_results,
            "task_averages": average_task_metrics(task_results),
            "roadmap": roadmap_results,
        }
        print(roadmap["name"])

    output_path = roadmap_run / "evaluation_results.json"
    output_path.write_text(json.dumps(_json_value(output), indent=2) + "\n")
    print(f"Saved evaluation results to {output_path}")


if __name__ == "__main__":
    main()
