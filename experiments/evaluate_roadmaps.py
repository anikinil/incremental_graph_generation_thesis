
import json
from pathlib import Path
import pickle
import sys

import networkx as nx
import numpy as np
from networkx.algorithms.connectivity import local_edge_connectivity


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
        "outbound_length": outbound_length,
        "inbound_length": inbound_length,
        "task_route_length": outbound_length + inbound_length
    }
    
def _evaluate_route_redundancy(roadmap, source, access, destination):
    graph = roadmap["graph"]
    return {
        "outbound_edge_connectivity": local_edge_connectivity(graph, source, access),
        "inbound_edge_connectivity": local_edge_connectivity(graph, access, destination),
    }


def evaluate_roadmap(roadmap, number_of_tasks=100, seed=41):
    """Evaluate sampled tasks and return one result dictionary per task."""
    rng = np.random.default_rng(seed)
    results = []

    for _ in range(number_of_tasks):
        source, access, destination = _sample_two_leg_task(roadmap, rng)
        result = {
            "task": {
                "source_interaction": source["position"],
                "access": access["position"],
                "destination_interaction": destination["position"]
            },
            "path_length": _evaluate_path_length(roadmap, source, access, destination),
            "route_redundancy": _evaluate_route_redundancy(roadmap, source["node"], access["node"], destination["node"]),
        }
        
        results.append(result)
        
    return results



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

    if data_type == "graph":  # gvd graph
        saved_graph = data["graph"]
        for node, position in saved_graph["nodes"].items():
            graph.add_node(node, pos=tuple(position))
        _add_weighted_edges(graph, saved_graph["edges"])

    elif isinstance(data["nodes"], dict):  # visibility graph and king's graph
        for node, attributes in data["nodes"].items():
            graph.add_node(node, pos=tuple(attributes["pos"]))
        _add_weighted_edges(graph, data["edges"])

    elif data_type == "drm_graph": # drm
        for index, position in enumerate(data["nodes"]):
            if data["node_present"][index] and data["node_active"][index]:
                graph.add_node(index, pos=tuple(position))
        active_edges = [edge for edge in data["edges"] if data["edge_active"][edge]]
        _add_weighted_edges(graph, active_edges)

    elif data_type == "dynamic_vgraph": # dynamic vgraph
        for node in data["nodes"]:
            graph.add_node(node, pos=tuple(node))
        active_edges = [edge for edge in data["edges"] if data["edge_active"][edge]]
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


def _load_roadmaps(roadmap_run):
    """Load every saved roadmap graph from a roadmap-generation run."""
    graph_data_types = {
        "vgraph",
        "dynamic_vgraph",
        "prm_graph",
        "drm_graph",
        "incremental_kings_grid",
        "graph",
    }
    roadmaps = []

    for stage_dir in sorted(roadmap_run.glob("stage_*")):
        for algorithm_dir in sorted(path for path in stage_dir.iterdir() if path.is_dir()):
            records = []
            for path in algorithm_dir.glob("data/*.pkl"):
                with path.open("rb") as file:
                    record = pickle.load(file)
                if record["data_type"] in graph_data_types:
                    records.append(record)
                    
            record = records[-1]
            graph = _build_graph(record)
            access_nodes, interaction_nodes = _special_nodes(record, graph)
            roadmaps.append({
                "name": f"{stage_dir.name}/{algorithm_dir.name}",
                "stage": stage_dir.name,
                "algorithm": algorithm_dir.name,
                "graph": graph,
                "access_points": _special_points(graph, access_nodes),
                "interaction_points": _special_points(graph, interaction_nodes),
            })

    return roadmaps


def _summary(results):
    """Create a small per-roadmap summary for the two current metrics."""
    route_lengths = [result["path_length"]["task_route_length"] for result in results]
    redundancy_values = []
    
    for result in results:
        for value in result["route_redundancy"].values():
            redundancy_values.append(value)
    
    return {
        "mean_task_route_length": float(np.mean(route_lengths)),
        "mean_route_redundancy": float(np.mean(redundancy_values)),
        "minimum_route_redundancy": int(min(redundancy_values)),
    }


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
    roadmaps = _load_roadmaps(roadmap_run)

    output = {}
    for roadmap in roadmaps:
        results = evaluate_roadmap(roadmap)
        output[roadmap["name"]] = {
            "summary": _summary(results),
            "tasks": results,
        }
        print(roadmap["name"], output[roadmap["name"]]["summary"])

    output_path = roadmap_run / "evaluation_results.json"
    output_path.write_text(json.dumps(_json_value(output), indent=2) + "\n")
    print(f"Saved evaluation results to {output_path}")


if __name__ == "__main__":
    main()
