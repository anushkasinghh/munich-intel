"""CLI entrypoint: python scripts/build_graph.py

Builds the networkx graph from data/entities/ + companies.yaml (VISION.md step 4)
and prints a summary — the sanity check before trusting the graph for anything else.
Does not scrape, extract, or persist a snapshot; see VISION.md for those steps.
"""

import sys
from collections import Counter
from pathlib import Path

from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from munich_intel.graph import build_graph

console = Console()


def main() -> None:
    graph = build_graph()

    node_kinds = Counter(kind for _, kind in graph.nodes(data="kind"))
    edge_types = Counter(edge_type for _, _, edge_type in graph.edges(data="edge_type"))

    nodes_table = Table(title="Nodes", show_lines=False)
    nodes_table.add_column("Kind", style="cyan")
    nodes_table.add_column("Count", justify="right")
    for kind, count in node_kinds.most_common():
        nodes_table.add_row(kind, str(count))
    nodes_table.add_row("[bold]total[/bold]", f"[bold]{graph.number_of_nodes()}[/bold]")

    edges_table = Table(title="Edges", show_lines=False)
    edges_table.add_column("Type", style="cyan")
    edges_table.add_column("Count", justify="right")
    for edge_type, count in edge_types.most_common():
        edges_table.add_row(edge_type, str(count))
    edges_table.add_row("[bold]total[/bold]", f"[bold]{graph.number_of_edges()}[/bold]")

    top_table = Table(title="Top 5 companies by degree")
    top_table.add_column("Company", style="cyan")
    top_table.add_column("Degree", justify="right")
    companies = [(n, d) for n, d in graph.degree() if graph.nodes[n]["kind"] == "company"]
    for node_id, degree in sorted(companies, key=lambda x: x[1], reverse=True)[:5]:
        top_table.add_row(graph.nodes[node_id]["name"], str(degree))

    console.print(nodes_table)
    console.print(edges_table)
    console.print(top_table)


if __name__ == "__main__":
    main()
