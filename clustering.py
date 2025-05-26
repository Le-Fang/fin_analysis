import networkx as nx
import matplotlib.pyplot as plt
import json
import community as community_louvain

class Clustering:
    """
    A class to read graph data from a JSON file, create a NetworkX graph,
    cluster the graph using the Louvain method, and visualize the clusters.
    """
    def __init__(self, file_path="data/graph.json"):
        self.file_path = file_path
        self.graph_data = None
    
    def read_json(self):
        """Read articles from the JSON file."""
        try:
            with open(self.file_path, 'r', encoding='utf-8-sig') as f:
                self.graph_data = json.load(f)
        except FileNotFoundError:
            print(f"File {self.file_path} not found.")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from file {self.file_path}.")
            print(f"Error: {e}")
            
        return self.graph_data

    def create_graph(self):
        """
        Create a NetworkX graph from the JSON data.
        :return: A NetworkX graph object.
        """
        G = nx.Graph()

        # Add nodes
        for node in self.graph_data["nodes"]:
            G.add_node(node["id"], labels=node["labels"], **node["properties"])

        # Add edges
        for rel in self.graph_data["relationships"]:
            if rel["start"] in G and rel["end"] in G:
                # Add edge with properties
                G.add_edge(rel["start"], rel["end"], type=rel["type"], name=rel['properties']['type'])

        return G
    
    def cluster_graph(self, G):
        """
        Cluster the graph using the Louvain method.
        :param G: The graph to cluster.
        :return: A dictionary mapping nodes to their cluster labels.
        """
        partition = community_louvain.best_partition(G)
        return partition

    def visualize_clusters(self, G, partition):
        """
        Visualize the clusters in the graph.
        :param G: The graph to visualize.
        :param partition: The partition of nodes into clusters.
        """
        pos = nx.spring_layout(G)
        plt.figure(figsize=(12, 8))

        # Draw nodes with colors based on their cluster
        cmap = plt.get_cmap('viridis', max(partition.values()) + 1)
        nx.draw_networkx_nodes(G, pos, node_color=list(partition.values()), cmap=cmap, node_size=50)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, alpha=0.5)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=8)

        plt.title("Graph Clustering Visualization")
        plt.show()
    
    def process(self):
        """Process the graph data and visualize clusters."""
        self.read_json()
        G = self.create_graph()
        partition = self.cluster_graph(G)
        self.visualize_clusters(G, partition)
        return partition
    
if __name__ == "__main__":
    clustering = Clustering()
    clustering.process()
    
    
