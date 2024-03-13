import torch
import networkx as nx
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity


class Graph:
    def __init__(self, conversation, model_name = "bert-base-uncased",
                 similarity_threshold = 0.8):
        self.conversation = conversation
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.encoded_vectors = self.encode_text_to_vectors()
        self.G = self.build_graph_from_vectors()

    def encode_text_to_vectors(self):
        utterances = [utterance['content'] for utterance in
                      self.conversation['dialog']]
        encoded_vectors = []
        for text in utterances:
            tokens = self.tokenizer(text, return_tensors = "pt", padding = True,
                                    truncation = True)
            with torch.no_grad():
                outputs = self.model(**tokens)
            vector = outputs.last_hidden_state.mean(dim = 1).squeeze().numpy()
            encoded_vectors.append(vector)
        return encoded_vectors

    def build_graph_from_vectors(self, window_size = 3):
        G = nx.Graph()
        for i, vector_i in enumerate(self.encoded_vectors):
            G.add_node(i, vector = vector_i)

        for i in range(len(self.encoded_vectors)):
            for j in range(i + 1,
                           min(i + window_size + 1, len(self.encoded_vectors))):
                sim = cosine_similarity([self.encoded_vectors[i]],
                                        [self.encoded_vectors[j]])[0, 0]
                if i % 2 == 0 and j % 2 == 0:
                    G.add_edge(i, j, weight = 2)
                elif i % 2 == 1 and j % 2 == 1:
                    G.add_edge(i, j, weight = 3)
                else:
                    G.add_edge(i, j, weight = 1)
                # if sim > self.similarity_threshold:
                #     # .2 for each weight of the edge
                #     G.add_edge(i, j, weight = round(sim, 2))
                # if < threshold, set weight to 0
                # else:
                #     G.add_edge(i, j, weight = 0)
        return G

    def plot_graph(self):
        nodes_with_edges = [node for node, degree in
                            dict(self.G.degree()).items() if degree > 0]

        # Create a subgraph with only nodes that have edges
        subgraph = self.G.subgraph(nodes_with_edges)
        # subgraph = self.G

        # Adjust the layout for better visualization
        pos = nx.kamada_kawai_layout(subgraph, scale = 5)

        # Plot the subgraph
        nx.draw(subgraph, pos, with_labels = True, font_weight = 'bold',
                node_size = 700, node_color = 'skyblue',
                font_color = 'black', font_size = 10)
        edge_labels = nx.get_edge_attributes(subgraph, 'weight')
        nx.draw_networkx_edge_labels(subgraph, pos, edge_labels = edge_labels)
        plt.title("Conversation Graph")
        plt.show()

    def get_adjacency_matrix(self):
        # return a sparse matrix
        adjacency_matrix = nx.adjacency_matrix(self.G)
        return adjacency_matrix if adjacency_matrix.shape[
                                       0] > 0 else 'No edges found!q'
