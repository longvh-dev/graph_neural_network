import json
import os
from pprint import pprint

import nltk
import torch
import networkx as nx
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity


def encode_text_to_vectors(text, model_name = "bert-base-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    tokens = tokenizer(text, return_tensors = "pt", padding = True,
                       truncation = True)
    with torch.no_grad():
        outputs = model(**tokens)
    vector = outputs.last_hidden_state.mean(dim = 1).squeeze().numpy()
    return vector


class Node:
    def __init__(self, utterance, speaker):
        self.utterance = utterance
        self.encoded_vector = encode_text_to_vectors(utterance)
        self.speaker = speaker
        # self.strategy

    def get_speaker(self):
        return self.speaker

    def get_vector(self):
        return self.encoded_vector

    def get_utterance(self):
        return self.utterance


class Graph:
    def __init__(self, conversation,
                 similarity_threshold = 0.8, sliding_window = 3):
        self.conversation = conversation
        self.similarity_threshold = similarity_threshold
        self.nodes = self.build_nodes()
        # self.speaker = self.get_speaker()
        self.sliding_window = sliding_window

        if not os.path.exists("graph.edgelist"):
            self.build_starter_graph()

        # graph get from graph.edgelist
        with open('graph.edgelist', 'r') as f:
            self.G = nx.read_weighted_edgelist(f)
        self.build_graph()

    def build_nodes(self):
        nodes = []
        for turn in self.conversation['dialog']:
            dialogue = turn['content']
            speaker = turn['speaker']
            # utterances = nltk.sent_tokenize(dialogue)
            nodes.append(Node(dialogue, speaker))

        print(nodes[i].get_utterance() for i in range(len(nodes)))
        return nodes

    def build_starter_graph(self):
        G = nx.Graph()
        for i, node_i in enumerate(self.nodes):
            G.add_node(i, vector = node_i.get_vector())
        for i in range(len(self.nodes)):
            for j in range(i + 1, len(self.nodes)):
                # sim = cosine_similarity([self.nodes[i].get_vector()],
                #                         [self.nodes[j].get_vector()])[0, 0]
                if self.nodes[i].get_speaker() == self.nodes[j].get_speaker():
                    if self.nodes[i].get_speaker() == 'seeker':
                        G.add_edge(i, j, weight = 0)
                    else:
                        G.add_edge(i, j, weight = 0)
                elif abs(i - j) <= self.sliding_window:
                    G.add_edge(i, j, weight = 0)
                else:
                    G.add_edge(i, j, weight = 0)

        # export to json
        nx.write_weighted_edgelist(G, "graph.edgelist")
        # return G

    def build_graph(self):
        # set new weight of each node by cosin similarity
        for i in range(len(self.nodes)):
            for j in range(i + 1, len(self.nodes)):
                sim = cosine_similarity([self.nodes[i].get_vector()],
                                        [self.nodes[j].get_vector()])[0, 0]
                if sim >= self.similarity_threshold:
                    self.G.add_edge(i, j, weight = sim)
                else:
                    self.G.add_edge(i, j, weight = 0)
        nx.write_weighted_edgelist(self.G, "graph.edgelist")

    def plot_graph(self):
        edges_with_weight = [(u, v) for u, v, d in
                             self.G.edges(data = True) if d['weight'] != 0]

        # Create a subgraph with only edges that have non-zero weight
        subgraph = self.G.edge_subgraph(edges_with_weight)

        # Adjust the layout for better visualization
        pos = nx.kamada_kawai_layout(subgraph, scale = 5)

        for u, v, d in subgraph.edges(data = True):
            nx.draw_networkx_edges(subgraph, pos, edgelist = [(u, v)],
                                   width = d['weight'])

        # Plot the subgraph
        nx.draw(subgraph, pos, with_labels = True, font_weight = 'bold',
                node_size = 500, node_color = 'skyblue',
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
