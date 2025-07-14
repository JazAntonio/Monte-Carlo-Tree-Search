import random
from copy import deepcopy

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from pgmpy.estimators import BicScore
from pgmpy.models import BayesianModel
from pgmpy.sampling import BayesianModelSampling
from sklearn.datasets import load_iris


class BayesianNetworkMCST:
    def __init__(self, data):
        self.data = data
        self.network = BayesianModel()
        self.score = -np.inf

    def calculate_score(self):
        scorer = BicScore(self.data)
        score = scorer.score(self.network)
        return score

    # def modify_structure(self):
    #     nodes = list(self.data.columns)
    #     node_a, node_b = random.sample(nodes, 2)
    #     if self.network.has_edge(node_a, node_b):
    #         self.network.remove_edge(node_a, node_b)
    #     else:
    #         self.network.add_edge(node_a, node_b)

    def modify_structure(self):
        nodes = list(self.data.columns)
        node_a, node_b = random.sample(nodes, 2)

        if self.network.has_edge(node_a, node_b):
            self.network.remove_edge(node_a, node_b)
        elif not self.network.has_edge(node_b, node_a):
            temp_network = self.network.copy()
            temp_network.add_edge(node_a, node_b)
            if nx.is_directed_acyclic_graph(
                temp_network
            ):  # Verificando si sigue siendo un DAG
                self.network.add_edge(node_a, node_b)

    def simulate(self):
        sampling = BayesianModelSampling(self.network)
        sample = sampling.forward_sample(size=100)
        return sample


class Node:
    def __init__(self, state):
        self.state = state
        self.children = []
        self.parent = None
        self.visits = 0
        self.reward = 0

    def expand(self):
        new_state = deepcopy(self.state)
        new_state.modify_structure()
        child_node = Node(new_state)
        child_node.parent = self
        self.children.append(child_node)

    def is_fully_expanded(self):
        return len(self.children) > 10  # Ajustar este límite según sea necesario

    def best_child(self, c_param=1.4):
        choices_weights = [
            c.reward / c.visits
            + c_param * np.sqrt((2 * np.log(self.visits) / c.visits))
            if c.visits > 0
            else float("inf")  # Esto evitará la división por cero
            for c in self.children
        ]
        return self.children[np.argmax(choices_weights)]


class MCTS:
    def __init__(self, data):
        self.root = Node(BayesianNetworkMCST(data))

    def select(self):
        current_node = self.root
        while current_node.is_fully_expanded():
            current_node = current_node.best_child()
        return current_node

    def simulate(self, node):
        simulated_sample = node.state.simulate()
        # Aquí puedes implementar lógica para evaluar el rendimiento del modelo
        # Por ejemplo, utilizando el BIC score
        return node.state.calculate_score()

    def backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.reward += reward
            node = node.parent

    def search(self, iterations):
        for _ in range(iterations):
            selected_node = self.select()
            if not selected_node.is_fully_expanded():
                selected_node.expand()
            reward = self.simulate(selected_node)
            self.backpropagate(selected_node, reward)
        return self.root.best_child().state.network


def plot_bayesian_network(
    bn_model, node_size=2000, font_size=12, graph_layout="spring"
):
    """
    Visualiza una red bayesiana utilizando NetworkX y Matplotlib.

    :param bn_model: Modelo de red bayesiana (BayesianModel de pgmpy)
    :param node_size: Tamaño de los nodos en el gráfico
    :param font_size: Tamaño de la fuente para los nombres de los nodos
    :param graph_layout: Layout para el gráfico ('spring', 'circular', 'kamada_kawai', 'random')
    """
    G = nx.DiGraph()
    G.add_edges_from(bn_model.edges())

    if graph_layout == "spring":
        pos = nx.spring_layout(G)
    elif graph_layout == "circular":
        pos = nx.circular_layout(G)
    elif graph_layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    else:  # random layout
        pos = nx.random_layout(G)

    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=node_size,
        font_size=font_size,
        node_color="skyblue",
        edge_color="black",
        linewidths=1,
        font_weight="bold",
    )
    plt.title("Bayesian Network Graph")
    plt.show()


iris = load_iris()
iris_data = pd.DataFrame(iris.data, columns=iris.feature_names)

# Instanciando y ejecutando MCTS
mcts = MCTS(iris_data)
optimal_structure = mcts.search(iterations=10)

print("Optimal Structure:", optimal_structure.edges())
plot_bayesian_network(optimal_structure)
