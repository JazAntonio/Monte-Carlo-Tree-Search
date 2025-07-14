import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from pgmpy.models import BayesianModel
from pgmpy.estimators import HillClimbSearch, BicScore, MaximumLikelihoodEstimator
import random
import math

class BayesianNetworkNode:
    def __init__(self, structure):
        self.children = []
        self.parent = None
        self.structure = structure
        self.wins = 0
        self.visits = 0

    def select_child(self):
        best_score = -1
        best_child = None
        for child in self.children:
            if child.visits == 0:
                return child
            ucb_score = (child.wins / child.visits) + math.sqrt(2 * math.log(self.visits) / child.visits)
            if ucb_score > best_score:
                best_score = ucb_score
                best_child = child
        return best_child

    def expand(self, possible_edges):
        new_edge = random.choice(possible_edges)
        new_structure = self.structure + [new_edge]
        new_node = BayesianNetworkNode(new_structure)
        new_node.parent = self
        self.children.append(new_node)

    def simulate(self, train_data, test_data):
        model = BayesianModel(self.structure)
        model.fit(train_data, estimator=MaximumLikelihoodEstimator)
        # Evaluar el modelo aquí (por ejemplo, usando validación cruzada)
        score = BicScore(train_data).score(model)
        return score

    def update(self, result):
        self.wins += result
        self.visits += 1

def mcts(root, train_data, test_data, iterations=1000):
    possible_edges = [('sepal length (cm)', 'sepal width (cm)'), ('petal length (cm)', 'petal width (cm)'), ...]  # Define todas las posibles aristas
    for _ in range(iterations):
        node = root
        while node.children:
            node = node.select_child()
        if node.visits < 10:  # Un umbral para decidir cuándo expandir
            node.expand(possible_edges)
        result = node.simulate(train_data, test_data)
        while node:
            node.update(result)
            node = node.parent
    return max(root.children, key=lambda c: c.wins / c.visits).structure

# Cargar el conjunto de datos de Iris y dividirlo en entrenamiento y prueba
iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Inicializar y ejecutar MCTS
root_structure = []
root = BayesianNetworkNode(root_structure)
best_structure = mcts(root, train_data, test_data)
print("Mejor estructura encontrada:", best_structure)
