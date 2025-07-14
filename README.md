## 🔧 Development Overview

The algorithm constructs a tree structure to explore graph states using **Monte Carlo Tree Search (MCTS)**. Each tree node stores:

- An adjacency matrix (current graph state)
- A list of child nodes
- A reference to its parent
- A cumulative value
- The number of visits

The MCTS algorithm proceeds as follows:

1. **Selection**: Starting from the root, the child with the highest **UCT** value is selected recursively until a leaf node is reached. Nodes with 0 visits are prioritized to ensure exploration.
2. **Expansion**: From the selected node, new child nodes are created by removing edges from the graph. Only those resulting in **connected graphs** are added.
3. **Simulation**: A child node is chosen (always the first, if available), and edges are randomly removed until the graph becomes disconnected. This defines the stopping condition.
4. **Backpropagation**: The weight `W` of the final graph is calculated and propagated back to the root by updating the value and visit count of each ancestor node.
5. **Iteration**: Steps 1–4 are repeated for `N = 100` iterations. The best child (based on value) is selected, and the tree is discarded. This process is repeated `M` times to move through graph states optimally.

> 💡 Alternative selection criteria can include the number of visits or a combination of value and visits, as discussed in the literature.
