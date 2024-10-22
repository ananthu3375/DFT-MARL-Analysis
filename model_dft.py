import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define the fault tree graph
G = nx.DiGraph()

# Add basic, intermediate, and top event nodes
basic_events = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
intermediate_events = ['G2', 'G3', 'C1', 'C2', 'CSP1', 'CSP2', 'M1', 'M2', 'M3', 'FDEP', 'PS']
top_event = 'G1'

# Add nodes to the graph
G.add_nodes_from(basic_events + intermediate_events + [top_event])

# Define edges based on the XML precedence relationships
edges = [
    ('A', 'PS'), ('B', 'PS'), ('PS', 'FDEP'),
    ('FDEP', 'C1'), ('FDEP', 'C2'),
    ('C', 'C1'), ('D', 'C1'),
    ('E', 'M1'), ('F', 'M1'),
    ('G', 'M3'), ('H', 'M3'),
    ('I', 'M2'), ('J', 'M2'),
    ('K', 'C2'), ('L', 'C2'),
    ('M1', 'CSP1'), ('M2', 'CSP2'),
    ('M3', 'CSP1'), ('M3', 'CSP2'),
    ('C1', 'G2'), ('CSP1', 'G2'),
    ('C2', 'G3'), ('CSP2', 'G3'),
    ('G2', 'G1'), ('G3', 'G1')
]

G.add_edges_from(edges)

# Define positions for nodes (adjusted for clarity)
pos = {
    'A': (0, 0), 'B': (1, 0), 'C': (2, 0), 'D': (3, 0), 'E': (4, 0), 'F': (5, 0),
    'G': (6, 0), 'H': (7, 0), 'I': (8, 0), 'J': (9, 0), 'K': (10, 0), 'L': (11, 0),
    'PS': (0.5, 1), 'FDEP': (3, 1), 'C1': (1, 2), 'C2': (9, 2),
    'M1': (4.5, 2), 'M2': (8.5, 2), 'M3': (6.5, 2),
    'CSP1': (3, 3), 'CSP2': (8, 3), 'G2': (4, 4), 'G3': (9, 4),
    'G1': (6.5, 5)
}

# Initialize all nodes as green
node_colors = {node: 'green' for node in G.nodes()}

# Action sequence (from the game)
action_sequence = [
    ("red_agent", "E"), ("blue_agent", "No Action"),
    ("red_agent", "H"), ("blue_agent", "No Action"),
    ("red_agent", "J"), ("blue_agent", "E"),
    ("red_agent", "C"), ("blue_agent", "C"),
    ("red_agent", "F"), ("blue_agent", "F"),
    ("red_agent", "C"), ("blue_agent", "C"),
    ("red_agent", "F"), ("blue_agent", "F"),
    ("red_agent", "C"), ("blue_agent", "C"),
    ("red_agent", "F"), ("blue_agent", "F"),
    ("red_agent", "C"), ("blue_agent", "C"),
    ("red_agent", "F"), ("blue_agent", "F"),
    ("red_agent", "L"), ("blue_agent", "No Action"),
    ("red_agent", "C"), ("blue_agent", "C"),
    ("red_agent", "F"), ("blue_agent", "F"),
    ("red_agent", "C"), ("blue_agent", "C"),
    ("red_agent", "F"), ("blue_agent", "F"),
    ("red_agent", "C"), ("blue_agent", "C"),
    ("red_agent", "F"), ("blue_agent", "F"),
    ("red_agent", "C"), ("blue_agent", "C"),
    ("red_agent", "F"), ("blue_agent", "F"),
    ("red_agent", "C"), ("blue_agent", "C"),
    ("red_agent", "F"), ("blue_agent", "F"),
    ("red_agent", "D"), ("blue_agent", "D"),
    ("red_agent", "F"), ("blue_agent", "F"),
    ("red_agent", "C"), ("blue_agent", "C"),
    ("red_agent", "F"), ("blue_agent", "F"),
    ("red_agent", "C"), ("blue_agent", "C"),
    ("red_agent", "F"), ("blue_agent", "F"),
    ("red_agent", "C"), ("blue_agent", "C"),
    ("red_agent", "F"), ("blue_agent", "F"),
    ("red_agent", "C"), ("blue_agent", "C"),
    ("red_agent", "F"), ("blue_agent", "F"),
    ("red_agent", "C"), ("blue_agent", "L"),
    ("red_agent", "A"), ("blue_agent", "No Action"),
    ("red_agent", "K"), ("blue_agent", "C"),
    ("red_agent", "L"), ("blue_agent", "No Action"),
    ("red_agent", "I"), ("blue_agent", "No Action"),
    ("red_agent", "E"), ("blue_agent", "E"),
    ("red_agent", "B"), ("blue_agent", "No Action"),
    ("red_agent", "G")
]

# Create the figure and axis
fig, ax = plt.subplots()


def update(frame):
    ax.clear()
    action = action_sequence[frame]
    agent, node = action

    if agent == "red_agent":
        node_colors[node] = 'red'
    elif agent == "blue_agent" and node != "No Action":
        node_colors[node] = 'blue'

    # Draw the graph with updated colors
    nx.draw(G, pos, with_labels=True, node_color=[node_colors[node] for node in G.nodes()], ax=ax)


# Set up the animation
ani = FuncAnimation(fig, update, frames=len(action_sequence), interval=1000, repeat=False)
plt.show()
