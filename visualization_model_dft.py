import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Initialize the fault tree as a directed graph
G = nx.DiGraph()

# Define basic, intermediate, and top events
basic_events = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
intermediate_events = {
    'PS': 'AND', 'FDEP': None, 'C1': 'OR', 'C2': 'OR',
    'M1': 'OR', 'M2': 'OR', 'M3': 'OR', 'CSP1': 'CSP', 'CSP2': 'CSP', 'G2': 'OR', 'G3': 'OR'
}
top_event = 'G1'

gate_types = {**intermediate_events, top_event: 'AND'}
G.add_nodes_from(basic_events + list(intermediate_events.keys()) + [top_event])  # Add nodes and edges to the graph

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
    'A': (0, 0), 'B': (2, 0), 'C': (4, 0), 'D': (6, 0), 'E': (8, 0), 'F': (10, 0),  # BE on the lowest section
    'G': (12, 0), 'H': (14, 0), 'I': (16, 0), 'J': (18, 0), 'K': (20, 0), 'L': (22, 0),
    'PS': (1, 1), 'FDEP': (1, 2.5),  # PS below and FDEP top:
    'M1': (9, 2), 'M2': (17, 2), 'M3': (13, 2),
    'C1': (5, 4), 'CSP1': (11, 4), 'CSP2': (15, 4), 'C2': (21, 4),
    'G2': (8, 6), 'G3': (18, 6),
    'G1': (13, 8)
}

# Initialize node colors
node_colors = {node: 'green' for node in G.nodes()}

# Action sequence (from the game)
action_sequence = [
    ("red_agent", "E"), ("blue_agent", "No Action"),
    ("red_agent", "H"), ("blue_agent", "No Action"),
    ("red_agent", "J"), ("blue_agent", "H"),
    ("red_agent", "C"), ("blue_agent", "J"),
    ("red_agent", "K"), ("blue_agent", "No Action"),
    # ("red_agent", "A"), ("blue_agent", "F"),
    # ("red_agent", "F"), ("blue_agent", "F"),
    ("red_agent", "B"), ("blue_agent", "No Action"),
    # ("red_agent", "F"), ("blue_agent", "No Action"),
    # ("red_agent", "J"), ("blue_agent", "No Action"),
    # ("red_agent", "F"), ("blue_agent", "F"),
    # ("red_agent", "L"), ("blue_agent", "No Action"),
    # ("red_agent", "C"), ("blue_agent", "C"),
    # ("red_agent", "F"), ("blue_agent", "F"),
    # ("red_agent", "C"), ("blue_agent", "C"),
    # ("red_agent", "F"), ("blue_agent", "F"),
    # ("red_agent", "C"), ("blue_agent", "C"),
    # ("red_agent", "F"), ("blue_agent", "F"),
    # ("red_agent", "C"), ("blue_agent", "C"),
    # ("red_agent", "F"), ("blue_agent", "F"),
    # ("red_agent", "C"), ("blue_agent", "C"),
    # ("red_agent", "F"), ("blue_agent", "F"),
    # ("red_agent", "D"), ("blue_agent", "D"),
    # ("red_agent", "F"), ("blue_agent", "F"),
    # ("red_agent", "C"), ("blue_agent", "C"),
    # ("red_agent", "F"), ("blue_agent", "F"),
    # ("red_agent", "C"), ("blue_agent", "C"),
    # ("red_agent", "F"), ("blue_agent", "F"),
    # ("red_agent", "C"), ("blue_agent", "C"),
    # ("red_agent", "F"), ("blue_agent", "F"),
    # ("red_agent", "C"), ("blue_agent", "C"),
    # ("red_agent", "F"), ("blue_agent", "F"),
    # ("red_agent", "C"), ("blue_agent", "L"),
    # ("red_agent", "A"), ("blue_agent", "No Action"),
    # ("red_agent", "K"), ("blue_agent", "C"),
    # ("red_agent", "L"), ("blue_agent", "No Action"),
    # ("red_agent", "I"), ("blue_agent", "No Action"),
    # ("red_agent", "E"), ("blue_agent", "E"),
    # ("red_agent", "B"), ("blue_agent", "No Action"),
    # ("red_agent", "G")
]


# Helper functions for drawing and updating
def draw_graph():
    nx.draw_networkx_nodes(
        G, pos, nodelist=basic_events, node_shape='o', node_size=400,
        node_color=[node_colors[node] for node in basic_events], ax=ax
    )
    nx.draw_networkx_nodes(
        G, pos, nodelist=list(intermediate_events.keys()) + [top_event], node_shape='s', node_size=900,
        node_color=[node_colors[node] for node in list(intermediate_events.keys()) + [top_event]], ax=ax
    )
    nx.draw_networkx_edges(
        G, pos, edgelist=[(u, v) for u, v in edges if (u, v) not in [('FDEP', 'C1'), ('FDEP', 'C2')]],
        edge_color='black', style='solid', width=1.5, ax=ax
    )
    nx.draw_networkx_edges(
        G, pos, edgelist=[('FDEP', 'C1'), ('FDEP', 'C2')],
        edge_color='orange', style='dotted', width=1.5, ax=ax
    )
    nx.draw_networkx_labels(G, pos, ax=ax)


# Cascading failure logic
def check_cascading_failures():
    for node in G.nodes:
        if node not in gate_types:  # Skip basic events with no gate type
            continue
        gate_type = gate_types[node]
        children = list(G.predecessors(node))

        # PS Failure Logic - triggers FDEP which causes both C1 and C2 to fail
        if node == 'PS' and node_colors[node] == 'red':
            node_colors['FDEP'] = 'red'
            node_colors['C1'] = 'red'
            node_colors['C2'] = 'red'
            continue

        if gate_type == 'AND':
            # AND gate: all children must fail for this node to fail
            if all(node_colors[child] == 'red' for child in children):
                node_colors[node] = 'red'
            elif any(node_colors[child] != 'red' for child in children):
                node_colors[node] = 'green'

        elif gate_type == 'OR':
            # OR gate: any child failure causes this node to fail
            if any(node_colors[child] == 'red' for child in children):
                node_colors[node] = 'red'
            elif all(node_colors[child] != 'red' for child in children):
                node_colors[node] = 'green'

        elif gate_type == 'CSP':
            # Updated CSP logic: CSP1 and CSP2 fail if two or more of M1, M2, M3 fail
            intermediate_events = ['M1', 'M2', 'M3']
            failed_count = sum(1 for event in intermediate_events if node_colors[event] == 'red')
            if failed_count >= 2:
                node_colors['CSP1'] = 'red'
                node_colors['CSP2'] = 'red'
            else:
                node_colors['CSP1'] = 'green'
                node_colors['CSP2'] = 'green'


# Animation update function
def update(frame):
    ax.clear()
    agent, node = action_sequence[frame]
    if agent == "red_agent":
        node_colors[node] = 'red'
    elif agent == "blue_agent" and node != "No Action":
        node_colors[node] = 'blue'
    check_cascading_failures()
    draw_graph()


# Set up the figure and axis
fig, ax = plt.subplots()
ani = FuncAnimation(fig, update, frames=len(action_sequence), interval=1000, repeat=False)
plt.show()
