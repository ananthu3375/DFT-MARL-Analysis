import os
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

save_dir = '03_100K_DFT_MARL-ddqn_analysisGraphs'
animation_path = os.path.join(save_dir, 'ld_fault_propagation_animation.gif')
fig, ax = plt.subplots(figsize=(18, 12))

G = nx.DiGraph()  # Initializing the fault tree as a directed graph

# Basic, Intermediate, and Top Events
basic_events = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
intermediate_events = {
    'PS': 'AND', 'FDEP': '', 'C1': 'OR', 'C2': 'OR',
    'M1': 'OR', 'M2': 'OR', 'M3': 'OR', 'CSP1': 'CSP', 'CSP2': 'CSP', 'G2': 'OR', 'G3': 'OR'
}
top_event = 'G1'
gate_types = {**intermediate_events, top_event: 'AND'}

G.add_nodes_from(basic_events + list(intermediate_events.keys()) + [top_event])  # Add nodes and edges to the graph

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

# Positions for nodes
pos = {
    'A': (0, 0), 'B': (2, 0), 'C': (4, 0), 'D': (6, 0), 'E': (8, 0), 'F': (10, 0),  # BE on the lowest section
    'G': (12, 0), 'H': (14, 0), 'I': (16, 0), 'J': (18, 0), 'K': (20, 0), 'L': (22, 0),
    'PS': (1, 1), 'FDEP': (1, 2.5),  # PS below and FDEP top:
    'M1': (9, 2), 'M2': (17, 2), 'M3': (13, 2),
    'C1': (5, 4), 'CSP1': (11, 4), 'CSP2': (15, 4), 'C2': (21, 4),
    'G2': (8, 6), 'G3': (18, 6),
    'G1': (13, 8)
}

node_colors = {node: 'green' for node in G.nodes()}

# Simulated Action Sequence
action_sequence = [
    ("red_agent", "No Action"), ("blue_agent", "No Action"),
    ("red_agent", "E"), ("blue_agent", "No Action"),
    ("red_agent", "H"), ("blue_agent", "No Action"),
    ("red_agent", "J"), ("blue_agent", "H"),
    ("red_agent", "C"), ("blue_agent", "J"),
    ("red_agent", "K"), ("blue_agent", "No Action"),
    ("red_agent", "A"), ("blue_agent", "K"),
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
        G, pos, nodelist=basic_events, node_shape='o', node_size=900,
        node_color=[node_colors[node] for node in basic_events], ax=ax
    )
    nx.draw_networkx_nodes(
        G, pos, nodelist=list(intermediate_events.keys()) + [top_event], node_shape='s', node_size=1500,
        node_color=[node_colors[node] for node in list(intermediate_events.keys()) + [top_event]], ax=ax
    )
    nx.draw_networkx_edges(
        G, pos, edgelist=[(u, v) for u, v in edges if (u, v) not in [('FDEP', 'C1'), ('FDEP', 'C2'), ('PS', 'FDEP')]],
        edge_color='black', style='solid', width=1.5, ax=ax
    )
    nx.draw_networkx_edges(
        G, pos, edgelist=[('FDEP', 'C1'), ('FDEP', 'C2'), ('PS', 'FDEP')],
        edge_color='orange', style='dotted', width=1.5, ax=ax
    )

    basic_labels = {node: node for node in basic_events}
    nx.draw_networkx_labels(G, pos, labels=basic_labels, font_size=12, ax=ax)  # Font Size - BE

    for node in gate_types.keys():  # Gate type of intermediate and top events
        x, y = pos[node]  # Position of each node
        if node == 'FDEP':
            ax.text(x, y, node, ha='center', va='center', fontsize=12, fontweight='normal', color='orange')
            continue

        ax.text(x, y + 0.1, node, ha='center', va='center', fontsize=12, fontweight='bold', color='black')  # Event name
        ax.text(x, y - 0.15, f"{gate_types[node]}", ha='center', va='center', fontsize=10, color='black')  # Gate type

    legend_elements = [
        # Circles for initial state and status of BE
        mlines.Line2D([], [], marker='o', color='green', markersize=10, label='Initial state of BE : all working', markerfacecolor='green', linestyle='None'),  # Green circle
        mpatches.Rectangle((0, 0), width=0.2, height=0.2, facecolor='green', edgecolor='green', label='Initial state of Intermediate Events'),  # Green square
        mlines.Line2D([], [], marker='o', color='red', markersize=10, label='Failed BE', markerfacecolor='red', linestyle='None'),  # Red circle
        mlines.Line2D([], [], marker='o', color='blue', markersize=10, label='Fixed BE', markerfacecolor='blue', linestyle='None'),  # Blue circle
        mpatches.Rectangle((0, 0), width=0.2, height=0.2, facecolor='red', edgecolor='red', label='Fault Propagation'),  # Red square
        mlines.Line2D([], [], color='orange', linestyle=':', label='FDEP Trigger')  # Dotted orange line
    ]

    # Legend in the top-left corner
    ax.legend(handles=legend_elements, loc='upper left', fontsize=8, handlelength=1.2, handleheight=1.2)


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
    ax.set_title("Fault Injection and Fault Propagation in the Original DFT", fontsize=16, fontweight='normal')


ani = FuncAnimation(fig, update, frames=len(action_sequence), interval=1000, repeat=False)
ani.save(animation_path, writer='pillow', fps=1, dpi=150)
plt.show()
