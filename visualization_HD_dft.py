import os
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

save_dir = '03_100K_DFT_MARL-ddqn_analysisGraphs'
animation_path = os.path.join(save_dir, 'hd_fault_propagation_animation.gif')
fig, ax = plt.subplots(figsize=(14, 12))

G = nx.DiGraph()  # Initializing the fault tree as a directed graph

# Basic, Intermediate, and Top Events
basic_events = {  # 'Basic Event': MTTR values
    'A': 3, 'B': 7, 'C01': 2, 'C02': 2, 'D': 2, 'E01': 2, 'E02': 3, 'F': 2, 'G': 3, 'H': 2,
    'I01': 2, 'I02': 2, 'J': 2, 'K': 2, 'L01': 3, 'L02': 2
    }
intermediate_events = {  # 'Intermediate Event': Gate type
    'PS': 'AND', 'FDEP': '', 'C_re': 'AND', 'E_re': 'AND', 'I_re': 'AND', 'L_re': 'AND',
    'C1': 'OR', 'C2': 'OR', 'M1': 'OR', 'M2': 'OR', 'M3': 'OR', 'CSP1': 'CSP', 'CSP2': 'CSP', 'G2': 'OR', 'G3': 'OR'
    }
top_event = 'G1'
gate_types = {**intermediate_events, top_event: 'AND'}

G.add_nodes_from(list(basic_events.keys()) + list(intermediate_events.keys()) + [top_event])  # Adding nodes and edges to the graph

edges = [
    ('A', 'PS'), ('B', 'PS'), ('PS', 'FDEP'),
    ('FDEP', 'C1'), ('FDEP', 'C2'),
    ('C01', 'C_re'), ('C02', 'C_re'),
    ('C_re', 'C1'), ('D', 'C1'),
    ('E01', 'E_re'), ('E02', 'E_re'),
    ('E_re', 'M1'), ('F', 'M1'),
    ('G', 'M3'), ('H', 'M3'),
    ('I01', 'I_re'), ('I02', 'I_re'),
    ('I_re', 'M2'), ('J', 'M2'),
    ('L01', 'L_re'), ('L02', 'L_re'),
    ('K', 'C2'), ('L_re', 'C2'),
    ('I01', 'F'), ('L02', 'K'),
    ('M1', 'CSP1'), ('M2', 'CSP2'),
    ('M3', 'CSP1'), ('M3', 'CSP2'),
    ('C1', 'G2'), ('CSP1', 'G2'),
    ('C2', 'G3'), ('CSP2', 'G3'),
    ('G2', 'G1'), ('G3', 'G1')
]

G.add_edges_from(edges)

# Positions for the nodes
pos = {
    'C01': (3, 0), 'C02': (5, 0), 'E01': (7, 0), 'E02': (9, 0), 'I01': (15, 0), 'I02': (17, 0), 'L01': (21, 0), 'L02': (23, 0),
    'A': (0, 1), 'B': (2, 1), 'C_re': (4, 1), 'D': (6, 1), 'E_re': (8, 1), 'F': (10, 1),  # BE on the lowest section
    'G': (12, 1), 'H': (14, 1), 'I_re': (16, 1), 'J': (18, 1), 'K': (20, 1), 'L_re': (22, 1),
    'PS': (1, 2), 'FDEP': (1, 3.5),  # PS below and FDEP top:
    'M1': (9, 3), 'M2': (17, 3), 'M3': (13, 3),
    'C1': (5, 5), 'CSP1': (11, 5), 'CSP2': (15, 5), 'C2': (21, 5),
    'G2': (8, 7), 'G3': (18, 7),
    'G1': (13, 9)
}

node_colors = {node: 'green' for node in G.nodes()}
repair_timers = {node: 0 for node in basic_events}  # To track MTTR countdowns for blue agent repairs

# Simulated Action Sequence
action_sequence = [
    ("red_agent", "No Action"), ("blue_agent", "No Action"),
    ("red_agent", "A"), ("blue_agent", "No Action"),
    ("red_agent", "B"), ("blue_agent", "A"),
    ("red_agent", "C01"), ("blue_agent", "B"),
    ("red_agent", "C02"), ("blue_agent", "No Action"),
    ("red_agent", "L02"), ("blue_agent", "No Action"),
    ("red_agent", "K"), ("blue_agent", "C01"),
    ("red_agent", "E01"), ("blue_agent", "C02"),
    ("red_agent", "E02"), ("blue_agent", "K"),
    ("red_agent", "I01"), ("blue_agent", "No Action"),
    ("red_agent", "I02"), ("blue_agent", "E01"),
    ("red_agent", "F"), ("blue_agent", "I01"),
    ("red_agent", "L01"), ("blue_agent", "F"),
    ("red_agent", "No Action"), ("blue_agent", "E02"),
    ("red_agent", "D"), ("blue_agent", "I02"),
    ("red_agent", "No Action"), ("blue_agent", "L01"),
    ("red_agent", "No Action"), ("blue_agent", "D"),
    ("red_agent", "No Action"), ("blue_agent", "L02")
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
        G, pos, edgelist=[(u, v) for u, v in edges if (u, v) not in [('FDEP', 'C1'), ('FDEP', 'C2'), ('PS', 'FDEP'), ('I01', 'F'), ('L02', 'K')]],
        edge_color='black', style='solid', width=1.5, ax=ax
    )
    nx.draw_networkx_edges(
        G, pos, edgelist=[('FDEP', 'C1'), ('FDEP', 'C2'), ('PS', 'FDEP')],
        edge_color='orange', style='dotted', width=1.5, ax=ax
    )
    nx.draw_networkx_edges(
        G, pos, edgelist=[('I01', 'F'), ('L02', 'K')],
        edge_color='olive', style='dotted', width=1.5, ax=ax
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
        mlines.Line2D([], [], marker='o', color='green', markersize=10, label='Initial state of BE : all working', markerfacecolor='green', linestyle='None'),
        mpatches.Rectangle((0, 0), width=0.2, height=0.2, facecolor='green', edgecolor='green', label='Initial state of Intermediate Events'),
        mlines.Line2D([], [], marker='o', color='red', markersize=10, label='Failed BE', markerfacecolor='red', linestyle='None'),
        mlines.Line2D([], [], marker='o', color='blue', markersize=10, label='Fixed BE', markerfacecolor='blue', linestyle='None'),
        mpatches.Rectangle((0, 0), width=0.2, height=0.2, facecolor='red', edgecolor='red', label='Fault Propagation'),
        mlines.Line2D([], [], color='orange', linestyle=':', label='FDEP Trigger'),
        mlines.Line2D([], [], color='olive', linestyle=':', label='Conditioned Event')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=8, handlelength=1.2, handleheight=1.2)


# Cascading failure logic
def check_cascading_failures():
    for node in G.nodes:
        if node not in gate_types:  # Skip basic events with no gate type
            continue
        gate_type = gate_types[node]
        children = list(G.predecessors(node))

        # PS Failure Logic - triggers FDEP which causes both C1 and C2 to fail
        if node == 'PS':
            # If either A or B (children of PS) is green, PS should be green
            if any(node_colors[child] == 'green' for child in children):
                node_colors['PS'] = 'green'
                node_colors['FDEP'] = 'green'
                node_colors['C1'] = 'green'
                node_colors['C2'] = 'green'
            else:
                # Otherwise, if both children are red, PS turns red and triggers FDEP and cascades
                node_colors['PS'] = 'red'
                node_colors['FDEP'] = 'red'
                node_colors['C1'] = 'red'
                node_colors['C2'] = 'red'
            continue

        if gate_type == 'AND':
            # AND gate: all children must fail for this node to fail
            if all(node_colors[child] == 'red' for child in children):
                node_colors[node] = 'red'
            elif any(node_colors[child] == 'green' for child in children):
                node_colors[node] = 'green'

        elif gate_type == 'OR':
            # OR gate: any child failure causes this node to fail
            if all(node_colors[child] == 'green' for child in children):
                node_colors[node] = 'green'
            elif any(node_colors[child] == 'red' or 'blue' for child in children):
                node_colors[node] = 'red'

        elif gate_type == 'CSP':
            # Updated CSP logic: CSP1 and CSP2 fail if two or more of M1, M2, M3 fails
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
    if node == "F":
        if node_colors["F"] != 'red' and node_colors["I01"] != 'red':  # F is conditioned on I01
            return
    if node == "K":
        if node_colors["K"] != 'red' and node_colors["L02"] != 'red':  # K is conditioned on L02
            return

    if agent == "red_agent":
        node_colors[node] = 'red'
    elif agent == "blue_agent" and node != "No Action":
        node_colors[node] = 'blue'
        repair_timers[node] = basic_events[node]  # Set mttr timer
    for n, timer in repair_timers.items():
        if timer > 0:
            repair_timers[n] -= 1
            if repair_timers[n] == 0:
                node_colors[n] = 'green'  # Turn BE back to the default color after repair
    check_cascading_failures()
    draw_graph()
    ax.set_title("Fault Injection and Fault Propagation in the Improved DFT", fontsize=16, fontweight='normal')


ani = FuncAnimation(fig, update, frames=len(action_sequence), interval=1000, repeat=False)
ani.save(animation_path, writer='pillow', fps=1, dpi=100)
plt.show()
