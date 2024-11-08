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
    'A': (0, 0), 'B': (2, 0), 'C': (4, 0), 'D': (6, 0), 'E': (8, 0), 'F': (10, 0),         # BE on the lowest section
    'G': (12, 0), 'H': (14, 0), 'I': (16, 0), 'J': (18, 0), 'K': (20, 0), 'L': (22, 0),
    'PS': (1, 1), 'FDEP': (1, 2.5),                                                        # PS below and FDEP top:
    'M1': (9, 2), 'M2': (17, 2), 'M3': (13, 2),
    'C1': (5, 4), 'CSP1': (11, 4), 'CSP2': (15, 4), 'C2': (21, 4),
    'G2': (8, 6), 'G3': (18, 6),
    'G1': (13, 8)
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


# Separate node drawing by shape
def draw_nodes():
    # Draw basic events as circles
    nx.draw_networkx_nodes(
        G, pos, nodelist=basic_events, node_shape='o', node_size=400, node_color=[node_colors[node] for node in basic_events], ax=ax
    )
    # Draw intermediate and top events as squares
    nx.draw_networkx_nodes(
        G, pos, nodelist=intermediate_events + [top_event], node_shape='s', node_size=900,
        node_color=[node_colors[node] for node in intermediate_events + [top_event]], ax=ax
    )


# Draw edges with specific styles
def draw_edges():
    # Draw dotted lines for FDEP to C1 and C2
    nx.draw_networkx_edges(
        G, pos, edgelist=[('FDEP', 'C1'), ('FDEP', 'C2')],
        edge_color='black', style='dotted', width=1.5, ax=ax
    )
    # Draw solid lines for all other edges
    remaining_edges = [(u, v) for u, v in edges if (u, v) not in [('FDEP', 'C1'), ('FDEP', 'C2')]]
    nx.draw_networkx_edges(
        G, pos, edgelist=remaining_edges,
        edge_color='black', style='solid', width=1.5, ax=ax
    )


def check_cascading_failures():
    # Check if A and B are red, triggering PS and downstream failures
    if node_colors['A'] == 'red' and node_colors['B'] == 'red':
        affected_nodes = ['PS', 'FDEP', 'C1', 'C2', 'G2', 'G3', 'G1']
        for node in affected_nodes:
            node_colors[node] = 'red'

    # Check C1 (OR gate) - fails if C or D fails, triggering downstream nodes
    if node_colors['C'] == 'red' or node_colors['D'] == 'red':
        affected_nodes = ['C1', 'G2', 'G1']
        for node in affected_nodes:
            node_colors[node] = 'red'

    # Check C2 (OR gate) - fails if K or L fails, triggering downstream nodes
    if node_colors['K'] == 'red' or node_colors['L'] == 'red':
        affected_nodes = ['C2', 'G3', 'G1']
        for node in affected_nodes:
            node_colors[node] = 'red'

    # Check M1 (OR gate) - fails if E or F fails, affecting CSP1
    if node_colors['E'] == 'red' or node_colors['F'] == 'red':
        affected_nodes = ['M1', 'CSP1']
        for node in affected_nodes:
            node_colors[node] = 'red'

    # Check M2 (OR gate) - fails if I or J fails, affecting CSP2
    if node_colors['I'] == 'red' or node_colors['J'] == 'red':
        affected_nodes = ['M2', 'CSP2']
        for node in affected_nodes:
            node_colors[node] = 'red'

    # Check M3 (OR gate) - fails if G or H fails, affecting both CSP1 and CSP2
    if node_colors['G'] == 'red' or node_colors['H'] == 'red':
        affected_nodes = ['M3', 'CSP1', 'CSP2']
        for node in affected_nodes:
            node_colors[node] = 'red'

    # Check CSP1 and CSP2 failures if any two of M1, M2, or M3 fail
    failed_M_gates = sum(1 for m in ['M1', 'M2', 'M3'] if node_colors[m] == 'red')
    if failed_M_gates >= 2:
        affected_nodes = ['CSP1', 'CSP2', 'G2', 'G3', 'G1']
        for node in affected_nodes:
            node_colors[node] = 'red'

    # Ensure G2 fails if C1 or CSP1 is red
    if node_colors['C1'] == 'red' or node_colors['CSP1'] == 'red':
        affected_nodes = ['G2', 'G1']
        for node in affected_nodes:
            node_colors[node] = 'red'

    # Ensure G3 fails if C2 or CSP2 is red
    if node_colors['C2'] == 'red' or node_colors['CSP2'] == 'red':
        affected_nodes = ['G3', 'G1']
        for node in affected_nodes:
            node_colors[node] = 'red'

    # Final top event: G1 fails if either G2 or G3 fails
    if node_colors['G2'] == 'red' or node_colors['G3'] == 'red':
        node_colors['G1'] = 'red'



def update(frame):
    ax.clear()
    action = action_sequence[frame]
    agent, node = action

    # Update node colors based on agent actions
    if agent == "red_agent":
        node_colors[node] = 'red'
    elif agent == "blue_agent" and node != "No Action":
        node_colors[node] = 'blue'

    # Check for cascading failures after each action
    # check_cascading_failures()

    # Draw nodes and edges with updated colors
    draw_nodes()
    nx.draw_networkx_edges(G, pos, ax=ax)
    nx.draw_networkx_labels(G, pos, ax=ax)


# Set up the animation
ani = FuncAnimation(fig, update, frames=len(action_sequence), interval=1000, repeat=False)
plt.show()
