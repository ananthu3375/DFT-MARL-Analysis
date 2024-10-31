import matplotlib.pyplot as plt
import os

# Reliability data
events = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
reliabilities = [0.9818, 0.9880, 0.8418, 0.8927, 0.8423, 0.8507, 0.9656, 0.9816, 0.8728, 0.8650, 0.9718, 0.9620]

# Sort events and reliabilities by reliability in descending order
sorted_indices = sorted(range(len(reliabilities)), key=lambda i: reliabilities[i], reverse=True)
events = [events[i] for i in sorted_indices]
reliabilities = [reliabilities[i] for i in sorted_indices]

# Plot the reliability as dots
plt.figure(figsize=(10, 6))
plt.plot(events, reliabilities, 'o', color='blue', markersize=8, label='Reliability')
plt.plot(events, reliabilities, linestyle=':', color='lightblue', linewidth=1)
# Set y-axis limits from 0.6 to 1.2
plt.ylim(0.6, 1.2)

# Add labels and title
plt.xlabel('Event')
plt.ylabel('Reliability')
plt.title('Reliability of Events for Red Agent (Sorted by Reliability)')

# Display values above each point
for i, reliability in enumerate(reliabilities):
    plt.text(i, reliability + 0.02, f"{reliability:.4f}", ha='center', va='bottom', rotation=90)
plt.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)

# Ensure the save directory exists
save_dir = '2_100K_DFT_MARL-ddqn_analysisGraphs'
os.makedirs(save_dir, exist_ok=True)

# Save the plot
plt.grid(True)
save_path = os.path.join(save_dir, 'reliability_plot.png')
plt.savefig(save_path)
plt.show()

print(f"Plot saved to {save_path}")
