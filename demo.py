# import matplotlib.pyplot as plt
# import numpy as np
#
# # Define events and their usage counts for Red and Blue Agents
# events = ['A', 'C01', 'C02', 'D', 'E01', 'E02', 'F', 'I01', 'I02', 'J', 'K', 'L01', 'L02', 'No Action']
# red_values = [217, 486, 499, 571, 500, 124, 177, 167, 199, 623, 489, 500, 8320, 0]  # 0 for 'No Action' in Red
# blue_values = [162, 0, 5, 0, 0, 0, 0, 0, 0, 225, 4, 0, 699, 11277]  # 0 for missing events in Blue
#
# # Set up figure and bar width
# plt.figure(figsize=(12, 7))
# bar_width = 0.45
# index = np.arange(len(events))
#
# # Plot bars for Red Agent and Blue Agent
# plt.bar(index, red_values, bar_width, label='Red Agent', color='red')
# plt.bar(index + bar_width, blue_values, bar_width, label='Blue Agent', color='blue')
#
# # Use logarithmic scale on the y-axis
# plt.yscale('log')
# plt.xlabel('Event')
# plt.ylabel('Count')
# plt.title('Game Event Usage')
#
# # Set x-ticks to show event names
# plt.xticks(index + bar_width / 2, events, rotation=45)
#
# # Add a legend to differentiate between Red and Blue Agent bars
# plt.legend()
#
# # Show the plot
# plt.tight_layout()
# plt.show()
import matplotlib.pyplot as plt
import numpy as np
import os

save_dir = '03_100K_DFT_MARL-ddqn_analysisGraphs'
os.makedirs(save_dir, exist_ok=True)

# Define events and their usage counts for Red and Blue Agents
events = ['A', 'C01', 'C02', 'D', 'E01', 'E02', 'F', 'I01', 'I02', 'J', 'K', 'L01', 'L02', 'No Action']
red_values = [217, 486, 499, 571, 500, 124, 177, 167, 199, 623, 489, 500, 8320, 0]  # 0 for 'No Action' in Red
blue_values = [162, 0, 5, 0, 0, 0, 0, 0, 0, 225, 4, 0, 699, 11277]  # 0 for missing events in Blue

# Set up figure and main plot parameters
fig, ax = plt.subplots(figsize=(12, 7))
bar_width = 0.45
index = np.arange(len(events))

# Plot bars for Red Agent and Blue Agent on the main plot
ax.bar(index, red_values, bar_width, label='Red Agent', color='red')
ax.bar(index + bar_width, blue_values, bar_width, label='Blue Agent', color='blue')

# Focus the main plot on the lower value range
ax.set_ylim(0, 1000)  # Adjust as needed to show moderate values
ax.set_xlabel('Events', labelpad=10)
ax.set_ylabel('Count')
ax.set_title('Game Event Usage')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(events, rotation=45)

# Add a legend for clarity
ax.legend()

# Inset plot to show higher values separately
ax_inset = fig.add_axes([0.5, 0.61, 0.25, 0.25])  # Adjust position and size of the inset
ax_inset.bar(index, red_values, bar_width, color='red')
ax_inset.bar(index + bar_width, blue_values, bar_width, color='blue')

# Set y-limit of inset to focus on high values
ax_inset.set_ylim(1000, 12000)  # Adjust this range to capture high-value events
ax_inset.set_xticks(index + bar_width / 2)
ax_inset.set_xticklabels(events, rotation=90, fontsize=8)  # Rotate for readability in inset

save_path = os.path.join(save_dir, 'game_event_usage_with_inset.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()
