import matplotlib.pyplot as plt
import os

save_dir = '03_100K_DFT_MARL-ddqn_analysisGraphs'
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, 'win_percentage_chart_combined_2.png')

# Data for both pie charts
data1 = {'Red Agent': 485, 'Blue Agent': 15}
data2 = {'Red Agent': 405, 'Blue Agent': 95}
colors = ['red', 'blue']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# To plot the first pie chart
sizes1 = list(data1.values())
ax1.pie(
    sizes1,
    labels=[''] * len(data1),
    colors=colors,
    autopct='%1.1f%%',
    pctdistance=1.2,
    startangle=140,
    textprops={'fontsize': 12}
)
ax1.set_title("Over Original DFT", fontsize=14, y=-0.1)  # Move title down

# To plot the second pie chart
sizes2 = list(data2.values())
ax2.pie(
    sizes2,
    labels=[''] * len(data2),
    colors=colors,
    autopct='%1.1f%%',
    pctdistance=1.2,
    startangle=140,
    textprops={'fontsize': 12}
)
ax2.set_title("Over Improved DFT", fontsize=14, y=-0.1)  # Move title down

# Adjust legend position
ax2.scatter([], [], color='red', label='Red Agent', s=80)
ax2.scatter([], [], color='blue', label='Blue Agent', s=80)
ax2.legend(loc='lower right', fontsize=8, bbox_to_anchor=(1.2, -0.2))  # Move legend further right

plt.suptitle("Average Win Percentage of Agents with blue_agent having 20% more Resources", fontsize=16)
plt.savefig(save_path, format='png')
plt.show()
