"""
File:     improvement_potential.py
Author:   Ananthu Ramesh S
Purpose:  To evaluate the reliability of the events and the overall system
          To calculate Improvement Potential of each Basic Event
"""

import pandas as pd
from matplotlib import pyplot as plt
import xml.etree.ElementTree as ET
from evaluation import calculate_and_plot_event_usage, actual_game


class ReliabilityAnalysis:
    def __init__(self, actions, wins, model_file):
        self.actions = actions
        self.wins = wins
        self.failure_rate = self.calculate_system_failure_rate()
        self.red_agent_event_count = {}
        self.total_event_usages = 0
        self.events = {}
        self.gates = {}
        self.dependencies = {}
        self.event_reliabilities = {}
        self.BE_failure_rates = {}
        self.gate_reliabilities = {}
        self.gate_failure_rates = {}
        self.R_system = None
        self.load_model(model_file)

    def load_model(self, model_file):
        """
        Load and parse the XML model to extract gates, events, and dependencies.
        :param model_file: Path to the XML file defining the DFT
        """
        tree = ET.parse(model_file)
        root = tree.getroot()

        for event in root.findall('event'):
            event_name = event.get('name')
            event_type = event.get('type')
            gate_type = event.get('gate_type')
            if event_type == "BASIC":
                self.events[event_name] = {
                    'type': event_type,
                }
                # Assign reliability from user input for basic events
                self.event_reliabilities[event_name] = None
                self.BE_failure_rates[event_name] = None
            elif event_type == "INTERMEDIATE":
                self.gates[event_name] = {
                    'type': event_type,
                    'gate_type': gate_type,
                    'children': []
                }
            elif event_type == "TOP":
                # Store the top event for system reliability calculation
                self.gates[event_name] = {
                    'type': 'TOP',
                    'gate_type': gate_type,
                    'children': []
                }

        for precedence in root.findall('precedence'):
            source = precedence.get('source')
            target = precedence.get('target')
            ptype = precedence.get('type')

            if ptype == 'NORMAL':
                self.gates[target]['children'].append(source)
            elif ptype == 'FDEP':
                if 'fdep_sources' not in self.gates[target]:
                    self.gates[target]['fdep_sources'] = []
                self.gates[target]['fdep_sources'].append(source)
            elif ptype == 'CSP':
                competitor = precedence.get('competitor')
                self.gates[target]['competitor'] = competitor
                self.gates[target]['children'].append(source)

    def calculate_system_failure_rate(self):
        """
        Calculate the system failure rate as the number of wins by the red agent
        divided by the total number of games played.
        """
        total_games = self.wins["red_agent"] + self.wins["blue_agent"]
        system_failures = self.wins["red_agent"]  # Red agent's win indicates system failure
        failure_rate = system_failures / total_games
        print(f"\n\nSystem Failure Rate: {failure_rate:.4f}")
        return failure_rate

    def calculate_total_event_usage(self):
        """
        Calculate the total number of event usages by the red agent.
        """
        for game_actions in self.actions["red_agent"]:
            for event in game_actions:
                if event in self.red_agent_event_count:
                    self.red_agent_event_count[event] += 1
                else:
                    self.red_agent_event_count[event] = 1
        self.total_event_usages = sum(self.red_agent_event_count.values())
        print(f"Total Event Usages by Red Agent: {self.total_event_usages}")

    def calculate_reliability(self):
        """
        Calculate the reliability of each event for the red agent using the formula:
        Reliability of Event_i = 1 - (Usage of Event_i / Total Usages) * Failure Rate (System)
        """
        print("\nReliability of Events for Red Agent:")
        for event in sorted(self.red_agent_event_count.keys()):
            usage_count = self.red_agent_event_count[event]
            reliability = 1 - (usage_count / self.total_event_usages) * self.failure_rate
            self.event_reliabilities[event] = reliability
            print(f"Event {event}: {reliability:.4f}")

    def calculate_BE_failure_rate(self):
        """
        Calculate the failure rate of each event for the red agent using the formula:
        Failure Rate Event_i = (Usage of Event_i / Total Usages) * Failure Rate (System)
        """
        print("\n\nFailure Rates of Basic Events using Red Agent Policy:")
        for event in sorted(self.red_agent_event_count.keys()):
            usage_count = self.red_agent_event_count[event]
            be_failure_rate = (usage_count / self.total_event_usages) * self.failure_rate
            self.BE_failure_rates[event] = be_failure_rate
            print(f"Event {event}: {be_failure_rate:.4f}")

    def calculate_intermediate_top_failure_rates(self):
        """
        Calculate the failure rates for all intermediate events (PS, C1, C2, M1, M2, M3, CSP1, CSP2, G2, G3, G1).
        This function calculates based on previously defined dependent events in the system.
        """
        print("\nFailure Rates of Intermediate Events using Red Agent Policy:")
        # Calculate PS failure rate
        failure_rate_A = self.BE_failure_rates.get("A", 0)
        failure_rate_B = self.BE_failure_rates.get("B", 0)
        ps_failure_rate = failure_rate_A * failure_rate_B
        self.gate_failure_rates['PS'] = ps_failure_rate
        print(f"PS = {ps_failure_rate:.4f}")

        # Calculate C1 and C2 failure rates using OR gates
        failure_rate_C = self.BE_failure_rates.get("C", 0)
        failure_rate_D = self.BE_failure_rates.get("D", 0)
        failure_rate_K = self.BE_failure_rates.get("K", 0)
        failure_rate_L = self.BE_failure_rates.get("L", 0)
        failure_rate_PS = self.gate_failure_rates.get('PS', 0)

        # OR gate logic for C1 and C2
        failure_rate_C1 = (failure_rate_C + failure_rate_D + failure_rate_PS
                           - failure_rate_C * failure_rate_D
                           - failure_rate_C * failure_rate_PS
                           - failure_rate_D * failure_rate_PS
                           + failure_rate_C * failure_rate_D * failure_rate_PS)

        failure_rate_C2 = (failure_rate_K + failure_rate_L + failure_rate_PS
                           - failure_rate_K * failure_rate_L
                           - failure_rate_K * failure_rate_PS
                           - failure_rate_L * failure_rate_PS
                           + failure_rate_K * failure_rate_L * failure_rate_PS)

        self.gate_failure_rates["C1"] = failure_rate_C1
        self.gate_failure_rates["C2"] = failure_rate_C2
        print(f"C1 = {failure_rate_C1:.4f}")
        print(f"C2 = {failure_rate_C2:.4f}")

        # Calculate M1, M2, M3 failure rates based on child events
        intermediate_events = ["M1", "M2", "M3"]
        for event in intermediate_events:
            if event in self.gates:
                children = self.gates[event]['children']
                child_failure_rates = [self.BE_failure_rates.get(child, 0) for child in children]
                if child_failure_rates:
                    failure_rate = child_failure_rates[0] + child_failure_rates[1] - (
                                child_failure_rates[0] * child_failure_rates[1])
                    self.gate_failure_rates[event] = failure_rate
                    print(f"{event} = {failure_rate:.4f}")

        # Calculate CSP1 and CSP2 failure rates
        failure_rate_M1 = self.gate_failure_rates.get("M1", 0)
        failure_rate_M2 = self.gate_failure_rates.get("M2", 0)
        failure_rate_M3 = self.gate_failure_rates.get("M3", 0)

        csp1_failure_rate = failure_rate_M1 * (failure_rate_M3 + (1 - failure_rate_M3) * failure_rate_M2)
        self.gate_failure_rates["CSP1"] = csp1_failure_rate
        print(f"CSP1 = {csp1_failure_rate:.4f}")

        csp2_failure_rate = failure_rate_M2 * (failure_rate_M3 + (1 - failure_rate_M3) * failure_rate_M1)
        self.gate_failure_rates["CSP2"] = csp2_failure_rate
        print(f"CSP2 = {csp2_failure_rate:.4f}")

        # Calculate G2 and G3 failure rates based on child events
        for event in ["G2", "G3"]:
            if event in self.gates:
                children = self.gates[event]['children']
                child_failure_rates = [self.gate_failure_rates.get(child, 0) for child in children]
                if child_failure_rates:
                    failure_rate = child_failure_rates[0] + child_failure_rates[1] - (
                                child_failure_rates[0] * child_failure_rates[1])
                    self.gate_failure_rates[event] = failure_rate
                    print(f"{event} = {failure_rate:.4f}")

        # Calculate G1 (Top Event) failure rate
        print("\nFailure Rates of Top Event using Red Agent Policy:")
        failure_rate_G2 = self.gate_failure_rates.get("G2", 0)
        failure_rate_G3 = self.gate_failure_rates.get("G3", 0)
        g1_failure_rate = failure_rate_G2 * failure_rate_G3
        self.gate_failure_rates['G1'] = g1_failure_rate
        print(f"G1 = {g1_failure_rate:.4f}")

    def run(self):
        self.calculate_total_event_usage()
        self.calculate_reliability()
        self.calculate_BE_failure_rate()
        self.calculate_intermediate_top_failure_rates()


class ImprovementPotential:
    def __init__(self, reliability_analysis):
        """
        Initialize with an instance of ReliabilityAnalysis to access the calculated
        event and system failure rates.
        """
        self.reliability_analysis = reliability_analysis
        self.BE_failure_rates = self.reliability_analysis.BE_failure_rates
        self.improvement_potentials = {}

    def improvement_potential(self):
        """
        Calculate the Improvement Potential for each basic event by setting the 
        event's failure rate to zero and recalculating the TOP event failure rate.
        """
        original_top_failure_rate = self.reliability_analysis.gate_failure_rates.get("G1", 0)
        original_failure_rates = self.BE_failure_rates.copy()  # Store original failure rates

        print("\nImprovement Potentials for each Basic Event:")
        for event in original_failure_rates.keys():
            self.reliability_analysis.BE_failure_rates[event] = 0  # Temporarily set the failure rate of the event to 0 for that event
            self.reliability_analysis.calculate_intermediate_top_failure_rates()  # Recalculate the intermediate and top event failure rates
            new_top_failure_rate = self.reliability_analysis.gate_failure_rates.get("G1", 0)
            improvement_potential_value = (original_top_failure_rate - new_top_failure_rate) * 10  # a factor 10 for better understanding of values
            self.improvement_potentials[event] = improvement_potential_value
            print(f"Event {event}: Improvement Potential = {improvement_potential_value:.4f}")
            self.reliability_analysis.BE_failure_rates[event] = original_failure_rates[event]  # Restore the original failure rate for the event
        self.reliability_analysis.calculate_intermediate_top_failure_rates()  # Reset the intermediate and top event failure rates back to original values
        print("\nImprovement Potentials for each Basic Event (Summary):")
        for event, potential in self.improvement_potentials.items():
            print(f"{event} = {potential:.4f}")


if __name__ == "__main__":
    # Run the game to generate the action data
    results = actual_game()

    # Extract the actions and wins from actual_game output
    actions = results['actions']
    wins = results['wins']
    print(f"\n\nCALCULATIONS BASED ON RED AGENT'S POLICY")

    # Calculate reliability
    reliability_analysis = ReliabilityAnalysis(actions, wins, 'model.xml')
    reliability_analysis.run()

    # Calculate improvement potential
    improvement_potential_analysis = ImprovementPotential(reliability_analysis)
    improvement_potential_analysis.improvement_potential()

    # Visualization of improvement potentials
    events = list(improvement_potential_analysis.improvement_potentials.keys())
    improvement_potentials = list(improvement_potential_analysis.improvement_potentials.values())
    event_usages = [reliability_analysis.red_agent_event_count.get(event, 0) for event in events]
    data = pd.DataFrame({"Event": events, "Improvement Potential": improvement_potentials, "Event Usage": event_usages})
    data = data.sort_values(by="Improvement Potential", ascending=False)

    fig, ax2 = plt.subplots(figsize=(12, 6))

    ax2.bar(data["Event"], data["Event Usage"], color="#CD1C18", alpha=1.0, width=0.7, zorder=1)  # Event Usage on the left y-axis
    ax2.set_ylabel("Event Usage by Red Agent", color="red", alpha=1.0)
    ax2.tick_params(axis="y", labelcolor="red")

    ax1 = ax2.twinx()  # Improvement Potential on the right y-axis
    ax1.scatter(data["Event"], data["Improvement Potential"], color="#069C56", s=70, edgecolor="#069C56", linewidth=0.6, zorder=2)
    ax1.set_ylabel("Improvement Potential", color="green")
    ax1.tick_params(axis="y", labelcolor="green")
    ax1.plot(data["Event"], data["Improvement Potential"], linestyle=':', color='#069C56', alpha=0.5, zorder=1)

    ax2.set_xlabel("Basic Events")
    ax1.legend(['--> IP'], loc='upper right', fontsize='medium', handlelength=2, handletextpad=0.5, frameon=False)
    ax2.grid(False)

    plt.title("Improvement Potentials and Event Usages for Each Basic Event")
    save_dir = '2_100K_DFT_MARL-ddqn_analysisGraphs'
    plt.savefig(f"{save_dir}/improvement_potentials_event_usages.png", bbox_inches='tight')
    # plt.show()
