"""
File:     reliability_analysis.py
Author:   Ananthu Ramesh S
Purpose:  To evaluate the reliability of the events and the overall system
          To calculate Birnbaum's Importance and Improvement Potential
"""

import numpy as np
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
        self.gate_reliabilities = {}
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
        print(f"\nTotal Event Usages by Red Agent: {self.total_event_usages}")

    def calculate_reliability(self):
        """
        Calculate the reliability of each event for the red agent using the formula:
        Reliability of Event_i = 1 - (Usage of Event_i / Total Usages) * Failure Rate (System)
        """
        print("\n\nReliability of Events for Red Agent:")
        for event in sorted(self.red_agent_event_count.keys()):
            usage_count = self.red_agent_event_count[event]
            reliability = 1 - (usage_count / self.total_event_usages) * self.failure_rate
            self.event_reliabilities[event] = reliability
            print(f"Event {event}: {reliability:.4f}")

    def calculate_ps_reliability(self):
        """
        Calculate the reliability of the PS event based on the reliabilities of basic events A and B.
        """
        R_A = self.event_reliabilities.get("A")
        R_B = self.event_reliabilities.get("B")
        if R_A is None or R_B is None:
            print("Reliabilities for events A and B are not defined.")
            return
        R_PS = R_A + R_B - R_A * R_B
        self.gate_reliabilities["PS"] = R_PS
        print(f"\nIntermediate Event:\nPS = {R_PS:.4f}")

    def calculate_intermediate_event_reliabilities(self):
        """
        Calculate the reliabilities of the intermediate events C1, C2, M1, M2, M3 based on their child events.
        """
        intermediate_events = ["C1", "C2", "M1", "M2", "M3"]

        # print("\nCalculating Intermediate Event Reliabilities:")
        for event in intermediate_events:
            if event in self.gates:
                children = self.gates[event]['children']
                child_reliabilities = [self.event_reliabilities.get(child, 0) for child in children]
                if child_reliabilities:
                    # Reliability of an OR gate is: R = 1 - (1 - R1)(1 - R2)...(1 - Rn). Since the OR gate logic is
                    # seen from the failure perspective, we have to take the AND gate logic for calculating reliability
                    # AND gate logic: Reliability R = R1 * R2 * ...... * Rn
                    reliability = np.prod(child_reliabilities)
                    self.gate_reliabilities[event] = reliability
                    print(f"{event} = {reliability:.4f}")
                else:
                    print(f"Intermediate Event {event} has no child events.")

    def run(self):
        self.calculate_total_event_usage()
        self.calculate_reliability()
        self.calculate_ps_reliability()
        self.calculate_intermediate_event_reliabilities()


if __name__ == "__main__":
    # Run the game to generate the action data
    results = actual_game()

    # Extract the actions and wins from actual_game output
    actions = results['actions']
    wins = results['wins']

    # Calculate reliability
    reliability_analysis = ReliabilityAnalysis(actions, wins, 'model.xml')
    reliability_analysis.run()
