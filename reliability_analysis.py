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

    def calculate_csp_reliability(self, M1_reliability, M2_reliability, M3_reliability):
        P_fail_M1 = 1 - M1_reliability
        P_fail_M2 = 1 - M2_reliability
        P_fail_M3 = 1 - M3_reliability

        P_fail_two = (P_fail_M1 * P_fail_M2 * M3_reliability) + \
                     (P_fail_M1 * P_fail_M3 * M2_reliability) + \
                     (P_fail_M2 * P_fail_M3 * M1_reliability)
        P_fail_all_three = P_fail_M1 * P_fail_M2 * P_fail_M3
        P_fail_CSP = P_fail_two + P_fail_all_three

        R_CSP1 = 1 - P_fail_CSP
        R_CSP2 = 1 - P_fail_CSP

        return R_CSP1, R_CSP2

    def evaluate_gate(self, gate_name, visited=None):
        if visited is None:
            visited = set()

        if gate_name in visited:
            print(f"Warning: Circular reference detected at gate: {gate_name}")
            return 1

        visited.add(gate_name)
        gate = self.gates[gate_name]
        gate_type = gate['gate_type']

        print(f"Evaluating gate {gate_name} (type: {gate_type}) with children: {gate['children']}")  ###

        if gate_type == 'CSP':
            M1_reliability = self.get_event_or_gate_reliability('M1', visited)
            M2_reliability = self.get_event_or_gate_reliability('M2', visited)
            M3_reliability = self.get_event_or_gate_reliability('M3', visited)
            print(f"CSP gate {gate_name}: M1={M1_reliability}, M2={M2_reliability}, M3={M3_reliability}")  ###

            R_CSP1, R_CSP2 = self.calculate_csp_reliability(M1_reliability, M2_reliability, M3_reliability)
            reliability = R_CSP1 if gate_name == 'CSP1' else R_CSP2

        else:
            children_reliabilities = [self.get_event_or_gate_reliability(child, visited) for child in gate['children']]
            print(f"{gate_type} gate {gate_name} children reliabilities: {children_reliabilities}")  ###
            if gate_type == 'OR':  # OR gate reliability: R1 * R2 * ... * Rn
                reliability = np.prod(children_reliabilities)
            elif gate_type == 'AND':  # AND gate reliability: 1 - (1 - R1) * (1 - R2) * ... * (1 - Rn) : since the
                # gates are implemented from a fault propagation perspective
                reliability = 1 - np.prod([1 - r for r in children_reliabilities])
            elif gate_type == 'FDEP':
                fdep_sources = gate.get('fdep_sources', [])
                source_reliabilities = [self.get_event_or_gate_reliability(src, visited) for src in fdep_sources]
                reliability = np.prod(source_reliabilities) * np.prod(children_reliabilities)
                print(f"FDEP gate {gate_name}: source reliabilities: {source_reliabilities}")  ###
            else:
                raise ValueError(f"Unsupported gate type: {gate_type}")

        self.gate_reliabilities[gate_name] = reliability
        visited.remove(gate_name)
        return reliability

    def get_event_or_gate_reliability(self, name, visited):
        if name in self.event_reliabilities:
            return self.event_reliabilities[name]
        else:
            return self.evaluate_gate(name, visited)

    def calculate_intermediate_reliabilities(self):
        for event in self.gates:
            self.evaluate_gate(event)

    def calculate_system_reliability(self, visited=None):
        top_gate = [name for name, gate in self.gates.items() if gate['type'] == 'TOP'][0]
        self.R_system = self.evaluate_gate(top_gate, visited)
        return self.R_system

    def run(self):
        self.calculate_total_event_usage()
        self.calculate_reliability()
        self.calculate_intermediate_reliabilities()

        print("\n\nReliability of Intermediate Events:")
        for event, reliability in self.event_reliabilities.items():
            if event in self.gates:
                print(f"Intermediate Event {event}: {reliability:.4f}")

        system_reliability = self.calculate_system_reliability()
        print(f"\n\nSystem Reliability: {system_reliability:.4f}")


if __name__ == "__main__":
    # Run the game to generate the action data
    results = actual_game()

    # Extract the actions and wins from actual_game output
    actions = results['actions']
    wins = results['wins']

    # Calculate reliability
    reliability_analysis = ReliabilityAnalysis(actions, wins, 'model.xml')
    reliability_analysis.run()
