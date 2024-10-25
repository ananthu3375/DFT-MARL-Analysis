"""
File:     reliability_analysis.py
Author:   Ananthu Ramesh S
Purpose:  To evaluate the reliability of the events and the overall system
          To calculate Birnbaum's Importance and Improvement Potential
"""

import numpy as np
import xml.etree.ElementTree as ET


class ReliabilityAnalysis:
    def __init__(self, model_file, time=1, delta=1e-5):
        """
        Initialized the ReliabilityAnalysis class with the XML model file and time.
        :param model_file: Path to the XML file defining the DFT
        :param time: Time at which reliability is calculated(default is 1)
        :param delta: Small change for numerical differentiation (default is 1e-5)
        """
        self.time = time
        self.delta = delta
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
            if event_type == 'BASIC':
                mttr = float(event.get('mttr'))
                repair_cost = float(event.get('repair_cost', 1))
                failure_cost = float(event.get('failure_cost', 1))
                self.events[event_name] = {
                    'type': event_type,
                    'mttr': mttr,
                    'repair_cost': repair_cost,
                    'failure_cost': failure_cost
                }
            else:
                gate_type = event.get('gate_type')
                self.gates[event_name] = {
                    'type': event_type,
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

    def calculate_reliability(self, mttr, repair_cost, failure_cost, alpha=0.5):
        """
        To calculate the reliability of a basic event based on MTTR, repair cost, and failure cost.
        :param mttr: Mean Time to Repair (MTTR) value for the event
        :param repair_cost: Cost of repairing the event after failure
        :param failure_cost: Cost of failure of the event
        :param alpha: Weighting factor for failure cost influence
        :return: Reliability value at time t
        """
        lambda_ = (1 / mttr) * (1 + alpha * (failure_cost / repair_cost))
        return np.exp(-lambda_ * self.time)

    def calculate_event_reliabilities(self):
        """
        To calculate the reliability for each basic event in the system.
        :return: Dictionary of event reliabilities
        """
        self.event_reliabilities = {
            event: self.calculate_reliability(data['mttr'], data['repair_cost'], data['failure_cost'])
            for event, data in self.events.items()
        }

    def calculate_ps_gate_reliability(self):
        """
        Calculates the reliability of the PS gate based on the reliabilities of A and B.
        Considers their MTTR, repair cost, and failure cost while calculating their lambda.
        Uses the priority AND (PS) gate formula.
        """
        event_A = self.events['A']
        event_B = self.events['B']
        reliability_A = self.calculate_reliability(event_A['mttr'], event_A['repair_cost'], event_A['failure_cost'])
        reliability_B = self.calculate_reliability(event_B['mttr'], event_B['repair_cost'], event_B['failure_cost'])
        # lambda_A = (1 / event_A['mttr']) * (1 + 0.5 * (event_A['failure_cost'] / event_A['repair_cost']))
        # lambda_B = (1 / event_B['mttr']) * (1 + 0.5 * (event_B['failure_cost'] / event_B['repair_cost']))

        # PS gate reliability formula
        # reliability_PS = reliability_A + reliability_B - (reliability_A * (lambda_A / (lambda_A + lambda_B)) * reliability_B)
        reliability_PS = reliability_A * reliability_B
        self.gate_reliabilities['PS'] = reliability_PS
        return reliability_PS

    def calculate_csp_reliability(self, M1_reliability, M2_reliability, M3_reliability):
        # Failure probabilities for M1, M2, M3
        P_fail_M1 = 1 - M1_reliability
        P_fail_M2 = 1 - M2_reliability
        P_fail_M3 = 1 - M3_reliability

        # Probability of exactly two failures
        P_fail_two = (P_fail_M1 * P_fail_M2 * M3_reliability) + \
                     (P_fail_M1 * P_fail_M3 * M2_reliability) + \
                     (P_fail_M2 * P_fail_M3 * M1_reliability)

        # Probability of all three failing
        P_fail_all_three = P_fail_M1 * P_fail_M2 * P_fail_M3

        # Total probability of CSP gates failing (two or more failures)
        P_fail_CSP = P_fail_two + P_fail_all_three

        # Reliability of CSP1 and CSP2
        R_CSP1 = 1 - P_fail_CSP
        R_CSP2 = 1 - P_fail_CSP

        return R_CSP1, R_CSP2

    def evaluate_gate(self, gate_name, visited=None):
        global reliability
        if visited is None:
            visited = set()

        if gate_name in visited:
            print(f"Warning: Circular reference detected at gate: {gate_name}")
            return 1

        visited.add(gate_name)

        gate = self.gates[gate_name]
        gate_type = gate['gate_type']

        if gate_type == 'CSP':
            # Get reliabilities of M1, M2, M3
            M1_reliability = self.get_event_or_gate_reliability('M1', visited)
            M2_reliability = self.get_event_or_gate_reliability('M2', visited)
            M3_reliability = self.get_event_or_gate_reliability('M3', visited)

            # Calculate CSP gate reliability
            R_CSP1, R_CSP2 = self.calculate_csp_reliability(M1_reliability, M2_reliability, M3_reliability)

            # Return the appropriate CSP reliability based on which gate is being evaluated
            if gate_name == 'CSP1':
                reliability = R_CSP1
            elif gate_name == 'CSP2':
                reliability = R_CSP2

        else:
            # Handle other gate types (AND, OR, FDEP)
            children_reliabilities = [self.get_event_or_gate_reliability(child, visited) for child in gate['children']]
            if gate_type == 'AND':
                reliability = np.prod(children_reliabilities)
            elif gate_type == 'OR':
                reliability = 1 - np.prod([1 - r for r in children_reliabilities])
            elif gate_type == 'FDEP':
                fdep_sources = gate.get('fdep_sources', [])
                source_reliabilities = [self.get_event_or_gate_reliability(src, visited) for src in fdep_sources]
                children_reliabilities = [self.get_event_or_gate_reliability(child, visited) for child in
                                          gate['children']]
                reliability = np.prod(source_reliabilities) * np.prod(children_reliabilities)
            else:
                raise ValueError(f"Unsupported gate type: {gate_type}")
        self.gate_reliabilities[gate_name] = reliability
        visited.remove(gate_name)
        return reliability

    def get_event_or_gate_reliability(self, name, visited):
        """
        Retrieve the reliability of a basic event or calculate it for a gate.
        :param name: Name of the event or gate
        :return: Reliability value
        """
        if name in self.event_reliabilities:
            return self.event_reliabilities[name]
        else:
            return self.evaluate_gate(name, visited)

    def calculate_system_reliability(self, visited=None):
        """
        To calculate the reliability of the system using the top gate.
        :return: The system reliability value
        """
        top_gate = [name for name, gate in self.gates.items() if gate['type'] == 'TOP'][0]
        self.R_system = self.evaluate_gate(top_gate, visited)
        return self.R_system

    def perform_analysis(self):
        """
        To perform the full analysis to calculate event reliabilities and system reliability.
        """
        self.calculate_event_reliabilities()
        self.calculate_ps_gate_reliability()
        self.calculate_system_reliability()
        self.print_results()

    def print_results(self):
        """
        To print the calculated values for event reliabilities, gate reliabilities, and system reliability.
        """
        print("Event Reliabilities:")
        for event, reliability in self.event_reliabilities.items():
            print(f"{event}: {reliability:.6f}")

        print("\nGate Reliabilities (Intermediate Events):")
        for gate, reliability in self.gate_reliabilities.items():
            print(f"{gate}: {reliability:.6f}")

        print("\nSystem Reliability:", self.R_system)


class ImportanceAnalysis:
    def __init__(self, reliability_analysis, delta=1e-2):
        """
        Initialize the ImportanceAnalysis class with the results from ReliabilityAnalysis.
        :param reliability_analysis: Instance of ReliabilityAnalysis with completed reliability calculations
        :param delta: Small change for numerical differentiation
        """
        self.reliability_analysis = reliability_analysis
        self.delta = delta
        self.R_system = reliability_analysis.R_system
        self.original_event_reliabilities = reliability_analysis.event_reliabilities.copy()

    def calculate_birnbaum_importance(self):
        """
        Calculate Birnbaum Importance for all basic events.
        """
        birnbaum_importances = {}
        for event in self.original_event_reliabilities:
            original_reliability = self.original_event_reliabilities[event]
            perturbed_reliability = min(1.0, original_reliability + self.delta)

            # Perturb the event reliability
            self.reliability_analysis.event_reliabilities[event] = perturbed_reliability

            # Recalculate system reliability
            perturbed_system_reliability = self.reliability_analysis.calculate_system_reliability()

            # Restore original reliability
            self.reliability_analysis.event_reliabilities[event] = original_reliability

            birnbaum_importances[event] = (perturbed_system_reliability - self.R_system) / self.delta

        return birnbaum_importances

    def calculate_improvement_potential(self):
        """
        Calculate Improvement Potential for all basic events.
        """
        improvement_potentials = {}
        for event in self.original_event_reliabilities:
            original_reliability = self.original_event_reliabilities[event]

            # Set event reliability to 1
            self.reliability_analysis.event_reliabilities[event] = 1.0

            # Recalculate system reliability
            improved_system_reliability = self.reliability_analysis.calculate_system_reliability()

            # Restore original reliability
            self.reliability_analysis.event_reliabilities[event] = original_reliability

            improvement_potentials[event] = improved_system_reliability - self.R_system

        return improvement_potentials

    def perform_importance_analysis(self):
        """
        Perform both Birnbaum Importance and Improvement Potential analysis.
        """
        birnbaum_importances = self.calculate_birnbaum_importance()
        improvement_potentials = self.calculate_improvement_potential()
        self.print_importance_results(birnbaum_importances, improvement_potentials)

    def print_importance_results(self, birnbaum_importances, improvement_potentials):
        """
        Print Birnbaum Importance and Improvement Potential results.
        """
        print("\nBirnbaum Importances:")
        for event, importance in birnbaum_importances.items():
            print(f"{event}: {importance:.6f}")

        print("\nImprovement Potentials:")
        for event, potential in improvement_potentials.items():
            print(f"{event}: {potential:.6f}")


if __name__ == '__main__':
    reliability_analysis = ReliabilityAnalysis('model.xml')
    reliability_analysis.perform_analysis()

    importance_analysis = ImportanceAnalysis(reliability_analysis)
    importance_analysis.perform_importance_analysis()
