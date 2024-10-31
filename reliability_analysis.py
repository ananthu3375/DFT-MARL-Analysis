"""
File:     reliability_analysis.py
Author:   Ananthu Ramesh S
Purpose:  To evaluate the reliability of the events and the overall system
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

    # def calculate_ps_reliability(self):
    #     """
    #     Calculate the reliability of the PS event based on the reliabilities of basic events A and B.
    #     """
    #     R_A = self.event_reliabilities.get("A")
    #     R_B = self.event_reliabilities.get("B")
    #     if R_A is None or R_B is None:
    #         print("Reliabilities for events A and B are not defined.")
    #         return
    #     R_PS = R_A + R_B - R_A * R_B
    #     self.gate_reliabilities["PS"] = R_PS
    #     print(f"\nIntermediate Event:\nPS = {R_PS:.4f}")
    #
    # def calculate_intermediate_Cs_Ms_reliabilities(self):
    #     """
    #     Calculate the reliabilities of the intermediate events C1, C2, M1, M2, M3 based on their child events.
    #     """
    #     intermediate_events = ["C1", "C2", "M1", "M2", "M3"]
    #
    #     # print("\nCalculating Intermediate Event Reliabilities:")
    #     for event in intermediate_events:
    #         if event in self.gates:
    #             children = self.gates[event]['children']
    #             child_reliabilities = [self.event_reliabilities.get(child, 0) for child in children]
    #             if child_reliabilities:
    #                 # Reliability of an OR gate is: R = 1 - (1 - R1)(1 - R2)...(1 - Rn). Since the OR gate logic is
    #                 # seen from the failure perspective, we have to take the AND gate logic for calculating reliability
    #                 # AND gate logic: Reliability R = R1 * R2 * ...... * Rn
    #                 reliability = np.prod(child_reliabilities)
    #                 self.gate_reliabilities[event] = reliability
    #                 print(f"{event} = {reliability:.4f}")
    #             else:
    #                 print(f"Intermediate Event {event} has no child events.")
    #
    # def calculate_csp_reliability(self):
    #     """
    #     Calculate the reliability of the CSP1 and CSP2 gates based on the reliabilities of M1, M2, and M3.
    #     For a 2-out-of-3 CSP gate, the gate fails if at least two out of the three components fail.
    #     """
    #     # Retrieve reliabilities for M1, M2, and M3
    #     R_M1 = self.gate_reliabilities.get("M1")
    #     R_M2 = self.gate_reliabilities.get("M2")
    #     R_M3 = self.gate_reliabilities.get("M3")
    #
    #     if None in [R_M1, R_M2, R_M3]:
    #         print("Reliabilities for M1, M2, or M3 are not defined.")
    #         return
    #
    #     F_M1 = 1 - R_M1
    #     F_M2 = 1 - R_M2
    #     F_M3 = 1 - R_M3
    #
    #     # Probability of failure for the 2-out-of-3 scenario
    #     P_failure_CSP = (F_M1 * F_M2) + (F_M1 * F_M3) + (F_M2 * F_M3)
    #
    #     # Calculate reliability of CSP gates
    #     R_CSP = 1 - P_failure_CSP
    #
    #     # Assign calculated reliabilities to CSP1 and CSP2
    #     self.gate_reliabilities["CSP1"] = R_CSP
    #     self.gate_reliabilities["CSP2"] = R_CSP
    #
    #     # Print results
    #     print(f"CSP1 = {R_CSP:.4f}")
    #     print(f"CSP2 = {R_CSP:.4f}")
    #
    # def calculate_g2_g3_reliability(self):
    #     """
    #     Calculate the reliability of the G2 and G3 events based on the reliabilities
    #     of intermediate events C1, C2, CSP1, and CSP2.
    #     """
    #     # Retrieve reliabilities for C1, C2, CSP1, and CSP2
    #     R_C1 = self.gate_reliabilities.get("C1")
    #     R_CSP1 = self.gate_reliabilities.get("CSP1")
    #     R_C2 = self.gate_reliabilities.get("C2")
    #     R_CSP2 = self.gate_reliabilities.get("CSP2")
    #
    #     if None in [R_C1, R_CSP1, R_C2, R_CSP2]:
    #         print("Reliabilities for C1, C2, CSP1, or CSP2 are not defined.")
    #         return
    #     R_G2 = R_C1 * R_CSP1
    #     self.gate_reliabilities["G2"] = R_G2
    #     R_G3 = R_C2 * R_CSP2
    #     self.gate_reliabilities["G3"] = R_G3
    #     print(f"G2 = {R_G2:.4f}")
    #     print(f"G3 = {R_G3:.4f}")

    def calculate_ps_failure_rate(self):
        """
        Calculate the failure rate for the PS gate, which fails if both A and B fail.
        """
        print(f"\nFailure Rates of Intermediate Events using Red Agent Policy:")
        failure_rate_A = self.BE_failure_rates.get("A", 0)
        failure_rate_B = self.BE_failure_rates.get("B", 0)
        ps_failure_rate = failure_rate_A * failure_rate_B
        self.gate_failure_rates['PS'] = ps_failure_rate
        print(f"PS = {ps_failure_rate:.4f}")

    def calculate_C1_C2_failure_rates(self):
        """
        Calculate the failure rates for the intermediate events C1 and C2,
        both of which are OR gates, using the previously calculated PS failure rate.
        """
        failure_rate_C = self.BE_failure_rates.get("C", 0)
        failure_rate_D = self.BE_failure_rates.get("D", 0)
        failure_rate_K = self.BE_failure_rates.get("K", 0)
        failure_rate_L = self.BE_failure_rates.get("L", 0)
        failure_rate_PS = self.gate_failure_rates.get('PS', 0)  # Use the PS failure rate already calculated

        # OR gate failure rate for C1 (union of C, D, and PS failures)
        failure_rate_C1 = (failure_rate_C + failure_rate_D + failure_rate_PS
                           - failure_rate_C * failure_rate_D
                           - failure_rate_C * failure_rate_PS
                           - failure_rate_D * failure_rate_PS
                           + failure_rate_C * failure_rate_D * failure_rate_PS)

        # OR gate failure rate for C2 (union of K, L, and PS failures)
        failure_rate_C2 = (failure_rate_K + failure_rate_L + failure_rate_PS
                           - failure_rate_K * failure_rate_L
                           - failure_rate_K * failure_rate_PS
                           - failure_rate_L * failure_rate_PS
                           + failure_rate_K * failure_rate_L * failure_rate_PS)

        self.gate_failure_rates["C1"] = failure_rate_C1
        self.gate_failure_rates["C2"] = failure_rate_C2
        print(f"C1 = {failure_rate_C1:.4f}")
        print(f"C2 = {failure_rate_C2:.4f}")

    def calculate_M1_M2_M3_failure_rates(self):
        """
        Calculate the reliabilities of the intermediate events M1, M2, M3 based on their child events.
        """
        intermediate_events = ["M1", "M2", "M3"]

        # print("\nCalculating Intermediate Event Reliabilities:")
        for event in intermediate_events:
            if event in self.gates:
                children = self.gates[event]['children']
                child_failure_rates = [self.BE_failure_rates.get(child, 0) for child in children]
                if child_failure_rates:
                    failure_rate = child_failure_rates[0] + child_failure_rates[1] - (child_failure_rates[0] * child_failure_rates[1])
                    self.gate_failure_rates[event] = failure_rate
                    print(f"{event} = {failure_rate:.4f}")
                else:
                    print(f"Intermediate Event {event} has no child events.")

    def calculate_csp_failure_rate(self):
        """
        Calculate the failure rates for CSP1 and CSP2 using the failure rates of M1, M2, and M3.
        """
        # Ensure M1, M2, and M3 have failure rates
        failure_rate_M1 = self.gate_failure_rates.get("M1", 0)
        failure_rate_M2 = self.gate_failure_rates.get("M2", 0)
        failure_rate_M3 = self.gate_failure_rates.get("M3", 0)

        # Calculate failure rate for CSP1
        # CSP1 fails if M1 fails and (M3 is either failed or being used by CSP2)
        csp1_failure_rate = failure_rate_M1 * (failure_rate_M3 + (1 - failure_rate_M3) * failure_rate_M2)
        self.gate_failure_rates["CSP1"] = csp1_failure_rate
        print(f"CSP1 = {csp1_failure_rate:.4f}")

        # Calculate failure rate for CSP2
        # CSP2 fails if M2 fails and (M3 is either failed or being used by CSP1)
        csp2_failure_rate = failure_rate_M2 * (failure_rate_M3 + (1 - failure_rate_M3) * failure_rate_M1)
        self.gate_failure_rates["CSP2"] = csp2_failure_rate
        print(f"CSP2 = {csp2_failure_rate:.4f}")

    # def calculate_csp_failure_rate(self):
    #     """
    #     Calculate the failure rates for CSP1 and CSP2 using the failure rates of M1, M2, and M3.
    #     Both CSP1 and CSP2 will fail if:
    #     - M1 and M2 fail
    #     - M1 and M3 fail
    #     - M2 and M3 fail
    #     - All three M1, M2, and M3 fail
    #     """
    #     # Retrieve failure rates for M1, M2, and M3
    #     failure_rate_M1 = self.gate_failure_rates.get("M1", 0)
    #     failure_rate_M2 = self.gate_failure_rates.get("M2", 0)
    #     failure_rate_M3 = self.gate_failure_rates.get("M3", 0)
    #
    #     # Calculate CSP1 and CSP2 failure rates based on the logic provided
    #     # We calculate the union of all possible failure conditions:
    #
    #     # Probability of M1 and M2 failing
    #     failure_M1_M2 = failure_rate_M1 * failure_rate_M2
    #
    #     # Probability of M1 and M3 failing
    #     failure_M1_M3 = failure_rate_M1 * failure_rate_M3
    #
    #     # Probability of M2 and M3 failing
    #     failure_M2_M3 = failure_rate_M2 * failure_rate_M3
    #
    #     # Probability of M1, M2, and M3 all failing
    #     failure_M1_M2_M3 = failure_rate_M1 * failure_rate_M2 * failure_rate_M3
    #
    #     # Calculate failure rate for CSP1 and CSP2 as the union of these events
    #     csp_failure_rate = (failure_M1_M2 + failure_M1_M3 + failure_M2_M3
    #                         + failure_M1_M2_M3)  # Union of all combinations
    #
    #     # Store and print the calculated failure rates for CSP1 and CSP2
    #     self.gate_failure_rates["CSP1"] = csp_failure_rate
    #     self.gate_failure_rates["CSP2"] = csp_failure_rate
    #     print(f"CSP1 Failure Rate: {csp_failure_rate:.4f}")
    #     print(f"CSP2 Failure Rate: {csp_failure_rate:.4f}")

    # def calculate_csp_failure_rate(self):
    #     """
    #     Calculate the failure rates for CSP1 and CSP2 as 2-out-of-3 gates
    #     with child events M1, M2, and M3.
    #     """
    #     failure_rate_M1 = self.gate_failure_rates.get("M1", 0)
    #     failure_rate_M2 = self.gate_failure_rates.get("M2", 0)
    #     failure_rate_M3 = self.gate_failure_rates.get("M3", 0)
    #
    #     failure_M1_M2_only = failure_rate_M1 * failure_rate_M2 * (1 - failure_rate_M3)
    #     failure_M1_M3_only = failure_rate_M1 * failure_rate_M3 * (1 - failure_rate_M2)
    #     failure_M2_M3_only = failure_rate_M2 * failure_rate_M3 * (1 - failure_rate_M1)
    #     failure_M1_M2_M3 = failure_rate_M1 * failure_rate_M2 * failure_rate_M3
    #     csp_failure_rate = (failure_M1_M2_only + failure_M1_M3_only
    #                         + failure_M2_M3_only + failure_M1_M2_M3)
    #     self.gate_failure_rates["CSP1"] = csp_failure_rate
    #     self.gate_failure_rates["CSP2"] = csp_failure_rate
    #     print(f"CSP1 = {csp_failure_rate:.4f}")
    #     print(f"CSP2 = {csp_failure_rate:.4f}")

    def calculate_G2_G3_failure_rates(self):
        """
        Calculate the reliabilities of the intermediate events G2, G3 based on their child events.
        """
        intermediate_events = ["G2", "G3"]
        for event in intermediate_events:
            if event in self.gates:
                children = self.gates[event]['children']
                child_failure_rates = [self.gate_failure_rates.get(child, 0) for child in children]
                if child_failure_rates:
                    failure_rate = child_failure_rates[0] + child_failure_rates[1] - (child_failure_rates[0] * child_failure_rates[1])
                    self.gate_failure_rates[event] = failure_rate
                    print(f"{event} = {failure_rate:.4f}")
                else:
                    print(f"Intermediate Event {event} has no child events.")

    def calculate_G1_failure_rate(self):
        """
        Calculate the reliability of the top event G1 based on its child events.
        """
        failure_rate_G2 = self.gate_failure_rates.get("G2", 0)
        failure_rate_G3 = self.gate_failure_rates.get("G3", 0)
        g1_failure_rate = failure_rate_G2 * failure_rate_G3
        self.gate_failure_rates['G1'] = g1_failure_rate
        print(f"\nTop Event Failure Rate: G1 = {g1_failure_rate:.4f}")

    def run(self):
        self.calculate_total_event_usage()
        self.calculate_reliability()
        self.calculate_BE_failure_rate()
        # self.calculate_ps_reliability()
        # self.calculate_intermediate_Cs_Ms_reliabilities()
        # self.calculate_csp_reliability()
        # self.calculate_g2_g3_reliability()
        self.calculate_ps_failure_rate()
        self.calculate_C1_C2_failure_rates()
        self.calculate_M1_M2_M3_failure_rates()
        self.calculate_csp_failure_rate()
        self.calculate_G2_G3_failure_rates()
        self.calculate_G1_failure_rate()


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
