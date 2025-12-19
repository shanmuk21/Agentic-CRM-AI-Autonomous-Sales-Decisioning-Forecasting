"""
agent_engine.py
Author: Shanmukha Rao Bodala
Description: Implementation of the Agentic Reasoning Layer (Action Layer).
This script utilizes the win-probability from the Perception Model to 
execute autonomous business logic via a ReAct (Reasoning + Acting) loop.
"""

import numpy as np
import time

class CRMAgent:
    def __init__(self, model, preprocessor, threshold=0.60):
        """
        Initialize the Agent.
        :param model: The trained Keras/TensorFlow perception model.
        :param preprocessor: The fitted ColumnTransformer for data scaling.
        :param threshold: The probability threshold for autonomous intervention.
        """
        self.model = model
        self.preprocessor = preprocessor
        self.threshold = threshold

    def analyze_and_act(self, raw_deal_data):
        """
        The Core ReAct Loop: Observe -> Reason -> Act.
        """
        # 1. OBSERVE: Preprocess the raw CRM record
        processed_data = self.preprocessor.transform(raw_deal_data)
        
        # 2. PERCEIVE: Get win probability from the Neural Network
        win_prob = self.model.predict(processed_data, verbose=0)[0][0]
        
        print(f"\n[Observation] Current Win Probability: {win_prob:.2%}")
        
        # 3. REASON: Evaluate against business constraints
        decision_log = self._reasoning_logic(win_prob, raw_deal_data)
        
        # 4. ACT: Execute autonomous intervention if necessary
        self._execute_action(decision_log)
        
        return decision_log

    def _reasoning_logic(self, win_prob, raw_data):
        """
        Internal reasoning engine to determine the 'Next-Best-Action'.
        """
        logic_output = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "probability": win_prob,
            "action_required": False,
            "action_type": "NONE",
            "justification": ""
        }

        # Logic for at-risk deals
        if win_prob < self.threshold:
            logic_output["action_required"] = True
            
            # Sub-reasoning based on deal size (Example of Agentic nuance)
            deal_value = raw_data['Opportunity Amount USD'].values[0]
            
            if deal_value > 50000:
                logic_output["action_type"] = "AUTONOMOUS_DISCOUNT_TRIGGER"
                logic_output["justification"] = f"Probability ({win_prob:.2%}) below threshold for high-value deal. Triggering 5% save-discount."
            else:
                logic_output["action_type"] = "MANAGER_ALERT_TASK"
                logic_output["justification"] = "Low probability detected. Escalating to Sales Manager for review."
        
        else:
            logic_output["justification"] = "Deal health is optimal. No autonomous intervention required."

        return logic_output

    def _execute_action(self, decision_log):
        """
        Simulates the execution of an action via CRM API/Webhooks.
        """
        if decision_log["action_required"]:
            print(f"[Action Executed] {decision_log['action_type']}")
            print(f"[Justification] {decision_log['justification']}")
        else:
            print("[Status] Monitoring active. No intervention needed.")

# --- DEMO EXECUTION ---
if __name__ == "__main__":
    print("Agent Engine Mode: Standalone Simulation")
    # Note: In production, you would load the pre-trained model and preprocessor:
    # model