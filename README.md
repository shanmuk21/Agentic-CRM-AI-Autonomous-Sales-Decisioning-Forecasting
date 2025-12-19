# Agentic AI for Enterprise CRM Sales Forecasting

This repository contains the implementation of the **Perception–Action Loop** described in the paper:

> **"Improving Sales Forecast Accuracy and Autonomous Decisioning Using Agentic AI in Enterprise CRM Platforms."**

---

## 🚀 Overview

This system combines **Neural Perception** and **Agentic Reasoning** to improve enterprise sales forecasting and enable autonomous CRM decision-making.

A **Feedforward Neural Network (FNN)** predicts the win probability of sales opportunities, while an **Agentic AI layer** executes autonomous **Next-Best-Actions (NBA)** based on real-time predictions and explainability signals.

---

## ✨ Key Features

- **Neural Perception**
  - Deep learning–based win/loss prediction
  - High accuracy over traditional baseline models

- **Autonomous Agency**
  - ReAct-style agentic reasoning loop
  - Real-time CRM interventions

- **Explainability**
  - SHAP-based feature attribution
  - Transparent justification for every autonomous decision

---

## 📂 Project Structure

```text
├── model.py          # Neural Network architecture and training pipeline
├── agent.py          # Agentic reasoning engine and action logic
├── main.py           # End-to-end demo execution
├── requirements.txt  # Python dependencies
├── data/             # Synthetic / anonymized CRM dataset
└── notebooks/        # Model comparisons (FNN vs RF vs Linear Regression)
🛠️ Installation & Setup
1️⃣ Clone the repository
bash
Copy code
git clone https://github.com/your-username/agentic-crm-ai.git
cd agentic-crm-ai
2️⃣ Install dependencies
bash
Copy code
pip install -r requirements.txt
3️⃣ Run the demo
bash
Copy code
python main.py
📊 Model Performance
The neural network model achieves a 17% improvement in MAE compared to traditional baseline models.

Model	MAE	RMSE
Linear Regression	1.12	1.45
Neural Network	0.74	1.05

🤖 Agentic Logic Example
The agent continuously monitors the win_probability signal.

When the probability drops below 0.60, the agent autonomously:

Calculates top negative feature drivers using SHAP

Triggers a Competitive Save Discount (5%)

Updates the CRM task owner with a Justification Report

This closed-loop Perception → Reasoning → Action cycle enables proactive and explainable CRM interventions.

📌 Use Cases
Enterprise CRM sales forecasting

Autonomous deal risk mitigation

AI-assisted sales enablement

Explainable AI decision support

📜 License
This project is intended for research and enterprise experimentation.
Please review licensing requirements before production use.
