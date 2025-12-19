# Agentic AI for Enterprise CRM Sales Forecasting

This repository contains the implementation of the **Perception–Action Loop** described in the paper:

> **"Improving Sales Forecast Accuracy and Autonomous Decisioning Using Agentic AI in Enterprise CRM Platforms."**

---

## Project Overview

Traditional CRM forecasting is passive. This project introduces an Agentic AI Framework that shifts the paradigm from simple prediction to Autonomous Decisioning. The architecture consists of a two-layer system:

- **Perception Laye:r** A Feedforward Neural Network (FNN) that identifies non-linear deal patterns, achieving a 17% reduction in Mean Absolute Error (MAE).

- **Action Layer:** A ReAct (Reasoning + Acting) Agent that autonomously executes business logic (e.g., discount triggers, resource allocation) based on forecast confidence and SHAP explainability scores.

##  Architecture
The system follows a -**Perception-Action Loop** integrated via API into a standard CRM (like Salesforce or SAP):

graph LR
    A[CRM Data] --> B[Neural Perception]
    B --> C[SHAP Explainability]
    C --> D[Agentic Reasoning Engine]
    D --> E[Autonomous Action]
    E --> A

---

## 📂 Repository Structure

```text
├── perception_model.py     # Neural Network architecture and training pipeline
├── agent_engine.py         # The ReAct logic and autonomous decision-making scripts.
├── data_processor.py       # Feature engineering, One-Hot Encoding, and Scaling.
├── demo_notebook.ipynb     # End-to-end walkthrough for reviewers.
├── requirements.txt        # Dependency list (TensorFlow, SHAP, Scikit-Learn).
└── models/                 # Pre-trained model weights (.keras format).
