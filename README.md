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
